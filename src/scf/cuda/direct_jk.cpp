#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_jk_plan.hpp"
#include "scf/cuda/metadata_upload.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/direct_task_layout.hpp"

namespace vibeqc::scf {

namespace {
using namespace cuda_execution;
}

CudaDirectJkPlan::~CudaDirectJkPlan() {
  if (device_id >= 0) (void)cudaSetDevice(device_id);
  if (stream) (void)cudaStreamSynchronize(stream);
  generated_coulomb.reset();  // Release the borrower before its stream/metadata.
  for (void* pointer : allocations) (void)runtime::resource_cuda_free(pointer);
  if (stream) (void)cudaStreamDestroy(stream);
}

namespace {

struct DirectJkFailure {
  vibeqc_status status;
  std::string detail;
};
void direct_jk_check(cudaError_t error) {
  if (error != cudaSuccess)
    throw DirectJkFailure{source_cuda_status(error), cudaGetErrorString(error)};
}
void direct_jk_require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
std::size_t direct_jk_product(std::size_t a, std::size_t b) {
  std::size_t out;
  if (!vibeqc::runtime::checked_multiply(a, b, out)) throw std::bad_alloc();
  return out;
}
void direct_jk_finite(const std::vector<double>& values) {
  for (double value : values) direct_jk_require(std::isfinite(value), "nonfinite direct J/K data");
}
void direct_jk_finite_result(const std::vector<double>& values) {
  for (double value : values)
    if (!std::isfinite(value))
      throw DirectJkFailure{VIBEQC_STATUS_NUMERICAL_FAILURE, "nonfinite direct J/K result"};
}
/** Fence before local download buffers unwind on a CUDA exception. The outer
 * status guard alone runs too late to protect buffers owned inside its lambda.
 */
struct DirectJkDownloadFence {
  cudaStream_t stream;
  ~DirectJkDownloadFence() {
    if (stream) (void)cudaStreamSynchronize(stream);
  }
  void complete() {
    direct_jk_check(cudaStreamSynchronize(stream));
    stream = nullptr;
  }
};
/** Always fence failed uploads too: caller-owned pageable buffers may die on return. */
template <class Function>
vibeqc_status direct_jk_guard(CudaDirectJkPlan* plan, std::string& detail, Function function) {
  detail.clear();
  try {
    function();
    return VIBEQC_STATUS_SUCCESS;
  } catch (const DirectJkFailure& failure) {
    if (plan && plan->stream) (void)cudaStreamSynchronize(plan->stream);
    detail = failure.detail;
    return failure.status;
  } catch (const std::bad_alloc&) {
    if (plan && plan->stream) (void)cudaStreamSynchronize(plan->stream);
    detail = "direct J/K allocation exceeds available capacity";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (cudaError_t error) {
    if (plan && plan->stream) (void)cudaStreamSynchronize(plan->stream);
    detail = cudaGetErrorString(error);
    return source_cuda_status(error);
  } catch (const std::exception& error) {
    if (plan && plan->stream) (void)cudaStreamSynchronize(plan->stream);
    detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
}

DirectCoulombRange direct_exchange_range(const FockTermSpec& term) {
  if (!term.present || term.op == FockOperator::FullRange) return DirectCoulombRange::Full;
  if (term.op == FockOperator::ShortRange) return DirectCoulombRange::Short;
  if (term.op == FockOperator::LongRange) return DirectCoulombRange::Long;
  throw std::invalid_argument("unknown exact-exchange radial operator");
}
FockBuildSpec direct_jk_strategy(const CudaDirectJkPlan* plan, FockBuildSpec spec,
                                 std::size_t begin, std::size_t count) {
  direct_jk_require(plan != nullptr, "null direct J/K plan");
  direct_jk_require(begin < plan->diagnostic.batch_size && count > 0 &&
                        count <= plan->diagnostic.batch_size - begin,
                    "direct J/K item range is invalid");
  spec = resolve_fock_build(spec, FockBackend::Cuda).spec;
  for (const auto* term : {&spec.coulomb, &spec.exchange})
    direct_jk_require(!term->present || term->approximation == FockApproximation::Exact,
                      "exact direct source cannot execute a fitted provider");
  direct_jk_require(!spec.coulomb.present || spec.coulomb.op == FockOperator::FullRange,
                    "direct CUDA Coulomb supports only the full-range operator");
  // The existing full-range Schwarz matrix is a conservative bound for both
  // erf(omega r)/r and erfc(omega r)/r: their Fourier multipliers are
  // nonnegative and bounded above by the full Coulomb multiplier.
  direct_jk_require(spec.derivative_order <= plan->derivative_order,
                    "direct source lacks requested derivative capability");
  return spec;
}
FockBuildSpec direct_jk_spec(const CudaDirectJkPlan* plan, FockBuildSpec spec,
                             const std::vector<double>& density, const std::vector<double>& beta,
                             std::size_t begin, std::size_t count) {
  spec = direct_jk_strategy(plan, spec, begin, count);
  direct_jk_require(
      density.size() == count * plan->diagnostic.nbf * plan->diagnostic.nbf &&
          (spec.spin == FockSpin::Unrestricted ? beta.size() == density.size() : beta.empty()),
      "direct J/K density/spin dimensions mismatch");
  direct_jk_finite(density);
  direct_jk_finite(beta);
  return spec;
}
void direct_jk_upload_density(CudaDirectJkPlan& plan, const std::vector<double>& density,
                              const std::vector<double>& beta, std::size_t offset) {
  direct_jk_check(cudaSetDevice(plan.device_id));
  const std::size_t bytes = density.size() * sizeof(double);
  direct_jk_check(cudaMemcpyAsync(plan.density + offset, density.data(), bytes,
                                  cudaMemcpyHostToDevice, plan.stream));
  if (!beta.empty())
    direct_jk_check(cudaMemcpyAsync(plan.beta + offset, beta.data(), bytes, cudaMemcpyHostToDevice,
                                    plan.stream));
}
}  // namespace

std::size_t cuda_direct_jk_device_bytes(std::size_t batch, std::size_t nao, std::size_t atoms,
                                        std::size_t shells, std::size_t primitives,
                                        unsigned derivative_order) {
  direct_jk_require(batch && nao && atoms && shells && primitives && derivative_order <= 1,
                    "invalid direct J/K resource shape");
  std::size_t bytes = sizeof(int);
  const auto add = [&](std::size_t count, std::size_t width) {
    const auto term = direct_jk_product(count, width);
    if (!vibeqc::runtime::checked_add(bytes, term, bytes)) throw std::bad_alloc();
  };
  const auto aos = direct_jk_product(batch, nao);
  const auto matrices = direct_jk_product(aos, nao);
  direct_jk_require(matrices <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
                        atoms <= static_cast<std::size_t>(std::numeric_limits<int>::max()) / 3,
                    "direct J/K resource shape exceeds launch dimensions");
  // These are the VIBEQC_DIRECT_METADATA fields, with the packer's fixed
  // three-term public AO expansion. No Cartesian quartet task table uploads.
  add(batch, sizeof(std::int64_t));
  add(1, sizeof(std::int64_t));  // terminal atom offset
  add(atoms, sizeof(std::int32_t) + 3 * sizeof(double));
  add(shells, sizeof(std::int32_t) + sizeof(std::uint8_t) + sizeof(std::int64_t));
  add(1, sizeof(std::int64_t));  // terminal primitive offset
  add(aos, sizeof(std::int32_t) + 10 * sizeof(std::uint8_t) + 3 * sizeof(double));
  add(primitives, 2 * sizeof(double));
  add(matrices, 6 * sizeof(double));
  if (derivative_order) add(atoms, 9 * sizeof(double));
  return bytes;
}

std::size_t cuda_direct_coulomb_device_bytes(std::size_t batch, std::size_t nao, std::size_t atoms,
                                             std::size_t shells, std::size_t primitives) {
  auto bytes = cuda_direct_jk_device_bytes(batch, nao, atoms, shells, primitives, 0);
  const auto add = [&](std::size_t n, std::size_t width) {
    bytes = runtime::size_add(bytes, runtime::size_mul(n, width));
  };
  // Shape-only upper bound for the optional spd Cartesian source. The public
  // spherical-to-Cartesian ratio is at most 6/5; 2 keeps integer admission
  // conservative without needing angular metadata or a geometry upload.
  const auto cart = runtime::size_mul(nao, 2);
  const auto public_elements = runtime::size_mul(batch, runtime::size_mul(nao, nao));
  const auto cart_elements = runtime::size_mul(batch, runtime::size_mul(cart, cart));
  add(cart_elements, 3 * sizeof(double));
  add(public_elements, 6 * sizeof(double));  // Two rectangular and two public matrices.
  add(runtime::size_mul(batch, cart), sizeof(std::int32_t) + 3 + sizeof(double));
  add(runtime::size_mul(shells, shells),
      3 * sizeof(std::int32_t) + sizeof(std::int64_t) + sizeof(std::uint32_t) + sizeof(double));
  add(runtime::size_mul(primitives, primitives), sizeof(cuda_execution::PrimitivePairData));
  add(shells, sizeof(std::int64_t));
  add(batch + 1, 2 * sizeof(std::int64_t) + 10 * sizeof(std::uint32_t));
  add(batch, sizeof(std::uint8_t));
  add(1, sizeof(cuda_execution::GeneratedShellPairStream) + 2 * sizeof(std::int64_t) +
             detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t));
  return bytes;
}

vibeqc_status create_cuda_direct_jk_plan(int device_id, const std::vector<core::System>& systems,
                                         unsigned derivative_order, double screening_tolerance,
                                         std::size_t budget, CudaDirectJkPlan** output,
                                         CudaDirectJkDiagnostic& diagnostic, std::string& detail) {
  if (output) *output = nullptr;
  diagnostic = {};
  return direct_jk_guard(nullptr, detail, [&] {
    direct_jk_require(output && device_id >= 0 && !systems.empty() && derivative_order <= 1 &&
                          std::isfinite(screening_tolerance) && screening_tolerance >= 0.0 &&
                          budget > 0,
                      "invalid direct J/K preparation request");
    const std::size_t coordinates = direct_jk_product(systems.front().atoms.size(), 3);
    direct_jk_require(coordinates > 0, "direct J/K source requires atoms");
    for (const auto& system : systems) {
      direct_jk_require(system.atoms.size() == systems.front().atoms.size(),
                        "direct J/K coordinate counts differ");
      direct_jk_require(system.basis_representation == VIBEQC_BASIS_CARTESIAN ||
                            system.basis_representation == VIBEQC_BASIS_SPHERICAL,
                        "unknown direct J/K AO representation");
      // Check direct C++ inputs before AO counting/packing or any CUDA work.
      for (const auto& atom : system.atoms)
        for (double coordinate : atom.position)
          direct_jk_require(std::isfinite(coordinate), "nonfinite direct J/K geometry");
      for (const auto& shell : system.shells) {
        direct_jk_require(shell.angular_momentum <= 3 && shell.atom_index < system.atoms.size() &&
                              !shell.primitives.empty(),
                          "invalid direct J/K shell");
        for (const auto& primitive : shell.primitives)
          direct_jk_require(std::isfinite(primitive.exponent) && primitive.exponent > 0.0 &&
                                std::isfinite(primitive.coefficient),
                            "invalid direct J/K primitive");
      }
    }
    HostBatch host;
    direct_jk_require(
        pack_host_batch(systems, std::vector<const std::vector<double>*>(systems.size()), host,
                        true),
        "direct J/K basis cannot be packed");
    const std::size_t matrix = direct_jk_product(host.nbf, host.nbf);
    (void)direct_jk_product(matrix, matrix);
    const std::size_t elements = direct_jk_product(matrix, systems.size());
    const std::size_t coord_elements = direct_jk_product(coordinates, systems.size());
    direct_jk_require(
        elements <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
            coord_elements <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "direct J/K launch dimensions exceed CUDA limits");
    std::size_t metadata = 0;
    auto count = [&](const auto& values) {
      const auto bytes = direct_jk_product(values.size(), sizeof(values[0]));
      if (!vibeqc::runtime::checked_add(metadata, bytes, metadata)) throw std::bad_alloc();
    };
#define VIBEQC_DIRECT_METADATA(F) \
  F(atom_offsets);                \
  F(atom_systems);                \
  F(positions);                   \
  F(shell_atoms);                 \
  F(shell_angular);               \
  F(shell_primitive_offsets);     \
  F(ao_shells);                   \
  F(ao_term_counts);              \
  F(ao_term_angular);             \
  F(ao_term_coefficients);        \
  F(primitive_exponents);         \
  F(primitive_coefficients)
#define VIBEQC_DIRECT_COUNT(field) count(host.field)
    VIBEQC_DIRECT_METADATA(VIBEQC_DIRECT_COUNT);
#undef VIBEQC_DIRECT_COUNT
    const auto matrix_bytes = direct_jk_product(elements, sizeof(double));
    const auto gradient_bytes =
        derivative_order ? direct_jk_product(direct_jk_product(coord_elements, 3), sizeof(double))
                         : 0;
    std::size_t required = direct_jk_product(matrix_bytes, 6);
    if (!vibeqc::runtime::checked_add(required, gradient_bytes, required) ||
        !vibeqc::runtime::checked_add(required, sizeof(int), required) ||
        !vibeqc::runtime::checked_add(required, metadata, required) || required > budget)
      throw std::bad_alloc();
    direct_jk_require(
        required == cuda_direct_jk_device_bytes(systems.size(), host.nbf,
                                                host.atomic_numbers.size(), host.shell_atoms.size(),
                                                host.primitive_exponents.size(), derivative_order),
        "direct J/K allocation inventory drifted from packed storage");
    direct_jk_check(cudaSetDevice(device_id));
    auto plan = std::make_unique<CudaDirectJkPlan>();
    plan->device_id = device_id;
    plan->derivative_order = derivative_order;
    plan->matrix_elements = elements;
    plan->coordinate_elements = coord_elements;
    plan->coordinates_per_item = coordinates;
    plan->screening_tolerance = screening_tolerance;
    plan->batch.batch_size = static_cast<std::int32_t>(systems.size());
    plan->batch.nbf = static_cast<std::int32_t>(host.nbf);
    plan->batch.direct_nbf = static_cast<std::int32_t>(host.direct_nbf);
    plan->batch.total_atoms = static_cast<std::int64_t>(host.atomic_numbers.size());
    plan->batch.total_shells = static_cast<std::int64_t>(host.shell_atoms.size());
    direct_jk_check(cudaStreamCreateWithFlags(&plan->stream, cudaStreamNonBlocking));
    auto upload = [&](const void* values, std::size_t bytes) -> void* {
      void* pointer{};
      const auto status = source_upload(*plan, values, bytes, &pointer, detail);
      if (status != VIBEQC_STATUS_SUCCESS) throw DirectJkFailure{status, detail};
      return pointer;
    };
#define VIBEQC_DIRECT_UPLOAD(field)                             \
  plan->batch.field = static_cast<decltype(plan->batch.field)>( \
      upload(host.field.data(), host.field.size() * sizeof(host.field[0])))
    VIBEQC_DIRECT_METADATA(VIBEQC_DIRECT_UPLOAD);
#undef VIBEQC_DIRECT_UPLOAD
#undef VIBEQC_DIRECT_METADATA
    // source_upload is a shared metadata helper; scratch has no host initializer.
    auto scratch = [&](std::size_t bytes) -> double* {
      if (!bytes) return nullptr;
      void* pointer{};
      direct_jk_check(runtime::resource_cuda_malloc(&pointer, bytes));
      try {
        plan->allocations.push_back(pointer);
      } catch (...) {
        runtime::resource_cuda_free(pointer);
        throw;
      }
      plan->device_bytes += bytes;
      return static_cast<double*>(pointer);
    };
    plan->density = scratch(matrix_bytes);
    plan->beta = scratch(matrix_bytes);
    plan->coulomb = scratch(matrix_bytes);
    plan->alpha_exchange = scratch(matrix_bytes);
    plan->beta_exchange = scratch(matrix_bytes);
    plan->bounds = scratch(matrix_bytes);
    plan->derivative = scratch(gradient_bytes);
    plan->numerical_failure = reinterpret_cast<int*>(scratch(sizeof(int)));
    direct_jk_check(cudaMemsetAsync(plan->numerical_failure, 0, sizeof(int), plan->stream));
    const auto blocks =
        static_cast<unsigned>((elements + kIndependentJkThreads - 1) / kIndependentJkThreads);
    launch_independent_jk_bounds_kernel(blocks, kIndependentJkThreads, 0, plan->stream, plan->batch,
                                        plan->bounds, plan->numerical_failure);
    direct_jk_check(cudaGetLastError());
    int numerical_failure = 0;
    direct_jk_check(cudaMemcpyAsync(&numerical_failure, plan->numerical_failure, sizeof(int),
                                    cudaMemcpyDeviceToHost, plan->stream));
    direct_jk_check(cudaStreamSynchronize(plan->stream));
    if (numerical_failure)
      throw DirectJkFailure{VIBEQC_STATUS_NUMERICAL_FAILURE, "nonfinite direct J/K Schwarz bound"};
    if (derivative_order == 0 && budget > required)
      plan->generated_coulomb = prepare_generated_coulomb(
          host, plan->batch, plan->stream, device_id, screening_tolerance, budget - required);
    auto& info = plan->diagnostic;
    info.batch_size = systems.size();
    info.nbf = host.nbf;
    info.coordinates_per_item = coordinates;
    info.device_bytes = plan->device_bytes;
    info.host_bytes = sizeof(*plan) + plan->allocations.capacity() * sizeof(void*);
    // HostBatch may reserve direct-HF queue metadata while reusing the common packer.
    // Include those temporary capacities even though this provider uploads only AO data.
    info.host_preparation_bytes =
        info.host_bytes + sizeof(host) +
        runtime::vector_capacities(
            host.atom_offsets, host.atom_systems, host.atomic_numbers, host.positions,
            host.system_shell_offsets, host.shell_atoms, host.shell_angular, host.shell_ao_offsets,
            host.shell_direct_ao_offsets, host.shell_primitive_offsets,
            host.system_shell_pair_offsets, host.system_shell_quartet_offsets,
            host.system_shell_pair_block_offsets, host.system_shell_pair_block_quartet_offsets,
            host.shell_pair_systems, host.shell_pair_first, host.shell_pair_second,
            host.shell_pair_primitive_offsets, host.psss_resident_tasks,
            host.psss_resident_ket_pairs, host.ao_shells, host.ao_term_counts, host.ao_term_angular,
            host.ao_term_coefficients, host.direct_ao_shells, host.direct_ao_angular,
            host.direct_ao_coefficients, host.ao_to_direct_transform, host.primitive_exponents,
            host.primitive_coefficients, host.occupied, host.warm_mask, host.warm_density);
    if (plan->generated_coulomb) {
      info.device_bytes += plan->generated_coulomb->device_bytes;
      info.host_bytes += sizeof(GeneratedCoulombPlan) +
                         runtime::vector_bytes(plan->generated_coulomb->allocations);
      info.host_preparation_bytes += plan->generated_coulomb->host_preparation_bytes;
      info.schedule = "generated-shell-coulomb/generic-jk-fallback";
    }
    info.derivative_order = derivative_order;
    info.screening_tolerance = screening_tolerance;
    diagnostic = info;
    *output = plan.release();
  });
}
void destroy_cuda_direct_jk_plan(CudaDirectJkPlan* plan) noexcept { delete plan; }

cudaStream_t cuda_direct_jk_stream(const CudaDirectJkPlan* plan) {
  direct_jk_require(plan != nullptr, "null direct J/K plan");
  return plan->stream;
}
int cuda_direct_jk_device(const CudaDirectJkPlan* plan) noexcept {
  return plan ? plan->device_id : -1;
}

static vibeqc_status enqueue_cuda_direct_jk_device_impl(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                                        const double* density, const double* beta,
                                                        std::size_t elements, double* coulomb,
                                                        double* alpha_exchange,
                                                        double* beta_exchange, int* numerical_error,
                                                        bool mixed_j, std::string& detail) {
  return direct_jk_guard(plan, detail, [&] {
    direct_jk_require(plan != nullptr, "null direct J/K plan");
    spec = direct_jk_strategy(plan, spec, 0, plan->diagnostic.batch_size);
    direct_jk_require(spec.derivative_order == 0 && elements == plan->matrix_elements,
                      "device direct J/K requires full-plan value dimensions");
    const bool unrestricted = spec.spin == FockSpin::Unrestricted;
    direct_jk_require(
        (unrestricted ? beta != nullptr : beta == nullptr) &&
            (spec.coulomb.present ? coulomb != nullptr : coulomb == nullptr) &&
            (spec.exchange.present ? alpha_exchange != nullptr : alpha_exchange == nullptr) &&
            (spec.exchange.present && unrestricted ? beta_exchange != nullptr
                                                   : beta_exchange == nullptr),
        "device direct J/K spin or selected-output mismatch");
    int current = -1;
    direct_jk_check(cudaGetDevice(&current));
    direct_jk_require(current == plan->device_id, "device direct J/K current device mismatch");
    const auto pointer = [&](const void* value) {
      direct_jk_require(value != nullptr, "null device direct J/K buffer");
      cudaPointerAttributes attributes{};
      direct_jk_check(cudaPointerGetAttributes(&attributes, value));
      direct_jk_require(attributes.type == cudaMemoryTypeDevice && attributes.device == current,
                        "direct J/K requires buffers on the current CUDA device");
    };
    pointer(density);
    pointer(numerical_error);
    if (unrestricted) pointer(beta);
    const auto bytes = direct_jk_product(elements, sizeof(double));
    const auto disjoint = [&](const void* a, std::size_t na, const void* b, std::size_t nb) {
      if (!a || !b) return;
      const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
      direct_jk_require(x <= std::numeric_limits<std::uintptr_t>::max() - na &&
                            y <= std::numeric_limits<std::uintptr_t>::max() - nb &&
                            (x + na <= y || y + nb <= x),
                        "device direct J/K writable buffers alias");
    };
    const double* inputs[]{density, beta};
    double* outputs[]{coulomb, alpha_exchange, beta_exchange};
    for (const auto* input : inputs) disjoint(input, bytes, numerical_error, sizeof(int));
    for (unsigned i = 0; i < 3; ++i) {
      if (!outputs[i]) continue;
      pointer(outputs[i]);
      for (const auto* input : inputs) disjoint(input, bytes, outputs[i], bytes);
      disjoint(outputs[i], bytes, numerical_error, sizeof(int));
      for (unsigned j = 0; j < i; ++j) disjoint(outputs[i], bytes, outputs[j], bytes);
    }
    direct_jk_check(cudaMemsetAsync(numerical_error, 0, sizeof(int), plan->stream));
    for (const auto* input : inputs)
      if (input) {
        launch_independent_jk_finite_kernel(plan->stream, input, elements, numerical_error);
        direct_jk_check(cudaGetLastError());
      }
    if (spec.coulomb.present || spec.exchange.present) {
      const auto dispatch = direct_jk_value_dispatch(
          plan->generated_coulomb != nullptr, spec.coulomb.present, spec.exchange.present, mixed_j);
      if (dispatch.generated_coulomb)
        direct_jk_check(
            enqueue_generated_coulomb(*plan->generated_coulomb, density, beta, coulomb));
      if (dispatch.generic_coulomb || dispatch.generic_exchange) {
        launch_independent_jk_kernel(
            static_cast<unsigned>(elements), kIndependentJkThreads, 0, plan->stream, plan->batch, 0,
            dispatch.generic_coulomb, dispatch.generic_exchange, unrestricted, mixed_j,
            direct_exchange_range(spec.exchange), spec.exchange.present ? spec.exchange.omega : 0.0,
            plan->screening_tolerance, plan->bounds, density, beta, coulomb, alpha_exchange,
            beta_exchange);
        direct_jk_check(cudaGetLastError());
      }
      for (const auto* output : outputs)
        if (output) {
          launch_independent_jk_finite_kernel(plan->stream, output, elements, numerical_error);
          direct_jk_check(cudaGetLastError());
        }
    }
  });
}

vibeqc_status enqueue_cuda_direct_jk_device(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                            const double* density, const double* beta,
                                            std::size_t elements, double* coulomb,
                                            double* alpha_exchange, double* beta_exchange,
                                            int* numerical_error, std::string& detail) {
  return enqueue_cuda_direct_jk_device_impl(plan, spec, density, beta, elements, coulomb,
                                            alpha_exchange, beta_exchange, numerical_error, false,
                                            detail);
}

vibeqc_status enqueue_cuda_direct_jk_device_mixed_j(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                                    const double* density, const double* beta,
                                                    std::size_t elements, double* coulomb,
                                                    double* alpha_exchange, double* beta_exchange,
                                                    int* numerical_error, std::string& detail) {
  return enqueue_cuda_direct_jk_device_impl(plan, spec, density, beta, elements, coulomb,
                                            alpha_exchange, beta_exchange, numerical_error, true,
                                            detail);
}

static vibeqc_status execute_cuda_direct_jk_range(
    CudaDirectJkPlan* plan, std::size_t begin, std::size_t count, FockBuildSpec spec,
    const std::vector<double>& density, const std::vector<double>& beta,
    std::vector<double>& coulomb, std::vector<double>& alpha_exchange,
    std::vector<double>& beta_exchange, std::string& detail) {
  return direct_jk_guard(plan, detail, [&] {
    spec = direct_jk_spec(plan, spec, density, beta, begin, count);
    const auto offset = begin * plan->diagnostic.nbf * plan->diagnostic.nbf;
    const bool unrestricted = spec.spin == FockSpin::Unrestricted;
    std::vector<double> j(spec.coulomb.present ? density.size() : 0);
    std::vector<double> ka(spec.exchange.present ? density.size() : 0);
    std::vector<double> kb(spec.exchange.present && unrestricted ? density.size() : 0);
    if (spec.coulomb.present || spec.exchange.present) {
      DirectJkDownloadFence fence{plan->stream};
      direct_jk_upload_density(*plan, density, beta, offset);
      launch_independent_jk_kernel(
          static_cast<unsigned>(density.size()), kIndependentJkThreads, 0, plan->stream,
          plan->batch, begin, spec.coulomb.present, spec.exchange.present, unrestricted, false,
          direct_exchange_range(spec.exchange), spec.exchange.present ? spec.exchange.omega : 0.0,
          plan->screening_tolerance, plan->bounds, plan->density, plan->beta, plan->coulomb,
          plan->alpha_exchange, plan->beta_exchange);
      direct_jk_check(cudaGetLastError());
      auto download = [&](std::vector<double>& out, const double* input) {
        if (!out.empty())
          direct_jk_check(cudaMemcpyAsync(out.data(), input + offset, out.size() * sizeof(double),
                                          cudaMemcpyDeviceToHost, plan->stream));
      };
      download(j, plan->coulomb);
      download(ka, plan->alpha_exchange);
      download(kb, plan->beta_exchange);
      fence.complete();
      direct_jk_finite_result(j);
      direct_jk_finite_result(ka);
      direct_jk_finite_result(kb);
    }
    coulomb = std::move(j);
    alpha_exchange = std::move(ka);
    beta_exchange = std::move(kb);
  });
}

static vibeqc_status execute_cuda_direct_energy_derivative_range(
    CudaDirectJkPlan* plan, std::size_t begin, std::size_t count, FockBuildSpec spec,
    const std::vector<double>& density, const std::vector<double>& beta,
    std::vector<double>& derivative, std::string& detail) {
  return direct_jk_guard(plan, detail, [&] {
    spec = direct_jk_spec(plan, spec, density, beta, begin, count);
    const auto offset = begin * plan->diagnostic.nbf * plan->diagnostic.nbf;
    direct_jk_require(spec.derivative_order == 1,
                      "direct J/K first derivatives were not requested");
    std::vector<double> result(count * plan->coordinates_per_item);
    const double cj = spec.coulomb.present ? spec.coulomb.coefficient : 0.0;
    const double ck = spec.exchange.present ? spec.exchange.coefficient : 0.0;
    if (cj != 0.0 || ck != 0.0) {
      DirectJkDownloadFence fence{plan->stream};
      direct_jk_upload_density(*plan, density, beta, offset);
      direct_jk_check(cudaMemsetAsync(plan->derivative + begin * plan->coordinates_per_item, 0,
                                      result.size() * sizeof(double), plan->stream));
      launch_independent_jk_derivative_kernel(
          static_cast<unsigned>(result.size()), kIndependentJkThreads, 0, plan->stream, plan->batch,
          plan->coordinates_per_item, begin, cj, ck, spec.spin == FockSpin::Unrestricted,
          direct_exchange_range(spec.exchange), spec.exchange.present ? spec.exchange.omega : 0.0,
          plan->screening_tolerance, plan->bounds, plan->density, plan->beta, plan->derivative);
      direct_jk_check(cudaGetLastError());
      direct_jk_check(
          cudaMemcpyAsync(result.data(), plan->derivative + begin * plan->coordinates_per_item,
                          result.size() * sizeof(double), cudaMemcpyDeviceToHost, plan->stream));
      fence.complete();
      direct_jk_finite_result(result);
    }
    derivative = std::move(result);
  });
}

CudaDirectJkDiagnostic cuda_direct_jk_plan_diagnostic(const CudaDirectJkPlan* plan) noexcept {
  return plan ? plan->diagnostic : CudaDirectJkDiagnostic{};
}
vibeqc_status execute_cuda_direct_jk(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                     const std::vector<double>& density,
                                     const std::vector<double>& beta, std::vector<double>& j,
                                     std::vector<double>& ka, std::vector<double>& kb,
                                     std::string& detail) {
  return execute_cuda_direct_jk_range(plan, 0, plan ? plan->diagnostic.batch_size : 0, spec,
                                      density, beta, j, ka, kb, detail);
}
vibeqc_status execute_cuda_direct_jk_item(CudaDirectJkPlan* plan, std::size_t item,
                                          FockBuildSpec spec, const std::vector<double>& density,
                                          const std::vector<double>& beta, std::vector<double>& j,
                                          std::vector<double>& ka, std::vector<double>& kb,
                                          std::string& detail) {
  return execute_cuda_direct_jk_range(plan, item, 1, spec, density, beta, j, ka, kb, detail);
}
vibeqc_status execute_cuda_direct_energy_derivative(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                                    const std::vector<double>& density,
                                                    const std::vector<double>& beta,
                                                    std::vector<double>& derivative,
                                                    std::string& detail) {
  return execute_cuda_direct_energy_derivative_range(
      plan, 0, plan ? plan->diagnostic.batch_size : 0, spec, density, beta, derivative, detail);
}
vibeqc_status execute_cuda_direct_energy_derivative_item(CudaDirectJkPlan* plan, std::size_t item,
                                                         FockBuildSpec spec,
                                                         const std::vector<double>& density,
                                                         const std::vector<double>& beta,
                                                         std::vector<double>& derivative,
                                                         std::string& detail) {
  return execute_cuda_direct_energy_derivative_range(plan, item, 1, spec, density, beta, derivative,
                                                     detail);
}

vibeqc_status execute_cuda_direct_rsh_energy_derivatives_item(
    CudaDirectJkPlan* plan, std::size_t item, FockSpin spin, double coulomb_coefficient,
    double short_exchange_coefficient, double long_exchange_coefficient, double omega,
    const std::vector<double>& density, const std::vector<double>& beta,
    std::vector<double>& derivatives, std::string& detail) {
  return direct_jk_guard(plan, detail, [&] {
    direct_jk_require(plan != nullptr && item < plan->diagnostic.batch_size,
                      "invalid fused RSH derivative item");
    direct_jk_require(
        std::isfinite(coulomb_coefficient) && std::isfinite(short_exchange_coefficient) &&
            std::isfinite(long_exchange_coefficient) && std::isfinite(omega) && omega >= 0.0,
        "nonfinite fused RSH derivative coefficient");
    FockBuildSpec spec;
    spec.spin = spin;
    spec.derivative_order = 1;
    spec.coulomb = {true, coulomb_coefficient};
    spec.exchange = {true, short_exchange_coefficient, FockOperator::ShortRange, omega,
                     FockApproximation::Exact};
    spec = direct_jk_spec(plan, spec, density, beta, item, 1);
    direct_jk_require(spec.derivative_order == 1,
                      "direct J/K first derivatives were not requested");
    const std::size_t n = plan->diagnostic.nbf;
    for (const auto* spin_density : {&density, &beta}) {
      if (spin_density->empty()) continue;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
          direct_jk_require(
              std::abs((*spin_density)[i * n + j] - (*spin_density)[j * n + i]) <= 1e-10,
              "symmetry-reduced RSH derivatives require symmetric densities");
    }

    const std::size_t coordinates = plan->coordinates_per_item;
    const std::size_t coordinate_offset = item * coordinates;
    const std::size_t matrix_offset = item * plan->diagnostic.nbf * plan->diagnostic.nbf;
    std::vector<double> result(3 * coordinates);
    if (coulomb_coefficient != 0.0 || short_exchange_coefficient != 0.0 ||
        long_exchange_coefficient != 0.0) {
      DirectJkDownloadFence fence{plan->stream};
      direct_jk_upload_density(*plan, density, beta, matrix_offset);
      for (unsigned source = 0; source < 3; ++source)
        direct_jk_check(cudaMemsetAsync(
            plan->derivative + source * plan->coordinate_elements + coordinate_offset, 0,
            coordinates * sizeof(double), plan->stream));
      launch_independent_rsh_derivative_kernel(
          static_cast<unsigned>(coordinates), kIndependentJkThreads, 0, plan->stream, plan->batch,
          coordinates, item, plan->coordinate_elements, coulomb_coefficient,
          short_exchange_coefficient, long_exchange_coefficient,
          spec.spin == FockSpin::Unrestricted, omega, plan->screening_tolerance, plan->bounds,
          plan->density, plan->beta, plan->derivative);
      direct_jk_check(cudaGetLastError());
      for (unsigned source = 0; source < 3; ++source)
        direct_jk_check(cudaMemcpyAsync(
            result.data() + source * coordinates,
            plan->derivative + source * plan->coordinate_elements + coordinate_offset,
            coordinates * sizeof(double), cudaMemcpyDeviceToHost, plan->stream));
      fence.complete();
      direct_jk_finite_result(result);
    }
    derivatives = std::move(result);
  });
}

}  // namespace vibeqc::scf

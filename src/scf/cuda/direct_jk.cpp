#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/basis_transform_kernels.hpp"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_jk_plan.hpp"
#include "scf/cuda/metadata_upload.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/direct_task_layout.hpp"

namespace generativeqc::scf {

namespace {
using namespace cuda_execution;
}

CudaDirectJkPlan::~CudaDirectJkPlan() {
  if (device_id >= 0) (void)cudaSetDevice(device_id);
  if (stream) (void)cudaStreamSynchronize(stream);
  generated_exchange.reset();
  generated_coulomb.reset();  // Release borrowers before their stream/metadata.
  for (void* pointer : allocations) (void)runtime::resource_cuda_free(pointer);
  if (stream) (void)cudaStreamDestroy(stream);
}

namespace {

struct DirectJkFailure {
  generativeqc_status status;
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
  if (!generativeqc::runtime::checked_multiply(a, b, out)) throw std::bad_alloc();
  return out;
}
void direct_jk_require_disjoint(const void* a, std::size_t na, const void* b, std::size_t nb) {
  if (!a || !b) return;
  const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
  direct_jk_require(x <= std::numeric_limits<std::uintptr_t>::max() - na &&
                        y <= std::numeric_limits<std::uintptr_t>::max() - nb &&
                        (x + na <= y || y + nb <= x),
                    "device direct J/K writable buffers alias");
}
void direct_jk_finite(const std::vector<double>& values) {
  for (double value : values) direct_jk_require(std::isfinite(value), "nonfinite direct J/K data");
}
void direct_jk_finite_result(const std::vector<double>& values) {
  for (double value : values)
    if (!std::isfinite(value))
      throw DirectJkFailure{GENERATIVEQC_STATUS_NUMERICAL_FAILURE, "nonfinite direct J/K result"};
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
/** Roll back an optional allocation group without releasing an earlier owner.
 * Preparation must fence its local upload/download staging before unwinding.
 * The second fence here also checks asynchronous errors before any device free.
 */
template <class Prepare, class Restore>
void direct_jk_optional_storage(CudaDirectJkPlan& plan, bool optional, std::string& detail,
                                Prepare prepare, Restore restore,
                                const int* numerical_failure = nullptr) {
  const auto retained_allocations = plan.allocations.size();
  const auto retained_bytes = plan.device_bytes;
  try {
    prepare();
    return;
  } catch (const DirectJkFailure& failure) {
    if (!optional || failure.status != GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw;
  } catch (const std::bad_alloc&) {
    if (!optional) throw;
  } catch (cudaError_t error) {
    if (!optional || error != cudaErrorMemoryAllocation) throw;
  }
  direct_jk_check(cudaStreamSynchronize(plan.stream));
  // A queued bound-status download can complete only while the failed
  // preparation unwinds. Allocation fallback must not hide that earlier fault.
  if (numerical_failure && *numerical_failure)
    throw DirectJkFailure{GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                          "nonfinite canonical Cartesian Schwarz bound"};
  while (plan.allocations.size() > retained_allocations) {
    direct_jk_check(runtime::resource_cuda_free(plan.allocations.back()));
    plan.allocations.pop_back();
  }
  plan.device_bytes = retained_bytes;
  restore();
  const auto error = cudaGetLastError();
  if (error != cudaErrorMemoryAllocation) direct_jk_check(error);
  detail.clear();
}
/** Always fence failed uploads too: caller-owned pageable buffers may die on return. */
template <class Function>
generativeqc_status direct_jk_guard(CudaDirectJkPlan* plan, std::string& detail,
                                    Function function) {
  detail.clear();
  try {
    function();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const DirectJkFailure& failure) {
    if (plan && plan->stream) (void)cudaStreamSynchronize(plan->stream);
    detail = failure.detail;
    return failure.status;
  } catch (const std::bad_alloc&) {
    if (plan && plan->stream) (void)cudaStreamSynchronize(plan->stream);
    detail = "direct J/K allocation exceeds available capacity";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (cudaError_t error) {
    if (plan && plan->stream) (void)cudaStreamSynchronize(plan->stream);
    detail = cudaGetErrorString(error);
    return source_cuda_status(error);
  } catch (const std::exception& error) {
    if (plan && plan->stream) (void)cudaStreamSynchronize(plan->stream);
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
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
    if (!generativeqc::runtime::checked_add(bytes, term, bytes)) throw std::bad_alloc();
  };
  const auto aos = direct_jk_product(batch, nao);
  const auto matrices = direct_jk_product(aos, nao);
  direct_jk_require(matrices <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
                        atoms <= static_cast<std::size_t>(std::numeric_limits<int>::max()) / 3,
                    "direct J/K resource shape exceeds launch dimensions");
  // These are the GENERATIVEQC_DIRECT_METADATA fields, with the packer's fixed
  // three-term public AO expansion. No Cartesian quartet task table uploads.
  add(batch, sizeof(std::int64_t));
  add(1, sizeof(std::int64_t));  // terminal atom offset
  // atom_systems + atomic_numbers + Cartesian positions.
  add(atoms, 2 * sizeof(std::int32_t) + 3 * sizeof(double));
  add(shells, sizeof(std::int32_t) + sizeof(std::uint8_t) + sizeof(std::int64_t));
  add(1, sizeof(std::int64_t));  // terminal primitive offset
  add(aos, sizeof(std::int32_t) + 10 * sizeof(std::uint8_t) + 3 * sizeof(double));
  add(primitives, 2 * sizeof(double));
  add(matrices, 6 * sizeof(double));
  if (derivative_order) add(atoms, 9 * sizeof(double));
  return bytes;
}

std::size_t cuda_direct_coulomb_device_bytes(std::size_t batch, std::size_t nao, std::size_t atoms,
                                             std::size_t shells, std::size_t primitives,
                                             unsigned derivative_order) {
  auto bytes = cuda_direct_jk_device_bytes(batch, nao, atoms, shells, primitives, derivative_order);
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
  // Generated shell topology also retains public shell AO offsets so
  // one-electron derivatives can borrow the same immutable geometry owner.
  add(shells + 1, sizeof(std::int64_t));
  add(batch + 1, 2 * sizeof(std::int64_t) + 10 * sizeof(std::uint32_t));
  add(batch, sizeof(std::uint8_t));
  add(1, sizeof(cuda_execution::GeneratedShellPairStream) + 2 * sizeof(std::int64_t) +
             detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t));
  if (derivative_order) {
    // Retain the generated full-range exchange owner as a stationary shell
    // derivative lease. Bounds use total shell counts as conservative shape-only
    // envelopes; no topology enumeration occurs in this query.
    const auto pairs = runtime::size_mul(shells, shells);
    const auto rectangular = runtime::size_mul(batch, runtime::size_mul(nao, cart));
    add(cart_elements, 4 * sizeof(double));
    add(public_elements, 4 * sizeof(double));
    add(rectangular, 4 * sizeof(double));
    add(pairs, 3 * sizeof(double) + sizeof(std::uint32_t) + sizeof(double));
    add(batch, 11 * sizeof(double));
    add(detail::kDirectQuartetShellClassCount, sizeof(std::uint32_t));
    add(atoms, 9 * sizeof(double));
    add(batch + 1, 2 * sizeof(std::int64_t));
    add(1, sizeof(cuda_execution::GeneratedShellPairStream) + sizeof(unsigned long long));
  }
  return bytes;
}

generativeqc_status create_cuda_direct_jk_plan(
    int device_id, const std::vector<core::System>& systems, unsigned derivative_order,
    double screening_tolerance, std::size_t budget, CudaDirectJkPlan** output,
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
      direct_jk_require(system.basis_representation == GENERATIVEQC_BASIS_CARTESIAN ||
                            system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL,
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
    // This provider uploads AO/shell data and constructs its own generated or
    // bounded shell schedules. Building Direct-HF's unused PSSS catalog here
    // adds quartic setup work; skipping it must retain the direct AO transform.
    direct_jk_require(
        pack_host_batch(systems, std::vector<const std::vector<double>*>(systems.size()), host,
                        true, false, true, ResidentPsssPolicy::Skip),
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
      if (!generativeqc::runtime::checked_add(metadata, bytes, metadata)) throw std::bad_alloc();
    };
#define GENERATIVEQC_DIRECT_METADATA(F) \
  F(atom_offsets);                      \
  F(atom_systems);                      \
  F(atomic_numbers);                    \
  F(positions);                         \
  F(shell_atoms);                       \
  F(shell_angular);                     \
  F(shell_primitive_offsets);           \
  F(ao_shells);                         \
  F(ao_term_counts);                    \
  F(ao_term_angular);                   \
  F(ao_term_coefficients);              \
  F(primitive_exponents);               \
  F(primitive_coefficients)
#define GENERATIVEQC_DIRECT_COUNT(field) count(host.field)
    GENERATIVEQC_DIRECT_METADATA(GENERATIVEQC_DIRECT_COUNT);
#undef GENERATIVEQC_DIRECT_COUNT
    const auto matrix_bytes = direct_jk_product(elements, sizeof(double));
    const auto gradient_bytes =
        derivative_order ? direct_jk_product(direct_jk_product(coord_elements, 3), sizeof(double))
                         : 0;
    std::size_t required = direct_jk_product(matrix_bytes, 6);
    if (!generativeqc::runtime::checked_add(required, gradient_bytes, required) ||
        !generativeqc::runtime::checked_add(required, sizeof(int), required) ||
        !generativeqc::runtime::checked_add(required, metadata, required) || required > budget)
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
      if (status != GENERATIVEQC_STATUS_SUCCESS) throw DirectJkFailure{status, detail};
      return pointer;
    };
#define GENERATIVEQC_DIRECT_UPLOAD(field)                       \
  plan->batch.field = static_cast<decltype(plan->batch.field)>( \
      upload(host.field.data(), host.field.size() * sizeof(host.field[0])))
    GENERATIVEQC_DIRECT_METADATA(GENERATIVEQC_DIRECT_UPLOAD);
#undef GENERATIVEQC_DIRECT_UPLOAD
#undef GENERATIVEQC_DIRECT_METADATA
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
      throw DirectJkFailure{GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                            "nonfinite direct J/K Schwarz bound"};
    // Derivative capability is orthogonal to the value schedule. When budgeted,
    // retain the same generated full-range exchange owner for stationary shell
    // derivatives; value-only plans keep its force scratch disabled.
    // Full-range SPD coverage does not cover SR/LR exchange. Reserve its
    // existing generated owner first, then use only the remaining budget for
    // canonical range sources. This keeps an optional range acceleration from
    // displacing the established full-range value/force provider.
    const bool through_f = std::any_of(host.shell_angular.begin(), host.shell_angular.end(),
                                       [](auto angular) { return angular == 3; });
    if (!through_f && budget > plan->device_bytes) {
      const auto optional_budget = budget - plan->device_bytes;
      plan->generated_exchange = prepare_generated_exchange(
          host, plan->batch, plan->stream, device_id, screening_tolerance, optional_budget,
          derivative_order != 0, false);
      if (!plan->generated_exchange)
        plan->generated_coulomb = prepare_generated_coulomb(
            host, plan->batch, plan->stream, device_id, screening_tolerance, optional_budget);
      // Generated owners account for their own allocations; diagnostic totals
      // add them below. Reduce this local allowance without double charging.
      if (plan->generated_exchange)
        budget -= plan->generated_exchange->device_bytes;
      else if (plan->generated_coulomb)
        budget -= plan->generated_coulomb->device_bytes;
    }
    std::size_t span_host_preparation_bytes = 0;
    std::size_t canonical_host_preparation_bytes = 0;
    const auto prepare_canonical = [&] {
      const auto cart_elements =
          direct_jk_product(systems.size(), direct_jk_product(host.direct_nbf, host.direct_nbf));
      const auto cart_matrix_bytes = direct_jk_product(cart_elements, sizeof(double));
      const auto rectangular_elements =
          direct_jk_product(systems.size(), direct_jk_product(host.nbf, host.direct_nbf));
      const auto rectangular_bytes = direct_jk_product(rectangular_elements, sizeof(double));
      const auto cart_pairs = direct_jk_product(
          systems.size(), direct_jk_product(host.direct_nbf, host.direct_nbf + 1U) / 2U);
      const auto cart_metadata_bytes =
          direct_jk_product(systems.size() * host.direct_nbf,
                            sizeof(std::int32_t) + 3U * sizeof(std::uint8_t) + sizeof(double));
      const auto cart_shell_bytes =
          direct_jk_product(host.shell_direct_ao_offsets.size(), sizeof(std::int64_t));
      const auto projection_span_bytes = direct_jk_product(
          direct_jk_product(systems.size(), host.nbf + host.direct_nbf), 2U * sizeof(std::int32_t));
      const bool needs_projection = host.nbf != host.direct_nbf;
      const auto projection_bytes =
          needs_projection
              ? runtime::size_add(
                    direct_jk_product(matrix_bytes, 5U),
                    runtime::size_add(direct_jk_product(rectangular_bytes, 3U),
                                      runtime::size_add(projection_span_bytes, systems.size())))
              : 0U;
      const auto cart_extra = runtime::size_add(
          cart_matrix_bytes,
          runtime::size_add(projection_bytes,
                            runtime::size_add(cart_metadata_bytes, cart_shell_bytes)));
      const auto cart_optional = runtime::size_add(
          cart_extra, runtime::size_add(direct_jk_product(cart_pairs, 2U * sizeof(std::int32_t)),
                                        direct_jk_product(cart_matrix_bytes, 6U)));
      plan->canonical_cartesian =
          cart_elements <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
          cart_optional <= budget - plan->device_bytes;
      const auto source_dimension = plan->canonical_cartesian ? host.direct_nbf : host.nbf;
      const auto source_matrix_bytes = plan->canonical_cartesian ? cart_matrix_bytes : matrix_bytes;
      const auto& source_shells =
          plan->canonical_cartesian ? host.direct_ao_shells : host.ao_shells;
      const auto pairs_per_item = direct_jk_product(source_dimension, source_dimension + 1U) / 2U;
      const auto pairs_count = direct_jk_product(systems.size(), pairs_per_item);
      const auto pairs_bytes = direct_jk_product(pairs_count, 2U * sizeof(std::int32_t));
      const auto scratch_bytes = direct_jk_product(source_matrix_bytes, 6U);
      const auto optional_bytes = runtime::size_add(
          pairs_bytes,
          runtime::size_add(scratch_bytes, plan->canonical_cartesian ? cart_extra : 0U));
      if (optional_bytes <= budget - plan->device_bytes) {
        std::vector<std::int32_t> pairs;
        pairs.reserve(direct_jk_product(pairs_count, 2U));
        canonical_host_preparation_bytes = sizeof(pairs) + runtime::vector_bytes(pairs);
        plan->canonical_pair_offsets.resize(systems.size());
        for (std::size_t item = 0; item < systems.size(); ++item) {
          auto& offsets = plan->canonical_pair_offsets[item];
          for (unsigned order = 0; order < 7U; ++order) {
            offsets[order] = pairs.size() / 2U;
            for (std::size_t first = 0; first < source_dimension; ++first)
              for (std::size_t second = 0; second <= first; ++second) {
                const auto first_shell = source_shells[item * source_dimension + first];
                const auto second_shell = source_shells[item * source_dimension + second];
                if (host.shell_angular[first_shell] + host.shell_angular[second_shell] != order)
                  continue;
                pairs.push_back(static_cast<std::int32_t>(first));
                pairs.push_back(static_cast<std::int32_t>(second));
              }
          }
          offsets[7] = pairs.size() / 2U;
        }
        direct_jk_require(pairs.size() == pairs_count * 2U,
                          "canonical J/K source-AO pair inventory drift");
        std::vector<std::int32_t> projection_spans;
        DirectJkDownloadFence staging_fence{plan->stream};
        plan->canonical_batch = plan->batch;
        plan->canonical_bounds = plan->bounds;
        if (plan->canonical_cartesian) {
          auto& source = plan->canonical_batch;
          source.nbf = static_cast<std::int32_t>(source_dimension);
          source.shell_direct_ao_offsets = static_cast<const std::int64_t*>(
              upload(host.shell_direct_ao_offsets.data(), cart_shell_bytes));
          source.direct_ao_shells = static_cast<const std::int32_t*>(upload(
              host.direct_ao_shells.data(), host.direct_ao_shells.size() * sizeof(std::int32_t)));
          source.direct_ao_angular = static_cast<const std::uint8_t*>(upload(
              host.direct_ao_angular.data(), host.direct_ao_angular.size() * sizeof(std::uint8_t)));
          source.direct_ao_coefficients = static_cast<const double*>(
              upload(host.direct_ao_coefficients.data(),
                     host.direct_ao_coefficients.size() * sizeof(double)));
          source.ao_shells = source.direct_ao_shells;
          if (needs_projection) {
            direct_jk_require(host.ao_to_direct_transform.size() == rectangular_elements,
                              "Cartesian source requires a complete HF projection transform");
            plan->canonical_transform = static_cast<const double*>(
                upload(host.ao_to_direct_transform.data(), rectangular_bytes));
            projection_spans.reserve(projection_span_bytes / sizeof(std::int32_t));
            span_host_preparation_bytes +=
                sizeof(projection_spans) + runtime::vector_bytes(projection_spans);
            for (bool public_rows : {true, false}) {
              const auto dimension = public_rows ? host.nbf : host.direct_nbf;
              const auto target_dimension = public_rows ? host.direct_nbf : host.nbf;
              const auto& ao_shells = public_rows ? host.ao_shells : host.direct_ao_shells;
              const auto& shell_offsets =
                  public_rows ? host.shell_direct_ao_offsets : host.shell_ao_offsets;
              for (std::size_t item = 0; item < systems.size(); ++item)
                for (std::size_t ao = 0; ao < dimension; ++ao) {
                  const auto shell = ao_shells[item * dimension + ao];
                  const auto begin = shell_offsets[shell] - item * target_dimension;
                  const auto end = shell_offsets[shell + 1U] - item * target_dimension;
                  direct_jk_require(begin < end && end <= target_dimension,
                                    "invalid shell-local canonical projection span");
                  projection_spans.push_back(static_cast<std::int32_t>(begin));
                  projection_spans.push_back(static_cast<std::int32_t>(end));
                }
            }
            plan->canonical_projection_spans = static_cast<const std::int32_t*>(
                upload(projection_spans.data(), projection_span_bytes));
            plan->canonical_active = reinterpret_cast<std::uint8_t*>(scratch(systems.size()));
            plan->canonical_public_density = scratch(direct_jk_product(matrix_bytes, 2U));
            plan->canonical_public_output = scratch(direct_jk_product(matrix_bytes, 2U));
            plan->canonical_projection = scratch(direct_jk_product(rectangular_bytes, 2U));
            plan->canonical_zero = scratch(matrix_bytes);
            direct_jk_check(
                cudaMemsetAsync(plan->canonical_active, 1, systems.size(), plan->stream));
            direct_jk_check(cudaMemsetAsync(plan->canonical_zero, 0, matrix_bytes, plan->stream));
          }
          plan->canonical_bounds = scratch(source_matrix_bytes);
          launch_independent_jk_bounds_kernel(
              static_cast<unsigned>((cart_elements + kIndependentJkThreads - 1U) /
                                    kIndependentJkThreads),
              kIndependentJkThreads, 0, plan->stream, source, plan->canonical_bounds,
              plan->numerical_failure, true);
          direct_jk_check(cudaGetLastError());
          direct_jk_check(cudaMemcpyAsync(&numerical_failure, plan->numerical_failure, sizeof(int),
                                          cudaMemcpyDeviceToHost, plan->stream));
        }
        plan->canonical_pairs = static_cast<const std::int32_t*>(upload(pairs.data(), pairs_bytes));
        plan->canonical_density = scratch(direct_jk_product(source_matrix_bytes, 2U));
        plan->canonical_coulomb = scratch(direct_jk_product(source_matrix_bytes, 2U));
        plan->canonical_exchange = scratch(direct_jk_product(source_matrix_bytes, 2U));
        staging_fence.complete();
        if (numerical_failure)
          throw DirectJkFailure{GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                                "nonfinite canonical Cartesian Schwarz bound"};

        // Sort/scan capacity is optional independently of canonical contraction.
        // Oversized CUB inventories and constrained budgets keep the dense path.
        if (pairs_count <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
            systems.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max() / 7)) {
          direct_jk_optional_storage(
              *plan, !through_f, detail,
              [&] {
                std::vector<int> segment_offsets;
                segment_offsets.reserve(systems.size() * 7U + 1U);
                std::size_t largest_bucket = 0;
                for (const auto& offsets : plan->canonical_pair_offsets)
                  for (unsigned order = 0; order < 7U; ++order) {
                    segment_offsets.push_back(static_cast<int>(offsets[order]));
                    largest_bucket = std::max(largest_bucket, offsets[order + 1U] - offsets[order]);
                  }
                segment_offsets.push_back(static_cast<int>(pairs_count));
                span_host_preparation_bytes +=
                    sizeof(segment_offsets) + runtime::vector_bytes(segment_offsets);
                std::size_t workspace_bytes = 0;
                direct_jk_check(canonical_pair_workspace(
                    static_cast<int>(pairs_count), static_cast<int>(systems.size() * 7U),
                    static_cast<int>(largest_bucket), workspace_bytes));
                const auto row_bytes = direct_jk_product(pairs_count, 7U * sizeof(std::uint64_t));
                const auto key_bytes = direct_jk_product(pairs_count, sizeof(double));
                const auto order_bytes = direct_jk_product(pairs_count, sizeof(std::int32_t));
                const auto offset_bytes = direct_jk_product(segment_offsets.size(), sizeof(int));
                const auto span_bytes = runtime::size_add(
                    runtime::size_add(row_bytes, workspace_bytes),
                    runtime::size_add(
                        offset_bytes,
                        direct_jk_product(runtime::size_add(key_bytes, order_bytes), 2U)));
                if (span_bytes <= budget - plan->device_bytes) {
                  DirectJkDownloadFence sort_fence{plan->stream};
                  const auto* offsets =
                      static_cast<const int*>(upload(segment_offsets.data(), offset_bytes));
                  auto* input_keys = scratch(key_bytes);
                  auto* sorted_keys = scratch(key_bytes);
                  auto* input_order = reinterpret_cast<std::int32_t*>(scratch(order_bytes));
                  auto* sorted_order = reinterpret_cast<std::int32_t*>(scratch(order_bytes));
                  auto* prefix = reinterpret_cast<std::uint64_t*>(scratch(row_bytes));
                  auto* workspace = scratch(workspace_bytes);
                  direct_jk_check(prepare_canonical_pair_order(
                      plan->stream, plan->canonical_batch, plan->canonical_pairs,
                      plan->canonical_bounds, static_cast<int>(pairs_count),
                      static_cast<int>(systems.size() * 7U), offsets, input_keys, input_order,
                      sorted_keys, sorted_order, workspace, workspace_bytes));
                  for (const auto& item_offsets : plan->canonical_pair_offsets)
                    for (unsigned first = 0; first < 7U; ++first)
                      for (unsigned second = 0; second <= first; ++second)
                        direct_jk_check(prepare_canonical_pair_rows(
                            plan->stream, sorted_keys, item_offsets[first],
                            item_offsets[first + 1U] - item_offsets[first], item_offsets[second],
                            item_offsets[second + 1U] - item_offsets[second], first == second,
                            screening_tolerance,
                            prefix + second * pairs_count + item_offsets[first], workspace,
                            workspace_bytes));
                  sort_fence.complete();
                  plan->canonical_pair_order = sorted_order;
                  plan->canonical_row_prefix = prefix;
                }
              },
              [&] {
                plan->canonical_pair_order = nullptr;
                plan->canonical_row_prefix = nullptr;
              });
        }
      }
    };
    direct_jk_optional_storage(
        *plan, !through_f, detail, prepare_canonical,
        [&] {
          plan->canonical_pairs = nullptr;
          plan->canonical_cartesian = false;
          plan->canonical_batch = {};
          plan->canonical_bounds = nullptr;
          plan->canonical_transform = nullptr;
          plan->canonical_projection_spans = nullptr;
          plan->canonical_active = nullptr;
          plan->canonical_public_density = nullptr;
          plan->canonical_public_output = nullptr;
          plan->canonical_projection = nullptr;
          plan->canonical_zero = nullptr;
          plan->canonical_pair_offsets.clear();
          plan->canonical_pair_order = nullptr;
          plan->canonical_row_prefix = nullptr;
          plan->canonical_density = nullptr;
          plan->canonical_coulomb = nullptr;
          plan->canonical_exchange = nullptr;
        },
        &numerical_failure);
    if (plan->canonical_pairs && derivative_order) {
      const auto retained_batch = plan->batch;
      direct_jk_optional_storage(
          *plan, !through_f, detail,
          [&] {
            // HF's shell-warp one-electron consumer needs only three topology arrays.
            // Borrow the already packed public AO data and the retained force scratch,
            // not a second generated exchange owner or per-call metadata upload.
            const auto offsets_bytes =
                direct_jk_product(host.shell_ao_offsets.size(), sizeof(std::int64_t));
            const auto pair_bytes =
                direct_jk_product(host.shell_pair_first.size(), sizeof(std::int32_t));
            const auto one_electron_bytes =
                runtime::size_add(offsets_bytes, direct_jk_product(pair_bytes, 2U));
            if (one_electron_bytes <= budget - plan->device_bytes) {
              DirectJkDownloadFence metadata_fence{plan->stream};
              plan->batch.shell_ao_offsets = static_cast<const std::int64_t*>(
                  upload(host.shell_ao_offsets.data(), offsets_bytes));
              plan->batch.shell_pair_first = static_cast<const std::int32_t*>(
                  upload(host.shell_pair_first.data(), pair_bytes));
              plan->batch.shell_pair_second = static_cast<const std::int32_t*>(
                  upload(host.shell_pair_second.data(), pair_bytes));
              plan->batch.total_shell_pairs =
                  static_cast<std::int64_t>(host.shell_pair_first.size());
              metadata_fence.complete();
            }
          },
          [&] { plan->batch = retained_batch; });
    }
    if (through_f && budget > plan->device_bytes) {
      const auto optional_budget = budget - plan->device_bytes;
      // Generated/native streaming owns its qualified classes; through-f plans
      // retain HF's bounded shell dispatcher for the remaining higher-l value
      // classes and, when requested, for stationary derivatives.
      const bool bounded_through_f = through_f;
      plan->generated_exchange = prepare_generated_exchange(
          host, plan->batch, plan->stream, device_id, screening_tolerance, optional_budget,
          derivative_order != 0, bounded_through_f);
      if (!plan->generated_exchange)
        plan->generated_coulomb = prepare_generated_coulomb(
            host, plan->batch, plan->stream, device_id, screening_tolerance, optional_budget);
    }
    auto& info = plan->diagnostic;
    info.batch_size = systems.size();
    info.nbf = host.nbf;
    info.coordinates_per_item = coordinates;
    info.device_bytes = plan->device_bytes;
    info.host_bytes = sizeof(*plan) + plan->allocations.capacity() * sizeof(void*) +
                      runtime::vector_capacities(plan->canonical_pair_offsets);
    // HostBatch may reserve direct-HF queue metadata while reusing the common packer.
    // Include those temporary capacities even though this provider uploads only AO data.
    info.host_preparation_bytes =
        info.host_bytes + sizeof(host) + span_host_preparation_bytes +
        canonical_host_preparation_bytes +
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
    if (plan->generated_exchange) {
      info.device_bytes += plan->generated_exchange->device_bytes;
      info.host_bytes += sizeof(GeneratedExchangePlan) +
                         runtime::vector_bytes(plan->generated_exchange->allocations) +
                         sizeof(GeneratedCoulombPlan) +
                         runtime::vector_bytes(plan->generated_exchange->shared->allocations);
      info.host_preparation_bytes += plan->generated_exchange->host_preparation_bytes;
      info.schedule =
          plan->generated_exchange->shared->value_capability
              ? (plan->canonical_pairs ? "generated-shell-coulomb+exchange/canonical-range-fallback"
                                       : "generated-shell-coulomb+exchange/generic-jk-fallback")
              : (direct_jk_bounded_value_enabled(*plan)
                     ? "generated-shell+bounded-through-f-jk/canonical-range-fallback"
                     : "retained-shell-owner/generic-jk-fallback");
    } else if (plan->generated_coulomb) {
      info.device_bytes += plan->generated_coulomb->device_bytes;
      info.host_bytes += sizeof(GeneratedCoulombPlan) +
                         runtime::vector_bytes(plan->generated_coulomb->allocations);
      info.host_preparation_bytes += plan->generated_coulomb->host_preparation_bytes;
      info.schedule = "generated-shell-coulomb/generic-jk-fallback";
    }
    if (plan->canonical_pairs && !direct_jk_generated_full_range_value_available(*plan))
      info.schedule = plan->canonical_cartesian
                          ? (plan->canonical_row_prefix
                                 ? "canonical-cartesian-jk/screened-rows/automatic"
                                 : "canonical-cartesian-jk/dense-angular-bucketed/automatic")
                          : (plan->canonical_row_prefix
                                 ? "canonical-public-ao-jk/screened-rows/automatic"
                                 : "canonical-public-ao-jk/dense-angular-bucketed/automatic");
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

/** Each ket bucket owns one O(P) page; each bra bucket's scan starts at zero. */
static CanonicalPairRows direct_jk_pair_rows(const CudaDirectJkPlan* plan, unsigned second,
                                             std::size_t first_begin) {
  if (!plan->canonical_row_prefix) return {};
  const auto pairs = plan->canonical_pair_offsets.back()[7];
  return {plan->canonical_pair_order, plan->canonical_row_prefix + second * pairs + first_begin};
}

/** Stage public spin blocks and reuse HF's D_cart = C^T D_public C projection.
 * An item-local force request must not read uninitialized densities elsewhere
 * in a batch. The active mask is restored for each request on the owning stream.
 */
static void direct_jk_canonical_density(CudaDirectJkPlan* plan, bool unrestricted,
                                        const double* density, const double* beta,
                                        std::size_t begin, std::size_t count) {
  const auto dimension = plan->diagnostic.nbf;
  const auto matrix = direct_jk_product(dimension, dimension);
  const auto bytes = direct_jk_product(matrix, sizeof(double));
  const auto spin_count = unrestricted ? 2U : 1U;
  auto* staged =
      plan->canonical_transform ? plan->canonical_public_density : plan->canonical_density;
  for (std::size_t item = begin; item < begin + count; ++item) {
    direct_jk_check(cudaMemcpyAsync(staged + item * spin_count * matrix, density + item * matrix,
                                    bytes, cudaMemcpyDeviceToDevice, plan->stream));
    if (unrestricted)
      direct_jk_check(cudaMemcpyAsync(staged + (item * spin_count + 1U) * matrix,
                                      beta + item * matrix, bytes, cudaMemcpyDeviceToDevice,
                                      plan->stream));
  }
  if (!plan->canonical_transform) return;
  const auto batch_size = plan->batch.batch_size;
  const auto source_dimension = plan->canonical_batch.nbf;
  const auto states = direct_jk_product(plan->diagnostic.batch_size, spin_count);
  const auto rectangular =
      direct_jk_product(states, direct_jk_product(dimension, source_dimension));
  const auto source_elements =
      direct_jk_product(states, direct_jk_product(source_dimension, source_dimension));
  const auto blocks = [](std::size_t elements) {
    return dim3(
        static_cast<unsigned>((elements + kIndependentJkThreads - 1U) / kIndependentJkThreads));
  };
  direct_jk_check(
      cudaMemsetAsync(plan->canonical_active, 0, plan->diagnostic.batch_size, plan->stream));
  direct_jk_check(cudaMemsetAsync(plan->canonical_active + begin, 1, count, plan->stream));
  launch_transform_density_to_direct_right_kernel(
      blocks(rectangular), kIndependentJkThreads, 0, plan->stream, batch_size, spin_count,
      plan->batch.nbf, source_dimension, plan->canonical_transform, staged, plan->canonical_active,
      plan->canonical_projection, plan->canonical_projection_spans);
  direct_jk_check(cudaGetLastError());
  launch_transform_density_to_direct_left_kernel(
      blocks(source_elements), kIndependentJkThreads, 0, plan->stream, batch_size, spin_count,
      plan->batch.nbf, source_dimension, plan->canonical_transform, plan->canonical_projection,
      plan->canonical_active, plan->canonical_density, plan->canonical_projection_spans);
  direct_jk_check(cudaGetLastError());
}

/** Project a whole batch once, then export separate public spin outputs.
 * HF's Fock projection adds hcore; a resident zero matrix makes this a pure
 * two-electron projection. J has identical spin blocks, so only one is exported.
 */
static void direct_jk_canonical_output(CudaDirectJkPlan* plan, bool unrestricted,
                                       const double* source, double* alpha, double* beta) {
  const auto dimension = plan->diagnostic.nbf;
  const auto matrix = direct_jk_product(dimension, dimension);
  const auto bytes = direct_jk_product(matrix, sizeof(double));
  const auto spin_count = unrestricted ? 2U : 1U;
  const double* projected = source;
  if (plan->canonical_transform) {
    const auto rectangular =
        direct_jk_product(direct_jk_product(plan->diagnostic.batch_size, spin_count),
                          direct_jk_product(dimension, plan->canonical_batch.nbf));
    const auto blocks = [](std::size_t elements) {
      return dim3(
          static_cast<unsigned>((elements + kIndependentJkThreads - 1U) / kIndependentJkThreads));
    };
    launch_transform_direct_fock_left_kernel(
        blocks(rectangular), kIndependentJkThreads, 0, plan->stream, plan->batch.batch_size,
        spin_count, plan->batch.nbf, plan->canonical_batch.nbf, plan->canonical_transform, source,
        plan->canonical_active, plan->canonical_projection, plan->canonical_projection_spans);
    direct_jk_check(cudaGetLastError());
    launch_transform_direct_fock_right_kernel(
        blocks(plan->matrix_elements * spin_count), kIndependentJkThreads, 0, plan->stream,
        plan->batch.batch_size, spin_count, plan->batch.nbf, plan->canonical_batch.nbf,
        plan->canonical_transform, plan->canonical_projection, plan->canonical_zero,
        plan->canonical_active, plan->canonical_public_output, plan->canonical_projection_spans);
    direct_jk_check(cudaGetLastError());
    projected = plan->canonical_public_output;
  }
  for (std::size_t item = 0; item < plan->diagnostic.batch_size; ++item) {
    direct_jk_check(cudaMemcpyAsync(alpha + item * matrix, projected + item * spin_count * matrix,
                                    bytes, cudaMemcpyDeviceToDevice, plan->stream));
    if (beta)
      direct_jk_check(cudaMemcpyAsync(beta + item * matrix,
                                      projected + (item * spin_count + 1U) * matrix, bytes,
                                      cudaMemcpyDeviceToDevice, plan->stream));
  }
}

generativeqc_status enqueue_cuda_direct_eri_tile(CudaDirectJkPlan* plan, std::size_t item,
                                                 const std::array<std::size_t, 4>& begin,
                                                 const std::array<std::size_t, 4>& count,
                                                 double* output, std::size_t elements,
                                                 cudaStream_t caller_stream, std::string& detail) {
  detail.clear();
  try {
    direct_jk_require(plan != nullptr, "null direct ERI source");
    direct_jk_require(output != nullptr && caller_stream != nullptr,
                      "direct ERI tile requires device output and stream");
    direct_jk_require(item < plan->diagnostic.batch_size, "direct ERI source item is out of range");
    const auto n = static_cast<std::size_t>(plan->diagnostic.nbf);
    std::size_t requested = 1;
    for (std::size_t axis = 0; axis < 4; ++axis) {
      direct_jk_require(count[axis] > 0 && begin[axis] <= n && count[axis] <= n - begin[axis],
                        "direct ERI tile is outside the public AO basis");
      requested = direct_jk_product(requested, count[axis]);
    }
    direct_jk_require(requested == elements, "direct ERI tile element count mismatch");

    direct_jk_check(cudaSetDevice(plan->device_id));
    int current = -1;
    direct_jk_check(cudaGetDevice(&current));
    direct_jk_require(current == plan->device_id, "direct ERI current device mismatch");
    cudaPointerAttributes attributes{};
    direct_jk_check(cudaPointerGetAttributes(&attributes, output));
    direct_jk_require(attributes.type == cudaMemoryTypeDevice && attributes.device == current,
                      "direct ERI output is not on the prepared CUDA device");

    // A full public-basis tile contains every mate of each ERI orbit. Partial
    // rectangles do not, so retain their bounded independent producer. This
    // only changes evaluation work inside the caller's existing output buffer.
    const bool full_basis =
        std::all_of(begin.begin(), begin.end(), [](auto value) { return value == 0; }) &&
        std::all_of(count.begin(), count.end(), [n](auto value) { return value == n; });
    if (full_basis)
      cuda_execution::launch_build_eri_system_orbits(
          caller_stream, plan->batch, static_cast<std::int32_t>(item), elements, output);
    else
      cuda_execution::launch_independent_eri_tile(caller_stream, plan->batch,
                                                  static_cast<std::int32_t>(item), begin, count,
                                                  elements, output);
    direct_jk_check(cudaGetLastError());
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const DirectJkFailure& failure) {
    detail = failure.detail;
    return failure.status;
  } catch (const std::bad_alloc&) {
    detail = "direct ERI tile extent exceeds numeric capacity";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (cudaError_t error) {
    detail = cudaGetErrorString(error);
    return source_cuda_status(error);
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
}

static generativeqc_status enqueue_cuda_direct_jk_device_impl(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const double* density, const double* beta,
    std::size_t elements, double* coulomb, double* alpha_exchange, double* beta_exchange,
    int* numerical_error, bool mixed_j, std::uint64_t* mixed_coulomb_work_count,
    std::string& detail) {
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
    if (mixed_coulomb_work_count) {
      pointer(mixed_coulomb_work_count);
      direct_jk_require(
          reinterpret_cast<std::uintptr_t>(mixed_coulomb_work_count) % alignof(std::uint64_t) == 0,
          "device direct J/K mixed-work counter is misaligned");
    }
    const auto bytes = direct_jk_product(elements, sizeof(double));
    const double* inputs[]{density, beta};
    double* outputs[]{coulomb, alpha_exchange, beta_exchange};
    for (const auto* input : inputs) {
      direct_jk_require_disjoint(input, bytes, numerical_error, sizeof(int));
      direct_jk_require_disjoint(input, bytes, mixed_coulomb_work_count, sizeof(std::uint64_t));
    }
    direct_jk_require_disjoint(numerical_error, sizeof(int), mixed_coulomb_work_count,
                               sizeof(std::uint64_t));
    for (unsigned i = 0; i < 3; ++i) {
      if (!outputs[i]) continue;
      pointer(outputs[i]);
      for (const auto* input : inputs) direct_jk_require_disjoint(input, bytes, outputs[i], bytes);
      direct_jk_require_disjoint(outputs[i], bytes, numerical_error, sizeof(int));
      direct_jk_require_disjoint(outputs[i], bytes, mixed_coulomb_work_count,
                                 sizeof(std::uint64_t));
      for (unsigned j = 0; j < i; ++j)
        direct_jk_require_disjoint(outputs[i], bytes, outputs[j], bytes);
    }
    direct_jk_check(cudaMemsetAsync(numerical_error, 0, sizeof(int), plan->stream));
    if (mixed_coulomb_work_count)
      direct_jk_check(
          cudaMemsetAsync(mixed_coulomb_work_count, 0, sizeof(std::uint64_t), plan->stream));
    if (plan->canonical_work_count)
      direct_jk_check(
          cudaMemsetAsync(plan->canonical_work_count, 0, 2U * sizeof(std::uint64_t), plan->stream));
    for (const auto* input : inputs)
      if (input) {
        launch_independent_jk_finite_kernel(plan->stream, input, elements, numerical_error);
        direct_jk_check(cudaGetLastError());
      }
    auto* generated_coulomb = plan->generated_exchange ? plan->generated_exchange->shared.get()
                                                       : plan->generated_coulomb.get();
    const bool generated_coulomb_available =
        !mixed_j && (plan->generated_exchange
                         ? direct_jk_generated_full_range_value_available(*plan)
                         : generated_coulomb != nullptr && generated_coulomb->value_capability);
    const bool generated_exchange_available =
        !mixed_j && direct_jk_generated_exchange_value_available(*plan, spec);
    const bool shell_values_cover_request =
        !mixed_j && (!spec.coulomb.present || generated_coulomb_available) &&
        (!spec.exchange.present || generated_exchange_available);
    if ((spec.coulomb.present || spec.exchange.present) && plan->canonical_pairs && !mixed_j &&
        !shell_values_cover_request) {
      const auto dimension = static_cast<std::size_t>(plan->canonical_batch.nbf);
      const auto matrix = direct_jk_product(dimension, dimension);
      const auto spin_count = unrestricted ? 2U : 1U;
      const auto scratch_bytes =
          direct_jk_product(matrix * plan->diagnostic.batch_size * spin_count, sizeof(double));
      direct_jk_canonical_density(plan, unrestricted, density, beta, 0,
                                  plan->diagnostic.batch_size);
      if (spec.coulomb.present)
        direct_jk_check(cudaMemsetAsync(plan->canonical_coulomb, 0, scratch_bytes, plan->stream));
      if (spec.exchange.present)
        direct_jk_check(cudaMemsetAsync(plan->canonical_exchange, 0, scratch_bytes, plan->stream));
      for (std::size_t item = 0; item < plan->diagnostic.batch_size; ++item) {
        const auto& offsets = plan->canonical_pair_offsets[item];
        for (unsigned first = 0; first < 7U; ++first)
          for (unsigned second = 0; second <= first; ++second) {
            launch_canonical_jk_kernel(
                plan->stream, plan->canonical_batch, plan->canonical_cartesian,
                static_cast<std::int32_t>(item), first + second, plan->canonical_pairs,
                direct_jk_pair_rows(plan, second, offsets[first]), offsets[first],
                offsets[first + 1U] - offsets[first], offsets[second],
                offsets[second + 1U] - offsets[second], first == second, spec.coulomb.present,
                spec.exchange.present, unrestricted, direct_exchange_range(spec.exchange),
                spec.exchange.present ? spec.exchange.omega : 0.0, plan->screening_tolerance,
                plan->canonical_bounds, plan->canonical_density, plan->canonical_coulomb,
                plan->canonical_exchange, plan->canonical_work_count);
            direct_jk_check(cudaGetLastError());
          }
      }
      if (spec.coulomb.present)
        direct_jk_canonical_output(plan, unrestricted, plan->canonical_coulomb, coulomb, nullptr);
      if (spec.exchange.present)
        direct_jk_canonical_output(plan, unrestricted, plan->canonical_exchange, alpha_exchange,
                                   unrestricted ? beta_exchange : nullptr);
      for (const auto* output : outputs)
        if (output) {
          launch_independent_jk_finite_kernel(plan->stream, output, elements, numerical_error);
          direct_jk_check(cudaGetLastError());
        }
    } else if (spec.coulomb.present || spec.exchange.present) {
      // Derivative capability is an owner maximum, not a value-schedule request.
      // Full-range value ownership can combine generated/native classes with
      // the bounded higher-l shell fallback.
      const auto dispatch =
          direct_jk_value_dispatch(generated_coulomb_available, generated_exchange_available,
                                   spec.coulomb.present, spec.exchange.present, mixed_j);
      if (dispatch.generated_coulomb) {
        if (plan->generated_exchange && !plan->generated_exchange->shared->value_capability)
          direct_jk_check(enqueue_generated_coulomb(*plan->generated_exchange, unrestricted,
                                                    density, beta, coulomb));
        else
          direct_jk_check(enqueue_generated_coulomb(*generated_coulomb, density, beta, coulomb));
      }
      if (dispatch.generated_exchange)
        direct_jk_check(enqueue_generated_exchange(
            *plan->generated_exchange, unrestricted, density, beta, alpha_exchange, beta_exchange,
            direct_exchange_range(spec.exchange),
            spec.exchange.present ? spec.exchange.omega : 0.0));
      if (dispatch.generic_coulomb || dispatch.generic_exchange) {
        launch_independent_jk_kernel(
            static_cast<unsigned>(elements), kIndependentJkThreads, 0, plan->stream, plan->batch, 0,
            dispatch.generic_coulomb, dispatch.generic_exchange, unrestricted, mixed_j,
            direct_exchange_range(spec.exchange), spec.exchange.present ? spec.exchange.omega : 0.0,
            plan->screening_tolerance, plan->bounds, density, beta, coulomb, alpha_exchange,
            beta_exchange, mixed_coulomb_work_count);
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

generativeqc_status enqueue_cuda_direct_rsh_values_device(
    CudaDirectJkPlan* plan, FockBuildSpec primary, FockBuildSpec correction, const double* density,
    const double* beta, std::size_t elements, double* coulomb, double* full_alpha_exchange,
    double* full_beta_exchange, double* range_alpha_exchange, double* range_beta_exchange,
    int* primary_error, int* range_error, std::string& detail) {
  if (plan == nullptr || !direct_jk_bounded_value_enabled(*plan)) {
    detail = "prepared Direct bounded range values are unavailable or not opted in";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  return direct_jk_guard(plan, detail, [&] {
    primary = direct_jk_strategy(plan, primary, 0, plan->diagnostic.batch_size);
    correction = direct_jk_strategy(plan, correction, 0, plan->diagnostic.batch_size);
    const bool unrestricted = primary.spin == FockSpin::Unrestricted;
    direct_jk_require(
        primary.derivative_order == 0 && primary.coulomb.present &&
            primary.coulomb.approximation == FockApproximation::Exact &&
            primary.coulomb.op == FockOperator::FullRange && primary.exchange.present &&
            primary.exchange.approximation == FockApproximation::Exact &&
            primary.exchange.op == FockOperator::FullRange && correction.derivative_order == 0 &&
            correction.spin == primary.spin && !correction.coulomb.present &&
            correction.exchange.present &&
            correction.exchange.approximation == FockApproximation::Exact &&
            (correction.exchange.op == FockOperator::ShortRange ||
             correction.exchange.op == FockOperator::LongRange) &&
            std::isfinite(correction.exchange.omega) && correction.exchange.omega > 0.0,
        "resident fused RSH value request has incompatible scientific identity");
    direct_jk_require(elements == plan->matrix_elements && density != nullptr &&
                          coulomb != nullptr && full_alpha_exchange != nullptr &&
                          range_alpha_exchange != nullptr && primary_error != nullptr &&
                          range_error != nullptr &&
                          (unrestricted ? beta != nullptr && full_beta_exchange != nullptr &&
                                              range_beta_exchange != nullptr
                                        : beta == nullptr && full_beta_exchange == nullptr &&
                                              range_beta_exchange == nullptr),
                      "resident fused RSH value buffers or dimensions are invalid");

    int current = -1;
    direct_jk_check(cudaGetDevice(&current));
    direct_jk_require(current == plan->device_id, "resident fused RSH current device mismatch");
    const auto device_pointer = [&](const void* value) {
      cudaPointerAttributes attributes{};
      direct_jk_check(cudaPointerGetAttributes(&attributes, value));
      direct_jk_require(attributes.type == cudaMemoryTypeDevice && attributes.device == current,
                        "resident fused RSH requires current-device buffers");
    };
    const void* buffers[]{density,
                          beta,
                          coulomb,
                          full_alpha_exchange,
                          full_beta_exchange,
                          range_alpha_exchange,
                          range_beta_exchange};
    for (const auto* value : buffers)
      if (value) device_pointer(value);
    device_pointer(primary_error);
    device_pointer(range_error);

    const auto bytes = direct_jk_product(elements, sizeof(double));
    const double* inputs[]{density, beta};
    double* outputs[]{coulomb, full_alpha_exchange, full_beta_exchange, range_alpha_exchange,
                      range_beta_exchange};
    for (const auto* input : inputs) {
      direct_jk_require_disjoint(input, bytes, primary_error, sizeof(int));
      direct_jk_require_disjoint(input, bytes, range_error, sizeof(int));
    }
    direct_jk_require_disjoint(primary_error, sizeof(int), range_error, sizeof(int));
    for (unsigned i = 0; i < 5; ++i) {
      for (const auto* input : inputs) direct_jk_require_disjoint(input, bytes, outputs[i], bytes);
      direct_jk_require_disjoint(outputs[i], bytes, primary_error, sizeof(int));
      direct_jk_require_disjoint(outputs[i], bytes, range_error, sizeof(int));
      for (unsigned j = 0; j < i; ++j)
        direct_jk_require_disjoint(outputs[i], bytes, outputs[j], bytes);
    }

    direct_jk_check(cudaMemsetAsync(primary_error, 0, sizeof(int), plan->stream));
    direct_jk_check(cudaMemsetAsync(range_error, 0, sizeof(int), plan->stream));
    if (plan->canonical_work_count)
      direct_jk_check(
          cudaMemsetAsync(plan->canonical_work_count, 0, 2U * sizeof(std::uint64_t), plan->stream));
    launch_independent_jk_finite_kernel(plan->stream, density, elements, primary_error);
    direct_jk_check(cudaGetLastError());
    if (beta) {
      launch_independent_jk_finite_kernel(plan->stream, beta, elements, primary_error);
      direct_jk_check(cudaGetLastError());
    }

    direct_jk_check(enqueue_generated_rsh_values(
        *plan->generated_exchange, unrestricted, density, beta, coulomb, full_alpha_exchange,
        full_beta_exchange, range_alpha_exchange, range_beta_exchange,
        direct_exchange_range(correction.exchange), correction.exchange.omega));

    for (const auto* output : {coulomb, full_alpha_exchange, full_beta_exchange})
      if (output) {
        launch_independent_jk_finite_kernel(plan->stream, output, elements, primary_error);
        direct_jk_check(cudaGetLastError());
      }
    for (const auto* output : {range_alpha_exchange, range_beta_exchange})
      if (output) {
        launch_independent_jk_finite_kernel(plan->stream, output, elements, range_error);
        direct_jk_check(cudaGetLastError());
      }
  });
}

generativeqc_status enqueue_cuda_direct_jk_device(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                                  const double* density, const double* beta,
                                                  std::size_t elements, double* coulomb,
                                                  double* alpha_exchange, double* beta_exchange,
                                                  int* numerical_error, std::string& detail) {
  return enqueue_cuda_direct_jk_device_impl(plan, spec, density, beta, elements, coulomb,
                                            alpha_exchange, beta_exchange, numerical_error, false,
                                            nullptr, detail);
}

generativeqc_status enqueue_cuda_direct_jk_device_mixed_j(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const double* density, const double* beta,
    std::size_t elements, double* coulomb, double* alpha_exchange, double* beta_exchange,
    int* numerical_error, std::string& detail, std::uint64_t* mixed_coulomb_work_count) {
  return enqueue_cuda_direct_jk_device_impl(plan, spec, density, beta, elements, coulomb,
                                            alpha_exchange, beta_exchange, numerical_error, true,
                                            mixed_coulomb_work_count, detail);
}

static generativeqc_status execute_cuda_direct_jk_range(
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
          plan->alpha_exchange, plan->beta_exchange, nullptr);
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

static generativeqc_status execute_cuda_direct_energy_derivative_range(
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
generativeqc_status execute_cuda_direct_jk(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                           const std::vector<double>& density,
                                           const std::vector<double>& beta, std::vector<double>& j,
                                           std::vector<double>& ka, std::vector<double>& kb,
                                           std::string& detail) {
  return execute_cuda_direct_jk_range(plan, 0, plan ? plan->diagnostic.batch_size : 0, spec,
                                      density, beta, j, ka, kb, detail);
}
generativeqc_status execute_cuda_direct_jk_item(CudaDirectJkPlan* plan, std::size_t item,
                                                FockBuildSpec spec,
                                                const std::vector<double>& density,
                                                const std::vector<double>& beta,
                                                std::vector<double>& j, std::vector<double>& ka,
                                                std::vector<double>& kb, std::string& detail) {
  return execute_cuda_direct_jk_range(plan, item, 1, spec, density, beta, j, ka, kb, detail);
}
generativeqc_status execute_cuda_direct_energy_derivative(
    CudaDirectJkPlan* plan, FockBuildSpec spec, const std::vector<double>& density,
    const std::vector<double>& beta, std::vector<double>& derivative, std::string& detail) {
  return execute_cuda_direct_energy_derivative_range(
      plan, 0, plan ? plan->diagnostic.batch_size : 0, spec, density, beta, derivative, detail);
}
generativeqc_status execute_cuda_direct_energy_derivative_item(CudaDirectJkPlan* plan,
                                                               std::size_t item, FockBuildSpec spec,
                                                               const std::vector<double>& density,
                                                               const std::vector<double>& beta,
                                                               std::vector<double>& derivative,
                                                               std::string& detail) {
  return execute_cuda_direct_energy_derivative_range(plan, item, 1, spec, density, beta, derivative,
                                                     detail);
}

generativeqc_status execute_cuda_direct_shell_full_range_derivatives_device(
    CudaDirectJkPlan* plan, FockSpin spin, double coulomb_coefficient, double exchange_coefficient,
    const double* density, const double* beta, std::size_t matrix_elements,
    std::vector<double>& derivatives, std::string& detail) {
  if (plan == nullptr || plan->generated_exchange == nullptr ||
      !plan->generated_exchange->force_capability) {
    detail = "prepared Direct owner has no retained shell derivative lease";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  return direct_jk_guard(plan, detail, [&] {
    direct_jk_require(plan->diagnostic.batch_size == 1,
                      "resident shell derivative requires one prepared item");
    direct_jk_require(plan->diagnostic.derivative_order >= 1,
                      "direct shell first derivatives were not retained");
    direct_jk_require(std::isfinite(coulomb_coefficient) && std::isfinite(exchange_coefficient),
                      "nonfinite resident shell derivative coefficient");
    const auto n = plan->diagnostic.nbf;
    direct_jk_require(density != nullptr && matrix_elements == n * n,
                      "resident shell derivative density shape is invalid");
    const bool unrestricted = spin == FockSpin::Unrestricted;
    direct_jk_require(unrestricted ? beta != nullptr : beta == nullptr,
                      "resident shell derivative spin storage is invalid");
    direct_jk_check(cudaSetDevice(plan->device_id));
    direct_jk_check(cuda_execution::execute_generated_full_range_energy_derivatives(
        *plan->generated_exchange, unrestricted, density, beta, coulomb_coefficient,
        exchange_coefficient, derivatives));
    direct_jk_finite_result(derivatives);
  });
}

generativeqc_status execute_cuda_direct_shell_rsh_energy_derivatives_device(
    CudaDirectJkPlan* plan, FockSpin spin, double coulomb_coefficient,
    double short_exchange_coefficient, double long_exchange_coefficient, double omega,
    const double* density, const double* beta, std::size_t matrix_elements,
    std::vector<double>& derivatives, std::string& detail) {
  if (plan == nullptr || plan->generated_exchange == nullptr ||
      !plan->generated_exchange->force_capability) {
    detail = "prepared Direct owner has no retained shell derivative lease";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  return direct_jk_guard(plan, detail, [&] {
    direct_jk_require(plan->diagnostic.batch_size == 1,
                      "resident shell RSH derivative requires one prepared item");
    direct_jk_require(plan->diagnostic.derivative_order >= 1,
                      "direct shell first derivatives were not retained");
    direct_jk_require(
        std::isfinite(coulomb_coefficient) && std::isfinite(short_exchange_coefficient) &&
            std::isfinite(long_exchange_coefficient) && std::isfinite(omega) && omega > 0.0,
        "nonfinite resident shell RSH derivative coefficient");
    const auto n = plan->diagnostic.nbf;
    direct_jk_require(density != nullptr && matrix_elements == n * n,
                      "resident shell RSH derivative density shape is invalid");
    const bool unrestricted = spin == FockSpin::Unrestricted;
    direct_jk_require(unrestricted ? beta != nullptr : beta == nullptr,
                      "resident shell RSH derivative spin storage is invalid");
    direct_jk_check(cudaSetDevice(plan->device_id));
    direct_jk_check(cuda_execution::execute_generated_rsh_energy_derivatives(
        *plan->generated_exchange, unrestricted, density, beta, coulomb_coefficient,
        short_exchange_coefficient, long_exchange_coefficient, omega, derivatives));
    direct_jk_finite_result(derivatives);
  });
}

/** Values and RSH forces share the same geometry-only admission schedule. */
static void direct_rsh_derivatives(CudaDirectJkPlan* plan, std::size_t item, double cj,
                                   double short_ck, double long_ck, bool unrestricted, double omega,
                                   const double* density, const double* beta) {
  if (plan->canonical_pairs) {
    if (plan->canonical_cartesian) {
      direct_jk_canonical_density(plan, unrestricted, density, beta, item, 1U);
      const auto source_dimension = static_cast<std::size_t>(plan->canonical_batch.nbf);
      const auto matrix = direct_jk_product(source_dimension, source_dimension);
      // The derivative helper adds item*matrix. UKS storage is interleaved,
      // so this additional item offset selects the correct alpha/beta blocks.
      density = plan->canonical_density + (unrestricted ? item * matrix : 0U);
      beta = unrestricted ? density + matrix : nullptr;
    }
    if (plan->canonical_work_count)
      direct_jk_check(
          cudaMemsetAsync(plan->canonical_work_count, 0, 2U * sizeof(std::uint64_t), plan->stream));
    const auto& offsets = plan->canonical_pair_offsets[item];
    for (unsigned first = 0; first < 7U; ++first)
      for (unsigned second = 0; second <= first; ++second) {
        launch_canonical_rsh_derivative_kernel(
            plan->stream, plan->canonical_batch, plan->canonical_cartesian,
            static_cast<std::int32_t>(item), first + second, plan->canonical_pairs,
            direct_jk_pair_rows(plan, second, offsets[first]), offsets[first],
            offsets[first + 1U] - offsets[first], offsets[second],
            offsets[second + 1U] - offsets[second], first == second, plan->coordinate_elements, cj,
            short_ck, long_ck, unrestricted, omega, plan->screening_tolerance,
            plan->canonical_bounds, density, beta, plan->derivative, plan->canonical_work_count);
        direct_jk_check(cudaGetLastError());
      }
  } else {
    launch_independent_rsh_derivative_kernel(
        static_cast<unsigned>(plan->coordinates_per_item), kIndependentJkThreads, 0, plan->stream,
        plan->batch, plan->coordinates_per_item, item, plan->coordinate_elements, cj, short_ck,
        long_ck, unrestricted, omega, plan->screening_tolerance, plan->bounds, density, beta,
        plan->derivative);
    direct_jk_check(cudaGetLastError());
  }
}

generativeqc_status execute_cuda_direct_rsh_energy_derivatives_device(
    CudaDirectJkPlan* plan, FockSpin spin, double coulomb_coefficient,
    double short_exchange_coefficient, double long_exchange_coefficient, double omega,
    const double* density, const double* beta, std::size_t matrix_elements,
    std::vector<double>& derivatives, std::string& detail) {
  return direct_jk_guard(plan, detail, [&] {
    direct_jk_require(plan != nullptr && plan->diagnostic.batch_size == 1,
                      "resident fused RSH derivative requires one prepared item");
    direct_jk_require(
        std::isfinite(coulomb_coefficient) && std::isfinite(short_exchange_coefficient) &&
            std::isfinite(long_exchange_coefficient) && std::isfinite(omega) && omega >= 0.0,
        "nonfinite resident fused RSH derivative coefficient");
    direct_jk_require(plan->diagnostic.derivative_order >= 1,
                      "direct J/K first derivatives were not retained");
    const auto n = plan->diagnostic.nbf;
    direct_jk_require(density != nullptr && matrix_elements == n * n,
                      "resident fused RSH derivative density shape is invalid");
    const bool unrestricted = spin == FockSpin::Unrestricted;
    direct_jk_require(unrestricted ? beta != nullptr : beta == nullptr,
                      "resident fused RSH derivative spin storage is invalid");

    const auto coordinates = plan->coordinates_per_item;
    std::vector<double> result(3 * coordinates);
    if (coulomb_coefficient != 0.0 || short_exchange_coefficient != 0.0 ||
        long_exchange_coefficient != 0.0) {
      direct_jk_check(cudaSetDevice(plan->device_id));
      DirectJkDownloadFence fence{plan->stream};
      for (unsigned source = 0; source < 3; ++source)
        direct_jk_check(cudaMemsetAsync(plan->derivative + source * plan->coordinate_elements, 0,
                                        coordinates * sizeof(double), plan->stream));
      direct_rsh_derivatives(plan, 0, coulomb_coefficient, short_exchange_coefficient,
                             long_exchange_coefficient, unrestricted, omega, density, beta);
      for (unsigned source = 0; source < 3; ++source)
        direct_jk_check(cudaMemcpyAsync(result.data() + source * coordinates,
                                        plan->derivative + source * plan->coordinate_elements,
                                        coordinates * sizeof(double), cudaMemcpyDeviceToHost,
                                        plan->stream));
      fence.complete();
      direct_jk_finite_result(result);
    }
    derivatives = std::move(result);
  });
}

generativeqc_status execute_cuda_direct_rsh_energy_derivatives_item(
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
      direct_rsh_derivatives(plan, item, coulomb_coefficient, short_exchange_coefficient,
                             long_exchange_coefficient, spec.spin == FockSpin::Unrestricted, omega,
                             plan->density, plan->beta);
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

}  // namespace generativeqc::scf

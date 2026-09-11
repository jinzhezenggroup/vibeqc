// Included at scf namespace scope after the shared contracted-ERI evaluator and
// HostBatch packing helpers. This provider adds consumers, not recurrence formulas.
#pragma once

struct CudaDirectJkPlan {
  int device_id{-1};
  DeviceBatch batch{};
  cudaStream_t stream{};
  unsigned derivative_order{};
  std::size_t matrix_elements{}, coordinates_per_item{}, coordinate_elements{};
  double screening_tolerance{};
  double *density{}, *beta{}, *coulomb{}, *alpha_exchange{}, *beta_exchange{}, *bounds{},
      *derivative{};
  int* numerical_failure{};
  std::vector<void*> allocations;
  std::size_t device_bytes{};
  CudaDirectJkDiagnostic diagnostic{};
  ~CudaDirectJkPlan() {
    if (device_id >= 0) (void)cudaSetDevice(device_id);
    if (stream) (void)cudaStreamSynchronize(stream);
    for (void* pointer : allocations) (void)runtime::resource_cuda_free(pointer);
    if (stream) (void)cudaStreamDestroy(stream);
  }
};

namespace {
constexpr unsigned kIndependentJkThreads = 32;

/** Schwarz bounds in public AO order, including sparse spherical expansions. */
__global__ void independent_jk_bounds_kernel(DeviceBatch batch, double* bounds, int* failure) {
  const std::size_t n = batch.nbf, matrix = n * n;
  const std::size_t item = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (item >= static_cast<std::size_t>(batch.batch_size) * matrix) return;
  const auto system = static_cast<std::int32_t>(item / matrix);
  const auto i = static_cast<std::int32_t>((item % matrix) / n);
  const auto j = static_cast<std::int32_t>(item % n);
  const double value = contracted_eri<double>(batch, system, i, j, i, j, -1);
  // NaN bounds must never masquerade as screened-out quartets.
  if (!isfinite(value)) atomicExch(failure, 1);
  bounds[item] = sqrt(fabs(value));
}

/** One output owner reduces all density pairs; no ERI tensor or atomics.
 * Full pair traversal preserves nonsymmetric input orientation. Integral and
 * sparse spherical expansion arithmetic is exactly the existing evaluator.
 */
__global__ void independent_jk_kernel(DeviceBatch batch, std::size_t system_begin, bool want_j,
                                      bool want_k, bool unrestricted, double screening,
                                      const double* bounds, const double* density,
                                      const double* beta, double* j_out, double* ka_out,
                                      double* kb_out) {
  __shared__ double sums[3][kIndependentJkThreads];
  const std::size_t n = batch.nbf, matrix = n * n;
  const std::size_t item = system_begin * matrix + blockIdx.x;
  const auto system = static_cast<std::int32_t>(item / matrix);
  const std::size_t offset = static_cast<std::size_t>(system) * matrix;
  const auto i = static_cast<std::int32_t>((item % matrix) / n);
  const auto j = static_cast<std::int32_t>(item % n);
  double coulomb = 0.0, alpha_exchange = 0.0, beta_exchange = 0.0;
  for (std::size_t kl = threadIdx.x; kl < matrix; kl += blockDim.x) {
    const auto k = static_cast<std::int32_t>(kl / n), l = static_cast<std::int32_t>(kl % n);
    const double a = density[offset + kl], b = unrestricted ? beta[offset + kl] : 0.0;
    if (want_j && bounds[item] * bounds[offset + kl] >= screening && a + b != 0.0)
      coulomb += (a + b) * contracted_eri<double>(batch, system, i, j, k, l, -1);
    if (want_k && bounds[offset + i * n + k] * bounds[offset + j * n + l] >= screening &&
        (a != 0.0 || b != 0.0)) {
      const double value = contracted_eri<double>(batch, system, i, k, j, l, -1);
      alpha_exchange += a * value;
      beta_exchange += b * value;
    }
  }
  sums[0][threadIdx.x] = coulomb;
  sums[1][threadIdx.x] = alpha_exchange;
  sums[2][threadIdx.x] = beta_exchange;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      for (unsigned term = 0; term < 3; ++term)
        sums[term][threadIdx.x] += sums[term][threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (want_j) j_out[item] = sums[0][0];
    if (want_k) ka_out[item] = sums[1][0];
    if (want_k && unrestricted) kb_out[item] = sums[2][0];
  }
}

/** Differentiate the same screened discrete energy at fixed spin densities.
 * Relabel exchange indices so a single ERI derivative serves J and both K
 * terms. The 1/2 energy factor is separate from signed Fock coefficients.
 */
__global__ void independent_jk_derivative_kernel(DeviceBatch batch,
                                                 std::size_t coordinates_per_item,
                                                 std::size_t system_begin, double cj, double ck,
                                                 bool unrestricted, double screening,
                                                 const double* bounds, const double* density,
                                                 const double* beta, double* out) {
  __shared__ double sums[kIndependentJkThreads];
  const std::size_t n = batch.nbf, matrix = n * n, quartets = matrix * matrix;
  const std::size_t coordinate = system_begin * coordinates_per_item + blockIdx.x;
  const auto system = static_cast<std::int32_t>(coordinate / coordinates_per_item);
  const std::size_t offset = static_cast<std::size_t>(system) * matrix;
  double sum = 0.0;
  for (std::size_t quartet = threadIdx.x; quartet < quartets; quartet += blockDim.x) {
    const std::size_t ij = quartet / matrix, kl = quartet % matrix;
    if (bounds[offset + ij] * bounds[offset + kl] < screening) continue;
    const auto i = static_cast<std::int32_t>(ij / n), j = static_cast<std::int32_t>(ij % n);
    const auto k = static_cast<std::int32_t>(kl / n), l = static_cast<std::int32_t>(kl % n);
    double weight = 0.0;
    // An absent/zero-weight term must not evaluate a quadratic that can
    // overflow, even when the requested total-density contribution is finite.
    if (cj != 0.0) {
      const double total_ij = density[offset + ij] + (unrestricted ? beta[offset + ij] : 0.0);
      const double total_kl = density[offset + kl] + (unrestricted ? beta[offset + kl] : 0.0);
      weight += cj * total_ij * total_kl;
    }
    if (ck != 0.0) {
      const std::size_t ik = i * n + k, jl = j * n + l;
      const double exchange = density[offset + ik] * density[offset + jl] +
                              (unrestricted ? beta[offset + ik] * beta[offset + jl] : 0.0);
      weight += ck * exchange;
    }
    weight *= 0.5;
    if (weight != 0.0)
      sum += weight *
             contracted_eri<Dual>(batch, system, i, j, k, l, static_cast<std::int64_t>(coordinate))
                 .derivative;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[coordinate] = sums[0];
}

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
  if (!checked_multiply(a, b, out)) throw std::bad_alloc();
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
  } catch (const std::exception& error) {
    if (plan && plan->stream) (void)cudaStreamSynchronize(plan->stream);
    detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
}
FockBuildSpec direct_jk_spec(const CudaDirectJkPlan* plan, FockBuildSpec spec,
                             const std::vector<double>& density, const std::vector<double>& beta,
                             std::size_t begin, std::size_t count) {
  direct_jk_require(plan != nullptr, "null direct J/K plan");
  direct_jk_require(begin < plan->diagnostic.batch_size && count > 0 &&
                        count <= plan->diagnostic.batch_size - begin,
                    "direct J/K item range is invalid");
  spec = resolve_fock_build(spec, FockBackend::Cpu).spec;  // Mathematical validation only.
  for (const auto* term : {&spec.coulomb, &spec.exchange})
    direct_jk_require(!term->present || term->approximation == FockApproximation::Exact,
                      "exact direct source cannot execute a fitted provider");
  direct_jk_require(spec.derivative_order <= plan->derivative_order,
                    "direct source lacks requested derivative capability");
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
      if (!checked_add(metadata, bytes, metadata)) throw std::bad_alloc();
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
        derivative_order ? direct_jk_product(coord_elements, sizeof(double)) : 0;
    std::size_t required = direct_jk_product(matrix_bytes, 6);
    if (!checked_add(required, gradient_bytes, required) ||
        !checked_add(required, sizeof(int), required) ||
        !checked_add(required, metadata, required) || required > budget)
      throw std::bad_alloc();
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
    independent_jk_bounds_kernel<<<blocks, kIndependentJkThreads, 0, plan->stream>>>(
        plan->batch, plan->bounds, plan->numerical_failure);
    direct_jk_check(cudaGetLastError());
    int numerical_failure = 0;
    direct_jk_check(cudaMemcpyAsync(&numerical_failure, plan->numerical_failure, sizeof(int),
                                    cudaMemcpyDeviceToHost, plan->stream));
    direct_jk_check(cudaStreamSynchronize(plan->stream));
    if (numerical_failure)
      throw DirectJkFailure{VIBEQC_STATUS_NUMERICAL_FAILURE, "nonfinite direct J/K Schwarz bound"};
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
    info.derivative_order = derivative_order;
    info.screening_tolerance = screening_tolerance;
    diagnostic = info;
    *output = plan.release();
  });
}
void destroy_cuda_direct_jk_plan(CudaDirectJkPlan* plan) noexcept { delete plan; }

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
      independent_jk_kernel<<<static_cast<unsigned>(density.size()), kIndependentJkThreads, 0,
                              plan->stream>>>(
          plan->batch, begin, spec.coulomb.present, spec.exchange.present, unrestricted,
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
      independent_jk_derivative_kernel<<<static_cast<unsigned>(result.size()),
                                         kIndependentJkThreads, 0, plan->stream>>>(
          plan->batch, plan->coordinates_per_item, begin, cj, ck,
          spec.spin == FockSpin::Unrestricted, plan->screening_tolerance, plan->bounds,
          plan->density, plan->beta, plan->derivative);
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

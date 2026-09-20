#include <cfloat>
#include <cmath>
#include <cstdint>

#include "dft/dispersion/d4_cuda.hpp"

namespace vibeqc::dft::dispersion {
namespace {

constexpr int kThreadsPerBlock = 256;

__device__ double atomic_add_fp64(double* address, double value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (true) {
    const unsigned long long assumed = old;
    const double current = __longlong_as_double(static_cast<long long>(assumed));
    old = atomicCAS(bits, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(current + value)));
    if (old == assumed) return current;
  }
#endif
}

__device__ void record_status(D4Status* statuses, std::uint32_t system, D4Status status) {
  atomicCAS(reinterpret_cast<int*>(statuses + system), static_cast<int>(D4Status::success),
            static_cast<int>(status));
}

__device__ bool active_member(const D4CudaBatch& batch, std::uint32_t system) {
  return batch.active == nullptr || batch.active[system] == 1u;
}

__device__ bool successful_member(const D4CudaBatch& batch, const D4CudaResult& result,
                                  std::uint32_t system) {
  return active_member(batch, system) && result.statuses[system] == D4Status::success;
}

__device__ void unpack_pair(std::uint32_t packed, int& first, int& second) {
  int hi = static_cast<int>((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(packed))) * 0.5);
  while (hi * (hi - 1) / 2 > static_cast<int>(packed)) --hi;
  while ((hi + 1) * hi / 2 <= static_cast<int>(packed)) ++hi;
  first = hi;
  second = static_cast<int>(packed) - hi * (hi - 1) / 2;
}

__device__ std::uint64_t choose3(std::uint64_t n) { return n < 3 ? 0 : n * (n - 1) * (n - 2) / 6; }

__device__ std::uint64_t choose2(std::uint64_t n) { return n < 2 ? 0 : n * (n - 1) / 2; }

__device__ void unpack_triple(std::uint64_t packed, int n, int& first, int& second, int& third) {
  int lo = 2;
  int hi = n - 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo + 1) / 2;
    if (choose3(static_cast<std::uint64_t>(mid)) <= packed)
      lo = mid;
    else
      hi = mid - 1;
  }
  first = lo;
  const std::uint64_t remainder = packed - choose3(static_cast<std::uint64_t>(first));
  lo = 1;
  hi = first - 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo + 1) / 2;
    if (choose2(static_cast<std::uint64_t>(mid)) <= remainder)
      lo = mid;
    else
      hi = mid - 1;
  }
  second = lo;
  third = static_cast<int>(remainder - choose2(static_cast<std::uint64_t>(second)));
}

__device__ void add_pair_gradient_atomic(int first, int second, const double* vector, double scale,
                                         double* gradient) {
  for (int axis = 0; axis < 3; ++axis) {
    atomic_add_fp64(gradient + 3 * first + axis, scale * vector[axis]);
    atomic_add_fp64(gradient + 3 * second + axis, -scale * vector[axis]);
  }
}

__device__ void compute_weights_atom(int atom, const std::int32_t* atomic_numbers,
                                     const double* coordination, const double* charges,
                                     bool zero_charge, const D4Parameters& parameters,
                                     D4Tables tables, double* weights, double* cn_derivatives,
                                     double* charge_derivatives) {
  const auto element = tables.elements[atomic_numbers[atom] - 1];
  double norm = 0.0;
  double norm_derivative = 0.0;
  double maximum_cn = -DBL_MAX;
  for (int ref_index = 0; ref_index < element.reference_count; ++ref_index) {
    const auto reference = tables.references[element.reference_offset + ref_index];
    maximum_cn = fmax(maximum_cn, reference.coordination_number);
    const double delta = coordination[atom] - reference.coordination_number;
    for (int gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double value = exp(-6.0 * gaussian * delta * delta);
      norm += value;
      norm_derivative -= 12.0 * gaussian * delta * value;
    }
  }
  const double inverse = norm > 1.4916681462400413e-154 ? 1.0 / norm : 0.0;
  for (int ref_index = 0; ref_index < element.reference_count; ++ref_index) {
    const auto reference = tables.references[element.reference_offset + ref_index];
    const double delta = coordination[atom] - reference.coordination_number;
    double numerator = 0.0;
    double numerator_derivative = 0.0;
    for (int gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double value = exp(-6.0 * gaussian * delta * delta);
      numerator += value;
      numerator_derivative -= 12.0 * gaussian * delta * value;
    }
    const double gaussian_weight =
        inverse ? numerator * inverse
                : (fabs(maximum_cn - reference.coordination_number) < 1.0e-12 ? 1.0 : 0.0);
    const double gaussian_derivative =
        inverse ? inverse * (numerator_derivative - numerator * norm_derivative * inverse) : 0.0;
    double scale = 0.0;
    double scale_derivative = 0.0;
    d4_detail::charge_scale(parameters.ga, parameters.gc * element.hardness,
                            reference.charge + element.effective_charge,
                            (zero_charge ? 0.0 : charges[atom]) + element.effective_charge, scale,
                            scale_derivative);
    weights[7 * atom + ref_index] = gaussian_weight * scale;
    cn_derivatives[7 * atom + ref_index] = gaussian_derivative * scale;
    charge_derivatives[7 * atom + ref_index] =
        zero_charge ? 0.0 : gaussian_weight * scale_derivative;
  }
}

// Offsets describe shared storage, so a malformed partition is a batch error.
// Validate once before any per-member workspace access, not only at neighbors.
__global__ void validate_partition_kernel(D4CudaBatch batch, D4CudaResult result) {
  __shared__ int invalid;
  if (threadIdx.x == 0)
    invalid = batch.offsets[0] != 0u || batch.offsets[batch.systems] != batch.total_atoms;
  __syncthreads();
  for (std::size_t system = threadIdx.x; system < batch.systems; system += blockDim.x)
    if (batch.offsets[system] > batch.offsets[system + 1] ||
        batch.offsets[system + 1] > batch.total_atoms)
      atomicExch(&invalid, 1);
  __syncthreads();
  if (invalid)
    for (std::size_t system = threadIdx.x; system < batch.systems; system += blockDim.x)
      result.statuses[system] = D4Status::invalid_argument;
}

__global__ void validate_and_coordination_kernel(D4CudaBatch batch, D4Parameters parameters,
                                                 D4Tables tables, double* workspace,
                                                 D4CudaResult result) {
  const std::uint32_t system = blockIdx.x;
  if (system >= batch.systems || result.statuses[system] != D4Status::success) return;
  __shared__ std::uint32_t begin;
  __shared__ std::uint32_t end;
  __shared__ int run;
  if (threadIdx.x == 0) {
    begin = batch.offsets[system];
    end = batch.offsets[system + 1];
    run = 1;
    if ((system == 0 && begin != 0u) || (system > 0 && batch.offsets[system - 1] > begin) ||
        (system + 1 == batch.systems && end != batch.total_atoms) || begin > end ||
        end > batch.total_atoms) {
      record_status(result.statuses, system, D4Status::invalid_argument);
      run = 0;
    } else if (end - begin > static_cast<std::uint32_t>(kD4MaximumAtoms)) {
      record_status(result.statuses, system, D4Status::unsupported);
      run = 0;
    } else if (batch.active != nullptr && batch.active[system] > 1u) {
      record_status(result.statuses, system, D4Status::invalid_argument);
      run = 0;
    } else if (!active_member(batch, system)) {
      run = 0;
    }
  }
  __syncthreads();
  if (!run) return;

  const int n = static_cast<int>(end - begin);
  if (n == 0) return;
  const auto* z = batch.atomic_numbers + begin;
  const auto* xyz = batch.coordinates + 3 * begin;
  const auto* q = batch.charges + begin;
  double* local_workspace = workspace + 27u * begin;
  double* coordination = local_workspace;

  for (int atom = threadIdx.x; atom < n; atom += blockDim.x) {
    if (z[atom] < 1 || z[atom] > static_cast<int>(tables.element_count) ||
        !d4_detail::finite(q[atom]) || !d4_detail::finite(xyz[3 * atom]) ||
        !d4_detail::finite(xyz[3 * atom + 1]) || !d4_detail::finite(xyz[3 * atom + 2])) {
      record_status(result.statuses, system,
                    z[atom] < 1 || z[atom] > static_cast<int>(tables.element_count)
                        ? D4Status::unsupported
                        : D4Status::invalid_argument);
    }
  }
  __syncthreads();
  if (result.statuses[system] != D4Status::success) return;

  const std::uint32_t pair_count = static_cast<std::uint32_t>(n * (n - 1) / 2);
  for (std::uint32_t packed = threadIdx.x; packed < pair_count; packed += blockDim.x) {
    int first = 0;
    int second = 0;
    unpack_pair(packed, first, second);
    double vector[3];
    const double r2 = d4_detail::distance2(xyz, first, second, vector);
    if (!d4_detail::finite(r2) || r2 < 1.0e-12) {
      record_status(result.statuses, system, D4Status::invalid_argument);
      continue;
    }
    const double distance = sqrt(r2);
    if (distance <= parameters.cn_cutoff) {
      double value = 0.0;
      double derivative = 0.0;
      d4_detail::cn_pair(first, second, z, tables, distance, value, derivative);
      atomic_add_fp64(coordination + first, value);
      atomic_add_fp64(coordination + second, value);
    }
  }
}

__global__ void weights_kernel(D4CudaBatch batch, D4Parameters parameters, D4Tables tables,
                               double* workspace, D4CudaResult result, bool zero_charge) {
  const std::uint32_t system = blockIdx.x;
  if (system >= batch.systems || !successful_member(batch, result, system)) return;
  const std::uint32_t begin = batch.offsets[system];
  const int n = static_cast<int>(batch.offsets[system + 1] - begin);
  if (n == 0) return;
  const auto* z = batch.atomic_numbers + begin;
  const auto* q = batch.charges + begin;
  double* base = workspace + 27u * begin;
  double* coordination = base;
  double* weights = coordination + n;
  double* cn_derivatives = weights + 7 * n;
  double* charge_derivatives = cn_derivatives + 7 * n;
  for (int atom = threadIdx.x; atom < n; atom += blockDim.x)
    compute_weights_atom(atom, z, coordination, q, zero_charge, parameters, tables, weights,
                         cn_derivatives, charge_derivatives);
}

__global__ void two_body_kernel(D4CudaBatch batch, D4Parameters parameters, D4Tables tables,
                                double* workspace, D4CudaResult result) {
  const std::uint32_t system = blockIdx.x;
  if (system >= batch.systems || !successful_member(batch, result, system)) return;
  const std::uint32_t begin = batch.offsets[system];
  const int n = static_cast<int>(batch.offsets[system + 1] - begin);
  if (n == 0) return;
  const auto* z = batch.atomic_numbers + begin;
  const auto* xyz = batch.coordinates + 3 * begin;
  double* base = workspace + 27u * begin;
  double* coordination = base;
  double* weights = coordination + n;
  double* cn_derivatives = weights + 7 * n;
  double* charge_derivatives = cn_derivatives + 7 * n;
  double* coordination_adjoints = charge_derivatives + 7 * n;
  double* gradient = coordination_adjoints + n;
  double* dedq = gradient + 3 * n;

  const std::uint32_t pair_count = static_cast<std::uint32_t>(n * (n - 1) / 2);
  for (std::uint32_t packed = threadIdx.x; packed < pair_count; packed += blockDim.x) {
    int first = 0;
    int second = 0;
    unpack_pair(packed, first, second);
    double vector[3];
    const double r2 = d4_detail::distance2(xyz, first, second, vector);
    const double distance = sqrt(r2);
    if (distance > parameters.pair_cutoff) continue;
    const auto coefficient = d4_detail::coefficient(first, second, z, tables, weights,
                                                    cn_derivatives, charge_derivatives);
    const double rr =
        3.0 * tables.elements[z[first] - 1].r4r2 * tables.elements[z[second] - 1].r4r2;
    const double r0 = d4_detail::radius(first, second, z, tables, parameters);
    const double u = r0 * r0;
    const double t6 = 1.0 / (r2 * r2 * r2 + u * u * u);
    const double t8 = 1.0 / (r2 * r2 * r2 * r2 + u * u * u * u);
    const double damping = parameters.s6 * t6 + parameters.s8 * rr * t8;
    const double damping_derivative = -6.0 * parameters.s6 * r2 * r2 * t6 * t6 -
                                      8.0 * parameters.s8 * rr * r2 * r2 * r2 * t8 * t8;
    atomic_add_fp64(result.energies + 2 * system, -coefficient.c6 * damping);
    atomic_add_fp64(dedq + first, -coefficient.qi * damping);
    atomic_add_fp64(dedq + second, -coefficient.qj * damping);
    atomic_add_fp64(coordination_adjoints + first, -coefficient.ci * damping);
    atomic_add_fp64(coordination_adjoints + second, -coefficient.cj * damping);
    add_pair_gradient_atomic(first, second, vector, -coefficient.c6 * damping_derivative, gradient);
  }
}

__global__ void atm_kernel(D4CudaBatch batch, D4Parameters parameters, D4Tables tables,
                           double* workspace, D4CudaResult result) {
  const std::uint32_t system = blockIdx.x;
  if (system >= batch.systems || parameters.s9 == 0.0 || !successful_member(batch, result, system))
    return;
  const std::uint32_t begin = batch.offsets[system];
  const int n = static_cast<int>(batch.offsets[system + 1] - begin);
  if (n < 3) return;
  const auto* z = batch.atomic_numbers + begin;
  const auto* xyz = batch.coordinates + 3 * begin;
  double* base = workspace + 27u * begin;
  double* coordination = base;
  double* weights = coordination + n;
  double* cn_derivatives = weights + 7 * n;
  double* charge_derivatives = cn_derivatives + 7 * n;
  double* coordination_adjoints = charge_derivatives + 7 * n;
  double* gradient = coordination_adjoints + n;

  const std::uint64_t triple_count = choose3(static_cast<std::uint64_t>(n));
  for (std::uint64_t packed = threadIdx.x; packed < triple_count; packed += blockDim.x) {
    int first = 0;
    int second = 0;
    int third = 0;
    unpack_triple(packed, n, first, second, third);
    double first_second[3];
    const double a = d4_detail::distance2(xyz, first, second, first_second);
    if (sqrt(a) > parameters.atm_cutoff) continue;
    const auto c12 = d4_detail::coefficient(first, second, z, tables, weights, cn_derivatives,
                                            charge_derivatives);
    double first_third[3];
    double second_third[3];
    const double b = d4_detail::distance2(xyz, first, third, first_third);
    const double c = d4_detail::distance2(xyz, second, third, second_third);
    if (sqrt(b) > parameters.atm_cutoff || sqrt(c) > parameters.atm_cutoff) continue;
    const auto c13 = d4_detail::coefficient(first, third, z, tables, weights, cn_derivatives,
                                            charge_derivatives);
    const auto c23 = d4_detail::coefficient(second, third, z, tables, weights, cn_derivatives,
                                            charge_derivatives);
    if (!(c12.c6 > 0.0 && c13.c6 > 0.0 && c23.c6 > 0.0)) {
      record_status(result.statuses, system, D4Status::numerical_failure);
      continue;
    }
    const double r2_product = a * b * c;
    const double r1_product = sqrt(r2_product);
    const double r3_product = r2_product * r1_product;
    const double r5_product = r3_product * r2_product;
    const double ratio = d4_detail::radius(first, second, z, tables, parameters) *
                         d4_detail::radius(first, third, z, tables, parameters) *
                         d4_detail::radius(second, third, z, tables, parameters) / r1_product;
    const double rp = pow(ratio, 16.0 / 3.0);
    const double damping = 1.0 / (1.0 + 6.0 * rp);
    const double angle =
        0.375 * (a + c - b) * (a - c + b) * (-a + c + b) / r5_product + 1.0 / r3_product;
    const double c9 = -parameters.s9 * sqrt(c12.c6 * c13.c6 * c23.c6);
    const double delta_energy = angle * damping * c9;
    const double damping_derivative = -32.0 * rp * damping * damping;
    atomic_add_fp64(result.energies + 2 * system + 1, -delta_energy);
    add_pair_gradient_atomic(
        first, second, first_second,
        d4_detail::atm_radial(a, c, b, r5_product, damping, angle, damping_derivative, c9),
        gradient);
    add_pair_gradient_atomic(
        first, third, first_third,
        d4_detail::atm_radial(b, c, a, r5_product, damping, angle, damping_derivative, c9),
        gradient);
    add_pair_gradient_atomic(
        second, third, second_third,
        d4_detail::atm_radial(c, b, a, r5_product, damping, angle, damping_derivative, c9),
        gradient);
    atomic_add_fp64(coordination_adjoints + first,
                    -0.5 * delta_energy * (c12.ci / c12.c6 + c13.ci / c13.c6));
    atomic_add_fp64(coordination_adjoints + second,
                    -0.5 * delta_energy * (c12.cj / c12.c6 + c23.ci / c23.c6));
    atomic_add_fp64(coordination_adjoints + third,
                    -0.5 * delta_energy * (c13.cj / c13.c6 + c23.cj / c23.c6));
  }
}

__global__ void coordination_response_kernel(D4CudaBatch batch, D4Parameters parameters,
                                             D4Tables tables, double* workspace,
                                             D4CudaResult result) {
  const std::uint32_t system = blockIdx.x;
  if (system >= batch.systems || !successful_member(batch, result, system)) return;
  const std::uint32_t begin = batch.offsets[system];
  const int n = static_cast<int>(batch.offsets[system + 1] - begin);
  if (n == 0) return;
  const auto* z = batch.atomic_numbers + begin;
  const auto* xyz = batch.coordinates + 3 * begin;
  double* base = workspace + 27u * begin;
  double* coordination_adjoints = base + n + 21 * n;
  double* gradient = coordination_adjoints + n;

  const std::uint32_t pair_count = static_cast<std::uint32_t>(n * (n - 1) / 2);
  for (std::uint32_t packed = threadIdx.x; packed < pair_count; packed += blockDim.x) {
    int first = 0;
    int second = 0;
    unpack_pair(packed, first, second);
    double vector[3];
    const double distance = sqrt(d4_detail::distance2(xyz, first, second, vector));
    if (distance > parameters.cn_cutoff) continue;
    double value = 0.0;
    double derivative = 0.0;
    d4_detail::cn_pair(first, second, z, tables, distance, value, derivative);
    add_pair_gradient_atomic(
        first, second, vector,
        derivative * (coordination_adjoints[first] + coordination_adjoints[second]) / distance,
        gradient);
  }
}

__global__ void finalize_kernel(D4CudaBatch batch, double* workspace, D4CudaResult result) {
  const std::uint32_t system = blockIdx.x;
  if (system >= batch.systems) return;
  // Admission errors precede every numerical phase. Their outputs are already
  // zero; never use an invalid interval to clear another member's storage.
  if (result.statuses[system] == D4Status::invalid_argument ||
      result.statuses[system] == D4Status::unsupported)
    return;
  const std::uint32_t begin = batch.offsets[system];
  const std::uint32_t end = batch.offsets[system + 1];
  if (begin > end || end > batch.total_atoms) {
    if (threadIdx.x == 0) {
      result.energies[2 * system] = 0.0;
      result.energies[2 * system + 1] = 0.0;
    }
    return;
  }
  const int n = static_cast<int>(end - begin);
  if (n == 0) return;
  double* base = workspace + 27u * begin;
  double* coordination_adjoints = base + n + 21 * n;
  double* gradient = coordination_adjoints + n;
  double* dedq = gradient + 3 * n;

  if (active_member(batch, system) && result.statuses[system] == D4Status::success) {
    if (threadIdx.x == 0 && (!d4_detail::finite(result.energies[2 * system]) ||
                             !d4_detail::finite(result.energies[2 * system + 1])))
      record_status(result.statuses, system, D4Status::numerical_failure);
    for (int atom = threadIdx.x; atom < n; atom += blockDim.x) {
      if (!d4_detail::finite(dedq[atom]) || !d4_detail::finite(gradient[3 * atom]) ||
          !d4_detail::finite(gradient[3 * atom + 1]) || !d4_detail::finite(gradient[3 * atom + 2]))
        record_status(result.statuses, system, D4Status::numerical_failure);
    }
  }
  __syncthreads();

  if (!active_member(batch, system) || result.statuses[system] != D4Status::success) {
    if (threadIdx.x == 0) {
      result.energies[2 * system] = 0.0;
      result.energies[2 * system + 1] = 0.0;
    }
    for (int atom = threadIdx.x; atom < n; atom += blockDim.x) {
      result.dedq[begin + atom] = 0.0;
      result.gradients[3 * (begin + atom)] = 0.0;
      result.gradients[3 * (begin + atom) + 1] = 0.0;
      result.gradients[3 * (begin + atom) + 2] = 0.0;
    }
    return;
  }
  for (int atom = threadIdx.x; atom < n; atom += blockDim.x) {
    result.dedq[begin + atom] = dedq[atom];
    result.gradients[3 * (begin + atom)] = gradient[3 * atom];
    result.gradients[3 * (begin + atom) + 1] = gradient[3 * atom + 1];
    result.gradients[3 * (begin + atom) + 2] = gradient[3 * atom + 2];
  }
}

cudaError_t launch_status() { return cudaPeekAtLastError(); }

}  // namespace

cudaError_t launch_d4_fixed_charge_batched_cuda(const D4CudaBatch& batch,
                                                const D4Parameters& parameters, D4Tables tables,
                                                double* workspace, std::size_t workspace_elements,
                                                const D4CudaResult& result, cudaStream_t stream) {
  if (batch.systems == 0) return cudaSuccess;
  if (batch.offsets == nullptr || result.statuses == nullptr || result.energies == nullptr ||
      !d4_detail::valid_parameters(parameters) ||
      parameters.reference_model != tables.reference_model ||
      fabs(parameters.ga - tables.ga) > 1.0e-15 || fabs(parameters.gc - tables.gc) > 1.0e-15 ||
      tables.elements == nullptr || tables.references == nullptr ||
      tables.reference_c6 == nullptr || tables.element_count != data::kElementCount ||
      tables.reference_count != data::kReferenceCount ||
      tables.reference_c6_count != data::kReferenceCount * (data::kReferenceCount + 1) / 2)
    return cudaErrorInvalidValue;
  if (batch.total_atoms != 0 &&
      (batch.atomic_numbers == nullptr || batch.coordinates == nullptr ||
       batch.charges == nullptr || workspace == nullptr || result.gradients == nullptr ||
       result.dedq == nullptr ||
       workspace_elements < d4_cuda_workspace_elements(batch.total_atoms)))
    return cudaErrorInvalidValue;

  cudaError_t error = cudaMemsetAsync(result.statuses, 0, batch.systems * sizeof(D4Status), stream);
  if (error != cudaSuccess) return error;
  error = cudaMemsetAsync(result.energies, 0, 2u * batch.systems * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  if (batch.total_atoms != 0) {
    error = cudaMemsetAsync(workspace, 0,
                            d4_cuda_workspace_elements(batch.total_atoms) * sizeof(double), stream);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(result.gradients, 0, 3u * batch.total_atoms * sizeof(double), stream);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(result.dedq, 0, batch.total_atoms * sizeof(double), stream);
    if (error != cudaSuccess) return error;
  }

  validate_partition_kernel<<<1, kThreadsPerBlock, 0, stream>>>(batch, result);
  if ((error = launch_status()) != cudaSuccess) return error;
  validate_and_coordination_kernel<<<batch.systems, kThreadsPerBlock, 0, stream>>>(
      batch, parameters, tables, workspace, result);
  if ((error = launch_status()) != cudaSuccess) return error;
  weights_kernel<<<batch.systems, kThreadsPerBlock, 0, stream>>>(batch, parameters, tables,
                                                                 workspace, result, false);
  if ((error = launch_status()) != cudaSuccess) return error;
  two_body_kernel<<<batch.systems, kThreadsPerBlock, 0, stream>>>(batch, parameters, tables,
                                                                  workspace, result);
  if ((error = launch_status()) != cudaSuccess) return error;
  if (parameters.s9 != 0.0) {
    weights_kernel<<<batch.systems, kThreadsPerBlock, 0, stream>>>(batch, parameters, tables,
                                                                   workspace, result, true);
    if ((error = launch_status()) != cudaSuccess) return error;
    atm_kernel<<<batch.systems, kThreadsPerBlock, 0, stream>>>(batch, parameters, tables, workspace,
                                                               result);
    if ((error = launch_status()) != cudaSuccess) return error;
  }
  coordination_response_kernel<<<batch.systems, kThreadsPerBlock, 0, stream>>>(
      batch, parameters, tables, workspace, result);
  if ((error = launch_status()) != cudaSuccess) return error;
  finalize_kernel<<<batch.systems, kThreadsPerBlock, 0, stream>>>(batch, workspace, result);
  return launch_status();
}

}  // namespace vibeqc::dft::dispersion

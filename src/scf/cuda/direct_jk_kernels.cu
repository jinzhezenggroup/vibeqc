#include <algorithm>
#include <cmath>

#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_native_contraction.cuh"

namespace vibeqc::scf {
namespace {
using namespace cuda_execution;
}

namespace {

vibeqc::integrals::CoulombRange integral_range(DirectCoulombRange range) {
  switch (range) {
    case DirectCoulombRange::Full:
      return vibeqc::integrals::CoulombRange::Full;
    case DirectCoulombRange::Long:
      return vibeqc::integrals::CoulombRange::Long;
    case DirectCoulombRange::Short:
      return vibeqc::integrals::CoulombRange::Short;
  }
  return vibeqc::integrals::CoulombRange::Full;
}

__global__ void independent_jk_finite_kernel(const double* values, std::size_t count,
                                             int* failure) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(values[i])) atomicExch(failure, 1);
}

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
template <bool MixedJ>
__global__ void independent_jk_kernel(DeviceBatch batch, std::size_t system_begin, bool want_j,
                                      bool want_k, bool unrestricted,
                                      vibeqc::integrals::CoulombRange exchange_range,
                                      double exchange_omega, double screening, const double* bounds,
                                      const double* density, const double* beta, double* j_out,
                                      double* ka_out, double* kb_out) {
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
    if (want_j && bounds[item] * bounds[offset + kl] >= screening && a + b != 0.0) {
      const double value =
          MixedJ ? scalar_value(contracted_eri<MixedPrecisionFloat>(batch, system, i, j, k, l, -1))
                 : contracted_eri<double>(batch, system, i, j, k, l, -1);
      coulomb += (a + b) * value;
    }
    if (want_k && bounds[offset + i * n + k] * bounds[offset + j * n + l] >= screening &&
        (a != 0.0 || b != 0.0)) {
      const double value =
          contracted_eri<double>(batch, system, i, k, j, l, -1, exchange_range, exchange_omega);
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
 *
 * Traverse each ordered AO quartet once, then differentiate only the unique
 * nuclear centers that actually occur in that quartet. Dual3 carries x/y/z
 * together and translational invariance reconstructs the final center. This
 * removes the previous coordinate-by-AO^4 scan without changing screening,
 * public-AO spherical expansion, coefficients, or radial operators.
 */
__global__ void independent_jk_derivative_kernel(DeviceBatch batch, std::size_t system_begin,
                                                 std::size_t system_count, double cj, double ck,
                                                 bool unrestricted,
                                                 vibeqc::integrals::CoulombRange exchange_range,
                                                 double exchange_omega, double screening,
                                                 const double* bounds, const double* density,
                                                 const double* beta, double* out) {
  const std::size_t n = batch.nbf, matrix = n * n, quartets = matrix * matrix;
  const std::size_t work_count = system_count * quartets;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < work_count; work += stride) {
    const std::size_t local_system = work / quartets;
    const auto system = static_cast<std::int32_t>(system_begin + local_system);
    const std::size_t quartet = work % quartets;
    const std::size_t offset = static_cast<std::size_t>(system) * matrix;
    const std::size_t ij = quartet / matrix, kl = quartet % matrix;
    if (bounds[offset + ij] * bounds[offset + kl] < screening) continue;

    const auto i = static_cast<std::int32_t>(ij / n), j = static_cast<std::int32_t>(ij % n);
    const auto k = static_cast<std::int32_t>(kl / n), l = static_cast<std::int32_t>(kl % n);
    double full_weight = 0.0, range_weight = 0.0;
    // An absent/zero-weight term must not evaluate a quadratic that can
    // overflow, even when the requested total-density contribution is finite.
    if (cj != 0.0) {
      const double total_ij = density[offset + ij] + (unrestricted ? beta[offset + ij] : 0.0);
      const double total_kl = density[offset + kl] + (unrestricted ? beta[offset + kl] : 0.0);
      full_weight += 0.5 * cj * total_ij * total_kl;
    }
    if (ck != 0.0) {
      const std::size_t ik = static_cast<std::size_t>(i) * n + k;
      const std::size_t jl = static_cast<std::size_t>(j) * n + l;
      const double exchange = density[offset + ik] * density[offset + jl] +
                              (unrestricted ? beta[offset + ik] * beta[offset + jl] : 0.0);
      if (exchange_range == vibeqc::integrals::CoulombRange::Full)
        full_weight += 0.5 * ck * exchange;
      else
        range_weight = 0.5 * ck * exchange;
    }
    if (full_weight == 0.0 && range_weight == 0.0) continue;

    const std::size_t base = static_cast<std::size_t>(system) * n;
    const std::int32_t center_atoms[4] = {
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(i)]],
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(j)]],
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(k)]],
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(l)]],
    };
    std::int32_t unique_atoms[4];
    unsigned unique_count = 0;
    for (unsigned center = 0; center < 4; ++center) {
      bool duplicate = false;
      for (unsigned previous = 0; previous < unique_count; ++previous)
        duplicate = duplicate || unique_atoms[previous] == center_atoms[center];
      if (!duplicate) unique_atoms[unique_count++] = center_atoms[center];
    }
    if (unique_count <= 1) continue;

    double reconstructed[3]{};
    for (unsigned center = 0; center + 1 < unique_count; ++center) {
      const std::int64_t coordinate = static_cast<std::int64_t>(unique_atoms[center]) * 3;
      double derivative[3]{};
      if (full_weight != 0.0) {
        const Dual3 value = contracted_eri<Dual3>(batch, system, i, j, k, l, coordinate);
        derivative[0] += full_weight * value.derivative_x;
        derivative[1] += full_weight * value.derivative_y;
        derivative[2] += full_weight * value.derivative_z;
      }
      if (range_weight != 0.0) {
        const Dual3 value = contracted_eri<Dual3>(batch, system, i, j, k, l, coordinate,
                                                  exchange_range, exchange_omega);
        derivative[0] += range_weight * value.derivative_x;
        derivative[1] += range_weight * value.derivative_y;
        derivative[2] += range_weight * value.derivative_z;
      }
      for (unsigned axis = 0; axis < 3; ++axis) {
        reconstructed[axis] += derivative[axis];
        if (derivative[axis] != 0.0)
          atomicAdd(out + static_cast<std::size_t>(coordinate) + axis, derivative[axis]);
      }
    }
    const std::size_t final_coordinate =
        static_cast<std::size_t>(unique_atoms[unique_count - 1]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis)
      if (reconstructed[axis] != 0.0)
        atomicAdd(out + final_coordinate + axis, -reconstructed[axis]);
  }
}


/** One RSH force pass: J uses the full Coulomb derivative, while SR/LR K
 * share Full = Short + Long. Evaluate Full and Long once per participating
 * center and form Short by subtraction.
 */
__global__ void independent_rsh_derivative_kernel(
    DeviceBatch batch, std::size_t system_begin, std::size_t system_count,
    std::size_t source_stride, double cj, double short_ck, double long_ck, bool unrestricted,
    double omega, double screening, const double* bounds, const double* density,
    const double* beta, double* out) {
  const std::size_t n = batch.nbf, matrix = n * n, quartets = matrix * matrix;
  const std::size_t work_count = system_count * quartets;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t work = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work < work_count; work += stride) {
    const std::size_t local_system = work / quartets;
    const auto system = static_cast<std::int32_t>(system_begin + local_system);
    const std::size_t quartet = work % quartets;
    const std::size_t offset = static_cast<std::size_t>(system) * matrix;
    const std::size_t ij = quartet / matrix, kl = quartet % matrix;
    if (bounds[offset + ij] * bounds[offset + kl] < screening) continue;

    const auto i = static_cast<std::int32_t>(ij / n), j = static_cast<std::int32_t>(ij % n);
    const auto k = static_cast<std::int32_t>(kl / n), l = static_cast<std::int32_t>(kl % n);
    const double total_ij = density[offset + ij] + (unrestricted ? beta[offset + ij] : 0.0);
    const double total_kl = density[offset + kl] + (unrestricted ? beta[offset + kl] : 0.0);
    const double j_weight = 0.5 * cj * total_ij * total_kl;
    const std::size_t ik = static_cast<std::size_t>(i) * n + k;
    const std::size_t jl = static_cast<std::size_t>(j) * n + l;
    const double exchange = density[offset + ik] * density[offset + jl] +
                            (unrestricted ? beta[offset + ik] * beta[offset + jl] : 0.0);
    const double short_weight = 0.5 * short_ck * exchange;
    const double long_weight = 0.5 * long_ck * exchange;
    if (j_weight == 0.0 && short_weight == 0.0 && long_weight == 0.0) continue;

    const std::size_t base = static_cast<std::size_t>(system) * n;
    const std::int32_t center_atoms[4] = {
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(i)]],
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(j)]],
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(k)]],
        batch.shell_atoms[batch.ao_shells[base + static_cast<std::size_t>(l)]],
    };
    std::int32_t unique_atoms[4];
    unsigned unique_count = 0;
    for (unsigned center = 0; center < 4; ++center) {
      bool duplicate = false;
      for (unsigned previous = 0; previous < unique_count; ++previous)
        duplicate = duplicate || unique_atoms[previous] == center_atoms[center];
      if (!duplicate) unique_atoms[unique_count++] = center_atoms[center];
    }
    if (unique_count <= 1) continue;

    double reconstructed[3][3]{};
    for (unsigned center = 0; center + 1 < unique_count; ++center) {
      const std::int64_t coordinate = static_cast<std::int64_t>(unique_atoms[center]) * 3;
      double full[3]{}, long_range[3]{};
      if (j_weight != 0.0 || short_weight != 0.0) {
        const Dual3 value = contracted_eri<Dual3>(batch, system, i, j, k, l, coordinate);
        full[0] = value.derivative_x;
        full[1] = value.derivative_y;
        full[2] = value.derivative_z;
      }
      if (short_weight != 0.0 || long_weight != 0.0) {
        const Dual3 value =
            contracted_eri<Dual3>(batch, system, i, j, k, l, coordinate,
                                  vibeqc::integrals::CoulombRange::Long, omega);
        long_range[0] = value.derivative_x;
        long_range[1] = value.derivative_y;
        long_range[2] = value.derivative_z;
      }
      for (unsigned axis = 0; axis < 3; ++axis) {
        const double source[3] = {
            j_weight * full[axis],
            short_weight * (full[axis] - long_range[axis]),
            long_weight * long_range[axis],
        };
        for (unsigned term = 0; term < 3; ++term) {
          reconstructed[term][axis] += source[term];
          if (source[term] != 0.0)
            atomicAdd(out + term * source_stride + static_cast<std::size_t>(coordinate) + axis,
                      source[term]);
        }
      }
    }
    const std::size_t final_coordinate =
        static_cast<std::size_t>(unique_atoms[unique_count - 1]) * 3;
    for (unsigned term = 0; term < 3; ++term)
      for (unsigned axis = 0; axis < 3; ++axis)
        if (reconstructed[term][axis] != 0.0)
          atomicAdd(out + term * source_stride + final_coordinate + axis,
                    -reconstructed[term][axis]);
  }
}

}  // namespace

namespace cuda_execution {

void launch_independent_jk_finite_kernel(cudaStream_t stream, const double* values,
                                         std::size_t count, int* failure) {
  const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((count + 127) / 128, 65535));
  independent_jk_finite_kernel<<<blocks, 128, 0, stream>>>(values, count, failure);
}

void launch_independent_jk_bounds_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, DeviceBatch batch, double* bounds,
                                         int* failure) {
  independent_jk_bounds_kernel<<<grid, block, shared_bytes, stream>>>(batch, bounds, failure);
}

void launch_independent_jk_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, DeviceBatch batch, std::size_t system_begin,
                                  bool want_j, bool want_k, bool unrestricted, bool mixed_j,
                                  DirectCoulombRange exchange_range, double exchange_omega,
                                  double screening, const double* bounds, const double* density,
                                  const double* beta, double* j_out, double* ka_out,
                                  double* kb_out) {
  if (mixed_j)
    independent_jk_kernel<true><<<grid, block, shared_bytes, stream>>>(
        batch, system_begin, want_j, want_k, unrestricted, integral_range(exchange_range),
        exchange_omega, screening, bounds, density, beta, j_out, ka_out, kb_out);
  else
    independent_jk_kernel<false><<<grid, block, shared_bytes, stream>>>(
        batch, system_begin, want_j, want_k, unrestricted, integral_range(exchange_range),
        exchange_omega, screening, bounds, density, beta, j_out, ka_out, kb_out);
}

void launch_independent_jk_derivative_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t coordinates_per_item, std::size_t system_begin, double cj, double ck,
    bool unrestricted, DirectCoulombRange exchange_range, double exchange_omega, double screening,
    const double* bounds, const double* density, const double* beta, double* out) {
  const std::size_t matrix = static_cast<std::size_t>(batch.nbf) * batch.nbf;
  const std::size_t system_count =
      coordinates_per_item == 0 ? 0 : static_cast<std::size_t>(grid.x) / coordinates_per_item;
  const std::size_t quartet_count = system_count * matrix * matrix;
  const unsigned blocks =
      static_cast<unsigned>(std::min<std::size_t>((quartet_count + block.x - 1) / block.x, 65535));
  if (blocks == 0) return;
  independent_jk_derivative_kernel<<<blocks, block, shared_bytes, stream>>>(
      batch, system_begin, system_count, cj, ck, unrestricted, integral_range(exchange_range),
      exchange_omega, screening, bounds, density, beta, out);
}

void launch_independent_rsh_derivative_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t coordinates_per_item, std::size_t system_begin, std::size_t source_stride,
    double cj, double short_ck, double long_ck, bool unrestricted, double omega, double screening,
    const double* bounds, const double* density, const double* beta, double* out) {
  const std::size_t matrix = static_cast<std::size_t>(batch.nbf) * batch.nbf;
  const std::size_t system_count =
      coordinates_per_item == 0 ? 0 : static_cast<std::size_t>(grid.x) / coordinates_per_item;
  const std::size_t quartet_count = system_count * matrix * matrix;
  const unsigned blocks =
      static_cast<unsigned>(std::min<std::size_t>((quartet_count + block.x - 1) / block.x, 65535));
  if (blocks == 0) return;
  independent_rsh_derivative_kernel<<<blocks, block, shared_bytes, stream>>>(
      batch, system_begin, system_count, source_stride, cj, short_ck, long_ck, unrestricted, omega,
      screening, bounds, density, beta, out);
}

}  // namespace cuda_execution

}  // namespace vibeqc::scf

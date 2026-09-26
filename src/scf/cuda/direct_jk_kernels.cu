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
 * Relabel exchange indices so a single ERI derivative serves J and both K
 * terms. The 1/2 energy factor is separate from signed Fock coefficients.
 */
__global__ void independent_jk_derivative_kernel(
    DeviceBatch batch, std::size_t coordinates_per_item, std::size_t system_begin, unsigned chunks,
    double cj, double ck, bool unrestricted, vibeqc::integrals::CoulombRange exchange_range,
    double exchange_omega, double screening, const double* bounds, const double* density,
    const double* beta, double* out) {
  __shared__ double sums[kIndependentJkThreads];
  const std::size_t n = batch.nbf, matrix = n * n, quartets = matrix * matrix;
  const std::size_t coordinate = system_begin * coordinates_per_item + blockIdx.x / chunks;
  const auto system = static_cast<std::int32_t>(coordinate / coordinates_per_item);
  const std::size_t offset = static_cast<std::size_t>(system) * matrix;
  double sum = 0.0;
  // Several blocks share a coordinate so small molecules can occupy the GPU.
  // Each quartet still belongs to exactly one lane; only the bounded block
  // sums are atomically accumulated into the zeroed coordinate output.
  for (std::size_t quartet = (blockIdx.x % chunks) * blockDim.x + threadIdx.x; quartet < quartets;
       quartet += static_cast<std::size_t>(chunks) * blockDim.x) {
    const std::size_t ij = quartet / matrix, kl = quartet % matrix;
    if (bounds[offset + ij] * bounds[offset + kl] < screening) continue;
    const auto i = static_cast<std::int32_t>(ij / n), j = static_cast<std::int32_t>(ij % n);
    const auto k = static_cast<std::int32_t>(kl / n), l = static_cast<std::int32_t>(kl % n);
    // A coordinate absent from the quartet has an exactly zero derivative.
    // Reject it before the primitive recurrence, including spherical expansion.
    const auto atom = static_cast<std::int32_t>(coordinate / 3);
    const auto base = static_cast<std::size_t>(system) * n;
    if (batch.shell_atoms[batch.ao_shells[base + i]] != atom &&
        batch.shell_atoms[batch.ao_shells[base + j]] != atom &&
        batch.shell_atoms[batch.ao_shells[base + k]] != atom &&
        batch.shell_atoms[batch.ao_shells[base + l]] != atom)
      continue;
    double weight = 0.0, range_weight = 0.0;
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
      if (exchange_range == vibeqc::integrals::CoulombRange::Full)
        weight += ck * exchange;
      else
        range_weight = 0.5 * ck * exchange;
    }
    weight *= 0.5;
    if (weight != 0.0)
      sum += weight *
             contracted_eri<Dual>(batch, system, i, j, k, l, static_cast<std::int64_t>(coordinate))
                 .derivative;
    if (range_weight != 0.0)
      sum += range_weight * contracted_eri<Dual>(batch, system, i, j, k, l,
                                                 static_cast<std::int64_t>(coordinate),
                                                 exchange_range, exchange_omega)
                                .derivative;
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out + coordinate, sums[0]);
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
  const unsigned chunks = static_cast<unsigned>(std::min<std::size_t>(
      {256, (matrix * matrix + block.x - 1) / block.x, 2147483647U / grid.x}));
  const dim3 parallel_grid(grid.x * chunks);
  independent_jk_derivative_kernel<<<parallel_grid, block, shared_bytes, stream>>>(
      batch, coordinates_per_item, system_begin, chunks, cj, ck, unrestricted,
      integral_range(exchange_range), exchange_omega, screening, bounds, density, beta, out);
}

}  // namespace cuda_execution

}  // namespace vibeqc::scf

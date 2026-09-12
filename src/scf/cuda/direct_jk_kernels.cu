#include <cmath>

#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_native_contraction.cuh"

namespace vibeqc::scf {
namespace {
using namespace cuda_execution;
}

namespace {

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

}  // namespace

namespace cuda_execution {

void launch_independent_jk_bounds_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, DeviceBatch batch, double* bounds,
                                         int* failure) {
  independent_jk_bounds_kernel<<<grid, block, shared_bytes, stream>>>(batch, bounds, failure);
}

void launch_independent_jk_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, DeviceBatch batch, std::size_t system_begin,
                                  bool want_j, bool want_k, bool unrestricted, double screening,
                                  const double* bounds, const double* density, const double* beta,
                                  double* j_out, double* ka_out, double* kb_out) {
  independent_jk_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, system_begin, want_j, want_k, unrestricted, screening, bounds, density, beta, j_out,
      ka_out, kb_out);
}

void launch_independent_jk_derivative_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                             cudaStream_t stream, DeviceBatch batch,
                                             std::size_t coordinates_per_item,
                                             std::size_t system_begin, double cj, double ck,
                                             bool unrestricted, double screening,
                                             const double* bounds, const double* density,
                                             const double* beta, double* out) {
  independent_jk_derivative_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, coordinates_per_item, system_begin, cj, ck, unrestricted, screening, bounds, density,
      beta, out);
}

}  // namespace cuda_execution

}  // namespace vibeqc::scf

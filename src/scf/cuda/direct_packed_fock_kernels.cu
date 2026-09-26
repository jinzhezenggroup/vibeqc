#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_native_contraction.cuh"
#include "scf/cuda/direct_packed_fock_kernels.hpp"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void build_fock_direct_packed_kernel(
    DeviceBatch batch, double screening_tolerance, const double* hcore,
    const std::int32_t* ao_pair_first, const std::int32_t* ao_pair_second, std::size_t pair_count,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock) {
  extern __shared__ double pair_sums[];
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t matrix_element = static_cast<std::size_t>(blockIdx.x);
  if (matrix_element >= static_cast<std::size_t>(batch.batch_size) * matrix_size) {
    return;
  }
  const std::int32_t system = static_cast<std::int32_t>(matrix_element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local_matrix = matrix_element % matrix_size;
  const std::size_t i = local_matrix % n;
  const std::size_t j = local_matrix / n;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double bound_ij = schwarz_bounds[matrix_offset + matrix_index(i, j, n)];

  double contribution = 0.0;
  for (std::size_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
    const std::size_t k = static_cast<std::size_t>(ao_pair_first[pair]);
    const std::size_t l = static_cast<std::size_t>(ao_pair_second[pair]);
    const double pkl = density[matrix_offset + matrix_index(k, l, n)];
    if (pkl == 0.0) continue;

    if (bound_ij * schwarz_bounds[matrix_offset + matrix_index(k, l, n)] >= screening_tolerance) {
      const double pair_weight = k == l ? pkl : 2.0 * pkl;
      contribution += pair_weight * contracted_eri<double>(
                                        batch, system, static_cast<std::int32_t>(i),
                                        static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
                                        static_cast<std::int32_t>(l), -1);
    }
    if (schwarz_bounds[matrix_offset + matrix_index(i, k, n)] *
            schwarz_bounds[matrix_offset + matrix_index(j, l, n)] >=
        screening_tolerance) {
      contribution -= 0.5 * pkl *
                      contracted_eri<double>(
                          batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(k),
                          static_cast<std::int32_t>(j), static_cast<std::int32_t>(l), -1);
    }
    if (k != l && schwarz_bounds[matrix_offset + matrix_index(i, l, n)] *
                          schwarz_bounds[matrix_offset + matrix_index(j, k, n)] >=
                      screening_tolerance) {
      contribution -= 0.5 * pkl *
                      contracted_eri<double>(
                          batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(l),
                          static_cast<std::int32_t>(j), static_cast<std::int32_t>(k), -1);
    }
  }

  pair_sums[threadIdx.x] = contribution;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      pair_sums[threadIdx.x] += pair_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    fock[matrix_element] = hcore[matrix_element] + pair_sums[0];
  }
}

__global__ void build_uhf_fock_direct_packed_kernel(
    DeviceBatch batch, double screening_tolerance, const double* hcore,
    const std::int32_t* ao_pair_first, const std::int32_t* ao_pair_second, std::size_t pair_count,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock) {
  extern __shared__ double pair_sums[];
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t matrix_element = static_cast<std::size_t>(blockIdx.x);
  if (matrix_element >= static_cast<std::size_t>(batch.batch_size) * 2 * matrix_size) {
    return;
  }
  const std::size_t state = matrix_element / matrix_size;
  const std::size_t system = state / 2;
  if (active != nullptr && active[system] == 0) return;
  const std::size_t spin = state % 2;
  const std::size_t local_matrix = matrix_element % matrix_size;
  const std::size_t i = local_matrix % n;
  const std::size_t j = local_matrix / n;
  const std::size_t physical_offset = system * matrix_size;
  const std::size_t alpha_offset = system * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t spin_offset = alpha_offset + spin * matrix_size;
  const double bound_ij = schwarz_bounds[physical_offset + matrix_index(i, j, n)];

  double contribution = 0.0;
  for (std::size_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
    const std::size_t k = static_cast<std::size_t>(ao_pair_first[pair]);
    const std::size_t l = static_cast<std::size_t>(ao_pair_second[pair]);
    const std::size_t kl = matrix_index(k, l, n);
    const double alpha = density[alpha_offset + kl];
    const double beta = density[beta_offset + kl];
    const double same_spin = density[spin_offset + kl];
    const double total = alpha + beta;
    if (total == 0.0 && same_spin == 0.0) continue;

    if (total != 0.0 && bound_ij * schwarz_bounds[physical_offset + kl] >= screening_tolerance) {
      const double pair_weight = k == l ? total : 2.0 * total;
      contribution += pair_weight * contracted_eri<double>(batch, static_cast<std::int32_t>(system),
                                                           static_cast<std::int32_t>(i),
                                                           static_cast<std::int32_t>(j),
                                                           static_cast<std::int32_t>(k),
                                                           static_cast<std::int32_t>(l), -1);
    }
    if (same_spin != 0.0 && schwarz_bounds[physical_offset + matrix_index(i, k, n)] *
                                    schwarz_bounds[physical_offset + matrix_index(j, l, n)] >=
                                screening_tolerance) {
      contribution -= same_spin * contracted_eri<double>(batch, static_cast<std::int32_t>(system),
                                                         static_cast<std::int32_t>(i),
                                                         static_cast<std::int32_t>(k),
                                                         static_cast<std::int32_t>(j),
                                                         static_cast<std::int32_t>(l), -1);
    }
    if (k != l && same_spin != 0.0 &&
        schwarz_bounds[physical_offset + matrix_index(i, l, n)] *
                schwarz_bounds[physical_offset + matrix_index(j, k, n)] >=
            screening_tolerance) {
      contribution -= same_spin * contracted_eri<double>(batch, static_cast<std::int32_t>(system),
                                                         static_cast<std::int32_t>(i),
                                                         static_cast<std::int32_t>(l),
                                                         static_cast<std::int32_t>(j),
                                                         static_cast<std::int32_t>(k), -1);
    }
  }

  pair_sums[threadIdx.x] = contribution;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      pair_sums[threadIdx.x] += pair_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    fock[matrix_element] = hcore[physical_offset + local_matrix] + pair_sums[0];
  }
}

void launch_build_fock_direct_packed_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    double screening_tolerance, const double* hcore, const std::int32_t* ao_pair_first,
    const std::int32_t* ao_pair_second, std::size_t pair_count, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock) {
  build_fock_direct_packed_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, screening_tolerance, hcore, ao_pair_first, ao_pair_second, pair_count, schwarz_bounds,
      density, active, fock);
}

void launch_build_uhf_fock_direct_packed_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    double screening_tolerance, const double* hcore, const std::int32_t* ao_pair_first,
    const std::int32_t* ao_pair_second, std::size_t pair_count, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock) {
  build_uhf_fock_direct_packed_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, screening_tolerance, hcore, ao_pair_first, ao_pair_second, pair_count, schwarz_bounds,
      density, active, fock);
}

}  // namespace vibeqc::scf::cuda_execution

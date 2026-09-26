#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "scf/cuda/df_jk_kernels.hpp"

namespace vibeqc::scf::cuda_df {

__global__ void mirror_exchange_triangle(std::size_t n, double* matrix) {
  const auto k = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (k >= n * n) return;
  const auto i = k % n, j = k / n;
  if (i > j) matrix[j + i * n] = matrix[k];
}

void launch_mirror_exchange_triangle(dim3 grid, dim3 block, cudaStream_t stream, std::size_t n,
                                     double* matrix) {
  mirror_exchange_triangle<<<grid, block, 0, stream>>>(n, matrix);
}

// Existing DF arithmetic and reduction order; host orchestration compiles separately.
__global__ void sum_spin_density_kernel(std::size_t elements, const double* alpha,
                                        const double* beta, double* total) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  total[element] = alpha[element] + beta[element];
}

__global__ void transpose_density_kernel(std::size_t dimension, const double* row_major,
                                         double* column_major) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t elements = dimension * dimension;
  if (element >= elements) return;
  const std::size_t row = element / dimension;
  const std::size_t column = element % dimension;
  column_major[row + column * dimension] = row_major[element];
}

__global__ void gather_auxiliary_tile_kernel(std::size_t matrix_elements, std::size_t naux,
                                             std::size_t system, std::size_t auxiliary_begin,
                                             std::size_t auxiliary_count,
                                             const double* three_center, double* tile) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t tile_elements = auxiliary_count * matrix_elements;
  if (element >= tile_elements) return;
  const std::size_t auxiliary = element / matrix_elements;
  const std::size_t pair = element % matrix_elements;
  tile[element] =
      three_center[(system * matrix_elements + pair) * naux + auxiliary_begin + auxiliary];
}

__global__ void transpose_streamed_df_tile_kernel(std::size_t pair_count,
                                                  std::size_t auxiliary_count,
                                                  const double* pair_major,
                                                  double* auxiliary_major) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = pair_count * auxiliary_count;
  if (element >= total) return;
  const std::size_t pair = element / auxiliary_count;
  const std::size_t auxiliary = element % auxiliary_count;
  auxiliary_major[auxiliary * pair_count + pair] = pair_major[pair * auxiliary_count + auxiliary];
}

__global__ void accumulate_streamed_auxiliary_density_kernel(std::size_t pair_count,
                                                             std::size_t auxiliary_count,
                                                             const double* tile,
                                                             const double* density,
                                                             double* auxiliary_density) {
  const std::size_t auxiliary = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (auxiliary >= auxiliary_count) return;
  double value = 0.0;
  for (std::size_t pair = 0; pair < pair_count; ++pair) {
    value += tile[pair * auxiliary_count + auxiliary] * density[pair];
  }
  auxiliary_density[auxiliary] += value;
}

__global__ void build_streamed_coulomb_tile_kernel(std::size_t pair_count,
                                                   std::size_t auxiliary_count, const double* tile,
                                                   const double* auxiliary_density,
                                                   double* coulomb) {
  const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pair_count) return;
  double value = 0.0;
  for (std::size_t auxiliary = 0; auxiliary < auxiliary_count; ++auxiliary) {
    value += tile[pair * auxiliary_count + auxiliary] * auxiliary_density[auxiliary];
  }
  // Auxiliary tiles are visited sequentially; retain the partial sum from
  // earlier tiles while each AO-pair segment remains disjoint.
  coulomb[pair] += value;
}

__global__ void reduce_exchange_tile_kernel(std::size_t matrix_elements,
                                            std::size_t auxiliary_count, std::size_t system,
                                            const double* contributions, double* exchange,
                                            bool continue_sum) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_elements) return;
  auto& output = exchange[system * matrix_elements + element];
  // Resident panels split storage, not the reduction: continue the running
  // Q sum so the original full-tensor addition order survives each boundary.
  double value = continue_sum ? output : 0.0;
  for (std::size_t auxiliary = 0; auxiliary < auxiliary_count; ++auxiliary) {
    value += contributions[auxiliary * matrix_elements + element];
  }
  output = continue_sum ? value : output + value;
}

__global__ void reduce_exchange_row_tile_kernel(std::size_t nbf, std::size_t row_begin,
                                                std::size_t row_count, std::size_t column_begin,
                                                std::size_t column_count,
                                                std::size_t auxiliary_count, std::size_t system,
                                                const double* contributions, double* exchange) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t tile_elements = row_count * column_count;
  if (element >= tile_elements) return;
  const std::size_t row = element / column_count;
  const std::size_t column = element % column_count;
  double value = 0.0;
  for (std::size_t auxiliary = 0; auxiliary < auxiliary_count; ++auxiliary) {
    // cuBLAS writes the row_count x column_count result column-major.  Read
    // it as [column][row] while scattering the public row-major [row][column]
    // tile, preserving the orientation for non-symmetric test densities.
    value += contributions[auxiliary * tile_elements + column * row_count + row];
  }
  exchange[system * nbf * nbf + (row_begin + row) * nbf + column_begin + column] += value;
}

void launch_sum_spin_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                    cudaStream_t stream, std::size_t elements, const double* alpha,
                                    const double* beta, double* total) {
  sum_spin_density_kernel<<<grid, block, shared_bytes, stream>>>(elements, alpha, beta, total);
}
void launch_transpose_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                     cudaStream_t stream, std::size_t dimension,
                                     const double* row_major, double* column_major) {
  transpose_density_kernel<<<grid, block, shared_bytes, stream>>>(dimension, row_major,
                                                                  column_major);
}
void launch_gather_auxiliary_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::size_t matrix_elements,
                                         std::size_t naux, std::size_t system,
                                         std::size_t auxiliary_begin, std::size_t auxiliary_count,
                                         const double* three_center, double* tile) {
  gather_auxiliary_tile_kernel<<<grid, block, shared_bytes, stream>>>(
      matrix_elements, naux, system, auxiliary_begin, auxiliary_count, three_center, tile);
}
void launch_transpose_streamed_df_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                              cudaStream_t stream, std::size_t pair_count,
                                              std::size_t auxiliary_count, const double* pair_major,
                                              double* auxiliary_major) {
  transpose_streamed_df_tile_kernel<<<grid, block, shared_bytes, stream>>>(
      pair_count, auxiliary_count, pair_major, auxiliary_major);
}
void launch_accumulate_streamed_auxiliary_density_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::size_t pair_count,
    std::size_t auxiliary_count, const double* tile, const double* density,
    double* auxiliary_density) {
  accumulate_streamed_auxiliary_density_kernel<<<grid, block, shared_bytes, stream>>>(
      pair_count, auxiliary_count, tile, density, auxiliary_density);
}
void launch_build_streamed_coulomb_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                               cudaStream_t stream, std::size_t pair_count,
                                               std::size_t auxiliary_count, const double* tile,
                                               const double* auxiliary_density, double* coulomb) {
  build_streamed_coulomb_tile_kernel<<<grid, block, shared_bytes, stream>>>(
      pair_count, auxiliary_count, tile, auxiliary_density, coulomb);
}
void launch_reduce_exchange_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                        cudaStream_t stream, std::size_t matrix_elements,
                                        std::size_t auxiliary_count, std::size_t system,
                                        const double* contributions, double* exchange,
                                        bool continue_sum) {
  reduce_exchange_tile_kernel<<<grid, block, shared_bytes, stream>>>(
      matrix_elements, auxiliary_count, system, contributions, exchange, continue_sum);
}
void launch_reduce_exchange_row_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                            cudaStream_t stream, std::size_t nbf,
                                            std::size_t row_begin, std::size_t row_count,
                                            std::size_t column_begin, std::size_t column_count,
                                            std::size_t auxiliary_count, std::size_t system,
                                            const double* contributions, double* exchange) {
  reduce_exchange_row_tile_kernel<<<grid, block, shared_bytes, stream>>>(
      nbf, row_begin, row_count, column_begin, column_count, auxiliary_count, system, contributions,
      exchange);
}
}  // namespace vibeqc::scf::cuda_df

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::cuda_df {

/** Mirror the computed column-major lower triangle; never average with the
 * uninitialized upper triangle of a BLAS symmetric rank-k result. */
void launch_mirror_exchange_triangle(dim3 grid, dim3 block, cudaStream_t stream, std::size_t n,
                                     double* matrix);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_sum_spin_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                    cudaStream_t stream, std::size_t elements, const double* alpha,
                                    const double* beta, double* total);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_transpose_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                     cudaStream_t stream, std::size_t dimension,
                                     const double* row_major, double* column_major);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_gather_auxiliary_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::size_t matrix_elements,
                                         std::size_t naux, std::size_t system,
                                         std::size_t auxiliary_begin, std::size_t auxiliary_count,
                                         const double* three_center, double* tile);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_transpose_streamed_df_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                              cudaStream_t stream, std::size_t pair_count,
                                              std::size_t auxiliary_count, const double* pair_major,
                                              double* auxiliary_major);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_accumulate_streamed_auxiliary_density_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::size_t pair_count,
    std::size_t auxiliary_count, const double* tile, const double* density,
    double* auxiliary_density);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_build_streamed_coulomb_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                               cudaStream_t stream, std::size_t pair_count,
                                               std::size_t auxiliary_count, const double* tile,
                                               const double* auxiliary_density, double* coulomb);

/** Forward the launch on its existing stream. continue_sum carries the
 * running auxiliary sum across storage panels without reassociating it. */
void launch_reduce_exchange_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                        cudaStream_t stream, std::size_t matrix_elements,
                                        std::size_t auxiliary_count, std::size_t system,
                                        const double* contributions, double* exchange,
                                        bool continue_sum = false);

/** Forward the caller's exact launch configuration on its existing stream. */
void launch_reduce_exchange_row_tile_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                            cudaStream_t stream, std::size_t nbf,
                                            std::size_t row_begin, std::size_t row_count,
                                            std::size_t column_begin, std::size_t column_count,
                                            std::size_t auxiliary_count, std::size_t system,
                                            const double* contributions, double* exchange);

}  // namespace vibeqc::scf::cuda_df

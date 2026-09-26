#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::cuda_execution {

/** Preserve launch geometry, stream and per-item state routing. */
void launch_build_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                 cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                 const std::int32_t* occupied, const double* coefficients,
                                 const std::uint8_t* active, double* density);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_build_spin_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::int32_t batch_size,
                                      std::int32_t spin_count, std::int32_t nbf,
                                      const std::int32_t* occupied, const double* coefficients,
                                      const std::uint8_t* active, double* density);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_mix_open_shell_guess_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                        cudaStream_t stream, std::int32_t batch_size,
                                        std::int32_t nbf, const std::int32_t* occupied,
                                        const std::uint8_t* active, double* coefficients);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_apply_warm_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, std::int32_t batch_size,
                                      std::int32_t nbf, const std::int32_t* occupied,
                                      const std::uint8_t* warm_mask, const double* warm_density,
                                      const double* overlap, double* density,
                                      std::uint8_t* warm_invalid);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_apply_uhf_warm_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          std::int32_t nbf, const std::int32_t* occupied,
                                          const std::uint8_t* warm_mask, const double* warm_density,
                                          const double* overlap, double* density,
                                          std::uint8_t* warm_invalid);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_build_weighted_density_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          std::int32_t nbf, const std::int32_t* occupied,
                                          const double* coefficients,
                                          const double* orbital_energies,
                                          const std::uint8_t* active, double* weighted_density);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_build_spin_weighted_density_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t nbf, const std::int32_t* occupied, const double* coefficients,
    const double* orbital_energies, const std::uint8_t* active, double* weighted_density);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_sum_uhf_spin_matrices_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::int32_t batch_size,
                                         std::int32_t nbf, const double* spin_matrices,
                                         const std::uint8_t* active, double* total_matrices);

}  // namespace vibeqc::scf::cuda_execution

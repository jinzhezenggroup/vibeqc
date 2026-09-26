#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf::cuda_execution {

/** Preserve launch geometry, stream and per-item state routing. */
void launch_initialize_direct_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, std::int32_t batch_size,
                                          std::int32_t matrices_per_system, std::int32_t nbf,
                                          const double* hcore, const std::uint8_t* active,
                                          double* fock);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_transform_density_to_direct_right_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t spin_count, std::int32_t nbf, std::int32_t direct_nbf, const double* transform,
    const double* density, const std::uint8_t* active, double* temporary);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_transform_density_to_direct_left_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t spin_count, std::int32_t nbf, std::int32_t direct_nbf, const double* transform,
    const double* temporary, const std::uint8_t* active, double* direct_density);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_transform_direct_fock_left_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                              cudaStream_t stream, std::int32_t batch_size,
                                              std::int32_t spin_count, std::int32_t nbf,
                                              std::int32_t direct_nbf, const double* transform,
                                              const double* direct_fock, const std::uint8_t* active,
                                              double* temporary);

/** Preserve launch geometry, stream and per-item state routing. */
void launch_transform_direct_fock_right_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                               cudaStream_t stream, std::int32_t batch_size,
                                               std::int32_t spin_count, std::int32_t nbf,
                                               std::int32_t direct_nbf, const double* transform,
                                               const double* temporary, const double* hcore,
                                               const std::uint8_t* active, double* fock);

}  // namespace vibeqc::scf::cuda_execution

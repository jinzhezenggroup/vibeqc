#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_reduce_shell_pair_density_bounds_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, const double* density, const std::uint8_t* active,
    ShellPairDensityBounds* shell_pair_density_bounds);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_reduce_bounded_shell_pair_block_bounds_kernel(dim3 grid, dim3 block,
                                                          std::size_t shared_bytes,
                                                          cudaStream_t stream, DeviceBatch batch,
                                                          const std::uint32_t* shell_pair_order,
                                                          const double* shell_pair_bounds,
                                                          double* shell_pair_block_bounds);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_reduce_bounded_system_density_bounds_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const ShellPairDensityBounds* shell_pair_density_bounds, double* system_density_bounds,
    double* system_pair_density_bounds);

}  // namespace vibeqc::scf::cuda_execution

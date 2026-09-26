#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_schwarz_bounds_packed_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                               cudaStream_t stream, DeviceBatch batch,
                                               std::size_t pair_count, double* schwarz_bounds);

/** Forward the resolved native route with unchanged geometry and borrowed buffers. */
void launch_build_schwarz_and_shell_pair_bounds_packed_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t pair_count, double* schwarz_bounds, double* shell_pair_bounds);

}  // namespace vibeqc::scf::cuda_execution

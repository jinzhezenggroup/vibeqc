#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Preserve the retained kernel geometry, shared workspace and stream. */
void launch_build_nuclear_repulsion_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                           cudaStream_t stream, DeviceBatch batch,
                                           double* nuclear_repulsion);

/** Preserve the retained kernel geometry, shared workspace and stream. */
void launch_nuclear_force_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                 cudaStream_t stream, DeviceBatch batch, const std::uint8_t* active,
                                 double* forces);

}  // namespace vibeqc::scf::cuda_execution

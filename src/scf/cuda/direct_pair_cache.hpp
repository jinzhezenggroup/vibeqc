#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Preserve the retained kernel geometry, shared workspace and stream. */
void launch_build_shell_primitive_pair_cache_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                    cudaStream_t stream, DeviceBatch batch,
                                                    PrimitivePairData* shell_primitive_pairs);

}  // namespace vibeqc::scf::cuda_execution

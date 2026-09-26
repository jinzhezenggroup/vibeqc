#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_validate_direct_tile_descriptors_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                    cudaStream_t stream, DeviceBatch batch,
                                                    const std::uint32_t* active_tile_offsets,
                                                    const std::uint32_t* active_tile_counts,
                                                    const ActiveShellQuartetTile* active_tiles,
                                                    std::size_t total_tile_capacity,
                                                    DirectTileValidationRecord* record);

}  // namespace vibeqc::scf::cuda_execution

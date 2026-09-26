#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_clear_active_shell_quartet_tile_counts_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::uint32_t* active_shell_quartet_tile_counts, std::uint32_t* persistent_fock_task_heads,
    std::uint32_t* fp32_shell_quartet_tile_counts, std::uint32_t* fp32_persistent_fock_task_heads);

/** Forward resolved queue policy; grid.y == batch_size selects system-major equal-count traversal.
 */
void launch_compact_active_shell_quartet_tiles_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint8_t* active, const std::uint32_t* active_shell_quartet_tile_offsets,
    std::uint32_t* active_shell_quartet_tile_counts,
    ActiveShellQuartetTile* active_shell_quartet_tiles, bool mixed_precision_enabled,
    double mixed_precision_cutoff_ceiling, double mixed_precision_budget_error,
    const std::uint32_t* mixed_precision_item_census,
    const std::uint32_t* fp32_shell_quartet_tile_offsets,
    std::uint32_t* fp32_shell_quartet_tile_counts,
    ActiveShellQuartetTile* fp32_shell_quartet_tiles);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_compact_generic_order5_tiles_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t generated_shell_class_mask,
    const std::uint64_t* generated_shell_class_mask_pointer, std::uint32_t* generic_tile_count,
    ActiveShellQuartetTile* generic_tiles);

}  // namespace vibeqc::scf::cuda_execution

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Materialize the bounded queue with unchanged geometry, stream and buffers.
 * This launch exposes only the materializing specializations used by the
 * driver. Per-class overflow remains visible for exact-class page recovery.
 */
void launch_compact_bounded_generated_tasks_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint8_t* active,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    std::uint64_t excluded_mask, const std::uint32_t* selected_classes,
    const std::uint32_t* selected_any, unsigned long long* global_cursor, GeneratedShellTask* tasks,
    std::uint32_t* task_counts, const std::uint32_t* task_offsets, std::uint32_t* overflow);

}  // namespace vibeqc::scf::cuda_execution

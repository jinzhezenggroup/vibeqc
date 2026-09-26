#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_classify_generated_shell_tasks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t total_tile_capacity, const std::uint32_t* active_shell_quartet_tile_offsets,
    const std::uint32_t* active_shell_quartet_tile_counts,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t enabled_shell_class_mask, const std::uint64_t* enabled_shell_class_mask_pointer,
    bool exclude_resident_ppps, std::uint32_t* generated_task_counts,
    std::uint8_t* generated_shell_classes, std::uint64_t low_order_signature_mask,
    std::uint32_t* low_order_signature_counts);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_prefix_generated_shell_task_counts_kernel(dim3 grid, dim3 block,
                                                      std::size_t shared_bytes, cudaStream_t stream,
                                                      const std::uint32_t* generated_task_counts,
                                                      std::uint32_t* generated_task_offsets,
                                                      std::uint32_t* generated_task_write_counts,
                                                      std::uint32_t* generated_task_heads);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_prefix_low_order_signature_counts_kernel(dim3 grid, dim3 block,
                                                     std::size_t shared_bytes, cudaStream_t stream,
                                                     const std::uint32_t* generated_task_offsets,
                                                     std::uint64_t low_order_signature_mask,
                                                     std::uint32_t* low_order_signature_counts,
                                                     std::uint32_t* low_order_signature_offsets);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_materialize_generated_shell_tasks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t total_tile_capacity, const ActiveShellQuartetTile* active_shell_quartet_tiles,
    const std::uint8_t* generated_shell_classes, const std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_write_counts, GeneratedShellTask* generated_tasks,
    std::uint64_t low_order_signature_mask, const std::uint32_t* low_order_signature_offsets,
    std::uint32_t* low_order_signature_write_counts);

}  // namespace vibeqc::scf::cuda_execution

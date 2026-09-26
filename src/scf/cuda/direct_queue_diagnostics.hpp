#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_accumulate_fock_precision_work_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                  cudaStream_t stream,
                                                  const std::uint32_t* page_count,
                                                  unsigned long long* total_count);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_start_bounded_fock_class_timer_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                  cudaStream_t stream, unsigned shell_class,
                                                  std::uint64_t* starts);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_finish_bounded_fock_class_timer_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                   cudaStream_t stream, unsigned shell_class,
                                                   const std::uint64_t* starts,
                                                   std::uint64_t* elapsed, std::uint32_t* launches);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_profile_active_shell_quartet_tiles_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t total_tile_capacity, const std::uint32_t* active_shell_quartet_tile_offsets,
    const std::uint32_t* active_shell_quartet_tile_counts,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    DeviceShellClassProfileEntry* profile);

/** Forward resolved queue policy with unchanged geometry, stream and buffers. */
void launch_profile_bounded_generated_tasks_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                   cudaStream_t stream, DeviceBatch batch,
                                                   const GeneratedShellTask* tasks,
                                                   const std::uint32_t* task_offset,
                                                   const std::uint32_t* task_count,
                                                   DeviceShellClassProfileEntry* profile);

}  // namespace vibeqc::scf::cuda_execution

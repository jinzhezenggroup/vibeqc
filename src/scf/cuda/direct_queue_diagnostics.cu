#include <cmath>

#include "scf/cuda/device_timer.cuh"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_queue_diagnostics.hpp"
#include "scf/cuda/direct_queue_profile.cuh"

namespace vibeqc::scf::cuda_execution {

/** Add one compacted page count to a profiling-only 64-bit accumulator. */
__global__ void accumulate_fock_precision_work_kernel(const std::uint32_t* page_count,
                                                      unsigned long long* total_count) {
  if (blockIdx.x == 0U && threadIdx.x == 0U && page_count != nullptr && total_count != nullptr) {
    atomicAdd(total_count, static_cast<unsigned long long>(*page_count));
  }
}

/** Record a stream-ordered timestamp immediately before one Fock class. */
__global__ void start_bounded_fock_class_timer_kernel(unsigned shell_class, std::uint64_t* starts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  starts[shell_class] = globaltimer_nanoseconds();
}

/** Accumulate exact stream time consumed by one bounded Fock class launch. */
__global__ void finish_bounded_fock_class_timer_kernel(unsigned shell_class,
                                                       const std::uint64_t* starts,
                                                       std::uint64_t* elapsed,
                                                       std::uint32_t* launches) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const std::uint64_t stop = globaltimer_nanoseconds();
  elapsed[shell_class] += stop - starts[shell_class];
  ++launches[shell_class];
}

/**
 * Summarize the exact tile list consumed by the final Fock and force kernels.
 *
 * The fixed grid walks topology capacity, but only slots below each compacted
 * angular partition's active count contribute. Profiling is opt-in, so these
 * atomics and the partition lookup never enter production timing runs.
 */
__global__ void profile_active_shell_quartet_tiles_kernel(
    DeviceBatch batch, std::size_t total_tile_capacity,
    const std::uint32_t* active_shell_quartet_tile_offsets,
    const std::uint32_t* active_shell_quartet_tile_counts,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    DeviceShellClassProfileEntry* profile) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= total_tile_capacity) return;

  unsigned angular_order = 0;
  while (angular_order + 1 < detail::kDirectQuartetAngularOrderCount &&
         slot >= active_shell_quartet_tile_offsets[angular_order + 1]) {
    ++angular_order;
  }
  const std::size_t partition_begin = active_shell_quartet_tile_offsets[angular_order];
  if (slot - partition_begin >= active_shell_quartet_tile_counts[angular_order]) {
    return;
  }

  const ActiveShellQuartetTile task = active_shell_quartet_tiles[slot];
  const std::int32_t first_shell = batch.shell_pair_first[task.first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[task.first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[task.second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[task.second_pair];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (shell_class >= detail::kDirectQuartetShellClassCount) return;

  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, task.first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, task.second_pair);
  const std::size_t ao_quartet_count = task.first_pair == task.second_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;
  const std::size_t tile_begin =
      static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize;
  if (tile_begin >= ao_quartet_count) return;
  const std::size_t tile_ao_quartets =
      min(detail::kDirectQuartetTileSize, ao_quartet_count - tile_begin);

  unsigned long long primitive_quartets = static_cast<unsigned long long>(tile_ao_quartets);
  const std::int32_t shells[4] = {first_shell, second_shell, third_shell, fourth_shell};
  for (const std::int32_t shell : shells) {
    primitive_quartets *= static_cast<unsigned long long>(batch.shell_primitive_offsets[shell + 1] -
                                                          batch.shell_primitive_offsets[shell]);
  }

  DeviceShellClassProfileEntry& entry = profile[shell_class];
  if (task.tile == 0) atomicAdd(&entry.shell_quartets, 1ULL);
  atomicAdd(&entry.tiles, 1ULL);
  atomicAdd(&entry.ao_quartets, static_cast<unsigned long long>(tile_ao_quartets));
  atomicAdd(&entry.primitive_quartets, primitive_quartets);
}

/** Profile a successfully materialized bounded generated queue once. */
__global__ void profile_bounded_generated_tasks_kernel(DeviceBatch batch,
                                                       const GeneratedShellTask* tasks,
                                                       const std::uint32_t* task_offset,
                                                       const std::uint32_t* task_count,
                                                       DeviceShellClassProfileEntry* profile) {
  const std::uint32_t count = *task_count;
  const std::uint32_t offset = *task_offset;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t task = blockIdx.x * blockDim.x + threadIdx.x; task < count; task += stride) {
    profile_bounded_direct_shell_quartet(
        batch, {tasks[offset + task].shell_pair[0], tasks[offset + task].shell_pair[1], 0U},
        profile);
  }
}

void launch_accumulate_fock_precision_work_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                  cudaStream_t stream,
                                                  const std::uint32_t* page_count,
                                                  unsigned long long* total_count) {
  accumulate_fock_precision_work_kernel<<<grid, block, shared_bytes, stream>>>(page_count,
                                                                               total_count);
}

void launch_start_bounded_fock_class_timer_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                  cudaStream_t stream, unsigned shell_class,
                                                  std::uint64_t* starts) {
  start_bounded_fock_class_timer_kernel<<<grid, block, shared_bytes, stream>>>(shell_class, starts);
}

void launch_finish_bounded_fock_class_timer_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                   cudaStream_t stream, unsigned shell_class,
                                                   const std::uint64_t* starts,
                                                   std::uint64_t* elapsed,
                                                   std::uint32_t* launches) {
  finish_bounded_fock_class_timer_kernel<<<grid, block, shared_bytes, stream>>>(shell_class, starts,
                                                                                elapsed, launches);
}

void launch_profile_active_shell_quartet_tiles_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t total_tile_capacity, const std::uint32_t* active_shell_quartet_tile_offsets,
    const std::uint32_t* active_shell_quartet_tile_counts,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    DeviceShellClassProfileEntry* profile) {
  profile_active_shell_quartet_tiles_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, total_tile_capacity, active_shell_quartet_tile_offsets,
      active_shell_quartet_tile_counts, active_shell_quartet_tiles, profile);
}

void launch_profile_bounded_generated_tasks_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                   cudaStream_t stream, DeviceBatch batch,
                                                   const GeneratedShellTask* tasks,
                                                   const std::uint32_t* task_offset,
                                                   const std::uint32_t* task_count,
                                                   DeviceShellClassProfileEntry* profile) {
  profile_bounded_generated_tasks_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, tasks, task_offset, task_count, profile);
}

}  // namespace vibeqc::scf::cuda_execution

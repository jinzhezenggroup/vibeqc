#include <cmath>

#include "scf/cuda/direct_bounded_tasks.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/direct_task_encoding.cuh"

namespace vibeqc::scf::cuda_execution {

/**
 * Materialize every enabled exact class in one hierarchical scan.
 *
 * Each class owns a fixed slice whose setup-time weight comes from shell-pair
 * angular histograms. Overflow is recorded per class so a later exact-class
 * page stream can recover only that class without discarding unrelated
 * generated routes or repeating a whole-topology integral evaluation.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose, bool Materialize>
__global__ void compact_bounded_generated_tasks_kernel(
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint8_t* active, const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask, std::uint64_t excluded_mask, const std::uint32_t* selected_classes,
    const std::uint32_t* selected_any, unsigned long long* global_cursor, GeneratedShellTask* tasks,
    std::uint32_t* task_counts, const std::uint32_t* task_offsets, std::uint32_t* overflow) {
  __shared__ unsigned long long block_quartet;
  if (selected_any != nullptr && *selected_any == 0U) return;
  const std::size_t total = static_cast<std::size_t>(batch.total_shell_pair_block_quartets);
  while (true) {
    if (threadIdx.x == 0) block_quartet = atomicAdd(global_cursor, 1ULL);
    __syncthreads();
    if (block_quartet >= total) return;

    const std::size_t packed_block_quartet = static_cast<std::size_t>(block_quartet);
    const std::int32_t system = shell_pair_block_quartet_system(batch, packed_block_quartet);
    if (active != nullptr && active[system] == 0) continue;
    const std::size_t local_block_quartet =
        packed_block_quartet -
        static_cast<std::size_t>(batch.system_shell_pair_block_quartet_offsets[system]);
    std::size_t first_block_local = 0;
    std::size_t second_block_local = 0;
    decode_lower_triangle(local_block_quartet, first_block_local, second_block_local);
    const std::size_t system_block_begin =
        static_cast<std::size_t>(batch.system_shell_pair_block_offsets[system]);
    const std::size_t first_block = system_block_begin + first_block_local;
    const std::size_t second_block = system_block_begin + second_block_local;
    if (!bounded_direct_block_pair_survives_screening<Purpose>(
            first_block, second_block, system, screening_tolerance, shell_pair_block_bounds,
            system_density_bounds)) {
      continue;
    }

    const std::size_t system_pair_begin =
        static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
    const std::size_t system_pair_end =
        static_cast<std::size_t>(batch.system_shell_pair_offsets[system + 1]);
    const std::size_t first_ordered_begin =
        system_pair_begin + first_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t second_ordered_begin =
        system_pair_begin + second_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t first_count =
        min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - first_ordered_begin);
    const std::size_t second_count =
        min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - second_ordered_begin);
    const bool same_block = first_block == second_block;
    const std::size_t candidate_count =
        same_block ? first_count * (first_count + 1) / 2 : first_count * second_count;
    for (std::size_t candidate = threadIdx.x; candidate < candidate_count;
         candidate += blockDim.x) {
      std::size_t first_local = 0;
      std::size_t second_local = 0;
      if (same_block) {
        decode_lower_triangle(candidate, first_local, second_local);
      } else {
        first_local = candidate / second_count;
        second_local = candidate % second_count;
      }
      const std::size_t first_pair = shell_pair_order[first_ordered_begin + first_local];
      const std::size_t second_pair = shell_pair_order[second_ordered_begin + second_local];
      if (!direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
              batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds)) {
        continue;
      }
      const std::int32_t first_shell = batch.shell_pair_first[first_pair];
      const std::int32_t second_shell = batch.shell_pair_second[first_pair];
      const std::int32_t third_shell = batch.shell_pair_first[second_pair];
      const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
      const unsigned shell_class = direct_quartet_shell_class_device(
          batch.shell_angular[first_shell], batch.shell_angular[second_shell],
          batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
      if (!bounded_generated_class_enabled(shell_class, enabled_mask_pointer, enabled_mask)) {
        continue;
      }
      if ((excluded_mask & (std::uint64_t{1} << shell_class)) != 0U) {
        continue;
      }
      if (selected_classes != nullptr && selected_classes[shell_class] == 0U) {
        continue;
      }
      if constexpr (Materialize) {
        const std::uint32_t class_slot = atomicAdd(task_counts + shell_class, 1U);
        const std::uint32_t class_capacity =
            task_offsets[shell_class + 1U] - task_offsets[shell_class];
        if (class_slot >= class_capacity) {
          atomicExch(overflow + shell_class, 1U);
          continue;
        }
        const std::uint32_t slot = task_offsets[shell_class] + class_slot;
        const ActiveShellQuartetTile tile{static_cast<std::uint32_t>(first_pair),
                                          static_cast<std::uint32_t>(second_pair), 0U};
        populate_generated_shell_task(batch, tile, tasks[slot]);
      } else {
        atomicAdd(task_counts + shell_class, 1U);
      }
    }
    __syncthreads();
  }
}

void launch_compact_bounded_generated_tasks_kernel(
    bool unrestricted, DirectScreeningPurpose purpose, dim3 grid, dim3 block,
    std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order, const double* shell_pair_block_bounds,
    const double* system_density_bounds, const std::uint8_t* active,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    std::uint64_t excluded_mask, const std::uint32_t* selected_classes,
    const std::uint32_t* selected_any, unsigned long long* global_cursor, GeneratedShellTask* tasks,
    std::uint32_t* task_counts, const std::uint32_t* task_offsets, std::uint32_t* overflow) {
  if (unrestricted == true) {
    if (purpose == DirectScreeningPurpose::Fock) {
      compact_bounded_generated_tasks_kernel<true, DirectScreeningPurpose::Fock, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds, active,
              enabled_mask_pointer, enabled_mask, excluded_mask, selected_classes, selected_any,
              global_cursor, tasks, task_counts, task_offsets, overflow);
    } else {
      compact_bounded_generated_tasks_kernel<true, DirectScreeningPurpose::Force, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds, active,
              enabled_mask_pointer, enabled_mask, excluded_mask, selected_classes, selected_any,
              global_cursor, tasks, task_counts, task_offsets, overflow);
    }
  } else {
    if (purpose == DirectScreeningPurpose::Fock) {
      compact_bounded_generated_tasks_kernel<false, DirectScreeningPurpose::Fock, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds, active,
              enabled_mask_pointer, enabled_mask, excluded_mask, selected_classes, selected_any,
              global_cursor, tasks, task_counts, task_offsets, overflow);
    } else {
      compact_bounded_generated_tasks_kernel<false, DirectScreeningPurpose::Force, true>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
              shell_pair_order, shell_pair_block_bounds, system_density_bounds, active,
              enabled_mask_pointer, enabled_mask, excluded_mask, selected_classes, selected_any,
              global_cursor, tasks, task_counts, task_offsets, overflow);
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution

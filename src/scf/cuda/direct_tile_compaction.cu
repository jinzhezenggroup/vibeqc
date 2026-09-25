#include <cmath>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/direct_tile_compaction.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void clear_active_shell_quartet_tile_counts_kernel(
    std::uint32_t* active_shell_quartet_tile_counts, std::uint32_t* persistent_fock_task_heads,
    std::uint32_t* fp32_shell_quartet_tile_counts, std::uint32_t* fp32_persistent_fock_task_heads) {
  const std::size_t order = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (order < detail::kDirectQuartetAngularOrderCount) {
    active_shell_quartet_tile_counts[order] = 0;
    if (fp32_shell_quartet_tile_counts != nullptr) {
      fp32_shell_quartet_tile_counts[order] = 0;
    }
  }
  // Reset queue state in the same captured Graph node as the active counts.
  // A separate tiny kernel here would be replayed for every SCF iteration.
  if (order < kPersistentFockAngularOrderCount) {
    persistent_fock_task_heads[order] = 0;
    if (fp32_persistent_fock_task_heads != nullptr) {
      fp32_persistent_fock_task_heads[order] = 0;
    }
  }
}

/**
 * Contribution cutoff for one item's FP32 tiles: the largest cutoff whose
 * worst-case accumulation `eps32 * cutoff * census` fits the reserved error.
 * A zero census keeps the whole item on the exact FP64 path, and an
 * item-agnostic diagnostic cutoff (no per-item budget) is returned unchanged.
 */
__device__ __forceinline__ double mixed_fock_item_cutoff(double cutoff_ceiling, double budget_error,
                                                         const std::uint32_t* item_census,
                                                         std::int32_t item) {
  if (!(cutoff_ceiling > 0.0) || item_census == nullptr) return 0.0;
  const std::uint32_t census = item_census[item];
  if (census == 0U) return 0.0;
  if (!(budget_error > 0.0)) return cutoff_ceiling;
  return fmin(cutoff_ceiling,
              budget_error / (kMixedPrecisionFloat32UnitRoundoff * static_cast<double>(census)));
}

template <bool Unrestricted, DirectScreeningPurpose Purpose>
__global__ void compact_active_shell_quartet_tiles_kernel(
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint8_t* active,
    const std::uint32_t* active_shell_quartet_tile_offsets,
    std::uint32_t* active_shell_quartet_tile_counts,
    ActiveShellQuartetTile* active_shell_quartet_tiles, bool mixed_precision_enabled,
    double mixed_precision_cutoff_ceiling, double mixed_precision_budget_error,
    const std::uint32_t* mixed_precision_item_census,
    const std::uint32_t* fp32_shell_quartet_tile_offsets,
    std::uint32_t* fp32_shell_quartet_tile_counts,
    ActiveShellQuartetTile* fp32_shell_quartet_tiles) {
  // An equal-quartet-count batch is launched system-major (grid.y == batch size).
  // Resolve ownership from blockIdx.y in that case instead of performing one
  // O(log(batch)) binary search for every shell quartet. Ragged batches keep
  // the flat traversal and its exact offset lookup.
  std::int32_t system = 0;
  std::size_t local_quartet = 0;
  const bool system_major =
      batch.batch_size > 1 && gridDim.y == static_cast<unsigned>(batch.batch_size);
  if (system_major) {
    system = static_cast<std::int32_t>(blockIdx.y);
    const std::size_t begin = static_cast<std::size_t>(batch.system_shell_quartet_offsets[system]);
    const std::size_t end =
        static_cast<std::size_t>(batch.system_shell_quartet_offsets[system + 1]);
    local_quartet = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_quartet >= end - begin) return;
  } else {
    const std::size_t shell_quartet =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (shell_quartet >= static_cast<std::size_t>(batch.total_shell_quartets)) return;
    system = shell_quartet_system(batch, shell_quartet);
    local_quartet =
        shell_quartet - static_cast<std::size_t>(batch.system_shell_quartet_offsets[system]);
  }
  if (active != nullptr && active[system] == 0) return;
  std::size_t first_pair_local = 0;
  std::size_t second_pair_local = 0;
  decode_lower_triangle(local_quartet, first_pair_local, second_pair_local);
  const std::size_t pair_begin = static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
  const std::size_t first_pair = pair_begin + first_pair_local;
  const std::size_t second_pair = pair_begin + second_pair_local;
  double contribution_bound = 0.0;
  if (!direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
          batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds,
          Purpose == DirectScreeningPurpose::Fock ? &contribution_bound : nullptr))
    return;

  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];

  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const std::size_t ao_quartet_count = first_pair == second_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;
  const std::uint32_t tile_count = static_cast<std::uint32_t>(
      (ao_quartet_count + detail::kDirectQuartetTileSize - 1) / detail::kDirectQuartetTileSize);
  const unsigned angular_order =
      batch.shell_angular[first_shell] + batch.shell_angular[second_shell] +
      batch.shell_angular[third_shell] + batch.shell_angular[fourth_shell];
  if (angular_order >= detail::kDirectQuartetAngularOrderCount) return;

  // Compaction expands each active shell quartet into only its populated AO
  // tiles inside a fixed angular-order partition. Exact shell-class dispatch
  // happens inside the consumer so Graph replay retains only 13 launch nodes.
  // Order within one partition need not be stable because consumers use
  // double atomics and promise numerical, rather than bitwise, replay.
  bool use_fp32 = false;
  if constexpr (Purpose == DirectScreeningPurpose::Fock) {
    // Low-order shell-fused workers remain FP64; routing them through the
    // generic evaluator would conflate precision with a scheduling regression.
    // The cutoff is per item: a system without a certified census keeps every
    // one of its tiles in the FP64 list regardless of its batch neighbors.
    const double item_cutoff =
        mixed_fock_item_cutoff(mixed_precision_cutoff_ceiling, mixed_precision_budget_error,
                               mixed_precision_item_census, system);
    use_fp32 = mixed_precision_enabled && item_cutoff > 0.0 &&
               angular_order >= kMixedFockMinimumAngularOrder &&
               fp32_shell_quartet_tile_counts != nullptr && contribution_bound < item_cutoff;
  }
  std::uint32_t* selected_counts =
      use_fp32 ? fp32_shell_quartet_tile_counts : active_shell_quartet_tile_counts;
  ActiveShellQuartetTile* selected_tiles =
      use_fp32 ? fp32_shell_quartet_tiles : active_shell_quartet_tiles;
  const std::uint32_t* selected_offsets =
      use_fp32 ? fp32_shell_quartet_tile_offsets : active_shell_quartet_tile_offsets;
  const std::uint32_t slot =
      selected_offsets[angular_order] + atomicAdd(selected_counts + angular_order, tile_count);
  for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
    selected_tiles[slot + tile] = {static_cast<std::uint32_t>(first_pair),
                                   static_cast<std::uint32_t>(second_pair), tile};
  }
}

/**
 * Compact the order-five fallback after excluding currently enabled AOT
 * classes. This stays separate from exact-class compaction so runtime masks
 * such as ``none``, ``dppp``, and ``all`` retain a correct generic fallback.
 */
__global__ void compact_generic_order5_tiles_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t generated_shell_class_mask,
    const std::uint64_t* generated_shell_class_mask_pointer, std::uint32_t* generic_tile_count,
    ActiveShellQuartetTile* generic_tiles) {
  // Fock graph replay uploads its runtime selection to device memory, while
  // the final force path supplies a host-resolved value outside the graph.
  if (generated_shell_class_mask_pointer != nullptr) {
    generated_shell_class_mask = *generated_shell_class_mask_pointer;
  }
  const std::size_t active_tile = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (active_tile >= static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const ActiveShellQuartetTile tile = active_shell_quartet_tiles[active_tile];
  const std::int32_t first_shell = batch.shell_pair_first[tile.first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[tile.first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[tile.second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[tile.second_pair];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (shell_class < 64U && (generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    return;
  }
  const std::uint32_t slot = atomicAdd(generic_tile_count, 1U);
  generic_tiles[slot] = tile;
}

void launch_clear_active_shell_quartet_tile_counts_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::uint32_t* active_shell_quartet_tile_counts, std::uint32_t* persistent_fock_task_heads,
    std::uint32_t* fp32_shell_quartet_tile_counts, std::uint32_t* fp32_persistent_fock_task_heads) {
  clear_active_shell_quartet_tile_counts_kernel<<<grid, block, shared_bytes, stream>>>(
      active_shell_quartet_tile_counts, persistent_fock_task_heads, fp32_shell_quartet_tile_counts,
      fp32_persistent_fock_task_heads);
}

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
    ActiveShellQuartetTile* fp32_shell_quartet_tiles) {
  if (unrestricted == true) {
    if (purpose == DirectScreeningPurpose::Fock) {
      compact_active_shell_quartet_tiles_kernel<true, DirectScreeningPurpose::Fock>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds, active,
              active_shell_quartet_tile_offsets, active_shell_quartet_tile_counts,
              active_shell_quartet_tiles, mixed_precision_enabled, mixed_precision_cutoff_ceiling,
              mixed_precision_budget_error, mixed_precision_item_census,
              fp32_shell_quartet_tile_offsets, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles);
    } else {
      compact_active_shell_quartet_tiles_kernel<true, DirectScreeningPurpose::Force>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds, active,
              active_shell_quartet_tile_offsets, active_shell_quartet_tile_counts,
              active_shell_quartet_tiles, mixed_precision_enabled, mixed_precision_cutoff_ceiling,
              mixed_precision_budget_error, mixed_precision_item_census,
              fp32_shell_quartet_tile_offsets, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles);
    }
  } else {
    if (purpose == DirectScreeningPurpose::Fock) {
      compact_active_shell_quartet_tiles_kernel<false, DirectScreeningPurpose::Fock>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds, active,
              active_shell_quartet_tile_offsets, active_shell_quartet_tile_counts,
              active_shell_quartet_tiles, mixed_precision_enabled, mixed_precision_cutoff_ceiling,
              mixed_precision_budget_error, mixed_precision_item_census,
              fp32_shell_quartet_tile_offsets, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles);
    } else {
      compact_active_shell_quartet_tiles_kernel<false, DirectScreeningPurpose::Force>
          <<<grid, block, shared_bytes, stream>>>(
              batch, screening_tolerance, shell_pair_bounds, shell_pair_density_bounds, active,
              active_shell_quartet_tile_offsets, active_shell_quartet_tile_counts,
              active_shell_quartet_tiles, mixed_precision_enabled, mixed_precision_cutoff_ceiling,
              mixed_precision_budget_error, mixed_precision_item_census,
              fp32_shell_quartet_tile_offsets, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles);
    }
  }
}

void launch_compact_generic_order5_tiles_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t generated_shell_class_mask,
    const std::uint64_t* generated_shell_class_mask_pointer, std::uint32_t* generic_tile_count,
    ActiveShellQuartetTile* generic_tiles) {
  compact_generic_order5_tiles_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles,
      generated_shell_class_mask, generated_shell_class_mask_pointer, generic_tile_count,
      generic_tiles);
}

}  // namespace vibeqc::scf::cuda_execution

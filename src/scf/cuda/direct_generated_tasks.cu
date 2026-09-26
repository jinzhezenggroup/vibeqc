#include <cmath>
#include <limits>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_generated_tasks.hpp"
#include "scf/cuda/direct_task_encoding.cuh"

namespace vibeqc::scf::cuda_execution {

// Slots without an enabled exact class remain on the existing native route.
constexpr std::uint8_t kNoGeneratedShellClass = std::numeric_limits<std::uint8_t>::max();

/**
 * Classify every active logical quartet once for all generated consumers.
 *
 * The byte tag is retained across the device-side prefix sum so task
 * materialization does not repeat exact-class decoding. Slots for AO tiles
 * beyond tile zero and classes disabled by the runtime mask remain tagged as
 * unclassified and fall through to the handwritten consumers.
 */
__global__ void classify_generated_shell_tasks_kernel(
    DeviceBatch batch, std::size_t total_tile_capacity,
    const std::uint32_t* active_shell_quartet_tile_offsets,
    const std::uint32_t* active_shell_quartet_tile_counts,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t enabled_shell_class_mask, const std::uint64_t* enabled_shell_class_mask_pointer,
    bool exclude_resident_ppps, std::uint32_t* generated_task_counts,
    std::uint8_t* generated_shell_classes, std::uint64_t low_order_signature_mask,
    std::uint32_t* low_order_signature_counts) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= total_tile_capacity) return;
  generated_shell_classes[slot] = kNoGeneratedShellClass;

  unsigned angular_order = 0;
  while (angular_order + 1 < detail::kDirectQuartetAngularOrderCount &&
         slot >= active_shell_quartet_tile_offsets[angular_order + 1]) {
    ++angular_order;
  }
  const std::size_t partition_begin = active_shell_quartet_tile_offsets[angular_order];
  if (slot - partition_begin >= active_shell_quartet_tile_counts[angular_order]) {
    return;
  }

  const ActiveShellQuartetTile tile = active_shell_quartet_tiles[slot];
  if (tile.tile != 0U) return;

  const std::int32_t first_shell = batch.shell_pair_first[tile.first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[tile.first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[tile.second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[tile.second_pair];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (enabled_shell_class_mask_pointer != nullptr) {
    enabled_shell_class_mask = *enabled_shell_class_mask_pointer;
  }
  if (shell_class >= detail::kDirectQuartetShellClassCount ||
      (enabled_shell_class_mask & (std::uint64_t{1} << shell_class)) == 0U) {
    return;
  }
  // Force preparation can route eligible canonical ppps quartets through the
  // resident-bra consumer.  Leave oversized bra primitive lists in this
  // ordinary class queue; otherwise their force contribution would vanish.
  if (exclude_resident_ppps && shell_class == kPppsShellClass) {
    std::uint32_t bra_pair = 0;
    if (resident_ppps_bra_pair(batch, tile, bra_pair)) return;
  }
  generated_shell_classes[slot] = static_cast<std::uint8_t>(shell_class);
  atomicAdd(generated_task_counts + shell_class, 1U);
  if (low_order_signature_counts != nullptr &&
      (low_order_signature_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    const unsigned signature = generated_low_order_signature_bucket(batch, tile);
    atomicAdd(
        low_order_signature_counts + generated_low_order_signature_index(shell_class, signature),
        1U);
  }
}

/** Build compact class slices and reset their materialization/worker cursors. */
__global__ void prefix_generated_shell_task_counts_kernel(
    const std::uint32_t* generated_task_counts, std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_write_counts, std::uint32_t* generated_task_heads) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  std::uint32_t offset = 0;
  generated_task_offsets[0] = 0;
  for (unsigned shell_class = 0; shell_class < detail::kDirectQuartetShellClassCount;
       ++shell_class) {
    generated_task_write_counts[shell_class] = 0;
    generated_task_heads[shell_class] = 0;
    offset += generated_task_counts[shell_class];
    generated_task_offsets[shell_class + 1] = offset;
  }
}

/** Prefix selected scalar signature slices inside their exact-class ranges. */
__global__ void prefix_low_order_signature_counts_kernel(
    const std::uint32_t* generated_task_offsets, std::uint64_t low_order_signature_mask,
    std::uint32_t* low_order_signature_counts, std::uint32_t* low_order_signature_offsets) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  for (unsigned class_slot = 0; class_slot < kLowOrderSignatureClassCount; ++class_slot) {
    const unsigned shell_class = class_slot == 0U ? kPspsShellClass : kPpssShellClass;
    if ((low_order_signature_mask & (std::uint64_t{1} << shell_class)) == 0U) {
      continue;
    }
    std::uint32_t offset = generated_task_offsets[shell_class];
    const unsigned signature_begin = class_slot * kLowOrderSignatureBucketsPerClass;
    for (unsigned signature = 0; signature < kLowOrderSignatureBucketsPerClass; ++signature) {
      const unsigned index = signature_begin + signature;
      const std::uint32_t count = low_order_signature_counts[index];
      low_order_signature_offsets[index] = offset;
      // Reuse the count array as the scatter cursor after preserving the class
      // total in generated_task_counts for the persistent worker.
      low_order_signature_counts[index] = 0U;
      offset += count;
    }
  }
}

/** Canonicalize classified quartets into contiguous exact-class slices. */
__global__ void materialize_generated_shell_tasks_kernel(
    DeviceBatch batch, std::size_t total_tile_capacity,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    const std::uint8_t* generated_shell_classes, const std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_write_counts, GeneratedShellTask* generated_tasks,
    std::uint64_t low_order_signature_mask, const std::uint32_t* low_order_signature_offsets,
    std::uint32_t* low_order_signature_write_counts) {
  const std::size_t active_tile = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (active_tile >= total_tile_capacity) return;
  const unsigned shell_class = generated_shell_classes[active_tile];
  if (shell_class == kNoGeneratedShellClass) return;

  const ActiveShellQuartetTile tile = active_shell_quartet_tiles[active_tile];
  std::uint32_t task_index = 0U;
  if (low_order_signature_offsets != nullptr && low_order_signature_write_counts != nullptr &&
      (low_order_signature_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    const unsigned signature = generated_low_order_signature_bucket(batch, tile);
    const unsigned index = generated_low_order_signature_index(shell_class, signature);
    task_index = low_order_signature_offsets[index] +
                 atomicAdd(low_order_signature_write_counts + index, 1U);
  } else {
    task_index = generated_task_offsets[shell_class] +
                 atomicAdd(generated_task_write_counts + shell_class, 1U);
  }
  populate_generated_shell_task(batch, tile, generated_tasks[task_index]);
}

void launch_classify_generated_shell_tasks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t total_tile_capacity, const std::uint32_t* active_shell_quartet_tile_offsets,
    const std::uint32_t* active_shell_quartet_tile_counts,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t enabled_shell_class_mask, const std::uint64_t* enabled_shell_class_mask_pointer,
    bool exclude_resident_ppps, std::uint32_t* generated_task_counts,
    std::uint8_t* generated_shell_classes, std::uint64_t low_order_signature_mask,
    std::uint32_t* low_order_signature_counts) {
  classify_generated_shell_tasks_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, total_tile_capacity, active_shell_quartet_tile_offsets,
      active_shell_quartet_tile_counts, active_shell_quartet_tiles, enabled_shell_class_mask,
      enabled_shell_class_mask_pointer, exclude_resident_ppps, generated_task_counts,
      generated_shell_classes, low_order_signature_mask, low_order_signature_counts);
}

void launch_prefix_generated_shell_task_counts_kernel(dim3 grid, dim3 block,
                                                      std::size_t shared_bytes, cudaStream_t stream,
                                                      const std::uint32_t* generated_task_counts,
                                                      std::uint32_t* generated_task_offsets,
                                                      std::uint32_t* generated_task_write_counts,
                                                      std::uint32_t* generated_task_heads) {
  prefix_generated_shell_task_counts_kernel<<<grid, block, shared_bytes, stream>>>(
      generated_task_counts, generated_task_offsets, generated_task_write_counts,
      generated_task_heads);
}

void launch_prefix_low_order_signature_counts_kernel(dim3 grid, dim3 block,
                                                     std::size_t shared_bytes, cudaStream_t stream,
                                                     const std::uint32_t* generated_task_offsets,
                                                     std::uint64_t low_order_signature_mask,
                                                     std::uint32_t* low_order_signature_counts,
                                                     std::uint32_t* low_order_signature_offsets) {
  prefix_low_order_signature_counts_kernel<<<grid, block, shared_bytes, stream>>>(
      generated_task_offsets, low_order_signature_mask, low_order_signature_counts,
      low_order_signature_offsets);
}

void launch_materialize_generated_shell_tasks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t total_tile_capacity, const ActiveShellQuartetTile* active_shell_quartet_tiles,
    const std::uint8_t* generated_shell_classes, const std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_write_counts, GeneratedShellTask* generated_tasks,
    std::uint64_t low_order_signature_mask, const std::uint32_t* low_order_signature_offsets,
    std::uint32_t* low_order_signature_write_counts) {
  materialize_generated_shell_tasks_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, total_tile_capacity, active_shell_quartet_tiles, generated_shell_classes,
      generated_task_offsets, generated_task_write_counts, generated_tasks,
      low_order_signature_mask, low_order_signature_offsets, low_order_signature_write_counts);
}

}  // namespace vibeqc::scf::cuda_execution

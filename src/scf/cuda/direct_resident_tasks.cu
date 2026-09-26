#include <cmath>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_resident_tasks.hpp"
#include "scf/cuda/direct_task_encoding.cuh"

namespace vibeqc::scf::cuda_execution {

/**
 * Count force-eligible canonical ppps tiles by their ``pp`` bra pair.
 *
 * This is intentionally indexed by the global shell-pair ordinal rather than
 * by a host-built map.  A direct batch can contain roughly 18k shell pairs;
 * three compact uint32 arrays make that histogram inexpensive and, more
 * importantly, keep active-system and force-screening decisions on device.
 */
__global__ void count_ppps_resident_bra_tasks_kernel(
    DeviceBatch batch, std::size_t active_tile_capacity,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t enabled_shell_class_mask, std::uint32_t* resident_bra_counts,
    std::uint32_t* resident_signature_counts) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= active_tile_capacity || slot >= *active_shell_quartet_tile_count) return;
  const ActiveShellQuartetTile tile = active_shell_quartet_tiles[slot];
  if ((enabled_shell_class_mask & (std::uint64_t{1} << kPppsShellClass)) == 0U) {
    return;
  }
  std::uint32_t bra_pair = 0;
  if (resident_ppps_bra_pair(batch, tile, bra_pair)) {
    atomicAdd(resident_bra_counts + bra_pair, 1U);
    if (resident_signature_counts != nullptr) {
      const unsigned signature = resident_ppps_signature_bucket(batch, tile, bra_pair);
      atomicAdd(resident_signature_counts +
                    static_cast<std::size_t>(bra_pair) * kPppsSignatureBucketCount + signature,
                1U);
    }
  }
}

/**
 * Prefix the ppps bra histogram and initialize one descriptor per bra.
 *
 * Descriptors are stored at their shell-pair ordinal. Inactive ordinals have
 * a zero ``ket_count`` and are harmless when the resident launch uses the
 * fixed shell-pair capacity; this avoids a device-to-host count readback and
 * keeps the force path graph/replay safe. ``resident_bra_offsets`` indexes
 * the transient ppps-sized tail of the generated-task arena. The final-force
 * stream launches the resident consumer before ordinary preparation is
 * allowed to overwrite that tail.
 */
__global__ void prefix_ppps_resident_bra_tasks_kernel(std::size_t total_shell_pairs,
                                                      const std::uint32_t* resident_bra_counts,
                                                      std::uint32_t* resident_bra_offsets,
                                                      std::uint32_t* resident_bra_write_counts,
                                                      GeneratedPppsResidentTask* resident_tasks) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  std::uint32_t offset = 0U;
  resident_bra_offsets[0] = 0U;
  for (std::size_t bra_pair = 0; bra_pair < total_shell_pairs; ++bra_pair) {
    resident_bra_write_counts[bra_pair] = 0U;
    const std::uint32_t count = resident_bra_counts[bra_pair];
    resident_tasks[bra_pair] = {static_cast<std::uint32_t>(bra_pair), offset, count};
    // Host topology validation bounds the resident allocation below
    // UINT32_MAX: every resident ket is one active ppps tile and the tile
    // capacity is checked before this kernel is launched.
    offset += count;
    resident_bra_offsets[bra_pair + 1U] = offset;
    resident_tasks[bra_pair].ket_begin = resident_bra_offsets[bra_pair];
  }
}

/** Build per-bra orientation/primitive bucket offsets for stable scattering. */
__global__ void prefix_ppps_resident_signature_buckets_kernel(
    std::size_t total_shell_pairs, const std::uint32_t* resident_bra_offsets,
    std::uint32_t* resident_signature_counts, std::uint32_t* resident_signature_offsets) {
  const std::size_t bra_pair = blockIdx.x;
  if (bra_pair >= total_shell_pairs || threadIdx.x != 0U) return;
  std::uint32_t offset = resident_bra_offsets[bra_pair];
  const std::size_t bucket_begin = bra_pair * kPppsSignatureBucketCount;
  for (unsigned bucket = 0U; bucket < kPppsSignatureBucketCount; ++bucket) {
    const std::size_t index = bucket_begin + bucket;
    const std::uint32_t count = resident_signature_counts[index];
    resident_signature_offsets[index] = offset;
    resident_signature_counts[index] = 0U;
    offset += count;
  }
}

/** Materialize eligible ppps tasks into the bra-grouped resident array. */
__global__ void materialize_ppps_resident_bra_tasks_kernel(
    DeviceBatch batch, std::size_t active_tile_capacity,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    const std::uint32_t* resident_bra_offsets, std::uint32_t* resident_bra_write_counts,
    const std::uint32_t* resident_signature_offsets, std::uint32_t* resident_signature_write_counts,
    GeneratedShellTask* resident_ket_tasks, std::uint32_t* resident_ket_signatures) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= active_tile_capacity || slot >= *active_shell_quartet_tile_count) return;
  const ActiveShellQuartetTile tile = active_shell_quartet_tiles[slot];
  std::uint32_t bra_pair = 0;
  if (!resident_ppps_bra_pair(batch, tile, bra_pair)) return;
  const bool pair_exchanged = bra_pair == tile.second_pair;
  std::uint32_t ket_index = 0U;
  if (resident_signature_offsets != nullptr && resident_signature_write_counts != nullptr) {
    const unsigned signature = resident_ppps_signature_bucket(batch, tile, bra_pair);
    const std::size_t bucket_index =
        static_cast<std::size_t>(bra_pair) * kPppsSignatureBucketCount + signature;
    ket_index = resident_signature_offsets[bucket_index] +
                atomicAdd(resident_signature_write_counts + bucket_index, 1U);
  } else {
    ket_index =
        resident_bra_offsets[bra_pair] + atomicAdd(resident_bra_write_counts + bra_pair, 1U);
  }
  populate_generated_shell_task(batch, tile, resident_ket_tasks[ket_index]);
  if (resident_ket_signatures != nullptr) {
    const GeneratedShellTask& task = resident_ket_tasks[ket_index];
    const std::int64_t ket_begin = batch.shell_pair_primitive_offsets[task.shell_pair[1]];
    const std::int64_t ket_end = batch.shell_pair_primitive_offsets[task.shell_pair[1] + 1U];
    const std::uint64_t ket_count =
        ket_end > ket_begin ? static_cast<std::uint64_t>(ket_end - ket_begin) : 0U;
    constexpr std::uint32_t kCountMask = 0x7fffffffU;
    const std::uint32_t encoded_count =
        ket_count > kCountMask ? kCountMask : static_cast<std::uint32_t>(ket_count);
    const std::uint32_t orientation = pair_exchanged ? 0x80000000U : 0U;
    resident_ket_signatures[ket_index] = orientation | encoded_count;
  }
}

void launch_count_ppps_resident_bra_tasks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t active_tile_capacity, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t enabled_shell_class_mask, std::uint32_t* resident_bra_counts,
    std::uint32_t* resident_signature_counts) {
  count_ppps_resident_bra_tasks_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, active_tile_capacity, active_shell_quartet_tile_count, active_shell_quartet_tiles,
      enabled_shell_class_mask, resident_bra_counts, resident_signature_counts);
}

void launch_prefix_ppps_resident_bra_tasks_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                  cudaStream_t stream,
                                                  std::size_t total_shell_pairs,
                                                  const std::uint32_t* resident_bra_counts,
                                                  std::uint32_t* resident_bra_offsets,
                                                  std::uint32_t* resident_bra_write_counts,
                                                  GeneratedPppsResidentTask* resident_tasks) {
  prefix_ppps_resident_bra_tasks_kernel<<<grid, block, shared_bytes, stream>>>(
      total_shell_pairs, resident_bra_counts, resident_bra_offsets, resident_bra_write_counts,
      resident_tasks);
}

void launch_prefix_ppps_resident_signature_buckets_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::size_t total_shell_pairs, const std::uint32_t* resident_bra_offsets,
    std::uint32_t* resident_signature_counts, std::uint32_t* resident_signature_offsets) {
  prefix_ppps_resident_signature_buckets_kernel<<<grid, block, shared_bytes, stream>>>(
      total_shell_pairs, resident_bra_offsets, resident_signature_counts,
      resident_signature_offsets);
}

void launch_materialize_ppps_resident_bra_tasks_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t active_tile_capacity, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    const std::uint32_t* resident_bra_offsets, std::uint32_t* resident_bra_write_counts,
    const std::uint32_t* resident_signature_offsets, std::uint32_t* resident_signature_write_counts,
    GeneratedShellTask* resident_ket_tasks, std::uint32_t* resident_ket_signatures) {
  materialize_ppps_resident_bra_tasks_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, active_tile_capacity, active_shell_quartet_tile_count, active_shell_quartet_tiles,
      resident_bra_offsets, resident_bra_write_counts, resident_signature_offsets,
      resident_signature_write_counts, resident_ket_tasks, resident_ket_signatures);
}

}  // namespace vibeqc::scf::cuda_execution

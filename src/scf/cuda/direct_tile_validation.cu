#include <cmath>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/direct_tile_validation.hpp"

namespace vibeqc::scf::cuda_execution {

__device__ bool direct_shell_ao_range_valid(const DeviceBatch& batch, std::int32_t shell,
                                            std::size_t system_ao_begin, std::size_t direct_nbf,
                                            std::size_t& count) {
  const std::int64_t begin_value = batch.shell_direct_ao_offsets[shell];
  const std::int64_t end_value = batch.shell_direct_ao_offsets[shell + 1];
  if (begin_value < 0 || end_value < begin_value) return false;
  const std::size_t begin = static_cast<std::size_t>(begin_value);
  const std::size_t end = static_cast<std::size_t>(end_value);
  if (begin < system_ao_begin || end < begin || end - system_ao_begin > direct_nbf) {
    return false;
  }
  count = end - begin;
  return true;
}

/** Record one validation failure without perturbing the production path. */
__device__ void record_direct_tile_validation_failure(
    DirectTileValidationRecord* record, DirectTileValidationError error, unsigned angular_order,
    std::size_t slot, const ActiveShellQuartetTile& tile, const std::int32_t* shells,
    std::size_t direct_nbf, std::size_t first_pair_count, std::size_t second_pair_count,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l, std::size_t active_tile_count,
    std::size_t partition_capacity, std::size_t partition_begin) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(&record->error, kDirectTileValidationNoError, code) !=
      kDirectTileValidationNoError) {
    return;
  }
  record->angular_order = angular_order;
  record->slot = static_cast<std::uint32_t>(slot);
  record->tile = tile.tile;
  record->first_pair = tile.first_pair;
  record->second_pair = tile.second_pair;
#pragma unroll
  for (unsigned center = 0; center < 4U; ++center) {
    record->shell[center] = shells == nullptr ? -1 : shells[center];
  }
  record->direct_nbf = static_cast<std::uint32_t>(direct_nbf);
  record->first_pair_count = static_cast<std::uint32_t>(first_pair_count);
  record->second_pair_count = static_cast<std::uint32_t>(second_pair_count);
  record->i = static_cast<std::uint32_t>(i);
  record->j = static_cast<std::uint32_t>(j);
  record->k = static_cast<std::uint32_t>(k);
  record->l = static_cast<std::uint32_t>(l);
  record->active_tile_count = static_cast<std::uint32_t>(active_tile_count);
  record->partition_capacity = static_cast<std::uint32_t>(partition_capacity);
  record->partition_begin = static_cast<std::uint32_t>(partition_begin);
}

/** Validate compact direct tiles before any generated or handwritten consumer. */
__global__ void validate_direct_tile_descriptors_kernel(DeviceBatch batch,
                                                        const std::uint32_t* active_tile_offsets,
                                                        const std::uint32_t* active_tile_counts,
                                                        const ActiveShellQuartetTile* active_tiles,
                                                        std::size_t total_tile_capacity,
                                                        DirectTileValidationRecord* record) {
  const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= total_tile_capacity) return;

  unsigned angular_order = 0;
  while (angular_order + 1U < detail::kDirectQuartetAngularOrderCount &&
         slot >= active_tile_offsets[angular_order + 1U]) {
    ++angular_order;
  }
  const std::size_t partition_begin = active_tile_offsets[angular_order];
  const std::size_t partition_end = active_tile_offsets[angular_order + 1U];
  const std::size_t partition_capacity =
      partition_end >= partition_begin ? partition_end - partition_begin : 0U;
  const std::size_t active_tile_count = active_tile_counts[angular_order];
  ActiveShellQuartetTile empty_tile{};
  if (partition_end < partition_begin || active_tile_count > partition_capacity) {
    record_direct_tile_validation_failure(
        record, DirectTileValidationError::count_exceeds_capacity, angular_order, slot, empty_tile,
        nullptr, static_cast<std::size_t>(batch.direct_nbf), 0, 0, 0, 0, 0, 0, active_tile_count,
        partition_capacity, partition_begin);
    return;
  }
  if (slot - partition_begin >= active_tile_count) return;

  const ActiveShellQuartetTile tile = active_tiles[slot];
  if (tile.first_pair >= static_cast<std::uint32_t>(batch.total_shell_pairs) ||
      tile.second_pair >= static_cast<std::uint32_t>(batch.total_shell_pairs)) {
    record_direct_tile_validation_failure(
        record, DirectTileValidationError::pair_out_of_bounds, angular_order, slot, tile, nullptr,
        static_cast<std::size_t>(batch.direct_nbf), 0, 0, 0, 0, 0, 0, active_tile_count,
        partition_capacity, partition_begin);
    return;
  }

  const std::int32_t shells[4] = {
      batch.shell_pair_first[tile.first_pair],
      batch.shell_pair_second[tile.first_pair],
      batch.shell_pair_first[tile.second_pair],
      batch.shell_pair_second[tile.second_pair],
  };
  for (unsigned center = 0; center < 4U; ++center) {
    if (shells[center] < 0 || shells[center] >= static_cast<std::int32_t>(batch.total_shells)) {
      record_direct_tile_validation_failure(
          record, DirectTileValidationError::shell_out_of_bounds, angular_order, slot, tile, shells,
          static_cast<std::size_t>(batch.direct_nbf), 0, 0, 0, 0, 0, 0, active_tile_count,
          partition_capacity, partition_begin);
      return;
    }
  }

  const std::int32_t system = batch.shell_pair_systems[tile.first_pair];
  const std::int32_t second_system = batch.shell_pair_systems[tile.second_pair];
  if (system < 0 || system >= batch.batch_size || second_system != system) {
    record_direct_tile_validation_failure(
        record, DirectTileValidationError::shell_out_of_bounds, angular_order, slot, tile, shells,
        static_cast<std::size_t>(batch.direct_nbf), 0, 0, 0, 0, 0, 0, active_tile_count,
        partition_capacity, partition_begin);
    return;
  }
  const std::size_t direct_nbf = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * direct_nbf;
  std::size_t shell_counts[4]{};
  for (unsigned center = 0; center < 4U; ++center) {
    if (!direct_shell_ao_range_valid(batch, shells[center], system_ao_begin, direct_nbf,
                                     shell_counts[center])) {
      record_direct_tile_validation_failure(
          record, DirectTileValidationError::ao_range_invalid, angular_order, slot, tile, shells,
          direct_nbf, 0, 0, 0, 0, 0, 0, active_tile_count, partition_capacity, partition_begin);
      return;
    }
  }
  const std::size_t first_pair_ao_count = shells[0] == shells[1]
                                              ? shell_counts[0] * (shell_counts[0] + 1U) / 2U
                                              : shell_counts[0] * shell_counts[1];
  const std::size_t second_pair_ao_count = shells[2] == shells[3]
                                               ? shell_counts[2] * (shell_counts[2] + 1U) / 2U
                                               : shell_counts[2] * shell_counts[3];
  const std::size_t ao_quartet_count = tile.first_pair == tile.second_pair
                                           ? first_pair_ao_count * (first_pair_ao_count + 1U) / 2U
                                           : first_pair_ao_count * second_pair_ao_count;
  const std::size_t expected_tiles =
      (ao_quartet_count + detail::kDirectQuartetTileSize - 1U) / detail::kDirectQuartetTileSize;
  if (tile.tile >= expected_tiles || first_pair_ao_count == 0U || second_pair_ao_count == 0U) {
    record_direct_tile_validation_failure(record, DirectTileValidationError::tile_out_of_bounds,
                                          angular_order, slot, tile, shells, direct_nbf,
                                          first_pair_ao_count, second_pair_ao_count, 0, 0, 0, 0,
                                          active_tile_count, partition_capacity, partition_begin);
    return;
  }

  const std::size_t ordinal = static_cast<std::size_t>(tile.tile) * detail::kDirectQuartetTileSize;
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t k = 0;
  std::size_t l = 0;
  if (!decode_direct_tile_ao_ordinal(batch, tile, ordinal, first_pair_ao_count,
                                     second_pair_ao_count, system_ao_begin, direct_nbf, i, j, k,
                                     l)) {
    record_direct_tile_validation_failure(record, DirectTileValidationError::ao_range_invalid,
                                          angular_order, slot, tile, shells, direct_nbf,
                                          first_pair_ao_count, second_pair_ao_count, i, j, k, l,
                                          active_tile_count, partition_capacity, partition_begin);
    return;
  }
  const std::size_t last_ordinal =
      min(ao_quartet_count - 1U, ordinal + detail::kDirectQuartetTileSize - 1U);
  if (!decode_direct_tile_ao_ordinal(batch, tile, last_ordinal, first_pair_ao_count,
                                     second_pair_ao_count, system_ao_begin, direct_nbf, i, j, k,
                                     l)) {
    record_direct_tile_validation_failure(record, DirectTileValidationError::ao_range_invalid,
                                          angular_order, slot, tile, shells, direct_nbf,
                                          first_pair_ao_count, second_pair_ao_count, i, j, k, l,
                                          active_tile_count, partition_capacity, partition_begin);
  }
}

void launch_validate_direct_tile_descriptors_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                    cudaStream_t stream, DeviceBatch batch,
                                                    const std::uint32_t* active_tile_offsets,
                                                    const std::uint32_t* active_tile_counts,
                                                    const ActiveShellQuartetTile* active_tiles,
                                                    std::size_t total_tile_capacity,
                                                    DirectTileValidationRecord* record) {
  validate_direct_tile_descriptors_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, active_tile_offsets, active_tile_counts, active_tiles, total_tile_capacity, record);
}

}  // namespace vibeqc::scf::cuda_execution

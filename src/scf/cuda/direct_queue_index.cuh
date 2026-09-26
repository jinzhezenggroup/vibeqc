#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

__device__ inline void decode_lower_triangle(std::size_t packed, std::size_t& first,
                                             std::size_t& second) {
  first = static_cast<std::size_t>(0.5 * (sqrt(8.0 * static_cast<double>(packed) + 1.0) - 1.0));
  while ((first + 1) * (first + 2) / 2 <= packed) ++first;
  while (first * (first + 1) / 2 > packed) --first;
  second = packed - first * (first + 1) / 2;
}

__device__ inline std::int32_t shell_quartet_system(const DeviceBatch& batch, std::size_t quartet) {
  std::int32_t lower = 0;
  std::int32_t upper = batch.batch_size;
  while (lower + 1 < upper) {
    const std::int32_t middle = lower + (upper - lower) / 2;
    if (static_cast<std::size_t>(batch.system_shell_quartet_offsets[middle]) <= quartet) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

/** Resolve one packed lower-triangular shell-pair-block task to its system. */
__device__ inline std::int32_t shell_pair_block_quartet_system(const DeviceBatch& batch,
                                                               std::size_t block_quartet) {
  std::int32_t lower = 0;
  std::int32_t upper = batch.batch_size;
  while (lower + 1 < upper) {
    const std::int32_t middle = lower + (upper - lower) / 2;
    if (static_cast<std::size_t>(batch.system_shell_pair_block_quartet_offsets[middle]) <=
        block_quartet) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

__device__ inline std::size_t shell_ao_pair_count(const DeviceBatch& batch,
                                                  std::size_t shell_pair) {
  const std::int32_t first_shell = batch.shell_pair_first[shell_pair];
  const std::int32_t second_shell = batch.shell_pair_second[shell_pair];
  const std::size_t first_count = static_cast<std::size_t>(
      batch.shell_direct_ao_offsets[first_shell + 1] - batch.shell_direct_ao_offsets[first_shell]);
  const std::size_t second_count =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[second_shell + 1] -
                               batch.shell_direct_ao_offsets[second_shell]);
  return first_shell == second_shell ? first_count * (first_count + 1) / 2
                                     : first_count * second_count;
}

__device__ inline void decode_shell_ao_pair(const DeviceBatch& batch, std::size_t shell_pair,
                                            std::size_t ordinal, std::size_t system_ao_begin,
                                            std::size_t& first, std::size_t& second) {
  const std::int32_t first_shell = batch.shell_pair_first[shell_pair];
  const std::int32_t second_shell = batch.shell_pair_second[shell_pair];
  const std::size_t first_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[first_shell]);
  const std::size_t second_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[second_shell]);
  const std::size_t second_count =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[second_shell + 1]) - second_begin;
  std::size_t first_component = 0;
  std::size_t second_component = 0;
  if (first_shell == second_shell) {
    decode_lower_triangle(ordinal, first_component, second_component);
  } else {
    first_component = ordinal / second_count;
    second_component = ordinal % second_count;
  }
  first = first_begin + first_component - system_ao_begin;
  second = second_begin + second_component - system_ao_begin;
}

/** Shared AO-quartet indexing for one pair-of-shell-pairs task. */
struct DirectShellAoQuartetLayout {
  std::size_t second_pair_ao_count{};
  std::size_t quartet_count{};
  bool same_shell_pair{};
};

__device__ inline DirectShellAoQuartetLayout direct_shell_ao_quartet_layout(
    const DeviceBatch& batch, std::size_t first_pair, std::size_t second_pair) {
  const std::size_t first_pair_ao_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_pair_ao_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  return {
      second_pair_ao_count,
      same_shell_pair ? first_pair_ao_count * (first_pair_ao_count + 1) / 2
                      : first_pair_ao_count * second_pair_ao_count,
      same_shell_pair,
  };
}

__device__ inline void decode_shell_ao_quartet(const DeviceBatch& batch, std::size_t first_pair,
                                               std::size_t second_pair,
                                               const DirectShellAoQuartetLayout& layout,
                                               std::size_t ordinal, std::size_t system_ao_begin,
                                               std::size_t (&raw_ao)[4]) {
  std::size_t first_ao_pair = 0;
  std::size_t second_ao_pair = 0;
  if (layout.same_shell_pair) {
    decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
  } else {
    first_ao_pair = ordinal / layout.second_pair_ao_count;
    second_ao_pair = ordinal % layout.second_pair_ao_count;
  }
  decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, raw_ao[0], raw_ao[1]);
  decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, raw_ao[2], raw_ao[3]);
}

/** Return the packed lower-triangle index for two shells in one system. */
__device__ inline std::size_t system_shell_pair_index(const DeviceBatch& batch, std::int32_t system,
                                                      std::int32_t first_shell,
                                                      std::int32_t second_shell) {
  const std::size_t shell_begin = static_cast<std::size_t>(batch.system_shell_offsets[system]);
  const std::size_t first = static_cast<std::size_t>(first_shell) - shell_begin;
  const std::size_t second = static_cast<std::size_t>(second_shell) - shell_begin;
  const std::size_t high = first > second ? first : second;
  const std::size_t low = first > second ? second : first;
  return static_cast<std::size_t>(batch.system_shell_pair_offsets[system]) + high * (high + 1) / 2 +
         low;
}

/** Match the host planner's symmetry-reduced s/p/d/f shell-class encoding. */
__host__ __device__ constexpr unsigned direct_triangular_class_high(unsigned index) {
  unsigned high = 0;
  while ((high + 1) * (high + 2) / 2 <= index) ++high;
  return high;
}

/** Resolve one class template to its exact Coulomb recurrence order. */
__host__ __device__ constexpr unsigned direct_shell_class_angular_order(unsigned shell_class) {
  const unsigned first_pair = direct_triangular_class_high(shell_class);
  const unsigned second_pair = shell_class - first_pair * (first_pair + 1) / 2;
  const unsigned first_high = direct_triangular_class_high(first_pair);
  const unsigned first_low = first_pair - first_high * (first_high + 1) / 2;
  const unsigned second_high = direct_triangular_class_high(second_pair);
  const unsigned second_low = second_pair - second_high * (second_high + 1) / 2;
  return first_high + first_low + second_high + second_low;
}

__host__ __device__ constexpr unsigned direct_shell_pair_class_cuda(unsigned first,
                                                                    unsigned second) {
  const unsigned high = first > second ? first : second;
  const unsigned low = first > second ? second : first;
  return high * (high + 1) / 2 + low;
}

__device__ inline unsigned direct_quartet_shell_class_device(unsigned first, unsigned second,
                                                             unsigned third, unsigned fourth) {
  const unsigned first_pair = direct_shell_pair_class_cuda(first, second);
  const unsigned second_pair = direct_shell_pair_class_cuda(third, fourth);
  const unsigned high_pair = max(first_pair, second_pair);
  const unsigned low_pair = min(first_pair, second_pair);
  return high_pair * (high_pair + 1) / 2 + low_pair;
}

__device__ inline bool decode_direct_tile_ao_ordinal(
    const DeviceBatch& batch, const ActiveShellQuartetTile& tile, std::size_t ordinal,
    std::size_t first_pair_ao_count, std::size_t second_pair_ao_count, std::size_t system_ao_begin,
    std::size_t direct_nbf, std::size_t& i, std::size_t& j, std::size_t& k, std::size_t& l) {
  if (first_pair_ao_count == 0U || second_pair_ao_count == 0U) {
    return false;
  }
  std::size_t first_ao_pair = ordinal / second_pair_ao_count;
  std::size_t second_ao_pair = ordinal % second_pair_ao_count;
  if (tile.first_pair == tile.second_pair) {
    decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
  }
  decode_shell_ao_pair(batch, tile.first_pair, first_ao_pair, system_ao_begin, i, j);
  decode_shell_ao_pair(batch, tile.second_pair, second_ao_pair, system_ao_begin, k, l);
  return i < direct_nbf && j < direct_nbf && k < direct_nbf && l < direct_nbf;
}

}  // namespace vibeqc::scf::cuda_execution

#ifndef VIBEQC_SCF_DIRECT_TASK_LAYOUT_HPP
#define VIBEQC_SCF_DIRECT_TASK_LAYOUT_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace vibeqc::scf::detail {

/**
 * Threads assigned to one symmetry-unique direct-J/K AO-quartet subtile.
 *
 * Exact Fock and force consumers sit near the per-thread register ceiling.
 * One full warp per block exposes independent work for latency hiding;
 * smaller sub-warp blocks waste execution lanes. A compact descriptor still
 * covers 256 quartets. Consumers expand only the number of virtual subtiles
 * reachable by each total angular order, avoiding empty low-order blocks
 * without multiplying fixed-topology arena metadata.
 */
inline constexpr std::size_t kDirectQuartetThreads = 32;
inline constexpr std::size_t kDirectQuartetTileSize = 256;
inline constexpr std::size_t kDirectQuartetSubtilesPerTile =
    kDirectQuartetTileSize / kDirectQuartetThreads;

static_assert(kDirectQuartetTileSize % kDirectQuartetThreads == 0);

/** Maximum populated 32-thread subtiles for one tile at each total order. */
#if defined(__CUDACC__)
__host__ __device__
#endif
    inline constexpr std::size_t direct_quartet_subtiles_per_tile(
        std::size_t angular_order) noexcept {
  // Cartesian component products reach at most 27 through order three,
  // 81 at order four, and 162 at order five. Higher orders can fill all 256
  // entries of a logical tile and therefore retain the full eight subtiles.
  return angular_order <= 3   ? 1
         : angular_order == 4 ? 3
         : angular_order == 5 ? 6
                              : kDirectQuartetSubtilesPerTile;
}

static_assert(direct_quartet_subtiles_per_tile(0) == 1);
static_assert(direct_quartet_subtiles_per_tile(3) == 1);
static_assert(direct_quartet_subtiles_per_tile(4) == 3);
static_assert(direct_quartet_subtiles_per_tile(5) == 6);
static_assert(direct_quartet_subtiles_per_tile(6) == 8);

/** Total shell angular orders from ssss (0) through ffff (12). */
inline constexpr std::size_t kDirectQuartetAngularOrderCount = 13;
inline constexpr std::uint8_t kDirectQuartetMaximumShellAngular = 3;

/** Canonical unordered shell-pair and pair-of-pairs classes for s/p/d/f. */
inline constexpr std::size_t kDirectShellPairClassCount = 10;
inline constexpr std::size_t kDirectQuartetShellClassCount = 55;

/** Fixed descriptor queue shared by one bounded-streaming CUDA CTA. */
inline constexpr std::size_t kBoundedDirectQueueCapacity = 256;

/** Shell pairs grouped under one conservative bounded-path coarse gate. */
inline constexpr std::size_t kBoundedDirectShellPairBlockSize = 32;

/**
 * Largest exact tile arena whose eight-subtile CUDA grids fit in `unsigned`.
 *
 * Every shell quartet owns at least one logical tile. A quartet count above
 * this limit therefore proves that the exact topology cannot be represented,
 * without first performing an O(N_shell^4) host enumeration.
 */
inline constexpr std::size_t kDirectFixedTopologyTileLimit =
    std::numeric_limits<std::uint32_t>::max() / kDirectQuartetSubtilesPerTile;

inline constexpr bool direct_topology_requires_bounded_streaming(
    std::size_t shell_quartet_count) noexcept {
  return shell_quartet_count > kDirectFixedTopologyTileLimit;
}

/** Number of fixed-capacity cursor rounds needed to visit candidate work. */
inline constexpr std::size_t bounded_direct_queue_refill_count(
    std::size_t candidate_count, std::size_t capacity = kBoundedDirectQueueCapacity) noexcept {
  return capacity == 0 ? 0 : candidate_count / capacity + (candidate_count % capacity == 0 ? 0 : 1);
}

/** Encode an unordered angular pair as ss, ps, pp, ds, ..., ff. */
inline constexpr std::size_t direct_shell_pair_class(std::uint8_t first,
                                                     std::uint8_t second) noexcept {
  const std::size_t high = std::max(first, second);
  const std::size_t low = std::min(first, second);
  return high * (high + 1) / 2 + low;
}

/** Encode an ERI shell class after applying pair and pair-exchange symmetry. */
inline constexpr std::size_t direct_quartet_shell_class(std::uint8_t first, std::uint8_t second,
                                                        std::uint8_t third,
                                                        std::uint8_t fourth) noexcept {
  const std::size_t first_pair = direct_shell_pair_class(first, second);
  const std::size_t second_pair = direct_shell_pair_class(third, fourth);
  const std::size_t high = std::max(first_pair, second_pair);
  const std::size_t low = std::min(first_pair, second_pair);
  return high * (high + 1) / 2 + low;
}

/** Decode a triangular class index into its canonical high/low members. */
inline constexpr std::array<std::size_t, 2> decode_direct_triangular_class(
    std::size_t index) noexcept {
  std::size_t high = 0;
  while ((high + 1) * (high + 2) / 2 <= index) ++high;
  return {high, index - high * (high + 1) / 2};
}

/** Return the total angular order represented by a canonical shell class. */
inline constexpr std::size_t direct_quartet_shell_class_angular_order(
    std::size_t shell_class) noexcept {
  const auto pair_classes = decode_direct_triangular_class(shell_class);
  const auto first_pair = decode_direct_triangular_class(pair_classes[0]);
  const auto second_pair = decode_direct_triangular_class(pair_classes[1]);
  return first_pair[0] + first_pair[1] + second_pair[0] + second_pair[1];
}

/** Fixed-topology capacity required by geometry-dependent tile compaction. */
struct DirectQuartetTaskLayout {
  std::size_t shell_quartet_count{};
  // Sum of ceil(unique AO quartets / logical tile size) for every quartet.
  std::size_t exact_tile_count{};
  // Fixed topology partitions used by angular-specialized CUDA consumers.
  std::array<std::size_t, kDirectQuartetAngularOrderCount> angular_order_tile_counts{};
  std::array<std::size_t, kDirectQuartetAngularOrderCount + 1> angular_order_tile_offsets{};
  // Finer fixed partitions used by generated exact shell-class consumers.
  std::array<std::size_t, kDirectQuartetShellClassCount> shell_class_tile_counts{};
  std::array<std::size_t, kDirectQuartetShellClassCount + 1> shell_class_tile_offsets{};
  // Previous padding multiplier derived from max(shell-pair AOs)^2.
  std::size_t maximum_tiles_per_shell_quartet{};
  std::size_t uniform_tile_count{};
};

inline bool checked_task_add(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first > std::numeric_limits<std::size_t>::max() - second) return false;
  result = first + second;
  return true;
}

inline bool checked_task_multiply(std::size_t first, std::size_t second,
                                  std::size_t& result) noexcept {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

/**
 * Count exact direct-J/K tiles for a canonical ragged shell-pair topology.
 *
 * Each system owns a lower-triangular list of shell pairs. The returned exact
 * capacity stores one descriptor per non-empty AO-quartet tile; the uniform
 * count reports the previous global-maximum tiling for regression diagnostics.
 */
inline bool make_direct_quartet_task_layout(
    const std::vector<std::int64_t>& shell_ao_offsets,
    const std::vector<std::uint8_t>& shell_angular,
    const std::vector<std::int64_t>& system_shell_pair_offsets,
    const std::vector<std::int32_t>& shell_pair_first,
    const std::vector<std::int32_t>& shell_pair_second, DirectQuartetTaskLayout& layout) {
  if (shell_ao_offsets.empty() || shell_angular.size() != shell_ao_offsets.size() - 1 ||
      system_shell_pair_offsets.empty() || shell_pair_first.empty() ||
      shell_pair_first.size() != shell_pair_second.size() ||
      system_shell_pair_offsets.front() != 0 || system_shell_pair_offsets.back() < 0 ||
      static_cast<std::size_t>(system_shell_pair_offsets.back()) != shell_pair_first.size()) {
    return false;
  }

  std::vector<std::size_t> shell_pair_ao_counts(shell_pair_first.size());
  std::vector<std::size_t> shell_pair_angular_orders(shell_pair_first.size());
  std::vector<std::size_t> shell_pair_classes(shell_pair_first.size());
  std::size_t maximum_shell_pair_ao_count = 0;
  for (std::size_t pair = 0; pair < shell_pair_first.size(); ++pair) {
    const std::int32_t first_shell = shell_pair_first[pair];
    const std::int32_t second_shell = shell_pair_second[pair];
    if (first_shell < 0 || second_shell < 0 ||
        static_cast<std::size_t>(first_shell) + 1 >= shell_ao_offsets.size() ||
        static_cast<std::size_t>(second_shell) + 1 >= shell_ao_offsets.size()) {
      return false;
    }
    if (shell_angular[first_shell] > kDirectQuartetMaximumShellAngular ||
        shell_angular[second_shell] > kDirectQuartetMaximumShellAngular) {
      return false;
    }
    const std::size_t pair_angular_order = static_cast<std::size_t>(shell_angular[first_shell]) +
                                           static_cast<std::size_t>(shell_angular[second_shell]);
    if (pair_angular_order >= kDirectQuartetAngularOrderCount) return false;
    const std::int64_t first_begin = shell_ao_offsets[first_shell];
    const std::int64_t first_end = shell_ao_offsets[first_shell + 1];
    const std::int64_t second_begin = shell_ao_offsets[second_shell];
    const std::int64_t second_end = shell_ao_offsets[second_shell + 1];
    if (first_begin < 0 || second_begin < 0 || first_end <= first_begin ||
        second_end <= second_begin) {
      return false;
    }
    const std::size_t first_count = static_cast<std::size_t>(first_end - first_begin);
    const std::size_t second_count = static_cast<std::size_t>(second_end - second_begin);
    std::size_t ao_pair_count = 0;
    if (first_shell == second_shell) {
      std::size_t first_plus_one = 0;
      if (!checked_task_add(first_count, 1, first_plus_one) ||
          !checked_task_multiply(first_count, first_plus_one, ao_pair_count)) {
        return false;
      }
      ao_pair_count /= 2;
    } else if (!checked_task_multiply(first_count, second_count, ao_pair_count)) {
      return false;
    }
    shell_pair_ao_counts[pair] = ao_pair_count;
    shell_pair_angular_orders[pair] = pair_angular_order;
    shell_pair_classes[pair] =
        direct_shell_pair_class(shell_angular[first_shell], shell_angular[second_shell]);
    maximum_shell_pair_ao_count = std::max(maximum_shell_pair_ao_count, ao_pair_count);
  }

  DirectQuartetTaskLayout made{};
  for (std::size_t system = 0; system + 1 < system_shell_pair_offsets.size(); ++system) {
    const std::int64_t begin_value = system_shell_pair_offsets[system];
    const std::int64_t end_value = system_shell_pair_offsets[system + 1];
    if (begin_value < 0 || end_value < begin_value) return false;
    const std::size_t begin = static_cast<std::size_t>(begin_value);
    const std::size_t end = static_cast<std::size_t>(end_value);
    if (end > shell_pair_ao_counts.size()) return false;
    for (std::size_t first_pair = begin; first_pair < end; ++first_pair) {
      for (std::size_t second_pair = begin; second_pair <= first_pair; ++second_pair) {
        std::size_t ao_quartet_count = 0;
        if (first_pair == second_pair) {
          std::size_t first_plus_one = 0;
          if (!checked_task_add(shell_pair_ao_counts[first_pair], 1, first_plus_one) ||
              !checked_task_multiply(shell_pair_ao_counts[first_pair], first_plus_one,
                                     ao_quartet_count)) {
            return false;
          }
          ao_quartet_count /= 2;
        } else if (!checked_task_multiply(shell_pair_ao_counts[first_pair],
                                          shell_pair_ao_counts[second_pair], ao_quartet_count)) {
          return false;
        }
        std::size_t rounded = 0;
        if (ao_quartet_count == 0 ||
            !checked_task_add(ao_quartet_count, kDirectQuartetTileSize - 1, rounded)) {
          return false;
        }
        const std::size_t tile_count = rounded / kDirectQuartetTileSize;
        const std::size_t angular_order =
            shell_pair_angular_orders[first_pair] + shell_pair_angular_orders[second_pair];
        const std::size_t reachable_low_order_quartets =
            direct_quartet_subtiles_per_tile(angular_order) * kDirectQuartetThreads;
        // Orders zero through five fit in one logical tile and have tighter
        // Cartesian component maxima than its 256-entry storage capacity.
        // Keep that scheduling invariant checked beside the authoritative
        // topology counts so a future basis-layout change cannot drop work.
        if (angular_order <= 5 &&
            (tile_count != 1 || ao_quartet_count > reachable_low_order_quartets)) {
          return false;
        }
        const std::size_t high_pair_class =
            std::max(shell_pair_classes[first_pair], shell_pair_classes[second_pair]);
        const std::size_t low_pair_class =
            std::min(shell_pair_classes[first_pair], shell_pair_classes[second_pair]);
        const std::size_t shell_class =
            high_pair_class * (high_pair_class + 1) / 2 + low_pair_class;
        if (angular_order >= kDirectQuartetAngularOrderCount ||
            shell_class >= kDirectQuartetShellClassCount ||
            !checked_task_add(made.angular_order_tile_counts[angular_order], tile_count,
                              made.angular_order_tile_counts[angular_order]) ||
            !checked_task_add(made.shell_class_tile_counts[shell_class], tile_count,
                              made.shell_class_tile_counts[shell_class])) {
          return false;
        }
        if (!checked_task_add(made.shell_quartet_count, 1, made.shell_quartet_count) ||
            !checked_task_add(made.exact_tile_count, tile_count, made.exact_tile_count)) {
          return false;
        }
      }
    }
  }
  for (std::size_t order = 0; order < kDirectQuartetAngularOrderCount; ++order) {
    made.angular_order_tile_offsets[order + 1] = made.angular_order_tile_offsets[order];
    if (!checked_task_add(made.angular_order_tile_offsets[order + 1],
                          made.angular_order_tile_counts[order],
                          made.angular_order_tile_offsets[order + 1])) {
      return false;
    }
  }
  if (made.angular_order_tile_offsets.back() != made.exact_tile_count) {
    return false;
  }
  for (std::size_t shell_class = 0; shell_class < kDirectQuartetShellClassCount; ++shell_class) {
    made.shell_class_tile_offsets[shell_class + 1] = made.shell_class_tile_offsets[shell_class];
    if (!checked_task_add(made.shell_class_tile_offsets[shell_class + 1],
                          made.shell_class_tile_counts[shell_class],
                          made.shell_class_tile_offsets[shell_class + 1])) {
      return false;
    }
  }
  if (made.shell_class_tile_offsets.back() != made.exact_tile_count) {
    return false;
  }
  std::size_t maximum_uniform_ao_quartets = 0;
  std::size_t rounded_uniform_ao_quartets = 0;
  if (!checked_task_multiply(maximum_shell_pair_ao_count, maximum_shell_pair_ao_count,
                             maximum_uniform_ao_quartets) ||
      !checked_task_add(maximum_uniform_ao_quartets, kDirectQuartetTileSize - 1,
                        rounded_uniform_ao_quartets)) {
    return false;
  }
  made.maximum_tiles_per_shell_quartet = rounded_uniform_ao_quartets / kDirectQuartetTileSize;
  if (!checked_task_multiply(made.shell_quartet_count, made.maximum_tiles_per_shell_quartet,
                             made.uniform_tile_count)) {
    return false;
  }
  layout = made;
  return true;
}

}  // namespace vibeqc::scf::detail

#endif

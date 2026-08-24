#ifndef QCE_SCF_DIRECT_TASK_LAYOUT_HPP
#define QCE_SCF_DIRECT_TASK_LAYOUT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace qce::scf::detail {

/** Threads assigned to one symmetry-unique direct-J/K AO-quartet tile. */
inline constexpr std::size_t kDirectQuartetThreads = 256;

/** Fixed-topology capacity required by geometry-dependent tile compaction. */
struct DirectQuartetTaskLayout {
  std::size_t shell_quartet_count{};
  // Sum of ceil(unique AO quartets / threads) for every shell quartet.
  std::size_t exact_tile_count{};
  // Previous padding multiplier derived from max(shell-pair AOs)^2.
  std::size_t maximum_tiles_per_shell_quartet{};
  std::size_t uniform_tile_count{};
};

inline bool checked_task_add(std::size_t first,
                             std::size_t second,
                             std::size_t& result) noexcept {
  if (first > std::numeric_limits<std::size_t>::max() - second) return false;
  result = first + second;
  return true;
}

inline bool checked_task_multiply(std::size_t first,
                                  std::size_t second,
                                  std::size_t& result) noexcept {
  if (first != 0 &&
      second > std::numeric_limits<std::size_t>::max() / first) {
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
    const std::vector<std::int64_t>& system_shell_pair_offsets,
    const std::vector<std::int32_t>& shell_pair_first,
    const std::vector<std::int32_t>& shell_pair_second,
    DirectQuartetTaskLayout& layout) {
  if (shell_ao_offsets.empty() || system_shell_pair_offsets.empty() ||
      shell_pair_first.empty() ||
      shell_pair_first.size() != shell_pair_second.size() ||
      system_shell_pair_offsets.front() != 0 ||
      system_shell_pair_offsets.back() < 0 ||
      static_cast<std::size_t>(system_shell_pair_offsets.back()) !=
          shell_pair_first.size()) {
    return false;
  }

  std::vector<std::size_t> shell_pair_ao_counts(shell_pair_first.size());
  std::size_t maximum_shell_pair_ao_count = 0;
  for (std::size_t pair = 0; pair < shell_pair_first.size(); ++pair) {
    const std::int32_t first_shell = shell_pair_first[pair];
    const std::int32_t second_shell = shell_pair_second[pair];
    if (first_shell < 0 || second_shell < 0 ||
        static_cast<std::size_t>(first_shell) + 1 >=
            shell_ao_offsets.size() ||
        static_cast<std::size_t>(second_shell) + 1 >=
            shell_ao_offsets.size()) {
      return false;
    }
    const std::int64_t first_begin = shell_ao_offsets[first_shell];
    const std::int64_t first_end = shell_ao_offsets[first_shell + 1];
    const std::int64_t second_begin = shell_ao_offsets[second_shell];
    const std::int64_t second_end = shell_ao_offsets[second_shell + 1];
    if (first_begin < 0 || second_begin < 0 || first_end <= first_begin ||
        second_end <= second_begin) {
      return false;
    }
    const std::size_t first_count =
        static_cast<std::size_t>(first_end - first_begin);
    const std::size_t second_count =
        static_cast<std::size_t>(second_end - second_begin);
    std::size_t ao_pair_count = 0;
    if (first_shell == second_shell) {
      std::size_t first_plus_one = 0;
      if (!checked_task_add(first_count, 1, first_plus_one) ||
          !checked_task_multiply(first_count, first_plus_one,
                                 ao_pair_count)) {
        return false;
      }
      ao_pair_count /= 2;
    } else if (!checked_task_multiply(first_count, second_count,
                                      ao_pair_count)) {
      return false;
    }
    shell_pair_ao_counts[pair] = ao_pair_count;
    maximum_shell_pair_ao_count =
        std::max(maximum_shell_pair_ao_count, ao_pair_count);
  }

  DirectQuartetTaskLayout made{};
  for (std::size_t system = 0;
       system + 1 < system_shell_pair_offsets.size(); ++system) {
    const std::int64_t begin_value = system_shell_pair_offsets[system];
    const std::int64_t end_value = system_shell_pair_offsets[system + 1];
    if (begin_value < 0 || end_value < begin_value) return false;
    const std::size_t begin = static_cast<std::size_t>(begin_value);
    const std::size_t end = static_cast<std::size_t>(end_value);
    if (end > shell_pair_ao_counts.size()) return false;
    for (std::size_t first_pair = begin; first_pair < end; ++first_pair) {
      for (std::size_t second_pair = begin; second_pair <= first_pair;
           ++second_pair) {
        std::size_t ao_quartet_count = 0;
        if (first_pair == second_pair) {
          std::size_t first_plus_one = 0;
          if (!checked_task_add(shell_pair_ao_counts[first_pair], 1,
                                first_plus_one) ||
              !checked_task_multiply(shell_pair_ao_counts[first_pair],
                                     first_plus_one, ao_quartet_count)) {
            return false;
          }
          ao_quartet_count /= 2;
        } else if (!checked_task_multiply(shell_pair_ao_counts[first_pair],
                                          shell_pair_ao_counts[second_pair],
                                          ao_quartet_count)) {
          return false;
        }
        std::size_t rounded = 0;
        if (ao_quartet_count == 0 ||
            !checked_task_add(ao_quartet_count,
                              kDirectQuartetThreads - 1, rounded)) {
          return false;
        }
        const std::size_t tile_count = rounded / kDirectQuartetThreads;
        if (!checked_task_add(made.shell_quartet_count, 1,
                              made.shell_quartet_count) ||
            !checked_task_add(made.exact_tile_count, tile_count,
                              made.exact_tile_count)) {
          return false;
        }
      }
    }
  }
  std::size_t maximum_uniform_ao_quartets = 0;
  std::size_t rounded_uniform_ao_quartets = 0;
  if (!checked_task_multiply(maximum_shell_pair_ao_count,
                             maximum_shell_pair_ao_count,
                             maximum_uniform_ao_quartets) ||
      !checked_task_add(maximum_uniform_ao_quartets,
                        kDirectQuartetThreads - 1,
                        rounded_uniform_ao_quartets)) {
    return false;
  }
  made.maximum_tiles_per_shell_quartet =
      rounded_uniform_ao_quartets / kDirectQuartetThreads;
  if (!checked_task_multiply(made.shell_quartet_count,
                             made.maximum_tiles_per_shell_quartet,
                             made.uniform_tile_count)) {
    return false;
  }
  layout = made;
  return true;
}

}  // namespace qce::scf::detail

#endif

#include "scf/cuda/queue_plan.hpp"

#include <algorithm>
#include <limits>

#include "runtime/bounded_workspace.hpp"
#include "scf/aot_shell_registry.hpp"
#include "scf/cuda/direct_constants.hpp"

namespace vibeqc::scf::cuda_execution {

/** Host queue partitioning and partial-page arithmetic. Keep enumeration and truncation rules
 * identical to the device consumer. */
/**
 * Partition the bounded generated-task cache among non-streaming classes.
 *
 * Potential quartet counts are derived from the ten shell-pair angular
 * histograms, so setup stays O(N_shell_pairs) even when the exact quartet
 * topology has billions of entries. Dominant Fock classes need no slice
 * because they enumerate pair segments directly. A class that outgrows its
 * proportional slice is replayed independently through exact-class paged
 * compaction and its generated consumer; it cannot invalidate faster routes
 * for unrelated classes.
 */
std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1>
make_bounded_generated_task_offsets(
    const HostBatch& host, std::size_t task_capacity,
    std::array<std::uint64_t, detail::kDirectQuartetShellClassCount>* upper_bounds) {
  std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1> offsets{};
  if (task_capacity == 0) return offsets;

  std::array<bool, detail::kDirectQuartetShellClassCount> force_compiled{};
  std::array<bool, detail::kDirectQuartetShellClassCount> fock_compiled{};
  const auto include_kernels = [](std::array<bool, detail::kDirectQuartetShellClassCount>& compiled,
                                  const generated::ShellKernelMetadata* kernels,
                                  std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
      if (kernels[index].shell_class < compiled.size()) {
        compiled[kernels[index].shell_class] = true;
      }
    }
  };
  std::size_t force_count = 0;
  const generated::ShellKernelMetadata* force_kernels =
      generated::selected_shell_kernels(force_count);
  include_kernels(force_compiled, force_kernels, force_count);
  std::size_t fock_count = 0;
  const generated::ShellKernelMetadata* fock_kernels =
      generated::selected_fock_shell_kernels(fock_count);
  include_kernels(fock_compiled, fock_kernels, fock_count);
  std::array<bool, detail::kDirectQuartetShellClassCount> compiled{};
  for (std::size_t shell_class = 0; shell_class < compiled.size(); ++shell_class) {
    const bool queued_fock =
        fock_compiled[shell_class] &&
        (kStreamingFockShellClassMask & (std::uint64_t{1} << shell_class)) == 0U;
    // Fock-only spd rows still compile an exact dormant force symbol for the
    // bounded path.  Include their fixed-capacity slices without advertising
    // them to ordinary fixed-topology force planning.
    const bool queued_force =
        (force_compiled[shell_class] ||
         (fock_compiled[shell_class] &&
          (kStreamingFockShellClassMask & (std::uint64_t{1} << shell_class)) != 0U)) &&
        (kDdddShellClassMask & (std::uint64_t{1} << shell_class)) == 0U;
    // Direct pair-class streams consume no generated descriptor slice.
    compiled[shell_class] = queued_force || queued_fock;
  }

  std::array<std::uint64_t, detail::kDirectQuartetShellClassCount> weights{};
  for (std::size_t system = 0; system + 1 < host.system_shell_pair_offsets.size(); ++system) {
    std::array<std::uint64_t, detail::kDirectShellPairClassCount> pair_class_counts{};
    const std::size_t pair_begin = static_cast<std::size_t>(host.system_shell_pair_offsets[system]);
    const std::size_t pair_end =
        static_cast<std::size_t>(host.system_shell_pair_offsets[system + 1]);
    for (std::size_t pair = pair_begin; pair < pair_end; ++pair) {
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      const std::size_t pair_class = detail::direct_shell_pair_class(
          host.shell_angular[first_shell], host.shell_angular[second_shell]);
      ++pair_class_counts[pair_class];
    }
    for (std::size_t high = 0; high < pair_class_counts.size(); ++high) {
      for (std::size_t low = 0; low <= high; ++low) {
        const std::uint64_t high_count = pair_class_counts[high];
        const std::uint64_t low_count = pair_class_counts[low];
        const std::uint64_t quartets =
            high == low ? high_count * (high_count + 1U) / 2U : high_count * low_count;
        weights[high * (high + 1U) / 2U + low] += quartets;
      }
    }
  }
  if (upper_bounds != nullptr) *upper_bounds = weights;

  std::size_t compiled_count = 0;
  std::uint64_t total_weight = 0;
  for (std::size_t shell_class = 0; shell_class < compiled.size(); ++shell_class) {
    if (!compiled[shell_class]) continue;
    ++compiled_count;
    total_weight += weights[shell_class];
  }
  if (compiled_count == 0) return offsets;

  std::array<std::uint64_t, detail::kDirectQuartetShellClassCount> capacities{};
  const std::uint64_t minimum_per_class =
      std::min<std::uint64_t>(4096U, task_capacity / compiled_count);
  const std::uint64_t reserved = minimum_per_class * compiled_count;
  const std::uint64_t proportional = task_capacity - reserved;
  std::uint64_t assigned = 0;
  for (std::size_t shell_class = 0; shell_class < compiled.size(); ++shell_class) {
    if (!compiled[shell_class]) continue;
    const std::uint64_t share = total_weight == 0
                                    ? proportional / compiled_count
                                    : proportional * weights[shell_class] / total_weight;
    capacities[shell_class] = minimum_per_class + share;
    assigned += capacities[shell_class];
  }
  // Integer division leaves fewer than one task per compiled class. Assign
  // those slots round-robin; exact proportions do not depend on the tail.
  for (std::size_t shell_class = 0; assigned < task_capacity;
       shell_class = (shell_class + 1U) % compiled.size()) {
    if (!compiled[shell_class]) continue;
    ++capacities[shell_class];
    ++assigned;
  }

  std::uint64_t cursor = 0;
  for (std::size_t shell_class = 0; shell_class < compiled.size(); ++shell_class) {
    offsets[shell_class] = static_cast<std::uint32_t>(cursor);
    cursor += capacities[shell_class];
  }
  offsets[detail::kDirectQuartetShellClassCount] = static_cast<std::uint32_t>(cursor);
  return offsets;
}

/** Return shell classes that can actually occur within this batch topology. */

std::uint64_t present_direct_shell_class_mask(const HostBatch& host) {
  static_assert(detail::kDirectQuartetShellClassCount <= 64U);
  std::uint64_t mask = 0U;
  for (std::size_t system = 0; system + 1U < host.system_shell_pair_offsets.size(); ++system) {
    std::array<bool, detail::kDirectShellPairClassCount> present_pairs{};
    const std::size_t pair_begin = static_cast<std::size_t>(host.system_shell_pair_offsets[system]);
    const std::size_t pair_end =
        static_cast<std::size_t>(host.system_shell_pair_offsets[system + 1U]);
    for (std::size_t pair = pair_begin; pair < pair_end; ++pair) {
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      present_pairs[detail::direct_shell_pair_class(host.shell_angular[first_shell],
                                                    host.shell_angular[second_shell])] = true;
    }
    for (std::size_t high = 0; high < present_pairs.size(); ++high) {
      if (!present_pairs[high]) continue;
      for (std::size_t low = 0; low <= high; ++low) {
        if (!present_pairs[low]) continue;
        const std::size_t shell_class = high * (high + 1U) / 2U + low;
        mask |= std::uint64_t{1} << shell_class;
      }
    }
  }
  return mask;
}

/** Build class-major/system-major shell-pair segments for AOT streaming. */
bool make_bounded_stream_shell_pair_order(const HostBatch& host,
                                          std::vector<std::uint32_t>& pair_order,
                                          std::vector<std::uint32_t>& pair_class_offsets) {
  if (host.system_shell_pair_offsets.empty() ||
      host.shell_pair_first.size() != host.shell_pair_second.size() ||
      host.shell_pair_first.size() >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return false;
  }
  const std::size_t batch_size = host.system_shell_pair_offsets.size() - 1U;
  const std::size_t stride = batch_size + 1U;
  const std::size_t total_pairs = host.shell_pair_first.size();
  const std::size_t class_count = detail::kDirectShellPairClassCount;
  // Count each pair exactly once, then fill the class-major segments from
  // those prefix offsets.  The previous class-at-a-time implementation
  // revisited every pair for all ten classes, which made host setup needlessly
  // sensitive to the number of angular classes present in a large bucket.
  std::vector<std::uint32_t> class_counts(class_count * batch_size, 0U);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::size_t pair_begin = static_cast<std::size_t>(host.system_shell_pair_offsets[system]);
    const std::size_t pair_end =
        static_cast<std::size_t>(host.system_shell_pair_offsets[system + 1U]);
    for (std::size_t pair = pair_begin; pair < pair_end; ++pair) {
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      const std::size_t pair_class = detail::direct_shell_pair_class(
          host.shell_angular[first_shell], host.shell_angular[second_shell]);
      ++class_counts[pair_class * batch_size + system];
    }
  }

  pair_class_offsets.assign(class_count * stride, 0U);
  std::vector<std::uint32_t> class_write_offsets(class_count * batch_size);
  std::size_t cursor = 0;
  for (std::size_t pair_class = 0; pair_class < class_count; ++pair_class) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const std::size_t segment = pair_class * batch_size + system;
      pair_class_offsets[pair_class * stride + system] = static_cast<std::uint32_t>(cursor);
      class_write_offsets[segment] = static_cast<std::uint32_t>(cursor);
      cursor += class_counts[segment];
    }
    pair_class_offsets[pair_class * stride + batch_size] = static_cast<std::uint32_t>(cursor);
  }
  if (cursor != total_pairs) return false;

  pair_order.resize(total_pairs);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::size_t pair_begin = static_cast<std::size_t>(host.system_shell_pair_offsets[system]);
    const std::size_t pair_end =
        static_cast<std::size_t>(host.system_shell_pair_offsets[system + 1U]);
    for (std::size_t pair = pair_begin; pair < pair_end; ++pair) {
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      const std::size_t pair_class = detail::direct_shell_pair_class(
          host.shell_angular[first_shell], host.shell_angular[second_shell]);
      const std::size_t segment = pair_class * batch_size + system;
      pair_order[class_write_offsets[segment]++] = static_cast<std::uint32_t>(pair);
    }
  }
  return pair_order.size() == host.shell_pair_first.size();
}

/** Return the row containing one packed lower-triangle ordinal. */
std::uint64_t bounded_lower_triangle_row(std::uint64_t ordinal, std::uint64_t row_count) {
  std::uint64_t lower = 0U;
  std::uint64_t upper = row_count;
  while (lower < upper) {
    const std::uint64_t middle = lower + (upper - lower) / 2U;
    if (middle * (middle + 1U) / 2U <= ordinal) {
      lower = middle + 1U;
    } else {
      upper = middle;
    }
  }
  return lower == 0U ? 0U : lower - 1U;
}

/** Return the number of triangle rows whose start is below a prefix. */
std::uint64_t bounded_lower_triangle_row_end(std::uint64_t prefix, std::uint64_t row_count) {
  std::uint64_t lower = 0U;
  std::uint64_t upper = row_count;
  while (lower < upper) {
    const std::uint64_t middle = lower + (upper - lower) / 2U;
    if (middle * (middle + 1U) / 2U < prefix) {
      lower = middle + 1U;
    } else {
      upper = middle;
    }
  }
  return lower;
}

BoundedGeneratedPageRange bounded_generated_page_range(
    const std::vector<std::uint32_t>& pair_class_offsets, std::size_t batch_size,
    unsigned high_pair_class, unsigned low_pair_class, std::uint64_t page_begin,
    std::uint32_t page_capacity) {
  const std::size_t stride = batch_size + 1U;
  const std::uint64_t page_end = page_begin > std::numeric_limits<std::uint64_t>::max() -
                                                  static_cast<std::uint64_t>(page_capacity)
                                     ? std::numeric_limits<std::uint64_t>::max()
                                     : page_begin + static_cast<std::uint64_t>(page_capacity);
  std::uint64_t candidate_cursor = 0U;
  std::uint64_t bra_cursor = 0U;
  bool found_begin = false;
  bool found_end = false;
  std::uint64_t bra_begin = 0U;
  std::uint64_t bra_end = 0U;
  for (std::size_t system = 0; system < batch_size; ++system) {
    const std::uint32_t high_begin = pair_class_offsets[high_pair_class * stride + system];
    const std::uint32_t high_end = pair_class_offsets[high_pair_class * stride + system + 1U];
    const std::uint32_t low_begin = pair_class_offsets[low_pair_class * stride + system];
    const std::uint32_t low_end = pair_class_offsets[low_pair_class * stride + system + 1U];
    const std::uint64_t high_count = high_end - high_begin;
    const std::uint64_t low_count = low_end - low_begin;
    const bool same_pair_class = high_pair_class == low_pair_class;
    const std::uint64_t system_candidates =
        same_pair_class ? high_count * (high_count + 1U) / 2U : high_count * low_count;
    if (system_candidates == 0U) {
      bra_cursor += high_count;
      continue;
    }

    const std::uint64_t system_end = candidate_cursor + system_candidates;
    if (!found_begin && page_begin < system_end) {
      const std::uint64_t local_begin =
          page_begin > candidate_cursor ? page_begin - candidate_cursor : 0U;
      const std::uint64_t local_bra = same_pair_class
                                          ? bounded_lower_triangle_row(local_begin, high_count)
                                          : local_begin / low_count;
      bra_begin = bra_cursor + std::min(high_count, local_bra);
      found_begin = true;
    }
    if (found_begin && !found_end && page_end <= system_end) {
      const std::uint64_t local_end = page_end - candidate_cursor;
      // ``ceil`` keeps a row whose final ket falls inside the page.  The
      // exact ket loop clips the row to the page interval below.
      const std::uint64_t rows_end =
          same_pair_class ? bounded_lower_triangle_row_end(local_end, high_count)
                          : std::min(high_count, (local_end + low_count - 1U) / low_count);
      bra_end = bra_cursor + rows_end;
      found_end = true;
    }
    candidate_cursor = system_end;
    bra_cursor += high_count;
  }
  if (!found_begin) {
    bra_begin = bra_cursor;
    bra_end = bra_cursor;
  } else if (!found_end) {
    bra_end = bra_cursor;
  }
  return {
      candidate_cursor,
      static_cast<std::uint32_t>(bra_begin),
      static_cast<std::uint32_t>(bra_end),
  };
}

}  // namespace vibeqc::scf::cuda_execution

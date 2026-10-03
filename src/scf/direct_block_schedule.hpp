#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "scf/direct_task_layout.hpp"

namespace generativeqc::scf::detail {

/** Geometry-only preparation; density and exact shell/AO gates stay downstream. */
struct BoundedDirectHostSchedule {
  std::vector<std::uint32_t> pair_order;
  std::vector<std::uint64_t> block_prefix;
};

/** Sort each system's Schwarz bounds and index only geometry-live block pairs.
 * Preparation costs O(P log P + B log B), with P shell pairs and B pair blocks.
 * Consumers perform O(S log B) row decoding for S admitted block products,
 * rather than enumerating all B(B+1)/2 products. The worst case remains dense.
 * Strict-less rejection, ties, zero screening and FP64 multiplication match
 * the existing first block gate. No density-dependent state is cached. */
inline BoundedDirectHostSchedule make_bounded_direct_schedule(
    const std::vector<std::int64_t>& system_offsets, const std::vector<double>& bounds,
    double screening) {
  if (system_offsets.empty() || system_offsets.front() != 0 || system_offsets.back() < 0 ||
      static_cast<std::size_t>(system_offsets.back()) != bounds.size() ||
      !std::is_sorted(system_offsets.begin(), system_offsets.end()) ||
      bounds.size() > std::numeric_limits<std::uint32_t>::max() || !std::isfinite(screening) ||
      screening < 0)
    throw std::invalid_argument("invalid bounded Schwarz schedule domain");
  for (double bound : bounds)
    if (!std::isfinite(bound) || bound < 0)
      throw std::invalid_argument("invalid bounded Schwarz bound");
  BoundedDirectHostSchedule result;
  result.pair_order.resize(bounds.size());
  std::iota(result.pair_order.begin(), result.pair_order.end(), 0U);
  result.block_prefix.push_back(0U);
  constexpr auto block_size = kBoundedDirectShellPairBlockSize;
  for (std::size_t system = 0; system + 1U < system_offsets.size(); ++system) {
    const auto begin = static_cast<std::size_t>(system_offsets[system]);
    const auto end = static_cast<std::size_t>(system_offsets[system + 1U]);
    std::sort(result.pair_order.begin() + begin, result.pair_order.begin() + end,
              [&](auto first, auto second) {
                return bounds[first] == bounds[second] ? first < second
                                                       : bounds[first] > bounds[second];
              });
    const auto blocks = (end - begin + block_size - 1U) / block_size;
    for (std::size_t row = 0; row < blocks; ++row) {
      const double first = bounds[result.pair_order[begin + row * block_size]];
      std::size_t lower = 0, upper = row + 1U;
      while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2U;
        const double second = bounds[result.pair_order[begin + middle * block_size]];
        if (first * second < screening)
          upper = middle;
        else
          lower = middle + 1U;
      }
      result.block_prefix.push_back(result.block_prefix.back() + lower);
    }
  }
  return result;
}

}  // namespace generativeqc::scf::detail

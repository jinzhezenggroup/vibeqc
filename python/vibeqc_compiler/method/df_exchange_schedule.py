"""Capacity and source-work schedule for exact streamed DF occupied exchange.

The native owner supplies the dense fallback traversal and four disjoint buffer
capacities. This compiler policy chooses raw-source projection before metric
whitening only when it reduces generated values. It knows no SCF reference,
device, orbital lifetime, or density acceptance policy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectedExchangeSchedule:
    """A zero row count means that the existing dense traversal must be used."""

    rows: int = 0
    blocks: int = 0
    generated_rows: int = 0


def projected_exchange_schedule(
    n: int,
    auxiliaries: int,
    rank: int,
    capacity: int,
    dense_row_blocks: int,
    dense_output_blocks: int,
    triangular: bool,
) -> ProjectedExchangeSchedule:
    """Balance row blocks to minimize rereading earlier triangular panels.

    Each buffer holds ``capacity`` doubles. A row needs both ``a * rank``
    projection elements and at least ``n`` raw elements. For a fixed number of
    blocks, the smallest feasible row width minimizes repeated prefix rows;
    the last, possibly shorter block is generated only once. Full K has equal
    source work at all widths with the same block count. This balanced choice
    also leaves more room for raw auxiliary reuse.
    """
    if min(n, auxiliaries, rank, dense_row_blocks, dense_output_blocks) <= 0:
        return ProjectedExchangeSchedule()
    limit = (1 << 31) - 1
    if n > limit // n or rank > n or auxiliaries > limit // rank:
        return ProjectedExchangeSchedule()
    maximum_rows = min(n, capacity // (auxiliaries * rank), capacity // n)
    if maximum_rows <= 0:
        return ProjectedExchangeSchedule()
    blocks = (n + maximum_rows - 1) // maximum_rows
    rows = (n + blocks - 1) // blocks
    generated = n + rows * blocks * (blocks - 1) // 2 if triangular else n * blocks
    if generated >= n * dense_row_blocks * dense_output_blocks:
        return ProjectedExchangeSchedule()
    return ProjectedExchangeSchedule(rows, blocks, generated)


def native_header() -> str:
    """Emit the same shape-only policy for native allocation and execution."""
    return r"""// Generated from vibeqc_compiler.method.df_exchange_schedule; do not edit.
#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
namespace vibeqc::scf::generated {
struct ProjectedExchangeSchedule {
  std::size_t rows{}, blocks{}, generated_rows{};
};
inline ProjectedExchangeSchedule projected_exchange_schedule(
    std::size_t n, std::size_t a, std::size_t rank, std::size_t capacity,
    std::size_t dense_row_blocks, std::size_t dense_output_blocks, bool triangular) noexcept {
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (!n || !a || !rank || !dense_row_blocks || !dense_output_blocks ||
      n > limit / n || rank > n || a > limit / rank) return {};
  const auto maximum_rows = std::min({n, capacity / (a * rank), capacity / n});
  if (!maximum_rows) return {};
  const auto blocks = 1 + (n - 1) / maximum_rows;
  const auto rows = 1 + (n - 1) / blocks;
  // n*n <= INT_MAX bounds this triangular row census in size_t.
  const auto generated = triangular ? n + rows * blocks * (blocks - 1) / 2 : n * blocks;
  const auto dense_rows = static_cast<long double>(n) * dense_row_blocks * dense_output_blocks;
  if (generated >= dense_rows) return {};
  return {rows, blocks, generated};
}
}  // namespace vibeqc::scf::generated
"""

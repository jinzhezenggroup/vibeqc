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
    blocks, the smallest feasible row width minimizes repeated prefix rows.
    Triangular traversal retains the preceding row panel across the next row
    load, so only earlier prefixes need regeneration. The last, possibly
    shorter block is generated only once. Full K has equal
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
    generated = (
        n + rows * (blocks - 1) * max(0, blocks - 2) // 2 if triangular else n * blocks
    )
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
  const auto prefixes = blocks > 2 ? (blocks - 1) * (blocks - 2) / 2 : 0;
  const auto generated = triangular ? n + rows * prefixes : n * blocks;
  const auto dense_rows = static_cast<long double>(n) * dense_row_blocks * dense_output_blocks;
  if (generated >= dense_rows) return {};
  return {rows, blocks, generated};
}

// Execute an admitted shape with two retained projection slots. Project and
// contract callbacks bind storage/BLAS in the native owner and return false on
// failure. No pointer or cache identity survives this invocation; stream order
// protects every reuse, including a captured graph replay with new coefficients.
template <class Project, class Contract>
bool visit_projected_exchange(std::size_t n, std::size_t rows, bool triangular,
                              Project&& project, Contract&& contract) {
  if (!n || !rows || rows > n) return false;
  for (std::size_t r = 0, block = 0; r < n; r += rows, ++block) {
    const auto nr = std::min(rows, n - r);
    const std::size_t left = triangular ? block % 2 : 0;
    const auto right = 1 - left;
    // The preceding outer row is still in the other slot. Visit it before
    // overwriting that slot with an earlier prefix; the new row stays live.
    if (!project(r, nr, left)) return false;
    if (triangular) {
      if (!contract(r, nr, r, nr, left, left, true)) return false;
      for (std::size_t c = r; c != 0;) {
        c -= rows;
        const bool retained = c + rows == r;
        if (!retained && !project(c, rows, right)) return false;
        if (!contract(r, nr, c, rows, left, right, retained)) return false;
      }
    } else {
      // Preserve the explicit full-matrix traversal and its source census.
      for (std::size_t c = 0; c < n; c += rows) {
        const auto nc = std::min(rows, n - c);
        const bool diagonal = c == r;
        if (!diagonal && !project(c, nc, right)) return false;
        if (!contract(r, nr, c, nc, left, diagonal ? left : right, diagonal)) return false;
      }
    }
  }
  return true;
}
}  // namespace vibeqc::scf::generated
"""

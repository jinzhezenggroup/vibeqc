"""Host-executed census and scratch-lifetime contracts; no GPU performance claim."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_response_projection_work_and_budget(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "response_projection.cpp"
    source.write_text(r"""
#include "scf/df_response_projection_policy.hpp"
#include <array>
#include <cassert>
#include <limits>

// Exception type is part of the contract: invalid shapes are not capacity
// failures, and arithmetic overflow must never turn into a smaller allocation.
template<class Error, class F> void rejects(F operation) {
  bool rejected = false;
  try { operation(); }
  catch (const Error&) { rejected = true; }
  assert(rejected);
}

int main() {
  using namespace vibeqc::scf;
  const auto five = df_fitted_panel_work(3712, 771, 295296);
  if (five.panels != 5 || five.reader_calls != 25 ||
      five.projected_columns != 18560 || five.staging_elements != 5480693760ULL) return 1;
  const auto eight = df_fitted_panel_work(3712, 481, 295296);
  if (eight.panels != 8 || eight.reader_calls != 64 ||
      eight.projected_columns != 29696) return 2;
  const auto full = df_fitted_panel_work(9, 20, 7);
  if (full.panels != 1 || full.reader_calls != 1 || full.projected_columns != 9) return 3;
  // Enumerate the actual outer read and off-diagonal rereads independently of
  // the closed-form census, including ragged and oversized panels.
  for (std::size_t a : {1U, 9U, 37U}) {
    for (std::size_t tile : {1U, 3U, 8U, 64U}) {
      std::size_t panels = 0, calls = 0, columns = 0;
      for (std::size_t p = 0; p < a; p += tile) {
        ++panels;
        ++calls;
        columns += std::min(tile, a - p);
        for (std::size_t q = 0; q < a; q += tile) {
          if (q == p) continue;
          ++calls;
          columns += std::min(tile, a - q);
        }
      }
      const auto work = df_fitted_panel_work(a, tile, 7);
      assert(work.panels == panels && work.reader_calls == calls);
      assert(work.projected_columns == columns && work.staging_elements == columns * 7);
    }
  }
  const auto capacity = 3712ULL * 160 * 160;
  for (bool pair_major : {false, true}) {
    for (std::size_t n : {1U, 7U, 16U, 768U}) {
      for (std::size_t r : {std::size_t{0}, n / 2, n}) {
        for (std::size_t cap : {1U, 3U, 64U}) {
          const auto p = plan_df_occupied_projection_batch(n, 3712, r, capacity, cap, pair_major);
          if (p.columns > cap || p.total_elements > capacity ||
              p.total_elements != p.input_elements + p.staging_elements + p.product_elements)
            return 4;
          if (p.columns) {
            const auto exact = plan_df_occupied_projection_batch(
                n, 3712, r, p.total_elements, cap, pair_major);
            const auto less = plan_df_occupied_projection_batch(
                n, 3712, r, p.total_elements - 1, cap, pair_major);
            if (exact.columns != p.columns || less.columns != p.columns - 1) return 5;
          }
        }
      }
    }
  }
  const auto bounded = plan_df_occupied_projection_batch(768, 3712, 160, capacity, 64, false);
  if (bounded.columns != 64 || bounded.total_elements != 45613056) return 6;
  const auto empty = plan_df_occupied_projection_batch(7, 9, 3, 0, 64, false);
  if (empty.columns || empty.total_elements) return 7;
  // More scratch cannot admit columns past the end of the auxiliary axis.
  const auto tail = plan_df_occupied_projection_batch(7, 9, 3, capacity, 64, true);
  assert(tail.columns == 9 && tail.input_elements == 441 &&
         tail.staging_elements == 441 && tail.product_elements == 189 &&
         tail.total_elements == 1071);
  for (bool pair_major : {false, true}) {
    const auto zero_rank = plan_df_occupied_projection_batch(1, 1, 0, 2, 1, pair_major);
    assert(zero_rank.columns == 1 && zero_rank.product_elements == 0);
    assert(zero_rank.total_elements == (pair_major ? 2U : 1U));
  }
  for (const auto shape : {std::array<std::size_t, 3>{0, 1, 1},
                           std::array<std::size_t, 3>{1, 0, 1},
                           std::array<std::size_t, 3>{1, 1, 0}})
    rejects<std::invalid_argument>([&] { df_fitted_panel_work(shape[0], shape[1], shape[2]); });
  for (const auto shape : {std::array<std::size_t, 4>{0, 1, 0, 1},
                           std::array<std::size_t, 4>{1, 0, 0, 1},
                           std::array<std::size_t, 4>{1, 1, 2, 1},
                           std::array<std::size_t, 4>{1, 1, 0, 0}})
    rejects<std::invalid_argument>([&] {
      plan_df_occupied_projection_batch(shape[0], shape[1], shape[2], 0, shape[3], false);
    });
  const auto huge = std::numeric_limits<std::size_t>::max();
  assert(df_response_checked_product(huge, 0) == 0);
  assert(df_response_checked_product(huge, 1) == huge);
  assert(df_response_checked_sum(huge, 0) == huge);
  const auto single = df_fitted_panel_work(huge, huge, 1);
  assert(single.panels == 1 && single.projected_columns == huge);
  rejects<std::overflow_error>([&] { df_fitted_panel_work(huge, 1, 1); });
  rejects<std::overflow_error>([&] { df_fitted_panel_work(2, 2, huge); });
  rejects<std::overflow_error>([&] {
    plan_df_occupied_projection_batch(huge, 1, 0, 1, 1, false);
  });
  // Each matrix fits on its own; the combined scratch stride does not.
  const auto side = (std::size_t{1} << (std::numeric_limits<std::size_t>::digits / 2)) - 1;
  rejects<std::overflow_error>([&] {
    plan_df_occupied_projection_batch(side, 1, 0, huge, 1, true);
  });
  rejects<std::overflow_error>([&] {
    plan_df_occupied_projection_batch(side, 1, side, huge, 1, false);
  });
  return 0;
}
""")
    executable = tmp_path / "response_projection"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)

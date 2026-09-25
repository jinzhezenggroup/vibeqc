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
#include <limits>
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
  bool overflow = false, invalid = false;
  try { (void)plan_df_occupied_projection_batch(
      std::numeric_limits<std::size_t>::max(), 1, 0, 1, 1, false); }
  catch (const std::overflow_error&) { overflow = true; }
  try { (void)df_fitted_panel_work(1, 0, 1); }
  catch (const std::invalid_argument&) { invalid = true; }
  return overflow && invalid ? 0 : 8;
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

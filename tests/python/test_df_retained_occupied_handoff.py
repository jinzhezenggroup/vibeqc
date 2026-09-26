"""Execute the native owner-selection block without allocating CUDA storage."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_retained_occupied_handoff_selects_one_bridge_representation(
    tmp_path: Path,
) -> None:
    """Validate views after selection, including invalid-token and auto fallbacks."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/cuda/df_force_response.cpp").read_text()
    start = source.index("    CudaDfOccupiedResponseView owned_factors;")
    end = source.index("    // The diagnostic upload route", start)
    selection = source[start:end]
    unit = tmp_path / "handoff.cpp"
    unit.write_text(
        r"""#include <cstddef>
#include <string>
#include <string_view>
constexpr int VIBEQC_STATUS_SUCCESS = 0;
struct Plan { bool integral_source = true, streamed = false; };
struct Metric { bool full_rank = true; };
struct TensorView { const double* data = nullptr; };
struct CudaDfOccupiedResponseView { unsigned owner_identity = 0; };
int selected_calls = 0;
int select_occupied_response_factors(Plan&, int, bool valid_token, int, std::size_t,
                                    CudaDfOccupiedResponseView& view, std::string&) {
  ++selected_calls;
  if (valid_token) view.owner_identity = 7;
  return VIBEQC_STATUS_SUCCESS;
}
int select_corrected_occupied_response_factor(Plan&, int, bool, int, std::size_t,
                                             CudaDfOccupiedResponseView&, std::string&) {
  return VIBEQC_STATUS_SUCCESS;
}
int run(bool streamed, std::string_view space, bool valid_token, bool borrow,
        bool expected_occupied, bool expected_select) {
  Plan storage{true, streamed};
  auto* plan = &storage;
  const int system = 0, terms = 0;
  const bool final_state = valid_token;
  const std::size_t maximum_bytes = 4096;
  Metric metric;
  std::string detail;
  double retained = 1;
  TensorView whitened{streamed ? nullptr : &retained};
  TensorView packed_raw{streamed ? nullptr : &retained};
  selected_calls = 0;
"""
        + selection
        + r"""
  if ((owned_factors.owner_identity != 0) != expected_occupied) return 1;
  if ((selected_calls != 0) != expected_select) return 2;
  // This is the bridge's mutual-exclusion requirement, checked after the real
  // caller block has selected and revoked its views.
  if (owned_factors.owner_identity && (whitened.data || packed_raw.data)) return 3;
  if (!streamed && !expected_occupied &&
      (whitened.data != &retained || packed_raw.data != &retained)) return 4;
  return 0;
}
int main() {
  // Both fusion modes share this handoff; fusion only affects the consumer.
  for (bool valid : {false, true}) {
    if (run(false, "occupied", valid, false, valid, true)) return 10;
    if (run(false, "auto", valid, false, false, false)) return 11;
    if (run(false, "dense", valid, false, false, false)) return 12;
    if (run(false, "occupied", valid, true, false, false)) return 13;
    if (run(true, "auto", valid, false, valid, true)) return 14;
    if (run(true, "occupied", valid, false, valid, true)) return 15;
  }
}
"""
    )
    binary = tmp_path / "handoff"
    subprocess.run(
        [compiler, "-std=c++20", "-O2", str(unit), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(binary)], check=True, timeout=10)

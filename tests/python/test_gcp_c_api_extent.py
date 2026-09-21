"""The real gCP C entrypoint must reject unrepresentable extents before execution."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_gcp_c_api_rejects_wrapped_and_signed_coordinate_extents(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    headers = {
        "vibeqc/vibeqc.h": """
#pragma once
#include <cstdint>
using vibeqc_status = int;
constexpr int VIBEQC_STATUS_SUCCESS=0, VIBEQC_STATUS_INVALID_ARGUMENT=1;
constexpr int VIBEQC_STATUS_NOT_IMPLEMENTED=2, VIBEQC_STATUS_NUMERICAL_FAILURE=3;
constexpr int VIBEQC_STATUS_INTERNAL_ERROR=4;
""",
        "api/error.hpp": """
#pragma once
#include "vibeqc/vibeqc.h"
namespace vibeqc::api {
inline vibeqc_status map_exception() { return VIBEQC_STATUS_INTERNAL_ERROR; }
}
""",
        "dft/dispersion/gcp_r2scan3c.hpp": """
#pragma once
#include <cstdint>
namespace vibeqc::dft::dispersion {
enum class GCPStatus { success, invalid_argument, unsupported_element,
                       coincident_atoms, numerical_failure };
inline unsigned calls=0;
inline int r2scan3c_gcp_parameters() { return 0; }
inline GCPStatus evaluate_r2scan3c_gcp(int, const std::int32_t*, const double*,
                                    int, double*, double*) {
  ++calls;
  return GCPStatus::success;
}
}
""",
    }
    for name, text in headers.items():
        header = tmp_path / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text(text)
    # Only the downstream chemistry is a sentinel. Compile the production ABI
    # unchanged, so the test cannot pass against a copied validation predicate.
    source = ROOT / "src/api/c_api_gcp.cpp"
    driver = tmp_path / "driver.cpp"
    driver.write_text(
        '#include "'
        + source.as_posix()
        + '"\n'
        + r"""
#include <iostream>
#include <limits>
int main() {
  const auto signed_limit = static_cast<std::uint32_t>(INT_MAX / 3);
  const auto wrapped = std::numeric_limits<std::uint32_t>::max() / 3u + 1u;
  std::int32_t z=1;
  double xyz=1, energy=19, gradient=23;
  struct Case { std::uint32_t n, coordinates, gradients; bool want_gradient, valid; };
  const Case cases[] = {
    {wrapped, 3u*wrapped, 3u*wrapped, true, false},
    {wrapped, 3u*wrapped, 0, false, false},
    {signed_limit+1u, 3u*(signed_limit+1u), 3u*(signed_limit+1u), true, false},
    {static_cast<std::uint32_t>(INT_MAX), 3u*static_cast<std::uint32_t>(INT_MAX),
     3u*static_cast<std::uint32_t>(INT_MAX), true, false},
    {signed_limit, 3u*signed_limit, 3u*signed_limit, true, true},
    {1, 3, 3, true, true}, {1, 3, 0, false, true},
    {1, 4, 4, true, false}, {1, 3, 2, true, false}, {0, 0, 0, false, false},
  };
  for (const auto& c : cases) {
    vibeqc::dft::dispersion::calls=0;
    const int status=vibeqc_r2scan3c_gcp_evaluate(
        &z,c.n,&xyz,c.coordinates,&energy,c.want_gradient?&gradient:nullptr,c.gradients);
    if (status != (c.valid?VIBEQC_STATUS_SUCCESS:VIBEQC_STATUS_INVALID_ARGUMENT) ||
        vibeqc::dft::dispersion::calls != (c.valid?1u:0u)) {
      std::cerr << "invalid admission: n=" << c.n << " count=" << c.coordinates << '\n';
      return 1;
    }
    if (energy!=19 || gradient!=23) return 2;
  }
}
"""
    )
    executable = tmp_path / "gcp_extent"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(tmp_path),
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr or result.stdout

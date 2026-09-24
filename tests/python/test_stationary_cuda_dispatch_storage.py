"""The generated ordered derivative table must be readable by device functions."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.integral.first_derivative_schedule import (
    COMPONENT_LABELS,
    derivative_dispatch_entries,
)
from vibeqc_compiler.method.stationary_cuda import _emit_stationary_dispatch

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def dispatch() -> tuple:
    rows = derivative_dispatch_entries(COMPONENT_LABELS)
    return rows, _emit_stationary_dispatch(rows)


def test_full_spd_dispatch_uses_device_global_storage(dispatch: tuple) -> None:
    rows, source = dispatch
    assert len(rows) == 10301
    # The complete table exceeds constant space even before padding.
    assert len(rows) * 3 * 4 > 64 * 1024
    assert (
        "static __device__ const StationaryDispatchEntry stationary_dispatch[]"
        in source
    )
    assert "__constant__" not in source


def test_every_dispatch_preserves_kind_and_permutations(
    dispatch: tuple, tmp_path: Path
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    rows, source = dispatch
    expected = []
    for identifier, kind, centers, axes in rows:
        center_values = (*centers, *([-1] * (4 - len(centers))))
        center_text = ",".join(map(str, center_values))
        axes_text = ",".join(map(str, axes))
        expected.append(
            f"{{{identifier},{kind},{len(centers)},{{{center_text}}},{{{axes_text}}}}}"
        )
    prefix = r"""
#include <algorithm>
#include <cstddef>
#define __device__
unsigned expected_kind, expected_rank;
double expected_e[4], expected_c[12];
bool first_derivative(unsigned kind, const double* e, const double* c, double* out) {
  if (kind != expected_kind) return false;
  for (unsigned i = 0; i < 4; ++i) if (e[i] != expected_e[i]) return false;
  for (unsigned i = 0; i < 12; ++i) if (c[i] != expected_c[i]) return false;
  for (unsigned i = 0; i < 3 * expected_rank; ++i) out[i] = 19.0 + 7.0 * i;
  return true;
}
"""
    driver = r"""
struct Expected { unsigned id, kind, rank; int centers[4], axes[3]; };
static const Expected expected[] = { @ROWS@ };
int main() {
  const double e[4] = {0.3, 0.7, 1.2, 1.8};
  const double c[12] = {-2, 3, 5, 7, -11, 13, 17, 19, -23, 29, 31, 37};
  for (const auto& row : expected) {
    expected_kind = row.kind;
    expected_rank = row.rank;
    std::fill_n(expected_e, 4, 1.0);
    std::fill_n(expected_c, 12, 0.0);
    double actual[12], wanted[12];
    std::fill_n(actual, 12, -777.0);
    std::fill_n(wanted, 12, -777.0);
    for (unsigned i = 0; i < row.rank; ++i) {
      expected_e[i] = e[row.centers[i]];
      for (unsigned a = 0; a < 3; ++a) {
        const auto physical = 3 * row.centers[i] + row.axes[a];
        expected_c[3 * i + a] = c[physical];
        wanted[physical] = 19.0 + 7.0 * (3 * i + a);
      }
    }
    if (!vibeqc_stationary_cuda::stationary_first_derivative(row.id, e, c, actual)) return 1;
    for (unsigned i = 0; i < 12; ++i) if (actual[i] != wanted[i]) return 2;
  }
  for (unsigned missing : {9999u, 19999u, 29999u, 40001u}) {
    double out[12]; std::fill_n(out, 12, -777.0);
    if (vibeqc_stationary_cuda::stationary_first_derivative(missing, e, c, out)) return 3;
    for (double value : out) if (value != -777.0) return 4;
  }
}
"""
    path, executable = tmp_path / "dispatch.cpp", tmp_path / "dispatch"
    path.write_text(prefix + source + driver.replace("@ROWS@", ",\n".join(expected)))
    subprocess.run(
        [compiler, "-std=c++17", "-O0", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_emitted_dispatch_compiles_with_nvcc_without_a_device(
    dispatch: tuple, tmp_path: Path
) -> None:
    compiler = os.environ.get("CUDACXX") or shutil.which("nvcc")
    if compiler is None:
        pytest.skip("requires nvcc; no GPU is needed for this compile gate")
    _, source = dispatch
    path = tmp_path / "dispatch.cu"
    path.write_text(
        "#include <cuda_runtime.h>\n"
        "__device__ bool first_derivative(unsigned, const double*, const double*, double*) "
        "{ return true; }\n" + source
    )
    subprocess.run(
        [compiler, "-std=c++17", "-dc", str(path), "-o", str(tmp_path / "dispatch.o")],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

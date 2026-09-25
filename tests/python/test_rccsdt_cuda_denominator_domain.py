"""Host regression for the emitted CUDA denominator-domain guard.

This executes the actual constant f-string region, not a rewritten scientific
formula. It does not qualify CUDA launches, reductions, or complete RCCSD(T).
"""

from __future__ import annotations

import ctypes
import math
import shutil
import subprocess
import typing
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def guard(tmp_path_factory: pytest.TempPathFactory) -> typing.Any:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source_path = Path(__file__).resolve().parents[2] / "tools/generate_rccsdt_cuda.py"
    text = source_path.read_text()
    start = text.index("    const double physical_denominator =")
    end = text.index("    double contribution = 0.0;", start)
    # These escaped braces are literal C++ in the source() f-string.
    body = text[start:end].replace("{{", "{").replace("}}", "}")
    directory = tmp_path_factory.mktemp("triples-denominator")
    cpp = directory / "guard.cpp"
    cpp.write_text(
        """
#include <algorithm>
#include <cmath>
#include <cstddef>
using std::isfinite;
void fail_once(int* error, int code) { if (!*error) *error = code; }
void atomic_min_positive(double* value, double other) {
  *value = std::min(*value, other);
}
extern "C" int guard(double occupied_energy, double virtual_energy,
    std::size_t a, std::size_t b, std::size_t c, double denominator_threshold,
    double* reciprocal) {
  const double eps_o[1]{occupied_energy};
  const double eps_v[3]{virtual_energy,virtual_energy,virtual_energy};
  const std::size_t i=0,j=0,k=0;
  int error_value=0;
  int* error=&error_value;
  double minimum_value=INFINITY;
  double* minimum=&minimum_value;
  for (int once=0;once<1;++once) {
"""
        + body
        + """
    *reciprocal=1.0/denominator;
  }
  return error_value;
}
"""
    )
    lib = directory / "guard.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            str(cpp),
            "-o",
            str(lib),
        ],
        check=True,
        timeout=30,
    )
    library = ctypes.CDLL(str(lib))
    fn = library.guard
    fn.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    fn.restype = ctypes.c_int
    return fn


@pytest.mark.parametrize("indices", [(0, 0, 0), (1, 1, 0)])
def test_finite_gap_with_overflowing_multiplicity_fails_closed(
    guard: typing.Any, indices: tuple[int, int, int]
) -> None:
    # Every input and the -1.2e308 physical gap is finite. Multiplication
    # by either 6 or 2 overflows; old code accepted 1 / -inf == -0.0.
    out = ctypes.c_double(math.nan)
    code = guard(-4e307, 0, *indices, 1e-10, ctypes.byref(out))
    assert code == 3, (
        f"accepted nonfinite scaled denominator: status={code}, reciprocal={out.value}"
    )
    assert math.isnan(out.value), "failed response must not publish a value"


@pytest.mark.parametrize(
    "indices,multiplicity", [((0, 0, 0), 6), ((1, 1, 0), 2), ((2, 1, 0), 1)]
)
def test_ordinary_denominator_is_unchanged(
    guard: typing.Any, indices: tuple[int, int, int], multiplicity: int
) -> None:
    out = ctypes.c_double(math.nan)
    assert guard(-1.0, 1.0, *indices, 1e-10, ctypes.byref(out)) == 0
    assert out.value == 1.0 / (-6.0 * multiplicity)


def test_extreme_nonoverflowing_denominator_is_still_accepted(
    guard: typing.Any,
) -> None:
    out = ctypes.c_double(math.nan)
    assert guard(-4e307, 0, 2, 1, 0, 1e-10, ctypes.byref(out)) == 0
    assert math.isfinite(out.value) and out.value < 0


@pytest.mark.parametrize("occupied,virtual", [(0.0, 0.0), (1.0, 0.0), (-math.inf, 0.0)])
def test_original_invalid_physical_gaps_still_fail(
    guard: typing.Any, occupied: float, virtual: float
) -> None:
    out = ctypes.c_double(math.nan)
    assert guard(occupied, virtual, 0, 0, 0, 1e-10, ctypes.byref(out)) == 1
    assert math.isnan(out.value)

"""Check the live CC force stability certificate against known rotated spectra."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc import _native

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def curvature_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Link the shared numerical owner and compile the actual force wrapper.

    Extracting only this private wrapper avoids duplicating either eigensolver
    mathematics or the complete force pipeline in a small spectral unit test.
    Full molecular/budget tests independently exercise its production caller.
    """
    compiler = shutil.which("c++")
    if compiler is None or sys.platform == "win32":
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/cc/rccsdt_force.cpp").read_text()
    function = source.split("double minimum_symmetric_eigenvalue", 1)[1].split(
        "\n}\n", 1
    )[0]
    body = "double minimum_symmetric_eigenvalue" + function + "\n}\n"
    directory = tmp_path_factory.mktemp("cc-curvature")
    path = directory / "probe.cpp"
    path.write_text(PREFIX + body + MAIN)
    library = Path(_native.load_library(device="cpu")._name).resolve()
    executable = directory / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            str(path),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable


@pytest.mark.parametrize("scale", (1.0, 1e4))
@pytest.mark.parametrize("minimum", (-0.2, 5e-9, 2e-8, 0.3))
def test_force_curvature_rotated_stability_boundary(
    curvature_probe: Path, minimum: float, scale: float
) -> None:
    """Dense rotations preserve the minimum on both sides of the physical gate."""
    rng = np.random.default_rng(1751)
    q, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    spectrum = scale * np.linspace(0.4, 4.0, len(q))
    spectrum[0] = minimum
    # Include exact internal multiplicities; the stability gate concerns only
    # the smallest curvature, never eigenvector identity within a repeated block.
    spectrum[8:12] = 1.5 * scale
    matrix = (q * spectrum) @ q.T
    data = str(len(q)) + "\n" + " ".join(format(x, ".17g") for x in matrix.ravel())
    result = subprocess.run(
        [str(curvature_probe)],
        input=data,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    observed = float(result.stdout)
    # For the ill-conditioned fixture, constructing Q diag(lambda) Q^T in
    # FP64 perturbs its tiny minimum. Compare the actual stored matrix against
    # independent LAPACK, and separately require the same stability decision.
    expected = float(np.linalg.eigvalsh(matrix)[0])
    tolerance = max(2e-12, 64 * np.finfo(float).eps * max(abs(spectrum)))
    assert observed == pytest.approx(expected, abs=tolerance)
    assert (observed > 1e-8) == (expected > 1e-8) == (minimum > 1e-8)


@pytest.mark.parametrize("center", (1.01e-8, 1.04e-8))
@pytest.mark.parametrize("rotation", ("identity", "low_block", "dense"))
def test_force_curvature_clustered_small_eigenvalues(
    curvature_probe: Path, center: float, rotation: str
) -> None:
    """A remote large eigenvalue must not hide a near-threshold low block."""
    matrix = np.array([[center, 3e-10, 0.0], [3e-10, center, 0.0], [0.0, 0.0, 4e4]])
    if rotation == "low_block":
        angle = 0.31
        c, s = np.cos(angle), np.sin(angle)
        q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        matrix = q @ matrix @ q.T
    elif rotation == "dense":
        q, _ = np.linalg.qr(np.random.default_rng(1753).normal(size=(3, 3)))
        matrix = q @ matrix @ q.T
    result = subprocess.run(
        [str(curvature_probe)],
        input="3\n" + " ".join(format(x, ".17g") for x in matrix.ravel()),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    observed = float(result.stdout)
    expected = float(np.linalg.eigvalsh(matrix)[0])
    tolerance = 2e-11 if rotation == "dense" else 2e-13
    assert observed == pytest.approx(expected, abs=tolerance)
    assert (observed > 1e-8) == (expected > 1e-8) == (center - 3e-10 > 1e-8)


PREFIX = r"""
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include "tensor/cpu_linalg.hpp"
namespace tensor = generativeqc::tensor;
std::size_t square(std::size_t n) { return n*n; }
"""
MAIN = r"""
int main() {
  std::size_t n; std::cin >> n;
  std::vector<double> matrix(n*n);
  for(auto& value : matrix) std::cin >> value;
  if(!std::cin) return 2;
  std::cout << std::setprecision(17) << minimum_symmetric_eigenvalue(std::move(matrix), n) << '\n';
}
"""

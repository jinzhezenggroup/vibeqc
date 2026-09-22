"""Compile the production COSX derivative diagnostic without requiring a GPU."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_becke_pair_derivative_contraction_stays_sparse() -> None:
    source = (ROOT / "src/dft/grid.cpp").read_text()
    body = _definition(
        source, "std::vector<double> MolecularGrid::contract_weight_derivative("
    )

    # Each Becke pair depends on at most the owner, a, and b atom blocks.  A
    # nuclear-coordinate sweep inside the pair loop adds an avoidable atom-count
    # factor to every molecular COSX force evaluation.
    assert "std::vector<double> pair_derivative(ncoord)" not in body
    assert "for (double& derivative : pair_derivative)" not in body
    assert body.count("coordinate < ncoord") == 2


@pytest.mark.parametrize("requested_tile", (1, 7, 64))
def test_molecular_cosx_diagnostic_accounts_for_all_device_buffers(
    tmp_path: Path, requested_tile: int
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/dft/cuda_cosx_derivative.cu").read_text()
    header = (ROOT / "src/dft/cuda_cosx.hpp").read_text()
    # Only geometry metadata is synthetic; compile the actual production
    # checked arithmetic and diagnostic, including its public result layout.
    definitions = "\n".join(
        _definition(source, signature)
        for signature in (
            "std::size_t add(",
            "std::size_t mul(",
            "std::size_t derivative_grid_bytes(",
            "CudaCosxMolecularDerivativeDiagnostic cuda_cosx_molecular_derivative_diagnostic(",
        )
    )
    harness = r"""
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "dft/grid.hpp"
using GridOwner = std::remove_cvref_t<decltype(
    std::declval<const vibeqc::dft::MolecularGrid&>().owners())>::value_type;
struct System { std::size_t natom=3, nprimitive=8, nao=5; };
struct AoBasis : System { explicit AoBasis(const System& s) : System(s) {} };
struct MolecularGrid {
  System value;
  const System& system() const { return value; }
  std::size_t point_count() const { return 23; }
};
RESULT_STRUCT;
DEFINITIONS
int main() {
  MolecularGrid grid;
  const auto report = cuda_cosx_molecular_derivative_diagnostic(grid, TILE);
  const auto t = std::min(grid.point_count(), std::size_t{TILE});
  const auto n = grid.system().nao;
  // Independent allocation ledger from the molecular executor: density, ESP,
  // four projected/potential arrays, weights AND sensitivity, nuclear gradient,
  // owner indices, and the device error flag.
  const auto expected = sizeof(double) * (n*n + t*n*n + 4*t*n + 2*t + 3*grid.system().natom)
                      + t*sizeof(GridOwner) + sizeof(int);
  if (report.derivative_device_bytes != expected) {
    std::cerr << report.derivative_device_bytes << " != " << expected;
    return 1;
  }
  if (report.device_bytes != report.grid_device_bytes + expected) return 2;
  if (report.tile_points != t) return 3;
}
"""
    harness = (
        harness.replace(
            "RESULT_STRUCT",
            _definition(header, "struct CudaCosxMolecularDerivativeDiagnostic"),
        )
        .replace("DEFINITIONS", definitions)
        .replace("TILE", str(requested_tile))
    )
    path = tmp_path / "diagnostic.cpp"
    path.write_text(harness)
    binary = tmp_path / "diagnostic"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            str(path),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=10
    )

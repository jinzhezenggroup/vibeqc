"""Run the real completion body against malformed finite metric contractions.

The rejection fixture is deliberately not a physical overlap matrix: it checks
that NaN cross-column contractions cannot be reported as zero orthogonality
error. Positive controls use SPD metrics and independent occupied subspaces.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
using Matrix = std::vector<double>;
namespace integrals { struct IntegralData { std::size_t nbf; Matrix overlap; }; }
struct OccupiedCompletionResult {
  Matrix density, coefficients;
  std::size_t added_orbitals = 0;
  double minimum_added_norm = 1, metric_orthogonality_error = 0;
};
std::size_t index(std::size_t i, std::size_t j, std::size_t n) { return i*n+j; }
"""
DRIVER = r"""
int main(int argc, char** argv) {
  if (argc != 4) return 99;
  const std::string mode = argv[1];
  const int rotation = std::atoi(argv[2]), sign = std::atoi(argv[3]);
  integrals::IntegralData target{3, {}};
  Matrix seed, reference{1,0,0,0,1,0,0,0,1};
  std::size_t seeded = 2, occupied = 2;
  bool reject = mode == "nan-cross" || mode == "nonempty-zero-seed";
  if (mode == "nan-cross" || mode == "finite-cross") {
    const double c = 1e154, small = 0.5e-308;
    const double cross = mode == "nan-cross" ? sign*1e155 : 0.0;
    const Matrix metric{small,0,cross,0,small,-cross,0,0,1};
    const Matrix columns{c,0,c,0,0,1};
    target.overlap.resize(9);
    seed.resize(6);
    for (std::size_t i = 0; i < 3; ++i) {
      const auto row = (i + rotation) % 3;
      for (std::size_t j = 0; j < 3; ++j)
        target.overlap[row*3 + (j+rotation)%3] = metric[i*3+j];
      for (std::size_t j = 0; j < 2; ++j) seed[row*2+j] = columns[i*2+j];
    }
  } else {
    target.overlap = {4,0,0,0,1,0,0,0,0.25};
    reference = {0.5,0,0,0,1,0,0,0,2};
    seeded = 0;
    if (mode == "nonempty-zero-seed") seed = {1};
    if (mode == "seeded-completion") { seeded = 1; seed = {0.5,0,0}; }
    if (mode == "empty-completion") occupied = 0;
  }
  const auto original_metric = target.overlap, original_seed = seed;
  try {
    const auto result = complete_occupied_density(
        target, seed, seeded, reference, occupied, 1.0, 1e-6);
    if (reject) {
      std::cerr << "invalid inputs published with error="
                << result.metric_orthogonality_error << '\n';
      return 1;
    }
    if (result.metric_orthogonality_error > 1e-13 ||
        result.added_orbitals != occupied-seeded ||
        !std::all_of(result.density.begin(), result.density.end(),
                     [](double x) { return std::isfinite(x); })) return 2;
    if (mode != "finite-cross") {
      const Matrix expected = occupied ? Matrix{0.25,0,0,0,1,0,0,0,0} : Matrix(9,0);
      for (std::size_t i = 0; i < 9; ++i)
        if (std::abs(result.density[i]-expected[i]) > 1e-13) return 3;
    }
  } catch (const std::invalid_argument& error) {
    if (!reject) { std::cerr << error.what(); return 4; }
    const std::string expected = mode == "nan-cross"
        ? "occupied completion metric contraction is non-finite"
        : "occupied completion has inconsistent or non-finite inputs";
    if (error.what() != expected) { std::cerr << error.what(); return 5; }
  }
  if (seed != original_seed || target.overlap != original_metric) return 6;
}
"""


@pytest.fixture(scope="module")
def completion_metric_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    source = (ROOT / "src/scf/initial_guess/density.cpp").read_text()
    start = source.index("OccupiedCompletionResult complete_occupied_density(")
    end = source.index("std::pair<Matrix, Matrix> prepare_initial_uhf_density(", start)
    directory = tmp_path_factory.mktemp("completion-metric")
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(PREFIX + source[start:end] + DRIVER)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-fsanitize=undefined",
            str(unit),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize(
    "mode,rotation,sign",
    [
        *(("nan-cross", rotation, sign) for rotation in range(3) for sign in (-1, 1)),
        *(("finite-cross", rotation, 1) for rotation in range(3)),
        ("nonempty-zero-seed", 0, 1),
        ("seeded-completion", 0, 1),
        ("unseeded-completion", 0, 1),
        ("empty-completion", 0, 1),
    ],
)
def test_completion_metric_validation(
    completion_metric_probe: Path, mode: str, rotation: int, sign: int
) -> None:
    result = subprocess.run(
        [str(completion_metric_probe), mode, str(rotation), str(sign)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

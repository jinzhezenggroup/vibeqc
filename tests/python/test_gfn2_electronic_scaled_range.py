"""Emitted pair arithmetic preserves finite results before contraction."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = tmp_path_factory.mktemp("electronic-scaled-range")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_gfn2_electronic_cuda.py"),
            "--output",
            str(root / "generated.cuh"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    source = root / "probe.cpp"
    source.write_text(r"""
#define __device__
#include "generated.cuh"
#include <cmath>
#include <cstdlib>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  using namespace vibeqc::xtb::generated;
  Gfn2ElectronicPairIntegrals x{};
  Gfn2ElectronicPairPotentials p{};
  double expected{};
  switch (std::atoi(argv[1])) {
    case 0:
      x.overlap=0.01; p.row_scalar=1e308; p.column_scalar=1e308;
      expected=std::fma(-0.5*x.overlap,p.column_scalar,
                       std::fma(-0.5*x.overlap,p.row_scalar,0.0)); break;
    case 1:
      x.overlap=1e308; p.row_scalar=3.0;
      expected=(-0.5*x.overlap)*p.row_scalar; break;
    case 2:
      x.dipole_forward[0]=1e308; p.dipole_column[0]=3.0;
      expected=(-0.5*x.dipole_forward[0])*p.dipole_column[0]; break;
    case 3:
      x.quadrupole_reverse[0]=1e308; p.quadrupole_row[0]=3.0;
      expected=(-0.5*x.quadrupole_reverse[0])*p.quadrupole_row[0]; break;
    default: return 2;
  }
  double actual{};
  if (!evaluate_gfn2_electronic_pair(x,p,actual)) return 3;
  return std::isfinite(actual) && std::abs(actual/expected-1.0) < 1e-14 ? 0 : 4;
}
""")
    executable = root / "probe"
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return executable


@pytest.mark.parametrize("case", range(4))
def test_generated_pair_preserves_scaled_finite_result(probe: Path, case: int) -> None:
    result = subprocess.run(
        [str(probe), str(case)], capture_output=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stderr.decode()

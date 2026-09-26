"""Execute the actual completion body at finite-input publication boundaries."""

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
#include <vector>
using Matrix = std::vector<double>;
namespace integrals { struct IntegralData { std::size_t nbf; Matrix overlap; }; }
struct OccupiedCompletionResult {
 Matrix density, coefficients;
 std::size_t added_orbitals = 0;
 double minimum_added_norm = 1, metric_orthogonality_error = 0;
};
std::size_t index(std::size_t i,std::size_t j,std::size_t n){ return i*n+j; }
"""
DRIVER = r"""
int main(int argc,char**argv){
 if(argc!=6)return 99;
 const double s=std::strtod(argv[1],nullptr),c=std::strtod(argv[2],nullptr),w=std::strtod(argv[3],nullptr);
 const bool reject=std::atoi(argv[4]);
 const std::size_t seeded=std::atoi(argv[5]);
 try {
  const integrals::IntegralData target{1,{s}};
  const Matrix seed=seeded ? Matrix{c} : Matrix{}, reference{c};
  const auto out=complete_occupied_density(target,seed,seeded,reference,1,w,1e-6);
  if(reject){std::cerr<<"published nonfinite density "<<out.density[0]<<'\n';return 1;}
  if(out.density.size()!=1||!std::isfinite(out.density[0]))return 2;
  const long double expected=(long double)w*c*c;
  if(std::abs((long double)out.density[0]-expected)>1e-13L*std::max(1.0L,std::abs(expected)))return 3;
 } catch(const std::invalid_argument& e){
  if(!reject){std::cerr<<e.what()<<'\n';return 4;}
  if(std::string(e.what()).find("non-finite target density")==std::string::npos){std::cerr<<e.what()<<'\n';return 5;}
 }
}
"""


@pytest.fixture(scope="module")
def completion_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    source = (ROOT / "src/scf/initial_guess/density.cpp").read_text()
    start = source.index("OccupiedCompletionResult complete_occupied_density(")
    end = source.index("std::pair<Matrix, Matrix> prepare_initial_uhf_density(", start)
    directory = tmp_path_factory.mktemp("completion-publication")
    unit, exe = directory / "probe.cpp", directory / "probe"
    unit.write_text(PREFIX + source[start:end] + DRIVER)
    subprocess.run(
        [compiler, "-std=c++20", "-O2", str(unit), "-o", str(exe)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return exe


@pytest.mark.parametrize("seeded", (0, 1))
@pytest.mark.parametrize(
    "s,c,w,reject",
    [
        (1e-308, 1e154, 2, 1),
        (1e-310, 1e155, 1, 1),
        (1e-308, -1e154, 2, 1),
        (1, 1, 2, 0),
        (4, 0.5, 2, 0),
        (0.25, 2, 2, 0),
        (1e-308, 1e154, 1, 0),
    ],
)
def test_completion_rejects_nonfinite_publication(
    completion_probe: Path,
    s: float,
    c: float,
    w: float,
    reject: int,
    seeded: int,
) -> None:
    result = subprocess.run(
        [str(completion_probe), str(s), str(c), str(w), str(reject), str(seeded)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

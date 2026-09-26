"""Execute real projection admission; the one-column Gram oracle is analytic.

Finite SPD metrics and finite coefficients may overflow internally. A NaN must
not be reduced to zero error by std::max, even if target projection is finite.
This is a host arithmetic/validation test, not native eigensolver qualification.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
using Matrix=std::vector<double>;
namespace integrals { struct IntegralData { std::size_t nbf; Matrix overlap; }; }
struct OccupiedProjectionResult {
 Matrix density,coefficients;
 double source_metric_orthogonality_error=0,minimum_projected_norm=0;
 double projection_residual=0,target_metric_orthogonality_error=0;
};
std::size_t index(std::size_t i,std::size_t j,std::size_t n){return i*n+j;}
namespace reference {
 struct EigenResult { Matrix values,vectors; };
 EigenResult symmetric_eigen(Matrix gram,std::size_t n){
  if(n!=1||gram.size()!=1)throw std::logic_error("test oracle is only for a 1x1 Gram");
  return {Matrix{gram[0]},Matrix{1.0}};
 }
}
"""
DRIVER = r"""
int main(int argc,char**argv){
 if(argc!=4)return 99;
 const double magnitude=std::strtod(argv[1],nullptr),sign=std::strtod(argv[2],nullptr);
 const bool invalid=std::atoi(argv[3]);
 const integrals::IntegralData target{1,{1}};
 const Matrix x{1}, s=invalid ? Matrix{1e308,-sign*5e307,-sign*5e307,1e308}:Matrix{1,0,0,1};
 const Matrix c=invalid ? Matrix{magnitude,0,sign*magnitude,1}:Matrix{sign,0,0,1};
 const Matrix cross=invalid ? Matrix{0.5/magnitude,sign*0.5/magnitude}:Matrix{sign,0};
 try {
  auto out=project_occupied_density(target,x,s,cross,2,c,1,2.0,0.5);
  if(invalid){std::cerr<<"accepted nonfinite source metric as orthogonal: "<<out.source_metric_orthogonality_error<<" density="<<out.density[0]<<'\n';return 1;}
  if(std::abs(out.density[0]-2)>1e-12)return 2;
 }catch(const std::invalid_argument& e){
  if(!invalid){std::cerr<<e.what();return 3;}
  if(std::string(e.what())!="source occupied metric contraction is non-finite"){std::cerr<<e.what();return 4;}
 }
}
"""


@pytest.fixture(scope="module")
def projection_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    source = (ROOT / "src/scf/initial_guess/density.cpp").read_text()
    start = source.index("OccupiedProjectionResult project_occupied_density(")
    end = source.index("OccupiedCompletionResult complete_occupied_density(", start)
    directory = tmp_path_factory.mktemp("projection-metric")
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


@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("magnitude,invalid", ((4, 1), (8, 1), (100, 1), (1, 0)))
def test_nonfinite_source_metric_cannot_pass_orthogonality(
    projection_probe: Path, magnitude: int, sign: int, invalid: int
) -> None:
    result = subprocess.run(
        [str(projection_probe), str(magnitude), str(sign), str(invalid)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

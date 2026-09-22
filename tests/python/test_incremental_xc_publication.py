"""Exercise the actual incremental accumulator at its point-evaluator boundary."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("overflow", ["potential", "difference", "electrons", "finite"])
def test_incremental_xc_rejects_nonfinite_accumulated_outputs(
    tmp_path: Path, overflow: str
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    source = (ROOT / "src/dft/xc.cpp").read_text()
    begin = source.index(
        "ExactIncrementalXcIntegral integrate_pbe_rks_incremental_exact("
    )
    end = source.index("\nXcIntegral integrate_pbe_rks_with_tail_scaled(", begin)
    function = source[begin:end]
    # All per-point outputs are finite. The stub is only the prequalified
    # point-provider boundary; the actual production accumulation is compiled.
    flags = {"potential": 0, "difference": 1, "electrons": 2, "finite": 3}
    harness = tmp_path / "publication.cpp"
    harness.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
struct AoBasis {
  std::size_t nao=1;
  void evaluate(const double*, std::size_t count, int, int, std::size_t,
                double* out, std::size_t size) const {
    std::fill(out,out+size,0.0);
    std::fill(out,out+count,1.0);
  }
};
struct MolecularGrid {
  std::vector<double> p{0,0,0,1,0,0}, w{1,1};
  std::size_t point_count() const {return w.size();}
  const auto& points() const {return p;}
  const auto& weights() const {return w;}
};
struct Diagnostic {std::size_t npoint{},ingredient_mask{},active_ao{},borrowed_density_bytes{};};
struct XcIntegral {std::vector<double> potential;std::size_t points{};Diagnostic density_diagnostic;double energy{},electrons{};};
struct ExactIncrementalXcIntegral {XcIntegral total;double anchor_energy{},energy_difference{};std::vector<double> potential_difference;};
namespace runtime {
std::size_t add_capacity(std::size_t a,std::size_t b){return a+b;}
std::size_t vector_bytes(const std::vector<double>& v){return v.capacity()*sizeof(double);}
}
void validate_density_matrix(const AoBasis&,const MolecularGrid&,const std::vector<double>&,std::size_t){}
void sample_xc_capacity(XcIntegral&,const std::vector<double>&,std::size_t){}
std::array<double,5> rks_features(const double*,const std::array<const double*,3>&,
    std::size_t,const std::vector<double>& d,const void*,unsigned){return {d[0],0,0,0,0};}
struct Point {bool valid=true;double energy=1;double rho[2]{};double gradient[2][3]{};};
"""
        + f"constexpr int scenario={flags[overflow]};\n"
        + r"""
Point evaluate_generated_pbe_point(const double* rho,const double (*)[3],double,double){
  Point out;
  if(scenario==0) out.rho[0]=1e308;
  if(scenario==1) out.rho[0]=rho[0]<0.75 ? 1e308 : -1e308;
  return out;
}
"""
        + function
        + r"""
int main(){
  AoBasis basis;MolecularGrid grid;
  if(scenario==1){grid.w.resize(1);grid.p.resize(3);}
  const std::vector<double> anchor{scenario==2 ? 1e308 : 1.0};
  const std::vector<double> delta{scenario==1 ? 1.0 : 0.0};
  try {
    (void)integrate_pbe_rks_incremental_exact(basis,grid,anchor,delta,1,1.0,1.0);
  } catch(const std::runtime_error&){return scenario==3 ? 1 : 0;}
  return scenario==3 ? 0 : 1;
}
"""
    )
    executable = tmp_path / "publication"
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(harness), "-o", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    checked = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=5
    )
    assert checked.returncode == 0, (
        f"incremental XC publication gate failed for {overflow}"
    )

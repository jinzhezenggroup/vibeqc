"""Compiled generated electronic helpers preserve the native FMA contract."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_electronic_fma_cancellation_and_publication(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = Path(__file__).resolve().parents[2]
    header = tmp_path / "generated_gfn2_electronic_native.hpp"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(root / "tools/generate_gfn2_electronic_native.py"),
            "--output",
            str(header),
        ],
        check=True,
        capture_output=True,
        timeout=45,
    )
    source = tmp_path / "fused.cpp"
    source.write_text(r"""
#include <cmath>
#include <limits>
#include <initializer_list>
#include "generated_gfn2_electronic_native.hpp"
int main() {
  using namespace vibeqc::xtb::generated;
  double out=123.;
  if(!gfn2_population_update_tensor(1e308,2.,1e308,out) || out!=std::fma(-1e308,2.,1e308)) return 1;
  if(!gfn2_core_energy_update_tensor(1e308,2.,-1e308,out) || out!=std::fma(1e308,2.,-1e308)) return 2;
  double old=0.;
  for(double potential : {1e308,1e308,-1e308,-1e308}) old=std::fma(-0.25,potential,old);
  if(!gfn2_scalar_hamiltonian_update_tensor(0.5,1e308,1e308,-1e308,-1e308,0.,out) || out!=old) return 3;
  for(double x : {0.,0.1,-0.3,1e-200,1e200}) {
    const double expected=std::fma(-x,0.2,0.17);
    if(!gfn2_population_update_tensor(x,0.2,0.17,out) || out!=expected) return 4;
    const double expected_h=std::fma(-0.5*x,0.3,std::fma(-0.5*x,-0.2,0.13));
    if(!gfn2_multipole_hamiltonian_update_tensor(x,x,0.3,-0.2,0.13,out) || out!=expected_h) return 5;
  }
  out=123.;
  if(gfn2_core_energy_update_tensor(1e308,2.,1e308,out) || out!=123.) return 6;
  if(gfn2_population_update_tensor(std::numeric_limits<double>::quiet_NaN(),1.,1.,out) || out!=123.) return 7;
}
""")
    binary = tmp_path / "fused"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=45,
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)

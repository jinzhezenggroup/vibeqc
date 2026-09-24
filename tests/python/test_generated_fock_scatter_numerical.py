"""Generated Direct-Fock scatter matches an independent dense ERI contraction."""

import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_fock_accumulation_header,
)

ROOT = Path(__file__).resolve().parents[2]


def test_generated_scatter_all_canonical_index_coincidences(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    (tmp_path / "cuda_runtime.h").write_text(
        "#pragma once\n#define __device__\n#define __host__\n#define __forceinline__ inline\ninline void atomicAdd(double* p,double x){*p+=x;}\n"
    )
    (tmp_path / "scatter.hpp").write_text(emit_direct_fock_accumulation_header())
    driver = tmp_path / "scatter.cpp"
    driver.write_text(
        r"""
#include "scatter.hpp"
#include <set>
#include <vector>
#include <iostream>
#include <array>
#include <cmath>
int main() {
  using namespace vibeqc::scf::cuda_execution;
  constexpr size_t n=4, m=n*n, off=5, spin=40;
  auto index=[](size_t p,size_t q,size_t r,size_t s){return ((p*n+q)*n+r)*n+s;};
"""
        + r"""
  size_t cases=0;
  for(size_t i=0;i<n;++i) for(size_t j=0;j<=i;++j)
  for(size_t k=0;k<n;++k) for(size_t l=0;l<=k;++l) {
    if(i*(i+1)/2+j < k*(k+1)/2+l) continue;
    const double value=.125+.003*(i+j+k+l);
    std::set<std::array<size_t,4>> orbit{{i,j,k,l},{j,i,k,l},{i,j,l,k},{j,i,l,k},
                                     {k,l,i,j},{l,k,i,j},{k,l,j,i},{l,k,j,i}};
    std::vector<double> eri(n*n*n*n), density(100);
    for(auto a:orbit) eri[index(a[0],a[1],a[2],a[3])]=value;
    for(size_t p=0;p<n;++p) for(size_t q=0;q<n;++q) {
      density[off+p+q*n]=((p+q)%3==0 ? 0.0 : .03*(p+2*q+1));
      density[spin+p+q*n]=.04*(p+2*q+1);
      density[spin+m+p+q*n]=-.01*(2*p+q+2);
    }
    for(bool unrestricted : {false,true}) for(bool coulomb_only : {false,true}) {
      std::vector<double> actual(100), expected(100);
      if(unrestricted) accumulate_direct_fock_integral<true>(n,off,spin,density.data(),actual.data(),i,j,k,l,value,coulomb_only);
      else accumulate_direct_fock_integral<false>(n,off,spin,density.data(),actual.data(),i,j,k,l,value,coulomb_only);
      for(size_t p=0;p<n;++p) for(size_t q=0;q<n;++q)
      for(size_t r=0;r<n;++r) for(size_t s=0;s<n;++s) {
        double J=eri[index(p,q,r,s)], K=eri[index(p,r,q,s)];
        if(coulomb_only) K=0;
        if(unrestricted) {
          double a=density[spin+r+s*n], b=density[spin+m+r+s*n];
          expected[spin+p+q*n]+=(a+b)*J-a*K;
          expected[spin+m+p+q*n]+=(a+b)*J-b*K;
        } else expected[off+p+q*n]+=density[off+r+s*n]*(J-.5*K);
      }
      for(size_t a=0;a<actual.size();++a) if(std::abs(actual[a]-expected[a])>2e-13) return 1;
      ++cases;
    }
  }
  if(cases!=220) return 2;
  std::cout<<cases<<" independent dense RHF/UHF scatter comparisons passed\n";
}
"""
    )
    executable = tmp_path / "scatter"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""Execute emitted geometry/partition code against the original quotient on host.

This checks binary64 arithmetic and indexing, not CUDA scheduling/device libm.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.xc.quadrature_cuda import emit_quadrature_cuda

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def partition_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    directory = tmp_path_factory.mktemp("inverse-partition")
    (directory / "cuda_runtime.h").write_text(
        """#pragma once
#include <cmath>
#define __device__
#define __global__
using std::isfinite;
struct dim3 { unsigned x,y,z; constexpr dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
inline dim3 blockIdx{0,0,0},threadIdx{0,0,0},blockDim{1,1,1},gridDim{1,1,1};
"""
    )
    source = emit_quadrature_cuda().split("__global__ void normalize_kernel", 1)[0]
    source += "\n} // namespace\n#endif\n"
    source += r"""
#include <array>
#include <cstdlib>
#include <iostream>
using namespace vibeqc::generated::quadrature;
template<unsigned It> int run(double sep,double tolerance) {
  constexpr size_t count=4;
  const double centers[6]{0,0,0,sep,0,0};
  std::array<double,6> geometry{-123,0,0,0,0,456};
  geometry_kernel(centers,2,tolerance,geometry.data()+1);
  const double distances[8]{sep*.25,sep*.5,sep*.75,1,sep*.75,sep*.5,sep*.25,1};
  std::array<double,10> logs{}; logs.front()=-123;logs.back()=456;
  partition_kernel<It>(count,2,distances,geometry.data()+1,logs.data()+1);
  for(size_t p=0;p<count;++p) {
    const double mu=sep>tolerance?fmin(1.,fmax(-1.,(distances[count+p]-distances[p])/sep)):0.;
    const double pair=fmin(1.,fmax(0.,becke<It>(mu)));
    for(size_t a=0;a<2;++a) {
      const double expected=a?pair:1-pair,actual=exp(logs[1+a*count+p]);
      if(!std::isfinite(actual) || std::abs(actual-expected)>3e-13) {
        std::cerr<<"iteration="<<It<<" point="<<p<<" actual="<<actual<<" expected="<<expected<<'\n';
        return 1;
      }
    }
  }
  if(geometry.front()!=-123 || geometry.back()!=456 || logs.front()!=-123 || logs.back()!=456) return 2;
  for(size_t i=1;i<5;++i) if(!std::isfinite(geometry[i])) return 3;
  return 0;
}
int main(int argc,char**argv) {
  if(argc!=3)return 99;
  const double sep=std::strtod(argv[1],nullptr),tol=std::strtod(argv[2],nullptr);
  return run<1>(sep,tol)||run<2>(sep,tol)||run<3>(sep,tol)||run<4>(sep,tol)||run<5>(sep,tol);
}
"""
    cpp = directory / "partition.cpp"
    cpp.write_text(source)
    binary = directory / "partition"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-D__CUDACC__",
            f"-I{directory}",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize(
    "separation,tolerance",
    [
        (1e-310, 0.0),
        (1e-310, 1e-320),
        (1e-308, 0.0),
        (0.0, 0.0),
        (5e-13, 1e-12),
        (2.0, 1e-12),
        (1e-100, 0.0),
    ],
)
def test_emitted_partition_retains_finite_quotient_boundary(
    partition_binary: Path, separation: float, tolerance: float
) -> None:
    subprocess.run(
        [str(partition_binary), str(separation), str(tolerance)], check=True, timeout=10
    )

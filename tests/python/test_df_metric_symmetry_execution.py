"""Execute the actual metric launch and kernels with a serial host index shim.

The shared decoder has one barrier and no later cross-thread dependence. Running
its first lane before the rest models that dataflow, not CUDA synchronization.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_SHIM = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#define __global__
#define __shared__ static
#define __syncthreads() ((void)0)
using cudaStream_t = int;
struct dim3 { unsigned x,y,z; dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
dim3 blockIdx,threadIdx,blockDim,gridDim;
std::vector<unsigned> hits;
template<class F,class... Args>
void simulate(F kernel,dim3 grid,dim3 block,std::size_t,cudaStream_t,Args... args) {
  gridDim=grid;blockDim=block;
  for(unsigned z=0;z<grid.z;++z) for(unsigned y=0;y<grid.y;++y) for(unsigned x=0;x<grid.x;++x) {
    blockIdx=dim3(x,y,z);
    for(unsigned tz=0;tz<block.z;++tz) for(unsigned ty=0;ty<block.y;++ty) for(unsigned tx=0;tx<block.x;++tx) {
      threadIdx=dim3(tx,ty,tz);kernel(args...);
    }
  }
}
"""

_DRIVER = r"""
int main(int argc,char** argv) {
  if(argc!=6) return 99;
  const unsigned n=std::atoi(argv[1]), bx=std::atoi(argv[2]), by=std::atoi(argv[3]);
  const unsigned systems=std::atoi(argv[4]), clipped=std::atoi(argv[5]);
  dim3 grid((n+bx-1)/bx,(n+by-1)/by,systems),block(bx,by);
  if(clipped && grid.y>1) --grid.y;
  const std::size_t elements=std::size_t(systems)*n*n;
  std::vector<double> values(elements+8,-999.0);
  for(std::size_t i=0;i<elements;++i) values[i]=double((i*17+3)%197)-81.0;
  auto expected=values;
  std::vector<unsigned> expected_hits(elements+8,0);
  for(unsigned s=0;s<systems;++s) for(unsigned row=0;row<std::min(n,grid.y*by);++row)
    for(unsigned col=row;col<std::min(n,grid.x*bx);++col) {
      const auto a=(std::size_t(s)*n+row)*n+col,b=(std::size_t(s)*n+col)*n+row;
      const double average=0.5*(expected[a]+expected[b]);
      expected[a]=expected[b]=average;++expected_hits[a];
    }
  hits.assign(elements+8,0);
  launch_symmetrize_metrics_kernel(grid,block,0,0,n,values.data());
  for(std::size_t i=0;i<values.size();++i) {
    if(values[i]!=expected[i] || hits[i]!=expected_hits[i]) {
      std::cerr << "mismatch at " << i << " value " << values[i] << " expected " << expected[i]
                << " visits " << hits[i] << " expected " << expected_hits[i] << '\n';
      return 1;
    }
  }
}
"""


@pytest.fixture(scope="module")
def metric_execution(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    text = (ROOT / "src/scf/cuda/df_metric_kernels.cu").read_text(encoding="utf-8")
    start = text.index("__global__ void symmetrize_metrics_kernel")
    end = text.index("__global__ void scale_eigenvectors_kernel", start)
    kernels = text[start:end]
    marker = "__global__ void symmetrize_metrics_fallback_kernel"
    if marker in text:
        start = text.index(marker)
        kernels += text[
            start : text.index("void launch_symmetrize_metrics_kernel", start)
        ]
    start = text.index("void launch_symmetrize_metrics_kernel")
    wrapper = text[start : text.index("void launch_scale_eigenvectors_kernel", start)]
    wrapper, count = re.subn(
        r"(\w+)<<<([\s\S]*?)>>>\(([\s\S]*?)\);",
        r"simulate(\1, \2, \3);",
        wrapper,
    )
    assert count in (1, 2)
    kernels = kernels.replace(
        "metrics[first] = symmetric;", "++hits[first]; metrics[first] = symmetric;"
    )
    directory = tmp_path_factory.mktemp("metric-symmetry-execution")
    unit, executable = directory / "metric.cpp", directory / "metric"
    unit.write_text(_SHIM + kernels + wrapper + _DRIVER, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
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
    "dimension,bx,by,systems,clipped",
    [
        (1, 16, 16, 1, 0),
        (17, 16, 16, 2, 0),
        (33, 16, 16, 3, 0),
        (33, 16, 8, 2, 0),
        (65, 32, 8, 2, 0),
        (49, 8, 16, 1, 0),
        (49, 16, 16, 2, 1),
        (928, 16, 16, 1, 0),
        (3712, 16, 16, 1, 0),
    ],
)
def test_symmetry_matches_original_domain_and_visits_once(
    metric_execution: Path,
    dimension: int,
    bx: int,
    by: int,
    systems: int,
    clipped: int,
) -> None:
    result = subprocess.run(
        [
            str(metric_execution),
            str(dimension),
            str(bx),
            str(by),
            str(systems),
            str(clipped),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr

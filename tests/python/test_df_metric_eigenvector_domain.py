"""Execute the actual DF scaler launch adapter with a serial host CUDA shim."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src/scf/cuda/df_metric_kernels.cu"
SHIM = r"""
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
dim3 blockIdx, threadIdx, blockDim, gridDim;
bool capture_only=false;
unsigned column_launches=0;
template<class F, class... A>
void simulate(F fn,dim3 grid,dim3 block,std::size_t,cudaStream_t,A... args) {
 gridDim=grid;blockDim=block;
 if(capture_only)return;
 for(unsigned z=0;z<grid.z;++z)for(unsigned y=0;y<grid.y;++y)for(unsigned x=0;x<grid.x;++x){
  blockIdx=dim3(x,y,z);
  for(unsigned tz=0;tz<block.z;++tz)for(unsigned ty=0;ty<block.y;++ty)for(unsigned tx=0;tx<block.x;++tx){
   threadIdx=dim3(tx,ty,tz);fn(args...);
  }
 }
}
"""
DRIVER = r"""
int main(int argc,char**argv){
 if(argc!=7)return 99;
 const std::size_t n=std::strtoull(argv[1],nullptr,10), systems=std::strtoull(argv[2],nullptr,10);
 const unsigned threads=std::atoi(argv[3]),blocks=std::atoi(argv[4]),gy=std::atoi(argv[5]),gz=std::atoi(argv[6]);
 const std::size_t total=n*n*systems;
 if(n>100000){
  capture_only=true;
  launch_scale_eigenvectors_kernel(dim3(blocks,gy,gz),dim3(threads),0,0,total,n,nullptr,nullptr,nullptr);
  if(column_launches){std::cerr<<"invalid CUDA column grid admitted\n";return 1;}return 0;
 }
 std::vector<double> input(total),scales(n*systems),output(total+8,-999),expected=output;
 for(std::size_t i=0;i<total;++i)input[i]=double(int(i%19)-7)/8;
 for(std::size_t i=0;i<scales.size();++i)scales[i]=double(i%13+1)/4;
 if(gy&&gz)for(std::size_t i=0;i<std::min(total,std::size_t(blocks)*threads);++i)expected[i]=input[i]*scales[i/n];
 launch_scale_eigenvectors_kernel(dim3(blocks,gy,gz),dim3(threads),0,0,total,n,input.data(),scales.data(),output.data());
 const bool full=gy==1&&gz==1&&std::size_t(blocks)*threads>=total;
 if(bool(column_launches)!=full){std::cerr<<"wrong schedule\n";return 2;}
 for(std::size_t i=0;i<output.size();++i)if(output[i]!=expected[i]){std::cerr<<"write-domain mismatch "<<i<<'\n';return 3;}
}
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    text = SOURCE.read_text()
    kernels = text[
        text.index("__global__ void scale_eigenvectors_flat_kernel") : text.index(
            "/** Scale a bounded pair stripe"
        )
    ]
    start = text.index("void launch_scale_eigenvectors_kernel")
    wrapper = text[start : text.index("void launch_scale_metric_projection_to", start)]
    wrapper = wrapper.replace(
        "scale_eigenvectors_column_kernel<<<",
        "++column_launches; scale_eigenvectors_column_kernel<<<",
    )
    wrapper, count = re.subn(
        r"(\w+)<<<([\s\S]*?)>>>\(([\s\S]*?)\);",
        r"simulate(\1, \2, \3);",
        wrapper,
    )
    assert count == 2
    directory = tmp_path_factory.mktemp("eigenvector-domain")
    unit = directory / "probe.cpp"
    exe = directory / "probe"
    unit.write_text(SHIM + kernels + wrapper + DRIVER)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-fsanitize=undefined",
            str(unit),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return exe


@pytest.mark.parametrize(
    "args",
    [
        (1, 1, 32, 1, 1, 1),
        (4, 1, 4, 4, 1, 1),
        (4, 1, 4, 1, 1, 1),
        (5, 2, 8, 7, 1, 1),
        (17, 2, 8, 71, 1, 1),
        (17, 2, 8, 73, 1, 1),
        (4, 1, 4, 4, 2, 1),
        (4, 1, 4, 4, 1, 2),
        (4, 1, 4, 0, 1, 1),
        (928, 1, 256, 3364, 1, 1),
        (2147483648, 1, 256, 1, 1, 1),
    ],
)
def test_eigenvector_launch_domain(probe: Path, args: tuple[int, ...]) -> None:
    completed = subprocess.run(
        [str(probe), *map(str, args)], capture_output=True, text=True, timeout=15
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

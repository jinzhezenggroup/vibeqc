"""Execute the CUDA density kernel and launcher with a serial index shim.

This checks write-domain ownership and scalar arithmetic, not GPU execution or
synchronization. Clipped launches must preserve the original flat-prefix work.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#define __global__
using cudaStream_t = int;
struct dim3 {
  unsigned x,y,z;
  dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){}
};
dim3 blockIdx,threadIdx,blockDim,gridDim;
std::vector<unsigned> hits;
template<class F,class... A>
void simulate(F fn,dim3 grid,dim3 block,std::size_t,cudaStream_t,A... args) {
  gridDim=grid;blockDim=block;
  for(unsigned x=0;x<grid.x;++x) {
    blockIdx=dim3(x);
    for(unsigned t=0;t<block.x;++t) {threadIdx=dim3(t);fn(args...);}
  }
}
"""
DRIVER = r"""
int main(int argc,char** argv) {
  if(argc!=6)return 99;
  const std::size_t n=std::strtoul(argv[1],nullptr,10),batch=std::strtoul(argv[2],nullptr,10);
  const unsigned threads=std::strtoul(argv[3],nullptr,10),blocks=std::strtoul(argv[4],nullptr,10);
  const double weight=std::strtod(argv[5],nullptr);
  const std::size_t total=batch*n*n,visited=std::min(total,std::size_t(threads)*blocks);
  std::vector<double> c(total),out(total+8,-991.0),expected=out;
  std::vector<std::int32_t> occupied(batch);
  for(std::size_t s=0;s<batch;++s)occupied[s]=s%3==1 ? 0 : (s%3==2 ? n : (n+1)/2);
  for(std::size_t i=0;i<total;++i)c[i]=double(int(i%31)-13)/8.0;
  const auto saved=c;
  for(std::size_t i=0;i<visited;++i) {
    const auto s=i/(n*n),local=i%(n*n),row=local%n,col=local/n,offset=s*n*n;
    double value=0;
    for(std::int32_t k=0;k<occupied[s];++k)
      value+=weight*c[offset+row+k*n]*c[offset+col+k*n];
    expected[i]=value;
  }
  hits.assign(out.size(),0);
  launch_build_device_density_kernel(dim3(blocks),dim3(threads),0,0,batch,n,
                                     occupied.data(),c.data(),weight,out.data());
  for(std::size_t i=0;i<out.size();++i) {
    if(out[i]!=expected[i] || hits[i]!=(i<visited?1U:0U)) {
      std::cerr<<"write-domain mismatch at "<<i<<": "<<out[i]<<" expected "<<expected[i]
               <<" writes "<<hits[i]<<'\n';return 1;
    }
  }
  if(c!=saved)return 2;
}
"""


@pytest.fixture(scope="module")
def density_domain_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    source = (ROOT / "src/scf/cuda/df_scf_kernels.cu").read_text()
    begin = source.index("__global__ void build_device_density_kernel")
    end = source.index("__global__ void compute_device_energy_kernel", begin)
    kernel = source[begin:end].replace(
        "density[element] = value;", "++hits[element]; density[element] = value;"
    )
    kernel = kernel.replace(
        "density[offset + column + row * nbf] = value;",
        "{ ++hits[offset + column + row * nbf]; "
        "density[offset + column + row * nbf] = value; }",
    )
    begin = source.index("void launch_build_device_density_kernel")
    end = source.index("void launch_compute_device_energy_kernel", begin)
    wrapper, count = re.subn(
        r"(\w+)<<<([\s\S]*?)>>>\(([\s\S]*?)\);",
        r"simulate(\1, \2, \3);",
        source[begin:end],
    )
    assert count == 1
    directory = tmp_path_factory.mktemp("df-density-domain")
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(PREFIX + kernel + wrapper + DRIVER)
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


@pytest.mark.parametrize("weight", (1.0, 2.0, 0.5, 3.0))
@pytest.mark.parametrize(
    "n,batch,threads,blocks",
    [
        (1, 1, 32, 1),
        (4, 1, 4, 1),
        (4, 1, 4, 3),
        (4, 1, 4, 4),
        (5, 3, 8, 4),
        (5, 3, 8, 10),
        (17, 3, 32, 19),
        (17, 3, 32, 28),
        (4, 1, 4, 0),
        (4, 0, 4, 1),
    ],
)
def test_density_preserves_requested_launch_domain(
    density_domain_probe: Path,
    n: int,
    batch: int,
    threads: int,
    blocks: int,
    weight: float,
) -> None:
    result = subprocess.run(
        [
            str(density_domain_probe),
            str(n),
            str(batch),
            str(threads),
            str(blocks),
            str(weight),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

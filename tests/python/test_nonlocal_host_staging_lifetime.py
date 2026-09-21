"""Fault-inject the actual CUDA wrapper's host staging lifetime, without a GPU."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def staging_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    body = source[source.index("void execute_vv10_cuda(") :].rsplit(
        "}  // namespace vibeqc::dft::nlc", 1
    )[0]
    body = re.sub(r"<<<.*?>>>", "", body, flags=re.DOTALL)
    folder = tmp_path_factory.mktemp("nonlocal-host-staging")
    cpp, binary = folder / "probe.cpp", folder / "probe"
    cpp.write_text(PREFIX + body + SUFFIX)
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(cpp), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return binary


@pytest.mark.parametrize("failed_download", (0, 2, 3, 4, 5, 6))
def test_queued_downloads_outlive_host_staging(
    staging_probe: Path, failed_download: int
) -> None:
    result = subprocess.run(
        [str(staging_probe), str(failed_download)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
void* watched=nullptr; bool pending=false,freed_pending=false;
int fail_download=0,downloads=0;
void operator delete(void* p) noexcept {
 if(p==watched && pending) freed_pending=true;
 std::free(p);
}
void operator delete(void* p,std::size_t) noexcept { ::operator delete(p); }
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
int cudaMemcpyAsync(void* dst,const void* src,std::size_t bytes,int kind,int) {
 if(kind==cudaMemcpyDeviceToHost) {
  ++downloads;
  if(downloads==fail_download) return 7;
  if(downloads==1) { watched=dst;pending=true; }
 }
 std::memcpy(dst,src,bytes);return 0;
}
int cudaMemsetAsync(void* dst,int value,std::size_t n,int) {std::memset(dst,value,n);return 0;}
int cudaGetLastError() {return 0;}
namespace runtime {
void cuda_resource_check(int status) {if(status) throw std::runtime_error("injected copy failure");}
std::size_t size_add(std::size_t a,std::size_t b,const char*) {return a+b;}
std::size_t size_mul(std::size_t a,std::size_t b,const char*) {return a*b;}
struct CudaDeviceScope {explicit CudaDeviceScope(int) {}};
struct OwnedCudaStream {
 explicit OwnedCudaStream(int) {}
 int get() const {return 1;}
 void synchronize() const {pending=false;}
 ~OwnedCudaStream() {synchronize();}
};
template<class T> struct OwnedCudaBuffer {
 T* data;std::size_t count;
 OwnedCudaBuffer(int,std::size_t n,int):data(static_cast<T*>(std::calloc(n,sizeof(T)))),count(n) {}
 T* get() {return data;}
 ~OwnedCudaBuffer() {pending=false;std::free(data);}
};
}
namespace vibeqc::dft::nlc {
struct Vv10Parameters {int variant=1;double b=6.0,c=0.01,coefficient=1.0;};
unsigned launch_blocks(std::size_t,unsigned) {return 1;}
template<class... T> void local_scales_kernel(T&&...) {}
template<class... T> void pair_kernel_ordered(T&&...) {}
"""

SUFFIX = r"""
}
int main(int argc,char** argv) {
 fail_download=argc>1 ? std::atoi(argv[1]):0;
 double points[3]{},weight=1.0,density=1.0,gradient[3]{},energy=0.0;
 double vrho=0.0,vsigma=0.0,point[3]{},weight_derivative=0.0;
 bool threw=false;
 try {vibeqc::dft::nlc::execute_vv10_cuda(points,&weight,&density,gradient,1,1,{},0,
  energy,&vrho,&vsigma,point,&weight_derivative);}
 catch(const std::exception&) {threw=true;}
 if(threw!=(fail_download!=0)) {std::cerr<<"unexpected status";return 2;}
 if(freed_pending) {std::cerr<<"host energy buffer freed before pending D2H drained";return 3;}
 if(pending) {std::cerr<<"work left queued";return 4;}
}
"""

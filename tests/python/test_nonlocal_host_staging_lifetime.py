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
    body = source[
        source.index("Vv10CudaDeviceLayout vv10_cuda_device_layout(") :
    ].rsplit("}  // namespace vibeqc::dft::nlc", 1)[0]
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


@pytest.mark.parametrize("failed_download", (0, 1, 2, 3, 4, 5, 6))
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


def test_molecular_domain_padding_is_resident_and_fail_closed() -> None:
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    body = source.split("void enqueue_vv10_molecular_domain_cuda(", 1)[1].split(
        "Vv10CudaDeviceLayout vv10_cuda_device_layout(", 1
    )[0]
    forbidden = (
        "OwnedCudaStream",
        "OwnedCudaBuffer",
        "cudaMemcpyAsync",
        "cudaMemcpyHostToDevice",
        "cudaMemcpyDeviceToHost",
        "cudaStreamSynchronize",
    )
    assert all(token not in body for token in forbidden)
    assert "cudaMemsetAsync" in body
    kernel = (
        source.split("__global__ void molecular_domain_kernel(", 1)[1].split(
            "__global__ void reduce_energy_ordered_kernel(", 1
        )[0]
        if "__global__ void reduce_energy_ordered_kernel("
        in source.split("__global__ void molecular_domain_kernel(", 1)[1]
        else source.split("__global__ void molecular_domain_kernel(", 1)[1].split(
            "}  // namespace", 1
        )[0]
    )
    assert "rho < threshold" in kernel
    assert "effective_weights[i] = inactive ? 0.0 : weight" in kernel
    assert "effective_density[i] = inactive ? 1.0 : rho" in kernel
    assert "inactive ? 0.0 : gx" in kernel
    assert "if (!valid) atomicExch(failed, 1)" in kernel


def test_resident_enqueue_has_no_hidden_allocation_transfer_or_fence() -> None:
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    body = source.split("void enqueue_vv10_cuda_device(", 1)[1].split(
        "void execute_vv10_cuda(", 1
    )[0]
    forbidden = (
        "OwnedCudaStream",
        "OwnedCudaBuffer",
        "cudaMemcpyAsync",
        "cudaMemcpyHostToDevice",
        "cudaMemcpyDeviceToHost",
        "cudaStreamSynchronize",
    )
    assert all(token not in body for token in forbidden)
    assert "cudaMemsetAsync" in body
    assert "reduce_energy_ordered_kernel" in body


PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
struct TestCudaStream {};
using cudaStream_t = TestCudaStream*;
TestCudaStream test_stream;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
int cudaMemcpyAsync(void* dst,const void* src,std::size_t bytes,int kind,cudaStream_t) {
 if(kind==cudaMemcpyDeviceToHost) {
  ++downloads;
  if(downloads==fail_download) return 7;
  if(downloads==1) { watched=dst;pending=true; }
 }
 std::memcpy(dst,src,bytes);return 0;
}
int cudaMemsetAsync(void* dst,int value,std::size_t n,cudaStream_t) {std::memset(dst,value,n);return 0;}
int cudaGetLastError() {return 0;}
namespace runtime {
void cuda_resource_check(int status) {if(status) throw std::runtime_error("injected copy failure");}
std::size_t size_add(std::size_t a,std::size_t b,const char*) {return a+b;}
std::size_t size_mul(std::size_t a,std::size_t b,const char*) {return a*b;}
struct CudaDeviceScope {explicit CudaDeviceScope(int) {}};
struct OwnedCudaStream {
 explicit OwnedCudaStream(int) {}
 cudaStream_t get() const {return &test_stream;}
 void synchronize() const {pending=false;}
 ~OwnedCudaStream() {synchronize();}
};
template<class T> struct OwnedCudaBuffer {
 T* data;std::size_t count;
 OwnedCudaBuffer(int,std::size_t n,cudaStream_t):data(static_cast<T*>(std::calloc(n,sizeof(T)))),count(n) {}
 T* get() {return data;}
 ~OwnedCudaBuffer() {pending=false;std::free(data);}
};
}
namespace vibeqc::dft::nlc {
enum class Vv10Variant {vv10=1,rvv10=2};
struct Vv10Parameters {Vv10Variant variant=Vv10Variant::vv10;double b=6.0,c=0.01,coefficient=1.0;};
struct Vv10CudaDeviceLayout {
 std::size_t point_count{},tile_points{},workspace_bytes{};
 bool features{},geometry{};
};
unsigned launch_blocks(std::size_t,unsigned) {return 1;}
template<class... T> void local_scales_kernel(T&&...) {}
template<class... T> void pair_kernel_ordered(T&&...) {}
template<class... T> void reduce_energy_ordered_kernel(T&&...) {}
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

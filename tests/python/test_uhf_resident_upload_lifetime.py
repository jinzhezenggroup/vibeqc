"""Compile native upload/cleanup control flow with deferred-copy lifetime probes."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def upload_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/api/c_api_fock.cpp").read_text()
    create = source.split(
        'extern "C" vibeqc_status vibeqc_uhf_response_resident_create(', 1
    )[1]
    body = create.split("    owner->allocation_bytes = bytes;\n", 1)[1]
    body = body.split("    *output = owner.release();", 1)[0]
    # Only deallocation is instrumented: retain every native upload, catch and
    # stream-drain statement, plus the actual vector allocation/filling order.
    body = body.replace("std::vector<double>", "TrackedVector")
    owner = source.split("struct vibeqc_uhf_response_resident {", 1)[1].split(
        "\n};", 1
    )[0]
    harness = r"""
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#define VIBEQC_HAS_CUDA 1
using cudaStream_t = int;
using cublasHandle_t = void*;
struct vibeqc_fock_plan;
namespace vibeqc::scf { struct CudaDirectJkPlan; }
struct vibeqc_uhf_response_resident { OWNER_BODY
};
struct Pending { void* dst; const void* src; std::size_t bytes; };
Pending pending[6]{};
int pending_count=0, copy_calls=0, sync_calls=0, fail_copy=0, fail_sync=0;
int allocations=0, handles=0;
bool dead_source=false;
struct TrackedVector : std::vector<double> {
  using std::vector<double>::vector;
  ~TrackedVector() {
    for(int i=0;i<pending_count;++i)
      if(pending[i].src==data()) dead_source=true;
  }
};
constexpr int cudaMemcpyHostToDevice=1, CUBLAS_POINTER_MODE_HOST=0;
int cudaSetDevice(int) { return 0; }
int cublasCreate(void** p) { *p=reinterpret_cast<void*>(1); ++handles; return 0; }
int cublasSetStream(void*,int) { return 0; }
int cublasSetPointerMode(void*,int) { return 0; }
int cublasDestroy(void*) { --handles; return 0; }
int cudaMemcpyAsync(void* dst,const void* src,std::size_t bytes,int,int) {
  ++copy_calls;
  if(copy_calls==fail_copy) return 2;
  pending[pending_count++]={dst,src,bytes};
  return 0;
}
int cudaStreamSynchronize(int) {
  ++sync_calls;
  if(fail_sync && sync_calls==1) return 3;
  // Never dereference a released source: the probe records the ordering bug.
  if(!dead_source)
    for(int i=0;i<pending_count;++i)
      std::memcpy(pending[i].dst,pending[i].src,pending[i].bytes);
  pending_count=0;
  return 0;
}
namespace vibeqc::runtime {
int resource_cuda_malloc(void** p,std::size_t n) {
  *p=std::malloc(n); if(!*p) return 4; ++allocations; return 0;
}
int resource_cuda_free(void* p) { std::free(p); --allocations; return 0; }
}
void resident_cuda(int code) { if(code) throw std::runtime_error("injected CUDA failure"); }
void resident_blas(int code) { if(code) throw std::runtime_error("injected BLAS failure"); }
void uhf_resident_sync(vibeqc_uhf_response_resident* o) {
  resident_cuda(cudaStreamSynchronize(o->stream)); ++o->synchronizations;
}
void create() {
  const std::size_t n=3,matrix=9,oa=1,va=2,ob=1,vb=2,slots=32,alpha_dim=2,beta_dim=2;
  const std::size_t bytes=4096;
  const double coefficients_alpha[9]={1,2,3,4,5,6,7,8,9};
  const double coefficients_beta[9]={9,8,7,6,5,4,3,2,1};
  const double orbital_energies_alpha[3]={-1,0,1};
  const double orbital_energies_beta[3]={-2,0,2};
  auto owner=std::make_unique<vibeqc_uhf_response_resident>();
  owner->stream=17;
  NATIVE_BODY
  if(copy_calls!=6 || pending_count || owner->h2d_bytes!=224)
    throw std::runtime_error("successful upload accounting changed");
  if(owner->coefficients_alpha[1]!=4 || owner->coefficients_beta[1]!=6)
    throw std::runtime_error("uploaded coefficient transpose changed");
  vibeqc::runtime::resource_cuda_free(owner->allocation);
  cublasDestroy(owner->blas);
}
int main(int argc,char** argv) {
  if(argc!=3) return 10;
  fail_copy=std::atoi(argv[1]); fail_sync=std::atoi(argv[2]);
  bool threw=false;
  try { create(); } catch(const std::runtime_error&) { threw=true; }
  if(dead_source) { std::cerr<<"host source released before stream drain\n"; return 1; }
  if(pending_count || allocations || handles) return 2;
  if(threw != bool(fail_copy || fail_sync)) return 3;
  return 0;
}
"""
    harness = harness.replace("OWNER_BODY", owner).replace("NATIVE_BODY", body)
    directory = tmp_path_factory.mktemp("uhf-upload-lifetime")
    cpp, executable = directory / "probe.cpp", directory / "probe"
    cpp.write_text(harness)
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(cpp), "-o", str(executable)], check=True
    )
    return executable


@pytest.mark.parametrize("failed_copy", range(7))
def test_uhf_native_upload_keeps_staging_alive_until_drain(
    upload_probe: Path, failed_copy: int
) -> None:
    subprocess.run([str(upload_probe), str(failed_copy), "0"], check=True)


def test_uhf_native_upload_keeps_staging_alive_after_sync_failure(
    upload_probe: Path,
) -> None:
    subprocess.run([str(upload_probe), "0", "1"], check=True)

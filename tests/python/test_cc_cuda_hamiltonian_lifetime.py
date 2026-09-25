"""Execute the response owner with delayed-copy CUDA doubles, without a GPU.

This tests exception cleanup and device ownership, not generated CC equations or
CUDA driver behavior. Local result vectors detect destruction before a drain.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
struct Stream { int device; };
using cudaStream_t = Stream*;
constexpr int cudaSuccess=0, cudaStreamNonBlocking=1;
constexpr int cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
struct Copy { void* dst; const void* src; std::size_t size; int kind; };
std::vector<Copy> queued;
int active_device=0, copies=0, fail_copy=0, sync_count=0, fail_sync=0;
int allocations=0, streams=0, premature_destructions=0;
bool fail_generated=false;
int cudaGetDevice(int* d) { *d=active_device; return 0; }
int cudaSetDevice(int d) { active_device=d; return 0; }
int cudaStreamCreateWithFlags(cudaStream_t* s,int) {
  *s=new Stream{active_device}; ++streams; return 0;
}
struct Allocation { void* p; int device; };
std::vector<Allocation> device_allocations;
int cudaMalloc(void** p,std::size_t n) {
  *p=std::calloc(1,n); if(!*p)return 2;
  device_allocations.push_back({*p,active_device}); ++allocations; return 0;
}
int cudaFree(void* p) {
  auto it=std::find_if(device_allocations.begin(),device_allocations.end(),
      [&](auto a){return a.p==p;});
  if(it==device_allocations.end() || it->device!=active_device)return 7;
  device_allocations.erase(it); std::free(p); --allocations; return 0;
}
int cudaStreamDestroy(cudaStream_t s) {
  if(s->device!=active_device)return 7;
  delete s; --streams; return 0;
}
int cudaMemcpyAsync(void* dst,const void* src,std::size_t n,int kind,cudaStream_t s) {
  if(s->device!=active_device)return 7;
  if(++copies==fail_copy)return 13;
  queued.push_back({dst,src,n,kind}); return 0;
}
int cudaMemsetAsync(void* p,int value,std::size_t n,cudaStream_t s) {
  if(s->device!=active_device)return 7;
  std::memset(p,value,n); return 0;
}
int cudaStreamSynchronize(cudaStream_t s) {
  if(s->device!=active_device)return 7;
  if(++sync_count==fail_sync)return 13;
  // No scientific result is modeled. Draining retires every outstanding borrow.
  queued.clear(); return 0;
}
void cuda_check(int error) { if(error)throw std::runtime_error("injected CUDA failure"); }
namespace vibeqc_tensor { struct DeviceAllocationError : std::bad_alloc {}; }
struct TrackingVector : std::vector<double> {
  using std::vector<double>::vector;
  TrackingVector()=default;
  TrackingVector(const TrackingVector&)=default;
  TrackingVector(TrackingVector&&)=default;
  ~TrackingVector() {
    for(const auto& copy:queued)
      if(copy.kind==cudaMemcpyDeviceToHost && copy.dst==data() && size())
        ++premature_destructions;
  }
};
struct CudaRawHamiltonianView { std::span<const double> density,g,h,rotation; };
struct CudaParameterResponseView {
  std::span<const double> foo,fov,fvv,ovov,ovvo,oovv,ovvv,ovoo,oooo,vvvv;
};
struct CudaHamiltonianResponseResult {
  TrackingVector hcore,eri,overlap,rotation_gradient,stationarity,orbital_rhs;
};
struct CudaHamiltonianResponseOwner { struct Impl; };
namespace generated {
struct CudaState {
  double *density{},*g{},*h{},*rotation{},*bar_foo{},*bar_fov{},*bar_fvv{},
    *bar_ovov{},*bar_ovvo{},*bar_oovv{},*bar_ovvv{},*bar_ovoo{},*bar_oooo{},
    *bar_vvvv{},*bar_reference_electronic_energy{},*bar_fock{},*d_rotation{},
    *response_arena{};
  int* error{}; std::size_t o{},v{}; cudaStream_t stream{};
};
struct DeviceHamiltonianOutputs {
  double *hcore,*eri,*overlap,*rotation_gradient,*stationarity,*orbital_rhs;
};
struct DeviceOrbitalOutputs { double* d_fov; };
std::size_t hamiltonian_weights_arena_elements(std::size_t,std::size_t) {return 1024;}
std::size_t fock_weights_arena_elements(std::size_t,std::size_t) {return 1024;}
std::size_t orbital_jvp_arena_elements(std::size_t,std::size_t) {return 1024;}
DeviceHamiltonianOutputs run_hamiltonian_weights_cuda(CudaState& s) {
  if(fail_generated)throw std::runtime_error("injected generated failure");
  auto* p=s.response_arena; return {p,p,p,p,p,p};
}
DeviceHamiltonianOutputs run_fock_weights_cuda(CudaState& s) {
  return run_hamiltonian_weights_cuda(s);
}
DeviceOrbitalOutputs run_orbital_jvp_cuda(CudaState& s) {
  if(fail_generated)throw std::runtime_error("injected generated failure");
  return {s.response_arena};
}
std::size_t checked_add(std::size_t a,std::size_t b) {
  if(a>std::numeric_limits<std::size_t>::max()-b)throw std::length_error("overflow");
  return a+b;
}
}
"""
DRIVER = r"""
int main(int argc,char** argv) {
  if(argc!=4)return 99;
  const std::string mode=argv[1], op=argv[2]; const int index=std::atoi(argv[3]);
  std::vector<double> matrix(4,1.0), tensor(16,1.0), scalar(1,1.0);
  CudaRawHamiltonianView raw{matrix,tensor,matrix,matrix};
  active_device=0;
  auto owner=std::make_unique<CudaHamiltonianResponseOwner::Impl>(1,1,raw,1,1U<<20);
  if(mode=="construct")return active_device==0 ? 0 : 1;
  if(mode=="destroy") {
    active_device=2; owner.reset();
    return active_device==2 && !allocations && !streams ? 0 : 2;
  }
  active_device=mode=="device"?2:1;
  const int before_device=active_device;
  copies=sync_count=0;
  fail_copy=mode=="copy"?index:0;
  fail_sync=mode=="sync"?1:0;
  fail_generated=mode=="generated";
  std::array<std::span<const double>,10> p;
  p.fill(scalar);
  if(mode=="invalid")p[index]={};
  CudaParameterResponseView parameters{p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9]};
  bool threw=false;
  try {
    if(op=="hamiltonian") (void)owner->hamiltonian(parameters,1.0);
    else if(op=="fock") (void)owner->fock(matrix);
    else (void)owner->orbital_jvp(matrix);
  } catch(const std::exception&) {threw=true;}
  const bool expected_failure=mode=="copy" || mode=="sync" || mode=="generated" || mode=="invalid";
  if(threw!=expected_failure) {std::cerr<<"unexpected outcome";return 3;}
  if(premature_destructions) {std::cerr<<"destroyed local DMA destination before drain";return 4;}
  if(!queued.empty()) {std::cerr<<"host borrow survived the call";return 5;}
  if(active_device!=before_device) {std::cerr<<"changed caller device";return 6;}
  if(mode=="invalid" && copies) {std::cerr<<"uploaded before full validation";return 7;}
  if(!expected_failure && sync_count!=1) {std::cerr<<"extra normal-path fence";return 8;}
  // The same retained owner must remain reusable after an injected failure.
  fail_copy=fail_sync=0;fail_generated=false;
  try {(void)owner->fock(matrix);}catch(...) {return 9;}
  return queued.empty() && active_device==before_device ? 0 : 10;
}
"""


@pytest.fixture(scope="module")
def lifetime_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++20 compiler")
    source = (ROOT / "src/cc/lambda_response_cuda.cu").read_text()
    helpers = source[source.index("std::size_t checked_add(") : source.index("struct AmplitudeLayout")]
    reserve = source[source.index("std::size_t reserve(") : source.index("class CudaLambdaActions")]
    owner = source[
        source.index("struct CudaHamiltonianResponseOwner::Impl {") : source.index(
            "CudaHamiltonianResponseOwner::CudaHamiltonianResponseOwner("
        )
    ].replace("std::vector<double>", "TrackingVector")
    folder = tmp_path_factory.mktemp("cc-hamiltonian-lifetime")
    unit, binary = folder / "probe.cpp", folder / "probe"
    unit.write_text(PREFIX + helpers + reserve + owner + DRIVER)
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", str(unit), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize(
    "mode,operation,index",
    [
        ("construct", "fock", 0),
        ("destroy", "fock", 0),
        *((mode, op, 0) for mode in ("success", "device", "sync", "generated") for op in ("hamiltonian", "fock", "orbital")),
        *(("copy", op, i) for op, count in (("hamiltonian", 18), ("fock", 8), ("orbital", 3)) for i in range(1, count + 1)),
        *(("invalid", "hamiltonian", i) for i in range(10)),
    ],
)
def test_response_call_owns_device_and_host_lifetimes(
    lifetime_probe: Path, mode: str, operation: str, index: int
) -> None:
    result = subprocess.run(
        [str(lifetime_probe), mode, operation, str(index)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""Check native Lambda host-buffer lifetimes with delayed-copy CUDA doubles.

The extracted production owner runs without a GPU. Every queued transfer keeps
its host borrow until a fence, so failures expose premature output destruction
without relying on pageable-memory staging behavior in a particular driver.
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

struct Problem {
  std::size_t nocc=1,nvir=1;
  TrackingVector foo{1.0},fov{1.0},fvv{1.0},ovov{1.0},ovvo{1.0},oovv{1.0},
    ovvv{1.0},ovoo{1.0},oooo{1.0},vvvv{1.0},d1{1.0},d2{1.0};
};
struct SolverResult { TrackingVector t1{1.0},t2{1.0}; };
struct LambdaOptions { std::size_t max_bytes=1U<<20; };
std::size_t lambda_cpu_numeric_capacity(const Problem&,const SolverResult&,
                                       const LambdaOptions&,bool) {return 1024;}
"""
DRIVER = r"""
int main(int argc,char** argv) {
  if(argc!=4)return 99;
  const std::string op=argv[1], mode=argv[2]; const int index=std::atoi(argv[3]);
  Problem p; SolverResult cc; LambdaOptions options;
  {
    CudaLambdaActions owner(p,cc,options,1,false,true);
    copies=sync_count=0;
    fail_copy=mode=="copy"?index:0;
    fail_sync=mode=="sync"?1:0;
    fail_generated=mode=="generated";
    bool threw=false;
    try {
      TrackingVector one,two;
      if(op=="replay") {
        double energy=0.0;
        owner.fresh_replay(energy,one,two);
      } else if(op=="seeds") {
        owner.set_parameter_seeds(cc.t1,cc.t2);
      } else if(op=="parameter") {
        (void)owner.parameter(generated::parameter_stub,1);
      } else if(op=="rhs" || op=="independent-rhs") {
        owner.rhs(op=="independent-rhs",one,two);
      } else {
        owner.transpose(op=="independent-transpose",cc.t1,cc.t2,one,two);
      }
    } catch(const std::exception&) {threw=true;}
    if(threw!=(mode!="success")) {std::cerr<<"unexpected exception outcome";return 1;}
    if(premature_destructions) {std::cerr<<"DMA target destroyed before drain";return 2;}
    if(!queued.empty()) {std::cerr<<"host borrow outlived response call";return 3;}
    if(mode=="success" && sync_count!=1) {std::cerr<<"extra normal-path fence";return 4;}
    // A handled failure must not poison the retained action owner.
    fail_copy=fail_sync=0; fail_generated=false;
    TrackingVector one,two;
    owner.rhs(false,one,two);
    if(!queued.empty())return 5;
  }
  return allocations || streams || active_device!=0 ? 6 : 0;
}
"""


@pytest.fixture(scope="module")
def lambda_lifetime_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++20 compiler")
    source = (ROOT / "src/cc/lambda_response_cuda.cu").read_text()
    owner = source[
        source.index("std::size_t checked_add(") : source.index("double max_abs(")
    ].replace("std::vector<double>", "TrackingVector")
    header = (ROOT / "src/cc/cuda_solver_support.cuh").read_text()
    state = header[
        header.index("struct CudaState {") : header.index(
            "DeviceIterationOutputs run_iteration_cuda"
        )
    ]
    generated = (
        "namespace generated {\n"
        + state
        + r"""
std::size_t checked_add(std::size_t a,std::size_t b) {
  if(a>std::numeric_limits<std::size_t>::max()-b)throw std::length_error("overflow");
  return a+b;
}
DeviceReplayOutputs run_replay_cuda(CudaState& s) {
  if(fail_generated)throw std::runtime_error("injected generated failure");
  return {s.replay_arena,s.replay_arena,s.replay_arena};
}
DeviceParameterOutput parameter_stub(CudaState& s) {
  if(fail_generated)throw std::runtime_error("injected generated failure");
  return {s.response_arena};
}
"""
    )
    response_programs = (
        "lambda_rhs",
        "lambda_transpose",
        "lambda_independent_rhs",
        "lambda_independent_transpose",
    )
    parameters = (
        "foo",
        "fov",
        "fvv",
        "ovov",
        "ovvo",
        "oovv",
        "ovvv",
        "ovoo",
        "oooo",
        "vvvv",
    )
    for name in ("replay", *response_programs, *(f"parameter_{p}" for p in parameters)):
        generated += (
            f"std::size_t {name}_arena_elements(std::size_t,std::size_t)"
            "{return 1024;}\n"
        )
    for name in response_programs:
        generated += (
            f"DeviceLambdaOutputs run_{name}_cuda(CudaState& s){{"
            'if(fail_generated)throw std::runtime_error("injected generated failure");'
            "return {s.response_arena,s.response_arena};}\n"
        )
    generated += "}\n"
    folder = tmp_path_factory.mktemp("cc-lambda-lifetime")
    unit, binary = folder / "probe.cpp", folder / "probe"
    unit.write_text(PREFIX + generated + owner + DRIVER)
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(unit),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


OPERATIONS = (
    ("replay", 4),
    ("rhs", 4),
    ("independent-rhs", 4),
    ("transpose", 5),
    ("independent-transpose", 5),
    ("seeds", 3),
    ("parameter", 2),
)


@pytest.mark.parametrize(
    "operation,mode,index",
    [
        *(
            (op, mode, 0)
            for op, _ in OPERATIONS
            for mode in ("success", "sync", "generated")
            if op != "seeds" or mode != "generated"
        ),
        *(
            (op, "copy", index)
            for op, count in OPERATIONS
            for index in range(1, count + 1)
        ),
    ],
)
def test_lambda_call_drains_borrowed_host_buffers(
    lambda_lifetime_probe: Path, operation: str, mode: str, index: int
) -> None:
    result = subprocess.run(
        [str(lambda_lifetime_probe), operation, mode, str(index)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

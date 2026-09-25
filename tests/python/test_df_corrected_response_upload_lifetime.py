"""Failure/rejection must drain borrowed host transfers before returning.

Execute the actual selector with delayed CUDA operations. This checks ownership
and failure propagation, not numerical density reconstruction or GPU kernels.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_SHIM = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
using cudaStream_t = int;
using vibeqc_status = int;
constexpr int cudaSuccess=0, cudaMemcpyHostToDevice=1;
constexpr int VIBEQC_STATUS_SUCCESS=0, VIBEQC_STATUS_OUT_OF_MEMORY=2, cudaFailure=3;
int mode=0, syncs=0, factor_calls=0;
std::vector<std::function<void()>> pending;
int cudaSetDevice(int) { return mode==1 ? cudaFailure : 0; }
int cudaMemcpyAsync(void* to,const void* from,std::size_t n,int,cudaStream_t) {
  if(mode==2) return cudaFailure;
  pending.emplace_back([=] { std::memcpy(to,from,n); }); return 0;
}
int cudaMemsetAsync(void* p,int value,std::size_t n,cudaStream_t) {
  if(mode==3) return cudaFailure;
  pending.emplace_back([=] { std::memset(p,value,n); }); return 0;
}
int cudaStreamSynchronize(cudaStream_t) {
  ++syncs;
  for(auto& operation:pending) operation();
  pending.clear(); return 0;
}
namespace runtime::cuda_trace {
struct TraceOperation { TraceOperation(const char*,int,std::initializer_list<std::size_t>) {} };
template<class T> void trace_counter(const char*,T) {}
}
struct Identity {
  struct { std::uint64_t basis=9,reference=1,density_generation=1,orbital_generation=1; } factor;
  std::uint64_t solve_epoch=1,model=1;
  std::vector<std::size_t> occupied{1};
};
struct CudaDfFinalStateToken { unsigned version=1; Identity identity; };
struct PersistentScfState {
  bool unrestricted=false,occupied_exchange=true,final_frames_available=true;
  std::uint32_t generation=7;
  std::uint32_t* d_alpha_factor_generation=&generation;
  double storage[2]{};
  double* d_alpha_factor=storage;
};
struct CudaDensityFittingJkPlan {
  void* persistent_scf_state=nullptr;
  bool streamed=true;
  void* integral_source=reinterpret_cast<void*>(1);
  std::size_t batch_size=1,matrix_elements=4,nbf=2,naux=2,factor_basis_identity=9;
  cudaStream_t stream=1;
  int device_id=0;
  double storage[4]{};
  double* primary_density=storage;
};
struct DensityFittingDensityResponse {
  std::span<const double> density;
  double coulomb_coefficient=1.0,exchange_coefficient=0.25;
};
struct CudaDfOccupiedResponseView {
  struct Factor { const double* coefficients{}; std::size_t rank{}; double density_scale{}; };
  std::array<Factor,3> factors{};
  std::size_t nbf{},naux{},owner_identity{};
};
bool qualified_value_rhf_exchange(const CudaDensityFittingJkPlan&,std::size_t) { return true; }
int cuda_density_fitting_final_state_token(CudaDensityFittingJkPlan*,std::size_t,
    CudaDfFinalStateToken& token,std::string&) { token={}; return 0; }
int cuda_failure(int error,const char*,std::string&) { return error; }
int factor_density_for_exchange(CudaDensityFittingJkPlan& plan,PersistentScfState&,
    const double*,bool& accepted,std::size_t& rank,std::string&) {
  ++factor_calls;
  if(mode==4) return cudaFailure;
  if(mode==5) throw std::bad_alloc();
  if(mode==6) throw std::runtime_error("factor failure");
  if(mode==7) return 0; // Rejected by a pre-eigensolve qualification check.
  cudaStreamSynchronize(plan.stream);
  accepted=mode!=8; rank=mode==9 ? 2 : 1;
  return 0;
}
"""

_DRIVER = r"""
int main(int argc,char** argv) {
  if(argc!=2) return 99;
  mode=std::atoi(argv[1]);
  PersistentScfState state;
  CudaDensityFittingJkPlan plan;
  plan.persistent_scf_state=&state;
  CudaDfFinalStateToken requested;
  requested.identity.factor.density_generation=2;
  requested.identity.factor.orbital_generation=2;
  if(mode==12) requested.identity.solve_epoch=2;
  double host[4]{1,0,0,1};
  const DensityFittingDensityResponse term{host};
  CudaDfOccupiedResponseView view;
  std::string detail;
  int status=0,exception=0;
  try {
    status=select_corrected_occupied_response_factor(plan,0,mode==11 ? nullptr : &requested,
        {&term,1},mode==10 ? 0 : 1024,view,detail);
  } catch(const std::bad_alloc&) { exception=5; }
    catch(const std::runtime_error&) { exception=6; }
  if(!pending.empty()) return 1; // Callers may release `host` once the API returns.
  if(mode>=1 && mode<=4 && status!=cudaFailure) return 2;
  if((mode==5 || mode==6) && exception!=mode) return 3;
  if(mode==0) {
    if(status || !view.owner_identity || view.factors[0].rank!=1 ||
       view.factors[0].density_scale!=1.0 || syncs!=1) return 4;
  } else if(view.owner_identity) return 5;
  if((mode==1 || mode==11 || mode==12) && (syncs || factor_calls)) return 6;
  return 0;
}
"""


@pytest.fixture(scope="module")
def selector_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text(encoding="utf-8")
    begin = source.index("vibeqc_status select_corrected_occupied_response_factor(")
    end = source.index("\n}  // namespace", begin)
    body = source[begin:end]
    directory = tmp_path_factory.mktemp("corrected-upload-lifetime")
    unit, executable = directory / "selector.cpp", directory / "selector"
    unit.write_text(_SHIM + body + _DRIVER, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
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


@pytest.mark.parametrize("mode", range(13))
def test_corrected_factor_upload_is_drained_on_failure_or_rejection(
    selector_probe: Path, mode: int
) -> None:
    result = subprocess.run(
        [str(selector_probe), str(mode)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

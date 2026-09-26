"""Execute corrected-final J/K ownership with delayed transfers, not GPU math."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_SHIM = r"""
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
using cudaStream_t=int;
using vibeqc_status=int;
constexpr int cudaSuccess=0,VIBEQC_STATUS_SUCCESS=0,failure=3;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
int mode=0,syncs=0,copies=0,factor_calls=0;
std::vector<std::function<void()>> pending;
int cudaSetDevice(int) {return mode==1 ? failure : 0;}
int cudaMemcpyAsync(void* dst,const void* src,std::size_t n,int,cudaStream_t) {
  ++copies;
  if ((mode==2 && copies==1) || (mode==13 && copies==2) || (mode==14 && copies==3)) return failure;
  pending.emplace_back([=]{std::memcpy(dst,src,n);}); return 0;
}
int cudaMemsetAsync(void* p,int v,std::size_t n,cudaStream_t) {
  if(mode==3) return failure;
  pending.emplace_back([=]{std::memset(p,v,n);}); return 0;
}
int cudaStreamSynchronize(cudaStream_t) {
  ++syncs; for(auto& op:pending) op(); pending.clear(); return 0;
}
struct TraceOperation { TraceOperation(const char*,int,std::initializer_list<std::size_t>) {} };
template<class T> void trace_counter(const char*,T) {}
namespace runtime::df_progress { void label(const char*,const char*) {} }
struct Plan {
  std::size_t nbf=2,naux=2,matrix_elements=4;
  int stream=1,device_id=0;
  void* integral_source=reinterpret_cast<void*>(1);
  bool streamed=true;
  double density[4]{},j[4]{},k[4]{};
  double *primary_density=density,*coulomb=j,*alpha_exchange=k;
};
struct State {
  std::uint32_t generation=7;
  std::uint32_t* d_alpha_factor_generation=&generation;
  double storage[2]{};
  double* d_alpha_factor=storage;
};
struct Token {
  struct {
    struct { std::uint64_t density_generation=1; } factor;
    std::array<std::size_t,1> occupied{1};
  } identity;
};
int cuda_failure(int status,const char*,std::string&) { return status; }
int factor_density_for_exchange(Plan& p,State&,const double*,bool& accepted,std::size_t& rank,std::string&) {
  ++factor_calls;
  if(mode==4) return failure;
  if(mode==5) throw std::bad_alloc();
  if(mode==6) return 0;
  cudaStreamSynchronize(p.stream);
  accepted=mode!=7; rank=mode==8 ? 2 : 1; return 0;
}
int build_coulomb(Plan& p,const double*,std::string&) {
  auto* data=p.coulomb; pending.emplace_back([=]{data[0]=2;});
  if(mode==9) return failure;
  if(mode==10) throw std::runtime_error("J failure");
  return 0;
}
int build_occupied_exchange(Plan& p,std::size_t,const double*,std::size_t,bool,double weight,double*,std::string&) {
  if(weight!=1) throw std::logic_error("changed factor weight");
  auto* data=p.alpha_exchange; pending.emplace_back([=]{data[0]=3;});
  if(mode==11) return failure;
  if(mode==12) throw std::runtime_error("K failure");
  return 0;
}
int run(Plan* plan,State* state,bool& used,std::vector<double>& coulomb,std::vector<double>& exchange) {
  const bool corrected_streamed=true, download=mode!=15;
  const auto bytes=plan->matrix_elements*sizeof(double);
  Token current,expected; expected.identity.factor.density_generation=2;
  std::vector<double> density{1,0,0,1};
  std::string detail;
  int status=0;
  auto fallback=[](const char*) {return 0;};
"""

_DRIVER = r"""
  return 99;
}
int main(int argc,char** argv) {
  if(argc!=2) return 99;
  mode=std::atoi(argv[1]);
  Plan plan; State state;
  bool used=false;
  int status=0,exception=0;
  std::vector<double> j,k;
  try { status=run(&plan,&state,used,j,k); }
  catch(const std::bad_alloc&) {exception=5;}
  catch(const std::runtime_error&) {exception=mode;}
  // The host density inside run has ended its lifetime; no copy may remain.
  if(!pending.empty()) return 1;
  const bool successful=mode==0 || mode==15;
  if(used!=successful) return 2;
  if(mode==5 || mode==10 || mode==12) {if(exception!=mode) return 3;}
  else if(mode==1 || mode==2 || mode==3 || mode==4 || mode==9 || mode==11 || mode==13 || mode==14) {
    if(status!=failure) return 4;
  } else if(status || exception) return 5;
  if(successful && syncs!=2) return 6; // Factor validation + final drain only.
  if(mode==0 && (j.size()!=4 || k.size()!=4 || j[0]!=2 || k[0]!=3)) return 7;
  if(mode==15 && (!j.empty() || !k.empty() || plan.j[0]!=2 || plan.k[0]!=3)) return 8;
  if(mode==1 && (syncs || factor_calls)) return 9;
  return 0;
}
"""


@pytest.fixture(scope="module")
def final_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    text = (ROOT / "src/scf/cuda/df_scf_final_state.cpp").read_text(encoding="utf-8")
    start = text.index("  if (corrected_streamed) {")
    stop = text.index("\n  TraceOperation trace(", start)
    directory = tmp_path_factory.mktemp("corrected-final-lifetime")
    unit, executable = directory / "final.cpp", directory / "final"
    unit.write_text(_SHIM + text[start:stop] + _DRIVER, encoding="utf-8")
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


@pytest.mark.parametrize("mode", range(16))
def test_corrected_final_drains_before_return(final_probe: Path, mode: int) -> None:
    result = subprocess.run(
        [str(final_probe), str(mode)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

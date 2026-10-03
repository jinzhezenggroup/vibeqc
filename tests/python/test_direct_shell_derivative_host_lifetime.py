"""Fault-inject the production shell derivative wrappers without CUDA hardware."""

import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.lowering.fock_accumulation import (
    emit_direct_force_density_coefficient,
)

ROOT = Path(__file__).resolve().parents[2]


def _extract_function(source: str, symbol: str) -> str:
    start = source.index(symbol)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {symbol}")


@pytest.fixture(scope="module")
def host_lifetime_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    bodies = []
    for route in ("full_range", "rsh"):
        symbol = f"cudaError_t execute_generated_{route}_energy_derivatives("
        bodies.append(_extract_function(source, symbol))
    body = "\n".join(bodies)
    folder = tmp_path_factory.mktemp("direct-shell-host-lifetime")
    (folder / "cuda_runtime.h").write_text(
        "#pragma once\n#define __device__\n#define __forceinline__ inline\n"
    )
    cpp, binary = folder / "probe.cpp", folder / "probe"
    prefix = PREFIX.replace(
        "// PRODUCTION_DENSITY_COEFFICIENT", emit_direct_force_density_coefficient()
    )
    cpp.write_text(prefix + body + SUFFIX)
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
            str(cpp),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize("throw_error", [False, True])
@pytest.mark.parametrize(
    ("route", "failed_step"),
    [
        (route, step)
        for route, count in (
            ("full_range", 8),
            ("rsh", 8),
            ("rsh_split", 14),
            ("rsh_zero", 8),
        )
        for step in range(count)
    ],
)
def test_pending_downloads_outlive_early_returns_and_exceptions(
    host_lifetime_probe: Path, route: str, failed_step: int, throw_error: bool
) -> None:
    result = subprocess.run(
        [str(host_lifetime_probe), str(failed_step), str(int(throw_error)), route],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "coulomb,short,long,density_fixture",
    [
        (1.0, 0.0, 0.0, False),
        (0.0, 0.0, 0.0, False),
        (1.0, 0.0, 0.0, True),
        (0.0, 0.0, 0.0, True),
        (1.0, -0.1, 0.0, False),
        (1.0, 0.0, -0.5, False),
    ],
)
def test_split_rsh_preserves_disabled_sources(
    host_lifetime_probe: Path,
    coulomb: float,
    short: float,
    long: float,
    density_fixture: bool,
) -> None:
    """Compile real weights for opposing finite UKS spins; no ERI is evaluated."""
    result = subprocess.run(
        [
            str(host_lifetime_probe),
            "0",
            "0",
            "rsh_split",
            str(coulomb),
            str(short),
            str(long),
            str(int(density_fixture)),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_preparation_staging_outlives_the_stream_draining_owner() -> None:
    source = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    body = source.split(
        "std::unique_ptr<GeneratedExchangePlan> prepare_generated_exchange(", 1
    )[1].split("namespace {", 1)[0]
    assert body.index("std::vector<std::uint32_t> bounded_pair_order;") < body.index(
        "auto plan = std::make_unique<GeneratedExchangePlan>();"
    )


PREFIX = r"""
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <utility>
#include <tuple>
#include <vector>
#include "scf/cuda/direct_eri_symmetry.cuh"
#include "scf/cuda/matrix_index.cuh"
using cudaError_t = int;
using cudaStream_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaMemcpyDeviceToHost=2;
int step=0, fail_step=0, syncs=0; bool throw_error=false;
unsigned source_count=2, submitted_copies=0;
bool tracking=false, pending=false, freed_pending=false;
void* watched[32]{}; unsigned watched_count=0; bool split=false;
bool density_fixture=false; unsigned range_calls=0;
void* operator new(std::size_t n) {
  void* p=std::malloc(n);
  if(!p) throw std::bad_alloc();
  if(tracking && (n==3*sizeof(double) || n==6*sizeof(double) || n==9*sizeof(double)))
    watched[watched_count++ % 32]=p;
  return p;
}
void operator delete(void* p) noexcept {
  for(auto* allocated:watched) if(p==allocated && pending) freed_pending=true;
  std::free(p);
}
void operator delete(void* p,std::size_t) noexcept { ::operator delete(p); }
int operation() {
  if(++step!=fail_step) return cudaSuccess;
  if(throw_error) throw std::runtime_error("injected CUDA wrapper exception");
  return 7;
}
struct Copy {void* dst; double values[9]; std::size_t bytes;};
Copy copies[3]{}; unsigned copy_count=0;
int cudaMemsetAsync(void* dst,int value,std::size_t n,cudaStream_t) {
  int error=operation(); if(error) return error;
  std::memset(dst,value,n); return cudaSuccess;
}
int cudaGetLastError() { return operation(); }
int cudaMemcpyAsync(void* dst,const void* src,std::size_t n,int,cudaStream_t) {
  int error=operation(); if(error) return error;
  const auto expected=(split ? (submitted_copies==0 ? 6U : 3U) : 3*source_count)*sizeof(double);
  if(n!=expected || copy_count>=1U)
    throw std::runtime_error("bad copy");
  copies[copy_count].dst=dst;
  copies[copy_count].bytes=n;
  std::memcpy(copies[copy_count++].values,src,n);
  ++submitted_copies; pending=true; return cudaSuccess;
}
int cudaStreamSynchronize(cudaStream_t) {
  ++syncs;
  int error=operation(); if(error) return error;
  for(unsigned i=0;i<copy_count;++i)
    std::memcpy(copies[i].dst,copies[i].values,copies[i].bytes);
  pending=false; copy_count=0; return cudaSuccess;
}
namespace generativeqc::scf::cuda_execution {
namespace detail { constexpr unsigned kDirectQuartetShellClassCount=1; }
enum class DirectCoulombRange { Short, Long };
struct Batch { int total_atoms=1; };
struct Shared {
  Batch batch; cudaStream_t stream=1; unsigned worker_blocks=1;
  double screening=0, *shell_bounds=nullptr, *schwarz=nullptr;
  std::uint8_t* active=nullptr;
};
struct GeneratedExchangePlan {
  Shared* shared; bool force_capability=true;
  std::uint32_t* bounded_pair_order;
  double *shell_pair_block_bounds, *force;
  unsigned long long* force_cursor;
  std::uint32_t* heads;
  double *shell_pair_density_bounds=nullptr, *system_density_bounds=nullptr;
  double* direct_spin=nullptr;
  int bounded_block_domain=0;
};
// PRODUCTION_DENSITY_COEFFICIENT
int prepare_generated_exchange_density(GeneratedExchangePlan& plan,bool,const double* alpha,const double*) {
  if(density_fixture) plan.direct_spin=const_cast<double*>(alpha);
  return operation();
}
// Independent labelled source values expose both coefficient and force-sign
// mistakes in the host decomposition, without evaluating any integral kernel.
template<class... Args> void launch_bounded_shell_energy_derivative(Args&&... args) {
  const auto values=std::make_tuple(args...);
  auto* force=std::get<14>(values);
  const auto* density=std::get<12>(values);
  const double j=density_fixture ? direct_force_density_coefficient_scaled<true>(
      2,0,0,density,1,1,0,0,std::get<16>(values),0.0) : std::get<16>(values);
  const double k=density_fixture ? direct_force_density_coefficient_scaled<true>(
      2,0,0,density,1,1,0,0,0.0,std::get<17>(values)) : std::get<17>(values);
  for(unsigned i=0;i<3;++i) {
    force[i]=-j*(1+i);
    force[3+i]=-k*(10+i);
  }
}
template<class... Args> void launch_bounded_shell_range_exchange_derivative(Args&&... args) {
  ++range_calls;
  const auto values=std::make_tuple(args...);
  auto* force=std::get<14>(values);
  const double k=density_fixture ? direct_force_density_coefficient_scaled<true>(
      2,0,0,std::get<12>(values),1,1,0,0,0.0,std::get<18>(values)) : std::get<18>(values);
  for(unsigned i=0;i<3;++i) force[i]=-k*(20+i);
}
template<class... Args> void launch_bounded_shell_rsh_derivatives(Args&&... args) {
  const auto values=std::make_tuple(args...);
  auto* force=std::get<14>(values);
  for(unsigned i=0;i<3;++i) {
    force[i]=-std::get<17>(values)*(1+i);
    force[3+i]=-std::get<18>(values)*(-10.0);
    force[6+i]=-std::get<19>(values)*(20+i);
  }
}
"""

SUFFIX = r"""
}
int main(int argc,char** argv) {
  fail_step=argc>1 ? std::atoi(argv[1]) : 0;
  throw_error=argc>2 && std::atoi(argv[2]);
  const bool zero_exchange=argc>3 && std::strcmp(argv[3],"rsh_zero")==0;
  split=zero_exchange || (argc>3 && std::strcmp(argv[3],"rsh_split")==0);
  source_count=(argc>3 && std::strcmp(argv[3],"full_range")!=0) ? 3U : 2U;
  const double cj=argc>4 ? std::strtod(argv[4],nullptr) : 1.0;
  const double cs=argc>5 ? std::strtod(argv[5],nullptr) : (zero_exchange ? 0.0 : -0.1);
  const double cl=argc>6 ? std::strtod(argv[6],nullptr) : (zero_exchange ? 0.0 : -0.5);
  density_fixture=argc>7 && std::atoi(argv[7]);
  const bool want_range=split && (cs!=0.0 || cl!=0.0);
  const int expected_syncs=want_range ? 2 : 1;
  using namespace generativeqc::scf::cuda_execution;
  Shared shared;
  std::uint32_t pair=0,head=0; unsigned long long cursor=0;
  double force[9]{}, bound=1, density=1;
  // The total density is small; the unused same-spin exchange squares overflow.
  double spin_density[]{1.0,1e200,1e200,1.0,1.0,-1e200,-1e200,1.0};
  if(density_fixture && (direct_force_density_coefficient_scaled<true>(
      2,0,0,spin_density,1,1,0,0,1.0,0.0)!=4.0 ||
      std::isfinite(direct_force_density_coefficient_scaled<true>(
          2,0,0,spin_density,1,1,0,0,0.0,1.0)))) return 9;
  GeneratedExchangePlan plan{&shared,true,&pair,&bound,force,&cursor,&head};
  std::vector<double> output{99.0};
  const auto execute = [&]() {
    submitted_copies=0; range_calls=0;
    return source_count==2
        ? execute_generated_full_range_energy_derivatives(
            plan,false,&density,nullptr,1.0,-0.5,output)
        : execute_generated_rsh_energy_derivatives(
            plan,density_fixture,density_fixture ? spin_density : &density,
            density_fixture ? spin_density+4 : nullptr,cj,cs,cl,split ? 0.3 : 0.4,output);
  };
  tracking=true;
  int status=0; bool threw=false;
  try { status=execute(); }
  catch(const std::exception&) { threw=true; }
  tracking=false;
  if(freed_pending) {std::cerr<<"result freed before queued D2H drained";return 2;}
  if(pending) {std::cerr<<"D2H still pending at API return";return 3;}
  if(fail_step) {
    if(throw_error ? !threw : status!=7) {std::cerr<<"lost injected failure";return 4;}
    if(output!=std::vector<double>{99.0}) {std::cerr<<"published partial result";return 5;}
  } else {
    if(threw || status || output.size()!=3*source_count) return 6;
    for(unsigned i=0;i<3;++i) {
      if(!std::isfinite(output[i]) || !std::isfinite(output[3+i]) ||
         std::abs(output[i]-cj*(density_fixture ? 4 : 1)*(1+i))>1e-12 ||
         std::abs(output[3+i]-(source_count==2 ? -0.5*(10+i) : -10.0*cs))>1e-12 ||
         (source_count==3 && (!std::isfinite(output[6+i]) ||
          std::abs(output[6+i]-cl*(20+i))>1e-12))) {
        std::cerr<<"published derivative source/sign/coefficient changed";return 6;
      }
    }
    if(range_calls!=(want_range ? 1U : 0U)) {std::cerr<<"unused LR worker executed";return 10;}
    if(syncs!=expected_syncs) {std::cerr<<"extra success-path synchronization";return 7;}
  }
  // Reuse the same retained owner after the failed call.
  step=0;fail_step=0;syncs=0;throw_error=false;
  if(execute()!=cudaSuccess || pending || syncs!=expected_syncs ||
     range_calls!=(want_range ? 1U : 0U)) {
    std::cerr<<"owner did not recover";return 8;
  }
}
"""

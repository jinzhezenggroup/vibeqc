"""Deferred H2D copies must borrow owner storage, not a returned stack frame."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_host_xc_staging_keeps_copy_sources_alive(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    body = source.split("  CudaXcView stage_xc(", 1)[1].split(
        "\n  void enqueue_legacy()", 1
    )[0]
    harness = r"""
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iostream>
#include "dft/semilocal_family.hpp"
using vibeqc::dft::SemilocalFamily;
using vibeqc::dft::semilocal_family_from_code;
int selected_route = -1;
namespace scf { struct ScfOptions {
 enum class XcExecutionSchedule { DeviceFused, HostUnfused };
 XcExecutionSchedule xc_execution_schedule=XcExecutionSchedule::HostUnfused;
}; }
struct CudaXcView { std::uint64_t generation; std::size_t n; unsigned spins;
 double* potential; double* totals; int* error; int stream; };
struct DeviceXC { void enqueue(double*,std::size_t,std::uint64_t) {}
 CudaXcView view(std::uint64_t) { return {}; } };
struct XcIntegral { std::vector<double> potential=std::vector<double>(4,3);
 double energy=2.5, electrons=2; };
struct SpinXcIntegral { std::array<std::vector<double>,2> potential{
 std::vector<double>(4,3),std::vector<double>(4,4)};
 double energy=2.5; std::array<double,2> electrons{1,1}; };
template<class... T> XcIntegral integrate_lda_xc_pw_rks(T&&...) { selected_route=0; return {}; }
template<class... T> XcIntegral integrate_pbe_rks_with_tail(T&&...) { selected_route=1; return {}; }
template<class... T> XcIntegral integrate_r2scan_rks(T&&...) { selected_route=2; return {}; }
template<class... T> SpinXcIntegral integrate_lda_xc_pw_uks(T&&...) { selected_route=3; return {}; }
template<class... T> SpinXcIntegral integrate_pbe_uks(T&&...) { selected_route=4; return {}; }
template<class... T> SpinXcIntegral integrate_r2scan_uks(T&&...) { selected_route=5; return {}; }
constexpr int cudaMemcpyDeviceToHost=1,cudaMemcpyHostToDevice=2;
struct Region { std::uintptr_t begin; std::size_t size; };
std::vector<Region> owned;
struct Copy { void* destination; const void* source; std::size_t bytes; };
std::vector<Copy> queued;
int cudaMemcpyAsync(void* dst,const void* src,std::size_t bytes,int kind,int) {
 if(kind==cudaMemcpyDeviceToHost) { std::memcpy(dst,src,bytes); return 0; }
 const auto address=reinterpret_cast<std::uintptr_t>(src);
 bool retained=false;
 for(auto region:owned) retained |= address>=region.begin &&
  address+bytes>=address && address+bytes<=region.begin+region.size;
 if(!retained) throw std::runtime_error("H2D source escapes plan-owned storage");
 queued.push_back({dst,src,bytes}); return 0;
}
int cudaStreamSynchronize(int) { return 0; }
void check(int status) { if(status) throw std::runtime_error("cuda error"); }
struct Owner {
 scf::ScfOptions options;
 DeviceXC* xc=nullptr;
 int basis=0,grid=0,stream=0;
 unsigned spins=1;
 SemilocalFamily functional=SemilocalFamily::Lda;
 std::size_t n=2,matrix=4,elements=4;
 struct { std::size_t tile_points=2; } xc_layout;
 struct { std::uint64_t xc_host_d2h_bytes=0,xc_host_h2d_bytes=0,
  xc_host_synchronizations=0,synchronizations=0; } movement;
 std::vector<double> host_xc_density=std::vector<double>(8),
  host_xc_alpha=std::vector<double>(4),host_xc_beta=std::vector<double>(4),
  host_xc_potential=std::vector<double>(8);
 std::array<double,3> host_xc_totals{};
 int host_xc_error=0;
 double d[8]{},p[8]{},t[3]{};
 int e=-9;
 double* density=d; double* tmp1=p; double* staged_xc_totals=t;
 int* staged_xc_error=&e;
 CudaXcView stage_xc(STAGE_BODY
};
int main() {
 for(unsigned spins:{1U,2U}) for(unsigned functional:{0U,1U,2U}) {
  auto owner=std::make_unique<Owner>();
  owner->spins=spins; owner->elements=4*spins; owner->functional=semilocal_family_from_code(functional);
  selected_route=-1;
  owned.clear(); queued.clear();
  owned.push_back({reinterpret_cast<std::uintptr_t>(owner.get()),sizeof(Owner)});
  owned.push_back({reinterpret_cast<std::uintptr_t>(owner->host_xc_potential.data()),
   owner->host_xc_potential.size()*sizeof(double)});
  try {
   const auto result=owner->stage_xc(7);
   if(selected_route!=static_cast<int>(functional+3*(spins-1)))
    throw std::runtime_error("wrong semilocal route");
   for(auto copy:queued) std::memcpy(copy.destination,copy.source,copy.bytes);
   if(result.generation!=7 || result.totals[0]!=2.5 || result.totals[1]!=1 ||
      result.totals[2]!=1 || *result.error!=0 || result.potential[0]!=3 ||
      (spins==2 && result.potential[4]!=4)) throw std::runtime_error("copy content");
   if(owner->movement.xc_host_h2d_bytes!=8*owner->elements+28)
    throw std::runtime_error("transfer accounting");
  } catch(const std::exception& e) { std::cerr<<e.what();return 1; }
 }
}
"""
    harness = harness.replace("STAGE_BODY", body)
    cpp, binary = tmp_path / "staging.cpp", tmp_path / "staging"
    cpp.write_text(harness)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "src"),
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = subprocess.run(
        [str(binary)], check=False, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr

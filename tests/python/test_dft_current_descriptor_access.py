"""Compile the two DFT descriptor readers without legacy prefix helpers.

Minimal dependent types isolate descriptor defaults/copy ownership; full native
DFT integration remains covered by the ordinary library build and native suite.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _source() -> str:
    return (ROOT / "src/methods/dft_method.cpp").read_text(encoding="utf-8")


def test_current_dft_readers_do_not_probe_legacy_prefixes() -> None:
    assert "field_present(" not in _source()


def test_current_dft_descriptor_readers_compile_and_keep_defaults(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    text = _source()
    grid = text[text.index("dft::GridSpec ks_grid_options("):text.index("Result adapt_result(")]
    auxiliary = text[
        text.index("std::optional<core::System> ks_auxiliary_template("):
        text.index("void validate_ks_auxiliary_geometry(")
    ]
    unit = tmp_path / "descriptor.cpp"
    unit.write_text(
        r'''
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>
constexpr unsigned VIBEQC_ABI_VERSION=1;
constexpr int VIBEQC_STATUS_ABI_MISMATCH=2,VIBEQC_STATUS_NOT_IMPLEMENTED=3,VIBEQC_STATUS_INVALID_ARGUMENT=4;
constexpr unsigned VIBEQC_XC_EXECUTION_DEVICE_FUSED=0,VIBEQC_XC_EXECUTION_HOST_UNFUSED=1;
struct MethodError:std::runtime_error { MethodError(int,const char* m):std::runtime_error(m){} };
struct NativeKsExecutionPlan {};
const char* expected_scf_domain(const NativeKsExecutionPlan&) {return "current-domain";}
namespace core {struct System {std::vector<int> atoms;};}
namespace scf {struct ScfOptions {
 enum class XcExecutionSchedule {DeviceFused,HostUnfused};
 std::uint64_t xc_tile_points=256;
 XcExecutionSchedule xc_execution_schedule=XcExecutionSchedule::DeviceFused;
};}
namespace dft {
struct GridSpec {
 unsigned version=1,radial_points=4,angular_polar=3,angular_azimuth=2,partition_iterations=3;
 double coincident_tolerance=1e-12;
 std::array<double,119> element_radii{};
};
void validate_grid_spec(const GridSpec& g) {if(!g.radial_points)throw std::invalid_argument("radial");}
}
struct vibeqc_ks_options {
 std::size_t struct_size=sizeof(vibeqc_ks_options); unsigned abi_version=1;
 const char* scf_domain="current-domain"; std::uint64_t tile_points=37;
 unsigned xc_execution_schedule=VIBEQC_XC_EXECUTION_HOST_UNFUSED;
 unsigned grid_version=2,radial_points=7,angular_polar=5,angular_azimuth=9,partition_iterations=4;
 double coincident_tolerance=1e-10; const double* element_radii=nullptr; std::size_t element_radius_count=0;
};
struct Auxiliary {core::System data;};
struct vibeqc_method_descriptor {
 std::size_t struct_size=sizeof(vibeqc_method_descriptor);
 const vibeqc_ks_options* ks_options=nullptr;
 const Auxiliary* density_fitting_auxiliary_basis=nullptr;
};
'''
        + grid + auxiliary + r'''
int main() {
 vibeqc_method_descriptor d; scf::ScfOptions options; NativeKsExecutionPlan plan;
 auto defaults=ks_grid_options(d,options,plan);
 assert(defaults.version==1 && options.xc_tile_points==256);
 assert(!ks_auxiliary_template(d));
 vibeqc_ks_options input; d.ks_options=&input;
 auto copied=ks_grid_options(d,options,plan);
 assert(copied.version==2 && copied.radial_points==7 && options.xc_tile_points==37);
 assert(options.xc_execution_schedule==scf::ScfOptions::XcExecutionSchedule::HostUnfused);
 unsigned rejected=0;
 for(int mode=0;mode<6;++mode) {
  auto bad=input;
  if(mode==0)bad.struct_size=0;
  if(mode==1)bad.abi_version=0;
  if(mode==2)bad.scf_domain="wrong";
  if(mode==3)bad.tile_points=0;
  if(mode==4)bad.xc_execution_schedule=9;
  if(mode==5)bad.element_radius_count=119;
  d.ks_options=&bad;
  try {(void)ks_grid_options(d,options,plan);}catch(const std::exception&){++rejected;}
 }
 assert(rejected==6);
 Auxiliary aux{{{1,8}}}; d.density_fitting_auxiliary_basis=&aux;
 auto owned=ks_auxiliary_template(d); aux.data.atoms[0]=7;
 assert(owned && owned->atoms[0]==1 && owned->atoms[1]==8);
}
''', encoding="utf-8",
    )
    binary = tmp_path / "descriptor"
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(unit), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    subprocess.run([str(binary)], check=True, timeout=10)

"""Host-execute the shared preparation gate without allocating molecular tensors."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>
int allocations=0, executions=0;
constexpr int GENERATIVEQC_BACKEND_CPU_REFERENCE=1, GENERATIVEQC_STATUS_OUT_OF_MEMORY=2, GENERATIVEQC_STATUS_NOT_IMPLEMENTED=3;
namespace core { struct System { bool df_supported=true; }; }
namespace runtime {
struct ExecutionContext {
  bool cuda=false;
  int backend() const { return cuda ? 3 : GENERATIVEQC_BACKEND_CPU_REFERENCE; }
  bool cuda_requested() const { return cuda; }
  int device_id() const { return 0; }
};
}
struct generativeqc_method_descriptor { std::size_t budget=100; bool valid=true; };
struct MethodError : std::runtime_error {
  MethodError(int,const std::string& msg) : std::runtime_error(msg) {}
};
struct Reference { std::size_t reference_memory_budget_bytes=100; int diis_history=8; double screening_tolerance=0; };
namespace scf {
namespace cuda_execution {
bool cuda_df_value_domain(const core::System& orbital,const core::System& system,std::string& detail) {
  detail="unsupported DF source basis";
  return orbital.df_supported && system.df_supported;
}
}
enum class FockSpin { Restricted };
enum class FockBackend { Cpu, Cuda };
// Match the real derivative default so value-only admission is tested.
struct FockBuildSpec { unsigned derivative_order=1; };
FockBuildSpec make_hf_fock_spec(FockSpin) { return {}; }
int resolve_fock_build(FockBuildSpec spec,FockBackend,double) {
  if (spec.derivative_order != 0) throw std::runtime_error("unused derivative source");
  return 0;
}
struct PreparedFockPlan {
  PreparedFockPlan(const core::System&,std::nullptr_t,int,int=-1) { ++allocations; }
};
}
namespace posthf {
std::size_t source_capacity(const core::System&) { return 0; }
std::size_t rhf_reference_capacity(const core::System&,int,bool) { return 80; }
std::size_t checked_add(std::size_t a,std::size_t b) { return a+b; }
}
struct Diagnostic { std::size_t numeric_capacity_bytes=80; };
struct RccsdNativeState {
  bool cached;
  std::size_t external_reservation_bytes=0;
  Diagnostic diagnostic{};
};
void validate_descriptor(const generativeqc_method_descriptor& d,const runtime::ExecutionContext&) {
  if (!d.valid) throw std::invalid_argument("invalid descriptor");
}
std::size_t correlation_budget(const generativeqc_method_descriptor& d) { return d.budget; }
int cc_options(const generativeqc_method_descriptor&,std::size_t) { return 0; }
Reference reference_options(const generativeqc_method_descriptor&,std::size_t) { return {}; }
RccsdNativeState execute_rccsd_prepared(runtime::ExecutionContext&,const core::System&,
                                      Reference,int,std::size_t,scf::PreparedFockPlan* p,
                                      const std::vector<double>*, bool*,
                                      std::unique_ptr<scf::PreparedFockPlan>*, const core::System*) {
  ++executions;
  return {p != nullptr,0,{80}};
}
"""

MAIN = r"""
int main(int argc,char** argv) {
  if (argc != 2) return 1;
  const int mode=std::atoi(argv[1]);
  runtime::ExecutionContext execution;
  core::System system;
  generativeqc_method_descriptor descriptor;
  std::unique_ptr<scf::PreparedFockPlan> cache;
  if (mode == 0) descriptor.budget=79;
  if (mode == 1) descriptor.valid=false;
  if (mode == 2) execution.cuda=true;
  if (mode == 5) {
    execution.cuda=true;
    core::System auxiliary; auxiliary.df_supported=false;
    try {
      (void)run_rccsd_native_state(execution,system,descriptor,&cache,nullptr,nullptr,0,&auxiliary);
      return 9;
    } catch (const MethodError&) {
      return executions || allocations ? 10 : 0;
    }
  }
  try {
    auto result=run_rccsd_native_state(execution,system,descriptor,mode==3 ? nullptr : &cache,
                                        nullptr,nullptr,0,nullptr);
    if (mode < 2) return 2;
    // CUDA source preparation belongs after native RHF, inside execution.
    const bool expect_cache = mode >= 4;
    if (result.cached != expect_cache) return 3;
    if (allocations != (expect_cache ? 1 : 0) || executions != 1) return 4;
    if (expect_cache) {
      auto* first=cache.get();
      result=run_rccsd_native_state(execution,system,descriptor,&cache,nullptr,nullptr,0,nullptr);
      if (!result.cached || cache.get()!=first || allocations!=1 || executions!=2) return 5;
      descriptor.budget=79;
      try { (void)run_rccsd_native_state(execution,system,descriptor,&cache,nullptr,nullptr,0,nullptr); return 6; }
      catch (const MethodError&) {}
      if (cache.get()!=first || allocations!=1 || executions!=2) return 7;
    }
  } catch (const std::exception&) {
    if (mode >= 2 || allocations || executions || cache) return 8;
  }
}
"""


def test_rccsd_admits_before_creating_or_reusing_exact_cache(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    start = source.index("RccsdNativeState run_rccsd_native_state(")
    end = source.index("\ngenerativeqc_status validate_rccsd_system", start)
    program = PREFIX + source[start:end] + MAIN
    path, executable = tmp_path / "probe.cpp", tmp_path / "probe"
    path.write_text(program)
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-DGENERATIVEQC_HAS_CUDA=1",
            str(path),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    for mode in range(6):
        result = subprocess.run(
            [str(executable), str(mode)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, (mode, result.returncode, result.stderr)
    consumer = (ROOT / "src/methods/rccsdt_method.cpp").read_text()
    assert (
        "run_rccsd_native_state(execution_, system_, descriptor_, &cpu_exact_plan_,"
        in consumer
    )
    assert "make_unique<scf::PreparedFockPlan>" not in consumer

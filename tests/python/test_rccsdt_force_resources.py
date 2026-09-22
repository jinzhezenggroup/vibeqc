"""Trace real native RCCSD(T) force allocations and enforce its endpoint cap."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from vibeqc import _native

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def force_resource_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Link the real library; allocator interception observes nested kernels too."""
    compiler = shutil.which("c++")
    if compiler is None or sys.platform == "win32":
        pytest.skip("requires a host C++ compiler with ELF/Mach-O linking")
    directory = tmp_path_factory.mktemp("rccsdt-force-resources")
    source = directory / "probe.cpp"
    source.write_text(CPP)
    library = Path(_native.load_library(device="cpu")._name).resolve()
    executable = directory / "probe"
    subprocess.run(
        [
            compiler, "-std=c++20", "-O2", "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"), str(source), str(library),
            "-Wl,-rpath," + str(library.parent), "-o", str(executable),
        ],
        check=True, capture_output=True, text=True, timeout=90,
    )
    return executable


@pytest.mark.parametrize("case", ("h2o", "nh3"))
def test_native_force_exact_cap_and_nested_live_allocations(
    force_resource_probe: Path, tmp_path: Path, case: str
) -> None:
    """Use physical converged states large enough to expose the omitted arenas."""
    inputs = json.loads(
        (ROOT / "tests/reference_data/cc/gradients" / f"{case}.json").read_text()
    )["inputs"]
    rows = [f"{len(inputs['atomic_numbers'])} {len(inputs['shells'])}"]
    rows.extend(
        " ".join(map(str, (z, *xyz)))
        for z, xyz in zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True)
    )
    for shell in inputs["shells"]:
        rows.append(
            f"{shell['atom_index']} {shell['angular_momentum']} {len(shell['primitives'])}"
        )
        rows.extend(" ".join(map(str, pair)) for pair in shell["primitives"])
    path = tmp_path / "system.txt"
    path.write_text("\n".join(rows) + "\n")
    result = subprocess.run(
        [str(force_resource_probe), str(path)],
        capture_output=True, text=True, timeout=120,
    )
    (tmp_path / "trace.json").write_text(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout)
    assert record["nested_peak"] + record["retained"] <= record["planned_peak"]
    # Ensure allocator interposition actually observed generated response work.
    assert record["largest_allocation"] > 500_000


CPP = r"""
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include "cc/rccsdt_force.hpp"
#include "methods/rccsd_method.hpp"
#include "molecule/basis.hpp"
#include "runtime/execution_context.hpp"

namespace trace {
struct alignas(std::max_align_t) Header { std::size_t bytes, epoch; };
std::size_t epoch=0, live=0, peak=0, largest=0;
bool active=false;
void start() { ++epoch; live=peak=largest=0; active=true; }
}
void* operator new(std::size_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max()-sizeof(trace::Header))
    throw std::bad_alloc();
  auto* h=static_cast<trace::Header*>(std::malloc(sizeof(trace::Header)+bytes));
  if(!h) throw std::bad_alloc();
  *h={bytes, trace::active ? trace::epoch : 0};
  if(h->epoch) {
    trace::live+=bytes; trace::peak=std::max(trace::peak,trace::live);
    trace::largest=std::max(trace::largest,bytes);
  }
  return h+1;
}
void operator delete(void* p) noexcept {
  if(!p) return;
  auto* h=static_cast<trace::Header*>(p)-1;
  if(h->epoch && h->epoch==trace::epoch) trace::live-=h->bytes;
  std::free(h);
}
void operator delete(void* p,std::size_t) noexcept { ::operator delete(p); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p,std::size_t) noexcept { ::operator delete(p); }

int main(int argc,char** argv) {
  try {
    if(argc!=2) return 1;
    std::ifstream input(argv[1]); std::size_t atoms=0,shells=0;
    input >> atoms >> shells;
    vibeqc::core::System system; system.atoms.resize(atoms);system.shells.resize(shells);
    for(auto& atom:system.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    for(auto& shell:system.shells) {
      std::size_t count=0; input >> shell.atom_index >> shell.angular_momentum >> count;
      shell.primitives.resize(count);
      for(auto& p:shell.primitives) input >> p.exponent >> p.coefficient;
    }
    if(!input) return 2;
    std::string detail;
    if(vibeqc::molecule::validate_and_normalize(system,detail)!=VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail);
    vibeqc::core::ContextState context;
    vibeqc::runtime::ExecutionContext execution(context);
    vibeqc_method_descriptor method{sizeof(vibeqc_method_descriptor),VIBEQC_ABI_VERSION,
                                    VIBEQC_METHOD_RCCSD_T,200,8,1e-13,1e-11,0.0};
    method.ccsd_max_iterations=150;method.ccsd_diis_history=6;
    method.ccsd_energy_tolerance=1e-13;method.ccsd_residual_tolerance=1e-11;
    method.correlation_memory_budget_bytes=256ULL<<20;
    auto state=vibeqc::methods::detail::run_rccsd_native_state(execution,system,method);
    if(!state.solved.converged() || !state.reference) return 3;
    const auto plan=vibeqc::cc::plan_rccsdt_force_cpu(system,*state.reference,state.problem,
                                                    state.solved,256ULL<<20);
    // Refuse before even the first triples-response output is materialized.
    trace::start(); bool refused=false;
    try {
      (void)vibeqc::cc::rccsdt_force_cpu(system,*state.reference,state.problem,state.solved,
                                       state.eps_o,state.eps_v,plan.peak_bytes-1);
    } catch(const std::length_error&) { refused=true; }
    trace::active=false;
    if(!refused || trace::largest>=4096) return 4;
    trace::start();
    const auto force=vibeqc::cc::rccsdt_force_cpu(system,*state.reference,state.problem,state.solved,
                                                state.eps_o,state.eps_v,plan.peak_bytes);
    trace::active=false;
    if(force.numeric_capacity_bytes!=plan.peak_bytes ||
       trace::peak+plan.retained_input_bytes>plan.peak_bytes) return 5;
    const auto old_capacity=state.problem.foo.capacity();
    state.problem.foo.reserve(old_capacity+32);
    const auto enlarged=vibeqc::cc::plan_rccsdt_force_cpu(system,*state.reference,state.problem,
                                                        state.solved,256ULL<<20);
    if(enlarged.peak_bytes-plan.peak_bytes !=
       (state.problem.foo.capacity()-old_capacity)*sizeof(double)) return 6;
    std::cout << "{\"nested_peak\":" << trace::peak
              << ",\"largest_allocation\":" << trace::largest
              << ",\"retained\":" << plan.retained_input_bytes
              << ",\"planned_peak\":" << plan.peak_bytes << "}\n";
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 7; }
}
"""

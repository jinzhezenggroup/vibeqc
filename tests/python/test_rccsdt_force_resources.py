"""Trace real native RCCSD(T) force allocations and enforce its endpoint cap."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc import _native

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
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(source),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return executable


@pytest.mark.parametrize("case", ("h2o", "nh3", "water2", "methane"))
def test_native_force_exact_cap_and_nested_live_allocations(
    force_resource_probe: Path, tmp_path: Path, case: str
) -> None:
    """Use physical converged states large enough to expose the omitted arenas."""
    if case in ("water2", "methane"):
        records = json.loads(
            (
                ROOT / "tests/reference_data/cc/gradients/water_clusters_ccsdt.json"
            ).read_text()
        )
        atoms = next(
            row["inputs"]
            for row in records["rows"]
            if row["atoms"] == 6 and row["geometry"] == "original"
        )
        if case == "methane":
            atoms = json.loads(
                (
                    ROOT / "tests/reference_data/cc/gradients/ch4_degenerate.json"
                ).read_text()
            )["inputs"]
        elements = json.loads(
            (ROOT / "python/generativeqc/data/basis_pack.json").read_text()
        )["bases"]["sto-3g"]["elements"]
        numbers = [{"H": 1, "C": 6, "O": 8}[z] for z, _ in atoms]
        inputs = {
            "atomic_numbers": numbers,
            "coordinates": [xyz for _, xyz in atoms],
            "shells": [
                {
                    "atom_index": i,
                    "angular_momentum": shell["angular_momentum"],
                    "primitives": list(
                        zip(shell["exponents"], shell["coefficients"], strict=True)
                    ),
                }
                for i, z in enumerate(numbers)
                for shell in elements[str(z)]
            ],
        }
    else:
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
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "GENERATIVEQC_DF_PROGRESS_TRACE": str(tmp_path / "progress.jsonl"),
        },
    )
    (tmp_path / "trace.json").write_text(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout)
    assert record["nested_peak"] + record["retained"] <= record["minimum_peak"]
    assert record["source_reads"] > 0
    # Tie observation to the current generated response arena, whose size may
    # shrink when dead intermediates share storage. An arbitrary historical
    # byte floor incorrectly rejects a successful memory optimization.
    assert record["triples_arena_bytes"] > 0
    assert record["largest_allocation"] >= record["triples_arena_bytes"]
    assert (record["triples_fock_phase_bytes"] > 0) == bool(
        record["full_triples_fock_response"]
    )
    if case == "methane":
        assert record["full_triples_fock_response"] == 1
    progress = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    curvatures = [
        float(row["value"])
        for row in progress
        if row.get("key") == "minimum_orbital_curvature"
    ]
    assert curvatures and min(curvatures) > 1e-8


CPP = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include "cc/rccsdt_force.hpp"
#include "cc/triples_response.hpp"
#include "methods/rccsd_method.hpp"
#include "molecule/basis.hpp"
#include "posthf/raw_source.hpp"
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

class CountingSource final : public generativeqc::integrals::ElectronInteractionSource {
 public:
  explicit CountingSource(const generativeqc::posthf::RawSource& source) : source_(source) {}
  const generativeqc::core::System& orbital() const override { return source_.orbital(); }
  std::size_t nbf() const override { return source_.nbf(); }
  std::size_t naux() const override { return source_.naux(); }
  std::size_t retained_numeric_bytes() const override { return source_.retained_numeric_bytes(); }
  bool supports(Operator op) const noexcept override { return source_.supports(op); }
  void read(Operator op,const std::array<std::size_t,4>& begin,
            const std::array<std::size_t,4>& count,double* out,
            std::size_t elements) const override {
    ++reads;
    source_.read(op,begin,count,out,elements);
  }
  mutable std::size_t reads=0;
 private:
  const generativeqc::posthf::RawSource& source_;
};

int main(int argc,char** argv) {
  try {
    if(argc!=2) return 1;
    std::ifstream input(argv[1]); std::size_t atoms=0,shells=0;
    input >> atoms >> shells;
    generativeqc::core::System system; system.atoms.resize(atoms);system.shells.resize(shells);
    for(auto& atom:system.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    for(auto& shell:system.shells) {
      std::size_t count=0; input >> shell.atom_index >> shell.angular_momentum >> count;
      shell.primitives.resize(count);
      for(auto& p:shell.primitives) input >> p.exponent >> p.coefficient;
    }
    if(!input) return 2;
    std::string detail;
    if(generativeqc::molecule::validate_and_normalize(system,detail)!=GENERATIVEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail);
    generativeqc::core::ContextState context;
    generativeqc::runtime::ExecutionContext execution(context);
    generativeqc_method_descriptor method{sizeof(generativeqc_method_descriptor),GENERATIVEQC_ABI_VERSION,
                                    GENERATIVEQC_METHOD_RCCSD_T,200,8,1e-13,1e-11,0.0};
    method.ccsd_max_iterations=150;method.ccsd_diis_history=6;
    method.ccsd_energy_tolerance=1e-13;method.ccsd_residual_tolerance=1e-11;
    method.correlation_memory_budget_bytes=256ULL<<20;
    auto state=generativeqc::methods::detail::run_rccsd_native_state(execution,system,method);
    if(!state.solved.converged() || !state.reference) return 3;
    generativeqc::posthf::RawSource raw_source(system);
    CountingSource force_source(raw_source);
    const auto plan=generativeqc::cc::plan_rccsdt_force_cpu(
        system,force_source,*state.reference,state.problem,state.solved,256ULL<<20);
    // Refuse before even the first triples-response output or source tile is materialized.
    trace::start(); bool refused=false;
    try {
      (void)generativeqc::cc::rccsdt_force_cpu(system,force_source,*state.reference,state.problem,
                                       state.solved,state.eps_o,state.eps_v,plan.minimum_peak_bytes-1);
    } catch(const std::length_error&) { refused=true; }
    trace::active=false;
    if(!refused || trace::largest>=4096 || force_source.reads!=0) return 4;
    // Obtain the actual arena contract independently of the complete-force
    // allocation trace, and release these outputs before tracing that endpoint.
    std::size_t triples_arena_bytes=0;
    {
      generativeqc::cc::TriplesResponseOptions options;
      options.max_bytes=256ULL<<20;
      const auto triples=generativeqc::cc::triples_response_cpu(
          state.problem,state.solved,state.eps_o,state.eps_v,options);
      triples_arena_bytes=triples.arena_bytes;
    }
    trace::start();
    const auto force=generativeqc::cc::rccsdt_force_cpu(system,force_source,*state.reference,
                                                state.problem,state.solved,state.eps_o,state.eps_v,
                                                plan.peak_bytes);
    trace::active=false;
    if(force.numeric_capacity_bytes!=plan.peak_bytes || force_source.reads==0 ||
       trace::peak+plan.retained_input_bytes>plan.peak_bytes) return 5;
    const auto tiles=(state.reference->nbf+plan.raw_provider_axis_tile-1)/
                     plan.raw_provider_axis_tile;
    if(force_source.reads!=tiles*tiles*tiles*tiles ||
       force.raw_source_reads!=force_source.reads || force.raw_device_source_reads!=0 ||
       force.raw_source_values!=state.reference->nbf*state.reference->nbf*
                                state.reference->nbf*state.reference->nbf) return 8;
    if(plan.minimum_peak_bytes<plan.peak_bytes) {
      const auto minimum=generativeqc::cc::plan_rccsdt_force_cpu(
          system,force_source,*state.reference,state.problem,state.solved,plan.minimum_peak_bytes);
      force_source.reads=0;
      trace::start();
      const auto constrained=generativeqc::cc::rccsdt_force_cpu(
          system,force_source,*state.reference,state.problem,state.solved,state.eps_o,state.eps_v,
          plan.minimum_peak_bytes);
      trace::active=false;
      if(constrained.numeric_capacity_bytes>plan.minimum_peak_bytes ||
         trace::peak+minimum.retained_input_bytes>plan.minimum_peak_bytes ||
         force_source.reads<force.raw_source_reads) return 9;
      for(std::size_t i=0;i<force.forces.size();++i)
        if(std::abs(force.forces[i]-constrained.forces[i])>1e-11) return 10;
    }
    const auto old_capacity=state.problem.foo.capacity();
    state.problem.foo.reserve(old_capacity+32);
    const auto enlarged=generativeqc::cc::plan_rccsdt_force_cpu(
        system,force_source,*state.reference,state.problem,state.solved,256ULL<<20);
    if(enlarged.peak_bytes-plan.peak_bytes !=
       (state.problem.foo.capacity()-old_capacity)*sizeof(double)) return 6;
    std::cout << "{\"nested_peak\":" << trace::peak
              << ",\"full_triples_fock_response\":" << plan.full_triples_fock_response
              << ",\"triples_fock_phase_bytes\":" << plan.triples_fock_phase_bytes
              << ",\"largest_allocation\":" << trace::largest
              << ",\"triples_arena_bytes\":" << triples_arena_bytes
              << ",\"source_reads\":" << force.raw_source_reads
              << ",\"minimum_source_reads\":" << force_source.reads
              << ",\"source_tile\":" << plan.raw_provider_axis_tile
              << ",\"transform_fmas\":" << force.raw_transform_fmas
              << ",\"minimum_peak\":" << plan.minimum_peak_bytes
              << ",\"retained\":" << plan.retained_input_bytes
              << ",\"planned_peak\":" << plan.peak_bytes << "}\n";
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 7; }
}
"""

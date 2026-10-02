"""Exercise the live force planner's CUDA MO-provider lifetime without a GPU."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_force_provider_backend_and_complete_cap(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/cc/rccsdt_force.cpp").read_text()
    planner = source.split("static RccsdtForcePlan plan_relaxed_rccsd_force_cpu", 1)[1]
    planner = (
        "static RccsdtForcePlan plan_relaxed_rccsd_force_cpu"
        + planner.split("\nRccsdtForcePlan plan_rccsd_force_cpu", 1)[0]
    )
    helpers = source[
        source.index("std::size_t checked_add(") : source.index("bool finite(")
    ]
    parameters = source[
        source.index("std::size_t parameter_elements(") : source.index(
            "ParameterWeights parameter_vjp("
        )
    ]
    # Isolate the newly composed raw phase. Other scientific phases deliberately
    # reserve zero scratch here; the real generated provider model remains live.
    names = sorted(set(re.findall(r"generated::(\w+)\(", planner)))
    generated = (
        "namespace generated {\nstd::size_t scratch = 0;\n"
        + "\n".join(
            f"template<class... T> std::size_t {name}(T...) {{ return scratch; }}"
            for name in names
        )
        + "\n}\n"
    )
    program = PREFIX + helpers + parameters + generated + planner + MAIN
    path, executable = tmp_path / "planner.cpp", tmp_path / "planner"
    path.write_text(program)
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(path),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cuda_force_uses_its_admitted_provider_allowance() -> None:
    source = (ROOT / "src/cc/rccsdt_force.cpp").read_text()
    execution = source.split("static RccsdtForceResult relaxed_rccsd_force_impl", 1)[1]
    assert "source.retained_numeric_bytes(), cuda_derivative)" in execution
    assert execution.index("const auto resources") < execution.index("triples.emplace(")
    assert "resources.raw_provider_budget_bytes" in execution
    assert "resources.raw_provider_axis_tile" in execution
    assert (
        "NativeBlockProvider provider(source, ref, provider_budget, axis_tile,"
        in source
    )
    assert "posthf::AOTileDomain::Basis" in source


PREFIX = r"""
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "cc/rccsdt_force.hpp"
#include "cc/triples_response_internal.hpp"
#include "hf/reference.hpp"
#include "posthf/block_capacity_generated.hpp"

namespace generativeqc::molecule {
std::size_t ao_count(const core::System& system) noexcept { return system.shells.size(); }
}
namespace generativeqc::response {
GmresPlan prepare_gmres(std::size_t, const GmresOptions&) { return {}; }
}
namespace generativeqc::cc {
namespace detail {
// This harness isolates the raw MO-provider phase. Like the generated response
// arena stubs, the separately owned triples layout contributes no scratch here.
TriplesResponseLayout triples_response_layout(std::size_t, std::size_t, std::size_t, bool) {
  return {};
}
}
std::size_t problem_host_bytes(const Problem&) { return 1024; }
std::size_t lambda_cpu_numeric_capacity(const Problem& p, const SolverResult& result,
                                      const LambdaOptions&, bool) {
  return p.reference_retained_bytes + problem_host_bytes(p) +
         8 * (result.t1.capacity() + result.t2.capacity());
}
"""

MAIN = r"""
}
int main() {
  using namespace generativeqc;
  for (std::size_t n=2; n<=12; ++n) {
    core::System system; system.atoms.resize(1); system.shells.resize(n);
    hf::PhysicalReference reference; reference.nbf=n;
    for (auto* values : {&reference.overlap, &reference.hcore, &reference.fock,
                         &reference.coefficients, &reference.density}) values->resize(n*n);
    reference.orbital_energies.resize(n);
    for (std::size_t o=1; o<n; ++o) for (bool triples : {false,true}) {
      reference.nocc=o;
      cc::Problem problem; problem.nocc=o; problem.nvir=n-o;
      problem.reference_retained_bytes=8*(5*n*n+n);
      cc::SolverResult result; result.t1.resize(o*(n-o));
      result.t2.resize(o*o*(n-o)*(n-o));
      constexpr std::size_t source_bytes=8192, budget=256ULL<<20;
      const auto cpu=cc::plan_relaxed_rccsd_force_cpu(
          system,reference,problem,result,budget,triples,source_bytes,false);
      const auto gpu=cc::plan_relaxed_rccsd_force_cpu(
          system,reference,problem,result,budget,triples,source_bytes,true);
      const auto provider=posthf::numeric_block_plan(
          n,8*(5*n*n+n),source_bytes,{n,n,n,n},{n,n,n,n},true);
      if (gpu.raw_phase_bytes-cpu.raw_phase_bytes!=provider.device_bytes) return 1;
      if (gpu.raw_provider_budget_bytes!=provider.host_bytes+provider.device_bytes) return 2;
      if (gpu.raw_provider_budget_bytes>=gpu.raw_phase_bytes) return 3;
      if (gpu.peak_bytes<gpu.raw_phase_bytes) return 4;
      const auto exact=cc::plan_relaxed_rccsd_force_cpu(
          system,reference,problem,result,gpu.peak_bytes,triples,source_bytes,true);
      if (exact.peak_bytes!=gpu.peak_bytes) return 5;
      const auto tight=cc::plan_relaxed_rccsd_force_cpu(
          system,reference,problem,result,gpu.peak_bytes-1,triples,source_bytes,true);
      if (tight.peak_bytes>=gpu.peak_bytes || tight.raw_provider_axis_tile>=n) return 11;
      const auto minimum=cc::plan_relaxed_rccsd_force_cpu(
          system,reference,problem,result,gpu.minimum_peak_bytes,triples,source_bytes,true);
      if (minimum.raw_provider_axis_tile!=1 || minimum.peak_bytes!=gpu.minimum_peak_bytes) return 12;
      bool rejected=false;
      try {
        (void)cc::plan_relaxed_rccsd_force_cpu(
            system,reference,problem,result,gpu.minimum_peak_bytes-1,triples,source_bytes,true);
      } catch (const std::length_error&) { rejected=true; }
      if (!rejected) return 6;
    }
  }
  // Expensive response arenas may dominate even the full-basis provider.
  // Their unavoidable peak must still reject one byte less before execution.
  cc::generated::scratch = 1ULL << 28;
  for (std::size_t n : {14, 28}) for (bool cuda : {false, true}) {
    core::System system; system.atoms.resize(1); system.shells.resize(n);
    hf::PhysicalReference reference; reference.nbf=n; reference.nocc=n/2;
    cc::Problem problem; problem.nocc=n/2; problem.nvir=n-n/2;
    cc::SolverResult result;
    constexpr std::size_t budget=8ULL<<30, source_bytes=8192;
    const auto wide=cc::plan_relaxed_rccsd_force_cpu(
        system,reference,problem,result,budget,true,source_bytes,cuda);
    if (wide.raw_provider_axis_tile!=n || wide.raw_phase_bytes>wide.peak_bytes) return 7;
    const auto provider=posthf::numeric_block_plan(
        n,8*(5*n*n+n),source_bytes,{n,n,n,n},{n,n,n,n},cuda);
    if (wide.raw_provider_budget_bytes!=provider.host_bytes+provider.device_bytes) return 8;
    const auto exact=cc::plan_relaxed_rccsd_force_cpu(
        system,reference,problem,result,wide.peak_bytes,true,source_bytes,cuda);
    if (exact.peak_bytes!=wide.peak_bytes || exact.raw_provider_axis_tile!=n) return 9;
    bool rejected=false;
    try {
      (void)cc::plan_relaxed_rccsd_force_cpu(
          system,reference,problem,result,wide.peak_bytes-1,true,source_bytes,cuda);
    } catch (const std::length_error&) { rejected=true; }
    if (!rejected) return 10;
  }
}
"""

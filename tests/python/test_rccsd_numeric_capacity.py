"""Actual native solver admission includes transient DIIS and result storage."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_cpu_solver_reserves_all_known_live_numeric_buffers(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    # The standalone generator isolates its tooling import environment. Do not
    # leak those generation-only shims into the running pytest interpreter.
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_rccsd_native.py"),
            "--cpu-header",
            str(tmp_path / "generated_rccsd_cpu.hpp"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    source = tmp_path / "capacity.cpp"
    source.write_text(CPP)
    executable = tmp_path / "capacity"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-DVIBEQC_HAS_CUDA=0",
            "-I" + str(root / "src"),
            "-I" + str(tmp_path),
            str(root / "src/cc/solver.cpp"),
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr


CPP = r"""
#include "cc/solver.hpp"
#include "generated_rccsd_cpu.hpp"
#include <iostream>
#include <stdexcept>
int main() {
  using namespace vibeqc::cc;
  Problem p; p.nocc = 1; p.nvir = 2;
  p.foo = {-1}; p.fov.resize(2); p.fvv = {1,0,0,1};
  p.ovov.resize(4); p.ovvo.resize(4); p.oovv.resize(4);
  p.ovvv.resize(8); p.ovoo.resize(2); p.oooo.resize(1); p.vvvv.resize(16);
  p.d1.assign(2,-2); p.d2.assign(4,-4); p.initial_t1.resize(2); p.initial_t2.resize(4);
  const std::size_t elements = 6;
  const auto base = problem_host_bytes(p) + sizeof(double) *
      (generated::iteration_arena_elements(1,2) + generated::replay_arena_elements(1,2));
  for (unsigned history : {0U,2U}) {
    SolverOptions options; options.diis_size = history;
    options.max_bytes = base + sizeof(double) * (2 + 2 * history) * elements;
    bool refused = false;
    try { solve_cpu(p,options); } catch (const std::length_error&) { refused=true; }
    if (!refused) { std::cerr << "transient CPU/DIIS buffers were not reserved\n"; return 1; }
    // Current/trial/error and a copied history vector can coexist before trim.
    // Gram, original+copied augmented solve and two RHS vectors are separate.
    const auto h = std::size_t(history), n = h + 1;
    const auto scratch = h ? h*h + 2*n*n + 2*n : 0;
    const auto exact = base + sizeof(double) * ((4 + 2*h)*elements + scratch);
    options.max_bytes = exact - 1; refused=false;
    try { solve_cpu(p,options); } catch (const std::length_error&) { refused=true; }
    if (!refused) return 2;
    options.max_bytes = exact;
    const auto result=solve_cpu(p,options);
    if (!result.converged() || result.diagnostic.numeric_capacity_bytes != exact) return 3;
  }
  const auto before=problem_host_bytes(p), old_capacity=p.fvv.capacity();
  p.fvv.reserve(old_capacity+32);
  if (problem_host_bytes(p) != before + (p.fvv.capacity()-old_capacity)*sizeof(double)) return 4;
  std::cout << "CPU transient/history/exact-budget and retained-capacity gates passed\n";
}
"""

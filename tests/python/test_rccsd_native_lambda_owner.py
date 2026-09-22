"""Native RCCSD Lambda owner reuses generated actions and bounded GMRES."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_lambda_owner_fresh_replay_and_budget(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    header = tmp_path / "generated_rccsd_cpu.hpp"
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_rccsd_native.py"),
            "--cpu-header",
            str(header),
        ],
        cwd=root,
        check=True,
        timeout=60,
    )
    source = tmp_path / "lambda_owner.cpp"
    source.write_text(CPP)
    executable = tmp_path / "lambda_owner"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-DVIBEQC_HAS_CUDA=0",
            "-I" + str(root / "src"),
            "-I" + str(tmp_path),
            str(root / "src/cc/solver.cpp"),
            str(root / "src/cc/lambda_response.cpp"),
            str(root / "src/response/native_gmres.cpp"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


CPP = r"""
#include "cc/lambda_response.hpp"
#include "cc/solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace vibeqc::cc;
  Problem p;
  p.nocc = 1;
  p.nvir = 2;
  p.foo = {-1.0};
  p.fov = {0.0, 0.0};
  p.fvv = {1.0, 0.0, 0.0, 1.5};
  p.ovov.assign(4, 0.0);
  p.ovvo.assign(4, 0.0);
  p.oovv.assign(4, 0.0);
  p.ovvv.assign(8, 0.0);
  p.ovoo.assign(2, 0.0);
  p.oooo.assign(1, 0.0);
  p.vvvv.assign(16, 0.0);
  p.d1 = {-2.0, -2.5};
  p.d2 = {-4.0, -4.5, -4.5, -5.0};
  p.initial_t1.assign(2, 0.0);
  p.initial_t2.assign(4, 0.0);

  SolverOptions cc_options;
  cc_options.diis_size = 0;
  cc_options.max_bytes = 64ULL << 20;
  const auto cc = solve_cpu(p, cc_options);
  if (!cc.converged()) {
    std::cerr << "CC did not converge: " << cc.reason << "\n";
    return 1;
  }

  LambdaOptions refused;
  refused.max_bytes = 1;
  bool rejected = false;
  try {
    (void)solve_lambda_cpu(p, cc, refused);
  } catch (const std::length_error&) {
    rejected = true;
  }
  if (!rejected) return 2;

  LambdaOptions options;
  options.max_bytes = 64ULL << 20;
  const auto lambda = solve_lambda_cpu(p, cc, options);
  if (!lambda.converged()) return 3;
  if (lambda.lambda1.size() != 2 || lambda.lambda2.size() != 4) return 4;
  if (std::any_of(lambda.lambda1.begin(), lambda.lambda1.end(),
                  [](double x) { return x != 0.0; }))
    return 5;
  if (std::any_of(lambda.lambda2.begin(), lambda.lambda2.end(),
                  [](double x) { return x != 0.0; }))
    return 6;
  if (lambda.diagnostic.cc_r1_max != 0.0 || lambda.diagnostic.cc_r2_max != 0.0 ||
      lambda.diagnostic.lambda_residual_norm != 0.0 ||
      lambda.diagnostic.independent_residual_norm != 0.0 ||
      lambda.diagnostic.independent_residual_max != 0.0)
    return 7;
  if (!lambda.diagnostic.shared_program_hash ||
      !lambda.diagnostic.independent_program_hash)
    return 8;
  std::cout << "native Lambda owner replay/budget/audit passed\n";
  return 0;
}
"""

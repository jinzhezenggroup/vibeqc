from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_response_symmetry_contract_compiles_and_runs(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")

    source = tmp_path / "response_symmetry_contract.cpp"
    executable = tmp_path / "response_symmetry_contract"
    source.write_text(
        r'''
#include <array>
#include <span>
#include <stdexcept>

#include "response/linear_problem.hpp"

int main() {
  using vibeqc::response::LinearResponseProblem;
  using vibeqc::response::LinearResponseSymmetry;

  auto general_apply = [](std::span<const double> input, std::span<double> output) {
    output[0] = input[0] + 2.0 * input[1];
    output[1] = 3.0 * input[0] + 4.0 * input[1];
  };
  LinearResponseProblem general(2, general_apply);
  if (general.symmetry() != LinearResponseSymmetry::General || general.has_transpose()) return 1;
  try {
    (void)general.apply_transpose();
    return 2;
  } catch (const std::logic_error&) {
  }

  LinearResponseProblem symmetric(2, general_apply, LinearResponseSymmetry::Symmetric);
  if (symmetric.symmetry() != LinearResponseSymmetry::Symmetric || !symmetric.has_transpose())
    return 3;
  const std::array<double, 2> input{1.0, 2.0};
  std::array<double, 2> output{};
  symmetric.apply_transpose()(input, output);
  if (output[0] != 5.0 || output[1] != 11.0) return 4;

  auto transpose_apply = [](std::span<const double> input, std::span<double> output) {
    output[0] = input[0] + 3.0 * input[1];
    output[1] = 2.0 * input[0] + 4.0 * input[1];
  };
  LinearResponseProblem nonsymmetric(
      2, general_apply, LinearResponseSymmetry::General, transpose_apply);
  if (nonsymmetric.symmetry() != LinearResponseSymmetry::General || !nonsymmetric.has_transpose())
    return 5;
  output = {};
  nonsymmetric.apply_transpose()(input, output);
  if (output[0] != 7.0 || output[1] != 10.0) return 6;

  return 0;
}
'''
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([str(executable)], cwd=ROOT, check=True)


def test_mp2_response_adapter_declares_symmetric_operator() -> None:
    source = (ROOT / "src/posthf/mp2_force.cpp").read_text()
    start = source.index("response::LinearResponseProblem response_problem")
    end = source.index("ConventionalForceResult conventional_force_impl", start)
    adapter = source[start:end]

    assert "response::LinearResponseSymmetry::Symmetric" in adapter

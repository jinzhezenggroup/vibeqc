"""Preparation bounds distinguish properties, providers, copies and tiny budgets."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_preparation_shapes_and_value_response_partition(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "budget.cpp"
    source.write_text(
        r"""
#include "scf/df_preparation_budget.hpp"
#include <iostream>
int main() {
  using namespace vibeqc::scf;
  // 96 atoms: the measured 800 Cartesian / 768 public AO workload. This
  // shape-only check must not allocate its multi-gigabyte derivative tensors.
  DfPreparationShape shape{800, 768, 96, 448, 736, 448, 736, false, false};
  const auto energy = df_preparation_storage(shape);
  shape.include_derivatives = true;
  const auto force = df_preparation_storage(shape);
  shape.generated_one_electron = true;
  const auto generated = df_preparation_storage(shape);
  shape.public_nbf = shape.cartesian_nbf;
  shape.generated_one_electron = false;
  const auto cartesian = df_preparation_storage(shape);
  shape.cartesian_nbf = std::numeric_limits<std::size_t>::max();
  const auto overflow = df_preparation_storage(shape);
  std::cout << "[" << energy.peak_bytes << "," << force.peak_bytes << ","
            << generated.peak_bytes << "," << cartesian.peak_bytes << ","
            << force.retained_bytes << "," << overflow.peak_bytes << ","
            << df_value_budget(17, false) << "," << df_value_budget(17, true) << ","
            << df_force_budget(17) << "," << df_value_budget(0, true) << ","
            << df_force_budget(0) << "," << df_value_budget(1, true) << ","
            << df_force_budget(1) << "]\n";
}
"""
    )
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "budget"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    values = json.loads(subprocess.check_output([str(executable)], text=True))
    energy, force, generated, cartesian, retained, overflow = values[:6]
    assert energy < 1 << 30
    assert generated < 1 << 30
    # Native input and transformed output both remain live during conversion.
    assert force > 8 * 2 * 288 * (800**2 + 768**2) > 5 << 30
    assert retained > 8 * 2 * 288 * 768**2
    assert cartesian > force  # Identity conversion still returns a deep copy.
    assert force - energy > 8 * 2 * 288 * (800**2 + 768**2)
    assert generated - energy == 2 * 8 * 288  # Nuclear derivatives only.
    assert overflow == 2**64 - 1
    assert values[6:] == [17, 8, 8, 0, 128 << 20, 1, 1]

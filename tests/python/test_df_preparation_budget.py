"""Preparation bounds distinguish properties, providers, copies and tiny budgets."""

import json
import shutil
import subprocess
import typing
from pathlib import Path

import pytest


def test_preparation_shapes_and_resource_policy_envelopes(
    tmp_path: typing.Any,
) -> None:
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
  const auto explicit_force =
      resolve_df_budget({24, 60, 5, 2, 6, true}, {}, 17);
  const auto explicit_energy =
      resolve_df_budget({24, 60, 5, 2, 6, false}, {}, 17);
  const auto small =
      resolve_df_budget({12, 24, 3, 1, 4, true}, {}, 0);
  const auto large =
      resolve_df_budget({300, 700, 80, 8, 8, true}, {}, 0);
  const auto small_repeat =
      resolve_df_budget({12, 24, 3, 1, 4, true}, {}, 0);
  const auto constrained =
      resolve_df_budget({300, 700, 80, 8, 8, true},
                        {400U << 20, 8ULL << 30, true}, 0);
  const auto roomy =
      resolve_df_budget({300, 700, 80, 8, 8, true},
                        {8ULL << 30, 16ULL << 30, true}, 0);
  const auto impossible =
      resolve_df_budget({24, 60, 5, 2, 6, true}, {}, 1);
  std::cout << "[" << energy.peak_bytes << "," << force.peak_bytes << ","
            << generated.peak_bytes << "," << cartesian.peak_bytes << ","
            << force.retained_bytes << "," << overflow.peak_bytes << ","
            << explicit_force.total_bytes << "," << explicit_force.value_bytes << ","
            << explicit_force.response_bytes << "," << explicit_energy.value_bytes << ","
            << explicit_energy.response_bytes << "," << small.total_bytes << ","
            << large.total_bytes << "," << small_repeat.total_bytes << ","
            << constrained.total_bytes << "," << constrained.reserved_headroom_bytes << ","
            << roomy.total_bytes << "," << roomy.reserved_headroom_bytes << ","
            << impossible.feasible << "," << impossible.value_bytes << ","
            << impossible.response_bytes << "]\n";
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
    (
        explicit_total,
        explicit_value,
        explicit_response,
        energy_value,
        energy_response,
        small,
        large,
        small_repeat,
        constrained,
        constrained_headroom,
        roomy,
        roomy_headroom,
        impossible,
        impossible_value,
        impossible_response,
    ) = values[6:]
    assert explicit_total == 17
    assert explicit_value + explicit_response == explicit_total
    assert 0 < explicit_response < explicit_total
    assert energy_value == 17 and energy_response == 0
    assert small == small_repeat
    assert 32 << 20 <= small < large <= 1 << 30
    assert constrained <= (400 << 20)
    assert constrained_headroom == 200 << 20  # Actual half-free fallback reservation.
    assert roomy >= constrained
    assert roomy_headroom >= 256 << 20
    assert [impossible, impossible_value, impossible_response] == [0, 1, 0]


def test_constrained_headroom_reports_the_actual_reservation(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "headroom.cpp"
    source.write_text(r"""
#include "scf/df_preparation_budget.hpp"
int main() {
  using namespace vibeqc::scf;
  const DfBudgetWorkload shape{300,700,80,8,8,true};
  const auto tight = resolve_df_budget(shape,{400ULL<<20,8ULL<<30,true},0);
  if (tight.reserved_headroom_bytes > tight.observed_free_bytes) return 1;
  if (tight.total_bytes != (150ULL<<20)) return 2;
  if (tight.reserved_headroom_bytes != (200ULL<<20)) return 3;
  for (std::size_t free : {std::size_t(0),std::size_t(1),std::size_t(2),std::size_t(7)}) {
    const auto budget = resolve_df_budget(shape,{free,8ULL<<30,true},0);
    if (budget.reserved_headroom_bytes > free) return 4;
    if (budget.total_bytes > free-budget.reserved_headroom_bytes) return 5;
  }
  return 0;
}
""")
    output = tmp_path / "headroom"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(output),
        ],
        check=True,
    )
    subprocess.run([str(output)], check=True)

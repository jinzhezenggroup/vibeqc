"""Preparation bounds distinguish properties, providers, copies and tiny budgets."""

import json
import os
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


def test_live_auto_budget_can_retain_large_values_without_widening_caps(
    tmp_path: Path,
) -> None:
    """The no-probe safety ceiling must not force roomy devices to stream."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "resident_budget.cpp"
    source.write_text(r"""
#include "scf/df_preparation_budget.hpp"
int main() {
  using namespace vibeqc::scf;
  const DfResourceEnvelope roomy{30ULL<<30,32ULL<<30,true};
  for (bool forces : {false,true}) {
    const DfBudgetWorkload shape{768,768,96,1,6,forces};
    const auto live = resolve_df_budget(shape,roomy,0);
    const auto fallback = resolve_df_budget(shape,{},0);
    const auto tensor = 768ULL*768ULL*768ULL*sizeof(double);
    if (!live.feasible || live.value_bytes < 4*tensor) return 1;
    if (live.total_bytes != live.value_bytes+live.response_bytes) return 2;
    const auto available = roomy.free_bytes-live.reserved_headroom_bytes;
    if (live.total_bytes > available-available/4) return 3;
    if (fallback.total_bytes > (1ULL<<30)) return 4;
    if (fallback != resolve_df_budget(shape,{},0)) return 5;
    for (std::size_t cap : {std::size_t(1),std::size_t(32ULL<<20),
                            std::size_t(1ULL<<30),std::size_t(2ULL<<30)}) {
      const auto bounded = resolve_df_budget(shape,roomy,cap);
      if (bounded.total_bytes != cap) return 6;
      if (bounded.value_bytes+bounded.response_bytes != cap) return 7;
    }
    const auto tight = resolve_df_budget(shape,{400ULL<<20,8ULL<<30,true},0);
    if (tight.total_bytes != (150ULL<<20)) return 8;
  }
  const auto huge = std::numeric_limits<std::size_t>::max();
  const auto saturated = resolve_df_budget({huge,huge,huge,huge,12,true},roomy,0);
  if (!saturated.feasible || saturated.total_bytes > roomy.free_bytes) return 9;
  if (saturated.value_bytes+saturated.response_bytes != saturated.total_bytes) return 10;
}
""")
    executable = tmp_path / "resident_budget"
    root = Path(__file__).resolve().parents[2]
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
    subprocess.run([str(executable)], check=True)


def test_roomy_live_budget_admits_the_batch_resident_value_owner(
    tmp_path: Path,
) -> None:
    """A roomy device must not stream a full resident batch for lack of budget."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "resident_batch_budget.cpp"
    source.write_text(r"""
#include "scf/df_preparation_budget.hpp"
int main() {
  using namespace vibeqc::scf;
  const DfBudgetWorkload work{96, 96, 12, 4, 8, true};
  const auto floor = df_resident_value_admission_floor(work);
  const auto roomy = resolve_df_budget(work, {30ULL << 30, 32ULL << 30, true}, 0);
  const auto tight = resolve_df_budget(work, {400ULL << 20, 8ULL << 30, true}, 0);
  if (!roomy.feasible || roomy.value_bytes < floor) return 1;
  // The same shape on a constrained live envelope retains the streamed
  // workload target instead of borrowing the response owner for residency.
  if (tight.value_bytes >= floor || tight.total_bytes >= roomy.total_bytes) return 2;
  if (roomy.value_bytes + roomy.response_bytes != roomy.total_bytes) return 3;
  return 0;
}
""")
    executable = tmp_path / "resident_batch_budget"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_single_packed_value_owner_has_distinct_capacity_and_identity(
    tmp_path: Path,
) -> None:
    """A single fitted owner cannot silently reserve or advertise raw storage."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "single_packed.cpp"
    source.write_text(r"""
#include "scf/df_value_storage.hpp"
#include <cstdlib>
int main() {
  using namespace vibeqc::scf;
  const auto full = df_packed_value_capacity(1,768,3712,160,8);
  const auto single = df_packed_value_capacity(1,768,3712,160,8,false);
  if (single.factor_bytes != 8769110016ULL ||
      single.factor_bytes != full.factor_bytes ||
      single.scratch_bytes != full.scratch_bytes) return 1;
  if (setenv("VIBEQC_DF_VALUE_STORAGE","packed-single",1)) return 2;
  const auto selected = requested_df_pair_storage();
  if (!df_packed_pairs(selected) || df_retains_packed_raw(selected) ||
      selected == DfPairStorage::SymmetricLower) return 3;
  if (df_packed_pairs(static_cast<DfPairStorage>(123))) return 4;
  return 0;
}
""")
    executable = tmp_path / "single_packed"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_single_packed_96_atom_plan_keeps_values_when_occupied_scratch_does_not_fit(
    tmp_path: Path,
) -> None:
    """A 96-atom default allowance admits B without forcing raw regeneration."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    # CI passes the binary directory of its CPU build. Local preset/out-of-tree
    # builds can supply the same variable; never borrow another preset's output.
    binary_dir = Path(os.environ.get("VIBEQC_BUILD_DIR", root / "build"))
    generated = binary_dir / "generated"
    if not (generated / "generated_df_exchange_schedule.hpp").exists():
        pytest.skip(f"requires generated DF source schedule in {generated}")
    source = tmp_path / "single_packed_plan.cpp"
    source.write_text(r"""
#include <iostream>
#include "src/scf/density_fitting.cpp"
int main() {
  using namespace vibeqc::scf;
  const auto default_budget = plan_packed_density_fitting_tiles(
      1,768,3712,160,13685173124ULL,0,160,false);
  if (!default_budget.stores_full_three_center ||
      default_budget.value_storage.pairs != DfPairStorage::SymmetricLowerSingle ||
      default_budget.value_storage.rank_capacity != 0 ||
      default_budget.peak_workspace_bytes > 13685173124ULL) return 1;
  const auto more_values = plan_packed_density_fitting_tiles(
      1,768,3712,160,16421977600ULL,0,160,false);
  if (more_values.value_storage.rank_capacity != 160 ||
      more_values.automatic_rhf_rank != 160 ||
      more_values.peak_workspace_bytes > 16421977600ULL) return 2;
  try {
    (void)plan_packed_density_fitting_tiles(1,768,3712,160,1,0,160,false);
    return 3;
  } catch (const DensityFittingBudgetError&) {
  }
  const auto narrow = plan_packed_density_fitting_tiles(1,1,256,0,0,0,0,false);
  if (narrow.auxiliary_tile != 256) return 4;
  try {
    (void)plan_packed_density_fitting_tiles(
        1,1,256,0,narrow.peak_workspace_bytes-1,0,0,false);
    return 5;
  } catch (const DensityFittingBudgetError&) {
  }
  return 0;
}
""")
    executable = tmp_path / "single_packed_plan"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            "-I" + str(root),
            "-I" + str(root / "include"),
            "-I" + str(root / "src"),
            "-I" + str(generated),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)

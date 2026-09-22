"""Qualification and source-retirement gates for compiler-owned GFN2 ES2 math."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.method.gfn2_es2_runtime import (
    build_gfn2_es2_arithmetic_hardness_program,
    build_gfn2_es2_cached_gradient_weight_program,
    build_gfn2_es2_energy_update_program,
    build_gfn2_es2_gradient_projection_program,
    build_gfn2_es2_pair_primal,
    build_gfn2_es2_pair_vjp,
    build_gfn2_es2_potential_update_program,
)
from vibeqc_compiler.tensor import Program, execute

ROOT = Path(__file__).resolve().parents[2]


def _scalar_outputs(program: Program, **feeds: float) -> dict[str, float]:
    result = execute(
        program,
        {name: np.asarray(value, dtype=np.float64) for name, value in feeds.items()},
    ).outputs
    return {name: float(np.asarray(value)) for name, value in result.items()}


@pytest.mark.parametrize(
    ("dx", "dy", "dz", "hardness", "q1", "q2"),
    (
        (1.2, -0.7, 0.4, 0.43, 0.8, -0.3),
        (-2.4, 0.2, 1.1, 0.71, -1.2, 0.6),
        (0.03, -0.04, 0.05, 1.17, 0.2, 0.9),
    ),
)
def test_cached_gradient_lowering_matches_generated_pair_vjp(
    dx: float, dy: float, dz: float, hardness: float, q1: float, q2: float
) -> None:
    primal_inputs = {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "pair_hardness": hardness,
        "first_charge": q1,
        "second_charge": q2,
    }
    primal = _scalar_outputs(build_gfn2_es2_pair_primal(), **primal_inputs)
    reverse = _scalar_outputs(
        build_gfn2_es2_pair_vjp().program,
        **primal_inputs,
        bar_pair_energy=1.0,
    )
    weight = _scalar_outputs(
        build_gfn2_es2_cached_gradient_weight_program(),
        kernel=primal["kernel"],
        first_charge=q1,
        second_charge=q2,
    )["weight"]
    projected = _scalar_outputs(
        build_gfn2_es2_gradient_projection_program(),
        weight=weight,
        dx=dx,
        dy=dy,
        dz=dz,
    )
    np.testing.assert_allclose(
        [projected["gx"], projected["gy"], projected["gz"]],
        [reverse["bar_dx"], reverse["bar_dy"], reverse["bar_dz"]],
        rtol=2e-15,
        atol=2e-16,
    )


def test_arithmetic_hardness_graph_matches_gfn2_mean() -> None:
    actual = _scalar_outputs(
        build_gfn2_es2_arithmetic_hardness_program(),
        first_hardness=0.43,
        second_hardness=0.71,
    )["average"]
    assert actual == np.float64(0.5 * 0.43 + 0.5 * 0.71)


def test_primal_matches_softened_coulomb_equation() -> None:
    feeds = {
        "dx": 1.1,
        "dy": -0.6,
        "dz": 0.8,
        "pair_hardness": 0.52,
        "first_charge": 0.7,
        "second_charge": -0.4,
    }
    actual = _scalar_outputs(build_gfn2_es2_pair_primal(), **feeds)
    softened = math.sqrt(
        feeds["dx"] ** 2
        + feeds["dy"] ** 2
        + feeds["dz"] ** 2
        + (1.0 / feeds["pair_hardness"]) ** 2
    )
    kernel = 1.0 / softened
    assert actual["kernel"] == pytest.approx(kernel, rel=2e-16)
    assert actual["pair_energy"] == pytest.approx(
        feeds["first_charge"] * feeds["second_charge"] * kernel, rel=2e-16
    )


def test_serial_potential_and_energy_updates_preserve_runtime_order() -> None:
    potential = _scalar_outputs(
        build_gfn2_es2_potential_update_program(),
        kernel=0.31,
        charge=-0.27,
        accumulator=0.14,
    )["updated"]
    assert potential == np.float64(0.14 + 0.31 * -0.27)
    energy = _scalar_outputs(
        build_gfn2_es2_energy_update_program(),
        row_charge=-0.62,
        potential=potential,
        accumulator=0.09,
    )["updated"]
    assert energy == np.float64(0.09 + 0.5 * -0.62 * potential)


def test_native_es2_codegen_needs_no_site_packages(tmp_path: Path) -> None:
    output = tmp_path / "generated_gfn2_es2_native.hpp"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools" / "generate_gfn2_es2_native.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = output.read_text(encoding="utf-8")
    assert "gfn2_es2_arithmetic_hardness_logical_hash" in source
    assert "gfn2_es2_pair_primal_logical_hash" in source
    assert "gfn2_es2_pair_vjp_hash" in source
    assert "evaluate_gfn2_es2_kernel_from_hardness" in source
    assert "accumulate_gfn2_es2_potential" in source
    assert "accumulate_gfn2_es2_energy" in source
    assert "evaluate_gfn2_es2_cached_gradient_weight" in source
    assert "project_gfn2_es2_gradient" in source
    assert "__host__ __device__" in source


def test_production_es2_consumers_do_not_restore_handwritten_science() -> None:
    cpu = (ROOT / "src/xtb/gfn2_runtime/src/model/gfn2/es2.cpp").read_text(
        encoding="utf-8"
    )
    cuda = (ROOT / "src/xtb/gfn2_runtime/src/backends/cuda/gfn2_es2.cu").read_text(
        encoding="utf-8"
    )
    assert "generated_gfn2_es2_native.hpp" in cpu
    assert "generated_gfn2_es2_native.hpp" in cuda
    forbidden = (
        "0.5 * row_charge * potential",
        "0.5 * shell_charges",
        "pair_contribution = -weighted",
        "contribution *= kernel",
        "const double softened_distance",
    )
    for source in (cpu, cuda):
        for formula in forbidden:
            assert formula not in source

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.geometry.gfn2_pair import (
    build_gfn2_runtime_pair_kernel,
    build_gfn2_runtime_pair_primal,
)
from vibeqc_compiler.tensor import execute

ROOT = Path(__file__).resolve().parents[2]


def _primal_inputs(distance: float, light_pair: float) -> dict[str, np.ndarray]:
    return {
        "distance": np.asarray(distance, dtype=np.float64),
        "radius": np.asarray(2.55, dtype=np.float64),
        "pair_alpha": np.asarray(1.17, dtype=np.float64),
        "pair_charge": np.asarray(5.2, dtype=np.float64),
        "light_pair": np.asarray(light_pair, dtype=np.float64),
    }


@pytest.mark.parametrize("light_pair", (0.0, 1.0))
def test_gfn2_runtime_pair_kernel_matches_primal_and_centered_difference(
    light_pair: float,
) -> None:
    distance = 2.3
    step = 1.0e-6
    primal = build_gfn2_runtime_pair_primal()
    kernel = build_gfn2_runtime_pair_kernel()

    inputs = _primal_inputs(distance, light_pair)
    kernel_inputs = {
        **inputs,
        "d_distance": np.asarray(1.0, dtype=np.float64),
    }
    values = execute(kernel, kernel_inputs).outputs
    direct = execute(primal, inputs).outputs
    plus = execute(primal, _primal_inputs(distance + step, light_pair)).outputs
    minus = execute(primal, _primal_inputs(distance - step, light_pair)).outputs

    assert values["coordination"] == direct["coordination"]
    assert values["repulsion_energy"] == direct["repulsion_energy"]
    for output, derivative in (
        ("coordination", "coordination_distance_derivative"),
        ("repulsion_energy", "repulsion_distance_derivative"),
    ):
        finite_difference = (plus[output] - minus[output]) / (2.0 * step)
        assert values[derivative] == pytest.approx(
            finite_difference, rel=2.0e-8, abs=2.0e-10
        )


def test_gfn2_runtime_pair_primal_matches_documented_scalar_equations() -> None:
    distance = 2.3
    radius = 2.55
    pair_alpha = 1.17
    pair_charge = 5.2
    inputs = _primal_inputs(distance, 0.0)
    values = execute(build_gfn2_runtime_pair_primal(), inputs).outputs

    first = 1.0 / (1.0 + math.exp(-10.0 * (radius / distance - 1.0)))
    second = 1.0 / (1.0 + math.exp(-20.0 * ((radius + 2.0) / distance - 1.0)))
    repulsion = (
        pair_charge * math.exp(-pair_alpha * distance * math.sqrt(distance)) / distance
    )
    assert values["coordination"] == pytest.approx(first * second, rel=1.0e-15)
    assert values["repulsion_energy"] == pytest.approx(repulsion, rel=1.0e-15)


def test_gfn2_native_pair_codegen_needs_no_site_packages(tmp_path: Path) -> None:
    output = tmp_path / "generated_gfn2_pair_native.hpp"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools" / "generate_gfn2_pair_native.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = output.read_text(encoding="utf-8")
    assert "gfn2_pair_primal_logical_hash" in source
    assert "gfn2_pair_distance_jvp_hash" in source
    assert "evaluate_gfn2_coordination_pair" in source
    assert "evaluate_gfn2_repulsion_pair" in source

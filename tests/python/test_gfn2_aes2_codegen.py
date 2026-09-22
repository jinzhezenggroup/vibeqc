import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.method.gfn2_aes2 import (
    PAIR_GEOMETRY_NAMES,
    PAIR_MULTIPOLE_NAMES,
    build_gfn2_aes2_kernel_program,
    build_gfn2_aes2_kernel_vjp_program,
    build_gfn2_aes2_onsite_energy_program,
    build_gfn2_aes2_onsite_potential_program,
    build_gfn2_aes2_pair_energy_program,
    build_gfn2_aes2_pair_geometry_vjp_program,
    build_gfn2_aes2_pair_potential_program,
    build_gfn2_aes2_radius_from_fraction_program,
)
from vibeqc_compiler.tensor import execute
from vibeqc_compiler.tensor.program import Program

ROOT = Path(__file__).resolve().parents[2]


def _scalar(value: float) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _pair_inputs() -> dict[str, np.ndarray]:
    values = {
        "dx": 1.2,
        "dy": -0.7,
        "dz": 0.9,
        "kernel3": 0.031,
        "kernel5": 0.0067,
        "first_charge": 0.3,
        "second_charge": -0.2,
        "first_dipole_x": 0.1,
        "first_dipole_y": -0.2,
        "first_dipole_z": 0.3,
        "second_dipole_x": -0.2,
        "second_dipole_y": 0.05,
        "second_dipole_z": 0.1,
        "first_quadrupole_0": 0.01,
        "first_quadrupole_1": 0.02,
        "first_quadrupole_2": -0.03,
        "first_quadrupole_3": 0.04,
        "first_quadrupole_4": 0.05,
        "first_quadrupole_5": -0.02,
        "second_quadrupole_0": -0.02,
        "second_quadrupole_1": 0.01,
        "second_quadrupole_2": 0.03,
        "second_quadrupole_3": -0.01,
        "second_quadrupole_4": 0.02,
        "second_quadrupole_5": 0.04,
    }
    return {name: _scalar(value) for name, value in values.items()}


def _centered(
    program: Program, inputs: dict[str, np.ndarray], name: str, step: float = 1.0e-6
) -> float:
    plus = dict(inputs)
    minus = dict(inputs)
    plus[name] = _scalar(float(inputs[name]) + step)
    minus[name] = _scalar(float(inputs[name]) - step)
    return float(
        (
            execute(program, plus).outputs["energy"]
            - execute(program, minus).outputs["energy"]
        )
        / (2.0 * step)
    )


def test_aes2_pair_reverse_ad_matches_centered_differences() -> None:
    primal = build_gfn2_aes2_pair_energy_program()
    multipole_vjp = build_gfn2_aes2_pair_potential_program().program
    geometry_vjp = build_gfn2_aes2_pair_geometry_vjp_program().program
    inputs = _pair_inputs()

    multipole = execute(multipole_vjp, {**inputs, "bar_energy": _scalar(1.0)}).outputs
    geometry = execute(geometry_vjp, {**inputs, "bar_energy": _scalar(1.0)}).outputs
    for name in (
        "first_charge",
        "first_dipole_y",
        "second_quadrupole_4",
    ):
        assert float(multipole[f"bar_{name}"]) == pytest.approx(
            _centered(primal, inputs, name), rel=2.0e-9, abs=2.0e-11
        )
    for name in PAIR_GEOMETRY_NAMES:
        assert float(geometry[f"bar_{name}"]) == pytest.approx(
            _centered(primal, inputs, name), rel=2.0e-9, abs=2.0e-11
        )


def test_aes2_onsite_potential_is_energy_reverse_ad() -> None:
    primal = build_gfn2_aes2_onsite_energy_program()
    vjp = build_gfn2_aes2_onsite_potential_program().program
    inputs = {
        "dipole_kernel": _scalar(0.13),
        "quadrupole_kernel": _scalar(0.07),
        "dipole_x": _scalar(0.1),
        "dipole_y": _scalar(-0.2),
        "dipole_z": _scalar(0.3),
        **{f"quadrupole_{i}": _scalar(0.01 * (i - 2)) for i in range(6)},
    }
    values = execute(vjp, {**inputs, "bar_energy": _scalar(1.0)}).outputs
    for name in ("dipole_x", "dipole_z", "quadrupole_1", "quadrupole_4"):
        assert float(values[f"bar_{name}"]) == pytest.approx(
            _centered(primal, inputs, name), rel=2.0e-9, abs=2.0e-11
        )


def test_aes2_kernel_reverse_ad_and_radius_graph() -> None:
    kernel = build_gfn2_aes2_kernel_program()
    kernel_vjp = build_gfn2_aes2_kernel_vjp_program().program
    distance = 2.3
    radius = 2.8
    values = execute(
        kernel_vjp,
        {
            "distance": _scalar(distance),
            "radius": _scalar(radius),
            "bar_kernel3": _scalar(0.7),
            "bar_kernel5": _scalar(-0.4),
        },
    ).outputs

    def objective(r: float, a: float) -> float:
        out = execute(kernel, {"distance": _scalar(r), "radius": _scalar(a)}).outputs
        return 0.7 * float(out["kernel3"]) - 0.4 * float(out["kernel5"])

    step = 1.0e-6
    dr = (objective(distance + step, radius) - objective(distance - step, radius)) / (
        2 * step
    )
    da = (objective(distance, radius + step) - objective(distance, radius - step)) / (
        2 * step
    )
    assert float(values["bar_distance"]) == pytest.approx(dr, rel=2.0e-8, abs=2.0e-10)
    assert float(values["bar_radius"]) == pytest.approx(da, rel=2.0e-8, abs=2.0e-10)

    argument = 4.0 * (1.9 - 1.1 - 1.2)
    fraction = 1.0 / (1.0 + math.exp(-argument))
    radius_values = execute(
        build_gfn2_aes2_radius_from_fraction_program(),
        {"base_radius": _scalar(2.2), "fraction": _scalar(fraction)},
    ).outputs
    expected = 2.2 + (5.0 - 2.2) * fraction
    derivative = (5.0 - 2.2) * 4.0 * fraction * (1.0 - fraction)
    assert float(radius_values["radius"]) == pytest.approx(expected)
    assert float(radius_values["cn_derivative"]) == pytest.approx(derivative)


def test_aes2_native_codegen_needs_no_site_packages(tmp_path: Path) -> None:
    cpu = tmp_path / "generated_gfn2_aes2_native.hpp"
    cuda = tmp_path / "generated_gfn2_aes2_native.cuh"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools" / "generate_gfn2_aes2_native.py"),
            "--cpu-output",
            str(cpu),
            "--cuda-output",
            str(cuda),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    cpu_source = cpu.read_text(encoding="utf-8")
    cuda_source = cuda.read_text(encoding="utf-8")
    for marker in (
        "gfn2_aes2_pair_energy_logical_hash",
        "gfn2_aes2_pair_potential_vjp_hash",
        "gfn2_aes2_pair_geometry_vjp_hash",
        "evaluate_gfn2_aes2_pair_vjp",
    ):
        assert marker in cpu_source
        assert marker in cuda_source
    assert "__device__ inline bool gfn2_aes2_pair_energy_tensor" in cuda_source
    assert "__device__ inline bool evaluate_gfn2_aes2_pair_vjp" in cuda_source
    assert "import numpy" not in cpu_source
    assert set(PAIR_MULTIPOLE_NAMES)


def test_aes2_runtime_contains_no_duplicate_scientific_formulas() -> None:
    cpu = (ROOT / "src/xtb/gfn2_runtime/src/model/gfn2/aes2.cpp").read_text(
        encoding="utf-8"
    )
    cuda = (ROOT / "src/xtb/gfn2_runtime/src/backends/cuda/gfn2_aes2.cu").read_text(
        encoding="utf-8"
    )
    for source in (cpu, cuda):
        assert "packed_pair_tensor" not in source
        assert "packed_dot" not in source
        assert "kernel3_distance_derivative" not in source
        assert "kernel5_radius_derivative" not in source
        assert "double logistic(" not in source
        assert "evaluate_gfn2_aes2_pair_energy" in source
        assert "evaluate_gfn2_aes2_pair_vjp" in source
    assert "evaluate_gfn2_aes2_pair_potential" in cpu
    assert "evaluate_gfn2_aes2_pair_potential" in cuda

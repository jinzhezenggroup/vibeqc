"""Correctness tests for build-time shell-class symbolic code generation."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys

import pytest

from tools.qce_codegen import NvrtcCacheSpec, build_psss_kernel, nvrtc_cache_key
from tools.qce_codegen.shell_class import AXES, CENTERS, emit_psss_cuda


def boys_values(argument: float, count: int = 3) -> list[float]:
    """Reference Boys sequence sufficient for the psss pilot."""

    if argument < 1.0e-8:
        return [
            sum(
                (-argument) ** term
                / (math.factorial(term) * (2 * order + 2 * term + 1))
                for term in range(18)
            )
            for order in range(count)
        ]
    root = math.sqrt(argument)
    values = [0.5 * math.sqrt(math.pi / argument) * math.erf(root)]
    exponential = math.exp(-argument)
    for order in range(count - 1):
        values.append(
            ((2 * order + 1) * values[-1] - exponential) / (2 * argument)
        )
    return values


def sample_variables() -> dict[str, float]:
    values = {
        "alpha": 1.3,
        "beta": 0.7,
        "gamma": 0.9,
        "delta": 0.5,
        "kPi": math.pi,
    }
    positions = {
        "first": (0.1, -0.3, 0.2),
        "second": (-0.4, 0.2, 0.5),
        "third": (0.6, -0.1, -0.2),
        "fourth": (-0.2, 0.4, -0.6),
    }
    for center, position in positions.items():
        for axis, coordinate in zip(AXES, position, strict=True):
            values[f"{center}_{axis}"] = coordinate
    return values


def evaluate_value(kernel, values: dict[str, float]) -> float:
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument)):
        values[f"boys_{order}"] = value
    return kernel.graph.evaluate(kernel.value, values)


@pytest.mark.parametrize("p_axis", AXES)
def test_psss_symbolic_gradients_match_finite_difference(p_axis: str):
    kernel = build_psss_kernel(p_axis)
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument)):
        values[f"boys_{order}"] = value

    step = 2.0e-6
    for center_index, center in enumerate(CENTERS[:3]):
        for axis_index, axis in enumerate(AXES):
            variable = f"{center}_{axis}"
            plus = dict(values)
            minus = dict(values)
            plus[variable] += step
            minus[variable] -= step
            numerical = (
                evaluate_value(kernel, plus) - evaluate_value(kernel, minus)
            ) / (2.0 * step)
            analytic = kernel.graph.evaluate(
                kernel.gradients[center_index][axis_index], values
            )
            assert analytic == pytest.approx(numerical, rel=2.0e-8, abs=2.0e-9)


def test_psss_fourth_center_uses_exact_translation_recovery():
    kernel = build_psss_kernel("x")
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument)):
        values[f"boys_{order}"] = value
    for axis in range(3):
        total = sum(
            kernel.graph.evaluate(kernel.gradients[center][axis], values)
            for center in range(4)
        )
        assert total == pytest.approx(0.0, abs=2.0e-14)


def test_cuda_emission_is_deterministic_and_runtime_ad_free():
    first = emit_psss_cuda(build_psss_kernel("z"))
    second = emit_psss_cuda(build_psss_kernel("z"))
    assert first == second
    assert "boys_values<2>" in first
    assert "generated_psss_z_gradient" in first
    assert "Dual3" not in first


def test_nvrtc_cache_key_covers_binary_compatibility_inputs():
    specification = NvrtcCacheSpec(
        generator_abi="1",
        shell_class=(2, 1, 2, 0),
        derivative_centers=(0, 1, 2),
        precision_policy="fp64",
        screening_policy="density-v1",
        source_digest="abc123",
        compute_capability="sm_120",
        nvrtc_version="12.9",
        driver_version="580.95.05",
    )
    key = nvrtc_cache_key(specification)
    assert len(key) == 64
    assert key == nvrtc_cache_key(specification)
    assert key != nvrtc_cache_key(
        replace(specification, precision_policy="mixed-fp32-control")
    )


def test_codegen_cli_writes_deterministic_aot_candidate(tmp_path: Path):
    output = tmp_path / "generated" / "psss_x.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "psss",
        "--axis",
        "x",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert "generated_psss_x_gradient" in first

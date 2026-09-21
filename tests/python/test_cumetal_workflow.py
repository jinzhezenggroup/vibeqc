"""Exercise CuMetal toolkit version rewrites without an Apple GPU."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/cumetal-cuda.yml"
CUMETAL_APPLE_CMAKE_CONFORMANCE = "73ee848cdb88f15518f05278ce16d55edfbbbc44"


def test_cumetal_jobs_share_apple_cmake_conformance_pin() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    pins = re.findall(r"CUMETAL_COMMIT: ([0-9a-f]{40})", workflow)
    assert pins == [CUMETAL_APPLE_CMAKE_CONFORMANCE] * 2


@pytest.mark.skipif(shutil.which("sed") is None, reason="sed is not installed")
def test_cumetal_toolkit_version_rewrites_match_literal_dots() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    expressions = re.findall(r"sed -i '' '([^']+)'", workflow)
    version_rewrites = [
        expression for expression in expressions if "/12.9/" in expression
    ]
    assert len(version_rewrites) == 2, (
        "validate both runtime and benchmark toolkit setup"
    )
    source = "CUDA Version 12.2.0\nrelease 12.2, V12.2.0\n12x2 must stay unchanged\n"
    expected = "CUDA Version 12.9.0\nrelease 12.9, V12.9.0\n12x2 must stay unchanged\n"
    for expression in version_rewrites:
        result = subprocess.run(
            ["sed", expression],
            input=source,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        assert result.stdout == expected, f"invalid toolkit rewrite: {expression}"


def test_real_endpoint_benchmarks_do_not_enable_per_launch_logging() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    measured = workflow.split(
        "- name: Run CodSpeed real VibeQC CuMetal endpoint suite", 1
    )[1]
    measured = measured.split("\n      - name:", 1)[0]
    assert "CUMETAL_DEBUG_REGISTRATION" not in measured
    assert "CUMETAL_DEBUG_LAUNCH" not in measured
    diagnostics = workflow.split("- name: Diagnose failed real CuMetal endpoints", 1)[1]
    diagnostics = diagnostics.split("\n      - name:", 1)[0]
    assert "failure()" in diagnostics
    assert "steps.real_endpoints.outcome == 'failure'" in diagnostics
    assert 'CUMETAL_DEBUG_REGISTRATION: "1"' in diagnostics
    assert 'CUMETAL_DEBUG_LAUNCH: "1"' in diagnostics
    assert "--codspeed" not in diagnostics


def test_real_endpoint_build_disables_ptx_jump_tables() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    build = workflow.split("- name: Build VibeQC CuMetal endpoint library", 1)[1]
    build = build.split("\n      - name:", 1)[0]
    assert "-DCMAKE_CUDA_FLAGS=-fno-jump-tables" in build


@pytest.mark.parametrize(
    "step_name",
    (
        "Run CodSpeed real VibeQC CuMetal endpoint suite",
        "Diagnose failed real CuMetal endpoints",
    ),
)
def test_real_endpoint_selects_typed_ieee64_backend(step_name: str) -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split(f"- name: {step_name}", 1)[1]
    step = step.split("\n      - name:", 1)[0]
    assert "CUMETAL_PTX_BACKEND: cumetal-ir" in step
    assert "CUMETAL_FP64_MODE: ieee64" in step


def test_real_endpoint_benchmark_environment_does_not_build_native_wheel() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    job = workflow.split("CuMetal performance (CodSpeed / Apple GPU)", 1)[1]
    prepare = job.split("- name: Prepare benchmark environment", 1)[1]
    prepare = prepare.split("\n      - name:", 1)[0]
    assert "--editable ." not in prepare
    assert (
        "uv pip install --python .venv/bin/python numpy pytest-codspeed==5.0.3"
        in prepare
    )
    assert "PYTHONPATH: ${{ github.workspace }}/python:${{ github.workspace }}" in job


def test_one_electron_cuda_entrypoints_flatten_typed_device_views() -> None:
    values = (ROOT / "src/scf/cuda/one_electron_values.cu").read_text(encoding="utf-8")
    derivatives = (ROOT / "src/scf/cuda/one_electron_derivatives.cu").read_text(
        encoding="utf-8"
    )

    assert "thread_pairs_flat<<<" in values
    assert "shell_warp_pairs_flat<<<" in values
    assert "pairs::thread_pairs<Policy" not in values
    assert "pairs::shell_warp_pairs<Policy" not in values
    for kernel in (
        "nucleus_cooperative_gradient",
        "thread_gradient",
        "shell_warp_gradient",
        "serial_gradient",
    ):
        signature = derivatives.split(f"__global__ void {kernel}(", 1)[1].split(
            ") {", 1
        )[0]
        assert "VIBEQC_ONE_ELECTRON_VIEW_KERNEL_PARAMETERS" in signature
        assert "OneElectronDeviceView" not in signature
        assert "OneElectronWeightView" not in signature

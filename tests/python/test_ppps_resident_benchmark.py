"""Correctness and timing gates for the isolated ppps resident-bra kernel."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.vibeqc_codegen.benchmark import (
    emit_ppps_resident_bra_benchmark_cuda,
    emit_shell_class_benchmark_cuda,
)
from tools.vibeqc_codegen.fused_schedule import build_fused_shell_plan
from tools.vibeqc_codegen.production import load_production_kernel_selections
from tools.vibeqc_codegen.shell_spec import FUSED_SHELL_SPEC_BY_NAME


def test_ppps_resident_benchmark_groups_contiguous_ket_tasks():
    """Keep the synthetic 1110 descriptor and independent oracle visible."""

    source = emit_ppps_resident_bra_benchmark_cuda(512, 2, 1, 3, 3)
    assert "resident_task_count" in source
    assert "GeneratedPppsResidentTask" in source
    assert "generated_ppps_resident_bra_force_rhf_kernel<<<" in source
    assert "generated_ppps_component_recompute_rhf_kernel<<<" in source
    assert "resident_bra_1110" in source
    assert "VIBEQC_TASK_COUNT" not in source
    assert "VIBEQC_FUSED_LAUNCH" not in source


def test_ppps_resident_benchmark_runs_when_nvcc_is_configured(tmp_path: Path):
    """Compile locally and schedule every real-GPU check through Slurm."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the resident GPU benchmark")
    architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_120")
    source_path = tmp_path / "generated_ppps_resident_benchmark.cu"
    executable = tmp_path / "generated_ppps_resident_benchmark"
    # Two 256-task chunks leave almost the entire RTX 5090 idle.  Use enough
    # independent resident chunks to measure steady-state throughput while
    # retaining the same per-bra primitive reuse and compact synthetic data.
    task_count = 32768
    source_path.write_text(
        emit_ppps_resident_bra_benchmark_cuda(task_count, 2, 1, 3, 3),
        encoding="utf-8",
    )
    compiled = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={architecture}",
            "-O3",
            str(source_path),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr

    environment = dict(os.environ)
    cuda_library = Path(nvcc).parent.parent / "lib64"
    previous = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(cuda_library) if not previous else f"{cuda_library}:{previous}"
    )
    run = subprocess.run(
        [
            "srun",
            "--partition=main",
            "--gres=gpu:5090:1",
            "--nodes=1",
            "--ntasks=1",
            "--time=00:03:00",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=210,
        env=environment,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    payload = json.loads(run.stdout.strip().splitlines()[-1])
    assert payload["topology"] == "resident_bra_1110"
    assert payload["maximum_force_error"] <= (
        2.0e-10 * max(1.0, payload["maximum_force"])
    )
    assert payload["fused_ms"] > 0.0

    repository_root = Path(__file__).resolve().parents[2]
    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    selection = next(
        item
        for item in load_production_kernel_selections(
            repository_root
            / "tools"
            / "vibeqc_codegen"
            / "production_shell_classes.json",
            "sm_120",
        )
        if item.spec == spec
    )
    production_plan = build_fused_shell_plan(
        spec,
        consumers=selection.consumers,
        schedule=selection.schedule,
        recurrence=selection.recurrence,
    )
    production_source = tmp_path / "generated_ppps_production_benchmark.cu"
    production_executable = tmp_path / "generated_ppps_production_benchmark"
    production_source.write_text(
        emit_shell_class_benchmark_cuda(
            spec,
            task_count=task_count,
            primitive_count=2,
            warmups=1,
            iterations=3,
            samples=3,
            plan=production_plan,
            benchmark_kernel_only=True,
            persistent_kernel=True,
        ),
        encoding="utf-8",
    )
    production_compiled = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={architecture}",
            "-O3",
            str(production_source),
            "-o",
            str(production_executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert production_compiled.returncode == 0, (
        production_compiled.stdout + production_compiled.stderr
    )
    production_run = subprocess.run(
        [
            "srun",
            "--partition=main",
            "--gres=gpu:5090:1",
            "--nodes=1",
            "--ntasks=1",
            "--time=00:03:00",
            str(production_executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=210,
        env=environment,
    )
    assert production_run.returncode == 0, production_run.stdout + production_run.stderr
    production = json.loads(production_run.stdout.strip().splitlines()[-1])
    assert production["maximum_force_error"] <= (
        2.0e-10 * max(1.0, production["maximum_force"])
    )
    comparison = {
        "resident": payload,
        "component_lanes": production,
        "speedup_vs_component_lanes": production["fused_ms"] / payload["fused_ms"],
    }
    print(json.dumps(comparison, sort_keys=True))

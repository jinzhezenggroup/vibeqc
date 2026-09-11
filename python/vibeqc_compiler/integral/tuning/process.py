"""Finite compiler invocation, runtime environment and artifact-size helpers.

These helpers report execution facts; resource rejection and candidate ranking
are separate policies and cannot be inferred from process success alone."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..batch_benchmark import parse_ptxas_resources
from ..cuda_adapter import CudaCompilerAdapter
from ..cuda_target import (
    cuda_target_info,
)
from .policy import ScheduleTrial


def _compile_trial(
    nvcc: Path,
    architecture: str,
    directory: Path,
    trial: ScheduleTrial,
    compile_timeout: float = 300,
    artifact_suffix: str = "",
) -> dict[str, object]:
    """Compile one schedule and retain diagnostics for resource gates.

    NVCC launches ``cicc`` and ``ptxas`` children.  A timeout therefore kills
    the fresh process group rather than only the NVCC parent, otherwise large
    shell trials leave orphan compiler processes consuming CPU while later
    candidates start.
    """

    stem = (
        f"{trial.spec.name}_{trial.schedule_id}{trial.integral_suffix}{artifact_suffix}"
    )
    source = directory / f"{stem}.cu"
    obj = directory / f"{stem}.o"
    compiler = CudaCompilerAdapter(
        nvcc=nvcc,
        target=cuda_target_info(architecture),
        compile_timeout=compile_timeout,
    )
    result = compiler.compile(source, obj)
    diagnostics = result.stdout + result.stderr
    marker = f"{trial.symbol_prefix}_shell_class_{trial.consumer.value}_"
    return {
        "key": trial.key,
        "object": obj,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_seconds": result.duration_seconds,
        "source_bytes": _artifact_size(source),
        "object_bytes": _artifact_size(obj),
        "diagnostics": diagnostics,
        "resources": parse_ptxas_resources(
            diagnostics,
            trial.spec.name,
            symbol_prefix=marker,
        ),
    }


def _runtime_environment(nvcc: Path) -> dict[str, str]:
    """Expose the selected CUDA runtime without changing GPU visibility."""

    environment = dict(os.environ)
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        environment.pop("CUDA_VISIBLE_DEVICES")
    library = nvcc.parent.parent / "lib64"
    previous = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(library) if not previous else f"{library}:{previous}"
    )
    return environment


def _tool_version(command: Path) -> str:
    """Return one compiler version line for tuning-artifact provenance."""

    try:
        result = subprocess.run(
            [str(command), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError:
        return ""
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines()]
    return next((line for line in reversed(lines) if line), "")


def _artifact_size(path: Path) -> int | None:
    """Return a generated artifact's byte size, or ``None`` if unavailable.

    Failed or timed-out compiler invocations do not necessarily leave an
    output file behind.  Keeping the absence explicit lets a tuning report
    distinguish a zero-byte artifact from a compile that produced nothing.
    """

    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        pass
    return None

"""Shared reproducibility helpers for executable benchmark scripts."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git_output(*arguments: str) -> str | None:
    """Return one Git value without making benchmark execution depend on Git."""

    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _distribution_version(names: Iterable[str]) -> str | None:
    """Resolve the first installed distribution name from a compatibility list."""

    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _command_output(arguments: tuple[str, ...]) -> str | None:
    """Return normalized tool output without making metadata collection fatal."""

    try:
        completed = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip() or completed.stderr.strip()
    return "\n".join(line.rstrip() for line in output.splitlines()) or None


def _cuda_tool_path(name: str) -> str | None:
    """Resolve one CUDA tool from the selected toolkit before consulting PATH."""

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate = Path(cuda_path) / "bin" / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


def _toolchain_metadata() -> dict[str, Any]:
    """Record the compilers that make a GPU benchmark reproducible."""

    tools: dict[str, Any] = {}
    for name in ("nvcc", "ptxas", "cuobjdump"):
        path = _cuda_tool_path(name)
        tools[name] = {
            "path": path,
            "version": _command_output((path, "--version")) if path else None,
        }
    cxx = os.environ.get("CXX") or shutil.which("c++")
    tools["host_cxx"] = {
        "path": cxx,
        "version": _command_output((cxx, "--version")) if cxx else None,
    }
    return tools


def _visible_nvidia_device(device_id: int) -> str:
    """Map a process-local CUDA ordinal to the scheduler-visible GPU token."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = tuple(item.strip() for item in visible.split(","))
        if device_id < len(devices) and devices[device_id]:
            return devices[device_id]
    return str(device_id)


def _nvidia_smi_state(device_id: int) -> dict[str, Any] | None:
    """Capture post-benchmark clocks, power, temperature, and performance state."""

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    fields = (
        "pstate",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks.current.memory",
        "temperature.gpu",
    )
    output = _command_output(
        (
            nvidia_smi,
            f"--id={_visible_nvidia_device(device_id)}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    if output is None:
        return None
    row = tuple(item.strip() for item in output.splitlines()[0].split(","))
    if len(row) != len(fields):
        return None

    def number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "performance_state": row[0],
        "power_draw_watts": number(row[1]),
        "power_limit_watts": number(row[2]),
        "sm_clock_mhz": number(row[3]),
        "memory_clock_mhz": number(row[4]),
        "temperature_celsius": number(row[5]),
        "sampling_point": "after benchmark measurements",
    }


def _source_status_payload(
    tracked_status: str | None,
    untracked_paths: str | None,
) -> dict[str, Any]:
    """Separate source dirtiness from newly generated result artifacts.

    A benchmark matrix commonly writes several new JSON files before they are
    committed together. Earlier files in that same matrix must not make later
    runs look as though they used modified scientific source. Tracked changes
    always count as dirty; only untracked ``benchmarks/results/*.json`` files
    are classified as pending generated evidence instead of source changes.
    """

    if tracked_status is None or untracked_paths is None:
        return {
            "dirty": None,
            "pending_generated_benchmark_artifacts": None,
        }
    untracked = [line for line in untracked_paths.splitlines() if line]
    pending_results = [
        path
        for path in untracked
        if path.startswith("benchmarks/results/") and path.endswith(".json")
    ]
    source_untracked = [path for path in untracked if path not in pending_results]
    return {
        "dirty": bool(tracked_status or source_untracked),
        "pending_generated_benchmark_artifacts": len(pending_results),
    }


def environment_metadata(
    *,
    distributions: dict[str, tuple[str, ...]] | None = None,
    accelerator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the source and runtime that produced a benchmark result.

    Benchmark numbers are useful only when they can be tied to exact source,
    dependency, and device state. Missing optional metadata is represented by
    ``null`` rather than preventing a run on minimal CPU installations.
    """

    head = _git_output("rev-parse", "HEAD")
    tracked_status = _git_output("status", "--porcelain=v1", "--untracked-files=no")
    untracked_paths = _git_output("ls-files", "--others", "--exclude-standard")
    source_status = _source_status_payload(tracked_status, untracked_paths)
    package_versions = {
        label: _distribution_version(candidates)
        for label, candidates in (distributions or {}).items()
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": head,
            **source_status,
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": package_versions,
        "runtime": {
            "vibeqc_library": os.environ.get("VIBEQC_LIBRARY"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "toolchain": _toolchain_metadata(),
        "accelerator": accelerator,
    }


def cuda_accelerator_metadata(cupy_module: Any) -> dict[str, Any]:
    """Return stable JSON fields for the CUDA device used by a benchmark."""

    device_id = cupy_module.cuda.Device().id
    properties = cupy_module.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace").rstrip("\x00")
    return {
        "backend": "cuda",
        "device_id": device_id,
        "name": name,
        "compute_capability": [
            properties.get("major"),
            properties.get("minor"),
        ],
        "total_global_memory_bytes": properties.get("totalGlobalMem"),
        "multiprocessor_count": properties.get("multiProcessorCount"),
        "clock_rate_khz": properties.get("clockRate"),
        "driver_version": cupy_module.cuda.runtime.driverGetVersion(),
        "runtime_version": cupy_module.cuda.runtime.runtimeGetVersion(),
        "nvidia_smi": _nvidia_smi_state(device_id),
    }


def benchmark_gate_failures(
    *,
    speedup: float,
    maximum_energy_error: float,
    maximum_force_error: float,
    vibeqc_converged: bool = True,
    reference_converged: bool = True,
    minimum_speedup: float | None = None,
    maximum_vibeqc_over_reference: float | None = None,
    maximum_energy_error_limit: float | None = None,
    maximum_force_error_limit: float | None = None,
) -> list[str]:
    """Return actionable failures for optional accuracy/performance gates."""

    failures = []
    if not vibeqc_converged:
        failures.append("one or more VIBEQC systems did not converge")
    if not reference_converged:
        failures.append("one or more GPU4PySCF reference systems did not converge")
    if minimum_speedup is not None and speedup < minimum_speedup:
        failures.append(f"warm speedup {speedup:.6g}x is below {minimum_speedup:.6g}x")
    if maximum_vibeqc_over_reference is not None and (
        speedup <= 0.0 or 1.0 / speedup > maximum_vibeqc_over_reference
    ):
        ratio = float("inf") if speedup <= 0.0 else 1.0 / speedup
        failures.append(
            "VibeQC/reference warm ratio "
            f"{ratio:.6g}x exceeds "
            f"{maximum_vibeqc_over_reference:.6g}x"
        )
    if (
        maximum_energy_error_limit is not None
        and maximum_energy_error > maximum_energy_error_limit
    ):
        failures.append(
            "maximum energy error "
            f"{maximum_energy_error:.6g} Eh exceeds "
            f"{maximum_energy_error_limit:.6g} Eh"
        )
    if (
        maximum_force_error_limit is not None
        and maximum_force_error > maximum_force_error_limit
    ):
        failures.append(
            "maximum force error "
            f"{maximum_force_error:.6g} Eh/bohr exceeds "
            f"{maximum_force_error_limit:.6g} Eh/bohr"
        )
    return failures


def write_result(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a stable, human-readable JSON benchmark artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination

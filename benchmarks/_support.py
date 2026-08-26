"""Shared reproducibility helpers for executable benchmark scripts."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable


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
    status = _git_output("status", "--porcelain=v1")
    package_versions = {
        label: _distribution_version(candidates)
        for label, candidates in (distributions or {}).items()
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": head,
            "dirty": None if status is None else bool(status),
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
    }


def benchmark_gate_failures(
    *,
    speedup: float,
    maximum_energy_error: float,
    maximum_force_error: float,
    vibeqc_converged: bool = True,
    reference_converged: bool = True,
    minimum_speedup: float | None = None,
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
        failures.append(
            f"warm speedup {speedup:.6g}x is below {minimum_speedup:.6g}x"
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

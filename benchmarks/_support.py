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
            "qce_library": os.environ.get("QCE_LIBRARY"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "accelerator": accelerator,
    }


def write_result(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a stable, human-readable JSON benchmark artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


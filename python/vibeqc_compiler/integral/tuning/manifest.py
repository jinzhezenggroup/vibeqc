"""Validate provenance and atomically publish accepted tuning manifest entries.

Existing promotion requirements remain authoritative; emitting or compiling
a candidate does not provide its numerical or endpoint acceptance evidence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from ..cuda_schedule import (
    ScheduleIR,
)
from ..ir import KernelConsumer
from .policy import schedule_payload
from .shared import _PROVENANCE_FIELDS


def _validated_provenance(
    name: str, metrics: Mapping[str, object]
) -> dict[str, object]:
    """Validate compiler/runtime measurements before persisting a manifest row."""

    validated: dict[str, object] = {}
    for field in _PROVENANCE_FIELDS:
        if field not in metrics:
            continue
        value = metrics[field]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                f"{name} provenance {field} must be finite and non-negative"
            )
        validated[field] = value
    return validated


def update_manifest_payload(
    payload: dict[str, object],
    architecture: str,
    winners: Mapping[str, ScheduleIR],
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
    provenance: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Return a v2 manifest with measured winners installed for one GPU.

    ``provenance`` may carry endpoint and compiler measurements for each
    winner.  Keeping those values beside the schedule lets a later promotion
    apply the compile/runtime Pareto policy instead of silently optimizing only
    a synthetic kernel timer.
    """

    if payload.get("schema_version") != 2:
        raise ValueError("autotuning requires a schema-v2 production manifest")
    architectures = payload.get("architectures")
    if not isinstance(architectures, dict):
        raise TypeError("production manifest requires an architectures object")
    if architecture not in architectures:
        raise ValueError(f"production manifest has no profile for {architecture}")
    profile = architectures.get(architecture)
    if not isinstance(profile, dict):
        raise TypeError("architecture profile must be a JSON object")
    kernels = profile.get("kernels")
    if not isinstance(kernels, list):
        raise TypeError("architecture profile requires a kernels list")

    selected_consumer = KernelConsumer(consumer)
    # Force and Fock may deliberately use different execution geometries for
    # the same shell class.  A Fock autotune must therefore update the
    # optional ``fock_schedule`` field while preserving the force schedule;
    # otherwise a batch Fock search silently retunes the force path too.
    schedule_field = (
        "fock_schedule" if selected_consumer == KernelConsumer.FOCK else "schedule"
    )
    required_consumers = (
        (KernelConsumer.FOCK, KernelConsumer.FORCE)
        if selected_consumer == KernelConsumer.FOCK
        else (KernelConsumer.FORCE,)
    )
    installed = set()
    for row in kernels:
        if not isinstance(row, dict):
            raise TypeError("production kernel entry must be a JSON object")
        name = row.get("shell_class")
        if isinstance(name, str) and name in winners:
            row[schedule_field] = schedule_payload(winners[name])
            raw_consumers = row.get("consumers", [])
            if not isinstance(raw_consumers, list):
                raise TypeError(f"{name} consumers must be a list")
            current = {KernelConsumer(item) for item in raw_consumers}
            current.update(required_consumers)
            row["consumers"] = [
                item.value for item in KernelConsumer if item in current
            ]
            if provenance is not None:
                metrics = provenance.get(name, {})
                if not isinstance(metrics, Mapping):
                    raise TypeError(f"{name} provenance must be a mapping")
                validated = _validated_provenance(name, metrics)
                for field, value in validated.items():
                    if value is not None:
                        row[field] = value
            installed.add(name)
    for name, schedule in winners.items():
        if name not in installed:
            entry = {
                "shell_class": name,
                "consumers": [item.value for item in required_consumers],
            }
            # Omitting ``schedule`` for a newly added Fock row lets the
            # production parser derive the canonical force plan, while the
            # measured winner is installed only in ``fock_schedule``.
            entry[schedule_field] = schedule_payload(schedule)
            if provenance is not None:
                metrics = provenance.get(name, {})
                if not isinstance(metrics, Mapping):
                    raise TypeError(f"{name} provenance must be a mapping")
                entry.update(_validated_provenance(name, metrics))
            kernels.append(entry)
    return payload


def write_tuned_manifest(
    input_path: Path,
    output_path: Path,
    architecture: str,
    winners: Mapping[str, ScheduleIR],
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
    provenance: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Write architecture winners without mutating the source manifest in place.

    The replacement is atomic within the destination directory.  A batch run
    can therefore be interrupted (or fail after compiling a subset of
    candidates) without leaving a truncated JSON manifest that looks
    production-ready to the next build.
    """

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    updated = update_manifest_payload(
        payload,
        architecture,
        winners,
        consumer,
        provenance,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(updated, indent=2, sort_keys=False))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass

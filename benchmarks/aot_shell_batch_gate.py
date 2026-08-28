"""Run a fixed-dm0 A/B endpoint benchmark for generated shell classes.

The baseline and candidate are executed in one prepared VibeQC batch.  A
candidate-union capacity prime happens before the measured cold baseline so a
larger candidate cannot be penalized (or fail) merely because the immutable
CUDA task arena was sized from the smaller baseline selection.  The measured
replays then use one frozen post-cold density and an interleaved ABBA/AB
sequence.  GPU4PySCF is intentionally not involved: this tool isolates the
VibeQC shell-class dispatch and its SCF/force path.

The module keeps all GPU imports inside :func:`main`; parser, dry-run, and
pure control-flow tests can therefore run on scheduler login nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

_SHELL_ENVIRONMENT = "VIBEQC_AOT_SHELL_CLASSES"
_FOCK_SHELL_ENVIRONMENT = "VIBEQC_AOT_FOCK_SHELL_CLASSES"
_RESERVED_ENVIRONMENTS = frozenset(
    {_SHELL_ENVIRONMENT, _FOCK_SHELL_ENVIRONMENT}
)
BASELINE = "baseline"
CANDIDATE = "candidate"


def _class_list(value: str) -> tuple[str, ...]:
    """Parse a non-empty, deterministic comma-separated class selection.

    Empty fields are rejected instead of silently discarded.  Silently
    accepting ``"dppp,,dpds"`` makes a typo look like a valid benchmark and
    is particularly dangerous when a zero-class selection changes the native
    fallback path.
    """

    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError(
            "shell classes must be non-empty and unique"
        )
    fields = tuple(field.strip() for field in value.split(","))
    if any(not field for field in fields) or len(set(fields)) != len(fields):
        raise argparse.ArgumentTypeError(
            "shell classes must be non-empty and unique"
        )
    return fields


def _ordered_union(*selections: Sequence[str]) -> tuple[str, ...]:
    """Return selections concatenated without changing their first order."""

    result: list[str] = []
    seen: set[str] = set()
    for selection in selections:
        for name in selection:
            if name not in seen:
                result.append(name)
                seen.add(name)
    return tuple(result)


def _parse_environment_overrides(
    values: Sequence[str] | None,
) -> dict[str, str]:
    """Parse repeated ``NAME=VALUE`` options into an ordered mapping.

    Values may contain additional ``=`` characters, which is useful for
    options whose value is itself a serialized setting.  A duplicate key on
    one side of an A/B comparison is almost certainly a typo, so reject it
    rather than letting the last occurrence silently win.  The two sides are
    parsed independently and may intentionally use the same key with
    different values.
    """

    overrides: dict[str, str] = {}
    for raw_value in values or ():
        if not isinstance(raw_value, str) or "=" not in raw_value:
            raise ValueError(
                "environment overrides must use NAME=VALUE syntax"
            )
        name, value = raw_value.split("=", 1)
        if not name:
            raise ValueError("environment override names must be non-empty")
        if "=" in name:
            raise ValueError("environment override names cannot contain '='")
        if "\x00" in name or "\x00" in value:
            raise ValueError("environment overrides cannot contain NUL bytes")
        if name in _RESERVED_ENVIRONMENTS:
            raise ValueError(
                f"environment override {name!r} is reserved for shell selection"
            )
        if name in overrides:
            raise ValueError(f"duplicate environment override {name!r}")
        overrides[name] = value
    return overrides


def _validated_environment_overrides(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate a programmatic override mapping before touching ``os.environ``."""

    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError("environment overrides must be a mapping")
    validated: dict[str, str] = {}
    for name, value in overrides.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError(
                "environment override names and values must be strings"
            )
        if not name:
            raise ValueError("environment override names must be non-empty")
        if "=" in name:
            raise ValueError("environment override names cannot contain '='")
        if "\x00" in name or "\x00" in value:
            raise ValueError("environment overrides cannot contain NUL bytes")
        if name in _RESERVED_ENVIRONMENTS:
            raise ValueError(
                f"environment override {name!r} is reserved for shell selection"
            )
        validated[name] = value
    return validated


def _argument_environment_overrides(
    arguments: argparse.Namespace, side: str
) -> dict[str, str]:
    """Return normalized overrides from a parser namespace.

    ``_dry_run_payload`` is also used directly by pure-Python tests and small
    inspection tools, so tolerate a namespace that has not gone through
    ``_validate_arguments`` yet.
    """

    attribute = f"{side}_environment_overrides"
    if hasattr(arguments, attribute):
        value = getattr(arguments, attribute)
        return {} if value is None else dict(value)
    return _parse_environment_overrides(
        getattr(arguments, f"{side}_env", ())
    )


def interleaved_selection_order(
    repeats: int, style: str = "abba"
) -> tuple[str, ...]:
    """Return exactly ``repeats`` baseline and candidate labels.

    ``abba`` uses balanced four-sample blocks and is the default because it
    places each selection on both sides of its counterpart.  ``ab`` is useful
    when a profiler needs a strictly alternating stream.  A short final block
    is truncated without ever changing the requested sample count.
    """

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if style not in {"abba", "ab"}:
        raise ValueError("selection order must be 'abba' or 'ab'")
    block = (BASELINE, CANDIDATE, CANDIDATE, BASELINE)
    if style == "ab":
        block = (BASELINE, CANDIDATE)
    counts = {BASELINE: 0, CANDIDATE: 0}
    order: list[str] = []
    while counts[BASELINE] < repeats or counts[CANDIDATE] < repeats:
        for label in block:
            if counts[label] >= repeats:
                continue
            order.append(label)
            counts[label] += 1
    return tuple(order)


@contextmanager
def _aot_selection(
    shell_classes: Sequence[str],
    fock_classes: Sequence[str] | None = None,
    environment_overrides: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Install one deterministic registry selection and restore caller state.

    An omitted Fock selection means the registry default (all available Fock
    classes), not an ambient selection inherited from the invoking shell.
    Clearing that variable makes benchmark artifacts reproducible while the
    context manager still restores the caller's value afterwards.
    """

    overrides = _validated_environment_overrides(environment_overrides)
    values = {
        _SHELL_ENVIRONMENT: ",".join(shell_classes),
        _FOCK_SHELL_ENVIRONMENT: (
            None if fock_classes is None else ",".join(fock_classes)
        ),
    }
    values.update(overrides)
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _synchronize(cupy_module: Any) -> None:
    """Synchronize the stream used by the native VibeQC CUDA plan."""

    cupy_module.cuda.Stream.null.synchronize()


def _result_payload(result: Any) -> dict[str, Any]:
    """Serialize all diagnostics needed to identify an SCF branch."""

    items = []
    for item in result.items:
        items.append(
            {
                "converged": bool(item.converged),
                "iterations": int(item.iterations),
                "energy_change_hartree": float(item.energy_change),
                "density_rms": float(item.density_rms),
                "warm_start_used": bool(item.warm_start_used),
                "warm_start_fallback": bool(item.warm_start_fallback),
                "executed_backend": item.executed_backend,
                "bucket_id": int(item.bucket_id),
            }
        )
    energies = getattr(result, "energies", None)
    if energies is None:
        energies = [item.energy for item in result.items]
    return {
        "energies_hartree": np.asarray(energies, dtype=np.float64).tolist(),
        "forces_hartree_per_bohr": [
            None
            if item.forces is None
            else np.asarray(item.forces, dtype=np.float64).tolist()
            for item in result.items
        ],
        "convergence": items,
        "iteration_branches": [item["iterations"] for item in items],
    }


def _execute_once(
    batch: Any,
    cupy_module: Any,
    shell_classes: Sequence[str],
    *,
    fock_classes: Sequence[str] | None = None,
    environment_overrides: Mapping[str, str] | None = None,
) -> tuple[Any, float]:
    """Execute one synchronized replay under an exact registry selection."""

    with _aot_selection(
        shell_classes,
        fock_classes,
        environment_overrides=environment_overrides,
    ):
        _synchronize(cupy_module)
        start = time.perf_counter()
        result = batch.execute(strict=True)
        _synchronize(cupy_module)
        elapsed = time.perf_counter() - start
    return result, elapsed


def _execute(
    batch: Any,
    cupy_module: Any,
    selection: tuple[str, ...],
    repeats: int,
    *,
    environment_overrides: Mapping[str, str] | None = None,
) -> tuple[Any, list[float]]:
    """Compatibility helper for a contiguous selection replay.

    New measurements use :func:`_alternating_replays`; this helper remains
    useful for quick local probes and preserves the old script's small API.
    """

    if repeats < 1:
        raise ValueError("repeats must be positive")
    timings = []
    result = None
    for _ in range(repeats):
        result, elapsed = _execute_once(
            batch,
            cupy_module,
            selection,
            environment_overrides=environment_overrides,
        )
        timings.append(elapsed)
    assert result is not None
    return result, timings


def _accuracy(reference: Any, candidate: Any) -> tuple[float, float]:
    """Return maximum energy and Cartesian force differences."""

    reference_forces = [item.forces for item in reference.items]
    candidate_forces = [item.forces for item in candidate.items]
    if any(force is None for force in (*reference_forces, *candidate_forces)):
        raise ValueError("accuracy comparison requires successful force results")
    return (
        float(
            np.max(
                np.abs(
                    np.asarray(reference.energies, dtype=np.float64)
                    - np.asarray(candidate.energies, dtype=np.float64)
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    np.asarray(reference_forces, dtype=np.float64)
                    - np.asarray(candidate_forces, dtype=np.float64)
                )
            )
        ),
    )


def iteration_branch(sample: dict[str, Any]) -> tuple[int, ...]:
    """Return the per-system iteration tuple for one serialized sample."""

    return tuple(int(value) for value in sample["iteration_branches"])


def pairwise_accuracy(
    baseline_samples: Sequence[dict[str, Any]],
    candidate_samples: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair ordinal A/B replays and retain both accuracy and branch parity."""

    if len(baseline_samples) != len(candidate_samples):
        raise ValueError("baseline and candidate sample counts must match")
    pairs = []
    for repeat, (baseline, candidate) in enumerate(
        zip(baseline_samples, candidate_samples, strict=True)
    ):
        baseline_energies = np.asarray(
            baseline["energies_hartree"], dtype=np.float64
        )
        candidate_energies = np.asarray(
            candidate["energies_hartree"], dtype=np.float64
        )
        baseline_forces = baseline["forces_hartree_per_bohr"]
        candidate_forces = candidate["forces_hartree_per_bohr"]
        if any(force is None for force in (*baseline_forces, *candidate_forces)):
            raise ValueError("accuracy comparison requires successful force results")
        pairs.append(
            {
                "repeat": repeat,
                "iteration_branches_match": (
                    iteration_branch(baseline) == iteration_branch(candidate)
                ),
                "baseline_iteration_branch": list(iteration_branch(baseline)),
                "candidate_iteration_branch": list(iteration_branch(candidate)),
                "maximum_energy_error_hartree": float(
                    np.max(np.abs(baseline_energies - candidate_energies))
                ),
                "maximum_force_error_hartree_per_bohr": float(
                    np.max(
                        np.abs(
                            np.asarray(baseline_forces, dtype=np.float64)
                            - np.asarray(candidate_forces, dtype=np.float64)
                        )
                    )
                ),
            }
        )
    return pairs


def timing_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw synchronized samples without discarding any sample."""

    seconds = [float(sample["seconds"]) for sample in samples]
    if not seconds:
        raise ValueError("at least one timing sample is required")
    return {
        "samples": len(seconds),
        "median_seconds": float(statistics.median(seconds)),
        "minimum_seconds": float(min(seconds)),
        "maximum_seconds": float(max(seconds)),
        "raw_seconds": seconds,
    }


def _sample(
    batch: Any,
    cupy_module: Any,
    label: str,
    shell_classes: Sequence[str],
    sequence_index: int,
    *,
    fock_classes: Sequence[str] | None = None,
    environment_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run and serialize one replay while keeping the result for A/B pairing."""

    result, seconds = _execute_once(
        batch,
        cupy_module,
        shell_classes,
        fock_classes=fock_classes,
        environment_overrides=environment_overrides,
    )
    payload = _result_payload(result)
    return {
        "sequence_index": sequence_index,
        "selection": label,
        "shell_classes": list(shell_classes),
        "fock_classes": None if fock_classes is None else list(fock_classes),
        "environment_overrides": dict(environment_overrides or {}),
        "seconds": float(seconds),
        **payload,
    }


def _alternating_replays(
    batch: Any,
    cupy_module: Any,
    baseline_classes: tuple[str, ...],
    candidate_classes: tuple[str, ...],
    repeats: int,
    *,
    order_style: str = "abba",
    baseline_fock_classes: tuple[str, ...] | None = None,
    candidate_fock_classes: tuple[str, ...] | None = None,
    baseline_environment_overrides: Mapping[str, str] | None = None,
    candidate_environment_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Collect fixed-dm0 A/B samples in a deterministic interleaved order."""

    order = interleaved_selection_order(repeats, order_style)
    samples = []
    for sequence_index, label in enumerate(order):
        if label == BASELINE:
            classes = baseline_classes
            fock_classes = baseline_fock_classes
            environment_overrides = baseline_environment_overrides
        else:
            classes = candidate_classes
            fock_classes = candidate_fock_classes
            environment_overrides = candidate_environment_overrides
        samples.append(
            _sample(
                batch,
                cupy_module,
                label,
                classes,
                sequence_index,
                fock_classes=fock_classes,
                environment_overrides=environment_overrides,
            )
        )
    baseline_samples = [
        sample for sample in samples if sample["selection"] == BASELINE
    ]
    candidate_samples = [
        sample for sample in samples if sample["selection"] == CANDIDATE
    ]
    pairs = pairwise_accuracy(baseline_samples, candidate_samples)
    baseline_timing = timing_summary(baseline_samples)
    candidate_timing = timing_summary(candidate_samples)
    baseline_median = baseline_timing["median_seconds"]
    candidate_median = candidate_timing["median_seconds"]
    speedup = float(baseline_median / candidate_median)
    return {
        "measurement_order": list(order),
        "raw_samples": samples,
        "baseline_samples": baseline_samples,
        "candidate_samples": candidate_samples,
        "iteration_branches": {
            "baseline": [
                list(iteration_branch(sample)) for sample in baseline_samples
            ],
            "candidate": [
                list(iteration_branch(sample)) for sample in candidate_samples
            ],
        },
        "timing_summary": {
            "baseline": baseline_timing,
            "candidate": candidate_timing,
            "speedup": speedup,
        },
        "pairwise_accuracy": pairs,
    }


def _gate_measurement(
    measurement: dict[str, Any],
    *,
    maximum_energy_error: float,
    maximum_force_error: float,
    minimum_speedup: float,
) -> dict[str, Any]:
    """Attach explicit accuracy, convergence, and performance gate fields."""

    pairs = measurement["pairwise_accuracy"]
    maximum_energy = max(
        pair["maximum_energy_error_hartree"] for pair in pairs
    )
    maximum_force = max(
        pair["maximum_force_error_hartree_per_bohr"] for pair in pairs
    )
    baseline_converged = all(
        item["converged"]
        for sample in measurement["baseline_samples"]
        for item in sample["convergence"]
    )
    candidate_converged = all(
        item["converged"]
        for sample in measurement["candidate_samples"]
        for item in sample["convergence"]
    )
    iteration_branches_match = all(
        pair["iteration_branches_match"] for pair in pairs
    )
    speedup = measurement["timing_summary"]["speedup"]
    failures = []
    if not baseline_converged:
        failures.append("baseline SCF convergence")
    if not candidate_converged:
        failures.append("candidate SCF convergence")
    if not iteration_branches_match:
        failures.append("SCF iteration branch parity")
    if maximum_energy > maximum_energy_error:
        failures.append("energy accuracy")
    if maximum_force > maximum_force_error:
        failures.append("force accuracy")
    if speedup < minimum_speedup:
        failures.append("performance")
    measurement["accuracy"] = {
        "maximum_energy_error_hartree": maximum_energy,
        "maximum_force_error_hartree_per_bohr": maximum_force,
        "pair_count": len(pairs),
    }
    measurement["convergence"] = {
        "baseline_all_converged": baseline_converged,
        "candidate_all_converged": candidate_converged,
        "all_iteration_branches_match": iteration_branches_match,
    }
    measurement["gate"] = {
        "minimum_speedup": minimum_speedup,
        "maximum_energy_error_hartree": maximum_energy_error,
        "maximum_force_error_hartree_per_bohr": maximum_force_error,
        "passed": not failures,
        "failures": failures,
    }
    # Keep a compact compatibility view for scripts that consumed the old
    # gate's flat fields, while the structured fields above retain all pairs.
    measurement["speedup"] = speedup
    measurement["maximum_energy_error_hartree"] = maximum_energy
    measurement["maximum_force_error_hartree_per_bohr"] = maximum_force
    measurement["baseline_seconds"] = measurement["timing_summary"][
        "baseline"
    ]["raw_seconds"]
    measurement["candidate_seconds"] = measurement["timing_summary"][
        "candidate"
    ]["raw_seconds"]
    measurement["baseline_median_seconds"] = measurement["timing_summary"][
        "baseline"
    ]["median_seconds"]
    measurement["candidate_median_seconds"] = measurement["timing_summary"][
        "candidate"
    ]["median_seconds"]
    measurement["iterations"] = measurement["iteration_branches"]["candidate"][-1]
    measurement["classes"] = measurement["candidate_samples"][-1][
        "shell_classes"
    ]
    measurement["passed"] = not failures
    measurement["failures"] = failures
    return measurement


def _cold_baseline_and_freeze(
    batch: Any,
    cupy_module: Any,
    baseline_classes: tuple[str, ...],
    *,
    baseline_fock_classes: tuple[str, ...] | None = None,
    baseline_environment_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute exactly one measured cold baseline and freeze its post-cold dm0."""

    result, seconds = _execute_once(
        batch,
        cupy_module,
        baseline_classes,
        fock_classes=baseline_fock_classes,
        environment_overrides=baseline_environment_overrides,
    )
    payload = _result_payload(result)
    batch.set_warm_start_updates(False)
    return {
        "seconds": float(seconds),
        "shell_classes": list(baseline_classes),
        "fock_classes": (
            None
            if baseline_fock_classes is None
            else list(baseline_fock_classes)
        ),
        "environment_overrides": dict(baseline_environment_overrides or {}),
        "warm_start_updates_after_run": False,
        **payload,
    }


def _fixed_dm0_measurement(
    batch: Any,
    cupy_module: Any,
    baseline_classes: tuple[str, ...],
    candidate_classes: tuple[str, ...],
    repeats: int,
    *,
    order_style: str,
    warmups: int,
    baseline_fock_classes: tuple[str, ...] | None = None,
    candidate_fock_classes: tuple[str, ...] | None = None,
    baseline_environment_overrides: Mapping[str, str] | None = None,
    candidate_environment_overrides: Mapping[str, str] | None = None,
    maximum_energy_error: float,
    maximum_force_error: float,
    minimum_speedup: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze one cold dm0, then return warmups and a gated A/B measurement."""

    cold = _cold_baseline_and_freeze(
        batch,
        cupy_module,
        baseline_classes,
        baseline_fock_classes=baseline_fock_classes,
        baseline_environment_overrides=baseline_environment_overrides,
    )
    warmup_samples = []
    for index in range(warmups):
        warmup_samples.append(
            _sample(
                batch,
                cupy_module,
                "warmup",
                baseline_classes,
                index,
                fock_classes=baseline_fock_classes,
                environment_overrides=baseline_environment_overrides,
            )
        )
    measurement = _alternating_replays(
        batch,
        cupy_module,
        baseline_classes,
        candidate_classes,
        repeats,
        order_style=order_style,
        baseline_fock_classes=baseline_fock_classes,
        candidate_fock_classes=candidate_fock_classes,
        baseline_environment_overrides=baseline_environment_overrides,
        candidate_environment_overrides=candidate_environment_overrides,
    )
    _gate_measurement(
        measurement,
        maximum_energy_error=maximum_energy_error,
        maximum_force_error=maximum_force_error,
        minimum_speedup=minimum_speedup,
    )
    measurement["cold_baseline"] = cold
    measurement["warmups"] = warmup_samples
    measurement["fixed_dm0"] = {
        "enabled": True,
        "source": "measured baseline cold result",
        "warm_start_updates": False,
    }
    return measurement, warmup_samples


def _measurement(
    batch: Any,
    cupy_module: Any,
    baseline_classes: tuple[str, ...],
    candidate_classes: tuple[str, ...],
    repeats: int,
    maximum_energy_error: float,
    maximum_force_error: float,
    minimum_speedup: float,
    *,
    baseline_environment_overrides: Mapping[str, str] | None = None,
    candidate_environment_overrides: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run the fixed-dm0 measurement with the public legacy signature."""

    measurement, _ = _fixed_dm0_measurement(
        batch,
        cupy_module,
        baseline_classes,
        candidate_classes,
        repeats,
        order_style="abba",
        warmups=0,
        baseline_environment_overrides=baseline_environment_overrides,
        candidate_environment_overrides=candidate_environment_overrides,
        maximum_energy_error=maximum_energy_error,
        maximum_force_error=maximum_force_error,
        minimum_speedup=minimum_speedup,
    )
    return measurement


def _bisect_regression(
    batch: Any,
    cupy_module: Any,
    baseline: tuple[str, ...],
    extras: tuple[str, ...],
    arguments: argparse.Namespace,
    measurements: list[dict[str, object]],
) -> list[list[str]]:
    """Find failing shell-class groups while preserving the frozen dm0."""

    if not extras:
        return []
    selection = _ordered_union(baseline, extras)
    measurement = _alternating_replays(
        batch,
        cupy_module,
        baseline,
        selection,
        arguments.bisection_repeats,
        order_style=arguments.order,
        baseline_fock_classes=getattr(arguments, "baseline_fock_classes", None),
        candidate_fock_classes=getattr(arguments, "candidate_fock_classes", None),
        baseline_environment_overrides=getattr(
            arguments, "baseline_environment_overrides", {}
        ),
        candidate_environment_overrides=getattr(
            arguments, "candidate_environment_overrides", {}
        ),
    )
    _gate_measurement(
        measurement,
        maximum_energy_error=arguments.maximum_energy_error,
        maximum_force_error=arguments.maximum_force_error,
        minimum_speedup=arguments.minimum_speedup,
    )
    measurement["bisection"] = True
    measurement["candidate_classes"] = list(selection)
    measurements.append(measurement)
    if measurement["passed"]:
        return []
    if len(extras) == 1:
        return [list(extras)]
    middle = len(extras) // 2
    first = extras[:middle]
    second = extras[middle:]
    first_failures = _bisect_regression(
        batch, cupy_module, baseline, first, arguments, measurements
    )
    second_failures = _bisect_regression(
        batch, cupy_module, baseline, second, arguments, measurements
    )
    if not first_failures and not second_failures:
        # Neither half regresses independently, so retain the interaction
        # group instead of blaming an exact class without evidence.
        return [list(extras)]
    return [*first_failures, *second_failures]


def _selection_payload(
    shell_classes: tuple[str, ...],
    fock_classes: tuple[str, ...] | None,
    environment_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a stable JSON representation of a runtime registry selection."""

    return {
        "shell_classes": list(shell_classes),
        "fock_classes": None if fock_classes is None else list(fock_classes),
        "fock_environment": (
            "default-all" if fock_classes is None else "explicit"
        ),
        "environment_overrides": dict(environment_overrides or {}),
    }


def _capacity_fock_selection(
    baseline: tuple[str, ...] | None,
    candidate: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    """Return a capacity-prime Fock selection covering both A/B paths.

    ``None`` selects every registered Fock class and is therefore the safe
    superset whenever either side uses the default registry.  Two explicit
    selections can use their smaller ordered union.
    """

    if baseline is None or candidate is None:
        return None
    return _ordered_union(baseline, candidate)


def _dry_run_payload(arguments: argparse.Namespace) -> dict[str, Any]:
    """Build a no-import plan description for login-node inspection."""

    batches = tuple(arguments.batch or (1, 4))
    return {
        "schema_version": 2,
        "benchmark": "aot_shell_batch_gate",
        "protocol": "fixed_dm0_interleaved_ab",
        "case": arguments.case,
        "batches": list(batches),
        "baseline_selection": _selection_payload(
            arguments.baseline_classes,
            arguments.baseline_fock_classes,
            _argument_environment_overrides(arguments, "baseline"),
        ),
        "candidate_selection": _selection_payload(
            arguments.candidate_classes,
            arguments.candidate_fock_classes,
            _argument_environment_overrides(arguments, "candidate"),
        ),
        "measurement_order": list(
            interleaved_selection_order(arguments.repeats, arguments.order)
        ),
        "settings": {
            "warmups": arguments.warmups,
            "repeats": arguments.repeats,
            "bisection_repeats": arguments.bisection_repeats,
            "order": arguments.order,
        },
    }


def _parser() -> argparse.ArgumentParser:
    """Construct the CLI parser without importing benchmark or GPU modules."""

    parser = argparse.ArgumentParser(
        description=(
            "fixed-dm0 interleaved A/B gate for VibeQC generated shell classes"
        )
    )
    parser.add_argument(
        "--case", default="water-tetramer-def2-svp-spherical"
    )
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument(
        "--baseline-classes",
        type=_class_list,
        default=("dppp", "dpds"),
    )
    parser.add_argument(
        "--candidate-classes",
        type=_class_list,
        default=("dppp", "dpds", "ppps", "dpps", "dsps", "dspp"),
    )
    parser.add_argument("--baseline-fock-classes", type=_class_list)
    parser.add_argument("--candidate-fock-classes", type=_class_list)
    parser.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="temporarily set a runtime environment variable for baseline samples",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="temporarily set a runtime environment variable for candidate samples",
    )
    parser.add_argument(
        "--order",
        choices=("abba", "ab"),
        default="abba",
        help="warm replay order; ABBA balances endpoint placement",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bisection-repeats", type=int, default=3)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    parser.add_argument("--maximum-energy-error", type=float, default=1.0e-10)
    parser.add_argument("--maximum-force-error", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--screening-tolerance", type=float, default=1.0e-14)
    parser.add_argument(
        "--shell-class-profiling",
        action="store_true",
        help="collect the final screened shell-class profile after each batch",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the measurement plan without importing GPU packages",
    )
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> None:
    """Reject invalid numerical settings before any CUDA import or allocation."""

    batches = tuple(arguments.batch or (1, 4))
    if any(value < 1 for value in batches):
        parser.error("--batch values must be positive")
    if arguments.warmups < 0:
        parser.error("--warmups must be non-negative")
    if arguments.repeats < 1 or arguments.bisection_repeats < 1:
        parser.error("repeat counts must be positive")
    if arguments.minimum_speedup <= 0.0:
        parser.error("--minimum-speedup must be positive")
    if arguments.maximum_energy_error < 0.0:
        parser.error("--maximum-energy-error must be non-negative")
    if arguments.maximum_force_error < 0.0:
        parser.error("--maximum-force-error must be non-negative")
    if arguments.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    for name in (
        "energy_tolerance",
        "density_tolerance",
        "screening_tolerance",
    ):
        if getattr(arguments, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        arguments.baseline_environment_overrides = _parse_environment_overrides(
            getattr(arguments, "baseline_env", ())
        )
        arguments.candidate_environment_overrides = _parse_environment_overrides(
            getattr(arguments, "candidate_env", ())
        )
    except ValueError as error:
        parser.error(str(error))


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    _validate_arguments(parser, arguments)
    if arguments.dry_run:
        payload = _dry_run_payload(arguments)
        output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if arguments.output is None:
            print(output, end="")
        else:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(output, encoding="utf-8")
            print(f"JSON result: {arguments.output}")
        return

    # Import runtime/GPU dependencies only after parsing and dry-run handling.
    import cupy as cp
    from _cases import benchmark_cases
    from _support import cuda_accelerator_metadata, environment_metadata
    from compare_gpu4pyscf_batch import scaled_geometries
    from vibeqc import Calculator

    cases = benchmark_cases()
    if arguments.case not in cases:
        parser.error(
            f"unknown --case {arguments.case!r}; choose from {', '.join(sorted(cases))}"
        )
    case = cases[arguments.case]
    batches = tuple(arguments.batch or (1, 4))
    payload: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "aot_shell_batch_gate",
        "protocol": "fixed_dm0_interleaved_ab",
        "case": arguments.case,
        "baseline_selection": _selection_payload(
            arguments.baseline_classes,
            arguments.baseline_fock_classes,
            arguments.baseline_environment_overrides,
        ),
        "candidate_selection": _selection_payload(
            arguments.candidate_classes,
            arguments.candidate_fock_classes,
            arguments.candidate_environment_overrides,
        ),
        "settings": {
            "warmups": arguments.warmups,
            "repeats": arguments.repeats,
            "bisection_repeats": arguments.bisection_repeats,
            "order": arguments.order,
            "max_iterations": arguments.max_iterations,
            "energy_tolerance": arguments.energy_tolerance,
            "density_tolerance": arguments.density_tolerance,
            "screening_tolerance": arguments.screening_tolerance,
            "shell_class_profiling": arguments.shell_class_profiling,
        },
        "gates": {
            "minimum_speedup": arguments.minimum_speedup,
            "maximum_energy_error_hartree": arguments.maximum_energy_error,
            "maximum_force_error_hartree_per_bohr": arguments.maximum_force_error,
        },
        "batches": [],
    }
    gate_passed = True
    union_classes = _ordered_union(
        arguments.baseline_classes, arguments.candidate_classes
    )
    union_fock_classes = _capacity_fock_selection(
        arguments.baseline_fock_classes,
        arguments.candidate_fock_classes,
    )

    for batch_size in batches:
        systems = scaled_geometries(case.atoms, batch_size)
        calculator = Calculator(
            method=case.method,
            basis=case.vibeqc_basis,
            basis_representation=case.basis_representation,
            device="cuda",
            max_iterations=arguments.max_iterations,
            energy_tolerance=arguments.energy_tolerance,
            density_tolerance=arguments.density_tolerance,
            screening_tolerance=arguments.screening_tolerance,
        )
        measurements: list[dict[str, Any]] = []
        with calculator.prepare_batch(
            systems,
            charges=[case.charge] * batch_size,
            multiplicities=[case.multiplicity] * batch_size,
            warm_start=True,
            shell_class_profiling=arguments.shell_class_profiling,
        ) as prepared:
            # The first CUDA execution sizes the immutable generated-task arena
            # from the active registry.  Prime with the complete A/B union,
            # then clear its density so the measured baseline is genuinely cold.
            capacity_prime, prime_seconds = _execute_once(
                prepared,
                cp,
                union_classes,
                fock_classes=union_fock_classes,
                # Prime with the candidate runtime too: a candidate-only
                # dispatch path may allocate a larger task arena than the
                # baseline even when both sides share the same shell union.
                environment_overrides=arguments.candidate_environment_overrides,
            )
            prepared.clear_warm_starts()
            capacity_payload = {
                "seconds": float(prime_seconds),
                "selection": "capacity_prime",
                "shell_classes": list(union_classes),
                "fock_classes": (
                    None
                    if union_fock_classes is None
                    else list(union_fock_classes)
                ),
                "environment_overrides": dict(
                    arguments.candidate_environment_overrides
                ),
                **_result_payload(capacity_prime),
            }
            measurement, _ = _fixed_dm0_measurement(
                prepared,
                cp,
                arguments.baseline_classes,
                arguments.candidate_classes,
                arguments.repeats,
                order_style=arguments.order,
                warmups=arguments.warmups,
                baseline_fock_classes=arguments.baseline_fock_classes,
                candidate_fock_classes=arguments.candidate_fock_classes,
                baseline_environment_overrides=(
                    arguments.baseline_environment_overrides
                ),
                candidate_environment_overrides=(
                    arguments.candidate_environment_overrides
                ),
                maximum_energy_error=arguments.maximum_energy_error,
                maximum_force_error=arguments.maximum_force_error,
                minimum_speedup=arguments.minimum_speedup,
            )
            measurement["capacity_prime"] = capacity_payload
            measurements.append(measurement)

            regression_groups: list[list[str]] = []
            if not measurement["passed"]:
                extras = tuple(
                    name
                    for name in arguments.candidate_classes
                    if name not in arguments.baseline_classes
                )
                regression_groups = _bisect_regression(
                    prepared,
                    cp,
                    arguments.baseline_classes,
                    extras,
                    arguments,
                    measurements,
                )
            profile = None
            if arguments.shell_class_profiling:
                profile = [
                    {
                        "class": entry.label,
                        "shell_angular": list(entry.shell_angular),
                        "shell_quartets": entry.shell_quartets,
                        "tiles": entry.tiles,
                        "ao_quartets": entry.ao_quartets,
                        "primitive_quartets": entry.primitive_quartets,
                    }
                    for entry in prepared.last_shell_class_profile()
                    if entry.shell_quartets or entry.tiles or entry.ao_quartets
                ]
        gate_passed = gate_passed and bool(measurement["passed"])
        payload["batches"].append(
            {
                "batch_size": batch_size,
                "full_candidate": measurement,
                "regression_groups": regression_groups,
                "measurements": measurements,
                "active_shell_class_profile": profile,
            }
        )

    payload["environment"] = environment_metadata(
        distributions={
            "numpy": ("numpy",),
            "cupy": ("cupy-cuda12x", "cupy"),
        },
        accelerator=cuda_accelerator_metadata(cp),
    )
    payload["passed"] = gate_passed
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(output, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output, encoding="utf-8")
        print(f"JSON result: {arguments.output}")
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

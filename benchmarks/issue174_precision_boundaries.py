"""Measure #174 fixed-density Fock and complete CUDA output boundaries.

Run this script inside a Slurm GPU allocation. One prepared batch converges an
FP64 density, freezes warm-state updates, and replays exactly that density for
the FP64, automatic, and broad experimental-FP32 Fock variants. Complete cold
SCF measurements use the public energy-only and energy-plus-force endpoints;
profiling switches are disabled for those clean wall-time samples.

The experimental threshold is a diagnostic request, not a product policy or
an accuracy certificate. Sensitive and unsupported work remains FP64, and the
JSON record reports actual FP32/FP64 tile counts emitted by the native profiler.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    # Direct ``python benchmarks/...`` execution otherwise exposes only the
    # benchmarks directory, not its namespace-package parent.
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_MIXED_KEY = "VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD"
_DIAGNOSTIC_ENVIRONMENT = {
    "VIBEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC": "1",
    "VIBEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE": "1",
    "VIBEQC_BOUNDED_DIRECT_STREAMING": "force",
}
_CLASS_HEADER_PATTERN = re.compile(
    r"^bounded-direct-fock-class-profile "
    r"total_gpu_ms=(?P<gpu_ms>[0-9.eE+-]+) classes=(?P<classes>\d+)$"
)
_CLASS_PATTERN = re.compile(
    r"^\s+(?P<name>\S+)\s+class=(?P<shell_class>\d+)\s+"
    r"launches=(?P<launches>\d+)\s+gpu_ms=(?P<gpu_ms>[0-9.eE+-]+)\s+"
    r"share=(?P<share>[0-9.eE+-]+)%$"
)
_PRECISION_HEADER_PATTERN = re.compile(
    r"^bounded-direct-fock-precision-profile enabled=(?P<enabled>[01]) "
    r"threshold=(?P<threshold>[0-9.eE+-]+)$"
)
_PRECISION_WORK_PATTERN = re.compile(
    r"^\s+(?P<name>\S+)\s+class=(?P<shell_class>\d+)\s+"
    r"fp64_quartets=(?P<fp64>\d+)\s+fp32_quartets=(?P<fp32>\d+)\s+"
    r"mixed_capable=(?P<mixed_capable>[01])$"
)


@contextmanager
def _environment(overrides: dict[str, str | None]) -> Iterator[None]:
    """Apply one process-local runtime policy and restore it on every exit."""

    previous = {name: os.environ.get(name) for name in overrides}
    for name, value in overrides.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _capture_native_stderr(function: Callable[[], _T]) -> tuple[_T, str]:
    """Capture C/C++ fd-2 diagnostics without redirecting benchmark stdout."""

    sys.stderr.flush()
    saved_stderr = os.dup(2)
    try:
        with tempfile.TemporaryFile(mode="w+b") as stream:
            os.dup2(stream.fileno(), 2)
            try:
                result = function()
            finally:
                sys.stderr.flush()
                os.dup2(saved_stderr, 2)
            stream.seek(0)
            diagnostic = stream.read().decode("utf-8", errors="replace")
    finally:
        os.close(saved_stderr)
    return result, diagnostic


def _parse_fock_profile(diagnostic: str) -> dict[str, Any]:
    """Convert each native operator evaluation into typed JSON fields."""

    evaluations: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in diagnostic.splitlines():
        match = _CLASS_HEADER_PATTERN.match(line)
        if match:
            if current is not None:
                evaluations.append(current)
            current = {
                "class_gpu_milliseconds": float(match.group("gpu_ms")),
                "reported_class_count": int(match.group("classes")),
                "classes": [],
                "precision": None,
            }
            continue
        match = _CLASS_PATTERN.match(line)
        if match and current is not None:
            current["classes"].append(
                {
                    "name": match.group("name"),
                    "shell_class": int(match.group("shell_class")),
                    "launches": int(match.group("launches")),
                    "gpu_milliseconds": float(match.group("gpu_ms")),
                    "timed_share_percent": float(match.group("share")),
                }
            )
            continue
        match = _PRECISION_HEADER_PATTERN.match(line)
        if match and current is not None:
            current["precision"] = {
                "enabled": bool(int(match.group("enabled"))),
                "threshold": float(match.group("threshold")),
                "shell_classes": [],
            }
            continue
        match = _PRECISION_WORK_PATTERN.match(line)
        if match and current is not None and current["precision"] is not None:
            current["precision"]["shell_classes"].append(
                {
                    "name": match.group("name"),
                    "shell_class": int(match.group("shell_class")),
                    "fp64_quartets": int(match.group("fp64")),
                    "fp32_quartets": int(match.group("fp32")),
                    "mixed_capable": bool(int(match.group("mixed_capable"))),
                }
            )
    if current is not None:
        evaluations.append(current)
    if not evaluations or any(item["precision"] is None for item in evaluations):
        raise RuntimeError("native Fock diagnostic omitted precision work counts")
    for item in evaluations:
        if len(item["classes"]) != item["reported_class_count"]:
            raise RuntimeError(
                "native Fock diagnostic reported an incomplete class profile"
            )
    return {
        "operator_evaluation_count": len(evaluations),
        "operator_evaluations": evaluations,
    }


def _calculator(case: Any, arguments: argparse.Namespace) -> Any:
    """Construct one calculator with controls shared by every measurement."""

    from vibeqc import Calculator

    return Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=arguments.max_iterations,
        energy_tolerance=arguments.energy_tolerance,
        density_tolerance=arguments.density_tolerance,
        screening_tolerance=arguments.screening_tolerance,
    )


def _mode_environment(
    selection: str | None, *, profiling: bool
) -> dict[str, str | None]:
    """Resolve one arithmetic mode while clearing unrelated profile switches."""

    environment: dict[str, str | None] = {
        _MIXED_KEY: selection,
        **{name: None for name in _DIAGNOSTIC_ENVIRONMENT},
    }
    if profiling:
        environment.update(_DIAGNOSTIC_ENVIRONMENT)
    return environment


def _validate_fixed_density_sample(item: Any, diagnostic: str) -> dict[str, Any]:
    """Reject a cold retry or stale plan instead of labeling it fixed-density work."""

    if (
        item.status_message != "SCF did not converge"
        or not item.warm_start_used
        or item.warm_start_fallback
    ):
        raise RuntimeError("Fock diagnostic did not preserve the frozen warm density")
    profile = _parse_fock_profile(diagnostic)
    if profile["operator_evaluation_count"] != 1:
        raise RuntimeError("fixed-density timing requires exactly one Fock evaluation")
    return {
        "warm_start_used": item.warm_start_used,
        "warm_start_fallback": item.warm_start_fallback,
        **profile,
    }


def _fixed_density_profiles(
    case: Any,
    arguments: argparse.Namespace,
    modes: tuple[tuple[str, str | None], ...],
    cupy: Any,
) -> dict[str, Any]:
    """Replay one frozen FP64 density through all isolated Fock variants."""

    calculator = _calculator(case, arguments)
    records: dict[str, Any] = {}
    with calculator.prepare_batch(
        [case.atoms],
        charges=[case.charge],
        multiplicities=[case.multiplicity],
        warm_start=True,
    ) as batch:
        with _environment(_mode_environment(None, profiling=False)):
            cold = batch.execute(strict=True)
            cupy.cuda.Stream.null.synchronize()
        batch.set_warm_start_updates(False)
        fixed = cold.items[0]

        for name, selection in modes:
            samples: list[dict[str, Any]] = []
            with _environment(_mode_environment(selection, profiling=True)):
                # Changing arithmetic invalidates the device plan. Exclude one
                # setup replay so every reported sample is the same steady
                # fixed-density Fock operation.
                setup, setup_diagnostic = _capture_native_stderr(
                    lambda: batch.execute(strict=False)
                )
                cupy.cuda.Stream.null.synchronize()
                _validate_fixed_density_sample(setup.items[0], setup_diagnostic)
                for _ in range(arguments.repeats):
                    cupy.cuda.Stream.null.synchronize()
                    started = time.perf_counter()
                    result, diagnostic = _capture_native_stderr(
                        lambda: batch.execute(strict=False)
                    )
                    cupy.cuda.Stream.null.synchronize()
                    item = result.items[0]
                    samples.append(
                        {
                            "synchronized_wall_seconds": time.perf_counter() - started,
                            **_validate_fixed_density_sample(item, diagnostic),
                        }
                    )
            records[name] = {
                "requested_threshold": selection,
                "samples": samples,
            }
    return {
        "fixed_density_origin": {
            "arithmetic": "fp64",
            "energy": fixed.energy,
            "iterations": fixed.iterations,
            "warm_start_updates_frozen": True,
        },
        "variants": records,
    }


def _complete_endpoints(
    case: Any,
    arguments: argparse.Namespace,
    modes: tuple[tuple[str, str | None], ...],
    cupy: Any,
) -> dict[str, Any]:
    """Measure clean complete calls rather than subtracting a force estimate."""

    endpoints: dict[str, Any] = {}
    for name, selection in modes:
        calculator = _calculator(case, arguments)
        mode_record: dict[str, Any] = {"requested_threshold": selection}
        with _environment(_mode_environment(selection, profiling=False)):
            for endpoint, properties in (
                ("energy_only", ("energy",)),
                ("energy_plus_forces", ("energy", "forces")),
            ):
                samples: list[dict[str, Any]] = []
                for _ in range(arguments.repeats):
                    cupy.cuda.Stream.null.synchronize()
                    started = time.perf_counter()
                    result = calculator.singlepoint(
                        case.atoms,
                        charge=case.charge,
                        multiplicity=case.multiplicity,
                        properties=properties,
                    )
                    cupy.cuda.Stream.null.synchronize()
                    samples.append(
                        {
                            "synchronized_wall_seconds": time.perf_counter() - started,
                            "energy": result.energy,
                            "iterations": result.iterations,
                            "converged": result.converged,
                            "force_max_abs": (
                                None
                                if result.forces is None
                                else float(abs(result.forces).max())
                            ),
                        }
                    )
                mode_record[endpoint] = {
                    "properties": list(properties),
                    "samples": samples,
                }
        endpoints[name] = mode_record
    return endpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="water-tetramer-def2-svp-spherical")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--screening-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--experimental-fp32-threshold", type=float, default=1.0e300)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or arguments.max_iterations < 1:
        parser.error("--repeats and --max-iterations must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("this real-GPU benchmark must run inside a Slurm allocation")

    # Delay the GPU import until after argument validation so --help remains
    # safe on login nodes and never initializes a device outside the scheduler.
    import cupy as cp

    from benchmarks._cases import benchmark_cases
    from benchmarks._support import (
        cuda_accelerator_metadata,
        environment_metadata,
        write_result,
    )

    try:
        case = benchmark_cases()[arguments.case]
    except KeyError as error:
        parser.error(f"unknown benchmark case: {arguments.case}")
        raise AssertionError("unreachable") from error
    modes = (
        ("fp64", None),
        ("auto", "auto"),
        ("experimental_fp32", f"{arguments.experimental_fp32_threshold:.17g}"),
    )
    payload = {
        "schema_version": 1,
        "benchmark": "issue174_precision_boundaries",
        "case": arguments.case,
        "case_description": case.description,
        "controls": {
            "max_iterations": arguments.max_iterations,
            "energy_tolerance": arguments.energy_tolerance,
            "density_tolerance": arguments.density_tolerance,
            "screening_tolerance": arguments.screening_tolerance,
            "repeats": arguments.repeats,
        },
        "fixed_density_fock": _fixed_density_profiles(case, arguments, modes, cp),
        "complete_endpoints": _complete_endpoints(case, arguments, modes, cp),
        "environment": environment_metadata(
            distributions={"numpy": ("numpy",), "cupy": ("cupy-cuda12x", "cupy")},
            accelerator=cuda_accelerator_metadata(cp),
        ),
        "limitations": [
            "The experimental threshold is not a promoted automatic policy.",
            "Class GPU global-timer rows cover instrumented generated Fock workers; synchronized wall time covers the complete isolated replay.",
            "This A-slice records output boundaries and arithmetic work counts; controller/refinement evidence belongs to later #174 slices.",
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        destination = write_result(arguments.output, payload)
        print(f"JSON result: {destination}")


if __name__ == "__main__":
    main()

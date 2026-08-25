"""Accept a generated shell-class batch against the current production set.

The benchmark changes ``QCE_AOT_SHELL_CLASSES`` inside one process, so the
baseline and candidate share one prepared QCE batch, CUDA context, and GPU
allocation.  If the complete candidate set fails, recursive subset checks
identify the smallest observed regression group without rebuilding CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from _cases import benchmark_cases
from compare_gpu4pyscf_batch import scaled_geometries
from qce import Calculator


def _class_list(value: str) -> tuple[str, ...]:
    """Parse a deterministic comma-separated exact shell-class selection."""

    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("shell classes must be non-empty and unique")
    return result


def _execute(batch, cupy, selection: tuple[str, ...], repeats: int):
    """Time one runtime selection and retain the final converged result."""

    os.environ["QCE_AOT_SHELL_CLASSES"] = ",".join(selection)
    timings = []
    result = None
    for _ in range(repeats):
        cupy.cuda.Stream.null.synchronize()
        begin = time.perf_counter()
        result = batch.execute(strict=True)
        cupy.cuda.Stream.null.synchronize()
        timings.append(time.perf_counter() - begin)
    assert result is not None
    return result, timings


def _accuracy(reference, candidate) -> tuple[float, float]:
    """Return maximum energy and Cartesian force differences."""

    reference_forces = np.stack([item.forces for item in reference.items])
    candidate_forces = np.stack([item.forces for item in candidate.items])
    return (
        float(np.max(np.abs(reference.energies - candidate.energies))),
        float(np.max(np.abs(reference_forces - candidate_forces))),
    )


def _measurement(
    batch,
    cupy,
    baseline_classes: tuple[str, ...],
    candidate_classes: tuple[str, ...],
    repeats: int,
    maximum_energy_error: float,
    maximum_force_error: float,
    minimum_speedup: float,
) -> dict[str, object]:
    """Measure a selection relative to the exact current production baseline."""

    baseline_result, baseline_seconds = _execute(
        batch, cupy, baseline_classes, repeats
    )
    candidate_result, candidate_seconds = _execute(
        batch, cupy, candidate_classes, repeats
    )
    energy_error, force_error = _accuracy(baseline_result, candidate_result)
    baseline_median = statistics.median(baseline_seconds)
    candidate_median = statistics.median(candidate_seconds)
    speedup = baseline_median / candidate_median
    failures = []
    if energy_error > maximum_energy_error:
        failures.append("energy accuracy")
    if force_error > maximum_force_error:
        failures.append("force accuracy")
    if speedup < minimum_speedup:
        failures.append("performance")
    if not all(item.converged for item in candidate_result.items):
        failures.append("SCF convergence")
    return {
        "classes": list(candidate_classes),
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "baseline_median_seconds": baseline_median,
        "candidate_median_seconds": candidate_median,
        "speedup": speedup,
        "maximum_energy_error_hartree": energy_error,
        "maximum_force_error_hartree_per_bohr": force_error,
        "iterations": [item.iterations for item in candidate_result.items],
        "passed": not failures,
        "failures": failures,
    }


def _bisect_regression(
    batch,
    cupy,
    baseline: tuple[str, ...],
    extras: tuple[str, ...],
    arguments: argparse.Namespace,
    measurements: list[dict[str, object]],
) -> list[list[str]]:
    """Recursively identify failing subsets, retaining interaction groups."""

    if not extras:
        return []
    selection = (*baseline, *extras)
    measurement = _measurement(
        batch,
        cupy,
        baseline,
        selection,
        arguments.bisection_repeats,
        arguments.maximum_energy_error,
        arguments.maximum_force_error,
        arguments.minimum_speedup,
    )
    measurement["bisection"] = True
    measurements.append(measurement)
    if measurement["passed"]:
        return []
    if len(extras) == 1:
        return [list(extras)]
    middle = len(extras) // 2
    first = extras[:middle]
    second = extras[middle:]
    first_failures = _bisect_regression(
        batch, cupy, baseline, first, arguments, measurements
    )
    second_failures = _bisect_regression(
        batch, cupy, baseline, second, arguments, measurements
    )
    if not first_failures and not second_failures:
        # Neither half regresses independently, so preserve the interaction
        # group instead of incorrectly blaming one exact class.
        return [list(extras)]
    return [*first_failures, *second_failures]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", default="water-tetramer-def2-svp-spherical"
    )
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument(
        "--baseline-classes", type=_class_list, default=("dppp", "dpds")
    )
    parser.add_argument(
        "--candidate-classes",
        type=_class_list,
        default=(
            "dppp",
            "dpds",
            "ppps",
            "dpps",
            "dsps",
            "dspp",
        ),
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
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    batches = tuple(arguments.batch or (1, 4))
    if any(value < 1 for value in batches):
        parser.error("--batch values must be positive")
    if arguments.repeats < 1 or arguments.bisection_repeats < 1:
        parser.error("repeat counts must be positive")

    import cupy as cp

    case = benchmark_cases()[arguments.case]
    payload: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "aot_shell_batch_gate",
        "case": arguments.case,
        "baseline_classes": list(arguments.baseline_classes),
        "candidate_classes": list(arguments.candidate_classes),
        "gates": {
            "minimum_speedup": arguments.minimum_speedup,
            "maximum_energy_error_hartree": arguments.maximum_energy_error,
            "maximum_force_error_hartree_per_bohr": arguments.maximum_force_error,
        },
        "batches": [],
    }
    gate_passed = True
    for batch_size in batches:
        systems = scaled_geometries(case.atoms, batch_size)
        calculator = Calculator(
            method=case.method,
            basis=case.qce_basis,
            basis_representation=case.basis_representation,
            device="cuda",
            max_iterations=arguments.max_iterations,
            energy_tolerance=arguments.energy_tolerance,
            density_tolerance=arguments.density_tolerance,
            screening_tolerance=arguments.screening_tolerance,
        )
        measurements: list[dict[str, object]] = []
        with calculator.prepare_batch(
            systems,
            charges=[case.charge] * batch_size,
            multiplicities=[case.multiplicity] * batch_size,
            warm_start=True,
        ) as prepared:
            _execute(
                prepared,
                cp,
                arguments.baseline_classes,
                arguments.warmups,
            )
            full = _measurement(
                prepared,
                cp,
                arguments.baseline_classes,
                arguments.candidate_classes,
                arguments.repeats,
                arguments.maximum_energy_error,
                arguments.maximum_force_error,
                arguments.minimum_speedup,
            )
            measurements.append(full)
            regression_groups = []
            if not full["passed"]:
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
        gate_passed = gate_passed and bool(full["passed"])
        payload["batches"].append(
            {
                "batch_size": batch_size,
                "full_candidate": full,
                "regression_groups": regression_groups,
                "measurements": measurements,
            }
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

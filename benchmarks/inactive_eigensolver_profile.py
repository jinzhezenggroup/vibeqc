"""Measure inactive-state work in divergent RHF/UHF eigensolver fleets.

The endpoint samples use an uninstrumented batch. A separate opt-in plan
records `%globaltimer` intervals around each eigensolve in the device-tail SCF
Graph. Both plans replay the same frozen base-geometry density against a
deterministic family of perturbed geometries, so some systems converge before
their peers without introducing host-side iteration control.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from _cases import benchmark_cases
from _support import cuda_accelerator_metadata, environment_metadata, write_result
from vibeqc import BatchResult, Calculator, InactiveEigensolverProfileEntry


def _csv_choices(value: str, allowed: set[str]) -> tuple[str, ...]:
    """Parse an ordered comma-separated selection without duplicates."""

    selected = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not selected or len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError("selection must be non-empty and unique")
    invalid = set(selected) - allowed
    if invalid:
        raise argparse.ArgumentTypeError(
            "unsupported selection: " + ", ".join(sorted(invalid))
        )
    return selected


def _batch_sizes(value: str) -> tuple[int, ...]:
    """Parse the exact positive fleet sizes requested by issue 51."""

    try:
        sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch sizes must be integers") from error
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive and unique")
    return sizes


def divergent_coordinates(
    atoms: Sequence[tuple[str, Sequence[float]]],
    batch_size: int,
    maximum_distortion_bohr: float,
) -> tuple[np.ndarray, ...]:
    """Return compatible geometries with deliberately different warm difficulty.

    Every eighth system keeps the frozen reference geometry. The other seven
    levels perturb hydrogens more than heavy atoms while keeping the cluster's
    center fixed. This produces a repeatable convergence spread without
    changing atom order, basis topology, charge, or spin occupation.
    """

    base = np.asarray([position for _, position in atoms], dtype=np.float64)
    center = np.mean(base, axis=0)
    levels = np.asarray((0.0, 0.04, 0.09, 0.16, 0.28, 0.45, 0.68, 1.0))
    geometries = []
    for system in range(batch_size):
        amplitude = maximum_distortion_bohr * levels[system % len(levels)]
        coordinates = base.copy()
        if amplitude != 0.0:
            for atom_index, (element, _) in enumerate(atoms):
                phase = (system + 1) * (atom_index + 1)
                direction = np.asarray(
                    (
                        np.sin(0.73 * phase),
                        np.cos(1.11 * phase),
                        np.sin(1.37 * phase + 0.2),
                    )
                )
                direction /= np.linalg.norm(direction)
                weight = 0.35 if element.upper() != "H" else 1.0
                coordinates[atom_index] += amplitude * weight * direction
            # Remove the small net translation introduced by unequal weights.
            coordinates -= np.mean(coordinates, axis=0) - center
        geometries.append(coordinates)
    return tuple(geometries)


def _timed_execute(
    batch: Any, coordinates: Sequence[np.ndarray]
) -> tuple[BatchResult, float]:
    """Time one synchronous native execution and require scientific success."""

    start = time.perf_counter()
    result = batch.execute(coordinates, strict=True)
    return result, time.perf_counter() - start


def _convergence(result: BatchResult) -> list[dict[str, Any]]:
    return [
        {
            "index": item.index,
            "iterations": item.iterations,
            "energy_change_hartree": item.energy_change,
            "density_rms": item.density_rms,
            "converged": item.converged,
        }
        for item in result.items
    ]


def profile_summary(
    records: Sequence[InactiveEigensolverProfileEntry],
    endpoint_median_seconds: float,
) -> dict[str, Any]:
    """Derive issue-51 gate quantities from exact per-iteration records."""

    provider_records = [record for record in records if record.provider_invoked]
    provider_nanoseconds = sum(
        record.solver_elapsed_nanoseconds for record in provider_records
    )
    divergent_records = [
        record
        for record in provider_records
        if record.active_solver_count < record.solver_batch_count
    ]
    post_first_convergence_nanoseconds = sum(
        record.solver_elapsed_nanoseconds for record in divergent_records
    )
    # A fixed-batch provider exposes only total batch time. Weighting each
    # measured interval by its inactive fraction is the explicit upper-bound
    # estimate used by the issue's decision gate; it is not reported as a
    # measured compacted-provider runtime.
    estimated_inactive_nanoseconds = sum(
        record.solver_elapsed_nanoseconds * record.inactive_fraction
        for record in provider_records
    )
    estimated_inactive_share = (
        estimated_inactive_nanoseconds / provider_nanoseconds
        if provider_nanoseconds
        else 0.0
    )
    plausible_endpoint_improvement_fraction = (
        estimated_inactive_nanoseconds * 1.0e-9 / endpoint_median_seconds
        if endpoint_median_seconds > 0.0
        else 0.0
    )
    inactive_share_gate = estimated_inactive_share >= 0.20
    endpoint_gate = plausible_endpoint_improvement_fraction >= 0.01
    submission_finite = all(
        record.inactive_submission_nonfinite_count == 0 for record in records
    )
    terminal_status_isolated = all(
        record.inactive_info_nonzero_count == 0 for record in records
    )
    return {
        "record_count": len(records),
        "all_solver_device_nanoseconds": sum(
            record.solver_elapsed_nanoseconds for record in records
        ),
        "provider_record_count": len(provider_records),
        "first_divergent_iteration": (
            divergent_records[0].iteration if divergent_records else None
        ),
        "maximum_inactive_fraction": max(
            (record.inactive_fraction for record in provider_records),
            default=0.0,
        ),
        "provider_device_nanoseconds": provider_nanoseconds,
        "provider_device_nanoseconds_after_first_convergence": (
            post_first_convergence_nanoseconds
        ),
        "estimated_inactive_device_nanoseconds": estimated_inactive_nanoseconds,
        "estimated_inactive_share_of_provider_time": estimated_inactive_share,
        "plausible_endpoint_improvement_fraction": (
            plausible_endpoint_improvement_fraction
        ),
        "inactive_input_nonfinite_matrices_before_repair": sum(
            record.inactive_input_nonfinite_count for record in records
        ),
        "inactive_submission_nonfinite_matrices": sum(
            record.inactive_submission_nonfinite_count for record in records
        ),
        "inactive_nonzero_info_slots": sum(
            record.inactive_info_nonzero_count for record in records
        ),
        "inactive_touches": sorted(
            {touch for record in records for touch in record.inactive_touches}
        ),
        "validation": {
            "inactive_provider_submissions_finite": submission_finite,
            "inactive_info_cannot_change_terminal_status": terminal_status_isolated,
            "passed": submission_finite and terminal_status_isolated,
        },
        "decision_gate": {
            "minimum_inactive_share": 0.20,
            "minimum_plausible_endpoint_improvement_fraction": 0.01,
            "inactive_share_gate_met": inactive_share_gate,
            "endpoint_gate_met": endpoint_gate,
            "met": inactive_share_gate or endpoint_gate,
        },
    }


def measure_workload(
    *,
    method: str,
    batch_size: int,
    repeats: int,
    maximum_distortion_bohr: float,
    max_iterations: int,
    energy_tolerance: float,
    density_tolerance: float,
    screening_tolerance: float,
) -> dict[str, Any]:
    """Measure one RHF or UHF 96-AO divergent fleet."""

    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    systems = [case.atoms] * batch_size
    charge = 0 if method == "rhf" else 1
    multiplicity = 1 if method == "rhf" else 2
    charges = [charge] * batch_size
    multiplicities = [multiplicity] * batch_size
    coordinates = divergent_coordinates(case.atoms, batch_size, maximum_distortion_bohr)
    calculator = Calculator(
        method=method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=max_iterations,
        energy_tolerance=energy_tolerance,
        density_tolerance=density_tolerance,
        screening_tolerance=screening_tolerance,
    )

    endpoint_samples = []
    endpoint_result: BatchResult | None = None
    with calculator.prepare_batch(
        systems,
        charges=charges,
        multiplicities=multiplicities,
        warm_start=True,
    ) as batch:
        cold = batch.execute(strict=True)
        batch.set_warm_start_updates(False)
        # Prime geometry-derived state outside the endpoint sample set.
        _timed_execute(batch, coordinates)
        for _ in range(repeats):
            endpoint_result, seconds = _timed_execute(batch, coordinates)
            endpoint_samples.append(seconds)

    assert endpoint_result is not None
    endpoint_median = statistics.median(endpoint_samples)

    with calculator.prepare_batch(
        systems,
        charges=charges,
        multiplicities=multiplicities,
        warm_start=True,
        inactive_eigensolver_profiling=True,
    ) as batch:
        batch.execute(strict=True)
        batch.set_warm_start_updates(False)
        _timed_execute(batch, coordinates)
        profiled_result, profiled_seconds = _timed_execute(batch, coordinates)
        records = batch.last_inactive_eigensolver_profile()
        diagnostics = [
            diagnostic.to_dict() for diagnostic in batch.last_eigensolver_diagnostics()
        ]

    expected_record_count = max(item.iterations for item in profiled_result.items)
    if len(records) != expected_record_count:
        raise RuntimeError(
            "iteration profile length does not match the longest SCF branch: "
            f"{len(records)} != {expected_record_count}"
        )
    summary = profile_summary(records, endpoint_median)
    return {
        "method": method,
        "batch_size": batch_size,
        "matrix_dimension": 96,
        "charge": charge,
        "multiplicity": multiplicity,
        "distortion_levels_bohr": [
            float(np.max(np.abs(coordinates[index] - coordinates[0])))
            for index in range(batch_size)
        ],
        "cold_convergence": _convergence(cold),
        "endpoint": {
            "instrumented": False,
            "samples_seconds": endpoint_samples,
            "median_seconds": endpoint_median,
            "systems_per_second": batch_size / endpoint_median,
            "convergence": _convergence(endpoint_result),
        },
        "profiled_execution": {
            "seconds": profiled_seconds,
            "convergence": _convergence(profiled_result),
            "eigensolver_diagnostics": diagnostics,
            "iteration_records": [record.to_dict() for record in records],
            "summary": summary,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        type=lambda value: _csv_choices(value, {"rhf", "uhf"}),
        default=("rhf", "uhf"),
    )
    parser.add_argument("--batch-sizes", type=_batch_sizes, default=(4, 16, 64))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maximum-distortion-bohr", type=float, default=0.15)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--screening-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--comparison-baseline",
        type=Path,
        help="provider-profile JSON to compare with an override run",
    )
    return parser


def candidate_comparisons(
    baseline: dict[str, Any], candidate_workloads: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare overlapping candidate workloads against provider evidence."""

    baseline_by_key = {
        (workload["method"], workload["batch_size"]): workload
        for workload in baseline["workloads"]
    }
    comparisons = []
    for candidate in candidate_workloads:
        key = (candidate["method"], candidate["batch_size"])
        if key not in baseline_by_key:
            continue
        provider = baseline_by_key[key]
        provider_iterations = [
            item["iterations"] for item in provider["endpoint"]["convergence"]
        ]
        candidate_iterations = [
            item["iterations"] for item in candidate["endpoint"]["convergence"]
        ]
        provider_solver_nanoseconds = provider["profiled_execution"]["summary"].get(
            "all_solver_device_nanoseconds",
            provider["profiled_execution"]["summary"]["provider_device_nanoseconds"],
        )
        candidate_solver_nanoseconds = candidate["profiled_execution"]["summary"][
            "all_solver_device_nanoseconds"
        ]
        endpoint_ratio = (
            candidate["endpoint"]["median_seconds"]
            / provider["endpoint"]["median_seconds"]
        )
        comparisons.append(
            {
                "method": key[0],
                "batch_size": key[1],
                "iteration_branches_match": (
                    candidate_iterations == provider_iterations
                ),
                "provider_endpoint_median_seconds": provider["endpoint"][
                    "median_seconds"
                ],
                "candidate_endpoint_median_seconds": candidate["endpoint"][
                    "median_seconds"
                ],
                "candidate_to_provider_endpoint_ratio": endpoint_ratio,
                "provider_solver_device_nanoseconds": provider_solver_nanoseconds,
                "candidate_solver_device_nanoseconds": candidate_solver_nanoseconds,
                "candidate_to_provider_solver_ratio": (
                    candidate_solver_nanoseconds / provider_solver_nanoseconds
                ),
                "candidate_improves_endpoint": endpoint_ratio < 1.0,
            }
        )
    return comparisons


def main() -> None:
    args = _parser().parse_args()
    if args.repeats <= 0 or args.max_iterations <= 0:
        raise ValueError("--repeats and --max-iterations must be positive")
    if args.maximum_distortion_bohr < 0.0:
        raise ValueError("--maximum-distortion-bohr must be non-negative")

    workloads = []
    for method in args.methods:
        for batch_size in args.batch_sizes:
            print(f"measuring {method.upper()} batch {batch_size}", flush=True)
            workload = measure_workload(
                method=method,
                batch_size=batch_size,
                repeats=args.repeats,
                maximum_distortion_bohr=args.maximum_distortion_bohr,
                max_iterations=args.max_iterations,
                energy_tolerance=args.energy_tolerance,
                density_tolerance=args.density_tolerance,
                screening_tolerance=args.screening_tolerance,
            )
            workloads.append(workload)
            summary = workload["profiled_execution"]["summary"]
            print(
                "  iterations="
                + ",".join(
                    str(item["iterations"])
                    for item in workload["endpoint"]["convergence"]
                )
            )
            print(
                "  inactive-share="
                f"{summary['estimated_inactive_share_of_provider_time']:.2%}, "
                "endpoint-upper-bound="
                f"{summary['plausible_endpoint_improvement_fraction']:.3%}",
                flush=True,
            )

    try:
        import cupy as cp

        accelerator = cuda_accelerator_metadata(cp)
    except (ImportError, RuntimeError):
        accelerator = None
    current_run_gate_met = any(
        workload["profiled_execution"]["summary"]["decision_gate"]["met"]
        for workload in workloads
    )
    validation_passed = all(
        workload["profiled_execution"]["summary"]["validation"]["passed"]
        for workload in workloads
    )
    comparisons = []
    baseline_gate_met: bool | None = None
    baseline_validation_passed: bool | None = None
    if args.comparison_baseline is not None:
        baseline = json.loads(args.comparison_baseline.read_text(encoding="utf-8"))
        comparisons = candidate_comparisons(baseline, workloads)
        baseline_decision = baseline.get("decision", {})
        baseline_gate_met = bool(
            baseline_decision.get("any_workload_met_optimization_gate", False)
        )
        baseline_validation_passed = bool(
            baseline_decision.get("all_safety_validations_passed", False)
        )
    # A masked-solver candidate has no provider records of its own. Its
    # eligibility therefore comes from the provider measurement being compared,
    # while its own records still prove convergence and timing behavior.
    effective_gate_met = (
        baseline_gate_met if baseline_gate_met is not None else current_run_gate_met
    )
    effective_validation_passed = validation_passed and (
        baseline_validation_passed if baseline_validation_passed is not None else True
    )
    candidate_improves_endpoint = any(
        comparison["candidate_improves_endpoint"] for comparison in comparisons
    )
    if comparisons and not candidate_improves_endpoint:
        decision_reason = (
            "reject the measured candidate because it did not improve any "
            "overlapping endpoint"
        )
    elif effective_gate_met:
        decision_reason = (
            "the measurement gate permits candidate evaluation; promotion still "
            "requires a measured endpoint win and regression gates"
        )
    else:
        decision_reason = "publish measurement only; no workload met the issue-51 gate"
    payload = {
        "schema_version": 1,
        "benchmark": "inactive_eigensolver_profile",
        "issue": 51,
        "environment": environment_metadata(
            distributions={"cupy": ("cupy-cuda12x", "cupy"), "numpy": ("numpy",)},
            accelerator=accelerator,
        ),
        "settings": {
            "methods": list(args.methods),
            "batch_sizes": list(args.batch_sizes),
            "repeats": args.repeats,
            "maximum_distortion_bohr": args.maximum_distortion_bohr,
            "max_iterations": args.max_iterations,
            "energy_tolerance": args.energy_tolerance,
            "density_tolerance": args.density_tolerance,
            "screening_tolerance": args.screening_tolerance,
            "graph_eigensolver_override": os.environ.get(
                "VIBEQC_GRAPH_EIGENSOLVER_OVERRIDE"
            ),
            "warm_start_policy": (
                "freeze the common base-geometry converged density, then replay "
                "deterministic geometry perturbations"
            ),
        },
        "workloads": workloads,
        "candidate_comparisons": comparisons,
        "decision": {
            "all_safety_validations_passed": effective_validation_passed,
            "current_run_any_workload_met_optimization_gate": (current_run_gate_met),
            "baseline_met_optimization_gate": baseline_gate_met,
            "any_workload_met_optimization_gate": effective_gate_met,
            "evaluate_scheduler_candidates": (
                effective_validation_passed and effective_gate_met
            ),
            "promote_scheduler_change": (
                effective_validation_passed
                and effective_gate_met
                and candidate_improves_endpoint
            ),
            "reason": decision_reason,
        },
    }
    destination = write_result(args.output, payload)
    print(f"JSON result: {destination}")


if __name__ == "__main__":
    main()

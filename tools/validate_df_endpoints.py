"""Measure complete old/new CUDA DF energy-force endpoints under Slurm.

Cold execution includes setup and SCF; changed geometry invalidates the source;
warm samples reuse the fixed-geometry plan and converged density. These timings
are reported separately so cached work cannot masquerade as integral speedup.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from vibeqc import Calculator

from tools.vibeqc_validation.f_shell_numerics import numerical_error
from tools.vibeqc_validation.schema import file_hash


def run_endpoint(method, batch, budget, route, repeats):
    """Use public prepared batches, including changed coordinates and forces."""
    os.environ["VIBEQC_DF_VALUES"] = (
        "reference" if route == "reference" else "generated"
    )
    os.environ["VIBEQC_DF_VALUE_MAPPING"] = route
    geometry = [("O", [0.0, 0.0, 0.0]), ("H", [0.0, 1.43, 1.11])]
    if method == "rhf":
        geometry.append(("H", [0.0, -1.43, 1.11]))
    systems = [
        [(element, [x, y, z + item * 0.01]) for element, (x, y, z) in geometry]
        for item in range(batch)
    ]
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        device="cuda",
        basis_representation="spherical",
        density_fitting="cuda",
        auxiliary_basis="def2-svp",
        density_fitting_relative_threshold=1e-12,
        density_fitting_memory_budget_bytes=budget,
        max_iterations=160,
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
    )
    phases = {}
    start = time.perf_counter()
    with calculator.prepare_batch(
        systems, multiplicities=[1 if method == "rhf" else 2] * batch
    ) as prepared:
        phases["prepare_ms"] = (time.perf_counter() - start) * 1000

        def execute(coordinates=None):
            begin = time.perf_counter()
            result = prepared.execute(coordinates, strict=True)
            elapsed = (time.perf_counter() - begin) * 1000
            if not all(
                item.converged and item.executed_backend == "cuda"
                for item in result.items
            ):
                raise RuntimeError("DF endpoint did not converge on CUDA")
            record = {
                "milliseconds": elapsed,
                "energies": result.energies.tolist(),
                "forces": [item.forces.tolist() for item in result.items],
                "iterations": [item.iterations for item in result.items],
                "metrics": [
                    d.to_dict()
                    for d in prepared.last_density_fitting_metric_diagnostics()
                ],
            }
            # Force-response scratch has a separate conservative peak; the
            # positive planner budget bounds persistent source/plan storage.
            if budget and any(
                d["device_resident_bytes"] > budget for d in record["metrics"]
            ):
                raise RuntimeError(
                    "DF resident plan exceeds its configured memory budget"
                )
            return record

        phases["cold"] = execute()
        phases["warm"] = [execute() for _ in range(repeats)]
        changed = [np.array([xyz for _, xyz in system]) for system in systems]
        for item, xyz in enumerate(changed):
            xyz[1, 2] += 0.025 + item * 0.01
        phases["changed_geometry"] = execute(changed)
        # Omitting coordinates restores the prepared input geometry; keep the
        # updated coordinates explicit so this phase reuses the changed source.
        phases["changed_warm"] = [execute(changed) for _ in range(repeats)]
        for cold_phase, warm_phase in (
            ("cold", "warm"),
            ("changed_geometry", "changed_warm"),
        ):
            for sample in phases[warm_phase]:
                for quantity in ("energies", "forces"):
                    error = numerical_error(
                        np.asarray(sample[quantity]),
                        np.asarray(phases[cold_phase][quantity]),
                        atol=2e-8,
                        rtol=2e-9,
                    )
                    if not error["passed"]:
                        raise RuntimeError(
                            f"{warm_phase} changed {quantity} at fixed geometry"
                        )
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[0, 8 * 1024**2, 16 * 1024**2]
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--methods", nargs="+", choices=("rhf", "uhf"), default=["rhf", "uhf"]
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=("auxiliary", "component", "primitive"),
        default=["auxiliary", "component", "primitive"],
    )
    args = parser.parse_args()
    if (
        args.repeats < 1
        or any(n < 1 for n in args.batches)
        or any(b < 0 for b in args.budgets)
    ):
        parser.error(
            "positive repeats/batches and non-negative memory budgets are required"
        )
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run this real GPU endpoint through srun")
    report = {
        "schema": "vibeqc.df_endpoint_validation",
        "version": 1,
        "library": os.environ["VIBEQC_LIBRARY"],
        "library_hash": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "runs": [],
        "production_promoted": False,
        "context_preconditioned": True,
        "placement": {
            "positive_budget": "device public-basis transform; metric D2H/H2D setup staging",
            "zero_budget": "device Cartesian values; host public-basis transform and raw staging",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        for batch in args.batches:
            for budget in args.budgets:
                # Prime this route's CUDA modules/context before both sides.
                # Each measured run still creates a fresh basis/source/plan,
                # so source setup is cold without charging context startup
                # exclusively to the first reference measurement.
                run_endpoint(method, batch, budget, "reference", 1)
                baseline = run_endpoint(
                    method, batch, budget, "reference", args.repeats
                )
                # Bulk generation has one mapping; repeating it under each
                # source-only override would duplicate the same measurement.
                routes = args.routes if budget else args.routes[:1]
                for route in routes:
                    actual = run_endpoint(method, batch, budget, route, args.repeats)
                    errors = {}
                    for phase in ("cold", "changed_geometry", "warm", "changed_warm"):
                        expected_samples = (
                            baseline[phase] if "warm" in phase else [baseline[phase]]
                        )
                        actual_samples = (
                            actual[phase] if "warm" in phase else [actual[phase]]
                        )
                        # Every replay is a numerical gate, including early
                        # warm samples that a final-sample check would miss.
                        for sample, (expected_phase, actual_phase) in enumerate(
                            zip(expected_samples, actual_samples, strict=True)
                        ):
                            for quantity in ("energies", "forces"):
                                errors[f"{phase}_{sample}_{quantity}"] = (
                                    numerical_error(
                                        np.asarray(actual_phase[quantity]),
                                        np.asarray(expected_phase[quantity]),
                                        atol=2e-8,
                                        rtol=2e-9,
                                    )
                                )
                            errors[f"{phase}_{sample}_rank"] = {
                                "passed": [
                                    d["effective_rank"]
                                    for d in expected_phase["metrics"]
                                ]
                                == [
                                    d["effective_rank"] for d in actual_phase["metrics"]
                                ]
                            }
                    speedups = {
                        phase: baseline[phase]["milliseconds"]
                        / actual[phase]["milliseconds"]
                        for phase in ("cold", "changed_geometry")
                    }
                    speedups.update(
                        {
                            phase: float(
                                np.median([r["milliseconds"] for r in baseline[phase]])
                                / np.median([r["milliseconds"] for r in actual[phase]])
                            )
                            for phase in ("warm", "changed_warm")
                        }
                    )
                    row = {
                        "method": method,
                        "batch": batch,
                        "budget": budget,
                        "route": route,
                        # The bulk compatibility builder uses one thread per
                        # output. Cooperative mappings belong to the bounded
                        # source, so never label bulk timings as warp timings.
                        "effective_mapping": route if budget else "auxiliary_bulk",
                        "baseline": baseline,
                        "generated": actual,
                        "errors": errors,
                        "speedup": speedups,
                        "passed": all(error["passed"] for error in errors.values()),
                    }
                    report["runs"].append(row)
                    args.output.write_text(
                        json.dumps(report, indent=2, sort_keys=True) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "method",
                                    "batch",
                                    "budget",
                                    "route",
                                    "passed",
                                    "speedup",
                                )
                            }
                        ),
                        flush=True,
                    )
    report["passed"] = all(row["passed"] for row in report["runs"])
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

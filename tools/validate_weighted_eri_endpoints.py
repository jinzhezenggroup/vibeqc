"""Compare retained/generated psss with complete public RHF/UHF force endpoints.

Each route starts in a fresh process and prepared plan, preserving the selected
library and frozen resident-bra policy. Mixed batches include an s-only system.
Changed-geometry warm replays always supply the changed coordinates explicitly.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from tools.vibeqc_validation.f_shell_numerics import numerical_error
from tools.vibeqc_validation.schema import file_hash


def endpoint(config):
    """Execute cold, fixed warm, changed cold, and explicit changed warm phases."""
    from vibeqc import Calculator

    method, batch = config["method"], config["batch"]
    water = [
        ("O", [0.0, 0.0, 0.0]),
        ("H", [0.0, 1.43, 1.11]),
        ("H", [0.0, -1.43, 1.11]),
    ]
    # OH + two closed-shell waters gives a single radical without the nearly
    # degenerate independent spins of three separated OH radicals. Both
    # methods exceed the native 16-AO persistent-ERI cutoff even in STO-3G.
    geometry = [
        (element, [x + 7.0 * fragment, y, z])
        for fragment in range(3)
        for element, (x, y, z) in (water if method == "rhf" or fragment else water[:2])
    ]
    systems = [
        [(element, [x + item * 0.01, y, z]) for element, (x, y, z) in geometry]
        for item in range(batch)
    ]
    if batch > 1:
        systems[1] = [("H", [0.0, 0.0, 0.0])]
        if method == "rhf":
            systems[1].append(("H", [0.0, 0.0, 1.4]))
    calculator = Calculator(
        method=method,
        basis=config["basis"],
        device="cuda",
        basis_representation="spherical",
        max_iterations=160,
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
    )
    phases = {}
    start = time.perf_counter()
    multiplicities = [1 if method == "rhf" else 2] * batch
    phases["inputs"] = {
        "systems": systems,
        "multiplicities": multiplicities,
        "coordinates_unit": "bohr",
    }
    with calculator.prepare_batch(
        systems,
        multiplicities=multiplicities,
        shell_class_profiling=config.get("profile_only", False),
    ) as prepared:
        phases["prepare_ms"] = (time.perf_counter() - start) * 1000

        def execute(coordinates=None):
            start = time.perf_counter()
            result = prepared.execute(coordinates, strict=True)
            elapsed = (time.perf_counter() - start) * 1000
            if not all(
                item.converged and item.executed_backend == "cuda"
                for item in result.items
            ):
                raise RuntimeError("weighted ERI endpoint failed to converge on CUDA")
            return {
                "milliseconds": elapsed,
                "energies": result.energies.tolist(),
                "forces": [item.forces.tolist() for item in result.items],
                "iterations": [item.iterations for item in result.items],
            }

        phases["cold"] = execute()
        if config.get("profile_only"):
            work = [asdict(row) for row in prepared.last_shell_class_profile()]
            psss = next(row for row in work if row["shell_angular"] == (1, 0, 0, 0))
            if psss["primitive_quartets"] <= 0:
                raise RuntimeError("endpoint did not execute psss primitive work")
            return {"work": work, "cold": phases["cold"]}
        phases["warm"] = [execute() for _ in range(config["repeats"])]
        changed = [np.array([xyz for _, xyz in system]) for system in systems]
        for item, xyz in enumerate(changed):
            xyz[-1, 2] += 0.025 + item * 0.01
        phases["changed_geometry"] = execute(changed)
        phases["changed_warm"] = [execute(changed) for _ in range(config["repeats"])]
        for cold, warm in (("cold", "warm"), ("changed_geometry", "changed_warm")):
            for sample in phases[warm]:
                require_equal(sample, phases[cold])
    return phases


def require_equal(actual, reference):
    """Compare ragged force batches per molecule, plus all molecular energies."""
    checks = [
        numerical_error(
            np.asarray(actual["energies"]),
            np.asarray(reference["energies"]),
            atol=2e-8,
            rtol=2e-10,
        )
    ]
    checks.extend(
        numerical_error(np.asarray(a), np.asarray(b), atol=2e-7, rtol=2e-8)
        for a, b in zip(actual["forces"], reference["forces"], strict=True)
    )
    if not all(c["passed"] for c in checks):
        raise RuntimeError(f"weighted ERI endpoint numerical mismatch: {checks}")
    return checks


def cross_schedule_checks(runs):
    """Gate identical physics across fixed/resident/pages and mixed batches.

    The first molecule has identical coordinates in every batch size, while
    the additional molecules intentionally have different shapes/geometries.
    Compare it separately rather than broadcasting ragged batch arrays.
    """
    references, batch_references = {}, {}
    checks = []
    for run in runs:
        key = (run["method"], run["basis"])
        reference = references.setdefault(key, run["samples"]["reference"])
        same_batch = batch_references.setdefault(
            (*key, run["batch"]), run["samples"]["reference"]
        )
        for samples in run["samples"].values():
            if samples["inputs"]["systems"][0] != reference["inputs"]["systems"][0]:
                raise RuntimeError("cross-schedule first-molecule coordinates differ")
            if samples["inputs"] != same_batch["inputs"]:
                raise RuntimeError("cross-schedule batch inputs differ")
            for phase in ("cold", "changed_geometry"):
                checks.extend(require_equal(samples[phase], same_batch[phase]))

                def first(result):
                    return {
                        "energies": result["energies"][:1],
                        "forces": result["forces"][:1],
                    }

                checks.extend(
                    require_equal(first(samples[phase]), first(reference[phase]))
                )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--methods", nargs="+", choices=["rhf", "uhf"], default=["rhf", "uhf"]
    )
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=["fixed", "resident", "paged"],
        default=["fixed", "resident", "paged"],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--bases", nargs="+", default=["sto-3g", "def2-svp"])
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run this endpoint validator inside a Slurm GPU allocation")
    if args.worker:
        print(json.dumps(endpoint(json.loads(args.worker))))
        return
    if args.output is None or args.repeats < 1 or any(b < 1 for b in args.batches):
        parser.error("output, positive repeats, and positive batches are required")
    current_library = Path(os.environ["VIBEQC_LIBRARY"])
    report = {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "library_sha256": file_hash(current_library),
        "baseline_library_sha256": file_hash(args.baseline_library)
        if args.baseline_library
        else None,
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        for basis in args.bases:
            for batch in args.batches:
                for schedule in args.schedules:
                    resident = schedule == "resident"
                    config = {
                        "method": method,
                        "basis": basis,
                        "batch": batch,
                        "resident_bra": resident,
                        "schedule": schedule,
                        "repeats": args.repeats,
                    }
                    routes = ["reference", "generated"]
                    if args.baseline_library:
                        routes.insert(0, "baseline")
                    samples = {}
                    for route in routes:
                        env = {
                            **os.environ,
                            "VIBEQC_PSSS_WEIGHTED": route,
                            "VIBEQC_PSSS_RESIDENT_BRA": str(int(resident)),
                            "VIBEQC_BOUNDED_DIRECT_STREAMING": "force"
                            if schedule == "paged"
                            else "none",
                            "VIBEQC_LIBRARY": str(
                                args.baseline_library
                                if route == "baseline"
                                else current_library
                            ),
                        }
                        completed = subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker",
                                json.dumps(config),
                            ],
                            env=env,
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        if completed.returncode:
                            raise RuntimeError(
                                f"{config}, route={route}: {completed.stderr}"
                            )
                        samples[route] = json.loads(completed.stdout)
                    profile_config = {
                        **config,
                        "profile_only": True,
                        "profile_schedule": "fixed",
                    }
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            json.dumps(profile_config),
                        ],
                        # Native class counters currently cover fixed queues;
                        # page traversal is gated separately by endpoint equality.
                        env={**env, "VIBEQC_BOUNDED_DIRECT_STREAMING": "none"},
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    profile = {"schedule": "fixed", **json.loads(completed.stdout)}
                    checks, speedups = {}, {}
                    for route in routes:
                        checks[route] = []
                        speedups[route] = {}
                        for phase in (
                            "cold",
                            "warm",
                            "changed_geometry",
                            "changed_warm",
                        ):
                            reference = samples["reference"][phase]
                            candidate = samples[route][phase]
                            if not isinstance(reference, list):
                                reference, candidate = [reference], [candidate]
                            for a, b in zip(candidate, reference, strict=True):
                                checks[route].extend(require_equal(a, b))
                            speedups[route][phase] = float(
                                np.median([s["milliseconds"] for s in reference])
                                / np.median([s["milliseconds"] for s in candidate])
                            )
                    report["runs"].append(
                        {
                            **config,
                            "samples": samples,
                            "profile": profile,
                            "checks": checks,
                            "reference_over_route_speedup": speedups,
                        }
                    )
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({**config, "speedups": speedups}), flush=True)
    report["cross_schedule_batch_checks"] = cross_schedule_checks(report["runs"])
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

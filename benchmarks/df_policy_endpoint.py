"""Interleave DF execution policies from one frozen engine-local warm density.

Every policy transition receives an untimed replay to prime/rebuild its plan;
that cost is retained separately. Clean samples exclude instrumentation. A
separate traced pass records executed branches and device component counters.
Run only in a finite Slurm GPU allocation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from vibeqc import Calculator, _native

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import convergence_payload
from benchmarks.df_component_ledger import aggregate, read_trace

CASES = {
    96: "water-tetramer-def2-svp-spherical",
    192: "water-octamer-s4-def2-svp-spherical",
    384: "water-hexadecamer-2s4-def2-svp-spherical",
    768: "water-32mer-4s4-def2-svp-spherical",
}


def main():
    """Retain each numerical result before enforcing unchanged strict gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aos", type=int, choices=CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--control", default="VIBEQC_DF_EXCHANGE")
    parser.add_argument("--policies", nargs="+", default=["dense", "occupied"])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--journal", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source-patch", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or args.repeats < 1:
        parser.error("requires Slurm and positive repeats")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    case = benchmark_cases()[CASES[args.aos]]
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    native = _native.load_library()
    native.vibeqc_get_source_identity.restype = ctypes.c_char_p
    # A pinned library may outlive subsequent local edits. Prefer its frozen
    # source patch so queued runs describe the code actually linked.
    frozen_patch = args.source_patch or library.with_name("source.patch")
    patch = (
        frozen_patch.read_bytes()
        if frozen_patch.exists()
        else subprocess.check_output(
            ["git", "diff", "HEAD", "--binary", "--", "src", "python"]
        )
    )
    args.output.with_suffix(".source.patch").write_bytes(patch)
    payload = {
        "case": CASES[args.aos],
        "aos": args.aos,
        "scope": "intrusive diagnostic"
        if args.trace or args.journal
        else "clean endpoint",
        "native_source_identity": native.vibeqc_get_source_identity().decode(),
        "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controls": {k: v for k, v in os.environ.items() if k.startswith("VIBEQC_")},
        "scientific_settings": {
            "basis": str(case.vibeqc_basis),
            "basis_representation": case.basis_representation,
            "auxiliary_basis": "same as orbital basis",
            "metric_relative_threshold": 1e-10,
            "method": case.method,
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "max_iterations": 100,
            "screening_tolerance": 1e-12,
            "geometries_bohr": case.atoms,
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "control": args.control,
        "policies": args.policies,
        "warm_policy": "one frozen post-cold density, prime every policy transition",
        "samples": [],
    }

    def save():
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    def execute(batch):
        start = time.perf_counter()
        result = batch.execute(strict=True, properties=("energy", "forces"))
        seconds = time.perf_counter() - start
        return result, seconds

    os.environ[args.control] = args.policies[0]
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        density_fitting="cuda",
        auxiliary_basis=case.vibeqc_basis,
        density_fitting_memory_budget_bytes=0,
        screening_tolerance=1e-12,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    prepare_start = time.perf_counter()
    with calculator.prepare_batch([case.atoms]) as batch:
        payload["prepare_seconds"] = time.perf_counter() - prepare_start
        cold, payload["cold_seconds"] = execute(batch)
        payload["complete_cold_seconds"] = (
            payload["prepare_seconds"] + payload["cold_seconds"]
        )
        payload["cold_convergence"] = convergence_payload(cold)
        batch.set_warm_start_updates(False)
        expected_energy = cold.energies
        expected_forces = np.array([item.forces for item in cold.items])
        if args.reference:
            reference = json.loads(args.reference.read_text())
            if reference["workload"]["case"] != CASES[args.aos]:
                raise RuntimeError("independent reference case differs")
            expected_energy = np.asarray(reference["gpu4pyscf"]["energies_hartree"])
            expected_forces = np.asarray(
                reference["gpu4pyscf"]["forces_hartree_per_bohr"]
            )
            payload["reference_sha256"] = hashlib.sha256(
                args.reference.read_bytes()
            ).hexdigest()
        save()
        for repeat in range(args.repeats):
            policies = args.policies if repeat % 2 == 0 else args.policies[::-1]
            for policy in policies:
                os.environ[args.control] = policy
                prime, prime_seconds = execute(batch)
                trace = args.output.with_suffix(f".{repeat}-{policy}.jsonl")
                if args.trace:
                    os.environ["VIBEQC_DF_TRACE"] = str(trace.resolve())
                journal = args.output.with_suffix(f".{repeat}-{policy}.journal.jsonl")
                if args.journal:
                    os.environ["VIBEQC_DF_PROGRESS_TRACE"] = str(journal.resolve())
                result, seconds = execute(batch)
                os.environ.pop("VIBEQC_DF_PROGRESS_TRACE", None)
                os.environ.pop("VIBEQC_DF_TRACE", None)
                forces = np.array([item.forces for item in result.items])
                sample = {
                    "policy": policy,
                    "repeat": repeat,
                    "seconds": seconds,
                    "prime_seconds": prime_seconds,
                    "prime_iterations": [item.iterations for item in prime.items],
                    "iterations": [item.iterations for item in result.items],
                    "convergence": convergence_payload(result),
                    "energies_hartree": result.energies.tolist(),
                    "forces_hartree_per_bohr": forces.tolist(),
                    "maximum_energy_error": float(
                        np.max(np.abs(result.energies - expected_energy))
                    ),
                    "maximum_force_error": float(
                        np.max(np.abs(forces - expected_forces))
                    ),
                    "metric": [
                        d.to_dict()
                        for d in batch.last_density_fitting_metric_diagnostics()
                    ],
                }
                if args.trace:
                    sample["components"] = aggregate(read_trace(trace))
                if args.journal:
                    sample["final_state_observations"] = [
                        row
                        for line in journal.read_text().splitlines()
                        if (row := json.loads(line)).get("key", "").startswith("final_")
                    ]
                payload["samples"].append(sample)
                save()
                print(policy, repeat, seconds, sample["iterations"], flush=True)
                if (
                    sample["maximum_energy_error"] > 1e-9
                    or sample["maximum_force_error"] > 1e-8
                ):
                    raise RuntimeError("unchanged DF energy/force gate failed")


if __name__ == "__main__":
    main()

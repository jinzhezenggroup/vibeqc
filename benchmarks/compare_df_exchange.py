"""Compare dense/occupied RI-K on one immutable warm-density snapshot.

Policy switches rebuild the device SCF state. Each switch is primed outside
the timed sample; both the priming cost and the steady replay are retained.
This keeps setup visible while measuring the execution policy on identical D.
Run only inside a finite Slurm GPU allocation, without concurrent compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
from pathlib import Path

import numpy as np
from _cases import benchmark_cases
from _support import cuda_accelerator_metadata, environment_metadata, write_result
from compare_gpu4pyscf_batch import _vibeqc_sample, scaled_geometries
from vibeqc import Calculator


def main() -> None:
    """Retain all ABBA samples and reject nonfinite/unconverged parity data."""
    parser = argparse.ArgumentParser(description=__doc__)
    cases = benchmark_cases()
    parser.add_argument("--case", choices=cases, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--energy-only", action="store_true")
    parser.add_argument("--memory-budget-bytes", type=int, default=1 << 30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("a finite Slurm GPU allocation is required")
    if min(args.batch, args.repeats, args.memory_budget_bytes) < 1:
        parser.error("batch, repeats and memory budget must be positive")

    import cupy as cp

    case = cases[args.case]
    forces = not args.energy_only
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    payload = {
        "schema": "vibeqc.df_exchange_ab.v1",
        "case": args.case,
        "batch": args.batch,
        "ao_count": case.expected_ao_count,
        "properties": ["energy", "forces"] if forces else ["energy"],
        "memory_budget_bytes": args.memory_budget_bytes,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": environment_metadata(),
        "accelerator": cuda_accelerator_metadata(cp),
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "warm_policy": "one prepared batch, frozen post-cold dense snapshot; prime after each policy switch",
        "trace_enabled": bool(os.environ.get("VIBEQC_DF_TRACE")),
        "samples": [],
        "passed": False,
    }
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        density_fitting="cuda",
        auxiliary_basis=case.vibeqc_basis,
        density_fitting_memory_budget_bytes=args.memory_budget_bytes,
    )
    original_policy = os.environ.get("VIBEQC_DF_EXCHANGE")
    try:
        with calculator.prepare_batch(
            scaled_geometries(case.atoms, args.batch),
            charges=[case.charge] * args.batch,
            multiplicities=[case.multiplicity] * args.batch,
            warm_start=True,
        ) as batch:
            os.environ["VIBEQC_DF_EXCHANGE"] = "dense"
            payload["cold_dense"] = _vibeqc_sample(batch, cp, -1, forces)
            batch.set_warm_start_updates(False)
            payload["value_plan_diagnostics"] = [
                diagnostic.to_dict()
                for diagnostic in batch.last_density_fitting_metric_diagnostics()
            ]
            write_result(args.output, payload)
            for repeat in range(args.repeats):
                order = ("dense", "occupied")
                if repeat % 2:
                    order = order[::-1]
                for policy in order:
                    os.environ["VIBEQC_DF_EXCHANGE"] = policy
                    prime = _vibeqc_sample(batch, cp, -1, forces)
                    sample = _vibeqc_sample(batch, cp, len(payload["samples"]), forces)
                    payload["samples"].append(
                        {"policy": policy, "repeat": repeat, "prime": prime, **sample}
                    )
                    write_result(args.output, payload)
    finally:
        if original_policy is None:
            os.environ.pop("VIBEQC_DF_EXCHANGE", None)
        else:
            os.environ["VIBEQC_DF_EXCHANGE"] = original_policy

    errors = {"energies_hartree": 0.0}
    if forces:
        errors["forces_hartree_per_bohr"] = 0.0
    branches_match = True
    for repeat in range(args.repeats):
        pair = [s for s in payload["samples"] if s["repeat"] == repeat]
        assert len(pair) == 2
        branches = []
        for sample in pair:
            for result in (sample, sample["prime"]):
                if not all(c["converged"] for c in result["convergence"]):
                    raise ValueError("unconverged SCF cannot pass exchange parity")
            branches.append(tuple(c["iterations"] for c in sample["convergence"]))
        branches_match &= branches[0] == branches[1]
        for key, previous in errors.items():
            a, b = (np.asarray(s[key]) for s in pair)
            if (
                a.shape != b.shape
                or not a.size
                or not all(np.isfinite(x).all() for x in (a, b))
            ):
                raise ValueError(f"invalid {key} arrays")
            errors[key] = max(previous, float(np.max(np.abs(a - b))))
    medians = {
        policy: statistics.median(
            s["seconds"] for s in payload["samples"] if s["policy"] == policy
        )
        for policy in ("dense", "occupied")
    }
    payload.update(
        maximum_errors=errors,
        iteration_branches_match=branches_match,
        median_seconds=medians,
        dense_over_occupied=medians["dense"] / medians["occupied"],
        passed=errors["energies_hartree"] <= 1e-9
        and errors.get("forces_hartree_per_bohr", 0) <= 1e-8,
    )
    write_result(args.output, payload)
    if not payload["passed"]:
        raise ValueError(f"dense/occupied endpoint numerical gate failed: {errors}")


if __name__ == "__main__":
    main()

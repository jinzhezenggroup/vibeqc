"""Cold, changed-geometry and fixed-geometry validation of native DF value policies.

Run within Slurm. Every arm owns a fresh prepared batch; cold and changed calls
include complete SCF work and setup. Fixed-geometry samples use the same supplied
checkpoint across policies when one is supplied. Raw arrays and component traces
are separate evidence, never substitutes for these complete endpoint samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import convergence_payload
from benchmarks.df_component_ledger import aggregate, read_host_trace, read_trace
from benchmarks.df_policy_endpoint import CASES, independent_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aos", type=int, choices=CASES, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--energy-only", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or args.repeats < 1 or args.budget < 0:
        parser.error("requires Slurm and finite positive repeats/nonnegative budget")
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case = benchmark_cases()[CASES[args.aos]]
    reference = json.loads(args.reference.read_text())
    positions = np.array([r for _, r in case.atoms])
    changed = positions.copy()
    changed[1, 0] += 0.001
    library = Path(os.environ["VIBEQC_LIBRARY"])
    payload = {
        "case": CASES[args.aos],
        "aos": args.aos,
        "budget": args.budget,
        "scope": "intrusive diagnostic" if args.trace else "clean endpoints",
        "basis": case.vibeqc_basis,
        "basis_representation": case.basis_representation,
        "metric_relative_threshold": 1e-10,
        "coordinates_bohr": positions.tolist(),
        "changed_coordinates_bohr": changed.tolist(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_patch_sha256": hashlib.sha256(
            library.with_name("source.patch").read_bytes()
        ).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples": [],
    }
    properties = ("energy",) if args.energy_only else ("energy", "forces")
    payload["properties"] = properties
    changed_reference = None
    for repeat in range(args.repeats):
        for policy in (
            ("generic", "candidate") if repeat % 2 == 0 else ("candidate", "generic")
        ):
            os.environ["VIBEQC_DF_VALUE_MATH"] = policy
            os.environ["VIBEQC_DF_VALUE_RAW_MAPPING"] = (
                "scalar" if policy == "generic" else "candidate"
            )
            calc = Calculator(
                method="rhf",
                basis=case.vibeqc_basis,
                basis_representation=case.basis_representation,
                device="cuda",
                density_fitting="cuda",
                auxiliary_basis=case.vibeqc_basis,
                density_fitting_memory_budget_bytes=args.budget,
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                max_iterations=100,
            )
            start = time.perf_counter()
            with calc.prepare_batch([case.atoms]) as batch:
                prepare_seconds = time.perf_counter() - start
                for phase in ("cold", "changed", "warm", "warm", "warm"):
                    if phase == "changed" and args.checkpoint:
                        batch.load_checkpoint(args.checkpoint, allow_warm=True)
                        batch.set_warm_start_updates(False)
                    if phase == "warm":
                        # Return to the declared original geometry and density.
                        batch.execute([positions], properties=properties, strict=True)
                        if args.checkpoint:
                            batch.load_checkpoint(args.checkpoint, allow_warm=True)
                        batch.set_warm_start_updates(False)
                    suffix = f"{repeat}-{policy}-{phase}-{len(payload['samples'])}"
                    if args.trace:
                        trace = args.output.with_suffix(f".{suffix}.jsonl")
                        host = args.output.with_suffix(f".{suffix}.host.jsonl")
                        os.environ["VIBEQC_DF_TRACE"] = str(trace)
                        os.environ["VIBEQC_DF_HOST_TRACE"] = str(host)
                    start = time.perf_counter()
                    result = batch.execute(
                        [changed if phase == "changed" else positions],
                        properties=properties,
                        strict=True,
                    )
                    seconds = time.perf_counter() - start
                    if phase == "cold":
                        seconds += prepare_seconds
                    item = result.items[0]
                    forces = None if args.energy_only else np.asarray(item.forces)
                    row = {
                        "policy": policy,
                        "repeat": repeat,
                        "phase": phase,
                        "seconds": seconds,
                        "iterations": item.iterations,
                        "convergence": convergence_payload(result),
                        "energy": item.energy,
                        "forces": None if forces is None else forces.tolist(),
                        "metric": [
                            x.to_dict()
                            for x in batch.last_density_fitting_metric_diagnostics()
                        ],
                    }
                    if phase == "changed":
                        if changed_reference is None:
                            changed_reference = (item.energy, forces)
                        expected_energy, expected_forces = changed_reference
                    else:
                        expected_energy, expected_forces = independent_reference(
                            reference,
                            args.aos,
                            result.energies,
                            np.asarray(
                                reference["gpu4pyscf"]["forces_hartree_per_bohr"]
                            )
                            if args.energy_only
                            else forces[None],
                            [item.basis_metadata],
                        )
                        expected_energy = expected_energy[0]
                        if expected_forces is not None:
                            expected_forces = expected_forces[0]
                    row["energy_error"] = abs(item.energy - expected_energy)
                    row["force_error"] = (
                        None
                        if forces is None
                        else float(np.max(np.abs(forces - expected_forces)))
                    )
                    if args.trace:
                        os.environ.pop("VIBEQC_DF_TRACE")
                        os.environ.pop("VIBEQC_DF_HOST_TRACE")
                        row["components"] = aggregate(read_trace(trace))
                        row["host"] = read_host_trace(host)
                    payload["samples"].append(row)
                    args.output.write_text(json.dumps(payload, indent=2) + "\n")
                    print(repeat, policy, phase, seconds, item.iterations, flush=True)
                    if (
                        not np.isfinite(item.energy)
                        or (forces is not None and not np.isfinite(forces).all())
                        or row["energy_error"] > 1e-9
                        or (
                            row["force_error"] is not None and row["force_error"] > 1e-8
                        )
                    ):
                        raise RuntimeError(
                            "strict independent/paired numerical gate failed"
                        )


if __name__ == "__main__":
    main()

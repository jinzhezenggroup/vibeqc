"""Held-out screening curve for separated WATER8, with a fresh PySCF force oracle."""

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
from benchmarks.df_component_ledger import aggregate, read_host_trace, read_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--separation", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or args.repeats < 5
        or not 1 < args.separation <= 10
    ):
        parser.error("requires Slurm, at least five repeats and separation in (1,10]")
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    from pyscf import gto, scf

    case = benchmark_cases()["water-octamer-s4-def2-svp-spherical"]
    coordinates = np.array([r for _, r in case.atoms])
    for begin in range(0, len(coordinates), 3):
        coordinates[begin : begin + 3] += (args.separation - 1) * coordinates[
            begin
        ].copy()
    atoms = [(z, tuple(r)) for (z, _), r in zip(case.atoms, coordinates)]
    mol = gto.M(atom=atoms, basis="def2-svp", unit="Bohr", verbose=0)
    oracle = scf.RHF(mol).density_fit(auxbasis="def2-svp")
    oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-12, 1e-10, 100
    oracle.kernel()
    if not oracle.converged:
        raise RuntimeError("independent reference failed to converge")
    force = -oracle.nuc_grad_method().kernel()
    library = Path(os.environ["VIBEQC_LIBRARY"])
    payload = {
        "case": "water8-separated",
        "aos": 192,
        "separation": args.separation,
        "atoms_bohr": atoms,
        "basis": "def2-svp",
        "representation": "spherical",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "controls": {k: v for k, v in os.environ.items() if k.startswith("VIBEQC_")},
        "independent_energy": oracle.e_tot,
        "independent_forces": force.tolist(),
        "samples": [],
        "diagnostics": [],
    }
    calc = Calculator(
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    os.environ["VIBEQC_DF_FORCE_SCREEN_ABS"] = "off"
    policies = ("off", "1e-8", "1e-6", "1e-4")
    with calc.prepare_batch([atoms]) as batch:
        initial = batch.execute(strict=True).items[0]
        batch.set_warm_start_updates(False)
        payload["cold_energy_error"] = abs(initial.energy - oracle.e_tot)
        strict_force = np.asarray(initial.forces)
        jobs = [
            (False, repeat, p)
            for repeat in range(args.repeats)
            for p in (policies if repeat % 2 == 0 else policies[::-1])
        ]
        jobs += [(True, 0, p) for p in policies]
        for diagnostic, repeat, policy in jobs:
            os.environ["VIBEQC_DF_FORCE_SCREEN_ABS"] = policy
            batch.execute(strict=True)  # Exclude plan transition from clean replay.
            if diagnostic:
                trace = args.output.with_suffix(f".{policy}.jsonl")
                host = args.output.with_suffix(f".{policy}.host.jsonl")
                os.environ.update(
                    VIBEQC_DF_TRACE=str(trace),
                    VIBEQC_DF_HOST_TRACE=str(host),
                    VIBEQC_DF_SHELL_COUNTERS="1",
                )
            start = time.perf_counter()
            actual = batch.execute(strict=True).items[0]
            seconds = time.perf_counter() - start
            observed = np.asarray(actual.forces)
            error = float(np.max(np.abs(observed - strict_force)))
            independent = float(np.max(np.abs(observed - force)))
            row = {
                "policy": policy,
                "repeat": repeat,
                "seconds": seconds,
                "scope": "intrusive diagnostic" if diagnostic else "clean endpoint",
                "iterations": actual.iterations,
                "energy": actual.energy,
                "forces": observed.tolist(),
                "maximum_force_error": error,
                "rms_force_error": float(
                    np.sqrt(np.mean((observed - strict_force) ** 2))
                ),
                "independent_force_error": independent,
                "translation_error": float(np.max(np.abs(observed.sum(axis=0)))),
            }
            if diagnostic:
                row["components"] = aggregate(read_trace(trace))
                row["force_stage_seconds"] = (
                    sum(
                        r["wall_ms"]
                        for x in read_host_trace(host)
                        for r in x["regions"]
                        if r["name"] == "force_response"
                    )
                    / 1000
                )
                for key in (
                    "VIBEQC_DF_TRACE",
                    "VIBEQC_DF_HOST_TRACE",
                    "VIBEQC_DF_SHELL_COUNTERS",
                ):
                    os.environ.pop(key)
            payload["diagnostics" if diagnostic else "samples"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(policy, repeat, seconds, error, flush=True)
            budget = 0 if policy == "off" else float(policy)
            if (
                abs(actual.energy - oracle.e_tot) > 1e-9
                or independent > budget + 1e-8
                or error > budget + 1e-10
                or row["translation_error"] > 1e-8
            ):
                raise RuntimeError("screening force budget or independent gate failed")


if __name__ == "__main__":
    main()

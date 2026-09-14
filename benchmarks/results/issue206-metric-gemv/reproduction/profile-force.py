"""Instrument cold/warm force ownership after the clean endpoint matrix."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import scaled_geometries
from benchmarks.df_component_ledger import (
    aggregate,
    aggregate_host,
    read_host_trace,
    read_trace,
)
from benchmarks.df_progress_ledger import read_progress, summarize_progress

root = Path(".artifacts/issue206-metric-gemv/force-profile")
root.mkdir(exist_ok=False)
results = []
for case_name, aos in [
    ("water-tetramer-def2-svp-spherical", 96),
    ("water-octamer-s4-def2-svp-spherical", 192),
]:
    case = benchmark_cases()[case_name]
    for serial in (False, True):
        budget = 0
        os.environ["VIBEQC_DF_HOST_RESPONSE_WEIGHTS"] = "0"
        os.environ["VIBEQC_DF_SERIAL_RESPONSE_DOT"] = "1" if serial else "0"
        stem = f"{aos}ao-b1-serial{serial}"
        calc = Calculator(
            basis=case.vibeqc_basis,
            basis_representation="spherical",
            device="cuda",
            density_fitting="cuda",
            density_fitting_memory_budget_bytes=budget,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            max_iterations=100,
        )
        with calc.prepare_batch(scaled_geometries(case.atoms, 1)) as batch:
            for stage in ("cold", "warm"):
                base = root / f"{stem}-{stage}"
                for kind, key in [
                    ("cuda", "VIBEQC_DF_TRACE"),
                    ("host", "VIBEQC_DF_HOST_TRACE"),
                    ("progress", "VIBEQC_DF_PROGRESS_TRACE"),
                ]:
                    os.environ[key] = str(
                        base.with_suffix("." + kind + ".jsonl").resolve()
                    )
                start = time.perf_counter()
                result = batch.execute(strict=True, properties=("energy", "forces"))
                seconds = time.perf_counter() - start
                for key in (
                    "VIBEQC_DF_TRACE",
                    "VIBEQC_DF_HOST_TRACE",
                    "VIBEQC_DF_PROGRESS_TRACE",
                ):
                    del os.environ[key]
                row = {
                    "case": case_name,
                    "aos": aos,
                    "batch": 1,
                    "budget": budget,
                    "stage": stage,
                    "serial_metric_dot": serial,
                    "seconds": seconds,
                    "energy": result.items[0].energy,
                    "forces": result.items[0].forces.tolist(),
                    "iterations": result.items[0].iterations,
                    "metric": [
                        d.to_dict()
                        for d in batch.last_density_fitting_metric_diagnostics()
                    ],
                    "host": aggregate_host(
                        read_host_trace(base.with_suffix(".host.jsonl"))
                    ),
                    "cuda": aggregate(read_trace(base.with_suffix(".cuda.jsonl"))),
                    "progress": summarize_progress(
                        read_progress(base.with_suffix(".progress.jsonl"))
                    ),
                }
                results.append(row)
                (root / "summary.json").write_text(
                    json.dumps(
                        {
                            "scope": "Intrusive component attribution, separate from clean matrix",
                            "slurm_job": os.environ["SLURM_JOB_ID"],
                            "git_head": subprocess.check_output(
                                ["git", "rev-parse", "HEAD"], text=True
                            ).strip(),
                            "library_sha256": hashlib.sha256(
                                Path(os.environ["VIBEQC_LIBRARY"]).read_bytes()
                            ).hexdigest(),
                            "results": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(stem, stage, seconds, flush=True)

import numpy as np

checks = []
for aos in (96, 192):
    for stage in ("cold", "warm"):
        rows = [r for r in results if r["aos"] == aos and r["stage"] == stage]
        assert len(rows) == 2
        left, right = rows
        energy_error = abs(left["energy"] - right["energy"])
        force_error = float(
            np.max(np.abs(np.array(left["forces"]) - np.array(right["forces"])))
        )
        check = {
            "aos": aos,
            "stage": stage,
            "maximum_energy_error_hartree": energy_error,
            "maximum_force_error_hartree_per_bohr": force_error,
            "iterations": [left["iterations"], right["iterations"]],
        }
        checks.append(check)
        assert energy_error <= 1e-9 and force_error <= 1e-8, check
        assert left["iterations"] == right["iterations"], check
(root / "ablation-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
(root.parent / "validation-success.json").write_text(
    json.dumps(
        {
            "slurm_job": os.environ["SLURM_JOB_ID"],
            "library_sha256": hashlib.sha256(
                Path(os.environ["VIBEQC_LIBRARY"]).read_bytes()
            ).hexdigest(),
            "profile_ablation_passed": True,
        },
        indent=2,
    )
    + "\n"
)

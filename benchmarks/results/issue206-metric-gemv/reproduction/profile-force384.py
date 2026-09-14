"""Instrument cold/warm force ownership after the clean endpoint matrix."""

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


def sample_memory(stop, samples):
    """Intrusive process samples; they never overlap a clean comparison."""
    while not stop.is_set():
        sample = {"time_unix": time.time()}
        try:
            run = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            sample["gpu_query_returncode"] = run.returncode
            sample["gpu_process_rows"] = [
                line
                for line in run.stdout.splitlines()
                if line.split(",")[0].strip() == str(os.getpid())
            ]
            sample["gpu_query_error"] = run.stderr
            sample["host_status"] = [
                line
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith(("VmHWM:", "VmRSS:"))
            ]
        # Preserve the measured runner's best-effort sampling: query failures
        # are observations and must remain visible in the output record.
        except Exception as error:  # noqa: BLE001
            sample["error"] = repr(error)
        samples.append(sample)
        stop.wait(0.1)


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

root = Path(".artifacts/issue206-metric-gemv/large-force-profile")
root.mkdir(exist_ok=False)
results = []
for case_name, aos in [("water-hexadecamer-2s4-def2-svp-spherical", 384)]:
    case = benchmark_cases()[case_name]
    for host in (False,):
        budget = 0
        os.environ["VIBEQC_DF_HOST_RESPONSE_WEIGHTS"] = "1" if host else "0"
        stem = f"{aos}ao-b1-blas"
        os.environ["VIBEQC_DF_SERIAL_RESPONSE_DOT"] = "0"
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
                samples = []
                stop = threading.Event()
                monitor = threading.Thread(target=sample_memory, args=(stop, samples))
                monitor.start()
                start = time.perf_counter()
                try:
                    result = batch.execute(strict=True, properties=("energy", "forces"))
                    seconds = time.perf_counter() - start
                finally:
                    stop.set()
                    monitor.join()
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
                    "serial_metric_dot": False,
                    "seconds": seconds,
                    "memory_samples": samples,
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

reference = json.loads(
    Path(".artifacts/issue206-metric-gemv/clean/384ao-b1-forces-blas.json").read_text()
)["gpu4pyscf"]
for row in results:
    row["independent_reference_errors"] = {
        "energy_hartree": abs(row["energy"] - reference["energies_hartree"][0]),
        "maximum_force_hartree_per_bohr": float(
            np.max(
                np.abs(
                    np.array(row["forces"])
                    - np.array(reference["forces_hartree_per_bohr"][0])
                )
            )
        ),
    }
    assert row["independent_reference_errors"]["energy_hartree"] < 1e-9
    assert row["independent_reference_errors"]["maximum_force_hartree_per_bohr"] < 1e-8
payload = json.loads((root / "summary.json").read_text())
payload["results"] = results
payload["scope"] = (
    "Intrusive component attribution and sampled process memory; separate from every clean endpoint. Host high-water values are cumulative for this process."
)
(root / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

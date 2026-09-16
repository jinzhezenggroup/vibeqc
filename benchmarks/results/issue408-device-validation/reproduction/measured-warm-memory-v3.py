"""Sample warm DF endpoint memory separately from all clean timing samples."""

import argparse
import ctypes
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from vibeqc import Calculator, _native

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import convergence_payload
from benchmarks.df_policy_endpoint import CASES, independent_reference

p = argparse.ArgumentParser()
p.add_argument("--aos", type=int, choices=(384, 768), required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
a = p.parse_args()
assert os.environ.get("SLURM_JOB_ID")
case = benchmark_cases()[CASES[a.aos]]
reference_path = Path("benchmarks/results/issue377-379-df/gpu4pyscf") / (
    CASES[a.aos] + ".json"
)
library = Path(os.environ["VIBEQC_LIBRARY"])
lib = _native.load_library()
lib.vibeqc_get_source_identity.restype = ctypes.c_char_p
payload = {
    "aos": a.aos,
    "case": CASES[a.aos],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "native_source_identity": lib.vibeqc_get_source_identity().decode(),
    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "scope": "Intrusive diagnostics, excluded from clean timing. Warm device process peaks use nvidia-smi samples at >=50 ms intervals and may miss shorter transients. Host resident samples use /proc/self/statm. Lifetime host high-water includes initialization. Each arm is primed outside sampling; resident readings retain the live prepared owner.",
    "samples": [],
}


def memory():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    device = {
        int(pid): int(mib) * 1024**2
        for line in raw.splitlines()
        for pid, mib in [line.split(",")]
    }[os.getpid()]
    host = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
        "SC_PAGE_SIZE"
    )
    return device, host


calc = Calculator(
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
with calc.prepare_batch([case.atoms]) as batch:
    payload["restore"] = batch.load_checkpoint(a.checkpoint, allow_warm=True)
    batch.set_warm_start_updates(False)
    initial = batch.execute(strict=True, properties=("energy", "forces"))
    energy, force = independent_reference(
        json.loads(reference_path.read_text()),
        a.aos,
        initial.energies,
        np.array([item.forces for item in initial.items]),
        [item.basis_metadata for item in initial.items],
    )
    for properties in [("energy",), ("energy", "forces")]:
        for policy in ["1", "0"]:
            os.environ["VIBEQC_DF_REFERENCE_FINAL_VALIDATION"] = policy
            batch.execute(strict=True, properties=properties)
            name = f"{a.aos}-{policy}-" + (
                "forces" if len(properties) == 2 else "energy"
            )
            prefix = a.output.parent / name
            os.environ["VIBEQC_DF_TRACE"] = str(prefix) + ".memory.cuda.jsonl"
            os.environ["VIBEQC_DF_HOST_TRACE"] = str(prefix) + ".memory.host.jsonl"
            os.environ["VIBEQC_DF_PROGRESS_TRACE"] = (
                str(prefix) + ".memory.journal.jsonl"
            )
            samples = []
            failures = []
            stop = threading.Event()

            def sample(stop=stop, samples=samples, failures=failures):
                try:
                    while not stop.is_set():
                        device, host = memory()
                        samples.append([time.monotonic_ns(), device, host])
                        stop.wait(0.05)
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    subprocess.SubprocessError,
                ) as error:
                    failures.append(str(error))

            worker = threading.Thread(target=sample)
            before = memory()
            worker.start()
            try:
                result = batch.execute(strict=True, properties=properties)
            finally:
                stop.set()
                worker.join()
                for variable in [
                    "VIBEQC_DF_TRACE",
                    "VIBEQC_DF_HOST_TRACE",
                    "VIBEQC_DF_PROGRESS_TRACE",
                ]:
                    os.environ.pop(variable, None)
            after = memory()
            assert samples and not failures, failures
            error_energy = float(np.max(np.abs(result.energies - energy)))
            error_force = (
                float(
                    np.max(np.abs(np.array([i.forces for i in result.items]) - force))
                )
                if len(properties) == 2
                else None
            )
            row = {
                "properties": properties,
                "policy": policy,
                "iterations": [i.iterations for i in result.items],
                "convergence": convergence_payload(result),
                "maximum_energy_error": error_energy,
                "maximum_force_error": error_force,
                "prepared_resident_before": {"device": before[0], "host": before[1]},
                "prepared_resident_after": {"device": after[0], "host": after[1]},
                "sampled_warm_device_peak_bytes": max(
                    before[0], after[0], *(s[1] for s in samples)
                ),
                "sampled_warm_host_peak_bytes": max(
                    before[1], after[1], *(s[2] for s in samples)
                ),
                "lifetime_host_peak_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
                "memory_samples": samples,
            }
            payload["samples"].append(row)
            a.output.write_text(json.dumps(payload, indent=2) + "\n")
            assert row["iterations"] == [3]
            assert error_energy <= 1e-9 and (error_force is None or error_force <= 1e-8)
            print(
                name,
                row["sampled_warm_device_peak_bytes"],
                row["sampled_warm_host_peak_bytes"],
                flush=True,
            )

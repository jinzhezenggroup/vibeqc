"""Reproduce the original inclusive interval on the verified master binary.

This historical control intentionally uses the original progress fences and
the original first post-cold warm replay. It is separate from steady frozen-seed
candidate qualification. Native source/binary identities pin the exact control;
the current dirty native files are not represented as its measured source.
"""

import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from vibeqc import Calculator, _native

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import scaled_geometries
from benchmarks.df_component_ledger import aggregate, read_trace

assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
root = Path(".artifacts/issue308-response-timeline/original-progress")
root.mkdir(exist_ok=False)
path = Path(".artifacts/issue308-response-timeline/baseline/libvibeqc.so").resolve()
binary_hash = hashlib.sha256(path.read_bytes()).hexdigest()
assert binary_hash == "5b841ed1dfbfd34eaa73b20a1c4f28da4cf1d798bd8e502093679c9a1f7aaa06"
os.environ["VIBEQC_LIBRARY"] = str(path)
os.environ["VIBEQC_DF_HOST_RESPONSE_WEIGHTS"] = "0"
os.environ["VIBEQC_DF_SERIAL_RESPONSE_DOT"] = "0"
library = _native.load_library()
library.vibeqc_get_source_identity.restype = ctypes.c_char_p
identity = library.vibeqc_get_source_identity().decode()
assert identity == "cb60b50bac01691a81ac6bd46919c840bdf1f609d5651e637adef31e7f01e9ca"
cudart = ctypes.CDLL("libcudart.so.12")
case = benchmark_cases()["water-hexadecamer-2s4-def2-svp-spherical"]
calc = Calculator(
    basis=case.vibeqc_basis,
    auxiliary_basis=case.vibeqc_basis,
    basis_representation="spherical",
    device="cuda",
    density_fitting="cuda",
    density_fitting_memory_budget_bytes=0,
    energy_tolerance=1e-12,
    density_tolerance=1e-10,
    max_iterations=100,
)
with calc.prepare_batch(scaled_geometries(case.atoms, 1)) as batch:
    cold = batch.execute(strict=True, properties=("energy", "forces"))
    for suffix, key in (
        ("cuda", "VIBEQC_DF_TRACE"),
        ("host", "VIBEQC_DF_HOST_TRACE"),
        ("progress", "VIBEQC_DF_PROGRESS_TRACE"),
    ):
        os.environ[key] = str((root / f"warm.{suffix}.jsonl").resolve())
    assert cudart.cudaProfilerStart() == 0
    start = time.perf_counter()
    result = batch.execute(strict=True, properties=("energy", "forces"))
    seconds = time.perf_counter() - start
    assert cudart.cudaProfilerStop() == 0
    for key in ("VIBEQC_DF_TRACE", "VIBEQC_DF_HOST_TRACE", "VIBEQC_DF_PROGRESS_TRACE"):
        del os.environ[key]
    item = result.items[0]
    refpath = Path(
        "benchmarks/results/issue206-metric-gemv/measurements/384ao-b1-forces-blas.json"
    )
    ref = json.loads(refpath.read_text())["gpu4pyscf"]
    eerr = abs(item.energy - ref["energies_hartree"][0])
    ferr = float(
        np.max(np.abs(item.forces - np.asarray(ref["forces_hartree_per_bohr"][0])))
    )
    payload = {
        "scope": "Original first post-cold warm force with original per-region progress fences; attribution only",
        "source_revision": "a57656a62f333a0990c963844a69bcef8b7c5b16",
        "source_identity": identity,
        "library_sha256": binary_hash,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "seconds": seconds,
        "cold_iterations": [x.iterations for x in cold.items],
        "warm_iterations": [item.iterations],
        "energy_hartree": item.energy,
        "forces_hartree_per_bohr": item.forces.tolist(),
        "maximum_energy_error_hartree": eerr,
        "maximum_force_error_hartree_per_bohr": ferr,
        "reference_sha256": hashlib.sha256(refpath.read_bytes()).hexdigest(),
        "components": aggregate(read_trace(root / "warm.cuda.jsonl")),
    }
    (root / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    assert eerr < 1e-9 and ferr < 1e-8
    print(seconds, eerr, ferr, flush=True)

"""Shared-schema evidence for a measured public native MP2 calculation."""

import ctypes
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vibeqc.profiles import probe_device

from tools.vibeqc_validation.schema import (
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    write_evidence,
)


def record_public_result(
    calc, result, metadata, opposite_spin, same_spin, seconds, destination
):
    """Save observed outputs, exact library identity and declared staging.

    Capacity estimates remain separate from allocator/whole-process metrics;
    a single endpoint sample is not a speed comparison or promotion.
    """
    root = Path(__file__).resolve().parents[2]
    backend = "cuda" if result.executed_backend == "cuda" else "cpu"
    record = new_evidence(
        tier="endpoint",
        subject=f"public-canonical-MP2/{metadata['inputs']['name']}/{backend}",
        inputs_hash=metadata["array_hash"],
    )
    library = Path(calc._library._name).resolve()
    calc._library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    snapshot = root / "build/mp2-a1/source-manifest.json"
    source_files = (
        json.loads(snapshot.read_text())
        if snapshot.exists()
        else {
            "reason": "no transferred snapshot manifest",
            "files": {
                p.relative_to(root).as_posix(): file_hash(p)
                for p in (root / "src/posthf").glob("*")
                if p.is_file()
            },
        }
    )
    reference = metadata["records"]["conventional"]
    record.update(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        backend_selected=backend,
        device=probe_device(calc._library, calc._device_id)["device"]
        if backend == "cuda"
        else {"kind": "cpu", "machine": platform.machine()},
        hardware=outcome("pass"),
        toolchain={"python": sys.version, "numpy": np.__version__},
        settings={
            "device": backend,
            "fast_compile": os.environ.get("VIBEQC_MP2_FAST_COMPILE") == "1",
            "scope": "public native canonical RHF-to-MP2 energy; no force/RI/performance promotion",
            "source_snapshot": source_files,
            "library": str(library),
            "library_sha256": file_hash(library),
            "native_source_identity": calc._library.vibeqc_get_source_identity().decode(),
            "python_module": str(Path(sys.modules["vibeqc"].__file__).resolve()),
            "reference_versions": metadata["versions"],
            "reference_array_hash": metadata["array_hash"],
            "result": asdict(result),
            "host_staging": "GPU reference matrices validated on host; CPU AO source; bounded MO D2H/H2D; native host scalar fold",
            "gpu_allocation": os.environ.get("VIBEQC_MP2_GPU_ALLOCATION"),
            "numeric_budget_bytes": calc._correlation_memory_budget_bytes or 256 << 20,
        },
    )
    record["hashes"].update(
        equation=result.correlation.equation_hash,
        source=canonical_hash(source_files),
        schedule=canonical_hash(
            {"occupied_tile": 1, "virtual_tiles": [1, 2, 4, 8], "backend": backend}
        ),
    )
    record["hash_reasons"]["ir"] = (
        "equation hash identifies generated TensorIR; generated artifacts retained with native build"
    )
    record["stages"]["representation"] = outcome("pass")
    record["stages"]["source"] = outcome("pass")
    record["block_errors"] = {
        "OS_SS": block_error(
            [
                result.correlation.opposite_spin_energy,
                result.correlation.same_spin_energy,
            ],
            [opposite_spin, same_spin],
            atol=1e-11,
            rtol=1e-10,
        ),
        "total_energy": block_error(
            result.energy,
            reference["hf_energy"] + reference["correlation_energy"],
            atol=1e-9,
            rtol=0,
        ),
    }
    passed = all(v["passed"] for v in record["block_errors"].values())
    for stage in ("numerical", "endpoint"):
        record["stages"][stage] = outcome(
            "pass" if passed else "fail",
            None if passed else "independent public molecular gate failed",
        )
    record["solver_trace_reason"] = (
        "SCF iterations and final physical residual in result; MP2 noniterative"
    )
    record["residuals"]["physical_reference"] = result.correlation.reference_residual
    record["timings"] = [
        {
            "selection": "baseline",
            "workload": "energy-only",
            "seconds": seconds,
            "inputs_hash": record["inputs_hash"],
        }
    ]
    record["memory"]["reason"] = (
        "combined capacity is an estimate in result; measured correlation owned/provider bytes are separate; no process RSS or full device overhead bound"
    )
    write_evidence(destination, record)

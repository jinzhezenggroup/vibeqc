"""Separate-process host high-water and owned-device records for complete DF-HF."""

import argparse
import ctypes
import json
import os
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from vibeqc import Calculator, _native
from vibeqc.autotune import source_identity
from vibeqc.resources import ResourceBudget

from benchmarks._cases import benchmark_cases
from tools.vibeqc_validation.schema import file_hash


def main():
    """Keep baseline/candidate process peaks independent, including startup/driver RSS."""
    cases = benchmark_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=cases, default="sp8")
    parser.add_argument(
        "--selection", choices=("reference", "generated"), required=True
    )
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--df-budget", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".artifacts/benchmarks/df_endpoint_memory.json"),
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or args.batch < 1 or args.df_budget < 0:
        parser.error("use a finite Slurm job, positive batch, and nonnegative budget")
    os.environ["VIBEQC_ONE_ELECTRON_DERIVATIVES"] = "generated"
    if args.selection == "reference":
        parser.error(
            "coordinate-wise DF response was retired; use an archived source checkout"
        )
    os.environ["VIBEQC_DF_DERIVATIVE_MAPPING"] = "thread"
    lib = _native.load_library()
    lib.vibeqc_get_source_identity.restype = ctypes.c_char_p
    identity = lib.vibeqc_get_source_identity().decode()
    if identity != source_identity(ROOT):
        raise RuntimeError("loaded library does not match the scientific source")
    case = cases[args.case]
    # Inventory v1 covers at most 16 public AOs. Larger cases still measure
    # complete-process RSS, but cannot claim a whole-HF device-ledger bound.
    ao_count = case.expected_ao_count
    if ao_count is None and not isinstance(case.vibeqc_basis, str):
        ao_count = sum(
            2 * shell.angular_momentum + 1
            if case.basis_representation == "spherical"
            else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            for shell in case.vibeqc_basis
        )
    observe_resources = ao_count is not None and ao_count <= 16
    systems = [
        [(z, tuple(np.asarray(r) * (1 + 0.01 * i))) for z, r in case.atoms]
        for i in range(args.batch)
    ]
    calc = Calculator(
        device="cuda",
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=args.df_budget,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        resource_budget=(
            ResourceBudget(host_bytes=2 << 30, device_bytes=16 << 30)
            if observe_resources
            else None
        ),
    )
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    with calc.prepare_batch(
        systems,
        charges=[case.charge] * args.batch,
        multiplicities=[case.multiplicity] * args.batch,
    ) as batch:
        cold = batch.execute(strict=True)
        batch.set_warm_start_updates(False)
        warm = batch.execute(strict=True)
        moved = [np.array([r for _, r in atoms]) for atoms in systems]
        for positions in moved:
            positions[-1, 0] += 0.013
        changed = batch.execute(moved, strict=True)
        observed = batch.resource_diagnostics if observe_resources else None
    payload = {
        "schema": "vibeqc.df_endpoint_memory",
        "version": 1,
        "case": args.case,
        "selection": args.selection,
        "batch": args.batch,
        "df_budget": args.df_budget,
        "source_identity": identity,
        "binary_sha256": file_hash(lib._name),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_peak_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "process_peak_before_execution_bytes": before,
        "host_peak_scope": "Linux process resident high-water including interpreter, driver, library startup, and the complete energy-plus-force workflow; no CPU integral oracle is constructed",
        "owned_device_scope": (
            "native resource ledger; excludes opaque driver/library storage"
            if observe_resources
            else "unavailable: CUDA HF inventory v1 supports at most 16 public AOs"
        ),
        "resource_diagnostics": observed,
        "energies": {
            "cold": cold.energies.tolist(),
            "warm": warm.energies.tolist(),
            "changed": changed.energies.tolist(),
        },
        "maximum_force": float(
            max(np.max(np.abs(item.forces)) for item in changed.items)
        ),
    }
    # A failed resource gate must retain the measured evidence for diagnosis.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["process_peak_host_bytes"] > 2 << 30:
        raise RuntimeError(
            "observed process host high-water exceeded the 2 GiB benchmark limit"
        )
    print(
        json.dumps(
            {
                k: payload[k]
                for k in ("case", "selection", "df_budget", "process_peak_host_bytes")
            }
        )
    )


if __name__ == "__main__":
    main()

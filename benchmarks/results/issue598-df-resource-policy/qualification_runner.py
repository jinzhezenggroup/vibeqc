#!/usr/bin/env python3
"""Real-NVIDIA qualification for issue #598 automatic DF resource policy."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_trace
from benchmarks.df_policy_endpoint import cpu_reference, endpoint_errors

MIB = 1024**2
GIB = 1024**3
RESOURCE_KEYS = {
    "resource_policy_version",
    "resolved_total_budget_bytes",
    "resolved_value_budget_bytes",
    "resolved_response_budget_bytes",
    "resource_reserved_headroom_bytes",
    "resource_observed_free_bytes",
    "resource_observed_total_bytes",
    "resource_probe_live",
}

OUT = Path(os.environ["VIBEQC_ISSUE598_EVIDENCE"]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "qualification.json"

runtime = ctypes.CDLL("libcudart.so.12")
runtime.cudaMemGetInfo.argtypes = [
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
runtime.cudaMemGetInfo.restype = ctypes.c_int
runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
runtime.cudaMalloc.restype = ctypes.c_int
runtime.cudaFree.argtypes = [ctypes.c_void_p]
runtime.cudaFree.restype = ctypes.c_int
runtime.cudaDeviceSynchronize.argtypes = []
runtime.cudaDeviceSynchronize.restype = ctypes.c_int


def check(status: int, label: str) -> None:
    if status:
        raise RuntimeError(f"{label} failed with CUDA status {status}")


def mem_info() -> tuple[int, int]:
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    check(runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
    return int(free.value), int(total.value)


def malloc(size: int) -> ctypes.c_void_p:
    pointer = ctypes.c_void_p()
    check(runtime.cudaMalloc(ctypes.byref(pointer), size), f"cudaMalloc({size})")
    return pointer


def free_all(pointers: list[ctypes.c_void_p]) -> None:
    for pointer in reversed(pointers):
        check(runtime.cudaFree(pointer), "cudaFree")
    if pointers:
        check(runtime.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


def hold_until_free(
    target_free: int,
) -> tuple[list[ctypes.c_void_p], int, list[dict[str, int]]]:
    pointers: list[ctypes.c_void_p] = []
    allocated = 0
    snapshots: list[dict[str, int]] = []
    while True:
        free, total = mem_info()
        snapshots.append(
            {"free_bytes": free, "total_bytes": total, "allocated_bytes": allocated}
        )
        excess = free - target_free
        if excess <= 2 * MIB:
            return pointers, allocated, snapshots
        requested = min(1 * GIB, max(1 * MIB, excess - 1 * MIB))
        size = requested
        while size >= 1 * MIB:
            pointer = ctypes.c_void_p()
            status = runtime.cudaMalloc(ctypes.byref(pointer), size)
            if status == 0:
                pointers.append(pointer)
                allocated += size
                break
            size //= 2
        else:
            raise RuntimeError(
                f"could not apply live-memory pressure: free={free} target={target_free}"
            )


def process_memory_bytes() -> int | None:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in text.splitlines():
        pieces = [part.strip() for part in line.split(",")]
        if len(pieces) == 2 and int(pieces[0]) == os.getpid():
            return int(pieces[1]) * MIB
    return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(batch: Any) -> dict[str, Any]:
    rows = [entry.to_dict() for entry in batch.last_density_fitting_metric_diagnostics()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one DF diagnostic, got {len(rows)}")
    return rows[0]


def progress_policy(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        key = record.get("key")
        if record.get("event") == "VALUE" and key in RESOURCE_KEYS:
            values[key] = record.get("value")
    missing = RESOURCE_KEYS - values.keys()
    if missing:
        raise RuntimeError(f"progress trace missing resource fields: {sorted(missing)}")
    return values


def response_scratch(path: Path) -> int:
    records = [row for row in read_trace(path) if row["operation"] == "force_response"]
    if not records:
        raise RuntimeError("force response trace is missing")
    return max(int(row["counters"].get("response_scratch_bytes", 0)) for row in records)


def policy_identity(policy: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(policy[key] for key in sorted(RESOURCE_KEYS))


def execute(
    batch: Any,
    *,
    label: str,
    reference_energy: np.ndarray,
    reference_force: np.ndarray,
) -> dict[str, Any]:
    trace = OUT / f"{label}.trace.jsonl"
    progress = OUT / f"{label}.progress.jsonl"
    for path in (trace, progress):
        path.unlink(missing_ok=True)
    os.environ["VIBEQC_DF_TRACE"] = str(trace)
    os.environ["VIBEQC_DF_PROGRESS_TRACE"] = str(progress)
    try:
        result = batch.execute(strict=True, properties=("energy", "forces"))
        check(runtime.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
    finally:
        os.environ.pop("VIBEQC_DF_TRACE", None)
        os.environ.pop("VIBEQC_DF_PROGRESS_TRACE", None)

    forces = np.asarray([item.forces for item in result.items])
    energy_error, force_error = endpoint_errors(
        result.energies, forces, reference_energy, reference_force
    )
    if energy_error > 1e-9 or force_error is None or force_error > 1e-8:
        raise RuntimeError(
            f"{label} independent endpoint gate failed: "
            f"energy={energy_error}, force={force_error}"
        )
    row = metric(batch)
    policy = progress_policy(progress)
    total = int(policy["resolved_total_budget_bytes"])
    split_total = int(policy["resolved_value_budget_bytes"]) + int(
        policy["resolved_response_budget_bytes"]
    )
    if total != split_total:
        raise RuntimeError(f"{label} resolved value/response split does not sum to total")
    if int(policy["resource_probe_live"]) != 1:
        raise RuntimeError(f"{label} did not use the live CUDA resource probe")
    peak_device = int(row["peak_device_bytes"])
    value_budget = int(policy["resolved_value_budget_bytes"])
    if peak_device > value_budget:
        raise RuntimeError(
            f"{label} DF value peak {peak_device} exceeds "
            f"resolved value budget {value_budget}"
        )
    scratch = response_scratch(trace)
    response_budget = int(policy["resolved_response_budget_bytes"])
    if scratch > response_budget:
        raise RuntimeError(
            f"{label} response scratch {scratch} exceeds "
            f"resolved response budget {response_budget}"
        )
    return {
        "label": label,
        "energy_hartree": result.energies.tolist(),
        "forces_hartree_per_bohr": forces.tolist(),
        "iterations": [item.iterations for item in result.items],
        "energy_error_hartree": energy_error,
        "force_error_hartree_per_bohr": force_error,
        "metric": row,
        "resource_policy": policy,
        "resolved_total_budget_bytes": total,
        "response_scratch_bytes": scratch,
        "process_device_resident_bytes": process_memory_bytes(),
        "actual_free_after_endpoint_bytes": mem_info()[0],
        "trace_sha256": sha256(trace),
        "progress_sha256": sha256(progress),
    }


def run_envelope(
    case_name: str,
    reference_energy: np.ndarray,
    reference_force: np.ndarray,
    *,
    mode: str,
    roomy_total: int | None = None,
) -> dict[str, Any]:
    case = benchmark_cases()[case_name]
    holders: list[ctypes.c_void_p] = []
    pressure: dict[str, Any] = {"mode": mode}
    before_free, total_device = mem_info()
    pressure["free_before_pressure_bytes"] = before_free
    pressure["total_device_bytes"] = total_device

    if mode == "constrained":
        if roomy_total is None:
            raise ValueError("constrained run requires roomy_total")
        headroom = max(256 * MIB, total_device // 8)
        desired = max(2 * MIB, math.floor(roomy_total * 0.90))
        target_free = headroom + math.ceil(desired * 4 / 3)
        holders, allocated, snapshots = hold_until_free(target_free)
        pressure.update(
            {
                "policy_headroom_bytes": headroom,
                "desired_resolved_budget_bytes": desired,
                "target_free_bytes": target_free,
                "external_pressure_bytes": allocated,
                "allocation_snapshots": snapshots,
            }
        )
    pressure["free_before_prepare_bytes"] = mem_info()[0]

    try:
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
        with calculator.prepare_batch(
            [case.atoms],
            charges=[case.charge],
            multiplicities=[case.multiplicity],
            warm_start=True,
        ) as batch:
            cold = execute(
                batch,
                label=f"{case_name}-{mode}-cold",
                reference_energy=reference_energy,
                reference_force=reference_force,
            )
            batch.set_warm_start_updates(False)
            identity = policy_identity(cold["resource_policy"])
            actual_free_before_perturb = mem_info()[0]
            post_prepare: list[ctypes.c_void_p] = []
            if mode == "roomy":
                perturb = min(
                    2 * GIB, max(256 * MIB, actual_free_before_perturb // 16)
                )
                post_prepare.append(malloc(perturb))
                check(runtime.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
                actual_free_after_perturb = mem_info()[0]
                if actual_free_before_perturb - actual_free_after_perturb < 128 * MIB:
                    raise RuntimeError("post-prepare memory perturbation was too small")
            else:
                actual_free_after_perturb = actual_free_before_perturb

            try:
                warm1 = execute(
                    batch,
                    label=f"{case_name}-{mode}-warm1",
                    reference_energy=reference_energy,
                    reference_force=reference_force,
                )
                warm2 = execute(
                    batch,
                    label=f"{case_name}-{mode}-warm2",
                    reference_energy=reference_energy,
                    reference_force=reference_force,
                )
            finally:
                free_all(post_prepare)

            for warm in (warm1, warm2):
                if policy_identity(warm["resource_policy"]) != identity:
                    raise RuntimeError(
                        f"{case_name}/{mode} warm replay changed frozen "
                        "DF resource policy"
                    )
            if mode == "roomy":
                observed = int(cold["resource_policy"]["resource_observed_free_bytes"])
                if int(warm1["resource_policy"]["resource_observed_free_bytes"]) != observed:
                    raise RuntimeError("warm replay re-probed live free memory")
                if (
                    actual_free_after_perturb
                    >= actual_free_before_perturb - 128 * MIB
                ):
                    raise RuntimeError(
                        "actual live free memory did not change for freeze proof"
                    )

            return {
                "case": case_name,
                "method": case.method,
                "charge": case.charge,
                "multiplicity": case.multiplicity,
                "expected_ao_count": case.expected_ao_count,
                "pressure": pressure,
                "actual_free_before_post_prepare_perturb_bytes": (
                    actual_free_before_perturb
                ),
                "actual_free_after_post_prepare_perturb_bytes": (
                    actual_free_after_perturb
                ),
                "cold": cold,
                "warm1": warm1,
                "warm2": warm2,
            }
    finally:
        free_all(holders)


def main() -> None:
    # The qz platform exposes this campaign as a finite Inspire GPU batch job,
    # not necessarily as Slurm.  Real-device identity is asserted below via
    # nvidia-smi/CUDA runtime and retained together with the platform job record.
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "issue": 598,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines(),
        "library": str(library),
        "library_sha256": sha256(library),
        "allocation_kind": "inspire-gpu-batch-job",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "nvcc_version": subprocess.check_output(
            ["nvcc", "--version"], text=True
        ).splitlines(),
        "results": [],
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n")

    case_names = (
        "water-tetramer-def2-svp-spherical",
        "oh-def2-svp-spherical-uhf",
    )
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case_name in case_names:
        case = benchmark_cases()[case_name]
        energy, force, metadata = cpu_reference(
            case, case.pyscf_basis, case.pyscf_basis
        )
        references[case_name] = (energy, force)
        payload.setdefault("cpu_references", {})[case_name] = metadata
        SUMMARY.write_text(json.dumps(payload, indent=2) + "\n")

    for case_name in case_names:
        reference_energy, reference_force = references[case_name]
        roomy = run_envelope(
            case_name, reference_energy, reference_force, mode="roomy"
        )
        payload["results"].append(roomy)
        SUMMARY.write_text(json.dumps(payload, indent=2) + "\n")
        constrained = run_envelope(
            case_name,
            reference_energy,
            reference_force,
            mode="constrained",
            roomy_total=int(roomy["cold"]["resolved_total_budget_bytes"]),
        )
        if int(constrained["cold"]["resolved_total_budget_bytes"]) >= int(
            roomy["cold"]["resolved_total_budget_bytes"]
        ):
            raise RuntimeError(
                f"{case_name} constrained live envelope did not reduce "
                "resolved budget"
            )
        payload["results"].append(constrained)
        SUMMARY.write_text(json.dumps(payload, indent=2) + "\n")

    payload["qualification"] = {
        "real_cuda_probe": True,
        "rhf_force_endpoint": True,
        "uhf_force_endpoint": True,
        "roomy_and_constrained": True,
        "value_peak_within_budget": True,
        "response_scratch_within_budget": True,
        "warm_policy_frozen_after_actual_free_memory_change": True,
        "independent_energy_force_gates": True,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["qualification"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            existing = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
            existing["failure"] = failure
            SUMMARY.write_text(json.dumps(existing, indent=2) + "\n")
        finally:
            print(json.dumps(failure, indent=2), file=sys.stderr)
        raise

"""Capture complete warm DF forces, with separate clean and intrusive runs.

Use Nsight's cudaProfilerApi capture range to exclude preparation and priming.
The component trace is optional; the progress trace is deliberately prohibited
because it fences each region and changes pageable-copy/stream interactions.
This runner checks every complete force against the retained independent #206
reference. It never interprets inclusive host copy time as DMA duration.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from vibeqc import Calculator, _native
from vibeqc.autotune import source_identity

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import scaled_geometries
from benchmarks.df_component_ledger import (
    aggregate,
    aggregate_host,
    read_host_trace,
    read_trace,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    19: "oh-def2-svp-spherical-uhf",
    192: "water-octamer-s4-def2-svp-spherical",
    384: "water-hexadecamer-2s4-def2-svp-spherical",
}
CANDIDATES = {
    "generic": ("generic", "warp", "scalar", "pageable"),
    "shell-warp": ("shell", "warp", "scalar", "pageable"),
    "shell-packed": ("shell", "packed", "scalar", "pageable"),
    "shell-compact": ("shell", "compact", "scalar", "pageable"),
    "blas": ("generic", "warp", "blas", "pageable"),
    "pinned": ("generic", "warp", "blas", "pinned-panels"),
    "combined-warp": ("shell", "warp", "blas", "pinned-panels"),
    "combined-packed": ("shell", "packed", "blas", "pinned-panels"),
    "combined-compact": ("shell", "compact", "blas", "pinned-panels"),
}
CANDIDATE_CONTROLS = (
    "VIBEQC_DF_WEIGHTED_EXECUTION",
    "VIBEQC_DF_SHELL_SCHEDULE",
    "VIBEQC_DF_RESPONSE_ALGEBRA",
    "VIBEQC_DF_RAW_STAGING",
)


def main():
    """Require source/binary agreement and fresh artifacts before any GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aos", type=int, choices=CASES, default=384)
    parser.add_argument("--batch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-trace", action="store_true")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument(
        "--candidate",
        action="append",
        choices=CANDIDATES,
        help="Sweep named force consumers on one fixed post-cold density; repeat to select order",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.nsys and args.repeats != 1:
        parser.error("capture one warm replay per profiler invocation")
    if args.nsys and args.candidate and len(args.candidate) != 1:
        parser.error("capture one candidate per profiler invocation")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("run inside a finite Slurm GPU allocation")
    for key in ("VIBEQC_DF_TRACE", "VIBEQC_DF_HOST_TRACE", "VIBEQC_DF_PROGRESS_TRACE"):
        if os.environ.get(key):
            parser.error(f"unset ambient {key}; select instrumentation explicitly")
    if args.candidate:
        if any(
            os.environ.get(k)
            for k in (
                "VIBEQC_DF_RESPONSE_UPLOAD_PROBE",
                "VIBEQC_DF_RESPONSE_SCATTER_PROBE",
            )
        ):
            parser.error("candidate sweeps cannot be combined with attribution probes")
        os.environ.update(
            zip(CANDIDATE_CONTROLS, CANDIDATES[args.candidate[0]], strict=True)
        )
    args.output.mkdir(parents=True, exist_ok=False)
    library_path = args.library.resolve()
    os.environ["VIBEQC_LIBRARY"] = str(library_path)
    library = _native.load_library()
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    identity = library.vibeqc_get_source_identity().decode()
    if identity != source_identity(ROOT):
        raise RuntimeError(
            "native library does not match the current scientific source"
        )
    reference_path = (
        ROOT
        / "benchmarks/results/issue206-metric-gemv/measurements"
        / f"{args.aos}ao-b{args.batch}-forces-blas.json"
    )
    reference = json.loads(reference_path.read_text())
    case = benchmark_cases()[CASES[args.aos]]
    geometries = scaled_geometries(case.atoms, args.batch)
    inputs = reference["workload"]
    serialized_geometries = [
        [
            {"element": element, "coordinates_bohr": list(position)}
            for element, position in atoms
        ]
        for atoms in geometries
    ]
    if inputs["geometries"] != serialized_geometries:
        raise RuntimeError("retained independent reference has different coordinates")
    expected_settings = {
        "case": CASES[args.aos],
        "ao_count": args.aos,
        "batch_size": args.batch,
        "method": case.method,
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "basis_representation": case.basis_representation,
        "auxiliary_basis": "same as orbital basis",
        "density_fitting_relative_threshold": 1e-10,
        "density_tolerance": 1e-10,
        "energy_tolerance": 1e-12,
        "max_iterations": 100,
        "vibeqc_screening_tolerance": 1e-12,
    }
    if any(inputs[key] != value for key, value in expected_settings.items()):
        raise RuntimeError(
            "retained independent reference has different scientific settings"
        )
    payload = {
        "scope": "intrusive attribution"
        if args.nsys or args.component_trace
        else "clean endpoint",
        "case": CASES[args.aos],
        "aos": args.aos,
        "batch": args.batch,
        "source_identity": identity,
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "library": str(library_path),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "controls": {k: v for k, v in os.environ.items() if k.startswith("VIBEQC_")},
        "reference": str(reference_path.relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "scientific_settings": expected_settings,
        "warm_start_policy": "fixed post-cold density, one untimed replay before measurement",
        "samples": [],
        "candidate_order": args.candidate,
    }
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
    (args.output / "source.patch").write_bytes(patch)
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    payload["source_patch_sha256"] = hashlib.sha256(patch).hexdigest()
    cudart = ctypes.CDLL("libcudart.so.12") if args.nsys else None

    def save():
        (args.output / "result.json").write_text(json.dumps(payload, indent=2) + "\n")

    save()
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
        geometries,
        charges=[case.charge] * args.batch,
        multiplicities=[case.multiplicity] * args.batch,
    ) as batch:
        # Prime the same prepared state; cold work is outside every timed sample.
        cold = batch.execute(strict=True, properties=("energy", "forces"))
        payload["cold_iterations"] = [item.iterations for item in cold.items]
        batch.set_warm_start_updates(False)
        prime = batch.execute(strict=True, properties=("energy", "forces"))
        payload["prime_iterations"] = [item.iterations for item in prime.items]
        save()
        runs = [
            (candidate, repeat)
            for candidate in (args.candidate or [None])
            for repeat in range(args.repeats)
        ]
        expected_branch = None
        for candidate, repeat in runs:
            if candidate and repeat == 0:
                os.environ.update(
                    zip(CANDIDATE_CONTROLS, CANDIDATES[candidate], strict=True)
                )
                # The force execution selector changes no SCF equation. Warm
                # snapshots stay frozen; this untimed replay primes allocations
                # and records the branch before measuring the selected consumer.
                candidate_prime = batch.execute(
                    strict=True, properties=("energy", "forces")
                )
                payload.setdefault("candidate_prime_iterations", {})[candidate] = [
                    item.iterations for item in candidate_prime.items
                ]
            prefix = f"{candidate}-" if candidate else ""
            trace_path = args.output / f"{prefix}warm-{repeat}.cuda.jsonl"
            host_path = args.output / f"{prefix}warm-{repeat}.host.jsonl"
            if args.component_trace:
                os.environ["VIBEQC_DF_TRACE"] = str(trace_path.resolve())
                os.environ["VIBEQC_DF_HOST_TRACE"] = str(host_path.resolve())
            if cudart and cudart.cudaProfilerStart() != 0:
                raise RuntimeError("cudaProfilerStart failed")
            start = time.perf_counter()
            try:
                result = batch.execute(strict=True, properties=("energy", "forces"))
                seconds = time.perf_counter() - start
            finally:
                if cudart and cudart.cudaProfilerStop() != 0:
                    raise RuntimeError("cudaProfilerStop failed")
                os.environ.pop("VIBEQC_DF_TRACE", None)
                os.environ.pop("VIBEQC_DF_HOST_TRACE", None)
            energies = np.array([item.energy for item in result.items])
            forces = np.array([item.forces for item in result.items])
            ref = reference["gpu4pyscf"]
            energy_error = float(np.max(np.abs(energies - ref["energies_hartree"])))
            force_error = float(np.max(np.abs(forces - ref["forces_hartree_per_bohr"])))
            sample = {
                "candidate": candidate,
                "controls": {k: os.environ.get(k) for k in CANDIDATE_CONTROLS},
                "seconds": seconds,
                "iterations": [item.iterations for item in result.items],
                "warm_start_used": [item.warm_start_used for item in result.items],
                "warm_start_fallback": [
                    item.warm_start_fallback for item in result.items
                ],
                "energies_hartree": energies.tolist(),
                "forces_hartree_per_bohr": forces.tolist(),
                "maximum_energy_error_hartree": energy_error,
                "maximum_force_error_hartree_per_bohr": force_error,
                "metric": [
                    d.to_dict() for d in batch.last_density_fitting_metric_diagnostics()
                ],
            }
            if candidate:
                if expected_branch is None:
                    expected_branch = sample["iterations"]
                if (
                    sample["iterations"] != expected_branch
                    or not all(sample["warm_start_used"])
                    or any(sample["warm_start_fallback"])
                ):
                    raise RuntimeError(
                        "candidate sweep changed the fixed-density warm SCF branch"
                    )
            if args.component_trace:
                records = read_trace(trace_path)
                responses = [r for r in records if r["operation"] == "force_response"]
                if len(responses) != args.batch:
                    raise RuntimeError("missing per-item force response trace")
                probe = os.environ.get("VIBEQC_DF_RESPONSE_UPLOAD_PROBE", "")
                scatter_probe = os.environ.get("VIBEQC_DF_RESPONSE_SCATTER_PROBE", "")
                for response in responses:
                    n, a, counters = (
                        response["nbf"],
                        response["naux"],
                        response["counters"],
                    )
                    if response["source_backed"] or n != args.aos or a != args.aos:
                        raise RuntimeError(
                            "unexpected response value provider or dimensions"
                        )
                    if counters["three_center_derivative_weights"] != n * n * a:
                        raise RuntimeError("incomplete three-center response work")
                    if counters["metric_derivative_weights"] != a * a:
                        raise RuntimeError("incomplete metric response work")
                    if candidate:
                        execution, _, algebra, staging = CANDIDATES[candidate]
                        if bool(counters.get("three_center_shell_panels")) != (
                            execution == "shell"
                        ):
                            raise RuntimeError(
                                "selected generated shell consumer did not execute"
                            )
                        if bool(counters.get("response_charge_blas_dots")) != (
                            algebra == "blas"
                        ):
                            raise RuntimeError(
                                "selected response algebra did not execute"
                            )
                        if bool(counters.get("raw_panel_pinned_host_bytes")) != (
                            staging == "pinned-panels"
                        ):
                            raise RuntimeError(
                                "selected raw panel staging did not execute"
                            )
                    slices, remainder = divmod(
                        counters["raw_value_upload_bytes"], n * n * 8
                    )
                    if remainder or slices == 0:
                        raise RuntimeError("invalid raw response byte count")
                    if counters.get("raw_probe_prior_stream_drains", 0) != (
                        slices if probe else 0
                    ):
                        raise RuntimeError(
                            "upload drain ablation did not execute as requested"
                        )
                    if counters.get("raw_probe_gather_elements", 0) != (
                        slices * n * n if probe == "packed" else 0
                    ):
                        raise RuntimeError(
                            "upload gather ablation did not execute as requested"
                        )
                    if counters.get("derivative_probe_gradient_copies", 0) != (
                        128 if scatter_probe == "sharded" else 0
                    ):
                        raise RuntimeError(
                            "gradient destination ablation did not execute as requested"
                        )
                sample["components"] = aggregate(records)
                sample["trace_sha256"] = hashlib.sha256(
                    trace_path.read_bytes()
                ).hexdigest()
                sample["host_components"] = aggregate_host(read_host_trace(host_path))
                if sample["host_components"]["reference_eigensolves"]:
                    raise RuntimeError(
                        "warm replay executed a CPU-reference eigensolve"
                    )
                sample["host_trace_sha256"] = hashlib.sha256(
                    host_path.read_bytes()
                ).hexdigest()
            payload["samples"].append(sample)
            save()
            if not (energy_error < 1e-9 and force_error < 1e-8):
                raise RuntimeError(
                    "complete endpoint failed independent reference parity"
                )
            print(
                args.aos,
                args.batch,
                candidate,
                repeat,
                seconds,
                energy_error,
                force_error,
                flush=True,
            )


if __name__ == "__main__":
    main()

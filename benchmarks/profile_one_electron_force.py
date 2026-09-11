"""Capture warm VibeQC energy-plus-force replays for kernel attribution.

Run this helper under Nsight Systems with ``--capture-range=cudaProfilerApi``.
The cold execution prepares the fixed-topology plan and warm density before the
capture starts, so the trace contains only comparable production replays.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from _cases import benchmark_cases
from compare_gpu4pyscf_batch import scaled_geometries
from vibeqc import Calculator

_SCALAR_ENVIRONMENT = "VIBEQC_ONE_ELECTRON_FORCE_SCALAR"


def main() -> None:
    """Capture the owning CUDA stream with explicit derivative implementations."""
    cases = benchmark_cases()
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(cases), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("scalar", "cooperative", "generated_thread", "generated_shell_warp"),
        required=True,
    )
    parser.add_argument("--fitted", action="store_true")
    parser.add_argument("--df-budget", type=int, default=0)
    parser.add_argument(
        "--df-response",
        choices=("reference", "generated"),
        help="DF two-electron response selector, independent of one-electron mode",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--screening-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run real-GPU profiling inside a finite Slurm allocation")
    if args.batch < 1 or args.repeats < 1:
        raise ValueError("--batch and --repeats must be positive")
    if (
        args.energy_tolerance <= 0.0
        or args.density_tolerance <= 0.0
        or args.screening_tolerance <= 0.0
    ):
        raise ValueError("SCF and screening tolerances must be positive")

    if args.df_response is not None:
        if not args.fitted:
            parser.error("--df-response requires --fitted")
        if args.df_response == "reference":
            parser.error(
                "coordinate-wise DF response was retired; use an archived source checkout"
            )

    if args.mode == "scalar":
        os.environ[_SCALAR_ENVIRONMENT] = "1"
    else:
        os.environ.pop(_SCALAR_ENVIRONMENT, None)
    generated = args.mode.startswith("generated_")
    os.environ["VIBEQC_ONE_ELECTRON_DERIVATIVES"] = (
        "generated" if generated else "reference"
    )
    os.environ["VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING"] = (
        args.mode.removeprefix("generated_") if generated else "thread"
    )

    # The profiler needs only the runtime API, avoiding a second array runtime
    # and allocator in the measured process. Synchronize all owning streams.
    runtime = ctypes.CDLL("libcudart.so.12")

    def cuda_call(name):
        function = getattr(runtime, name)
        function.restype = ctypes.c_int
        function.argtypes = []
        status = function()
        if status:
            raise RuntimeError(f"{name} failed with CUDA status {status}")

    case = cases[args.case]
    systems = scaled_geometries(case.atoms, args.batch)
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=100,
        energy_tolerance=args.energy_tolerance,
        density_tolerance=args.density_tolerance,
        screening_tolerance=args.screening_tolerance,
        density_fitting="cuda" if args.fitted else "none",
        density_fitting_memory_budget_bytes=args.df_budget,
    )
    with calculator.prepare_batch(
        systems,
        charges=[case.charge] * args.batch,
        multiplicities=[case.multiplicity] * args.batch,
        warm_start=True,
    ) as batch:
        batch.execute(strict=True)
        batch.set_warm_start_updates(False)
        cuda_call("cudaDeviceSynchronize")
        cuda_call("cudaProfilerStart")
        samples = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            result = batch.execute(strict=True)
            cuda_call("cudaDeviceSynchronize")
            elapsed = time.perf_counter() - started
            samples.append(
                {
                    "repeat": repeat,
                    "seconds": elapsed,
                    "iterations": [item.iterations for item in result.items],
                }
            )
        cuda_call("cudaProfilerStop")

    warm_seconds = [sample["seconds"] for sample in samples]
    payload = {
        "schema_version": 1,
        "benchmark": "profile_one_electron_force",
        "case": args.case,
        "ao_count": case.expected_ao_count,
        "batch_size": args.batch,
        "mode": args.mode,
        "fitted": args.fitted,
        "df_budget": args.df_budget,
        "df_response": args.df_response,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "energy_tolerance": args.energy_tolerance,
        "density_tolerance": args.density_tolerance,
        "screening_tolerance": args.screening_tolerance,
        "repeats": args.repeats,
        "warm_samples": samples,
        "warm_median_seconds": statistics.median(warm_seconds),
        "warm_minimum_seconds": min(warm_seconds),
        "energies_hartree": result.energies.tolist(),
        "maximum_absolute_force_hartree_per_bohr": float(
            np.max(np.abs(np.stack([item.forces for item in result.items])))
        ),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()

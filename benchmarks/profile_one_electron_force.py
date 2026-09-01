"""Capture warm VibeQC energy-plus-force replays for kernel attribution.

Run this helper under Nsight Systems with ``--capture-range=cudaProfilerApi``.
The cold execution prepares the fixed-topology plan and warm density before the
capture starts, so the trace contains only comparable production replays.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import cupy as cp
import numpy as np
from _cases import benchmark_cases
from compare_gpu4pyscf_batch import nvtx_range, scaled_geometries
from vibeqc import Calculator

_SCALAR_ENVIRONMENT = "VIBEQC_ONE_ELECTRON_FORCE_SCALAR"


def main() -> None:
    cases = benchmark_cases()
    supported_cases = tuple(
        name for name, case in cases.items() if case.expected_ao_count is not None
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=supported_cases, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mode", choices=("scalar", "cooperative"), required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--screening-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1 or args.repeats < 1:
        raise ValueError("--batch and --repeats must be positive")
    if (
        args.energy_tolerance <= 0.0
        or args.density_tolerance <= 0.0
        or args.screening_tolerance <= 0.0
    ):
        raise ValueError("SCF and screening tolerances must be positive")

    if args.mode == "scalar":
        os.environ[_SCALAR_ENVIRONMENT] = "1"
    else:
        os.environ.pop(_SCALAR_ENVIRONMENT, None)

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
    )
    with calculator.prepare_batch(
        systems,
        charges=[case.charge] * args.batch,
        multiplicities=[case.multiplicity] * args.batch,
        warm_start=True,
    ) as batch:
        batch.execute(strict=True)
        cp.cuda.Stream.null.synchronize()
        cp.cuda.profiler.start()
        samples = []
        for repeat in range(args.repeats):
            with nvtx_range(cp, "vibeqc/warm/energy-plus-force"):
                started = time.perf_counter()
                result = batch.execute(strict=True)
                cp.cuda.Stream.null.synchronize()
                elapsed = time.perf_counter() - started
            samples.append(
                {
                    "repeat": repeat,
                    "seconds": elapsed,
                    "iterations": [item.iterations for item in result.items],
                }
            )
        cp.cuda.profiler.stop()

    warm_seconds = [sample["seconds"] for sample in samples]
    payload = {
        "schema_version": 1,
        "benchmark": "profile_one_electron_force",
        "case": args.case,
        "ao_count": case.expected_ao_count,
        "batch_size": args.batch,
        "mode": args.mode,
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

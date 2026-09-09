"""Interleaved one-electron candidate endpoint gates on actual HF constructions.

Run inside a finite Slurm allocation. Cold and changed-geometry samples execute
integral construction; unchanged geometry is a separate cache non-regression
control. Whole-SCF timings must never be reported as integral-kernel speedups.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from vibeqc import Calculator, _native
from vibeqc.autotune import source_identity

from benchmarks._cases import benchmark_cases
from tools.vibeqc_validation.performance import assess_comparison, measure_interleaved
from tools.vibeqc_validation.schema import canonical_hash, file_hash


def main():
    """Keep exact inputs, every SCF residual/iteration count and raw A/B samples."""
    cases = benchmark_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=cases, default="sp8")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mapping", choices=("thread", "shell_warp"), default="thread")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run this real-GPU gate inside Slurm")
    if args.batch < 1 or args.repeats < 5:
        parser.error("batch must be positive and at least five repeats are required")
    case = cases[args.case]
    systems = [
        [
            (element, tuple(np.asarray(position) * (1 + 0.01 * i)))
            for element, position in case.atoms
        ]
        for i in range(args.batch)
    ]
    base_positions = [
        np.array([position for _, position in system]) for system in systems
    ]
    options = {
        "method": case.method,
        "basis": case.vibeqc_basis,
        "basis_representation": case.basis_representation,
        "device": "cuda",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-14,
    }
    inputs_hash = canonical_hash(
        {
            "case": args.case,
            "systems": systems,
            "charge": case.charge,
            "multiplicity": case.multiplicity,
            "settings": {k: v for k, v in options.items() if k != "basis"},
            "basis": repr(case.vibeqc_basis),
        }
    )
    os.environ["VIBEQC_ONE_ELECTRON_VALUE_MAPPING"] = args.mapping
    library = _native.load_library()
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    if library.vibeqc_get_source_identity().decode() != source_identity(ROOT):
        raise RuntimeError(
            "native library does not match the current scientific source"
        )
    # Execution is already synchronous, but explicit runtime synchronization
    # preserves the shared timing protocol if the API later becomes async.
    cudart = ctypes.CDLL("libcudart.so.12")
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int

    def synchronize():
        if cudart.cudaDeviceSynchronize() != 0:
            raise RuntimeError("CUDA synchronization failed")

    def select(selection):
        os.environ["VIBEQC_ONE_ELECTRON_VALUES"] = (
            "generated" if selection == "candidate" else "reference"
        )

    def prepare():
        return Calculator(**options).prepare_batch(
            systems,
            charges=[case.charge] * args.batch,
            multiplicities=[case.multiplicity] * args.batch,
            inactive_eigensolver_profiling=True,
        )

    def diagnostics(result, batch):
        return {
            "energies": result.energies.tolist(),
            "iteration_history": [
                entry.to_dict() for entry in batch.last_inactive_eigensolver_profile()
            ],
            "iterations": [item.iterations for item in result.items],
            "energy_change": [item.energy_change for item in result.items],
            "density_rms": [item.density_rms for item in result.items],
            "backends": [item.executed_backend for item in result.items],
        }

    samples = []

    def cold(selection):
        with prepare() as batch:
            return diagnostics(batch.execute(strict=True), batch)

    # Prime module loading once on both sides. The timed cold workload still
    # constructs and destroys a fresh prepared plan for every sample.
    for side in ("baseline", "candidate"):
        select(side)
        cold(side)
    samples += measure_interleaved(
        cold,
        synchronize,
        workload="cold-start",
        inputs_hash=inputs_hash,
        repeats=args.repeats,
        prepare=select,
    )
    memory = {}
    with ExitStack() as stack:
        batches = {}
        for side in ("baseline", "candidate"):
            select(side)
            batches[side] = stack.enter_context(prepare())
            batches[side].execute(strict=True)
            # Freeze both the density and its convergence seed for A/B replay.
            batches[side].set_warm_start_updates(False)
        for workload in ("unchanged-geometry", "changed-geometry"):
            counts = {side: 0 for side in batches}

            def replay(side, *, counts=counts, workload=workload):
                counts[side] += 1
                positions = base_positions
                if workload == "changed-geometry":
                    # Alternate exact coordinates so every replay invalidates
                    # the geometry cache without giving either side new inputs.
                    delta = 0.003 if counts[side] % 2 else -0.003
                    positions = [
                        r + np.arange(r.size).reshape(r.shape) * delta
                        for r in base_positions
                    ]
                return diagnostics(
                    batches[side].execute(positions, strict=True), batches[side]
                )

            samples += measure_interleaved(
                replay,
                synchronize,
                workload=workload,
                inputs_hash=inputs_hash,
                repeats=args.repeats,
                prepare=select,
            )
        for side, batch in batches.items():
            memory[side] = batch.resource_diagnostics
    report = {
        "schema": "vibeqc.one_electron_endpoint",
        "version": 1,
        "case": args.case,
        "batch": args.batch,
        "mapping": args.mapping,
        "source_identity": source_identity(ROOT),
        "binary_hash": file_hash(library._name),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip()
        ),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "samples": samples,
        "memory": memory,
        "comparison": assess_comparison(samples),
        "timing_scope": "whole HF energy-plus-force endpoint, not integral kernels",
        "production_promoted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()

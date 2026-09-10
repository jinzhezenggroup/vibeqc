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
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from vibeqc import Calculator, Primitive, Shell, _native
from vibeqc.autotune import source_identity
from vibeqc.resources import ResourceBudget

from benchmarks._cases import benchmark_cases
from tools.vibeqc_validation.performance import assess_comparison, measure_interleaved
from tools.vibeqc_validation.schema import canonical_hash, file_hash


def main():
    """Keep exact inputs, every SCF residual/iteration count and raw A/B samples."""
    cases = benchmark_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=cases, default="sp8")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--mapping", choices=("thread", "shell_warp", "serial"), default="thread"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--derivatives", action="store_true")
    parser.add_argument(
        "--df-derivatives",
        action="store_true",
        help="compare complete DF derivative responses while sharing the generated one-electron backend",
    )
    parser.add_argument("--fitted", action="store_true")
    parser.add_argument("--df-budget", type=int, default=0)
    parser.add_argument("--observe-resources", action="store_true")
    parser.add_argument(
        "--contraction-length",
        type=int,
        default=0,
        help="replace each explicit shell's radial contraction with this many primitives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".artifacts/benchmarks/one_electron_values_gate.json"),
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run this real-GPU gate inside Slurm")
    if args.batch < 1 or args.repeats < 5:
        parser.error("batch must be positive and at least five repeats are required")
    if args.df_derivatives or not args.derivatives:
        parser.error(
            "the reference value/DF response was retired; use tools/benchmark_cuda_ownership.py "
            "with explicit archived baseline and candidate checkouts"
        )
    if args.mapping == "serial":
        parser.error("one-electron derivatives use thread/shell_warp mapping")
    case = cases[args.case]
    basis = case.vibeqc_basis
    if args.contraction_length:
        if args.contraction_length < 1 or isinstance(basis, str):
            parser.error(
                "contraction-length requires positive length and an explicit-shell case"
            )
        basis = tuple(
            Shell(
                s.atom_index,
                s.angular_momentum,
                tuple(
                    Primitive(
                        s.primitives[0].exponent * (1 + 0.2 * k),
                        (-1.0 if k % 3 == 2 else 1.0) / (k + 1),
                    )
                    for k in range(args.contraction_length)
                ),
            )
            for s in basis
        )
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
        "basis": basis,
        "basis_representation": case.basis_representation,
        "device": "cuda",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-14,
        "density_fitting": "cuda" if args.fitted else "none",
        "density_fitting_memory_budget_bytes": args.df_budget,
    }
    inputs_hash = canonical_hash(
        {
            "case": args.case,
            "systems": systems,
            "charge": case.charge,
            "multiplicity": case.multiplicity,
            "settings": {k: v for k, v in options.items() if k != "basis"},
            "basis": repr(basis),
        }
    )
    selection_variable = (
        "VIBEQC_ONE_ELECTRON_DERIVATIVES"
        if args.derivatives
        else "VIBEQC_ONE_ELECTRON_VALUES"
    )
    mapping_variable = (
        "VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING"
        if args.derivatives
        else "VIBEQC_ONE_ELECTRON_VALUE_MAPPING"
    )
    os.environ[mapping_variable] = args.mapping
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
        os.environ[selection_variable] = (
            "generated" if selection == "candidate" else "reference"
        )

    def prepare():
        resource_budget = (
            ResourceBudget(host_bytes=2 << 30, device_bytes=16 << 30)
            if args.observe_resources
            else None
        )
        return Calculator(**options, resource_budget=resource_budget).prepare_batch(
            systems,
            charges=[case.charge] * args.batch,
            multiplicities=[case.multiplicity] * args.batch,
            # Resource plans deliberately exclude optional profiling buffers.
            inactive_eigensolver_profiling=not (args.fitted or args.observe_resources),
        )

    def diagnostics(result, batch):
        return {
            "energies": result.energies.tolist(),
            "forces": [item.forces.tolist() for item in result.items],
            "iteration_history": [
                entry.to_dict() for entry in batch.last_inactive_eigensolver_profile()
            ]
            if not (args.fitted or args.observe_resources)
            else "iteration history unavailable for DF or resource-budgeted execution",
            "df_diagnostics": [
                asdict(item) for item in batch.last_density_fitting_metric_diagnostics()
            ]
            if args.fitted
            else [],
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
    paired_errors = {}
    for workload in ("cold-start", "unchanged-geometry", "changed-geometry"):
        sides = {
            side: [
                r["diagnostics"]
                for r in samples
                if r["workload"] == workload and r["selection"] == side
            ]
            for side in ("baseline", "candidate")
        }
        paired_errors[workload] = {
            "energy": max(
                float(np.max(np.abs(np.asarray(a["energies"]) - b["energies"])))
                for a, b in zip(sides["baseline"], sides["candidate"], strict=True)
            ),
            "force": max(
                float(np.max(np.abs(np.asarray(a["forces"]) - b["forces"])))
                for a, b in zip(sides["baseline"], sides["candidate"], strict=True)
            ),
        }
    passed = all(
        e["energy"] <= 3e-10 and e["force"] <= 3e-9 for e in paired_errors.values()
    )
    report = {
        "schema": "vibeqc.df_derivative_endpoint"
        if args.df_derivatives
        else "vibeqc.one_electron_endpoint",
        "version": 1,
        "case": args.case,
        "batch": args.batch,
        "contraction_length_override": args.contraction_length,
        "mapping": args.mapping,
        "operator": "df_derivatives"
        if args.df_derivatives
        else ("derivatives" if args.derivatives else "values"),
        "fitted": args.fitted,
        "df_budget": args.df_budget,
        "resource_observation_enabled": args.observe_resources,
        "paired_errors": paired_errors,
        "accuracy_passed": passed,
        "source_identity": source_identity(ROOT),
        "df_derivatives": args.df_derivatives,
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
    if not passed:
        raise SystemExit("one-electron endpoint accuracy gate failed")


if __name__ == "__main__":
    main()

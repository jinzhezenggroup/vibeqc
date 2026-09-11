"""Compare complete source+projection+target HF cost with cold/warm targets."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from vibeqc import Calculator, projected_singlepoint
from vibeqc.autotune import source_identity
from vibeqc.progressive import _retained_density

from tools.vibeqc_validation.performance import assess_comparison, measure_interleaved
from tools.vibeqc_validation.schema import canonical_hash, file_hash

CASES = {
    "h2-rhf-small-large": ("rhf", "sto-3g", "def2-svp", 0, 1),
    "h2-uhf-small-large": ("uhf", "sto-3g", "def2-svp", 1, 2),
    "h2-rhf-same": ("rhf", "sto-3g", "sto-3g", 0, 1),
    "h2-rhf-large-small": ("rhf", "def2-svp", "sto-3g", 0, 1),
}


def item_record(item, density):
    """Retain actual target density so degenerate/different roots are visible."""
    return {
        "energy": item.energy,
        "forces": item.forces.tolist(),
        "density": density.tolist(),
        "iterations": item.iterations,
        "fock_builds": item.fock_builds,
        "energy_change": item.energy_change,
        "density_rms": item.density_rms,
        "converged": item.converged,
        "backend": item.executed_backend,
        "restart_origin": item.restart_origin,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="h2-rhf-small-large")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".artifacts/benchmarks/basis_projection_gate.json"),
    )
    args = parser.parse_args()
    if args.device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("real GPU benchmarks require a finite Slurm allocation")
    method, source_basis, target_basis, charge, multiplicity = CASES[args.case]
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))]
    settings = {
        "method": method,
        "device": args.device,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-14,
    }
    source, target = (
        Calculator(basis=source_basis, **settings),
        Calculator(basis=target_basis, **settings),
    )
    library = target._library
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    identity = source_identity(ROOT)
    if library.vibeqc_get_source_identity().decode() != identity:
        raise RuntimeError(
            "benchmark library does not match the current source identity"
        )
    inputs = {
        "atoms": atoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "settings": settings,
        "source_basis": source_basis,
        "target_basis": target_basis,
    }
    inputs_hash = canonical_hash(inputs)
    runtime = ctypes.CDLL("libcudart.so.12") if args.device == "cuda" else None

    def synchronize():
        if runtime is not None and runtime.cudaDeviceSynchronize() != 0:
            raise RuntimeError("CUDA synchronization failed")

    def candidate():
        result = projected_singlepoint(
            target, source, atoms, charge=charge, multiplicity=multiplicity
        )
        return {
            "target": item_record(result.target, result.target_density),
            "source_fock_builds": result.source.fock_builds,
            "source_iterations": result.source.iterations,
            "stages": result.diagnostics,
        }

    def cold():
        with target.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity]
        ) as batch:
            item = batch.execute(strict=True).items[0]
            return {
                "target": item_record(item, _retained_density(batch)),
                "source_fock_builds": 0,
            }

    # Prime modules on both sides. Each timed cold call still constructs and
    # destroys the entire source/projection/target or target-only workflow.
    reference = cold()["target"]
    candidate()
    samples = measure_interleaved(
        lambda side: candidate() if side == "candidate" else cold(),
        synchronize,
        workload="cold-start",
        inputs_hash=inputs_hash,
        repeats=args.repeats,
    )
    with target.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity]
    ) as warm:
        warm.execute(strict=True)
        warm.set_warm_start_updates(False)

        def replay(side):
            if side == "candidate":
                return candidate()
            item = warm.execute(strict=True).items[0]
            return {
                "target": item_record(item, _retained_density(warm)),
                "source_fock_builds": 0,
            }

        samples += measure_interleaved(
            replay,
            synchronize,
            workload="unchanged-geometry",
            inputs_hash=inputs_hash,
            repeats=args.repeats,
        )
    for row in samples:
        state = row["diagnostics"]["target"]
        errors = {
            "energy": abs(state["energy"] - reference["energy"]),
            "force": float(
                np.max(np.abs(np.array(state["forces"]) - reference["forces"]))
            ),
            "density": float(
                np.max(np.abs(np.array(state["density"]) - reference["density"]))
            ),
        }
        state["reference_errors"] = errors
        state["equivalent_target_state"] = (
            errors["energy"] <= 3e-10
            and errors["force"] <= 3e-9
            and errors["density"] <= 1e-7
        )
        state["energy_relative_to_cold"] = state["energy"] - reference["energy"]
    counts = {}
    for workload in ("cold-start", "unchanged-geometry"):
        baseline = [
            s["diagnostics"]["target"]["fock_builds"]
            for s in samples
            if s["workload"] == workload and s["selection"] == "baseline"
        ]
        candidate_counts = [
            s["diagnostics"]["target"]["fock_builds"]
            for s in samples
            if s["workload"] == workload and s["selection"] == "candidate"
        ]
        counts[workload] = {
            "baseline_target": baseline,
            "projected_target": candidate_counts,
            "saved_target_fock_builds": [
                b - c for b, c in zip(baseline, candidate_counts, strict=True)
            ]
            if all(v is not None for v in [*baseline, *candidate_counts])
            else None,
        }
    report = {
        "schema": "vibeqc.basis_projection_endpoint",
        "version": 1,
        "case": args.case,
        "inputs": inputs,
        "samples": samples,
        "comparison": assess_comparison(samples),
        "fock_builds": counts,
        "all_target_states_equivalent": all(
            r["diagnostics"]["target"]["equivalent_target_state"] for r in samples
        ),
        "source_identity": identity,
        "binary_sha256": file_hash(library._name),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "counter_boundary": "measured CPU physical Fock builds; CUDA and incomplete retries are unavailable",
        "timing_scope": "full energy/force endpoints; candidate always includes fresh source solve, projection and target solve; warm baseline retains its target plan",
        "iteration_history": "not exported by this public solver; final counts/residuals and physical Fock evaluations retained",
        "production_promoted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "case": args.case,
                "passed": report["all_target_states_equivalent"],
                "output": str(args.output),
            }
        )
    )
    if not report["all_target_states_equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

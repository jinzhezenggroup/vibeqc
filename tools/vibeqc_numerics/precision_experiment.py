"""Audit the existing experimental CUDA mixed-Fock override at fixed SCF controls.

Run as a standalone, single-threaded driver inside a GPU allocation. Environment
changes are confined to this diagnostic process and restored on every exit.
Requested thresholds are not FP32 execution counts or observable-error budgets;
#174 owns instrumentation, policy promotion and automatic refinement.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from vibeqc import Calculator, ResolvedModel, TargetAccuracy, compare_observables
from vibeqc.profiles import canonical_hash, file_hash

from tools.vibeqc_numerics.audit import ProbeControls, StrictHFAudit, probe_hf
from tools.vibeqc_numerics.replay import replay
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.fixtures import calculator_inputs


@contextmanager
def _mixed_override(threshold):
    """Restore the process policy even after a failed native experiment."""
    key = "VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD"
    old = os.environ.get(key)
    os.environ[key] = str(threshold)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def run(baseline_path: Path, output: Path):
    """Compare matched iteration settings with an independently strict baseline.

    Only the arithmetic request changes between each matched pair. Both are
    compared with the fully relaxed strict FP64-requested target, and every
    converged density is audited using unscreened CPU raw integrals. The saved
    states use the same operator-replay schema as the baseline experiment.
    """
    started = time.perf_counter()
    if not replay(baseline_path)["passed"]:
        raise ValueError("precision experiment requires a verified baseline archive")
    baseline = json.loads(baseline_path.read_text())
    if baseline["backend"] != "cuda":
        raise ValueError("precision experiment requires a CUDA baseline")
    output.mkdir(parents=True, exist_ok=True)
    target = TargetAccuracy.from_dict(baseline["target"])
    cases, states = [], {}
    with np.load(
        baseline_path.parent / baseline["states"]["file"], allow_pickle=False
    ) as arrays:
        for original in baseline["cases"]:
            case = {**original, "rows": []}
            inputs = case["inputs"]
            atoms = list(
                zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True)
            )
            calc = Calculator(**calculator_inputs(inputs))
            model = ResolvedModel.from_dict(case["model"])
            name = case["name"]
            strict_density = arrays[name + "__strict__density"]
            strict_forces = arrays[name + "__strict__forces"]
            states[name + "__strict__density"] = strict_density
            states[name + "__strict__forces"] = strict_forces
            with NativeSource(atoms, calc._basis, charge=inputs["charge"]) as source:
                audit = StrictHFAudit(source, model)
                for matched in original["rows"]:
                    if (
                        matched["source"] != "coupled_iteration"
                        or matched["status"] != "converged"
                    ):
                        continue
                    for threshold in (1e-6, 1e-3, 0.1):
                        label = matched["label"] + f"__mixed_{threshold:g}"
                        row = {
                            "label": label,
                            "source": "experimental_arithmetic_at_matched_iteration_controls",
                            "controls": matched["controls"],
                            "requested_mixed_fock_threshold": threshold,
                            "matched_fp64_row": matched["label"],
                            "matched_fp64_solve_seconds": matched["solve_seconds"],
                            "matched_fp64_relaxed_errors": matched["relaxed_errors"],
                        }
                        try:
                            with _mixed_override(threshold):
                                probe = probe_hf(
                                    source,
                                    model,
                                    ProbeControls(**matched["controls"]),
                                    backend="cuda",
                                    experimental_mixed_fock_threshold=threshold,
                                )
                            row.update(probe.scalars())
                            row["status"] = (
                                "converged" if probe.converged else "unconverged"
                            )
                            if probe.converged:
                                physical = audit.evaluate(probe)
                                comparison = compare_observables(
                                    model,
                                    model,
                                    model,
                                    target,
                                    {"energy": probe.energy, "forces": probe.forces},
                                    {
                                        "energy": case["strict"]["energy"],
                                        "forces": strict_forces,
                                    },
                                    scope="relaxed_target",
                                    provenance=(
                                        ("strict_reference", case["strict_id"]),
                                    ),
                                    converged=True,
                                )
                                row.update(
                                    physical_audit=physical,
                                    assessment=comparison.to_dict(),
                                    relaxed_errors={
                                        "energy": abs(
                                            probe.energy - case["strict"]["energy"]
                                        ),
                                        "force_max": float(
                                            np.max(np.abs(probe.forces - strict_forces))
                                        ),
                                    },
                                    matched_energy_difference=abs(
                                        probe.energy - matched["energy"]
                                    ),
                                    matched_force_max_difference=float(
                                        np.max(
                                            np.abs(
                                                probe.forces
                                                - arrays[
                                                    name
                                                    + "__"
                                                    + matched["label"]
                                                    + "__forces"
                                                ]
                                            )
                                        )
                                    ),
                                    density_change_from_strict=float(
                                        np.max(np.abs(probe.density - strict_density))
                                    ),
                                )
                                states[name + "__" + label + "__density"] = (
                                    probe.density
                                )
                                states[name + "__" + label + "__forces"] = probe.forces
                        except (RuntimeError, ValueError, MemoryError) as error:
                            row.update(status="failed", reason=str(error))
                        case["rows"].append(row)
            cases.append(case)
            print(json.dumps({"case": name, "rows": len(case["rows"])}), flush=True)
    archive = output / "states.npz"
    np.savez_compressed(archive, **states)
    record = {
        **baseline,
        "cases": cases,
        "baseline_record_hash": baseline["record_hash"],
        "runner_sha256": file_hash(Path(__file__)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "total_seconds": time.perf_counter() - started,
        "calibration": None,
        "holdout": None,
        "states": {
            "file": archive.name,
            "sha256": file_hash(archive),
            "bytes": archive.stat().st_size,
            "array_shapes": {k: list(v.shape) for k, v in states.items()},
        },
        "boundaries": {
            **baseline["boundaries"],
            "arithmetic": "Existing experimental mixed-Fock request; actual FP32 workload count remains unreported. No policy promotion or empirical coverage claim.",
            "cost": "Matched single-pass complete native probe timings; diagnostic comparison only, not a performance gate.",
        },
    }
    record.pop("record_hash")
    record["record_hash"] = canonical_hash(record)
    (output / "report.json").write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.baseline, arguments.output)


if __name__ == "__main__":
    main()

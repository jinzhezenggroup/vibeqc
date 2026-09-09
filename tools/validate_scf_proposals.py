"""Reproducible native HF traces and complete-solve traditional baselines.

Run ``python -m tools.validate_scf_proposals --output PATH`` with the CPU
library. The explicit output path opts into local molecular trace export.
Reference generation is separate and requires pinned PySCF only when invoked.
"""

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from vibeqc import (
    Calculator,
    ObservableTarget,
    TargetAccuracy,
    _native,
    compare_observables,
)
from vibeqc.profiles import canonical_hash, file_hash

from tools.vibeqc_numerics.audit import ProbeControls, StrictHFAudit, probe_hf
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_scf import solve
from tools.vibeqc_scf.baselines import transported_density
from tools.vibeqc_scf.proposals import diagonal_preconditioner, diis_density, mixing
from tools.vibeqc_scf.replay import TargetOperator, counterfactual, export_trace
from tools.vibeqc_validation.fixtures import calculator_inputs

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "tests/reference_data/scf-proposals/references.json"
TARGET = TargetAccuracy(
    (
        ObservableTarget("energy", "absolute", "Eh", 1e-9),
        ObservableTarget("forces", "max_abs", "Eh/bohr", 1e-8),
    )
)
CONTROLS = ProbeControls(energy_tolerance=1e-12, density_tolerance=1e-10)


def load_references(path=REFERENCES):
    record = json.loads(path.read_text())
    digest = record.pop("record_hash")
    if (
        record["schema"] != "vibeqc.scf_reference_families"
        or record["schema_version"] != 1
        or canonical_hash(record) != digest
        or record["provenance"]["pyscf"] != "2.14.0"
    ):
        raise ValueError("invalid SCF independent reference archive")
    assignments = {}
    for case in record["cases"]:
        prior = assignments.setdefault(case["family"], case["split"])
        if prior != case["split"]:
            raise ValueError("whole-family split leakage")
    return record


def agreement(energy, forces, model, reference):
    """Observed #173 comparisons are distinct from solver convergence."""
    if energy is None or forces is None:
        return {"status": "unavailable", "reason": "failed solve"}
    assessment = compare_observables(
        model,
        model,
        model,
        TARGET,
        {"energy": energy, "forces": forces},
        {"energy": reference["energy"], "forces": np.asarray(reference["forces"])},
        scope="relaxed_target",
        provenance=(
            ("independent_reference", canonical_hash(reference)),
            ("comparator", "pinned-pyscf-2.14.0"),
        ),
        converged=True,
    )
    return assessment.to_dict()


def run(output, *, repeats=3, names=None, environment_note=None):
    """Measure equal controls and record information available to each baseline."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be positive")
    references = load_references()
    if names and set(names) - {c["trajectory"] for c in references["cases"]}:
        raise ValueError("unknown SCF benchmark trajectory")
    if output.exists():
        raise ValueError("benchmark output must be a new directory")
    output.mkdir(parents=True)
    rows, history = [], {}
    for case in references["cases"]:
        if names and case["trajectory"] not in names:
            continue
        inputs = case["inputs"]
        atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
        # The public calculator requires a positive screening control; CPU
        # conventional integrals are unscreened in both execution paths.
        calc = Calculator(
            **{
                **calculator_inputs(inputs),
                **asdict(CONTROLS),
                "screening_tolerance": 1e-14,
            }
        )
        model = calc.resolved_model(
            atoms, charge=inputs["charge"], multiplicity=inputs["multiplicity"]
        )
        previous = history.setdefault(case["trajectory"], [])
        with NativeSource(atoms, calc._basis, charge=inputs["charge"]) as source:
            audit = StrictHFAudit(source, model)
            variants = [
                ("cold_diis", None, CONTROLS, None, "current geometry only", None),
                (
                    "fixed_point",
                    None,
                    replace(CONTROLS, diis_history=0),
                    None,
                    "current geometry only",
                    None,
                ),
                (
                    "safeguarded_diis",
                    diis_density,
                    CONTROLS,
                    None,
                    "current geometry only",
                    None,
                ),
                (
                    "safeguarded_mixing",
                    mixing(0.5),
                    CONTROLS,
                    None,
                    "current geometry only",
                    None,
                ),
                (
                    "diagonal_ov_preconditioner",
                    diagonal_preconditioner,
                    CONTROLS,
                    None,
                    "current geometry only",
                    None,
                ),
            ]
            if previous:
                for label, older in (
                    ("projected_orbitals", None),
                    (
                        "extrapolated_density",
                        previous[-2][0] if len(previous) > 1 else None,
                    ),
                ):
                    if label == "extrapolated_density" and len(previous) < 2:
                        continue
                    seed, repair = transported_density(
                        previous[-1][0], model, audit.overlap, older=older
                    )
                    variants.append(
                        (
                            label,
                            None,
                            CONTROLS,
                            seed,
                            "previous geometry state"
                            if older is None
                            else "two previous geometry states",
                            repair,
                        )
                    )
            current = None
            for label, proposer, controls, seed, information, repair in variants:
                timings, results = [], []
                for _ in range(repeats):
                    measured = solve(
                        source, model, controls, proposer=proposer, initial_density=seed
                    )
                    timings.append(
                        measured.solve_seconds
                        + (repair["repair_seconds"] if repair else 0)
                    )
                    results.append(measured)
                result = results[0]
                row = {
                    "case": case["name"],
                    "family": case["family"],
                    "split": case["split"],
                    "baseline": label,
                    "available_information": information,
                    "controls": asdict(controls),
                    "repair": repair,
                    **result.metrics(),
                    "latency_samples_seconds": timings,
                    "median_total_seconds": statistics.median(timings),
                    "throughput_solves_per_second": 1 / statistics.median(timings),
                    "comparison": agreement(
                        result.energy, result.forces, model, case["reference"]
                    ),
                    "independent_competing_solution": case["competing_solution"],
                    "repeat_work_counts": [
                        [r.iterations, r.fock_builds, r.status] for r in results
                    ],
                }
                if result.converged:
                    row["physical_audit"] = audit.evaluate(result.as_probe())
                rows.append(row)
                if label == "cold_diis":
                    current = result
                    traced = solve(source, model, controls, capture=True)
                    export_trace(
                        traced,
                        source,
                        output / (case["name"] + "-cold.json"),
                        provenance={
                            "family": case["family"],
                            "split": case["split"],
                            "reference_hash": canonical_hash(case["reference"]),
                        },
                    )
                    row["trace_total_seconds"] = traced.solve_seconds
                    row["trace_copy_validation_seconds"] = traced.trace_seconds
                    row["trace_preserves_result"] = (
                        traced.energy == result.energy
                        and traced.iterations == result.iterations
                    )
                    # Legacy timing starts at its native call; compare it with
                    # the same outer wall-clock envelope as this bridge.
                    legacy_times = []
                    for _ in range(repeats):
                        started = time.perf_counter()
                        legacy = probe_hf(source, model, controls)
                        legacy_times.append(time.perf_counter() - started)
                    row["legacy_total_samples_seconds"] = legacy_times
                    row["disabled_over_legacy_latency"] = statistics.median(
                        timings
                    ) / statistics.median(legacy_times)
                    row["disabled_preserves_result"] = (
                        legacy.energy == result.energy
                        and legacy.iterations == result.iterations
                    )
                    operator = TargetOperator(source, model)
                    row["counterfactuals"] = [
                        counterfactual(s, diis_density, operator)
                        for s in traced.snapshots[:3]
                    ]
                    if traced.converged:
                        retained_state = traced.snapshots[-1]
                if label == "safeguarded_diis":
                    traced = solve(
                        source, model, controls, proposer=proposer, capture=True
                    )
                    export_trace(
                        traced,
                        source,
                        output / (case["name"] + "-proposal.json"),
                        provenance={"family": case["family"], "split": case["split"]},
                    )
            # Same-geometry warm density has strictly more initial information
            # than cold DIIS, so it is an explicitly separate comparison group.
            if current is not None and current.converged:
                warm = solve(source, model, CONTROLS, initial_density=current.density)
                rows.append(
                    {
                        "case": case["name"],
                        "family": case["family"],
                        "split": case["split"],
                        "baseline": "same_geometry_warm_density",
                        "available_information": "converged current-geometry density",
                        **warm.metrics(),
                        "prior_solve_seconds": current.solve_seconds,
                        "prior_fock_builds": current.fock_builds,
                        "comparison": agreement(
                            warm.energy, warm.forces, model, case["reference"]
                        ),
                    }
                )
            # Compare moving-geometry warm starts through the existing public
            # FleetPlan. Without a fallback, the validated CPU loop count is
            # exactly iterations+2. Fallback work is explicitly unavailable in
            # this older public result ABI and cannot support a speedup claim.
            if previous:
                prior_inputs = previous[-1][1]
                prior_atoms = list(
                    zip(
                        inputs["atomic_numbers"],
                        prior_inputs["coordinates"],
                        strict=True,
                    )
                )
                with calc.prepare_batch(
                    [prior_atoms],
                    charges=[model.charge],
                    multiplicities=[model.multiplicity],
                ) as batch:
                    initial = batch.execute()
                    started = time.perf_counter()
                    moved = batch.execute([np.asarray(inputs["coordinates"])]).items[0]
                    elapsed = time.perf_counter() - started
                    known_count = not moved.warm_start_fallback and moved.status in (
                        _native.STATUS_SUCCESS,
                        _native.STATUS_SCF_NOT_CONVERGED,
                    )
                    rows.append(
                        {
                            "case": case["name"],
                            "family": case["family"],
                            "split": case["split"],
                            "baseline": "existing_fleet_warm_density",
                            "available_information": "previous geometry state",
                            "status": "converged" if moved.succeeded else "failed",
                            "energy": moved.energy,
                            "iterations": moved.iterations,
                            "fock_builds": moved.iterations
                            + (2 if moved.converged else 0)
                            if known_count
                            else None,
                            "fock_count_source": "audited CPU iterations plus final rebuilds on convergence"
                            if known_count
                            else "unavailable from the older public ABI after fallback/error",
                            "warm_start_used": moved.warm_start_used,
                            "warm_start_fallback": moved.warm_start_fallback,
                            "prior_converged": initial.items[0].succeeded,
                            "solve_seconds": elapsed,
                            "comparison": agreement(
                                moved.energy, moved.forces, model, case["reference"]
                            ),
                        }
                    )
            failed = solve(
                source, model, replace(CONTROLS, max_iterations=1), capture=True
            )
            export_trace(
                failed,
                source,
                output / (case["name"] + "-failed.json"),
                provenance={"family": case["family"], "split": case["split"]},
            )
            rows.append(
                {
                    "case": case["name"],
                    "family": case["family"],
                    "split": case["split"],
                    "baseline": "deliberate_iteration_failure",
                    **failed.metrics(),
                }
            )
            if current is not None and current.converged:
                previous.append((retained_state, inputs))
        print(case["name"], "complete", flush=True)
    sources = [
        Path(__file__),
        ROOT / "src/scf/rhf.cpp",
        ROOT / "src/scf/types.hpp",
        ROOT / "src/scf/proposals.hpp",
        ROOT / "src/scf/proposal_bridge.hpp",
        ROOT / "src/posthf/bridge.cpp",
        ROOT / "tools/vibeqc_numerics/audit.py",
        ROOT / "tools/vibeqc_posthf/sources.py",
        ROOT / "python/vibeqc/accuracy.py",
        ROOT / "python/vibeqc/profiles.py",
        *sorted((ROOT / "tools/vibeqc_scf").glob("*.py")),
    ]
    report = {
        "schema": "vibeqc.scf_complete_solve_benchmark",
        "schema_version": 1,
        "rows": rows,
        "target": TARGET.to_dict(),
        "references_sha256": file_hash(REFERENCES),
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "library_sha256": file_hash(Path(source._library._name)),
            "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "thread_environment": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "environment_note": environment_note,
        },
        "second_order_baseline": "unavailable: native CPU solver exposes DIIS and fixed-point iteration, no Newton/SOSCF",
        "proposal_family": "deterministic traditional baselines; no learned model",
        "timing_scope": "complete CPU solve with integral preparation, iteration, failed proposals and forces; repeats exclude independent audits and trace export",
        "replay_timing_scope": "snapshot counterfactual only; never a complete solve",
    }
    report["record_hash"] = canonical_hash(report)
    (output / "report.json").write_text(
        json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--environment-note")
    args = parser.parse_args()
    run(
        args.output,
        repeats=args.repeats,
        names=args.names,
        environment_note=args.environment_note,
    )

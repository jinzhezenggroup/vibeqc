"""Run controlled native HF accuracy experiments and whole-family calibration.

Invoke with ``python -m tools.validate_accuracy --output PATH``. CPU execution
is the first calibration domain. CUDA runs require an external GPU allocation
and retain their own out-of-domain estimator status. No numerical policy is
changed automatically and no external reference dependency is imported.
"""

from __future__ import annotations

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import ctypes as ct
import json
import os
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from vibeqc import Calculator, ObservableTarget, TargetAccuracy, compare_observables
from vibeqc.accuracy_estimator import (
    EmpiricalHFEstimator,
    HFCalibrationDomain,
    HFCalibrationSample,
)
from vibeqc.profiles import canonical_hash, file_hash

from tools.vibeqc_numerics.audit import (
    ProbeControls,
    StrictHFAudit,
    error_features,
    probe_hf,
)
from tools.vibeqc_numerics.fixtures import FAMILIES, TRAINING_FAMILIES, accuracy_suite
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.fixtures import calculator_inputs

TARGET = TargetAccuracy(
    (
        ObservableTarget("energy", "absolute", "Eh", 1e-6),
        ObservableTarget("forces", "max_abs", "Eh/bohr", 1e-7),
    )
)
STRICT_ENERGY_GATE = 1e-9
STRICT_FORCE_GATE = 1e-8


def variants(backend):
    """Change one source at a time, then coupled SCF controls, at fixed identity."""
    strict = ProbeControls()
    rows = []
    for tolerance in (1e-9, 1e-7, 1e-6, 1e-5):
        rows.extend(
            (
                (
                    f"energy_{tolerance:g}",
                    "iteration_energy",
                    replace(strict, energy_tolerance=tolerance),
                ),
                (
                    f"density_{tolerance:g}",
                    "iteration_density",
                    replace(strict, density_tolerance=tolerance),
                ),
                (
                    f"coupled_{tolerance:g}",
                    "coupled_iteration",
                    replace(
                        strict, energy_tolerance=tolerance, density_tolerance=tolerance
                    ),
                ),
            )
        )
        if backend == "cuda":
            rows.append(
                (
                    f"screening_{tolerance:g}",
                    "screening",
                    replace(strict, screening_tolerance=tolerance),
                )
            )
            rows.append(
                (
                    f"coupled_screening_{tolerance:g}",
                    "coupled_screening_iteration",
                    replace(
                        strict,
                        screening_tolerance=tolerance,
                        energy_tolerance=tolerance,
                        density_tolerance=tolerance,
                    ),
                )
            )
    rows.append(
        ("deliberate_nonconvergence", "failure", replace(strict, max_iterations=1))
    )
    return rows


def run(output, *, backend="cpu", names=None):
    """Write replayable, checksum-linked scientific states and all failed rows."""
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cases, arrays, samples = [], {}, []
    requested = set(names) if names else None
    fixtures = accuracy_suite()
    if requested and requested - {r["inputs"]["name"] for r in fixtures}:
        raise ValueError("unknown accuracy fixture")
    for reference in fixtures:
        inputs = reference["inputs"]
        name = inputs["name"]
        if requested and name not in requested:
            continue
        atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
        calc = Calculator(**calculator_inputs(inputs))
        model = calc.resolved_model(
            atoms, charge=inputs["charge"], multiplicity=inputs["multiplicity"]
        )
        case = {
            "name": name,
            "family": FAMILIES[name],
            "split": "training" if FAMILIES[name] in TRAINING_FAMILIES else "holdout",
            "inputs": inputs,
            "model": model.to_dict(),
            "independent_reference_hash": canonical_hash(reference),
            "rows": [],
        }
        with NativeSource(atoms, calc._basis, charge=inputs["charge"]) as source:
            audit = StrictHFAudit(source, model)
            strict = probe_hf(source, model, backend=backend)
            if not strict.converged:
                case.update(status="strict_reference_failed", strict=strict.scalars())
                cases.append(case)
                continue
            strict_audit = audit.evaluate(strict)
            energy_error = abs(strict.energy - reference["data"]["energy"])
            force_error = float(
                np.max(np.abs(strict.forces - reference["data"]["forces"]))
            )
            strict_passed = (
                energy_error <= STRICT_ENERGY_GATE
                and force_error <= STRICT_FORCE_GATE
                and strict_audit["orthonormal_commutator_max"] <= 1e-8
            )
            strict_id = canonical_hash(
                {
                    "model": model.identity,
                    "controls": asdict(strict.controls),
                    "energy": strict.energy,
                    "forces": strict.forces.tolist(),
                    "density": strict.density.tolist(),
                }
            )
            case.update(
                status="pass" if strict_passed else "strict_gate_failed",
                strict=strict.scalars(),
                strict_controls=asdict(strict.controls),
                strict_id=strict_id,
                strict_audit=strict_audit,
                independent_errors={"energy": energy_error, "force_max": force_error},
                audit_setup_seconds=audit.setup_seconds,
                audit_resident_array_bytes=audit.resident_array_bytes,
            )
            arrays[name + "__strict__density"] = strict.density
            arrays[name + "__strict__forces"] = strict.forces
            for label, source_name, controls in variants(backend):
                record = {
                    "label": label,
                    "source": source_name,
                    "controls": asdict(controls),
                }
                try:
                    probe = probe_hf(source, model, controls, backend=backend)
                    record.update(probe.scalars())
                    record["status"] = "converged" if probe.converged else "unconverged"
                    if not probe.converged:
                        case["rows"].append(record)
                        continue
                    physical = audit.evaluate(probe)
                    features_started = time.perf_counter()
                    features = error_features(source, probe, physical)
                    features_seconds = time.perf_counter() - features_started
                    errors = (
                        abs(probe.energy - strict.energy),
                        float(np.max(np.abs(probe.forces - strict.forces))),
                    )
                    report = compare_observables(
                        model,
                        model,
                        model,
                        TARGET,
                        {"energy": probe.energy, "forces": probe.forces},
                        {"energy": strict.energy, "forces": strict.forces},
                        scope="relaxed_target",
                        provenance=(
                            ("strict_reference", strict_id),
                            ("independent_reference", canonical_hash(reference)),
                        ),
                        converged=probe.converged and strict.converged,
                    )
                    sample_id = canonical_hash(
                        {
                            "case": name,
                            "label": label,
                            "model": model.identity,
                            "controls": asdict(controls),
                            "backend": backend,
                        }
                    )
                    record.update(
                        sample_id=sample_id,
                        physical_audit=physical,
                        features=asdict(features),
                        features_seconds=features_seconds,
                        relaxed_errors={"energy": errors[0], "force_max": errors[1]},
                        assessment=report.to_dict(),
                        density_change_from_strict=float(
                            np.max(np.abs(probe.density - strict.density))
                        ),
                        intended_root_certified=False,
                    )
                    arrays[name + "__" + label + "__density"] = probe.density
                    arrays[name + "__" + label + "__forces"] = probe.forces
                    if strict_passed:
                        samples.append(
                            HFCalibrationSample(
                                FAMILIES[name],
                                sample_id,
                                model,
                                features,
                                *errors,
                                strict_id,
                            )
                        )
                except (RuntimeError, ValueError, MemoryError) as error:
                    record.update(status="failed", reason=str(error))
                case["rows"].append(record)
        cases.append(case)
        print(
            json.dumps(
                {"case": name, "status": case["status"], "rows": len(case["rows"])}
            ),
            flush=True,
        )

    calibration = None
    holdout = None
    if backend == "cpu":
        basis_family_id = Calculator(basis="sto-3g")._basis.identity
        domain = HFCalibrationDomain(basis_family_id)
        training = [s for s in samples if s.family in TRAINING_FAMILIES]
        if {s.family for s in training} == set(TRAINING_FAMILIES):
            estimator = EmpiricalHFEstimator.fit(domain, training)
            calibration = estimator.to_dict()
            held_out = [s for s in samples if s.family not in TRAINING_FAMILIES]
            if held_out:
                holdout = estimator.evaluate_holdout(held_out, force_tolerance=1e-7)
                holdout["target_sweep"] = []
                for tolerance in (1e-9, 1e-7, 1e-6, 1e-5):
                    reliability = estimator.evaluate_holdout(
                        held_out, energy_tolerance=tolerance, force_tolerance=tolerance
                    )
                    holdout["target_sweep"].append(
                        {
                            "energy_abs_Eh": tolerance,
                            "force_max_abs_Eh_per_bohr": tolerance,
                            **{k: v for k, v in reliability.items() if k != "rows"},
                        }
                    )
            predictions = {}
            reference_norms = {
                case["model"]["identity"]: (
                    abs(case["strict"]["energy"]),
                    float(np.max(np.abs(arrays[case["name"] + "__strict__forces"]))),
                )
                for case in cases
                if case["status"] == "pass"
            }
            for sample in samples:
                began = time.perf_counter()
                try:
                    prediction = estimator.predict(
                        sample.model,
                        sample.features,
                        energy_reference_norm=reference_norms[sample.model.identity][0],
                        force_reference_norm=reference_norms[sample.model.identity][1],
                    )
                    result = {
                        "status": "empirical",
                        "evidence": [e.to_dict() for e in prediction],
                    }
                except ValueError as error:
                    result = {"status": "outside_domain", "reason": str(error)}
                predictions[sample.sample_id] = {
                    **result,
                    "evaluation_seconds": time.perf_counter() - began,
                }
            for case in cases:
                for row in case["rows"]:
                    if row.get("sample_id") in predictions:
                        row["prediction"] = predictions[row["sample_id"]]

    archive = output / "states.npz"
    np.savez_compressed(archive, **arrays)
    library = Calculator()._library
    library.vibeqc_get_source_identity.restype = ct.c_char_p
    result = {
        "schema": "vibeqc.accuracy_experiment",
        "schema_version": 1,
        "backend": backend,
        "target": TARGET.to_dict(),
        "cases": cases,
        "strict_gates": {
            "energy_abs": STRICT_ENERGY_GATE,
            "force_max_abs": STRICT_FORCE_GATE,
            "physical_residual_max": 1e-8,
        },
        "calibration": calibration,
        "holdout": holdout,
        "source_identity": library.vibeqc_get_source_identity().decode(),
        "runner_sha256": file_hash(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "thread_policy": {
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "total_seconds": time.perf_counter() - start,
        "states": {
            "file": archive.name,
            "sha256": file_hash(archive),
            "bytes": archive.stat().st_size,
            "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        },
        "boundaries": {
            "screening": "CPU oracle is unscreened; no CPU screening speed/error claim"
            if backend == "cpu"
            else "explicit same-model screening sweeps",
            "arithmetic": "FP64 baseline; actual FP32 execution/refinement belongs to #174 instrumentation",
            "fixed_density_audit": "CPU native raw values plus NumPy contractions, timed separately",
            "relaxed_audit": "separate strict native solve, validated against pinned independent PySCF",
            "model_variation": "changing fitting metric/grid/provider changes model identity and is rejected by the accuracy assessment",
            "calibration": "limited CPU FP64 domain; never a certified observable bound",
        },
    }
    result["record_hash"] = canonical_hash(result)
    # The full per-item evidence is intentionally retained. Compact JSON keeps
    # the CUDA sweep below the repository's per-artifact size limit without
    # dropping failed rows or model/evidence identities.
    (output / "report.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    if calibration:
        (output / "estimator.json").write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cases", nargs="+")
    arguments = parser.parse_args()
    result = run(arguments.output, backend=arguments.backend, names=arguments.cases)
    print(
        json.dumps(
            {
                "seconds": result["total_seconds"],
                "holdout": None
                if result["holdout"] is None
                else {k: v for k, v in result["holdout"].items() if k != "rows"},
            },
            indent=2,
        )
    )
    if any(case["status"] != "pass" for case in result["cases"]):
        raise SystemExit("strict HF reference gates failed; see preserved report")


if __name__ == "__main__":
    main()

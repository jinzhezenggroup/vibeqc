#!/usr/bin/env python3
"""NUM03 force-aware SCF-effort geometry-optimization research harness.

This is deliberately a research tool, not a public geometry optimizer.  It
compares one deterministic FIRE trajectory using fixed-strict SCF against the
same FIRE update with an opt-in NUM03 next-step SCF convergence policy.  Grid,
screening, arithmetic precision and the finite-basis electronic model stay
fixed so incomplete-SCF error is not conflated with other numerical sources.

Every optimization finishes with an independent strict energy+force cleanup.
Calibration/reference work is reported separately from production-policy time.
Negative convergence or performance results are retained in the JSON output.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Self

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from vibeqc import (
    Calculator,
    ForceAwareScfPolicy,
    ObservableDelta,
    ScfEffortLevel,
    ScfForceCalibrationSample,
    ScfForceErrorEstimator,
    TargetErrorBudget,
)
from vibeqc.profiles import canonical_hash


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    basis: str = "sto-3g"
    charge: int = 0
    multiplicity: int = 1


@dataclass(frozen=True)
class Endpoint:
    level: str
    energy: float
    forces: tuple[tuple[float, float, float], ...]
    converged: bool
    iterations: int
    energy_change: float
    density_rms: float
    physical_residual_rms: float | None
    seconds: float
    restart_origin: str
    requested_precision: str
    precision: dict[str, Any] | None


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def levels() -> tuple[ScfEffortLevel, ...]:
    return (
        ScfEffortLevel("loose", 0, 1e-6, 1e-4, 60),
        ScfEffortLevel("medium", 1, 1e-8, 1e-6, 90),
        ScfEffortLevel("strict", 2, 1e-10, 1e-8, 140, strict=True),
    )


def _atoms(symbols: tuple[str, ...], coordinates: np.ndarray) -> tuple:
    return tuple(
        (symbol, tuple(float(value) for value in row))
        for symbol, row in zip(symbols, coordinates, strict=True)
    )


def _case(
    name: str,
    family: str,
    symbols: tuple[str, ...],
    xyz: list[list[float]],
    *,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
) -> Case:
    return Case(
        name,
        family,
        tuple((symbol, tuple(row)) for symbol, row in zip(symbols, xyz, strict=True)),
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
    )


def cases() -> dict[str, Case]:
    water = _case(
        "water",
        "water",
        ("O", "H", "H"),
        [[0.0, 0.0, 0.0], [0.18, 1.62, 1.18], [-0.12, -1.48, 1.04]],
    )
    methane = _case(
        "methane",
        "methane",
        ("C", "H", "H", "H", "H"),
        [
            [0.0, 0.0, 0.0],
            [1.30, 1.30, 1.30],
            [-1.42, -1.22, 1.36],
            [-1.28, 1.45, -1.19],
            [1.40, -1.30, -1.38],
        ],
    )
    ethane = _case(
        "ethane-flexible",
        "ethane",
        ("C", "C", "H", "H", "H", "H", "H", "H"),
        [
            [-1.45, 0.0, 0.0],
            [1.45, 0.0, 0.0],
            [-2.10, 1.72, 0.20],
            [-2.05, -0.72, 1.58],
            [-2.15, -1.02, -1.40],
            [2.10, -1.65, -0.30],
            [2.02, 0.55, 1.68],
            [2.18, 1.10, -1.34],
        ],
    )
    return {
        "h2": _case(
            "h2",
            "h2",
            ("H", "H"),
            [[0.0, 0.0, -1.10], [0.0, 0.0, 1.10]],
        ),
        "lih": _case(
            "lih",
            "lih",
            ("Li", "H"),
            [[0.0, 0.0, -2.20], [0.0, 0.0, 2.20]],
        ),
        "water": water,
        "methane": methane,
        "ethane-flexible": ethane,
        "water-def2-svp": dataclasses.replace(
            water, name="water-def2-svp", family="water-basis-holdout", basis="def2-svp"
        ),
    }


def calibration_geometries(case: Case) -> tuple[tuple[str, np.ndarray], ...]:
    xyz = np.asarray([position for _, position in case.atoms], dtype=float)
    if case.name == "h2":
        return tuple(
            (label, np.array([[0.0, 0.0, -bond / 2], [0.0, 0.0, bond / 2]]))
            for label, bond in (("early", 2.4), ("intermediate", 1.8), ("near", 1.45))
        )
    if case.name == "lih":
        return tuple(
            (label, np.array([[0.0, 0.0, -bond / 2], [0.0, 0.0, bond / 2]]))
            for label, bond in (("early", 4.8), ("intermediate", 3.8), ("near", 3.1))
        )
    # Holdouts use small deterministic perturbations of the declared geometry.
    direction = np.arange(1, xyz.size + 1, dtype=float).reshape(xyz.shape)
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    return (
        ("early", xyz),
        ("intermediate", xyz + 0.08 * direction),
        ("near", xyz - 0.04 * direction),
    )


def _calculator(
    case: Case,
    level: ScfEffortLevel,
    device: str,
    *,
    precision: str = "fp64",
) -> Calculator:
    return Calculator(
        method="rhf",
        basis=case.basis,
        device=device,
        precision=precision,
        screening_tolerance=1e-12,
        max_iterations=level.max_iterations,
        energy_tolerance=level.energy_tolerance,
        density_tolerance=level.density_tolerance,
    )


@cache
def basis_calibration_id(case: Case) -> str:
    """Calibration-domain basis family, not the exact molecular basis target."""
    return canonical_hash(
        {
            "basis_family": case.basis.lower().replace("_", "-"),
            "representation": "cartesian",
        }
    )


def scientific_model_id(case: Case, device: str = "cpu") -> str:
    """Exact finite-basis target; device is deliberately excluded."""
    calc = _calculator(case, levels()[-1], device, precision="fp64")
    metadata = calc.basis_metadata(
        case.atoms, charge=case.charge, multiplicity=case.multiplicity
    )
    return canonical_hash(
        {
            "method": "rhf",
            "basis": metadata["orbital"]["mathematical_identity"],
            "precision": "fp64",
            "screening_tolerance": 1e-12,
        }
    )


def single_endpoint(
    case: Case,
    coordinates: np.ndarray,
    level: ScfEffortLevel,
    device: str,
    *,
    precision: str = "fp64",
) -> Endpoint:
    calc = _calculator(case, level, device, precision=precision)
    started = time.perf_counter()
    result = calc.singlepoint(
        _atoms(tuple(symbol for symbol, _ in case.atoms), coordinates),
        charge=case.charge,
        multiplicity=case.multiplicity,
        properties=("energy", "forces"),
    )
    elapsed = time.perf_counter() - started
    if result.forces is None:
        raise RuntimeError("RHF endpoint returned no analytic forces")
    return Endpoint(
        level.name,
        float(result.energy),
        tuple(tuple(float(value) for value in row) for row in result.forces),
        bool(result.converged),
        int(result.iterations),
        float(result.energy_change),
        float(result.density_rms),
        (
            None
            if result.physical_residual_rms is None
            else float(result.physical_residual_rms)
        ),
        elapsed,
        "singlepoint",
        precision,
        _jsonable(result.precision),
    )


def calibration_samples(
    training: tuple[Case, ...], device: str
) -> tuple[tuple[ScfForceCalibrationSample, ...], dict[str, Any]]:
    samples = []
    rows = {}
    strict_level = levels()[-1]
    for case in training:
        case_rows = []
        calibration_basis = basis_calibration_id(case)
        for geometry_label, coordinates in calibration_geometries(case):
            strict = single_endpoint(case, coordinates, strict_level, device)
            if not strict.converged:
                raise RuntimeError(
                    f"strict calibration failed for {case.name}:{geometry_label}"
                )
            for level in levels()[:-1]:
                candidate = single_endpoint(case, coordinates, level, device)
                if not candidate.converged:
                    case_rows.append(
                        {
                            "geometry": geometry_label,
                            "level": level.name,
                            "status": "not_converged",
                        }
                    )
                    continue
                delta = ObservableDelta.between(
                    candidate.energy,
                    candidate.forces,
                    strict.energy,
                    strict.forces,
                )
                samples.append(
                    ScfForceCalibrationSample(
                        case.family,
                        f"{case.name}:{geometry_label}:{level.name}",
                        "rhf",
                        calibration_basis,
                        "density_rms",
                        candidate.density_rms,
                        delta,
                    )
                )
                case_rows.append(
                    {
                        "geometry": geometry_label,
                        "level": level.name,
                        "candidate": _jsonable(candidate),
                        "strict": _jsonable(strict),
                        "actual_error": _jsonable(delta),
                    }
                )
        rows[case.name] = case_rows
    return tuple(samples), rows


def holdout_samples(case: Case, device: str) -> tuple[ScfForceCalibrationSample, ...]:
    strict_level = levels()[-1]
    samples = []
    calibration_basis = basis_calibration_id(case)
    for geometry_label, coordinates in calibration_geometries(case):
        strict = single_endpoint(case, coordinates, strict_level, device)
        for level in levels()[:-1]:
            candidate = single_endpoint(case, coordinates, level, device)
            if not candidate.converged:
                continue
            samples.append(
                ScfForceCalibrationSample(
                    case.family,
                    f"{case.name}:{geometry_label}:{level.name}",
                    "rhf",
                    calibration_basis,
                    "density_rms",
                    candidate.density_rms,
                    ObservableDelta.between(
                        candidate.energy,
                        candidate.forces,
                        strict.energy,
                        strict.forces,
                    ),
                )
            )
    return tuple(samples)


class PreparedLevels:
    """One warm-start-capable prepared batch per fixed SCF effort level."""

    def __init__(self, case: Case, device: str, precision: str) -> None:
        self.case, self.device, self.precision = case, device, precision
        self._stack = contextlib.ExitStack()
        self._batches = {}

    def __enter__(self) -> Self:
        for level in levels():
            calc = _calculator(self.case, level, self.device, precision=self.precision)
            batch = self._stack.enter_context(
                calc.prepare_batch(
                    [self.case.atoms],
                    charges=[self.case.charge],
                    multiplicities=[self.case.multiplicity],
                    warm_start=True,
                )
            )
            self._batches[level.name] = batch
        return self

    def __exit__(self, *args: object) -> None:
        self._stack.close()

    def evaluate(self, coordinates: np.ndarray, level: ScfEffortLevel) -> Endpoint:
        started = time.perf_counter()
        result = self._batches[level.name].execute(
            coordinates=(coordinates,),
            strict=True,
            properties=("energy", "forces"),
        )
        elapsed = time.perf_counter() - started
        item = result.items[0]
        if not item.succeeded or item.forces is None:
            raise RuntimeError(
                f"SCF endpoint failed at {level.name}: status={item.status}"
            )
        return Endpoint(
            level.name,
            float(item.energy),
            tuple(tuple(float(value) for value in row) for row in item.forces),
            bool(item.converged),
            int(item.iterations),
            float(item.energy_change),
            float(item.density_rms),
            (
                None
                if item.physical_residual_rms is None
                else float(item.physical_residual_rms)
            ),
            elapsed,
            item.restart_origin,
            self.precision,
            _jsonable(item.precision),
        )


def fire_optimize(
    case: Case,
    *,
    device: str,
    policy: ForceAwareScfPolicy,
    adaptive: bool,
    precision: str,
    max_steps: int,
) -> dict[str, Any]:
    policy_started = time.perf_counter()
    coordinates = np.asarray([position for _, position in case.atoms], dtype=float)
    initial_centroid = coordinates.mean(axis=0)
    velocity = np.zeros_like(coordinates)
    dt, dt_max = 0.08, 0.30
    alpha, alpha_start = 0.10, 0.10
    positive_steps = 0
    state = policy.initial_state()
    selected = policy.strict_level
    records = []
    retries = 0
    production_seconds = 0.0
    total_iterations = 0
    converged_geometry = False
    calibration_basis = basis_calibration_id(case)
    with PreparedLevels(case, device, precision) as prepared:
        for step in range(max_steps):
            try:
                endpoint = prepared.evaluate(coordinates, selected)
            except RuntimeError as error:
                if selected.strict:
                    records.append(
                        {"step": step, "status": "strict_failure", "error": str(error)}
                    )
                    break
                retries += 1
                selected = policy.strict_level
                state = dataclasses.replace(state, level_index=policy.strict_index)
                endpoint = prepared.evaluate(coordinates, selected)
            production_seconds += endpoint.seconds
            total_iterations += endpoint.iterations
            forces = np.asarray(endpoint.forces, dtype=float)
            force_max = float(np.max(np.abs(forces)))
            force_rms = float(np.sqrt(np.mean(forces**2)))
            record = {
                "step": step,
                "level": selected.name,
                "endpoint": _jsonable(endpoint),
                "force_max_abs": force_max,
                "force_rms": force_rms,
                "coordinates": coordinates.tolist(),
            }
            if force_max <= policy.optimizer_force_tolerance:
                converged_geometry = True
                records.append(record)
                break

            if adaptive:
                decision = policy.propose_next(
                    state,
                    method="rhf",
                    basis_id=calibration_basis,
                    geometry_id=canonical_hash(
                        {
                            "case": case.name,
                            "step": step,
                            "coordinates": coordinates.tolist(),
                        }
                    ),
                    step_index=step,
                    current_force_max=force_max,
                )
                state = decision.state
                selected = decision.level
                record["next_effort"] = _jsonable(decision)
            else:
                selected = policy.strict_level

            # Deterministic FIRE update with unit fictitious masses.
            velocity += dt * forces
            power = float(np.sum(velocity * forces))
            velocity_norm = float(np.linalg.norm(velocity))
            force_norm = float(np.linalg.norm(forces))
            if power > 0.0 and force_norm > 0.0:
                positive_steps += 1
                velocity = (1.0 - alpha) * velocity + alpha * (
                    velocity_norm / force_norm
                ) * forces
                if positive_steps > 5:
                    dt = min(dt * 1.1, dt_max)
                    alpha *= 0.99
            else:
                positive_steps = 0
                dt *= 0.5
                alpha = alpha_start
                velocity.fill(0.0)
            displacement = dt * velocity
            atom_steps = np.linalg.norm(displacement, axis=1)
            max_step = float(np.max(atom_steps))
            if max_step > 0.20:
                displacement *= 0.20 / max_step
            coordinates += displacement
            # Remove pure translation without changing internal geometry.
            coordinates += initial_centroid - coordinates.mean(axis=0)
            record["fire"] = {
                "dt": dt,
                "alpha": alpha,
                "power": power,
                "max_displacement_bohr": float(
                    np.max(np.linalg.norm(displacement, axis=1))
                ),
            }
            records.append(record)

    cleanup_started = time.perf_counter()
    strict_final = single_endpoint(
        case,
        coordinates,
        policy.strict_level,
        device,
        precision="fp64",
    )
    cleanup_seconds = time.perf_counter() - cleanup_started
    final_forces = np.asarray(strict_final.forces)
    final_force_max = float(np.max(np.abs(final_forces)))
    model_id = scientific_model_id(case, device)
    verification = policy.verify_final(
        target_model_id=model_id,
        observed_model_id=model_id,
        level=policy.strict_level,
        converged=strict_final.converged,
        energy=strict_final.energy,
        energy_change=strict_final.energy_change,
        density_rms=strict_final.density_rms,
        force_max_abs=final_force_max,
    )
    return {
        "case": case.name,
        "adaptive": adaptive,
        "requested_precision": precision,
        "converged_before_cleanup": converged_geometry,
        "records": records,
        "steps": len(records),
        "retries": retries,
        "scf_iterations": total_iterations,
        "production_seconds_before_cleanup": cleanup_started - policy_started,
        "successful_endpoint_seconds": production_seconds,
        "strict_cleanup_seconds": cleanup_seconds,
        "complete_policy_seconds": time.perf_counter() - policy_started,
        "strict_final": _jsonable(strict_final),
        "strict_final_force_max_abs": final_force_max,
        "final_coordinates": coordinates.tolist(),
        "final_verification": _jsonable(verification),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--force-tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--optimization-cases",
        default="water,methane,ethane-flexible,water-def2-svp",
        help="comma-separated case names",
    )
    parser.add_argument(
        "--holdout-cases",
        default="water,methane,ethane-flexible,water-def2-svp",
        help="comma-separated calibration-holdout case names",
    )
    args = parser.parse_args()

    all_cases = cases()
    training = (all_cases["h2"], all_cases["lih"])
    calibration_started = time.perf_counter()
    samples, calibration_rows = calibration_samples(training, args.device)
    estimator = ScfForceErrorEstimator.fit(samples, safety_factor=1.5)
    calibration_seconds = time.perf_counter() - calibration_started

    policy = ForceAwareScfPolicy(
        levels(),
        estimator,
        optimizer_force_tolerance=args.force_tolerance,
        intermediate_energy_error=2e-5,
        intermediate_force_fraction=0.15,
        max_intermediate_force_error=5e-3,
        near_stationary_factor=5.0,
        relax_after=2,
    )
    holdout_budget = TargetErrorBudget(
        energy_abs=policy.intermediate_energy_error,
        force_max_abs=policy.max_intermediate_force_error,
    )
    holdout = {}
    for name in filter(None, args.holdout_cases.split(",")):
        case = all_cases[name]
        started = time.perf_counter()
        measured = holdout_samples(case, args.device)
        calibration_basis = basis_calibration_id(case)
        holdout[name] = {
            "report": estimator.evaluate_holdout(measured, holdout_budget),
            "seconds": time.perf_counter() - started,
            "basis_calibration_id": calibration_basis,
            "production_policy_basis_admitted": calibration_basis
            in estimator.basis_ids,
        }

    optimizations = {}
    for name in filter(None, args.optimization_cases.split(",")):
        case = all_cases[name]
        arms = {
            "fixed_strict": fire_optimize(
                case,
                device=args.device,
                policy=policy,
                adaptive=False,
                precision="fp64",
                max_steps=args.max_steps,
            ),
            "mixed_arithmetic_only": fire_optimize(
                case,
                device=args.device,
                policy=policy,
                adaptive=False,
                precision="auto",
                max_steps=args.max_steps,
            ),
            "tolerance_schedule_only": fire_optimize(
                case,
                device=args.device,
                policy=policy,
                adaptive=True,
                precision="fp64",
                max_steps=args.max_steps,
            ),
            "combined": fire_optimize(
                case,
                device=args.device,
                policy=policy,
                adaptive=True,
                precision="auto",
                max_steps=args.max_steps,
            ),
        }
        baseline = arms["fixed_strict"]
        baseline_ok = baseline["final_verification"]["status"] == "observed_met"
        comparisons = {}
        for arm_name, arm in arms.items():
            arm_ok = arm["final_verification"]["status"] == "observed_met"
            speedup = (
                baseline["complete_policy_seconds"] / arm["complete_policy_seconds"]
                if baseline_ok and arm_ok
                else None
            )
            comparisons[arm_name] = {
                "matched_final_accuracy": baseline_ok and arm_ok,
                "complete_endpoint_speedup_vs_fixed_strict": speedup,
                "performance_success": bool(speedup is not None and speedup > 1.0),
            }
        optimizations[name] = {"arms": arms, "comparisons": comparisons}

    payload = {
        "schema": "vibeqc.num03-scf-effort-geomopt/v1",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "device": args.device,
        "ownership": {
            "public_geometry_optimizer_added": False,
            "shared_progressive_controller_owner": "github-issue-192",
            "arithmetic_precision_owner": "github-issue-174",
            "policy_scope": "SCF convergence effort only",
        },
        "optimizer_contract": {
            "algorithm": "deterministic FIRE research harness",
            "termination": {
                "force_max_abs_eh_per_bohr": args.force_tolerance,
                "max_steps": args.max_steps,
            },
            "maximum_single_atom_step_bohr": 0.20,
            "public_optimizer_api": False,
        },
        "calibration": {
            "estimator": _jsonable(estimator),
            "seconds": calibration_seconds,
            "rows": calibration_rows,
            "certified": False,
        },
        "holdout": holdout,
        "optimizations": optimizations,
        "interpretation": {
            "no_nve_hessian_ts_generalization": True,
            "negative_performance_results_are_retained": True,
            "calibration_reference_cost_is_research_evidence_not_production_policy_cost": True,
            "strict_cleanup_is_included_in_complete_policy_seconds": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    failures = [
        f"{name}:{arm_name}"
        for name, result in optimizations.items()
        for arm_name, arm in result["arms"].items()
        if arm["final_verification"]["status"] != "observed_met"
    ]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

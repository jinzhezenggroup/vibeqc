"""Qualify public conventional canonical RHF-MP2 force endpoints.

The driver writes a fresh run directory and never overwrites prior evidence.
PySCF is imported only while executing independent reference calculations; the
case registry and evidence validators remain usable in ordinary CPU CI.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from vibeqc import Primitive, Shell

RUN_SCHEMA = "vibeqc.mp2-public-force-validation.v1"
CASE_SCHEMA = "vibeqc.mp2-public-force-case.v1"

PYSCF_FORCE_ATOL = 2.0e-6
FD_FINE_ATOL = 5.0e-6
TRANSLATION_ATOL = 2.0e-8
TORQUE_ATOL = 2.0e-7
ROTATION_ATOL = 2.0e-6
BATCH_ATOL = 2.0e-9
CPU_CUDA_ATOL = 2.0e-9
ENERGY_FORCE_IDENTITY_ATOL = 2.0e-8


@dataclass(frozen=True)
class PublicForceCase:
    """One closed-shell all-electron conventional MP2 qualification case."""

    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    vibeqc_basis: str | tuple[Shell, ...]
    pyscf_basis: str | dict[str, list]
    basis_representation: str = "cartesian"
    charge: int = 0
    minimum_ao_count: int = 1


def validation_cases() -> dict[str, PublicForceCase]:
    """Return the fixed B2 scientific case matrix in review order."""

    return {
        "h2": PublicForceCase(
            atoms=(("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))),
            vibeqc_basis="sto-3g",
            pyscf_basis="sto-3g",
        ),
        "lih": PublicForceCase(
            atoms=(("Li", (0.0, 0.0, 0.0)), ("H", (0.2, -0.1, 3.0))),
            vibeqc_basis="sto-3g",
            pyscf_basis="sto-3g",
            basis_representation="spherical",
        ),
        "h2o": PublicForceCase(
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (1.4, 0.1, 1.1)),
                ("H", (-1.2, 0.2, 1.3)),
            ),
            vibeqc_basis="sto-3g",
            pyscf_basis="sto-3g",
        ),
        "f-shell": PublicForceCase(
            atoms=(("He", (0.0, 0.0, 0.0)), ("H", (0.3, -0.2, 1.7))),
            vibeqc_basis=(
                Shell(0, 0, (Primitive(1.3, 1.0),)),
                Shell(0, 3, (Primitive(0.7, 1.0),)),
                Shell(1, 0, (Primitive(0.6, 1.0),)),
            ),
            pyscf_basis={
                "He": [[0, [1.3, 1.0]], [3, [0.7, 1.0]]],
                "H": [[0, [0.6, 1.0]]],
            },
            basis_representation="spherical",
            charge=1,
        ),
        "water-def2-svp": PublicForceCase(
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, -1.43233673, 1.10715266)),
                ("H", (0.0, 1.43233673, 1.10715266)),
            ),
            vibeqc_basis="def2-svp",
            pyscf_basis="def2-svp",
            minimum_ao_count=13,
        ),
    }


def parse_fd_steps(value: str | Sequence[float]) -> tuple[float, ...]:
    """Parse the required three-or-more positive, distinct FD step sizes."""

    try:
        values = (
            tuple(float(item.strip()) for item in value.split(","))
            if isinstance(value, str)
            else tuple(float(item) for item in value)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "finite differences require three distinct positive steps"
        ) from error
    if (
        len(values) < 3
        or len(set(values)) != len(values)
        or any(not math.isfinite(item) or item <= 0 for item in values)
    ):
        raise ValueError("finite differences require three distinct positive steps")
    return values


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def initialize_run_directory(
    destination: Path,
    *,
    backend: str,
    cases: Sequence[str],
    fd_steps: Sequence[float],
    command: Sequence[str],
) -> Path:
    """Create a fresh evidence root and its initial machine-readable manifest."""

    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")
    selected = tuple(cases)
    known = validation_cases()
    if not selected or any(name not in known for name in selected):
        raise ValueError("validation cases contain an unknown or empty selection")
    steps = parse_fd_steps(fd_steps)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(
        destination / "manifest.json",
        {
            "schema": RUN_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "cases": list(selected),
            "fd_steps_bohr": list(steps),
            "command": list(command),
            "status": "running",
            "case_records": [],
        },
    )
    return destination


def validate_case_record(record: dict, *, expected_steps: Sequence[float]) -> None:
    """Reject incomplete or internally inconsistent scientific case records."""

    if record.get("schema") != CASE_SCHEMA or record.get("status") != "pass":
        raise ValueError("case record is not a passing public-force result")
    if record.get("backend") not in {"cpu", "cuda"} or not record.get("case"):
        raise ValueError("case record lacks backend or case identity")
    forces = record.get("forces", {})
    if not forces.get("public") or not forces.get("pyscf"):
        raise ValueError("case record lacks public or PySCF forces")
    errors = record.get("errors", {})
    finite = errors.get("finite_difference", [])
    actual_steps = tuple(row.get("step_bohr") for row in finite)
    if actual_steps != tuple(expected_steps):
        raise ValueError("case record finite-difference steps are incomplete")
    for key in ("pyscf_max_abs", "pyscf_rms"):
        value = errors.get(key)
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError("case record has invalid independent-reference errors")
    for row in finite:
        if any(
            type(row.get(key)) not in {int, float}
            or not math.isfinite(row[key])
            or row[key] < 0
            for key in ("step_bohr", "max_abs", "rms")
        ):
            raise ValueError("case record has invalid finite-difference errors")
    invariants = record.get("invariants", {})
    required_invariants = {
        "translation_force_norm",
        "torque_norm",
        "rotational_covariance_max_abs",
        "changed_geometry_energy_delta",
        "changed_geometry_force_delta",
        "energy_force_identity_error",
    }
    if not required_invariants <= invariants.keys() or any(
        type(invariants[key]) not in {int, float}
        or not math.isfinite(invariants[key])
        or invariants[key] < 0
        for key in required_invariants
    ):
        raise ValueError("case record has incomplete invariants")
    diagnostic = record.get("diagnostics", {})
    integer_fields = {
        "response_iterations",
        "planned_endpoint_peak_bytes",
        "measured_endpoint_peak_bytes",
        "numeric_capacity_bytes",
        "force_provenance_flags",
    }
    if any(
        type(diagnostic.get(key)) is not int or diagnostic[key] < 0
        for key in integer_fields
    ):
        raise ValueError("case record has invalid resource diagnostics")
    if diagnostic["measured_endpoint_peak_bytes"] == 0:
        raise ValueError("case record requires a measured endpoint peak")
    if not (
        diagnostic["measured_endpoint_peak_bytes"]
        <= diagnostic["planned_endpoint_peak_bytes"]
        <= diagnostic["numeric_capacity_bytes"]
    ):
        raise ValueError("case record resource bounds are inconsistent")
    for key in ("response_absolute_residual", "response_relative_residual"):
        value = diagnostic.get(key)
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError("case record has invalid response diagnostics")
    if (
        len(diagnostic.get("equation_hash", "")) != 64
        or diagnostic.get("response_operator_hash") != "rhf-canonical-response-v1"
        or diagnostic.get("force_provenance_flags") != 7
    ):
        raise ValueError("case record has invalid force provenance")


def central_finite_difference_forces(
    calculator,
    atoms: Sequence[tuple[str | int, Sequence[float]]],
    *,
    charge: int,
    steps: Sequence[float],
) -> list[dict]:
    """Fully re-solve every Cartesian displacement at every requested step."""

    displacements = _finite_difference_displacements(atoms, steps)
    records = []
    for step, rows in displacements:
        forces = np.zeros((len(atoms), 3), dtype=np.float64)
        for atom_index, axis, plus_atoms, minus_atoms in rows:
            plus_energy = calculator.singlepoint(
                plus_atoms, charge=charge, properties=("energy",)
            ).energy
            minus_energy = calculator.singlepoint(
                minus_atoms, charge=charge, properties=("energy",)
            ).energy
            forces[atom_index, axis] = -(float(plus_energy) - float(minus_energy)) / (
                2 * step
            )
        records.append({"step_bohr": step, "forces": forces.tolist()})
    return records


def _finite_difference_displacements(
    atoms: Sequence[tuple[str | int, Sequence[float]]],
    steps: Sequence[float],
) -> list[tuple[float, list[tuple[int, int, tuple, tuple]]]]:
    """Materialize every ordered plus/minus geometry without evaluating it."""

    checked_steps = parse_fd_steps(steps)
    elements = tuple(atom[0] for atom in atoms)
    positions = np.asarray([atom[1] for atom in atoms], dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not np.isfinite(positions).all()
    ):
        raise ValueError("finite-difference atoms require finite Cartesian coordinates")
    records = []
    for step in checked_steps:
        rows = []
        for atom_index in range(len(positions)):
            for axis in range(3):
                plus = positions.copy()
                minus = positions.copy()
                plus[atom_index, axis] += step
                minus[atom_index, axis] -= step
                plus_atoms = tuple(
                    (element, tuple(position))
                    for element, position in zip(elements, plus, strict=True)
                )
                minus_atoms = tuple(
                    (element, tuple(position))
                    for element, position in zip(elements, minus, strict=True)
                )
                rows.append((atom_index, axis, plus_atoms, minus_atoms))
        records.append((step, rows))
    return records


def _finite_difference_energy_task(payload: tuple) -> float:
    """Evaluate one fresh CPU energy in an isolated spawned worker."""

    case_name, budget, charge, atoms = payload
    case = validation_cases()[case_name]
    calculator = _calculator(case, "cpu", budget)
    return float(
        calculator.singlepoint(atoms, charge=charge, properties=("energy",)).energy
    )


def parallel_central_finite_difference_forces(
    name: str,
    case: PublicForceCase,
    *,
    budget: int,
    steps: Sequence[float],
    workers: int,
) -> list[dict]:
    """Run the same full CPU FD matrix across isolated spawned processes."""

    if type(workers) is not int or workers < 2:
        raise ValueError("parallel finite differences require at least two workers")
    displacements = _finite_difference_displacements(case.atoms, steps)
    tasks = []
    for _, rows in displacements:
        for _, _, plus_atoms, minus_atoms in rows:
            tasks.append((name, budget, case.charge, plus_atoms))
            tasks.append((name, budget, case.charge, minus_atoms))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        energies = iter(executor.map(_finite_difference_energy_task, tasks))
        records = []
        for step, rows in displacements:
            forces = np.zeros((len(case.atoms), 3), dtype=np.float64)
            for atom_index, axis, _, _ in rows:
                plus_energy = next(energies)
                minus_energy = next(energies)
                forces[atom_index, axis] = -(plus_energy - minus_energy) / (2 * step)
            records.append({"step_bohr": step, "forces": forces.tolist()})
    return records


def force_invariants(positions, forces) -> dict[str, float]:
    """Return translation and rotational sum-rule residual norms."""

    coordinates = np.asarray(positions, dtype=np.float64)
    values = np.asarray(forces, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1:] != (3,)
        or values.shape != coordinates.shape
        or not np.isfinite(coordinates).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("force invariants require matching finite Cartesian arrays")
    translation = np.sum(values, axis=0)
    torque = np.sum(np.cross(coordinates, values), axis=0)
    return {
        "translation_force_norm": float(np.linalg.norm(translation)),
        "torque_norm": float(np.linalg.norm(torque)),
    }


def _error_metrics(actual, expected) -> dict[str, float]:
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(
        expected, dtype=np.float64
    )
    if difference.size == 0 or not np.isfinite(difference).all():
        raise ValueError("error metrics require nonempty finite arrays")
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference**2))),
    }


def _git_identity() -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        revision, dirty = "unknown", True
    return {"revision": revision, "dirty": dirty}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_matrix() -> np.ndarray:
    axis = np.asarray((1.0, 2.0, -1.0), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.371
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )


def _atoms_with_positions(case: PublicForceCase, positions) -> tuple:
    values = np.asarray(positions, dtype=np.float64)
    return tuple(
        (element, tuple(float(component) for component in position))
        for (element, _), position in zip(case.atoms, values, strict=True)
    )


def _pyscf_reference(case: PublicForceCase) -> dict:
    import pyscf
    from pyscf import gto, mp, scf

    molecule = gto.M(
        atom=list(case.atoms),
        basis=case.pyscf_basis,
        unit="Bohr",
        cart=case.basis_representation == "cartesian",
        charge=case.charge,
        spin=0,
        verbose=0,
    )
    if molecule.nao_nr() < case.minimum_ao_count:
        raise RuntimeError(
            f"case has {molecule.nao_nr()} AOs, below required {case.minimum_ao_count}"
        )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1.0e-13
    mean_field.conv_tol_grad = 1.0e-11
    mean_field.max_cycle = 200
    mean_field.direct_scf_tol = 0.0
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("independent PySCF RHF reference did not converge")
    correlation = mp.MP2(mean_field, frozen=None)
    correlation.kernel()
    if getattr(correlation, "converged", True) is not True:
        raise RuntimeError("independent PySCF MP2 reference did not converge")
    gradient = np.asarray(correlation.nuc_grad_method().kernel(), dtype=np.float64)
    if not np.isfinite(gradient).all():
        raise RuntimeError("independent PySCF MP2 gradient is nonfinite")
    return {
        "pyscf_version": pyscf.__version__,
        "ao_count": int(molecule.nao_nr()),
        "energy": float(mean_field.e_tot + correlation.e_corr),
        "forces": (-gradient).tolist(),
        "settings": {
            "scf_energy_tolerance": mean_field.conv_tol,
            "scf_gradient_tolerance": mean_field.conv_tol_grad,
            "direct_scf_tolerance": mean_field.direct_scf_tol,
            "frozen_core": None,
            "cartesian": bool(molecule.cart),
        },
    }


def _calculator(case: PublicForceCase, backend: str, budget: int):
    from vibeqc import Calculator

    return Calculator(
        method="mp2",
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device=backend,
        correlation_memory_budget_bytes=budget,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
        max_iterations=200,
        screening_tolerance=0.0,
    )


def _changed_geometry(case: PublicForceCase) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([position for _, position in case.atoms], dtype=np.float64)
    direction = np.arange(1, positions.size + 1, dtype=np.float64).reshape(
        positions.shape
    )
    direction -= direction.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(direction)
    if not norm > 0:
        raise ValueError("changed-geometry direction is singular")
    displacement = 1.0e-3 * direction / norm
    return positions + displacement, displacement


def _run_failure_matrix(case: PublicForceCase, backend: str) -> dict:
    checks = {}

    def expect(name, expected, operation):
        try:
            operation()
        except expected as error:
            checks[name] = {
                "status": "pass",
                "exception": type(error).__name__,
                "detail": str(error),
            }
        except Exception as error:  # noqa: BLE001 - retain unexpected failures
            checks[name] = {
                "status": "fail",
                "exception": type(error).__name__,
                "detail": str(error),
            }
        else:
            checks[name] = {"status": "fail", "detail": "request unexpectedly passed"}

    from vibeqc import Calculator

    force_request = {"charge": case.charge, "properties": ("energy", "forces")}
    expect(
        "denominator",
        RuntimeError,
        lambda: Calculator(
            method="mp2",
            basis=case.vibeqc_basis,
            basis_representation=case.basis_representation,
            device=backend,
            mp2_denominator_threshold=100.0,
        ).singlepoint(case.atoms, **force_request),
    )
    expect(
        "minimum_budget",
        (RuntimeError, MemoryError),
        lambda: Calculator(
            method="mp2",
            basis=case.vibeqc_basis,
            basis_representation=case.basis_representation,
            device=backend,
            correlation_memory_budget_bytes=1024,
        ).singlepoint(case.atoms, **force_request),
    )
    invalid_atoms = list(case.atoms)
    invalid_atoms[0] = (invalid_atoms[0][0], (math.nan, *invalid_atoms[0][1][1:]))
    expect(
        "nonfinite_geometry",
        (ValueError, RuntimeError),
        lambda: Calculator(
            method="mp2",
            basis=case.vibeqc_basis,
            basis_representation=case.basis_representation,
            device=backend,
        ).singlepoint(invalid_atoms, **force_request),
    )
    expect(
        "unsupported_ri_force",
        NotImplementedError,
        lambda: Calculator(
            method="mp2",
            basis=case.vibeqc_basis,
            basis_representation=case.basis_representation,
            device=backend,
            density_fitting="cuda" if backend == "cuda" else "cpu",
        ).singlepoint(case.atoms, **force_request),
    )
    checks["response_iteration_exhaustion"] = {
        "status": "covered-by-native-test",
        "evidence": "vibeqc_mp2_gradient_tests bounded GMRES maximum-iteration status",
        "reason": "the public method intentionally has no response-iteration fault-injection control",
    }
    return checks


def run_case(
    name: str,
    case: PublicForceCase,
    *,
    backend: str,
    fd_steps: Sequence[float],
    budget: int,
    fd_workers: int = 1,
) -> dict:
    """Run one independent analytic, FD, covariance, batch and resource gate."""

    calculator = _calculator(case, backend, budget)
    base = calculator.singlepoint(
        case.atoms, charge=case.charge, properties=("energy", "forces")
    )
    public_forces = np.asarray(base.forces, dtype=np.float64)
    reference = _pyscf_reference(case)
    pyscf_forces = np.asarray(reference["forces"], dtype=np.float64)
    independent_error = _error_metrics(public_forces, pyscf_forces)
    finite_records = (
        parallel_central_finite_difference_forces(
            name, case, budget=budget, steps=fd_steps, workers=fd_workers
        )
        if fd_workers > 1
        else central_finite_difference_forces(
            calculator, case.atoms, charge=case.charge, steps=fd_steps
        )
    )
    finite_errors = []
    for row in finite_records:
        metrics = _error_metrics(row["forces"], public_forces)
        finite_errors.append(
            {"step_bohr": row["step_bohr"], **metrics, "forces": row["forces"]}
        )

    positions = np.asarray([position for _, position in case.atoms], dtype=np.float64)
    invariants = force_invariants(positions, public_forces)
    rotation = _rotation_matrix()
    rotated_atoms = _atoms_with_positions(case, positions @ rotation.T)
    rotated = calculator.singlepoint(
        rotated_atoms, charge=case.charge, properties=("energy", "forces")
    )
    invariants["rotational_covariance_max_abs"] = _error_metrics(
        rotated.forces, public_forces @ rotation.T
    )["max_abs"]

    changed_positions, displacement = _changed_geometry(case)
    changed_atoms = _atoms_with_positions(case, changed_positions)
    changed = calculator.singlepoint(
        changed_atoms, charge=case.charge, properties=("energy", "forces")
    )
    changed_forces = np.asarray(changed.forces, dtype=np.float64)
    invariants["changed_geometry_energy_delta"] = abs(changed.energy - base.energy)
    invariants["changed_geometry_force_delta"] = float(
        np.max(np.abs(changed_forces - public_forces))
    )
    invariants["energy_force_identity_error"] = abs(
        (changed.energy - base.energy)
        + 0.5 * float(np.vdot(public_forces + changed_forces, displacement))
    )

    batch = calculator.batch_singlepoint(
        [case.atoms, changed_atoms],
        charges=[case.charge, case.charge],
        strict=True,
    )
    batch_force_error = max(
        _error_metrics(batch.items[0].forces, public_forces)["max_abs"],
        _error_metrics(batch.items[1].forces, changed_forces)["max_abs"],
    )
    batch_energy_error = max(
        abs(batch.items[0].energy - base.energy),
        abs(batch.items[1].energy - changed.energy),
    )

    cpu_parity = None
    if backend == "cuda":
        cpu = _calculator(case, "cpu", budget).singlepoint(
            case.atoms, charge=case.charge, properties=("energy", "forces")
        )
        cpu_parity = {
            "energy_abs": abs(cpu.energy - base.energy),
            **_error_metrics(base.forces, cpu.forces),
        }

    diagnostic = asdict(base.correlation)
    gates = {
        "pyscf": independent_error["max_abs"] <= PYSCF_FORCE_ATOL,
        "finite_difference": finite_errors[-1]["max_abs"] <= FD_FINE_ATOL,
        "translation": invariants["translation_force_norm"] <= TRANSLATION_ATOL,
        "torque": invariants["torque_norm"] <= TORQUE_ATOL,
        "rotation": invariants["rotational_covariance_max_abs"] <= ROTATION_ATOL,
        "changed_geometry": invariants["changed_geometry_force_delta"] > 0,
        "energy_force_identity": invariants["energy_force_identity_error"]
        <= ENERGY_FORCE_IDENTITY_ATOL,
        "batch": max(batch_force_error, batch_energy_error) <= BATCH_ATOL,
        "response": diagnostic["response_absolute_residual"] <= 1.0e-10,
        "resource_plan": 0
        < diagnostic["measured_endpoint_peak_bytes"]
        <= diagnostic["planned_endpoint_peak_bytes"]
        <= diagnostic["numeric_capacity_bytes"],
        "provenance": diagnostic["force_provenance_flags"] == 7,
        "cpu_cuda_parity": cpu_parity is None
        or max(cpu_parity["max_abs"], cpu_parity["energy_abs"]) <= CPU_CUDA_ATOL,
    }
    record = {
        "schema": CASE_SCHEMA,
        "case": name,
        "backend": backend,
        "status": "pass" if all(gates.values()) else "fail",
        "model": {
            "method": "canonical RHF-MP2",
            "approximation": "conventional unscreened four-center",
            "all_electron": True,
            "frozen_core": False,
            "basis_representation": case.basis_representation,
            "charge": case.charge,
            "ao_count": reference["ao_count"],
            "units": {
                "coordinates": "Bohr",
                "energy": "Hartree",
                "force": "Hartree/Bohr",
            },
        },
        "energies": {
            "public": base.energy,
            "pyscf": reference["energy"],
            "absolute_error": abs(base.energy - reference["energy"]),
        },
        "forces": {"public": public_forces.tolist(), "pyscf": reference["forces"]},
        "errors": {
            "pyscf_max_abs": independent_error["max_abs"],
            "pyscf_rms": independent_error["rms"],
            "finite_difference": finite_errors,
        },
        "invariants": invariants,
        "batch": {
            "force_max_abs": batch_force_error,
            "energy_max_abs": batch_energy_error,
            "item_statuses": [item.status for item in batch.items],
        },
        "cpu_parity": cpu_parity,
        "diagnostics": diagnostic,
        "independent_reference": reference,
        "gates": gates,
    }
    if record["status"] == "pass":
        validate_case_record(record, expected_steps=fd_steps)
    return record


def _environment_record(calculator=None) -> dict:
    library = None
    if calculator is not None:
        library = Path(calculator._library._name).resolve()
    record = {
        "source": _git_identity(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "toolchain": {"python": sys.version, "numpy": np.__version__},
        "library": None,
        "environment": {
            "VIBEQC_LIBRARY": os.environ.get("VIBEQC_LIBRARY"),
            "VIBEQC_PROFILE": os.environ.get("VIBEQC_PROFILE"),
        },
    }
    if library and library.exists():
        record["library"] = {
            "path": str(library),
            "bytes": library.stat().st_size,
            "sha256": _file_sha256(library),
        }
        getter = getattr(calculator._library, "vibeqc_get_source_identity", None)
        if getter is not None:
            getter.restype = ctypes.c_char_p
            value = getter()
            record["library"]["source_identity"] = value.decode() if value else None
    return record


def execute_validation(
    *,
    backend: str,
    selected_cases: Sequence[str],
    fd_steps: Sequence[float],
    output: Path,
    budget: int,
    fd_workers: int,
    command: Sequence[str],
) -> dict:
    """Execute the B2 matrix, preserving every case result in a fresh directory."""

    cases = validation_cases()
    output = initialize_run_directory(
        output,
        backend=backend,
        cases=selected_cases,
        fd_steps=fd_steps,
        command=command,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["finite_difference_workers"] = fd_workers
    first_calculator = _calculator(cases[selected_cases[0]], backend, budget)
    manifest["environment"] = _environment_record(first_calculator)
    manifest["tolerances"] = {
        "pyscf_force_max_abs": PYSCF_FORCE_ATOL,
        "finite_difference_fine_max_abs": FD_FINE_ATOL,
        "translation_force_norm": TRANSLATION_ATOL,
        "torque_norm": TORQUE_ATOL,
        "rotational_covariance_max_abs": ROTATION_ATOL,
        "batch_max_abs": BATCH_ATOL,
        "cpu_cuda_max_abs": CPU_CUDA_ATOL,
        "energy_force_identity_abs": ENERGY_FORCE_IDENTITY_ATOL,
    }
    failures = _run_failure_matrix(cases[selected_cases[0]], backend)
    manifest["failure_matrix"] = failures
    all_passed = all(
        row["status"] in {"pass", "covered-by-native-test"} for row in failures.values()
    )
    try:
        for name in selected_cases:
            record = run_case(
                name,
                cases[name],
                backend=backend,
                fd_steps=fd_steps,
                budget=budget,
                fd_workers=fd_workers,
            )
            relative = f"cases/{name}.json"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_json(target, record)
            manifest["case_records"].append(
                {
                    "case": name,
                    "path": relative,
                    "sha256": _file_sha256(target),
                    "bytes": target.stat().st_size,
                    "status": record["status"],
                }
            )
            all_passed = all_passed and record["status"] == "pass"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "detail": str(error)}
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
        raise
    manifest["status"] = "pass" if all_passed else "fail"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    if not all_passed:
        raise RuntimeError("one or more public MP2 force qualification gates failed")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    parser.add_argument("--cases", default=",".join(validation_cases()))
    parser.add_argument("--fd-steps", default="0.004,0.002,0.001")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-bytes", type=int, default=512 << 20)
    parser.add_argument("--fd-workers", type=int, default=1)
    parser.add_argument("--list-cases", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_cases:
        for name, case in validation_cases().items():
            print(f"{name}\tminimum_ao_count={case.minimum_ao_count}")
        return 0
    if args.backend is None or args.output is None:
        parser.error("--backend and --output are required unless --list-cases is used")
    selected = tuple(item.strip() for item in args.cases.split(",") if item.strip())
    unknown = set(selected) - set(validation_cases())
    if not selected or unknown:
        parser.error(f"unknown or empty case selection: {sorted(unknown)}")
    if args.budget_bytes < 1:
        parser.error("--budget-bytes must be positive")
    if args.fd_workers < 1:
        parser.error("--fd-workers must be positive")
    if args.backend == "cuda" and args.fd_workers != 1:
        parser.error("CUDA validation requires --fd-workers 1")
    execute_validation(
        backend=args.backend,
        selected_cases=selected,
        fd_steps=parse_fd_steps(args.fd_steps),
        output=args.output,
        budget=args.budget_bytes,
        fd_workers=args.fd_workers,
        command=(sys.executable, *sys.argv),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

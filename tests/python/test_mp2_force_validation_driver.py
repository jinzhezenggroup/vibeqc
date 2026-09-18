"""Contract tests for the public conventional MP2 force evidence driver."""

import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_validation_cases_cover_required_public_force_domain():
    from tools.validate_mp2_public_force import validation_cases

    cases = validation_cases()
    assert tuple(cases) == ("h2", "lih", "h2o", "f-shell", "water-def2-svp")
    assert any(shell.angular_momentum == 3 for shell in cases["f-shell"].vibeqc_basis)
    assert cases["water-def2-svp"].minimum_ao_count > 12


def test_fd_steps_require_three_distinct_positive_values():
    from tools.validate_mp2_public_force import parse_fd_steps

    assert parse_fd_steps("0.004,0.002,0.001") == (0.004, 0.002, 0.001)
    for value in ("0.004,0.002", "0.004,0.004,0.001", "0.004,0,-0.001"):
        with pytest.raises(ValueError, match="three distinct positive"):
            parse_fd_steps(value)


def test_run_directory_is_fresh_and_manifest_is_machine_readable(tmp_path):
    from tools.validate_mp2_public_force import initialize_run_directory

    destination = tmp_path / "run"
    initialize_run_directory(
        destination,
        backend="cpu",
        cases=("h2",),
        fd_steps=(0.004, 0.002, 0.001),
        command=("python", "tools/validate_mp2_public_force.py"),
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["schema"] == "vibeqc.mp2-public-force-validation.v1"
    assert manifest["backend"] == "cpu"
    assert manifest["cases"] == ["h2"]
    assert manifest["fd_steps_bohr"] == [0.004, 0.002, 0.001]
    assert manifest["status"] == "running"
    with pytest.raises(FileExistsError):
        initialize_run_directory(
            destination,
            backend="cpu",
            cases=("h2",),
            fd_steps=(0.004, 0.002, 0.001),
            command=("python",),
        )


def test_case_record_requires_complete_scientific_and_resource_gates():
    from tools.validate_mp2_public_force import validate_case_record

    record = {
        "schema": "vibeqc.mp2-public-force-case.v1",
        "case": "h2",
        "backend": "cpu",
        "status": "pass",
        "forces": {
            "public": [[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]],
            "pyscf": [[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]],
        },
        "errors": {
            "pyscf_max_abs": 0.0,
            "pyscf_rms": 0.0,
            "finite_difference": [
                {"step_bohr": step, "max_abs": 1.0e-8, "rms": 1.0e-8}
                for step in (0.004, 0.002, 0.001)
            ],
        },
        "invariants": {
            "translation_force_norm": 0.0,
            "torque_norm": 0.0,
            "rotational_covariance_max_abs": 0.0,
            "changed_geometry_energy_delta": 1.0e-3,
            "changed_geometry_force_delta": 1.0e-3,
            "energy_force_identity_error": 1.0e-12,
        },
        "diagnostics": {
            "response_iterations": 1,
            "response_absolute_residual": 1.0e-12,
            "response_relative_residual": 1.0e-12,
            "planned_endpoint_peak_bytes": 1024,
            "measured_endpoint_peak_bytes": 1024,
            "numeric_capacity_bytes": 2048,
            "equation_hash": "a" * 64,
            "response_operator_hash": "rhf-canonical-response-v1",
            "force_provenance_flags": 7,
        },
    }
    validate_case_record(record, expected_steps=(0.004, 0.002, 0.001))
    record["diagnostics"]["measured_endpoint_peak_bytes"] = 0
    with pytest.raises(ValueError, match="requires a measured endpoint peak"):
        validate_case_record(record, expected_steps=(0.004, 0.002, 0.001))
    record["diagnostics"]["measured_endpoint_peak_bytes"] = 1024
    record["errors"]["finite_difference"].pop()
    with pytest.raises(ValueError, match="finite-difference steps"):
        validate_case_record(record, expected_steps=(0.004, 0.002, 0.001))


def test_full_cartesian_finite_difference_recomputes_every_displacement():
    from tools.validate_mp2_public_force import central_finite_difference_forces

    class QuadraticCalculator:
        def __init__(self):
            self.calls = 0

        def singlepoint(self, atoms, *, charge=0, properties=("energy",)):
            del charge, properties
            self.calls += 1
            coordinates = np.asarray([position for _, position in atoms])
            return SimpleNamespace(energy=float(np.sum(coordinates**2)))

    calculator = QuadraticCalculator()
    atoms = (("H", (0.2, -0.1, 0.3)), ("H", (-0.4, 0.5, -0.6)))
    records = central_finite_difference_forces(
        calculator, atoms, charge=0, steps=(0.004, 0.002, 0.001)
    )
    assert calculator.calls == 2 * 2 * 3 * 3
    for record in records:
        np.testing.assert_allclose(
            record["forces"], -2 * np.asarray([a[1] for a in atoms])
        )


def test_fd_displacements_preserve_full_ordered_matrix_and_input():
    from tools.validate_mp2_public_force import _finite_difference_displacements

    atoms = (("H", (0.2, -0.1, 0.3)), ("H", (-0.4, 0.5, -0.6)))
    records = _finite_difference_displacements(atoms, (0.004, 0.002, 0.001))
    assert [step for step, _ in records] == [0.004, 0.002, 0.001]
    assert all(len(rows) == 2 * 3 for _, rows in records)
    first = records[0][1][0]
    assert first[:2] == (0, 0)
    np.testing.assert_allclose(first[2][0][1], (0.204, -0.1, 0.3))
    np.testing.assert_allclose(first[3][0][1], (0.196, -0.1, 0.3))
    assert atoms == (("H", (0.2, -0.1, 0.3)), ("H", (-0.4, 0.5, -0.6)))


def test_parallel_fd_requires_cpu_worker_pool_and_cuda_rejects_it(tmp_path):
    from tools.validate_mp2_public_force import (
        main,
        parallel_central_finite_difference_forces,
        validation_cases,
    )

    with pytest.raises(ValueError, match="at least two workers"):
        parallel_central_finite_difference_forces(
            "h2",
            validation_cases()["h2"],
            budget=1 << 20,
            steps=(0.004, 0.002, 0.001),
            workers=1,
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--backend",
                "cuda",
                "--output",
                str(tmp_path / "cuda"),
                "--fd-workers",
                "2",
            ]
        )


def test_parallel_fd_preserves_serial_force_order(monkeypatch):
    import tools.validate_mp2_public_force as driver

    class ImmediateExecutor:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 2
            assert mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, tasks):
            return tuple(function(task) for task in tasks)

    def quadratic_energy(payload):
        _, _, _, atoms = payload
        coordinates = np.asarray([position for _, position in atoms])
        return float(np.sum(coordinates**2))

    monkeypatch.setattr(driver, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(driver, "_finite_difference_energy_task", quadratic_energy)
    case = driver.PublicForceCase(
        atoms=(("H", (0.2, -0.1, 0.3)), ("H", (-0.4, 0.5, -0.6))),
        vibeqc_basis="sto-3g",
        pyscf_basis="sto-3g",
    )
    records = driver.parallel_central_finite_difference_forces(
        "h2",
        case,
        budget=1 << 20,
        steps=(0.004, 0.002, 0.001),
        workers=2,
    )
    for record in records:
        np.testing.assert_allclose(
            record["forces"], -2 * np.asarray([atom[1] for atom in case.atoms])
        )


def test_force_invariants_report_translation_and_torque_norms():
    from tools.validate_mp2_public_force import force_invariants

    positions = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    forces = np.asarray([[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]])
    metrics = force_invariants(positions, forces)
    assert metrics == {"translation_force_norm": 0.0, "torque_norm": 0.0}

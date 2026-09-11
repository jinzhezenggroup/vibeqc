"""Explicit Slurm GPU gates for generic weights and complete Direct/DF forces."""

import copy
import os

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

from tools.vibeqc_validation.one_electron_gradient import (
    execute_gradient,
    reference_matrices,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_ONE_ELECTRON_DERIVATIVE_CUDA_TEST") != "1",
    reason="explicit Slurm one-electron derivative tier",
)


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_arbitrary_nonsymmetric_weights_raw_fused_and_bounded_schedules(representation):
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    inputs = {
        "atomic_numbers": [2, 1],
        "coordinates": [[0.2, -0.3, -0.7], [0.1, 0.2, 0.7]],
        "basis_representation": representation,
        "charge": 1,
        "multiplicity": 1,
        "shells": [
            {
                "atom_index": 0,
                "angular_momentum": angular_momentum,
                "primitives": [
                    [0.7 + 0.2 * angular_momentum, 0.8],
                    [1.8 + 0.1 * angular_momentum, -0.13],
                ],
            }
            for angular_momentum in (0, 1, 2, 3)
        ]
        + [{"atom_index": 1, "angular_momentum": 0, "primitives": [[1.2, 1.0]]}],
    }
    calculator = Calculator(
        device="cuda",
        basis_representation=representation,
        basis=[
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                tuple(Primitive(*p) for p in s["primitives"]),
            )
            for s in inputs["shells"]
        ],
    )
    atoms = [(z, tuple(r)) for z, r in zip(("He", "H"), inputs["coordinates"])]
    values, derivative = reference_matrices(inputs)
    weights = np.random.default_rng(141).normal(size=values.shape)
    expected = np.einsum(
        "aoij,oij->a", derivative.reshape(6, 3, *values.shape[1:]), weights
    ).reshape(2, 3)
    outputs = []
    for schedule in (0, 1, 2):
        actual, resources = execute_gradient(
            calculator,
            atoms,
            weights,
            schedule=schedule,
            maximum_bytes=128 << 10,
            charge=1,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-11)
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-11)
        assert resources["device_to_host_bytes"] == 3 * len(atoms) * 8
        assert resources["device_bytes"] <= 128 << 10
        assert resources["host_numeric_bytes"] <= 128 << 10
        assert resources["stream_synchronizations"] == 1
        outputs.append(actual)
    serial, _ = execute_gradient(
        calculator, atoms, weights, schedule=2, maximum_bytes=128 << 10, charge=1
    )
    np.testing.assert_array_equal(serial, outputs[2])
    # A smaller arena must reject before executing an unbounded fallback.
    with pytest.raises(RuntimeError, match="budget|maximum_bytes"):
        execute_gradient(calculator, atoms, weights, maximum_bytes=32, charge=1)
    for step in (2e-4, 5e-5):
        for atom in range(2):
            for axis in range(3):
                plus, minus = copy.deepcopy(inputs), copy.deepcopy(inputs)
                plus["coordinates"][atom][axis] += step
                minus["coordinates"][atom][axis] -= step
                finite_difference = np.sum(
                    weights
                    * (reference_matrices(plus)[0] - reference_matrices(minus)[0])
                ) / (2 * step)
                assert expected[atom, axis] == pytest.approx(
                    finite_difference, abs=2e-6, rel=2e-6
                )


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("count", [1, 3])
def test_generated_derivatives_preserve_complete_scf_forces(
    monkeypatch, method, representation, fitted, count
):
    from test_one_electron_values_cuda import run_case

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    options = {
        "method": method,
        "representation": representation,
        "fitted": fitted,
        "count": count,
    }
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "reference")
    reference = run_case(monkeypatch, mapping="thread", **options)
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "generated")
    for mapping in ("thread", "shell_warp"):
        monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING", mapping)
        actual = run_case(monkeypatch, mapping="thread", **options)
        for left, right in zip(reference, actual):
            np.testing.assert_allclose(
                left.energies, right.energies, atol=3e-10, rtol=0
            )
            for expected, found in zip(left.items, right.items):
                assert found.executed_backend == "cuda"
                np.testing.assert_allclose(
                    found.forces, expected.forces, atol=3e-9, rtol=0
                )


@pytest.mark.parametrize("fitted", [False, True])
def test_derivative_selectors_on_reused_plan_match_fresh_execution(monkeypatch, fitted):
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    calc = Calculator(
        device="cuda",
        density_fitting="cuda" if fitted else "none",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "reference")
    with calc.prepare_batch([atoms]) as batch:
        batch.execute(strict=True)
        for selection, mapping in (
            ("generated", "thread"),
            ("generated", "shell_warp"),
            ("reference", "thread"),
        ):
            monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", selection)
            monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING", mapping)
            actual = batch.execute(strict=True).items[0]
            expected = calc.singlepoint(atoms)
            np.testing.assert_allclose(
                actual.forces, expected.forces, atol=3e-9, rtol=0
            )


@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
@pytest.mark.parametrize("fitted", [False, True])
def test_generated_target_forces_against_independent_pyscf(
    monkeypatch, method, charge, multiplicity, fitted
):
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0.2, 0.7))]
    mol = gto.M(
        atom=atoms,
        basis="def2-svp",
        unit="Bohr",
        cart=True,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    reference = (scf.RHF if method == "rhf" else scf.UHF)(mol)
    if fitted:
        reference = reference.density_fit(auxbasis="def2-svp")
    reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-10
    reference.kernel()
    assert reference.converged
    forces = -reference.nuc_grad_method().kernel()
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "generated")
    for mapping in ("thread", "shell_warp"):
        monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING", mapping)
        result = Calculator(
            device="cuda",
            method=method,
            basis="def2-svp",
            density_fitting="cuda" if fitted else "none",
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
        ).singlepoint(atoms, charge=charge, multiplicity=multiplicity)
        np.testing.assert_allclose(result.energy, reference.e_tot, atol=3e-10, rtol=0)
        np.testing.assert_allclose(result.forces, forces, atol=3e-9, rtol=0)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        rotated_atoms = [
            (element, tuple(rotation @ np.array(r) + [1.2, -0.4, 0.8]))
            for element, r in atoms
        ]
        rotated = Calculator(
            device="cuda",
            method=method,
            basis="def2-svp",
            density_fitting="cuda" if fitted else "none",
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
        ).singlepoint(rotated_atoms, charge=charge, multiplicity=multiplicity)
        np.testing.assert_allclose(rotated.energy, result.energy, atol=3e-10, rtol=0)
        np.testing.assert_allclose(
            rotated.forces, result.forces @ rotation.T, atol=3e-9, rtol=0
        )


def test_failed_item_does_not_contaminate_generated_neighbor(monkeypatch):
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "generated")
    atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    other = [("He", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    calc = Calculator(device="cuda", max_iterations=3)
    with calc.prepare_batch([atoms, other], charges=[0, 1]) as batch:
        result = batch.execute()
        assert result.items[0].succeeded and not result.items[1].succeeded
        expected = Calculator(device="cuda").singlepoint(atoms)
        np.testing.assert_allclose(
            result.items[0].forces, expected.forces, atol=3e-9, rtol=0
        )


@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
@pytest.mark.parametrize("fitted", [False, True])
def test_screened_target_energy_force_domain(
    monkeypatch, method, charge, multiplicity, fitted
):
    """Measure total force errors; a value-screen threshold alone is no bound.

    One-electron contractions are always unscreened. The positive-threshold API
    uses 1e-30 for the near-unscreened two-electron reference on this fixture.
    """
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", "generated")
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.2, 0.7))]
    options = {
        "device": "cuda",
        "method": method,
        "basis": "def2-svp",
        "density_fitting": "cuda" if fitted else "none",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    reference = Calculator(**options, screening_tolerance=1e-30).singlepoint(
        atoms, charge=charge, multiplicity=multiplicity
    )
    for threshold in (1e-14, 1e-9):
        calc = Calculator(**options, screening_tolerance=threshold)
        actual = calc.singlepoint(atoms, charge=charge, multiplicity=multiplicity)
        np.testing.assert_allclose(actual.energy, reference.energy, atol=3e-10, rtol=0)
        np.testing.assert_allclose(actual.forces, reference.forces, atol=3e-9, rtol=0)
        for step in (2e-4, 5e-5):
            displaced = []
            for sign in (1, -1):
                geometry = [
                    (element, np.array(position)) for element, position in atoms
                ]
                geometry[1][1][2] += sign * step
                displaced.append(
                    calc.singlepoint(
                        geometry, charge=charge, multiplicity=multiplicity
                    ).energy
                )
            finite_force = -(displaced[0] - displaced[1]) / (2 * step)
            assert actual.forces[1, 2] == pytest.approx(finite_force, abs=3e-7, rel=0)

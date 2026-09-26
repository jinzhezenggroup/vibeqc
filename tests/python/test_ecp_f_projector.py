"""Independent gates for f projectors, distinct from the orbital-f capability."""

import os
import typing

import numpy as np
import pytest
from test_ecp import detached_native, detached_reference, fixture, reference
from vibeqc import Calculator, ResourceBudget
from vibeqc.ecp import ecp_integrals


def require_device(device: typing.Any) -> None:
    if device == "cuda" and os.environ.get("VIBEQC_ECP_CUDA_TEST") != "1":
        pytest.skip("requires an allocated CUDA device")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("power", range(5))
def test_f_projector_detached_all_center_jets_and_weights(
    device: typing.Any, power: typing.Any
) -> None:
    require_device(device)
    xyz = np.array([[0.13, -0.21, 0.17], [0.43, 0.19, 1.2], [-0.21, 0.11, -0.7]])
    options = {"power": power, "orbital": 3, "d_projector": True, "f_projector": True}
    actual = detached_native(xyz, device=device, **options)
    expected = detached_reference(xyz, **options)
    np.testing.assert_allclose(actual[:, 0], expected, atol=2e-9, rtol=1e-9)
    without_f = detached_reference(xyz, **(options | {"f_projector": False}))
    assert np.max(np.abs(expected[1] - without_f[1])) > 1e-4
    jets = actual[:, 1:].reshape(2, 3, 3, *actual.shape[-2:])
    np.testing.assert_allclose(jets.sum(axis=1), 0, atol=3e-12)
    weights = np.random.default_rng(17143).normal(size=actual.shape[-2:])
    for step in (2e-4, 7e-5):
        for atom in range(3):
            for axis in range(3):
                delta = np.zeros_like(xyz)
                delta[atom, axis] = step
                fd = (
                    detached_reference(xyz + delta, **options)
                    - detached_reference(xyz - delta, **options)
                ) / (2 * step)
                np.testing.assert_allclose(
                    jets[:, atom, axis], fd, atol=3e-7, rtol=3e-6
                )
                np.testing.assert_allclose(
                    np.einsum("pij,ij->p", jets[:, atom, axis], weights),
                    np.einsum("pij,ij->p", fd, weights),
                    atol=2e-6,
                    rtol=3e-6,
                )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("spin", [0, 1])
def test_f_projector_complete_hf(
    device: typing.Any, representation: typing.Any, spin: typing.Any
) -> None:
    require_device(device)
    scf = pytest.importorskip("pyscf.scf")
    atoms, basis, mol = fixture(
        representation=representation, spin=spin, f_shell=True, f_projector=True
    )
    raw = ecp_integrals(atoms, basis, device=device)
    np.testing.assert_allclose(raw.matrix, reference(mol), atol=2e-9, rtol=1e-9)
    oracle = (scf.UHF(mol) if spin else scf.RHF(mol)).run(conv_tol=1e-12)
    assert oracle.converged
    result = Calculator(
        method="uhf" if spin else "rhf", basis=basis, device=device
    ).singlepoint(atoms, charge=spin, multiplicity=spin + 1)
    assert result.converged
    np.testing.assert_allclose(result.energy, oracle.e_tot, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        result.forces, -oracle.nuc_grad_method().kernel(), atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-7)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_f_projector_budgeted_replay_and_energy_difference(
    device: typing.Any,
) -> None:
    require_device(device)
    atoms, basis, mol = fixture(f_projector=True)
    calculator = Calculator(basis=basis, device=device)
    plan = calculator.estimate_resources([atoms]).require_feasible()
    bounded = Calculator(
        basis=basis,
        device=device,
        resource_budget=ResourceBudget(
            host_bytes=plan.peak_bytes["host"],
            device_bytes=plan.peak_bytes.get("device"),
        ),
    )
    direction = np.array([[0.23, -0.31, 0.17], [-0.19, 0.11, 0.29]])
    with bounded.prepare_batch([atoms]) as batch:
        initial = batch.execute(strict=True).items[0]
        assert initial.converged
        for step in (2e-4, 7e-5):
            energies = []
            for sign in (-1, 1):
                xyz = mol.atom_coords() + sign * step * direction
                replay = batch.execute(coordinates=[xyz], strict=True).items[0]
                assert replay.converged
                energies.append(replay.energy)
            assert (energies[1] - energies[0]) / (2 * step) == pytest.approx(
                -np.sum(initial.forces * direction), abs=2e-6
            )
        if device == "cuda":
            ledger = batch.resource_diagnostics["observation"]["device_ledger"]
            assert ledger["rejected_allocations"] == 0
            assert 0 < ledger["peak_bytes"] <= ledger["limit_bytes"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_f_projector_f_orbital_refinement(
    device: typing.Any, representation: typing.Any
) -> None:
    require_device(device)
    atoms, basis, mol = fixture(
        representation=representation, f_shell=True, f_projector=True
    )
    raw = ecp_integrals(atoms, basis, device=device)
    fine = ecp_integrals(
        atoms, basis, device=device, radial_points=224, polar_points=44
    )
    value_only = ecp_integrals(atoms, basis, device=device, derivatives=False)
    np.testing.assert_allclose(raw.matrix, reference(mol), atol=2e-9, rtol=1e-9)
    for component in ("local", "nonlocal_"):
        np.testing.assert_allclose(
            getattr(raw, component), getattr(fine, component), atol=2e-9, rtol=0
        )
        np.testing.assert_allclose(
            getattr(raw, component), getattr(value_only, component), atol=2e-12, rtol=0
        )
    for component in ("local_derivative", "nonlocal_derivative"):
        np.testing.assert_allclose(
            getattr(raw, component), getattr(fine, component), atol=2e-8, rtol=0
        )

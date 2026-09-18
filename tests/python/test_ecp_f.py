"""Bounded orbital-f ECP extension; projector angular momentum remains <= d."""

import os

import numpy as np
import pytest
from test_ecp import detached_native, detached_reference, fixture, reference
from vibeqc import Calculator, ResourceBudget
from vibeqc.ecp import ecp_integrals


def require_device(device):
    if device == "cuda" and os.environ.get("VIBEQC_ECP_CUDA_TEST") != "1":
        pytest.skip("requires an allocated CUDA device")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_f_raw_matrices_all_center_derivatives_and_weights(device, representation):
    require_device(device)
    atoms, basis, mol = fixture(
        representation=representation, f_shell=True, f_on_h=True
    )
    result = ecp_integrals(atoms, basis, device=device)
    refined = ecp_integrals(
        atoms, basis, device=device, radial_points=224, polar_points=44
    )
    np.testing.assert_allclose(result.matrix, reference(mol), atol=2e-9, rtol=1e-9)
    jets = result.local_derivative + result.nonlocal_derivative
    np.testing.assert_allclose(result.local, refined.local, atol=2e-9, rtol=1e-9)
    np.testing.assert_allclose(
        result.nonlocal_, refined.nonlocal_, atol=2e-9, rtol=1e-9
    )
    np.testing.assert_allclose(
        result.local_derivative, refined.local_derivative, atol=2e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        result.nonlocal_derivative, refined.nonlocal_derivative, atol=2e-8, rtol=1e-8
    )
    np.testing.assert_allclose(jets.sum(axis=0), 0, atol=2e-12)
    xyz = mol.atom_coords()
    weights = np.random.default_rng(17130).normal(size=result.matrix.shape)
    for step in (2e-4, 7e-5):
        expected = np.zeros_like(jets)
        for atom in range(len(atoms)):
            for axis in range(3):
                delta = np.zeros_like(xyz)
                delta[atom, axis] = step
                plus = mol.copy().set_geom_(xyz + delta, unit="Bohr")
                minus = mol.copy().set_geom_(xyz - delta, unit="Bohr")
                expected[atom, axis] = (reference(plus) - reference(minus)) / (2 * step)
        np.testing.assert_allclose(jets, expected, atol=3e-7, rtol=3e-6)
        np.testing.assert_allclose(
            result.contract(weights),
            np.einsum("axij,ij->ax", expected, weights),
            atol=2e-6,
            rtol=3e-6,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_f_detached_ecp_center_and_d_projector(device):
    require_device(device)
    xyz = np.array([[0.13, -0.21, 0.17], [0.43, 0.19, 1.2], [-0.21, 0.11, -0.7]])
    actual = detached_native(xyz, device=device, orbital=3, d_projector=True)
    np.testing.assert_allclose(
        actual[:, 0],
        detached_reference(xyz, orbital=3, d_projector=True),
        atol=2e-9,
        rtol=1e-9,
    )
    for step in (2e-4, 7e-5):
        for axis in range(3):
            delta = np.zeros_like(xyz)
            delta[0, axis] = step
            expected = (
                detached_reference(xyz + delta, orbital=3, d_projector=True)
                - detached_reference(xyz - delta, orbital=3, d_projector=True)
            ) / (2 * step)
            np.testing.assert_allclose(
                actual[:, 1 + axis], expected, atol=3e-7, rtol=3e-6
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("spin", [0, 1])
def test_f_complete_hf_energy_and_force(device, representation, spin):
    require_device(device)
    scf = pytest.importorskip("pyscf.scf")
    atoms, basis, mol = fixture(representation=representation, f_shell=True, spin=spin)
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


def test_f_cuda_complete_energy_directional_finite_difference():
    require_device("cuda")
    atoms, basis, mol = fixture(f_shell=True)
    calculator = Calculator(basis=basis, device="cuda")
    actual = calculator.singlepoint(atoms)
    direction = np.array([[0.23, -0.31, 0.17], [-0.19, 0.11, 0.29]])
    for step in (2e-4, 7e-5):
        energies = []
        for sign in (-1, 1):
            xyz = mol.atom_coords() + sign * step * direction
            moved = [
                (symbol, tuple(x)) for (symbol, _), x in zip(atoms, xyz, strict=True)
            ]
            result = calculator.singlepoint(moved)
            assert result.converged
            energies.append(result.energy)
        derivative = (energies[1] - energies[0]) / (2 * step)
        assert derivative == pytest.approx(-np.sum(actual.forces * direction), abs=2e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_f_prepared_replay_with_planned_budget(device):
    require_device(device)
    # Complete HF above covers f on the ECP atom. This replay gate covers f
    # on the all-electron atom too, without repeating the very expensive
    # two-f-center CPU reference on every PR. Keep that larger case opt-in.
    both_centers = device == "cpu" and os.getenv("VIBEQC_ECP_LARGE_CPU_TEST") == "1"
    atoms, basis, _ = fixture(f_shell=both_centers, f_on_h=True)
    calculator = Calculator(basis=basis, device=device)
    if device == "cuda":
        # The existing small CUDA allocation inventory stops at 16 public
        # AOs. Orbital f does not qualify a larger resource provider.
        larger_atoms, larger_basis, _ = fixture(f_shell=True, f_on_h=True)
        larger = Calculator(basis=larger_basis, device=device)
        with pytest.raises(NotImplementedError, match="<=16 public AOs"):
            larger.estimate_resources([larger_atoms]).require_feasible()
    plan = calculator.estimate_resources([atoms]).require_feasible()
    bounded = Calculator(
        basis=basis,
        device=device,
        resource_budget=ResourceBudget(
            host_bytes=plan.peak_bytes["host"],
            device_bytes=plan.peak_bytes.get("device"),
        ),
    )
    with bounded.prepare_batch([atoms]) as batch:
        initial = batch.execute(strict=True).items[0]
        assert initial.converged
        moved = [
            (symbol, (x + 0.03 * i, y - 0.02 * i, z + 0.01))
            for i, (symbol, (x, y, z)) in enumerate(atoms)
        ]
        replay = batch.execute(coordinates=[[r for _, r in moved]], strict=True)
        fresh = calculator.singlepoint(moved)
        assert fresh.converged and replay.items[0].converged
        np.testing.assert_allclose(replay.energies[0], fresh.energy, atol=2e-8, rtol=0)
        np.testing.assert_allclose(
            replay.items[0].forces, fresh.forces, atol=2e-6, rtol=0
        )
        if device == "cuda":
            ledger = batch.resource_diagnostics["observation"]["device_ledger"]
            assert ledger["rejected_allocations"] == 0
            assert 0 < ledger["peak_bytes"] <= ledger["limit_bytes"]

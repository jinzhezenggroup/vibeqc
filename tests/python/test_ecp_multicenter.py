"""Mixed LANL2DZ Na / Stuttgart RLC K: two physical ECP centers."""

import json
import typing

import numpy as np
import pytest
from test_ecp_heavy import reference_components, require_device
from vibeqc import (
    Atom,
    BasisProvenance,
    BasisSet,
    BasisShell,
    Calculator,
    ElementBasis,
    ResourceBudget,
)
from vibeqc.ecp import ecp_integrals, resolve_ecp
from vibeqc.profiles import canonical_hash


def multicenter_fixture(
    *, spin: typing.Any = 0, reverse: typing.Any = False, displacement: typing.Any = 0.0
) -> typing.Any:
    pyscf = pytest.importorskip("pyscf")
    gto = pytest.importorskip("pyscf.gto")
    families = {"Na": "lanl2dz", "K": "stuttgart-dz"}
    orbitals = {s: gto.basis.load(f, s) for s, f in families.items()}
    potentials = {s: gto.basis.load_ecp(f, s) for s, f in families.items()}
    checksum = canonical_hash({"basis": orbitals, "ecp": potentials})
    assert (
        checksum == "e7860773555887760528a11fcbc37958a2923df0fd5d4e6f2874e17b5acf29c0"
    )
    elements = []
    for symbol, z, core in (("Na", 11, 10), ("K", 19, 18)):
        assert potentials[symbol][0] == core
        shells = tuple(
            BasisShell(
                shell[0],
                tuple(str(row[0]) for row in shell[1:]),
                tuple(
                    tuple(str(row[column]) for row in shell[1:])
                    for column in range(1, len(shell[1]))
                ),
            )
            for shell in orbitals[symbol]
        )
        channels = potentials[symbol][1]
        local_label = max(channel for channel, _ in channels) + 1
        records = []
        for channel, powers in channels:
            terms = [(n, a, c) for n, rows in enumerate(powers) for a, c in rows]
            # PySCF drops Stuttgart's explicit zero local residual.
            if channel == -1 and not terms:
                terms = [(2, 1.0, 0.0)]
            records.append(
                {
                    "ecp_type": "scalar_ecp",
                    "angular_momentum": [local_label if channel == -1 else channel],
                    "r_exponents": [n for n, _, _ in terms],
                    "gaussian_exponents": [str(a) for _, a, _ in terms],
                    "coefficients": [[str(c) for _, _, c in terms]],
                }
            )
        elements.append(
            ElementBasis(
                z, shells, ecp_core_electrons=core, ecp_data=json.dumps(records)
            )
        )
    basis = BasisSet(
        "LANL2DZ-Na/Stuttgart-RLC-K qualification",
        tuple(elements),
        BasisProvenance(
            "PySCF test-only installed basis library",
            pyscf.__version__,
            "upstream external test data; not redistributed",
            checksum,
        ),
        representation="spherical",
    )
    atoms = [("Na", (0.13, -0.21, 0.17)), ("K", (0.43, 0.29, 5.8 + displacement))]
    if reverse:
        atoms.reverse()
    mol = gto.M(
        atom=atoms,
        basis=orbitals,
        ecp=potentials,
        unit="Bohr",
        spin=spin,
        charge=spin,
        verbose=0,
    )
    return atoms, basis, mol


def center_components(mol: typing.Any, center: typing.Any) -> typing.Any:
    """Select one Libcint operator center while retaining every orbital center."""
    gto = pytest.importorskip("pyscf.gto")
    selected = mol.copy()
    selected._ecpbas = mol._ecpbas[mol._ecpbas[:, gto.ATOM_OF] == center]
    return reference_components(selected)


def calculator(
    basis: typing.Any, device: typing.Any, spin: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    return Calculator(
        basis=basis,
        device=device,
        method="uhf" if spin else "rhf",
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


def endpoint(
    device: typing.Any, spin: typing.Any, reverse: typing.Any = False
) -> typing.Any:
    require_device(device)
    scf = pytest.importorskip("pyscf.scf")
    atoms, basis, mol = multicenter_fixture(spin=spin, reverse=reverse)
    cores, terms = resolve_ecp(basis, tuple(Atom.from_value(a) for a in atoms))
    assert cores == ((18, 10) if reverse else (10, 18))
    assert {term[0] for term in terms} == {0, 1}
    assert mol.nelectron == 2 - spin
    assert mol.nao_nr() == 16
    np.testing.assert_array_equal(mol.atom_charges(), [1, 1])
    assert mol.energy_nuc() == pytest.approx(
        1 / np.linalg.norm(mol.atom_coords()[0] - mol.atom_coords()[1]), abs=1e-13
    )
    energies = []
    for guess in ("1e", "minao"):
        ref = scf.UHF(mol) if spin else scf.RHF(mol)
        ref.conv_tol, ref.conv_tol_grad = 1e-13, 1e-10
        ref.max_cycle, ref.init_guess = 150, guess
        ref.kernel()
        assert ref.converged
        energies.append(ref.e_tot)
    assert abs(energies[1] - energies[0]) < 2e-9
    result = calculator(basis, device, spin).singlepoint(
        atoms, charge=spin, multiplicity=spin + 1
    )
    assert result.converged
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    forces = -ref.nuc_grad_method().kernel()
    np.testing.assert_allclose(result.energy, ref.e_tot, atol=2e-8, rtol=0)
    np.testing.assert_allclose(result.forces, forces, atol=2e-6, rtol=0)
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-7)
    return atoms, basis, mol, result, ref.e_tot, forces


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("displacement", [0.0, 0.37])
def test_mixed_centers_raw_derivatives_and_operator_partition(
    device: typing.Any, displacement: typing.Any
) -> None:
    require_device(device)
    atoms, basis, mol = multicenter_fixture(displacement=displacement)
    raw = ecp_integrals(atoms, basis, device=device)
    fine = ecp_integrals(
        atoms, basis, device=device, radial_points=224, polar_points=44
    )
    values = ecp_integrals(atoms, basis, device=device, derivatives=False)
    reference = reference_components(mol)
    by_center = [center_components(mol, c) for c in range(2)]
    np.testing.assert_allclose(sum(by_center), reference, atol=2e-12, rtol=0)
    # Each distinct projector center must contribute; K's local residual is zero.
    assert np.max(np.abs(by_center[0][0])) > 1e-3
    np.testing.assert_array_equal(by_center[1][0], 0)
    assert all(np.max(np.abs(block[1])) > 1e-3 for block in by_center)
    np.testing.assert_allclose([raw.local, raw.nonlocal_], reference, atol=2e-9, rtol=0)
    for field in ("local", "nonlocal_"):
        np.testing.assert_allclose(
            getattr(raw, field), getattr(fine, field), atol=2e-9, rtol=0
        )
        np.testing.assert_allclose(
            getattr(raw, field), getattr(values, field), atol=2e-12, rtol=0
        )
    for field in ("local_derivative", "nonlocal_derivative"):
        np.testing.assert_allclose(
            getattr(raw, field), getattr(fine, field), atol=2e-8, rtol=0
        )
        np.testing.assert_allclose(getattr(raw, field).sum(axis=0), 0, atol=2e-11)
    jets = np.array([raw.local_derivative, raw.nonlocal_derivative])
    weights = np.random.default_rng(1711018).normal(size=raw.matrix.shape)
    for step in (2e-4, 7e-5):
        for atom in range(2):
            for axis in range(3):
                delta = np.zeros((2, 3))
                delta[atom, axis] = step
                plus = mol.copy().set_geom_(mol.atom_coords() + delta, unit="Bohr")
                minus = mol.copy().set_geom_(mol.atom_coords() - delta, unit="Bohr")
                fd = (reference_components(plus) - reference_components(minus)) / (
                    2 * step
                )
                np.testing.assert_allclose(
                    jets[:, atom, axis], fd, atol=3e-7, rtol=2e-6
                )
                assert raw.contract(weights)[atom, axis] == pytest.approx(
                    np.einsum("pij,ij->", fd, weights), abs=2e-6
                )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixed_centers_atom_and_ao_permutation(device: typing.Any) -> None:
    require_device(device)
    atoms, basis, mol = multicenter_fixture()
    reverse_atoms, reverse_basis, reverse_mol = multicenter_fixture(reverse=True)
    assert basis.identity == reverse_basis.identity
    first = ecp_integrals(atoms, basis, device=device)
    swapped = ecp_integrals(reverse_atoms, basis, device=device)
    slices = mol.aoslice_by_atom()[:, 2:]
    permutation = np.concatenate([np.arange(lo, hi) for lo, hi in slices[::-1]])
    np.testing.assert_allclose(
        reverse_mol.intor("int1e_ovlp"),
        mol.intor("int1e_ovlp")[np.ix_(permutation, permutation)],
        atol=2e-12,
        rtol=0,
    )
    for field in ("local", "nonlocal_", "local_derivative", "nonlocal_derivative"):
        expected = getattr(first, field)
        if expected.ndim == 4:
            expected = expected[::-1]
        expected = expected[..., permutation, :][..., permutation]
        np.testing.assert_allclose(
            getattr(swapped, field), expected, atol=2e-10, rtol=0
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("spin", [0, 1])
def test_mixed_centers_complete_hf_permutation(
    device: typing.Any, spin: typing.Any
) -> None:
    first = endpoint(device, spin)
    reverse = endpoint(device, spin, reverse=True)
    np.testing.assert_allclose(first[3].energy, reverse[3].energy, atol=2e-10, rtol=0)
    np.testing.assert_allclose(
        first[3].forces, reverse[3].forces[::-1], atol=2e-8, rtol=0
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("spin", [0, 1])
def test_mixed_centers_budgeted_replay_and_complete_difference(
    device: typing.Any, spin: typing.Any
) -> None:
    require_device(device)
    atoms, basis, mol = multicenter_fixture(spin=spin)
    plan = (
        calculator(basis, device, spin)
        .estimate_resources([atoms], charges=[spin], multiplicities=[spin + 1])
        .require_feasible()
    )
    calc = calculator(
        basis,
        device,
        spin,
        resource_budget=ResourceBudget(
            host_bytes=plan.peak_bytes["host"],
            device_bytes=plan.peak_bytes.get("device"),
        ),
    )
    direction = np.array([[0.23, -0.31, 0.17], [-0.19, 0.11, 0.29]])
    with calc.prepare_batch(
        [atoms], charges=[spin], multiplicities=[spin + 1]
    ) as batch:
        initial = batch.execute(strict=True).items[0]
        assert initial.converged
        for step in (2e-4, 7e-5):
            energies = []
            for sign in (-1, 1):
                moved = batch.execute(
                    coordinates=[mol.atom_coords() + sign * step * direction],
                    strict=True,
                ).items[0]
                assert moved.converged
                energies.append(moved.energy)
            assert (energies[1] - energies[0]) / (2 * step) == pytest.approx(
                -np.sum(initial.forces * direction), abs=2e-6
            )
        restored = batch.execute(coordinates=[mol.atom_coords()], strict=True).items[0]
        np.testing.assert_allclose(restored.energy, initial.energy, atol=2e-10, rtol=0)
        np.testing.assert_allclose(restored.forces, initial.forces, atol=2e-8, rtol=0)
        if device == "cuda":
            ledger = batch.resource_diagnostics["observation"]["device_ledger"]
            assert ledger["rejected_allocations"] == 0
            assert 0 < ledger["peak_bytes"] <= ledger["limit_bytes"]

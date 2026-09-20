"""Bounded real-parameter qualification: Stuttgart RLC Na/K with STO-3G hydrogen.

Parameters are loaded from the pinned test-only PySCF installation. This suite
does not establish coverage of other elements or ECP parameter families.
"""

import json
import os
import typing

import numpy as np
import pytest
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

# Atomic number, removed core, and an off-axis molecular geometry in bohr.
CASES = {"Na": (11, 10, 3.2), "K": (19, 18, 4.0)}
PARAMETER_SHA256 = {
    "Na": "72bf469c369370758895f0c70cf8a033346439ed0839c635e7baf5161b8ecf58",
    "K": "11efa860c4e39d99181857b0fac4e54fac4c7c108dc598ae1b49a2371319e77a",
}


def stuttgart_fixture(
    symbol: typing.Any, *, spin: typing.Any = 0, displacement: typing.Any = 0.0
) -> typing.Any:
    pyscf = pytest.importorskip("pyscf")
    gto = pytest.importorskip("pyscf.gto")
    z, core, bond_z = CASES[symbol]
    orbital = {
        symbol: gto.basis.load("stuttgart-dz", symbol),
        "H": gto.basis.load("sto-3g", "H"),
    }
    potentials = {symbol: gto.basis.load_ecp("stuttgart-dz", symbol)}
    assert potentials[symbol][0] == core
    assert {channel for channel, _ in potentials[symbol][1]} == {-1, 0, 1, 2}
    assert all(
        c == 0
        for channel, powers in potentials[symbol][1]
        if channel == -1
        for rows in powers
        for _, c in rows
    )
    checksum = canonical_hash({"basis": orbital, "ecp": potentials})
    assert checksum == PARAMETER_SHA256[symbol], "qualification parameters changed"
    atoms = [
        (symbol, (0.13, -0.21, 0.17)),
        ("H", (0.43, 0.19, bond_z + displacement)),
    ]
    mol = gto.M(
        atom=atoms,
        basis=orbital,
        ecp=potentials,
        unit="Bohr",
        spin=spin,
        charge=spin,
        verbose=0,
    )
    elements = []
    for label, atomic_number in ((symbol, z), ("H", 1)):
        shells = tuple(
            BasisShell(
                shell[0],
                tuple(str(row[0]) for row in shell[1:]),
                tuple(
                    tuple(str(row[column]) for row in shell[1:])
                    for column in range(1, len(shell[1]))
                ),
            )
            for shell in orbital[label]
        )
        records = []
        if label == symbol:
            channels = potentials[label][1]
            local_label = max(channel for channel, _ in channels) + 1
            for channel, powers in channels:
                terms = [(n, a, c) for n, rows in enumerate(powers) for a, c in rows]
                # PySCF discards the published zero UL coefficient. Preserve
                # that explicit local channel in the owned scalar schema.
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
                atomic_number,
                shells,
                ecp_core_electrons=core if label == symbol else 0,
                ecp_data=json.dumps(records) if records else None,
            )
        )
    basis = BasisSet(
        f"Stuttgart-RLC-{symbol}/STO-3G-H qualification",
        tuple(elements),
        BasisProvenance(
            "PySCF test-only installed basis library",
            pyscf.__version__,
            "upstream external test data; not redistributed",
            checksum,
        ),
        representation="spherical",
    )
    return atoms, basis, mol


def reference_components(mol: typing.Any) -> typing.Any:
    """Libcint local/nonlocal blocks selected independently of VibeQC terms."""
    gto = pytest.importorskip("pyscf.gto")
    norms = np.sqrt(mol.intor("int1e_ovlp").diagonal())
    blocks = []
    for local in (True, False):
        selected = mol.copy()
        selected._ecpbas = mol._ecpbas[(mol._ecpbas[:, gto.ANG_OF] == -1) == local]
        blocks.append(selected.intor("ECPscalar") / norms[:, None] / norms[None, :])
    return np.array(blocks)


def require_device(device: typing.Any) -> None:
    if device == "cuda" and os.environ.get("VIBEQC_ECP_CUDA_TEST") != "1":
        pytest.skip("requires an allocated CUDA device")


@pytest.mark.parametrize("symbol", CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stuttgart_components_refinement_and_all_center_derivatives(
    symbol: typing.Any, device: typing.Any
) -> None:
    require_device(device)
    for displacement in (0.0, 0.37):
        atoms, basis, mol = stuttgart_fixture(symbol, displacement=displacement)
        raw = ecp_integrals(atoms, basis, device=device)
        fine = ecp_integrals(
            atoms, basis, device=device, radial_points=224, polar_points=44
        )
        # This family has an exactly zero local residual, with a nonzero
        # Coulomb tail and signed s/p/d projector differences.
        np.testing.assert_array_equal(raw.local, 0)
        np.testing.assert_array_equal(raw.local_derivative, 0)
        assert np.max(np.abs(raw.nonlocal_)) > 1e-3
        values = ecp_integrals(atoms, basis, device=device, derivatives=False)
        np.testing.assert_allclose(
            [raw.local, raw.nonlocal_], reference_components(mol), atol=2e-9, rtol=0
        )
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
        weights = np.random.default_rng(1715546).normal(size=raw.matrix.shape)
        jets = np.array([raw.local_derivative, raw.nonlocal_derivative])
        xyz = mol.atom_coords()
        for step in (2e-4, 7e-5):
            for atom in range(2):
                for axis in range(3):
                    delta = np.zeros_like(xyz)
                    delta[atom, axis] = step
                    plus = mol.copy().set_geom_(xyz + delta, unit="Bohr")
                    minus = mol.copy().set_geom_(xyz - delta, unit="Bohr")
                    fd = (reference_components(plus) - reference_components(minus)) / (
                        2 * step
                    )
                    np.testing.assert_allclose(
                        jets[:, atom, axis], fd, atol=3e-7, rtol=2e-6
                    )
                    assert raw.contract(weights)[atom, axis] == pytest.approx(
                        np.einsum("pij,ij->", fd, weights), abs=2e-6
                    )


@pytest.mark.parametrize("symbol", CASES)
@pytest.mark.parametrize("spin", [0, 1])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stuttgart_complete_hf_and_core_bookkeeping(
    symbol: typing.Any, spin: typing.Any, device: typing.Any
) -> None:
    require_device(device)
    scf = pytest.importorskip("pyscf.scf")
    atoms, basis, mol = stuttgart_fixture(symbol, spin=spin)
    z, core, _ = CASES[symbol]
    cores, terms = resolve_ecp(basis, tuple(Atom.from_value(atom) for atom in atoms))
    assert cores == (core, 0)
    assert {term[0] for term in terms} == {0}
    assert mol.nelectron == z - core + 1 - spin == 2 - spin
    np.testing.assert_array_equal(mol.atom_charges(), [1, 1])
    assert mol.energy_nuc() == pytest.approx(
        1 / np.linalg.norm(mol.atom_coords()[0] - mol.atom_coords()[1]), abs=1e-13
    )
    oracle = (scf.UHF(mol) if spin else scf.RHF(mol)).run(conv_tol=1e-12)
    assert oracle.converged
    result = Calculator(
        method="uhf" if spin else "rhf", basis=basis, device=device
    ).singlepoint(atoms, charge=spin, multiplicity=spin + 1)
    assert result.converged
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    np.testing.assert_allclose(result.energy, oracle.e_tot, atol=2e-8, rtol=0)
    np.testing.assert_allclose(
        result.forces, -oracle.nuc_grad_method().kernel(), atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-7)


@pytest.mark.parametrize("symbol", CASES)
@pytest.mark.parametrize("spin", [0, 1])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stuttgart_budgeted_replay_complete_energy_difference(
    symbol: typing.Any, spin: typing.Any, device: typing.Any
) -> None:
    require_device(device)
    atoms, basis, mol = stuttgart_fixture(symbol, spin=spin)
    method = "uhf" if spin else "rhf"
    plan = Calculator(basis=basis, device=device, method=method).estimate_resources(
        [atoms], charges=[spin], multiplicities=[spin + 1]
    )
    plan.require_feasible()
    calculator = Calculator(
        basis=basis,
        device=device,
        method=method,
        resource_budget=ResourceBudget(
            host_bytes=plan.peak_bytes["host"],
            device_bytes=plan.peak_bytes.get("device"),
        ),
    )
    direction = np.array([[0.23, -0.31, 0.17], [-0.19, 0.11, 0.29]])
    with calculator.prepare_batch(
        [atoms], charges=[spin], multiplicities=[spin + 1]
    ) as batch:
        initial = batch.execute(strict=True).items[0]
        assert initial.converged
        for step in (2e-4, 7e-5):
            energies = []
            for sign in (-1, 1):
                xyz = mol.atom_coords() + sign * step * direction
                moved = batch.execute(coordinates=[xyz], strict=True).items[0]
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

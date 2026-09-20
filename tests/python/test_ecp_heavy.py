"""Bounded real-parameter qualification: LANL2DZ Rb/Cs/Au/Br/I with STO-3G hydrogen.

Parameters are loaded from the pinned test-only PySCF installation. This suite
does not establish coverage of other heavy elements or ECP parameter families.
"""

import json
import os
import typing
from dataclasses import replace

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
CASES = {
    "Rb": (37, 28, 4.4),
    "Cs": (55, 46, 4.8),
    "Au": (79, 60, 3.0),
    "Br": (35, 28, 2.85),
    "I": (53, 46, 3.25),
}
VALENCE_CHARGES = {"Rb": 9, "Cs": 9, "Au": 19, "Br": 7, "I": 7}
PARAMETER_SHA256 = {
    "Br": "befb5b5ae3a88d48ab3565c8c1afbb87bc27c4bd6a5fb271a0c24e600d0ec6a7",
    "I": "e270915093e0efcd4419608403ab23cc9bce5bfd9f0bb6aa3d972388c1025e5e",
    "Au": "618e1d76ee6ec1af1f846befb9ca9da96b54c8bbac109033cf5b34414b615642",
    "Rb": "9d8f07743d6859efb8fa6155fe1ac83e7e9de18bb453f51dcdb2846c202277aa",
    "Cs": "f5d99d7ab2ca5fae6d454127584aa3bc241de2ea8e3d4870d5889f8c23b0aa9e",
}


def molecular_charge(symbol: typing.Any, spin: typing.Any) -> typing.Any:
    # AuH+ has competing native SCF solutions and is not qualified here.
    return -spin if symbol == "Au" else spin


def scf_options(symbol: typing.Any) -> typing.Any:
    return (
        {"energy_tolerance": 1e-12, "density_tolerance": 1e-10, "max_iterations": 200}
        if symbol == "Au"
        else {}
    )


def reference_hf(symbol: typing.Any, mol: typing.Any) -> typing.Any:
    scf = pytest.importorskip("pyscf.scf")
    guesses = ("minao", "1e", "atom") if symbol == "Au" else ("minao",)
    solutions = []
    for guess in guesses:
        target = scf.UHF(mol) if mol.spin else scf.RHF(mol)
        if symbol == "Au":
            target.conv_tol_grad = 1e-9
            target.max_cycle = 200
        target.run(conv_tol=1e-12, init_guess=guess)
        assert target.converged
        solutions.append(target)
    energies = {guess: target.e_tot for guess, target in zip(guesses, solutions)}
    np.testing.assert_allclose(
        list(energies.values()), solutions[0].e_tot, atol=2e-9, rtol=0
    )
    return solutions[0], energies


def heavy_fixture(
    symbol: typing.Any, *, spin: typing.Any = 0, displacement: typing.Any = 0.0
) -> typing.Any:
    pyscf = pytest.importorskip("pyscf")
    gto = pytest.importorskip("pyscf.gto")
    z, core, bond_z = CASES[symbol]
    orbital = {
        symbol: gto.basis.load("lanl2dz", symbol),
        "H": gto.basis.load("sto-3g", "H"),
    }
    potentials = {symbol: gto.basis.load_ecp("lanl2dz", symbol)}
    assert potentials[symbol][0] == core
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
        charge=molecular_charge(symbol, spin),
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
        f"LANL2DZ-{symbol}/STO-3G-H qualification",
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
def test_heavy_components_refinement_and_all_center_derivatives(
    symbol: typing.Any, device: typing.Any
) -> None:
    require_device(device)
    for displacement in (0.0, 0.37):
        atoms, basis, mol = heavy_fixture(symbol, displacement=displacement)
        raw = ecp_integrals(atoms, basis, device=device)
        fine = ecp_integrals(
            atoms, basis, device=device, radial_points=224, polar_points=44
        )
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
def test_heavy_complete_hf_and_core_bookkeeping(
    symbol: typing.Any, spin: typing.Any, device: typing.Any
) -> None:
    require_device(device)
    atoms, basis, mol = heavy_fixture(symbol, spin=spin)
    z, core, _ = CASES[symbol]
    cores, terms = resolve_ecp(basis, tuple(Atom.from_value(atom) for atom in atoms))
    assert cores == (core, 0)
    assert {term[0] for term in terms} == {0}
    valence = VALENCE_CHARGES[symbol]
    assert mol.nelectron == z - core + 1 - mol.charge == valence + 1 - mol.charge
    np.testing.assert_array_equal(mol.atom_charges(), [valence, 1])
    assert mol.energy_nuc() == pytest.approx(
        valence / np.linalg.norm(mol.atom_coords()[0] - mol.atom_coords()[1]), abs=1e-13
    )
    oracle, _ = reference_hf(symbol, mol)
    result = Calculator(
        method="uhf" if spin else "rhf",
        basis=basis,
        device=device,
        **scf_options(symbol),
    ).singlepoint(atoms, charge=molecular_charge(symbol, spin), multiplicity=spin + 1)
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
def test_heavy_resource_boundary_replay_complete_energy_difference(
    symbol: typing.Any, spin: typing.Any, device: typing.Any
) -> None:
    require_device(device)
    atoms, basis, mol = heavy_fixture(symbol, spin=spin)
    method = "uhf" if spin else "rhf"
    plan = Calculator(
        basis=basis, device=device, method=method, **scf_options(symbol)
    ).estimate_resources([atoms], charges=[mol.charge], multiplicities=[spin + 1])
    if symbol == "Au" and device == "cuda":
        assert mol.nao_nr() == 23
        assert plan.status == "unsupported"
        with pytest.raises(NotImplementedError, match="<=16 public AOs"):
            plan.require_feasible()
        # Qualify numerical replay without inventing a larger CUDA inventory.
        budget = None
        with pytest.raises(NotImplementedError, match="<=16 public AOs"):
            Calculator(
                basis=basis,
                device=device,
                method=method,
                resource_budget=ResourceBudget(device_bytes=1 << 30),
            ).prepare_batch([atoms], charges=[mol.charge], multiplicities=[spin + 1])
    else:
        plan.require_feasible()
        budget = ResourceBudget(
            host_bytes=plan.peak_bytes["host"],
            device_bytes=plan.peak_bytes.get("device"),
        )
    calculator = Calculator(
        basis=basis,
        device=device,
        method=method,
        resource_budget=budget,
        **scf_options(symbol),
    )
    direction = np.array([[0.23, -0.31, 0.17], [-0.19, 0.11, 0.29]])
    with calculator.prepare_batch(
        [atoms], charges=[mol.charge], multiplicities=[spin + 1]
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
        if device == "cuda" and budget is not None:
            ledger = batch.resource_diagnostics["observation"]["device_ledger"]
            assert ledger["rejected_allocations"] == 0
            assert 0 < ledger["peak_bytes"] <= ledger["limit_bytes"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gold_real_f_channel_is_nonzero_and_independently_resolved(
    device: typing.Any,
) -> None:
    require_device(device)
    gto = pytest.importorskip("pyscf.gto")
    atoms, basis, mol = heavy_fixture("Au")
    gold = basis.by_element[79]
    records = json.loads(gold.ecp_data)
    assert {r["angular_momentum"][0] for r in records} == {0, 1, 2, 3, 4}
    assert max(shell.angular_momentum for shell in gold.shells) == 2
    raw = ecp_integrals(atoms, basis, device=device)
    for record in records:
        if record["angular_momentum"] == [3]:
            record["coefficients"] = [["0"] * len(record["r_exponents"])]
    without_f = replace(
        basis,
        elements=tuple(
            replace(element, ecp_data=json.dumps(records))
            if element.atomic_number == 79
            else element
            for element in basis.elements
        ),
    )
    reduced = ecp_integrals(atoms, without_f, device=device)
    selected = mol.copy()
    selected._ecpbas = mol._ecpbas[mol._ecpbas[:, gto.ANG_OF] == 3]
    norms = np.sqrt(mol.intor("int1e_ovlp").diagonal())
    expected = selected.intor("ECPscalar") / norms[:, None] / norms[None, :]
    assert np.max(np.abs(expected)) > 1e-6
    np.testing.assert_allclose(
        raw.nonlocal_ - reduced.nonlocal_, expected, atol=2e-9, rtol=0
    )
    np.testing.assert_array_equal(raw.local, reduced.local)
    xyz = mol.atom_coords()
    for step in (2e-4, 7e-5):
        for atom in range(2):
            for axis in range(3):
                delta = np.zeros_like(xyz)
                delta[atom, axis] = step
                plus = selected.copy().set_geom_(xyz + delta, unit="Bohr")
                minus = selected.copy().set_geom_(xyz - delta, unit="Bohr")
                fd = (plus.intor("ECPscalar") - minus.intor("ECPscalar")) / (2 * step)
                fd /= norms[:, None] * norms[None, :]
                np.testing.assert_allclose(
                    (raw.nonlocal_derivative - reduced.nonlocal_derivative)[atom, axis],
                    fd,
                    atol=3e-7,
                    rtol=2e-6,
                )

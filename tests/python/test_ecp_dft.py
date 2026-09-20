"""Matched-grid LDA/PBE ECP energies, independent of public force qualification."""

import os
import typing
from dataclasses import asdict

import numpy as np
import pytest
from test_ecp import fixture
from vibeqc import Atom, Calculator, ResourceBudget
from vibeqc_compiler.dft.grid import MolecularGrid

METHODS = ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks")


def require_device(device: typing.Any) -> None:
    if device == "cuda" and os.environ.get("VIBEQC_ECP_CUDA_TEST") != "1":
        pytest.skip("requires an allocated CUDA device")


def reference(
    mol: typing.Any, atoms: typing.Any, method: typing.Any, guess: typing.Any = "1e"
) -> typing.Any:
    """Independent Libcint/Libxc SCF on the exact declared molecular grid."""
    dft = pytest.importorskip("pyscf.dft")
    grid = MolecularGrid(
        tuple(Atom.from_value(a) for a in atoms),
        charge=mol.charge,
        multiplicity=mol.spin + 1,
    ).explicit()
    solver = dft.UKS(mol) if method.endswith("uks") else dft.RKS(mol)
    solver.xc = "PBE" if method.startswith("pbe") else "LDA_X,LDA_C_PW"
    solver.grids.coords = np.array(grid.points)
    solver.grids.weights = np.array(grid.weights)
    solver.small_rho_cutoff = 0
    solver.conv_tol, solver.conv_tol_grad = 1e-13, 1e-10
    solver.max_cycle = 150
    solver.init_guess = guess
    solver.kernel()
    assert solver.converged
    density = solver.make_rdm1()
    total = density.sum(axis=0) if mol.spin else density
    fock, overlap = solver.get_fock(dm=density), solver.get_ovlp()
    residual = fock @ density @ overlap - overlap @ density @ fock
    assert np.max(np.sqrt(np.mean(residual**2, axis=(-2, -1)))) < 1e-9
    components = {
        "nuclear": mol.energy_nuc(),
        "one_electron": float(np.einsum("ij,ji", total, solver.get_hcore())),
        "hartree": float(0.5 * np.einsum("ij,ji", total, solver.get_j(dm=total))),
        "xc": float(solver.get_veff(dm=density).exc),
    }
    # A missing ECP residual cannot accidentally pass the total energy gate.
    ecp_expectation = float(np.einsum("ij,ji", total, mol.intor("ECPscalar")))
    assert abs(ecp_expectation) > 1e-3
    return solver.e_tot, components, ecp_expectation, grid.identity


def calculator(
    basis: typing.Any, method: typing.Any, device: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    return Calculator(
        basis=basis,
        method=method,
        device=device,
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


def endpoint(
    method: typing.Any, device: typing.Any, representation: typing.Any
) -> typing.Any:
    require_device(device)
    spin = int(method.endswith("uks"))
    atoms, basis, mol = fixture(spin=spin, representation=representation)
    expected, components, residual_energy, grid_identity = reference(mol, atoms, method)
    second, _, _, _ = reference(mol, atoms, method, "minao")
    assert abs(second - expected) < 1e-8
    calc = calculator(basis, method, device)
    result = calc.singlepoint(
        atoms, charge=spin, multiplicity=spin + 1, properties=("energy",)
    )
    assert result.converged and result.forces is None
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert result.physical_residual_rms < 1e-9
    assert abs(result.energy - expected) < 1e-8
    diagnostic = result.ks_diagnostic
    assert diagnostic.occupations == ((1, 0) if spin else (1, 1))
    np.testing.assert_allclose(diagnostic.electrons, diagnostic.occupations, atol=1e-9)
    assert diagnostic.ao_order == int(method.startswith("pbe"))
    actual = asdict(diagnostic.components)
    errors = {key: abs(actual[key] - components[key]) for key in components}
    assert max(errors.values()) < 2e-8
    assert abs(sum(actual.values()) - result.energy) < 1e-12
    assert (
        abs(
            actual["nuclear"]
            - 1 / np.linalg.norm(mol.atom_coords()[0] - mol.atom_coords()[1])
        )
        < 1e-12
    )
    return {
        "method": method,
        "device": device,
        "representation": representation,
        "parameter_sha256": basis.provenance.checksum,
        "grid_identity": grid_identity,
        "nao": mol.nao_nr(),
        "nelectron": mol.nelectron,
        "energy": result.energy,
        "reference_energy": expected,
        "energy_error": abs(result.energy - expected),
        "physical_residual_rms": result.physical_residual_rms,
        "components": actual,
        "component_errors": errors,
        "ecp_expectation": residual_energy,
    }


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_ecp_dft_independent_energy_components(
    method: typing.Any, device: typing.Any, representation: typing.Any
) -> None:
    endpoint(method, device, representation)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_ecp_dft_budgeted_ragged_replay_and_isolation(
    method: typing.Any, device: typing.Any
) -> None:
    require_device(device)
    spin = int(method.endswith("uks"))
    atoms, basis, mol = fixture(spin=spin)
    # Mix an ECP molecule and an all-electron fragment in the same owner.
    fragment = [("H", (0, 0, 0))] if spin else [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    systems, charges, multiplicities = [atoms, fragment], [spin, 0], [spin + 1] * 2
    calc = calculator(basis, method, device)
    plan = calc.estimate_resources(
        systems, charges=charges, multiplicities=multiplicities
    ).require_feasible()
    bounded = calculator(
        basis,
        method,
        device,
        resource_budget=ResourceBudget(
            host_bytes=plan.peak_bytes["host"],
            device_bytes=plan.peak_bytes.get("device"),
        ),
    )
    with bounded.prepare_batch(
        systems, charges=charges, multiplicities=multiplicities
    ) as batch:
        cold = batch.execute(strict=True, properties=("energy",))
        warm = batch.execute(strict=True, properties=("energy",))
        assert all(item.warm_start_used for item in warm.items)
        np.testing.assert_allclose(warm.energies, cold.energies, atol=2e-9, rtol=0)
        xyz = mol.atom_coords()
        xyz[1] += [0.03, -0.02, 0.19]
        moved = batch.execute(
            coordinates=[xyz, None], strict=True, properties=("energy",)
        )
        moved_mol = mol.copy().set_geom_(xyz, unit="Bohr")
        moved_atoms = [
            (symbol, tuple(position)) for (symbol, _), position in zip(atoms, xyz)
        ]
        target, _, _, _ = reference(moved_mol, moved_atoms, method)
        assert abs(moved.items[0].energy - target) < 1e-8
        assert abs(moved.items[1].energy - cold.items[1].energy) < 2e-9
        bad = batch.execute(
            coordinates=[[0.0], None], strict=False, properties=("energy",)
        )
        assert not bad.items[0].succeeded and bad.items[1].succeeded
        restored = batch.execute(
            coordinates=[mol.atom_coords(), None], strict=True, properties=("energy",)
        )
        np.testing.assert_allclose(restored.energies, cold.energies, atol=2e-9, rtol=0)
        if device == "cuda":
            ledger = batch.resource_diagnostics["observation"]["device_ledger"]
            assert ledger["rejected_allocations"] == 0
            assert 0 < ledger["peak_bytes"] <= ledger["limit_bytes"]

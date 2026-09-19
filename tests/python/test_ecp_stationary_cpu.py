"""CPU ECP stationary gradients: independent full-grid-response and SCF oracles."""

from dataclasses import replace

import numpy as np
import pytest
from test_ecp import fixture
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import StationaryDerivativeContract, StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
from vibeqc_compiler.dft import NativeAO

GRID = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)


def test_ecp_cpu_ao_spatial_jets_do_not_promote_ecp_higher_derivatives():
    from vibeqc.basis_capabilities import basis_capability

    atoms, record, mol = fixture(representation="cartesian")
    points = np.array([[0.21, -0.15, 0.8], [0.43, 0.36, 2.1]])
    with NativeAO(atoms, basis=record) as basis:
        np.testing.assert_allclose(
            basis.evaluate(points, 3),
            mol.eval_gto("GTOval_cart_deriv3", points),
            atol=2e-12,
            rtol=2e-12,
        )
    assert basis_capability(record, atoms, operator="ao", derivative_order=3)[
        "eligible"
    ]
    assert not basis_capability(
        record, atoms, operator="nuclear_attraction", derivative_order=2
    )["eligible"]
    assert not basis_capability(
        record, atoms, backend="cuda", operator="ao", derivative_order=2
    )["eligible"]


def reference(mol, state, method):
    from pyscf import dft, lib

    lib.num_threads(1)
    solver = dft.UKS(mol) if method.endswith("uks") else dft.RKS(mol)
    solver.xc = "PBE" if method.startswith("pbe") else "LDA_X,LDA_C_PW"
    solver.grids.coords = np.array(state.grid.points)
    solver.grids.weights = np.array(state.grid.weights)
    solver.grids.radii_adjust = None
    owners = np.asarray(state.grid.owners)
    atomic = {
        mol.atom_symbol(a): (
            np.array(state.grid.points[owners == a] - mol.atom_coord(a)),
            np.array(state._source.atomic_weights[owners == a]),
        )
        for a in range(mol.natm)
    }
    solver.grids.gen_atomic_grids = lambda *args, **kwargs: atomic
    solver.small_rho_cutoff = 0
    solver.conv_tol, solver.conv_tol_grad = 1e-13, 1e-10
    solver.max_cycle = 150
    solver.init_guess = "1e"
    solver.kernel()
    assert solver.converged
    gradient = solver.nuc_grad_method()
    gradient.grid_response = True
    return solver.e_tot, gradient.kernel()


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_ecp_complete_cpu_gradient_analytic_fd_and_live_owner(method, record_property):
    spin = int(method.endswith("uks"))
    atoms, record, mol = fixture(spin=spin, representation="cartesian")
    calc = Calculator(
        basis=record,
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with (
        calc.prepare_batch([atoms], charges=[spin], multiplicities=[spin + 1]) as batch,
        NativeAO(
            atoms,
            basis=record,
            charge=spin,
            multiplicity=spin + 1,
        ) as basis,
    ):
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        assert state._source.metadata[0] == 4
        assert state._source.ecp_cores == (10, 0)
        assert state._source.hamiltonian == "scalar-semilocal-ecp"
        assert state.occupations.sum() == 2 - spin
        result = complete_rks_gradient_diagnostic(
            state,
            basis,
            cache=".cache/ecp-stationary-tests",
            execution="native",
        )
        expected_energy, expected = reference(mol, state, method)
        assert abs(energy - expected_energy) < 2e-8
        np.testing.assert_allclose(result.gradient, expected, atol=1e-7, rtol=0)
        if method == "pbe-uks":
            interpreted = complete_rks_gradient_diagnostic(
                state, basis, cache=".cache/ecp-stationary-tests", execution="reference"
            )
            np.testing.assert_allclose(
                interpreted.gradient, expected, atol=1e-7, rtol=0
            )
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=1e-9, rtol=0)
        ecp = result.components["ecp_local"] + result.components["ecp_nonlocal"]
        assert np.max(np.abs(ecp)) > 1e-3
        assert np.max(np.abs(result.gradient - ecp - expected)) > 1e-3
        # Effective ionic charges are +1/+1, not bare Na(+11)/H(+1).
        delta = mol.atom_coord(0) - mol.atom_coord(1)
        np.testing.assert_allclose(
            result.components["nuclear"][0],
            -delta / np.linalg.norm(delta) ** 3,
            atol=1e-13,
        )
        record_property(
            "analytic_max_error", float(np.max(np.abs(result.gradient - expected)))
        )
        direction = np.array([[0.2, -0.13, 0.07], [-0.11, 0.08, 0.19]])
        projection = float(np.sum(result.gradient * direction))
        estimates = []
        for step in (1e-3, 3e-4, 1e-4):
            energies = []
            for sign in (1, -1):
                moved = [
                    (a, np.asarray(r) + sign * step * d)
                    for (a, r), d in zip(atoms, direction)
                ]
                energies.append(
                    calc.singlepoint(
                        moved,
                        charge=spin,
                        multiplicity=spin + 1,
                        properties=("energy",),
                    ).energy
                )
            estimates.append((energies[0] - energies[1]) / (2 * step))
        np.testing.assert_allclose(estimates[-2:], projection, atol=2e-7, rtol=0)
        record_property("fd_errors", [abs(x - projection) for x in estimates])
        with pytest.raises(ValueError, match="identity"):
            forged = replace(
                state, identity=replace(state.identity, model_identity="all-electron")
            )
            StationaryDerivativeContract(forged.identity).validate(forged)
        # A fresh ECP owner must never authorize the old derivative proof.
        batch.execute(strict=True)
        with pytest.raises(ValueError, match="stale"):
            state._source.ecp_derivatives()
        current = StationaryKsState.from_native(batch, basis)
        assert current.identity.solve_epoch > state.identity.solve_epoch
        assert current._source.ecp_terms == state._source.ecp_terms
        with pytest.raises(ValueError, match="forces"):
            calc.singlepoint(
                atoms,
                charge=spin,
                multiplicity=spin + 1,
                properties=("energy", "forces"),
            )


def test_same_core_count_different_ecp_is_bound_to_actual_energy_owner():
    import json

    atoms, record, _ = fixture(representation="cartesian")
    changed = []
    for element in record.elements:
        if element.ecp_core_electrons:
            potentials = json.loads(element.ecp_data)
            potentials[0]["coefficients"][0][0] = str(
                float(potentials[0]["coefficients"][0][0]) * 1.01
            )
            element = replace(element, ecp_data=json.dumps(potentials))
        changed.append(element)
    other_record = replace(record, elements=tuple(changed))
    options = {
        "method": "lda-rks",
        "ks_options": KsOptions(grid=GRID),
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    with (
        Calculator(basis=record, **options).prepare_batch([atoms]) as batch,
        Calculator(basis=other_record, **options).prepare_batch([atoms]) as other,
        NativeAO(atoms, basis=record) as basis,
    ):
        batch.execute(strict=True)
        other.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        changed_state = StationaryKsState.from_native(other, basis)
        assert state.identity.basis_identity == changed_state.identity.basis_identity
        assert state._source.ecp_cores == changed_state._source.ecp_cores
        assert state._source.ecp_terms != changed_state._source.ecp_terms
        assert (
            np.max(
                np.abs(
                    state._source.ecp_derivatives()
                    - changed_state._source.ecp_derivatives()
                )
            )
            > 1e-6
        )
        with pytest.raises(ValueError, match="identity"):
            StationaryDerivativeContract(state.identity).validate(
                replace(state, _source=changed_state._source)
            )
    with pytest.raises(RuntimeError, match="closed"):
        state._source.ecp_derivatives()

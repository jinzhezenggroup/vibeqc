"""Complete diagnostic gradients: independent analytic and reconverged oracles.

PySCF is used only here. Its own libcint, libxc, SCF and analytic Becke response
evaluate the same input basis and atomic quadrature, without calling any of the
generated derivative graphs under test. Public DFT force capabilities stay off.
"""

import ctypes as ct
import typing
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import (
    BasisProvenance,
    BasisSet,
    BasisShell,
    Calculator,
    ElementBasis,
    GridPolicy,
    GridSpec,
    KsOptions,
    method_capabilities,
)
from vibeqc._dft_gradient import StationaryDerivativeContract, StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
from vibeqc_compiler.dft import NativeAO

ATOMS = [("O", (0.1, -0.1, 0.0)), ("H", (0.1, 0.2, 1.7)), ("H", (1.6, -0.2, -0.5))]
GRID = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
TRANSITION_METAL_GRID_BASIS_PAYLOAD = {
    1: ((0, ("1.2",), (("1",),)),),
    26: (
        (0, ("2.0",), (("1",),)),
        (1, ("1.0",), (("1",),)),
    ),
}


def transition_metal_grid_basis() -> BasisSet:
    """Synthetic Fe/H s-p fixture kept inside the qualified CPU gradient domain."""
    from vibeqc.profiles import canonical_hash

    elements = tuple(
        ElementBasis(
            atomic_number,
            tuple(BasisShell(*shell) for shell in shells),
        )
        for atomic_number, shells in sorted(TRANSITION_METAL_GRID_BASIS_PAYLOAD.items())
    )
    return BasisSet(
        "issue-596-fe-h-sp-grid-qualification",
        elements,
        BasisProvenance(
            "inline issue-596 transition-metal grid qualification fixture",
            "1",
            "CC0-1.0",
            canonical_hash(TRANSITION_METAL_GRID_BASIS_PAYLOAD),
        ),
    )


GRID_CONVERGENCE_GATES = {
    "standard": {
        "energy_hartree": 2e-6,
        "gradient_hartree_per_bohr": 7e-5,
        "maximum_dense_point_fraction": 0.35,
    },
    "tight": {
        "energy_hartree": 1e-6,
        "gradient_hartree_per_bohr": 2e-5,
        "maximum_dense_point_fraction": 0.75,
    },
}


def assert_production_grid_convergence(
    accuracy: str,
    production_points: int,
    reference_points: int,
    energy_error: float,
    gradient_error: float,
    record_property: typing.Any,
) -> None:
    """Bind promoted profiles to measured accuracy and point-cost envelopes."""
    gate = GRID_CONVERGENCE_GATES[accuracy]
    point_fraction = production_points / reference_points
    # pytest-xdist serializes user properties through execnet, which does not
    # accept NumPy scalar subclasses. Keep retained evidence transport-neutral.
    record_property("grid_accuracy", accuracy)
    record_property("production_points", int(production_points))
    record_property("independent_reference_points", int(reference_points))
    record_property("production_dense_point_fraction", float(point_fraction))
    record_property("energy_error_hartree", float(energy_error))
    record_property("gradient_error_hartree_per_bohr", float(gradient_error))
    record_property("energy_gate_hartree", float(gate["energy_hartree"]))
    record_property(
        "gradient_gate_hartree_per_bohr", float(gate["gradient_hartree_per_bohr"])
    )
    assert point_fraction < gate["maximum_dense_point_fraction"]
    assert energy_error < gate["energy_hartree"]
    assert gradient_error < gate["gradient_hartree_per_bohr"]


def calculator(method: typing.Any, **kwargs: typing.Any) -> typing.Any:
    return Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


def production_calculator(
    method: typing.Any, *, grid_accuracy: str = "standard", **kwargs: typing.Any
) -> typing.Any:
    return Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid_accuracy=grid_accuracy),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


def independent_gradient(
    basis: typing.Any, state: typing.Any, method: typing.Any
) -> typing.Any:
    """PySCF full-response RKS with native atomic quadrature, independent algebra."""
    from pyscf import dft, gto, lib
    from pyscf.data.elements import ELEMENTS
    from pyscf.grad import rks

    lib.num_threads(1)
    labels = [f"{ELEMENTS[a.atomic_number]}{i}" for i, a in enumerate(basis.atoms)]
    shells = {label: [] for label in labels}
    for shell in basis.shells:
        shells[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=[(label, a.position) for label, a in zip(labels, basis.atoms)],
        basis=shells,
        unit="Bohr",
        cart=True,
        charge=basis.charge,
        spin=basis.multiplicity - 1,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "PBE" if method == "pbe-rks" else "LDA_X,LDA_C_PW"
    mf.grids.coords = np.array(state.grid.points)
    mf.grids.weights = np.array(state.grid.weights)
    mf.grids.radii_adjust = None
    owners = np.asarray(state.grid.owners)
    tab = {
        mol.atom_symbol(a): (
            np.array(state.grid.points[owners == a] - mol.atom_coord(a)),
            np.array(state._source.atomic_weights[owners == a]),
        )
        for a in range(mol.natm)
    }
    mf.grids.gen_atomic_grids = lambda *args, **kwargs: tab
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 150
    mf.kernel()
    assert mf.converged
    grad = mf.nuc_grad_method()
    grad.grid_response = True
    total = grad.kernel()
    d, w = mf.make_rdm1(), grad.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    h, s, j = grad.hcore_generator(mol), grad.get_ovlp(mol), grad.get_j(mol, d)
    grid_and_weight, vxc = rks.get_vxc_full_response(
        mf._numint, mol, mf.grids, mf.xc, d
    )
    components = {
        name: np.zeros_like(total)
        for name in ("one_electron", "coulomb", "overlap_pulay", "xc_ao", "xc_weight")
    }
    for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        components["one_electron"][a] = np.einsum("xij,ij->x", h(a), d)
        components["coulomb"][a] = 2 * np.einsum("xij,ij->x", j[:, p0:p1], d[p0:p1])
        components["overlap_pulay"][a] = -2 * np.einsum(
            "xij,ij->x", s[:, p0:p1], w[p0:p1]
        )
        components["xc_ao"][a] = 2 * np.einsum("xij,ij->x", vxc[:, p0:p1], d[p0:p1])
    for points, _, dw in rks.grids_response_cc(mf.grids):
        ao = mf._numint.eval_ao(mol, points, deriv=1)
        kind = "GGA" if method == "pbe-rks" else "LDA"
        rho = mf._numint.eval_rho(mol, ao if kind == "GGA" else ao[0], d, xctype=kind)
        exc = mf._numint.eval_xc(mf.xc, rho, deriv=0)[0]
        density = rho[0] if kind == "GGA" else rho
        components["xc_weight"] += np.einsum("p,p,axp->ax", exc, density, dw)
    components["xc_grid"] = grid_and_weight - components["xc_weight"]
    components["nuclear"] = grad.grad_nuc()
    np.testing.assert_allclose(sum(components.values()), total, atol=2e-12, rtol=0)
    return mf.e_tot, total, components


def independent_converged_grid_reference(
    basis: typing.Any, method: typing.Any
) -> tuple[float, np.ndarray, int]:
    """Use PySCF's own dense unpruned quadrature as an independent grid oracle."""
    from pyscf import dft, gto, lib
    from pyscf.data.elements import ELEMENTS

    lib.num_threads(1)
    labels = [f"{ELEMENTS[a.atomic_number]}{i}" for i, a in enumerate(basis.atoms)]
    shells = {label: [] for label in labels}
    for shell in basis.shells:
        shells[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=[(label, a.position) for label, a in zip(labels, basis.atoms)],
        basis=shells,
        unit="Bohr",
        cart=True,
        charge=basis.charge,
        spin=basis.multiplicity - 1,
        verbose=0,
    )
    mf = dft.UKS(mol) if method.endswith("-uks") else dft.RKS(mol)
    mf.xc = "PBE" if method.startswith("pbe-") else "LDA_X,LDA_C_PW"
    # This deliberately does not consume VibeQC GridSpec/points/weights.  The
    # independent oracle uses PySCF's Treutler radial mapping, Lebedev angular
    # rule and Becke partition at a substantially denser unpruned resolution.
    mf.grids.atom_grid = (120, 974)
    mf.grids.prune = None
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    grad = mf.nuc_grad_method()
    grad.grid_response = True
    gradient = grad.kernel()
    assert mf.grids.coords is not None
    return mf.e_tot, gradient, len(mf.grids.coords)


@pytest.mark.parametrize("accuracy", ["standard", "tight"])
@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks"])
def test_production_grid_converges_against_independent_dense_quadrature(
    method: typing.Any, accuracy: typing.Any, record_property: typing.Any
) -> None:
    """Promoted production profiles stay accurate against an independent dense grid."""
    pytest.importorskip(
        "pyscf", reason="independent dense-grid reference requires PySCF"
    )
    calc = production_calculator(method, grid_accuracy=accuracy, max_iterations=200)
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        expected = GridPolicy(accuracy).resolve(method)
        assert state._source.grid_spec == expected
        result = complete_rks_gradient_diagnostic(
            state,
            basis,
            cache=f".cache/production-grid-convergence-{method}-{accuracy}",
            execution="native",
            tile_points=137,
            integral_terms=17,
            primitive_tile=29,
        )
        reference_energy, reference_gradient, reference_points = (
            independent_converged_grid_reference(basis, method)
        )
        production_points = len(state.grid.points)
        energy_error = abs(energy - reference_energy)
        gradient_error = float(np.max(np.abs(result.gradient - reference_gradient)))
        assert_production_grid_convergence(
            accuracy,
            production_points,
            reference_points,
            energy_error,
            gradient_error,
            record_property,
        )


@pytest.mark.parametrize("accuracy", ["standard", "tight"])
@pytest.mark.parametrize("method", ["lda-uks", "pbe-uks"])
def test_production_grid_open_shell_converges_against_independent_dense_quadrature(
    method: typing.Any, accuracy: typing.Any, record_property: typing.Any
) -> None:
    """Production v2 unrestricted profiles converge against an independent dense grid."""
    pytest.importorskip(
        "pyscf", reason="independent dense-grid reference requires PySCF"
    )
    charge, multiplicity = 1, 2
    calc = production_calculator(method, grid_accuracy=accuracy, max_iterations=200)
    with (
        calc.prepare_batch(
            [ATOMS], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis,
    ):
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        expected = GridPolicy(accuracy).resolve(method)
        assert state._source.grid_spec == expected
        assert state.density.shape[0] == 2
        result = complete_rks_gradient_diagnostic(
            state,
            basis,
            cache=f".cache/production-grid-convergence-{method}-{accuracy}",
            execution="native",
            tile_points=137,
            integral_terms=17,
            primitive_tile=29,
        )
        reference_energy, reference_gradient, reference_points = (
            independent_converged_grid_reference(basis, method)
        )
        production_points = len(state.grid.points)
        energy_error = abs(energy - reference_energy)
        gradient_error = float(np.max(np.abs(result.gradient - reference_gradient)))
        assert_production_grid_convergence(
            accuracy,
            production_points,
            reference_points,
            energy_error,
            gradient_error,
            record_property,
        )


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks"])
def test_production_grid_light_element_energy_and_force(method: typing.Any) -> None:
    """Production v2 survives the independent full-response water oracle."""
    pytest.importorskip("pyscf", reason="independent analytic reference requires PySCF")
    calc = production_calculator(method, max_iterations=200)
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        assert state._source.grid_spec.version == 2
        assert state._source.grid_provenance["policy_version"] == 2
        result = complete_rks_gradient_diagnostic(
            state,
            basis,
            cache=".cache/production-grid-cpu",
            execution="native",
            tile_points=137,
            integral_terms=17,
            primitive_tile=29,
        )
        reference_energy, reference, _ = independent_gradient(basis, state, method)
        assert energy == pytest.approx(reference_energy, abs=2e-9)
        np.testing.assert_allclose(result.gradient, reference, atol=1e-7, rtol=0)
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=3e-10, rtol=0)


def test_production_grid_transition_metal_energy_and_force() -> None:
    """Fe/H v2 grid resolves sourced Z=26 radii with an independent force oracle."""
    pytest.importorskip("pyscf", reason="independent analytic reference requires PySCF")

    atoms = [("Fe", (0.05, -0.02, 0.03)), ("H", (0.17, 0.11, 2.25))]
    charge, multiplicity = 25, 1
    basis_definition = transition_metal_grid_basis()
    calc = production_calculator("lda-rks", basis=basis_definition, max_iterations=250)
    with (
        calc.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(
            atoms,
            basis=basis_definition,
            charge=charge,
            multiplicity=multiplicity,
        ) as basis,
    ):
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        assert state._source.grid_spec.version == 2
        assert dict(state._source.grid_spec.element_radii)[26] > 0
        assert max(shell.angular_momentum for shell in basis.shells) == 1
        assert basis_definition.provenance.source.startswith("inline issue-596")
        result = complete_rks_gradient_diagnostic(
            state,
            basis,
            cache=".cache/production-grid-fe",
            execution="native",
            tile_points=137,
            integral_terms=17,
            primitive_tile=29,
        )
        reference_energy, reference, _ = independent_gradient(basis, state, "lda-rks")
        assert energy == pytest.approx(reference_energy, abs=3e-9)
        np.testing.assert_allclose(result.gradient, reference, atol=2e-7, rtol=0)
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=5e-10, rtol=0)


def independent_uks_gradient(
    basis: typing.Any, state: typing.Any, method: typing.Any
) -> typing.Any:
    """Independent PySCF full-grid-response UKS total gradient."""
    from pyscf import dft, gto, lib
    from pyscf.data.elements import ELEMENTS

    lib.num_threads(1)
    labels = [f"{ELEMENTS[a.atomic_number]}{i}" for i, a in enumerate(basis.atoms)]
    shells = {label: [] for label in labels}
    for shell in basis.shells:
        shells[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=[(label, a.position) for label, a in zip(labels, basis.atoms)],
        basis=shells,
        unit="Bohr",
        cart=True,
        charge=basis.charge,
        spin=basis.multiplicity - 1,
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.xc = "PBE" if method == "pbe-uks" else "LDA_X,LDA_C_PW"
    mf.grids.coords = np.array(state.grid.points)
    mf.grids.weights = np.array(state.grid.weights)
    mf.grids.radii_adjust = None
    owners = np.asarray(state.grid.owners)
    tab = {
        mol.atom_symbol(a): (
            np.array(state.grid.points[owners == a] - mol.atom_coord(a)),
            np.array(state._source.atomic_weights[owners == a]),
        )
        for a in range(mol.natm)
    }
    mf.grids.gen_atomic_grids = lambda *args, **kwargs: tab
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 150
    mf.kernel()
    assert mf.converged
    grad = mf.nuc_grad_method()
    grad.grid_response = True
    return mf.e_tot, grad.kernel()


def independent_semilocal_total_gradient(
    basis: typing.Any, state: typing.Any, method: typing.Any
) -> typing.Any:
    """Independent PySCF analytic total gradient on the identical explicit grid."""
    from pyscf import dft, gto, lib
    from pyscf.data.elements import ELEMENTS

    lib.num_threads(1)
    labels = [f"{ELEMENTS[a.atomic_number]}{i}" for i, a in enumerate(basis.atoms)]
    shells = {label: [] for label in labels}
    for shell in basis.shells:
        shells[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=[(label, a.position) for label, a in zip(labels, basis.atoms)],
        basis=shells,
        unit="Bohr",
        cart=True,
        charge=basis.charge,
        spin=basis.multiplicity - 1,
        verbose=0,
    )
    mf = dft.RKS(mol) if method.endswith("-rks") else dft.UKS(mol)
    mf.xc = "R2SCAN"
    mf.grids.coords = np.array(state.grid.points)
    mf.grids.weights = np.array(state.grid.weights)
    mf.grids.radii_adjust = None
    owners = np.asarray(state.grid.owners)
    tab = {
        mol.atom_symbol(a): (
            np.array(state.grid.points[owners == a] - mol.atom_coord(a)),
            np.array(state._source.atomic_weights[owners == a]),
        )
        for a in range(mol.natm)
    }
    mf.grids.gen_atomic_grids = lambda *args, **kwargs: tab
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    grad = mf.nuc_grad_method()
    grad.grid_response = True
    return mf.e_tot, grad.kernel()


@pytest.mark.parametrize(
    "method,charge,multiplicity",
    [("r2scan-rks", 0, 1), ("r2scan-uks", 1, 2)],
)
def test_complete_r2scan_cpu_gradient_reconverged_directional_fd(
    method: typing.Any,
    charge: typing.Any,
    multiplicity: typing.Any,
    record_property: typing.Any,
) -> None:
    """Hard gate: complete tau force agrees with fully reconverged energy differences."""
    calc = calculator(method, max_iterations=200)
    with (
        calc.prepare_batch(
            [ATOMS], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis,
    ):
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        assert state.identity.method == method
        result = complete_rks_gradient_diagnostic(
            state,
            basis,
            cache=".cache/r2scan-gradient-tests",
            execution="native",
            tile_points=137,
            integral_terms=17,
            primitive_tile=29,
        )
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=2e-9, rtol=0)

        xyz = np.asarray([position for _, position in ATOMS])
        direction = np.array(
            [[0.13, -0.07, 0.11], [-0.05, 0.17, 0.03], [0.09, 0.02, -0.14]]
        )
        estimates = []
        for step in (3e-4, 1e-4):
            energies = []
            for sign in (1, -1):
                moved = [
                    (atom[0], position)
                    for atom, position in zip(
                        ATOMS, xyz + sign * step * direction, strict=True
                    )
                ]
                energies.append(
                    calc.singlepoint(
                        moved,
                        charge=charge,
                        multiplicity=multiplicity,
                        properties=("energy",),
                    ).energy
                )
            estimates.append((energies[0] - energies[1]) / (2 * step))
        actual = float(np.sum(result.gradient * direction))
        assert abs(estimates[-1] - estimates[-2]) < 2e-6
        assert abs(estimates[-1] - actual) < 2e-6
        record_property("r2scan_fd_error", abs(estimates[-1] - actual))


@pytest.mark.parametrize(
    "method,charge,multiplicity",
    [("r2scan-rks", 0, 1), ("r2scan-uks", 1, 2)],
)
def test_complete_r2scan_cpu_gradient_independent_analytic(
    method: typing.Any,
    charge: typing.Any,
    multiplicity: typing.Any,
    record_property: typing.Any,
) -> None:
    """Independent PySCF/libxc analytic gate when that validation stack is installed."""
    pytest.importorskip("pyscf", reason="independent analytic reference requires PySCF")
    calc = calculator(method, max_iterations=200)
    with (
        calc.prepare_batch(
            [ATOMS], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis,
    ):
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        result = complete_rks_gradient_diagnostic(
            state,
            basis,
            cache=".cache/r2scan-gradient-tests",
            execution="native",
            tile_points=137,
            integral_terms=17,
            primitive_tile=29,
        )
        reference_energy, reference = independent_semilocal_total_gradient(
            basis, state, method
        )
        assert energy == pytest.approx(reference_energy, abs=2e-8)
        np.testing.assert_allclose(result.gradient, reference, atol=2e-6, rtol=0)
        record_property(
            "r2scan_analytic_max_error",
            float(np.max(np.abs(result.gradient - reference))),
        )


@pytest.mark.parametrize("execution", ["reference", "native"])
@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks"])
def test_complete_asymmetric_water_analytic_and_reconverged_fd(
    method: typing.Any, record_property: typing.Any, execution: typing.Any
) -> None:
    pytest.importorskip("pyscf", reason="independent analytic reference requires PySCF")
    calc = calculator(method)
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        result = complete_rks_gradient_diagnostic(
            state,
            basis,
            cache=".cache/b22-tests",
            execution=execution,
            max_ecp_pair_samples=1,
        )
        assert result.work["ecp_quadrature_pair_samples"] == 0
        assert result.work["primitive_records"] == result.work["primitive_record_bound"]
        reference_energy, reference, components = independent_gradient(
            basis, state, method
        )
        assert energy == pytest.approx(reference_energy, abs=2e-9)
        for name, value in result.components.items():
            np.testing.assert_allclose(
                value, components[name], atol=1e-7, rtol=0, err_msg=name
            )
        np.testing.assert_allclose(result.gradient, reference, atol=1e-7, rtol=0)
        record_property(
            "analytic_max_error", float(np.max(np.abs(result.gradient - reference)))
        )
        record_property(
            "source_max_error",
            float(
                max(
                    np.max(np.abs(value - components[name]))
                    for name, value in result.components.items()
                )
            ),
        )
        assert result.work["ordered_quartets"] == basis.nao**4
        assert result.work["grid_directional_points"] == (
            9 * len(state.grid.points) if execution == "reference" else 0
        )
        # All seven signed sources are nontrivial here. An omitted/reversed
        # grid, Pulay or nuclear term cannot pass by molecular symmetry.
        for name in ("xc_grid", "xc_weight", "overlap_pulay", "nuclear"):
            assert np.max(np.abs(result.components[name])) > 1e-4
            assert (
                np.max(
                    np.abs(result.gradient - 2 * result.components[name] - reference)
                )
                > 1e-4
            )
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=2e-10, rtol=0)

        coordinates = np.array([a[1] for a in ATOMS])
        estimates = []
        for step in (1e-3, 3e-4, 1e-4):
            fd = np.empty_like(coordinates)
            for a, axis in product_coordinates():
                direction = np.zeros_like(coordinates)
                direction[a, axis] = step
                energies = [
                    calc.singlepoint(
                        [
                            (label, pos)
                            for (label, _), pos in zip(
                                ATOMS, coordinates + sign * direction
                            )
                        ]
                    ).energy
                    for sign in (1, -1)
                ]
                fd[a, axis] = (energies[0] - energies[1]) / (2 * step)
            estimates.append(fd)
        # A multistep stable region and the full componentwise force gate;
        # every displaced density, grid, Fock and SCF solve is recomputed.
        np.testing.assert_allclose(estimates[-1], estimates[-2], atol=1e-6, rtol=0)
        np.testing.assert_allclose(result.gradient, estimates[-1], atol=1e-6, rtol=0)
        richardson = (9 * estimates[-1] - estimates[-2]) / 8
        np.testing.assert_allclose(result.gradient, richardson, atol=1e-7, rtol=0)
        record_property(
            "finite_difference_max_error",
            float(np.max(np.abs(result.gradient - estimates[-1]))),
        )
        record_property(
            "richardson_max_error", float(np.max(np.abs(result.gradient - richardson)))
        )

        shift, order = np.array([0.25, -0.37, 0.18]), [2, 0, 1]
        moved = [(ATOMS[i][0], coordinates[i] + shift) for i in order]
        with calc.prepare_batch([moved]) as other, NativeAO(moved) as moved_basis:
            other.execute(strict=True)
            moved_state = StationaryKsState.from_native(other, moved_basis)
            moved_result = complete_rks_gradient_diagnostic(
                moved_state,
                moved_basis,
                cache=".cache/b22-tests",
                execution=execution,
                tile_points=137,
                integral_terms=17,
                primitive_tile=29,
            )
            np.testing.assert_allclose(
                moved_result.gradient, result.gradient[order], atol=1e-8, rtol=0
            )
        # Warm replay must generate a new lease and reproduce the same gradient.
        replay = batch.execute(strict=True).items[0]
        assert replay.warm_start_used
        with pytest.raises(ValueError, match="stale"):
            complete_rks_gradient_diagnostic(
                state, basis, cache=".cache/b22-tests", execution=execution
            )
        current = StationaryKsState.from_native(batch, basis)
        warm = complete_rks_gradient_diagnostic(
            current, basis, cache=".cache/b22-tests", execution=execution
        )
        np.testing.assert_allclose(warm.gradient, result.gradient, atol=1e-9, rtol=0)


@pytest.mark.parametrize("execution", ["reference", "native"])
@pytest.mark.parametrize("method", ["lda-uks", "pbe-uks"])
def test_complete_open_shell_uks_analytic_and_reconverged_fd(
    method: typing.Any, execution: typing.Any
) -> None:
    """B3: asymmetric doublet uses the same seven-source plan without RKS factors."""
    pytest.importorskip("pyscf", reason="independent analytic reference requires PySCF")
    charge, multiplicity = 1, 2
    calc = calculator(method, max_iterations=200)
    with (
        calc.prepare_batch(
            [ATOMS], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis,
    ):
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        assert state.identity.method == method
        assert state.density.shape[0] == 2
        assert not np.allclose(state.density[0], state.density[1], atol=1e-12, rtol=0)
        result = complete_rks_gradient_diagnostic(
            state,
            basis,
            cache=".cache/b3-tests",
            execution=execution,
            tile_points=137,
            integral_terms=17,
            primitive_tile=29,
        )
        reference_energy, reference = independent_uks_gradient(basis, state, method)
        assert energy == pytest.approx(reference_energy, abs=2e-9)
        np.testing.assert_allclose(result.gradient, reference, atol=1e-7, rtol=0)
        np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=3e-10, rtol=0)
        assert result.work["ordered_quartets"] == basis.nao**4
        assert all(np.isfinite(value).all() for value in result.components.values())

        # Re-solve both spin channels and the physical moving grid at each displacement.
        # The directional test is kept on the compiled route to avoid duplicating the
        # expensive SCF matrix for the interpreter-only diagnostic.
        if execution == "native":
            xyz = np.asarray([position for _, position in ATOMS])
            direction = np.array(
                [[0.13, -0.07, 0.11], [-0.05, 0.17, 0.03], [0.09, 0.02, -0.14]]
            )
            estimates = []
            for step in (3e-4, 1e-4):
                energies = []
                for sign in (1, -1):
                    moved = [
                        (atom[0], position)
                        for atom, position in zip(
                            ATOMS, xyz + sign * step * direction, strict=True
                        )
                    ]
                    energies.append(
                        calc.singlepoint(
                            moved,
                            charge=charge,
                            multiplicity=multiplicity,
                            properties=("energy",),
                        ).energy
                    )
                estimates.append((energies[0] - energies[1]) / (2 * step))
            actual = float(np.sum(result.gradient * direction))
            assert abs(estimates[-1] - estimates[-2]) < 1e-6
            assert abs(estimates[-1] - actual) < 1e-6

        # A replay creates a new native identity and must revoke the old UKS lease.
        batch.execute(strict=True)
        with pytest.raises(ValueError, match="stale"):
            complete_rks_gradient_diagnostic(
                state, basis, cache=".cache/b3-tests", execution=execution
            )
        current = StationaryKsState.from_native(batch, basis)
        replay = complete_rks_gradient_diagnostic(
            current, basis, cache=".cache/b3-tests", execution=execution
        )
        np.testing.assert_allclose(replay.gradient, result.gradient, atol=1e-9, rtol=0)


def product_coordinates() -> typing.Any:
    return ((a, axis) for a in range(3) for axis in range(3))


def test_failure_isolation_native_malformed_geometry_and_detached_state() -> None:
    calc = calculator("pbe-rks")
    with calc.prepare_batch([ATOMS, ATOMS]) as batch, NativeAO(ATOMS) as basis:
        batch.execute(strict=True)
        old = StationaryKsState.from_native(batch, basis)
        contract = StationaryDerivativeContract(old.identity)
        with pytest.raises(ValueError, match="current native.*snapshot"):
            contract.validate(replace(old, _source=None))
        with pytest.raises(ValueError):
            contract.validate(
                replace(old, weighted_density=old.weighted_density * 1.01)
            )
        result = batch.execute(coordinates=[[0.0], None], strict=False)
        assert not result.items[0].succeeded and result.items[1].succeeded
        with pytest.raises(ValueError, match="stale"):
            contract.validate(old)
        live = StationaryKsState.from_native(batch, basis, index=1)
        # Bypass Python shape checks: malformed native invocation revokes all
        # prior results before descriptor validation, including the good neighbor.
        lib = batch._library
        from vibeqc import _native

        bad = _native.BatchInputDescriptor()
        status = lib.vibeqc_batch_execute(batch._batch, ct.byref(bad), 1, None, 0)
        assert status != 0
        with pytest.raises(ValueError, match="stale"):
            StationaryDerivativeContract(live.identity).validate(live)
        changed = np.array([a[1] for a in ATOMS])
        changed[1, 0] += 0.02
        batch.execute(coordinates=[changed, None], strict=True)
        with pytest.raises(ValueError, match="basis/overlap source"):
            StationaryKsState.from_native(batch, basis)
        moved = [(a[0], p) for a, p in zip(ATOMS, changed)]
        with NativeAO(moved) as moved_basis:
            new = StationaryKsState.from_native(batch, moved_basis)
            assert new.identity.owner != old.identity.owner
        for method in ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"):
            assert method_capabilities(method).supported_properties == frozenset(
                {"energy"}
            )
        with pytest.raises(ValueError, match="does not support properties"):
            calc.singlepoint(ATOMS, properties=("energy", "forces"))


@pytest.mark.parametrize("fail_publication", [False, True])
def test_compiler_source_publication_is_atomic(
    tmp_path: typing.Any, monkeypatch: typing.Any, fail_publication: typing.Any
) -> None:
    from vibeqc import _stationary_cpu as module

    path = tmp_path / "source.cpp"
    path.write_text("old complete source")
    replace_file = module.os.replace

    def check_then_publish(temporary: typing.Any, destination: typing.Any) -> None:
        assert path.read_text() == "old complete source"
        assert temporary.read_text() == "new complete source"
        if fail_publication:
            raise OSError("injected publication failure")
        replace_file(temporary, destination)

    monkeypatch.setattr(module.os, "replace", check_then_publish)
    if fail_publication:
        with pytest.raises(OSError, match="injected publication"):
            module._publish_source(path, "new complete source")
        assert path.read_text() == "old complete source"
    else:
        module._publish_source(path, "new complete source")
        assert path.read_text() == "new complete source"
    assert sorted(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("execution", ["reference", "native"])
def test_cpu_diagnostic_bounds_and_late_provider_failure(
    tmp_path: typing.Any, monkeypatch: typing.Any, execution: typing.Any
) -> None:
    from pathlib import Path

    from vibeqc import _stationary_cpu as module
    from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter

    atoms = [("H", (0.1, 0.2, -0.6)), ("H", (0.2, -0.1, 0.8))]
    calc = calculator("pbe-rks")
    compiler = CppCompilerAdapter(Path("c++"))
    with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        for name, value in (
            ("tile_points", 0),
            ("tile_points", True),
            ("integral_terms", 129),
            ("primitive_tile", 4097),
        ):
            with pytest.raises(ValueError, match=name):
                complete_rks_gradient_diagnostic(
                    state, basis, cache=tmp_path, execution=execution, **{name: value}
                )
        with pytest.raises(TypeError, match="compiler adapter"):
            complete_rks_gradient_diagnostic(
                state, basis, cache=tmp_path, execution=execution, compiler=object()
            )
        original = module._PrimitiveExecutor._run
        calls = 0

        def fail_late(
            self: typing.Any, kind: typing.Any, count: typing.Any
        ) -> typing.Any:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise ArithmeticError("injected late primitive failure")
            return original(self, kind, count)

        with monkeypatch.context() as patch:
            patch.setattr(module._PrimitiveExecutor, "_run", fail_late)
            with pytest.raises(ArithmeticError, match="late primitive"):
                complete_rks_gradient_diagnostic(
                    state, basis, cache=tmp_path, execution=execution, compiler=compiler
                )
        assert calls == 3
        assert StationaryDerivativeContract(state.identity).validate(state) is state
        good = complete_rks_gradient_diagnostic(
            state, basis, cache=tmp_path, execution=execution, compiler=compiler
        )
        assert np.isfinite(good.gradient).all()
        with pytest.raises(ValueError):
            good.gradient.flags.writeable = True
    with (
        NativeAO(ATOMS, basis="def2-svp") as d_basis,
        pytest.raises(NotImplementedError, match="s/p"),
    ):
        module._PrimitiveExecutor(d_basis, tmp_path, 128, compiler)


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks"])
def test_fresh_process_gradient_has_no_external_oracle_dependency(
    tmp_path: typing.Any, method: typing.Any
) -> None:
    import subprocess
    import sys

    code = r"""
import importlib.abc
import sys
import numpy as np
class BlockOracle(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyscf', 'gpu4pyscf', 'cupy'}:
            raise AssertionError('unexpected external oracle: ' + fullname)
sys.meta_path.insert(0, BlockOracle())
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc_compiler.dft import NativeAO
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
atoms = [('H', (.1, .2, -.6)), ('H', (.2, -.1, .8))]
calc = Calculator(method=sys.argv[2], device='cpu',
    ks_options=KsOptions(grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)),
    energy_tolerance=1e-12, density_tolerance=1e-10)
with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
    batch.execute(strict=True)
    state = StationaryKsState.from_native(batch, basis)
    def blocked(*args, **kwargs):
        raise AssertionError("native path invoked an interpreter")
    from vibeqc_compiler.xc.grid_response import GridResponseProgram
    from vibeqc_compiler.xc.coefficients import AOJetPullbackProgram
    GridResponseProgram.evaluate = blocked
    AOJetPullbackProgram.evaluate = blocked
    for module in tuple(sys.modules.values()):
        if module is not None and getattr(module, '__name__', '').startswith(('vibeqc.', 'vibeqc_compiler.')):
            for name in ('evaluate_array_graph',):
                if hasattr(module, name):
                    setattr(module, name, blocked)
            if getattr(module, '__name__', '') in {'vibeqc_compiler.tensor', 'vibeqc_compiler.tensor.interpreter', 'vibeqc_compiler.method.stationary_gradient', 'vibeqc._stationary_cpu'}:
                module.execute = blocked
    result = complete_rks_gradient_diagnostic(state, basis, cache=sys.argv[1], execution="native")
    assert np.isfinite(result.gradient).all()
    np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=1e-10)
assert not any(name.split('.')[0] in {'pyscf', 'gpu4pyscf', 'cupy'} for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), method],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_native_late_grid_failure_stale_lease_and_changed_geometry(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    from vibeqc import _stationary_cpu as module

    atoms = [("H", (0.1, 0.2, -0.6)), ("H", (0.2, -0.1, 0.8))]
    calc = calculator("pbe-rks")
    run = module.NativeGridContraction.contract
    with calc.prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        calls = 0

        def fail_late(
            self: typing.Any, *args: typing.Any, **kwargs: typing.Any
        ) -> typing.Any:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("injected late grid failure")
            return run(self, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(module.NativeGridContraction, "contract", fail_late)
            with pytest.raises(ValueError, match="late grid"):
                complete_rks_gradient_diagnostic(
                    state, basis, cache=tmp_path, execution="native"
                )
        assert calls == 2
        StationaryDerivativeContract(state.identity).validate(state)
        calls = 0

        def revoke(
            self: typing.Any, *args: typing.Any, **kwargs: typing.Any
        ) -> typing.Any:
            nonlocal calls
            calls += 1
            result = run(self, *args, **kwargs)
            if calls == 1:
                batch.execute(strict=True)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(module.NativeGridContraction, "contract", revoke)
            with pytest.raises(ValueError, match="stale"):
                complete_rks_gradient_diagnostic(
                    state, basis, cache=tmp_path, execution="native"
                )
        changed = [("H", (0.15, 0.2, -0.6)), atoms[1]]
        batch.execute(coordinates=[np.array([p for _, p in changed])], strict=True)
        with NativeAO(changed) as moved_basis:
            current = StationaryKsState.from_native(batch, moved_basis)
            native = complete_rks_gradient_diagnostic(
                current,
                moved_basis,
                cache=tmp_path,
                execution="native",
                tile_points=137,
            )
            reference = complete_rks_gradient_diagnostic(
                current,
                moved_basis,
                cache=tmp_path,
                execution="reference",
                tile_points=137,
            )
            assert native.plan_identity == reference.plan_identity
            for name in native.components:
                np.testing.assert_allclose(
                    native.components[name],
                    reference.components[name],
                    atol=1e-12,
                    rtol=0,
                )

"""Native UKS CPKS acceptance against independent libcint/Libxc and SCF."""

import typing
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._ks_snapshot import NativeKsSnapshot
from vibeqc_compiler.dft import ExplicitGrid, NativeAO
from vibeqc_compiler.xc import functional

from tools.vibeqc_response import (
    GMRESOptions,
    KrylovRecycleSpace,
    NativeRKSResponse,
    NativeUKSResponse,
    ResponseCompatibilityError,
    ResponseUnsupported,
    UHFResponseOperator,
    solve,
    solve_many,
)

LIH = [("Li", (0.1, -0.2, 0.0)), ("H", (0.1, -0.2, 2.7))]
H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
# These are fixed-grid response/oracle tests, not quadrature convergence tests.
# Keep a nontrivial atom-centred grid and the same strict independent SCF/fxc/FD
# checks; production-grid convergence is covered by test_grid_policy_convergence.
GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)


def _calculator(method: str) -> Calculator:
    return Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )


@pytest.fixture(params=("lda-uks", "pbe-uks"), scope="module")
def native(request: typing.Any) -> typing.Iterator[typing.Any]:
    with (
        _calculator(request.param).prepare_batch(
            [LIH], charges=[1], multiplicities=[2]
        ) as batch,
        NativeAO(LIH, charge=1, multiplicity=2) as basis,
    ):
        result = batch.execute(strict=True)
        with NativeUKSResponse.from_native(batch, basis, tile_points=257) as response:
            assert response.problem.reference.reference_energy == result.items[0].energy
            assert response.problem.reference.algorithm == "UKS"
            yield response, basis


def _independent_mf(response: typing.Any, basis: typing.Any) -> typing.Any:
    """Reuse only input shells and quadrature, never native integral/XC outputs."""
    pytest.importorskip("pyscf")
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
        atom=[
            (label, a.position) for label, a in zip(labels, basis.atoms, strict=True)
        ],
        basis=shells,
        charge=basis.charge,
        spin=basis.multiplicity - 1,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = dft.UKS(mol)
    mf.xc = "PBE" if response.state.identity.method == "pbe-uks" else "LDA_X,LDA_C_PW"
    mf.grids.coords = np.array(response.state.grid.points)
    mf.grids.weights = np.array(response.state.grid.weights)
    mf.small_rho_cutoff = 0
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.mo_coeff = response.state.coefficients.copy()
    mf.mo_energy = response.state.orbital_energies.copy()
    mf.mo_occ = response.state.occupations.copy()
    return mf


def _density_direction(response: typing.Any, vector: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            c @ response.problem.layout.density_matrix(spin, vector) @ c.T
            for spin, c in zip(
                ("alpha", "beta"), response.state.coefficients, strict=True
            )
        ]
    )


def _project(response: typing.Any, fock: np.ndarray) -> np.ndarray:
    """Independent projection of two AO potential blocks into the packed layout."""
    layout = response.problem.layout
    return layout.pack(
        *[
            (c.T @ f @ c)[np.ix_(layout.spaces(spin)[1], layout.spaces(spin)[0])].T
            for spin, c, f in zip(
                ("alpha", "beta"), response.state.coefficients, fock, strict=True
            )
        ]
    )


def _oracle_action(
    response: typing.Any, mf: typing.Any, vector: np.ndarray
) -> np.ndarray:
    layout = response.problem.layout
    blocks = layout.split(vector)
    gap = layout.pack(
        *[
            (
                eps[list(layout.spaces(spin)[1])][None, :]
                - eps[list(layout.spaces(spin)[0])][:, None]
            )
            * x
            for spin, eps, x in zip(
                ("alpha", "beta"), response.state.orbital_energies, blocks, strict=True
            )
        ]
    )
    return gap + _project(
        response, mf.gen_response(hermi=1)(_density_direction(response, vector))
    )


def test_spin_action_finite_rotations_and_transpose(native: typing.Any) -> None:
    from scipy.linalg import expm

    response, basis = native
    state, layout = response.state, response.problem.layout
    mf = _independent_mf(response, basis)
    np.testing.assert_allclose(mf.get_ovlp(), state.overlap, atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        mf.get_hcore() + mf.get_veff(dm=state.density), state.fock, atol=3e-9, rtol=0
    )
    x, y = np.random.default_rng(179).normal(size=(2, response.dimension)) * 0.1
    # Isolated alpha/beta inputs expose missing cross-spin blocks.
    na = layout.block_dimension("alpha")
    for vector in (
        x,
        np.r_[x[:na], np.zeros(response.dimension - na)],
        np.r_[np.zeros(na), x[na:]],
    ):
        np.testing.assert_allclose(
            response.apply(vector),
            _oracle_action(response, mf, vector),
            atol=3e-9,
            rtol=3e-9,
        )
    assert response.dot_identity(x, y) < 3e-12
    np.testing.assert_allclose(
        response.apply_transpose(x),
        _oracle_action(response, mf, x),
        atol=3e-9,
        rtol=3e-9,
    )
    for step in (1e-3, 3e-4, 1e-4):
        gradients = []
        for sign in (1, -1):
            coefficients = np.stack(
                [
                    c @ expm(-sign * step * layout.generator_matrix(spin, x))
                    for spin, c in zip(
                        ("alpha", "beta"), state.coefficients, strict=True
                    )
                ]
            )
            density = np.stack(
                [
                    (c * occ) @ c.T
                    for c, occ in zip(coefficients, state.occupations, strict=True)
                ]
            )
            fock = mf.get_hcore() + mf.get_veff(dm=density)
            gradients.append(
                layout.pack(
                    *[
                        (c.T @ f @ c)[
                            np.ix_(layout.spaces(spin)[1], layout.spaces(spin)[0])
                        ].T
                        for spin, c, f in zip(
                            ("alpha", "beta"), coefficients, fock, strict=True
                        )
                    ]
                )
            )
        np.testing.assert_allclose(
            response.apply(x),
            (gradients[0] - gradients[1]) / (2 * step),
            atol=3e-7,
            rtol=3e-7,
        )


def test_solve_reconverged_spin_densities_and_recycling(native: typing.Any) -> None:
    response, basis = native
    mf = _independent_mf(response, basis)
    mf.kernel(dm0=response.state.density)
    assert mf.converged
    np.testing.assert_allclose(
        mf.e_tot, response.problem.reference.reference_energy, atol=3e-9, rtol=0
    )
    perturbation = np.random.default_rng(180).normal(
        size=response.problem.reference.hcore.shape
    )
    perturbation = 0.05 * (perturbation + perturbation.T)
    rhs = -_project(response, np.stack([perturbation, perturbation]))
    options = GMRESOptions(atol=1e-12, rtol=1e-11, restart=20)
    result = solve(response, rhs, options=options, raise_on_failure=True)
    oracle = _independent_mf(response, basis)
    assert (
        np.linalg.norm(rhs - _oracle_action(response, oracle, result.solution)) < 3e-9
    )
    expected = _density_direction(response, result.solution)
    for step in (3e-4, 1e-4, 3e-5):
        densities = []
        for sign in (1, -1):
            displaced = _independent_mf(response, basis)
            displaced.get_hcore = lambda *_, sign=sign, step=step: (
                response.problem.reference.hcore + sign * step * perturbation
            )
            displaced.kernel(dm0=response.state.density)
            assert displaced.converged
            densities.append(displaced.make_rdm1())
        np.testing.assert_allclose(
            expected, (densities[0] - densities[1]) / (2 * step), atol=3e-6, rtol=3e-5
        )
    columns = np.column_stack([rhs, -0.4 * rhs, np.zeros_like(rhs)])
    # Keep full-size physical replay; the strategy matrix below uses H4.
    for strategy in ("recycled",):
        many = solve_many(
            response, columns, strategy=strategy, options=options, raise_on_failure=True
        )
        np.testing.assert_allclose(
            many.solution,
            result.solution[:, None] * np.array([1.0, -0.4, 0.0]),
            atol=3e-10,
        )


@pytest.mark.parametrize("method", ("lda-uks", "pbe-uks"))
def test_empty_beta_spin_and_live_identity(method: str) -> None:
    with (
        _calculator(method).prepare_batch(
            [H2], charges=[1], multiplicities=[2]
        ) as batch,
        NativeAO(H2, charge=1, multiplicity=2) as basis,
    ):
        with pytest.raises(RuntimeError):
            NativeUKSResponse.from_native(batch, basis)
        batch.execute(strict=True)
        with NativeUKSResponse.from_native(batch, basis) as response:
            assert response.problem.layout.block_dimension("beta") == 0
            mf = _independent_mf(response, basis)
            x = np.ones(response.dimension) * 0.1
            np.testing.assert_allclose(
                response.apply(x), _oracle_action(response, mf, x), atol=3e-9, rtol=3e-9
            )
            assert np.all(_density_direction(response, x)[1] == 0)
            wrong = functional(
                "LDA_XC_PW" if method == "pbe-uks" else "PBE", spin="polarized"
            )
            with pytest.raises(ValueError, match="functional"):
                NativeUKSResponse.from_native(batch, basis, functional=wrong)
            grid = response.state.grid
            with pytest.raises(ValueError, match="grid source"):
                NativeUKSResponse.from_native(
                    batch,
                    basis,
                    ExplicitGrid(grid.points, grid.weights * 1.01, grid.owners, {}),
                )
            with pytest.raises(ResponseUnsupported, match="UHF reference"):
                UHFResponseOperator(response.problem, response.backend)
            recycle = KrylovRecycleSpace(response.problem)
            solve(response, x, recycle=recycle, raise_on_failure=True)
            batch.execute(strict=True)
            for action in (
                lambda: response.apply(x),
                lambda: solve(response, np.zeros(response.dimension)),
                lambda: solve_many(
                    response, np.zeros((response.dimension, 2)), strategy="blocked"
                ),
            ):
                with pytest.raises(ValueError, match="stale"):
                    action()
            with NativeUKSResponse.from_native(batch, basis) as current:
                with pytest.raises(ResponseCompatibilityError):
                    solve(current, x, recycle=recycle)
                saved = current.state
                current.state = replace(saved, physical_residual=1e-3)
                with pytest.raises(ValueError, match="residual"):
                    current.validate_current()
                current.state = saved
                failed = batch.execute(coordinates=[[0.0]], strict=False)
                assert not failed.items[0].succeeded
                with pytest.raises(ValueError, match="stale"):
                    solve(current, np.zeros(current.dimension))
        batch.execute(strict=True)
        final = NativeUKSResponse.from_native(batch, basis)
    try:
        with pytest.raises(RuntimeError, match="closed"):
            solve(final, np.zeros(final.dimension))
    finally:
        final.close()


def test_spin_point_bridge_shape_domain_and_strides(native: typing.Any) -> None:
    """Check the spin wire layout and reject undefined directions before assembly."""
    response, _ = native
    source = response.state._source
    pbe = response.state.identity.method == "pbe-uks"
    rho = np.array([[0.7, 0.8], [0.3, 0.2]])
    grad = np.arange(12, dtype=float).reshape(2, 2, 3) * 0.01
    delta = rho * np.array([[0.17], [-0.09]])
    dg = grad * 0.1
    expected = source.evaluate_uks_response_points(pbe, rho, grad, delta, dg)
    reversed_points = source.evaluate_uks_response_points(
        pbe, rho[:, ::-1], grad[:, ::-1], delta[:, ::-1], dg[:, ::-1]
    )
    for name in ("rho", "gradient"):
        np.testing.assert_array_equal(reversed_points[name], expected[name][:, ::-1])
    with pytest.raises(ValueError, match="functional"):
        source.evaluate_uks_response_points(not pbe, rho, grad, delta, dg)
    with pytest.raises(ValueError, match="spin arrays"):
        source.evaluate_uks_response_points(pbe, rho.ravel(), grad, delta, dg)
    with pytest.raises(ValueError, match="finite"):
        source.evaluate_uks_response_points(pbe, rho, grad, delta * np.nan, dg)
    with pytest.raises(NotImplementedError, match="spin state"):
        source.evaluate_rks_response_points(
            pbe, rho.sum(axis=0), grad.sum(axis=0), delta.sum(axis=0), dg.sum(axis=0)
        )
    empty = rho.copy()
    empty[1] = 0
    zero = np.zeros_like(grad)
    with pytest.raises(RuntimeError):
        source.evaluate_uks_response_points(pbe, empty, zero, delta, zero)


@pytest.mark.parametrize("restricted", (True, False))
def test_meta_gga_snapshot_cannot_be_interpreted_as_pbe(restricted: bool) -> None:
    """The native functional wire code must remain a typed method identity."""
    charge, multiplicity = 0, 1
    method = "r2scan-rks" if restricted else "r2scan-uks"
    adapter = NativeRKSResponse if restricted else NativeUKSResponse
    with (
        _calculator(method).prepare_batch(
            [H2], charges=[charge], multiplicities=[multiplicity]
        ) as batch,
        NativeAO(H2, charge=charge, multiplicity=multiplicity) as basis,
    ):
        batch.execute(strict=True)
        with pytest.raises(ValueError, match="LDA/PBE"):
            adapter.from_native(batch, basis)
        source = NativeKsSnapshot(batch, 0)
        try:
            assert source.metadata[6] == 2
            evaluate = (
                source.evaluate_rks_response_points
                if restricted
                else source.evaluate_uks_response_points
            )
            shape = (1,) if restricted else (2, 1)
            rho, gradient = np.ones(shape), np.zeros((*shape, 3))
            with pytest.raises(NotImplementedError, match="LDA/PBE"):
                evaluate(True, rho, gradient, rho, gradient)
        finally:
            source.close()


@pytest.mark.parametrize("method", ("lda-uks", "pbe-uks"))
def test_native_multirhs_strategy_matrix_on_small_molecule(method: str) -> None:
    # H4+ has both occupied and virtual spaces in both spin channels. The larger
    # open-shell physical fixture above retains the independent three-step FD.
    atoms = [
        ("H", (0.0, 0.0, -0.7)),
        ("H", (0.1, 0.0, 0.7)),
        ("H", (3.0, 0.2, -0.65)),
        ("H", (3.2, 0.1, 0.65)),
    ]
    with (
        _calculator(method).prepare_batch(
            [atoms], charges=[1], multiplicities=[2]
        ) as batch,
        NativeAO(atoms, charge=1, multiplicity=2) as basis,
    ):
        batch.execute(strict=True)
        with NativeUKSResponse.from_native(batch, basis, tile_points=257) as response:
            assert response.problem.layout.block_dimension("alpha") == 4
            assert response.problem.layout.block_dimension("beta") == 3
            perturbation = np.random.default_rng(180).normal(
                size=response.problem.reference.hcore.shape
            )
            perturbation = 0.05 * (perturbation + perturbation.T)
            rhs = -_project(response, np.stack([perturbation, perturbation]))
            options = GMRESOptions(atol=1e-12, rtol=1e-11, restart=20)
            result = solve(response, rhs, options=options, raise_on_failure=True)
            oracle = _independent_mf(response, basis)
            assert (
                np.linalg.norm(rhs - _oracle_action(response, oracle, result.solution))
                < 3e-9
            )
            columns = np.column_stack([rhs, -0.4 * rhs, np.zeros_like(rhs)])
            for strategy in ("sequential", "blocked", "recycled"):
                many = solve_many(
                    response,
                    columns,
                    strategy=strategy,
                    options=options,
                    raise_on_failure=True,
                )
                np.testing.assert_allclose(
                    many.solution,
                    result.solution[:, None] * np.array([1.0, -0.4, 0.0]),
                    atol=3e-10,
                )

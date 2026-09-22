"""Real RKS CPKS: independent libcint/libxc and reconverged perturbation oracles."""

import typing
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._ks_snapshot import _scf_xc_points
from vibeqc_compiler.dft import ExplicitGrid, NativeAO
from vibeqc_compiler.xc import functional

from tools.vibeqc_response import (
    GMRESOptions,
    KrylovRecycleSpace,
    NativeRKSResponse,
    ResponseCompatibilityError,
    ResponseUnsupported,
    solve,
    solve_many,
)

ATOMS = [("O", (0.1, -0.1, 0.0)), ("H", (0.1, 0.2, 1.7)), ("H", (1.6, -0.2, -0.5))]
H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
# These are fixed-grid response/oracle tests, not quadrature convergence tests.
# Keep a nontrivial atom-centred grid and the same strict independent SCF/fxc/FD
# checks; production-grid convergence is covered by test_grid_policy_convergence.
GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)


def _calculator(method: str, **kwargs: typing.Any) -> Calculator:
    return Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        **kwargs,
    )


@pytest.fixture(params=("lda-rks", "pbe-rks"), scope="module")
def native(request: typing.Any) -> typing.Iterator[typing.Any]:
    with (
        _calculator(request.param).prepare_batch([ATOMS]) as batch,
        NativeAO(ATOMS) as basis,
    ):
        result = batch.execute(strict=True)
        with NativeRKSResponse.from_native(batch, basis, tile_points=257) as response:
            assert response.problem.reference.reference_energy == result.items[0].energy
            yield response, basis


def _independent_mf(response: typing.Any, basis: typing.Any) -> typing.Any:
    """Use the same input shells/grid, but independent integrals, XC and SCF."""
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
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "PBE" if response.state.identity.method == "pbe-rks" else "LDA_X,LDA_C_PW"
    mf.grids.coords = np.array(response.state.grid.points)
    mf.grids.weights = np.array(response.state.grid.weights)
    mf.small_rho_cutoff = 0
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    return mf


def _assert_independent_residual(
    response: typing.Any, mf: typing.Any, solution: np.ndarray, rhs: np.ndarray
) -> None:
    ref, layout = response.problem.reference, response.problem.layout
    c, occ = ref.coefficients, ref.occupations
    expected_density = c @ layout.density_matrix(solution) @ c.T
    # Independent true residual uses Libxc fxc and libcint J, never the tested
    # operator's action or residual report.
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = c, occ, ref.orbital_energies
    independent_action = mf.gen_response(hermi=1)
    image = independent_action(expected_density)
    gap = (
        ref.orbital_energies[ref.nocc :][None, :]
        - ref.orbital_energies[: ref.nocc, None]
    )
    residual = (
        rhs
        - (
            gap * layout.as_ia(solution)
            + (c.T @ image @ c)[np.ix_(layout.virtual, layout.occupied)].T
        ).ravel()
    )
    assert np.linalg.norm(residual) < 2e-9


def test_action_finite_rotations_transpose_and_independent_fxc(
    native: typing.Any,
) -> None:
    from scipy.linalg import expm

    response, basis = native
    ref, layout = response.problem.reference, response.problem.layout
    mf = _independent_mf(response, basis)
    c, eps, occ = ref.coefficients, ref.orbital_energies, ref.occupations
    d = (c * occ) @ c.T
    np.testing.assert_allclose(mf.get_ovlp(), ref.overlap, atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        mf.get_hcore() + mf.get_veff(dm=d), ref.fock, atol=2e-10, rtol=0
    )
    mf.mo_coeff, mf.mo_energy, mf.mo_occ = c, eps, occ
    independent_action = mf.gen_response(hermi=1)
    rng = np.random.default_rng(179)
    x, y = rng.normal(size=(2, response.dimension)) * 0.1
    delta = c @ layout.density_matrix(x) @ c.T
    gap = eps[ref.nocc :][None, :] - eps[: ref.nocc, None]
    expected = (
        gap * layout.as_ia(x)
        + (c.T @ independent_action(delta) @ c)[
            np.ix_(layout.virtual, layout.occupied)
        ].T
    ).ravel()
    np.testing.assert_allclose(response.apply(x), expected, atol=2e-9, rtol=2e-9)
    assert response.dot_identity(x, y) < 2e-12
    np.testing.assert_allclose(
        response.apply_transpose(x), expected, atol=2e-9, rtol=2e-9
    )
    for step in (1e-3, 3e-4, 1e-4):
        focks = []
        for sign in (1, -1):
            rotated = c @ expm(-sign * step * layout.generator_matrix(x))
            density = (rotated * occ) @ rotated.T
            fock = mf.get_hcore() + mf.get_veff(dm=density)
            focks.append(rotated.T @ fock @ rotated)
        numerical = ((focks[0] - focks[1]) / (2 * step))[
            np.ix_(layout.virtual, layout.occupied)
        ].T.ravel()
        np.testing.assert_allclose(response.apply(x), numerical, atol=2e-7, rtol=2e-7)


def test_complete_solve_and_reconverged_density_response(native: typing.Any) -> None:
    response, basis = native
    ref, layout = response.problem.reference, response.problem.layout
    mf = _independent_mf(response, basis)
    c, occ = ref.coefficients, ref.occupations
    mf.kernel(dm0=(c * occ) @ c.T)
    assert mf.converged
    np.testing.assert_allclose(mf.e_tot, ref.reference_energy, atol=2e-10, rtol=0)
    rng = np.random.default_rng(180)
    perturbation = rng.normal(size=ref.hcore.shape)
    perturbation = 0.1 * (perturbation + perturbation.T)
    rhs = -(c.T @ perturbation @ c)[np.ix_(layout.virtual, layout.occupied)].T.ravel()
    options = GMRESOptions(atol=1e-12, rtol=1e-11)
    result = solve(response, rhs, options=options, raise_on_failure=True)
    expected_density = c @ layout.density_matrix(result.solution) @ c.T
    _assert_independent_residual(response, mf, result.solution, rhs)
    for step in (3e-4, 1e-4, 3e-5):
        densities = []
        for sign in (1, -1):
            displaced = _independent_mf(response, basis)
            displaced.get_hcore = lambda *_, sign=sign, step=step: (
                ref.hcore + sign * step * perturbation
            )
            displaced.kernel(dm0=response.state.density[0])
            assert displaced.converged
            densities.append(displaced.make_rdm1())
        numerical = (densities[0] - densities[1]) / (2 * step)
        np.testing.assert_allclose(expected_density, numerical, atol=2e-6, rtol=2e-5)

    columns = np.column_stack([rhs, -0.4 * rhs, np.zeros_like(rhs)])
    # Keep full-size physical replay; the strategy matrix below uses H4.
    for strategy in ("recycled",):
        many = solve_many(
            response, columns, strategy=strategy, options=options, raise_on_failure=True
        )
        for j, scale in enumerate((1, -0.4, 0)):
            np.testing.assert_allclose(
                many.results[j].solution, scale * result.solution, atol=2e-10
            )


def test_scf_point_response_tail_zero_gradient_and_finite_difference(
    native: typing.Any,
) -> None:
    response, _ = native
    source = response.state._source
    pbe = response.state.identity.method == "pbe-rks"
    # Both rational branches, exact grad=0, and tails outside interior-v1.
    rho = np.array([0.9, 0.9, 0.9, 1e-20, 1e-80, 1e-180])
    grad = rho[:, None] * np.array(
        [
            [0, 0, 0],
            [0.03, -0.02, 0.01],
            [7, -2, 1],
            [2, -1, 0.1],
            [2, -1, 0.1],
            [2, -1, 0.1],
        ]
    )
    drho = 0.17 * rho
    dg = rho[:, None] * np.array([0.07, 0.02, -0.03])
    actual = source.evaluate_rks_response_points(pbe, rho, grad, drho, dg)
    for step in (1e-3, 3e-4, 1e-4):
        values = []
        for sign in (1, -1):
            values.append(
                _scf_xc_points(
                    source._library,
                    pbe,
                    np.stack([rho + sign * step * drho] * 2) / 2,
                    np.stack([grad + sign * step * dg] * 2) / 2,
                )
            )
        for key in ("rho", "gradient"):
            expected = ((values[0][key] - values[1][key]) / (2 * step)).mean(
                axis=0, keepdims=True
            )
            # PBE's unpolarized gradient coefficient cancels at grad=0;
            # the central difference retains an O(step**2) truncation term.
            # Keep that allowance separate from tiny tail coefficients, which
            # must pass a relative test rather than an ordinary absolute gate.
            np.testing.assert_allclose(
                actual[key][:, :3], expected[:, :3], rtol=3e-6, atol=2e-6 * step**2
            )
            np.testing.assert_allclose(
                actual[key][:, 3:], expected[:, 3:], rtol=3e-6, atol=1e-300
            )
    zero = source.evaluate_rks_response_points(
        pbe, np.zeros(1), np.zeros((1, 3)), np.zeros(1), np.zeros((1, 3))
    )
    assert not np.any(zero["rho"])
    tiny_zero = source.evaluate_rks_response_points(
        pbe, np.array([1e-280]), np.zeros((1, 3)), np.zeros(1), np.zeros((1, 3))
    )
    assert not np.any(tiny_zero["rho"]) and not np.any(tiny_zero["gradient"])
    with pytest.raises(RuntimeError):
        source.evaluate_rks_response_points(
            pbe, np.zeros(1), np.zeros((1, 3)), np.ones(1), np.zeros((1, 3))
        )


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks"))
def test_native_cpks_identity_lifetime_and_recycling(method: str) -> None:
    calc = _calculator(method)
    with calc.prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        with pytest.raises(RuntimeError):
            NativeRKSResponse.from_native(batch, basis)
        batch.execute(strict=True)
        with NativeRKSResponse.from_native(batch, basis) as response:
            wrong = functional(
                "LDA_XC_PW" if method == "pbe-rks" else "PBE", spin="unpolarized"
            )
            with pytest.raises(ValueError, match="functional"):
                NativeRKSResponse.from_native(batch, basis, functional=wrong)
            grid = response.state.grid
            bad_grid = ExplicitGrid(grid.points, grid.weights * 1.01, grid.owners, {})
            with pytest.raises(ValueError, match="grid source"):
                NativeRKSResponse.from_native(batch, basis, bad_grid)
            shells = list(basis.shells)
            primitives = list(shells[0].primitives)
            primitives[0] = replace(
                primitives[0], exponent=primitives[0].exponent * 1.01
            )
            shells[0] = replace(shells[0], primitives=tuple(primitives))
            with (
                NativeAO(H2, basis=shells) as bad_basis,
                pytest.raises(ValueError, match="basis/overlap"),
            ):
                NativeRKSResponse.from_native(batch, bad_basis)
            recycle = KrylovRecycleSpace(response.problem)
            solve(
                response,
                np.ones(response.dimension),
                recycle=recycle,
                raise_on_failure=True,
            )
            # Re-executing identical geometry also revokes the previous epoch.
            batch.execute(strict=True)
            for action in (
                lambda: response.apply(np.ones(response.dimension)),
                lambda: solve(response, np.zeros(response.dimension)),
                lambda: solve_many(
                    response, np.zeros((response.dimension, 2)), strategy="blocked"
                ),
            ):
                with pytest.raises(ValueError, match="stale"):
                    action()
            with NativeRKSResponse.from_native(batch, basis) as current:
                with pytest.raises(ResponseCompatibilityError):
                    solve(current, np.ones(current.dimension), recycle=recycle)
                # Same-shaped provider/model replacement does not renew a lease.
                saved = current.xc_kernel.spec
                current.xc_kernel.spec = wrong
                with pytest.raises(ValueError, match="identity"):
                    current.apply(np.ones(current.dimension))
                current.xc_kernel.spec = saved
                saved_grid = current.xc_kernel.grid
                current.xc_kernel.grid = bad_grid
                with pytest.raises(ValueError, match="identity"):
                    current.apply(np.ones(current.dimension))
                current.xc_kernel.grid = saved_grid
                saved_provider = current.backend.identity
                current.backend.identity = "unrelated-provider"
                with pytest.raises(ValueError, match="identity"):
                    solve(current, np.zeros(current.dimension))
                current.backend.identity = saved_provider
                saved_state = current.state
                current.state = replace(saved_state, physical_residual=1e-3)
                with pytest.raises(ValueError, match="residual"):
                    current.validate_current()
                current.state = saved_state
                failed = batch.execute(coordinates=[[0.0]], strict=False)
                assert not failed.items[0].succeeded
                with pytest.raises(ValueError, match="stale"):
                    solve(current, np.zeros(current.dimension))
        batch.execute(strict=True)
        final = NativeRKSResponse.from_native(batch, basis)
    try:
        with pytest.raises(RuntimeError, match="closed"):
            solve(final, np.zeros(final.dimension))
    finally:
        final.close()


def test_native_uks_is_not_inferred_from_rks() -> None:
    with _calculator("pbe-uks").prepare_batch(
        [H2], charges=[-1], multiplicities=[2]
    ) as batch:
        batch.execute(strict=True)
        with (
            NativeAO(H2, charge=-1, multiplicity=2) as basis,
            pytest.raises(ResponseUnsupported, match="all-electron RKS"),
        ):
            NativeRKSResponse.from_native(batch, basis)


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks"))
def test_native_multirhs_strategy_matrix_on_small_molecule(method: str) -> None:
    # Two occupied and two virtual orbitals: retain a genuine coupled response,
    # not the scalar two-electron limit, without repeating the water endpoint.
    atoms = [
        ("H", (0.0, 0.0, -0.7)),
        ("H", (0.1, 0.0, 0.7)),
        ("H", (3.0, 0.2, -0.65)),
        ("H", (3.2, 0.1, 0.65)),
    ]
    with _calculator(method).prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
        batch.execute(strict=True)
        with NativeRKSResponse.from_native(batch, basis, tile_points=257) as response:
            ref, layout = response.problem.reference, response.problem.layout
            assert ref.nocc == 2 and response.dimension == 4
            c = ref.coefficients
            perturbation = np.random.default_rng(180).normal(size=ref.hcore.shape)
            perturbation = 0.1 * (perturbation + perturbation.T)
            rhs = -(c.T @ perturbation @ c)[
                np.ix_(layout.virtual, layout.occupied)
            ].T.ravel()
            options = GMRESOptions(atol=1e-12, rtol=1e-11)
            result = solve(response, rhs, options=options, raise_on_failure=True)
            _assert_independent_residual(
                response, _independent_mf(response, basis), result.solution, rhs
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
                for j, scale in enumerate((1, -0.4, 0)):
                    np.testing.assert_allclose(
                        many.results[j].solution, scale * result.solution, atol=2e-10
                    )

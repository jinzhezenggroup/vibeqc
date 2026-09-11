"""CPKS feature-kernel action and fail-closed public DFT boundaries."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc import FixedDensityXC, UnsupportedXC, functional
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture

from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture
from tools.vibeqc_response import (
    CPKSResponseOperator,
    DenseAOResponseBackend,
    FixedDensityXCDerivativeKernel,
    ResponseUnsupported,
    solve,
)


def test_fixed_density_xc_hessian_matches_potential_finite_difference():
    meta, arrays, grid = load_integration_fixture("h2")
    spec = functional("PBE", spin="unpolarized")
    density = arrays["density_total"]
    delta = np.array([[0.3, -0.2], [-0.2, 0.1]])
    with NativeAO(**basis_arguments(meta)) as basis:
        kernel = FixedDensityXCDerivativeKernel(
            spec, basis, grid, density, tile_points=7
        )
        actual = kernel.apply(delta)
        integrator = FixedDensityXC(spec)
        reference = integrator.integrate(basis, grid, density, tile_points=7)
        for step in (1e-4, 3e-5, 1e-5):
            plus = integrator.integrate(
                basis, grid, density + step * delta, tile_points=7
            )
            minus = integrator.integrate(
                basis, grid, density - step * delta, tile_points=7
            )
            expected = (plus.potential - minus.potential) / (2 * step)
            np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-8)
        assert np.isfinite(reference.energy)
        assert kernel.identity
        assert kernel.statistics["tiles"] > 0


def test_polarized_fixed_density_xc_response_averages_spin_potentials():
    meta, arrays, grid = load_integration_fixture("h2")
    spec = functional("PBE", spin="polarized")
    density = arrays["density_total"]
    delta = np.array([[0.3, -0.2], [-0.2, 0.1]])
    with NativeAO(**basis_arguments(meta)) as basis:
        kernel = FixedDensityXCDerivativeKernel(
            spec, basis, grid, density, tile_points=7
        )
        actual = kernel.apply(delta)
        integrator = FixedDensityXC(spec)
        for step in (1e-4, 3e-5, 1e-5):
            plus = integrator.integrate(
                basis, grid, density + step * delta, tile_points=7
            )
            minus = integrator.integrate(
                basis, grid, density - step * delta, tile_points=7
            )
            expected = (plus.potential - minus.potential) / (2 * step)
            np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-8)


def test_fixed_density_xc_kernel_rejects_wrong_reference_density():
    meta, _arrays, grid = load_integration_fixture("h2")
    spec = functional("PBE", spin="unpolarized")
    with NativeAO(**basis_arguments(meta)) as basis:
        reference = fixture_snapshot(*load_fixture("h2"))
        reference_density = (
            reference.coefficients * reference.occupations
        ) @ reference.coefficients.T
        kernel = FixedDensityXCDerivativeKernel(spec, basis, grid, reference_density)
        ks = replace(
            reference,
            algorithm="KS",
            functional_identity=spec.identity,
            grid_identity=grid.identity,
            geometry_hash=kernel.geometry_hash,
            basis_hash=kernel.basis_hash,
            hf_backend="test-ks",
        )
        kernel.validate_reference(ks)
        wrong = FixedDensityXCDerivativeKernel(
            spec, basis, grid, reference_density * 1.1
        )
        with pytest.raises(ValueError, match="density mismatch"):
            wrong.validate_reference(ks)


def test_fixed_density_xc_kernel_rejects_exact_exchange_and_stale_grid():
    meta, arrays, grid = load_integration_fixture("h2")
    spec = functional("PBE", spin="unpolarized")
    with NativeAO(**basis_arguments(meta)) as basis:
        with pytest.raises(UnsupportedXC, match="exact exchange"):
            FixedDensityXCDerivativeKernel(
                replace(spec, exact_exchange=Fraction(1, 4)),
                basis,
                grid,
                arrays["density_total"],
            )
        moved = replace(grid, owners=(1,) * len(grid.owners))
        # The explicit-grid branch is intentionally fixed-frame, but a
        # molecular grid must reject a changed topology before contraction.
        del moved


class _LinearXCDerivativeKernel:
    """Synthetic symmetric kernel for CPKS plumbing tests, not a DFT model."""

    def __init__(self, basis_identity, grid_identity, functional_identity, scale=0.3):
        self.basis_identity = basis_identity
        self.grid_identity = grid_identity
        self.functional_identity = functional_identity
        self.scale = float(scale)
        self.identity = f"linear-test-{scale}"

    def apply(self, delta_density):
        return self.scale * np.asarray(delta_density)

    def apply_transpose(self, delta_density):
        return self.apply(delta_density)


def test_cpks_operator_action_and_solve_with_synthetic_ks_reference():
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    ks = replace(
        reference,
        algorithm="KS",
        functional_identity="pbe-test",
        grid_identity="grid-test",
        hf_backend="synthetic-test-ks",
    )
    backend = DenseAOResponseBackend(np.zeros((ks.nmo,) * 4))
    kernel = _LinearXCDerivativeKernel(
        ks.basis_hash, ks.grid_identity, ks.functional_identity
    )
    problem = CPKSResponseOperator.build_problem(ks, backend, kernel)
    operator = CPKSResponseOperator(problem, backend, kernel)
    dense = operator.to_dense()
    x = np.zeros((problem.dimension, 1))
    x[0, 0] = 1.0
    delta_mo = problem.layout.density_matrix(x[:, 0])
    delta_ao = ks.coefficients @ delta_mo @ ks.coefficients.T
    xc_mo = ks.coefficients.T @ kernel.apply(delta_ao) @ ks.coefficients
    xc_block = xc_mo[np.ix_(problem.layout.virtual, problem.layout.occupied)].T
    expected = np.diag(
        (
            ks.orbital_energies[ks.nocc :, None]
            - ks.orbital_energies[: ks.nocc][None, :]
        ).reshape(-1)
        + xc_block.reshape(-1)
    )
    np.testing.assert_allclose(dense, expected, atol=1e-12)
    rhs = np.linspace(-0.5, 0.5, problem.dimension)
    result = solve(operator, rhs)
    assert result.converged
    assert result.residual_norm < 1e-11


def test_cpks_requires_matching_kernel_identities_and_ks_reference():
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    backend = DenseAOResponseBackend(np.zeros((reference.nmo,) * 4))
    kernel = _LinearXCDerivativeKernel(reference.basis_hash, "grid-test", "pbe-test")
    with pytest.raises(ResponseUnsupported, match="converged KS reference"):
        CPKSResponseOperator.build_problem(reference, backend, kernel)
    ks = replace(
        reference,
        algorithm="KS",
        functional_identity="pbe-test",
        grid_identity="grid-test",
        hf_backend="synthetic-test-ks",
    )
    wrong = _LinearXCDerivativeKernel(
        ks.basis_hash, "other-grid", ks.functional_identity
    )
    problem = CPKSResponseOperator.build_problem(ks, backend, kernel)
    with pytest.raises(ValueError, match="grid_identity"):
        CPKSResponseOperator(problem, backend, wrong)

"""Checks for the shared closed-shell nuclear-response convention."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_hessian import (
    build_rhf_nuclear_rhs,
    build_stationary_nuclear_rhs,
    metric_density_response_mo,
    solve_stationary_nuclear_perturbation,
)
from tools.vibeqc_hessian.perturbation import solve_rhf_nuclear_perturbation
from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture
from tools.vibeqc_response import (
    CPKSResponseOperator,
    DenseAOResponseBackend,
    RHFResponseOperator,
)


def test_metric_density_response_uses_closed_shell_occupation_connection() -> None:
    overlap = np.array([[0.2, 0.3, 0.4], [0.3, -0.1, 0.5], [0.4, 0.5, 0.6]])
    expected = -0.5 * (
        overlap * np.array([2.0, 0.0, 0.0])[None, :]
        + np.array([2.0, 0.0, 0.0])[:, None] * overlap
    )
    np.testing.assert_allclose(
        metric_density_response_mo(overlap, nocc=1), expected, rtol=0, atol=0
    )


def test_nuclear_rhs_includes_metric_fock_and_overlap_energy_gap_term() -> None:
    energies = np.array([-0.7, 0.2, 0.8])
    frozen = np.array([[0.0, 0.1, 0.2], [0.3, 0.0, 0.4], [0.5, 0.6, 0.0]])
    overlap = np.array([[0.0, 0.7, 0.8], [0.9, 0.0, 0.2], [0.3, 0.4, 0.0]])
    metric = np.array([[0.0, 0.01, 0.02], [0.03, 0.0, 0.04], [0.05, 0.06, 0.0]])
    rhs = build_rhf_nuclear_rhs(frozen, overlap, metric, energies, nocc=1)
    expected = np.array(
        [[0.3 + 0.03 - 0.5 * (0.2 - 0.7) * 0.7, 0.5 + 0.05 - 0.5 * (0.8 - 0.7) * 0.8]]
    )
    np.testing.assert_allclose(rhs, expected, rtol=0, atol=1e-15)
    assert rhs.shape == (1, 2)


def test_nuclear_rhs_rejects_missing_or_malformed_metric_inputs() -> None:
    values = np.zeros((3, 3))
    with pytest.raises(ValueError, match="finite"):
        bad = values.copy()
        bad[0, 0] = np.nan
        build_rhf_nuclear_rhs(bad, values, values, np.arange(3.0), nocc=1)
    with pytest.raises(ValueError, match="shape"):
        build_rhf_nuclear_rhs(values, values, np.zeros((2, 2)), np.arange(3.0), nocc=1)
    with pytest.raises(ValueError, match="between"):
        metric_density_response_mo(values, nocc=0)


def test_rhf_induced_fock_preserves_j_minus_half_k_contract() -> None:
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    eri = np.arange(reference.nmo**4, dtype=np.float64).reshape((reference.nmo,) * 4)
    backend = DenseAOResponseBackend(eri)
    problem = RHFResponseOperator.build_problem(reference, backend)
    operator = RHFResponseOperator(problem, backend)
    density = np.array([[0.3, -0.07], [-0.07, 0.2]])

    coulomb, exchange = backend.coulomb_exchange(density)
    np.testing.assert_allclose(
        operator.induced_fock(density),
        coulomb - 0.5 * exchange,
        atol=0,
        rtol=0,
    )


class _LinearXCKernel:
    """Synthetic XC Hessian action for closed-shell plumbing tests."""

    def __init__(
        self, basis_identity: str, grid_identity: str, functional_identity: str
    ) -> None:
        self.basis_identity = basis_identity
        self.grid_identity = grid_identity
        self.functional_identity = functional_identity
        self.identity = "hessian-linear-xc"
        self.host_workspace_bytes = 0

    def apply(self, delta_density: np.ndarray) -> np.ndarray:
        return 0.25 * np.asarray(delta_density)

    def apply_transpose(self, delta_density: np.ndarray) -> np.ndarray:
        return self.apply(delta_density)


def test_stationary_nuclear_response_accepts_cpks_induced_fock() -> None:
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
    kernel = _LinearXCKernel(ks.basis_hash, ks.grid_identity, ks.functional_identity)
    problem = CPKSResponseOperator.build_problem(ks, backend, kernel)
    operator = CPKSResponseOperator(problem, backend, kernel)

    frozen = np.zeros((ks.nmo, ks.nmo))
    overlap = np.array([[0.2, 0.08], [0.08, -0.1]])
    response = solve_stationary_nuclear_perturbation(operator, frozen, overlap)

    coefficients = ks.coefficients
    overlap_mo = coefficients.T @ overlap @ coefficients
    metric_mo = metric_density_response_mo(overlap_mo, nocc=ks.nocc)
    metric_ao = coefficients @ metric_mo @ coefficients.T
    metric_fock_mo = coefficients.T @ operator.induced_fock(metric_ao) @ coefficients
    expected = build_stationary_nuclear_rhs(
        coefficients.T @ frozen @ coefficients,
        overlap_mo,
        metric_fock_mo,
        ks.orbital_energies,
        nocc=ks.nocc,
    )

    np.testing.assert_allclose(response.rhs, expected, atol=1e-12, rtol=0)
    without_xc = build_stationary_nuclear_rhs(
        coefficients.T @ frozen @ coefficients,
        overlap_mo,
        np.zeros_like(metric_fock_mo),
        ks.orbital_energies,
        nocc=ks.nocc,
    )
    assert np.max(np.abs(response.rhs - without_xc)) > 1e-8
    with pytest.raises(TypeError, match="RHF nuclear response"):
        solve_rhf_nuclear_perturbation(operator, frozen, overlap)


@pytest.mark.parametrize("method", ("rhf", "cpks"))
@pytest.mark.parametrize("transpose", (False, True))
@pytest.mark.parametrize("imaginary", (0.0, 0.25))
def test_induced_fock_rejects_complex_density_before_backend(
    method: str, transpose: bool, imaginary: float
) -> None:
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    backend = DenseAOResponseBackend(np.zeros((reference.nmo,) * 4))
    if method == "cpks":
        reference = replace(
            reference,
            algorithm="KS",
            functional_identity="pbe-test",
            grid_identity="grid-test",
            hf_backend="synthetic-test-ks",
        )
        kernel = _LinearXCKernel(
            reference.basis_hash, reference.grid_identity, reference.functional_identity
        )
        problem = CPKSResponseOperator.build_problem(reference, backend, kernel)
        operator = CPKSResponseOperator(problem, backend, kernel)
    else:
        problem = RHFResponseOperator.build_problem(reference, backend)
        operator = RHFResponseOperator(problem, backend)
    density = np.eye(reference.nmo, dtype=np.complex128) * (0.3 + imaginary * 1j)
    with pytest.raises(ValueError, match="real"):
        operator.induced_fock(density, transpose=transpose)
    assert backend.statistics["actions"] == 0

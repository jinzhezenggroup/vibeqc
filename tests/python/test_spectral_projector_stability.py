"""Projector response must not inherit conditioning/underflow of f(A)."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.method.matrix_function import SymmetricMatrixFunctionSpec
from vibeqc_compiler.method.spectral_projector import prepare_fixed_rank_projector


@pytest.mark.parametrize("exponent", (8, 10, 12))
def test_ill_conditioned_pseudoinverse_keeps_cross_gap(exponent: int) -> None:
    q, _ = np.linalg.qr(np.random.default_rng(812).normal(size=(5, 5)))
    matrix = (q * np.array([0.0, 10.0**-exponent, 1e-3, 0.3, 1.0])) @ q.T
    state = SymmetricMatrixFunctionSpec(
        5,
        "conditioned",
        function="pseudoinverse",
        relative_threshold=10.0 ** (-exponent - 1),
        branch_guard=10.0 ** (-exponent - 3),
    ).prepare(matrix)
    response = prepare_fixed_rank_projector(state)
    # Independent eigenvalue-only oracle; do not recover f(lambda) from f(A).
    values = np.linalg.eigvalsh(0.5 * matrix + 0.5 * matrix.T)
    assert state.rank == 4
    expected = 1.0 / (values[-1] - values[0])
    assert response.divided[-1, 0] == pytest.approx(expected, rel=3e-12)


@pytest.mark.parametrize(
    "function,scale", (("inverse_sqrt", 1e220), ("pseudoinverse", 1e170))
)
def test_parent_derivative_underflow_does_not_erase_projector_response(
    function: str, scale: float
) -> None:
    matrix = np.diag(np.array([0.2, 2.0, 4.0]) * scale)
    state = SymmetricMatrixFunctionSpec(
        3,
        "scaled",
        function=function,
        relative_threshold=0.1,
    ).prepare(matrix)
    response = prepare_fixed_rank_projector(state)
    # Multiplication by scale makes this an O(1) check without an absolute
    # tolerance that could silently accept a zero subnormal-scale coefficient.
    assert response.divided[1, 0] * scale == pytest.approx(1.0 / 1.8, rel=3e-15)
    assert response.divided[2, 0] * scale == pytest.approx(1.0 / 3.8, rel=3e-15)


@pytest.mark.parametrize("action", ("jvp", "vjp"))
def test_full_projector_huge_finite_seed_is_exact_zero(action: str) -> None:
    q = np.array([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
    matrix = (q * np.array([1.0, 2.0])) @ q.T
    state = SymmetricMatrixFunctionSpec(2, "full", relative_threshold=0.01).prepare(
        matrix
    )
    response = prepare_fixed_rank_projector(state)
    with np.errstate(over="raise", invalid="raise"):
        result = getattr(response, action)(np.full((2, 2), 1.7e308))
    np.testing.assert_array_equal(result, np.zeros((2, 2)))
    assert not result.flags.writeable
    with pytest.raises(ValueError, match="finite"):
        getattr(response, action)(np.full((2, 2), np.inf))


@pytest.mark.parametrize("function", ("inverse_sqrt", "pseudoinverse"))
def test_projector_multistep_difference_and_degenerate_gauge(function: str) -> None:
    q, _ = np.linalg.qr(np.random.default_rng(933).normal(size=(4, 4)))
    matrix = (q * np.array([0.2, 0.3, 2.0, 2.0])) @ q.T
    raw = np.random.default_rng(934).normal(size=(4, 4))
    tangent = 0.5 * raw + 0.5 * raw.T
    state = SymmetricMatrixFunctionSpec(
        4,
        "gauge",
        function=function,
        relative_threshold=0.4,
    ).prepare(matrix)
    response = prepare_fixed_rank_projector(state)
    analytic = response.jvp(tangent)
    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        projectors = []
        for sign in (1, -1):
            values, vectors = np.linalg.eigh(matrix + sign * step * tangent)
            keep = values > 0.4 * values[-1]
            assert np.count_nonzero(keep) == 2
            projectors.append(vectors[:, keep] @ vectors[:, keep].T)
        errors.append(
            np.max(np.abs((projectors[0] - projectors[1]) / (2 * step) - analytic))
        )
    assert errors[1] < 2e-8
    assert errors[2] < 2e-9
    internal = q[:, 2:] @ np.array([[0.2, 0.7], [0.7, -0.3]]) @ q[:, 2:].T
    np.testing.assert_allclose(response.jvp(internal), 0, atol=3e-15, rtol=0)


def test_spectrum_is_immutable_and_legacy_snapshot_fails_closed() -> None:
    state = SymmetricMatrixFunctionSpec(2, "spectrum", relative_threshold=0.1).prepare(
        np.diag(np.array([0.1, 2.0]))
    )
    assert state.eigenvalues is not None
    with pytest.raises(ValueError):
        state.eigenvalues.setflags(write=True)
    with pytest.raises(ValueError, match="prepared spectral"):
        prepare_fixed_rank_projector(replace(state, eigenvalues=None))

"""Fixed-rank spectral-projector response regressions for #185."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc_compiler.method.matrix_function import (
    MatrixFunctionEvaluation,
    SymmetricMatrixFunctionSpec,
)
from vibeqc_compiler.method.spectral_projector import (
    prepare_fixed_rank_projector,
    projector_logical_workspace_bytes,
)


def _state(matrix: np.ndarray) -> MatrixFunctionEvaluation:
    return SymmetricMatrixFunctionSpec(
        matrix.shape[0],
        "pair-density",
        relative_threshold=0.1,
        branch_guard=1e-8,
    ).prepare(matrix)


def _projector(state: MatrixFunctionEvaluation) -> np.ndarray:
    retained = np.asarray(state.retained, dtype=bool)
    vectors = state.vectors[:, retained]
    return vectors @ vectors.T


def test_projector_jvp_matches_recomputed_fixed_rank_finite_difference() -> None:
    matrix = np.diag(np.array([0.2, 2.0, 4.0]))
    tangent = np.array(
        [[0.0, 0.7, -0.2], [0.7, 0.3, 0.4], [-0.2, 0.4, -0.1]],
        dtype=np.float64,
    )
    state = _state(matrix)
    response = prepare_fixed_rank_projector(state)

    analytic = response.jvp(tangent)
    step = 1.0e-5
    plus = _projector(state.rebind(matrix + step * tangent))
    minus = _projector(state.rebind(matrix - step * tangent))
    finite_difference = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(analytic, finite_difference, atol=2e-9, rtol=0.0)
    assert response.rank == 2
    assert response.retained == (False, True, True)


def test_projector_response_ignores_internal_gauge_motion() -> None:
    matrix = np.diag(np.array([0.2, 2.0, 2.0]))
    retained_internal = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.0, 0.6, 0.0]],
        dtype=np.float64,
    )

    response = prepare_fixed_rank_projector(_state(matrix))

    np.testing.assert_allclose(response.jvp(retained_internal), 0.0, atol=2e-15)
    np.testing.assert_allclose(
        response.projector,
        np.diag(np.array([0.0, 1.0, 1.0])),
        atol=2e-15,
        rtol=0.0,
    )


def test_projector_jvp_and_vjp_are_full_frobenius_adjoint() -> None:
    matrix = np.diag(np.array([0.2, 2.0, 4.0]))
    tangent = np.array(
        [[0.3, -0.5, 0.1], [-0.5, 0.2, 0.4], [0.1, 0.4, -0.7]],
        dtype=np.float64,
    )
    cotangent = np.array(
        [[0.1, 0.7, -0.2], [-0.4, 0.3, 0.8], [0.5, -0.1, -0.6]],
        dtype=np.float64,
    )
    response = prepare_fixed_rank_projector(_state(matrix))

    lhs = np.vdot(cotangent, response.jvp(tangent))
    rhs = np.vdot(response.vjp(cotangent), tangent)

    assert abs(lhs - rhs) < 2e-14
    assert not response.vjp(cotangent).flags.writeable


def test_full_retained_branch_has_identity_projector_and_zero_response() -> None:
    matrix = np.diag(np.array([1.0, 1.5, 2.0]))
    state = SymmetricMatrixFunctionSpec(
        3,
        "full-retained",
        relative_threshold=0.01,
        branch_guard=1e-8,
    ).prepare(matrix)
    response = prepare_fixed_rank_projector(state)
    tangent = np.array(
        [[0.0, 0.2, -0.4], [0.2, 0.0, 0.3], [-0.4, 0.3, 0.0]],
        dtype=np.float64,
    )

    np.testing.assert_allclose(response.projector, np.eye(3), atol=2e-15, rtol=0.0)
    np.testing.assert_array_equal(response.jvp(tangent), np.zeros((3, 3)))


def test_projector_response_rejects_untruncated_or_invalid_inputs() -> None:
    matrix = np.diag(np.array([1.0, 2.0]))
    full_state = SymmetricMatrixFunctionSpec(2, "full").prepare(matrix)

    with pytest.raises(ValueError, match="fixed-rank threshold"):
        prepare_fixed_rank_projector(full_state)

    response = prepare_fixed_rank_projector(
        SymmetricMatrixFunctionSpec(
            2,
            "truncated",
            relative_threshold=0.2,
            branch_guard=1e-8,
        ).prepare(np.diag(np.array([0.1, 2.0])))
    )
    with pytest.raises(ValueError, match="symmetric"):
        response.jvp(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="float64"):
        response.jvp(np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        response.vjp(np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float64))


def test_projector_response_budget_is_preflighted() -> None:
    state = _state(np.diag(np.array([0.2, 2.0, 4.0])))
    required = projector_logical_workspace_bytes(3)

    with pytest.raises(ValueError, match="workspace budget exceeded"):
        prepare_fixed_rank_projector(state, max_bytes=required - 1)

    response = prepare_fixed_rank_projector(state, max_bytes=required)
    assert response.max_bytes == required


def test_projector_identity_tracks_rebound_parent_state() -> None:
    matrix = np.diag(np.array([0.2, 2.0, 4.0]))
    state = _state(matrix)
    first = prepare_fixed_rank_projector(state)
    moved = state.rebind(
        matrix
        + np.array(
            [[0.0, 1e-3, 0.0], [1e-3, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    )
    second = prepare_fixed_rank_projector(moved)

    assert first.parent_identity != second.parent_identity
    assert first.identity != second.identity
    assert not first.projector.flags.writeable
    assert not first.divided.flags.writeable
    assert not first.vectors.flags.writeable

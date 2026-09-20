"""CPU checks for the #434 fixed-density diagnostic algebra."""

from __future__ import annotations

import numpy as np

from benchmarks.issue434_fixed_density import coulomb_from_raw, residuals


def test_issue434_residuals_accept_exact_closed_shell_projector() -> None:
    overlap = np.eye(2)
    fock = np.diag([-1.0, 1.0])
    density = np.diag([2.0, 0.0])

    result = residuals(fock, overlap, density)

    assert result["occupied_count"] == 1
    assert result["commutator_maximum"] == 0.0
    assert result["fixed_point_density_maximum"] == 0.0
    assert result["fixed_point_density_rms"] == 0.0
    assert result["electron_trace_error"] == 0.0
    assert result["idempotency_error"] == 0.0


def test_issue434_coulomb_recomposition_matches_explicit_metric_inverse() -> None:
    raw = np.array(
        [
            [[1.0, 0.2], [0.3, -0.1]],
            [[0.3, -0.1], [0.7, 0.4]],
        ]
    )
    metric = np.array([[2.0, 0.25], [0.25, 1.5]])
    density = np.array([[1.2, 0.1], [0.1, 0.8]])

    charge = raw.reshape(4, 2).T @ density.reshape(-1)
    expected = (raw.reshape(4, 2) @ np.linalg.solve(metric, charge)).reshape(2, 2)

    np.testing.assert_allclose(
        coulomb_from_raw(raw, metric, density), expected, atol=1e-14, rtol=0
    )

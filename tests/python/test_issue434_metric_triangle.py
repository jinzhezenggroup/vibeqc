"""RI-J schedule comparisons must use the same authoritative metric triangle."""

import numpy as np
import pytest

from benchmarks.issue434_fixed_density import (
    coulomb_from_raw,
    coulomb_from_whitened_raw,
)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("lower_delta", [1.0e-12, -1.0e-12, 0.5, 10.0])
def test_whitening_uses_the_direct_solvers_upper_triangle(
    order: str, lower_delta: float
) -> None:
    raw = np.array([[[1.0, 0.2], [0.3, -0.1]], [[0.3, -0.1], [0.7, 0.4]]], order=order)
    density = np.array([[1.2, 0.1], [0.1, 0.8]], order=order)
    metric = np.array([[2.0, 0.25], [0.25, 1.5]], order=order)
    authoritative = metric.copy()
    metric[1, 0] += lower_delta
    before = metric.copy()
    matrix = raw.reshape(4, 2)
    expected = (
        matrix @ np.linalg.solve(authoritative, matrix.T @ density.ravel())
    ).reshape(2, 2)
    np.testing.assert_allclose(
        coulomb_from_raw(raw, metric, density), expected, atol=5e-15, rtol=1e-15
    )
    np.testing.assert_allclose(
        coulomb_from_whitened_raw(raw, metric, density),
        expected,
        atol=5e-15,
        rtol=1e-15,
    )
    np.testing.assert_array_equal(metric, before)


@pytest.mark.parametrize("order", ["C", "F"])
def test_whitening_does_not_ignore_authoritative_triangle_perturbations(
    order: str,
) -> None:
    raw = np.array([[[1.0, 0.2], [0.3, -0.1]], [[0.3, -0.1], [0.7, 0.4]]], order=order)
    density = np.array([[1.2, 0.1], [0.1, 0.8]], order=order)
    metric = np.array([[2.0, 0.25], [0.25, 1.5]], order=order)
    changed = metric.copy(order=order)
    changed[0, 1] += 1.0e-4
    expected = coulomb_from_raw(raw, changed, density) - coulomb_from_raw(
        raw, metric, density
    )
    actual = coulomb_from_whitened_raw(
        raw, changed, density
    ) - coulomb_from_whitened_raw(raw, metric, density)
    assert np.max(np.abs(expected)) > 1.0e-6
    np.testing.assert_allclose(actual, expected, atol=5e-15, rtol=1e-15)

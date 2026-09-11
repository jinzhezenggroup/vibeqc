"""Basis transport is tested against independent Euclidean least squares."""

import numpy as np
import pytest
from vibeqc.projection import (
    ProjectionPolicy,
    ProjectionRejected,
    project_density,
    project_occupied,
)


def spaces():
    """Two non-nested bases embedded in a common orthonormal ambient space."""
    rng = np.random.default_rng(42)
    source = rng.normal(size=(9, 6))
    target = rng.normal(size=(9, 7))
    metric = source.T @ source
    eig, vec = np.linalg.eigh(metric)
    coefficients = (vec / np.sqrt(eig))[:, :2]
    return source, target, coefficients


def test_nonnested_projection_matches_independent_least_squares_and_rotations():
    source, target, coefficients = spaces()
    source_metric, target_metric = source.T @ source, target.T @ target
    cross = target.T @ source
    policy = ProjectionPolicy(maximum_residual=1)
    result = project_occupied(
        source_metric, target_metric, cross, coefficients, policy=policy
    )
    expected = np.linalg.lstsq(
        target, source @ coefficients, rcond=policy.relative_threshold
    )[0]
    gram = expected.T @ target_metric @ expected
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    expected = expected @ (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    np.testing.assert_allclose(result.density, 2 * expected @ expected.T, atol=1e-13)
    np.testing.assert_allclose(
        result.coefficients.T @ target_metric @ result.coefficients,
        np.eye(2),
        atol=1e-13,
    )
    assert result.diagnostics.electron_trace == pytest.approx(4)
    assert result.diagnostics.projection_residual == pytest.approx(
        np.sqrt(1 - eigenvalues[0])
    )
    for rotation in (np.diag([-1, 1]), np.array([[0.6, -0.8], [0.8, 0.6]])):
        rotated = project_occupied(
            source_metric, target_metric, cross, coefficients @ rotation, policy=policy
        )
        np.testing.assert_allclose(rotated.density, result.density, atol=2e-13)
    with pytest.raises(ValueError):
        result.density.setflags(write=True)


@pytest.mark.parametrize("occupation", [1, 2])
def test_same_basis_and_density_reconstruction_preserve_metric_electron_properties(
    occupation,
):
    source, _, coefficients = spaces()
    metric = source.T @ source
    density = occupation * coefficients @ coefficients.T
    result = project_density(
        metric, metric, metric, density, occupied_orbitals=2, occupation=occupation
    )
    np.testing.assert_allclose(result.density, density, atol=1e-13)
    np.testing.assert_allclose(
        result.density @ metric @ result.density,
        occupation * result.density,
        atol=1e-13,
    )
    assert np.trace(result.density @ metric) == pytest.approx(2 * occupation)


def test_conditioning_discards_small_metric_modes_but_does_not_invent_missing_orbitals():
    source_metric = np.eye(2)
    target_metric = np.diag([1, 1, 1e-14])
    cross = np.array([[1, 0], [0, 1], [0, 0]])
    result = project_occupied(source_metric, target_metric, cross, np.eye(2))
    assert result.diagnostics.target_metric_rank == 2
    np.testing.assert_array_equal(result.density, np.diag([2, 2, 0]))
    with pytest.raises(ProjectionRejected, match="insufficient occupied rank"):
        project_occupied(source_metric, np.diag([1, 1e-14]), np.eye(2), np.eye(2))
    with pytest.raises(ProjectionRejected, match="lost rank"):
        project_occupied(source_metric, np.eye(2), np.diag([1, 0]), np.eye(2))


def test_poor_projection_inconsistent_overlap_and_source_states_are_rejected():
    metric = np.eye(2)
    occupied = np.eye(2)[:, :1]
    with pytest.raises(ProjectionRejected, match="projection residual"):
        project_occupied(metric, metric, 0.1 * metric, occupied)
    with pytest.raises(ProjectionRejected, match="inconsistent"):
        project_occupied(metric, metric, 2 * metric, occupied)
    with pytest.raises(ProjectionRejected, match="orthonormal"):
        project_occupied(metric, metric, metric, 2 * occupied)
    for density in (np.diag([1.5, 0.5]), np.diag([2, 2]), np.diag([0, 0])):
        with pytest.raises(ProjectionRejected, match=r"electron count|fractional"):
            project_density(metric, metric, metric, density, occupied_orbitals=1)
    with pytest.raises(ProjectionRejected, match="Hermitian"):
        project_density(metric, metric, metric, [[2, 1], [0, 0]], occupied_orbitals=1)
    with pytest.raises(ProjectionRejected, match="real"):
        project_occupied(metric, metric, metric, occupied.astype(complex))


def test_empty_spin_channel_preserves_zero_electrons():
    result = project_density(
        np.eye(2),
        np.eye(3),
        np.zeros((3, 2)),
        np.zeros((2, 2)),
        occupied_orbitals=0,
        occupation=1,
    )
    assert result.coefficients.shape == (3, 0)
    assert result.diagnostics.electron_trace == 0
    np.testing.assert_array_equal(result.density, np.zeros((3, 3)))

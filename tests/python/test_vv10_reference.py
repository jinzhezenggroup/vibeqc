"""Independent fixed-grid VV10/rVV10 reference gates for #491 slice A."""

import numpy as np
import pytest
from vibeqc_compiler.dft import (
    assemble_nonlocal_potential_reference,
    nonlocal_energy_density_reference,
    nonlocal_energy_reference,
    nonlocal_feature_derivatives_reference,
    nonlocal_kernel_matrix_reference,
)
from vibeqc_compiler.method import original_nonlocal_correlation


@pytest.fixture
def fixed_grid():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.2, -0.1],
            [-0.3, 0.8, 0.5],
            [1.1, -0.4, 0.9],
        ]
    )
    weights = np.array([0.3, 0.5, 0.4, 0.2])
    density = np.array([0.42, 0.31, 0.18, 0.27])
    gradient = np.array(
        [
            [0.05, -0.02, 0.01],
            [-0.03, 0.04, 0.02],
            [0.01, 0.02, -0.04],
            [-0.02, -0.01, 0.03],
        ]
    )
    return coords, weights, density, gradient


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("vv10", 0.002000475442872441),
        ("rvv10", 0.0018147083462731767),
    ],
)
def test_fixed_grid_energy_regression(fixed_grid, variant, expected):
    coords, weights, density, gradient = fixed_grid
    energy = nonlocal_energy_reference(
        coords,
        weights,
        density,
        gradient,
        original_nonlocal_correlation(variant),
        tile_size=2,
    )
    assert energy == pytest.approx(expected, rel=2e-14, abs=1e-16)


def test_reference_kernel_is_pair_symmetric(fixed_grid):
    coords, _, density, gradient = fixed_grid
    kernel = nonlocal_kernel_matrix_reference(
        coords,
        density,
        gradient,
        original_nonlocal_correlation("vv10"),
    )
    np.testing.assert_allclose(kernel, kernel.T, rtol=0.0, atol=1e-16)
    assert np.all(kernel < 0.0)


def test_tiling_and_grid_permutation_do_not_change_energy(fixed_grid):
    coords, weights, density, gradient = fixed_grid
    spec = original_nonlocal_correlation("vv10")
    reference = nonlocal_energy_reference(
        coords, weights, density, gradient, spec, tile_size=1
    )
    for tile_size in (2, 3, 8):
        candidate = nonlocal_energy_reference(
            coords, weights, density, gradient, spec, tile_size=tile_size
        )
        assert candidate == pytest.approx(reference, rel=2e-15, abs=1e-17)

    order = np.array([2, 0, 3, 1])
    permuted = nonlocal_energy_reference(
        coords[order],
        weights[order],
        density[order],
        gradient[order],
        spec,
        tile_size=2,
    )
    assert permuted == pytest.approx(reference, rel=2e-15, abs=1e-17)


def test_energy_is_weighted_density_contraction(fixed_grid):
    coords, weights, density, gradient = fixed_grid
    spec = original_nonlocal_correlation("rvv10")
    eps = nonlocal_energy_density_reference(
        coords, weights, density, gradient, spec, tile_size=2
    )
    energy = nonlocal_energy_reference(
        coords, weights, density, gradient, spec, tile_size=2
    )
    assert energy == pytest.approx(
        float(np.dot(weights * density, eps)), rel=0.0, abs=0.0
    )


def test_reference_oracle_fails_closed_outside_its_domain(fixed_grid):
    coords, weights, density, gradient = fixed_grid
    spec = original_nonlocal_correlation("vv10")
    with pytest.raises(ValueError, match="strictly positive"):
        nonlocal_energy_reference(
            coords, weights, density * np.array([1, 1, 0, 1]), gradient, spec
        )
    with pytest.raises(ValueError, match="positive integer"):
        nonlocal_energy_reference(coords, weights, density, gradient, spec, tile_size=0)
    with pytest.raises(ValueError, match="small fixed grids"):
        nonlocal_kernel_matrix_reference(coords, density, gradient, spec, max_points=3)


def test_feature_derivatives_match_fixed_grid_directional_difference(fixed_grid):
    coords, weights, density, gradient = fixed_grid
    spec = original_nonlocal_correlation("vv10")
    vrho, vsigma = nonlocal_feature_derivatives_reference(
        coords, weights, density, gradient, spec, tile_size=2
    )
    drho = np.array([0.03, -0.02, 0.01, -0.015])
    dgradient = np.array(
        [
            [0.01, 0.02, -0.01],
            [-0.02, 0.01, 0.015],
            [0.005, -0.01, 0.02],
            [0.01, 0.005, -0.015],
        ]
    )
    dsigma = 2.0 * np.einsum("pi,pi->p", gradient, dgradient)
    predicted = float(np.dot(weights, vrho * drho + vsigma * dsigma))

    errors = []
    for step in (2e-4, 1e-4, 5e-5):
        plus = nonlocal_energy_reference(
            coords,
            weights,
            density + step * drho,
            gradient + step * dgradient,
            spec,
            tile_size=2,
        )
        minus = nonlocal_energy_reference(
            coords,
            weights,
            density - step * drho,
            gradient - step * dgradient,
            spec,
            tile_size=2,
        )
        finite_difference = (plus - minus) / (2.0 * step)
        errors.append(abs(finite_difference - predicted))
    assert max(errors) < 2e-10


def _features_from_total_density(jets, density):
    phi = jets[0]
    weighted = phi @ density
    rho = np.sum(phi * weighted, axis=1)
    gradient = np.stack(
        [2.0 * np.sum(derivative * weighted, axis=1) for derivative in jets[1:4]],
        axis=1,
    )
    return rho, gradient


def test_ao_potential_is_derivative_of_same_nonlocal_energy():
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.6, 0.1, -0.2], [-0.4, 0.7, 0.3], [0.9, -0.5, 0.8]]
    )
    weights = np.array([0.4, 0.3, 0.5, 0.2])
    jets = np.array(
        [
            [[1.0, 0.2], [0.8, -0.1], [0.6, 0.3], [1.1, 0.05]],
            [[0.10, -0.04], [-0.03, 0.08], [0.05, 0.02], [-0.06, 0.04]],
            [[-0.02, 0.07], [0.06, -0.01], [-0.04, 0.03], [0.02, 0.05]],
            [[0.03, 0.01], [-0.05, 0.02], [0.07, -0.02], [0.01, -0.04]],
        ]
    )
    density_matrix = np.array([[0.9, 0.1], [0.1, 0.7]])
    direction = np.array([[0.04, -0.015], [-0.015, -0.03]])
    spec = original_nonlocal_correlation("rvv10")

    rho, gradient = _features_from_total_density(jets, density_matrix)
    vrho, vsigma = nonlocal_feature_derivatives_reference(
        coords, weights, rho, gradient, spec, tile_size=2
    )
    potential = assemble_nonlocal_potential_reference(
        jets, weights, gradient, vrho, vsigma
    )
    predicted = float(np.sum(potential * direction))

    errors = []
    for step in (2e-4, 1e-4, 5e-5):
        plus_rho, plus_gradient = _features_from_total_density(
            jets, density_matrix + step * direction
        )
        minus_rho, minus_gradient = _features_from_total_density(
            jets, density_matrix - step * direction
        )
        plus = nonlocal_energy_reference(
            coords, weights, plus_rho, plus_gradient, spec, tile_size=2
        )
        minus = nonlocal_energy_reference(
            coords, weights, minus_rho, minus_gradient, spec, tile_size=2
        )
        finite_difference = (plus - minus) / (2.0 * step)
        errors.append(abs(finite_difference - predicted))
    assert max(errors) < 2e-10

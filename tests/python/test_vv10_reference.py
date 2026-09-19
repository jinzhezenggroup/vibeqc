"""Independent fixed-grid VV10/rVV10 reference gates for #491 slice A."""

import numpy as np
import pytest
from vibeqc_compiler.dft import (
    nonlocal_energy_density_reference,
    nonlocal_energy_reference,
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

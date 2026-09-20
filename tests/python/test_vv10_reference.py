"""Independent fixed-grid VV10/rVV10 reference gates for #491 slice A."""

import typing

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
def fixed_grid() -> typing.Any:
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
    ],
)
def test_fixed_grid_energy_regression(
    fixed_grid: typing.Any, variant: typing.Any, expected: typing.Any
) -> None:
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


def test_reference_kernel_is_pair_symmetric(fixed_grid: typing.Any) -> None:
    coords, _, density, gradient = fixed_grid
    kernel = nonlocal_kernel_matrix_reference(
        coords,
        density,
        gradient,
        original_nonlocal_correlation("vv10"),
    )
    np.testing.assert_allclose(kernel, kernel.T, rtol=0.0, atol=1e-16)
    assert np.all(kernel < 0.0)


def test_tiling_and_grid_permutation_do_not_change_energy(
    fixed_grid: typing.Any,
) -> None:
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


def test_energy_is_weighted_density_contraction(fixed_grid: typing.Any) -> None:
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


def test_reference_oracle_fails_closed_outside_its_domain(
    fixed_grid: typing.Any,
) -> None:
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


@pytest.mark.parametrize("variant", ["vv10", "rvv10"])
def test_feature_derivatives_match_fixed_grid_directional_difference(
    fixed_grid: typing.Any, variant: typing.Any
) -> None:
    coords, weights, density, gradient = fixed_grid
    spec = original_nonlocal_correlation(variant)
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


def _features_from_total_density(jets: typing.Any, density: typing.Any) -> typing.Any:
    phi = jets[0]
    weighted = phi @ density
    rho = np.sum(phi * weighted, axis=1)
    gradient = np.stack(
        [2.0 * np.sum(derivative * weighted, axis=1) for derivative in jets[1:4]],
        axis=1,
    )
    return rho, gradient


@pytest.mark.parametrize("variant", ["vv10", "rvv10"])
def test_ao_potential_is_derivative_of_same_nonlocal_energy(
    variant: typing.Any,
) -> None:
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
    spec = original_nonlocal_correlation(variant)

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


@pytest.mark.parametrize("variant", ["vv10", "rvv10"])
@pytest.mark.parametrize("field", [0, 1, 2, 3])
def test_reference_rejects_complex_input_before_casting(
    variant: typing.Any, field: typing.Any
) -> None:
    from vibeqc_compiler.dft.nonlocal_reference import nonlocal_energy_reference

    args = [np.zeros((2, 3)), np.ones(2), np.array([0.1, 0.7]), np.zeros((2, 3))]
    args[field] = args[field].astype(complex) + 0.01j
    with pytest.raises(ValueError, match="real"):
        nonlocal_energy_reference(*args, original_nonlocal_correlation(variant))


@pytest.mark.parametrize("variant", ["vv10", "rvv10"])
def test_reference_rejects_nonfinite_intermediates(variant: typing.Any) -> None:
    with pytest.raises(
        (ValueError, FloatingPointError), match="finite|overflow|invalid|divide"
    ):
        nonlocal_energy_reference(
            np.zeros((2, 3)),
            np.ones(2),
            np.full(2, 1e-200),
            np.ones((2, 3)),
            original_nonlocal_correlation(variant),
        )


def test_rvv10_differs_from_reparameterized_vv10_for_unequal_densities() -> None:
    from dataclasses import replace

    spec = original_nonlocal_correlation("rvv10")
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.3, 0.0]])
    density = np.array([0.02, 0.9])
    gradient = np.array([[0.01, 0.02, 0.0], [0.07, -0.1, 0.2]])
    actual = nonlocal_kernel_matrix_reference(coords, density, gradient, spec)
    old = nonlocal_kernel_matrix_reference(
        coords, density, gradient, replace(spec, variant="vv10")
    )
    assert abs(actual[0, 1] - old[0, 1]) > 1e-7


def _published_rvv10_decimal(
    coords: typing.Any,
    weights: typing.Any,
    density: typing.Any,
    gradient: typing.Any,
    spec: typing.Any,
) -> typing.Any:
    """Independent scalar Decimal form of PRB 87, 041108 Eqs. (4)-(6)."""
    from decimal import Decimal as Dec
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 60
        pi = Dec("3.14159265358979323846264338327950288419716939937510582097494")
        rho = [Dec(str(x)) for x in density]
        xyz = [[Dec(str(x)) for x in row] for row in coords]
        grad = [[Dec(str(x)) for x in row] for row in gradient]
        w = [Dec(str(x)) for x in weights]
        b = Dec(spec.b.numerator) / Dec(spec.b.denominator)
        c = Dec(spec.c.numerator) / Dec(spec.c.denominator)
        k = [Dec("1.5") * b * pi * (r / (9 * pi)) ** (Dec(1) / 6) for r in rho]
        omega = [
            (c * sum(x * x for x in g) ** 2 / r**4 + 4 * pi * r / 3).sqrt()
            for r, g in zip(rho, grad)
        ]
        beta = (3 / b**2) ** Dec("0.75") / 32
        energy = beta * sum(wi * r for wi, r in zip(w, rho))
        for i in range(len(rho)):
            for j in range(len(rho)):
                r2 = sum((a - bb) ** 2 for a, bb in zip(xyz[i], xyz[j]))
                z, zp = 1 + omega[i] / k[i] * r2, 1 + omega[j] / k[j] * r2
                theta = rho[i] / k[i] ** Dec("1.5")
                theta_p = rho[j] / k[j] ** Dec("1.5")
                phi = -Dec("1.5") / (z * zp * (z + zp))
                energy += w[i] * w[j] * theta * theta_p * phi / 2
        return float(energy)


@pytest.mark.parametrize("tile_size", [1, 2, 7])
def test_rvv10_matches_published_density_rescaled_kernel(
    fixed_grid: typing.Any, tile_size: typing.Any
) -> None:
    spec = original_nonlocal_correlation("rvv10")
    expected = _published_rvv10_decimal(*fixed_grid, spec)
    actual = nonlocal_energy_reference(*fixed_grid, spec, tile_size=tile_size)
    assert actual == pytest.approx(expected, rel=2e-14, abs=1e-16)


def test_historical_reparameterized_vv10_value_is_not_rvv10(
    fixed_grid: typing.Any,
) -> None:
    from dataclasses import replace

    spec = replace(original_nonlocal_correlation("rvv10"), variant="vv10")
    assert nonlocal_energy_reference(*fixed_grid, spec) == pytest.approx(
        0.0018147083462731767, rel=2e-14, abs=1e-16
    )
    revised = nonlocal_energy_reference(
        *fixed_grid, original_nonlocal_correlation("rvv10")
    )
    assert abs(revised - 0.0018147083462731767) > 1e-9

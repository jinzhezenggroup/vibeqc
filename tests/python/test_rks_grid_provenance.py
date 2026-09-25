"""Dimensionless native-grid provenance remains meaningful at radial tails."""

import math

import numpy as np
import pytest
from vibeqc_compiler.xc.grid_response import partition_response

from tools.vibeqc_hessian.rks_directional import _validate_partition_provenance


@pytest.mark.parametrize("scale", [1.0, 306184.2636279619, 1e100])
def test_one_ulp_partition_difference_does_not_scale_with_atomic_measure(
    scale: float,
) -> None:
    raw = np.array([scale])
    fraction = np.array([2e-7])
    # Native and generated distance/polynomial arithmetic differ near an endpoint.
    perturbed = fraction + np.finfo(np.float64).eps
    _validate_partition_provenance(raw, raw * fraction, perturbed)


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_wrong_partition_is_rejected_at_every_measure_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="provenance"):
        _validate_partition_provenance([scale], [0.2 * scale], [0.3])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_invalid_atomic_measure_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        _validate_partition_provenance([bad], [0.0], [0.5])


def test_zero_measure_requires_exact_zero_weight() -> None:
    _validate_partition_provenance([0.0, 2.0], [0.0, 1.0], [0.3, 0.5])
    with pytest.raises(ValueError, match="zero atomic measure"):
        _validate_partition_provenance([0.0], [1e-100], [0.5])


@pytest.mark.parametrize(
    "weights,fractions", [([np.inf], [0.5]), ([1.0], [np.nan]), ([1.0], [1.1])]
)
def test_invalid_weight_or_fraction_is_rejected(
    weights: list[float], fractions: list[float]
) -> None:
    with pytest.raises(ValueError):
        _validate_partition_provenance([2.0], weights, fractions)


def test_h2_grid_matches_independent_scalar_partition_without_mass_amplification() -> (
    None
):
    centers = np.array([[0.0, 0.0, -0.72], [0.08, -0.03, 0.71]])
    radial, radial_weights = np.polynomial.legendre.leggauss(10)
    polar, polar_weights = np.polynomial.legendre.leggauss(4)
    points, raw, weights, owners = [], [], [], []

    def distance(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return math.hypot(math.hypot(d[0], d[1]), d[2])

    for owner in range(2):
        for node, wrule in zip(radial, radial_weights, strict=True):
            t = 0.5 * (node + 1.0)
            radius = t / (1.0 - t)
            wr = 0.5 * wrule * radius * radius / ((1.0 - t) * (1.0 - t))
            for z, wz in zip(polar, polar_weights, strict=True):
                ring = math.sqrt(max(0.0, 1.0 - z * z))
                for azimuth in range(8):
                    phi = 2.0 * math.pi * azimuth / 8
                    point = centers[owner] + [
                        radius * ring * math.cos(phi),
                        radius * ring * math.sin(phi),
                        radius * z,
                    ]
                    mu = np.clip(
                        (distance(point, centers[1]) - distance(point, centers[0]))
                        / distance(centers[1], centers[0]),
                        -1.0,
                        1.0,
                    )
                    for _ in range(3):
                        mu = 0.5 * mu * (3.0 - mu * mu)
                    pair = np.clip(0.5 * (1.0 - mu), 0.0, 1.0)
                    fraction = pair if owner else 1.0 - pair
                    measure = wr * wz * (2.0 * math.pi / 8)
                    points.append(point)
                    raw.append(measure)
                    weights.append(measure * fraction)
                    owners.append(owner)
    points = np.asarray(points)
    owners = np.asarray(owners)
    raw, weights = np.asarray(raw), np.asarray(weights)
    direction = np.array([[0.17, -0.09, 0.31], [-0.13, 0.07, -0.26]])
    response = partition_response(
        points,
        centers,
        point_motion=direction[owners],
        center_motion=direction,
    )
    fractions = response.weights[np.arange(len(points)), owners]
    # The oracle is a separate scalar hypot/Becke implementation, not response AD.
    assert np.max(np.abs(fractions - weights / raw)) < 2e-14
    _validate_partition_provenance(raw, weights, fractions)

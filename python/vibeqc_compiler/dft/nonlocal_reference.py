"""Small fixed-grid VV10/rVV10 CPU oracle for independent validation (#491).

This module is deliberately a reference evaluator, not a production runtime.
It performs the published real-space double sum in FP64 and tiles only to keep
the diagnostic memory bound explicit.
"""

from __future__ import annotations

import math

import numpy as np

from vibeqc_compiler.common.nonlocal_correlation import NonlocalCorrelationSpec


def _validated_arrays(coords, weights, density, gradient):
    coords = np.asarray(coords, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates require shape (n_grid, 3)")
    ngrid = coords.shape[0]
    if weights.shape != (ngrid,) or density.shape != (ngrid,):
        raise ValueError("weights and density require shape (n_grid,)")
    if gradient.shape != (ngrid, 3):
        raise ValueError("density gradient requires shape (n_grid, 3)")
    for name, array in (
        ("coordinates", coords),
        ("weights", weights),
        ("density", density),
        ("density gradient", gradient),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    if np.any(density <= 0):
        raise ValueError(
            "reference VV10 oracle requires strictly positive grid density"
        )
    return coords, weights, density, gradient


def _local_scales(density, gradient, spec):
    if not isinstance(spec, NonlocalCorrelationSpec):
        raise TypeError("spec requires NonlocalCorrelationSpec")
    b = float(spec.b)
    c = float(spec.c)
    sigma = np.einsum("pi,pi->p", gradient, gradient)
    omega = np.sqrt(
        c * np.square(sigma / np.square(density)) + (4.0 * math.pi / 3.0) * density
    )
    kappa = b * 1.5 * math.pi * np.power(density / (9.0 * math.pi), 1.0 / 6.0)
    beta = np.power(3.0 / (b * b), 0.75) / 32.0
    return omega, kappa, float(beta)


def nonlocal_kernel_matrix_reference(
    coords, density, gradient, spec, *, max_points=512
):
    """Materialize the small-grid pair kernel for symmetry diagnostics only."""
    coords = np.asarray(coords, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    ngrid = coords.shape[0] if coords.ndim else 0
    if ngrid > max_points:
        raise ValueError("kernel-matrix reference is limited to small fixed grids")
    _, _, density, gradient = _validated_arrays(
        coords, np.ones(ngrid), density, gradient
    )
    omega, kappa, _ = _local_scales(density, gradient, spec)
    delta = coords[:, None, :] - coords[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", delta, delta)
    gi = omega[:, None] * r2 + kappa[:, None]
    gj = omega[None, :] * r2 + kappa[None, :]
    return -1.5 / (gi * gj * (gi + gj))


def nonlocal_energy_density_reference(
    coords,
    weights,
    density,
    gradient,
    spec,
    *,
    tile_size=256,
):
    """Return VV10-family nonlocal correlation energy per electron on the grid."""
    coords, weights, density, gradient = _validated_arrays(
        coords, weights, density, gradient
    )
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    omega, kappa, beta = _local_scales(density, gradient, spec)
    eps = np.empty_like(density)
    weighted_density = weights * density
    ngrid = density.size

    for i in range(ngrid):
        total = 0.0
        for start in range(0, ngrid, tile_size):
            stop = min(start + tile_size, ngrid)
            delta = coords[start:stop] - coords[i]
            r2 = np.einsum("pi,pi->p", delta, delta)
            gi = omega[i] * r2 + kappa[i]
            gj = omega[start:stop] * r2 + kappa[start:stop]
            kernel = -1.5 / (gi * gj * (gi + gj))
            total += float(
                np.sum(
                    weighted_density[start:stop] * kernel,
                    dtype=np.float64,
                )
            )
        eps[i] = beta + 0.5 * total
    return eps


def nonlocal_energy_reference(
    coords,
    weights,
    density,
    gradient,
    spec,
    *,
    tile_size=256,
):
    """Evaluate the fixed-grid VV10/rVV10 nonlocal correlation energy in Eh."""
    _, weights, density, _ = _validated_arrays(coords, weights, density, gradient)
    eps = nonlocal_energy_density_reference(
        coords,
        weights,
        density,
        gradient,
        spec,
        tile_size=tile_size,
    )
    return float(np.dot(weights * density, eps))

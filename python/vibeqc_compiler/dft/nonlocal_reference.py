"""Small fixed-grid VV10/rVV10 CPU oracle for independent validation (#491).

This module is deliberately a reference evaluator, not a production runtime.
It performs the published real-space double sum in FP64 and tiles only to keep
the diagnostic memory bound explicit.
"""

from __future__ import annotations

import math
from functools import wraps

import numpy as np

from vibeqc_compiler.common.nonlocal_correlation import NonlocalCorrelationSpec


def _real_array(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    return np.asarray(raw, dtype=np.float64)


def _finite_arithmetic(function):
    @wraps(function)
    def checked(*args, **kwargs):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = function(*args, **kwargs)
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite nonlocal-correlation result")
        return result

    return checked


def _validated_arrays(coords, weights, density, gradient):
    coords = _real_array(coords, "coordinates")
    weights = _real_array(weights, "weights")
    density = _real_array(density, "density")
    gradient = _real_array(gradient, "density gradient")
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
    if not math.isfinite(b) or not math.isfinite(c) or b <= 0 or c <= 0:
        raise ValueError("nonlocal parameters must be finite positive FP64")
    sigma = np.einsum("pi,pi->p", gradient, gradient)
    omega = np.sqrt(
        c * np.square(sigma / np.square(density)) + (4.0 * math.pi / 3.0) * density
    )
    kappa = b * 1.5 * math.pi * np.power(density / (9.0 * math.pi), 1.0 / 6.0)
    beta = np.power(3.0 / (b * b), 0.75) / 32.0
    if (
        not np.isfinite(omega).all()
        or not np.isfinite(kappa).all()
        or np.any(kappa <= 0)
        or not np.isfinite(beta)
    ):
        raise FloatingPointError(
            "nonfinite or invalid local nonlocal-correlation scales"
        )
    return omega, kappa, float(beta)


def _pair_kernel(r2, wi, wj, ki, kj, spec):
    if spec.variant == "rvv10":
        # Sabatini et al., PRB 87, 041108 (2013), Eqs. (4)-(6).
        # Restore the kappa**(-3/2) factors to the ordinary density measure.
        zi, zj = wi / ki * r2 + 1, wj / kj * r2 + 1
        return -1.5 / ((ki * kj) ** 1.5 * zi * zj * (zi + zj))
    gi, gj = wi * r2 + ki, wj * r2 + kj
    return -1.5 / (gi * gj * (gi + gj))


@_finite_arithmetic
def nonlocal_kernel_matrix_reference(
    coords, density, gradient, spec, *, max_points=512
):
    """Materialize the small-grid pair kernel for symmetry diagnostics only."""
    coords = _real_array(coords, "coordinates")
    density = _real_array(density, "density")
    gradient = _real_array(gradient, "density gradient")
    ngrid = coords.shape[0] if coords.ndim else 0
    if ngrid > max_points:
        raise ValueError("kernel-matrix reference is limited to small fixed grids")
    _, _, density, gradient = _validated_arrays(
        coords, np.ones(ngrid), density, gradient
    )
    omega, kappa, _ = _local_scales(density, gradient, spec)
    delta = coords[:, None, :] - coords[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", delta, delta)
    return _pair_kernel(
        r2, omega[:, None], omega[None, :], kappa[:, None], kappa[None, :], spec
    )


@_finite_arithmetic
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
            kernel = _pair_kernel(
                r2, omega[i], omega[start:stop], kappa[i], kappa[start:stop], spec
            )
            total += float(
                np.sum(
                    weighted_density[start:stop] * kernel,
                    dtype=np.float64,
                )
            )
        eps[i] = beta + 0.5 * total
    return eps


@_finite_arithmetic
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


@_finite_arithmetic
def nonlocal_feature_derivatives_reference(
    coords,
    weights,
    density,
    gradient,
    spec,
    *,
    tile_size=256,
):
    """Return fixed-grid derivatives dE/d(rho,sigma) before quadrature weights."""
    coords, weights, density, gradient = _validated_arrays(
        coords, weights, density, gradient
    )
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    omega, kappa, beta = _local_scales(density, gradient, spec)
    sigma = np.einsum("pi,pi->p", gradient, gradient)
    c = float(spec.c)
    domega_drho = (
        (4.0 * math.pi / 3.0) - 4.0 * c * np.square(sigma) / np.power(density, 5)
    ) / (2.0 * omega)
    domega_dsigma = c * sigma / (omega * np.power(density, 4))
    dkappa_drho = kappa / (6.0 * density)

    vrho = np.empty_like(density)
    vsigma = np.empty_like(density)
    weighted_density = weights * density
    ngrid = density.size

    for i in range(ngrid):
        sum_phi = 0.0
        sum_rho = 0.0
        sum_sigma = 0.0
        for start in range(0, ngrid, tile_size):
            stop = min(start + tile_size, ngrid)
            delta = coords[start:stop] - coords[i]
            r2 = np.einsum("pi,pi->p", delta, delta)
            phi = _pair_kernel(
                r2, omega[i], omega[start:stop], kappa[i], kappa[start:stop], spec
            )
            if spec.variant == "rvv10":
                zi = 1 + omega[i] / kappa[i] * r2
                zj = 1 + omega[start:stop] / kappa[start:stop] * r2
                factor_z = 1 / zi + 1 / (zi + zj)
                dphi_domega = -phi * r2 / kappa[i] * factor_z
                dphi_dkappa = phi / kappa[i] * (-1.5 + (zi - 1) * factor_z)
            else:
                gi = omega[i] * r2 + kappa[i]
                gj = omega[start:stop] * r2 + kappa[start:stop]
                dphi_dgi = -phi * (1 / gi + 1 / (gi + gj))
                dphi_domega, dphi_dkappa = dphi_dgi * r2, dphi_dgi
            dphi_drho = dphi_domega * domega_drho[i] + dphi_dkappa * dkappa_drho[i]
            dphi_dsigma = dphi_domega * domega_dsigma[i]
            factor = weighted_density[start:stop]
            sum_phi += float(np.sum(factor * phi, dtype=np.float64))
            sum_rho += float(np.sum(factor * dphi_drho, dtype=np.float64))
            sum_sigma += float(np.sum(factor * dphi_dsigma, dtype=np.float64))
        vrho[i] = beta + sum_phi + density[i] * sum_rho
        vsigma[i] = density[i] * sum_sigma
    return vrho, vsigma


@_finite_arithmetic
def assemble_nonlocal_potential_reference(
    jets,
    weights,
    density_gradient,
    vrho,
    vsigma,
):
    """Assemble the total-density AO matrix for a fixed-grid VV10 potential."""
    jets = _real_array(jets, "jets")
    weights = _real_array(weights, "weights")
    density_gradient = _real_array(density_gradient, "density_gradient")
    vrho = _real_array(vrho, "vrho")
    vsigma = _real_array(vsigma, "vsigma")
    if jets.ndim != 3 or jets.shape[0] < 4:
        raise ValueError("nonlocal potential requires first-order AO jets")
    ngrid = jets.shape[1]
    if (
        weights.shape != (ngrid,)
        or density_gradient.shape != (ngrid, 3)
        or vrho.shape != (ngrid,)
        or vsigma.shape != (ngrid,)
    ):
        raise ValueError("nonlocal potential point shapes do not match")
    if not all(
        np.all(np.isfinite(array))
        for array in (jets, weights, density_gradient, vrho, vsigma)
    ):
        raise ValueError("nonlocal potential inputs must be finite")

    phi = jets[0]
    derivatives = jets[1:4]
    matrix = phi.T @ ((weights * vrho)[:, None] * phi)
    spatial = 2.0 * vsigma[:, None] * density_gradient
    panel = sum(spatial[:, k, None] * derivatives[k] for k in range(3))
    cross = phi.T @ (weights[:, None] * panel)
    matrix += cross + cross.T
    return 0.5 * (matrix + matrix.T)

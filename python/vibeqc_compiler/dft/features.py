"""Spin-resolved density invariants for arbitrary supplied real symmetric D."""

from __future__ import annotations

import numpy as np

from vibeqc_compiler.common.arrays import immutable


def spin_densities(density, nao):
    """Return [alpha,beta,AO,AO]; RHF input is total D and splits equally.

    D need not be an SCF solution or positive semidefinite. Negative diagnostic
    densities are preserved, never clipped/renormalized. Complex and asymmetric
    matrices fail explicitly. The caller controls electron occupations.
    """
    d = immutable(density)
    if d.shape == (nao, nao):
        d = np.stack((0.5 * d, 0.5 * d))
    elif d.shape != (2, nao, nao):
        raise ValueError("density must be total RHF (AO,AO) or (alpha/beta,AO,AO)")
    if not np.allclose(d, d.swapaxes(1, 2), atol=1e-12, rtol=1e-10):
        raise ValueError("density matrices must be symmetric")
    return immutable(0.5 * (d + d.swapaxes(1, 2)))


def _publish(rho, gradient, tau):
    sigma = np.stack(
        (
            np.sum(gradient[0] ** 2, axis=1),
            np.sum(gradient[0] * gradient[1], axis=1),
            np.sum(gradient[1] ** 2, axis=1),
        )
    )
    return {
        "rho": immutable(rho),
        "gradient": immutable(gradient),
        "sigma": immutable(sigma),
        "tau": immutable(tau),
    }


def density_features(jets, density, *, ingredients=None):
    """Contract rho, grad(rho), sigma(aa,ab,bb), tau=1/2 sum D gradχ·gradχ.

    rho/tau have shape [spin,point], gradient [spin,point,xyz], and sigma
    [aa/ab/bb,point]. The cross-spin sigma_ab carries no extra factor of two.
    This CPU reference uses matrix contractions and includes all active AOs.

    ``ingredients`` prunes unneeded reductions for semilocal consumers. A
    rho-only request accepts value-only jets; omitting tau avoids its three
    additional density-matrix products per spin. The default preserves the
    full diagnostic feature ABI.
    """
    requested = (
        ("rho", "gradient", "sigma", "tau")
        if ingredients is None
        else tuple(ingredients)
    )
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(k not in ("rho", "gradient", "sigma", "tau") for k in requested)
    ):
        raise ValueError("unsupported or duplicate density ingredient")
    need_gradient = "gradient" in requested or "sigma" in requested
    need_first = need_gradient or "tau" in requested
    jets = np.asarray(jets)
    if jets.ndim != 3 or jets.shape[0] not in (
        (4, 10, 20) if need_first else (1, 4, 10, 20)
    ):
        raise ValueError("density features require the requested AO derivative domain")
    if np.iscomplexobj(jets) or not np.isfinite(jets).all():
        raise ValueError("AO jets must be finite and real")
    d = spin_densities(density, jets.shape[2])
    value, derivatives = jets[0], jets[1:4]
    rho, gradient, tau = [], [], []
    for spin in d:
        w = value @ spin
        if "rho" in requested:
            rho.append(np.sum(value * w, axis=1))
        if need_gradient:
            gradient.append(
                np.stack(
                    [2 * np.sum(derivative * w, axis=1) for derivative in derivatives],
                    axis=-1,
                )
            )
        if "tau" in requested:
            tau.append(
                0.5
                * sum(
                    np.sum((derivative @ spin) * derivative, axis=1)
                    for derivative in derivatives
                )
            )
    values = {
        "rho": np.asarray(rho),
        "gradient": np.asarray(gradient),
        "tau": np.asarray(tau),
    }
    if "sigma" in requested:
        values["sigma"] = np.stack(
            [
                np.sum(values["gradient"][a] * values["gradient"][b], axis=1)
                for a, b in ((0, 0), (0, 1), (1, 1))
            ]
        )
    return {key: immutable(values[key]) for key in requested}


def orbital_features(jets, coefficients, occupations):
    """Independent occupied-orbital summation, with per-spin occupations.

    Coefficients are [spin,AO,orbital]; fractional/non-SCF orbital populations
    are legal. Unlike density_features this route never constructs D.
    """
    jets, c, occ = immutable(jets), immutable(coefficients), immutable(occupations)
    if (
        jets.ndim != 3
        or jets.shape[0] not in (4, 10, 20)
        or c.ndim != 3
        or c.shape[0] != 2
        or c.shape[1] != jets.shape[2]
        or occ.shape != (2, c.shape[2])
    ):
        raise ValueError("invalid spin orbital feature shapes")
    if np.any(occ < 0):
        raise ValueError("orbital occupations must be nonnegative")
    rho, gradient, tau = [], [], []
    for spin in range(2):
        orbital = np.einsum("jpm,mi->jpi", jets[:4], c[spin])
        rho.append(np.sum(orbital[0] ** 2 * occ[spin], axis=1))
        gradient.append(
            np.stack(
                [
                    np.sum(2 * orbital[0] * orbital[k] * occ[spin], axis=1)
                    for k in range(1, 4)
                ],
                axis=-1,
            )
        )
        tau.append(0.5 * np.sum(orbital[1:4] ** 2 * occ[spin], axis=(0, 2)))
    return _publish(np.asarray(rho), np.asarray(gradient), np.asarray(tau))

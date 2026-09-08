"""Spin-resolved density invariants for arbitrary supplied real symmetric D."""

from __future__ import annotations

import numpy as np

from tools.vibeqc_posthf.reference import immutable


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


def density_features(jets, density):
    """Contract rho, grad(rho), sigma(aa,ab,bb), tau=1/2 sum D gradχ·gradχ.

    rho/tau have shape [spin,point], gradient [spin,point,xyz], and sigma
    [aa/ab/bb,point]. The cross-spin sigma_ab carries no extra factor of two.
    This CPU reference uses matrix contractions and includes all active AOs.
    """
    jets = np.asarray(jets)
    if jets.ndim != 3 or jets.shape[0] not in (4, 10, 20):
        raise ValueError("density features require AO values and first derivatives")
    if np.iscomplexobj(jets) or not np.isfinite(jets).all():
        raise ValueError("AO jets must be finite and real")
    d = spin_densities(density, jets.shape[2])
    value, derivatives = jets[0], jets[1:4]
    rho, gradient, tau = [], [], []
    for spin in d:
        w = value @ spin
        rho.append(np.sum(value * w, axis=1))
        gradient.append(
            np.stack(
                [2 * np.sum(derivative * w, axis=1) for derivative in derivatives],
                axis=-1,
            )
        )
        tau.append(
            0.5
            * sum(
                np.sum((derivative @ spin) * derivative, axis=1)
                for derivative in derivatives
            )
        )
    return _publish(np.asarray(rho), np.asarray(gradient), np.asarray(tau))


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

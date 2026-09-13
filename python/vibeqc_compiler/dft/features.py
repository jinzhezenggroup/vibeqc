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


def requested_ingredients(ingredients=None):
    """Validate the common CPU/CUDA feature request, preserving output order."""
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
    return requested


def _feature_request(jets, ingredients):
    """Share the requested jet domain, independently of the contraction route."""
    requested = requested_ingredients(ingredients)
    need_gradient = "gradient" in requested or "sigma" in requested
    need_first = need_gradient or "tau" in requested
    jets = np.asarray(jets)
    if jets.ndim != 3 or jets.shape[0] not in (
        (4, 10, 20) if need_first else (1, 4, 10, 20)
    ):
        raise ValueError("density features require the requested AO derivative domain")
    if np.iscomplexobj(jets) or not np.isfinite(jets).all():
        raise ValueError("AO jets must be finite and real")
    return jets, requested, need_gradient


def _publish(requested, rho, gradient, tau):
    """Build nonlinear sigma only after each complete spin gradient is reduced."""
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
    jets, requested, need_gradient = _feature_request(jets, ingredients)
    d = spin_densities(density, jets.shape[2])
    value, derivatives = jets[0], jets[1:4]
    rho, gradient, tau = [], [], []
    for spin in d:
        if "rho" in requested or need_gradient:
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
    return _publish(requested, rho, gradient, tau)


def _spin_orbitals(coefficients, occupations, nao):
    """Own two real C/f blocks; spin channels may have different orbital counts."""
    if len(coefficients) != 2 or len(occupations) != 2:
        raise ValueError("orbitals require two spin blocks")
    c, occ = tuple(map(immutable, coefficients)), tuple(map(immutable, occupations))
    for cs, fs in zip(c, occ, strict=True):
        if cs.ndim != 2 or cs.shape[0] != nao or fs.shape != (cs.shape[1],):
            raise ValueError("invalid spin orbital feature shapes")
        if np.any(fs < 0):
            raise ValueError("orbital occupations must be nonnegative")
    return c, occ


def orbital_features(jets, coefficients, occupations, *, ingredients=None):
    """Independent occupied-orbital summation, with per-spin occupations.

    Coefficients are [spin,AO,orbital]; fractional/non-SCF orbital populations
    are legal. Alternatively supply two (AO,norb_spin) arrays and two occupation
    vectors, including an (AO,0) empty channel. Occupations are per spin even
    for closed shells; total occupations must be divided between the spins.
    This route never constructs D or discards any supplied orbital column.
    ``ingredients`` has the same output-pruning contract as density_features.
    """
    jets, requested, need_gradient = _feature_request(jets, ingredients)
    c, occ = _spin_orbitals(coefficients, occupations, jets.shape[2])
    rho, gradient, tau = [], [], []
    for spin in range(2):
        # Weight before collocation: a zero occupation must remain zero even
        # when squaring an unweighted coefficient would overflow. This also
        # makes the route explicitly Psi = Phi (C sqrt(f)).
        factor = c[spin] * np.sqrt(occ[spin])
        if "rho" in requested or need_gradient:
            value = jets[0] @ factor
        if "rho" in requested:
            rho.append(np.sum(value**2, axis=1))
        if need_gradient or "tau" in requested:
            derivatives = jets[1:4] @ factor
        if need_gradient:
            gradient.append(
                np.stack(
                    [
                        np.sum(2 * value * derivative, axis=1)
                        for derivative in derivatives
                    ],
                    axis=-1,
                )
            )
        if "tau" in requested:
            tau.append(0.5 * np.sum(derivatives**2, axis=(0, 2)))
    return _publish(requested, rho, gradient, tau)

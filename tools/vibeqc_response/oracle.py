"""Independent explicit-matrix and finite-rotation response oracles."""

from __future__ import annotations

import numpy as np

from tools.vibeqc_posthf.conventions import MOBlock
from tools.vibeqc_posthf.reference import immutable


def explicit_rhf_response_matrix(problem, provider):
    """Assemble the tiny MO response matrix from explicit chemists' ERIs.

    For the density-response parameterization in :class:`RotationLayout`:

    ``A[(i,a),(j,b)] = (eps_a-eps_i) delta_ij delta_ab
                      + 4(ai|bj) - (ab|ij) - (aj|ib)``.

    The provider must belong to the exact reference snapshot.  This routine is
    intentionally dense and tiny-system-only; it is the independent algebraic
    oracle for the matrix-free JVP.
    """
    if provider.snapshot.identity != problem.reference.identity:
        raise ValueError("explicit response provider/reference mismatch")
    layout = problem.layout
    o = layout.occupied
    v = layout.virtual
    g_ovov = provider.get(MOBlock((v, o, v, o))).to_host()
    g_vvoo = provider.get(MOBlock((v, v, o, o))).to_host()
    g_voov = provider.get(MOBlock((v, o, o, v))).to_host()
    eps = problem.reference.orbital_energies
    nocc, nvirt = layout.nocc, layout.nvirt
    matrix = np.zeros((nocc, nvirt, nocc, nvirt))
    for i in range(nocc):
        for a in range(nvirt):
            for j in range(nocc):
                for b in range(nvirt):
                    matrix[i, a, j, b] = (
                        (eps[v[a]] - eps[o[i]]) * (i == j) * (a == b)
                        + 4.0 * g_ovov[a, i, b, j]
                        - g_vvoo[a, b, i, j]
                        - g_voov[a, j, i, b]
                    )
    return immutable(matrix.reshape(layout.dimension, layout.dimension))


def _expm_small(matrix, *, terms=18):
    """Matrix exponential for the tiny finite-difference rotations used here."""
    value = np.asarray(matrix, dtype=np.float64)
    result = np.eye(value.shape[0])
    term = np.eye(value.shape[0])
    for order in range(1, terms + 1):
        term = term @ value / order
        result += term
        if np.max(np.abs(term)) < 1e-18:
            break
    return result


def finite_rotation_jvp(problem, backend, vector, *, step=1e-6, exchange_fraction=0.5):
    """Return the finite-rotation derivative of the occupied-virtual gradient.

    The parameterization uses ``C(t)=C exp(-t K)`` with the generator from
    :class:`RotationLayout`.  The returned value is ``-d g_ov/dt`` at ``t=0``,
    which equals the matrix-free Jacobian action by construction.
    """
    if problem.method != "rhf":
        raise ValueError("finite-rotation oracle currently implements RHF only")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("finite-difference step must be finite and positive")
    x = problem.layout.validate_vector(vector)
    generator = problem.layout.generator_matrix(x)
    reference = problem.reference
    coefficients = reference.coefficients
    hcore = reference.hcore
    occupied = problem.layout.occupied
    virtual = problem.layout.virtual

    def gradient(sign):
        rotated = coefficients @ _expm_small(-sign * step * generator)
        occupied_coefficients = rotated[:, occupied]
        density = 2.0 * occupied_coefficients @ occupied_coefficients.T
        coulomb, exchange = backend.coulomb_exchange(density)
        fock = hcore + coulomb - exchange_fraction * exchange
        return rotated.T @ fock @ rotated

    plus = gradient(1.0)
    minus = gradient(-1.0)
    derivative = (plus - minus) / (2.0 * step)
    response = derivative[np.ix_(virtual, occupied)].T
    return problem.layout.validate_vector(response.reshape(-1))

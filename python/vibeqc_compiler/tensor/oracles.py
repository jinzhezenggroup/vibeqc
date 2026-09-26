"""Tiny explicit-loop references independent of the TensorIR/NumPy einsum.

These equations are deliberately written out to check orbital ordering,
permutation signs, and prefactors. They are fragments, not a CCSD solver.
"""

import typing

import numpy as np


def matrix_product(left: typing.Any, right: typing.Any) -> typing.Any:
    """C[p,q] = sum_P A[p,P] B[P,q]."""
    result = np.zeros((left.shape[0], right.shape[1]))
    for p in range(left.shape[0]):
        for q in range(right.shape[1]):
            for auxiliary in range(left.shape[1]):
                result[p, q] += left[p, auxiliary] * right[auxiliary, q]
    return result


def mp2_energy(
    integrals: typing.Any, occupied_energy: typing.Any, virtual_energy: typing.Any
) -> typing.Any:
    """Spin-orbital fragment E = (1/4) sum_ijab g_ijab^2 / D_ijab."""
    result = 0.0
    for i, j, a, b in np.ndindex(integrals.shape):
        denominator = (
            occupied_energy[i]
            + occupied_energy[j]
            - virtual_energy[a]
            - virtual_energy[b]
        )
        result += 0.25 * integrals[i, j, a, b] ** 2 / denominator
    return np.asarray(result)


def virtual_residual(fock: typing.Any, amplitudes: typing.Any) -> typing.Any:
    """One antisymmetrized virtual Fock term, not the full CCSD residual."""
    result = np.zeros(amplitudes.shape)
    for i, j, a, b in np.ndindex(amplitudes.shape):
        for e in range(fock.shape[0]):
            result[i, j, a, b] += (
                fock[a, e] * amplitudes[i, j, e, b]
                - fock[b, e] * amplitudes[i, j, e, a]
            )
    return result


def restricted_pair_update(trial: typing.Any) -> typing.Any:
    """Spatial t2 pair exchange has a plus sign, with no separate antisymmetry."""
    result = np.zeros(trial.shape)
    for i, j, a, b in np.ndindex(trial.shape):
        result[i, j, a, b] = 0.5 * (trial[i, j, a, b] + trial[j, i, b, a])
    return result

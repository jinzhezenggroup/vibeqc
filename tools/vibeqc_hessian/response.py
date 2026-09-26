"""Closed-shell stationary nuclear-perturbation RHS construction.

The shared response solver owns the Jacobian action, but issue #179
explicitly assigns nuclear RHS construction to Hessian callers. This module
implements the symmetric metric-gauge convention fixed in ``docs/developer/hessian.md``
and returns the occupied-major/virtual-minor layout consumed by the shared
closed-shell RHF/CPKS response operators.
"""

from __future__ import annotations

import typing

import numpy as np

__all__ = [
    "build_rhf_nuclear_rhs",
    "build_stationary_nuclear_rhs",
    "metric_density_response_mo",
]


def _finite_matrix(values: typing.Any, *, name: str, nmo: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (nmo, nmo):
        raise ValueError(f"{name} must have shape ({nmo}, {nmo}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _orbital_energies(values: typing.Any, *, nmo: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (nmo,):
        raise ValueError(
            f"orbital_energies must have shape ({nmo},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("orbital_energies must be finite")
    return array


def metric_density_response_mo(
    overlap_derivative_mo: typing.Any, *, nocc: int
) -> np.ndarray:
    """Return the known MO density connection ``-1/2 (S_R D + D S_R)``.

    ``D`` is the closed-shell occupation matrix with value 2 on occupied
    orbitals and zero on virtual orbitals.  The returned matrix is in the MO
    basis and must be transformed to AO form before applying the backend
    Coulomb/exchange map.
    """

    overlap = np.asarray(overlap_derivative_mo, dtype=np.float64)
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap_derivative_mo must be a square matrix")
    if not np.isfinite(overlap).all():
        raise ValueError("overlap_derivative_mo must be finite")
    nmo = overlap.shape[0]
    if not isinstance(nocc, (int, np.integer)) or not 0 < nocc < nmo:
        raise ValueError("nocc must be an integer between 1 and nmo-1")
    occupations = np.zeros(nmo, dtype=np.float64)
    occupations[:nocc] = 2.0
    return -0.5 * (overlap * occupations[None, :] + occupations[:, None] * overlap)


def build_stationary_nuclear_rhs(
    frozen_fock_derivative_mo: typing.Any,
    overlap_derivative_mo: typing.Any,
    metric_fock_response_mo: typing.Any,
    orbital_energies: typing.Any,
    *,
    nocc: int,
) -> np.ndarray:
    """Build one closed-shell nuclear RHS in occupied-major/virtual-minor order.

    The inputs are MO matrices for the frozen Fock derivative, overlap
    derivative, and the Fock response to the known metric-density connection.
    The metric contribution is required explicitly so a caller cannot silently
    omit it. The returned RHS follows ``A x = -b``; method-specific Fock
    response has already been supplied by the response operator.
    """

    energies = _orbital_energies(orbital_energies, nmo=len(orbital_energies))
    nmo = energies.size
    frozen = _finite_matrix(
        frozen_fock_derivative_mo, name="frozen_fock_derivative_mo", nmo=nmo
    )
    overlap = _finite_matrix(
        overlap_derivative_mo, name="overlap_derivative_mo", nmo=nmo
    )
    metric = _finite_matrix(
        metric_fock_response_mo, name="metric_fock_response_mo", nmo=nmo
    )
    if not isinstance(nocc, (int, np.integer)) or not 0 < nocc < nmo:
        raise ValueError("nocc must be an integer between 1 and nmo-1")
    occupied = np.arange(nocc)
    virtual = np.arange(nocc, nmo)
    ai = (frozen + metric)[np.ix_(virtual, occupied)].T
    overlap_ia = overlap[np.ix_(occupied, virtual)]
    eps_i = energies[occupied, None]
    eps_a = energies[None, virtual]
    return ai - 0.5 * (eps_i + eps_a) * overlap_ia


def build_rhf_nuclear_rhs(
    frozen_fock_derivative_mo: typing.Any,
    overlap_derivative_mo: typing.Any,
    metric_fock_response_mo: typing.Any,
    orbital_energies: typing.Any,
    *,
    nocc: int,
) -> np.ndarray:
    """Backward-compatible RHF name for the common closed-shell RHS algebra."""
    return build_stationary_nuclear_rhs(
        frozen_fock_derivative_mo,
        overlap_derivative_mo,
        metric_fock_response_mo,
        orbital_energies,
        nocc=nocc,
    )

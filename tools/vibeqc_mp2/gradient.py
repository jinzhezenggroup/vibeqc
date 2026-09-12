"""Analytic MP2 energy adjoints used by complete-gradient assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from vibeqc_compiler.tensor import vjp

from tools.vibeqc_posthf.reference import immutable

from .equations import cpu_capacity, energy_program


@dataclass(frozen=True)
class TileEnergyAdjoint:
    """VJP weights for one ordered ``(i,j,a,b)`` MP2 energy tile."""

    direct: np.ndarray
    exchange: np.ndarray
    occupied_i: np.ndarray
    occupied_j: np.ndarray
    virtual_a: np.ndarray
    virtual_b: np.ndarray
    equation_hash: str


@dataclass(frozen=True)
class MP2EnergyAdjoint:
    """Global canonical weights for ``g[i,j,a,b]=(ia|jb)`` and ``epsilon[p]``."""

    integrals_iajb: np.ndarray
    orbital_energies: np.ndarray
    equation_hash: str


@dataclass(frozen=True)
class MP2OrbitalRHS:
    """Canonical response RHS ``-dE/dkappa`` and its MO weights."""

    response_rhs: np.ndarray
    one_electron: np.ndarray
    two_electron: np.ndarray


def tile_energy_adjoint(feeds, *, max_bytes=None):
    """Differentiate total OS+SS energy without differentiating solver history.

    ``g`` is ``(ia|jb)`` and ``x`` is ``(ib|ja)`` in the exact tile layout
    consumed by :func:`energy_program`. The four energy-vector cotangents retain
    repeated occupied/virtual contributions for later global scatter-add.
    """

    arrays = {
        name: np.asarray(value, dtype=np.float64) for name, value in feeds.items()
    }
    if set(arrays) != {"g", "x", "ei", "ej", "ea", "eb"}:
        raise ValueError("MP2 adjoint requires g/x and ei/ej/ea/eb feeds")
    if arrays["g"].ndim != 4 or arrays["x"].shape != arrays["g"].shape:
        raise ValueError(
            "MP2 adjoint integral tiles must have the same rank-four shape"
        )
    shape = arrays["g"].shape
    if tuple(arrays[name].shape for name in ("ei", "ej", "ea", "eb")) != tuple(
        (shape[index],) for index in range(4)
    ):
        raise ValueError("MP2 adjoint orbital-energy vectors do not match the tile")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("MP2 adjoint feeds must be finite")
    program = energy_program(shape, differentiable=True)
    budget = cpu_capacity(program) * 4 if max_bytes is None else max_bytes
    reverse = vjp(
        program,
        arrays,
        {"opposite_spin": np.array(1.0), "same_spin": np.array(1.0)},
        max_bytes=budget,
    )
    bars = reverse.input_cotangents
    return TileEnergyAdjoint(
        *(immutable(bars[name]) for name in ("g", "x", "ei", "ej", "ea", "eb")),
        reverse.primal_logical_hash,
    )


def canonical_energy_adjoint(integrals_iajb, orbital_energies, occupied):
    """Return full canonical MP2 VJP weights with repeated feeds accumulated.

    The exchange feed is a transposed read of the same physical ``(ia|jb)``
    tensor. Its cotangent is therefore transposed back before addition. The
    occupied/virtual energy vectors occur twice in the denominator and are
    scatter-added into one global orbital-energy weight.
    """

    g = np.asarray(integrals_iajb, dtype=np.float64)
    eps = np.asarray(orbital_energies, dtype=np.float64)
    if g.ndim != 4 or g.shape[0] != g.shape[1] or g.shape[2] != g.shape[3]:
        raise ValueError("canonical MP2 integrals require [nocc,nocc,nvirt,nvirt]")
    no, nv = g.shape[0], g.shape[2]
    if type(occupied) is not int or occupied != no or eps.shape != (no + nv,):
        raise ValueError("canonical MP2 occupation/energy dimensions are inconsistent")
    tile = tile_energy_adjoint(
        {
            "g": g,
            "x": g.swapaxes(2, 3),
            "ei": eps[:no],
            "ej": eps[:no],
            "ea": eps[no:],
            "eb": eps[no:],
        }
    )
    energy_weight = np.empty_like(eps)
    energy_weight[:no] = tile.occupied_i + tile.occupied_j
    energy_weight[no:] = tile.virtual_a + tile.virtual_b
    return MP2EnergyAdjoint(
        immutable(tile.direct + tile.exchange.swapaxes(2, 3)),
        immutable(energy_weight),
        tile.equation_hash,
    )


def canonical_orbital_rhs(hcore_mo, eri_mo, adjoint, occupied):
    """Differentiate the MP2 correlation Lagrangian under MO rotations.

    Orbital-energy cotangents multiply the diagonal of the RHF Fock matrix.
    Expanding that Fock build produces one- and two-electron MO weights. The
    negative antisymmetric part of their rotation derivative is the explicit
    RHS in the shared solver convention ``A z = -dE/dkappa``.
    """

    h = np.asarray(hcore_mo, dtype=np.float64)
    eri = np.asarray(eri_mo, dtype=np.float64)
    if not isinstance(adjoint, MP2EnergyAdjoint):
        raise TypeError("canonical orbital RHS requires an MP2EnergyAdjoint")
    if h.ndim != 2 or h.shape[0] != h.shape[1] or eri.shape != h.shape * 2:
        raise ValueError("canonical orbital RHS requires square h and rank-four ERIs")
    n = h.shape[0]
    if type(occupied) is not int or not 0 < occupied < n:
        raise ValueError("canonical orbital RHS requires occupied and virtual orbitals")
    if adjoint.orbital_energies.shape != (n,) or adjoint.integrals_iajb.shape != (
        occupied,
        occupied,
        n - occupied,
        n - occupied,
    ):
        raise ValueError("MP2 adjoint dimensions do not match the MO Hamiltonian")
    if (
        not np.isfinite(h).all()
        or not np.isfinite(eri).all()
        or not np.isfinite(adjoint.orbital_energies).all()
        or not np.isfinite(adjoint.integrals_iajb).all()
    ):
        raise ValueError("canonical orbital RHS inputs must be finite")

    one = np.zeros_like(h)
    two = np.zeros_like(eri)
    diagonal = np.arange(n)
    one[diagonal, diagonal] = adjoint.orbital_energies
    for p, weight in enumerate(adjoint.orbital_energies):
        for i in range(occupied):
            two[p, p, i, i] += 2 * weight
            two[p, i, i, p] -= weight
    two[
        :occupied,
        occupied:,
        :occupied,
        occupied:,
    ] += adjoint.integrals_iajb.transpose(0, 2, 1, 3)

    rotation_gradient = np.einsum("pq,tq->tp", one, h, optimize=True)
    rotation_gradient += np.einsum("pq,pt->tq", one, h, optimize=True)
    rotation_gradient += np.einsum("pqrs,tqrs->tp", two, eri, optimize=True)
    rotation_gradient += np.einsum("pqrs,ptrs->tq", two, eri, optimize=True)
    rotation_gradient += np.einsum("pqrs,pqts->tr", two, eri, optimize=True)
    rotation_gradient += np.einsum("pqrs,pqrt->ts", two, eri, optimize=True)
    rhs = (
        rotation_gradient[occupied:, :occupied].T
        - rotation_gradient[:occupied, occupied:]
    )
    return MP2OrbitalRHS(immutable(rhs), immutable(one), immutable(two))

"""Analytic MP2 energy adjoints used by complete-gradient assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from vibeqc_compiler.tensor import vjp

from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_response import RHFResponseOperator, solve

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
    reference_identity: str
    hamiltonian_id: str


@dataclass(frozen=True)
class MP2OrbitalRHS:
    """Canonical response RHS ``-dE/dkappa`` and its MO weights."""

    energy_gradient: np.ndarray
    response_rhs: np.ndarray
    one_electron: np.ndarray
    two_electron: np.ndarray
    reference_identity: str
    hamiltonian_id: str


@dataclass(frozen=True)
class MP2LagrangianWeights:
    """Relaxed MO derivative weights after the shared RHF Z-vector solve."""

    one_electron: np.ndarray
    two_electron: np.ndarray
    overlap: np.ndarray
    stationarity_residual: float
    reference_identity: str
    hamiltonian_id: str
    operator_identity: str


@dataclass(frozen=True)
class MP2ResponseResult:
    """Converged shared response result bound to its exact MP2 problem."""

    solve_result: object
    reference_identity: str
    hamiltonian_id: str
    operator_identity: str

    @property
    def solution(self):
        return self.solve_result.solution

    @property
    def converged(self):
        return self.solve_result.converged

    @property
    def residual_norm(self):
        return self.solve_result.residual_norm


@dataclass(frozen=True)
class MP2AOLagrangianWeights:
    """Relaxed AO weights for dS, dh and conventional four-center dg."""

    overlap: np.ndarray
    one_electron: np.ndarray
    two_electron: np.ndarray
    reference_identity: str
    hamiltonian_id: str
    operator_identity: str


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


def canonical_energy_adjoint(
    integrals_iajb, orbital_energies, occupied, *, reference_identity, hamiltonian_id
):
    """Return full canonical MP2 VJP weights with repeated feeds accumulated.

    The exchange feed is a transposed read of the same physical ``(ia|jb)``
    tensor. Its cotangent is therefore transposed back before addition. The
    occupied/virtual energy vectors occur twice in the denominator and are
    scatter-added into one global orbital-energy weight.
    """

    g = np.asarray(integrals_iajb, dtype=np.float64)
    eps = np.asarray(orbital_energies, dtype=np.float64)
    for name, value in (
        ("reference_identity", reference_identity),
        ("hamiltonian_id", hamiltonian_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty identity")
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
        reference_identity,
        hamiltonian_id,
    )


def _rotation_gradient(one, two, h, eri):
    result = np.einsum("pq,tq->tp", one, h, optimize=True)
    result += np.einsum("pq,pt->tq", one, h, optimize=True)
    result += np.einsum("pqrs,tqrs->tp", two, eri, optimize=True)
    result += np.einsum("pqrs,ptrs->tq", two, eri, optimize=True)
    result += np.einsum("pqrs,pqts->tr", two, eri, optimize=True)
    result += np.einsum("pqrs,pqrt->ts", two, eri, optimize=True)
    return result


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

    rotation_gradient = _rotation_gradient(one, two, h, eri)
    energy_gradient = (
        rotation_gradient[:occupied, occupied:]
        - rotation_gradient[occupied:, :occupied].T
    )
    fock = h.copy()
    for i in range(occupied):
        fock += 2 * eri[:, :, i, i] - eri[:, i, i, :]
    if np.max(np.abs(fock - np.diag(np.diag(fock)))) > 1e-8:
        raise ValueError("MP2 orbital RHS requires a canonical RHF Fock matrix")
    energies = np.diag(fock)

    def add_negative_fock_multiplier(row, column, value):
        one[row, column] -= value
        for j in range(occupied):
            two[row, column, j, j] -= 2 * value
            two[row, j, j, column] += value

    for block in (tuple(range(occupied)), tuple(range(occupied, n))):
        for offset, p in enumerate(block):
            for q in block[offset + 1 :]:
                denominator = energies[p] - energies[q]
                if abs(denominator) <= 1e-10:
                    raise ValueError(
                        "same-space canonical MP2 response is near-degenerate"
                    )
                derivative = rotation_gradient[p, q] - rotation_gradient[q, p]
                add_negative_fock_multiplier(q, p, derivative / denominator)
    rotation_gradient = _rotation_gradient(one, two, h, eri)
    rhs = (
        rotation_gradient[occupied:, :occupied].T
        - rotation_gradient[:occupied, occupied:]
    )
    return MP2OrbitalRHS(
        immutable(energy_gradient),
        immutable(rhs),
        immutable(one),
        immutable(two),
        adjoint.reference_identity,
        adjoint.hamiltonian_id,
    )


def solve_canonical_orbital_response(reference, backend, orbital_rhs, *, options=None):
    """Solve the MP2 Z-vector with the shared bounded RHF response layer."""

    if not isinstance(orbital_rhs, MP2OrbitalRHS):
        raise TypeError("MP2 response requires an MP2OrbitalRHS")
    problem = RHFResponseOperator.build_problem(
        reference, backend, perturbation_labels=("mp2-orbital-lagrangian",)
    )
    if orbital_rhs.reference_identity != reference.identity:
        raise ValueError("MP2 response RHS belongs to a different reference")
    if orbital_rhs.hamiltonian_id != reference.hamiltonian_id:
        raise ValueError("MP2 response RHS Hamiltonian differs from the reference")
    if (
        hasattr(backend, "hamiltonian_id")
        and backend.hamiltonian_id != orbital_rhs.hamiltonian_id
    ):
        raise ValueError("MP2 response backend Hamiltonian differs from the RHS")
    expected = (problem.layout.nocc, problem.layout.nvirt)
    if orbital_rhs.response_rhs.shape != expected:
        raise ValueError(
            "MP2 response RHS does not match the reference rotation layout"
        )
    operator = RHFResponseOperator(problem, backend)
    result = solve(
        operator,
        orbital_rhs.response_rhs.reshape(-1),
        options=options,
        raise_on_failure=True,
    )
    return MP2ResponseResult(
        result,
        reference.identity,
        reference.hamiltonian_id,
        problem.operator_identity,
    )


def canonical_lagrangian_weights(hcore_mo, eri_mo, adjoint, response, occupied):
    """Combine HF, MP2 and Z-vector terms into relaxed MO derivative weights."""

    orbital = canonical_orbital_rhs(hcore_mo, eri_mo, adjoint, occupied)
    h = np.asarray(hcore_mo, dtype=np.float64)
    eri = np.asarray(eri_mo, dtype=np.float64)
    if not isinstance(response, MP2ResponseResult):
        raise TypeError("MP2 Lagrangian weights require a bound response result")
    if response.reference_identity != adjoint.reference_identity:
        raise ValueError("MP2 Z-vector belongs to a different reference")
    if response.hamiltonian_id != adjoint.hamiltonian_id:
        raise ValueError("MP2 Z-vector Hamiltonian differs from the energy adjoint")
    z = np.asarray(response.solution, dtype=np.float64)
    n = h.shape[0]
    if z.shape != (occupied * (n - occupied),) or not np.isfinite(z).all():
        raise ValueError("MP2 Z-vector does not match the occupied-virtual layout")
    z = z.reshape(occupied, n - occupied)
    one = np.array(orbital.one_electron, copy=True)
    two = np.array(orbital.two_electron, copy=True)

    def add_negative_fock_multiplier(row, column, value):
        one[row, column] -= value
        for j in range(occupied):
            two[row, column, j, j] -= 2 * value
            two[row, j, j, column] += value

    for i in range(occupied):
        one[i, i] += 2.0
        for j in range(occupied):
            two[i, i, j, j] += 2.0
            two[i, j, j, i] -= 1.0
    for i in range(occupied):
        for a in range(occupied, n):
            add_negative_fock_multiplier(a, i, z[i, a - occupied])
    gradient = _rotation_gradient(one, two, h, eri)
    stationarity = gradient - gradient.T
    overlap = -0.25 * (gradient + gradient.T)
    return MP2LagrangianWeights(
        immutable(one),
        immutable(two),
        immutable(overlap),
        float(np.linalg.norm(stationarity)),
        adjoint.reference_identity,
        adjoint.hamiltonian_id,
        response.operator_identity,
    )


def ao_lagrangian_weights(reference, weights):
    """Back-transform relaxed MO weights without changing derivative factors."""

    if not isinstance(weights, MP2LagrangianWeights):
        raise TypeError("AO transformation requires MP2LagrangianWeights")
    if weights.reference_identity != reference.identity:
        raise ValueError("MP2 Lagrangian weights belong to a different reference")
    if weights.hamiltonian_id != reference.hamiltonian_id:
        raise ValueError("MP2 Lagrangian Hamiltonian differs from the reference")
    c = reference.coefficients
    one = c @ weights.one_electron @ c.T
    overlap = c @ weights.overlap @ c.T
    two = np.einsum(
        "pqrs,up,vq,wr,xs->uvwx",
        weights.two_electron,
        c,
        c,
        c,
        c,
        optimize=True,
    )
    return MP2AOLagrangianWeights(
        immutable(overlap),
        immutable(one),
        immutable(two),
        weights.reference_identity,
        weights.hamiltonian_id,
        weights.operator_identity,
    )


def dense_molecular_gradient_oracle(
    reference, source, weights, *, output_budget_bytes=256 << 20
):
    """Contract the private <=12-AO derivative oracle for validation only."""

    ao = ao_lagrangian_weights(reference, weights)
    if (
        source.geometry_hash != reference.geometry_hash
        or source.basis_hash != reference.basis_hash
        or source.representation != reference.representation
    ):
        raise ValueError("MP2 derivative source differs from the reference")
    derivatives = source.integral_derivatives(output_budget_bytes=output_budget_bytes)
    gradient = derivatives["nuclear"].copy()
    gradient += np.einsum("xpq,pq->x", derivatives["overlap"], ao.overlap)
    gradient += np.einsum("xpq,pq->x", derivatives["hcore"], ao.one_electron)
    gradient += np.einsum("xpqrs,pqrs->x", derivatives["eri"], ao.two_electron)
    if not np.isfinite(gradient).all():
        raise ValueError("MP2 molecular gradient is nonfinite")
    return immutable(gradient.reshape(len(source.atoms), 3))

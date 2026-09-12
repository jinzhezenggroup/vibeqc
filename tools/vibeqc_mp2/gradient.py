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


@dataclass(frozen=True)
class MP2RILagrangianWeights:
    """Relaxed AO/DF weights for dS, dh, raw dA and metric dM."""

    overlap: np.ndarray
    one_electron: np.ndarray
    three_center: np.ndarray
    metric: np.ndarray
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


def _rotation_gradient_streamed(one, two, h, provider):
    """Contract the generalized-Fock rotation gradient without a full MO ERI."""

    from tools.vibeqc_posthf.conventions import MOBlock

    one = np.asarray(one, dtype=np.float64)
    two = np.asarray(two, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n = h.shape[0]
    if one.shape != (n, n) or two.shape != (n, n, n, n):
        raise ValueError("streamed rotation weights have inconsistent dimensions")
    result = np.einsum("pq,tq->tp", one, h, optimize=True)
    result += np.einsum("pq,pt->tq", one, h, optimize=True)
    orbitals = tuple(range(n))

    def read(slots):
        block = provider.get(MOBlock(slots)).to_host()
        value = np.asarray(block, dtype=np.float64)
        provider.clear()
        return value

    for t in range(n):
        first = read(((t,), orbitals, orbitals, orbitals))[0]
        result[t, :] += np.einsum("pqrs,qrs->p", two, first, optimize=True)
        del first
        second = read((orbitals, (t,), orbitals, orbitals))[:, 0]
        result[t, :] += np.einsum("pqrs,prs->q", two, second, optimize=True)
        del second
        third = read((orbitals, orbitals, (t,), orbitals))[:, :, 0]
        result[t, :] += np.einsum("pqrs,pqs->r", two, third, optimize=True)
        del third
        fourth = read((orbitals, orbitals, orbitals, (t,)))[:, :, :, 0]
        result[t, :] += np.einsum("pqrs,pqr->s", two, fourth, optimize=True)
        del fourth
    if not np.isfinite(result).all():
        raise ValueError("streamed MP2 rotation gradient is nonfinite")
    return result


def canonical_orbital_rhs_streamed(reference, hcore_mo, provider, adjoint, occupied):
    """Build the canonical MP2 response RHS from bounded provider blocks."""

    h = np.asarray(hcore_mo, dtype=np.float64)
    n = h.shape[0]
    if (
        h.shape != (n, n)
        or type(occupied) is not int
        or not 0 < occupied < n
        or provider.snapshot.identity != reference.identity
        or provider.snapshot.hamiltonian_id != reference.hamiltonian_id
        or adjoint.reference_identity != reference.identity
        or adjoint.hamiltonian_id != reference.hamiltonian_id
    ):
        raise ValueError("streamed MP2 orbital RHS reference/provider mismatch")
    if adjoint.orbital_energies.shape != (n,) or adjoint.integrals_iajb.shape != (
        occupied,
        occupied,
        n - occupied,
        n - occupied,
    ):
        raise ValueError("streamed MP2 adjoint dimensions are inconsistent")
    one = np.zeros_like(h)
    two = np.zeros((n,) * 4)
    diagonal = np.arange(n)
    one[diagonal, diagonal] = adjoint.orbital_energies
    for p, weight in enumerate(adjoint.orbital_energies):
        for i in range(occupied):
            two[p, p, i, i] += 2 * weight
            two[p, i, i, p] -= weight
    two[:occupied, occupied:, :occupied, occupied:] += adjoint.integrals_iajb.transpose(
        0, 2, 1, 3
    )
    rotation_gradient = _rotation_gradient_streamed(one, two, h, provider)
    energy_gradient = (
        rotation_gradient[:occupied, occupied:]
        - rotation_gradient[occupied:, :occupied].T
    )
    energies = reference.orbital_energies

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
    rotation_gradient = _rotation_gradient_streamed(one, two, h, provider)
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


def canonical_lagrangian_weights_streamed(
    reference, hcore_mo, provider, adjoint, response, occupied
):
    """Assemble relaxed weights while streaming every MO-ERI contraction."""

    orbital = canonical_orbital_rhs_streamed(
        reference, hcore_mo, provider, adjoint, occupied
    )
    h = np.asarray(hcore_mo, dtype=np.float64)
    if not isinstance(response, MP2ResponseResult):
        raise TypeError("streamed MP2 Lagrangian requires a bound response result")
    if (
        response.reference_identity != adjoint.reference_identity
        or response.hamiltonian_id != adjoint.hamiltonian_id
        or response.reference_identity != reference.identity
    ):
        raise ValueError("streamed MP2 Z-vector belongs to a different problem")
    z = np.asarray(response.solution, dtype=np.float64)
    n = h.shape[0]
    if z.shape != (occupied * (n - occupied),) or not np.isfinite(z).all():
        raise ValueError("streamed MP2 Z-vector has the wrong layout")
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
    gradient = _rotation_gradient_streamed(one, two, h, provider)
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


def ao_lagrangian_weights(reference, weights, *, output_budget_bytes=None):
    """Back-transform relaxed MO weights without changing derivative factors."""

    if not isinstance(weights, MP2LagrangianWeights):
        raise TypeError("AO transformation requires MP2LagrangianWeights")
    if weights.reference_identity != reference.identity:
        raise ValueError("MP2 Lagrangian weights belong to a different reference")
    if weights.hamiltonian_id != reference.hamiltonian_id:
        raise ValueError("MP2 Lagrangian Hamiltonian differs from the reference")
    c = reference.coefficients
    n = reference.nmo
    output_bytes = (2 * n * n + n**4) * 8
    if output_budget_bytes is not None:
        if type(output_budget_bytes) is not int or output_budget_bytes < 1:
            raise ValueError("AO Lagrangian output budget must be positive")
        if output_bytes > output_budget_bytes:
            raise MemoryError("AO Lagrangian weights exceed their output budget")
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


def fused_cuda_conventional_molecular_gradient(
    reference,
    source,
    weights,
    orbital_calculator,
    *,
    weight_output_budget_bytes=256 << 20,
    consumer_maximum_bytes=128 << 20,
    device_id=0,
):
    """Contract conventional weights through the #141/#144 CUDA consumers.

    The dense MO two-electron cotangent remains an explicit migration boundary.
    It is back-transformed one public AO shell quartet at a time and consumed
    immediately, so no molecular AO rank-four weight or nuclear-coordinate
    derivative tensor is formed. Published one-electron weights plus the
    largest local quartet have one output guard; transform scratch and caller
    MO weights are excluded. Each derivative consumer has a separate limit.
    """

    from tools.vibeqc_validation.one_electron_gradient import execute_gradient

    if (
        type(weight_output_budget_bytes) is not int
        or weight_output_budget_bytes < 1
        or type(consumer_maximum_bytes) is not int
        or consumer_maximum_bytes < 1
        or type(device_id) is not int
        or device_id < 0
    ):
        raise ValueError(
            "fused CUDA conventional gradient requires valid stage controls"
        )
    if not isinstance(weights, MP2LagrangianWeights):
        raise TypeError("conventional gradient requires MP2LagrangianWeights")
    if (
        reference.hamiltonian_id != "conventional-unscreened"
        or weights.hamiltonian_id != "conventional-unscreened"
    ):
        raise ValueError(
            "conventional gradient requires the unscreened exact Hamiltonian"
        )
    if weights.reference_identity != reference.identity:
        raise ValueError(
            "conventional gradient weights belong to a different reference"
        )
    if (
        source.electron_count != reference.electron_count
        or source.multiplicity != 1
        or reference.geometry_hash != source.geometry_hash
        or reference.basis_hash != source.basis_hash
        or reference.representation != source.representation
        or reference.nmo != source.nbf
    ):
        raise ValueError("conventional gradient source/reference identity mismatch")
    representation = (
        "spherical" if source.representation == "real_spherical" else "cartesian"
    )
    if (
        getattr(orbital_calculator, "_device_name", None) != "cuda"
        or getattr(orbital_calculator, "_device_id", None) != device_id
        or getattr(orbital_calculator, "_representation_name", None) != representation
        or tuple(orbital_calculator._shells_for_atoms(source.atoms)) != source.shells
    ):
        raise ValueError("conventional gradient calculator differs from the source")
    from itertools import product

    c = reference.coefficients
    maximum_shell = max(source.shell_sizes)
    maximum_local_bytes = maximum_shell**4 * 8
    # The one-electron phase temporarily holds the two transformed weights and
    # their three-block S/T/V publication. The later shell phase holds one
    # local quartet and the molecular two-electron gradient candidate.
    molecular_gradient_bytes = len(source.atoms) * 3 * 8
    weight_output_bytes = max(
        5 * reference.nmo**2 * 8 + molecular_gradient_bytes,
        maximum_local_bytes + 2 * molecular_gradient_bytes + 12 * 8,
    )
    if weight_output_bytes > weight_output_budget_bytes:
        raise MemoryError("tiled AO Lagrangian weights exceed their output budget")
    one_weight = c @ weights.one_electron @ c.T
    overlap_weight = c @ weights.overlap @ c.T
    one_blocks = np.stack((overlap_weight, one_weight, one_weight))
    one, one_resources = execute_gradient(
        orbital_calculator,
        source.atoms,
        one_blocks,
        maximum_bytes=consumer_maximum_bytes,
        device_id=device_id,
        charge=source.charge,
        multiplicity=source.multiplicity,
    )
    del one_blocks, one_weight, overlap_weight
    offsets = np.cumsum((0, *source.shell_sizes))
    two = np.zeros((len(source.atoms), 3))
    tiles = 0
    for shell_indices in product(range(len(source.shells)), repeat=4):
        panels = tuple(
            c[offsets[index] : offsets[index + 1], :] for index in shell_indices
        )
        local = np.einsum(
            "pqrs,up,vq,wr,xs->uvwx",
            weights.two_electron,
            *panels,
            optimize=True,
        )
        center_gradient = source.weighted_eri_shell_gradient_cuda(
            shell_indices,
            local,
            device_id=device_id,
            stage_budget_bytes=consumer_maximum_bytes,
        )
        for slot, shell in enumerate(shell_indices):
            two[source.shells[shell].atom_index] += center_gradient[slot]
        tiles += 1
        del center_gradient, local, panels
    gradient = _nuclear_repulsion_gradient(source.atoms) + one + two
    if not np.isfinite(gradient).all():
        raise ValueError("fused CUDA conventional MP2 molecular gradient is nonfinite")
    return immutable(gradient), {
        "one_electron": one_resources,
        "weighted_eri_stage_bytes": consumer_maximum_bytes,
        "weighted_eri_shell_tiles": tiles,
        "global_derivative_tensors": False,
        "dense_response_weights": True,
        "dense_ao_two_electron_weights": False,
        "weight_output_bytes": int(weight_output_bytes),
        "weight_output_budget_bytes": weight_output_budget_bytes,
        "excluded_from_bridge_budget": (
            "shell-local AO cotangent transformation scratch",
            "caller-owned dense MO response weights",
            "Python/native object metadata",
            "CUDA context and allocator overhead",
        ),
    }


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


def _inverse_sqrt_metric_response(metric, response, relative_threshold):
    """Apply the self-adjoint fixed-rank Frechet derivative of M**(-1/2)."""

    matrix = np.asarray(metric, dtype=np.float64)
    bar = np.asarray(response, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or bar.shape != matrix.shape
        or not np.isfinite(matrix).all()
        or not np.isfinite(bar).all()
        or not 0 < relative_threshold < 1
    ):
        raise ValueError("RI metric response dimensions or threshold are invalid")
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    largest = values[-1]
    if not largest > 0:
        raise ValueError("RI metric has no positive response subspace")
    cutoff = relative_threshold * largest
    scale = max(1.0, abs(largest))
    if np.min(np.abs(values - cutoff)) <= 1e-12 * scale:
        raise ValueError("RI metric rank is unresolved at the response cutoff")
    retained = values > cutoff
    function = np.zeros_like(values)
    function[retained] = values[retained] ** -0.5
    divided = np.empty((len(values), len(values)))
    for p, left in enumerate(values):
        for q, right in enumerate(values):
            if abs(left - right) <= 1e-12 * max(1.0, abs(left), abs(right)):
                divided[p, q] = -0.5 * left**-1.5 if retained[p] else 0.0
            else:
                divided[p, q] = (function[p] - function[q]) / (left - right)
    eigen_response = vectors.T @ (0.5 * (bar + bar.T)) @ vectors
    result = vectors @ (divided * eigen_response) @ vectors.T
    if not np.isfinite(result).all():
        raise ValueError("RI metric inverse-square-root response is nonfinite")
    return immutable(0.5 * (result + result.T))


def dense_ri_lagrangian_weights_oracle(
    reference, source, metric_factor, weights, *, output_budget_bytes=256 << 20
):
    """Reverse the relaxed RI Lagrangian to raw A/M weights for validation."""

    if not isinstance(weights, MP2LagrangianWeights):
        raise TypeError("RI transformation requires MP2LagrangianWeights")
    if (
        weights.reference_identity != reference.identity
        or weights.hamiltonian_id != reference.hamiltonian_id
        or metric_factor.hamiltonian_id != reference.hamiltonian_id
        or metric_factor.geometry_hash != source.geometry_hash
        or metric_factor.auxiliary_hash != source.auxiliary_hash
        or reference.geometry_hash != source.geometry_hash
        or reference.basis_hash != source.basis_hash
        or reference.representation != source.representation
        or reference.nmo != source.nbf
    ):
        raise ValueError("RI Lagrangian/reference/source/metric identity mismatch")
    n = reference.nmo
    na = source.naux
    # Published overlap/one/A/M cotangents coexist on return. Python integer
    # arithmetic is unbounded; the byte product is checked against the caller's
    # explicit output guard before any source read or NumPy allocation.
    elements = 2 * n * n + n * n * na + na * na
    if elements * 8 > output_budget_bytes:
        raise MemoryError("RI Lagrangian oracle exceeds its output budget")
    raw_a = source._read("three_center_eri", (0, 0, 0), (n, n, na))
    raw_m = source._read("coulomb_metric", (0, 0), (na, na))
    c = reference.coefficients
    transformed = np.einsum("mp,nq,mnP->Ppq", c, c, raw_a, optimize=True)
    inverse_root = metric_factor.inverse_square_root
    whitened = np.einsum("Ppq,PQ->Qpq", transformed, inverse_root, optimize=True)
    two = weights.two_electron
    bar_whitened = np.einsum("pqrs,Qrs->Qpq", two, whitened, optimize=True)
    bar_whitened += np.einsum("rspq,Qrs->Qpq", two, whitened, optimize=True)
    bar_transformed = np.einsum(
        "PQ,Qpq->Ppq", inverse_root, bar_whitened, optimize=True
    )
    bar_a = np.einsum("mp,nq,Ppq->mnP", c, c, bar_transformed, optimize=True)
    bar_inverse_root = np.einsum(
        "Ppq,Qpq->PQ", transformed, bar_whitened, optimize=True
    )
    bar_m = _inverse_sqrt_metric_response(
        raw_m, bar_inverse_root, metric_factor.relative_threshold
    )
    one = c @ weights.one_electron @ c.T
    overlap = c @ weights.overlap @ c.T
    return MP2RILagrangianWeights(
        immutable(overlap),
        immutable(one),
        immutable(bar_a),
        bar_m,
        weights.reference_identity,
        weights.hamiltonian_id,
        weights.operator_identity,
    )


def dense_ri_molecular_gradient_oracle(
    reference, source, metric_factor, weights, *, output_budget_bytes=256 << 20
):
    """Contract the private dense RI derivative oracle for validation only."""

    ao = dense_ri_lagrangian_weights_oracle(
        reference,
        source,
        metric_factor,
        weights,
        output_budget_bytes=output_budget_bytes,
    )
    derivatives = source.df_integral_derivatives(
        output_budget_bytes=output_budget_bytes
    )
    gradient = derivatives["nuclear"].copy()
    gradient += np.einsum("xpq,pq->x", derivatives["overlap"], ao.overlap)
    gradient += np.einsum("xpq,pq->x", derivatives["hcore"], ao.one_electron)
    gradient += np.einsum("xpqP,pqP->x", derivatives["three_center"], ao.three_center)
    gradient += np.einsum("xPQ,PQ->x", derivatives["metric"], ao.metric)
    if not np.isfinite(gradient).all():
        raise ValueError("RI-MP2 molecular gradient is nonfinite")
    return immutable(gradient.reshape(len(source.atoms), 3))


def _nuclear_repulsion_gradient(atoms):
    """Analytic nuclear-repulsion gradient in Hartree/Bohr."""

    numbers = np.asarray([atom.atomic_number for atom in atoms], dtype=np.float64)
    positions = np.asarray([atom.position for atom in atoms], dtype=np.float64)
    gradient = np.zeros_like(positions)
    for first in range(len(atoms)):
        for second in range(first):
            difference = positions[first] - positions[second]
            distance = float(np.linalg.norm(difference))
            if not distance > 0 or not np.isfinite(distance):
                raise ValueError("nuclear repulsion requires distinct finite centers")
            contribution = -numbers[first] * numbers[second] * difference / distance**3
            gradient[first] += contribution
            gradient[second] -= contribution
    return immutable(gradient)


def fused_cuda_ri_molecular_gradient(
    reference,
    source,
    metric_factor,
    weights,
    orbital_calculator,
    auxiliary_calculator,
    *,
    weight_output_budget_bytes=128 << 20,
    consumer_maximum_bytes=128 << 20,
    maximum_tile_elements=0,
    device_id=0,
):
    """Contract relaxed RI weights through the #141/#143 CUDA consumers.

    This path forms dense A/M cotangents but never forms nuclear-coordinate
    derivative tensors. ``weight_output_budget_bytes`` guards the published
    dense cotangents only; their transformation scratch is excluded.
    ``consumer_maximum_bytes`` independently bounds each sequential #141/#143
    consumer, whose descriptors exclude caller-owned weights. This is a
    migration bridge to a fully tiled method adjoint, not one whole-path peak
    bound; the private dense derivative oracle is not called.
    """

    from tools.vibeqc_validation.df_gradient import execute_df_gradient
    from tools.vibeqc_validation.one_electron_gradient import execute_gradient

    if (
        type(weight_output_budget_bytes) is not int
        or weight_output_budget_bytes < 1
        or type(consumer_maximum_bytes) is not int
        or consumer_maximum_bytes < 1
        or type(maximum_tile_elements) is not int
        or maximum_tile_elements < 0
        or type(device_id) is not int
        or device_id < 0
    ):
        raise ValueError("fused CUDA RI gradient requires valid stage budgets")
    if source.electron_count != reference.electron_count or source.multiplicity != 1:
        raise ValueError("fused RI gradient source/reference electron state mismatch")
    if (
        getattr(orbital_calculator, "_device_name", None) != "cuda"
        or getattr(auxiliary_calculator, "_device_name", None) != "cuda"
        or getattr(orbital_calculator, "_device_id", None) != device_id
        or getattr(auxiliary_calculator, "_device_id", None) != device_id
        or getattr(orbital_calculator, "_representation_name", None)
        != ("spherical" if source.representation == "real_spherical" else "cartesian")
        or getattr(auxiliary_calculator, "_representation_name", None)
        != ("spherical" if source.representation == "real_spherical" else "cartesian")
        or tuple(orbital_calculator._shells_for_atoms(source.atoms)) != source.shells
        or tuple(auxiliary_calculator._shells_for_atoms(source.atoms))
        != source.auxiliary_shells
    ):
        raise ValueError("fused CUDA RI gradient calculators differ from the source")
    ao = dense_ri_lagrangian_weights_oracle(
        reference,
        source,
        metric_factor,
        weights,
        output_budget_bytes=weight_output_budget_bytes,
    )
    one, one_resources = execute_gradient(
        orbital_calculator,
        source.atoms,
        np.stack((ao.overlap, ao.one_electron, ao.one_electron)),
        maximum_bytes=consumer_maximum_bytes,
        device_id=device_id,
        charge=source.charge,
        multiplicity=source.multiplicity,
    )
    fitted, df_resources = execute_df_gradient(
        orbital_calculator,
        auxiliary_calculator,
        source.atoms,
        ao.three_center,
        ao.metric,
        maximum_bytes=consumer_maximum_bytes,
        maximum_tile_elements=maximum_tile_elements,
        device_id=device_id,
        charge=source.charge,
        multiplicity=source.multiplicity,
    )
    gradient = _nuclear_repulsion_gradient(source.atoms) + one + fitted
    if not np.isfinite(gradient).all():
        raise ValueError("fused CUDA RI-MP2 molecular gradient is nonfinite")
    return immutable(gradient), {
        "one_electron": one_resources,
        "density_fitting": df_resources,
        "global_derivative_tensors": False,
        "dense_response_weights": True,
        "weight_output_bytes": int(
            ao.overlap.nbytes
            + ao.one_electron.nbytes
            + ao.three_center.nbytes
            + ao.metric.nbytes
        ),
        "weight_output_budget_bytes": weight_output_budget_bytes,
        "consumer_maximum_bytes": consumer_maximum_bytes,
        "excluded_from_bridge_budget": (
            "dense cotangent transformation scratch",
            "caller-owned response weights",
            "Python/native object metadata",
            "CUDA context and allocator overhead",
        ),
    }

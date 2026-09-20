"""Bounded native CPU analytic RHF Hessian through the #178/#179 layers.

SCF state, S/T/V and ERI values come from VibeQC. First-order nuclear sources
are generated from the existing compiler DAGs, explicit second derivatives
use #178, and #179 solves orbital response with streamed native J/K actions.
PySCF and the semi-numerical oracle are confined to external validation.

This is a tools integration for at most 12 Cartesian AOs/four atoms, not a
public production-size, CUDA, DF, ECP, DFT or matrix-free molecular HVP endpoint.
"""

from __future__ import annotations

import shutil
import typing
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.blocks import TensorLayout, WeightTile
from vibeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
)
from vibeqc_compiler.integral.second_derivatives_execute import (
    PreparedSecondDerivative,
    compile_second_derivative,
)
from vibeqc_compiler.integral.second_derivatives_inputs import (
    prepare_second_shell_stream,
)
from vibeqc_compiler.integral.second_order_layout import (
    SecondAtomMap,
    second_coordinate_tiles,
)
from vibeqc_compiler.integral.shell_spec import cartesian_components

from tools.vibeqc_response.backends import NativeJKBackend
from tools.vibeqc_response.operators import RHFResponseOperator

from .first_order import checked_direction
from .native import NativeRHFState
from .perturbation import solve_rhf_nuclear_perturbation

__all__ = [
    "analytic_hessian",
    "build_reference",
    "cphf_relaxation",
    "nuclear_closed_form",
    "nuclear_hvp",
    "provider_components",
    "provider_hvp_components",
]

_COMPILE_CACHE: dict = {}


# ---------------------------------------------------------------------------
# #178 second-integral provider driver
# ---------------------------------------------------------------------------


def _compile_cached(
    key: typing.Any,
    build_ir: typing.Any,
    ir_extra: typing.Any,
    adapter: typing.Any,
    cache: typing.Any,
    output_indices: typing.Any,
    component_indices: typing.Any,
) -> typing.Any:
    # The directory owns artifact lifetime; the compiler and ordered subsets
    # are part of execution/layout identity, not interchangeable cache hints.
    ck = (
        Path(cache).resolve(),
        adapter,
        key,
        tuple(output_indices),
        tuple(component_indices),
    )
    artifact = _COMPILE_CACHE.get(ck)
    if artifact is None or not artifact.native.library.is_file():
        ir = build_ir(**ir_extra)
        _COMPILE_CACHE[ck] = compile_second_derivative(
            ir,
            adapter,
            cache,
            output_indices=output_indices,
            component_indices=component_indices,
        )
    return _COMPILE_CACHE[ck]


def _tile_components(count: typing.Any, chunk: typing.Any = 64) -> typing.Any:
    for start in range(0, count, chunk):
        yield tuple(range(start, min(start + chunk, count)))


def _scatter(
    full: typing.Any, ci: typing.Any, center_atoms: typing.Any, data: typing.Any
) -> typing.Any:
    """Scatter a dense kernel result to a molecular Hessian tensor."""
    nat = data["state"].nat
    k = len(ci)
    mapping = SecondAtomMap(ci, center_atoms)
    blk = mapping.scatter_hessian(full.reshape(k * 3, k * 3))
    blk = blk.reshape(len(mapping.atom_indices), 3, len(mapping.atom_indices), 3)
    out = np.zeros((nat, 3, nat, 3))
    for i, ai in enumerate(mapping.atom_indices):
        for j, aj in enumerate(mapping.atom_indices):
            out[ai, :, aj, :] += blk[i, :, j, :]
    return out.transpose(0, 2, 1, 3)


def _scatter_hvp(
    full: typing.Any, ci: typing.Any, center_atoms: typing.Any, data: typing.Any
) -> typing.Any:
    """Scatter one recovered shell-center HVP onto physical atom rows."""
    nat = data["state"].nat
    k = len(ci)
    mapping = SecondAtomMap(ci, center_atoms)
    blk = mapping.scatter_hvp(full.reshape(k, 3))
    out = np.zeros((nat, 3))
    for i, atom in enumerate(mapping.atom_indices):
        out[atom] += blk[i]
    return out


def _run_kernel_summed(
    data: typing.Any,
    key: typing.Any,
    build_ir: typing.Any,
    ir_extra: typing.Any,
    adapter: typing.Any,
    cache: typing.Any,
    prims: typing.Any,
    centers: typing.Any,
    weight_full_flat: typing.Any,
    component_count: typing.Any,
    *,
    direction: typing.Any = None,
) -> typing.Any:
    """Run one shell tuple through the weighted Hessian or HVP provider.

    The directional path expands physical displacements to mathematical
    centers before execution and never materializes a coordinate Hessian.
    """
    ca = ir_extra.get("_center_atoms")
    extra = {k: v for k, v in ir_extra.items() if k != "_center_atoms"}
    ir = build_ir(**extra)
    ci = ir.requested_derivative_centers
    k = len(ci)
    hvp = direction is not None
    full = np.zeros(k * 3 if hvp else k * 3 * k * 3)
    sig = ir.signature
    tiles = list(second_coordinate_tiles(ci, packing="dense", hvp=hvp))
    center_direction = None
    if hvp:
        mapping = SecondAtomMap(ci, ca)
        center_direction = mapping.expand_direction(
            direction[list(mapping.atom_indices)]
        )
    for ao_chunk in _tile_components(component_count):
        wc = np.zeros(component_count)
        wc[list(ao_chunk)] = weight_full_flat[list(ao_chunk)]
        for oi in tiles:
            art = _compile_cached(
                (key, "hvp" if hvp else "hessian"),
                build_ir,
                extra,
                adapter,
                cache,
                oi,
                ao_chunk,
            )
            tile = WeightTile(TensorLayout(sig.tensor_indices, sig.component_shape), wc)
            stream = prepare_second_shell_stream(
                art,
                prims,
                centers,
                tile,
                public_signature=sig,
                projections=None,
                direction=center_direction,
            )
            with PreparedSecondDerivative(art, record_capacity=8) as plan:
                r = plan.contract(stream, profile=True)
            full[list(oi)] += np.asarray(r.values).sum(axis=0)
    if hvp:
        return _scatter_hvp(full, ci, ca, data)
    return _scatter(full, ci, ca, data)


def _provider_data(s: typing.Any) -> typing.Any:
    _validate_analytic_domain(s)
    C, eps = s.C, s.eps
    W_e = (C[:, : s.nocc] * (2 * eps[: s.nocc])) @ C[:, : s.nocc].T
    adapter = CppCompilerAdapter(Path(shutil.which("c++") or "c++"))
    return {
        "state": s,
        "shells": s.source.shells,
        "primitives": s.primitives,
        "adapter": adapter,
        "cache": s.cache / "second-cache",
        "W_e": W_e,
        "density": s.P0,
    }


def _run_one_electron(
    data: typing.Any,
    family: typing.Any,
    weight: typing.Any,
    *,
    direction: typing.Any = None,
) -> typing.Any:
    """Provider output for one one-electron family as Hessian or HVP."""
    state = data["state"]
    nat = state.nat
    adapter, cache = data["adapter"], data["cache"]
    shells = data["shells"]
    nbas = len(shells)
    loc = state.offsets
    z = state.Z
    output = "weighted_hvp" if direction is not None else "weighted_hessian"
    total = np.zeros((nat, 3) if direction is not None else (nat, nat, 3, 3))
    for a in range(nbas):
        for b in range(nbas):
            la = shells[a].angular_momentum
            lb = shells[b].angular_momentum
            na = len(cartesian_components(la))
            nb = len(cartesian_components(lb))
            wa = weight[loc[a] : loc[a] + na, loc[b] : loc[b] + nb].reshape(na, nb)
            prims = (data["primitives"][a], data["primitives"][b])
            ca_atom = shells[a].atom_index
            cb_atom = shells[b].atom_index
            if family == "nuclear_attraction":
                for nucleus in range(nat):
                    ir_extra = {
                        "family": family,
                        "angular": (la, lb),
                        "charge": float(z[nucleus]),
                        "output": output,
                        "_center_atoms": (ca_atom, cb_atom, nucleus),
                    }
                    key = (family, la, lb, float(z[nucleus]))
                    centers = np.array(
                        [
                            state.coords[ca_atom],
                            state.coords[cb_atom],
                            state.coords[nucleus],
                        ]
                    )
                    total += _run_kernel_summed(
                        data,
                        key,
                        build_one_electron_second_ir,
                        ir_extra,
                        adapter,
                        cache,
                        prims,
                        centers,
                        wa.ravel(),
                        na * nb,
                        direction=direction,
                    )
            else:
                ir_extra = {
                    "family": family,
                    "angular": (la, lb),
                    "output": output,
                    "_center_atoms": (ca_atom, cb_atom),
                }
                key = (family, la, lb)
                centers = np.array([state.coords[ca_atom], state.coords[cb_atom]])
                total += _run_kernel_summed(
                    data,
                    key,
                    build_one_electron_second_ir,
                    ir_extra,
                    adapter,
                    cache,
                    prims,
                    centers,
                    wa.ravel(),
                    na * nb,
                    direction=direction,
                )
    return total


def _run_eri(
    data: typing.Any, density: typing.Any, *, direction: typing.Any = None
) -> typing.Any:
    """Provider output for four-center ERIs as Hessian or HVP."""
    state = data["state"]
    nat = state.nat
    adapter, cache = data["adapter"], data["cache"]
    shells = data["shells"]
    nbas = len(shells)
    loc = state.offsets
    output = "weighted_hvp" if direction is not None else "weighted_hessian"
    total = np.zeros((nat, 3) if direction is not None else (nat, nat, 3, 3))
    for a in range(nbas):
        for b in range(nbas):
            for c in range(nbas):
                for d in range(nbas):
                    la = shells[a].angular_momentum
                    lb = shells[b].angular_momentum
                    lc = shells[c].angular_momentum
                    ld = shells[d].angular_momentum
                    na = len(cartesian_components(la))
                    nb = len(cartesian_components(lb))
                    nc = len(cartesian_components(lc))
                    nd = len(cartesian_components(ld))
                    sa, sb, sc, sd = (slice(loc[i], loc[i + 1]) for i in (a, b, c, d))
                    w4 = 0.5 * np.einsum(
                        "uv,wx->uvwx", density[sa, sb], density[sc, sd]
                    )
                    w4 -= 0.25 * np.einsum(
                        "uw,vx->uvwx", density[sa, sc], density[sb, sd]
                    )
                    prims = tuple(data["primitives"][i] for i in (a, b, c, d))
                    ca, cb, cc, cd = (shells[x].atom_index for x in (a, b, c, d))
                    centers = np.array(
                        [
                            state.coords[ca],
                            state.coords[cb],
                            state.coords[cc],
                            state.coords[cd],
                        ]
                    )
                    ir_extra = {
                        "angular": (la, lb, lc, ld),
                        "output": output,
                        "_center_atoms": (ca, cb, cc, cd),
                    }
                    key = ("eri", la, lb, lc, ld)
                    total += _run_kernel_summed(
                        data,
                        key,
                        build_eri_second_ir,
                        ir_extra,
                        adapter,
                        cache,
                        prims,
                        centers,
                        w4.ravel(),
                        na * nb * nc * nd,
                        direction=direction,
                    )
    return total


def provider_components(s: typing.Any) -> typing.Any:
    """Return the frozen-skeleton components from the #178 providers.

    Keys ``core`` (kinetic + nuclear_attraction, weight P0), ``pulay``
    (overlap, weight W_e, sign -1) and ``two_electron`` (ERI, weight W2),
    each shaped ``(nat, nat, 3, 3)``.
    """
    data = _provider_data(s)
    P0 = s.P0
    core = _run_one_electron(data, "kinetic", P0) + _run_one_electron(
        data, "nuclear_attraction", P0
    )
    pulay = -_run_one_electron(data, "overlap", data["W_e"])
    two_electron = _run_eri(data, data["density"])
    return {"core": core, "pulay": pulay, "two_electron": two_electron}


def provider_hvp_components(s: typing.Any, direction: typing.Any) -> typing.Any:
    """Return frozen-skeleton second-integral HVP components directly."""
    _validate_analytic_domain(s)
    vector = checked_direction(direction, s.nat)
    data = _provider_data(s)
    p0 = s.P0
    core = _run_one_electron(data, "kinetic", p0, direction=vector) + _run_one_electron(
        data, "nuclear_attraction", p0, direction=vector
    )
    pulay = -_run_one_electron(data, "overlap", data["W_e"], direction=vector)
    two_electron = _run_eri(data, data["density"], direction=vector)
    return {"core": core, "pulay": pulay, "two_electron": two_electron}


# ---------------------------------------------------------------------------
# closed-form nucleus-nucleus second derivative
# ---------------------------------------------------------------------------


def nuclear_closed_form(s: typing.Any) -> typing.Any:
    """Exact Coulomb second derivative: d^2 (Za Zb / |ra-rb|) / dx dy.

    For the pair contribution ``blk = Za Zb / d^3 (3 R R^T - I)`` the
    diagonal (atom with itself) accumulates over every other nucleus and the
    off-diagonal pair block is its negative.
    """
    Z = s.Z
    coords = s.coords
    nat = s.nat
    H = np.zeros((nat, nat, 3, 3))
    for a in range(nat):
        for b in range(a + 1, nat):
            r = coords[a] - coords[b]
            d = np.linalg.norm(r)
            R = r / d
            blk = Z[a] * Z[b] / d**3 * (3.0 * np.outer(R, R) - np.eye(3))
            H[a, a] += blk
            H[b, b] += blk
            H[a, b] -= blk
            H[b, a] -= blk.T
    return H


def nuclear_hvp(s: typing.Any, direction: typing.Any) -> typing.Any:
    """Apply the exact nucleus-nucleus Hessian to one Cartesian direction."""
    _validate_analytic_domain(s)
    vector = checked_direction(direction, s.nat)
    out = np.zeros((s.nat, 3))
    for a in range(s.nat):
        for b in range(a + 1, s.nat):
            r = s.coords[a] - s.coords[b]
            d = np.linalg.norm(r)
            unit = r / d
            block = s.Z[a] * s.Z[b] / d**3 * (3.0 * np.outer(unit, unit) - np.eye(3))
            contribution = block @ (vector[a] - vector[b])
            out[a] += contribution
            out[b] -= contribution
    return out


# ---------------------------------------------------------------------------
# #179 shared response operator for the electronic relaxation
# ---------------------------------------------------------------------------


def build_reference(s: typing.Any) -> typing.Any:
    """Reuse the caller's validated native snapshot without rerunning SCF."""
    _validate_analytic_domain(s)
    return s.reference


def _analytic_first_order_inputs(s: typing.Any) -> typing.Any:
    """Generated S/T/V/ERI contractions, tied to the same native SCF density."""
    _validate_analytic_domain(s)
    return s.first_order_inputs


def _validate_analytic_domain(s: typing.Any) -> None:
    if not isinstance(s, NativeRHFState):
        raise TypeError(
            "analytic Hessian requires NativeRHFState, not an oracle System"
        )
    s.validate()


def cphf_relaxation(s: typing.Any) -> typing.Any:
    """Electronic relaxation through #180 RHS contract and #179 solver.

    The first-order frozen Fock and overlap matrices are analytic. For every
    nuclear perturbation we build the mandatory metric-density Fock response,
    call build_rhf_nuclear_rhs, and solve A x = -b with #179 GMRES.

    x is the nonredundant symmetric-gauge response. The actual occupied
    MO-coefficient derivative is U_ai = x_ia.T - S_ai/2 and
    U_ij = -S_ij/2. Relaxation blocks for both (R,S) and (S,R) are evaluated
    independently; no triangular mirroring or post-hoc symmetrization occurs.
    """
    _validate_analytic_domain(s)
    C, eps = s.C, s.eps
    nocc = s.nocc
    occ = s.occ
    mocc = C[:, occ]
    e_i = eps[occ]
    nat, nbf = s.nat, s.nbf

    h1ao, s1ao_all = _analytic_first_order_inputs(s)
    ref = build_reference(s)
    backend = NativeJKBackend(s.source)
    op = RHFResponseOperator(RHFResponseOperator.build_problem(ref, backend), backend)
    mo1s = np.zeros((nat, 3, nbf, nocc))
    e1s = np.zeros((nat, 3, nocc, nocc))
    for ia in range(nat):
        for x in range(3):
            response = solve_rhf_nuclear_perturbation(op, h1ao[ia, x], s1ao_all[ia, x])
            mo1s[ia, x] = response.coefficient_derivative
            e1s[ia, x] = response.occupied_energy_derivative

    # Raw assembly: compute every perturbation ordering independently.
    relax = np.zeros((nat, nat, 3, 3))
    s1oo = np.einsum(
        "axpq,pi,qj->axij",
        s1ao_all,
        mocc,
        mocc,
    )
    for ia in range(nat):
        for ja in range(nat):
            for x in range(3):
                for y in range(3):
                    dm1 = mo1s[ja, y] @ mocc.T
                    dm1e = (mo1s[ja, y] * e_i[None, :]) @ mocc.T
                    relax[ia, ja, x, y] += np.einsum("pq,pq->", h1ao[ia, x], dm1) * 4
                    relax[ia, ja, x, y] -= (
                        np.einsum("pq,pq->", s1ao_all[ia, x], dm1e) * 4
                    )
                    relax[ia, ja, x, y] -= (
                        np.einsum("ij,ij->", s1oo[ia, x], e1s[ja, y]) * 2
                    )
    return relax


# ---------------------------------------------------------------------------
# top-level assembly + validation
# ---------------------------------------------------------------------------


def analytic_hessian(s: typing.Any, *, relax: typing.Any = None) -> typing.Any:
    """Return every component plus the total from the shared #178/#179 layers.

    This diagnostic integration is bounded to 12 AOs. A supplied relaxation
    must be a finite real (natoms, natoms, 3, 3) tensor.

    ``core``/``pulay``/``two_electron`` come from the #178 second-integral
    providers, ``nuclear`` from the closed-form Coulomb second derivative, and
    ``relaxation`` from #179's matrix-free RHF operator (or the supplied
    ``relax`` tensor, e.g. to swap in the FD oracle).
    """
    _validate_analytic_domain(s)
    if relax is not None:
        relax = np.asarray(relax)
        if (
            relax.shape != (s.nat, s.nat, 3, 3)
            or relax.dtype.kind not in "biuf"
            or not np.isfinite(relax).all()
        ):
            raise ValueError(
                "relaxation must be finite real with shape (natoms, natoms, 3, 3)"
            )
    comp = provider_components(s)
    comp["nuclear"] = nuclear_closed_form(s)
    comp["relaxation"] = relax if relax is not None else cphf_relaxation(s)
    total = (
        comp["nuclear"]
        + comp["core"]
        + comp["pulay"]
        + comp["two_electron"]
        + comp["relaxation"]
    )
    comp["total"] = total
    return comp

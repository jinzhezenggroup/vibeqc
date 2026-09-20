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
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.resources import ResourceBudget
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
    key: object,
    build_ir: object,
    ir_extra: object,
    adapter: object,
    cache: object,
    output_indices: object,
    component_indices: object,
) -> object:
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


def _tile_components(count: object, chunk: object = 64) -> object:
    for start in range(0, count, chunk):
        yield tuple(range(start, min(start + chunk, count)))


def _scatter(
    full: np.ndarray,
    ci: tuple[int, ...],
    center_atoms: tuple[int, ...],
    data: dict[str, object],
) -> np.ndarray:
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
    full: np.ndarray,
    ci: tuple[int, ...],
    center_atoms: tuple[int, ...],
    data: dict[str, object],
) -> np.ndarray:
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
    data: dict[str, object],
    key: object,
    build_ir: object,
    ir_extra: dict[str, object],
    adapter: object,
    cache: Path,
    prims: tuple[object, ...],
    centers: np.ndarray,
    weight_full_flat: np.ndarray,
    component_count: int,
    *,
    direction: np.ndarray | None = None,
) -> np.ndarray:
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
            with PreparedSecondDerivative(
                art,
                record_capacity=8,
                budget=data["resource_budget"],
                device_id=data["device_id"],
            ) as plan:
                r = plan.contract(stream, profile=True)
            data["second_executions"].append(r.diagnostics)
            full[list(oi)] += np.asarray(r.values).sum(axis=0)
    if hvp:
        return _scatter_hvp(full, ci, ca, data)
    return _scatter(full, ci, ca, data)


def _checked_second_hvp_options(
    backend: str, compiler: object, device_id: int, budget_bytes: int
) -> object:
    """Validate one explicit #178 weighted-HVP execution backend."""
    if backend not in ("cpu", "cuda"):
        raise ValueError("second-integral backend must be cpu or cuda")
    if type(budget_bytes) is not int or not 0 < budget_bytes < 2**63:
        raise ValueError("second-integral budget must be a positive int64")
    if backend == "cuda":
        from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter

        if not isinstance(compiler, CudaCompilerAdapter):
            raise TypeError(
                "CUDA second-integral HVPs require an explicit CudaCompilerAdapter"
            )
        if type(device_id) is not int or not 0 <= device_id < 2**31:
            raise ValueError("CUDA second-integral device_id must be a nonnegative int")
        resource_budget = ResourceBudget(
            host_bytes=budget_bytes,
            device_bytes=budget_bytes,
            per_device_bytes=((device_id, budget_bytes),),
        )
        return compiler, device_id, resource_budget
    if compiler is not None:
        raise ValueError("second-integral compiler is only meaningful for CUDA")
    return (
        CppCompilerAdapter(Path(shutil.which("c++") or "c++")),
        0,
        ResourceBudget(host_bytes=budget_bytes, device_bytes=0),
    )


def _provider_data(
    s: NativeRHFState,
    *,
    backend: str = "cpu",
    compiler: object = None,
    device_id: int = 0,
    budget_bytes: int = 64 << 20,
) -> dict[str, object]:
    _validate_analytic_domain(s)
    adapter, provider_device, resource_budget = _checked_second_hvp_options(
        backend, compiler, device_id, budget_bytes
    )
    C, eps = s.C, s.eps
    W_e = (C[:, : s.nocc] * (2 * eps[: s.nocc])) @ C[:, : s.nocc].T
    return {
        "state": s,
        "shells": s.source.shells,
        "primitives": s.primitives,
        "adapter": adapter,
        "cache": s.cache / f"second-cache-{backend}",
        "W_e": W_e,
        "density": s.P0,
        "backend": backend,
        "device_id": provider_device,
        "budget_bytes": budget_bytes,
        "resource_budget": resource_budget,
        "second_executions": [],
    }


def _second_provider_diagnostics(data: dict[str, object]) -> dict[str, object]:
    """Summarize the generated #178 provider without claiming hidden residency."""
    executions = data["second_executions"]
    timing_names = ("device_ms", "input_ms", "output_ms", "kernel_ms")
    timing = {
        name: sum(
            (item.get("device_timing") or {}).get(name, 0.0) for item in executions
        )
        for name in timing_names
    }
    chunks = sum(item["chunks"] for item in executions)
    cuda = data["backend"] == "cuda"
    return {
        "backend": f"{data['backend']}-generated-weighted-hvp",
        "provider_backend": data["backend"],
        "device_id": data["device_id"] if cuda else None,
        "budget_bytes": data["budget_bytes"],
        "program_identities": tuple(
            sorted({item["program_identity"] for item in executions})
        ),
        "native_artifacts": tuple(
            sorted({item["native_artifact"] for item in executions})
        ),
        "executions": len(executions),
        "primitive_records": sum(item["records"] for item in executions),
        "record_batches": chunks,
        "record_batch_uploads": chunks if cuda else 0,
        "result_tile_downloads": chunks if cuda else 0,
        "raw_hessian_downloads": 0,
        "intermediate_matrix_downloads": 0,
        "peak_host_bytes": max(
            (item["resources"]["peak_bytes"].get("host", 0) for item in executions),
            default=0,
        ),
        "peak_device_bytes": max(
            (item["resources"]["peak_bytes"].get("device", 0) for item in executions),
            default=0,
        ),
        "device_timing_ms": timing if cuda else None,
    }


def _run_one_electron(
    data: dict[str, object],
    family: str,
    weight: np.ndarray,
    *,
    direction: np.ndarray | None = None,
) -> np.ndarray:
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
    data: dict[str, object], density: np.ndarray, *, direction: np.ndarray | None = None
) -> np.ndarray:
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


def provider_components(s: NativeRHFState) -> dict[str, np.ndarray]:
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


def provider_hvp_components(
    s: NativeRHFState,
    direction: np.ndarray,
    *,
    backend: str = "cpu",
    compiler: object = None,
    device_id: int = 0,
    budget_bytes: int = 64 << 20,
    return_diagnostics: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return frozen-skeleton #178 HVP components on an explicit backend.

    CUDA reuses the already-qualified generated second-derivative provider. It
    streams packed primitive records to bounded device storage and downloads
    only contracted coordinate HVP tiles; no raw integral Hessian is published.
    """
    _validate_analytic_domain(s)
    vector = checked_direction(direction, s.nat)
    data = _provider_data(
        s,
        backend=backend,
        compiler=compiler,
        device_id=device_id,
        budget_bytes=budget_bytes,
    )
    p0 = s.P0
    core = _run_one_electron(data, "kinetic", p0, direction=vector) + _run_one_electron(
        data, "nuclear_attraction", p0, direction=vector
    )
    pulay = -_run_one_electron(data, "overlap", data["W_e"], direction=vector)
    two_electron = _run_eri(data, data["density"], direction=vector)
    components = {"core": core, "pulay": pulay, "two_electron": two_electron}
    if return_diagnostics:
        return components, _second_provider_diagnostics(data)
    return components


# ---------------------------------------------------------------------------
# closed-form nucleus-nucleus second derivative
# ---------------------------------------------------------------------------


def nuclear_closed_form(s: NativeRHFState) -> np.ndarray:
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


def nuclear_hvp(s: NativeRHFState, direction: np.ndarray) -> np.ndarray:
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


def build_reference(s: NativeRHFState) -> object:
    """Reuse the caller's validated native snapshot without rerunning SCF."""
    _validate_analytic_domain(s)
    return s.reference


def _analytic_first_order_inputs(s: NativeRHFState) -> tuple[np.ndarray, np.ndarray]:
    """Generated S/T/V/ERI contractions, tied to the same native SCF density."""
    _validate_analytic_domain(s)
    return s.first_order_inputs


def _validate_analytic_domain(s: NativeRHFState) -> None:
    if not isinstance(s, NativeRHFState):
        raise TypeError(
            "analytic Hessian requires NativeRHFState, not an oracle System"
        )
    s.validate()


def cphf_relaxation(s: NativeRHFState) -> np.ndarray:
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


def analytic_hessian(
    s: NativeRHFState, *, relax: np.ndarray | None = None
) -> dict[str, np.ndarray]:
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

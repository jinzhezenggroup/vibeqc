"""Compiler-generated perturbative-triples response sources for #154 A.

The standard closed-shell (T) energy already has one audited TensorIR owner in
tools.vibeqc_cc.triples_tiles. This module deliberately does not add hand-written
derivative algebra. It asks the common #151 TensorIR reverse generator for VJPs
of the existing tile energy and provides the minimal scatter needed to
accumulate prefix-bounded tile cotangents into full input tensors.

Two different notions of duplicate work matter here:

* Mathematical contributions are not duplicated: TriplesTileEnumerator
  partitions the triangular a>=b>=c energy domain. By linearity, the VJP of the
  full energy is exactly the sum of the VJPs of those disjoint energy tiles.
  Tests compare that identity directly against the untiled primal VJP.
* Arithmetic recomputation is a compiler scheduling decision. Exact CSE may
  share repeated expressions; bounded CUDA schedules may instead recompute
  intermediates to reduce peak memory. Such recomputation changes cost, not
  mathematical multiplicity, and is reported by the TensorIR planner.

This is a fixed-input response frontend only. Corrected Lambda, orbital
response, complete nuclear gradients and public force capabilities remain
outside this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np
from vibeqc_compiler.tensor import execute, optimize, transpose_program

from .triples import _check_denominators, _validate, build_triples_program
from .triples_tiles import (
    TriplesTileEnumerator,
    _tile_input_feeds,
    build_tile_triples_program,
)

TRIPLES_RESPONSE_INPUTS = (
    "t1",
    "t2",
    "ovvv",
    "ovoo",
    "ovov",
    "fov",
    "eps_o",
    "eps_v",
)

_INPUT_ORDER = ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")


def _selected_inputs(inputs: Iterable[str] | None) -> tuple[str, ...]:
    selected = TRIPLES_RESPONSE_INPUTS if inputs is None else tuple(inputs)
    if not selected:
        raise ValueError("at least one triples response input is required")
    if len(set(selected)) != len(selected):
        raise ValueError("triples response inputs must be unique")
    unknown = set(selected) - set(TRIPLES_RESPONSE_INPUTS)
    if unknown:
        raise ValueError(f"unknown triples response inputs: {sorted(unknown)}")
    return selected


def build_tile_triples_vjp(
    nocc: int,
    nvir: int,
    *,
    vir_chunk: tuple[int, int] | None = None,
    inputs: Iterable[str] | None = None,
    max_elements: int = 1_000_000,
) -> Any:
    """Generate one demand-driven VJP from the existing bounded (T) primal."""

    primal = build_tile_triples_program(nocc, nvir, vir_chunk=vir_chunk)
    return transpose_program(
        primal,
        ("triples_energy",),
        inputs=_selected_inputs(inputs),
        max_elements=max_elements,
    )


def build_full_triples_vjp(
    nocc: int,
    nvir: int,
    *,
    inputs: Iterable[str] | None = None,
    max_elements: int = 1_000_000,
) -> Any:
    """Generate the untiled tiny-reference VJP used to verify tile accumulation."""

    primal = build_triples_program(nocc, nvir)
    return transpose_program(
        primal,
        ("triples_energy",),
        inputs=_selected_inputs(inputs),
        max_elements=max_elements,
    )


def _arrays(
    ovvv: np.ndarray,
    ovoo: np.ndarray,
    ovov: np.ndarray,
    fov: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "ovvv": ovvv,
        "ovoo": ovoo,
        "ovov": ovov,
        "fov": fov,
        "t1": t1,
        "t2": t2,
        "eps_o": eps_o,
        "eps_v": eps_v,
    }


def tile_triples_vjp(
    nocc: int,
    nvir: int,
    ovvv: np.ndarray,
    ovoo: np.ndarray,
    ovov: np.ndarray,
    fov: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    *,
    vir_chunk: tuple[int, int] | None = None,
    inputs: Iterable[str] | None = None,
    denominator_threshold: float = 1e-10,
    optimize_graph: bool = False,
    max_elements: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Execute a compiler-generated unit-seeded VJP for one (T) energy tile."""

    _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    _check_denominators(eps_o, eps_v, denominator_threshold)
    selected = _selected_inputs(inputs)
    derivative = build_tile_triples_vjp(
        nocc,
        nvir,
        vir_chunk=vir_chunk,
        inputs=selected,
        max_elements=max_elements,
    )
    program = optimize(derivative.program) if optimize_graph else derivative.program
    a_end = nvir if vir_chunk is None else vir_chunk[1]
    feeds = _tile_input_feeds(
        _arrays(ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v),
        a_end,
    )
    result = execute(
        program,
        {**feeds, "bar_triples_energy": np.asarray(1.0, dtype=np.float64)},
    ).outputs
    return {name: np.asarray(result[f"bar_{name}"]) for name in selected}


def _scatter_prefix(full: np.ndarray, local: np.ndarray, name: str, a_end: int) -> None:
    """Accumulate one tile cotangent into the matching full-input coordinates."""

    if name == "ovvv":
        full[:, :a_end, :, :a_end] += local
    elif name == "ovoo":
        full[:, :a_end, :, :] += local
    elif name == "ovov":
        full[:, :a_end, :, :a_end] += local
    elif name in ("fov", "t1"):
        full[:, :a_end] += local
    elif name == "t2":
        full[:, :, :a_end, :] += local
    elif name == "eps_v":
        full[:a_end] += local
    elif name == "eps_o":
        full[...] += local
    else:
        raise ValueError(f"no triples response scatter rule for {name}")


def accumulate_tile_triples_vjp(
    nocc: int,
    nvir: int,
    ovvv: np.ndarray,
    ovoo: np.ndarray,
    ovov: np.ndarray,
    fov: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    *,
    vir_chunk_size: int | None = None,
    inputs: Iterable[str] | None = None,
    denominator_threshold: float = 1e-10,
    optimize_graph: bool = False,
    max_elements: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Sum VJPs of disjoint (T) energy tiles into full response tensors.

    Prefix input regions overlap between tiles because a later energy tile may
    depend on an earlier virtual label. Their derivative contributions must
    therefore be added in those shared coordinates. This is accumulation of
    the derivative of a disjoint energy partition, not duplicate counting.
    """

    _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    _check_denominators(eps_o, eps_v, denominator_threshold)
    selected = _selected_inputs(inputs)
    arrays = _arrays(ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    totals = {
        name: np.zeros_like(np.asarray(arrays[name]), dtype=np.float64)
        for name in selected
    }
    enumerator = TriplesTileEnumerator(nocc, nvir, vir_chunk_size=vir_chunk_size)
    for tile in enumerator:
        local = tile_triples_vjp(
            nocc,
            nvir,
            ovvv,
            ovoo,
            ovov,
            fov,
            t1,
            t2,
            eps_o,
            eps_v,
            vir_chunk=(tile.a_start, tile.a_end),
            inputs=selected,
            denominator_threshold=denominator_threshold,
            optimize_graph=optimize_graph,
            max_elements=max_elements,
        )
        for name in selected:
            _scatter_prefix(totals[name], local[name], name, tile.a_end)
    return totals


def full_triples_vjp(
    nocc: int,
    nvir: int,
    ovvv: np.ndarray,
    ovoo: np.ndarray,
    ovov: np.ndarray,
    fov: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    *,
    inputs: Iterable[str] | None = None,
    denominator_threshold: float = 1e-10,
    optimize_graph: bool = False,
    max_elements: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Execute the untiled VJP oracle for tiny validation cases."""

    _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    _check_denominators(eps_o, eps_v, denominator_threshold)
    selected = _selected_inputs(inputs)
    derivative = build_full_triples_vjp(
        nocc,
        nvir,
        inputs=selected,
        max_elements=max_elements,
    )
    program = optimize(derivative.program) if optimize_graph else derivative.program
    feeds = _arrays(ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    result = execute(
        program,
        {**feeds, "bar_triples_energy": np.asarray(1.0, dtype=np.float64)},
    ).outputs
    return {name: np.asarray(result[f"bar_{name}"]) for name in selected}

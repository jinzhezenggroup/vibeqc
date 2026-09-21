"""Bounded triples tile enumerator and per-tile CPU reference (CG11 slice B).

Tile decomposition of the (T) energy over the triangular virtual domain
``a >= b >= c``: a-chunks partition the outermost loop, while the occupied
contraction stays full within each tile.  The enumerator produces exactly one
tile descriptor per a-chunk so that every ``(a,b,c)`` triple is counted once
and the tile sum equals :func:`triples_energy` to machine precision.

No DIIS, no approximated denominator, no GPU dependency here.
"""

import typing
from fractions import Fraction

import numpy as np
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    divide,
    einsum,
    execute,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    runtime_indexed_select,
    transpose,
)

from .triples import (
    _LABELS,
    INVENTORY_HASH,
    OP,
    R3,
    SLOW_TABLE,
    VERSION,
    VP,
    _check_denominators,
    _degeneracy,
    _permuted,
    _v,
    _validate,
    _views,
    _w,
    r3,
)

# ---------------------------------------------------------------------------
# Tile descriptor and enumerator
# ---------------------------------------------------------------------------

DESCRIPTION = "bounded triples tiles: vir chunk by a-range, occ full within each tile"


class TileSpec:
    """One tile descriptor: (a_start, a_end) for the virtual chunk.

    The occupied space is always full within the tile.  The sub-block for
    this tile includes virtual indices [0, a_end) in the input tensors,
    but the triangular loops only process a in [a_start, a_end).
    """

    __slots__ = ("a_end", "a_start", "nvir")

    def __init__(
        self, a_start: typing.Any, a_end: typing.Any, nvir: typing.Any
    ) -> None:
        if not (0 <= a_start < a_end <= nvir):
            raise ValueError(f"invalid a-chunk [{a_start}, {a_end}) for nvir={nvir}")
        self.a_start = a_start
        self.a_end = a_end
        self.nvir = nvir

    def __repr__(self) -> typing.Any:
        return f"TileSpec(a=[{self.a_start},{self.a_end})/{self.nvir})"

    def __eq__(self, other: object) -> typing.Any:
        if not isinstance(other, TileSpec):
            return NotImplemented
        return (self.a_start, self.a_end, self.nvir) == (
            other.a_start,
            other.a_end,
            other.nvir,
        )

    def __hash__(self) -> typing.Any:
        return hash((self.a_start, self.a_end, self.nvir))

    @property
    def vir_subblock_size(self) -> typing.Any:
        return self.a_end

    @property
    def ntriples(self) -> typing.Any:
        """Count of ``a>=b>=c`` triples processed in this tile."""
        n = 0
        for a in range(self.a_start, self.a_end):
            n += (a + 1) * (a + 2) // 2
        return n

    def __iter__(self) -> typing.Any:
        """Yield each ``(a,b,c)`` triple with ``a>=b>=c`` in this tile."""
        for a in range(self.a_start, self.a_end):
            for b in range(a + 1):
                for c in range(b + 1):
                    yield a, b, c


class TriplesTileEnumerator:
    """Deterministic tile enumeration for the (T) virtual triple space.

    Parameters
    ----------
    nocc, nvir : int
        Full occupied and virtual dimensions.
    vir_chunk_size : int
        Maximum number of ``a`` values per tile.  Each tile covers
        ``vir_chunk_size`` consecutive ``a`` values; the last tile may
        be smaller.  Occupied space is never chunked — each tile uses
        the full ``nocc`` range.
    """

    def __init__(
        self, nocc: typing.Any, nvir: typing.Any, *, vir_chunk_size: typing.Any = None
    ) -> None:
        if any(type(n) is not int or n < 1 for n in (nocc, nvir)):
            raise ValueError("triples require nonempty occupied and virtual spaces")
        if vir_chunk_size is None:
            vir_chunk_size = nvir
        if type(vir_chunk_size) is not int or vir_chunk_size < 1:
            raise ValueError("vir_chunk_size must be a positive integer")
        self.nocc = nocc
        self.nvir = nvir
        self.vir_chunk_size = vir_chunk_size

    def __iter__(self) -> typing.Any:
        """Yield :class:`TileSpec` for each a-chunk."""
        nvir = self.nvir
        chunk = self.vir_chunk_size
        for a_start in range(0, nvir, chunk):
            a_end = min(a_start + chunk, nvir)
            yield TileSpec(a_start, a_end, nvir)

    def __len__(self) -> typing.Any:
        return (self.nvir + self.vir_chunk_size - 1) // self.vir_chunk_size

    def tiles(self) -> typing.Any:
        return list(self)


# ---------------------------------------------------------------------------
# Per-tile CPU reference
# ---------------------------------------------------------------------------


def tile_triples_energy(
    tile: typing.Any,
    nocc: typing.Any,
    ovvv: typing.Any,
    ovoo: typing.Any,
    ovov: typing.Any,
    fov: typing.Any,
    t1: typing.Any,
    t2: typing.Any,
    eps_o: typing.Any,
    eps_v: typing.Any,
    *,
    denominator_threshold: typing.Any = 1e-10,
) -> typing.Any:
    """CPU reference: (T) contribution of a single tile.

    Identical to :func:`triples_energy` but restricted to virtual triples
    whose ``a`` index falls in the tile's a-range.  The sub-block always
    includes virtual indices [0, a_end) from the full input tensors.

    The occupied space is processed in full.  This function is the per-tile
    ground truth for GPU-vs-CPU comparison (gate ≤ 1e-10).
    """
    nvir_full = len(eps_v)
    _validate(nocc, nvir_full, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    views = _views(ovvv, ovoo, ovov, fov, t1, t2)
    eijk = eps_o[:, None, None] + eps_o[None, :, None] + eps_o[None, None, :]
    et = 0.0
    for a in range(tile.a_start, tile.a_end):
        for b in range(a + 1):
            for c in range(b + 1):
                d3_val = eijk - eps_v[a] - eps_v[b] - eps_v[c]
                if denominator_threshold is not None:
                    if np.any(d3_val >= 0):
                        raise ValueError("noncanonical (T) denominator in tile")
                    if np.min(np.abs(d3_val)) <= denominator_threshold:
                        raise ValueError("near-zero (T) denominator in tile")
                d3 = d3_val * _degeneracy(a, b, c)
                ws = {lbl: _w(views, *_permuted((a, b, c), VP[lbl])) for lbl in _LABELS}
                vs = {lbl: _v(views, *_permuted((a, b, c), VP[lbl])) for lbl in _LABELS}
                zs = {lbl: r3(ws[lbl] + 0.5 * vs[lbl]) / d3 for lbl in _LABELS}
                for zlbl, row in SLOW_TABLE.items():
                    for wlbl, ost in row:
                        et += np.einsum(
                            "ijk,ijk", ws[wlbl].transpose(OP[ost]), zs[zlbl]
                        )
    return et * 2


def tile_triples_energy_masked(
    tile: typing.Any,
    nocc: typing.Any,
    ovvv_full: typing.Any,
    ovoo_full: typing.Any,
    ovov_full: typing.Any,
    fov_full: typing.Any,
    t1_full: typing.Any,
    t2_full: typing.Any,
    eps_o: typing.Any,
    eps_v: typing.Any,
) -> typing.Any:
    """CPU reference for per-tile comparison: zero out triples outside the tile.

    Unlike :func:`tile_triples_energy` which only loops over the tile's
    a-range, this function runs the *full* triangular sum but zeros out
    the contribution of every ``(a,b,c)`` triple whose ``a`` is outside
    ``[tile.a_start, tile.a_end)``.  This produces a ``per_tile_masked``
    value that is the exact counterpart of the CUDA tile scalar — same
    occupied space, same full input tensors, just restricted a-range.
    """
    nvir = len(eps_v)
    _validate(
        nocc,
        nvir,
        ovvv_full,
        ovoo_full,
        ovov_full,
        fov_full,
        t1_full,
        t2_full,
        eps_o,
        eps_v,
    )
    views = _views(ovvv_full, ovoo_full, ovov_full, fov_full, t1_full, t2_full)
    eijk = eps_o[:, None, None] + eps_o[None, :, None] + eps_o[None, None, :]
    et = 0.0
    for a in range(nvir):
        for b in range(a + 1):
            for c in range(b + 1):
                if a < tile.a_start or a >= tile.a_end:
                    continue  # zero-out: skip this triple
                d3_val = eijk - eps_v[a] - eps_v[b] - eps_v[c]
                d3 = d3_val * _degeneracy(a, b, c)
                ws = {lbl: _w(views, *_permuted((a, b, c), VP[lbl])) for lbl in _LABELS}
                vs = {lbl: _v(views, *_permuted((a, b, c), VP[lbl])) for lbl in _LABELS}
                zs = {lbl: r3(ws[lbl] + 0.5 * vs[lbl]) / d3 for lbl in _LABELS}
                for zlbl, row in SLOW_TABLE.items():
                    for wlbl, ost in row:
                        et += np.einsum(
                            "ijk,ijk", ws[wlbl].transpose(OP[ost]), zs[zlbl]
                        )
    return et * 2


# ---------------------------------------------------------------------------
# Tiny-TensorIR tile program (same inventory; differentiable; tile-sized)
# ---------------------------------------------------------------------------


def _t_views_tile(nodes: typing.Any) -> typing.Any:
    return {
        "t1T": transpose(nodes["t1"], (1, 0)),
        "t2T": transpose(nodes["t2"], (2, 3, 0, 1)),
        "vvov": transpose(nodes["ovvv"], (1, 3, 0, 2)),
        "vooo": transpose(nodes["ovoo"], (1, 0, 2, 3)),
        "vvoo": transpose(nodes["ovov"], (1, 3, 0, 2)),
        "fvo": transpose(nodes["fov"], (1, 0)),
    }


def _fix_tile(node: typing.Any, *selections: typing.Any) -> typing.Any:
    axes = []
    for axis, pos in selections:
        node = gather(node, axis, [pos])
        axes.append(axis)
    return reduce_sum(node, tuple(sorted(axes)))


def _w_node_tile(
    v: typing.Any, a: typing.Any, b: typing.Any, c: typing.Any
) -> typing.Any:
    ab = _fix_tile(v["vvov"], (0, a), (1, b))
    cc = _fix_tile(v["t2T"], (0, c))
    w1 = einsum("if,fkj->ijk", ab, cc)
    a0 = _fix_tile(v["vooo"], (0, a))
    bc = _fix_tile(v["t2T"], (0, b), (1, c))
    w2 = einsum("ijm,mk->ijk", a0, bc)
    return add(w1, w2, coefficients=(1, -1))


def _v_node_tile(
    v: typing.Any, a: typing.Any, b: typing.Any, c: typing.Any
) -> typing.Any:
    ab = _fix_tile(v["vvoo"], (0, a), (1, b))
    cc = _fix_tile(v["t1T"], (0, c))
    v1 = einsum("ij,k->ijk", ab, cc)
    ab2 = _fix_tile(v["t2T"], (0, a), (1, b))
    c2 = _fix_tile(v["fvo"], (0, c))
    v2 = einsum("ij,k->ijk", ab2, c2)
    return add(v1, v2, coefficients=(1, 1))


def _r3_node_tile(w: typing.Any) -> typing.Any:
    return add(
        *(transpose(w, perm) for _, perm in R3),
        coefficients=tuple(c for c, _ in R3),
    )


def _d3_node_tile(
    nodes: typing.Any,
    ijk: typing.Any,
    a: typing.Any,
    b: typing.Any,
    c: typing.Any,
    fac: typing.Any,
) -> typing.Any:
    e0 = broadcast(nodes["eps_o"], ijk, (0,))
    e1 = broadcast(nodes["eps_o"], ijk, (1,))
    e2 = broadcast(nodes["eps_o"], ijk, (2,))
    eijk = add(e0, e1, e2)
    ev = add(
        reduce_sum(gather(nodes["eps_v"], 0, [a]), (0,)),
        reduce_sum(gather(nodes["eps_v"], 0, [b]), (0,)),
        reduce_sum(gather(nodes["eps_v"], 0, [c]), (0,)),
    )
    return add(eijk, broadcast(ev, ijk, ()), coefficients=(fac, -fac))


def build_tile_triples_program(
    nocc: typing.Any, nvir: typing.Any, *, vir_chunk: typing.Any = None
) -> typing.Any:
    """Lower the (T) inventory for a tile to unshared TensorIR.

    ``nvir`` is the full virtual population.  The tile covers triangular
    ``a>=b>=c`` values for ``a`` in ``vir_chunk``; all label coordinates
    therefore lie in the prefix ``[0, a_end)``.

    Tensor specs use two distinct virtual index spaces: bounded label axes of
    extent ``a_end`` and a full summation axis of extent ``nvir``.  This keeps
    the W1 ``f`` contraction exact while allowing resident uploads to use
    exact prefix-shaped label tensors.
    """
    if any(type(n) is not int or n < 1 for n in (nocc, nvir)):
        raise ValueError("triples require nonempty occupied and virtual spaces")
    a_start = 0
    a_end = nvir
    if vir_chunk is not None:
        a_start, a_end = vir_chunk
        if not (0 <= a_start < a_end <= nvir):
            raise ValueError(f"invalid vir_chunk {vir_chunk} for nvir={nvir}")
    occ = IndexSpace("occupied", "occupied", nocc)
    label_vir = IndexSpace("tile_virtual", "virtual", a_end)
    sum_vir = IndexSpace("full_virtual", "virtual", nvir)

    def O(name: typing.Any) -> typing.Any:
        return Index(name, occ)

    def V(name: typing.Any) -> typing.Any:
        return Index(name, label_vir)

    def F(name: typing.Any) -> typing.Any:
        return Index(name, sum_vir)

    common = {
        "role": "parameter",
        "differentiable": True,
        "representation": "restricted_spatial",
    }
    nodes = {
        # ovvv.transpose(1, 3, 0, 2) is vvov[a,b,i,f], so original
        # axis 2 is the full f-summation axis while axes 1/3 are labels.
        "ovvv": input_tensor(
            "ovvv", TensorSpec((O("i0"), V("a0"), F("f0"), V("b0")), **common)
        ),
        "ovoo": input_tensor(
            "ovoo", TensorSpec((O("i1"), V("a1"), O("j1"), O("m1")), **common)
        ),
        "ovov": input_tensor(
            "ovov", TensorSpec((O("i2"), V("a2"), O("j2"), V("b2")), **common)
        ),
        "fov": input_tensor("fov", TensorSpec((O("k3"), V("c3")), **common)),
        "t1": input_tensor("t1", TensorSpec((O("i4"), V("a4")), **common)),
        # t2T[c,f,k,j] requires the second virtual t2 axis to remain full;
        # W2/V2 may still gather label coordinates from that full prefix.
        "t2": input_tensor(
            "t2", TensorSpec((O("i5"), O("j5"), V("a5"), F("f5")), **common)
        ),
        "eps_o": input_tensor("eps_o", TensorSpec((O("i6"),), **common)),
        "eps_v": input_tensor("eps_v", TensorSpec((V("a7"),), **common)),
    }
    ijk = (O("io"), O("jo"), O("ko"))
    views = _t_views_tile(nodes)

    scalar_terms = []
    for a in range(a_start, a_end):
        for b in range(a + 1):
            for c in range(b + 1):
                d3 = _d3_node_tile(nodes, ijk, a, b, c, _degeneracy(a, b, c))
                ws = {
                    lbl: _w_node_tile(views, *_permuted((a, b, c), VP[lbl]))
                    for lbl in _LABELS
                }
                vs = {
                    lbl: _v_node_tile(views, *_permuted((a, b, c), VP[lbl]))
                    for lbl in _LABELS
                }
                halves = Fraction(1, 2)
                zs = {
                    lbl: divide(
                        _r3_node_tile(add(ws[lbl], vs[lbl], coefficients=(1, halves))),
                        d3,
                    )
                    for lbl in _LABELS
                }
                for zlbl, row in SLOW_TABLE.items():
                    for wlbl, ost in row:
                        scalar_terms.append(
                            einsum(
                                "ijk,ijk->",
                                transpose(ws[wlbl], OP[ost]),
                                zs[zlbl],
                            )
                        )
    et = add(*scalar_terms)
    energy = add(et, coefficients=(2,))
    return Program(
        {"triples_energy": energy},
        provenance={
            "method": "RCCSD(T)",
            "slice": "B",
            "inventory_version": VERSION,
            "inventory_hash": INVENTORY_HASH,
            "source": "PySCF 2.14.0 ccsd_t_slow.py kernel + r3; see source_manifest.json",
            "note": (
                f"bounded tile TensorIR reference; nocc={nocc}, "
                f"label_nvir={a_end}, sum_nvir={nvir}, "
                f"a_range={a_start}:{a_end}"
            ),
        },
    )


def _tile_input_feeds(arrays: typing.Any, a_end: typing.Any) -> typing.Any:
    """Extract exact-shape TensorIR feeds for a prefix-bounded tile.

    Label axes use ``[0, a_end)``.  The W1 ``f`` summation stays full:
    ``ovvv`` axis 2 and ``t2`` axis 3 retain the complete virtual population.
    """
    ovvv = arrays["ovvv"]
    ovoo = arrays["ovoo"]
    ovov = arrays["ovov"]
    fov = arrays["fov"]
    t1 = arrays["t1"]
    t2 = arrays["t2"]
    eps_o = arrays["eps_o"]
    eps_v = arrays["eps_v"]
    return {
        "ovvv": np.ascontiguousarray(ovvv[:, :a_end, :, :a_end]),
        "ovoo": np.ascontiguousarray(ovoo[:, :a_end, :, :]),
        "ovov": np.ascontiguousarray(ovov[:, :a_end, :, :a_end]),
        "fov": np.ascontiguousarray(fov[:, :a_end]),
        "t1": np.ascontiguousarray(t1[:, :a_end]),
        "t2": np.ascontiguousarray(t2[:, :, :a_end, :]),
        "eps_o": np.ascontiguousarray(eps_o),
        "eps_v": np.ascontiguousarray(eps_v[:a_end]),
    }


# ---------------------------------------------------------------------------\n# Runtime-indexed TensorIR tile program (#783)\n# ---------------------------------------------------------------------------\n\n\ndef runtime_tile_capacity(\n    nocc: typing.Any, nvir: typing.Any, vir_chunk_size: typing.Any\n) -> int:\n    """Maximum triangular-domain lanes needed by one runtime a-chunk."""\n    tiles = tuple(\n        TriplesTileEnumerator(nocc, nvir, vir_chunk_size=vir_chunk_size)\n    )\n    return max(tile.ntriples for tile in tiles)\n\n\ndef runtime_tile_controls(tile: TileSpec, capacity: int) -> dict[str, np.ndarray]:\n    """Build bounded runtime maps/masks without changing program identity."""\n    if type(capacity) is not int or capacity < tile.ntriples:\n        raise ValueError("runtime triples capacity is smaller than the tile domain")\n    a_map = np.zeros(capacity, dtype=np.int64)\n    b_map = np.zeros(capacity, dtype=np.int64)\n    c_map = np.zeros(capacity, dtype=np.int64)\n    active = np.zeros(capacity, dtype=np.float64)\n    degeneracy = np.ones(capacity, dtype=np.float64)\n    for lane, (a, b, c) in enumerate(tile):\n        a_map[lane], b_map[lane], c_map[lane] = a, b, c\n        active[lane] = 1.0\n        degeneracy[lane] = float(_degeneracy(a, b, c))\n    return {\n        "a_map": a_map,\n        "b_map": b_map,\n        "c_map": c_map,\n        "active": active,\n        "degeneracy": degeneracy,\n    }\n\n\ndef runtime_tile_static_feeds(arrays: typing.Any) -> dict[str, np.ndarray]:\n    """Contiguous full-system scientific inputs uploaded once per owner."""\n    return {\n        name: np.ascontiguousarray(arrays[name])\n        for name in ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")\n    }\n\n\ndef _runtime_select(\n    value: typing.Any,\n    domain: Index,\n    *selections: tuple[int, typing.Any],\n) -> typing.Any:\n    return runtime_indexed_select(value, selections, domain)\n\n\ndef _runtime_w_node(\n    views: typing.Any, domain: Index, coordinates: tuple[typing.Any, ...]\n) -> typing.Any:\n    a, b, c = coordinates\n    ab = _runtime_select(views["vvov"], domain, (0, a), (1, b))\n    cc = _runtime_select(views["t2T"], domain, (0, c))\n    w1 = einsum("qif,qfkj->qijk", ab, cc)\n    a0 = _runtime_select(views["vooo"], domain, (0, a))\n    bc = _runtime_select(views["t2T"], domain, (0, b), (1, c))\n    w2 = einsum("qijm,qmk->qijk", a0, bc)\n    return add(w1, w2, coefficients=(1, -1))\n\n\ndef _runtime_v_node(\n    views: typing.Any, domain: Index, coordinates: tuple[typing.Any, ...]\n) -> typing.Any:\n    a, b, c = coordinates\n    ab = _runtime_select(views["vvoo"], domain, (0, a), (1, b))\n    cc = _runtime_select(views["t1T"], domain, (0, c))\n    v1 = einsum("qij,qk->qijk", ab, cc)\n    ab2 = _runtime_select(views["t2T"], domain, (0, a), (1, b))\n    c2 = _runtime_select(views["fvo"], domain, (0, c))\n    v2 = einsum("qij,qk->qijk", ab2, c2)\n    return add(v1, v2, coefficients=(1, 1))\n\n\ndef _runtime_r3_node(w: typing.Any) -> typing.Any:\n    return add(\n        *(\n            transpose(w, (0, *(axis + 1 for axis in permutation)))\n            for _, permutation in R3\n        ),\n        coefficients=tuple(coefficient for coefficient, _ in R3),\n    )\n\n\ndef build_runtime_tile_triples_program(\n    nocc: typing.Any, nvir: typing.Any, *, capacity: typing.Any\n) -> Program:\n    """Build one fixed-capacity triples graph reused across runtime tile ranges.\n\n    The graph contains no Python/IR loop over (a,b,c). Runtime int64 maps bind\n    triangular virtual triples to the leading domain axis; all W/V/R3 algebra\n    is vectorized over that domain and reduced only after the scientific body.\n    """\n    if any(type(n) is not int or n < 1 for n in (nocc, nvir, capacity)):\n        raise ValueError("runtime triples require positive occupied/virtual/domain sizes")\n    occ = IndexSpace("runtime_occupied", "occupied", nocc)\n    vir = IndexSpace("runtime_virtual", "virtual", nvir)\n    lanes = IndexSpace("runtime_triples", "batch", capacity)\n\n    def O(name: str) -> Index:\n        return Index(name, occ)\n\n    def V(name: str) -> Index:\n        return Index(name, vir)\n\n    q = Index("q", lanes)\n    common = {\n        "role": "parameter",\n        "differentiable": True,\n        "representation": "restricted_spatial",\n    }\n    nodes = {\n        "ovvv": input_tensor(\n            "ovvv", TensorSpec((O("i0"), V("a0"), V("f0"), V("b0")), **common)\n        ),\n        "ovoo": input_tensor(\n            "ovoo", TensorSpec((O("i1"), V("a1"), O("j1"), O("m1")), **common)\n        ),\n        "ovov": input_tensor(\n            "ovov", TensorSpec((O("i2"), V("a2"), O("j2"), V("b2")), **common)\n        ),\n        "fov": input_tensor("fov", TensorSpec((O("k3"), V("c3")), **common)),\n        "t1": input_tensor("t1", TensorSpec((O("i4"), V("a4")), **common)),\n        "t2": input_tensor(\n            "t2", TensorSpec((O("i5"), O("j5"), V("a5"), V("f5")), **common)\n        ),\n        "eps_o": input_tensor("eps_o", TensorSpec((O("i6"),), **common)),\n        "eps_v": input_tensor("eps_v", TensorSpec((V("a7"),), **common)),\n        "a_map": input_tensor(\n            "a_map", TensorSpec((q,), dtype="int64", role="input")\n        ),\n        "b_map": input_tensor(\n            "b_map", TensorSpec((q,), dtype="int64", role="input")\n        ),\n        "c_map": input_tensor(\n            "c_map", TensorSpec((q,), dtype="int64", role="input")\n        ),\n        "active": input_tensor(\n            "active",\n            TensorSpec((q,), role="input", representation="restricted_spatial"),\n        ),\n        "degeneracy": input_tensor(\n            "degeneracy",\n            TensorSpec((q,), role="input", representation="restricted_spatial"),\n        ),\n    }\n    views = _t_views_tile(nodes)\n    base_maps = (nodes["a_map"], nodes["b_map"], nodes["c_map"])\n    coordinates = {\n        label: tuple(base_maps[position] for position in VP[label])\n        for label in _LABELS\n    }\n    ws = {\n        label: _runtime_w_node(views, q, coordinates[label]) for label in _LABELS\n    }\n    vs = {\n        label: _runtime_v_node(views, q, coordinates[label]) for label in _LABELS\n    }\n\n    ijk = (O("io"), O("jo"), O("ko"))\n    qijk = (q, *ijk)\n    eijk = add(\n        broadcast(nodes["eps_o"], ijk, (0,)),\n        broadcast(nodes["eps_o"], ijk, (1,)),\n        broadcast(nodes["eps_o"], ijk, (2,)),\n    )\n    virtual_sum = add(\n        _runtime_select(nodes["eps_v"], q, (0, nodes["a_map"])),\n        _runtime_select(nodes["eps_v"], q, (0, nodes["b_map"])),\n        _runtime_select(nodes["eps_v"], q, (0, nodes["c_map"])),\n    )\n    denominator = multiply(\n        add(\n            broadcast(eijk, qijk, (1, 2, 3)),\n            broadcast(virtual_sum, qijk, (0,)),\n            coefficients=(1, -1),\n        ),\n        broadcast(nodes["degeneracy"], qijk, (0,)),\n    )\n    halves = Fraction(1, 2)\n    zs = {\n        label: divide(\n            _runtime_r3_node(add(ws[label], vs[label], coefficients=(1, halves))),\n            denominator,\n        )\n        for label in _LABELS\n    }\n    lane_terms = []\n    for zlabel, row in SLOW_TABLE.items():\n        for wlabel, occupied_order in row:\n            axes = (0, *(axis + 1 for axis in OP[occupied_order]))\n            lane_terms.append(\n                einsum("qijk,qijk->q", transpose(ws[wlabel], axes), zs[zlabel])\n            )\n    lane_energy = add(*lane_terms)\n    total = reduce_sum(multiply(lane_energy, nodes["active"]), (0,))\n    energy = add(total, coefficients=(2,))\n    return Program(\n        {"triples_energy": energy},\n        provenance={\n            "method": "RCCSD(T)",\n            "slice": "#783-runtime-indexed",\n            "inventory_version": VERSION,\n            "inventory_hash": INVENTORY_HASH,\n            "runtime_domain_capacity": capacity,\n            "runtime_domain": "triangular a>=b>=c supplied by int64 maps",\n        },\n    )\n\n\ndef runtime_tile_triples_energy_tensorir(\n    nocc: typing.Any,\n    nvir: typing.Any,\n    arrays: typing.Any,\n    *,\n    vir_chunk_size: int = 1,\n    denominator_threshold: typing.Any = 1e-10,\n) -> float:\n    """CPU-reference execution of one reusable runtime-indexed tile graph."""\n    _validate(\n        nocc,\n        nvir,\n        arrays["ovvv"],\n        arrays["ovoo"],\n        arrays["ovov"],\n        arrays["fov"],\n        arrays["t1"],\n        arrays["t2"],\n        arrays["eps_o"],\n        arrays["eps_v"],\n    )\n    _check_denominators(arrays["eps_o"], arrays["eps_v"], denominator_threshold)\n    tiles = tuple(\n        TriplesTileEnumerator(nocc, nvir, vir_chunk_size=vir_chunk_size)\n    )\n    capacity = max(tile.ntriples for tile in tiles)\n    program = build_runtime_tile_triples_program(nocc, nvir, capacity=capacity)\n    static = runtime_tile_static_feeds(arrays)\n    total = 0.0\n    for tile in tiles:\n        feeds = {**static, **runtime_tile_controls(tile, capacity)}\n        total += float(execute(program, feeds).outputs["triples_energy"])\n    return total\n\ndef tile_triples_energy_tensorir(
    nocc: typing.Any,
    nvir: typing.Any,
    ovvv: typing.Any,
    ovoo: typing.Any,
    ovov: typing.Any,
    fov: typing.Any,
    t1: typing.Any,
    t2: typing.Any,
    eps_o: typing.Any,
    eps_v: typing.Any,
    *,
    vir_chunk: typing.Any = None,
    denominator_threshold: typing.Any = 1e-10,
) -> typing.Any:
    """Build and execute the tile TensorIR lowering; returns the E_T scalar."""
    _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    _check_denominators(eps_o, eps_v, denominator_threshold)
    program = build_tile_triples_program(nocc, nvir, vir_chunk=vir_chunk)
    a_end = nvir if vir_chunk is None else vir_chunk[1]
    feeds = _tile_input_feeds(
        {
            "ovvv": ovvv,
            "ovoo": ovoo,
            "ovov": ovov,
            "fov": fov,
            "t1": t1,
            "t2": t2,
            "eps_o": eps_o,
            "eps_v": eps_v,
        },
        a_end,
    )
    result = execute(program, feeds)
    return float(result.outputs["triples_energy"])

"""Bounded resolvent moments of the canonical standard-(T) energy.

The audited W/V/R3 inventory supplies all numerators. The inverse derivative
``d D^-1 = -D^-1 (dD) D^-1`` supplies complete same-occupancy Fock cotangents,
including off-diagonals at internal degeneracies. Only occupied/virtual
separation is required; no same-occupancy orbital gap is divided here.

The caller sweeps b>=c and pages a, multiplies contributions by 2-delta_bc,
and accumulates occupied moments once per a page and virtual moments once per
page pair. At most two pages of resolvent vectors are simultaneously required.
The frontend defines mathematics and shapes, not a production force capability.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    divide,
    einsum,
    input_tensor,
    multiply,
    transpose,
)

from .triples import INVENTORY_HASH, OP, SLOW_TABLE, VP
from .triples_tiles import (
    _runtime_r3_node,
    _runtime_select,
    _runtime_triples_inputs,
    _runtime_v_node,
    _runtime_w_node,
    _t_views_tile,
)

if TYPE_CHECKING:
    from generativeqc_compiler.tensor.ir import Node


def build_runtime_triples_resolvent_program(
    nocc: int, nvir: int, *, capacity: int
) -> Program:
    """Return masked X=PW/D and Y=R3(P(W+V/2))/D for runtime a,b,c lanes.

    P is the sum over six simultaneous occupied/virtual pair permutations.
    Summing the complete ordered domain gives E=(1/3)<PW,Y>. R3 commutes with
    P, making all three occupied (or virtual) marginal contractions equal.
    Inactive lanes have valid maps and denominators, then both outputs are
    zeroed. In particular padded lanes must not contribute to cross-page moments.
    """
    nodes = _runtime_triples_inputs(nocc, nvir, capacity)
    q = nodes["a_map"].spec.indices[0]
    views = _t_views_tile(nodes)
    maps = tuple(nodes[f"{label}_map"] for label in "abc")
    ws, vs = {}, {}
    for label, permutation in VP.items():
        coordinates = tuple(maps[position] for position in permutation)
        ws[label] = _runtime_w_node(views, q, coordinates)
        vs[label] = _runtime_v_node(views, q, coordinates)

    def pair_sum(values: dict[str, Node]) -> Node:
        return add(
            *(
                transpose(values[label], (0, *(axis + 1 for axis in OP[order])))
                for label, order in SLOW_TABLE["abc"]
            )
        )

    left = pair_sum(ws)
    right = pair_sum(
        {
            label: add(ws[label], vs[label], coefficients=(1, Fraction(1, 2)))
            for label in ws
        }
    )
    qijk = left.spec.indices
    denominator = add(
        *(broadcast(nodes["eps_o"], qijk, (axis,)) for axis in (1, 2, 3)),
        *(
            broadcast(_runtime_select(nodes["eps_v"], q, (0, value)), qijk, (0,))
            for value in maps
        ),
        coefficients=(1, 1, 1, -1, -1, -1),
    )
    active = broadcast(nodes["active"], qijk, (0,))
    return Program(
        {
            "x": multiply(divide(left, denominator), active),
            "y": multiply(divide(_runtime_r3_node(right), denominator), active),
        },
        provenance={
            "method": "RCCSD(T)",
            "inventory_hash": INVENTORY_HASH,
            "scope": "separable Fock resolvent vectors",
            "runtime_domain_capacity": capacity,
            "runtime_domain": "a page for fixed b>=c; active masks padded lanes",
        },
    )


def build_triples_fock_moment_program(
    nocc: int, *, capacity: int, block: str
) -> Program:
    """Contract one oo moment or one cross-page vv moment from bounded vectors.

    oo uses only the left page. vv returns the left-by-right block, already
    symmetrized in its two factors. An off-diagonal page pair must be scattered
    to both matrix blocks once; a diagonal page pair must be scattered once.
    Multiplicity 2-delta_bc belongs to the caller, not to an individual vector.
    """
    if any(type(size) is not int or size < 1 for size in (nocc, capacity)):
        raise ValueError("triples Fock moments require positive occupied/page sizes")
    if block not in ("oo", "vv"):
        raise ValueError("triples Fock moment block must be oo or vv")
    occ = IndexSpace("runtime_occupied", "occupied", nocc)
    lanes = IndexSpace("runtime_triples", "batch", capacity)
    spec = TensorSpec(
        (Index("q", lanes), *(Index(label, occ) for label in "ijk")),
        role="parameter",
        differentiable=True,
        representation="restricted_spatial",
    )
    x_left, y_left = (input_tensor(name, spec) for name in ("x_left", "y_left"))
    if block == "oo":
        moment = einsum("qijk,qljk->il", x_left, y_left)
        output = add(
            moment, transpose(moment, (1, 0)), coefficients=(Fraction(-1, 2),) * 2
        )
    else:
        x_right, y_right = (input_tensor(name, spec) for name in ("x_right", "y_right"))
        output = add(
            einsum("pijk,qijk->pq", x_left, y_right),
            einsum("pijk,qijk->pq", y_left, x_right),
            coefficients=(Fraction(1, 2),) * 2,
        )
    return Program(
        {"foo" if block == "oo" else "fvv": output},
        provenance={
            "method": "RCCSD(T)",
            "inventory_hash": INVENTORY_HASH,
            "scope": "separable Fock resolvent moment",
            "block": block,
            "runtime_domain_capacity": capacity,
        },
    )

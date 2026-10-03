"""Retained DF-CC Hamiltonian blocks in the supplied spatial-MO frame.

The correlation factors share one complete source transform. Only blocks with
at most two virtual indices are assembled; the solver consumes Q-major B_ov
and B_vv directly for its virtual actions. Reference/Fock policy stays outside
this integral representation.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.ir import slice_tensor, transpose

FACTOR_NAMES = ("boo", "bov", "bvo", "bvv")
BLOCK_FACTORS = {
    "ovov": ("bov", "bov"),
    "ovvo": ("bov", "bvo"),
    "oovv": ("boo", "bvv"),
    "ovoo": ("bov", "boo"),
    "oooo": ("boo", "boo"),
}


def factor_program(
    occupied: int, virtuals: int, auxiliaries: int, *, symmetric_pairs: bool = False
) -> Program:
    """Select four Q-major sectors, optionally restoring physical pair symmetry.

    Native molecular integrals are symmetric in their two spatial-orbital
    indices. Independent FP64 contractions can lose that symmetry in an
    ill-conditioned orbital/metric frame. Project the complete factor before
    selecting sectors so every retained block and factorized virtual action
    uses the same Hamiltonian. Arbitrary supplied tensors retain their original
    values unless the physical source owner explicitly selects this contract.
    """
    if any(type(x) is not int or x <= 0 for x in (occupied, virtuals, auxiliaries)):
        raise ValueError("positive integer DF-CC source extents required")
    n = occupied + virtuals
    orbital = IndexSpace("df_source_orbital", "orbital", n)
    # The common native emitter calls its dynamic third dimension 'batch'.
    # Here that independent extent is the auxiliary coordinate, not molecules.
    auxiliary = IndexSpace("df_source_auxiliary", "batch", auxiliaries)
    p, q, Q = Index("p", orbital), Index("q", orbital), Index("Q", auxiliary)
    source = input_tensor(
        "bmo", TensorSpec((p, q, Q), role="parameter", differentiable=True)
    )
    if symmetric_pairs:
        source = add(
            source, transpose(source, (1, 0, 2)), coefficients=(Fraction(1, 2),) * 2
        )
    sectors = {"o": (0, occupied), "v": (occupied, n)}
    return Program(
        {
            name: transpose(
                slice_tensor(
                    source, (sectors[name[1]], sectors[name[2]], (0, auxiliaries))
                ),
                (2, 0, 1),
            )
            for name in FACTOR_NAMES
        },
        provenance={
            "method": "DF-CC source factor selection",
            "native_execution_order": "dependencies",
            "orbital_pair_projection": "symmetric_average"
            if symmetric_pairs
            else "none",
        },
    )


def block_program(occupied: int, virtuals: int, auxiliaries: int, name: str) -> Program:
    """Define one retained four-index block from the same fitted factors."""
    if name not in BLOCK_FACTORS:
        raise ValueError("only retained DF-CC blocks may be assembled")
    factors = factor_program(occupied, virtuals, auxiliaries)
    left_name, right_name = BLOCK_FACTORS[name]
    left = input_tensor(
        left_name, replace(factors.outputs[left_name].spec, role="parameter")
    )
    right = (
        left
        if left_name == right_name
        else input_tensor(
            right_name, replace(factors.outputs[right_name].spec, role="parameter")
        )
    )
    return Program(
        {name: einsum("Qpq,Qrs->pqrs", left, right)},
        provenance={"method": "DF-CC retained integral block"},
    )

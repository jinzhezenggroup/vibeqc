"""TensorIR ownership of the factorized DF-RCCSD virtual residual correction.

This module is the generated-algebra bridge for #157 C2b. It represents exactly
the ovvv/vvvv contributions externalized by build_ccsd_program(...,
external_virtual_correction=True) directly in terms of retained B_ov/B_vv
factors. The auxiliary index is reduced inside every output contraction; no
ovvv, vvvv, or Q-indexed residual tensor is materialized.
"""

from __future__ import annotations

import typing

from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    Symmetry,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    transpose,
)

from .equations import amplitude_specs

VERSION = 1


def _symmetric_pair(value: typing.Any) -> typing.Any:
    return add(value, transpose(value, (1, 0, 3, 2)))


def build_df_virtual_correction_program(
    nocc: typing.Any,
    nvir: typing.Any,
    naux: typing.Any,
) -> Program:
    """Return exact DF substitutions for every externalized ovvv/vvvv term.

    Inputs are B_ov[Q,i,a], B_vv[Q,a,b], T1 and T2. Outputs have exactly the
    singles/doubles residual layouts expected by the external-correction RCCSD
    TensorIR inventory. All Q reductions happen in the final einsums of each
    scientific term, so generated native execution never owns ovvv/vvvv and
    never creates a Q-indexed residual-sized temporary.
    """

    if any(type(n) is not int or n < 1 for n in (nocc, nvir, naux)):
        raise ValueError(
            "factorized DF RCCSD requires nonempty occupied, virtual and auxiliary spaces"
        )

    singles_spec, doubles_spec = amplitude_specs(nocc, nvir)
    occupied = singles_spec.indices[0].space
    virtual = singles_spec.indices[1].space
    auxiliary = IndexSpace("df_auxiliary", "auxiliary", naux)

    q = Index("Q", auxiliary)
    i = Index("i", occupied)
    a = Index("a", virtual)
    b = Index("b", virtual)

    bov = input_tensor(
        "bov",
        TensorSpec(
            (q, i, a),
            role="parameter",
            differentiable=True,
            representation="restricted_spatial_df",
        ),
    )
    bvv = input_tensor(
        "bvv",
        TensorSpec(
            (q, a, b),
            role="parameter",
            differentiable=True,
            representation="restricted_spatial_df",
            symmetries=(Symmetry((0, 2, 1)),),
        ),
    )
    t1 = input_tensor("t1", singles_spec)
    t2 = input_tensor("t2", doubles_spec)
    tau = add(t2, einsum("ia,jb->ijab", t1, t1))

    # Singles S09/S10/S13/S14 after substituting
    # g[p,q,r,s] = sum_Q B[Q,p,q] B[Q,r,s].
    r1 = add(
        einsum("Qkd,ikcd,Qac->ia", bov, t2, bvv, coefficient=2),
        einsum("Qkc,ikcd,Qad->ia", bov, t2, bvv, coefficient=-1),
        einsum("Qkd,kd,ic,Qac->ia", bov, t1, t1, bvv, coefficient=2),
        einsum("Qkc,kd,ic,Qad->ia", bov, t1, t1, bvv, coefficient=-1),
    )

    # Lvv(ovvv) -> Xfvv -> D06.
    d06 = add(
        einsum("Qkd,kd,Qac,ijcb->ijab", bov, t1, bvv, t2, coefficient=2),
        einsum("Qad,Qkc,kd,ijcb->ijab", bvv, bov, t1, t2, coefficient=-1),
    )

    # Wvoov(ovvv) ring/exchange.
    ring = einsum("Qad,id,Qkc,kjcb->ijab", bvv, t1, bov, t2, coefficient=2)
    exchange = einsum(
        "Qad,id,Qkc,kjbc->ijab", bvv, t1, bov, t2, coefficient=-1
    )

    # Wvovo(ovvv) ring/cross.
    vovo_ring = einsum(
        "Qac,Qkd,id,kjcb->ijab", bvv, bov, t1, t2, coefficient=-1
    )
    vovo_cross = einsum(
        "Qbc,Qkd,id,kjac->ijab", bvv, bov, t1, t2, coefficient=-1
    )

    # Xv(ovvv) -> D02.
    d02 = einsum("Qia,jc,Qcb->ijab", bov, t1, bvv)

    # Complete D05 virtual ladder. These three terms are exactly the part
    # removed together with Wvvvv in external-correction mode.
    d05_direct = einsum("Qac,ijcd,Qbd->ijab", bvv, tau, bvv)
    d05_left = einsum(
        "Qac,ijcd,kb,Qkd->ijab", bvv, tau, t1, bov, coefficient=-1
    )
    d05_right = einsum(
        "ka,Qkc,ijcd,Qbd->ijab", t1, bov, tau, bvv, coefficient=-1
    )

    r2 = add(
        _symmetric_pair(d06),
        _symmetric_pair(ring),
        _symmetric_pair(exchange),
        _symmetric_pair(vovo_ring),
        _symmetric_pair(vovo_cross),
        _symmetric_pair(d02),
        d05_direct,
        d05_left,
        d05_right,
    )

    return Program(
        {
            "df_virtual_singles": r1,
            "df_virtual_doubles": r2,
        },
        provenance={
            "method": "RCCSD",
            "slice": "157-C2b1",
            "df_virtual_correction_version": VERSION,
            "factorization": "g[pqrs]=sum_Q B[Q,pq] B[Q,rs]",
            "external_inventory": "df-ovvv-vvvv-residual-v1",
            "resident_ovvv": False,
            "resident_vvvv": False,
            "auxiliary_reduction": "direct-in-output-einsums",
        },
    )

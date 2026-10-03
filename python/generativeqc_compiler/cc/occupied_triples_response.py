"""Shared-AD reverse programs for the occupied-tile DF triples decomposition.

These graphs keep panel/moment derivatives in packed BLAS-sized coordinates.
They do not scatter a tile into a complete rank-six tensor. The native owner
must sum every occupied permutation, including repeated indices, and bind each
local cotangent to the corresponding strided physical view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    input_tensor,
    multiply,
    optimize,
    transpose_program,
)

from .occupied_triples import df_panel_program, energy_scalar_program, moment_program


def _pullback(primal: Program, inputs: Iterable[str] | None = None) -> Program:
    """Keep the shared reverse algebra while pruning unrequested output paths."""
    result = optimize(
        transpose_program(primal, tuple(primal.outputs), inputs=inputs).program
    )
    return Program(
        result.outputs,
        provenance={**result.provenance, "native_execution_order": "dependencies"},
    )


def panel_vjp(virtuals: int, auxiliaries: int) -> Program:
    """Two BLAS contractions from one accumulated occupied-panel cotangent.

    bar_bov_i addresses a strided Q-by-v view; bar_bvv accumulates across all
    occupied panels. Bvv's physical symmetric tangent is an upstream boundary,
    so this map returns its unprojected dense Frobenius cotangent.
    """
    return _pullback(df_panel_program(virtuals, auxiliaries))


def moment_vjp(occupied: int, virtuals: int, output: str) -> Program:
    """Four packed contractions for a W or V cube, without global scatters.

    W produces panel/t2_kj/ovoo_ij/t2_mk cotangents. V produces
    ovov_ij/t1_k/t2_ij/fov_k cotangents. Repeated physical views accumulate;
    arbitrary local seed arrays need not themselves have pair symmetry.
    """
    if output not in ("w", "v"):
        raise ValueError("occupied triples response requires W or V")
    primal = moment_program(occupied, virtuals)
    return _pullback(Program({output: primal.outputs[output]}))


def energy_scalar_vjp(inputs: Iterable[str] | None = None) -> Program:
    """Demand-driven scalar pullback of the exact occupied-triangle epilogue.

    The denominator input already contains the forward 6/2/1 multiplicity.
    The caller composes its derivative with gap * multiplicity exactly once.
    Virtual-coordinate permutations belong to the source binding: gathering
    their inverse maps avoids nondeterministic atomic accumulation into W.
    This scalar boundary alone is not a full same-space Fock response.
    """
    return _pullback(energy_scalar_program(), inputs)


def gap_vjp(virtuals: int) -> Program:
    """Reduce a gap-cube seed into three occupied scalars and one virtual vector.

    Keeping each occupied slot independent lets the native scatter account for
    repeated physical indices without atomics or a changed multiplicity.
    """
    if type(virtuals) is not int or virtuals < 1:
        raise ValueError("positive virtual extent required")
    vir = IndexSpace("occupied_response_virtual", "virtual", virtuals)
    axes = tuple(Index(x, vir) for x in "abc")
    ev = input_tensor(
        "eps_v", TensorSpec((axes[0],), role="parameter", differentiable=True)
    )
    occupied = [
        input_tensor("eps_" + x, TensorSpec((), role="parameter", differentiable=True))
        for x in "ijk"
    ]
    gap = add(
        *(broadcast(x, axes, ()) for x in occupied),
        *(broadcast(ev, axes, (axis,)) for axis in range(3)),
        coefficients=(1, 1, 1, -1, -1, -1),
    )
    return _pullback(Program({"gap": gap}))


def scaled_denominator_vjp() -> Program:
    """Differentiate gap times the forward occupied multiplicity exactly once."""
    gap, multiplicity = [
        input_tensor(x, TensorSpec((), role="parameter", differentiable=True))
        for x in ("gap", "multiplicity")
    ]
    return _pullback(Program({"denominator": multiply(gap, multiplicity)}), ("gap",))

"""Runtime-bound scalar GFN2 H0-force science generated from TensorIR.

The CUDA runtime keeps ragged shell/AO traversal, validation, accumulation,
failure publication, and stream ownership. This module owns the local H0
pair factor, its reverse-mode CN/radial response, and the AO contraction
arithmetic so those equations are not duplicated in handwritten CUDA.
"""

from __future__ import annotations

from vibeqc_compiler.tensor.ad_program import transpose_program
from vibeqc_compiler.tensor.ir import (
    Node,
    add,
    constant,
    divide,
    input_tensor,
    multiply,
    sqrt,
)
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

GFN2_H0_FORCE_RUNTIME_VERSION = "gfn2-h0-force-runtime-ir-v1"


def _input(name: str, *, differentiable: bool = False) -> Node:
    return input_tensor(
        name,
        TensorSpec((), role="input", differentiable=differentiable),
    )


def _one() -> Node:
    return constant(1, TensorSpec((), role="constant"))


def _half() -> Node:
    return constant("1/2", TensorSpec((), role="constant"))


def _average_level_inputs() -> tuple[Node, Node, Node, Node, Node, Node, Node]:
    first_shell_level = _input("first_shell_level")
    second_shell_level = _input("second_shell_level")
    first_cn_scale = _input("first_cn_scale")
    second_cn_scale = _input("second_cn_scale")
    first_cn = _input("first_cn", differentiable=True)
    second_cn = _input("second_cn", differentiable=True)
    first_level = add(
        first_shell_level,
        multiply(first_cn_scale, first_cn),
        coefficients=(1, -1),
    )
    second_level = add(
        second_shell_level,
        multiply(second_cn_scale, second_cn),
        coefficients=(1, -1),
    )
    # Match the native finite-range contract exactly: form the level sum first,
    # then apply 1/2. Reassociating this as 0.5*a + 0.5*b would accept cases
    # where the retired implementation intentionally reported overflow.
    average_level = multiply(add(first_level, second_level), _half())
    return (
        average_level,
        first_shell_level,
        second_shell_level,
        first_cn_scale,
        second_cn_scale,
        first_cn,
        second_cn,
    )


def build_gfn2_h0_onsite_factor_program() -> Program:
    """Build the same-atom H0 shell-pair factor."""

    average_level, *_ = _average_level_inputs()
    return Program(
        {"factor": average_level},
        provenance={
            "kind": "gfn2-runtime-h0-onsite-factor",
            "version": GFN2_H0_FORCE_RUNTIME_VERSION,
            "source_issue": 560,
        },
    )


def build_gfn2_h0_offsite_factor_program() -> Program:
    """Build the separated-atom H0 shell-pair factor."""

    average_level, *_ = _average_level_inputs()
    first_radius = _input("first_radius")
    second_radius = _input("second_radius")
    first_polynomial = _input("first_polynomial")
    second_polynomial = _input("second_polynomial")
    pair_scale = _input("pair_scale")
    distance = _input("distance", differentiable=True)

    radius_sum = add(first_radius, second_radius)
    reduced_distance = sqrt(divide(distance, radius_sum))
    first_shape = add(_one(), multiply(first_polynomial, reduced_distance))
    second_shape = add(_one(), multiply(second_polynomial, reduced_distance))
    # Preserve the retired CUDA/C++ left-to-right product order. This matters
    # for finite-range behavior when pair_scale rescales a large first shape.
    spatial_scale = multiply(multiply(pair_scale, first_shape), second_shape)
    factor = multiply(average_level, spatial_scale)
    return Program(
        {"factor": factor},
        provenance={
            "kind": "gfn2-runtime-h0-offsite-factor",
            "version": GFN2_H0_FORCE_RUNTIME_VERSION,
            "source_issue": 560,
        },
    )


def build_gfn2_h0_distance_program() -> Program:
    """Build the Cartesian separation norm used by offsite H0 science."""

    dx = _input("dx", differentiable=True)
    dy = _input("dy", differentiable=True)
    dz = _input("dz", differentiable=True)
    distance_squared = add(
        multiply(dx, dx),
        multiply(dy, dy),
        multiply(dz, dz),
    )
    distance = sqrt(distance_squared)
    return Program(
        {
            "distance_squared": distance_squared,
            "distance": distance,
        },
        provenance={
            "kind": "gfn2-runtime-h0-distance",
            "version": GFN2_H0_FORCE_RUNTIME_VERSION,
            "source_issue": 560,
        },
    )


def build_gfn2_h0_distance_vjp_program() -> Program:
    """Generate the Cartesian pullback of the same separation norm."""

    return transpose_program(
        build_gfn2_h0_distance_program(),
        ("distance",),
        inputs=("dx", "dy", "dz"),
    ).program


def build_gfn2_h0_pulay_seed_program() -> Program:
    """Build the stationary overlap/Pulay seed update."""

    seed = _input("seed")
    weighted = _input("weighted")
    return Program(
        {"pulay_seed": add(seed, weighted, coefficients=(1, -1))},
        provenance={
            "kind": "gfn2-runtime-h0-pulay-seed",
            "version": GFN2_H0_FORCE_RUNTIME_VERSION,
            "source_issue": 560,
        },
    )


def build_gfn2_h0_onsite_vjp_program() -> Program:
    """Generate same-atom CN adjoints from the onsite factor."""

    return transpose_program(
        build_gfn2_h0_onsite_factor_program(),
        ("factor",),
        inputs=("first_cn", "second_cn"),
    ).program


def build_gfn2_h0_offsite_vjp_program() -> Program:
    """Generate separated-atom CN and radial adjoints from the factor."""

    return transpose_program(
        build_gfn2_h0_offsite_factor_program(),
        ("factor",),
        inputs=("first_cn", "second_cn", "distance"),
    ).program


def build_gfn2_h0_ao_update_program() -> Program:
    """Build one AO contribution to overlap adjoint and shell-pair weight."""

    density = _input("density")
    overlap = _input("overlap")
    factor = _input("factor")
    overlap_adjoint = _input("overlap_adjoint")
    block_weight = _input("block_weight")
    overlap_adjoint_updated = add(
        overlap_adjoint,
        multiply(density, factor),
    )
    block_weight_updated = add(
        block_weight,
        multiply(density, overlap),
    )
    return Program(
        {
            "overlap_adjoint_updated": overlap_adjoint_updated,
            "block_weight_updated": block_weight_updated,
        },
        provenance={
            "kind": "gfn2-runtime-h0-ao-update",
            "version": GFN2_H0_FORCE_RUNTIME_VERSION,
            "source_issue": 560,
        },
    )

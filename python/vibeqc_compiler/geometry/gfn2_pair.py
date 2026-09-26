"""Compiler-owned runtime-bound scalar GFN2 pair science.

The full #504 GeometryIR graph owns topology, gather/scatter and ragged
reduction. Production runtimes already own those policies. This module factors
the pairwise scientific expression into scalar TensorIR so native CPU/CUDA
loops can consume generated math without specializing one Program per molecule.
"""

from __future__ import annotations

from vibeqc_compiler.tensor.ad_program import linearize
from vibeqc_compiler.tensor.ir import (
    Node,
    add,
    constant,
    divide,
    exp,
    input_tensor,
    multiply,
    sqrt,
)
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

GFN2_PAIR_RUNTIME_VERSION = "gfn2-runtime-pair-ir-v1"
GFN2_CN_FIRST_STEEPNESS = 10.0
GFN2_CN_SECOND_STEEPNESS = 20.0
GFN2_CN_SECOND_RADIUS_SHIFT_BOHR = 2.0
GFN2_REPULSION_KEXP = 1.5
GFN2_REPULSION_KLIGHT = 1.0


def gfn2_pair_equations(
    *,
    distance: Node,
    inverse_distance: Node,
    radius: Node,
    shifted_radius: Node,
    one: Node,
    minus_one: Node,
    first_steepness: Node,
    second_steepness: Node,
    light_pair: Node,
    heavy_pair: Node,
    pair_alpha: Node,
    pair_charge: Node,
) -> tuple[Node, Node]:
    """Return pair CN contribution and screened repulsion energy."""

    first_ratio = multiply(radius, inverse_distance)
    second_ratio = multiply(shifted_radius, inverse_distance)
    first_delta = add(first_ratio, one, coefficients=(1, -1))
    second_delta = add(second_ratio, one, coefficients=(1, -1))
    first_argument = multiply(first_steepness, first_delta)
    second_argument = multiply(second_steepness, second_delta)

    first_logistic = divide(one, add(one, exp(multiply(minus_one, first_argument))))
    second_logistic = divide(one, add(one, exp(multiply(minus_one, second_argument))))
    pair_coordination = multiply(first_logistic, second_logistic)

    heavy_distance = multiply(distance, sqrt(distance))
    distance_power = add(
        multiply(light_pair, distance),
        multiply(heavy_pair, heavy_distance),
    )
    decay_argument = multiply(minus_one, multiply(pair_alpha, distance_power))
    pair_repulsion = multiply(
        multiply(pair_charge, exp(decay_argument)),
        inverse_distance,
    )
    return pair_coordination, pair_repulsion


def build_gfn2_runtime_pair_primal() -> Program:
    """Build runtime-bound scalar pair energy/CN from the #504 equations."""

    differentiable = TensorSpec((), role="input", differentiable=True)
    parameter = TensorSpec((), role="input", differentiable=False)
    distance = input_tensor("distance", differentiable)
    radius = input_tensor("radius", parameter)
    pair_alpha = input_tensor("pair_alpha", parameter)
    pair_charge = input_tensor("pair_charge", parameter)
    light_pair = input_tensor("light_pair", parameter)

    one = constant(1)
    minus_one = constant(-1)
    inverse_distance = divide(one, distance)
    shifted_radius = add(radius, constant(repr(GFN2_CN_SECOND_RADIUS_SHIFT_BOHR)))
    heavy_pair = add(one, light_pair, coefficients=(1, -1))
    coordination, repulsion = gfn2_pair_equations(
        distance=distance,
        inverse_distance=inverse_distance,
        radius=radius,
        shifted_radius=shifted_radius,
        one=one,
        minus_one=minus_one,
        first_steepness=constant(repr(GFN2_CN_FIRST_STEEPNESS)),
        second_steepness=constant(repr(GFN2_CN_SECOND_STEEPNESS)),
        light_pair=light_pair,
        heavy_pair=heavy_pair,
        pair_alpha=pair_alpha,
        pair_charge=pair_charge,
    )
    return Program(
        {
            "coordination": coordination,
            "repulsion_energy": repulsion,
        },
        provenance={
            "kind": "gfn2-runtime-pair-primal",
            "version": GFN2_PAIR_RUNTIME_VERSION,
            "source": "#504",
        },
    )


def build_gfn2_runtime_pair_kernel() -> Program:
    """Attach compiler-generated d/dr outputs to the runtime pair primal."""

    primal = build_gfn2_runtime_pair_primal()
    jvp = linearize(
        primal,
        ("distance",),
        outputs=("coordination", "repulsion_energy"),
    )
    definitions = tuple(dict.fromkeys([*primal.nodes, *jvp.program.nodes]))
    return Program(
        {
            "coordination": primal.outputs["coordination"],
            "repulsion_energy": primal.outputs["repulsion_energy"],
            "coordination_distance_derivative": jvp.program.outputs["d_coordination"],
            "repulsion_distance_derivative": jvp.program.outputs["d_repulsion_energy"],
        },
        definitions,
        provenance={
            "kind": "gfn2-runtime-pair-kernel",
            "version": GFN2_PAIR_RUNTIME_VERSION,
            "primal_logical_hash": primal.logical_hash,
            "distance_jvp_hash": jvp.derivative_hash,
            "generation": "TensorIR forward AD",
        },
    )

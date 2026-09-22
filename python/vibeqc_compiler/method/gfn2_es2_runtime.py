"""Compiler-owned scalar GFN2 ES2 arithmetic and generated coordinate response.

The production ES2 runtimes own ragged shell topology, validation, reductions,
scratch publication, and stream/error policy.  This module owns only the
repeated scalar scientific equations so CPU and CUDA consumers share one
compiler identity instead of maintaining duplicate formulas.
"""

from __future__ import annotations

from vibeqc_compiler.tensor.ad_program import VJPProgram, transpose_program
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

GFN2_ES2_RUNTIME_VERSION = "gfn2-es2-runtime-ir-v1"


def _input(name: str, *, differentiable: bool = False) -> Node:
    return input_tensor(
        name,
        TensorSpec((), role="input", differentiable=differentiable),
    )


def build_gfn2_es2_arithmetic_hardness_program() -> Program:
    """Arithmetic shell-hardness mean used by GFN2 ES2."""

    first = _input("first_hardness")
    second = _input("second_hardness")
    average = add(first, second, coefficients=("1/2", "1/2"))
    return Program(
        {"average": average},
        provenance={
            "kind": "gfn2-runtime-es2-arithmetic-hardness",
            "version": GFN2_ES2_RUNTIME_VERSION,
        },
    )


def build_gfn2_es2_pair_primal() -> Program:
    """One off-site ES2 shell-pair energy from a bound pair hardness."""

    dx = _input("dx", differentiable=True)
    dy = _input("dy", differentiable=True)
    dz = _input("dz", differentiable=True)
    pair_hardness = _input("pair_hardness")
    first_charge = _input("first_charge")
    second_charge = _input("second_charge")

    one = constant(1)
    inverse_hardness = divide(one, pair_hardness)
    softened_squared = add(
        multiply(dx, dx),
        multiply(dy, dy),
        multiply(dz, dz),
        multiply(inverse_hardness, inverse_hardness),
    )
    kernel = divide(one, sqrt(softened_squared))
    pair_energy = multiply(multiply(first_charge, second_charge), kernel)
    return Program(
        {"kernel": kernel, "pair_energy": pair_energy},
        provenance={
            "kind": "gfn2-runtime-es2-pair-primal",
            "version": GFN2_ES2_RUNTIME_VERSION,
            "source": "GFN2 ES2 gexp=2 shell electrostatics",
        },
    )


def build_gfn2_es2_pair_vjp() -> VJPProgram:
    """Generate Cartesian dE/d(delta-R) for one off-site shell pair."""

    return transpose_program(
        build_gfn2_es2_pair_primal(),
        ("pair_energy",),
        inputs=("dx", "dy", "dz"),
    )


def build_gfn2_es2_potential_update_program() -> Program:
    """One serial shell-potential accumulation."""

    kernel = _input("kernel")
    charge = _input("charge")
    accumulator = _input("accumulator")
    updated = add(accumulator, multiply(kernel, charge))
    return Program(
        {"updated": updated},
        provenance={
            "kind": "gfn2-runtime-es2-potential-update",
            "version": GFN2_ES2_RUNTIME_VERSION,
        },
    )


def build_gfn2_es2_energy_update_program() -> Program:
    """One row-energy accumulation after the serial potential sweep."""

    row_charge = _input("row_charge")
    potential = _input("potential")
    accumulator = _input("accumulator")
    updated = add(
        accumulator,
        multiply(add(row_charge, coefficients=("1/2",)), potential),
    )
    return Program(
        {"updated": updated},
        provenance={
            "kind": "gfn2-runtime-es2-energy-update",
            "version": GFN2_ES2_RUNTIME_VERSION,
        },
    )


def build_gfn2_es2_cached_gradient_weight_program() -> Program:
    """Optimized cached-Gamma lowering of the generated pair VJP."""

    kernel = _input("kernel")
    first_charge = _input("first_charge")
    second_charge = _input("second_charge")
    # Preserve the runtime's scaling order before multiplying large charges.
    weight = multiply(first_charge, kernel)
    weight = multiply(weight, second_charge)
    weight = multiply(weight, kernel)
    weight = multiply(weight, kernel)
    return Program(
        {"weight": weight},
        provenance={
            "kind": "gfn2-runtime-es2-cached-gradient-weight",
            "version": GFN2_ES2_RUNTIME_VERSION,
            "derived_from": build_gfn2_es2_pair_vjp().derivative_hash,
        },
    )


def build_gfn2_es2_gradient_projection_program() -> Program:
    """Project a summed cached-Gamma weight onto one atom-pair displacement."""

    weight = _input("weight")
    dx = _input("dx")
    dy = _input("dy")
    dz = _input("dz")
    gradients = {
        "gx": add(multiply(weight, dx), coefficients=(-1,)),
        "gy": add(multiply(weight, dy), coefficients=(-1,)),
        "gz": add(multiply(weight, dz), coefficients=(-1,)),
    }
    return Program(
        gradients,
        provenance={
            "kind": "gfn2-runtime-es2-gradient-projection",
            "version": GFN2_ES2_RUNTIME_VERSION,
            "derived_from": build_gfn2_es2_pair_vjp().derivative_hash,
        },
    )

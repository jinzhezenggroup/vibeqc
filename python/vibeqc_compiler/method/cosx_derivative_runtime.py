"""Compiler-owned scalar TensorIR for COSX derivative contractions.

Native CUDA owns tile/index traversal, validation, stream/error handling and
atom scatter. These programs own the pure FP64 contraction arithmetic reused by
the explicit-point and molecular derivative consumers.
"""

from __future__ import annotations

from vibeqc_compiler.tensor.ir import Node, add, constant, input_tensor, multiply
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

COSX_DERIVATIVE_RUNTIME_VERSION = "cosx-derivative-contraction-ir-v1"


def _input(name: str) -> Node:
    return input_tensor(name, TensorSpec((), role="input"))


def _half() -> Node:
    return constant("1/2", TensorSpec((), role="constant"))


def _program(outputs: dict[str, Node], kind: str) -> Program:
    return Program(
        outputs,
        provenance={
            "kind": kind,
            "version": COSX_DERIVATIVE_RUNTIME_VERSION,
            "source_issue": 246,
        },
    )


def build_cosx_projection_update_program() -> Program:
    accumulator = _input("accumulator")
    left = _input("left")
    right = _input("right")
    return _program(
        {"updated": add(accumulator, multiply(left, right))},
        "cosx-projection-update",
    )


def build_cosx_esp_derivative_update_program() -> Program:
    accumulator = _input("accumulator")
    matrix_derivative = _input("matrix_derivative")
    projected_value = _input("projected_value")
    matrix_value = _input("matrix_value")
    projected_derivative = _input("projected_derivative")
    response = add(
        multiply(matrix_derivative, projected_value),
        multiply(matrix_value, projected_derivative),
    )
    return _program(
        {"updated": add(accumulator, response)},
        "cosx-esp-derivative-update",
    )


def build_cosx_symmetric_projection_update_program() -> Program:
    accumulator = _input("accumulator")
    ao = _input("ao")
    density_rc = _input("density_rc")
    density_cr = _input("density_cr")
    averaged = multiply(multiply(ao, _half()), add(density_rc, density_cr))
    return _program(
        {"updated": add(accumulator, averaged)},
        "cosx-symmetric-projection-update",
    )


def build_cosx_bidirectional_update_program() -> Program:
    right = _input("right")
    left = _input("left")
    matrix_rc = _input("matrix_rc")
    matrix_cr = _input("matrix_cr")
    projected = _input("projected")
    symmetric_projection = _input("symmetric_projection")
    return _program(
        {
            "right_updated": add(right, multiply(matrix_rc, projected)),
            "left_updated": add(left, multiply(matrix_cr, symmetric_projection)),
        },
        "cosx-bidirectional-update",
    )


def build_cosx_point_gradient_update_program() -> Program:
    accumulator = _input("accumulator")
    density = _input("density")
    phi_derivative_row = _input("phi_derivative_row")
    potential_column = _input("potential_column")
    phi_row = _input("phi_row")
    potential_derivative_column = _input("potential_derivative_column")
    phi_derivative_column = _input("phi_derivative_column")
    potential_row = _input("potential_row")
    phi_column = _input("phi_column")
    potential_derivative_row = _input("potential_derivative_row")
    raw_rc = add(
        multiply(phi_derivative_row, potential_column),
        multiply(phi_row, potential_derivative_column),
    )
    raw_cr = add(
        multiply(phi_derivative_column, potential_row),
        multiply(phi_column, potential_derivative_row),
    )
    contribution = multiply(
        multiply(density, _half()),
        add(raw_rc, raw_cr),
    )
    return _program(
        {"updated": add(accumulator, contribution)},
        "cosx-point-gradient-update",
    )


def build_cosx_molecular_ao_update_program() -> Program:
    from_left = _input("from_left")
    from_right = _input("from_right")
    density_rc = _input("density_rc")
    density_cr = _input("density_cr")
    potential = _input("potential")
    left_potential = _input("left_potential")
    left_contribution = multiply(
        multiply(_half(), add(density_rc, density_cr)),
        potential,
    )
    right_contribution = multiply(density_rc, left_potential)
    return _program(
        {
            "from_left_updated": add(from_left, left_contribution),
            "from_right_updated": add(from_right, right_contribution),
        },
        "cosx-molecular-ao-update",
    )


def build_cosx_molecular_cotangent_program() -> Program:
    energy_factor = _input("energy_factor")
    weight = _input("weight")
    from_left = _input("from_left")
    from_right = _input("from_right")
    cotangent = multiply(
        multiply(energy_factor, weight),
        add(from_left, from_right),
    )
    return _program({"cotangent": cotangent}, "cosx-molecular-cotangent")


def build_cosx_scale_program() -> Program:
    factor = _input("factor")
    value = _input("value")
    return _program({"scaled": multiply(factor, value)}, "cosx-scale")


def build_cosx_pair_scale_program() -> Program:
    first = _input("first")
    second = _input("second")
    value = _input("value")
    return _program(
        {"scaled": multiply(multiply(first, second), value)},
        "cosx-pair-scale",
    )

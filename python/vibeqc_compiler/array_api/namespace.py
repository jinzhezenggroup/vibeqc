"""Bounded Array-API-like namespace lowering directly to TensorIR."""

from __future__ import annotations

import typing
from fractions import Fraction

from vibeqc_compiler.tensor import ir as tensor_ir

from .array import ExactScalar, VibeArray


def _array(value: object, name: str = "operand") -> VibeArray:
    if not isinstance(value, VibeArray):
        raise TypeError(f"{name} must be a symbolic VibeArray")
    return value


def _exact(value: object, name: str = "scalar") -> Fraction:
    if type(value) not in (int, str, Fraction):
        raise TypeError(
            f"{name} must be an exact integer, Fraction, or rational string; "
            "floating-point scalar spelling is not accepted"
        )
    return Fraction(typing.cast("ExactScalar", value))


def _binary_arrays(
    left: object, right: object, name: str
) -> tuple[VibeArray, VibeArray]:
    return _array(left, f"{name} left operand"), _array(right, f"{name} right operand")


def add(x1: object, x2: object) -> VibeArray:
    """Elementwise add on identical TensorIR scientific domains."""
    left, right = _binary_arrays(x1, x2, "add")
    return VibeArray(tensor_ir.add(left.node, right.node))


def subtract(x1: object, x2: object) -> VibeArray:
    """Elementwise subtraction on identical TensorIR scientific domains."""
    left, right = _binary_arrays(x1, x2, "subtract")
    return VibeArray(tensor_ir.add(left.node, right.node, coefficients=(1, -1)))


def multiply(x1: object, x2: object) -> VibeArray:
    """Multiply arrays elementwise or scale one array by an exact scalar."""
    if isinstance(x1, VibeArray) and isinstance(x2, VibeArray):
        return VibeArray(tensor_ir.multiply(x1.node, x2.node))
    if isinstance(x1, VibeArray):
        factor = _exact(x2)
        return VibeArray(tensor_ir.add(x1.node, coefficients=(factor,)))
    if isinstance(x2, VibeArray):
        factor = _exact(x1)
        return VibeArray(tensor_ir.add(x2.node, coefficients=(factor,)))
    raise TypeError("multiply requires at least one symbolic VibeArray")


def divide(x1: object, x2: object) -> VibeArray:
    """Divide arrays elementwise or divide one array by an exact scalar."""
    if isinstance(x1, VibeArray) and isinstance(x2, VibeArray):
        return VibeArray(tensor_ir.divide(x1.node, x2.node))
    if isinstance(x1, VibeArray):
        denominator = _exact(x2, "divisor")
        if denominator == 0:
            raise ZeroDivisionError("exact scalar divisor cannot be zero")
        return VibeArray(
            tensor_ir.add(x1.node, coefficients=(Fraction(1, 1) / denominator,))
        )
    if isinstance(x2, VibeArray):
        raise TypeError(
            "scalar / VibeArray is not in the initial frontend subset because "
            "it would require an explicit domain-shaped scalar broadcast"
        )
    raise TypeError("divide requires at least one symbolic VibeArray")


def negative(x: object) -> VibeArray:
    value = _array(x)
    return VibeArray(tensor_ir.add(value.node, coefficients=(-1,)))


def pow(x: object, exponent: object) -> VibeArray:
    value = _array(x)
    return VibeArray(tensor_ir.power(value.node, _exact(exponent, "exponent")))


def exp(x: object) -> VibeArray:
    return VibeArray(tensor_ir.exp(_array(x).node))


def log(x: object) -> VibeArray:
    return VibeArray(tensor_ir.log(_array(x).node))


def sqrt(x: object) -> VibeArray:
    return VibeArray(tensor_ir.sqrt(_array(x).node))


def sum(
    x: object,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: object = None,
    keepdims: bool = False,
) -> VibeArray:
    """Reduce selected axes; dtype conversion and keepdims are not yet exposed."""
    value = _array(x)
    if dtype is not None:
        raise ValueError("frontend sum does not insert dtype conversions")
    if type(keepdims) is not bool or keepdims:
        raise ValueError("frontend sum currently requires keepdims=False")
    if axis is None:
        axes = tuple(range(value.ndim))
    elif type(axis) is int:
        axes = (axis,)
    elif isinstance(axis, tuple):
        axes = axis
    else:
        raise TypeError("axis must be an int, tuple of ints, or None")
    return VibeArray(tensor_ir.reduce_sum(value.node, axes=axes))


def permute_dims(x: object, axes: tuple[int, ...]) -> VibeArray:
    value = _array(x)
    return VibeArray(tensor_ir.transpose(value.node, axes))


def matmul(x1: object, x2: object) -> VibeArray:
    """Initial rank-2 matmul subset; batching is intentionally unsupported."""
    left, right = _binary_arrays(x1, x2, "matmul")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("frontend matmul currently supports rank-2 arrays only")
    return VibeArray(tensor_ir.einsum("ik,kj->ij", left.node, right.node))


def einsum(
    equation: str,
    *operands: object,
    coefficient: ExactScalar = 1,
) -> VibeArray:
    """VibeQC extension for general contractions absent from the core subset."""
    arrays = tuple(_array(value, "einsum operand") for value in operands)
    return VibeArray(
        tensor_ir.einsum(
            equation,
            *(value.node for value in arrays),
            coefficient=_exact(coefficient, "einsum coefficient"),
        )
    )

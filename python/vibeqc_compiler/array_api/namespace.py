"""Bounded Array-API-like namespace lowering directly to TensorIR."""

from __future__ import annotations

import builtins
import typing
from fractions import Fraction

from vibeqc_compiler.tensor import ir as tensor_ir
from vibeqc_compiler.tensor.types import Index

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


def _shape(shape: object, name: str = "shape") -> tuple[int, ...]:
    if not isinstance(shape, tuple) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise TypeError(f"{name} must be a tuple of nonnegative integers")
    return typing.cast("tuple[int, ...]", shape)


def _indices(
    indices: object,
    *,
    shape: tuple[int, ...],
    operation: str,
) -> tuple[Index, ...]:
    if indices is None:
        raise TypeError(
            f"{operation} requires explicit TensorIR indices in the preview; "
            "integer shape alone cannot establish scientific axis identity"
        )
    if not isinstance(indices, tuple) or any(
        not isinstance(index, Index) for index in indices
    ):
        raise TypeError(
            f"{operation} indices must be a tuple of TensorIR Index objects"
        )
    result = typing.cast("tuple[Index, ...]", indices)
    if tuple(index.extent for index in result) != shape:
        raise ValueError(f"{operation} shape must match the explicit TensorIR indices")
    return result


def _axis(axis: object, rank: int, operation: str) -> int:
    if type(axis) is not int:
        raise TypeError(f"{operation} axis must be an integer")
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise ValueError(f"{operation} axis is out of range")
    return normalized


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


def reshape(
    x: object,
    shape: tuple[int, ...],
    *,
    indices: tuple[Index, ...] | None = None,
) -> VibeArray:
    """Row-major reshape with explicit scientific domains for every result axis."""
    value = _array(x)
    result_shape = _shape(shape)
    result_indices = _indices(indices, shape=result_shape, operation="reshape")
    return VibeArray(tensor_ir.reshape(value.node, result_indices))


def permute_dims(x: object, axes: tuple[int, ...]) -> VibeArray:
    value = _array(x)
    return VibeArray(tensor_ir.transpose(value.node, axes))


def broadcast_to(
    x: object,
    shape: tuple[int, ...],
    *,
    indices: tuple[Index, ...] | None = None,
    axes: tuple[int, ...] | None = None,
) -> VibeArray:
    """Explicit broadcast; inserted axes and the input-to-output map are required."""
    value = _array(x)
    result_shape = _shape(shape)
    result_indices = _indices(indices, shape=result_shape, operation="broadcast_to")
    if axes is None:
        raise TypeError(
            "broadcast_to requires an explicit input-to-output axis map in the preview"
        )
    if not isinstance(axes, tuple) or any(type(axis) is not int for axis in axes):
        raise TypeError("broadcast_to axes must be a tuple of integers")
    return VibeArray(tensor_ir.broadcast(value.node, result_indices, axes))


def slice(x: object, key: object) -> VibeArray:
    """Static rank-preserving slicing with nonnegative unit-step slices only."""
    value = _array(x)
    items = key if isinstance(key, tuple) else (key,)
    if len(items) > value.ndim:
        raise IndexError("too many indices for symbolic VibeArray")
    items = (*items, *(builtins.slice(None) for _ in range(value.ndim - len(items))))
    ranges: list[tuple[int, int]] = []
    for item, extent in zip(items, value.shape):
        if not isinstance(item, builtins.slice):
            raise TypeError(
                "symbolic VibeArray indexing supports rank-preserving slices only"
            )
        if item.step not in (None, 1):
            raise ValueError("symbolic VibeArray slices require unit step")
        start = 0 if item.start is None else item.start
        stop = extent if item.stop is None else item.stop
        if type(start) is not int or type(stop) is not int:
            raise TypeError("symbolic VibeArray slice bounds must be integers or None")
        if start < 0 or stop < 0:
            raise ValueError("symbolic VibeArray slice bounds must be nonnegative")
        ranges.append((start, stop))
    return VibeArray(tensor_ir.slice_tensor(value.node, tuple(ranges)))


def take(x: object, indices: tuple[int, ...], *, axis: int) -> VibeArray:
    """Gather static local positions along one existing scientific axis."""
    value = _array(x)
    selected_axis = _axis(axis, value.ndim, "take")
    if not isinstance(indices, tuple) or any(
        type(index) is not int for index in indices
    ):
        raise TypeError("take indices must be a tuple of static integers")
    return VibeArray(tensor_ir.gather(value.node, selected_axis, indices))


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

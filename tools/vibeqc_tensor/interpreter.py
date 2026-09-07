"""Independent-node NumPy reference execution for development and debugging.

This is not a production execution plan. The conservative byte budget counts
logical retained arrays and returned copies, not NumPy's internal scratch or
Python object overhead. GPU planning and AD belong to subsequent issues.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .ir import Node
from .program import Program
from .types import checked_size


@dataclass(frozen=True)
class Execution:
    """Detached outputs/debug snapshots; caller inputs are never modified."""

    outputs: dict[str, np.ndarray]
    intermediates: dict[str, np.ndarray]
    logical_retained_bytes: int
    backend: str = "numpy-cpu-interpreter"


def _coefficient(pair, dtype):
    # Fraction conversion avoids overflowing large integer numerator and
    # denominator separately when their ratio is small and representable.
    return np.dtype(dtype).type(float(Fraction(*pair)))


def _evaluate(node: Node, operands: list[np.ndarray], feeds: Mapping) -> np.ndarray:
    a, op = node.attrs, node.op
    if op == "input":
        name = a["name"]
        if name not in feeds:
            raise ValueError(f"missing tensor input: {name}")
        value = np.asarray(feeds[name])
        if value.dtype != np.dtype(node.spec.dtype) or value.shape != node.spec.shape:
            raise ValueError(
                f"input {name} must have real dtype {node.spec.dtype} and shape {node.spec.shape}"
            )
        tolerance = 1e-11 if node.spec.dtype == "float64" else 1e-6
        for symmetry in node.spec.symmetries:
            if not np.allclose(
                value,
                symmetry.sign * value.transpose(symmetry.permutation),
                atol=tolerance,
                rtol=10 * tolerance,
            ):
                raise ValueError(f"input {name} violates its declared symmetry")
        return value
    if op == "constant":
        return np.array(
            [_coefficient(v, node.spec.dtype) for v in a["values"]],
            dtype=node.spec.dtype,
        ).reshape(node.spec.shape)
    if op == "add":
        result = np.zeros(node.spec.shape, dtype=node.spec.dtype)
        for operand, coefficient in zip(operands, a["coefficients"]):
            result += _coefficient(coefficient, node.spec.dtype) * operand
        return result
    if op == "multiply":
        return operands[0] * operands[1]
    if op == "divide":
        if np.any(operands[1] == 0):
            raise ValueError("tensor division by zero")
        return operands[0] / operands[1]
    if op == "einsum":
        arguments = []
        for value, labels in zip(operands, a["labels"]):
            arguments.extend((value, list(labels)))
        # optimize=False keeps a deterministic direct reference contraction;
        # contraction-tree selection is a separate lowering concern.
        result = np.einsum(*arguments, list(a["output"]), optimize=False)
        return result * _coefficient(a["coefficient"], node.spec.dtype)
    value = operands[0]
    if op == "transpose":
        return value.transpose(a["axes"])
    if op == "reshape":
        return value.reshape(node.spec.shape, order="C")
    if op == "slice":
        return value[tuple(slice(start, stop) for start, stop in a["ranges"])]
    if op == "gather":
        return np.take(value, np.asarray(a["positions"], dtype=np.intp), axis=a["axis"])
    if op == "reduce":
        return np.sum(value, axis=a["axes"], dtype=node.spec.dtype)
    if op == "broadcast":
        # The map need not preserve input order: transpose before inserting
        # singleton storage axes so each population lands on its declared slot.
        order = tuple(sorted(range(value.ndim), key=lambda i: a["axes"][i]))
        shape = [1] * len(node.spec.indices)
        for i, axis in enumerate(a["axes"]):
            shape[axis] = value.shape[i]
        return np.broadcast_to(value.transpose(order).reshape(shape), node.spec.shape)
    raise ValueError(f"unsupported interpreter primitive: {op}")


def execute(
    program: Program,
    feeds: Mapping,
    *,
    debug: bool = False,
    max_bytes: int = 256 * 1024 * 1024,
) -> Execution:
    """Evaluate live nodes in order, checking shapes, dtypes, and finiteness.

    Noncontiguous/negative-stride input arrays and read-only views are legal.
    Feed dictionaries may contain unused inputs so original and optimized
    programs share a fixture. Returned arrays never alias each other, inputs,
    or interpreter views. Every output/debug entry is an independent snapshot.
    """
    if not isinstance(program, Program) or not isinstance(feeds, Mapping):
        raise TypeError("execute requires a Program and input mapping")
    checked_size(max_bytes, "interpreter byte budget")
    nodes = program.live_nodes
    retained = sum(n.spec.size * n.spec.itemsize for n in nodes)
    retained += sum(n.spec.size * n.spec.itemsize for n in program.outputs.values())
    if debug:
        retained += sum(n.spec.size * n.spec.itemsize for n in nodes)
    if retained > max_bytes:
        raise ValueError("tensor interpreter logical retained-byte budget exceeded")
    values, snapshots = {}, {}
    names = program.debug_names
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        for node in nodes:
            try:
                value = np.asarray(
                    _evaluate(node, [values[n] for n in node.inputs], feeds)
                )
            except (FloatingPointError, OverflowError) as exc:
                raise ValueError(
                    f"non-finite arithmetic at {names[node]} ({node.op})"
                ) from exc
            if value.shape != node.spec.shape or value.dtype != np.dtype(
                node.spec.dtype
            ):
                raise ValueError(
                    f"interpreter result violates {node.op} shape/dtype contract"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite tensor at {names[node]} ({node.op})")
            # A fresh ndarray header prevents setflags from changing the
            # caller's input or the writeability of another shared view.
            value = value.view()
            value.flags.writeable = False
            values[node] = value
            if debug:
                snapshots[names[node]] = value.copy()
    return Execution(
        {name: values[node].copy() for name, node in program.outputs.items()},
        snapshots,
        retained,
    )

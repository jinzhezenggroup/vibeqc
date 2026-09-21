"""Independent-node NumPy reference execution for development and debugging.

This is not a production execution plan. The conservative byte budget counts
logical retained arrays and returned copies, not NumPy's internal scratch or
Python object overhead. GPU planning and AD belong to subsequent issues.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .precision import describe_precision
from .program import Program
from .scaled_arithmetic import scaled_bilinear_value
from .types import checked_size

if typing.TYPE_CHECKING:
    from .ir import Node


@dataclass(frozen=True)
class Execution:
    """Detached outputs/debug snapshots; caller inputs are never modified."""

    outputs: dict[str, np.ndarray]
    intermediates: dict[str, np.ndarray]
    logical_retained_bytes: int
    backend: str = "numpy-cpu-interpreter"


def _coefficient(pair: typing.Any, dtype: typing.Any) -> typing.Any:
    # Fraction conversion avoids overflowing large integer numerator and
    # denominator separately when their ratio is small and representable.
    return np.dtype(dtype).type(float(Fraction(*pair)))


def _mixed_einsum(
    node: Node, operands: list[np.ndarray], accumulation_dtype: str
) -> np.ndarray:
    """Reference FP32 term arithmetic with an explicit wider serial accumulator."""
    a = node.attrs
    compute = np.dtype(node.spec.dtype).type
    accumulate = np.dtype(accumulation_dtype).type
    domains: dict[int, int] = {}
    for value, labels in zip(operands, a["labels"], strict=True):
        domains.update(zip(labels, value.shape, strict=True))
    reduced = tuple(label for label in sorted(domains) if label not in a["output"])
    output_shape = tuple(domains[label] for label in a["output"])
    reduction_shape = tuple(domains[label] for label in reduced)
    result = np.empty(output_shape, dtype=node.spec.dtype)
    coefficient = compute(_coefficient(a["coefficient"], node.spec.dtype))
    for output_index in np.ndindex(output_shape):
        coordinates = dict(zip(a["output"], output_index, strict=True))
        total = accumulate(0.0)
        for reduction_index in np.ndindex(reduction_shape):
            coordinates.update(zip(reduced, reduction_index, strict=True))
            term = compute(1.0)
            for value, labels in zip(operands, a["labels"], strict=True):
                index = tuple(coordinates[label] for label in labels)
                term = compute(term * compute(value[index]))
            total = accumulate(total + accumulate(term))
        result[output_index] = compute(compute(total) * coefficient)
    return result


def _evaluate(
    node: Node,
    operands: list[np.ndarray],
    feeds: Mapping,
    accumulation_dtype: str | None = None,
) -> np.ndarray:
    a, op = node.attrs, node.op
    accumulation_dtype = (
        node.spec.dtype if accumulation_dtype is None else accumulation_dtype
    )
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
            if node.spec.dtype == "int64":
                raise ValueError("int64 TensorIR controls cannot declare symmetry")
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
    if op == "cast":
        return operands[0].astype(node.spec.dtype, copy=True)
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
    if op == "scaled_bilinear":
        return scaled_bilinear_value(*operands)
    if op in ("exp", "log", "sqrt", "power"):
        value = operands[0]
        if op in ("log", "power") and np.any(value <= 0):
            raise ValueError(f"tensor {op} domain requires strictly positive input")
        if op == "sqrt" and np.any(value < 0):
            raise ValueError("tensor sqrt domain requires nonnegative input")
        if op == "power":
            return np.power(value, _coefficient(a["exponent"], node.spec.dtype))
        return {"exp": np.exp, "log": np.log, "sqrt": np.sqrt}[op](value)
    if op == "einsum":
        if accumulation_dtype != node.spec.dtype:
            return _mixed_einsum(node, operands, accumulation_dtype)
        arguments = []
        for value, labels in zip(operands, a["labels"]):
            arguments.extend((value, list(labels)))
        # optimize=False keeps a deterministic direct reference contraction;
        # contraction-tree selection is a separate lowering concern.
        result = np.einsum(*arguments, list(a["output"]), optimize=False)
        return result * _coefficient(a["coefficient"], node.spec.dtype)
    if op == "runtime_indexed_select":
        value, maps = operands[0], operands[1:]
        axes = tuple(a["axes"])
        result = np.empty(node.spec.shape, dtype=node.spec.dtype)
        for mapping, axis in zip(maps, axes, strict=True):
            if np.any(mapping < 0) or np.any(mapping >= value.shape[axis]):
                raise ValueError(
                    "runtime_indexed_select coordinate is outside its source axis"
                )
        selected = dict(zip(axes, maps, strict=True))
        for domain_coordinate in range(node.spec.shape[0]):
            source = tuple(
                int(selected[axis][domain_coordinate])
                if axis in selected
                else slice(None)
                for axis in range(value.ndim)
            )
            result[domain_coordinate] = value[source]
        return result
    value = operands[0]
    if op == "transpose":
        return value.transpose(a["axes"])
    if op == "reshape":
        return value.reshape(node.spec.shape, order="C")
    if op == "slice":
        return value[tuple(slice(start, stop) for start, stop in a["ranges"])]
    if op in ("gather", "indexed_gather"):
        return np.take(value, np.asarray(a["positions"], dtype=np.intp), axis=a["axis"])
    if op == "scatter_add":
        result = np.zeros(node.spec.shape, dtype=node.spec.dtype)
        np.add.at(
            np.moveaxis(result, a["axis"], 0),
            np.asarray(a["positions"], dtype=np.intp),
            np.moveaxis(value, a["axis"], 0),
        )
        return result
    if op == "segment_sum":
        result = np.zeros(node.spec.shape, dtype=node.spec.dtype)
        source = np.moveaxis(value, a["axis"], 0)
        target = np.moveaxis(result, a["axis"], 0)
        for segment, (start, stop) in enumerate(zip(a["offsets"], a["offsets"][1:])):
            target[segment] = np.sum(source[start:stop], axis=0, dtype=node.spec.dtype)
        return result
    if op == "reduce":
        if accumulation_dtype == node.spec.dtype:
            return np.sum(value, axis=a["axes"], dtype=node.spec.dtype)
        # Match the generated CUDA reduction exactly: iterate reduced
        # coordinates in C order and widen each FP32 term before one serial
        # FP64 round-to-nearest addition. NumPy's pairwise sum is intentionally
        # avoided here because it is a different numerical program.
        axes = tuple(a["axes"])
        kept = tuple(axis for axis in range(value.ndim) if axis not in axes)
        output_shape = tuple(value.shape[axis] for axis in kept)
        reduction_shape = tuple(value.shape[axis] for axis in axes)
        result = np.empty(output_shape, dtype=node.spec.dtype)
        accumulate = np.dtype(accumulation_dtype).type
        compute = np.dtype(node.spec.dtype).type
        for output_index in np.ndindex(output_shape):
            coordinates = dict(zip(kept, output_index, strict=True))
            total = accumulate(0.0)
            for reduction_index in np.ndindex(reduction_shape):
                coordinates.update(zip(axes, reduction_index, strict=True))
                index = tuple(coordinates[axis] for axis in range(value.ndim))
                total = accumulate(total + accumulate(compute(value[index])))
            result[output_index] = compute(total)
        return result
    if op == "broadcast":
        # The map need not preserve input order: transpose before inserting
        # singleton storage axes so each population lands on its declared slot.
        order = tuple(sorted(range(value.ndim), key=lambda i: a["axes"][i]))
        shape = [1] * len(node.spec.indices)
        for i, axis in enumerate(a["axes"]):
            shape[axis] = value.shape[i]
        return np.broadcast_to(value.transpose(order).reshape(shape), node.spec.shape)
    raise ValueError(f"unsupported interpreter primitive: {op}")


def _run(
    program: Program,
    feeds: Mapping,
    *,
    debug: bool,
    max_bytes: int,
) -> tuple[dict[Node, np.ndarray], dict[str, np.ndarray], int]:
    """Shared single-pass evaluator used by execution and derivative rules."""
    if not isinstance(program, Program) or not isinstance(feeds, Mapping):
        raise TypeError("tensor evaluation requires a Program and input mapping")
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
    precision = (
        {value.name: value for value in describe_precision(program).values}
        if program.provenance.get("precision_execution") is not None
        else None
    )
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        for node in nodes:
            try:
                value = np.asarray(
                    _evaluate(
                        node,
                        [values[n] for n in node.inputs],
                        feeds,
                        (
                            None
                            if precision is None or names[node] not in precision
                            else precision[names[node]].accumulation_dtype
                        ),
                    )
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
    return values, snapshots, retained


def evaluate_nodes(
    program: Program,
    feeds: Mapping,
    *,
    max_bytes: int = 256 * 1024 * 1024,
) -> dict[Node, np.ndarray]:
    """Return one read-only primal value per live SSA node.

    This is the internal execution snapshot consumed by derivative rules.  It
    is deliberately not a public result object: callers must treat every array
    as immutable and must not retain it after the next evaluation, because
    values may be views into interpreter-owned storage.
    """
    values, _, _ = _run(program, feeds, debug=False, max_bytes=max_bytes)
    return values


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
    values, snapshots, retained = _run(program, feeds, debug=debug, max_bytes=max_bytes)
    return Execution(
        {name: values[node].copy() for name, node in program.outputs.items()},
        snapshots,
        retained,
    )

"""Bounded Array-namespace reference execution for TensorIR.

This module deliberately implements only operations with a portable immutable
Array API-style lowering (plus an optional NumPy-like ``einsum`` extension).
Unsupported primitives fail explicitly instead of falling back through NumPy,
which would silently move device arrays to the host.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from fractions import Fraction

from .precision import describe_precision
from .program import Program
from .types import checked_size

if typing.TYPE_CHECKING:
    from .ir import Node


class NamespaceCapabilityError(RuntimeError):
    """The requested array namespace cannot preserve an interpreter contract."""


def _name(namespace: typing.Any) -> str:
    return str(getattr(namespace, "__name__", type(namespace).__name__))


def _require(namespace: typing.Any, function: str) -> typing.Callable[..., typing.Any]:
    value = getattr(namespace, function, None)
    if not callable(value):
        raise NamespaceCapabilityError(
            f"array namespace {_name(namespace)!r} lacks required function {function!r}"
        )
    return value


def _dtype(namespace: typing.Any, name: str) -> typing.Any:
    dtype = getattr(namespace, name, None)
    if dtype is None:
        raise NamespaceCapabilityError(
            f"array namespace {_name(namespace)!r} lacks required dtype {name!r}"
        )
    return dtype


def _device(feeds: Mapping, names: set[str]) -> typing.Any | None:
    devices = [getattr(feeds[name], "device", None) for name in names if name in feeds]
    devices = [value for value in devices if value is not None]
    if not devices:
        return None
    first = devices[0]
    if any(value != first for value in devices[1:]):
        raise ValueError("tensor namespace interpreter requires feeds on one device")
    return first


def _placed_kwargs(device: typing.Any | None) -> dict[str, typing.Any]:
    return {} if device is None else {"device": device}


def _asarray(
    namespace: typing.Any,
    value: typing.Any,
    *,
    dtype: typing.Any | None = None,
    device: typing.Any | None = None,
) -> typing.Any:
    function = _require(namespace, "asarray")
    kwargs = _placed_kwargs(device)
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        return function(value, **kwargs)
    except TypeError as exc:
        if device is not None:
            raise NamespaceCapabilityError(
                f"array namespace {_name(namespace)!r} cannot preserve explicit device placement"
            ) from exc
        raise


def _copy(namespace: typing.Any, value: typing.Any) -> typing.Any:
    function = _require(namespace, "asarray")
    try:
        return function(value, copy=True)
    except TypeError:
        copier = getattr(value, "copy", None)
        if callable(copier):
            return copier()
        raise NamespaceCapabilityError(
            f"array namespace {_name(namespace)!r} cannot publish detached result copies"
        ) from None


def _scalar(
    namespace: typing.Any,
    pair: typing.Any,
    dtype_name: str,
    device: typing.Any | None,
) -> typing.Any:
    return _asarray(
        namespace,
        float(Fraction(*pair)),
        dtype=_dtype(namespace, dtype_name),
        device=device,
    )


def _truth(value: typing.Any) -> bool:
    try:
        return bool(value)
    except (TypeError, ValueError) as exc:
        raise NamespaceCapabilityError(
            "array namespace does not expose scalar truth values required for validation"
        ) from exc


def _validate_feed_device(raw: typing.Any, value: typing.Any, name: str) -> None:
    source = getattr(raw, "device", None)
    if source is None:
        return
    target = getattr(value, "device", None)
    if target is None or target != source:
        raise NamespaceCapabilityError(
            f"input {name} changed device while entering the requested array namespace"
        )


def _validate_symmetry(
    namespace: typing.Any,
    node: Node,
    value: typing.Any,
    name: str,
) -> None:
    if not node.spec.symmetries:
        return
    if node.spec.dtype == "int64":
        raise ValueError("int64 TensorIR controls cannot declare symmetry")
    permute = _require(namespace, "permute_dims")
    absolute = _require(namespace, "abs")
    all_ = _require(namespace, "all")
    tolerance = 1e-11 if node.spec.dtype == "float64" else 1e-6
    for symmetry in node.spec.symmetries:
        expected = symmetry.sign * permute(value, symmetry.permutation)
        error = absolute(value - expected)
        bound = tolerance + 10 * tolerance * absolute(expected)
        if not _truth(all_(error <= bound)):
            raise ValueError(f"input {name} violates its declared symmetry")


def _unsupported(namespace: typing.Any, op: str, reason: str) -> typing.NoReturn:
    raise NamespaceCapabilityError(
        f"TensorIR primitive {op!r} is not portable through array namespace "
        f"{_name(namespace)!r}: {reason}"
    )


def _evaluate(
    node: Node,
    operands: list[typing.Any],
    feeds: Mapping,
    namespace: typing.Any,
    device: typing.Any | None,
    accumulation_dtype: str | None,
) -> typing.Any:
    attrs, op = node.attrs, node.op
    dtype = _dtype(namespace, node.spec.dtype)
    accumulation_dtype = (
        node.spec.dtype if accumulation_dtype is None else accumulation_dtype
    )
    if accumulation_dtype != node.spec.dtype:
        _unsupported(
            namespace,
            op,
            "mixed accumulation order is a NumPy-reference numerical contract",
        )
    if op == "input":
        name = attrs["name"]
        if name not in feeds:
            raise ValueError(f"missing tensor input: {name}")
        raw = feeds[name]
        value = _asarray(namespace, raw)
        _validate_feed_device(raw, value, name)
        if value.dtype != dtype or tuple(value.shape) != node.spec.shape:
            raise ValueError(
                f"input {name} must have real dtype {node.spec.dtype} and shape {node.spec.shape}"
            )
        _validate_symmetry(namespace, node, value, name)
        return value
    if op == "constant":
        values = [float(Fraction(*value)) for value in attrs["values"]]
        value = _asarray(namespace, values, dtype=dtype, device=device)
        return _require(namespace, "reshape")(value, node.spec.shape)
    if op == "cast":
        return _require(namespace, "astype")(operands[0], dtype, copy=True)
    if op == "add":
        result = _require(namespace, "zeros")(
            node.spec.shape, dtype=dtype, **_placed_kwargs(device)
        )
        for operand, coefficient in zip(operands, attrs["coefficients"], strict=True):
            result = (
                result
                + _scalar(namespace, coefficient, node.spec.dtype, device) * operand
            )
        return result
    if op == "multiply":
        return operands[0] * operands[1]
    if op == "divide":
        if _truth(_require(namespace, "any")(operands[1] == 0)):
            raise ValueError("tensor division by zero")
        return operands[0] / operands[1]
    if op == "scaled_bilinear":
        _unsupported(
            namespace,
            op,
            "range-safe frexp/ldexp residual arithmetic has no declared portable lowering",
        )
    if op in ("exp", "log", "sqrt", "power"):
        value = operands[0]
        if op in ("log", "power") and _truth(_require(namespace, "any")(value <= 0)):
            raise ValueError(f"tensor {op} domain requires strictly positive input")
        if op == "sqrt" and _truth(_require(namespace, "any")(value < 0)):
            raise ValueError("tensor sqrt domain requires nonnegative input")
        if op == "power":
            return _require(namespace, "pow")(
                value, _scalar(namespace, attrs["exponent"], node.spec.dtype, device)
            )
        return _require(namespace, op)(value)
    if op == "einsum":
        einsum = getattr(namespace, "einsum", None)
        if not callable(einsum):
            _unsupported(
                namespace, op, "the namespace does not provide an einsum extension"
            )
        einsum = typing.cast("typing.Callable[..., typing.Any]", einsum)
        arguments: list[typing.Any] = []
        for value, labels in zip(operands, attrs["labels"], strict=True):
            arguments.extend((value, list(labels)))
        try:
            result = einsum(*arguments, list(attrs["output"]), optimize=False)
        except TypeError as exc:
            raise NamespaceCapabilityError(
                f"array namespace {_name(namespace)!r} einsum extension cannot "
                "preserve deterministic optimize=False execution"
            ) from exc
        return result * _scalar(
            namespace, attrs["coefficient"], node.spec.dtype, device
        )
    if op in {
        "runtime_indexed_select",
        "runtime_indexed_scatter_add",
        "scatter_add",
        "segment_sum",
    }:
        _unsupported(
            namespace,
            op,
            "portable immutable indexed update semantics are not declared for B2",
        )
    value = operands[0]
    if op == "transpose":
        return _require(namespace, "permute_dims")(value, attrs["axes"])
    if op == "reshape":
        return _require(namespace, "reshape")(value, node.spec.shape)
    if op == "slice":
        return value[tuple(slice(start, stop) for start, stop in attrs["ranges"])]
    if op in ("gather", "indexed_gather"):
        positions = _asarray(
            namespace,
            attrs["positions"],
            dtype=_dtype(namespace, "int64"),
            device=device,
        )
        return _require(namespace, "take")(value, positions, axis=attrs["axis"])
    if op == "reduce":
        return _require(namespace, "sum")(value, axis=attrs["axes"], dtype=dtype)
    if op == "broadcast":
        order = tuple(sorted(range(value.ndim), key=lambda index: attrs["axes"][index]))
        shape = [1] * len(node.spec.indices)
        for index, axis in enumerate(attrs["axes"]):
            shape[axis] = value.shape[index]
        value = _require(namespace, "permute_dims")(value, order)
        value = _require(namespace, "reshape")(value, tuple(shape))
        return _require(namespace, "broadcast_to")(value, node.spec.shape)
    raise ValueError(f"unsupported interpreter primitive: {op}")


def execute_namespace(
    program: Program,
    feeds: Mapping,
    namespace: typing.Any,
    *,
    debug: bool,
    max_bytes: int,
) -> tuple[dict[str, typing.Any], dict[str, typing.Any], int, str]:
    """Execute a bounded TensorIR subset without crossing through NumPy."""
    if not isinstance(program, Program) or not isinstance(feeds, Mapping):
        raise TypeError("tensor evaluation requires a Program and input mapping")
    if namespace is None:
        raise TypeError("array namespace must not be None")
    checked_size(max_bytes, "interpreter byte budget")
    nodes = program.live_nodes
    retained = sum(node.spec.size * node.spec.itemsize for node in nodes)
    retained += sum(
        node.spec.size * node.spec.itemsize for node in program.outputs.values()
    )
    if debug:
        retained += sum(node.spec.size * node.spec.itemsize for node in nodes)
    if retained > max_bytes:
        raise ValueError("tensor interpreter logical retained-byte budget exceeded")

    input_names = {node.attrs["name"] for node in nodes if node.op == "input"}
    device = _device(feeds, input_names)
    values: dict[Node, typing.Any] = {}
    snapshots: dict[str, typing.Any] = {}
    names = program.debug_names
    precision = (
        {value.name: value for value in describe_precision(program).values}
        if program.provenance.get("precision_execution") is not None
        else None
    )
    isfinite = _require(namespace, "isfinite")
    all_ = _require(namespace, "all")
    for node in nodes:
        try:
            value = _evaluate(
                node,
                [values[input_node] for input_node in node.inputs],
                feeds,
                namespace,
                device,
                (
                    None
                    if precision is None or names[node] not in precision
                    else precision[names[node]].accumulation_dtype
                ),
            )
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError(
                f"non-finite arithmetic at {names[node]} ({node.op})"
            ) from exc
        if tuple(value.shape) != node.spec.shape or value.dtype != _dtype(
            namespace, node.spec.dtype
        ):
            raise ValueError(
                f"interpreter result violates {node.op} shape/dtype contract"
            )
        if not _truth(all_(isfinite(value))):
            raise ValueError(f"non-finite tensor at {names[node]} ({node.op})")
        values[node] = value
        if debug:
            snapshots[names[node]] = _copy(namespace, value)

    outputs = {
        name: _copy(namespace, values[node]) for name, node in program.outputs.items()
    }
    return outputs, snapshots, retained, f"array-namespace:{_name(namespace)}"

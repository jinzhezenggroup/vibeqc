"""Primitive JVP/VJP rules for TensorIR (CPU reference, slice A of #151).

The rules are matrix-free: a VJP propagates one cotangent through one
primal evaluation and never materializes a Jacobian.  A JVP evaluates one
tangent direction through the same immutable SSA graph.  The module covers
every primitive in :data:`vibeqc_compiler.tensor.ir.PRIMITIVES`.

Slice A deliberately stops at dense general tensors.  Differentiating with
respect to a packed/symmetric parameter requires the transpose of the
pack/unpack maps and is slice B of #151.  Requests for such tangent or
cotangent spaces fail closed instead of silently using an unweighted
Euclidean adjoint.  Derivative-DAG generation, checkpointing, and CUDA
lowering are likewise outside this slice.
"""

from __future__ import annotations

import hashlib
import json
import math
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from types import MappingProxyType

import numpy as np

from .interpreter import evaluate_nodes
from .ir import Node, _execution_power_exponent
from .program import Program
from .scaled_arithmetic import scaled_bilinear_value
from .types import checked_size

AD_SCHEMA = "vibeqc.tensor.autodiff"
AD_VERSION = 1
AD_RULE_VERSION = 3
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
BACKEND = "numpy-cpu-autodiff"


def _coefficient(pair: typing.Any, dtype: typing.Any) -> typing.Any:
    """Convert a reduced rational pair without importing execution state."""
    return np.dtype(dtype).type(float(Fraction(*pair)))


def _digest(payload: Mapping) -> str:
    """Hash JSON-compatible provenance with a canonical byte spelling."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _identity_hash(primal_hash: str, mode: str, selection: Mapping) -> str:
    """Link one derivative computation to its primal equation identity."""
    return _digest(
        {
            "schema": AD_SCHEMA,
            "schema_version": AD_VERSION,
            "rule_version": AD_RULE_VERSION,
            "primal_logical_hash": primal_hash,
            "mode": mode,
            "selection": selection,
        }
    )


@dataclass(frozen=True)
class JVPResult:
    """Forward-direction derivatives keyed by public output name."""

    output_tangents: Mapping[str, np.ndarray]
    primal_logical_hash: str
    tangent_inputs: tuple[str, ...]
    output_names: tuple[str, ...]
    rule_version: int = AD_RULE_VERSION
    backend: str = BACKEND

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_tangents",
            MappingProxyType(dict(sorted(self.output_tangents.items()))),
        )
        object.__setattr__(self, "tangent_inputs", tuple(self.tangent_inputs))
        object.__setattr__(self, "output_names", tuple(self.output_names))

    @property
    def derivative_hash(self) -> str:
        """Content identity for this JVP, linked to the primal equation."""
        return _identity_hash(
            self.primal_logical_hash,
            "jvp",
            {
                "tangent_inputs": list(self.tangent_inputs),
                "output_names": list(self.output_names),
            },
        )

    def provenance(self) -> dict:
        """Return a JSON-serializable link back to the primal definition."""
        return {
            "schema": AD_SCHEMA,
            "schema_version": AD_VERSION,
            "rule_version": self.rule_version,
            "mode": "jvp",
            "primal_logical_hash": self.primal_logical_hash,
            "derivative_hash": self.derivative_hash,
            "tangent_inputs": list(self.tangent_inputs),
            "output_names": list(self.output_names),
            "backend": self.backend,
        }


@dataclass(frozen=True)
class VJPResult:
    """Reverse-direction derivatives keyed by differentiable input name."""

    input_cotangents: Mapping[str, np.ndarray]
    primal_logical_hash: str
    cotangent_outputs: tuple[str, ...]
    input_names: tuple[str, ...]
    rule_version: int = AD_RULE_VERSION
    backend: str = BACKEND

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_cotangents",
            MappingProxyType(dict(sorted(self.input_cotangents.items()))),
        )
        object.__setattr__(self, "cotangent_outputs", tuple(self.cotangent_outputs))
        object.__setattr__(self, "input_names", tuple(self.input_names))

    @property
    def derivative_hash(self) -> str:
        """Content identity for this VJP, linked to the primal equation."""
        return _identity_hash(
            self.primal_logical_hash,
            "vjp",
            {
                "cotangent_outputs": list(self.cotangent_outputs),
                "input_names": list(self.input_names),
            },
        )

    def provenance(self) -> dict:
        """Return a JSON-serializable link back to the primal definition."""
        return {
            "schema": AD_SCHEMA,
            "schema_version": AD_VERSION,
            "rule_version": self.rule_version,
            "mode": "vjp",
            "primal_logical_hash": self.primal_logical_hash,
            "derivative_hash": self.derivative_hash,
            "cotangent_outputs": list(self.cotangent_outputs),
            "input_names": list(self.input_names),
            "backend": self.backend,
        }


@dataclass(frozen=True)
class DotTestResult:
    """Result of ``<w, Jv> == <J^T w, v>`` for one tangent/cotangent pair."""

    lhs: float
    rhs: float
    absolute_error: float
    relative_error: float
    passed: bool
    atol: float
    rtol: float


def capabilities() -> dict:
    """Describe exactly what this slice implements; no promotion is implied."""
    return {
        "schema": AD_SCHEMA,
        "schema_version": AD_VERSION,
        "rule_version": AD_RULE_VERSION,
        "primitives": sorted(AD_PRIMITIVES),
        "modes": ["jvp", "vjp"],
        "backend": BACKEND,
        "packed_symmetry": False,
        "derivative_dag_generation": False,
        "bounded_recomputation": False,
        "cuda": False,
    }


def _input_groups(program: Program) -> dict[str, tuple[Node, ...]]:
    """Group every live occurrence of a public feed, before optional CSE.

    Program validates that same-name inputs have compatible logical types.
    Distinct SSA nodes still read one parameter, so their reverse contributions
    must be summed and all occurrences must seed derivative reachability.
    """
    groups: dict[str, list[Node]] = {}
    for node in program.live_nodes:
        if node.op == "input":
            groups.setdefault(node.attrs["name"], []).append(node)
    return {name: tuple(nodes) for name, nodes in groups.items()}


def _input_nodes(program: Program) -> dict[str, Node]:
    """Pick one representative for name/type validation, not accumulation."""
    return {name: nodes[0] for name, nodes in _input_groups(program).items()}


def _input_cotangent(nodes: typing.Any, bars: typing.Any) -> np.ndarray:
    """Return an owned cotangent summed over all uses of one public feed."""
    result = np.zeros(nodes[0].spec.shape, dtype=nodes[0].spec.dtype)
    for node in nodes:
        result += bars[node]
    if not np.isfinite(result).all():
        raise ValueError("non-finite VJP accumulation for a named input")
    return result


def _validate_vector(value: typing.Any, spec: typing.Any, label: str) -> np.ndarray:
    """Validate a tangent or cotangent against its exact logical contract."""
    array = np.asarray(value)
    if array.shape != spec.shape or array.dtype != np.dtype(spec.dtype):
        raise ValueError(
            f"{label} must have real dtype {spec.dtype} and shape {spec.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def _validate_tangents(program: Program, tangents: Mapping) -> dict[str, np.ndarray]:
    """Accept tangents only for declared differentiable, general inputs."""
    if not isinstance(tangents, Mapping):
        raise TypeError("tangents must be a mapping")
    inputs = _input_nodes(program)
    result = {}
    for name, value in tangents.items():
        if not isinstance(name, str) or name not in inputs:
            raise ValueError(f"unknown tangent input: {name!r}")
        node = inputs[name]
        if not node.spec.differentiable:
            raise ValueError(
                f"input {name} is not declared differentiable; refusing a tangent"
            )
        if node.spec.symmetries:
            raise ValueError(
                "packed/symmetric tangent spaces require slice B of #151; "
                f"input {name} declares symmetries"
            )
        result[name] = _validate_vector(value, node.spec, f"tangent {name}")
    return result


def _validate_cotangents(
    program: Program, cotangents: Mapping
) -> dict[str, np.ndarray]:
    """Accept cotangents only for outputs whose SSA value is differentiable."""
    if not isinstance(cotangents, Mapping):
        raise TypeError("cotangents must be a mapping")
    result = {}
    for name, value in cotangents.items():
        if not isinstance(name, str) or name not in program.outputs:
            raise ValueError(f"unknown cotangent output: {name!r}")
        node = program.outputs[name]
        if not node.spec.differentiable:
            raise ValueError(
                f"output {name} is not differentiable; refusing a cotangent"
            )
        result[name] = _validate_vector(value, node.spec, f"cotangent {name}")
    return result


def _select_names(mapping: Mapping, names: typing.Any, label: str) -> dict:
    """Select requested names without silently accepting typos or duplicates."""
    if names is None:
        return dict(mapping)
    if isinstance(names, str):
        raise TypeError(f"{label} names must be an iterable, not a string")
    selected = {}
    for name in names:
        if not isinstance(name, str) or name not in mapping:
            raise ValueError(f"unknown {label} name: {name!r}")
        if name in selected:
            raise ValueError(f"duplicate {label} name: {name}")
        selected[name] = mapping[name]
    if not selected:
        raise ValueError(f"at least one {label} name is required")
    return selected


def _require_general_inputs(program: Program, names: typing.Any) -> None:
    """Fail closed before returning an unweighted packed-space adjoint."""
    inputs = _input_nodes(program)
    for name in names:
        if inputs[name].spec.symmetries:
            raise ValueError(
                "packed/symmetric cotangent spaces require slice B of #151; "
                f"input {name} declares symmetries"
            )


def _check_budget(
    program: Program, arrays: Mapping[Node, np.ndarray], max_bytes: int
) -> None:
    """Conservatively bound retained primal values plus derivative arrays."""
    primal = sum(node.spec.size * node.spec.itemsize for node in program.live_nodes)
    derivative = sum(np.asarray(value).nbytes for value in arrays.values())
    if primal + derivative > max_bytes:
        raise ValueError("autodiff logical retained-byte budget exceeded")


def _zeros(spec: typing.Any) -> np.ndarray:
    return np.zeros(spec.shape, dtype=spec.dtype)


def _transcendental_partial(
    node: Node, values: typing.Any, weight: typing.Any
) -> np.ndarray:
    """Weighted scalar partials; primal/domain validation precedes reference AD."""
    x = values[0]
    if node.op == "exp":
        return np.exp(x) * weight
    if node.op == "log":
        # Divide the seed directly: forming 1/x first overflows on subnormals.
        return weight / x
    if node.op == "sqrt":
        if np.any(x == 0):
            raise ValueError("tensor sqrt derivative requires strictly positive input")
        return weight / (np.dtype(node.spec.dtype).type(2) * np.sqrt(x))
    p = _execution_power_exponent(node.attrs["exponent"], node.spec.dtype)
    if p == 0:
        return np.zeros_like(weight)
    if p == 1:
        return weight.copy()
    factor = np.dtype(node.spec.dtype).type(float(p))
    exponent = np.dtype(node.spec.dtype).type(float(p - 1))
    # Do not reuse rounded x**p / x: x**p may underflow while the slope does not.
    return (weight * np.power(x, exponent)) * factor


def _jvp_transcendental(
    node: Node, values: typing.Any, tangents: typing.Any
) -> np.ndarray:
    return _transcendental_partial(node, values, tangents[0])


def _vjp_transcendental(
    node: Node, values: typing.Any, bar: typing.Any
) -> list[np.ndarray]:
    return [_transcendental_partial(node, values, bar)]


def _jvp_cast(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    return tangents[0].astype(node.spec.dtype, copy=True)


def _vjp_cast(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    return [bar.astype(node.inputs[0].spec.dtype, copy=True)]


def _jvp_add(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    result = _zeros(node.spec)
    for tangent, coefficient in zip(tangents, node.attrs["coefficients"]):
        result += _coefficient(coefficient, node.spec.dtype) * tangent
    return result


def _jvp_multiply(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    return tangents[0] * values[1] + values[0] * tangents[1]


def _jvp_divide(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    x, y = values
    dx, dy = tangents
    return scaled_bilinear_value(dx, y, x, dy, y, y)


def _jvp_einsum(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    labels = node.attrs["labels"]
    output = list(node.attrs["output"])
    coefficient = _coefficient(node.attrs["coefficient"], node.spec.dtype)
    result = _zeros(node.spec)
    for differentiated in range(len(node.inputs)):
        arguments = []
        for operand, (value, tangent, label) in enumerate(
            zip(values, tangents, labels)
        ):
            arguments.extend(
                (tangent if operand == differentiated else value, list(label))
            )
        result += np.einsum(*arguments, output, optimize=False) * coefficient
    return result


def _jvp_transpose(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    return np.transpose(tangents[0], node.attrs["axes"])


def _jvp_reshape(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    return tangents[0].reshape(node.spec.shape, order="C")


def _jvp_slice(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    return tangents[0][
        tuple(slice(start, stop) for start, stop in node.attrs["ranges"])
    ]


def _jvp_gather(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    return np.take(
        tangents[0],
        np.asarray(node.attrs["positions"], dtype=np.intp),
        axis=node.attrs["axis"],
    )


def _jvp_scatter_add(
    node: Node,
    values: typing.Sequence[np.ndarray],
    tangents: typing.Sequence[np.ndarray],
) -> np.ndarray:
    result = _zeros(node.spec)
    index = [slice(None)] * result.ndim
    index[node.attrs["axis"]] = np.asarray(node.attrs["positions"], dtype=np.intp)
    np.add.at(result, tuple(index), tangents[0])
    return result


def _jvp_segment_sum(
    node: Node,
    values: typing.Sequence[np.ndarray],
    tangents: typing.Sequence[np.ndarray],
) -> np.ndarray:
    result = _zeros(node.spec)
    source = np.moveaxis(tangents[0], node.attrs["axis"], 0)
    target = np.moveaxis(result, node.attrs["axis"], 0)
    for segment, (start, stop) in enumerate(
        zip(node.attrs["offsets"], node.attrs["offsets"][1:])
    ):
        target[segment] = np.sum(source[start:stop], axis=0, dtype=node.spec.dtype)
    return result


def _jvp_runtime_indexed_select(
    node: Node,
    values: typing.Sequence[np.ndarray],
    tangents: typing.Sequence[np.ndarray],
) -> np.ndarray:
    source_tangent = tangents[0]
    maps = values[1:]
    axes = tuple(node.attrs["axes"])
    selected = dict(zip(axes, maps, strict=True))
    result = _zeros(node.spec)
    for domain_coordinate in range(node.spec.shape[0]):
        source = tuple(
            int(selected[axis][domain_coordinate]) if axis in selected else slice(None)
            for axis in range(source_tangent.ndim)
        )
        result[domain_coordinate] = source_tangent[source]
    return result


def _jvp_runtime_indexed_scatter_add(
    node: Node,
    values: typing.Sequence[np.ndarray],
    tangents: typing.Sequence[np.ndarray],
) -> np.ndarray:
    result = _zeros(node.spec)
    maps = values[1:]
    axes = tuple(node.attrs["axes"])
    selected = dict(zip(axes, maps, strict=True))
    source = tangents[0]
    for domain_coordinate in range(source.shape[0]):
        target = tuple(
            int(selected[axis][domain_coordinate]) if axis in selected else slice(None)
            for axis in range(result.ndim)
        )
        result[target] += source[domain_coordinate]
    return result


def _jvp_reduce(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    return np.sum(tangents[0], axis=node.attrs["axes"], dtype=node.spec.dtype)


def _jvp_broadcast(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    axes = node.attrs["axes"]
    order = tuple(sorted(range(len(axes)), key=lambda axis: axes[axis]))
    shape = [1] * len(node.spec.indices)
    for axis, target in enumerate(axes):
        shape[target] = tangents[0].shape[axis]
    return np.broadcast_to(tangents[0].transpose(order).reshape(shape), node.spec.shape)


def _vjp_scaled_bilinear(
    node: typing.Any,
    values: typing.Any,
    bar: typing.Any,
    active: typing.Any = (True,) * 6,
) -> typing.Any:
    a, b, c, d, e, f = values
    zero, one = _zeros(node.spec), np.ones(node.spec.shape, dtype=node.spec.dtype)
    result = []
    primal = scaled_bilinear_value(*values) if any(active[4:]) else zero
    for index, other in enumerate((b, a, d, c, primal, primal)):
        if not active[index]:
            result.append(zero)
            continue
        den1, den2 = (e, f) if index < 4 else (values[index], one)
        terms = (bar, other, zero, zero) if index < 2 else (zero, zero, bar, other)
        result.append(scaled_bilinear_value(*terms, den1, den2))
    return result


def _jvp_scaled_bilinear(
    node: typing.Any, values: typing.Any, tangents: typing.Any
) -> typing.Any:
    # Higher derivatives remain expressible. As with the ordinary add rule,
    # accumulation of six partials is not a global cancellation-safe reduction.
    result = _zeros(node.spec)
    for index, tangent in enumerate(tangents):
        active = tuple(i == index for i in range(6))
        result += _vjp_scaled_bilinear(node, values, tangent, active)[index]
    return result


_JVP_RULES = {
    "cast": _jvp_cast,
    "add": _jvp_add,
    "multiply": _jvp_multiply,
    "divide": _jvp_divide,
    "scaled_bilinear": _jvp_scaled_bilinear,
    **dict.fromkeys(("exp", "log", "sqrt", "power"), _jvp_transcendental),
    "einsum": _jvp_einsum,
    "transpose": _jvp_transpose,
    "reshape": _jvp_reshape,
    "slice": _jvp_slice,
    "gather": _jvp_gather,
    "indexed_gather": _jvp_gather,
    "scatter_add": _jvp_scatter_add,
    "segment_sum": _jvp_segment_sum,
    "runtime_indexed_select": _jvp_runtime_indexed_select,
    "runtime_indexed_scatter_add": _jvp_runtime_indexed_scatter_add,
    "reduce": _jvp_reduce,
    "broadcast": _jvp_broadcast,
}


def _jvp_node(node: Node, values: typing.Any, tangents: typing.Any) -> np.ndarray:
    """Dispatch one primitive and reject any future primitive without a rule."""
    try:
        rule = _JVP_RULES[node.op]
    except KeyError as exc:
        raise ValueError(f"no JVP rule for tensor primitive: {node.op}") from exc
    return rule(node, values, tangents)


def _vjp_add(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    return [
        _coefficient(coefficient, node.spec.dtype) * bar
        for coefficient in node.attrs["coefficients"]
    ]


def _vjp_multiply(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    return [bar * values[1], bar * values[0]]


def _vjp_divide(
    node: Node, values: typing.Any, bar: typing.Any, active: typing.Any = (True, True)
) -> list[np.ndarray]:
    x, y = values
    zero = _zeros(node.spec)
    return [
        bar / y if active[0] else zero,
        scaled_bilinear_value(zero, zero, bar, x, y, y) if active[1] else zero,
    ]


def _einsum_vjp_reference(
    node: Node, values: typing.Any, bar: typing.Any
) -> list[np.ndarray]:
    """Explicit-coordinate VJP for repeated labels within one operand.

    Repeated labels impose a diagonal constraint.  Renaming them to fresh
    labels changes the contraction, so this bounded reference enumerates the
    label space instead.  The fast path below handles the usual case where
    every operand has distinct labels.
    """
    labels = node.attrs["labels"]
    output = node.attrs["output"]
    sizes = {}
    for operand, operand_labels in zip(values, labels):
        for axis, label in enumerate(operand_labels):
            size = operand.shape[axis]
            if label in sizes and sizes[label] != size:
                raise ValueError("einsum label has inconsistent extent")
            sizes[label] = size
    order = sorted(sizes)
    contributions = [
        np.zeros(operand.shape, dtype=node.spec.dtype) for operand in values
    ]
    one = np.dtype(node.spec.dtype).type(1)
    for coordinates in product(*(range(sizes[label]) for label in order)):
        environment = dict(zip(order, coordinates))
        weight = bar[tuple(environment[label] for label in output)]
        if weight == 0:
            continue
        indices = [
            tuple(environment[label] for label in operand_labels)
            for operand_labels in labels
        ]
        factors = [values[operand][indices[operand]] for operand in range(len(values))]
        for differentiated in range(len(values)):
            product_value = one
            for operand in range(len(values)):
                if operand != differentiated:
                    product_value = product_value * factors[operand]
            contributions[differentiated][indices[differentiated]] += (
                weight * product_value
            )
    return contributions


def _vjp_einsum(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    labels = node.attrs["labels"]
    output = list(node.attrs["output"])
    coefficient = _coefficient(node.attrs["coefficient"], node.spec.dtype)
    if any(
        len(set(operand_labels)) != len(operand_labels) for operand_labels in labels
    ):
        return [
            contribution * coefficient
            for contribution in _einsum_vjp_reference(node, values, bar)
        ]
    contributions = []
    for differentiated, operand_labels in enumerate(labels):
        arguments = [bar, output]
        for operand, (value, label) in enumerate(zip(values, labels)):
            if operand != differentiated:
                arguments.extend((value, list(label)))
        # A ones operand carries the requested output labels.  It contributes
        # no numerical factor but makes summed labels legal einsum outputs.
        ones = np.ones(node.inputs[differentiated].spec.shape, dtype=node.spec.dtype)
        arguments.extend((ones, list(operand_labels)))
        contributions.append(
            np.einsum(*arguments, list(operand_labels), optimize=False) * coefficient
        )
    return contributions


def _vjp_transpose(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    inverse = tuple(np.argsort(node.attrs["axes"]))
    return [np.transpose(bar, inverse)]


def _vjp_reshape(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    return [bar.reshape(node.inputs[0].spec.shape, order="C")]


def _vjp_slice(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    result = _zeros(node.inputs[0].spec)
    result[tuple(slice(start, stop) for start, stop in node.attrs["ranges"])] = bar
    return [result]


def _vjp_gather(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    result = _zeros(node.inputs[0].spec)
    index = [slice(None)] * result.ndim
    index[node.attrs["axis"]] = np.asarray(node.attrs["positions"], dtype=np.intp)
    np.add.at(result, tuple(index), bar)
    return [result]


def _vjp_scatter_add(
    node: Node, values: typing.Sequence[np.ndarray], bar: np.ndarray
) -> list[np.ndarray]:
    return [
        np.take(
            bar,
            np.asarray(node.attrs["positions"], dtype=np.intp),
            axis=node.attrs["axis"],
        )
    ]


def _vjp_segment_sum(
    node: Node, values: typing.Sequence[np.ndarray], bar: np.ndarray
) -> list[np.ndarray]:
    offsets = node.attrs["offsets"]
    positions = np.repeat(
        np.arange(len(offsets) - 1, dtype=np.intp), np.diff(np.asarray(offsets))
    )
    return [np.take(bar, positions, axis=node.attrs["axis"])]


def _vjp_runtime_indexed_select(
    node: Node,
    values: typing.Sequence[np.ndarray],
    bar: np.ndarray,
) -> list[np.ndarray]:
    source = _zeros(node.inputs[0].spec)
    maps = values[1:]
    axes = tuple(node.attrs["axes"])
    selected = dict(zip(axes, maps, strict=True))
    for domain_coordinate in range(node.spec.shape[0]):
        target = tuple(
            int(selected[axis][domain_coordinate]) if axis in selected else slice(None)
            for axis in range(source.ndim)
        )
        source[target] += bar[domain_coordinate]
    return [source, *(_zeros(mapping.spec) for mapping in node.inputs[1:])]


def _vjp_runtime_indexed_scatter_add(
    node: Node,
    values: typing.Sequence[np.ndarray],
    bar: np.ndarray,
) -> list[np.ndarray]:
    maps = values[1:]
    axes = tuple(node.attrs["axes"])
    selected = dict(zip(axes, maps, strict=True))
    source = np.empty(node.inputs[0].spec.shape, dtype=node.inputs[0].spec.dtype)
    for domain_coordinate in range(source.shape[0]):
        target = tuple(
            int(selected[axis][domain_coordinate]) if axis in selected else slice(None)
            for axis in range(bar.ndim)
        )
        source[domain_coordinate] = bar[target]
    return [source, *(_zeros(mapping.spec) for mapping in node.inputs[1:])]


def _vjp_reduce(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    input_shape = node.inputs[0].spec.shape
    reduced = set(node.attrs["axes"])
    shape = [1 if axis in reduced else size for axis, size in enumerate(input_shape)]
    return [np.broadcast_to(bar.reshape(shape), input_shape)]


def _vjp_broadcast(node: Node, values: typing.Any, bar: typing.Any) -> list[np.ndarray]:
    axes = node.attrs["axes"]
    kept = set(axes)
    reduced = tuple(axis for axis in range(len(node.spec.indices)) if axis not in kept)
    summed = bar if not reduced else np.sum(bar, axis=reduced, dtype=node.spec.dtype)
    order = tuple(sorted(range(len(axes)), key=lambda axis: axes[axis]))
    return [summed.transpose(tuple(np.argsort(order)))]


_VJP_RULES = {
    "cast": _vjp_cast,
    "add": _vjp_add,
    "multiply": _vjp_multiply,
    "divide": _vjp_divide,
    "scaled_bilinear": _vjp_scaled_bilinear,
    **dict.fromkeys(("exp", "log", "sqrt", "power"), _vjp_transcendental),
    "einsum": _vjp_einsum,
    "transpose": _vjp_transpose,
    "reshape": _vjp_reshape,
    "slice": _vjp_slice,
    "gather": _vjp_gather,
    "indexed_gather": _vjp_gather,
    "scatter_add": _vjp_scatter_add,
    "segment_sum": _vjp_segment_sum,
    "runtime_indexed_select": _vjp_runtime_indexed_select,
    "runtime_indexed_scatter_add": _vjp_runtime_indexed_scatter_add,
    "reduce": _vjp_reduce,
    "broadcast": _vjp_broadcast,
}

# Derive the public capability set from the actual dispatch tables so a future
# primitive cannot be advertised before both derivative directions exist.
AD_RULES = {
    op: (_JVP_RULES[op], _VJP_RULES[op])
    for op in sorted(set(_JVP_RULES) & set(_VJP_RULES))
}
AD_PRIMITIVES = frozenset(AD_RULES)


def _vjp_node(
    node: Node, values: typing.Any, bar: typing.Any, active: typing.Any = None
) -> list[np.ndarray]:
    """Dispatch one primitive and reject any future primitive without a rule."""
    try:
        rule = _VJP_RULES[node.op]
    except KeyError as exc:
        raise ValueError(f"no VJP rule for tensor primitive: {node.op}") from exc
    if active is not None and node.op in ("divide", "scaled_bilinear"):
        return rule(node, values, bar, active)
    return rule(node, values, bar)


def _jvp_arrays(
    program: Program, values: Mapping[Node, np.ndarray], tangents: typing.Any
) -> dict[Node, np.ndarray]:
    """Evaluate every live SSA definition in forward tangent mode."""
    result = {}
    for node in program.live_nodes:
        if node.op == "input":
            result[node] = tangents.get(node.attrs["name"], _zeros(node.spec))
            continue
        if node.op == "constant":
            result[node] = _zeros(node.spec)
            continue
        operands = [values[operand] for operand in node.inputs]
        operand_tangents = [result[operand] for operand in node.inputs]
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                value = _jvp_node(node, operands, operand_tangents)
        except FloatingPointError as exc:
            raise ValueError(
                f"non-finite JVP arithmetic at primitive {node.op}"
            ) from exc
        value = np.asarray(value)
        if value.shape != node.spec.shape or value.dtype != np.dtype(node.spec.dtype):
            raise ValueError(
                f"JVP rule for {node.op} violates its shape/dtype contract"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite JVP value at primitive {node.op}")
        result[node] = value
    return result


def _vjp_arrays(
    program: Program,
    values: Mapping[Node, np.ndarray],
    cotangents: typing.Any,
    selected: typing.Any = None,
) -> dict[Node, np.ndarray]:
    """Propagate cotangents through one primal evaluation in reverse order."""
    live = program.live_nodes
    relevant = set(live) if selected is None else set()
    if selected is not None:
        for node in live:
            if (node.op == "input" and node.attrs["name"] in selected) or any(
                operand in relevant for operand in node.inputs
            ):
                relevant.add(node)
    bars = {node: _zeros(node.spec) for node in live}
    for name, cotangent in cotangents.items():
        bars[program.outputs[name]] += cotangent
    for node in reversed(live):
        if node.op in ("input", "constant") or node not in relevant:
            continue
        bar = bars[node]
        operands = [values[operand] for operand in node.inputs]
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                contributions = _vjp_node(
                    node, operands, bar, tuple(n in relevant for n in node.inputs)
                )
        except FloatingPointError as exc:
            raise ValueError(
                f"non-finite VJP arithmetic at primitive {node.op}"
            ) from exc
        if len(contributions) != len(node.inputs):
            raise ValueError(f"VJP rule for {node.op} returned the wrong arity")
        for operand, contribution in zip(node.inputs, contributions):
            if operand not in relevant:
                continue
            contribution_array = np.asarray(contribution)
            if (
                contribution_array.shape != operand.spec.shape
                or contribution_array.dtype != np.dtype(operand.spec.dtype)
            ):
                raise ValueError(
                    f"VJP rule for {node.op} violates its operand contract"
                )
            if not np.isfinite(contribution_array).all():
                raise ValueError(f"non-finite VJP contribution at primitive {node.op}")
            bars[operand] += contribution_array
    return bars


def jvp(
    program: Program,
    feeds: Mapping,
    tangents: Mapping,
    *,
    outputs: typing.Any = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> JVPResult:
    """Evaluate one forward tangent direction through a TensorIR program.

    Unspecified differentiable inputs receive zero tangents, so a caller can
    request a partial direction.  The result carries the primal logical hash
    and a derivative identity hash derived from it.  ``outputs`` selects which
    output tangents are returned; computing only that subset is slice B.
    """
    if not isinstance(program, Program):
        raise TypeError("jvp requires a Program")
    checked_size(max_bytes, "autodiff byte budget")
    tangents = _validate_tangents(program, tangents)
    selected = _select_names(program.outputs, outputs, "output")
    values = evaluate_nodes(program, feeds, max_bytes=max_bytes)
    arrays = _jvp_arrays(program, values, tangents)
    _check_budget(program, arrays, max_bytes)
    output_tangents = {
        name: np.array(arrays[node], dtype=node.spec.dtype, copy=True)
        for name, node in selected.items()
    }
    return JVPResult(
        output_tangents,
        program.logical_hash,
        tuple(sorted(tangents)),
        tuple(sorted(selected)),
    )


def vjp(
    program: Program,
    feeds: Mapping,
    cotangents: Mapping,
    *,
    inputs: typing.Any = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> VJPResult:
    """Propagate one cotangent without materializing a dense Jacobian.

    Cotangents may be supplied for a subset of outputs; omitted outputs are
    treated as zero.  ``inputs`` selects which differentiable input cotangents
    are returned. Unrequested paths and quotient partials are pruned before
    arithmetic so an unrequested overflowing quotient adjoint cannot fail a
    finite requested one.
    """
    if not isinstance(program, Program):
        raise TypeError("vjp requires a Program")
    checked_size(max_bytes, "autodiff byte budget")
    cotangents = _validate_cotangents(program, cotangents)
    differentiable = {
        name: node
        for name, node in _input_nodes(program).items()
        if node.spec.differentiable
    }
    selected = _select_names(differentiable, inputs, "input")
    _require_general_inputs(program, selected)
    values = evaluate_nodes(program, feeds, max_bytes=max_bytes)
    bars = _vjp_arrays(program, values, cotangents, selected)
    _check_budget(program, bars, max_bytes)
    groups = _input_groups(program)
    input_cotangents = {
        name: _input_cotangent(groups[name], bars) for name in sorted(selected)
    }
    return VJPResult(
        input_cotangents,
        program.logical_hash,
        tuple(sorted(cotangents)),
        tuple(sorted(selected)),
    )


def _inner(left: typing.Any, right: typing.Any) -> float:
    """Real Euclidean inner product accumulated in float64."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("inner-product operands must have identical shapes")
    return float(np.dot(left, right))


def _default_atol(
    program: Program, tangent_names: typing.Any, cotangent_names: typing.Any
) -> float:
    """Use a scale-aware absolute guard for the requested dense spaces."""
    inputs = _input_nodes(program)
    dtypes = {inputs[name].spec.dtype for name in tangent_names if name in inputs}
    dtypes.update(program.outputs[name].spec.dtype for name in cotangent_names)
    return 1e-12 if dtypes <= {"float64"} else 1e-6


def dot_test(
    program: Program,
    feeds: Mapping,
    tangents: Mapping,
    cotangents: Mapping,
    *,
    rtol: float = 1e-10,
    atol: float | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> DotTestResult:
    """Check ``<w, Jv> == <J^T w, v>`` on one fixed primal evaluation."""
    if not isinstance(program, Program):
        raise TypeError("dot_test requires a Program")
    if not math.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative")
    checked_size(max_bytes, "autodiff byte budget")
    tangents = _validate_tangents(program, tangents)
    cotangents = _validate_cotangents(program, cotangents)
    if atol is None:
        atol = _default_atol(program, tangents, cotangents)
    if not math.isfinite(atol) or atol < 0 or atol + rtol == 0:
        raise ValueError("atol/rtol must be finite, nonnegative, and not both zero")
    values = evaluate_nodes(program, feeds, max_bytes=max_bytes)
    forward = _jvp_arrays(program, values, tangents)
    reverse = _vjp_arrays(program, values, cotangents)
    _check_budget(program, forward, max_bytes)
    _check_budget(program, reverse, max_bytes)
    lhs = sum(
        _inner(cotangent, forward[program.outputs[name]])
        for name, cotangent in cotangents.items()
    )
    inputs = _input_groups(program)
    rhs = sum(
        _inner(reverse[node], tangent)
        for name, tangent in tangents.items()
        for node in inputs[name]
    )
    absolute_error = abs(lhs - rhs)
    scale = max(abs(lhs), abs(rhs))
    relative_error = absolute_error / scale if scale else 0.0
    if scale <= atol:
        passed = absolute_error <= atol
    else:
        passed = relative_error <= rtol
    return DotTestResult(
        lhs=lhs,
        rhs=rhs,
        absolute_error=absolute_error,
        relative_error=relative_error,
        passed=bool(passed),
        atol=atol,
        rtol=rtol,
    )

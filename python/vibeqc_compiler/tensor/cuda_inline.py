"""Bounded output-demand lowering for TensorIR values fused into CUDA consumers.

This is deliberately not a second TensorIR executor. It specializes one requested
scalar output through the shared optimizer, then emits only the tiny expression
subset qualified for embedding in an enclosing generated CUDA consumer.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod

from .optimize import optimize
from .program import Program


@dataclass(frozen=True)
class InlineCudaOutput:
    """One compiler-specialized scalar value for an enclosing CUDA consumer."""

    expression: str
    original_logical_hash: str
    specialization_logical_hash: str
    optimized_logical_hash: str
    optimizer_identity: str
    required_inputs: tuple[str, ...]
    pruning_diagnostics: typing.Mapping[str, typing.Any]


def _fraction(value: typing.Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value, 1)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(type(item) is int for item in value)
    ):
        return Fraction(*value)
    raise TypeError("inline CUDA lowering requires exact rational constants")


def exact_cuda_literal(value: typing.Any) -> str:
    """Spell one exact rational using the strict FP64 CUDA source convention."""

    value = _fraction(value)
    if value.denominator == 1:
        return f"{value.numerator}.0"
    return f"({value.numerator}.0/{value.denominator}.0)"


def _coordinates(linear: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        (linear // max(1, prod(shape[axis + 1 :]))) % max(1, extent)
        for axis, extent in enumerate(shape)
    )


def _flat(coordinates: typing.Iterable[int], shape: tuple[int, ...]) -> int:
    return sum(
        coordinate * prod(shape[axis + 1 :])
        for axis, coordinate in enumerate(coordinates)
    )


def _binding_values(value: typing.Any, size: int) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (tuple, list)) and all(
        isinstance(item, str) for item in value
    ):
        values = tuple(value)
    else:
        raise TypeError(
            "inline CUDA input bindings must be strings or string sequences"
        )
    if len(values) != size:
        raise ValueError("inline CUDA input binding size does not match TensorIR input")
    # Bindings are expressions, not necessarily identifiers or array accesses.
    # Group each leaf before inserting it into a surrounding sum/product.
    return tuple(f"({expression})" for expression in values)


def _reduce(
    values: tuple[str, ...], source_shape: tuple[int, ...], axes: tuple[int, ...]
) -> tuple[str, ...]:
    output_shape = tuple(
        extent for axis, extent in enumerate(source_shape) if axis not in axes
    )
    reduction_shape = tuple(source_shape[axis] for axis in axes)
    if any(extent == 0 for extent in reduction_shape):
        raise ValueError("inline CUDA lowering does not materialize empty reductions")
    output = []
    for linear in range(prod(output_shape)):
        remaining = iter(_coordinates(linear, output_shape))
        fixed = {
            axis: next(remaining)
            for axis in range(len(source_shape))
            if axis not in axes
        }
        terms = []
        for reduced in product(*(range(extent) for extent in reduction_shape)):
            reduction = dict(zip(axes, reduced, strict=True))
            coordinates = tuple(
                reduction[axis] if axis in reduction else fixed[axis]
                for axis in range(len(source_shape))
            )
            terms.append(values[_flat(coordinates, source_shape)])
        output.append("(" + " + ".join(terms) + ")")
    return tuple(output)


def lower_inline_cuda_output(
    program: Program,
    *,
    output: str,
    bindings: typing.Mapping[str, str | tuple[str, ...] | list[str]],
) -> InlineCudaOutput:
    """Specialize one scalar TensorIR output and fuse it into a CUDA expression.

    Output demand is resolved before bindings are inspected. An unrequested
    diagnostic branch may therefore contain inputs or operations unsupported by
    this bounded consumer and disappear safely before consumer lowering.
    """

    if not isinstance(program, Program):
        raise TypeError("inline CUDA lowering requires a TensorIR Program")
    if not isinstance(output, str):
        raise TypeError("inline CUDA requested output must be a string")
    specialized = optimize(program, requested_outputs=(output,))
    root = specialized.outputs[output]
    if root.spec.dtype != "float64" or root.spec.size != 1:
        raise ValueError("inline CUDA lowering requires one FP64 scalar output")

    values: dict[typing.Any, tuple[str, ...]] = {}
    required_inputs: list[str] = []
    for node in specialized.live_nodes:
        if node.spec.dtype != "float64":
            raise ValueError("inline CUDA lowering currently supports FP64 values only")
        if node.op == "input":
            name = node.attrs["name"]
            if name not in bindings:
                raise ValueError(f"missing inline CUDA input binding: {name}")
            values[node] = _binding_values(bindings[name], node.spec.size)
            required_inputs.append(name)
        elif node.op == "constant":
            raw = node.attrs["values"]
            if len(raw) != node.spec.size:
                raise ValueError(
                    "inline CUDA constant size does not match TensorIR spec"
                )
            values[node] = tuple(exact_cuda_literal(item) for item in raw)
        elif node.op == "reduce":
            values[node] = _reduce(
                values[node.inputs[0]],
                node.inputs[0].spec.shape,
                node.attrs["axes"],
            )
        elif node.op == "einsum":
            operands = [values[item] for item in node.inputs]
            domains: dict[int, int] = {}
            for child, labels in zip(node.inputs, node.attrs["labels"], strict=True):
                for label, extent in zip(labels, child.spec.shape, strict=True):
                    previous = domains.setdefault(label, extent)
                    if previous != extent:
                        raise ValueError("inconsistent inline CUDA einsum label extent")
            reduced = tuple(
                label for label in sorted(domains) if label not in node.attrs["output"]
            )
            # Inline only finite, tiny contractions (not a general tensor
            # executor). In particular the same-spin exchange weight reduces
            # one or two spin terms while retaining a singleton quartet axis.
            if prod(domains.values()) > 64 or any(
                extent < 1 for extent in domains.values()
            ):
                raise ValueError("inline CUDA einsum exceeds the 64-term work bound")
            coefficient = _fraction(node.attrs["coefficient"])
            outputs = []
            for linear in range(node.spec.size):
                fixed = dict(
                    zip(
                        node.attrs["output"],
                        _coordinates(linear, node.spec.shape),
                        strict=True,
                    )
                )
                terms = []
                for contracted in product(
                    *(range(domains[label]) for label in reduced)
                ):
                    coordinates = fixed | dict(zip(reduced, contracted, strict=True))
                    factors = [
                        operand[
                            _flat(
                                (coordinates[label] for label in labels),
                                child.spec.shape,
                            )
                        ]
                        for operand, child, labels in zip(
                            operands, node.inputs, node.attrs["labels"], strict=True
                        )
                    ]
                    if coefficient != 1:
                        factors.insert(0, exact_cuda_literal(coefficient))
                    terms.append("(" + " * ".join(factors) + ")")
                outputs.append(
                    terms[0] if len(terms) == 1 else "(" + " + ".join(terms) + ")"
                )
            values[node] = tuple(outputs)
        else:
            raise ValueError(f"unsupported TensorIR inline CUDA op: {node.op}")

    expression = values[root]
    if len(expression) != 1:
        raise ValueError("inline CUDA output did not lower to one scalar")
    provenance = specialized.provenance
    return InlineCudaOutput(
        expression=expression[0],
        original_logical_hash=program.logical_hash,
        specialization_logical_hash=provenance["specialized_logical_hash"],
        optimized_logical_hash=specialized.logical_hash,
        optimizer_identity=provenance["optimizer_identity"],
        required_inputs=tuple(required_inputs),
        pruning_diagnostics=provenance["pruning_diagnostics"],
    )

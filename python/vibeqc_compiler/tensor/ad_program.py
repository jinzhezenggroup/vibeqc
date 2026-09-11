"""Demand-driven TensorIR derivative programs for #151 slice B.

The forward and reverse rules in :mod:`vibeqc_compiler.tensor.autodiff` are
reference interpreters.  This module turns the same rules into immutable,
backend-independent :class:`~vibeqc_compiler.tensor.program.Program` DAGs that
can be executed by the CPU interpreter or lowered by the existing
``cuda_plan``/``cuda_execute`` path.

Generation is demand driven: only requested output tangents/cotangents and
only primal nodes on a requested dependency path produce new nodes.  A
requested output with no active tangent becomes an explicit zero-like node,
so provenance never depends on Python control-flow state.

Slice B generates forward and reverse programs for every primitive.
Gather/slice use exact incidence matrices and repeated einsum labels use exact
identity projections.  Packed parameters are expanded through an explicit
unpack DAG, and their reverse programs apply the weighted transpose; callers
request this with the ``packed=`` mapping.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from types import MappingProxyType

import numpy as np

from .autodiff import _input_groups, _input_nodes
from .ir import (
    Node,
    add,
    broadcast,
    constant,
    divide,
    einsum,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    reshape,
    slice_tensor,
    transpose,
)
from .packing import PackedLayout
from .program import Program
from .types import Index, IndexSpace, TensorSpec

GENERATION_SCHEMA = "vibeqc.tensor.ad_program"
GENERATION_VERSION = 1
TANGENT_PREFIX = "d_"
COTANGENT_PREFIX = "bar_"
DEFAULT_MAX_ELEMENTS = 1_000_000
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _select_names(mapping: Mapping, names, label: str) -> dict:
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


def _coefficient(pair) -> Fraction:
    return Fraction(*pair)


def _scale(node: Node, pair) -> Node:
    """Exact scalar multiple through a one-operand add node."""
    return add(node, coefficients=(_coefficient(pair),))


def _combine(terms) -> Node | None:
    """Exact ordered sum of non-None terms; None means identically zero."""
    terms = [term for term in terms if term is not None]
    if not terms:
        return None
    if len(terms) == 1:
        return terms[0]
    return add(*terms, coefficients=(1,) * len(terms))


def _zero_like(node: Node) -> Node:
    """A zero tensor with the same logical type, retaining its primal node."""
    return add(node, node, coefficients=(0, 0))


def _square(node: Node) -> Node:
    return multiply(node, node)


def _equation(labels, output) -> str:
    """Reconstruct an alphabetic equation from canonical integer labels."""
    unique = []
    for operand_labels in labels:
        for label in operand_labels:
            if label not in unique:
                unique.append(label)
    for label in output:
        if label not in unique:
            unique.append(label)
    if len(unique) > len(ALPHABET):
        raise ValueError("einsum equation exceeds the single-character alphabet")
    mapping = {label: ALPHABET[index] for index, label in enumerate(unique)}
    left = ",".join(
        "".join(mapping[label] for label in operand_labels) for operand_labels in labels
    )
    right = "".join(mapping[label] for label in output)
    return f"{left}->{right}"


def _derivative_input(node: Node, name: str) -> Node:
    """Create a read-only tangent/cotangent input with the primal logical type."""
    spec = replace(node.spec, role="input", differentiable=False)
    return input_tensor(name, spec)


def _constant_ones_for_axes(operand: Node, axes, *, max_elements: int) -> Node | None:
    """Create a minimal all-ones operand carrying only missing output labels."""
    axes = tuple(axes)
    if not axes:
        return None
    spec = TensorSpec(
        tuple(operand.spec.indices[axis] for axis in axes),
        dtype=operand.spec.dtype,
        representation=operand.spec.representation,
        role="constant",
    )
    if spec.size > max_elements:
        raise ValueError(
            "demand-driven VJP einsum needs an all-ones operand above the "
            "configured element budget; use the CPU interpreter reference"
        )
    return constant((1,) * spec.size, spec)


def _identity_constant(
    axis: Index, dtype: str, representation: str, *, max_elements: int
) -> Node:
    """Exact identity matrix on one repeated-label orbit."""
    size = axis.extent
    if size * size > max_elements:
        raise ValueError(
            "demand-driven VJP diagonal projection exceeds the configured "
            "element budget; use the CPU interpreter reference"
        )
    row = Index("identity_row", axis.space, axis.start, axis.stop, axis.selection)
    column = Index("identity_col", axis.space, axis.start, axis.stop, axis.selection)
    spec = TensorSpec(
        (row, column),
        dtype=dtype,
        representation=representation,
        role="constant",
    )
    values = tuple(1 if i == j else 0 for i in range(size) for j in range(size))
    return constant(values, spec)


def _incidence_constant(
    axis: Index,
    bar_axis: Index,
    positions,
    dtype: str,
    representation: str,
    *,
    max_elements: int,
) -> Node:
    """Exact gather/scatter incidence matrix for one logical axis."""
    rows, columns = axis.extent, bar_axis.extent
    if columns != len(positions):
        raise ValueError("incidence columns must match the gathered/sliced extent")
    if rows * columns > max_elements:
        raise ValueError(
            "demand-driven VJP incidence matrix exceeds the configured "
            "element budget; use the CPU interpreter reference"
        )
    row = Index("incidence_row", axis.space, axis.start, axis.stop, axis.selection)
    column = Index(
        "incidence_col",
        bar_axis.space,
        bar_axis.start,
        bar_axis.stop,
        bar_axis.selection,
    )
    spec = TensorSpec(
        (row, column),
        dtype=dtype,
        representation=representation,
        role="constant",
    )
    values = tuple(
        1 if row_index == positions[column_index] else 0
        for row_index in range(rows)
        for column_index in range(columns)
    )
    return constant(values, spec)


def _jvp_graph(node: Node, operand_tangents) -> Node | None:
    """Generate one forward tangent expression, or None for exact zero."""
    if node.op == "add":
        return _combine(
            _scale(tangent, coefficient)
            for tangent, coefficient in zip(
                operand_tangents, node.attrs["coefficients"]
            )
            if tangent is not None
        )
    if node.op == "multiply":
        left, right = node.inputs
        return _combine(
            [
                None
                if operand_tangents[0] is None
                else multiply(operand_tangents[0], right),
                None
                if operand_tangents[1] is None
                else multiply(left, operand_tangents[1]),
            ]
        )
    if node.op == "divide":
        numerator, denominator = node.inputs
        terms = []
        if operand_tangents[0] is not None:
            terms.append(multiply(operand_tangents[0], denominator))
        if operand_tangents[1] is not None:
            terms.append(_scale(multiply(numerator, operand_tangents[1]), (-1, 1)))
        combined = _combine(terms)
        return None if combined is None else divide(combined, _square(denominator))
    if node.op == "einsum":
        equation = _equation(node.attrs["labels"], node.attrs["output"])
        terms = []
        for differentiated, tangent in enumerate(operand_tangents):
            if tangent is None:
                continue
            operands = [
                tangent if index == differentiated else operand
                for index, operand in enumerate(node.inputs)
            ]
            terms.append(
                einsum(
                    equation,
                    *operands,
                    coefficient=_coefficient(node.attrs["coefficient"]),
                )
            )
        return _combine(terms)
    tangent = operand_tangents[0]
    if tangent is None:
        return None
    if node.op == "transpose":
        return transpose(tangent, node.attrs["axes"])
    if node.op == "reshape":
        return reshape(tangent, node.spec.indices)
    if node.op == "slice":
        return slice_tensor(tangent, node.attrs["ranges"])
    if node.op == "gather":
        return gather(tangent, node.attrs["axis"], node.attrs["positions"])
    if node.op == "reduce":
        return reduce_sum(tangent, node.attrs["axes"])
    if node.op == "broadcast":
        return broadcast(tangent, node.spec.indices, node.attrs["axes"])
    raise ValueError(f"no demand-driven JVP rule for primitive: {node.op}")


def _vjp_einsum(
    node: Node, bar: Node, active: tuple[bool, ...], *, max_elements: int
) -> list[Node | None]:
    """Build only requested operand adjoints, including diagonal embeddings."""
    labels = node.attrs["labels"]
    output = node.attrs["output"]
    coefficient = _coefficient(node.attrs["coefficient"])
    all_labels = [
        label for operand_labels in labels for label in operand_labels
    ] + list(output)
    next_label = max(all_labels, default=-1) + 1
    contributions = []
    for differentiated, operand_labels in enumerate(labels):
        if not active[differentiated]:
            contributions.append(None)
            continue
        operand = node.inputs[differentiated]
        occurrences = {}
        for axis, label in enumerate(operand_labels):
            occurrences.setdefault(label, []).append(axis)
        fresh = [0] * len(operand_labels)
        first = {}
        for label, axes in occurrences.items():
            if len(axes) == 1:
                fresh[axes[0]] = label
                first[label] = label
            else:
                for axis in axes:
                    fresh[axis] = next_label
                    next_label += 1
                first[label] = fresh[axes[0]]
        adjusted = {}
        for index, other_labels in enumerate(labels):
            if index != differentiated:
                adjusted[index] = tuple(
                    first.get(label, label) for label in other_labels
                )
        identity_nodes, identity_labels = [], []
        for axes in occurrences.values():
            for left_axis, right_axis in pairwise(axes):
                identity_nodes.append(
                    _identity_constant(
                        operand.spec.indices[left_axis],
                        operand.spec.dtype,
                        operand.spec.representation,
                        max_elements=max_elements,
                    )
                )
                identity_labels.append((fresh[left_axis], fresh[right_axis]))
        # A diagonal label can survive in the primal output. Its cotangent
        # axis must follow the same renaming as the other operand axes.
        bar_labels = tuple(first.get(label, label) for label in output)
        present = set(bar_labels)
        for other_labels in adjusted.values():
            present.update(other_labels)
        for pair in identity_labels:
            present.update(pair)
        missing = [label for label in fresh if label not in present]
        missing_axes = [axis for axis, label in enumerate(fresh) if label in missing]
        equation_labels = [bar_labels]
        operands = [bar]
        for index in range(len(labels)):
            if index != differentiated:
                equation_labels.append(adjusted[index])
                operands.append(node.inputs[index])
        for pair, identity in zip(identity_labels, identity_nodes):
            equation_labels.append(pair)
            operands.append(identity)
        ones = _constant_ones_for_axes(operand, missing_axes, max_elements=max_elements)
        if ones is not None:
            equation_labels.append(tuple(missing))
            operands.append(ones)
        contributions.append(
            einsum(
                _equation(equation_labels, tuple(fresh)),
                *operands,
                coefficient=coefficient,
            )
        )
    return contributions


def _embed_axis(
    bar: Node,
    axis: int,
    input_axis: Index,
    positions,
    dtype: str,
    *,
    max_elements: int,
) -> Node:
    """Adjoint of one-axis gather/slice: scatter-add through incidence."""
    incidence = _incidence_constant(
        input_axis,
        bar.spec.indices[axis],
        positions,
        dtype,
        bar.spec.representation,
        max_elements=max_elements,
    )
    bar_labels = list(range(len(bar.spec.indices)))
    gathered_label = bar_labels[axis]
    row_label = len(bar_labels)
    output_labels = list(bar_labels)
    output_labels[axis] = row_label
    return einsum(
        _equation(
            [tuple(bar_labels), (row_label, gathered_label)],
            tuple(output_labels),
        ),
        bar,
        incidence,
    )


def _slice_vjp_node(node: Node, bar: Node, *, max_elements: int) -> Node:
    """Adjoint of a unit-step contiguous slice, one axis at a time."""
    result = bar
    for axis, (start, stop) in enumerate(node.attrs["ranges"]):
        result = _embed_axis(
            result,
            axis,
            node.inputs[0].spec.indices[axis],
            tuple(range(start, stop)),
            node.inputs[0].spec.dtype,
            max_elements=max_elements,
        )
    return result


def _vjp_graph(
    node: Node, bar: Node, active: tuple[bool, ...], *, max_elements: int
) -> list[Node | None]:
    """Build active operand adjoints; prune before allocating any constants.

    Filtering afterward can exceed the element budget for an unrequested
    diagonal/scatter adjoint even when the requested derivative is scalar.
    """
    if not any(active):
        return [None] * len(node.inputs)
    if node.op == "add":
        return [
            _scale(bar, coefficient) if needed else None
            for coefficient, needed in zip(node.attrs["coefficients"], active)
        ]
    if node.op == "multiply":
        return [
            multiply(bar, node.inputs[1]) if active[0] else None,
            multiply(bar, node.inputs[0]) if active[1] else None,
        ]
    if node.op == "divide":
        numerator, denominator = node.inputs
        return [
            divide(bar, denominator) if active[0] else None,
            _scale(divide(multiply(bar, numerator), _square(denominator)), (-1, 1))
            if active[1]
            else None,
        ]
    if node.op == "einsum":
        return _vjp_einsum(node, bar, active, max_elements=max_elements)
    if node.op == "transpose":
        inverse = tuple(int(axis) for axis in np.argsort(node.attrs["axes"]))
        return [transpose(bar, inverse)]
    if node.op == "reshape":
        return [reshape(bar, node.inputs[0].spec.indices)]
    if node.op == "reduce":
        reduced = set(node.attrs["axes"])
        axes = [
            index
            for index, _ in enumerate(node.inputs[0].spec.indices)
            if index not in reduced
        ]
        return [broadcast(bar, node.inputs[0].spec.indices, tuple(axes))]
    if node.op == "broadcast":
        axes = node.attrs["axes"]
        kept = set(axes)
        reduced = tuple(
            axis for axis in range(len(node.spec.indices)) if axis not in kept
        )
        summed = bar if not reduced else reduce_sum(bar, reduced)
        order = tuple(sorted(range(len(axes)), key=lambda axis: axes[axis]))
        inverse = tuple(int(axis) for axis in np.argsort(order))
        return [transpose(summed, inverse)]
    if node.op == "slice":
        return [_slice_vjp_node(node, bar, max_elements=max_elements)]
    if node.op == "gather":
        axis = node.attrs["axis"]
        return [
            _embed_axis(
                bar,
                axis,
                node.inputs[0].spec.indices[axis],
                node.attrs["positions"],
                node.inputs[0].spec.dtype,
                max_elements=max_elements,
            )
        ]
    raise ValueError(f"no demand-driven VJP rule for primitive: {node.op}")


@dataclass(frozen=True)
class JVPProgram:
    """Generated forward derivative DAG plus replay/provenance metadata."""

    program: Program
    primal_logical_hash: str
    tangent_inputs: tuple[str, ...]
    output_names: tuple[str, ...]
    input_map: Mapping[str, str]
    output_map: Mapping[str, str]
    generation_version: int = GENERATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "tangent_inputs", tuple(self.tangent_inputs))
        object.__setattr__(self, "output_names", tuple(self.output_names))
        object.__setattr__(self, "input_map", MappingProxyType(dict(self.input_map)))
        object.__setattr__(self, "output_map", MappingProxyType(dict(self.output_map)))

    @property
    def derivative_hash(self) -> str:
        """The generated equation identity is the derivative identity."""
        return self.program.logical_hash

    def provenance(self) -> dict:
        return {
            "schema": GENERATION_SCHEMA,
            "schema_version": GENERATION_VERSION,
            "mode": "jvp",
            "primal_logical_hash": self.primal_logical_hash,
            "derivative_logical_hash": self.derivative_hash,
            "tangent_inputs": list(self.tangent_inputs),
            "output_names": list(self.output_names),
            "input_map": dict(self.input_map),
            "output_map": dict(self.output_map),
        }


@dataclass(frozen=True)
class VJPProgram:
    """Generated reverse derivative DAG plus replay/provenance metadata."""

    program: Program
    primal_logical_hash: str
    cotangent_outputs: tuple[str, ...]
    input_names: tuple[str, ...]
    input_map: Mapping[str, str]
    output_map: Mapping[str, str]
    generation_version: int = GENERATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "cotangent_outputs", tuple(self.cotangent_outputs))
        object.__setattr__(self, "input_names", tuple(self.input_names))
        object.__setattr__(self, "input_map", MappingProxyType(dict(self.input_map)))
        object.__setattr__(self, "output_map", MappingProxyType(dict(self.output_map)))

    @property
    def derivative_hash(self) -> str:
        return self.program.logical_hash

    def provenance(self) -> dict:
        return {
            "schema": GENERATION_SCHEMA,
            "schema_version": GENERATION_VERSION,
            "mode": "vjp",
            "primal_logical_hash": self.primal_logical_hash,
            "derivative_logical_hash": self.derivative_hash,
            "cotangent_outputs": list(self.cotangent_outputs),
            "input_names": list(self.input_names),
            "input_map": dict(self.input_map),
            "output_map": dict(self.output_map),
        }


def _ancestors(roots) -> set[Node]:
    needed, pending = set(), list(roots)
    while pending:
        node = pending.pop()
        if node not in needed:
            needed.add(node)
            pending.extend(node.inputs)
    return needed


def _descendants_of(program: Program, roots) -> set[Node]:
    """Nodes that can reach one of ``roots`` through primal edges."""
    roots = set(roots)
    users = {}
    for node in program.nodes:
        for operand in node.inputs:
            users.setdefault(operand, set()).add(node)
    reachable, pending = set(), list(roots)
    while pending:
        node = pending.pop()
        if node not in reachable:
            reachable.add(node)
            pending.extend(users.get(node, ()))
    return reachable


def _rebuild_node(node: Node, inputs) -> Node:
    """Recreate one primal primitive through its public constructor."""
    if node.op == "add":
        return add(
            *inputs,
            coefficients=(
                _coefficient(coefficient) for coefficient in node.attrs["coefficients"]
            ),
        )
    if node.op == "multiply":
        return multiply(*inputs)
    if node.op == "divide":
        return divide(*inputs)
    if node.op == "einsum":
        return einsum(
            _equation(node.attrs["labels"], node.attrs["output"]),
            *inputs,
            coefficient=_coefficient(node.attrs["coefficient"]),
        )
    if node.op == "transpose":
        return transpose(inputs[0], node.attrs["axes"])
    if node.op == "reshape":
        return reshape(inputs[0], node.spec.indices)
    if node.op == "slice":
        return slice_tensor(inputs[0], node.attrs["ranges"])
    if node.op == "gather":
        return gather(inputs[0], node.attrs["axis"], node.attrs["positions"])
    if node.op == "reduce":
        return reduce_sum(inputs[0], node.attrs["axes"])
    if node.op == "broadcast":
        return broadcast(inputs[0], node.spec.indices, node.attrs["axes"])
    raise ValueError(f"cannot rebuild primitive: {node.op}")


def _rebuild(
    program: Program,
    replacements: Mapping[Node, Node],
    *,
    extra_definitions=(),
    provenance=None,
) -> Program:
    """Rebuild a primal DAG with selected input definitions substituted."""
    mapping = {}
    for node in program.nodes:
        if node in replacements:
            mapping[node] = replacements[node]
        elif node.op in ("input", "constant"):
            mapping[node] = node
        else:
            mapping[node] = _rebuild_node(
                node, tuple(mapping[operand] for operand in node.inputs)
            )
    definitions = tuple(dict.fromkeys([*mapping.values(), *extra_definitions]))
    return Program(
        {name: mapping[node] for name, node in program.outputs.items()},
        definitions,
        program.provenance if provenance is None else provenance,
    )


def _layout_hash(layout: PackedLayout) -> str:
    encoded = json.dumps(layout.to_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _unpack_dag(original: Node, layout: PackedLayout):
    """Create packed input -> dense unpack DAG and its definitions."""
    dense_size = original.spec.size
    packed_space = IndexSpace(f"packed_{original.attrs['name']}", "batch", layout.size)
    packed_spec = TensorSpec(
        (Index("packed", packed_space),),
        dtype=original.spec.dtype,
        role=original.spec.role,
        representation=original.spec.representation,
        differentiable=True,
    )
    packed_input = input_tensor(original.attrs["name"], packed_spec)
    constant_spec = replace(
        original.spec,
        role="constant",
        symmetries=(),
        differentiable=False,
    )
    if layout.size == 0:
        dense = constant((0,) * dense_size, constant_spec)
        return packed_input, dense, (packed_input, dense)
    positions = tuple(
        layout.dense_to_packed[coordinate] if layout.signs[coordinate] else 0
        for coordinate in range(dense_size)
    )
    gathered = gather(packed_input, 0, positions)
    reshaped = reshape(gathered, original.spec.indices)
    signs = constant(tuple(layout.signs), constant_spec)
    dense = multiply(reshaped, signs)
    return packed_input, dense, (packed_input, gathered, reshaped, signs, dense)


def _expand_packed_inputs(program: Program, packed: Mapping) -> tuple[Program, dict]:
    """Replace packed parameters with explicit dense unpack DAGs."""
    if not packed:
        return program, {}
    inputs = _input_nodes(program)
    replacements, extra_definitions, layouts = {}, [], {}
    for name, layout in packed.items():
        if not isinstance(name, str) or name not in inputs:
            raise ValueError(f"unknown packed input: {name!r}")
        if not isinstance(layout, PackedLayout):
            raise TypeError("packed layouts must be PackedLayout objects")
        original = inputs[name]
        if not original.spec.differentiable:
            raise ValueError(f"packed input {name} is not differentiable")
        if layout.spec != original.spec:
            raise ValueError(
                f"packed layout for {name} does not match the primal input spec"
            )
        _packed_input, dense, nodes = _unpack_dag(original, layout)
        # All SSA occurrences read the same packed feed. Replace retained
        # dead definitions too, or Program would see dense and packed specs
        # for the same public name after rebuilding.
        replacements.update(
            (node, dense)
            for node in program.nodes
            if node.op == "input" and node.attrs["name"] == name
        )
        extra_definitions.extend(nodes)
        layouts[name] = layout
    provenance = {
        **program.provenance,
        "packed_inputs": {
            name: _layout_hash(layout) for name, layout in sorted(layouts.items())
        },
    }
    return (
        _rebuild(
            program,
            replacements,
            extra_definitions=extra_definitions,
            provenance=provenance,
        ),
        layouts,
    )


def _inverse_weights_constant(node: Node, layout: PackedLayout) -> Node:
    """Exact diagonal inverse packed metric for the weighted adjoint."""
    values = tuple(Fraction(1, weight) for weight in layout.weights)
    return constant(
        values,
        TensorSpec(
            node.spec.indices,
            dtype=node.spec.dtype,
            representation=node.spec.representation,
            role="constant",
        ),
    )


def linearize(
    program: Program,
    tangent_inputs,
    *,
    outputs=None,
    packed=None,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> JVPProgram:
    """Generate a demand-driven forward derivative :class:`Program`.

    The generated program takes the original inputs plus ``d_<name>`` tangent
    inputs and returns ``d_<output>`` tangents.  Only requested tangent inputs
    and output tangents are generated.
    """
    if not isinstance(program, Program):
        raise TypeError("linearize requires a Program")
    if type(max_elements) is not int or max_elements < 0:
        raise ValueError("max_elements must be a nonnegative integer")
    primal_hash = program.logical_hash
    program, _layouts = (
        _expand_packed_inputs(program, packed) if packed else (program, {})
    )
    inputs = _input_nodes(program)
    requested = _select_names(
        {name: node for name, node in inputs.items() if node.spec.differentiable},
        tangent_inputs,
        "tangent input",
    )
    selected_outputs = _select_names(program.outputs, outputs, "output")
    tangent_nodes = {}
    for name, node in requested.items():
        if node.spec.symmetries:
            raise NotImplementedError(
                "packed/symmetric tangent inputs need a boundary unpack map; "
                "slice B currently generates dense general tangents only"
            )
        generated = f"{TANGENT_PREFIX}{name}"
        if generated in inputs or generated in program.outputs:
            raise ValueError(
                f"generated tangent name collides with input/output: {generated}"
            )
        tangent_nodes[name] = _derivative_input(node, generated)
    needed = _ancestors(selected_outputs.values())
    tangents: dict[Node, Node | None] = {}
    generated_nodes = []
    for node in program.nodes:
        if node not in needed:
            continue
        if node.op == "input":
            tangent = tangent_nodes.get(node.attrs["name"])
        elif node.op == "constant":
            tangent = None
        else:
            tangent = _jvp_graph(node, [tangents[operand] for operand in node.inputs])
        tangents[node] = tangent
        if tangent is not None:
            generated_nodes.append(tangent)
    derivative_outputs = {}
    output_map = {}
    for name, node in selected_outputs.items():
        tangent = tangents[node]
        if tangent is None:
            tangent = _zero_like(node)
            generated_nodes.append(tangent)
        derivative_name = f"{TANGENT_PREFIX}{name}"
        derivative_outputs[derivative_name] = tangent
        output_map[derivative_name] = name
    definitions = tuple(
        dict.fromkeys([*program.nodes, *tangent_nodes.values(), *generated_nodes])
    )
    generated_program = Program(
        derivative_outputs,
        definitions,
        provenance={
            "schema": GENERATION_SCHEMA,
            "schema_version": GENERATION_VERSION,
            "mode": "jvp",
            "primal_logical_hash": primal_hash,
            "tangent_inputs": sorted(requested),
            "output_names": sorted(selected_outputs),
            "packed_inputs": program.provenance.get("packed_inputs", {}),
            "generation": "demand-driven",
        },
    )
    return JVPProgram(
        generated_program,
        primal_hash,
        tuple(sorted(requested)),
        tuple(sorted(selected_outputs)),
        {f"{TANGENT_PREFIX}{name}": name for name in sorted(requested)},
        output_map,
    )


def transpose_program(
    program: Program,
    cotangent_outputs,
    *,
    inputs=None,
    packed=None,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> VJPProgram:
    """Generate a demand-driven reverse derivative :class:`Program`.

    The generated program takes the original inputs plus ``bar_<output>``
    cotangent inputs and returns ``bar_<input>`` cotangents.  Reverse programs
    are generated only for primal nodes on a path from a requested input to a
    requested output.
    """
    if not isinstance(program, Program):
        raise TypeError("transpose_program requires a Program")
    if type(max_elements) is not int or max_elements < 0:
        raise ValueError("max_elements must be a nonnegative integer")
    primal_hash = program.logical_hash
    program, layouts = (
        _expand_packed_inputs(program, packed) if packed else (program, {})
    )
    selected_outputs = _select_names(
        program.outputs, cotangent_outputs, "cotangent output"
    )
    differentiable = {
        name: node
        for name, node in _input_nodes(program).items()
        if node.spec.differentiable
    }
    selected_inputs = _select_names(differentiable, inputs, "input")
    for name, node in selected_inputs.items():
        if node.spec.symmetries:
            raise NotImplementedError(
                "packed/symmetric cotangent inputs need a boundary transpose "
                "map; slice B currently generates dense general cotangents only"
            )
    groups = _input_groups(program)
    roots = (node for name in selected_inputs for node in groups[name])
    relevant = _ancestors(selected_outputs.values()) & _descendants_of(program, roots)
    bars: dict[Node, Node | None] = {}
    for name, node in selected_outputs.items():
        generated = f"{COTANGENT_PREFIX}{name}"
        if generated in _input_nodes(program) or generated in program.outputs:
            raise ValueError(
                f"generated cotangent name collides with input/output: {generated}"
            )
        cotangent = _derivative_input(node, generated)
        bars[node] = (
            cotangent if bars.get(node) is None else _combine([bars[node], cotangent])
        )
    generated_nodes = [bar for bar in bars.values() if bar is not None]
    for node in reversed(program.nodes):
        if node not in relevant or node.op in ("input", "constant"):
            continue
        bar = bars.get(node)
        if bar is None:
            continue
        contributions = _vjp_graph(
            node,
            bar,
            tuple(operand in relevant for operand in node.inputs),
            max_elements=max_elements,
        )
        for operand, contribution in zip(node.inputs, contributions):
            if contribution is None:
                continue
            generated_nodes.append(contribution)
            bars[operand] = (
                contribution
                if bars.get(operand) is None
                else _combine([bars[operand], contribution])
            )
    derivative_outputs = {}
    output_map = {}
    for name, node in selected_inputs.items():
        bar = _combine(bars.get(operand) for operand in groups[name])
        if bar is None:
            bar = _zero_like(node)
            generated_nodes.append(bar)
        if name in layouts:
            inverse = _inverse_weights_constant(node, layouts[name])
            bar = multiply(bar, inverse)
            generated_nodes.extend([inverse, bar])
        derivative_name = f"{COTANGENT_PREFIX}{name}"
        derivative_outputs[derivative_name] = bar
        output_map[derivative_name] = name
    definitions = tuple(dict.fromkeys([*program.nodes, *generated_nodes]))
    generated_program = Program(
        derivative_outputs,
        definitions,
        provenance={
            "schema": GENERATION_SCHEMA,
            "schema_version": GENERATION_VERSION,
            "mode": "vjp",
            "primal_logical_hash": primal_hash,
            "cotangent_outputs": sorted(selected_outputs),
            "input_names": sorted(selected_inputs),
            "packed_inputs": program.provenance.get("packed_inputs", {}),
            "generation": "demand-driven",
        },
    )
    return VJPProgram(
        generated_program,
        primal_hash,
        tuple(sorted(selected_outputs)),
        tuple(sorted(selected_inputs)),
        {f"{COTANGENT_PREFIX}{name}": name for name in sorted(selected_outputs)},
        {f"{COTANGENT_PREFIX}{name}": name for name in sorted(selected_inputs)},
    )

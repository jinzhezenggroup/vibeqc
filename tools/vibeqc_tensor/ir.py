"""Immutable tensor SSA and fail-closed primitive legality checks.

Every factory and deserialized node passes the same validation. There are no
mutable destinations, implicit broadcasting, conjugation, or iteration nodes.
Exact factors remain integer numerator/denominator pairs until interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from .types import Index, TensorSpec


def rational(value) -> tuple[int, int]:
    """Accept explicit exact coefficients, never approximate float spelling."""
    if type(value) not in (int, str, Fraction):
        raise TypeError("use an integer, Fraction, or rational string for coefficients")
    factor = Fraction(value)
    return factor.numerator, factor.denominator


def _fraction(pair) -> Fraction:
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(type(x) is not int for x in pair)
        or pair[1] <= 0
    ):
        raise ValueError("coefficient must be a normalized numerator/denominator pair")
    value = Fraction(*pair)
    if (value.numerator, value.denominator) != pair:
        raise ValueError("coefficient must be reduced")
    return value


def _freeze(value):
    """Only JSON data can enter attributes; executable objects cannot."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(x) for x in value)
    if type(value) in (int, str) or value is None:
        return value
    raise ValueError(
        "node attributes must contain only integers, strings, and sequences"
    )


@dataclass(frozen=True, eq=False)
class Node:
    """One typed SSA definition. Object identity preserves duplicate nodes.

    Structural equality is computed explicitly by the canonicalizer; Python
    recursion or a backend layout never decides mathematical equivalence.
    """

    op: str
    inputs: tuple[Node, ...]
    spec: TensorSpec
    attributes: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if any(not isinstance(n, Node) for n in self.inputs):
            raise TypeError("node operands must be tensor nodes")
        if not isinstance(self.spec, TensorSpec):
            raise TypeError("node requires a TensorSpec")
        attrs = tuple(sorted((k, _freeze(v)) for k, v in self.attributes))
        if any(not isinstance(k, str) for k, _ in attrs) or len(dict(attrs)) != len(
            attrs
        ):
            raise ValueError("duplicate or invalid node attributes")
        object.__setattr__(self, "attributes", attrs)
        _validate(self)

    @property
    def attrs(self) -> dict:
        """Return a detached attribute map, retaining immutable values."""
        return dict(self.attributes)


@dataclass(frozen=True)
class PrimitiveContract:
    """AD/lowering extension boundary; no derivative rules are implemented."""

    differentiable_operands: str
    accumulation: str
    storage: str = "fresh logical value; any physical aliases are read-only"


PRIMITIVES = {
    op: PrimitiveContract(
        "all real operands; integer indices and rational factors are static",
        "sum contributions when an operand is reused; gather repeats require scatter-add",
    )
    for op in (
        "add",
        "multiply",
        "divide",
        "einsum",
        "transpose",
        "reshape",
        "slice",
        "gather",
        "reduce",
        "broadcast",
    )
}


def _common(inputs: tuple[Node, ...]) -> TensorSpec:
    if not inputs:
        raise ValueError("operation requires operands")
    spec = inputs[0].spec
    if any(
        (n.spec.dtype, n.spec.representation) != (spec.dtype, spec.representation)
        for n in inputs
    ):
        raise ValueError("operand dtype and orbital representation must agree")
    return spec


def _result(inputs, *, indices=None, symmetries=()) -> TensorSpec:
    return _common(inputs).result(
        indices=indices,
        symmetries=symmetries,
        differentiable=any(n.spec.differentiable for n in inputs),
    )


def _axes(value, rank: int, *, permutation=False) -> tuple[int, ...]:
    value = tuple(value)
    if (
        any(type(i) is not int or not 0 <= i < rank for i in value)
        or len(set(value)) != len(value)
        or (permutation and len(value) != rank)
    ):
        raise ValueError("invalid or repeated axes")
    return value


def _einsum_domains(inputs, labels, output):
    if len(labels) != len(inputs):
        raise ValueError("einsum requires one label list per operand")
    domains = {}
    for node, names in zip(inputs, labels):
        if len(names) != len(node.spec.indices):
            raise ValueError("einsum label rank does not match operand")
        for name, index in zip(names, node.spec.indices):
            if type(name) is not int or name < 0:
                raise ValueError("einsum labels must be canonical nonnegative integers")
            if name in domains and domains[name].domain != index.domain:
                raise ValueError(
                    "einsum label reused across different index spaces/ranges"
                )
            domains[name] = index
    if len(set(output)) != len(output) or any(
        type(i) is not int or i not in domains for i in output
    ):
        raise ValueError("einsum outputs must be distinct labels present in the inputs")
    encountered = list(dict.fromkeys(i for names in labels for i in names))
    if encountered != list(range(len(encountered))):
        raise ValueError("einsum dummy labels must be normalized by first occurrence")
    return domains


def _infer(
    op: str, inputs: tuple[Node, ...], a: dict, declared: TensorSpec
) -> TensorSpec:
    """Infer safe result metadata; explicit view types are checked here too."""
    base = _common(inputs)
    if op in ("add", "multiply", "divide"):
        if any(
            tuple(i.domain for i in n.spec.indices)
            != tuple(i.domain for i in base.indices)
            for n in inputs
        ):
            raise ValueError("elementwise operands must have identical index domains")
        if op == "add":
            if len(a["coefficients"]) != len(inputs):
                raise ValueError("one exact coefficient is required per add operand")
            for pair in a["coefficients"]:
                _fraction(pair)
            symmetry = set(base.symmetries)
            for n in inputs[1:]:
                symmetry.intersection_update(n.spec.symmetries)
            return _result(inputs, symmetries=tuple(symmetry))
        if len(inputs) != 2:
            raise ValueError("elementwise multiply/divide require two operands")
        # Products of antisymmetric tensors need not remain antisymmetric.
        return _result(inputs)
    if len(inputs) != 1 and op != "einsum":
        raise ValueError(f"{op} requires exactly one operand")
    if op == "einsum":
        domains = _einsum_domains(inputs, a["labels"], a["output"])
        if tuple(i.domain for i in declared.indices) != tuple(
            domains[name].domain for name in a["output"]
        ):
            raise ValueError(
                "einsum output index domains/order do not match its labels"
            )
        _fraction(a["coefficient"])
        return _result(inputs, indices=declared.indices)
    if op == "transpose":
        order = _axes(a["axes"], len(base.indices), permutation=True)
        inverse = {old: new for new, old in enumerate(order)}
        symmetries = tuple(
            replace(s, permutation=tuple(inverse[s.permutation[i]] for i in order))
            for s in base.symmetries
        )
        return _result(
            inputs, indices=tuple(base.indices[i] for i in order), symmetries=symmetries
        )
    if op == "reshape":
        if base.size != declared.size:
            raise ValueError("reshape must preserve element count")
        return _result(inputs, indices=declared.indices)
    if op == "slice":
        if len(a["ranges"]) != len(base.indices):
            raise ValueError("slice requires one half-open range per axis")
        indices = []
        for index, bounds in zip(base.indices, a["ranges"]):
            if (
                len(bounds) != 2
                or any(type(i) is not int for i in bounds)
                or not 0 <= bounds[0] <= bounds[1] <= index.extent
            ):
                raise ValueError("slice range is outside its input axis")
            start, stop = bounds
            indices.append(
                replace(index, start=index.start + start, stop=index.start + stop)
                if index.selection is None
                else replace(index, selection=index.selection[start:stop])
            )
        return _result(inputs, indices=indices)
    if op == "gather":
        axis = _axes((a["axis"],), len(base.indices))[0]
        index = base.indices[axis]
        positions = a["positions"]
        if any(type(i) is not int or not 0 <= i < index.extent for i in positions):
            raise ValueError("gather position is outside its input axis")
        indices = list(base.indices)
        indices[axis] = replace(
            index, selection=tuple(index.coordinate(i) for i in positions)
        )
        return _result(inputs, indices=indices)
    if op == "reduce":
        axes = _axes(a["axes"], len(base.indices))
        if axes != tuple(sorted(axes)):
            raise ValueError("reduction axes must be in canonical order")
        return _result(
            inputs,
            indices=tuple(
                index for i, index in enumerate(base.indices) if i not in axes
            ),
        )
    if op == "broadcast":
        axes = _axes(a["axes"], len(declared.indices))
        if len(axes) != len(base.indices):
            raise ValueError("broadcast maps every input axis to one output axis")
        for source, axis in zip(base.indices, axes):
            target = declared.indices[axis]
            if source.domain != target.domain:
                raise ValueError(
                    "broadcast must preserve existing domains; insert new axes explicitly"
                )
        return _result(inputs, indices=declared.indices)
    raise ValueError(f"unsupported tensor primitive: {op}")


_ATTRS = {
    "input": {"name"},
    "constant": {"values"},
    "add": {"coefficients"},
    "multiply": set(),
    "divide": set(),
    "einsum": {"labels", "output", "coefficient"},
    "transpose": {"axes"},
    "reshape": set(),
    "slice": {"ranges"},
    "gather": {"axis", "positions"},
    "reduce": {"axes"},
    "broadcast": {"axes"},
}


def _validate(node: Node) -> None:
    if node.op not in _ATTRS:
        raise ValueError(
            f"unsupported tensor primitive (complex/conjugation included): {node.op}"
        )
    a, spec = node.attrs, node.spec
    if set(a) != _ATTRS[node.op]:
        raise ValueError(f"invalid attributes for {node.op}")
    if node.op in ("input", "constant"):
        if node.inputs:
            raise ValueError("input/constant nodes cannot have operands")
        if node.op == "input":
            if not isinstance(a["name"], str) or not a["name"].isidentifier():
                raise ValueError("input name must be an identifier")
            if spec.role not in ("input", "parameter"):
                raise ValueError("input node requires input or parameter role")
        else:
            if spec.role != "constant" or spec.symmetries:
                raise ValueError(
                    "literal constants require constant role and no declared symmetry"
                )
            if len(a["values"]) != spec.size:
                raise ValueError("constant length must match logical shape")
            for pair in a["values"]:
                _fraction(pair)
        return
    expected = _infer(node.op, node.inputs, a, spec)
    if spec != expected:
        raise ValueError(f"declared {node.op} result disagrees with inferred type")


def _make(op, inputs, attrs=(), *, indices=None) -> Node:
    inputs = tuple(inputs)
    declared = _result(inputs, indices=indices)
    attrs = dict(attrs)
    spec = _infer(op, inputs, attrs, declared)
    return Node(op, inputs, spec, tuple(attrs.items()))


def input_tensor(name: str, spec: TensorSpec) -> Node:
    """Declare a named immutable external tensor or trainable parameter."""
    return Node("input", (), spec, (("name", name),))


def constant(values, spec: TensorSpec | None = None) -> Node:
    """Define exact scalar or flattened row-major tensor literals."""
    if spec is None:
        spec = TensorSpec(role="constant")
    if type(values) in (int, str, Fraction):
        values = (values,)
    return Node("constant", (), spec, (("values", tuple(rational(x) for x in values)),))


def add(*inputs: Node, coefficients=None) -> Node:
    """Ordered rational-scaled sum, with no floating-point reassociation."""
    coefficients = (1,) * len(inputs) if coefficients is None else tuple(coefficients)
    return _make(
        "add", inputs, {"coefficients": tuple(rational(x) for x in coefficients)}
    )


def multiply(left: Node, right: Node) -> Node:
    """Elementwise product; broadcasting must be represented explicitly."""
    return _make("multiply", (left, right))


def divide(left: Node, right: Node) -> Node:
    """Elementwise quotient; a zero denominator is an execution error."""
    return _make("divide", (left, right))


def einsum(equation: str, *inputs: Node, coefficient=1) -> Node:
    """Explicit-output Einstein contraction without ellipses or conjugation.

    Alphabetic single-character labels are notation only. First-occurrence
    normalization makes dummy renaming immaterial to the logical equation.
    Repeated labels within an input take diagonals; omitted labels are summed.
    """
    if not isinstance(equation, str) or equation.count("->") != 1:
        raise ValueError("einsum requires an explicit 'inputs->output' equation")
    lhs, rhs = equation.replace(" ", "").split("->")
    terms = lhs.split(",")
    if any(not c.isascii() or not c.isalpha() for term in (*terms, rhs) for c in term):
        raise ValueError(
            "einsum supports alphabetic labels, without ellipses/conjugation"
        )
    mapping = {c: i for i, c in enumerate(dict.fromkeys("".join(terms)))}
    if any(c not in mapping for c in rhs):
        raise ValueError("einsum output label is absent from the inputs")
    labels = tuple(tuple(mapping[c] for c in term) for term in terms)
    output = tuple(mapping[c] for c in rhs)
    domains = _einsum_domains(inputs, labels, output)
    indices = tuple(replace(domains[mapping[c]], name=c) for c in rhs)
    return _make(
        "einsum",
        inputs,
        {"labels": labels, "output": output, "coefficient": rational(coefficient)},
        indices=indices,
    )


def transpose(value: Node, axes) -> Node:
    """Permute logical axes, carrying declared symmetry through the permutation."""
    return _make("transpose", (value,), {"axes": tuple(axes)})


def reshape(value: Node, indices: tuple[Index, ...]) -> Node:
    """Explicit row-major logical reshape; this is not an orbital transform."""
    return _make("reshape", (value,), indices=indices)


def slice_tensor(value: Node, ranges) -> Node:
    """Take unit-step, nonnegative half-open local ranges, including empties."""
    return _make("slice", (value,), {"ranges": tuple(tuple(r) for r in ranges)})


def gather(value: Node, axis: int, positions) -> Node:
    """Gather local positions, retaining repeated/reordered global coordinates."""
    return _make("gather", (value,), {"axis": axis, "positions": tuple(positions)})


def reduce_sum(value: Node, axes) -> Node:
    """Sum specified axes; reducing every axis produces a rank-zero scalar."""
    axes = _axes(axes, len(value.spec.indices))
    return _make("reduce", (value,), {"axes": tuple(sorted(axes))})


def broadcast(value: Node, indices: tuple[Index, ...], axes) -> Node:
    """Insert new axes with an explicit input-to-output axis map.

    Existing axes retain their populations/ranges. To expand a selected
    singleton across a different population, first reduce away that axis;
    equal numerical extents cannot authorize an orbital relabeling.
    """
    return _make("broadcast", (value,), {"axes": tuple(axes)}, indices=indices)

"""Symbolic TensorIR complexity analysis and opt-in contraction reassociation.

This module reasons only from declared index populations.  It never assumes
sparsity, screening, density fitting, low rank, or chemistry-specific identities.
Reassociation is therefore explicit opt-in because it changes floating-point
reduction order even when the mathematical contraction is identical.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, replace
from math import prod

from .ir import Node, _infer
from .precision import remap_precision_execution
from .program import Program

if typing.TYPE_CHECKING:
    from .types import Index

_SYMBOL_BY_KIND = {
    "ao": "N",
    "orbital": "N",
    "occupied": "O",
    "virtual": "V",
    "auxiliary": "A",
    "shell": "S",
    "atom": "R",
    "batch": "B",
    "history": "H",
    "matrix": "M",
    "pair": "P",
    "triplet": "T",
}
_SYMBOL_ORDER = tuple("NOVASRBHMPT")
_CONSTANT_KINDS = frozenset(("spin", "component", "cartesian"))


@dataclass(frozen=True)
class ComplexityMonomial:
    """One asymptotic product such as N^2 O V."""

    powers: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if any(
            not isinstance(symbol, str) or type(power) is not int or power <= 0
            for symbol, power in self.powers
        ):
            raise ValueError("complexity powers must be positive integer exponents")
        if len({symbol for symbol, _ in self.powers}) != len(self.powers):
            raise ValueError("complexity symbols must be unique")

    @property
    def degree(self) -> int:
        return sum(power for _, power in self.powers)

    @property
    def notation(self) -> str:
        if not self.powers:
            return "O(1)"
        factors = [
            symbol if power == 1 else f"{symbol}^{power}"
            for symbol, power in self.powers
        ]
        return "O(" + " ".join(factors) + ")"

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "notation": self.notation,
            "degree": self.degree,
            "powers": {symbol: power for symbol, power in self.powers},
        }


def _symbol(index: Index) -> str | None:
    if index.space.kind in _CONSTANT_KINDS:
        return None
    if (
        index.selection is not None
        or index.start != 0
        or index.stop != index.space.size
    ):
        return None
    return _SYMBOL_BY_KIND.get(index.space.kind)


def _ordered_powers(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    order = {symbol: position for position, symbol in enumerate(_SYMBOL_ORDER)}
    return tuple(
        sorted(
            counts.items(),
            key=lambda item: (order.get(item[0], len(order)), item[0]),
        )
    )


def _monomial(indices: typing.Iterable[Index]) -> ComplexityMonomial:
    counts: dict[str, int] = {}
    for index in indices:
        symbol = _symbol(index)
        if symbol is not None:
            counts[symbol] = counts.get(symbol, 0) + 1
    return ComplexityMonomial(_ordered_powers(counts))


def _einsum_domains(node: Node) -> dict[int, Index]:
    domains: dict[int, Index] = {}
    for child, labels in zip(node.inputs, node.attrs["labels"], strict=True):
        for label, index in zip(labels, child.spec.indices, strict=True):
            domains.setdefault(label, index)
    return domains


def _einsum_work(node: Node) -> tuple[ComplexityMonomial, int]:
    domains = _einsum_domains(node)
    labels = tuple(sorted(domains))
    monomial = _monomial(domains[label] for label in labels)
    elements = prod(domains[label].extent for label in labels)
    return monomial, len(node.inputs) * elements


@dataclass(frozen=True)
class NodeComplexity:
    storage: ComplexityMonomial
    work: ComplexityMonomial
    concrete_work: int

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "storage": self.storage.to_payload(),
            "work": self.work.to_payload(),
            "concrete_work": self.concrete_work,
        }


def node_complexity(node: Node) -> NodeComplexity:
    """Return shape-derived storage and one-materialization arithmetic order."""

    storage = _monomial(node.spec.indices)
    if node.op in ("input", "constant", "transpose", "reshape", "slice", "broadcast"):
        return NodeComplexity(storage, ComplexityMonomial(), 0)
    if node.op == "einsum":
        work, concrete = _einsum_work(node)
        return NodeComplexity(storage, work, concrete)
    if node.op == "reduce":
        source = node.inputs[0]
        return NodeComplexity(storage, _monomial(source.spec.indices), source.spec.size)
    return NodeComplexity(storage, storage, node.spec.size)


@dataclass(frozen=True)
class ComplexityEntry:
    name: str
    op: str
    output: bool
    complexity: NodeComplexity

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "op": self.op,
            "output": self.output,
            **self.complexity.to_payload(),
        }


@dataclass(frozen=True)
class ComplexityReport:
    entries: tuple[ComplexityEntry, ...]

    @property
    def max_storage_degree(self) -> int:
        return max(
            (entry.complexity.storage.degree for entry in self.entries), default=0
        )

    @property
    def max_work_degree(self) -> int:
        return max((entry.complexity.work.degree for entry in self.entries), default=0)

    def summary_payload(
        self, *, threshold: int = 4, limit: int = 32
    ) -> dict[str, typing.Any]:
        if type(threshold) is not int or threshold < 1:
            raise ValueError("complexity threshold must be a positive integer")
        if type(limit) is not int or limit < 1:
            raise ValueError("complexity diagnostic limit must be positive")

        materializations = [
            entry
            for entry in self.entries
            if entry.op not in ("input", "constant")
            and entry.complexity.storage.degree >= threshold
        ]
        work = [
            entry for entry in self.entries if entry.complexity.work.degree >= threshold
        ]

        def bounded(values: list[ComplexityEntry]) -> dict[str, typing.Any]:
            return {
                "items": [entry.to_payload() for entry in values[:limit]],
                "total": len(values),
                "truncated": max(0, len(values) - limit),
            }

        return {
            "schema": "generativeqc.tensor.symbolic-complexity.v1",
            "max_storage_degree": self.max_storage_degree,
            "max_work_degree": self.max_work_degree,
            "high_degree_materializations": bounded(materializations),
            "high_degree_work": bounded(work),
        }


def analyze_complexity(program: Program) -> ComplexityReport:
    """Analyze output-reachable TensorIR without changing the equation."""

    if not isinstance(program, Program):
        raise TypeError("complexity analysis requires a TensorIR Program")
    names = program.debug_names
    outputs = set(program.outputs.values())
    return ComplexityReport(
        tuple(
            ComplexityEntry(
                names[node],
                node.op,
                node in outputs,
                node_complexity(node),
            )
            for node in program.live_nodes
        )
    )


@dataclass(frozen=True)
class _TreePlan:
    node: Node
    labels: tuple[int, ...]
    max_degree: int
    concrete_work: int
    peak_elements: int
    signature: tuple[typing.Any, ...]


def _plan_key(plan: _TreePlan) -> tuple[typing.Any, ...]:
    return (
        plan.max_degree,
        plan.concrete_work,
        plan.peak_elements,
        plan.signature,
    )


def _dominates(left: _TreePlan, right: _TreePlan) -> bool:
    return (
        left.max_degree <= right.max_degree
        and left.concrete_work <= right.concrete_work
        and left.peak_elements <= right.peak_elements
    )


def _pareto_frontier(candidates: typing.Iterable[_TreePlan]) -> tuple[_TreePlan, ...]:
    frontier: list[_TreePlan] = []
    for candidate in sorted(candidates, key=_plan_key):
        if any(_dominates(existing, candidate) for existing in frontier):
            continue
        frontier = [
            existing for existing in frontier if not _dominates(candidate, existing)
        ]
        frontier.append(candidate)
    return tuple(frontier)


def _binary_einsum(
    left: _TreePlan,
    right: _TreePlan,
    output_labels: tuple[int, ...],
    domains: dict[int, Index],
    coefficient: tuple[int, int],
    final_spec: typing.Any = None,
) -> Node:
    encountered = tuple(dict.fromkeys(left.labels + right.labels))
    mapping = {label: position for position, label in enumerate(encountered)}
    labels = (
        tuple(mapping[label] for label in left.labels),
        tuple(mapping[label] for label in right.labels),
    )
    output = tuple(mapping[label] for label in output_labels)
    attrs = {"labels": labels, "output": output, "coefficient": coefficient}
    if final_spec is None:
        indices = tuple(
            replace(domains[label], name=f"c{position}")
            for position, label in enumerate(output_labels)
        )
        declared = left.node.spec.result(
            indices=indices,
            differentiable=(
                left.node.spec.differentiable or right.node.spec.differentiable
            ),
        )
    else:
        declared = final_spec
    spec = _infer("einsum", (left.node, right.node), attrs, declared)
    return Node("einsum", (left.node, right.node), spec, tuple(attrs.items()))


def _reassociate_node(
    node: Node, *, max_operands: int, max_intermediate_axes: dict[str, int]
) -> Node:
    if node.op != "einsum" or len(node.inputs) < 3 or len(node.inputs) > max_operands:
        return node
    labels_by_operand = tuple(tuple(labels) for labels in node.attrs["labels"])
    if any(len(labels) != len(set(labels)) for labels in labels_by_operand):
        return node

    domains = _einsum_domains(node)
    final_output = tuple(node.attrs["output"])
    final_set = set(final_output)
    operand_count = len(node.inputs)
    full_mask = (1 << operand_count) - 1

    appearances: dict[int, int] = {}
    for position, labels in enumerate(labels_by_operand):
        bit = 1 << position
        for label in labels:
            appearances[label] = appearances.get(label, 0) | bit

    def retained_labels(mask: int) -> tuple[int, ...]:
        if mask == full_mask:
            return final_output
        return tuple(
            label
            for label in sorted(domains)
            if appearances[label] & mask
            and (label in final_set or appearances[label] & (full_mask ^ mask))
        )

    plans: dict[int, tuple[_TreePlan, ...]] = {}
    for position, (child, labels) in enumerate(
        zip(node.inputs, labels_by_operand, strict=True)
    ):
        mask = 1 << position
        plans[mask] = (
            _TreePlan(
                child,
                labels,
                0,
                0,
                0,
                ("leaf", position),
            ),
        )

    masks = sorted(range(1, full_mask + 1), key=lambda value: value.bit_count())
    for mask in masks:
        if mask in plans:
            continue
        output_labels = retained_labels(mask)
        # A representation contract may forbid otherwise cheap intermediates
        # (for example, reconstructing four virtual axes from DF factors).
        # Prune this subset, not the entire search: another binary tree may
        # satisfy the contract. Public outputs and existing inputs are fixed.
        if mask != full_mask and any(
            sum(domains[label].space.kind == kind for label in output_labels) > limit
            for kind, limit in max_intermediate_axes.items()
        ):
            continue
        candidates: list[_TreePlan] = []
        left_mask = (mask - 1) & mask
        while left_mask:
            right_mask = mask ^ left_mask
            if (
                right_mask
                and left_mask < right_mask
                and left_mask in plans
                and right_mask in plans
            ):
                for left in plans[left_mask]:
                    for right in plans[right_mask]:
                        work_labels = tuple(dict.fromkeys(left.labels + right.labels))
                        work = _monomial(domains[label] for label in work_labels)
                        elements = prod(domains[label].extent for label in work_labels)
                        output_elements = prod(
                            (domains[label].extent for label in output_labels),
                            start=1,
                        )
                        coefficient = (
                            node.attrs["coefficient"] if mask == full_mask else (1, 1)
                        )
                        candidate_node = _binary_einsum(
                            left,
                            right,
                            output_labels,
                            domains,
                            coefficient,
                            node.spec if mask == full_mask else None,
                        )
                        candidates.append(
                            _TreePlan(
                                candidate_node,
                                output_labels,
                                max(left.max_degree, right.max_degree, work.degree),
                                left.concrete_work + right.concrete_work + 2 * elements,
                                max(
                                    left.peak_elements,
                                    right.peak_elements,
                                    output_elements,
                                ),
                                ("pair", left.signature, right.signature),
                            )
                        )
            left_mask = (left_mask - 1) & mask
        if not candidates:
            continue
        plans[mask] = _pareto_frontier(candidates)

    if full_mask not in plans:
        return node
    direct_degree = _monomial(domains.values()).degree
    best = min(plans[full_mask], key=_plan_key)
    return best.node if best.max_degree < direct_degree else node


def reassociate_einsums(
    program: Program,
    *,
    max_operands: int = 6,
    max_intermediate_axes: typing.Mapping[str, int] | None = None,
) -> Program:
    """Rewrite only n-ary contractions with a provably lower symbolic degree.

    This is explicit opt-in because the new binary tree changes floating-point
    reduction order.  The mathematical index contraction and exact rational
    coefficient are unchanged.

    ``max_intermediate_axes`` optionally bounds the number of axes of each
    index-space kind in new intermediates. It changes only scheduling, never
    existing inputs or public output shapes. If no lower-degree allowed tree
    exists, retain the direct contraction. This bounds representation rank,
    not total live memory, which remains the execution planner's responsibility.
    """

    if not isinstance(program, Program):
        raise TypeError("contraction reassociation requires a TensorIR Program")
    if "precision_execution" in program.provenance:
        raise ValueError(
            "contraction reassociation does not yet support explicit precision execution"
        )
    if type(max_operands) is not int or not 3 <= max_operands <= 8:
        raise ValueError("max_operands must lie in [3, 8]")
    from .types import SPACE_KINDS

    limits = dict(max_intermediate_axes or {})
    if any(
        kind not in SPACE_KINDS or type(limit) is not int or limit < 0
        for kind, limit in limits.items()
    ):
        raise ValueError(
            "intermediate axis limits require known kinds and nonnegative integers"
        )

    replacements: dict[Node, Node] = {}
    for node in program.nodes:
        inputs = tuple(replacements[child] for child in node.inputs)
        updated = node
        if inputs != node.inputs:
            spec = _infer(node.op, inputs, node.attrs, node.spec)
            updated = Node(node.op, inputs, spec, node.attributes)
        updated = _reassociate_node(
            updated, max_operands=max_operands, max_intermediate_axes=limits
        )
        replacements[node] = updated

    outputs = {name: replacements[node] for name, node in program.outputs.items()}
    definitions = tuple(replacements[node] for node in program.definitions)
    return Program(
        outputs,
        definitions,
        remap_precision_execution(program, replacements, outputs, definitions),
    )

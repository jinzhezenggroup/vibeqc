"""Small symbolic expression DAG with differentiation and structural CSE.

The production CUDA backend cannot afford runtime automatic-differentiation
objects in hot integral recurrences. This module moves that work to code
generation time: expressions are interned into a DAG, differentiated
symbolically, simplified locally, and emitted as scalar temporaries.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import cache

Coefficient = Fraction | float
Scalar = int | float | Fraction


@dataclass(frozen=True, slots=True)
class Node:
    """One immutable expression node owned by a :class:`Graph`."""

    operation: str
    arguments: tuple[int, ...] = ()
    payload: str | Coefficient | None = None


@dataclass(frozen=True, slots=True)
class SsaValueLifetime:
    """Definition and final-consumption events for one materialized value."""

    identifier: int
    operation: str
    definition_index: int
    last_use_index: int
    use_count: int


@dataclass(frozen=True, slots=True)
class SsaAnalysis:
    """Deterministic use/liveness summary for one ordered set of DAG roots.

    Constants and external variables are not materialized by the CUDA scalar
    emitter, so their dependency edges contribute to operation counts but not
    to the live-value estimate. Root reads are modeled as ordered output events
    after the last arithmetic definition, matching the current emitter shape.
    """

    root_count: int
    reachable_node_count: int
    operation_counts: tuple[tuple[str, int], ...]
    lifetimes: tuple[SsaValueLifetime, ...]
    arithmetic_operation_count: int
    peak_live_values: int

    @property
    def materialized_value_count(self) -> int:
        """Return the number of scalar temporaries emitted for these roots."""

        return len(self.lifetimes)

    @property
    def operation_count_by_kind(self) -> dict[str, int]:
        """Return reachable node counts keyed by deterministic operation name."""

        return dict(self.operation_counts)

    def to_payload(self) -> dict[str, object]:
        """Serialize aggregate static-model fields for tuning artifacts."""

        return {
            "root_count": self.root_count,
            "reachable_node_count": self.reachable_node_count,
            "operation_counts": self.operation_count_by_kind,
            "arithmetic_operation_count": self.arithmetic_operation_count,
            "materialized_value_count": self.materialized_value_count,
            "estimated_peak_live_values": self.peak_live_values,
        }


@dataclass(frozen=True, slots=True)
class RematerializationPolicy:
    """Bounded cost model controlling scalar CSE placement.

    ``inline_single_use`` shortens source-level live ranges without increasing
    arithmetic. ``rematerialize_multi_use`` may additionally duplicate cheap
    operations when their saved live range outweighs the configured operation
    cost. The arithmetic-growth limit prevents nested decisions from causing
    exponential expression expansion.
    """

    name: str
    inline_single_use: bool = False
    rematerialize_multi_use: bool = False
    cheap_operations: tuple[str, ...] = ("add", "multiply")
    operation_costs: tuple[tuple[str, float], ...] = (
        ("add", 1.0),
        ("multiply", 1.0),
        ("reciprocal", 4.0),
        ("power", 8.0),
        ("exp", 12.0),
    )
    live_range_weight: float = 1.0
    recomputation_weight: float = 1.0
    minimum_lifetime_span: int = 3
    maximum_extra_operation_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("rematerialization policy requires a name")
        if self.live_range_weight < 0.0 or self.recomputation_weight < 0.0:
            raise ValueError("rematerialization weights must be non-negative")
        if self.minimum_lifetime_span < 0:
            raise ValueError("minimum lifetime span must be non-negative")
        if self.maximum_extra_operation_fraction < 0.0:
            raise ValueError("operation-growth limit must be non-negative")
        if len(dict(self.operation_costs)) != len(self.operation_costs):
            raise ValueError("operation costs must use unique operation names")
        if any(cost < 0.0 for _, cost in self.operation_costs):
            raise ValueError("operation costs must be non-negative")

    @classmethod
    def materialized_cse(cls) -> RematerializationPolicy:
        """Preserve every structural CSE value as a scalar temporary."""

        return cls(name="materialized_cse")

    @classmethod
    def inline_single_use_values(cls) -> RematerializationPolicy:
        """Inline values that have exactly one source-level consumer."""

        return cls(name="inline_single_use", inline_single_use=True)

    @classmethod
    def pressure_rematerialized(cls) -> RematerializationPolicy:
        """Inline single-use values and selected cheap long-lived CSE nodes."""

        return cls(
            name="pressure_rematerialized",
            inline_single_use=True,
            rematerialize_multi_use=True,
        )

    def operation_cost(self, operation: str) -> float:
        """Return the target-independent relative cost of one operation."""

        return dict(self.operation_costs).get(operation, math.inf)


class AlgebraOrdering(str, Enum):
    """Ordering strategy for materialized scalar definitions."""

    TOPOLOGICAL = "topological"
    PRESSURE_AWARE = "pressure_aware"


class AlgebraFusion(str, Enum):
    """Arithmetic contraction strategy used by scalar lowering."""

    SEPARATE = "separate"
    FMA = "fma"


class AlgebraForm(str, Enum):
    """Associative representation selected before scalar optimization."""

    BINARY = "binary"
    CANONICAL_NARY = "canonical_nary"
    FACTORED_NARY = "factored_nary"


class PowerLowering(str, Enum):
    """Compile-time representation of small integral powers."""

    NATIVE = "native"
    SMALL_INTEGER = "small_integer"


@dataclass(frozen=True, slots=True)
class MaterializationDecision:
    """Explain whether one arithmetic DAG value remains a CUDA temporary."""

    identifier: int
    operation: str
    materialized: bool
    use_count: int
    lifetime_span: int
    operation_cost: float
    estimated_recomputation_cost: float
    estimated_live_range_cost: float
    reason: str


@dataclass(frozen=True, slots=True)
class MaterializationPlan:
    """Deterministic scalar placement plus exact emitted static metrics."""

    policy: RematerializationPolicy
    ordering: AlgebraOrdering
    fusion: AlgebraFusion
    root_identifiers: tuple[int, ...]
    decisions: tuple[MaterializationDecision, ...]
    emission_order: tuple[int, ...]
    fma_operations: tuple[tuple[int, int], ...]
    baseline_arithmetic_operation_count: int
    baseline_materialized_value_count: int
    baseline_peak_live_values: int
    operation_counts: tuple[tuple[str, int], ...]
    arithmetic_operation_count: int
    materialized_value_count: int
    peak_live_values: int

    @property
    def materialized_identifiers(self) -> frozenset[int]:
        """Return arithmetic node identifiers assigned to scalar temporaries."""

        return frozenset(
            decision.identifier for decision in self.decisions if decision.materialized
        )

    @property
    def inlined_value_count(self) -> int:
        """Return the number of canonical DAG values expanded at their uses."""

        return len(self.decisions) - self.materialized_value_count

    @property
    def rematerialized_value_count(self) -> int:
        """Return the number of multi-use DAG values deliberately recomputed."""

        return sum(
            not decision.materialized and decision.use_count > 1
            for decision in self.decisions
        )

    @property
    def reordered_value_count(self) -> int:
        """Return positions changed from the canonical topological order."""

        baseline_order = tuple(
            decision.identifier for decision in self.decisions if decision.materialized
        )
        return sum(
            baseline != emitted
            for baseline, emitted in zip(
                baseline_order, self.emission_order, strict=True
            )
        )

    @property
    def fma_operation_count(self) -> int:
        """Return the exact number of FMA occurrences in emitted expressions."""

        return dict(self.operation_counts).get("fma", 0)

    def to_payload(self) -> dict[str, object]:
        """Serialize compact pre/post placement metrics for tuning artifacts."""

        return {
            "policy": self.policy.name,
            "ordering": self.ordering.value,
            "fusion": self.fusion.value,
            "pre_optimization": {
                "arithmetic_operation_count": (
                    self.baseline_arithmetic_operation_count
                ),
                "materialized_value_count": self.baseline_materialized_value_count,
                "estimated_peak_live_values": self.baseline_peak_live_values,
            },
            "post_optimization": {
                "operation_counts": dict(self.operation_counts),
                "arithmetic_operation_count": self.arithmetic_operation_count,
                "materialized_value_count": self.materialized_value_count,
                "estimated_peak_live_values": self.peak_live_values,
            },
            "inlined_value_count": self.inlined_value_count,
            "rematerialized_value_count": self.rematerialized_value_count,
            "reordered_value_count": self.reordered_value_count,
            "fma_operation_count": self.fma_operation_count,
        }


class Expr:
    """Lightweight handle into one expression graph."""

    __slots__ = ("graph", "identifier")

    def __init__(self, graph: Graph, identifier: int) -> None:
        self.graph = graph
        self.identifier = identifier

    def __add__(self, other: Expr | Scalar) -> Expr:
        return self.graph.add(self, self.graph.coerce(other))

    def __radd__(self, other: Expr | Scalar) -> Expr:
        return self.graph.add(self.graph.coerce(other), self)

    def __sub__(self, other: Expr | Scalar) -> Expr:
        return self.graph.add(self, -self.graph.coerce(other))

    def __rsub__(self, other: Expr | Scalar) -> Expr:
        return self.graph.add(self.graph.coerce(other), -self)

    def __mul__(self, other: Expr | Scalar) -> Expr:
        return self.graph.multiply(self, self.graph.coerce(other))

    def __rmul__(self, other: Expr | Scalar) -> Expr:
        return self.graph.multiply(self.graph.coerce(other), self)

    def __truediv__(self, other: Expr | Scalar) -> Expr:
        return self.graph.multiply(
            self, self.graph.reciprocal(self.graph.coerce(other))
        )

    def __rtruediv__(self, other: Expr | Scalar) -> Expr:
        return self.graph.multiply(
            self.graph.coerce(other), self.graph.reciprocal(self)
        )

    def __neg__(self) -> Expr:
        return self.graph.multiply(self.graph.constant(-1), self)

    def pow(self, exponent: float) -> Expr:
        """Raise this expression to one compile-time scalar exponent."""

        return self.graph.power(self, exponent)


class Graph:
    """Interned expression DAG supporting forward symbolic differentiation."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self._identifiers: dict[Node, int] = {}
        self._constants: dict[Coefficient, Expr] = {}
        self._variables: dict[str, Expr] = {}

    def _intern(self, node: Node) -> Expr:
        identifier = self._identifiers.get(node)
        if identifier is None:
            identifier = len(self.nodes)
            self.nodes.append(node)
            self._identifiers[node] = identifier
        return Expr(self, identifier)

    def node(self, expression: Expr) -> Node:
        self._require_graph(expression)
        return self.nodes[expression.identifier]

    @staticmethod
    @cache
    def _exact_coefficient(value: Scalar) -> Coefficient:
        """Return an exact rational for every finite source-level scalar."""

        if isinstance(value, Fraction):
            return value
        if isinstance(value, int):
            return Fraction(value)
        numeric = float(value)
        if not math.isfinite(numeric):
            return numeric
        if numeric.is_integer():
            return Fraction(int(numeric))
        return Fraction(str(numeric))

    def _intern_constant(self, value: Coefficient) -> Expr:
        expression = self._constants.get(value)
        if expression is None:
            expression = self._intern(Node("constant", payload=value))
            self._constants[value] = expression
        return expression

    def constant(self, value: Scalar) -> Expr:
        """Intern one exact integer/decimal/rational algebra coefficient."""

        return self._intern_constant(self._exact_coefficient(value))

    def approximate_constant(self, value: float) -> Expr:
        """Intern a float already rounded by non-rational constant folding."""

        return self._intern_constant(float(value))

    def clone_constant(self, node: Node) -> Expr:
        """Copy one constant payload without reclassifying its exactness."""

        return self._intern_constant(self._constant_value(node))

    def variable(self, name: str) -> Expr:
        expression = self._variables.get(name)
        if expression is None:
            expression = self._intern(Node("variable", payload=name))
            self._variables[name] = expression
        return expression

    def coerce(self, value: Expr | Scalar) -> Expr:
        if isinstance(value, Expr):
            self._require_graph(value)
            return value
        return self.constant(value)

    def add(self, left: Expr, right: Expr) -> Expr:
        self._require_graph(left, right)
        if self.is_constant(left, 0.0):
            return right
        if self.is_constant(right, 0.0):
            return left
        left_node = self.node(left)
        right_node = self.node(right)
        if left_node.operation == "constant" and right_node.operation == "constant":
            return self._intern_constant(
                self._constant_value(left_node) + self._constant_value(right_node)
            )
        arguments = tuple(sorted((left.identifier, right.identifier)))
        return self._intern(Node("add", arguments))

    def add_many(self, values: Iterable[Expr]) -> Expr:
        """Build one flattened, constant-folded, deterministically sorted sum."""

        operands: list[Expr] = []
        constant: Coefficient = Fraction(0)
        pending = list(values)
        while pending:
            value = pending.pop()
            self._require_graph(value)
            node = self.node(value)
            if node.operation == "add":
                pending.extend(Expr(self, item) for item in node.arguments)
            elif node.operation == "constant":
                constant += self._constant_value(node)
            else:
                operands.append(value)
        if constant != 0:
            operands.append(self._intern_constant(constant))
        if not operands:
            return self.constant(0)
        if len(operands) == 1:
            return operands[0]
        return self._intern(
            Node("add", tuple(sorted(item.identifier for item in operands)))
        )

    def multiply(self, left: Expr, right: Expr) -> Expr:
        self._require_graph(left, right)
        if self.is_constant(left, 0.0) or self.is_constant(right, 0.0):
            return self.constant(0)
        if self.is_constant(left, 1.0):
            return right
        if self.is_constant(right, 1.0):
            return left
        left_node = self.node(left)
        right_node = self.node(right)
        if left_node.operation == "constant" and right_node.operation == "constant":
            return self._intern_constant(
                self._constant_value(left_node) * self._constant_value(right_node)
            )
        arguments = tuple(sorted((left.identifier, right.identifier)))
        return self._intern(Node("multiply", arguments))

    def multiply_many(self, values: Iterable[Expr]) -> Expr:
        """Build one flattened, constant-folded, deterministically sorted product."""

        operands: list[Expr] = []
        constant: Coefficient = Fraction(1)
        pending = list(values)
        while pending:
            value = pending.pop()
            self._require_graph(value)
            node = self.node(value)
            if node.operation == "multiply":
                pending.extend(Expr(self, item) for item in node.arguments)
            elif node.operation == "constant":
                factor = self._constant_value(node)
                if factor == 0:
                    return self.constant(0)
                constant *= factor
            else:
                operands.append(value)
        if constant != 1:
            operands.append(self._intern_constant(constant))
        if not operands:
            return self.constant(1)
        if len(operands) == 1:
            return operands[0]
        return self._intern(
            Node("multiply", tuple(sorted(item.identifier for item in operands)))
        )

    def factor_sum(self, values: Iterable[Expr]) -> Expr:
        """Greedily extract deterministic factors shared by two or more terms.

        Each iteration selects the factor present in the most terms, with the
        lowest graph identifier resolving ties. Replacing that group by one
        product strictly reduces the number of top-level terms, so recursive
        extraction terminates while exposing nested Horner-like structure.
        """

        flattened = self.add_many(values)
        node = self.node(flattened)
        if node.operation != "add":
            return flattened
        terms = [Expr(self, identifier) for identifier in node.arguments]
        occurrences: dict[int, list[int]] = {}
        for index, term in enumerate(terms):
            term_node = self.node(term)
            factors = (
                set(term_node.arguments)
                if term_node.operation == "multiply"
                else {term.identifier}
            )
            for factor in factors:
                occurrences.setdefault(factor, []).append(index)
        candidates = [
            (len(indices), -factor, factor, indices)
            for factor, indices in occurrences.items()
            if len(indices) >= 2
        ]
        if not candidates:
            return flattened
        _, _, factor, grouped_indices = max(candidates)
        grouped = set(grouped_indices)
        remainders = []
        for index in grouped_indices:
            term = terms[index]
            term_node = self.node(term)
            if term_node.operation != "multiply":
                remainders.append(self.constant(1))
                continue
            remaining = list(term_node.arguments)
            remaining.remove(factor)
            remainders.append(
                self.multiply_many(Expr(self, item) for item in remaining)
            )
        inner = self.factor_sum(remainders)
        factored = self.multiply_many((Expr(self, factor), inner))
        remaining_terms = [
            term for index, term in enumerate(terms) if index not in grouped
        ]
        remaining_terms.append(factored)
        return self.factor_sum(remaining_terms)

    def canonicalize_associative(
        self,
        roots: Sequence[Expr],
        *,
        factor_common: bool = False,
    ) -> tuple[Graph, tuple[Expr, ...]]:
        """Rebuild roots with canonical n-ary Add/Mul nodes.

        This pass runs after symbolic differentiation, so it can freely
        reassociate scalar arithmetic without changing derivative ownership.
        Constants are folded within each associative region and every operand
        tuple is sorted by deterministic target-graph identifiers.
        """

        normalized_roots = tuple(roots)
        for root in normalized_roots:
            self._require_graph(root)
        target = Graph()
        rebuilt: dict[int, Expr] = {}

        @cache
        def associative_arguments(
            identifier: int,
            operation: str,
        ) -> tuple[int, ...]:
            """Flatten one complete source-graph associative region."""

            node = self.nodes[identifier]
            if node.operation != operation:
                return (identifier,)
            return tuple(
                operand
                for argument in node.arguments
                for operand in associative_arguments(argument, operation)
            )

        @cache
        def structural_key(
            identifier: int,
        ) -> tuple[str, tuple[str, str], tuple[object, ...]]:
            """Order equal algebra independently of binary parenthesization."""

            node = self.nodes[identifier]
            if node.payload is None:
                payload = ("", "")
            elif isinstance(node.payload, Fraction):
                # Preserve the historical final-double ordering so exact
                # coefficients do not perturb greedy factor tie-breaks. The
                # rational suffix only resolves distinct exact values that
                # lower to the same double.
                payload = (
                    float(node.payload).hex(),
                    f"{node.payload.numerator}/{node.payload.denominator}",
                )
            elif isinstance(node.payload, float):
                payload = (node.payload.hex(), "")
            else:
                payload = (str(node.payload), "")
            if node.operation in ("add", "multiply"):
                arguments = associative_arguments(identifier, node.operation)
                child_keys = tuple(
                    sorted(structural_key(argument) for argument in arguments)
                )
            else:
                child_keys = tuple(
                    structural_key(argument) for argument in node.arguments
                )
            return node.operation, payload, child_keys

        def visit(identifier: int) -> Expr:
            cached = rebuilt.get(identifier)
            if cached is not None:
                return cached
            node = self.nodes[identifier]
            if node.operation == "constant":
                result = target.clone_constant(node)
            elif node.operation == "variable":
                result = target.variable(str(node.payload))
            elif node.operation == "add":
                source_arguments = sorted(
                    associative_arguments(identifier, "add"),
                    key=structural_key,
                )
                arguments = tuple(visit(item) for item in source_arguments)
                result = (
                    target.factor_sum(arguments)
                    if factor_common
                    else target.add_many(arguments)
                )
            elif node.operation == "multiply":
                source_arguments = sorted(
                    associative_arguments(identifier, "multiply"),
                    key=structural_key,
                )
                arguments = tuple(visit(item) for item in source_arguments)
                result = target.multiply_many(arguments)
            elif node.operation == "reciprocal":
                result = target.reciprocal(visit(node.arguments[0]))
            elif node.operation == "exp":
                result = target.exponential(visit(node.arguments[0]))
            elif node.operation == "power":
                result = target.power(
                    visit(node.arguments[0]),
                    float(node.payload),
                )
            else:
                raise ValueError(f"unsupported operation {node.operation!r}")
            rebuilt[identifier] = result
            return result

        return target, tuple(visit(root.identifier) for root in normalized_roots)

    def apply_algebra_form(
        self,
        roots: Sequence[Expr],
        form: AlgebraForm,
        power_lowering: PowerLowering = PowerLowering.NATIVE,
    ) -> tuple[Graph, tuple[Expr, ...]]:
        """Return roots in the requested power and associative representation."""

        normalized_roots = tuple(roots)
        graph = self
        if power_lowering == PowerLowering.SMALL_INTEGER:
            graph, normalized_roots = self.lower_small_integer_powers(normalized_roots)
        elif power_lowering != PowerLowering.NATIVE:
            raise ValueError(f"unsupported power lowering {power_lowering!r}")
        if form == AlgebraForm.BINARY:
            for root in normalized_roots:
                graph._require_graph(root)
            return graph, normalized_roots
        return graph.canonicalize_associative(
            normalized_roots,
            factor_common=form == AlgebraForm.FACTORED_NARY,
        )

    def lower_small_integer_powers(
        self,
        roots: Sequence[Expr],
        *,
        maximum_absolute_power: int = 4,
    ) -> tuple[Graph, tuple[Expr, ...]]:
        """Rebuild roots with bounded integer powers expanded into scalar ops.

        Positive powers use exponentiation by squaring, so fourth powers need
        two multiplies rather than three. Negative powers reuse the same
        positive addition chain beneath one reciprocal. Non-integral and
        larger powers remain explicit ``power`` nodes for CUDA ``pow``.
        """

        if maximum_absolute_power < 1:
            raise ValueError("maximum absolute power must be positive")
        normalized_roots = tuple(roots)
        for root in normalized_roots:
            self._require_graph(root)
        target = Graph()
        rebuilt: dict[int, Expr] = {}
        powers: dict[tuple[int, int], Expr] = {}

        def positive_power(base: Expr, exponent: int) -> Expr:
            key = (base.identifier, exponent)
            cached = powers.get(key)
            if cached is not None:
                return cached
            if exponent == 1:
                result = base
            else:
                half = positive_power(base, exponent // 2)
                result = target.multiply(half, half)
                if exponent & 1:
                    result = target.multiply(result, base)
            powers[key] = result
            return result

        def visit(identifier: int) -> Expr:
            cached = rebuilt.get(identifier)
            if cached is not None:
                return cached
            node = self.nodes[identifier]
            arguments = tuple(visit(item) for item in node.arguments)
            if node.operation == "constant":
                result = target.clone_constant(node)
            elif node.operation == "variable":
                result = target.variable(str(node.payload))
            elif node.operation == "add":
                result = (
                    target.add(arguments[0], arguments[1])
                    if len(arguments) == 2
                    else target.add_many(arguments)
                )
            elif node.operation == "multiply":
                result = (
                    target.multiply(arguments[0], arguments[1])
                    if len(arguments) == 2
                    else target.multiply_many(arguments)
                )
            elif node.operation == "reciprocal":
                result = target.reciprocal(arguments[0])
            elif node.operation == "exp":
                result = target.exponential(arguments[0])
            elif node.operation == "power":
                exponent = float(node.payload)
                if (
                    exponent.is_integer()
                    and 1 <= abs(exponent) <= maximum_absolute_power
                ):
                    integer_exponent = int(exponent)
                    result = positive_power(arguments[0], abs(integer_exponent))
                    if integer_exponent < 0:
                        result = target.reciprocal(result)
                else:
                    result = target.power(arguments[0], exponent)
            else:
                raise ValueError(f"unsupported operation {node.operation!r}")
            rebuilt[identifier] = result
            return result

        return target, tuple(visit(root.identifier) for root in normalized_roots)

    def reciprocal(self, value: Expr) -> Expr:
        self._require_graph(value)
        node = self.node(value)
        if node.operation == "constant":
            return self._intern_constant(1 / self._constant_value(node))
        return self._intern(Node("reciprocal", (value.identifier,)))

    def exponential(self, value: Expr) -> Expr:
        self._require_graph(value)
        node = self.node(value)
        if node.operation == "constant":
            return self.approximate_constant(
                math.exp(float(self._constant_value(node)))
            )
        return self._intern(Node("exp", (value.identifier,)))

    def power(self, value: Expr, exponent: float) -> Expr:
        self._require_graph(value)
        exponent = float(exponent)
        if exponent == 0.0:
            return self.constant(1)
        if exponent == 1.0:
            return value
        node = self.node(value)
        if node.operation == "constant":
            constant = self._constant_value(node)
            if isinstance(constant, Fraction) and exponent.is_integer():
                return self._intern_constant(constant ** int(exponent))
            return self.approximate_constant(float(constant) ** exponent)
        return self._intern(Node("power", (value.identifier,), exponent))

    def sum(self, values: Iterable[Expr]) -> Expr:
        result = self.constant(0)
        for value in values:
            result = result + value
        return result

    def is_constant(self, expression: Expr, value: Scalar | None = None) -> bool:
        node = self.node(expression)
        if node.operation != "constant":
            return False
        return value is None or float(self._constant_value(node)) == float(value)

    def differentiate(
        self,
        expression: Expr,
        variable: Expr,
        leaf_derivatives: Mapping[str, Expr] | None = None,
    ) -> Expr:
        """Differentiate one root while honoring custom external leaf rules.

        Boys values are supplied to generated kernels as a short sequence.
        Their derivative rule, ``dF_n(T) = -F_(n+1)(T) dT``, is therefore
        passed through ``leaf_derivatives`` instead of representing the Boys
        evaluator itself inside the algebra DAG.
        """

        self._require_graph(expression, variable)
        variable_node = self.node(variable)
        if variable_node.operation != "variable":
            raise ValueError("the differentiation target must be a variable")
        custom = dict(leaf_derivatives or {})
        memo: dict[int, Expr] = {}

        def visit(identifier: int) -> Expr:
            cached = memo.get(identifier)
            if cached is not None:
                return cached
            node = self.nodes[identifier]
            current = Expr(self, identifier)
            if node.operation == "constant":
                derivative = self.constant(0)
            elif node.operation == "variable":
                name = str(node.payload)
                derivative = custom.get(
                    name,
                    self.constant(1 if identifier == variable.identifier else 0),
                )
            elif node.operation == "add":
                if len(node.arguments) == 2:
                    derivative = visit(node.arguments[0]) + visit(node.arguments[1])
                else:
                    derivative = self.add_many(visit(item) for item in node.arguments)
            elif node.operation == "multiply":
                if len(node.arguments) == 2:
                    left = Expr(self, node.arguments[0])
                    right = Expr(self, node.arguments[1])
                    derivative = visit(left.identifier) * right + left * visit(
                        right.identifier
                    )
                else:
                    derivative = self.add_many(
                        self.multiply_many(
                            visit(argument)
                            if index == differentiated
                            else Expr(self, argument)
                            for index, argument in enumerate(node.arguments)
                        )
                        for differentiated in range(len(node.arguments))
                    )
            elif node.operation == "reciprocal":
                operand = Expr(self, node.arguments[0])
                derivative = -visit(operand.identifier) * operand.pow(-2.0)
            elif node.operation == "exp":
                operand = Expr(self, node.arguments[0])
                derivative = visit(operand.identifier) * current
            elif node.operation == "power":
                operand = Expr(self, node.arguments[0])
                exponent = float(node.payload)
                derivative = (
                    exponent * operand.pow(exponent - 1.0) * visit(operand.identifier)
                )
            else:
                raise ValueError(f"unsupported operation {node.operation!r}")
            memo[identifier] = derivative
            return derivative

        return visit(expression.identifier)

    def topological_order(self, roots: Sequence[Expr]) -> list[int]:
        """Return each reachable node once with dependencies first."""

        visited: set[int] = set()
        order: list[int] = []

        def visit(identifier: int) -> None:
            if identifier in visited:
                return
            visited.add(identifier)
            for argument in self.nodes[identifier].arguments:
                visit(argument)
            order.append(identifier)

        for root in roots:
            self._require_graph(root)
            visit(root.identifier)
        return order

    def analyze_ssa(self, roots: Sequence[Expr]) -> SsaAnalysis:
        """Return use counts, last uses, and peak materialized live values.

        The topological definition order is the same order consumed by
        :class:`~tools.vibeqc_codegen.cuda.CudaEmitter`. An input remains live
        through the event that defines its final consumer, so the estimate
        conservatively includes both operands and the result of that operation.
        """

        normalized_roots = tuple(roots)
        order = tuple(self.topological_order(normalized_roots))
        definition_index = {identifier: index for index, identifier in enumerate(order)}
        use_counts = {identifier: 0 for identifier in order}
        last_uses = dict(definition_index)
        for consumer in order:
            consumer_index = definition_index[consumer]
            for argument in self.nodes[consumer].arguments:
                use_counts[argument] += 1
                last_uses[argument] = max(last_uses[argument], consumer_index)

        output_begin = len(order)
        for offset, root in enumerate(normalized_roots):
            output_index = output_begin + offset
            use_counts[root.identifier] += 1
            last_uses[root.identifier] = max(last_uses[root.identifier], output_index)

        lifetimes = tuple(
            SsaValueLifetime(
                identifier=identifier,
                operation=self.nodes[identifier].operation,
                definition_index=definition_index[identifier],
                last_use_index=last_uses[identifier],
                use_count=use_counts[identifier],
            )
            for identifier in order
            if self.nodes[identifier].operation not in ("constant", "variable")
        )
        live_deltas: dict[int, int] = {}
        for lifetime in lifetimes:
            live_deltas[lifetime.definition_index] = (
                live_deltas.get(lifetime.definition_index, 0) + 1
            )
            release_index = lifetime.last_use_index + 1
            live_deltas[release_index] = live_deltas.get(release_index, 0) - 1
        live_values = 0
        peak_live_values = 0
        for event in sorted(live_deltas):
            live_values += live_deltas[event]
            peak_live_values = max(peak_live_values, live_values)

        counts = self.operation_counts(normalized_roots)
        operation_counts = tuple(sorted(counts.items()))
        arithmetic_operation_count = sum(
            self._node_arithmetic_operation_count(self.nodes[identifier])
            for identifier in order
        )
        return SsaAnalysis(
            root_count=len(normalized_roots),
            reachable_node_count=len(order),
            operation_counts=operation_counts,
            lifetimes=lifetimes,
            arithmetic_operation_count=arithmetic_operation_count,
            peak_live_values=peak_live_values,
        )

    def materialization_plan(
        self,
        roots: Sequence[Expr],
        policy: RematerializationPolicy | None = None,
        ordering: AlgebraOrdering = AlgebraOrdering.TOPOLOGICAL,
        fusion: AlgebraFusion = AlgebraFusion.SEPARATE,
    ) -> MaterializationPlan:
        """Choose scalar CSE values to retain under one bounded cost model.

        Decisions are made from canonical SSA use counts and lifetime spans.
        The returned post-plan operation and liveness metrics are then measured
        from the actual expression expansions that :class:`CudaEmitter` uses,
        including duplicated descendants of a rematerialized value.
        """

        normalized_roots = tuple(roots)
        selected_policy = policy or RematerializationPolicy.materialized_cse()
        baseline = self.analyze_ssa(normalized_roots)
        lifetimes = {item.identifier: item for item in baseline.lifetimes}
        materialized = set(lifetimes)
        reasons = {identifier: "structural_cse" for identifier in materialized}

        if selected_policy.inline_single_use:
            for lifetime in baseline.lifetimes:
                if lifetime.use_count == 1:
                    materialized.remove(lifetime.identifier)
                    reasons[lifetime.identifier] = "single_use"

        if fusion == AlgebraFusion.FMA:
            # A direct multiply consumed only by one add can be contracted
            # without recomputation. Select at most one multiply per binary add
            # so the lowering remains an ordinary three-operand FMA.
            for identifier in self.topological_order(normalized_roots):
                node = self.nodes[identifier]
                if node.operation != "add":
                    continue
                for argument in node.arguments:
                    lifetime = lifetimes.get(argument)
                    if (
                        lifetime is not None
                        and lifetime.operation == "multiply"
                        and lifetime.use_count == 1
                        and len(self.nodes[argument].arguments) == 2
                    ):
                        materialized.discard(argument)
                        reasons[argument] = "fma_operand"
                        break

        if selected_policy.rematerialize_multi_use:
            operation_budget = int(
                baseline.arithmetic_operation_count
                * selected_policy.maximum_extra_operation_fraction
            )
            candidates: list[tuple[float, int, int, int]] = []
            for lifetime in baseline.lifetimes:
                if lifetime.identifier not in materialized:
                    continue
                if lifetime.operation not in selected_policy.cheap_operations:
                    continue
                lifetime_span = lifetime.last_use_index - lifetime.definition_index
                if lifetime_span < selected_policy.minimum_lifetime_span:
                    continue
                operation_cost = selected_policy.operation_cost(
                    lifetime.operation
                ) * self._node_arithmetic_operation_count(
                    self.nodes[lifetime.identifier]
                )
                recomputation_cost = (
                    operation_cost
                    * max(0, lifetime.use_count - 1)
                    * selected_policy.recomputation_weight
                )
                live_range_cost = lifetime_span * selected_policy.live_range_weight
                benefit = live_range_cost - recomputation_cost
                if benefit <= 0.0:
                    continue
                # Higher benefit and longer spans win deterministic ties; a
                # lower identifier keeps source stable across Python versions.
                candidates.append(
                    (
                        benefit,
                        lifetime_span,
                        -lifetime.identifier,
                        max(0, lifetime.use_count - 1),
                    )
                )

            estimated_extra_operations = 0
            accepted: list[int] = []
            for _, _, negative_identifier, added_operations in sorted(
                candidates, reverse=True
            ):
                if estimated_extra_operations + added_operations > operation_budget:
                    continue
                identifier = -negative_identifier
                materialized.remove(identifier)
                reasons[identifier] = "live_range_benefit"
                accepted.append(identifier)
                estimated_extra_operations += added_operations

            # Nested inlining can duplicate more work than the local use-count
            # estimate. Enforce the budget against exact expanded arithmetic,
            # undoing the least valuable accepted decisions first.
            operation_counts = self._emitted_operation_counts(
                normalized_roots, materialized, fusion
            )
            while (
                sum(operation_counts.values())
                > baseline.arithmetic_operation_count + operation_budget
                and accepted
            ):
                identifier = accepted.pop()
                materialized.add(identifier)
                reasons[identifier] = "operation_growth_limit"
                operation_counts = self._emitted_operation_counts(
                    normalized_roots, materialized, fusion
                )
        else:
            operation_counts = self._emitted_operation_counts(
                normalized_roots, materialized, fusion
            )

        fma_operations = self._fma_operations(
            normalized_roots,
            materialized,
            fusion,
        )

        baseline_emission_order = tuple(
            identifier
            for identifier in self.topological_order(normalized_roots)
            if identifier in materialized
        )
        emission_order = baseline_emission_order
        baseline_plan_peak = self._materialized_peak_live_values(
            normalized_roots,
            materialized,
            baseline_emission_order,
        )
        if ordering == AlgebraOrdering.PRESSURE_AWARE:
            candidate_order = self._pressure_aware_materialized_order(
                normalized_roots,
                materialized,
                baseline_emission_order,
            )
            candidate_peak = self._materialized_peak_live_values(
                normalized_roots,
                materialized,
                candidate_order,
            )
            # A heuristic order is only actionable when the exact model proves
            # it lowers the peak; otherwise retain byte-stable topological code.
            if candidate_peak < baseline_plan_peak:
                emission_order = candidate_order
                peak_live_values = candidate_peak
            else:
                peak_live_values = baseline_plan_peak
        else:
            peak_live_values = baseline_plan_peak
        decisions = []
        for lifetime in baseline.lifetimes:
            lifetime_span = lifetime.last_use_index - lifetime.definition_index
            operation_cost = selected_policy.operation_cost(
                lifetime.operation
            ) * self._node_arithmetic_operation_count(self.nodes[lifetime.identifier])
            decisions.append(
                MaterializationDecision(
                    identifier=lifetime.identifier,
                    operation=lifetime.operation,
                    materialized=lifetime.identifier in materialized,
                    use_count=lifetime.use_count,
                    lifetime_span=lifetime_span,
                    operation_cost=operation_cost,
                    estimated_recomputation_cost=(
                        operation_cost * max(0, lifetime.use_count - 1)
                    ),
                    estimated_live_range_cost=(
                        lifetime_span * selected_policy.live_range_weight
                    ),
                    reason=reasons[lifetime.identifier],
                )
            )
        return MaterializationPlan(
            policy=selected_policy,
            ordering=ordering,
            fusion=fusion,
            root_identifiers=tuple(root.identifier for root in normalized_roots),
            decisions=tuple(decisions),
            emission_order=emission_order,
            fma_operations=fma_operations,
            baseline_arithmetic_operation_count=(baseline.arithmetic_operation_count),
            baseline_materialized_value_count=baseline.materialized_value_count,
            baseline_peak_live_values=baseline.peak_live_values,
            operation_counts=tuple(sorted(operation_counts.items())),
            arithmetic_operation_count=sum(operation_counts.values()),
            materialized_value_count=len(materialized),
            peak_live_values=peak_live_values,
        )

    def _emitted_operation_counts(
        self,
        roots: Sequence[Expr],
        materialized: set[int],
        fusion: AlgebraFusion,
    ) -> Counter[str]:
        """Count arithmetic occurrences after selective expression expansion."""

        fma_by_add = dict(self._fma_operations(roots, materialized, fusion))

        @cache
        def expression_counts(
            identifier: int,
            emit_materialized_root: bool = False,
        ) -> tuple[tuple[str, int], ...]:
            node = self.nodes[identifier]
            if node.operation in ("constant", "variable"):
                return ()
            if identifier in materialized and not emit_materialized_root:
                return ()
            fused_multiply = fma_by_add.get(identifier)
            if fused_multiply is not None:
                multiply = self.nodes[fused_multiply]
                counts = Counter({"fma": 1})
                remaining = list(node.arguments)
                remaining.remove(fused_multiply)
                if len(remaining) > 1:
                    counts["add"] += len(remaining) - 1
                for argument in (*multiply.arguments, *remaining):
                    counts.update(dict(expression_counts(argument)))
                return tuple(sorted(counts.items()))
            counts = Counter(
                {node.operation: self._node_arithmetic_operation_count(node)}
            )
            for argument in node.arguments:
                counts.update(dict(expression_counts(argument)))
            return tuple(sorted(counts.items()))

        counts: Counter[str] = Counter()
        for identifier in self.topological_order(roots):
            if identifier in materialized:
                counts.update(dict(expression_counts(identifier, True)))
        for root in roots:
            counts.update(dict(expression_counts(root.identifier)))
        return counts

    def _fma_operations(
        self,
        roots: Sequence[Expr],
        materialized: set[int],
        fusion: AlgebraFusion,
    ) -> tuple[tuple[int, int], ...]:
        """Return ``(add, multiply)`` pairs contracted by scalar lowering."""

        if fusion != AlgebraFusion.FMA:
            return ()
        operations = []
        for identifier in self.topological_order(roots):
            node = self.nodes[identifier]
            if node.operation != "add":
                continue
            for argument in node.arguments:
                if (
                    argument not in materialized
                    and self.nodes[argument].operation == "multiply"
                    and len(self.nodes[argument].arguments) == 2
                ):
                    operations.append((identifier, argument))
                    break
        return tuple(operations)

    @staticmethod
    def _node_arithmetic_operation_count(node: Node) -> int:
        """Return scalar instructions represented by one expression node."""

        if node.operation in ("constant", "variable"):
            return 0
        if node.operation in ("add", "multiply"):
            return len(node.arguments) - 1
        return 1

    def _materialized_peak_live_values(
        self,
        roots: Sequence[Expr],
        materialized: set[int],
        emission_order: Sequence[int],
    ) -> int:
        """Measure exact temporary liveness after inline references expand."""

        ordered = tuple(emission_order)
        if set(ordered) != materialized or len(ordered) != len(materialized):
            raise ValueError(
                "emission order must contain every materialized value once"
            )
        definition_index = {
            identifier: index for index, identifier in enumerate(ordered)
        }
        last_uses = dict(definition_index)

        @cache
        def referenced_values(identifier: int) -> tuple[tuple[int, int], ...]:
            node = self.nodes[identifier]
            if identifier in materialized:
                return ((identifier, 1),)
            if node.operation in ("constant", "variable"):
                return ()
            references: Counter[int] = Counter()
            for argument in node.arguments:
                references.update(dict(referenced_values(argument)))
            return tuple(sorted(references.items()))

        for consumer_index, consumer in enumerate(ordered):
            for argument in self.nodes[consumer].arguments:
                for identifier in dict(referenced_values(argument)):
                    last_uses[identifier] = max(last_uses[identifier], consumer_index)
        output_begin = len(ordered)
        for offset, root in enumerate(roots):
            for identifier in dict(referenced_values(root.identifier)):
                last_uses[identifier] = max(
                    last_uses[identifier], output_begin + offset
                )

        live_deltas: dict[int, int] = {}
        for identifier in ordered:
            definition = definition_index[identifier]
            live_deltas[definition] = live_deltas.get(definition, 0) + 1
            release = last_uses[identifier] + 1
            live_deltas[release] = live_deltas.get(release, 0) - 1
        live_values = 0
        peak_live_values = 0
        for event in sorted(live_deltas):
            live_values += live_deltas[event]
            peak_live_values = max(peak_live_values, live_values)
        return peak_live_values

    def _pressure_aware_materialized_order(
        self,
        roots: Sequence[Expr],
        materialized: set[int],
        baseline_order: Sequence[int],
    ) -> tuple[int, ...]:
        """List-schedule ready definitions to free their operands promptly.

        Inlined nodes are transparent: a materialized definition depends on
        every retained value reached through its expanded arguments. Candidate
        priority first minimizes the immediate live-value delta, then favors
        the longest downstream chain so independent subgraphs are not opened
        prematurely. Canonical order resolves all remaining ties.
        """

        @cache
        def referenced_values(identifier: int) -> frozenset[int]:
            node = self.nodes[identifier]
            if identifier in materialized:
                return frozenset((identifier,))
            if node.operation in ("constant", "variable"):
                return frozenset()
            references: set[int] = set()
            for argument in node.arguments:
                references.update(referenced_values(argument))
            return frozenset(references)

        dependencies: dict[int, frozenset[int]] = {}
        consumers = {identifier: set() for identifier in materialized}
        remaining_consumer_events = Counter[int]()
        for identifier in materialized:
            references: set[int] = set()
            for argument in self.nodes[identifier].arguments:
                references.update(referenced_values(argument))
            dependencies[identifier] = frozenset(references)
            for dependency in references:
                consumers[dependency].add(identifier)
                remaining_consumer_events[dependency] += 1
        for root in roots:
            for dependency in referenced_values(root.identifier):
                remaining_consumer_events[dependency] += 1

        canonical_index = {
            identifier: index for index, identifier in enumerate(baseline_order)
        }
        downstream_height: dict[int, int] = {}
        for identifier in reversed(tuple(baseline_order)):
            downstream_height[identifier] = 1 + max(
                (downstream_height[consumer] for consumer in consumers[identifier]),
                default=0,
            )

        unscheduled = set(materialized)
        ready = {
            identifier for identifier in materialized if not dependencies[identifier]
        }
        order = []
        while ready:

            def priority(identifier: int) -> tuple[int, int, int]:
                freed_operands = sum(
                    remaining_consumer_events[dependency] == 1
                    for dependency in dependencies[identifier]
                )
                return (
                    1 - freed_operands,
                    -downstream_height[identifier],
                    canonical_index[identifier],
                )

            selected = min(ready, key=priority)
            ready.remove(selected)
            unscheduled.remove(selected)
            order.append(selected)
            for dependency in dependencies[selected]:
                remaining_consumer_events[dependency] -= 1
            for consumer in consumers[selected]:
                if consumer in unscheduled and dependencies[consumer].isdisjoint(
                    unscheduled
                ):
                    ready.add(consumer)
        if unscheduled:
            raise RuntimeError("materialized expression dependencies contain a cycle")
        return tuple(order)

    def evaluate(self, expression: Expr, variables: Mapping[str, float]) -> float:
        """Evaluate one root for generator tests and finite-difference oracles."""

        values: dict[int, float] = {}
        for identifier in self.topological_order([expression]):
            node = self.nodes[identifier]
            if node.operation == "constant":
                result = float(self._constant_value(node))
            elif node.operation == "variable":
                result = float(variables[str(node.payload)])
            elif node.operation == "add":
                result = sum(values[item] for item in node.arguments)
            elif node.operation == "multiply":
                result = math.prod(values[item] for item in node.arguments)
            elif node.operation == "reciprocal":
                result = 1.0 / values[node.arguments[0]]
            elif node.operation == "exp":
                result = math.exp(values[node.arguments[0]])
            elif node.operation == "power":
                result = values[node.arguments[0]] ** float(node.payload)
            else:
                raise ValueError(f"unsupported operation {node.operation!r}")
            values[identifier] = result
        return values[expression.identifier]

    def operation_counts(self, roots: Sequence[Expr]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for identifier in self.topological_order(roots):
            operation = self.nodes[identifier].operation
            counts[operation] = counts.get(operation, 0) + 1
        return counts

    @staticmethod
    def _constant_value(node: Node) -> Coefficient:
        """Return the typed payload of one validated constant node."""

        if node.operation != "constant" or not isinstance(
            node.payload,
            (Fraction, float),
        ):
            raise TypeError("constant node has an invalid coefficient payload")
        return node.payload

    def _require_graph(self, *expressions: Expr) -> None:
        if any(expression.graph is not self for expression in expressions):
            raise ValueError("expressions from different graphs cannot be combined")

"""Small symbolic expression DAG with differentiation and structural CSE.

The production CUDA backend cannot afford runtime automatic-differentiation
objects in hot integral recurrences. This module moves that work to code
generation time: expressions are interned into a DAG, differentiated
symbolically, simplified locally, and emitted as scalar temporaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Node:
    """One immutable expression node owned by a :class:`Graph`."""

    operation: str
    arguments: tuple[int, ...] = ()
    payload: str | float | None = None


class Expr:
    """Lightweight handle into one expression graph."""

    __slots__ = ("graph", "identifier")

    def __init__(self, graph: Graph, identifier: int) -> None:
        self.graph = graph
        self.identifier = identifier

    def __add__(self, other: Expr | float) -> Expr:
        return self.graph.add(self, self.graph.coerce(other))

    def __radd__(self, other: Expr | float) -> Expr:
        return self.graph.add(self.graph.coerce(other), self)

    def __sub__(self, other: Expr | float) -> Expr:
        return self.graph.add(self, -self.graph.coerce(other))

    def __rsub__(self, other: Expr | float) -> Expr:
        return self.graph.add(self.graph.coerce(other), -self)

    def __mul__(self, other: Expr | float) -> Expr:
        return self.graph.multiply(self, self.graph.coerce(other))

    def __rmul__(self, other: Expr | float) -> Expr:
        return self.graph.multiply(self.graph.coerce(other), self)

    def __truediv__(self, other: Expr | float) -> Expr:
        return self.graph.multiply(
            self, self.graph.reciprocal(self.graph.coerce(other))
        )

    def __rtruediv__(self, other: Expr | float) -> Expr:
        return self.graph.multiply(
            self.graph.coerce(other), self.graph.reciprocal(self)
        )

    def __neg__(self) -> Expr:
        return self.graph.multiply(self.graph.constant(-1.0), self)

    def pow(self, exponent: float) -> Expr:
        """Raise this expression to one compile-time scalar exponent."""

        return self.graph.power(self, exponent)


class Graph:
    """Interned expression DAG supporting forward symbolic differentiation."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self._identifiers: dict[Node, int] = {}
        self._constants: dict[float, Expr] = {}
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

    def constant(self, value: float) -> Expr:
        value = float(value)
        expression = self._constants.get(value)
        if expression is None:
            expression = self._intern(Node("constant", payload=value))
            self._constants[value] = expression
        return expression

    def variable(self, name: str) -> Expr:
        expression = self._variables.get(name)
        if expression is None:
            expression = self._intern(Node("variable", payload=name))
            self._variables[name] = expression
        return expression

    def coerce(self, value: Expr | float) -> Expr:
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
            return self.constant(float(left_node.payload) + float(right_node.payload))
        arguments = tuple(sorted((left.identifier, right.identifier)))
        return self._intern(Node("add", arguments))

    def multiply(self, left: Expr, right: Expr) -> Expr:
        self._require_graph(left, right)
        if self.is_constant(left, 0.0) or self.is_constant(right, 0.0):
            return self.constant(0.0)
        if self.is_constant(left, 1.0):
            return right
        if self.is_constant(right, 1.0):
            return left
        left_node = self.node(left)
        right_node = self.node(right)
        if left_node.operation == "constant" and right_node.operation == "constant":
            return self.constant(float(left_node.payload) * float(right_node.payload))
        arguments = tuple(sorted((left.identifier, right.identifier)))
        return self._intern(Node("multiply", arguments))

    def reciprocal(self, value: Expr) -> Expr:
        self._require_graph(value)
        node = self.node(value)
        if node.operation == "constant":
            return self.constant(1.0 / float(node.payload))
        return self._intern(Node("reciprocal", (value.identifier,)))

    def exponential(self, value: Expr) -> Expr:
        self._require_graph(value)
        node = self.node(value)
        if node.operation == "constant":
            return self.constant(math.exp(float(node.payload)))
        return self._intern(Node("exp", (value.identifier,)))

    def power(self, value: Expr, exponent: float) -> Expr:
        self._require_graph(value)
        exponent = float(exponent)
        if exponent == 0.0:
            return self.constant(1.0)
        if exponent == 1.0:
            return value
        node = self.node(value)
        if node.operation == "constant":
            return self.constant(float(node.payload) ** exponent)
        return self._intern(Node("power", (value.identifier,), exponent))

    def sum(self, values: Iterable[Expr]) -> Expr:
        result = self.constant(0.0)
        for value in values:
            result = result + value
        return result

    def is_constant(self, expression: Expr, value: float | None = None) -> bool:
        node = self.node(expression)
        if node.operation != "constant":
            return False
        return value is None or float(node.payload) == value

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
                derivative = self.constant(0.0)
            elif node.operation == "variable":
                name = str(node.payload)
                derivative = custom.get(
                    name,
                    self.constant(1.0 if identifier == variable.identifier else 0.0),
                )
            elif node.operation == "add":
                derivative = visit(node.arguments[0]) + visit(node.arguments[1])
            elif node.operation == "multiply":
                left = Expr(self, node.arguments[0])
                right = Expr(self, node.arguments[1])
                derivative = visit(left.identifier) * right + left * visit(right.identifier)
            elif node.operation == "reciprocal":
                operand = Expr(self, node.arguments[0])
                derivative = -visit(operand.identifier) * operand.pow(-2.0)
            elif node.operation == "exp":
                operand = Expr(self, node.arguments[0])
                derivative = visit(operand.identifier) * current
            elif node.operation == "power":
                operand = Expr(self, node.arguments[0])
                exponent = float(node.payload)
                derivative = exponent * operand.pow(exponent - 1.0) * visit(
                    operand.identifier
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

    def evaluate(self, expression: Expr, variables: Mapping[str, float]) -> float:
        """Evaluate one root for generator tests and finite-difference oracles."""

        values: dict[int, float] = {}
        for identifier in self.topological_order([expression]):
            node = self.nodes[identifier]
            if node.operation == "constant":
                result = float(node.payload)
            elif node.operation == "variable":
                result = float(variables[str(node.payload)])
            elif node.operation == "add":
                result = values[node.arguments[0]] + values[node.arguments[1]]
            elif node.operation == "multiply":
                result = values[node.arguments[0]] * values[node.arguments[1]]
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

    def _require_graph(self, *expressions: Expr) -> None:
        if any(expression.graph is not self for expression in expressions):
            raise ValueError("expressions from different graphs cannot be combined")

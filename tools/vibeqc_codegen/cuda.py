"""CUDA scalar emitter for symbolic expression DAGs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .expr import Expr, Graph, MaterializationPlan


def format_constant(value: float) -> str:
    """Emit an unambiguous double literal accepted by NVCC."""

    if value == 0.0:
        return "0.0"
    if value == 1.0:
        return "1.0"
    if value == -1.0:
        return "-1.0"
    return f"{value:.17g}"


class CudaEmitter:
    """Emit selectively materialized DAG values with shared CSE state."""

    def __init__(
        self,
        graph: Graph,
        variables: Mapping[str, str],
        materialization_plan: MaterializationPlan | None = None,
    ) -> None:
        self.graph = graph
        self.variables = dict(variables)
        self.materialization_plan = materialization_plan
        self.names: dict[int, str] = {}
        self.lines: list[str] = []
        self._temporary = 0
        self._materialized: set[int] = set()

    def emit(self, roots: Sequence[Expr]) -> None:
        normalized_roots = tuple(roots)
        if self.materialization_plan is None:
            materialized = {
                identifier
                for identifier in self.graph.topological_order(normalized_roots)
                if self.graph.nodes[identifier].operation
                not in ("constant", "variable")
            }
        else:
            root_identifiers = tuple(root.identifier for root in normalized_roots)
            if root_identifiers != self.materialization_plan.root_identifiers:
                raise ValueError("materialization plan roots do not match emission roots")
            materialized = set(self.materialization_plan.materialized_identifiers)
        self._materialized = materialized
        for identifier in self.graph.topological_order(normalized_roots):
            if identifier in self.names:
                continue
            node = self.graph.nodes[identifier]
            if node.operation == "constant":
                self.names[identifier] = format_constant(float(node.payload))
                continue
            if node.operation == "variable":
                name = str(node.payload)
                self.names[identifier] = self.variables.get(name, name)
                continue
            if identifier not in materialized:
                continue
            arguments = [self._reference(item) for item in node.arguments]
            code = self._operation_code(identifier, arguments)
            name = f"v{self._temporary}"
            self._temporary += 1
            self.names[identifier] = name
            self.lines.append(f"  const double {name} = {code};")

    def _operation_code(self, identifier: int, arguments: Sequence[str]) -> str:
        """Lower one arithmetic node using already parenthesized arguments."""

        node = self.graph.nodes[identifier]
        if node.operation == "add":
            return f"{arguments[0]} + {arguments[1]}"
        if node.operation == "multiply":
            return f"{arguments[0]} * {arguments[1]}"
        if node.operation == "reciprocal":
            return f"1.0 / {arguments[0]}"
        if node.operation == "exp":
            return f"exp({arguments[0]})"
        if node.operation == "power":
            return (
                f"pow({arguments[0]}, "
                f"{format_constant(float(node.payload))})"
            )
        raise ValueError(f"unsupported CUDA operation {node.operation!r}")

    def _reference(self, identifier: int) -> str:
        """Return a leaf/CSE name or recursively expand an inlined value."""

        name = self.names.get(identifier)
        if name is not None:
            return name
        node = self.graph.nodes[identifier]
        if node.operation in ("constant", "variable"):
            raise RuntimeError("expression leaf was not initialized before use")
        if identifier in self._materialized:
            raise RuntimeError("materialized dependency was not emitted before use")
        arguments = [self._reference(item) for item in node.arguments]
        code = self._operation_code(identifier, arguments)
        if node.operation in ("exp", "power"):
            return code
        return f"({code})"

    def reference(self, expression: Expr) -> str:
        """Return the emitted scalar reference for one expression."""

        if expression.graph is not self.graph:
            raise ValueError("expression belongs to a different graph")
        return self._reference(expression.identifier)

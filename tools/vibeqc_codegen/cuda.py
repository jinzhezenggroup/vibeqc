"""CUDA scalar emitter for symbolic expression DAGs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .expr import Coefficient, Expr, Graph, MaterializationPlan


def format_constant(value: Coefficient) -> str:
    """Lower an exact/approximate coefficient to one CUDA double literal."""

    numeric = float(value)
    if numeric == 0.0:
        return "0.0"
    if numeric == 1.0:
        return "1.0"
    if numeric == -1.0:
        return "-1.0"
    return f"{numeric:.17g}"


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
        self._fma_by_add: dict[int, int] = {}

    def emit(self, roots: Sequence[Expr]) -> None:
        normalized_roots = tuple(roots)
        topological_order = tuple(self.graph.topological_order(normalized_roots))
        if self.materialization_plan is None:
            materialized = {
                identifier
                for identifier in topological_order
                if self.graph.nodes[identifier].operation
                not in ("constant", "variable")
            }
            emission_order = topological_order
        else:
            root_identifiers = tuple(root.identifier for root in normalized_roots)
            if root_identifiers != self.materialization_plan.root_identifiers:
                raise ValueError("materialization plan roots do not match emission roots")
            materialized = set(self.materialization_plan.materialized_identifiers)
            emission_order = self.materialization_plan.emission_order
            self._fma_by_add = dict(self.materialization_plan.fma_operations)
            # Reordered definitions may reference leaves that occur later in
            # the canonical walk, so initialize every external name first.
            for identifier in topological_order:
                node = self.graph.nodes[identifier]
                if node.operation == "constant":
                    self.names[identifier] = format_constant(float(node.payload))
                elif node.operation == "variable":
                    name = str(node.payload)
                    self.names[identifier] = self.variables.get(name, name)
        self._materialized = materialized
        for identifier in emission_order:
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
            code = self._operation_code(identifier)
            name = f"v{self._temporary}"
            self._temporary += 1
            self.names[identifier] = name
            self.lines.append(f"  const double {name} = {code};")

    def emit_assignment(self, expression: Expr, target: str) -> None:
        """Emit one expression and bind its root to an existing CUDA lvalue.

        Binding the root after the assignment lets later expressions reuse the
        stored field directly instead of retaining an otherwise dead temporary.
        This is useful for structured geometry records whose fields form the
        boundary between symbolic algebra and backend storage.
        """

        self.emit((expression,))
        self.lines.append(f"  {target} = {self.reference(expression)};")
        self.names[expression.identifier] = target

    def _operation_code(self, identifier: int) -> str:
        """Lower one arithmetic node using the plan's contraction decisions."""

        node = self.graph.nodes[identifier]
        fused_multiply = self._fma_by_add.get(identifier)
        if fused_multiply is not None:
            multiply = self.graph.nodes[fused_multiply]
            remaining = list(node.arguments)
            remaining.remove(fused_multiply)
            accumulator = (
                self._reference(remaining[0])
                if len(remaining) == 1
                else "(" + " + ".join(
                    self._reference(item) for item in remaining
                ) + ")"
            )
            arguments = [self._reference(item) for item in multiply.arguments]
            return f"fma({arguments[0]}, {arguments[1]}, {accumulator})"
        arguments = [self._reference(item) for item in node.arguments]
        if node.operation == "add":
            return " + ".join(arguments)
        if node.operation == "multiply":
            return " * ".join(arguments)
        if node.operation == "reciprocal":
            return f"1.0 / {arguments[0]}"
        if node.operation == "exp":
            return f"exp({arguments[0]})"
        if node.operation == "power":
            exponent = float(node.payload)
            if exponent == 0.5:
                return f"sqrt({arguments[0]})"
            return (
                f"pow({arguments[0]}, "
                f"{format_constant(exponent)})"
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
        code = self._operation_code(identifier)
        if node.operation in ("exp", "power") or identifier in self._fma_by_add:
            return code
        return f"({code})"

    def reference(self, expression: Expr) -> str:
        """Return the emitted scalar reference for one expression."""

        if expression.graph is not self.graph:
            raise ValueError("expression belongs to a different graph")
        return self._reference(expression.identifier)

"""CUDA scalar emitter for symbolic expression DAGs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .expr import Expr, Graph


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
    """Emit one assignment per non-leaf DAG node with shared CSE state."""

    def __init__(self, graph: Graph, variables: Mapping[str, str]) -> None:
        self.graph = graph
        self.variables = dict(variables)
        self.names: dict[int, str] = {}
        self.lines: list[str] = []
        self._temporary = 0

    def emit(self, roots: Sequence[Expr]) -> None:
        for identifier in self.graph.topological_order(roots):
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
            arguments = [self.names[item] for item in node.arguments]
            if node.operation == "add":
                code = f"{arguments[0]} + {arguments[1]}"
            elif node.operation == "multiply":
                code = f"{arguments[0]} * {arguments[1]}"
            elif node.operation == "reciprocal":
                code = f"1.0 / {arguments[0]}"
            elif node.operation == "exp":
                code = f"exp({arguments[0]})"
            elif node.operation == "power":
                code = f"pow({arguments[0]}, {format_constant(float(node.payload))})"
            else:
                raise ValueError(f"unsupported CUDA operation {node.operation!r}")
            name = f"v{self._temporary}"
            self._temporary += 1
            self.names[identifier] = name
            self.lines.append(f"  const double {name} = {code};")

    def reference(self, expression: Expr) -> str:
        """Return the emitted scalar reference for one expression."""

        return self.names[expression.identifier]

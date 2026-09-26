"""Runtime-independent scalar grid response IR shared by CUDA generation.

Keep NumPy evaluation lazy: uninstalled CMake generation only needs Graph and
standard-library metadata. The response consumer re-exports these canonical
objects so caches, scalar serialization and scientific identities stay shared.
"""

import typing
from dataclasses import dataclass
from functools import lru_cache

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.expr import Graph


@dataclass(frozen=True)
class GridResponseProgram:
    """Shared scalar primal/JVP roots, independent of runtime or molecule size."""

    graph: object
    roots: tuple
    identity: str

    def evaluate(self, **variables: typing.Any) -> typing.Any:
        from vibeqc_compiler.common.array_graph import evaluate_array_graph

        return evaluate_array_graph(self.graph, self.roots, variables)


@lru_cache(maxsize=8, typed=True)
def grid_response_program(kind: typing.Any, iterations: typing.Any = 3) -> typing.Any:
    """Generate local mathematics once, not one graph per nuclear coordinate."""
    if type(iterations) is not int or not 1 <= iterations <= 5:
        raise ValueError("partition iterations must be an integer in [1, 5]")
    graph = Graph()
    if kind == "norm":
        names = ("x", "y", "z")
        xyz = [graph.variable(name) for name in names]
        primal = graph.power(graph.sum(v * v for v in xyz), 0.5)
    elif kind == "ratio":
        names = ("a", "b")
        primal = graph.variable("a") / graph.variable("b")
    elif kind == "log":
        names = ("p",)
        primal = graph.stable_unary("log", graph.variable("p"))
    elif kind == "becke":
        names = ("mu",)
        mu = graph.variable("mu")
        for _ in range(iterations):
            mu = 0.5 * mu * (3 - mu * mu)
        primal = 0.5 * (1 - mu)
    else:
        raise ValueError("unknown grid response primitive")
    tangent = graph.differentiate(
        primal,
        graph.variable("direction"),
        {name: graph.variable(f"d{name}") for name in names},
    )
    roots = (primal, tangent)
    reachable = graph.topological_order(roots)
    indices = {node: i for i, node in enumerate(reachable)}
    identity = canonical_hash(
        {
            "schema": "vibeqc.grid-response-program/v1",
            "kind": kind,
            "nodes": [
                (
                    graph.nodes[i].operation,
                    [indices[j] for j in graph.nodes[i].arguments],
                    str(graph.nodes[i].payload),
                )
                for i in reachable
            ],
            "roots": [indices[root.identifier] for root in roots],
        }
    )
    return GridResponseProgram(graph, roots, identity)


@lru_cache(maxsize=8, typed=True)
def grid_mixed_response_program(
    kind: typing.Any, iterations: typing.Any = 3
) -> typing.Any:
    """Generate primal, two JVPs and their mixed directional derivative.

    The left/right tangent leaves are independent. mixed_* inputs describe
    a mixed derivative of an upstream leaf; ordinary Cartesian nuclear
    directions bind them to zero. This lets composed primitives retain the
    complete chain rule without introducing a second handwritten derivative.
    """
    if type(iterations) is not int or not 1 <= iterations <= 5:
        raise ValueError("partition iterations must be an integer in [1, 5]")
    graph = Graph()
    if kind == "norm":
        names = ("x", "y", "z")
        xyz = [graph.variable(name) for name in names]
        primal = graph.power(graph.sum(v * v for v in xyz), 0.5)
    elif kind == "ratio":
        names = ("a", "b")
        primal = graph.variable("a") / graph.variable("b")
    elif kind == "log":
        names = ("p",)
        primal = graph.stable_unary("log", graph.variable("p"))
    elif kind == "becke":
        names = ("mu",)
        mu = graph.variable("mu")
        for _ in range(iterations):
            mu = 0.5 * mu * (3 - mu * mu)
        primal = 0.5 * (1 - mu)
    else:
        raise ValueError("unknown grid response primitive")

    left = graph.differentiate(
        primal,
        graph.variable("left_direction"),
        {name: graph.variable(f"l{name}") for name in names},
    )
    right = graph.differentiate(
        primal,
        graph.variable("right_direction"),
        {name: graph.variable(f"r{name}") for name in names},
    )
    mixed = graph.differentiate(
        left,
        graph.variable("right_direction"),
        {
            **{name: graph.variable(f"r{name}") for name in names},
            **{f"l{name}": graph.variable(f"lr{name}") for name in names},
        },
    )
    roots = (primal, left, right, mixed)
    reachable = graph.topological_order(roots)
    indices = {node: i for i, node in enumerate(reachable)}
    identity = canonical_hash(
        {
            "schema": "vibeqc.grid-mixed-response-program/v1",
            "kind": kind,
            "nodes": [
                (
                    graph.nodes[i].operation,
                    [indices[j] for j in graph.nodes[i].arguments],
                    str(graph.nodes[i].payload),
                )
                for i in reachable
            ],
            "roots": [indices[root.identifier] for root in roots],
        }
    )
    return GridResponseProgram(graph, roots, identity)

"""Array interpretation of the existing scalar DAG for diagnostic consumers.

This module owns no expression representation or differentiation rules. It
executes the shared graph protocol and keeps scientific domain checks in the
caller. Native lowerings continue to consume the same graph directly.
"""

import math
import typing

import numpy as np


def _operation(node: typing.Any, args: typing.Any, variables: typing.Any) -> typing.Any:
    if node.operation == "constant":
        return float(node.payload)
    if node.operation == "variable":
        return variables[node.payload]
    if node.operation == "add":
        return sum(args)
    if node.operation == "multiply":
        value = 1
        for arg in args:
            value = value * arg
        return value
    if node.operation == "reciprocal":
        return 1 / args[0]
    if node.operation == "power":
        return args[0] ** float(node.payload)
    if node.operation in ("exp", "log", "log1p", "expm1"):
        return getattr(np, node.operation)(args[0])
    if node.operation == "atan":
        return np.arctan(args[0])
    if node.operation == "asinh":
        return np.arcsinh(args[0])
    if node.operation == "erf":
        return np.vectorize(math.erf, otypes=[np.float64])(args[0])
    raise ValueError(f"unsupported scalar primitive {node.operation!r}")


def _evaluate_piecewise(
    graph: typing.Any, roots: typing.Any, variables: typing.Any
) -> typing.Any:
    """Evaluate piecewise graphs lane-wise without touching inactive branches."""

    arrays = [np.asarray(value) for value in variables.values()]
    shape = np.broadcast_shapes(*(value.shape for value in arrays)) if arrays else ()
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    inputs = {
        name: np.broadcast_to(np.asarray(value), shape).reshape(size)
        for name, value in variables.items()
    }
    values = {}
    ready = {}

    def visit(identifier: typing.Any, lanes: typing.Any) -> typing.Any:
        if identifier not in values:
            values[identifier] = np.empty(size, dtype=np.float64)
            ready[identifier] = np.zeros(size, dtype=bool)
        missing = lanes[~ready[identifier][lanes]]
        if missing.size:
            node = graph.nodes[identifier]
            if node.operation == "select_le":
                left, right, if_true, if_false = node.arguments
                choose_true = visit(left, missing) <= visit(right, missing)
                true_lanes = missing[choose_true]
                false_lanes = missing[~choose_true]
                if true_lanes.size:
                    values[identifier][true_lanes] = visit(if_true, true_lanes)
                if false_lanes.size:
                    values[identifier][false_lanes] = visit(if_false, false_lanes)
            else:
                args = [visit(argument, missing) for argument in node.arguments]
                local = (
                    {node.payload: inputs[node.payload][missing]}
                    if node.operation == "variable"
                    else {}
                )
                values[identifier][missing] = _operation(node, args, local)
            ready[identifier][missing] = True
        return values[identifier][lanes]

    lanes = np.arange(size)
    with np.errstate(all="raise", under="ignore"):
        return tuple(visit(root.identifier, lanes).reshape(shape) for root in roots)


def evaluate_array_graph(
    graph: typing.Any, roots: typing.Any, variables: typing.Any
) -> typing.Any:
    """Evaluate shared intermediates once and return one value per root.

    Variables may be scalars or broadcast-compatible arrays. Constant roots remain
    scalars; the caller owns output shapes and any physical-domain restrictions.
    Floating-point exceptions propagate instead of silently regularizing inputs.
    """
    order = graph.topological_order(roots)
    if any(graph.nodes[index].operation == "select_le" for index in order):
        return _evaluate_piecewise(graph, roots, variables)
    values = {}
    with np.errstate(all="raise", under="ignore"):
        for index in order:
            node = graph.nodes[index]
            args = [values[i] for i in node.arguments]
            values[index] = _operation(node, args, variables)
    return tuple(values[root.identifier] for root in roots)

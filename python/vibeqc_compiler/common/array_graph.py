"""Array interpretation of the existing scalar DAG for diagnostic consumers.

This module owns no expression representation or differentiation rules. It
executes the shared graph protocol and keeps scientific domain checks in the
caller. Native lowerings continue to consume the same graph directly.
"""

import numpy as np


def evaluate_array_graph(graph, roots, variables):
    """Evaluate shared intermediates once and return one value per root.

    Variables may be scalars or broadcast-compatible arrays. Constant roots remain
    scalars; the caller owns output shapes and any physical-domain restrictions.
    Floating-point exceptions propagate instead of silently regularizing inputs.
    """
    values = {}
    with np.errstate(all="raise"):
        for index in graph.topological_order(roots):
            node = graph.nodes[index]
            args = [values[i] for i in node.arguments]
            if node.operation == "constant":
                value = float(node.payload)
            elif node.operation == "variable":
                value = variables[node.payload]
            elif node.operation == "add":
                value = sum(args)
            elif node.operation == "multiply":
                value = 1
                for arg in args:
                    value = value * arg
            elif node.operation == "reciprocal":
                value = 1 / args[0]
            elif node.operation == "power":
                value = args[0] ** float(node.payload)
            elif node.operation in ("exp", "log", "log1p", "expm1"):
                value = getattr(np, node.operation)(args[0])
            else:
                raise ValueError(f"unsupported scalar primitive {node.operation!r}")
            values[index] = value
    return tuple(values[root.identifier] for root in roots)

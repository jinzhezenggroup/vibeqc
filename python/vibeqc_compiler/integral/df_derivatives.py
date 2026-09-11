"""Generation-time DF derivatives with generic raw and external-weight outputs.

The interpreter differentiates the physical metric/three-center value DAG,
including Gaussian decay, center shifts and Boys arguments. The bounded CUDA
lowering uses the same Gaussian-moment definitions and the derivative of each
basis factor, then integrates polynomial coefficients against shared Boys
moments. Neither lowering carries automatic differentiation into runtime.
"""

import math
from dataclasses import dataclass, replace
from functools import cache

from .blocks import RawBlock, TensorLayout, WeightDescriptor, WeightedDerivative
from .df_values import (
    _reference_boys,
    build_df_axis_moment,
    build_df_component_kernel,
    build_df_value_ir,
)
from .expr import Expr, Graph, Node
from .shell_spec import AXES


def build_df_derivative_ir(family, angular, *, weighted=False):
    """Declare full (center,xyz,AO...) blocks with fixed bar_A or bar_M weights."""
    value = build_df_value_ir(family, angular)
    centers = len(value.operator.centers)
    layout = TensorLayout(("center", "xyz"), (centers, 3))
    if weighted:
        weights = WeightDescriptor(
            "bar_M" if centers == 2 else "bar_A", value.contractions[0].layout
        )
        consumer = WeightedDerivative(weights, layout, layout.storage_bytes)
    else:
        layout = TensorLayout(
            layout.indices + value.signature.tensor_indices,
            layout.shape + value.signature.component_shape,
        )
        consumer = RawBlock(layout, layout.storage_bytes)
    return replace(
        value, derivative=value.operator.nuclear_derivative(), contractions=(consumer,)
    )


@dataclass(frozen=True)
class DFDerivativeKernel:
    """One primitive's physical value DAG and independent-center derivatives."""

    graph: Graph
    value: Expr
    gradients: tuple
    boys_argument: Expr
    boys_count: int
    centers_count: int


def build_df_derivative_kernel(integral, components):
    """Differentiate physical centers before lowering recurrence boundaries."""
    if integral.derivative is None or integral.derivative.order != 1:
        raise ValueError("DF derivative lowering requires first nuclear derivatives")
    if any(
        not isinstance(c, (RawBlock, WeightedDerivative)) for c in integral.contractions
    ):
        raise ValueError("DF derivatives require raw or external-weight consumers")
    value_ir = build_df_value_ir(integral.operator.family, integral.signature.angular)
    value_ir = replace(integral, derivative=None, contractions=value_ir.contractions)
    kernel = build_df_component_kernel(value_ir, components)
    g = Graph()
    count = len(components)
    exponent = tuple(g.variable(f"exponent_{i}") for i in range(count))
    centers = tuple(
        tuple(g.variable(f"center_{i}_{axis}") for axis in AXES) for i in range(count)
    )
    p, q = sum(exponent[:-1]), exponent[-1]
    rho = p * q / (p + q)
    product = tuple(
        sum(exponent[i] * centers[i][axis] for i in range(count - 1)) / p
        for axis in range(3)
    )
    difference = tuple(product[axis] - centers[-1][axis] for axis in range(3))
    argument = rho * sum(d * d for d in difference)
    replacements = {"inverse_two_p": 0.5 / p, "inverse_two_q": 0.5 / q, "rho": rho}
    replacements.update({f"difference_{axis}": d for axis, d in zip(AXES, difference)})
    if count == 3:
        replacements.update(
            {
                f"p{i}_{axis}": product[a] - centers[i][a]
                for i in range(2)
                for a, axis in enumerate(AXES)
            }
        )

    # High-angular-momentum values contain deep addition chains. Preserve the
    # recursive clone's dependency order with an explicit stack so Python 3.11
    # and coverage tracing do not exhaust the interpreter recursion limit.
    cloned = {}
    pending = [(kernel.value.identifier, False)]
    while pending:
        identifier, ready = pending.pop()
        if identifier in cloned:
            continue
        node = kernel.graph.nodes[identifier]
        if node.arguments and not ready:
            pending.append((identifier, True))
            pending.extend((child, False) for child in reversed(node.arguments))
            continue
        if node.operation == "variable":
            cloned[identifier] = replacements.get(
                node.payload, g.variable(node.payload)
            )
        elif node.operation == "constant":
            cloned[identifier] = g.clone_constant(node)
        else:
            cloned[identifier] = g._intern(
                Node(
                    node.operation,
                    tuple(cloned[a].identifier for a in node.arguments),
                    node.payload,
                )
            )

    value = (
        (2 * math.pi**2.5)
        / (p * q * (p + q).pow(0.5))
        * cloned[kernel.value.identifier]
    )
    if count == 3:
        decay_argument = (
            -exponent[0]
            * exponent[1]
            / p
            * sum((a - b) * (a - b) for a, b in zip(centers[0], centers[1]))
        )
        value *= g.exponential(decay_argument)
    derivative = integral.derivative
    independent = derivative.independent_centers(integral.operator)
    roots = {}
    boys_count = sum(integral.signature.angular) + 2
    for center in independent:
        axes = []
        for coordinate in centers[center]:
            dt = g.differentiate(argument, coordinate)
            leaves = {
                f"boys_{n}": -g.variable(f"boys_{n + 1}") * dt
                for n in range(boys_count - 1)
            }
            axes.append(g.differentiate(value, coordinate, leaf_derivatives=leaves))
        roots[center] = tuple(axes)
    for recovered in derivative.recovered_centers(integral.operator):
        roots[recovered] = tuple(
            -sum(roots[c][axis] for c in independent) for axis in range(3)
        )
    return DFDerivativeKernel(
        g,
        value,
        tuple(roots[c] for c in derivative.requested_centers(integral.operator)),
        argument,
        boys_count,
        count,
    )


def evaluate_df_derivative(kernel, exponents, centers):
    """Evaluate generated roots for independent oracle/finite-difference tests."""
    if len(exponents) != kernel.centers_count or len(centers) != kernel.centers_count:
        raise ValueError("DF primitive centers/exponents do not match the signature")
    if any(not math.isfinite(a) or a <= 0 for a in exponents):
        raise ValueError("DF basis exponents must be finite and positive")
    variables = {f"exponent_{i}": a for i, a in enumerate(exponents)}
    variables.update(
        {
            f"center_{i}_{axis}": v
            for i, c in enumerate(centers)
            for axis, v in zip(AXES, c)
        }
    )
    argument = kernel.graph.evaluate(kernel.boys_argument, variables)
    variables.update(
        {
            f"boys_{i}": f
            for i, f in enumerate(_reference_boys(argument, kernel.boys_count))
        }
    )
    return tuple(
        tuple(kernel.graph.evaluate(root, variables) for root in axes)
        for axes in kernel.gradients
    )


def axis_polynomial(a, b, c):
    """Rewrite the shared Gaussian moment DAG as coefficients in u=t^2.

    Rys means and covariances are affine in u. Coefficient convolution therefore
    gives the exact same moment polynomial without differentiating fitted Rys
    roots. First derivatives need at most F_10 for public f/f/f, including the
    internal raising of one orbital Gaussian power.
    """
    source, root = build_df_axis_moment(a, b, c, internal_derivative=True)
    g = Graph()
    z = g.constant(0)
    pa, pb, dx, sx, sy, ip, iq = (
        g.variable(name) for name in ("pa", "pb", "dx", "sx", "sy", "ip", "iq")
    )
    variables = {
        "mean_0": (pa, -dx * sx),
        "mean_1": (pb, -dx * sx),
        "mean_2": (z, dx * sy),
        "variance_x": (ip, -ip * sx),
        "covariance_xy": (z, ip * sy),
        "variance_y": (iq, -iq * sy),
    }

    @cache
    def coefficients(identifier):
        node = source.nodes[identifier]
        if node.operation == "constant":
            return (g.clone_constant(node),)
        if node.operation == "variable":
            return variables[node.payload]
        children = [coefficients(i) for i in node.arguments]
        if node.operation == "add":
            return tuple(
                sum(child[i] if i < len(child) else z for child in children)
                for i in range(max(map(len, children)))
            )
        if node.operation == "multiply":
            result = (g.constant(1),)
            for child in children:
                result = tuple(
                    sum(
                        result[i] * child[k - i]
                        for i in range(len(result))
                        if 0 <= k - i < len(child)
                    )
                    for k in range(len(result) + len(child) - 1)
                )
            return result
        raise ValueError(f"non-polynomial Gaussian moment node: {node.operation}")

    return g, coefficients(root.identifier)

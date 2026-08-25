"""Symbolic shell-class kernel builders.

The first pilot is canonical ``(p s | s s)``. It is intentionally small
enough to validate differentiation, Boys-chain handling, CSE, and CUDA
emission against the existing hand-written analytic derivative before the
same machinery is extended to the dominant order-five classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .cuda import CudaEmitter
from .expr import Expr, Graph


AXES = ("x", "y", "z")
CENTERS = ("first", "second", "third", "fourth")


@dataclass(frozen=True, slots=True)
class PsssKernel:
    """One symbolic ``(p_axis s|s s)`` value and all-center gradients."""

    graph: Graph
    p_axis: str
    variables: Mapping[str, Expr]
    boys_argument: Expr
    value: Expr
    gradients: tuple[tuple[Expr, Expr, Expr], ...]


def _squared_distance(
    graph: Graph,
    coordinates: Mapping[str, Mapping[str, Expr]],
    first: str,
    second: str,
) -> Expr:
    return graph.sum(
        (coordinates[first][axis] - coordinates[second][axis]).pow(2.0)
        for axis in AXES
    )


def build_psss_kernel(p_axis: str) -> PsssKernel:
    """Build the canonical primitive value and symbolic nuclear derivatives."""

    if p_axis not in AXES:
        raise ValueError(f"unsupported p axis {p_axis!r}")
    graph = Graph()
    alpha = graph.variable("alpha")
    beta = graph.variable("beta")
    gamma = graph.variable("gamma")
    delta = graph.variable("delta")
    coordinates = {
        center: {
            axis: graph.variable(f"{center}_{axis}") for axis in AXES
        }
        for center in CENTERS
    }
    variables: dict[str, Expr] = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
    }
    for center in CENTERS:
        variables.update(
            {
                f"{center}_{axis}": coordinates[center][axis]
                for axis in AXES
            }
        )

    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    rho = p * q / (p + q)
    product_p = {
        axis: (
            alpha * coordinates["first"][axis]
            + beta * coordinates["second"][axis]
        )
        / p
        for axis in AXES
    }
    product_q = {
        axis: (
            gamma * coordinates["third"][axis]
            + delta * coordinates["fourth"][axis]
        )
        / q
        for axis in AXES
    }
    product_difference = {
        axis: product_p[axis] - product_q[axis] for axis in AXES
    }
    boys_argument = rho * graph.sum(
        product_difference[axis].pow(2.0) for axis in AXES
    )
    boys = tuple(graph.variable(f"boys_{order}") for order in range(3))
    pair_decay = graph.exponential(
        -mu * _squared_distance(graph, coordinates, "first", "second")
        - nu * _squared_distance(graph, coordinates, "third", "fourth")
    )
    pi = graph.variable("kPi")
    normalization = (
        2.0 * pi.pow(2.5) / (p * q * (p + q).pow(0.5))
    )
    pa = product_p[p_axis] - coordinates["first"][p_axis]
    primitive_value = (
        pa * boys[0] - (rho / p) * product_difference[p_axis] * boys[1]
    )
    value = normalization * pair_decay * primitive_value

    independent_gradients: list[tuple[Expr, Expr, Expr]] = []
    for center in CENTERS[:3]:
        center_gradients: list[Expr] = []
        for axis in AXES:
            variable = coordinates[center][axis]
            argument_derivative = graph.differentiate(boys_argument, variable)
            leaf_derivatives = {
                "boys_0": -boys[1] * argument_derivative,
                "boys_1": -boys[2] * argument_derivative,
            }
            center_gradients.append(
                graph.differentiate(value, variable, leaf_derivatives)
            )
        independent_gradients.append(tuple(center_gradients))
    fourth_gradient = tuple(
        -graph.sum(independent_gradients[center][axis] for center in range(3))
        for axis in range(3)
    )
    gradients = tuple(independent_gradients) + (fourth_gradient,)
    return PsssKernel(
        graph=graph,
        p_axis=p_axis,
        variables=variables,
        boys_argument=boys_argument,
        value=value,
        gradients=gradients,
    )


def emit_psss_cuda(kernel: PsssKernel) -> str:
    """Emit one AOT-ready CUDA device function for the pilot shell class."""

    variable_code = {
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
        "delta": "delta",
        "kPi": "kPi",
        "boys_0": "boys[0]",
        "boys_1": "boys[1]",
        "boys_2": "boys[2]",
    }
    for center in CENTERS:
        for axis in AXES:
            variable_code[f"{center}_{axis}"] = f"{center}.{axis}"
    emitter = CudaEmitter(kernel.graph, variable_code)
    emitter.emit([kernel.boys_argument])
    argument_reference = emitter.reference(kernel.boys_argument)
    emitter.lines.append("  double boys[3];")
    emitter.lines.append(
        f"  boys_values<2>({argument_reference}, boys);"
    )
    roots = [item for center in kernel.gradients for item in center]
    emitter.emit(roots)

    lines = [
        "/** Generated symbolic/CSE derivative for canonical "
        f"(p{kernel.p_axis} s|s s). */",
        f"__device__ void generated_psss_{kernel.p_axis}_gradient(",
        "    double alpha,",
        "    const Vec3<double>& first,",
        "    double beta,",
        "    const Vec3<double>& second,",
        "    double gamma,",
        "    const Vec3<double>& third,",
        "    double delta,",
        "    const Vec3<double>& fourth,",
        "    double (&gradient)[4][3]) {",
        *emitter.lines,
    ]
    for center, center_gradients in enumerate(kernel.gradients):
        for coordinate, expression in enumerate(center_gradients):
            lines.append(
                f"  gradient[{center}][{coordinate}] = "
                f"{emitter.reference(expression)};"
            )
    lines.append("}")
    return "\n".join(lines) + "\n"

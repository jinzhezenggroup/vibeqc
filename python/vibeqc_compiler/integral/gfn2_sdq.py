"""Compiler-owned primitive GFN2 S/D/Q integral algebra.

GFN2 dipole and quadrupole moments use the ket AO atom as the operator origin.
For primitive Cartesian Gaussians this is exactly the existing overlap Hermite
DAG with the ket polynomial raised by one or two Cartesian powers. Keeping that
identity explicit lets the compiler own S/D/Q values and coordinate
derivatives without introducing a second recurrence implementation.

This module intentionally stops at one primitive Cartesian AO pair. Contracted
basis coefficients, normalization, real-spherical transforms, ragged shell
traversal, screening, and runtime publication remain backend/runtime policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .expr import Expr, Graph
from .ir import OperatorFamily
from .one_electron_values import (
    build_one_electron_component_kernel,
    build_one_electron_value_ir,
)
from .shell_spec import AXES, cartesian_components

GFN2_SDQ_COMPONENTS = (
    "overlap",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "quadrupole_xx",
    "quadrupole_xy",
    "quadrupole_yy",
    "quadrupole_xz",
    "quadrupole_yz",
    "quadrupole_zz",
)

_DIPOLE_RAISES = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)
_SECOND_MOMENT_RAISES = (
    (2, 0, 0),
    (1, 1, 0),
    (0, 2, 0),
    (1, 0, 1),
    (0, 1, 1),
    (0, 0, 2),
)


def _component_powers(component: str) -> tuple[int, int, int]:
    if not isinstance(component, str) or any(axis not in AXES for axis in component):
        raise ValueError(f"invalid Cartesian component {component!r}")
    return (
        component.count("x"),
        component.count("y"),
        component.count("z"),
    )


def _component_from_powers(powers: tuple[int, int, int]) -> str:
    return "".join(axis * power for axis, power in zip(AXES, powers, strict=True))


def _raised_component(component: str, raises: tuple[int, int, int]) -> str:
    powers = _component_powers(component)
    return _component_from_powers(
        (
            powers[0] + raises[0],
            powers[1] + raises[1],
            powers[2] + raises[2],
        )
    )


def _overlap_root(
    graph: Graph,
    angular: tuple[int, int],
    components: tuple[str, str],
    ket_raise: tuple[int, int, int] = (0, 0, 0),
) -> Expr:
    order = sum(ket_raise)
    synthetic_angular = (angular[0], angular[1] + order)
    synthetic_components = (
        components[0],
        _raised_component(components[1], ket_raise),
    )
    integral = build_one_electron_value_ir(
        OperatorFamily.OVERLAP,
        synthetic_angular,
    )
    return build_one_electron_component_kernel(
        integral,
        synthetic_components,
        graph=graph,
    ).value


@dataclass(frozen=True)
class Gfn2SdqPrimitiveKernel:
    """One primitive Cartesian AO-pair S/D/Q graph and analytic center gradients."""

    angular: tuple[int, int]
    components: tuple[str, str]
    graph: Graph
    values: tuple[Expr, ...]
    gradients: tuple[tuple[tuple[Expr, ...], ...], ...]


def build_gfn2_sdq_primitive_kernel(
    angular: tuple[int, int],
    components: tuple[str, str],
    *,
    graph: Graph | None = None,
) -> Gfn2SdqPrimitiveKernel:
    """Build primitive overlap, ket-origin dipole/quadrupole and d/d(A,B).

    Public GFN2 basis shells are restricted to s/p/d. The internal ket power
    reaches g only while representing a second moment of a d function; that
    remains within the existing one-electron Hermite DAG's supported domain.
    """

    angular = tuple(angular)
    components = tuple(components)
    if len(angular) != 2 or any(
        type(order) is not int or not 0 <= order <= 2 for order in angular
    ):
        raise ValueError("GFN2 S/D/Q requires two public s/p/d shell orders")
    if len(components) != 2 or any(
        component not in cartesian_components(order)
        for component, order in zip(components, angular, strict=True)
    ):
        raise ValueError("GFN2 S/D/Q components do not match the shell orders")

    graph = Graph() if graph is None else graph

    overlap = _overlap_root(graph, angular, components)
    dipole = tuple(
        _overlap_root(graph, angular, components, raise_) for raise_ in _DIPOLE_RAISES
    )
    second = tuple(
        _overlap_root(graph, angular, components, raise_)
        for raise_ in _SECOND_MOMENT_RAISES
    )
    trace = second[0] + second[2] + second[5]
    quadrupole = (
        1.5 * second[0] - 0.5 * trace,
        1.5 * second[1],
        1.5 * second[2] - 0.5 * trace,
        1.5 * second[3],
        1.5 * second[4],
        1.5 * second[5] - 0.5 * trace,
    )
    values = (overlap, *dipole, *quadrupole)

    gradients = []
    for center in ("a", "b"):
        center_axes = []
        for axis in AXES:
            variable = graph.variable(f"{center}_{axis}")
            center_axes.append(
                tuple(graph.differentiate(value, variable) for value in values)
            )
        gradients.append(tuple(center_axes))

    return Gfn2SdqPrimitiveKernel(
        angular,
        components,
        graph,
        values,
        tuple(gradients),
    )


@dataclass(frozen=True)
class Gfn2SdqPrimitiveEvaluation:
    values: tuple[float, ...]
    gradients: tuple[tuple[tuple[float, ...], ...], ...]


def evaluate_gfn2_sdq_primitive(
    kernel: Gfn2SdqPrimitiveKernel,
    exponents: tuple[float, float],
    centers: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> Gfn2SdqPrimitiveEvaluation:
    """Interpret one generated primitive S/D/Q graph for qualification."""

    exponent_values = (float(exponents[0]), float(exponents[1]))
    center_values = (
        (float(centers[0][0]), float(centers[0][1]), float(centers[0][2])),
        (float(centers[1][0]), float(centers[1][1]), float(centers[1][2])),
    )
    if any(not isfinite(value) or value <= 0.0 for value in exponent_values):
        raise ValueError("GFN2 S/D/Q primitive exponents must be finite and positive")

    if any(not isfinite(value) for center in center_values for value in center):
        raise ValueError("GFN2 S/D/Q primitive centers must be finite")

    variables = {"alpha": exponent_values[0], "beta": exponent_values[1]}
    for center_name, position in zip(("a", "b"), center_values, strict=True):
        variables.update(
            {
                f"{center_name}_{axis}": value
                for axis, value in zip(AXES, position, strict=True)
            }
        )

    values = tuple(
        float(kernel.graph.evaluate(root, variables)) for root in kernel.values
    )
    gradients = tuple(
        tuple(
            tuple(float(kernel.graph.evaluate(root, variables)) for root in axis_roots)
            for axis_roots in center
        )
        for center in kernel.gradients
    )
    return Gfn2SdqPrimitiveEvaluation(values, gradients)

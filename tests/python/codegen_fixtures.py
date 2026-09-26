"""Shared immutable numerical fixtures for codegen tests."""

from __future__ import annotations

import math
import typing

from vibeqc_compiler.integral.shell_class import AXES


def boys_values(argument: float, count: int = 3) -> list[float]:
    """Reference Boys sequence sufficient for the psss pilot."""

    if argument < 1.0e-8:
        return [
            sum(
                (-argument) ** term
                / (math.factorial(term) * (2 * order + 2 * term + 1))
                for term in range(18)
            )
            for order in range(count)
        ]
    root = math.sqrt(argument)
    values = [0.5 * math.sqrt(math.pi / argument) * math.erf(root)]
    exponential = math.exp(-argument)
    for order in range(count - 1):
        values.append(((2 * order + 1) * values[-1] - exponential) / (2 * argument))
    return values


def sample_variables() -> dict[str, float]:
    values = {
        "alpha": 1.3,
        "beta": 0.7,
        "gamma": 0.9,
        "delta": 0.5,
        "kPi": math.pi,
    }
    positions = {
        "first": (0.1, -0.3, 0.2),
        "second": (-0.4, 0.2, 0.5),
        "third": (0.6, -0.1, -0.2),
        "fourth": (-0.2, 0.4, -0.6),
    }
    for center, position in positions.items():
        for axis, coordinate in zip(AXES, position, strict=True):
            values[f"{center}_{axis}"] = coordinate
    return values


def factored_dppp_variables(values: dict[str, float]) -> dict[str, float]:
    """Construct the common primitive geometry consumed by factored lowering."""

    alpha = values["alpha"]
    beta = values["beta"]
    gamma = values["gamma"]
    delta = values["delta"]
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    result = {
        "inverse_two_p": 0.5 / p,
        "inverse_two_q": 0.5 / q,
        "rho": p * q / (p + q),
        "first_product_scale": alpha / p,
        "second_product_scale": beta / p,
        "third_product_scale": gamma / q,
        "fourth_product_scale": delta / q,
    }
    product_p = {}
    product_q = {}
    pair_distance_squared = 0.0
    for axis in AXES:
        first = values[f"first_{axis}"]
        second = values[f"second_{axis}"]
        third = values[f"third_{axis}"]
        fourth = values[f"fourth_{axis}"]
        product_p[axis] = (alpha * first + beta * second) / p
        product_q[axis] = (gamma * third + delta * fourth) / q
        result[f"pa_{axis}"] = product_p[axis] - first
        result[f"pb_{axis}"] = product_p[axis] - second
        result[f"qc_{axis}"] = product_q[axis] - third
        result[f"qd_{axis}"] = product_q[axis] - fourth
        result[f"difference_{axis}"] = product_p[axis] - product_q[axis]
        first_difference = first - second
        second_difference = third - fourth
        pair_distance_squared += (
            -mu * first_difference * first_difference
            - nu * second_difference * second_difference
        )
        result[f"decay_first_{axis}"] = -2.0 * mu * first_difference
        result[f"decay_second_{axis}"] = 2.0 * mu * first_difference
        result[f"decay_third_{axis}"] = -2.0 * nu * second_difference
        result[f"decay_fourth_{axis}"] = 2.0 * nu * second_difference
    result["prefactor"] = (
        2.0
        * math.pi**2.5
        / (p * q * math.sqrt(p + q))
        * math.exp(pair_distance_squared)
    )
    argument = result["rho"] * sum(result[f"difference_{axis}"] ** 2 for axis in AXES)
    # Order-eight shell quartets such as DDDD require the ninth Boys moment
    # for their first-derivative oracle.
    for order, value in enumerate(boys_values(argument, 10)):
        result[f"boys_{order}"] = value
    return result


def evaluate_value(
    kernel: typing.Any, values: dict[str, float], boys_count: int = 3
) -> float:
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument, boys_count)):
        values[f"boys_{order}"] = value
    return kernel.graph.evaluate(kernel.value, values)

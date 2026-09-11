"""Test adapter for contracted Cartesian generator checks against libcint data.

This is geometry/normalization plumbing, not an independent integral oracle:
the actual recurrence is the existing generator evaluator. Reference tensors
come from libcint, whose contraction and derivative implementation is separate.
"""

from itertools import product
from math import exp, pi, prod, sqrt

import numpy as np
from vibeqc_compiler.integral.fused_schedule import evaluate_fused_shell_observables
from vibeqc_compiler.integral.rys import boys_values
from vibeqc_compiler.integral.shell_spec import FUSED_SHELL_SPEC_BY_NAME


def _geometry(exponents, centers):
    a, b, c, d = exponents
    p, q = a + b, c + d
    mu, nu = a * b / p, c * d / q
    first, second, third, fourth = np.asarray(centers)
    P, Q = (a * first + b * second) / p, (c * third + d * fourth) / q
    values = {
        "inverse_two_p": 0.5 / p,
        "inverse_two_q": 0.5 / q,
        "rho": p * q / (p + q),
        "first_product_scale": a / p,
        "second_product_scale": b / p,
        "third_product_scale": c / q,
        "fourth_product_scale": d / q,
        "prefactor": 2
        * pi**2.5
        / (p * q * sqrt(p + q))
        * exp(-mu * sum((first - second) ** 2) - nu * sum((third - fourth) ** 2)),
    }
    vectors = {
        "pa": P - first,
        "pb": P - second,
        "qc": Q - third,
        "qd": Q - fourth,
        "difference": P - Q,
        "decay_first": -2 * mu * (first - second),
        "decay_second": 2 * mu * (first - second),
        "decay_third": -2 * nu * (third - fourth),
        "decay_fourth": 2 * nu * (third - fourth),
    }
    for prefix, vector in vectors.items():
        values.update(
            {
                f"{prefix}_{axis}": value
                for axis, value in zip("xyz", vector, strict=True)
            }
        )
    values.update(
        {
            f"boys_{i}": x
            for i, x in enumerate(boys_values(values["rho"] * sum((P - Q) ** 2), 10))
        }
    )
    return values


def _primitive_weights(shell, component):
    l = shell["angular_momentum"]
    primitives = shell["primitives"]
    contraction_norm = sum(
        ca * cb * (2 * sqrt(a * b) / (a + b)) ** (l + 1.5)
        for a, ca in primitives
        for b, cb in primitives
    )
    double_factorials = prod(
        prod(range(1, 2 * component.count(axis), 2)) for axis in "xyz"
    )
    return [
        coefficient
        * (2 * exponent / pi) ** 0.75
        * sqrt((4 * exponent) ** l / double_factorials / contraction_norm)
        for exponent, coefficient in primitives
    ]


def evaluate_quartet(inputs: dict) -> dict:
    """Evaluate all Cartesian components and moving-shell derivatives in FP64."""
    if inputs["basis_representation"] != "cartesian":
        raise ValueError("generator host adapter currently accepts Cartesian quartets")
    shells = inputs["shells"]
    name = "".join("spdf"[s["angular_momentum"]] for s in shells)
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    values, gradients = [], []
    for component in spec.components:
        weights = [
            _primitive_weights(shell, axes)
            for shell, axes in zip(shells, component, strict=True)
        ]
        value, gradient = 0.0, np.zeros((4, 3))
        for indices in product(*(range(len(s["primitives"])) for s in shells)):
            exponents = [
                s["primitives"][i][0] for s, i in zip(shells, indices, strict=True)
            ]
            weight = prod(w[i] for w, i in zip(weights, indices, strict=True))
            result = evaluate_fused_shell_observables(
                spec, component, _geometry(exponents, inputs["coordinates"])
            )
            value += weight * result.value
            gradient += weight * np.asarray(result.gradients)
        values.append(value)
        gradients.append(gradient)
    shape = tuple(
        (s["angular_momentum"] + 1) * (s["angular_momentum"] + 2) // 2 for s in shells
    )
    return {
        "eri": np.asarray(values).reshape(shape),
        "gradient": np.asarray(gradients).transpose(1, 2, 0).reshape(4, 3, *shape),
    }

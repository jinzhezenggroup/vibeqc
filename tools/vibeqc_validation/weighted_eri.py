"""Independent geometry and raw-block references for external ERI derivatives."""

from __future__ import annotations

from math import exp, pi, sqrt

import numpy as np


def primitive_variables(exponents, centers, maximum_order: int) -> dict[str, float]:
    """Bind raw and geometry-factored DAGs to the same unnormalized primitive.

    Boys moments are integrated with a separate 64-point Gauss-Legendre rule.
    This small-fixture oracle is intentionally independent of the production
    Boys evaluator and is only used at the moderate arguments in these tests.
    """
    alpha, beta, gamma, delta = map(float, exponents)
    centers = np.asarray(centers, dtype=float)
    if (
        centers.shape != (4, 3)
        or not np.isfinite(centers).all()
        or any(a <= 0 or not np.isfinite(a) for a in exponents)
    ):
        raise ValueError("four finite centers and positive exponents are required")
    p, q = alpha + beta, gamma + delta
    mu, nu = alpha * beta / p, gamma * delta / q
    rho = p * q / (p + q)
    P, Q = (
        (alpha * centers[0] + beta * centers[1]) / p,
        (gamma * centers[2] + delta * centers[3]) / q,
    )
    difference = P - Q
    argument = rho * np.dot(difference, difference)
    if argument > 100:
        raise ValueError("weighted primitive fixture quadrature requires T <= 100")
    nodes, weights = np.polynomial.legendre.leggauss(64)
    nodes = (nodes + 1) / 2
    boys = [
        np.dot(weights, nodes ** (2 * n) * np.exp(-argument * nodes**2)) / 2
        for n in range(maximum_order + 1)
    ]
    result = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "kPi": pi,
        "rho": rho,
        "inverse_two_p": 0.5 / p,
        "inverse_two_q": 0.5 / q,
        "prefactor": 2
        * pi**2.5
        / (p * q * sqrt(p + q))
        * exp(
            -mu * np.dot(centers[0] - centers[1], centers[0] - centers[1])
            - nu * np.dot(centers[2] - centers[3], centers[2] - centers[3])
        ),
    }
    for center, name in enumerate(("first", "second", "third", "fourth")):
        result[f"{name}_product_scale"] = exponents[center] / (p if center < 2 else q)
        separation = centers[0] - centers[1] if center < 2 else centers[2] - centers[3]
        decay = (-2 if center % 2 == 0 else 2) * (mu if center < 2 else nu) * separation
        for axis, label in enumerate("xyz"):
            result[f"{name}_{label}"] = centers[center, axis]
            result[f"decay_{name}_{label}"] = decay[axis]
    for prefix, shift in zip(
        ("pa", "pb", "qc", "qd"),
        (P - centers[0], P - centers[1], Q - centers[2], Q - centers[3]),
    ):
        result.update(
            {f"{prefix}_{label}": shift[axis] for axis, label in enumerate("xyz")}
        )
    result.update(
        {f"difference_{label}": difference[axis] for axis, label in enumerate("xyz")}
    )
    result.update({f"boys_{n}": value for n, value in enumerate(boys)})
    return {key: float(value) for key, value in result.items()}

"""Shared scalar graph of compact AO-pair XC bilinears.

This build-time owner contains no array/runtime dependencies. Potential schedules
and AO-jet pullbacks derive from the same rho/gradient/kinetic expression.
"""

from typing import Any

from vibeqc_compiler.integral.expr import Graph


def ao_pair_bilinear(family: str) -> tuple[Any, ...]:
    """Return graph, AO legs, coefficients and their unweighted bilinear.

    Kinetic coefficients already include the tau one-half from the feature
    pullback. Consumers must apply quadrature weights exactly once.
    """
    if family not in ("lda", "gga", "mgga"):
        raise ValueError("AO bilinears support LDA/GGA/meta-GGA")
    graph = Graph()
    jets = 1 if family == "lda" else 4
    x = [graph.variable(f"x{j}") for j in range(jets)]
    y = [graph.variable(f"y{j}") for j in range(jets)]
    c = [graph.variable(f"c{j}") for j in range(jets + (family == "mgga"))]
    bilinear = c[0] * x[0] * y[0]
    for j in range(1, jets):
        bilinear += c[j] * (x[j] * y[0] + x[0] * y[j])
    if family == "mgga":
        for j in range(1, 4):
            bilinear += c[4] * x[j] * y[j]
    return graph, x, y, c, bilinear

"""Even-power lowering must retain bounded ratios while cancelling square roots."""

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc.libxc_maple import import_maple_source


@pytest.mark.parametrize("power", (2, 4))
@pytest.mark.parametrize("scale", (1.0, 1e-56))
@pytest.mark.parametrize("ratio", (1.0, 0.6))
def test_even_power_keeps_dimensionless_ratio_derivatives_finite(
    power: int, scale: float, ratio: float
) -> None:
    graph = Graph()
    a, b = graph.variable("a"), graph.variable("b")
    module = import_maple_source(f"f := (a,b) -> ((a-b)/(a+b))^{power}:")
    energy = module.call(graph, "f", a, b)
    first = graph.differentiate(energy, a)
    second = graph.differentiate(first, a)
    av, bv = scale, scale * ratio
    z = (av - bv) / (av + bv)
    dz = (2 * bv / (av + bv)) / (av + bv)
    ddz = (-4 * bv / (av + bv)) / (av + bv) / (av + bv)
    expected = (
        z**power,
        power * z ** (power - 1) * dz,
        power * (power - 1) * z ** (power - 2) * dz * dz
        + power * z ** (power - 1) * ddz,
    )
    result = evaluate_array_graph(graph, (energy, first, second), {"a": av, "b": bv})
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=0.0)


def test_even_power_still_cancels_gradient_square_root_at_zero() -> None:
    graph = Graph()
    sigma = graph.variable("sigma")
    module = import_maple_source("f := x -> (3*sqrt(x))^2:")
    energy = module.call(graph, "f", sigma)
    first = graph.differentiate(energy, sigma)
    second = graph.differentiate(first, sigma)
    result = evaluate_array_graph(graph, (energy, first, second), {"sigma": 0.0})
    np.testing.assert_array_equal(result, (0.0, 9.0, 0.0))

"""Admitted inverse-transcendental names reach the canonical scalar graph."""

import numpy as np
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc.libxc_maple import import_maple_source


def test_new_intrinsic_binding_preserves_values_and_derivatives() -> None:
    module = import_maple_source("f := x -> arcsinh(x):")
    graph = Graph()
    x = graph.variable("x")
    value = module.call(graph, "f", x)
    first = graph.differentiate(value, x)
    second = graph.differentiate(first, x)
    points = np.array([-20.0, -0.1, 0.0, 0.1, 20.0])
    actual = evaluate_array_graph(graph, (value, first, second), {"x": points})
    expected = (
        np.arcsinh(points),
        1 / np.sqrt(1 + points**2),
        -points / (1 + points**2) ** 1.5,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)


def test_new_intrinsic_name_still_allows_lexical_parameter_shadowing() -> None:
    module = import_maple_source(
        "h := y -> 2*y: f := (arcsinh,x) -> arcsinh(x): g := x -> f(h,x):"
    )
    graph = Graph()
    x = graph.variable("x")
    value = module.call(graph, "g", x)
    assert evaluate_array_graph(graph, (value,), {"x": 3.0})[0] == 6.0

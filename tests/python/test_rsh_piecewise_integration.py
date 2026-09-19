"""RSH scalar operations and MethodIR survive shared meta-GGA/VV10 integration."""

import math
from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.method import (
    METHOD_CATALOG,
    VV10,
    original_nonlocal_correlation,
    resolve_method,
)


@pytest.mark.parametrize("operation", ["atan", "asinh", "erf"])
def test_rsh_unary_works_inside_lazy_piecewise_and_iterative_graph(operation):
    graph = Graph()
    x = graph.variable("x")
    unary = graph.transcendental_unary(operation, x)
    root = graph.select_le(x, 0, unary, 1 / x)
    samples = np.array([-0.5, 0.0, 2.0])
    expected = np.array([getattr(math, operation)(-0.5), 0.0, 0.5])
    assert graph.evaluate(unary, {"x": -0.5}) == pytest.approx(expected[0])
    np.testing.assert_allclose(
        [graph.evaluate(root, {"x": v}) for v in samples], expected
    )
    np.testing.assert_allclose(
        evaluate_array_graph(graph, (root,), {"x": samples})[0], expected
    )


@pytest.mark.parametrize("with_dispersion", [False, True])
def test_range_exchange_and_nonlocal_correlation_are_canonically_composable(
    with_dispersion,
):
    from vibeqc_compiler.method import r2scan3c_d4_eeq

    spec = replace(
        METHOD_CATALOG["CAM-B3LYP"],
        dispersion=r2scan3c_d4_eeq() if with_dispersion else None,
        nonlocal_correlation=original_nonlocal_correlation(VV10),
    )
    combined = resolve_method(spec)
    assert combined.requirements["operators"] == (
        "semilocal-xc",
        "short-range-exchange",
        "long-range-exchange",
        "nonlocal-correlation",
    ) + (("geometry-d4-bj-eeq",) if with_dispersion else ())
    assert (
        combined.identity
        != resolve_method(replace(spec, nonlocal_correlation=None)).identity
    )

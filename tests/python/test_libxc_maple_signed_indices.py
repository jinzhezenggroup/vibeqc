"""Bound variables and intrinsic constants must not silently change mathematics."""

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc.libxc_maple import MapleImportError, import_maple_source


@pytest.mark.parametrize("body,expected", (("i^2", 2), ("i^4", 2), ("2^i", 3.5)))
def test_signed_bound_index_preserves_expression_precedence(
    body: str, expected: float
) -> None:
    module = import_maple_source(f"f := x -> add({body},i=-1..1)*x^2:")
    graph = Graph()
    x = graph.variable("x")
    value = module.call(graph, "f", x)
    first = graph.differentiate(value, x)
    second = graph.differentiate(first, x)
    actual = evaluate_array_graph(graph, (value, first, second), {"x": 2.0})
    np.testing.assert_allclose(
        actual, (4 * expected, 4 * expected, 2 * expected), rtol=0, atol=0
    )


@pytest.mark.parametrize("name", ("K_FACTOR_C", "MU_GE", "DBL_EPSILON"))
@pytest.mark.parametrize("binding", (False, True))
def test_intrinsic_constant_cannot_be_silently_overridden(
    name: str, binding: bool
) -> None:
    with pytest.raises(MapleImportError, match="reserved|binding name"):
        if binding:
            import_maple_source(f"f := x -> {name}*x:", bindings={name: 2})
        else:
            import_maple_source(f"{name} := 2: f := x -> {name}*x:")


@pytest.mark.parametrize("name", ("K_FACTOR_C", "MU_GE", "DBL_EPSILON"))
def test_intrinsic_constant_name_remains_legal_as_local_parameter(name: str) -> None:
    module = import_maple_source(f"f := ({name},x) -> {name}*x: g := x -> f(2,x):")
    graph = Graph()
    x = graph.variable("x")
    value = module.call(graph, "g", x)
    assert evaluate_array_graph(graph, (value,), {"x": 3.0})[0] == 6

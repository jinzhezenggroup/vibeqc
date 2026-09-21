"""Regression tests for the family-agnostic Libxc Maple importer v8 core."""

import math

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc import libxc_maple
from vibeqc_compiler.xc.libxc_maple import (
    IMPORTER_SEMANTICS,
    MapleImportError,
    import_maple_source,
)


def _value(source: str, name: str = "f", x: float = 2.0) -> float:
    module = import_maple_source(source)
    graph = Graph()
    variable = graph.variable("x")
    root = module.call(graph, name, variable)
    return float(evaluate_array_graph(graph, (root,), {"x": x})[0])


def test_v8_importer_identity_is_family_agnostic() -> None:
    assert IMPORTER_SEMANTICS == "libxc-maple-graph/v8"


@pytest.mark.parametrize("whitespace", ("", " ", "\t", " \t "))
def test_v8_admits_only_end_of_line_backslash_continuation(whitespace: str) -> None:
    source = "f := x -> x + " + "\\" + whitespace + "\n  1:"
    assert _value(source, x=3.0) == 4.0

    with pytest.raises(MapleImportError, match="unsupported Maple expression"):
        import_maple_source(r"f := x -> x \ 1:")


def test_v8_x_factor_c_is_reserved_and_matches_libxc_constant() -> None:
    expected = 3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
    assert _value("f := x -> X_FACTOR_C*x:", x=2.0) == pytest.approx(2 * expected)

    with pytest.raises(MapleImportError, match="reserved"):
        import_maple_source("X_FACTOR_C := 1: f := x -> X_FACTOR_C*x:")

    module = import_maple_source(
        "f := (X_FACTOR_C,x) -> X_FACTOR_C*x: g := x -> f(2,x):"
    )
    graph = Graph()
    x = graph.variable("x")
    assert graph.evaluate(module.call(graph, "g", x), {"x": 3.0}) == 6.0


@pytest.mark.parametrize(
    ("maple_name", "operation"),
    (("arcsinh", "asinh"), ("arctan", "atan")),
)
def test_v8_inverse_transcendentals_preserve_values_and_derivatives(
    maple_name: str, operation: str
) -> None:
    module = import_maple_source(f"f := x -> {maple_name}(x):")
    graph = Graph()
    x = graph.variable("x")
    value = module.call(graph, "f", x)
    first = graph.differentiate(value, x)
    second = graph.differentiate(first, x)
    points = np.array([-3.0, -0.1, 0.0, 0.2, 4.0])
    actual = np.asarray(
        evaluate_array_graph(graph, (value, first, second), {"x": points})
    )

    if maple_name == "arcsinh":
        expected = np.asarray(
            (
                np.arcsinh(points),
                1 / np.sqrt(1 + points**2),
                -points / (1 + points**2) ** 1.5,
            )
        )
    else:
        expected = np.asarray(
            (
                np.arctan(points),
                1 / (1 + points**2),
                -2 * points / (1 + points**2) ** 2,
            )
        )

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)
    operations = {
        graph.nodes[index].operation for index in graph.topological_order((value,))
    }
    assert operation in operations


@pytest.mark.parametrize("name", ("arcsinh", "arctan"))
def test_v8_inverse_transcendentals_allow_lexical_shadowing(name: str) -> None:
    module = import_maple_source(
        f"h := y -> 2*y: f := ({name},x) -> {name}(x): g := x -> f(h,x):"
    )
    graph = Graph()
    x = graph.variable("x")
    assert graph.evaluate(module.call(graph, "g", x), {"x": 3.0}) == 6.0


def test_importer_semantics_participate_in_transitive_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_maple_source("f := x -> x:")
    original = module.transitive_sha256
    monkeypatch.setattr(
        libxc_maple, "IMPORTER_SEMANTICS", IMPORTER_SEMANTICS + "/identity-probe"
    )
    assert module.transitive_sha256 != original

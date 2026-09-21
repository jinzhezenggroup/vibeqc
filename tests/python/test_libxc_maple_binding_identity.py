"""Preprocessing identity and stable rewrites preserve selected Maple semantics."""

from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc.libxc_maple import (
    MapleImportError,
    MapleModule,
    import_maple_file,
    import_maple_source,
)


def _value(module: MapleModule, name: str, x: float) -> float:
    graph = Graph()
    variable = graph.variable("x")
    expression = module.call(graph, name, variable)
    return float(evaluate_array_graph(graph, (expression,), {"x": x})[0])


@pytest.mark.parametrize("mode", ("memory", "file", "include"))
def test_late_define_does_not_alias_different_selected_branches(
    tmp_path: Path, mode: str
) -> None:
    source = "$ifdef MODE\nf := x -> x:\n$else\nf := x -> 2*x:\n$endif\n$define MODE\n"
    if mode == "include":
        (tmp_path / "branch.mpl").write_text(source, encoding="utf-8")
        source = '$include "branch.mpl"\n'
    (tmp_path / "entry.mpl").write_text(source, encoding="utf-8")
    modules = [
        import_maple_source(source, defines=defines)
        if mode == "memory"
        else import_maple_file(tmp_path, "entry.mpl", defines=defines)
        for defines in ((), ("MODE",))
    ]
    first, second = modules
    assert first.defines == second.defines == ("MODE",)
    assert first.source_hashes == second.source_hashes
    assert first.include_edges == second.include_edges
    assert _value(first, "f", 3.0) == 6.0
    assert _value(second, "f", 3.0) == 3.0
    assert first.transitive_sha256 != second.transitive_sha256


def test_initial_define_order_and_checkout_path_do_not_change_identity(
    tmp_path: Path,
) -> None:
    modules = []
    for name, defines in (("first", ("A", "B")), ("second", ("B", "A", "A"))):
        root = tmp_path / name
        root.mkdir()
        (root / "entry.mpl").write_text("f := x -> 2*x:", encoding="utf-8")
        modules.append(import_maple_file(root, "entry.mpl", defines=iter(defines)))
    assert modules[0].transitive_sha256 == modules[1].transitive_sha256


@pytest.mark.parametrize(
    "name,body,expected",
    (("log", "log(1+x)", 6.0), ("log", "log(x+1)", 6.0), ("exp", "exp(x)-1", 3.0)),
)
def test_stable_rewrite_does_not_capture_a_function_parameter(
    name: str, body: str, expected: float
) -> None:
    module = import_maple_source(
        f"h := y -> 2*y: f := ({name}, x) -> {body}: g := x -> f(h,x):"
    )
    graph = Graph()
    x = graph.variable("x")
    energy = module.call(graph, "g", x)
    first = graph.differentiate(energy, x)
    second = graph.differentiate(first, x)
    actual = evaluate_array_graph(graph, (energy, first, second), {"x": 2.0})
    np.testing.assert_array_equal(actual, (expected, 2.0, 0.0))


@pytest.mark.parametrize("name,body", (("log", "log(1+x)"), ("exp", "exp(x)-1")))
def test_stable_rewrite_does_not_make_a_scalar_parameter_callable(
    name: str, body: str
) -> None:
    module = import_maple_source(f"f := ({name}, x) -> {body}: g := x -> f(3,x):")
    with pytest.raises(MapleImportError, match="not callable"):
        _value(module, "g", 2.0)


@pytest.mark.parametrize(
    "body,operation",
    (("log(1+x)", "log1p"), ("log(x+1)", "log1p"), ("exp(x)-1", "expm1")),
)
def test_unshadowed_intrinsics_keep_cancellation_safe_lowering(
    body: str, operation: str
) -> None:
    module = import_maple_source(f"f := x -> {body}:")
    graph = Graph()
    x = graph.variable("x")
    energy = module.call(graph, "f", x)
    assert operation in {
        graph.nodes[i].operation for i in graph.topological_order((energy,))
    }
    actual = evaluate_array_graph(graph, (energy,), {"x": 1e-18})[0]
    assert actual == getattr(np, operation)(1e-18)

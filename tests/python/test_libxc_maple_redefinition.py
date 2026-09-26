"""File-mode overrides cannot rewrite values already captured by assignments."""

from pathlib import Path

import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc.libxc_maple import MapleImportError, import_maple_file


@pytest.mark.parametrize(
    "source",
    (
        "a := 2: captured := a: a := 3: f := x -> captured*x:",
        "a := 2: captured := [a, 4]: a := 3: f := x -> captured[1]*x:",
        "a := 2: b := a: captured := b: a := 3: f := x -> captured*x:",
        "a := 2: h := y -> a*y: captured := h(1): a := 3: f := x -> captured*x:",
        "h := y -> 2*y: captured := h(1): h := y -> 3*y: f := x -> captured*x:",
        "h := y -> 2*y: captured := h: h := y -> 3*y: f := x -> captured(x):",
    ),
)
def test_redefinition_rejects_changed_captured_assignment(
    tmp_path: Path, source: str
) -> None:
    (tmp_path / "entry.mpl").write_text(source, encoding="utf-8")
    with pytest.raises(MapleImportError, match="captured assignment"):
        import_maple_file(tmp_path, "entry.mpl")


@pytest.mark.parametrize(
    "source,expected",
    (
        ("a := 2: a := 3: f := x -> a*x:", 6.0),
        ("a := [2, 4]: a := [3, 4]: f := x -> a[1]*x:", 6.0),
        ("f := x -> 2*x: f := x -> 3*x:", 6.0),
        ("a := 2: f := x -> a*x: a := 3:", 6.0),
        (
            (
                "a := 2: h := (a,y) -> a*y: captured := h(3,2): a := 4: "
                "f := x -> captured*x:"
            ),
            12.0,
        ),
    ),
)
def test_independent_pinned_overrides_and_parameter_binding_remain_supported(
    tmp_path: Path, source: str, expected: float
) -> None:
    (tmp_path / "entry.mpl").write_text(source, encoding="utf-8")
    module = import_maple_file(tmp_path, "entry.mpl")
    graph = Graph()
    x = graph.variable("x")
    value = evaluate_array_graph(graph, (module.call(graph, "f", x),), {"x": 2.0})
    assert value == (expected,)

"""Built-in constants and bounded differentiation never silently rebind."""

from pathlib import Path

import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc.libxc_maple import (
    MapleImportError,
    import_maple_file,
    import_maple_source,
)


@pytest.mark.parametrize("name", ("MU_GE", "K_FACTOR_C", "DBL_EPSILON"))
@pytest.mark.parametrize("mode", ("source", "file", "binding"))
def test_meta_constants_reject_global_redefinition(
    tmp_path: Path, name: str, mode: str
) -> None:
    source = f"{name} := 2: f := x -> {name}*x:"
    (tmp_path / "entry.mpl").write_text(source, encoding="utf-8")
    message = "binding name" if mode == "binding" else "reserved"
    with pytest.raises(MapleImportError, match=message):
        if mode == "source":
            import_maple_source(source)
        elif mode == "file":
            import_maple_file(tmp_path, "entry.mpl")
        else:
            import_maple_source(f"f := x -> {name}*x:", bindings={name: 2})


@pytest.mark.parametrize("name", ("MU_GE", "K_FACTOR_C", "DBL_EPSILON"))
def test_meta_constants_still_allow_lexical_parameters(name: str) -> None:
    module = import_maple_source(f"f := ({name}, x) -> {name}*x: g := x -> f(2,x):")
    graph = Graph()
    x = graph.variable("x")
    root = module.call(graph, "g", x)
    first = graph.differentiate(root, x)
    assert evaluate_array_graph(graph, (root, first), {"x": 3.0}) == (6.0, 2.0)


def test_bounded_diff_rejects_aliased_placeholders() -> None:
    with pytest.raises(MapleImportError, match="distinct"):
        import_maple_source(
            "g := (a,b) -> a*b: f := x -> eval(diff(g(t,t),t), [t=x,t=x]):"
        )


@pytest.mark.parametrize("derivative", ("a", "b"))
def test_bounded_diff_preserves_distinct_placeholders(derivative: str) -> None:
    module = import_maple_source(
        f"g := (a,b) -> a*b: f := x -> eval(diff(g(a,b),{derivative}), [a=x,b=x]):"
    )
    graph = Graph()
    x = graph.variable("x")
    root = module.call(graph, "f", x)
    first = graph.differentiate(root, x)
    assert evaluate_array_graph(graph, (root, first), {"x": 3.0}) == (3.0, 1.0)

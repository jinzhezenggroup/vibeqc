"""Maple comments must not alter the lowered expression or hide definitions."""

import hashlib

import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc.libxc_maple import MapleImportError, import_maple_source


@pytest.mark.parametrize(
    "source",
    (
        "f := x -> x # inline comment\n + 1:",
        "# heading\nf := x -> x + 1:",
        "f := x -> x + 1: # trailing comment with : and :=\n",
        "f := x -> x # (* does not start a block\n + 1:",
        "f := x -> x (* # is not a line comment *) + 1:",
        "f := x -> x (* outer (* nested *) outer *) + 1:",
        "f := x -> x + # comment with (* and an unmatched quote '\n 1:",
        "f := x -> x # continued comment \\\n + 99\n + 1:",
        "f := x -> x # escaped backslash \\\\\n + 1:",
    ),
)
def test_maple_comments_preserve_arithmetic_and_raw_source_hash(source: str) -> None:
    module = import_maple_source(source)
    graph = Graph()
    value = module.call(graph, "f", graph.variable("x"))
    assert evaluate_array_graph(graph, (value,), {"x": 3.0}) == (4.0,)
    assert module.source_sha256 == hashlib.sha256(source.encode()).hexdigest()


@pytest.mark.parametrize("source", ("f := x -> x: (* open", "f := x -> x: *)"))
def test_maple_rejects_unbalanced_block_comments(source: str) -> None:
    with pytest.raises(MapleImportError, match="comment"):
        import_maple_source(source)


@pytest.mark.parametrize("name", ("Pi", "X2S", "sqrt", "exp", "gga_exchange"))
def test_maple_rejects_shadowed_compiler_intrinsic(name: str) -> None:
    with pytest.raises(MapleImportError, match="reserved"):
        import_maple_source(f"{name} := 2: f := x -> {name}*x:")

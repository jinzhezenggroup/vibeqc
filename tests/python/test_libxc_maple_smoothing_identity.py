"""The one admitted smooth-LR expansion must bind its actual source closure."""

from pathlib import Path

import pytest
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.xc.libxc_maple import MapleImportError, import_maple_source

SOURCE = Path(__file__).resolve().parents[2] / "external/libxc-7.0.0/attenuation.mpl"


@pytest.mark.parametrize(
    "old,new",
    (
        ("1 - 8/3*a*", "2 - 8/3*a*"),
        ("sqrt(Pi)*erf(1/(2*a))", "2*sqrt(Pi)*erf(1/(2*a))"),
        ("exp(-1/(4*a^2)) - 1", "exp(-1/(4*a^2)) - 2"),
        ("2*a^2*att_erf_aux2(a) + 1/2", "2*a^2*att_erf_aux2(a) + 1/3"),
    ),
)
def test_smooth_lr_rejects_changed_helper_or_dependency(old: str, new: str) -> None:
    source = SOURCE.read_text()
    assert old in source
    module = import_maple_source(source.replace(old, new, 1))
    graph = Graph()
    with pytest.raises(MapleImportError, match="source closure"):
        module.call(graph, "attenuation_erf", graph.constant(2))


def test_smooth_lr_rejects_same_named_unrelated_formula() -> None:
    module = import_maple_source(
        "attenuation_erf0 := a -> 2*a: "
        "f := a -> enforce_smooth_lr(attenuation_erf0,a,1.35,16):"
    )
    graph = Graph()
    with pytest.raises(MapleImportError, match="source closure"):
        module.call(graph, "f", graph.constant(2))


def test_smooth_lr_accepts_whitespace_only_source_changes() -> None:
    source = SOURCE.read_text()
    original = import_maple_source(source)
    formatted = import_maple_source(source.replace("2*a", "2 * a"))
    for point in (0.4, 1.35, 4.0):
        graph = Graph()
        expected = original.call(graph, "attenuation_erf", graph.constant(point))
        actual = formatted.call(graph, "attenuation_erf", graph.constant(point))
        assert graph.evaluate(actual, {}) == graph.evaluate(expected, {})

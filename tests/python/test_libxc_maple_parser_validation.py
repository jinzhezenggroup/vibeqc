"""Malformed Maple must not silently select another function or branch."""

import pytest
from vibeqc_compiler.xc.libxc_maple import MapleImportError, import_maple_source


@pytest.mark.parametrize("parameters", ("x,x", "x,,y", ",x", "x,"))
def test_maple_rejects_duplicate_or_empty_parameters(parameters: str) -> None:
    with pytest.raises(MapleImportError, match="parameter list"):
        import_maple_source(f"f := ({parameters}) -> x:")


@pytest.mark.parametrize("directive", ("$else", "$elif other"))
@pytest.mark.parametrize("defines", ((), ("enabled",)))
def test_maple_rejects_branches_after_else(
    directive: str, defines: tuple[str, ...]
) -> None:
    source = (
        "$ifdef enabled\n"
        "f := x -> x:\n"
        "$else\n"
        "f := x -> 2*x:\n"
        f"{directive}\n"
        "f := x -> 3*x:\n"
        "$endif\n"
    )
    with pytest.raises(MapleImportError, match="after \\$else"):
        import_maple_source(source, defines=defines)


@pytest.mark.parametrize(
    "defines,expression",
    (
        ((), "4*x"),
        (("outer",), "3*x"),
        (("outer", "inner"), "x"),
        (("outer", "other"), "2*x"),
        (("inner", "other"), "4*x"),
        (("outer", "inner", "other"), "x"),
    ),
)
def test_maple_nested_conditionals_preserve_parent_activity(
    defines: tuple[str, ...], expression: str
) -> None:
    module = import_maple_source(
        "$ifdef outer\n"
        "$ifdef inner\n"
        "f := x -> x:\n"
        "$elif other\n"
        "f := x -> 2*x:\n"
        "$else\n"
        "f := x -> 3*x:\n"
        "$endif\n"
        "$else\n"
        "f := x -> 4*x:\n"
        "$endif\n",
        defines=defines,
    )
    assert module.function_names == ("f",)
    assert dict(module.functions)["f"].expression == expression

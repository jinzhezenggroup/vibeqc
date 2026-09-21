"""Ordinary import syntax and nested packages must not bypass retirement."""

from pathlib import Path

import pytest

from tools.vibeqc_validation.xc_retirement import errors


@pytest.mark.parametrize(
    "source",
    [
        "from vibeqc_compiler.xc import expressions\n",
        "from vibeqc_compiler.xc import rsh_expressions as legacy\n",
        "from . import wb97mv_expressions\n",
        "from importlib import import_module\nimport_module('.expressions', 'vibeqc_compiler.xc')\n",
    ],
)
def test_common_import_forms_cannot_report_zero_consumers(
    tmp_path: Path, source: str
) -> None:
    path = tmp_path / "python/vibeqc_compiler/xc/new_consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    assert any("new legacy XC consumer" in error for error in errors(tmp_path))
    assert any(
        "legacy XC consumer remains" in error
        for error in errors(tmp_path, require_no_consumers=True)
    )


@pytest.mark.parametrize("name", ["expressions.py", "new_expressions.py"])
def test_nested_formula_modules_cannot_escape_inventory(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / "python/vibeqc_compiler/xc/nested" / name
    path.parent.mkdir(parents=True)
    path.write_text("pass\n")
    assert any("expression module" in error for error in errors(tmp_path))


def test_unrelated_relative_path_calls_are_not_imports(tmp_path: Path) -> None:
    source = tmp_path / "tools/loader.py"
    source.parent.mkdir(parents=True)
    source.write_text('load("../libvibeqc.so", "invalid")\n')
    assert errors(tmp_path) == []

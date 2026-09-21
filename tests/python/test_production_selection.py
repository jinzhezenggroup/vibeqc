"""Production selection policy stays separate from emission orchestration."""

from __future__ import annotations

import ast
from pathlib import Path

from vibeqc_compiler.integral import production, production_selection


def test_production_preserves_selection_compatibility_reexports() -> None:
    """Legacy production imports resolve to the canonical selection objects."""

    assert production.KernelSelection is production_selection.KernelSelection
    assert production._selection_integral is production_selection._selection_integral
    assert (
        production._SUPPORTED_RECURRENCES is production_selection._SUPPORTED_RECURRENCES
    )


def test_production_selection_has_leaf_dependencies() -> None:
    """Selection policy must not depend on production emission or bundle owners."""

    source = Path(production_selection.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    relative_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "production" not in relative_modules
    assert "production_cost" not in relative_modules
    assert "production_bundle" not in relative_modules


def test_production_selection_keeps_scientific_validation_local() -> None:
    """KernelSelection remains owned by the selection module, not the facade."""

    assert production_selection.KernelSelection.__module__.endswith(
        ".production_selection"
    )

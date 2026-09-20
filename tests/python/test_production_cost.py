"""Production compile-cost policy stays a leaf of orchestration."""

from __future__ import annotations

import ast
from pathlib import Path

from vibeqc_compiler.integral import production, production_cost


def test_production_preserves_cost_policy_compatibility_reexports() -> None:
    """Legacy production imports resolve to the canonical leaf policy objects."""

    assert production.production_compile_cost is production_cost.production_compile_cost
    assert production.stable_aot_shard_slot is production_cost.stable_aot_shard_slot
    assert (
        production._partition_production_selections
        is production_cost._partition_production_selections
    )
    assert production.shell_class_index is production_cost.shell_class_index


def test_production_cost_policy_has_leaf_dependencies() -> None:
    """Cost policy must not import production emission, benchmarks, or CLI code."""

    source = Path(production_cost.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    relative_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    absolute_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert relative_modules == {"shell_spec"}
    assert absolute_roots.isdisjoint({"benchmarks", "tools", "vibeqc"})

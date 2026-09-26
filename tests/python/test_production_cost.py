"""Production compile-cost policy stays a leaf of orchestration."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from vibeqc_compiler.integral import production, production_cost
from vibeqc_compiler.integral.shell_spec import FUSED_SHELL_SPEC_BY_NAME


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


def test_stable_aot_slots_preserve_eight_shard_identity_and_expand() -> None:
    """Larger shard counts must expose parallel buckets without moving default shards."""

    selections = [
        SimpleNamespace(spec=FUSED_SHELL_SPEC_BY_NAME[name])
        for name in production_cost._STABLE_AOT_SHARD_SLOTS
    ]
    for selection in selections:
        expected = production_cost._STABLE_AOT_SHARD_SLOTS[selection.spec.name]
        assert production_cost.stable_aot_shard_slot(selection) % 8 == expected

    for shard_count in (12, 16):
        shards = production_cost._partition_production_selections(
            selections, shard_count
        )
        used = {index for index, shard in enumerate(shards) if shard}
        assert len(used) > 8
        assert any(index >= 8 for index in used)

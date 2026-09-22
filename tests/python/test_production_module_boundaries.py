"""Ownership guards for the issue #487 production-module split."""

from __future__ import annotations

import ast
from pathlib import Path

from vibeqc_compiler.integral import production, production_bundle, production_emission

ROOT = Path(__file__).resolve().parents[2]
INTEGRAL = ROOT / "python" / "vibeqc_compiler" / "integral"


def _imports(module: str) -> set[str]:
    tree = ast.parse((INTEGRAL / f"{module}.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


def test_compatibility_facade_preserves_callable_identity() -> None:
    assert production.emit_production_shard is production_emission.emit_production_shard
    assert production.emit_profile_shard is production_emission.emit_profile_shard
    assert (
        production.write_production_bundle is production_bundle.write_production_bundle
    )
    assert (
        production.write_production_bundles
        is production_bundle.write_production_bundles
    )


def test_bundle_owner_does_not_import_cuda_emitters() -> None:
    imports = _imports("production_bundle")
    assert not any(
        "cuda_emitter" in name or "cuda_lowering" in name for name in imports
    )
    assert not any("benchmark" in name or "cli" in name for name in imports)


def test_emission_owner_does_not_import_bundle_or_filesystem_orchestration() -> None:
    imports = _imports("production_emission")
    assert not any("production_bundle" in name for name in imports)
    assert not any("benchmark" in name or "cli" in name for name in imports)
    source = (INTEGRAL / "production_emission.py").read_text(encoding="utf-8")
    assert "mkdir(" not in source
    assert "write_text(" not in source


def test_facade_is_narrow_and_contains_no_generation_implementation() -> None:
    source = (INTEGRAL / "production.py").read_text(encoding="utf-8")
    assert len(source.splitlines()) < 100
    assert "def emit_production_shard" not in source
    assert "def write_production_bundle" not in source

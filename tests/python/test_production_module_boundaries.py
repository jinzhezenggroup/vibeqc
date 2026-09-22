"""Ownership guards for the issue #487 production-module split."""

from __future__ import annotations

import ast
from graphlib import TopologicalSorter
from pathlib import Path

from vibeqc_compiler.integral import (
    production,
    production_bundle,
    production_emission,
    production_selection,
)

ROOT = Path(__file__).resolve().parents[2]
INTEGRAL = ROOT / "python" / "vibeqc_compiler" / "integral"
PRODUCTION_MODULES = {
    "production",
    "production_bundle",
    "production_cost",
    "production_emission",
    "production_profile",
    "production_registry",
    "production_selection",
}


def _imports(module: str) -> set[str]:
    tree = ast.parse((INTEGRAL / f"{module}.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
            else:
                imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


def _calls(module: str) -> set[str]:
    tree = ast.parse((INTEGRAL / f"{module}.py").read_text(encoding="utf-8"))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def _production_imports(module: str) -> set[str]:
    """Return direct imports within the production ownership graph."""
    tree = ast.parse((INTEGRAL / f"{module}.py").read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        candidates = (
            (node.module,) if node.module else tuple(alias.name for alias in node.names)
        )
        imports.update(
            candidate for candidate in candidates if candidate in PRODUCTION_MODULES
        )
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
    assert production._selection_integral is production_selection._selection_integral
    emission_aliases = (
        "CAPABILITY_LOCAL_PACKED_STREAMING_FOCK",
        "CAPABILITY_MIXED_FOCK",
        "CAPABILITY_STREAMING_FOCK",
        "GeneratedKernelArgument",
        "GeneratedKernelSignature",
        "KernelConsumer",
        "ScheduleIR",
        "ScheduleKind",
        "ShellClassSpec",
        "TYPE_CHECKING",
        "build_fused_shell_plan",
        "cuda_target_info",
        "emit_ppps_resident_bra_rys3_cuda",
        "emit_shell_class_fused_cuda",
        "re",
        "shell_pair_class",
    )
    for name in emission_aliases:
        assert getattr(production, name) is getattr(production_emission, name)
    for name in ("FUSED_SHELL_SPEC_BY_NAME", "normalize_cuda_architecture"):
        assert getattr(production, name) is getattr(production_bundle, name)


def test_production_ownership_graph_is_cycle_free() -> None:
    graph = {module: _production_imports(module) for module in PRODUCTION_MODULES}
    for module in PRODUCTION_MODULES - {"production"}:
        assert "production" not in graph[module]
    TopologicalSorter(graph).prepare()


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
    assert not imports & {"os", "pathlib", "shutil", "tempfile"}
    assert not _calls("production_emission") & {
        "makedirs",
        "mkdir",
        "open",
        "rename",
        "unlink",
        "write_bytes",
        "write_text",
    }


def test_facade_is_narrow_and_contains_no_generation_implementation() -> None:
    source = (INTEGRAL / "production.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert len(source.splitlines()) < 110
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body
    )
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
    assert assignments
    for assignment in assignments:
        assert isinstance(assignment.value, ast.Attribute)
        assert isinstance(assignment.value.value, ast.Name)
        assert assignment.value.value.id.startswith("_production_")

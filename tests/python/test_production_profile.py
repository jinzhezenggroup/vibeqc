"""Production manifest/profile ownership stays separate from emission orchestration."""

from __future__ import annotations

import ast
from pathlib import Path

from vibeqc_compiler.integral import production, production_profile


def test_production_preserves_profile_compatibility_reexports() -> None:
    """Legacy production imports resolve to the canonical profile objects."""

    assert production.ProfileMatch is production_profile.ProfileMatch
    assert (
        production._schedule_from_payload is production_profile._schedule_from_payload
    )
    assert (
        production.ResolvedProductionProfile
        is production_profile.ResolvedProductionProfile
    )
    assert (
        production.resolve_production_profile
        is production_profile.resolve_production_profile
    )
    assert (
        production.load_production_kernel_selections
        is production_profile.load_production_kernel_selections
    )
    assert (
        production.load_production_manifest
        is production_profile.load_production_manifest
    )
    assert (
        production.load_production_fock_manifest
        is production_profile.load_production_fock_manifest
    )


def test_production_profile_avoids_emission_orchestration_dependencies() -> None:
    """Manifest/profile parsing must not depend on emission or bundle owners."""

    source = Path(production_profile.__file__).read_text(encoding="utf-8")
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
    assert relative_modules.isdisjoint(
        {"production", "production_cost", "cuda_emitter", "cuda_lowering"}
    )
    assert absolute_roots.isdisjoint({"benchmarks", "tools", "vibeqc"})


def test_resolved_profile_is_owned_by_profile_module() -> None:
    """The canonical profile result type is no longer defined by the facade."""

    assert production_profile.ResolvedProductionProfile.__module__.endswith(
        ".production_profile"
    )

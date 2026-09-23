from __future__ import annotations

import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.xc import bulk_aot_cache


def _digest(label: str) -> str:
    return canonical_hash({"label": label})


def _recipe() -> dict:
    return {
        "emission_identity": _digest("emission"),
        "translation_unit_sha256": _digest("source"),
        "compiler": {
            "executable_sha256": _digest("cc"),
            "version": "Example Compiler 1.0",
        },
        "flags": ["-std=c99", "-O1", "-fPIC"],
    }


def _system_headers(label: str = "headers") -> bulk_aot_cache.CacheDependency:
    return bulk_aot_cache.CacheDependency(
        "system-header-manifest", "resolved-include-closure", _digest(label)
    )


def test_incomplete_closure_never_yields_reusable_key() -> None:
    closure = bulk_aot_cache.closure_from_probe_recipe(
        _recipe(),
        backend="cpu",
        target="x86_64",
        dependencies=(_system_headers(),),
        complete=False,
    )

    assert closure.cache_key is None
    assert closure.blocker == {
        "reason": "incomplete-dependency-closure",
        "closure_complete": False,
        "missing_roles": [],
    }


def test_missing_required_dependency_fails_closed_even_if_marked_complete() -> None:
    closure = bulk_aot_cache.closure_from_probe_recipe(
        _recipe(),
        backend="cuda",
        target="sm_120",
        dependencies=(_system_headers(),),
        complete=True,
    )

    assert closure.cache_key is None
    assert closure.blocker == {
        "reason": "incomplete-dependency-closure",
        "closure_complete": True,
        "missing_roles": ["cuda-device-toolchain-manifest"],
    }


def test_complete_closure_key_is_deterministic_and_order_independent() -> None:
    cuda = bulk_aot_cache.CacheDependency(
        "cuda-device-toolchain-manifest", "nvcc-ptxas-libdevice", _digest("cuda")
    )
    recipe = _recipe()
    first = bulk_aot_cache.closure_from_probe_recipe(
        recipe,
        backend="cuda",
        target="sm_120",
        dependencies=(_system_headers(), cuda),
        complete=True,
    )
    second = bulk_aot_cache.closure_from_probe_recipe(
        recipe,
        backend="cuda",
        target="sm_120",
        dependencies=(cuda, _system_headers()),
        complete=True,
    )

    assert first.reusable
    assert first.cache_key == second.cache_key
    assert first.blocker is None


def test_source_target_flags_and_dependency_changes_all_invalidate() -> None:
    base = bulk_aot_cache.closure_from_probe_recipe(
        _recipe(),
        backend="cpu",
        target="x86_64",
        dependencies=(_system_headers(),),
        complete=True,
    )

    changed_source_recipe = _recipe()
    changed_source_recipe["translation_unit_sha256"] = _digest("other-source")
    changed_source = bulk_aot_cache.closure_from_probe_recipe(
        changed_source_recipe,
        backend="cpu",
        target="x86_64",
        dependencies=(_system_headers(),),
        complete=True,
    )
    changed_flags_recipe = _recipe()
    changed_flags_recipe["flags"] = ["-std=c99", "-O2", "-fPIC"]
    changed_flags = bulk_aot_cache.closure_from_probe_recipe(
        changed_flags_recipe,
        backend="cpu",
        target="x86_64",
        dependencies=(_system_headers(),),
        complete=True,
    )
    changed_headers = bulk_aot_cache.closure_from_probe_recipe(
        _recipe(),
        backend="cpu",
        target="x86_64",
        dependencies=(_system_headers("other-headers"),),
        complete=True,
    )
    changed_target = bulk_aot_cache.closure_from_probe_recipe(
        _recipe(),
        backend="cpu",
        target="aarch64",
        dependencies=(_system_headers(),),
        complete=True,
    )

    assert bulk_aot_cache.invalidation_reasons(base, changed_source) == (
        "translation-unit",
    )
    assert bulk_aot_cache.invalidation_reasons(base, changed_flags) == (
        "compiler-flags",
    )
    assert bulk_aot_cache.invalidation_reasons(base, changed_headers) == (
        "dependency-content",
    )
    assert bulk_aot_cache.invalidation_reasons(base, changed_target) == ("target",)
    assert len(
        {
            base.cache_key,
            changed_source.cache_key,
            changed_flags.cache_key,
            changed_headers.cache_key,
            changed_target.cache_key,
        }
    ) == 5


def test_dependency_identity_changes_are_explicit_set_invalidation() -> None:
    before = bulk_aot_cache.closure_from_probe_recipe(
        _recipe(),
        backend="cpu",
        target="x86_64",
        dependencies=(_system_headers(),),
        complete=True,
    )
    renamed = bulk_aot_cache.CacheDependency(
        "system-header-manifest", "different-include-root", _digest("headers")
    )
    after = bulk_aot_cache.closure_from_probe_recipe(
        _recipe(),
        backend="cpu",
        target="x86_64",
        dependencies=(renamed,),
        complete=True,
    )

    assert bulk_aot_cache.invalidation_reasons(before, after) == ("dependency-set",)


def test_duplicate_dependency_role_identity_is_rejected() -> None:
    dep = _system_headers()
    with pytest.raises(ValueError, match="duplicate cache dependency"):
        bulk_aot_cache.closure_from_probe_recipe(
            _recipe(),
            backend="cpu",
            target="x86_64",
            dependencies=(dep, dep),
            complete=True,
        )


def test_probe_recipe_shape_and_digests_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid census probe recipe"):
        bulk_aot_cache.closure_from_probe_recipe(
            {}, backend="cpu", target="x86_64", dependencies=(), complete=False
        )
    recipe = _recipe()
    recipe["compiler"]["executable_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        bulk_aot_cache.closure_from_probe_recipe(
            recipe,
            backend="cpu",
            target="x86_64",
            dependencies=(_system_headers(),),
            complete=True,
        )

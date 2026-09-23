"""Runtime validation preserves the frozen, hash-bound cache closure contract."""

from dataclasses import dataclass, replace
from typing import Any

import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.xc.bulk_aot_cache import (
    CacheClosure,
    CacheDependency,
    closure_from_probe_recipe,
)


def closure(**changes: Any) -> CacheClosure:
    dependencies = tuple(
        CacheDependency(role, "test-owner", "a" * 64)
        for role in (
            "compiler-executable",
            "compiler-version",
            "system-header-manifest",
        )
    )
    values = {
        "emission_identity": "b" * 64,
        "translation_unit_sha256": "c" * 64,
        "backend": "cpu",
        "target": "x86_64",
        "flags": ("-O1",),
        "dependencies": dependencies,
        "complete": True,
    }
    return CacheClosure(**{**values, **changes})


@pytest.mark.parametrize("target", [True, 7, ["x86_64"], {"arch": "x86_64"}, " "])
def test_target_is_an_immutable_nonblank_string(target: Any) -> None:
    with pytest.raises(ValueError, match="target"):
        closure(target=target)


@pytest.mark.parametrize("field", ["role", "identity"])
@pytest.mark.parametrize("value", [True, 7, ["header"], {"owner": "header"}, " "])
def test_dependency_labels_are_nonblank_strings(field: str, value: Any) -> None:
    arguments = {"role": "system-header-manifest", "identity": "headers"}
    with pytest.raises(ValueError, match="dependency"):
        CacheDependency(**{**arguments, field: value}, content_sha256="d" * 64)


@pytest.mark.parametrize("container", [list, iter])
def test_closure_rejects_mutable_or_consumable_dependency_containers(
    container: Any,
) -> None:
    with pytest.raises(ValueError, match="dependencies"):
        closure(dependencies=container(closure().dependencies))


@dataclass(order=True)
class UnvalidatedDependency:
    role: str
    identity: str = "mutable-owner"
    content_sha256: str = "not-a-digest"

    def to_payload(self) -> dict[str, str]:
        return vars(self).copy()


def test_unvalidated_dependency_cannot_bypass_digest_validation() -> None:
    entries = tuple(UnvalidatedDependency(d.role) for d in closure().dependencies)
    with pytest.raises(ValueError, match="dependencies"):
        closure(dependencies=entries)


def test_valid_closure_identity_and_export_stay_unchanged() -> None:
    original = closure()
    expected = canonical_hash(original.to_payload())
    # Recorded from the pre-repair implementation for this exact valid payload.
    assert expected == "00afd67f85ac9f713797487b0dce3369fca5489a414b1eadc23cb20b4ce9359d"
    assert original.cache_key == expected
    exported = original.to_payload()
    exported["dependencies"][0]["content_sha256"] = "e" * 64
    exported["flags"].clear()
    assert original.cache_key == expected
    assert (
        replace(original, dependencies=original.dependencies[::-1]).cache_key
        == expected
    )
    assert replace(original, complete=False).cache_key is None


def test_recipe_bridge_keeps_valid_dependency_identity() -> None:
    recipe = {
        "emission_identity": "b" * 64,
        "translation_unit_sha256": "c" * 64,
        "compiler": {"executable_sha256": "a" * 64, "version": "test-compiler"},
        "flags": ["-O1"],
    }
    header = CacheDependency("system-header-manifest", "headers", "d" * 64)
    result = closure_from_probe_recipe(
        recipe, backend="cpu", target="x86_64", dependencies=(header,), complete=True
    )
    assert result.reusable and result.cache_key == canonical_hash(result.to_payload())
    recipe["flags"].append("-O2")
    assert result.flags == ("-O1",)

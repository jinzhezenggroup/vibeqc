from __future__ import annotations

import json
from pathlib import Path

import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.xc import bulk_aot_cache, bulk_aot_store


def _digest(label: str) -> str:
    return canonical_hash({"label": label})


def _closure(
    *, target: str = "x86_64", complete: bool = True
) -> bulk_aot_cache.CacheClosure:
    recipe = {
        "emission_identity": _digest("emission"),
        "translation_unit_sha256": _digest("source"),
        "compiler": {
            "executable_sha256": _digest("cc"),
            "version": "Example Compiler 1.0",
        },
        "flags": ["-std=c99", "-O1", "-fPIC"],
    }
    headers = bulk_aot_cache.CacheDependency(
        "system-header-manifest", "resolved-include-closure", _digest("headers")
    )
    return bulk_aot_cache.closure_from_probe_recipe(
        recipe,
        backend="cpu",
        target=target,
        dependencies=(headers,),
        complete=complete,
    )


def _object(tmp_path: Path, payload: bytes = b"ELF-test-object") -> Path:
    path = tmp_path / "compiled.o"
    path.write_bytes(payload)
    return path


def test_store_round_trip_verifies_manifest_and_object(tmp_path: Path) -> None:
    closure = _closure()
    result = bulk_aot_store.store_artifact(
        tmp_path / "cache", closure, _object(tmp_path)
    )

    assert result.status == "hit"
    assert result.cache_key == closure.cache_key
    assert result.artifact_path is not None
    assert result.artifact_path.read_bytes() == b"ELF-test-object"
    assert result.object_bytes == len(b"ELF-test-object")
    replay = bulk_aot_store.lookup_artifact(tmp_path / "cache", closure)
    assert replay == result


def test_incomplete_closure_is_rejected_without_creating_cache(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    result = bulk_aot_store.store_artifact(
        root, _closure(complete=False), _object(tmp_path)
    )

    assert result.status == "rejected"
    assert result.reason == "incomplete-dependency-closure"
    assert result.cache_key is None
    assert not root.exists()


def test_target_change_is_a_clean_miss_not_cross_target_reuse(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    bulk_aot_store.store_artifact(root, _closure(), _object(tmp_path))

    changed = bulk_aot_store.lookup_artifact(root, _closure(target="aarch64"))
    assert changed.status == "miss"
    assert changed.reason == "manifest-missing"
    assert changed.cache_key != _closure().cache_key


def test_object_corruption_is_detected_and_not_silently_repaired(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    closure = _closure()
    stored = bulk_aot_store.store_artifact(root, closure, _object(tmp_path))
    assert stored.artifact_path is not None
    stored.artifact_path.write_bytes(b"tampered")

    lookup = bulk_aot_store.lookup_artifact(root, closure)
    assert lookup.status == "corrupt"
    assert lookup.reason == "object-size"
    with pytest.raises(ValueError, match="refusing to overwrite corrupt cache entry"):
        bulk_aot_store.store_artifact(root, closure, _object(tmp_path, b"fresh"))


def test_manifest_closure_tampering_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    closure = _closure()
    stored = bulk_aot_store.store_artifact(root, closure, _object(tmp_path))
    assert stored.artifact_path is not None
    manifest_path = stored.artifact_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["closure"]["target"] = "other"
    manifest_path.write_text(json.dumps(manifest))

    lookup = bulk_aot_store.lookup_artifact(root, closure)
    assert lookup.status == "corrupt"
    assert lookup.reason == "manifest-closure"


def test_existing_verified_entry_wins_without_republication(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    closure = _closure()
    first = bulk_aot_store.store_artifact(root, closure, _object(tmp_path, b"first"))
    second_source = tmp_path / "second.o"
    second_source.write_bytes(b"second-different-object")
    second = bulk_aot_store.store_artifact(root, closure, second_source)

    assert second == first
    assert second.artifact_path is not None
    assert second.artifact_path.read_bytes() == b"first"

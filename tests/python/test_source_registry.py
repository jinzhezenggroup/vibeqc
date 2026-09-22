"""Repository-wide upstream scientific source registry contracts."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import pytest

from tools import source_registry


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_checked_in_registry_is_offline_verifiable() -> None:
    summary = source_registry.verify()
    assert summary["sources"] >= 7
    assert summary["local_files"] >= 43
    assert summary["products"] >= 5
    assert summary["derived_manifests"] == 4


REMOTE_REGENERATION_SOURCES = {
    "dftd4-reference",
    "gpu4pyscf-rys",
    "mctc-lib-eeq",
    "multicharge-eeq2019",
    "simple-dftd3-gcp",
}


def test_checked_in_sources_are_vendored_under_upstream_tree() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    for source_id, source in registry["sources"].items():
        if "local_root" not in source:
            continue
        assert source["kind"] != "remote-file-set", source_id
        assert source["local_root"].startswith("upstream/"), source_id
        root = source_registry.ROOT / source["local_root"]
        for name in source["files"]:
            assert (root / name).is_file()


def test_large_or_qualification_only_sources_stay_remote() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    for source_id in REMOTE_REGENERATION_SOURCES:
        source = registry["sources"][source_id]
        assert source["kind"] == "remote-file-set"
        assert "local_root" not in source


def test_libxc_registry_owns_every_pinned_source_file() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    source = registry["sources"]["libxc-7.0.0"]
    root = source_registry.ROOT / source["local_root"]
    derived = {
        Path(path).name
        for path, spec in registry["derived_manifests"].items()
        if spec.get("source") == "libxc-7.0.0"
    }
    actual = {path.name for path in root.iterdir() if path.is_file()} - derived
    assert actual == set(source["files"])
    assert set(source["collections"]) == {"core", "rsh", "wb97mv"}


def test_libxc_importer_semantics_are_pinned_separately() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    admission = registry["sources"]["libxc-7.0.0"]["admission"]
    importer = source_registry.ROOT / admission["importer"]
    from vibeqc_compiler.xc.libxc_maple import IMPORTER_SEMANTICS

    assert admission["semantics"] == IMPORTER_SEMANTICS
    assert admission["importer_sha256"] == source_registry._sha256(importer)
    assert registry["products"]["libxc-xc-admission"]["inputs"] == ["libxc-7.0.0"]


def test_gcp_canonical_input_matches_registered_upstream_provenance() -> None:
    registry = json.loads(source_registry.REGISTRY.read_text())
    source = registry["sources"]["simple-dftd3-gcp"]
    data = json.loads(
        (source_registry.ROOT / "external/r2scan3c/gcp-r2scan3c-h-ar.json").read_text()
    )
    upstream = data["upstream"]
    assert upstream["commit"] == source["revision"]
    assert upstream["license"] == source["license"]
    hashes = {
        item["upstream_path"]: item["sha256"] for item in source["files"].values()
    }
    assert hashes["src/dftd3/gcp/param.f90"] == upstream["param_sha256"]
    assert hashes["src/dftd3/gcp.f90"] == upstream["implementation_sha256"]
    assert hashes["src/dftd3/data/vdwrad.f90"] == upstream["vdwrad_sha256"]


def test_registry_derived_manifests_are_byte_stable() -> None:
    registry = source_registry._load()
    for relative, spec in registry["derived_manifests"].items():
        path = source_registry.ROOT / relative
        assert path.read_text(
            encoding="utf-8"
        ) == source_registry.render_derived_manifest(registry, spec)


def test_sync_is_pinned_and_normalizes_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = b"alpha  \n beta\t\n"
    normalized = b"alpha\n beta\n"
    registry = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {
            "sample": {
                "kind": "file-set",
                "repository": "https://example.invalid/sample",
                "revision": "rev-123",
                "license": "MIT",
                "local_root": "external/sample",
                "files": {
                    "data.txt": {
                        "upstream_path": "data.txt",
                        "url": "https://example.invalid/sample/rev-123/data.txt",
                        "upstream_sha256": _digest(upstream),
                        "sha256": _digest(normalized),
                        "normalization": "trailing-whitespace-only",
                    }
                },
            }
        },
        "products": {},
        "derived_manifests": {},
    }
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)
    monkeypatch.setattr(
        source_registry.urllib.request,
        "urlopen",
        lambda request, timeout: contextlib.closing(io.BytesIO(upstream)),
    )
    written = source_registry.sync_source("sample", registry_path)
    assert written == [tmp_path / "external/sample/data.txt"]
    assert written[0].read_bytes() == normalized
    assert source_registry.verify(registry_path)["local_files"] == 1


def test_sync_refuses_unexpected_upstream_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = b"expected\n"
    registry = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {
            "sample": {
                "kind": "remote-file-set",
                "repository": "https://example.invalid/sample",
                "revision": "rev-123",
                "license": "MIT",
                "files": {
                    "data.txt": {
                        "upstream_path": "data.txt",
                        "url": "https://example.invalid/sample/rev-123/data.txt",
                        "sha256": _digest(expected),
                    }
                },
            }
        },
        "products": {},
        "derived_manifests": {},
    }
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)
    monkeypatch.setattr(
        source_registry.urllib.request,
        "urlopen",
        lambda request, timeout: contextlib.closing(io.BytesIO(b"changed upstream\n")),
    )
    with pytest.raises(source_registry.SourceRegistryError, match="upstream digest"):
        source_registry.sync_source(
            "sample", registry_path, cache_root=tmp_path / ".cache/vibeqc-sources"
        )
    assert not (tmp_path / ".cache/vibeqc-sources/sample/data.txt").exists()


def test_registry_rejects_floating_or_unsafe_source_paths(tmp_path: Path) -> None:
    payload = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {
            "sample": {
                "kind": "file-set",
                "repository": "https://example.invalid/sample",
                "revision": "rev-123",
                "license": "MIT",
                "local_root": "../escape",
                "files": {
                    "data.txt": {
                        "url": "https://example.invalid/sample/main/data.txt",
                        "sha256": "0" * 64,
                    }
                },
            }
        },
        "products": {},
        "derived_manifests": {},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(source_registry.SourceRegistryError):
        source_registry.verify(path)


def test_update_requires_explicit_revision_and_invalidates_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"old\n"
    candidate = b"new\n"
    source = {
        "kind": "file-set",
        "repository": "https://github.com/example/sample",
        "revision": "rev-123",
        "license": "MIT",
        "local_root": "external/sample",
        "files": {
            "data.txt": {
                "upstream_path": "data.txt",
                "url": "https://raw.githubusercontent.com/example/sample/rev-123/data.txt",
                "sha256": _digest(original),
            }
        },
    }
    registry = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {"sample": source},
        "products": {
            "derived": {
                "inputs": ["sample"],
                "input_identity_sha256": source_registry._product_input_identity(
                    {"sample": source}, ["sample"]
                ),
                "generator": "tools/generator.py",
                "generator_sha256": "0" * 64,
                "outputs": {},
            }
        },
        "derived_manifests": {},
    }
    registry_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry))
    local = tmp_path / "external/sample/data.txt"
    local.parent.mkdir(parents=True)
    local.write_bytes(original)
    monkeypatch.setattr(source_registry, "ROOT", tmp_path)
    monkeypatch.setattr(
        source_registry.urllib.request,
        "urlopen",
        lambda request, timeout: contextlib.closing(io.BytesIO(candidate)),
    )

    with pytest.raises(source_registry.SourceRegistryError, match="immutable"):
        source_registry.update_source("sample", "master", registry_path)

    written = source_registry.update_source("sample", "rev-456", registry_path)
    assert written == [local]
    assert local.read_bytes() == candidate
    updated = json.loads(registry_path.read_text())
    item = updated["sources"]["sample"]["files"]["data.txt"]
    assert updated["sources"]["sample"]["revision"] == "rev-456"
    assert item["sha256"] == _digest(candidate)
    assert "/rev-456/data.txt" in item["url"]
    with pytest.raises(
        source_registry.SourceRegistryError, match="source inputs are stale"
    ):
        source_registry.verify(registry_path)

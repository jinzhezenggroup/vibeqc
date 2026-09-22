"""Every registry entry point enforces pinned revisions before network access."""

import hashlib
import json
from pathlib import Path

import pytest

from tools import source_registry as registry


@pytest.mark.parametrize(
    "revision", ("main", "master", "HEAD", "latest", "stable", "develop")
)
@pytest.mark.parametrize("operation", ("verify", "sync"))
def test_registered_floating_revision_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revision: str, operation: str
) -> None:
    data = b"pinned bytes\n"
    payload = {
        "schema": "vibeqc.scientific-source-registry",
        "schema_version": 1,
        "sources": {
            "sample": {
                "kind": "file-set",
                "repository": "https://example.invalid/sample",
                "revision": revision,
                "license": "MIT",
                "local_root": "upstream/sample",
                "files": {
                    "data": {
                        "url": f"https://example.invalid/sample/{revision}/data",
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                },
            }
        },
        "products": {},
        "derived_manifests": {},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    local = tmp_path / "upstream/sample/data"
    local.parent.mkdir(parents=True)
    local.write_bytes(data)
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    calls = []

    def fetch(*args: object, **kwargs: object) -> bytes:
        calls.append(args)
        return data

    monkeypatch.setattr(registry, "_fetch", fetch)
    with pytest.raises(registry.SourceRegistryError, match="immutable"):
        if operation == "verify":
            registry.verify(manifest)
        else:
            registry.sync_source("sample", manifest)
    assert not calls
    assert local.read_bytes() == data

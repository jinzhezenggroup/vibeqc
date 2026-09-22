"""Scientific-source synchronization must stay inside its selected roots."""

import hashlib
import json
from pathlib import Path

import pytest

from tools import source_registry as registry


@pytest.mark.parametrize("operation", ("sync", "update"))
@pytest.mark.parametrize("escape", ("source-id", "cache-link", "repository-link"))
def test_sync_and_update_reject_destination_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, escape: str
) -> None:
    root, cache, outside = (tmp_path / name for name in ("repo", "cache", "outside"))
    for path in (root, cache, outside):
        path.mkdir()
    payload = b"verified pinned scientific data\n"
    source_id = "../outside" if escape == "source-id" else "sample"
    source = {
        "kind": "remote-file-set",
        "repository": "https://github.com/example/source",
        "revision": "v1",
        "license": "MIT",
        "files": {
            "data.txt": {
                "url": "https://example.invalid/v1/data.txt",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    if escape == "cache-link":
        (cache / source_id).symlink_to(outside, target_is_directory=True)
    elif escape == "repository-link":
        (root / "linked").symlink_to(outside, target_is_directory=True)
        source["local_root"] = "linked"
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "vibeqc.scientific-source-registry",
                "schema_version": 1,
                "sources": {source_id: source},
                "products": {},
                "derived_manifests": {},
            }
        )
    )
    monkeypatch.setattr(registry, "ROOT", root)
    monkeypatch.setattr(registry, "_fetch", lambda *args, **kwargs: payload)
    with pytest.raises(registry.SourceRegistryError):
        if operation == "sync":
            registry.sync_source(source_id, manifest, cache_root=cache)
        else:
            registry.update_source(source_id, "v2", manifest, cache_root=cache)
    assert not (outside / "data.txt").exists()

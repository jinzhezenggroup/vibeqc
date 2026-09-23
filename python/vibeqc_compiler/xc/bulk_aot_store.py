"""Durable, fail-closed storage for reusable bulk Libxc AOT objects (#1123).

This module persists only artifacts whose :class:`CacheClosure` already proves a
complete compilation dependency closure. It does not compile code, discover a
toolchain, or promote runtime/public capability. Cache readers verify both the
manifest and object digest before returning a reusable path.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from vibeqc_compiler.common.provenance import atomic_json, file_hash

if TYPE_CHECKING:
    from .bulk_aot_cache import CacheClosure

STORE_SCHEMA = "vibeqc.libxc-aot-object-store/v1"
_OBJECT_NAME = "artifact.o"
_MANIFEST_NAME = "manifest.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CacheLookup:
    status: Literal["hit", "miss", "rejected", "corrupt"]
    reason: str | None
    cache_key: str | None
    artifact_path: Path | None = None
    object_sha256: str | None = None
    object_bytes: int | None = None


def _entry(root: Path, key: str) -> Path:
    if _SHA256.fullmatch(key) is None:
        raise ValueError("cache key must be a lowercase SHA-256 digest")
    return root / key[:2] / key


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".pending-object-", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def lookup_artifact(root: Path, closure: CacheClosure) -> CacheLookup:
    """Return a verified object path, never an unchecked cache candidate."""
    key = closure.cache_key
    if key is None:
        blocker = closure.blocker or {}
        return CacheLookup(
            status="rejected",
            reason=str(blocker.get("reason", "non-reusable-closure")),
            cache_key=None,
        )
    entry = _entry(Path(root), key)
    manifest_path = entry / _MANIFEST_NAME
    artifact_path = entry / _OBJECT_NAME
    if not manifest_path.is_file():
        return CacheLookup("miss", "manifest-missing", key)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return CacheLookup("corrupt", "manifest-unreadable", key)
    expected_closure = closure.to_payload()
    if not isinstance(manifest, dict) or manifest.get("schema") != STORE_SCHEMA:
        return CacheLookup("corrupt", "manifest-schema", key)
    if manifest.get("cache_key") != key:
        return CacheLookup("corrupt", "manifest-cache-key", key)
    if manifest.get("closure") != expected_closure:
        return CacheLookup("corrupt", "manifest-closure", key)
    object_record = manifest.get("object")
    if not isinstance(object_record, dict):
        return CacheLookup("corrupt", "manifest-object", key)
    digest = object_record.get("sha256")
    size = object_record.get("bytes")
    if (
        object_record.get("filename") != _OBJECT_NAME
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(size) is not int
        or size <= 0
    ):
        return CacheLookup("corrupt", "manifest-object", key)
    if not artifact_path.is_file():
        return CacheLookup("corrupt", "object-missing", key)
    try:
        observed_size = artifact_path.stat().st_size
        observed_digest = file_hash(artifact_path)
    except OSError:
        return CacheLookup("corrupt", "object-unreadable", key)
    if observed_size != size:
        return CacheLookup("corrupt", "object-size", key)
    if observed_digest != digest:
        return CacheLookup("corrupt", "object-digest", key)
    return CacheLookup(
        "hit",
        None,
        key,
        artifact_path=artifact_path,
        object_sha256=digest,
        object_bytes=size,
    )


def store_artifact(root: Path, closure: CacheClosure, object_path: Path) -> CacheLookup:
    """Atomically publish one verified object under an already reusable closure.

    Existing verified entries win. Corrupt entries are never silently replaced;
    callers must quarantine/remove them explicitly so corruption remains visible.
    A private staging directory holds both files until a single directory rename.
    Concurrent writers keep the first complete entry; they never replace its
    object separately from its manifest. Interrupted first writes remain misses.
    """
    current = lookup_artifact(root, closure)
    if current.status == "rejected":
        return current
    if current.status == "hit":
        return current
    if current.status == "corrupt":
        raise ValueError(f"refusing to overwrite corrupt cache entry: {current.reason}")
    source = Path(object_path)
    if not source.is_file():
        raise ValueError("object_path must identify a compiled object file")
    key = closure.cache_key
    if key is None:  # Kept local so a future CacheClosure change still fails closed.
        raise ValueError("cache closure is not reusable")
    entry = _entry(Path(root), key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-entry-", dir=entry.parent) as name:
        staged = Path(name)
        artifact_path = staged / _OBJECT_NAME
        _atomic_copy(source, artifact_path)
        # Hash the private copy, not a source that may change between reads.
        size = artifact_path.stat().st_size
        if size <= 0:
            raise ValueError("compiled object must be nonempty")
        manifest = {
            "schema": STORE_SCHEMA,
            "cache_key": key,
            "closure": closure.to_payload(),
            "object": {
                "filename": _OBJECT_NAME,
                "sha256": file_hash(artifact_path),
                "bytes": size,
            },
        }
        atomic_json(staged / _MANIFEST_NAME, manifest)
        try:
            # A committed entry is nonempty: rename cannot overwrite it.
            os.rename(staged, entry)
        except OSError:
            winner = lookup_artifact(root, closure)
            if winner.status == "hit":
                return winner
            if winner.status == "corrupt":
                raise ValueError(
                    f"refusing to overwrite corrupt cache entry: {winner.reason}"
                ) from None
            raise
    result = lookup_artifact(root, closure)
    if result.status != "hit":
        raise RuntimeError(
            f"published cache entry failed verification: {result.reason}"
        )
    return result

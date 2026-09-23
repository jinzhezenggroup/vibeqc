"""A cache object and its manifest must commit as one immutable entry."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.xc import bulk_aot_store as store
from vibeqc_compiler.xc.bulk_aot_cache import CacheClosure, CacheDependency

if TYPE_CHECKING:
    from pathlib import Path


def _closure() -> CacheClosure:
    return CacheClosure(
        "1" * 64,
        "2" * 64,
        "cpu",
        "x86_64",
        ("-O1",),
        tuple(
            CacheDependency(role, role, "3" * 64)
            for role in (
                "compiler-executable",
                "compiler-version",
                "system-header-manifest",
            )
        ),
        complete=True,
    )


@pytest.mark.parametrize("second_bytes", [b"first", b"other-compiled-object"])
def test_concurrent_first_writers_keep_one_complete_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_bytes: bytes
) -> None:
    root, closure = tmp_path / "cache", _closure()
    first, second = tmp_path / "first.o", tmp_path / "second.o"
    first.write_bytes(b"first")
    second.write_bytes(second_bytes)
    copied, resume = Event(), Event()
    copy = store._atomic_copy

    def interleave(source: Path, destination: Path) -> None:
        copy(source, destination)
        if source == first:
            copied.set()
            assert resume.wait(10), "second writer did not finish"

    monkeypatch.setattr(store, "_atomic_copy", interleave)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(store.store_artifact, root, closure, first)
        try:
            assert copied.wait(10), "first writer did not stage its object"
            winner = store.store_artifact(root, closure, second)
        finally:
            resume.set()
        loser = pending.result(timeout=10)
    assert winner == loser == store.lookup_artifact(root, closure)
    assert winner.artifact_path is not None
    assert winner.artifact_path.read_bytes() == second_bytes


def test_manifest_describes_copied_bytes_not_an_earlier_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.o"
    source.write_bytes(b"first")
    copy = store._atomic_copy

    def replace_before_copy(source: Path, destination: Path) -> None:
        source.write_bytes(b"new-object")
        copy(source, destination)

    monkeypatch.setattr(store, "_atomic_copy", replace_before_copy)
    result = store.store_artifact(tmp_path / "cache", _closure(), source)
    assert result.status == "hit"
    assert result.artifact_path is not None
    assert result.artifact_path.read_bytes() == b"new-object"
    assert result.object_bytes == len(b"new-object")


def test_failed_manifest_publication_remains_a_recoverable_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, closure = tmp_path / "cache", _closure()
    source = tmp_path / "source.o"
    source.write_bytes(b"object")
    publish = store.atomic_json

    def fail(*args: object) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(store, "atomic_json", fail)
    with pytest.raises(OSError, match="injected publication failure"):
        store.store_artifact(root, closure, source)
    assert store.lookup_artifact(root, closure).status == "miss"
    monkeypatch.setattr(store, "atomic_json", publish)
    assert store.store_artifact(root, closure, source).status == "hit"


def test_failed_source_open_closes_temporary_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptors: list[int] = []
    mkstemp = store.tempfile.mkstemp

    def record(*args: object, **kwargs: object) -> tuple[int, str]:
        descriptor, name = mkstemp(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor, name

    monkeypatch.setattr(store.tempfile, "mkstemp", record)
    with pytest.raises(FileNotFoundError):
        store._atomic_copy(tmp_path / "missing.o", tmp_path / "target.o")
    assert len(descriptors) == 1
    try:
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
    finally:
        try:
            os.close(descriptors[0])
        except OSError:
            pass


def test_concurrent_corrupt_entry_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, closure = tmp_path / "cache", _closure()
    source = tmp_path / "source.o"
    source.write_bytes(b"valid-object")
    assert closure.cache_key is not None
    entry = root / closure.cache_key[:2] / closure.cache_key
    copy = store._atomic_copy

    def corrupt_before_publication(source: Path, destination: Path) -> None:
        copy(source, destination)
        entry.mkdir(parents=True, exist_ok=True)
        (entry / "manifest.json").write_text("{}", encoding="utf-8")
        (entry / "artifact.o").write_bytes(b"retain-corrupt-evidence")

    monkeypatch.setattr(store, "_atomic_copy", corrupt_before_publication)
    with pytest.raises(ValueError, match="refusing to overwrite corrupt cache entry"):
        store.store_artifact(root, closure, source)
    assert (entry / "artifact.o").read_bytes() == b"retain-corrupt-evidence"
    assert (entry / "manifest.json").read_text() == "{}"


def test_existing_entry_does_not_read_replacement_source(tmp_path: Path) -> None:
    root, closure = tmp_path / "cache", _closure()
    source = tmp_path / "source.o"
    source.write_bytes(b"first")
    first = store.store_artifact(root, closure, source)
    second = store.store_artifact(root, closure, tmp_path / "missing.o")
    assert first == second


def test_empty_staged_object_leaves_a_clean_miss(tmp_path: Path) -> None:
    root, closure = tmp_path / "cache", _closure()
    source = tmp_path / "empty.o"
    source.write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty"):
        store.store_artifact(root, closure, source)
    assert store.lookup_artifact(root, closure).status == "miss"

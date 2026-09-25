"""Failed local publication must not leave an unretryable evidence destination."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import restore_retained_evidence as recovery


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, bytes]]:
    root = tmp_path / "repository"
    root.mkdir()
    payloads = {"campaign/first.bin": b"first\x00record", "campaign/second.bin": b"second\xffrecord"}
    for name, data in payloads.items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Evidence test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"],
        cwd=root, check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "vibeqc.git-object-snapshot.v1", "source_revision": revision,
        "file_count": len(payloads), "total_bytes": sum(map(len, payloads.values())),
        "files": [{"path": name, "bytes": len(data), "git_blob_sha1": hashlib.sha1(
            f"blob {len(data)}\0".encode() + data, usedforsecurity=False
        ).hexdigest()} for name, data in payloads.items()],
    }), encoding="utf-8")
    monkeypatch.setattr(recovery, "ROOT", root)
    return manifest, payloads


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_snapshot_copy_failure_is_clean_and_retryable(
    archive: tuple[Path, dict[str, bytes]], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, failure: type[BaseException],
) -> None:
    manifest, payloads = archive
    target = tmp_path / "snapshot"
    original = shutil.copyfile

    def fail_copy(src: str | Path, dst: str | Path, **kwargs: object) -> str | Path:
        if Path(dst).is_relative_to(target) and Path(src).name == "second.bin":
            Path(dst).write_bytes(b"incomplete")
            raise failure("injected publication failure")
        return original(src, dst, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", fail_copy)
        with pytest.raises((OSError, KeyboardInterrupt)):
            recovery.restore_snapshot(target, manifest=manifest)
    assert not target.exists()
    recovery.restore_snapshot(target, manifest=manifest)
    assert {name: (target / name).read_bytes() for name in payloads} == payloads


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_single_file_write_failure_is_clean_and_retryable(
    archive: tuple[Path, dict[str, bytes]], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, failure: type[BaseException],
) -> None:
    manifest, payloads = archive
    target = tmp_path / "single.bin"
    original = Path.open

    class BrokenWriter:
        def __init__(self, stream: object) -> None:
            self.stream = stream

        def __enter__(self) -> BrokenWriter:
            self.stream.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self.stream.__exit__(*args)

        def write(self, data: bytes) -> None:
            self.stream.write(data[:3])
            raise failure("injected publication failure")

    def fail_open(path: Path, *args: object, **kwargs: object) -> object:
        stream = original(path, *args, **kwargs)
        return BrokenWriter(stream) if path == target else stream

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_open)
        with pytest.raises(failure):
            recovery.restore("campaign/first.bin", target, manifest=manifest)
    assert not target.exists()
    recovery.restore("campaign/first.bin", target, manifest=manifest)
    assert target.read_bytes() == payloads["campaign/first.bin"]


@pytest.mark.parametrize("snapshot", [False, True])
def test_destination_created_during_verification_is_never_removed(
    archive: tuple[Path, dict[str, bytes]], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, snapshot: bool,
) -> None:
    manifest, _ = archive
    target = tmp_path / "already-owned"
    original = recovery._read

    def competing_destination(entry: dict) -> bytes:
        if not target.exists():
            if snapshot:
                target.mkdir()
                (target / "sentinel").write_bytes(b"keep")
            else:
                target.write_bytes(b"keep")
        return original(entry)

    monkeypatch.setattr(recovery, "_read", competing_destination)
    with pytest.raises(FileExistsError):
        if snapshot:
            recovery.restore_snapshot(target, manifest=manifest)
        else:
            recovery.restore("campaign/first.bin", target, manifest=manifest)
    assert ((target / "sentinel") if snapshot else target).read_bytes() == b"keep"


def test_corrupt_member_never_creates_destination(
    archive: tuple[Path, dict[str, bytes]], tmp_path: Path,
) -> None:
    manifest, _ = archive
    record = json.loads(manifest.read_text())
    record["files"][-1]["git_blob_sha1"] = "0" * 40
    manifest.write_text(json.dumps(record))
    target = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        recovery.restore_snapshot(target, manifest=manifest)
    assert not target.exists()

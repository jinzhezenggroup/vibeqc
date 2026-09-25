"""An interrupted publication compaction must retain the original evidence."""

from __future__ import annotations

import gzip
import json
import typing
from pathlib import Path

import pytest
from test_compact_evidence_safety import publication, snapshot

from tools import compact_evidence_publications as compact

# Explicitly re-export the fixture so pytest can discover it in this module.
__all__ = ["publication"]


def large_samples(root: Path) -> None:
    """Exercise the in-place evidence update, not only .json -> .json.gz."""
    directory = root / "campaign"
    samples = b"[" + b"1," * 500 + b"2]\n"
    (directory / "samples.json").write_bytes(samples)
    evidence_path = directory / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["attachments"][0].update(
        bytes=len(samples), sha256=compact.digest(samples)
    )
    evidence_path.write_bytes(compact.json_bytes(evidence))
    manifest_path = directory / "publication.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        data = (directory / entry["path"]).read_bytes()
        entry.update(bytes=len(data), sha256=compact.digest(data))
    manifest_path.write_bytes(compact.json_bytes(manifest))
    # Leave room for the rewritten path/metadata while keeping samples bulky.
    compact.THRESHOLD = len(evidence_path.read_bytes()) + 128


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("failure", ["second-output", "manifest", "retire-evidence"])
@pytest.mark.parametrize("exception", [OSError, KeyboardInterrupt])
def test_failure_restores_members_and_manifest(
    publication: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_place: bool,
    failure: str,
    exception: type[BaseException],
) -> None:
    if in_place:
        large_samples(publication)
    before = snapshot(publication)
    write_bytes = Path.write_bytes
    replace = Path.replace
    unlink = Path.unlink
    writes = 0
    injected = False

    def fail_once() -> None:
        nonlocal injected
        if not injected:
            injected = True
            raise exception("injected compaction failure")

    def write(path: Path, data: bytes) -> int:
        nonlocal writes
        writes += 1
        if failure == "second-output" and writes == 2:
            fail_once()
        if failure == "manifest" and path.name == "publication.json":
            fail_once()
        return write_bytes(path, data)

    def move(path: Path, target: Path) -> Path:
        if failure == "manifest" and Path(target).name == "publication.json":
            fail_once()
        if (
            failure == "retire-evidence"
            and path == publication / "campaign/evidence.json"
        ):
            fail_once()
        return replace(path, target)

    def remove(path: Path, missing_ok: bool = False) -> None:
        if (
            failure == "retire-evidence"
            and path == publication / "campaign/evidence.json"
        ):
            fail_once()
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "write_bytes", write)
    monkeypatch.setattr(Path, "replace", move)
    monkeypatch.setattr(Path, "unlink", remove)
    with pytest.raises(exception, match="injected compaction failure"):
        compact.compact_publication("campaign/publication.json")
    assert injected
    assert snapshot(publication) == before
    assert not list(publication.glob("campaign/.evidence-stage-*"))


@pytest.mark.parametrize("in_place", [False, True])
def test_success_keeps_payloads_and_idempotence(
    publication: Path, in_place: bool
) -> None:
    if in_place:
        large_samples(publication)
    old_samples = (publication / "campaign/samples.json").read_bytes()
    compact.compact_publication("campaign/publication.json")
    assert (
        gzip.decompress((publication / "campaign/samples.json.gz").read_bytes())
        == old_samples
    )
    manifest = json.loads((publication / "campaign/publication.json").read_text())
    for entry in manifest["files"]:
        data = (publication / "campaign" / entry["path"]).read_bytes()
        assert entry["bytes"] == len(data)
        assert entry["sha256"] == compact.digest(data)
    if in_place:
        assert (publication / "campaign/evidence.json").is_file()
    before = snapshot(publication)
    assert compact.compact_publication("campaign/publication.json", check=True) == []
    assert snapshot(publication) == before



def test_companion_created_after_preflight_is_not_overwritten(
    publication: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = snapshot(publication)
    target = publication / "campaign/evidence.json.gz"
    original_open = Path.open
    foreign = b"concurrent retained evidence"

    def open_with_race(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> typing.Any:
        if path == target and mode == "xb":
            with original_open(path, "wb") as stream:
                stream.write(foreign)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_race)
    with pytest.raises(FileExistsError):
        compact.compact_publication("campaign/publication.json")
    assert snapshot(publication) == {**before, "campaign/evidence.json.gz": foreign}


def test_failed_rollback_retains_original_backup(
    publication: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = publication / "campaign/evidence.json"
    original = evidence.read_bytes()
    manifest = publication / "campaign/publication.json"
    manifest_bytes = manifest.read_bytes()
    replace = Path.replace

    def blocked_replace(path: Path, target: Path) -> Path:
        if Path(target) == manifest:
            raise OSError("injected manifest commit failure")
        if path.name.startswith("old-") and Path(target) == evidence:
            raise OSError("injected rollback rename failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", blocked_replace)
    with pytest.raises(RuntimeError, match="rollback incomplete") as caught:
        compact.compact_publication("campaign/publication.json")
    assert manifest.read_bytes() == manifest_bytes
    recovery = list(publication.glob("campaign/.evidence-stage-*"))
    assert len(recovery) == 1
    assert any(path.read_bytes() == original for path in recovery[0].glob("old-*"))
    assert str(recovery[0]) in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)

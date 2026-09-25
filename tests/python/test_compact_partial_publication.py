"""A packed member does not mean that the entire publication is compacted."""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING

import pytest
from test_compact_evidence_safety import publication, snapshot
from test_compact_evidence_transaction import large_samples

from tools import compact_evidence_publications as compact

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["publication"]


def pack_one(root: Path, name: str) -> None:
    directory = root / "campaign"
    manifest_path = directory / "publication.json"
    manifest = json.loads(manifest_path.read_text())
    packed = gzip.compress((directory / name).read_bytes(), mtime=0)
    (directory / (name + ".gz")).write_bytes(packed)
    (directory / name).unlink()
    entry = next(row for row in manifest["files"] if row["path"] == name)
    entry.update(path=name + ".gz", bytes=len(packed), sha256=compact.digest(packed))
    if name == "samples.json":
        evidence_path = directory / "evidence.json"
        evidence = json.loads(evidence_path.read_bytes())
        evidence["attachments"][0].update(
            path=name + ".gz", bytes=len(packed), sha256=compact.digest(packed)
        )
        data = json.dumps(evidence).encode()
        evidence_path.write_bytes(data)
        record = next(row for row in manifest["files"] if row["role"] == "evidence")
        record.update(bytes=len(data), sha256=compact.digest(data))
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("name", ["samples.json", "evidence.json", "summary.json"])
def test_check_detects_remaining_plain_members(publication: Path, name: str) -> None:
    pack_one(publication, name)
    before = snapshot(publication)
    with pytest.raises(SystemExit, match="not compacted"):
        compact.main(["--check"])
    assert snapshot(publication) == before


@pytest.mark.parametrize("name", ["samples.json", "evidence.json", "summary.json"])
def test_partial_publication_finishes_without_double_compression(
    publication: Path, name: str
) -> None:
    samples = (publication / "campaign/samples.json").read_bytes()
    pack_one(publication, name)
    prior_packed = (publication / "campaign" / (name + ".gz")).read_bytes()
    changes = compact.compact_publication("campaign/publication.json")
    assert changes
    directory = publication / "campaign"
    assert not (directory / "samples.json").exists()
    assert not (directory / "evidence.json").exists()
    assert not list(directory.glob("*.gz.gz"))
    packed = (directory / "samples.json.gz").read_bytes()
    assert gzip.decompress(packed) == samples
    evidence = json.loads(
        gzip.decompress((directory / "evidence.json.gz").read_bytes())
    )
    assert evidence["energy"] == -1.25
    assert evidence["attachments"] == [
        {
            "path": "samples.json.gz",
            "bytes": len(packed),
            "sha256": compact.digest(packed),
        }
    ]
    manifest = json.loads((directory / "publication.json").read_bytes())
    for entry in manifest["files"]:
        raw = (directory / entry["path"]).read_bytes()
        assert len(raw) == entry["bytes"] and compact.digest(raw) == entry["sha256"]
    if name != "evidence.json":
        assert (directory / (name + ".gz")).read_bytes() == prior_packed
    before = snapshot(publication)
    assert compact.compact_publication("campaign/publication.json", check=True) == []
    assert snapshot(publication) == before


def test_packed_evidence_update_rolls_back_exact_original(
    publication: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack_one(publication, "evidence.json")
    before = snapshot(publication)
    manifest = publication / "campaign/publication.json"
    replace = type(manifest).replace

    def fail_manifest(path: Path, target: Path) -> Path:
        if target == manifest:
            raise OSError("injected manifest failure")
        return replace(path, target)

    monkeypatch.setattr(type(manifest), "replace", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        compact.compact_publication("campaign/publication.json")
    assert snapshot(publication) == before
    assert not list(publication.glob("campaign/.evidence-stage-*"))


def test_rewritten_evidence_crossing_threshold_is_compacted_in_one_pass(
    publication: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    large_samples(publication)
    directory = publication / "campaign"
    # The original evidence is just below the cap, but its rewritten storage
    # references increase its serialized size. Do not leave a second-pass job.
    monkeypatch.setattr(
        compact, "THRESHOLD", (directory / "evidence.json").stat().st_size + 1
    )
    compact.compact_publication("campaign/publication.json")
    assert (directory / "evidence.json.gz").is_file()
    before = snapshot(publication)
    assert compact.compact_publication("campaign/publication.json", check=True) == []
    assert snapshot(publication) == before

"""Read-only checks and original identities survive evidence compaction."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools import compact_evidence_publications as compact


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "campaign"
    directory.mkdir()
    samples = b'[1, 2, 3]\n'
    evidence = json.dumps({
        "energy": -1.25,
        "attachments": [{
            "path": "samples.json", "bytes": len(samples),
            "sha256": hashlib.sha256(samples).hexdigest(),
        }],
    }).encode()
    files = []
    for name, role, data in (
        ("samples.json", "samples", samples),
        ("evidence.json", "evidence", evidence),
        ("summary.json", "summary", b'{"passed": true}\n'),
    ):
        (directory / name).write_bytes(data)
        files.append({"path": name, "role": role, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    (directory / "publication.json").write_text(json.dumps({"files": files}))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "threshold_bytes": 1_000_000, "files": [],
    }))
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    monkeypatch.setattr(compact, "REVIEW", review)
    monkeypatch.setattr(compact, "TARGETS", ("campaign/publication.json",))
    monkeypatch.setattr(compact, "THRESHOLD", 1)
    return tmp_path


def test_check_never_rewrites_or_deletes_inputs(publication: Path) -> None:
    before = snapshot(publication)
    with pytest.raises(SystemExit, match="not compacted"):
        compact.main(["--check"])
    assert snapshot(publication) == before


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("name", ["samples.json", "evidence.json", "summary.json"])
@pytest.mark.parametrize("field", ["sha256", "bytes"])
def test_original_identity_is_checked_before_any_write(
    publication: Path, packed: bool, name: str, field: str
) -> None:
    if packed:
        compact.main([])
        if name != "summary.json":
            name += ".gz"
    path = publication / "campaign/publication.json"
    manifest = json.loads(path.read_text())
    entry = next(entry for entry in manifest["files"] if entry["path"] == name)
    entry[field] = "0" * 64 if field == "sha256" else entry["bytes"] + 1
    path.write_text(json.dumps(manifest))
    before = snapshot(publication)
    with pytest.raises(ValueError, match="identity mismatch"):
        compact.main([])
    assert snapshot(publication) == before


def test_existing_target_is_not_overwritten(publication: Path) -> None:
    (publication / "campaign/samples.json.gz").write_bytes(b"previous retained output")
    before = snapshot(publication)
    with pytest.raises(FileExistsError):
        compact.main([])
    assert snapshot(publication) == before


def test_success_preserves_data_and_repeated_check_is_read_only(publication: Path) -> None:
    old = snapshot(publication)
    compact.main([])
    manifest = json.loads((publication / "campaign/publication.json").read_text())
    for entry in manifest["files"]:
        data = (publication / "campaign" / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    samples = (publication / "campaign/samples.json.gz").read_bytes()
    assert gzip.decompress(samples) == old["campaign/samples.json"]
    evidence = json.loads(gzip.decompress(
        (publication / "campaign/evidence.json.gz").read_bytes()
    ))
    assert evidence["energy"] == -1.25
    attachment = evidence["attachments"][0]
    assert attachment["path"] == "samples.json.gz"
    assert attachment["sha256"] == hashlib.sha256(samples).hexdigest()
    before = snapshot(publication)
    compact.main(["--check"])
    assert snapshot(publication) == before

"""A Git blob identity must not disable an archive's declared SHA-256 check."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from tools import restore_retained_evidence as restore

if TYPE_CHECKING:
    from pathlib import Path

SCHEMAS = (
    "vibeqc.storage-migration.v1",
    "vibeqc.evidence-archive.v1",
    "vibeqc.git-snapshot.v1",
    "vibeqc.git-object-snapshot.v1",
)
DATA = b"\x00retained binary evidence\xff\r\n"
MEMBER = "campaign/data.bin"


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / MEMBER
    source.parent.mkdir()
    source.write_bytes(DATA)
    for arguments in (
        ["init", "--quiet"],
        ["add", "--", MEMBER],
        [
            "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgSign=false", "commit", "--quiet", "-m", "fixture",
        ],
    ):
        subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    source.unlink()
    monkeypatch.setattr(restore, "ROOT", repo)
    return tmp_path / "manifest.json", revision


def manifest(path: Path, revision: str, schema: str, checksum: str) -> None:
    blob = hashlib.sha1(
        f"blob {len(DATA)}\0".encode() + DATA, usedforsecurity=False
    ).hexdigest()
    entry = {
        "path": MEMBER,
        "bytes": len(DATA),
        "revision": revision,
        "sha256": hashlib.sha256(DATA).hexdigest(),
        "git_blob_sha1": blob,
    }
    if checksum == "bad-sha256":
        entry["sha256"] = "0" * 64
    elif checksum == "bad-blob":
        entry["git_blob_sha1"] = "0" * 40
    elif checksum == "single":
        extra = (
            "sha256" if schema == "vibeqc.git-object-snapshot.v1" else "git_blob_sha1"
        )
        del entry[extra]
    record_key = "archives" if schema == "vibeqc.storage-migration.v1" else "files"
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "source_revision": revision,
                "file_count": 1,
                "total_bytes": len(DATA),
                record_key: [entry],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("checksum", ["bad-sha256", "bad-blob", "dual", "single"])
def test_every_declared_digest_is_verified_before_publication(
    archive: tuple[Path, str], schema: str, snapshot: bool, checksum: str
) -> None:
    path, revision = archive
    manifest(path, revision, schema, checksum)
    target = path.parent / "restored"

    def run() -> Path:
        if snapshot:
            return restore.restore_snapshot(target, manifest=path)
        return restore.restore(MEMBER, target, manifest=path)

    if checksum.startswith("bad-"):
        with pytest.raises(ValueError, match="checksum/size mismatch"):
            run()
        assert not target.exists()
    else:
        assert run() == target
        assert (target / MEMBER if snapshot else target).read_bytes() == DATA

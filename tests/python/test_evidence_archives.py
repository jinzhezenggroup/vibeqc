"""Historical evidence remains intact and restores without overwriting files."""

import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.unpack_evidence import unpack

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks/results"


@pytest.mark.parametrize(
    "name", ["rccsd-148-a", "rccsd-148-b", "rccsd-148-c", "xc-integration-162"]
)
def test_committed_evidence_restores_exact_bytes(name, tmp_path):
    directory = RESULTS / name
    manifest = json.loads((directory / "raw-evidence.manifest.json").read_text())
    output = tmp_path / "restored"
    assert unpack(directory, output) == len(manifest["files"])
    for record in manifest["files"]:
        data = (output / record["path"]).read_bytes()
        assert sha256(data).hexdigest() == record["sha256"]
        if record["path"].endswith(".json"):
            json.loads(data)
    with pytest.raises(FileExistsError):
        unpack(directory, output)


def sample_archive(directory, name="record.json"):
    data = b'{"passed": true}\n'
    with ZipFile(directory / "raw-evidence.zip", "w") as archive:
        archive.writestr(name, data)
    manifest = {
        "schema": "vibeqc.evidence-archive.v1",
        "archive_sha256": sha256(
            (directory / "raw-evidence.zip").read_bytes()
        ).hexdigest(),
        "files": [
            {"path": name, "bytes": len(data), "sha256": sha256(data).hexdigest()}
        ],
    }
    (directory / "raw-evidence.manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("damage", ["archive", "member", "inventory"])
def test_corruption_fails_before_creating_output(tmp_path, damage):
    manifest = sample_archive(tmp_path)
    if damage == "archive":
        with (tmp_path / "raw-evidence.zip").open("ab") as stream:
            stream.write(b"corrupted")
    elif damage == "member":
        manifest["files"][0]["sha256"] = "0" * 64
    else:
        manifest["files"] = []
    (tmp_path / "raw-evidence.manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "restored"
    with pytest.raises(ValueError):
        unpack(tmp_path, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "a\\b"])
def test_unsafe_paths_are_rejected(tmp_path, name):
    sample_archive(tmp_path, name)
    # Windows' ZIP writer can normalize backslashes before our inventory check.
    with pytest.raises(ValueError, match="unsafe archive path|members differ"):
        unpack(tmp_path, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()

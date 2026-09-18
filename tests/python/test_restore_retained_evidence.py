"""Historical evidence restores offline with byte identity and safe output paths."""

import hashlib
import json
import subprocess

import pytest

from tools import restore_retained_evidence as module


@pytest.fixture
def historical(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return (
            subprocess.check_output(
                [
                    "git",
                    "-c",
                    "core.autocrlf=false",
                    "-c",
                    "user.name=Evidence test",
                    "-c",
                    "user.email=evidence@example.invalid",
                    *args,
                ],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

    git("init")
    name = "benchmarks/results/old/samples.json"
    payload = b'{"samples": [0.12345678901234567, 1e-14]}\r\n'
    source = root / name
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    git("add", name)
    git("commit", "--no-gpg-sign", "-m", "Historical fixture")
    revision = git("rev-parse", "HEAD")
    source.unlink()
    entry = {
        "path": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    audit = {
        "schema": "vibeqc.evidence-archive.v1",
        "source_revision": revision,
        "files": [entry],
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(audit))
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "AUDIT", manifest)
    return root, name, payload, manifest, audit


@pytest.mark.parametrize("legacy", [False, True])
def test_restore_keeps_original_bytes_and_refuses_overwrite(historical, legacy):
    root, name, payload, manifest, audit = historical
    if legacy:
        audit = {
            "schema": "vibeqc.storage-migration.v1",
            "archives": [{**audit["files"][0], "revision": audit["source_revision"]}],
        }
        manifest.write_text(json.dumps(audit))
    kwargs = {} if legacy else {"manifest": manifest}
    target = module.restore(name, **kwargs)
    assert target == root / ".artifacts/retention-restore" / name
    assert target.read_bytes() == payload
    with pytest.raises(FileExistsError):
        module.restore(name, **kwargs)
    assert target.read_bytes() == payload


@pytest.mark.parametrize("damage", ["hash", "size", "revision", "duplicate", "schema"])
def test_invalid_manifest_fails_before_writing(historical, damage):
    root, name, _, manifest, audit = historical
    if damage == "hash":
        audit["files"][0]["sha256"] = "0" * 64
    elif damage == "size":
        audit["files"][0]["bytes"] += 1
    elif damage == "revision":
        audit["source_revision"] = "master"
    elif damage == "duplicate":
        audit["files"] *= 2
    else:
        audit["schema"] = "unknown"
    manifest.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        module.restore(name, manifest=manifest)
    assert not (root / ".artifacts").exists()


def test_missing_git_object_does_not_fetch_implicitly(historical):
    root, name, _, manifest, audit = historical
    audit["source_revision"] = "a" * 40
    manifest.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="git fetch origin"):
        module.restore(name, manifest=manifest)
    assert not (root / ".artifacts").exists()


@pytest.mark.parametrize(
    "name", [".", "../escape", "/absolute", "a/../b", "a//b", "a\\b", "C:/b"]
)
def test_unsafe_path_is_rejected_before_git(historical, name):
    root, _, _, manifest, _ = historical
    with pytest.raises(ValueError, match="unsafe"):
        module.restore(name, manifest=manifest)
    assert not (root / ".artifacts").exists()


def test_default_destination_rejects_escaping_symlink(historical):
    root, name, _, manifest, _ = historical
    output = root / ".artifacts/retention-restore"
    output.mkdir(parents=True)
    outside = root / "outside"
    outside.mkdir()
    (output / "benchmarks").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        module.restore(name, manifest=manifest)
    assert not list(outside.iterdir())


def test_explicit_output_remains_supported(historical):
    root, name, payload, manifest, _ = historical
    target = root / "chosen" / "raw.json"
    assert module.restore(name, target, manifest=manifest) == target
    assert target.read_bytes() == payload

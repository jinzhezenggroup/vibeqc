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
    "name",
    [
        ".",
        "../escape",
        "/absolute",
        "a/../b",
        "a//b",
        "a\\b",
        "C:/b",
        "",
        "a\nfile",
        "a\0file",
    ],
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


@pytest.fixture
def snapshot(historical):
    root, name, payload, manifest, audit = historical
    binary = "benchmarks/results/old/arrays.bin"
    data = bytes(range(256)) + b"\x00\xff\r\n"
    source = root / binary
    source.write_bytes(data)
    subprocess.run(["git", "add", binary], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Evidence test",
            "-c",
            "user.email=evidence@example.invalid",
            "commit",
            "--no-gpg-sign",
            "-m",
            "Add binary snapshot member",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    audit["source_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    source.unlink()
    audit["schema"] = "vibeqc.git-snapshot.v1"
    audit["files"].append(
        {
            "path": binary,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    audit["file_count"] = len(audit["files"])
    audit["total_bytes"] = sum(entry["bytes"] for entry in audit["files"])
    manifest.write_text(json.dumps(audit))
    return root, manifest, audit, {name: payload, binary: data}


def test_complete_snapshot_preserves_text_binary_and_single_file_api(snapshot):
    root, manifest, _, originals = snapshot
    target = module.restore_snapshot(manifest=manifest)
    assert target == root / ".artifacts/retention-snapshot"
    assert {
        p.relative_to(target).as_posix(): p.read_bytes()
        for p in target.rglob("*")
        if p.is_file()
    } == originals
    for name, payload in originals.items():
        assert not (root / name).exists()
        assert module.restore(name, manifest=manifest).read_bytes() == payload
    with pytest.raises(FileExistsError):
        module.restore_snapshot(manifest=manifest)
    assert all((target / name).read_bytes() == data for name, data in originals.items())


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "size",
        "missing-last",
        "duplicate",
        "conflict",
        "unsafe",
        "file-count",
        "total-bytes",
        "empty",
    ],
)
def test_complete_snapshot_failure_leaves_no_partial_destination(snapshot, damage):
    root, manifest, audit, _ = snapshot
    last = audit["files"][-1]
    if damage == "hash":
        last["sha256"] = "0" * 64
    elif damage == "size":
        last["bytes"] += 1
        audit["total_bytes"] += 1
    elif damage == "missing-last":
        last["path"] = "benchmarks/results/old/missing.bin"
    elif damage == "duplicate":
        last["path"] = audit["files"][0]["path"]
    elif damage == "conflict":
        last["path"] = audit["files"][0]["path"] + "/child"
    elif damage == "unsafe":
        last["path"] = "../escape.bin"
    elif damage == "file-count":
        audit["file_count"] += 1
    elif damage == "total-bytes":
        audit["total_bytes"] += 1
    else:
        audit["files"] = []
    manifest.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        module.restore_snapshot(manifest=manifest)
    assert not (root / ".artifacts").exists()


def test_snapshot_git_reads_disable_network_and_lazy_fetch(snapshot, monkeypatch):
    _, manifest, _, originals = snapshot
    original_run = subprocess.run
    calls = []

    def offline_read(command, **kwargs):
        calls.append(command)
        assert command[:3] == ["git", "cat-file", "blob"]
        assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
        assert kwargs["env"]["GIT_ALLOW_PROTOCOL"] == ""
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        return original_run(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", offline_read)
    module.restore_snapshot(manifest=manifest)
    assert len(calls) == len(originals)


def test_snapshot_does_not_overwrite_destination_created_during_verification(
    snapshot, monkeypatch
):
    root, manifest, _, _ = snapshot
    target = root / "other-writer"
    read = module._read

    def concurrent_read(entry):
        if not target.exists():
            target.mkdir()
            (target / "keep.txt").write_text("unrelated work")
        return read(entry)

    monkeypatch.setattr(module, "_read", concurrent_read)
    with pytest.raises(FileExistsError):
        module.restore_snapshot(target, manifest=manifest)
    assert [p.name for p in target.iterdir()] == ["keep.txt"]
    assert (target / "keep.txt").read_text() == "unrelated work"


def test_snapshot_rejects_default_storage_symlink(snapshot):
    root, manifest, _, _ = snapshot
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    (root / ".artifacts").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        module.restore_snapshot(manifest=manifest)
    assert not list(elsewhere.iterdir())


def test_snapshot_cli_requires_exactly_one_selection(snapshot, capsys):
    root, manifest, _, _ = snapshot
    for selection in ([], ["some.json", "--all"]):
        with pytest.raises(SystemExit) as error:
            module.main(selection)
        assert error.value.code == 2
    target = root / "cli-copy"
    module.main(["--all", "--manifest", str(manifest), "--output", str(target)])
    assert str(target) in capsys.readouterr().out
    assert target.is_dir()


def test_checked_in_snapshot_has_only_git_identity():
    manifest = (
        module.ROOT / "benchmarks/results/retention-checkout/snapshot.manifest.json"
    )
    audit = json.loads(manifest.read_text())
    assert audit["schema"] == "vibeqc.git-snapshot.v1"
    assert not any(key.startswith("archive_") for key in audit)
    records = module._records(manifest)
    assert len(records) == audit["file_count"] == 955
    assert sum(record["bytes"] for record in records) == audit["total_bytes"]
    removed = [record for record in records if record["checkout"] == "git-history"]
    assert len(removed) == audit["moved_files"] == 209
    assert sum(record["bytes"] for record in removed) == audit["moved_bytes"]
    assert audit["history_rewritten"] is False

"""Storage policy cannot discard references or manufacture scientific passes."""

import json
import subprocess
from pathlib import Path

import pytest

from tools.vibeqc_validation.publication import publish, validate_publication
from tools.vibeqc_validation.retention import (
    check,
    classify,
    digest,
    extract_log,
    inventory,
    tracked_blobs,
)
from tools.vibeqc_validation.schema import block_error, new_evidence, outcome

ROOT = Path(__file__).resolve().parents[2]


def policy(**exceptions):
    return {
        "schema": "vibeqc.retention-policy.v1",
        "review_size_bytes": 32,
        "exceptions": exceptions,
    }


@pytest.mark.parametrize(
    "path",
    [
        "tests/reference_data/example.xml",
        "tests/data/checkpoint.npz",
        "external/libxc/a.log",
    ],
)
def test_reference_context_precedes_suffix(path):
    assert classify(path) == "reference"
    assert check({path: b"independent oracle"}, policy()) == []


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/results/run/a.log",
        "benchmarks/results/run/test.xml",
        "benchmarks/results/attempts/ok.json",
        "benchmarks/results/run/profile.sqlite.gz",
        "benchmarks/results/run/state.chk",
        "benchmarks/results/run/kernel.cubin",
    ],
)
def test_transient_patterns_need_explicit_exception(path):
    assert check({path: b"data"}, policy())
    exception = {
        "sha256": digest(b"data"),
        "reason": "Independent fixture",
        "owner": "validation",
    }
    assert check({path: b"data"}, policy(**{path: exception})) == []
    assert any(
        "stale" in error
        for error in check({path: b"changed"}, policy(**{path: exception}))
    )
    assert any("stale" in error for error in check({}, policy(**{path: exception})))


def test_size_guard_is_reviewable_and_covers_references():
    path = "tests/reference_data/large.json"
    data = b"x" * 33
    assert check({path: data}, policy())
    exception = {
        "sha256": digest(data),
        "reason": "Complete raw samples are required",
        "owner": "validation",
    }
    assert check({path: data}, policy(**{path: exception})) == []


def test_inventory_reads_staged_bytes_not_worktree_or_symlink_target(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "file.json"
    path.write_bytes(b"staged")
    subprocess.run(["git", "add", "file.json"], cwd=tmp_path, check=True)
    path.write_bytes(b"unstaged")
    link = tmp_path / "outside"
    link.symlink_to("/unreadable/external/path")
    subprocess.run(["git", "add", "outside"], cwd=tmp_path, check=True)
    blobs = tracked_blobs(tmp_path)
    assert blobs == {"file.json": b"staged", "outside": b"/unreadable/external/path"}
    report = inventory(blobs)
    assert sum(row["bytes"] for row in report["files"]) == sum(map(len, blobs.values()))
    assert report["classes"]["unknown"]["files"] == 1


def test_stdout_json_measurements_and_negative_diagnostics_survive():
    log = b'SLURM_JOB_ID=1\n{"seconds": [1.25, 0.5], "error": 1e-12}\nFAILED allocation budget\n[1/1] progress\n{"peak": 128}\n'
    result = extract_log(log)
    assert result["measurements"] == [
        {"seconds": [1.25, 0.5], "error": 1e-12},
        {"peak": 128},
    ]
    assert result["diagnostics"] == ["FAILED allocation budget"]


def publication_inputs(tmp_path):
    evidence = new_evidence(
        tier="cpu", subject="storage-protocol-test", inputs_hash="b" * 64
    )
    evidence.update(
        revision="a" * 40,
        device="CPU test fixture",
        backend_selected="cpu",
        toolchain={"python": "test"},
    )
    evidence["hashes"] = dict.fromkeys(evidence["hashes"], "c" * 64)
    evidence["stages"]["numerical"] = outcome("pass")
    evidence["block_errors"] = {
        "fixture": block_error([1.0], [1.0], atol=1e-11, rtol=1e-10)
    }
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    return {
        "source": {"revision": evidence["revision"], "dirty": False},
        "reproduction": {"command": ["python", "test_fixture.py"]},
        "files": [{"path": "evidence.json", "role": "evidence"}],
        "decision": {
            "status": "accepted",
            "scope": "numerical",
            "reason": "Exact independent test fixture",
        },
        "archives": [],
    }


def test_publication_selects_exact_bytes_and_never_overwrites(tmp_path):
    spec = publication_inputs(tmp_path)
    (tmp_path / "stdout.log").write_text("not selected")
    output = tmp_path / "published"
    assert publish(tmp_path, spec, output) == output
    assert sorted(p.name for p in output.iterdir()) == [
        "evidence.json",
        "publication.json",
    ]
    manifest = json.loads((output / "publication.json").read_text())
    data = (tmp_path / "evidence.json").read_bytes()
    assert (output / "evidence.json").read_bytes() == data
    validate_publication(manifest, {"evidence.json": data})
    with pytest.raises(FileExistsError):
        publish(tmp_path, spec, output)
    with pytest.raises(ValueError, match="checksum"):
        validate_publication(manifest, {"evidence.json": data + b" "})


@pytest.mark.parametrize(
    "damage",
    [
        "traversal",
        "symlink",
        "transient",
        "dirty",
        "revision",
        "false-promotion",
        "expiry",
        "duplicates",
    ],
)
def test_invalid_publication_fails_before_creating_output(tmp_path, damage):
    spec = publication_inputs(tmp_path)
    if damage == "traversal":
        spec["files"][0]["path"] = "../evidence.json"
    elif damage == "symlink":
        (tmp_path / "link").symlink_to(__file__)
        spec["files"].append({"path": "link", "role": "input"})
    elif damage == "transient":
        (tmp_path / "stdout.log").write_text("debug")
        spec["files"].append({"path": "stdout.log", "role": "summary"})
    elif damage == "dirty":
        spec["source"]["dirty"] = True
    elif damage == "revision":
        spec["source"]["revision"] = "d" * 40
    elif damage == "false-promotion":
        spec["decision"]["scope"] = "performance"
    elif damage == "duplicates":
        spec["files"] *= 2
    else:
        spec["archives"] = [
            {
                "uri": "https://example.org/artifacts/1",
                "sha256": "a" * 64,
                "bytes": 1,
                "retention": "14 days",
                "required_for_reproduction": True,
            }
        ]
    with pytest.raises(ValueError):
        publish(tmp_path, spec, tmp_path / "published")
    assert not (tmp_path / "published").exists()


def test_committed_publications_retain_valid_scientific_gates():
    """Manually edited publications must meet the same gates as the CLI."""
    for path in (ROOT / "benchmarks/results").rglob("publication.json"):
        manifest = json.loads(path.read_text())
        files = {
            entry["path"]: (path.parent / entry["path"]).read_bytes()
            for entry in manifest["files"]
        }
        validate_publication(manifest, files)


@pytest.mark.parametrize("damage", ["numerical", "attachment", "missing-tolerance"])
def test_publication_rejects_false_numerical_pass_or_lost_measurements(
    tmp_path, damage
):
    spec = publication_inputs(tmp_path)
    path = tmp_path / "evidence.json"
    evidence = json.loads(path.read_text())
    if damage == "numerical":
        evidence["block_errors"]["fixture"]["passed"] = False
    elif damage == "missing-tolerance":
        evidence["block_errors"]["fixture"].pop("atol")
    else:
        evidence["attachments"] = [
            {
                "path": "samples.json",
                "sha256": "b" * 64,
                "kind": "raw",
                "schema_version": 1,
            }
        ]
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        publish(tmp_path, spec, tmp_path / "published")
    assert not (tmp_path / "published").exists()


def test_migration_keeps_exported_profiler_evidence_and_xc_manifest():
    audit = json.loads(
        (ROOT / "benchmarks/results/retention-238/migration.json").read_text()
    )
    assert audit["removed_files"] == len(audit["files"])
    assert audit["removed_bytes"] == sum(row["bytes"] for row in audit["files"])
    for family in audit["summaries"].values():
        for record in family.values():
            for path in record.get("retained_exports", []):
                assert (ROOT / path).is_file()
    directory = ROOT / "benchmarks/results/xc-expressions-161"
    for path, expected in json.loads((directory / "manifest.json").read_text()).items():
        assert digest((directory / path).read_bytes()) == expected

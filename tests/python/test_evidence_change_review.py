"""Incoming evidence budgets cannot be hidden by deletions or unstaged edits."""

import json
import subprocess
import typing

import pytest

from tools.vibeqc_validation.retention import check, digest, tracked_blobs
from tools.vibeqc_validation.retention_review import (
    campaign_inventory,
    raw_json_markers,
    review_changes,
)

ROOT = "benchmarks/results/"


def policy(limit: typing.Any = 16, **exceptions: typing.Any) -> typing.Any:
    return {
        "schema": "vibeqc.retention-policy.v1",
        "review_size_bytes": 1 << 20,
        "change_review_max_bytes": limit,
        "exceptions": exceptions,
    }


def exception(data: typing.Any, *, review_change: typing.Any = False) -> typing.Any:
    return {
        "owner": "issue488-test",
        "reason": "Exact independent measurement inputs required for replay",
        "sha256": digest(data),
        "review_change": review_change,
    }


def test_complete_modified_bytes_count_and_deletions_are_not_credits() -> None:
    before = {ROOT + "old.json": b"0" * 100, ROOT + "changed.json": b"1" * 20}
    after = {
        ROOT + "changed.json": b"1" * 10,
        ROOT + "new.json": b"2" * 7,
        "tests/reference_data/oracle.json": b"3" * 100,
    }
    report = review_changes(before, after, policy())
    assert report["changed_bytes"] == report["review_bytes"] == 17
    assert report["changed_files"] == 2
    assert report["removed_bytes"] == 100
    assert report["removed_files"] == 1
    assert "17 new/modified bytes exceeds 16-byte" in report["errors"][0]
    assert review_changes(before, after, policy(17))["errors"] == []


def test_unchanged_history_does_not_consume_new_review_budget() -> None:
    blobs = {ROOT + "historical.json": b' {"launch_records":[{"duration":1}]} '}
    report = review_changes(blobs, blobs, policy(1))
    assert report["changed_files"] == report["review_bytes"] == 0
    assert report["errors"] == []


def test_many_small_files_and_renames_cannot_escape() -> None:
    before = {ROOT + "old.json": b"1" * 16}
    after = {ROOT + "new.json": b"1" * 16, ROOT + "publication.json": b"2"}
    report = review_changes(before, after, policy())
    assert report["review_bytes"] == 17
    assert report["errors"]
    assert all(row["change"] == "added" for row in report["files"])


@pytest.mark.parametrize("limit", [None, 0, -1, True, 2.5, "16"])
def test_invalid_change_budget_fails_closed(limit: typing.Any) -> None:
    with pytest.raises(ValueError, match="change_review_max_bytes"):
        review_changes({}, {}, policy(limit))


@pytest.mark.parametrize("damage", [None, "hash", "owner", "reason", "flag"])
def test_review_exception_is_opt_in_and_binds_exact_bytes(
    damage: typing.Any,
) -> None:
    path, data = ROOT + "sample.json", b"x" * 17
    entry = exception(data, review_change=True)
    if damage == "hash":
        entry["sha256"] = "a" * 64
    elif damage in {"owner", "reason"}:
        entry[damage] = " "
    elif damage == "flag":
        entry["review_change"] = "true"
    report = review_changes({}, {path: data}, policy(**{path: entry}))
    assert bool(report["errors"]) == (damage is not None)
    assert report["exempt_bytes"] == (17 if damage is None else 0)
    if damage is None:
        assert review_changes({}, {path: data + b"x"}, policy(**{path: entry}))[
            "errors"
        ]


def test_ordinary_exception_is_not_implicitly_a_change_budget_waiver() -> None:
    path, data = ROOT + "sample.json", b"x" * 17
    report = review_changes({}, {path: data}, policy(**{path: exception(data)}))
    assert report["review_bytes"] == 17
    assert report["errors"]


def test_review_exception_never_waives_checkout_or_hard_file_limits() -> None:
    path, data = ROOT + "sample.json", b"x" * ((1 << 20) + 1)
    rules = policy(**{path: exception(data, review_change=True)})
    rules["benchmark_results_max_bytes"] = 16
    assert review_changes({}, {path: data}, rules)["errors"] == []
    errors = check({path: data}, rules)
    assert any("hard" in error for error in errors)
    assert any("aggregate" in error for error in errors)


def test_invalid_exception_flag_is_rejected_even_without_new_evidence() -> None:
    data, path = b"1", ROOT + "sample.json"
    entry = {**exception(data), "review_change": 1}
    assert any(
        "boolean" in error for error in check({path: data}, policy(**{path: entry}))
    )


@pytest.mark.parametrize("field", ["launch_records", "traceEvents"])
def test_raw_profiler_json_needs_reason_even_when_renamed(
    field: typing.Any,
) -> None:
    path = ROOT + "misleading-summary.json"
    data = json.dumps({"nested": [{field: [{"duration": 1}]}]}).encode()
    assert raw_json_markers(path, data) == [field]
    assert review_changes({}, {path: data}, policy(4096))["errors"]
    assert (
        review_changes({}, {path: data}, policy(4096, **{path: exception(data)}))[
            "errors"
        ]
        == []
    )


@pytest.mark.parametrize(
    "data", [b'{"forces":[1,2]}', b'{"launch_records":[]}', b"not json"]
)
def test_no_blanket_numeric_array_ban(data: typing.Any) -> None:
    assert raw_json_markers(ROOT + "result.json", data) == []


def test_campaign_audit_does_not_call_arbitrary_json_accepted() -> None:
    path, data = ROOT + "legacy/array.npz", b"scientific-array"
    blobs = {
        path: data,
        ROOT + "legacy/negative.json": b'{"passed":false}',
        ROOT + "legacy/run.log": b"temporary",
        ROOT + "README.md": b"index",
        "tests/reference_data/permanent.json": b"reference",
    }
    report = campaign_inventory(blobs, policy(**{path: exception(data)}))
    assert report["total_files"] == 4
    assert report["total_bytes"] == sum(
        len(data) for path, data in blobs.items() if path.startswith(ROOT)
    )
    statuses = {row["path"]: row["review_status"] for row in report["files"]}
    assert statuses[path] == "hash-justified"
    assert statuses[ROOT + "legacy/negative.json"] == "manual-review"
    assert statuses[ROOT + "legacy/run.log"] == "transient-or-build"
    assert report["families"]["legacy"]["files"] == 3
    assert report["families"]["(root)"]["files"] == 1
    assert campaign_inventory(blobs, policy()) == campaign_inventory(
        dict(reversed(list(blobs.items()))), policy()
    )


@pytest.mark.parametrize(
    "damage", [None, "missing", "changed", "duplicate", "traversal", "malformed"]
)
def test_audit_marks_only_complete_hash_bound_publications(
    damage: typing.Any,
) -> None:
    prefix, data = ROOT + "bundle/", b"independently validated elsewhere"
    entries = [{"path": "data.json", "bytes": len(data), "sha256": digest(data)}]
    manifest = {"schema": "vibeqc.benchmark-publication.v1", "files": entries}
    blobs = {prefix + "data.json": data}
    if damage == "missing":
        blobs.clear()
    elif damage == "changed":
        blobs[prefix + "data.json"] += b" "
    elif damage == "duplicate":
        entries *= 2
    elif damage == "traversal":
        entries[0]["path"] = "../data.json"
    elif damage == "malformed":
        manifest["files"] = None
    blobs[prefix + "publication.json"] = json.dumps(manifest).encode()
    report = campaign_inventory(blobs, policy())
    assert all(
        (row["review_status"] == "publication-bound") == (damage is None)
        for row in report["files"]
    )


def git(root: typing.Any, *args: typing.Any) -> typing.Any:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def test_change_cli_reads_staged_bytes_and_never_skips_aggregate(
    tmp_path: typing.Any, monkeypatch: typing.Any, capsys: typing.Any
) -> None:
    from tools import evidence as cli

    git(tmp_path, "init", "-q")
    rules = policy(2)
    rules["benchmark_results_max_bytes"] = 100
    policy_path = tmp_path / "benchmarks/evidence-policy.json"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(json.dumps(rules))
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    base = git(tmp_path, "rev-parse", "HEAD")
    path = tmp_path / ROOT / "new.json"
    path.parent.mkdir(parents=True)
    path.write_text("123")
    git(tmp_path, "add", ".")
    path.write_text("1")  # Not staged: cannot hide the three bytes being committed.
    output = tmp_path / ".artifacts/review.json"
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "evidence.py",
            "--root",
            str(tmp_path),
            "check-change",
            "--base",
            base,
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 1
    assert json.loads(output.read_text())["review_bytes"] == 3
    assert "3 new/modified bytes exceeds 2-byte" in capsys.readouterr().out
    rules["change_review_max_bytes"] = 100
    rules["benchmark_results_max_bytes"] = 2
    policy_path.write_text(json.dumps(rules))
    git(tmp_path, "add", "benchmarks/evidence-policy.json")
    assert cli.main() == 1
    assert "aggregate budget" in capsys.readouterr().out
    assert any(
        "aggregate budget" in error
        for error in json.loads(output.read_text())["errors"]
    )
    with pytest.raises(subprocess.CalledProcessError):
        tracked_blobs(tmp_path, "missing-base-do-not-fetch")


def test_inventory_git_commands_cannot_lazily_fetch(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    calls = []

    def run(command: typing.Any, **kwargs: typing.Any) -> typing.Any:
        calls.append(kwargs["env"])
        return b""

    monkeypatch.setattr(subprocess, "check_output", run)
    assert tracked_blobs(tmp_path) == {}
    assert calls
    assert all(
        env["GIT_NO_LAZY_FETCH"] == "1" and env["GIT_ALLOW_PROTOCOL"] == ""
        for env in calls
    )

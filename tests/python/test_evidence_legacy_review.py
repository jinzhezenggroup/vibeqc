"""Legacy large benchmark evidence must stay hash-bound and explicitly reviewed."""

from __future__ import annotations

import json
import typing
from pathlib import Path

import pytest

from tools.restore_retained_evidence import _records
from tools.vibeqc_validation.retention import (
    POLICY_PATH,
    check,
    classify,
    digest,
    tracked_blobs,
)
from tools.vibeqc_validation.retention_review import (
    campaign_inventory,
    large_legacy_review_errors,
)

RESULT_ROOT = "benchmarks/results/"
REVIEW_PATH = "benchmarks/legacy-evidence-review.json"
DOCUMENT = "docs/legacy-review.md"


def policy(threshold: typing.Any = 16) -> typing.Any:
    return {
        "schema": "vibeqc.retention-policy.v1",
        "review_size_bytes": 1 << 20,
        "exceptions": {},
        "legacy_large_review_path": REVIEW_PATH,
        "legacy_large_review_threshold_bytes": threshold,
    }


def review_for(
    data: typing.Any = b"large-record", threshold: typing.Any = 8
) -> typing.Any:
    path = RESULT_ROOT + "legacy/result.json"
    review = {
        "schema": "vibeqc.legacy-large-evidence-review.v1",
        "threshold_bytes": threshold,
        "scope": "storage review only",
        "families": {
            "legacy": {
                "owner": "legacy",
                "review_document": DOCUMENT,
                "reason": "Reviewed retained endpoint evidence.",
            }
        },
        "files": [
            {
                "path": path,
                "family": "legacy",
                "bytes": len(data),
                "sha256": digest(data),
                "classification": "required-compact-accepted-evidence",
                "role": "endpoint-or-numerical-samples",
            }
        ],
    }
    blobs = {path: data, DOCUMENT: b"# Legacy review\n"}
    return path, review, blobs


def checked_blobs(
    data: typing.Any = b"large-record", threshold: typing.Any = 8
) -> typing.Any:
    path, review, blobs = review_for(data, threshold)
    blobs[REVIEW_PATH] = json.dumps(review).encode()
    return path, review, blobs


def test_valid_large_review_covers_exact_retained_bytes() -> None:
    path, review, blobs = review_for(threshold=8)
    assert large_legacy_review_errors(blobs, review, 8) == []
    complete = {**blobs, REVIEW_PATH: json.dumps(review).encode()}
    assert check(complete, policy(8)) == []
    statuses = {
        row["path"]: row["review_status"]
        for row in campaign_inventory(complete, policy(8))["files"]
    }
    assert statuses[path] == "legacy-large-reviewed"


@pytest.mark.parametrize(
    "damage",
    ["hash", "bytes", "missing-row", "family", "classification", "role", "document"],
)
def test_large_review_fails_closed_on_stale_or_incomplete_identity(
    damage: typing.Any,
) -> None:
    path, review, blobs = review_for(threshold=8)
    row = review["files"][0]
    if damage == "hash":
        row["sha256"] = "0" * 64
    elif damage == "bytes":
        row["bytes"] += 1
    elif damage == "missing-row":
        review["files"] = []
    elif damage == "family":
        row["family"] = "other"
    elif damage == "classification":
        row["classification"] = "transient/raw supporting artifact"
    elif damage == "role":
        row["role"] = " "
    else:
        blobs.pop(DOCUMENT)
    errors = large_legacy_review_errors(blobs, review, 8)
    assert errors, damage
    if damage == "missing-row":
        assert any("unreviewed large benchmark evidence" in error for error in errors)
    if damage == "hash":
        assert any("bytes changed" in error for error in errors)
    assert path in blobs


def test_policy_requires_review_file_and_matching_threshold() -> None:
    _, review, blobs = review_for(threshold=8)
    errors = check(blobs, policy(8))
    assert any("missing legacy large-evidence review" in error for error in errors)
    blobs[REVIEW_PATH] = json.dumps(review).encode()
    rules = policy(9)
    assert any("threshold differs" in error for error in check(blobs, rules))


def test_new_unreviewed_large_file_is_rejected_without_deletion_credit() -> None:
    _, review, blobs = checked_blobs(threshold=8)
    blobs[RESULT_ROOT + "new/large.json"] = b"new-large"
    errors = check(blobs, policy(8))
    assert any("unreviewed large benchmark evidence" in error for error in errors)
    assert large_legacy_review_errors(blobs, review, 8)


def test_reference_inputs_are_an_explicit_supported_classification() -> None:
    path, review, blobs = review_for(threshold=8)
    review["files"][0]["classification"] = "test-reference-input"
    review["files"][0]["role"] = "reference-input"
    assert large_legacy_review_errors(blobs, review, 8) == []
    assert path.endswith("result.json")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_checked_in_large_review_covers_every_large_result() -> None:
    blobs = tracked_blobs(REPOSITORY_ROOT)
    rules = json.loads(blobs[POLICY_PATH])
    assert check(blobs, rules) == []
    threshold = rules["legacy_large_review_threshold_bytes"]
    report = campaign_inventory(blobs, rules)
    assert not [
        row
        for row in report["files"]
        if row["bytes"] >= threshold and row["review_status"] == "manual-review"
    ]


def test_retention_488_snapshot_binds_removed_checkout_bytes() -> None:
    manifest = (
        REPOSITORY_ROOT / "benchmarks/results/retention-488/snapshot.manifest.json"
    )
    audit = json.loads(manifest.read_text(encoding="utf-8"))
    records = _records(manifest)
    assert audit["schema"] == "vibeqc.git-snapshot.v1"
    assert audit["history_rewritten"] is False
    assert len(records) == audit["file_count"] == audit["moved_files"] == 48
    assert (
        sum(record["bytes"] for record in records)
        == audit["total_bytes"]
        == audit["moved_bytes"]
        == 3_759_783
    )
    assert all(record["checkout"] == "git-history" for record in records)
    assert all(not (REPOSITORY_ROOT / record["path"]).exists() for record in records)


@pytest.mark.parametrize(
    "path",
    [
        RESULT_ROOT + "run/case.progress.jsonl",
        RESULT_ROOT + "run/case.progress.jsonl.gz",
        RESULT_ROOT + "run/case.journal.jsonl",
        RESULT_ROOT + "run/case.journal.jsonl.gz",
    ],
)
def test_execution_flow_streams_are_transient(path: typing.Any) -> None:
    assert classify(path) == "transient"
    rules = {
        "schema": "vibeqc.retention-policy.v1",
        "review_size_bytes": 1 << 20,
        "exceptions": {},
    }
    assert check({path: b"execution-flow"}, rules)


def test_generic_jsonl_is_not_blanket_transient() -> None:
    assert classify(RESULT_ROOT + "run/scientific-samples.jsonl") == "accepted-evidence"


def test_checked_in_results_have_no_execution_flow_streams() -> None:
    blobs = tracked_blobs(REPOSITORY_ROOT)
    endings = (
        ".progress.jsonl",
        ".progress.jsonl.gz",
        ".journal.jsonl",
        ".journal.jsonl.gz",
    )
    assert not [
        path
        for path in blobs
        if path.startswith(RESULT_ROOT) and path.lower().endswith(endings)
    ]

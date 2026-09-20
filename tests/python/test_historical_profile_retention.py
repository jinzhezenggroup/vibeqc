"""Compacted historical profiles keep all aggregates and offline recovery identity."""

import hashlib
import json
import typing
from pathlib import Path

import pytest

from tools.restore_retained_evidence import _records

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/results/retention-checkout/snapshot.manifest.json"
PROFILES = sorted(
    ROOT.glob("benchmarks/results/issue385-df-signatures/*/*-nsys-summary.json")
) + [
    ROOT / "benchmarks/results/issue404-407-df/initial-value-batch.json",
    ROOT / "benchmarks/results/issue404-407-df/value-batch.json",
]


def test_profile_inventory_is_complete() -> None:
    assert len(PROFILES) == 8
    snapshot = json.loads(MANIFEST.read_text())
    compacted = {
        entry["path"] for entry in snapshot["files"] if entry.get("compacted_fields")
    }
    assert compacted == {path.relative_to(ROOT).as_posix() for path in PROFILES}
    assert snapshot["compacted_files"] == len(compacted)


@pytest.mark.parametrize("path", PROFILES, ids=lambda path: str(path.relative_to(ROOT)))
def test_retained_aggregates_and_recovery_identity(path: typing.Any) -> None:
    # This validates the checkout without fetching any historical Git objects.
    records = {entry["path"]: entry for entry in _records(MANIFEST)}
    relative = path.relative_to(ROOT).as_posix()
    entry = records[relative]
    summary = json.loads(path.read_text())
    retention = summary.pop("retention")
    assert all(field not in summary for field in entry["compacted_fields"])
    assert retention["omitted_fields"] == entry["compacted_fields"]
    assert retention["source_path"] == relative
    assert retention["source_revision"] == entry["revision"]
    assert retention["source_bytes"] == entry["bytes"]
    assert retention["source_sha256"] == entry["sha256"]
    if "launch_records" in entry["compacted_fields"]:
        assert (
            retention["omitted_launch_record_count"]
            == entry["omitted_launch_record_count"]
        )
        assert retention["omitted_launch_record_count"] > 0
    assert (ROOT / retention["manifest"]).resolve() == MANIFEST.resolve()
    canonical = json.dumps(
        summary, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == entry["retained_fields_sha256"]


def test_snapshot_moved_totals_and_compacted_savings() -> None:
    snapshot = json.loads(MANIFEST.read_text())
    records = _records(MANIFEST)
    moved = [entry for entry in records if entry["checkout"] == "git-history"]
    assert snapshot["moved_files"] == len(moved)
    assert snapshot["moved_bytes"] == sum(entry["bytes"] for entry in moved)
    compacted = [entry for entry in records if entry.get("compacted_fields")]
    assert snapshot["compacted_bytes_saved"] == sum(
        entry["bytes"]
        - len((ROOT / entry["path"]).read_text(encoding="utf-8").encode())
        for entry in compacted
    )

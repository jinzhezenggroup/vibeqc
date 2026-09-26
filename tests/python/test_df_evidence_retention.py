"""Keep integration compaction lossless and production evidence available."""

import hashlib
import json
import typing
from pathlib import Path

from tools.restore_retained_evidence import _records

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/results/retention-df-integration/snapshot.manifest.json"


def test_df_snapshot_catalog_and_exact_removal_set() -> None:
    catalog = json.loads(MANIFEST.read_text())
    entries = _records(MANIFEST)
    assert catalog["source_revision"] == "5a7fdeb2689553c0a304dad3338ba184d850ef60"
    assert len(entries) == catalog["file_count"]
    removed = [row for row in entries if row["checkout"] == "git-history"]
    assert len(removed) == 29
    assert sum(row["bytes"] for row in removed) == 4_754_152
    assert all(not (ROOT / row["path"]).exists() for row in removed)


def test_retained_scientific_records_keep_original_bytes() -> None:
    for row in _records(MANIFEST):
        if row["checkout"] != "retained" or row["path"].endswith(".md"):
            continue
        data = (ROOT / row["path"]).read_bytes()
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]


def test_production_df_provenance_keeps_current_files() -> None:
    policy = json.loads(
        (
            ROOT / "python/vibeqc_compiler/integral/production_df_derivatives.json"
        ).read_text()
    )

    def visit(value: typing.Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.startswith("benchmarks/results/"):
            assert (ROOT / value).is_file(), value

    visit(policy)

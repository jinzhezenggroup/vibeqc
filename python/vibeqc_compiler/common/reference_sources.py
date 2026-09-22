"""Exact, audited import-only migrations for retained reference exporters.

These receipts do not rebind a measured record to a new execution. Numerical
archives and their metadata keep the exporter hashes from the original run.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .provenance import file_hash


def reference_source_matches(root: Path, path: str, recorded_sha256: str) -> bool:
    """Match current bytes or one explicitly reviewed historical-to-current pair.

    Any unrecorded source edit remains a mismatch, including import-only edits.
    This is intentionally separate from file_hash and artifact validation.
    """
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    try:
        actual = file_hash(root / relative)
    except OSError:
        return False
    if actual == recorded_sha256:
        return True
    receipt = root / "tests/reference_data/reference_source_import_migrations.json"
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict) or payload.get("schema") != (
        "vibeqc.reference-source-import-migrations.v1"
    ):
        return False
    records = payload.get("sources")
    if not isinstance(records, list):
        return False
    return any(
        isinstance(record, dict)
        and record.get("path") == relative.as_posix()
        and record.get("historical_sha256") == recorded_sha256
        and record.get("current_sha256") == actual
        for record in records
    )

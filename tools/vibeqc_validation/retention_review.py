"""Review new evidence and audit legacy families without certifying or deleting it.

This layer is stdlib-only so both index checks and source inventories can run
without a native build, NumPy, network access or an artifact service.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import PurePosixPath

from .retention import RESULT_ROOT, classify, digest, safe_relative


def _justified(path: str, data: bytes, policy: dict) -> bool:
    entry = policy.get("exceptions", {}).get(path, {})
    return (
        entry.get("sha256") == digest(data)
        and isinstance(entry.get("owner"), str)
        and bool(entry["owner"].strip())
        and isinstance(entry.get("reason"), str)
        and bool(entry["reason"].strip())
    )


def raw_json_markers(path: str, data: bytes) -> list[str]:
    """Recognize full profiler row arrays, not arbitrary numerical arrays.

    These markers request a human storage decision; they do not mean the data
    lack scientific value. NPZ measurements and negative results are not banned.
    """
    if not path.endswith(".json"):
        return []
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError, RecursionError):
        return []
    markers, pending = set(), [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, child in item.items():
                if (
                    key in {"launch_records", "traceEvents"}
                    and isinstance(child, list)
                    and child
                ):
                    markers.add(key)
                if isinstance(child, (list, dict)):
                    pending.append(child)
        elif isinstance(item, list):
            pending.extend(child for child in item if isinstance(child, (list, dict)))
    return sorted(markers)


def review_changes(
    before: dict[str, bytes], after: dict[str, bytes], policy: dict
) -> dict:
    """Count complete added/modified blobs, not net growth or diff line counts.

    Deletions cannot subsidize a new run dump. Unchanged historical evidence and
    permanent reference roots do not consume the new-evidence review budget.
    A hash-pinned scientific exception can explicitly waive *this* budget, but
    never the separate per-file or aggregate checkout limits.
    """
    limit = policy.get("change_review_max_bytes")
    if type(limit) is not int or limit <= 0:
        raise ValueError("change_review_max_bytes must be a positive integer")
    rows, errors = [], []
    for path, data in sorted(after.items()):
        if not path.startswith(RESULT_ROOT) or before.get(path) == data:
            continue
        justified = _justified(path, data, policy)
        exempt = justified and policy["exceptions"][path].get("review_change") is True
        markers = raw_json_markers(path, data)
        if markers and not justified:
            errors.append(
                f"{path}: raw profiler arrays ({', '.join(markers)}); publish a "
                "compact summary or justify these exact bytes in the retention policy"
            )
        rows.append(
            {
                "path": path,
                "change": "modified" if path in before else "added",
                "bytes": len(data),
                "previous_bytes": len(before.get(path, b"")),
                "sha256": digest(data),
                "review_exempt": exempt,
            }
        )
    total = sum(row["bytes"] for row in rows)
    charged = sum(row["bytes"] for row in rows if not row["review_exempt"])
    if charged > limit:
        errors.append(
            f"{RESULT_ROOT}: {charged} new/modified bytes exceeds {limit}-byte "
            "change review budget (deletions are not credits); retain compact "
            "evidence or add hash-pinned owner/reason/review_change exceptions"
        )
    removed = {
        path: data
        for path, data in before.items()
        if path.startswith(RESULT_ROOT) and path not in after
    }
    return {
        "schema": "vibeqc.evidence-change-review.v1",
        "changed_files": len(rows),
        "changed_bytes": total,
        "review_bytes": charged,
        "exempt_bytes": total - charged,
        "review_max_bytes": limit,
        "removed_files": len(removed),
        "removed_bytes": sum(map(len, removed.values())),
        "files": rows,
        "errors": errors,
    }


def _publication_members(blobs: dict[str, bytes]) -> set[str]:
    """Find complete hash-bound bundles, not independently validated claims."""
    covered = set()
    for path, data in blobs.items():
        if not path.startswith(RESULT_ROOT) or not path.endswith("/publication.json"):
            continue
        try:
            manifest = json.loads(data)
            if manifest.get("schema") != "vibeqc.benchmark-publication.v1":
                continue
            entries = manifest["files"]
            parent = str(PurePosixPath(path).parent)
            members = [parent + "/" + safe_relative(row["path"]) for row in entries]
            if not members or len(set(members)) != len(members) or path in members:
                continue
            if all(
                member in blobs
                and digest(blobs[member]) == row["sha256"]
                and type(row["bytes"]) is int
                and len(blobs[member]) == row["bytes"]
                for member, row in zip(members, entries, strict=True)
            ):
                covered.update([path, *members])
        except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
            continue
    return covered


LEGACY_LARGE_REVIEW_SCHEMA = "vibeqc.legacy-large-evidence-review.v1"
LEGACY_CLASSIFICATIONS = {
    "required-compact-accepted-evidence",
    "test-reference-input",
}


def _family(path: str) -> str:
    relative = path[len(RESULT_ROOT) :]
    return relative.split("/", 1)[0] if "/" in relative else "(root)"


def large_legacy_review_errors(
    blobs: dict[str, bytes], review: dict, threshold: int
) -> list[str]:
    """Require exact review identities for every retained large result blob.

    This is a storage review, not scientific re-acceptance.  Family reasons and
    review documents explain why large legacy bytes remain in the checkout; each
    file hash prevents a reviewed historical record from drifting silently.
    """
    errors: list[str] = []
    if review.get("schema") != LEGACY_LARGE_REVIEW_SCHEMA:
        return ["legacy large-evidence review has unsupported schema"]
    if type(threshold) is not int or threshold <= 0:
        return ["legacy_large_review_threshold_bytes must be a positive integer"]
    if review.get("threshold_bytes") != threshold:
        errors.append("legacy large-evidence review threshold differs from policy")
    families = review.get("families")
    rows = review.get("files")
    if not isinstance(families, dict) or not isinstance(rows, list):
        return [*errors, "legacy large-evidence review requires families and files"]

    for family, entry in sorted(families.items()):
        if not isinstance(entry, dict):
            errors.append(f"legacy review family {family!r} is not an object")
            continue
        for field in ("owner", "reason", "review_document"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"legacy review family {family!r} requires {field}")
        document = entry.get("review_document")
        if isinstance(document, str):
            try:
                safe_relative(document)
            except ValueError as error:
                errors.append(str(error))
            else:
                if document not in blobs:
                    errors.append(
                        f"legacy review family {family!r} missing review document {document}"
                    )

    indexed: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("legacy large-evidence review file row is not an object")
            continue
        path = row.get("path")
        try:
            safe_relative(path)
        except (TypeError, ValueError) as error:
            errors.append(str(error))
            continue
        if not path.startswith(RESULT_ROOT):
            errors.append(f"{path}: legacy review path is outside benchmark results")
            continue
        if path in indexed:
            errors.append(f"{path}: duplicate legacy large-evidence review row")
            continue
        indexed[path] = row
        expected_family = _family(path)
        if row.get("family") != expected_family:
            errors.append(f"{path}: legacy review family does not match path")
        if expected_family not in families:
            errors.append(f"{path}: legacy review family metadata is missing")
        if row.get("classification") not in LEGACY_CLASSIFICATIONS:
            errors.append(f"{path}: unsupported legacy evidence classification")
        if not isinstance(row.get("role"), str) or not row["role"].strip():
            errors.append(f"{path}: legacy review requires a file role")
        data = blobs.get(path)
        if data is None:
            errors.append(f"{path}: reviewed large evidence is missing")
            continue
        if len(data) != row.get("bytes") or digest(data) != row.get("sha256"):
            errors.append(f"{path}: reviewed large evidence bytes changed")
        if len(data) < threshold:
            errors.append(
                f"{path}: legacy review row is below the configured threshold"
            )

    expected = {
        path
        for path, data in blobs.items()
        if path.startswith(RESULT_ROOT) and len(data) >= threshold
    }
    missing = sorted(expected - indexed.keys())
    extra = sorted(indexed.keys() - expected)
    if missing:
        errors.append("unreviewed large benchmark evidence: " + ", ".join(missing))
    if extra:
        errors.append("stale legacy large-evidence review rows: " + ", ".join(extra))
    return errors


def large_legacy_review_members(blobs: dict[str, bytes], policy: dict) -> set[str]:
    """Return exact reviewed large paths only when the whole review is valid."""
    path = policy.get("legacy_large_review_path")
    threshold = policy.get("legacy_large_review_threshold_bytes")
    if not isinstance(path, str) or type(threshold) is not int or path not in blobs:
        return set()
    try:
        review = json.loads(blobs[path])
    except (ValueError, UnicodeError, TypeError):
        return set()
    if large_legacy_review_errors(blobs, review, threshold):
        return set()
    return {row["path"] for row in review["files"]}


def campaign_inventory(blobs: dict[str, bytes], policy: dict) -> dict:
    """List every results family and its actual retention-review evidence.

    A filename or JSON suffix alone is NOT an acceptance record. Unclassified
    legacy files need manual review; this inventory never authorizes deletion.
    Hash-bound publications still need the normal scientific validation suite.
    """
    covered = _publication_members(blobs)
    large_reviewed = large_legacy_review_members(blobs, policy)
    rows, families = [], {}
    for path, data in sorted(blobs.items()):
        if not path.startswith(RESULT_ROOT):
            continue
        family = _family(path)
        category = classify(path)
        justified = _justified(path, data, policy)
        status = (
            "hash-justified"
            if justified
            else "publication-bound"
            if path in covered
            else "legacy-large-reviewed"
            if path in large_reviewed
            else "transient-or-build"
            if category in {"transient", "generated-build"}
            else "manual-review"
        )
        row = {
            "path": path,
            "family": family,
            "bytes": len(data),
            "sha256": digest(data),
            "review_status": status,
            "raw_json_markers": raw_json_markers(path, data),
        }
        if justified:
            entry = policy["exceptions"][path]
            row.update(owner=entry["owner"], reason=entry["reason"])
        rows.append(row)
        summary = families.setdefault(
            family,
            {
                "files": 0,
                "bytes": 0,
                "status_files": Counter(),
                "status_bytes": Counter(),
            },
        )
        summary["files"] += 1
        summary["bytes"] += len(data)
        summary["status_files"][status] += 1
        summary["status_bytes"][status] += len(data)
    return {
        "schema": "vibeqc.evidence-campaign-inventory.v1",
        "scope": "tracked checkout blobs, not Git history or scientific acceptance",
        "total_files": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "families": families,
        "files": rows,
    }

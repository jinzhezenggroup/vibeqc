"""Deterministically gzip bulky JSON members of selected evidence publications."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from tools.vibeqc_validation.retention import safe_relative

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "benchmarks/legacy-evidence-review.json"
THRESHOLD = 128 << 10
ROLES = {"evidence", "samples"}
TARGETS = (
    "benchmarks/results/density-candidates/gpu/publication.json",
    "benchmarks/results/xc-contractions/publication.json",
    "benchmarks/results/spatial-density-sources/publication.json",
    "benchmarks/results/spatial-tasks/cpu/publication.json",
    "benchmarks/results/spatial-tasks/cuda/publication.json",
    "benchmarks/results/density-sources-gpu/publication.json",
    "benchmarks/results/fock-strategies/cpu/publication.json",
    "benchmarks/results/fock-strategies/cuda/publication.json",
    "benchmarks/results/cuda-ownership/one-electron/publication.json",
    "benchmarks/results/cuda-ownership/df/publication.json",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False).encode()
        + b"\n"
    )


def compressed(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=9, mtime=0)


def _update_storage_references(
    evidence: dict, replacements: dict[str, tuple[str, bytes]]
) -> bool:
    changed = False
    for attachment in evidence.get("attachments", []):
        old = attachment.get("path")
        if old in replacements:
            new, data = replacements[old]
            attachment["path"] = new
            attachment["sha256"] = digest(data)
            if "bytes" in attachment:
                attachment["bytes"] = len(data)
            changed = True
    for entries in evidence.get("record_parts", {}).values():
        for entry in entries:
            old = entry.get("path")
            if old in replacements:
                new, data = replacements[old]
                entry.update(path=new, bytes=len(data), sha256=digest(data))
                changed = True
    return changed


def compact_publication(
    relative: str, *, check: bool = False
) -> list[tuple[str, str, bytes]]:
    manifest_path = ROOT / safe_relative(relative)
    if not manifest_path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("publication manifest path escapes the checkout")
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["files"]
    # Storage identities authenticate bytes, not paths. Validate every member
    # before reads or writes so compaction cannot move/delete another bundle.
    names = set()
    for entry in entries:
        name = safe_relative(entry["path"])
        if name in names:
            raise ValueError("duplicate publication member path")
        names.add(name)
        if not (directory / name).resolve().is_relative_to(directory.resolve()):
            raise ValueError("publication member path escapes its directory")

    # Authenticate every original member before updating any storage identity.
    # Keep these exact bytes so compaction does not re-read unchecked content.
    contents = {}
    for entry in entries:
        path = directory / entry["path"]
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
            raise ValueError(f"publication identity mismatch: {path}")
        if str(entry["path"]).endswith(".json.gz"):
            json.loads(gzip.decompress(raw))
        contents[entry["path"]] = raw

    # Already compacted publications are validated but left byte-identical.
    if any(str(entry["path"]).endswith(".json.gz") for entry in entries):
        return []

    candidates = {
        entry["path"]: entry
        for entry in entries
        if entry["role"] in ROLES
        and entry["path"].endswith(".json")
        and entry["bytes"] >= THRESHOLD
    }
    if not candidates:
        return []

    evidence_entry = next(entry for entry in entries if entry["role"] == "evidence")
    evidence_name = evidence_entry["path"]
    replacements: dict[str, tuple[str, bytes]] = {}
    changes: list[tuple[str, str, bytes]] = []

    # Compress non-evidence members first so evidence storage references can
    # point at the final compressed identities.
    for old in candidates:
        if old == evidence_name:
            continue
        source = directory / old
        data = compressed(contents[old])
        new = old + ".gz"
        replacements[old] = (new, data)
        changes.append(
            (
                source.relative_to(ROOT).as_posix(),
                (directory / new).relative_to(ROOT).as_posix(),
                data,
            )
        )

    evidence_path = directory / evidence_name
    evidence = json.loads(contents[evidence_name])
    evidence_changed = _update_storage_references(evidence, replacements)
    evidence_data = (
        json_bytes(evidence) if evidence_changed else contents[evidence_name]
    )

    if evidence_name in candidates:
        new = evidence_name + ".gz"
        packed = compressed(evidence_data)
        replacements[evidence_name] = (new, packed)
        changes.append(
            (
                evidence_path.relative_to(ROOT).as_posix(),
                (directory / new).relative_to(ROOT).as_posix(),
                packed,
            )
        )
    elif evidence_changed:
        changes.append(
            (
                evidence_path.relative_to(ROOT).as_posix(),
                evidence_path.relative_to(ROOT).as_posix(),
                evidence_data,
            )
        )

    for entry in entries:
        old = entry["path"]
        if old in replacements:
            new, data = replacements[old]
            entry.update(path=new, bytes=len(data), sha256=digest(data))
        elif old == evidence_name and evidence_changed:
            entry.update(bytes=len(evidence_data), sha256=digest(evidence_data))

    # A pre-existing companion is not ours to overwrite, even in check mode.
    for old, new, _data in changes:
        target = ROOT / new
        if old != new and (target.exists() or target.is_symlink()):
            raise FileExistsError(target)
    if check:
        return changes

    for old, new, data in changes:
        target = ROOT / new
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if old != new:
            (ROOT / old).unlink()
    manifest_path.write_bytes(json_bytes(manifest))
    return changes


def update_legacy_review(changes: list[tuple[str, str, bytes]]) -> None:
    review = json.loads(REVIEW.read_text())
    rows = {row["path"]: row for row in review["files"]}
    for old, new, data in changes:
        row = rows.pop(old, None)
        if len(data) >= review["threshold_bytes"]:
            if row is None:
                raise ValueError(
                    f"missing legacy review row for compressed evidence: {old}"
                )
            row.update(path=new, bytes=len(data), sha256=digest(data))
            rows[new] = row
    review["files"] = sorted(rows.values(), key=lambda row: row["path"])
    REVIEW.write_bytes(json_bytes(review))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the compaction would modify the checkout",
    )
    args = parser.parse_args(argv)
    changes: list[tuple[str, str, bytes]] = []
    for target in TARGETS:
        changes.extend(compact_publication(target, check=args.check))
    if args.check and changes:
        raise SystemExit("retained evidence JSON is not compacted")
    if changes:
        update_legacy_review(changes)
    print(
        f"compressed evidence publications: {len(changes)} changed files"
        if changes
        else "compressed evidence publications: up to date"
    )


if __name__ == "__main__":
    main()

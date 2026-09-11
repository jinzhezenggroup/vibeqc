"""Git-index inventory and evidence retention checks, requiring only Python.

Classification is contextual: test fixtures and audited third-party sources
are references even when their suffix also occurs in temporary run output.
Exceptions name exact bytes, so an old justification cannot admit a new log.
This policy governs storage, never numerical acceptance or kernel promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath

POLICY_PATH = "benchmarks/evidence-policy.json"
REFERENCE_ROOTS = ("tests/reference_data/", "tests/data/", "external/")
RESULT_ROOT = "benchmarks/results/"
TRANSIENT_SUFFIXES = {
    ".log",
    ".xml",
    ".chk",
    ".checkpoint",
    ".nsys-rep",
    ".ncu-rep",
    ".qdrep",
    ".qdstrm",
    ".sqlite",
    ".sqlite3",
    ".out",
    ".err",
}
BUILD_SUFFIXES = {".o", ".obj", ".so", ".a", ".dll", ".dylib", ".cubin", ".ptx", ".pyc"}
CLASSES = (
    "reference",
    "accepted-evidence",
    "transient",
    "generated-build",
    "source",
    "unknown",
)


def digest(data: bytes) -> str:
    """Return the immutable content identity used by retention exceptions."""
    return hashlib.sha256(data).hexdigest()


def safe_relative(value: str) -> str:
    """Require a canonical portable path inside an evidence bundle."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {".", "..", ""} for p in value.split("/")):
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def classify(path: str) -> str:
    """Identify storage responsibility before looking at file suffixes."""
    if path.startswith(REFERENCE_ROOTS):
        return "reference"
    p = PurePosixPath(path)
    if p.suffix.lower() in BUILD_SUFFIXES or path.startswith(
        ("build/", ".artifacts/", ".cache/")
    ):
        return "generated-build"
    if path.startswith(RESULT_ROOT):
        if "attempts" in p.parts or any(
            s.lower() in TRANSIENT_SUFFIXES for s in p.suffixes
        ):
            return "transient"
        return "accepted-evidence"
    if path.startswith(
        (
            "src/",
            "include/",
            "python/",
            "compiler/",
            "tools/",
            "tests/",
            "benchmarks/",
            "docs/",
            "assets/",
            ".github/",
        )
    ):
        return "source"
    if len(p.parts) == 1 and (
        p.suffix in {".md", ".toml", ".json", ".yaml", ".yml", ".txt"}
        or p.name in {"LICENSE", ".gitignore", ".clang-format"}
    ):
        return "source"
    return "unknown"


def tracked_blobs(root: Path, revision: str | None = None) -> dict[str, bytes]:
    """Read actual index blobs (or one revision), never dereference symlinks.

    One cat-file batch avoids a subprocess per artifact. Inspecting index bytes
    also makes partially staged edits obey the policy that will be committed.
    """
    command = (
        ["git", "ls-tree", "-rz", revision]
        if revision
        else ["git", "ls-files", "--stage", "-z"]
    )
    rows = subprocess.check_output(command, cwd=root).split(b"\0")
    entries = []
    for row in filter(None, rows):
        metadata, path = row.split(b"\t", 1)
        fields = metadata.split()
        if (revision and fields[1] != b"blob") or (not revision and fields[2] != b"0"):
            raise ValueError("inventory requires ordinary files and a resolved index")
        oid = fields[2] if revision else fields[1]
        entries.append((path.decode("utf-8"), oid))
    raw = (
        subprocess.check_output(
            ["git", "cat-file", "--batch"],
            cwd=root,
            input=b"\n".join(oid for _, oid in entries) + b"\n",
        )
        if entries
        else b""
    )
    result, offset = {}, 0
    for path, oid in entries:
        end = raw.index(b"\n", offset)
        actual_oid, kind, length = raw[offset:end].split()
        if actual_oid != oid or kind != b"blob":
            raise ValueError("unexpected Git object in index")
        offset = end + 1
        size = int(length)
        result[path] = raw[offset : offset + size]
        offset += size + 1
    return result


def inventory(blobs: dict[str, bytes]) -> dict:
    """Return deterministic per-file and per-class tracked bytes/counts."""
    groups = {key: {"files": 0, "bytes": 0} for key in CLASSES}
    rows = []
    for path, data in sorted(blobs.items()):
        category = classify(path)
        groups[category]["files"] += 1
        groups[category]["bytes"] += len(data)
        rows.append(
            {
                "path": path,
                "class": category,
                "bytes": len(data),
                "sha256": digest(data),
            }
        )
    return {"schema": "vibeqc.retention-inventory.v1", "classes": groups, "files": rows}


def check(blobs: dict[str, bytes], policy: dict) -> list[str]:
    """Reject run debris, unreviewed large files and stale exceptions.

    No global JSON/XML ban applies: references precede pattern matching and
    any other scientific file can carry an explicit, hash-pinned exception.
    The review-size guard covers all tracked files, including source/oracles.
    """
    if policy.get("schema") != "vibeqc.retention-policy.v1":
        raise ValueError("unsupported retention policy")
    limit = policy["review_size_bytes"]
    if type(limit) is not int or limit <= 0:
        raise ValueError("review_size_bytes must be positive")
    exceptions = policy["exceptions"]
    errors = []
    for path, exception in exceptions.items():
        safe_relative(path)
        if (
            not isinstance(exception.get("reason"), str)
            or not exception["reason"].strip()
        ):
            errors.append(f"{path}: exception requires a scientific/storage reason")
        if not exception.get("owner"):
            errors.append(f"{path}: exception requires an owner")
        if path not in blobs or digest(blobs[path]) != exception.get("sha256"):
            errors.append(f"{path}: stale exception (missing or changed bytes)")
    for path, data in sorted(blobs.items()):
        violations = []
        category = classify(path)
        if category in {"transient", "generated-build"}:
            violations.append(category)
        if len(data) > limit:
            violations.append(f"exceeds {limit}-byte review guard")
        if violations and path not in exceptions:
            errors.append(
                f"{path}: {', '.join(violations)}; publish compact evidence or justify exact bytes in {POLICY_PATH}"
            )
    return errors


def extract_log(data: bytes) -> dict:
    """Preserve embedded JSON measurements and diagnostic conclusions.

    Historical runners sometimes printed whole timing/resource reports to
    stdout. Recover complete JSON documents without treating progress bars or
    scheduler receipts as measurements. Keep non-progress diagnostic lines so
    failed attempts and compiler resource reports remain auditable.
    """
    text = data.decode("utf-8", errors="replace")
    documents, lines = [], []
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        end = text.find("\n", offset)
        end = len(text) if end < 0 else end + 1
        candidate = text[offset:].lstrip()
        if candidate.startswith(("{", "[")):
            try:
                value, consumed = decoder.raw_decode(candidate)
            except ValueError:
                pass
            else:
                if isinstance(value, (dict, list)):
                    documents.append(value)
                    offset = len(text) - len(candidate) + consumed
                    continue
        line = text[offset:end].strip()
        if line and not re.match(
            r"(?:srun:|SLURM_JOB_ID=|\[\d+/\d+\]|Generating |Generated:|Capture range |/tmp/)",
            line,
        ):
            lines.append(line)
        offset = end
    return {"measurements": documents, "diagnostics": list(dict.fromkeys(lines))}


def migration_audit(blobs: dict[str, bytes], revision: str) -> dict:
    """Summarize removed run artifacts without altering retained measurements.

    Git revision/path plus SHA-256 is an immutable archive reference for old
    tracked debris; new runs use the external artifact workflow instead. XML
    case failures and skip reasons are kept alongside suite counts. Profiler
    databases are referenced by checksum; their compact CSV exports stay in Git.
    """
    from xml.etree import ElementTree

    rows, summaries = [], defaultdict(dict)
    for path, data in sorted(blobs.items()):
        if classify(path) != "transient":
            continue
        row = {
            "path": path,
            "bytes": len(data),
            "sha256": digest(data),
            "archive": f"git:{revision}:{path}",
        }
        family = path.split("/")[2]
        suffix = PurePosixPath(path).suffix
        if suffix == ".log":
            summary = extract_log(data)
            row["reason"] = (
                "Structured measurements and diagnostic conclusions extracted; routine stdout removed."
            )
        elif suffix == ".xml":
            tree = ElementTree.fromstring(data)
            summary = {
                "suites": [dict(node.attrib) for node in tree.iter("testsuite")],
                "nonpassing_cases": [
                    {
                        "case": dict(case.attrib),
                        "outcome": child.tag,
                        "details": dict(child.attrib),
                        "text": child.text,
                    }
                    for case in tree.iter("testcase")
                    for child in case
                    if child.tag in {"failure", "error", "skipped"}
                ],
            }
            row["reason"] = (
                "Suite totals, failures and skips extracted; routine test XML removed."
            )
        elif any(
            s in {".nsys-rep", ".ncu-rep", ".sqlite"}
            for s in PurePosixPath(path).suffixes
        ):
            stem = (
                path.split(".sqlite")[0]
                if ".sqlite" in path
                else str(PurePosixPath(path).with_suffix(""))
            )
            summary = {
                "retained_exports": sorted(
                    p for p in blobs if p.startswith(stem) and p.endswith(".csv")
                )
            }
            if not summary["retained_exports"]:
                raise ValueError(f"profiler database has no retained export: {path}")
            row["reason"] = (
                "Profiler database archived in baseline Git objects; human-readable exports and endpoint samples retained."
            )
        else:
            raise ValueError(f"artifact needs manual audit: {path}")
        summaries[family][path] = summary
        rows.append(row)
    return {
        "schema": "vibeqc.retention-migration.v1",
        "baseline": revision,
        "removed_files": len(rows),
        "removed_bytes": sum(r["bytes"] for r in rows),
        "files": rows,
        "summaries": dict(summaries),
    }

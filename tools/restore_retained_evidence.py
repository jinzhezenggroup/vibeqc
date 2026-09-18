"""Restore verified historical evidence from existing Git objects, offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "benchmarks/results/retention-size-limit/migration.json"


def _safe_path(path: str) -> None:
    if not isinstance(path, str):
        raise TypeError("unsafe historical evidence path")
    relative = PurePosixPath(path)
    if (
        not path
        or not relative.parts
        or relative.is_absolute()
        or relative.as_posix() != path
        or ".." in relative.parts
        or "\\" in path
        or ":" in path
        or any(ord(char) < 32 for char in path)
    ):
        raise ValueError("unsafe historical evidence path")


def _records(manifest: Path | None) -> list[dict]:
    audit = json.loads(Path(manifest or AUDIT).read_text())
    if not isinstance(audit, dict):
        raise TypeError("unsupported evidence migration manifest")
    schema = audit.get("schema")
    if schema == "vibeqc.storage-migration.v1":
        records = audit.get("archives")
    elif schema in {"vibeqc.evidence-archive.v1", "vibeqc.git-snapshot.v1"}:
        records = audit.get("files")
    else:
        raise ValueError("unsupported evidence migration manifest")
    if not isinstance(records, list) or not records:
        raise ValueError("empty or invalid historical evidence inventory")
    result = []
    paths = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("invalid historical evidence identity")
        entry = dict(record)
        if schema != "vibeqc.storage-migration.v1":
            entry["revision"] = audit.get("source_revision")
        _safe_path(entry.get("path"))
        if entry["path"] in paths:
            raise ValueError("path has no unique verified migration record")
        paths.add(entry["path"])
        if (
            not isinstance(entry.get("revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", entry["revision"])
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or type(entry.get("bytes")) is not int
            or entry["bytes"] < 0
        ):
            raise ValueError("invalid historical evidence identity")
        result.append(entry)
    if any(
        parent.as_posix() in paths
        for path in paths
        for parent in PurePosixPath(path).parents
    ):
        raise ValueError("historical evidence paths conflict")
    if schema == "vibeqc.git-snapshot.v1" and (
        type(audit.get("file_count")) is not int
        or audit["file_count"] != len(result)
        or type(audit.get("total_bytes")) is not int
        or audit["total_bytes"] != sum(entry["bytes"] for entry in result)
    ):
        raise ValueError("historical snapshot totals differ from inventory")
    return result


def _read(entry: dict) -> bytes:
    # Also block promisor-object lazy fetches in partial clones. Older Git
    # versions that ignore GIT_NO_LAZY_FETCH still cannot use any protocol.
    environment = {
        **os.environ,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_ALLOW_PROTOCOL": "",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{entry['revision']}:{entry['path']}"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            "historical object unavailable; fetch the recorded revision "
            f"with git fetch origin {entry['revision']} and retry"
        )
    data = result.stdout
    if (
        len(data) != entry["bytes"]
        or hashlib.sha256(data).hexdigest() != entry["sha256"]
    ):
        raise ValueError("historical evidence checksum/size mismatch")
    return data


def _target(output: Path | None, default: Path) -> Path:
    target = Path(output) if output is not None else default
    if output is None and not target.resolve().is_relative_to(
        ROOT.resolve() / ".artifacts"
    ):
        raise ValueError("restoration path escapes ignored storage")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    return target


def restore(
    path: str, output: Path | None = None, *, manifest: Path | None = None
) -> Path:
    """Verify historical bytes before writing; preserve the legacy interface."""
    _safe_path(path)
    matches = [entry for entry in _records(manifest) if entry["path"] == path]
    if len(matches) != 1:
        raise ValueError("path has no unique verified migration record")
    target = _target(output, ROOT / ".artifacts/retention-restore" / path)
    data = _read(matches[0])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    return target


def restore_snapshot(
    output: Path | None = None, *, manifest: Path | None = None
) -> Path:
    """Verify every member before creating a new complete snapshot directory.

    Stage one blob at a time in temporary local storage, bounding memory by the
    largest member. Invalid, corrupt or missing evidence leaves no destination.
    No Git fetch, external upload, archive extraction or script execution occurs.
    """
    records = _records(manifest)
    target = _target(output, ROOT / ".artifacts/retention-snapshot")
    with tempfile.TemporaryDirectory(prefix="vibeqc-evidence-restore-") as temporary:
        staged = Path(temporary)
        for entry in records:
            data = _read(entry)
            member = staged / entry["path"]
            member.parent.mkdir(parents=True, exist_ok=True)
            with member.open("xb") as stream:
                stream.write(data)
        # copytree refuses an existing destination, including one created while
        # validation was running. Never overwrite a checkout or a previous copy.
        shutil.copytree(staged, target)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="original repository-relative path")
    parser.add_argument("--all", action="store_true", help="restore the full inventory")
    parser.add_argument(
        "--output", type=Path, help="new file, or new directory with --all"
    )
    parser.add_argument(
        "--manifest", type=Path, help="migration, archive or Git snapshot manifest"
    )
    args = parser.parse_args(argv)
    if bool(args.path) == args.all:
        parser.error("choose exactly one historical path or --all")
    if args.all:
        print(restore_snapshot(args.output, manifest=args.manifest))
    else:
        print(restore(args.path, args.output, manifest=args.manifest))


if __name__ == "__main__":
    main()

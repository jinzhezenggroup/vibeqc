"""Restore migrated historical evidence from Git into ignored local storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "benchmarks/results/retention-size-limit/migration.json"


def restore(
    path: str, output: Path | None = None, *, manifest: Path | None = None
) -> Path:
    """Verify historical bytes before creating a file; never overwrite a copy.

    Normal clones retain these existing Git objects. A shallow clone may need
    to fetch the recorded revision explicitly. No network request, extraction
    or execution of archived scripts occurs here. The default manifest retains
    the original oversized-archive interface; --manifest also accepts a
    per-member evidence archive manifest with a pinned source_revision.
    """
    relative = PurePosixPath(path)
    if (
        not path
        or not relative.parts
        or relative.is_absolute()
        or relative.as_posix() != path
        or ".." in relative.parts
        or "\\" in path
        or ":" in path
    ):
        raise ValueError("unsafe historical evidence path")
    audit = json.loads(Path(manifest or AUDIT).read_text())
    if audit.get("schema") == "vibeqc.storage-migration.v1":
        records = audit["archives"]
    elif audit.get("schema") == "vibeqc.evidence-archive.v1":
        records = [
            {**record, "revision": audit.get("source_revision")}
            for record in audit["files"]
        ]
    else:
        raise ValueError("unsupported evidence migration manifest")
    matches = [item for item in records if item["path"] == path]
    if len(matches) != 1:
        raise ValueError("path has no unique verified migration record")
    entry = matches[0]
    if (
        not isinstance(entry.get("revision"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", entry["revision"])
        or not isinstance(entry.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        or type(entry.get("bytes")) is not int
        or entry["bytes"] < 0
    ):
        raise ValueError("invalid historical evidence identity")
    destination = ROOT / ".artifacts/retention-restore"
    target = Path(output) if output is not None else destination / path
    if output is None and not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError("restoration path escapes ignored storage")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    result = subprocess.run(
        ["git", "show", f"{entry['revision']}:{path}"],
        cwd=ROOT,
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
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="original repository-relative evidence path")
    parser.add_argument("--output", type=Path, help="new local evidence filename")
    parser.add_argument(
        "--manifest", type=Path, help="migration or evidence-archive manifest"
    )
    args = parser.parse_args()
    print(restore(args.path, args.output, manifest=args.manifest))

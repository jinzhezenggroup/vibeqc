"""Restore a migrated historical archive from Git into ignored local storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "benchmarks/results/retention-size-limit/migration.json"


def restore(path: str, output: Path | None = None) -> Path:
    """Verify historical bytes before creating a file; never overwrite a copy.

    Normal clones retain these existing Git objects. A shallow clone may need
    to fetch the recorded revision explicitly. No network request, extraction
    or execution of archived scripts occurs here.
    """
    records = json.loads(AUDIT.read_text())["archives"]
    entry = next((item for item in records if item["path"] == path), None)
    if entry is None or not re.fullmatch(r"[0-9a-f]{40}", entry["revision"]):
        raise ValueError("path has no verified migration record")
    target = Path(output) if output else ROOT / ".artifacts/retention-restore" / path
    if target.exists():
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
        raise ValueError("historical archive checksum/size mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="original repository-relative archive path")
    parser.add_argument("--output", type=Path, help="new local archive filename")
    args = parser.parse_args()
    print(restore(args.path, args.output))

"""Verify immutable evidence and recompute warm verdicts without loading CUDA."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def verify(data: bytes, record: dict) -> None:
    """Bind both compressed archives and their exact original member bytes."""
    if len(data) != record["bytes"] or (
        hashlib.sha256(data).hexdigest() != record["sha256"]
    ):
        raise ValueError(f"evidence identity mismatch: {record['path']}")


def main() -> None:
    """Extract only verified relative files into disposable CPU audit storage."""
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / "manifest.json").read_text())
    prefix = PurePosixPath(".artifacts/issue206-stable-response")
    with tempfile.TemporaryDirectory(prefix="vibeqc-206-evidence-") as temporary:
        root = Path(temporary)
        seen: set[str] = set()
        for archive in manifest["archives"]:
            name = archive["path"]
            if Path(name).name != name:
                raise ValueError("archive must be a direct bundle member")
            source = bundle / name
            verify(source.read_bytes(), archive)
            with zipfile.ZipFile(source) as contents:
                expected = [entry["path"] for entry in archive["members"]]
                if contents.namelist() != expected:
                    raise ValueError(f"changed archive member list: {name}")
                for entry in archive["members"]:
                    relative = PurePosixPath(entry["path"])
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or not relative.is_relative_to(prefix)
                        or entry["path"] in seen
                    ):
                        raise ValueError("unsafe or duplicate evidence path")
                    seen.add(entry["path"])
                    data = contents.read(entry["path"])
                    verify(data, entry)
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
        artifacts = root / prefix
        summary_path = artifacts / "mainline-clean-v14/summary-v1.json"
        expected_summary = json.loads(summary_path.read_text())
        if expected_summary != json.loads((bundle / "warm-summary.json").read_text()):
            raise ValueError("published warm summary differs from original")
        # The original audit creates its output exclusively. Remove only the
        # disposable extracted copy after retaining the expected result above.
        summary_path.unlink()
        subprocess.run(
            [sys.executable, str(artifacts / "audit-mainline-clean-v14.py")],
            cwd=root,
            check=True,
        )
        if json.loads(summary_path.read_text()) != expected_summary:
            raise ValueError("recomputed warm verdicts differ from retained audit")
        print(f"Verified {len(seen)} exact records; all 19 verdicts reproduced.")


if __name__ == "__main__":
    main()

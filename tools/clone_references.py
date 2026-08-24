#!/usr/bin/env python3
"""Clone review/oracle repositories outside the product source tree.

The script intentionally does not vendor or link cloned code. After cloning,
record the resolved commit and license checksum in references/manifest.toml
before adapting any implementation.
"""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path


def run(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "qc-references",
        help="directory outside the product repository",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="clone only named repositories; may be repeated",
    )
    args = parser.parse_args()

    manifest_path = Path(__file__).resolve().parents[1] / "references" / "manifest.toml"
    repositories = tomllib.loads(manifest_path.read_text())["repository"]
    selected = set(args.name)
    args.root.mkdir(parents=True, exist_ok=True)
    for repository in repositories:
        name = repository["name"]
        if selected and name not in selected:
            continue
        destination = args.root / name
        if destination.exists():
            print(f"skip {name}: {destination} already exists")
            continue
        run(
            "git",
            "clone",
            "--filter=blob:none",
            repository["url"],
            str(destination),
        )
        pinned_commit = repository.get("commit", "")
        if pinned_commit:
            run("git", "checkout", "--detach", pinned_commit, cwd=destination)
        resolved = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=destination, text=True
        ).strip()
        print(f"cloned {name} at {resolved}")


if __name__ == "__main__":
    main()

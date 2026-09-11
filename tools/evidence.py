"""Inventory/check tracked evidence, or explicitly publish selected run files."""

from __future__ import annotations

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vibeqc_validation.retention import (
    POLICY_PATH,
    check,
    inventory,
    migration_audit,
    tracked_blobs,
)


def main() -> int:
    """Keep source inventory stdlib-only; publication uses validation's NumPy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "audit-migration"):
        command = commands.add_parser(name)
        command.add_argument("--revision", required=name == "audit-migration")
        command.add_argument("--output", type=Path, required=True)
    commands.add_parser("check")
    publication = commands.add_parser("publish")
    publication.add_argument("--run-directory", type=Path, required=True)
    publication.add_argument("--specification", type=Path, required=True)
    publication.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "publish":
        from tools.vibeqc_validation.publication import publish

        print(
            publish(
                args.run_directory,
                json.loads(args.specification.read_text()),
                args.destination,
            )
        )
        return 0
    blobs = tracked_blobs(args.root, getattr(args, "revision", None))
    if args.command == "check":
        errors = check(blobs, json.loads(blobs[POLICY_PATH]))
        # Scientific envelope validation remains in normal Python CI, while
        # storage checks do not install numerical dependencies in pre-commit.
        for path, data in blobs.items():
            if path.startswith("benchmarks/results/") and path.endswith(
                "/publication.json"
            ):
                from tools.vibeqc_validation.retention import digest, safe_relative

                manifest = json.loads(data)
                parent = str(Path(path).parent)
                for entry in manifest["files"]:
                    target = parent + "/" + safe_relative(entry["path"])
                    if (
                        target not in blobs
                        or digest(blobs[target]) != entry["sha256"]
                        or len(blobs[target]) != entry["bytes"]
                    ):
                        errors.append(
                            f"{path}: missing/changed publication member {target}"
                        )
        print(
            "\n".join(errors)
            if errors
            else "Evidence retention and publication checks passed."
        )
        return int(bool(errors))
    result = (
        inventory(blobs)
        if args.command == "inventory"
        else migration_audit(blobs, args.revision)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in {"classes", "removed_files", "removed_bytes"}
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Enforce the dependency direction of shared native SCF modules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# These are implementation boundaries, independent of #231's scientific CUDA
# ownership inventory. A reference oracle must not acquire a method driver or
# generated backend dependency merely because a future consumer needs it.
ALLOWED = {
    "reference": ("scf/reference/",),
    "initial_guess": ("scf/reference/", "scf/initial_guess/", "core/", "integrals/"),
}
SUFFIXES = {".cpp", ".hpp", ".cu", ".cuh"}
ROOT = Path(__file__).resolve().parents[1]


def audit_scf_structure(root: Path = ROOT) -> dict:
    """Check quoted/angle source includes, including relative-path spellings.

    Standard-library headers are outside this source graph. Comments do not
    introduce edges. Report source sizes without treating a small line count as
    evidence that the full HF decomposition has been completed.
    """
    source = (root / "src").resolve()
    errors, edges, modules = [], [], []
    for owner, allowed in ALLOWED.items():
        for path in sorted((source / "scf" / owner).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            content = path.read_text()
            relative = path.relative_to(source).as_posix()
            modules.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "lines": len(content.splitlines()),
                }
            )
            # Preserve line numbers when dropping multiline comments.
            text = re.sub(
                r"/\*.*?\*/|//[^\n]*",
                lambda m: "\n" * m[0].count("\n"),
                content,
                flags=re.DOTALL,
            )
            for match in re.finditer(
                r'^\s*#\s*include\s*["<]([^">]+)[">]', text, re.MULTILINE
            ):
                header = match[1]
                candidate = source / header
                if not candidate.is_file():
                    candidate = path.parent / header
                if not candidate.is_file():
                    continue
                try:
                    target = candidate.resolve().relative_to(source).as_posix()
                except ValueError:
                    continue
                edges.append({"from": relative, "to": target})
                if not target.startswith(allowed):
                    line = text.count("\n", 0, match.start()) + 1
                    errors.append(
                        f"{relative}:{line}: forbidden {owner} dependency on {target}"
                    )
    return {"modules": modules, "edges": edges, "errors": errors}


def main():
    """Return failure for a dependency violation; expose an optional JSON inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_scf_structure()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for error in report["errors"]:
            print(error)
        print(
            f"Checked {len(report['modules'])} shared SCF modules; "
            f"{len(report['errors'])} dependency errors"
        )
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())

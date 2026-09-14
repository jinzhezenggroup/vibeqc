"""Regenerate and compare committed one-electron derivative references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.vibeqc_validation.one_electron_derivatives import (
    one_electron_derivative_matrix,
)
from tools.vibeqc_validation.one_electron_reference_io import (
    write_derivative_references,
)


def _scientific_identity(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    return {
        "schema": manifest["schema"],
        "schema_version": manifest["schema_version"],
        "fixture_count": manifest["fixture_count"],
        "fixtures": manifest["fixtures"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--compare",
        type=Path,
        help="committed reference directory that must have identical inputs/arrays",
    )
    args = parser.parse_args()

    fixtures = one_electron_derivative_matrix()
    write_derivative_references(fixtures, args.output)
    if args.compare is not None:
        generated = _scientific_identity(args.output)
        committed = _scientific_identity(args.compare)
        if generated != committed:
            raise SystemExit(
                "one-electron derivative references changed; inspect and regenerate "
                "the committed fixture"
            )


if __name__ == "__main__":
    main()

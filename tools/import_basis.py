"""Convert a local BSE complete JSON file to VibeQC's checked basis schema."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vibeqc.basis_import import import_bse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--representation", choices=("cartesian", "spherical"))
    args = parser.parse_args()
    basis = import_bse(
        args.input,
        source=args.source,
        source_version=args.source_version,
        license=args.license,
        representation=args.representation,
    )
    basis.write(args.output)
    print(basis.identity)


if __name__ == "__main__":
    main()

"""Generate scalar ECP AO jets, radial potentials and projector contractions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import argparse

from vibeqc_compiler.integral.ecp_projector import emit_ecp_quadrature_cpp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    text = emit_ecp_quadrature_cpp()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)


if __name__ == "__main__":
    main()

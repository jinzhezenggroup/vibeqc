"""Emit the shared arbitrary-weight primitive used by the native psss candidate."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vibeqc_codegen.weighted_eri_cuda import emit_psss_weighted_header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inline-single-use", action="store_true")
    args = parser.parse_args()
    source = emit_psss_weighted_header(inline_single_use=args.inline_single_use)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != source:
        args.output.write_text(source)


if __name__ == "__main__":
    main()

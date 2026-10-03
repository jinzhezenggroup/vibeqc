"""Emit the common native DF source transform from compiler-owned algebra."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.method.df_mo_source import native_header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(native_header())


if __name__ == "__main__":
    main()

"""Generate the compiler-owned DF source-reuse schedule without a runtime."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vibeqc_compiler.method.df_exchange_schedule import native_header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(native_header(), encoding="utf-8")


if __name__ == "__main__":
    main()

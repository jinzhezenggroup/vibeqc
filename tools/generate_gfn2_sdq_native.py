"""Generate native CPU GFN2 S/D/Q primitive helpers from compiler DAGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vibeqc_compiler.integral.gfn2_sdq_cpu import emit_gfn2_sdq_cpu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(emit_gfn2_sdq_cpu(), encoding="utf-8")


if __name__ == "__main__":
    main()

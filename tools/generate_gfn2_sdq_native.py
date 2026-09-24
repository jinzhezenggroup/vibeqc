"""Generate native CPU GFN2 S/D/Q primitive helpers from compiler DAGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vibeqc_compiler.integral.gfn2_sdq_cpu import emit_gfn2_sdq_cpu, emit_gfn2_sdq_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cuda-output", type=Path)
    args = parser.parse_args()
    if args.output is None and args.cuda_output is None:
        parser.error("at least one of --output or --cuda-output is required")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(emit_gfn2_sdq_cpu(), encoding="utf-8")
    if args.cuda_output is not None:
        args.cuda_output.parent.mkdir(parents=True, exist_ok=True)
        args.cuda_output.write_text(emit_gfn2_sdq_cuda(), encoding="utf-8")


if __name__ == "__main__":
    main()

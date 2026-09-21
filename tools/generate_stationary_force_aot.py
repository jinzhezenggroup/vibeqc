"""Emit one qualified stationary CUDA force translation unit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cuda
from vibeqc_compiler.method.stationary_cuda import (
    emit_stationary_aot_cuda,
    qualified_sp_requests,
)

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functional", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--spin", choices=("unpolarized", "polarized"), required=True)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    primitive_source = emit_first_derivative_cuda(qualified_sp_requests())
    write_if_changed(
        args.output,
        emit_stationary_aot_cuda(
            args.functional,
            primitive_source=primitive_source,
            spin=args.spin,
            iterations=args.iterations,
        ),
    )


if __name__ == "__main__":
    main()

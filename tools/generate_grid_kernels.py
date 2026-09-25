"""Emit the shared AO policy/runtime composition for the native CUDA build."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.dft.ao_cuda import emit_grid_source
from vibeqc_compiler.dft.xc_contraction_cuda import XcMatrixSchedule

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    """Use the shared emitter's native composition and preserve unchanged mtimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--xc-matrix-tile",
        type=int,
        choices=(8, 16, 32),
        default=16,
        help="compiler-owned CUDA XC matrix tile; production default is 16",
    )
    args = parser.parse_args()
    source, _, _ = emit_grid_source(
        native_ks=True,
        xc_matrix_schedule=XcMatrixSchedule(args.xc_matrix_tile),
    )
    write_if_changed(args.output, source)


if __name__ == "__main__":
    main()

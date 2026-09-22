"""Emit the compiler-owned CUDA molecular quadrature and resource schedule."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.xc.quadrature_cuda import emit_quadrature_cuda

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    """Generate deterministic source without loading a runtime or GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_if_changed(args.output, emit_quadrature_cuda())


if __name__ == "__main__":
    main()

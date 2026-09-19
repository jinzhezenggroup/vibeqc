"""Emit compiler-owned native LDA/PBE XC nuclear-gradient CUDA."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.xc.geometry_cuda import emit_native_geometry_cuda

from tools.generate_df_kernels import write_if_changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_if_changed(args.output, emit_native_geometry_cuda())


if __name__ == "__main__":
    main()

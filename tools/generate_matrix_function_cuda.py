"""Generate the runtime-sized CUDA lowering for matrix-function custom rules."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from vibeqc_compiler.method.matrix_function_cuda import (
    emit_symmetric_matrix_function_vjp_cuda,
)

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_if_changed(args.output, emit_symmetric_matrix_function_vjp_cuda())


if __name__ == "__main__":
    main()

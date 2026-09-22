"""Generate production native lowerings for the DF-HF stationary response."""

import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

# Build-time generation needs only immutable TensorIR/type/lowering modules.
# Avoid importing the NumPy-backed public TensorIR facade in bare CMake Python.
import vibeqc_compiler

_tensor_path = ROOT / "python" / "vibeqc_compiler" / "tensor"
_tensor_package = types.ModuleType("vibeqc_compiler.tensor")
_tensor_package.__path__ = [str(_tensor_path)]
_tensor_package.__package__ = "vibeqc_compiler.tensor"
sys.modules["vibeqc_compiler.tensor"] = _tensor_package
vibeqc_compiler.tensor = _tensor_package

from vibeqc_compiler.method.df_hf_response_cuda import (
    emit_df_hf_response_contract,
    emit_df_hf_response_cuda,
)

from tools.generate_df_kernels import write_if_changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-output", type=Path)
    parser.add_argument("--contract-output", type=Path)
    args = parser.parse_args()
    if args.cuda_output is None and args.contract_output is None:
        parser.error("at least one output is required")
    if args.cuda_output is not None:
        write_if_changed(args.cuda_output, emit_df_hf_response_cuda())
    if args.contract_output is not None:
        write_if_changed(args.contract_output, emit_df_hf_response_contract())


if __name__ == "__main__":
    main()

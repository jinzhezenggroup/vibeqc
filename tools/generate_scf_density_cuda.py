"""Generate compiler-owned CUDA SCF density kernels."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "python"))

# Build-time code generation must work with CMake's minimal Python interpreter.
# Avoid importing tensor.__init__ and its optional validation/interpreter stack.
if "vibeqc_compiler.tensor" not in sys.modules:
    import vibeqc_compiler

    tensor_path = ROOT / "python" / "vibeqc_compiler" / "tensor"
    tensor_package = types.ModuleType("vibeqc_compiler.tensor")
    tensor_package.__path__ = [str(tensor_path)]
    tensor_package.__package__ = "vibeqc_compiler.tensor"
    sys.modules["vibeqc_compiler.tensor"] = tensor_package
    vibeqc_compiler.tensor = tensor_package

from vibeqc_compiler.tensor.scf_cuda import emit_density_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(emit_density_cuda(), encoding="utf-8")


if __name__ == "__main__":
    main()

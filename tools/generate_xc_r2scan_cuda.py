"""Generate the native CUDA r2SCAN scalar point evaluator."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from tools.generate_xc_cpu import emit_r2scan_program, write_if_changed


def emit_r2scan_device() -> str:
    body = emit_r2scan_program(
        value_type="R2scanDeviceValue",
        function_name="r2scan_device",
        identity_constant="kR2scanDeviceExpressionIdentity",
        qualifier="__device__ inline",
    )
    return (
        "// Generated from audited MPL-2.0 r2SCAN expressions.\n"
        "#pragma once\n"
        "#include <cmath>\n"
        "namespace vibeqc::dft::generated {\n"
        f"{body}\n"
        "}  // namespace vibeqc::dft::generated\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_if_changed(args.output, emit_r2scan_device())


if __name__ == "__main__":
    main()

"""Generate the native CUDA r2SCAN point ABI from shared semilocal lowering."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.xc.semilocal_codegen import emit_r2scan_program


def emit_r2scan_device() -> str:
    """Preserve the native r2SCAN device ABI while sharing its scientific source."""

    body = emit_r2scan_program(
        value_type="R2scanDeviceValue",
        function_name="r2scan_device",
        identity_constant="kR2scanDeviceExpressionIdentity",
        qualifier="__device__ inline",
    )
    return "\n".join(
        [
            "// Generated from audited MPL-2.0 r2SCAN expressions.",
            "#pragma once",
            "#include <cmath>",
            "namespace vibeqc::dft::generated {",
            body.rstrip("\n"),
            "}  // namespace vibeqc::dft::generated",
            "",
        ]
    )


def write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_if_changed(args.output, emit_r2scan_device())


if __name__ == "__main__":
    main()

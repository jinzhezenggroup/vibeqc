"""Generate native CUDA B3LYP semilocal E/vxc from the shared MethodIR graph."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.method.spec import (
    ExactExchangePrimitive,
    SemilocalXCPrimitive,
    resolve_method,
)
from vibeqc_compiler.xc.semilocal_codegen import emit_polarized_semilocal


def emit_b3lyp_device() -> str:
    """Emit the production-tail B3LYP semilocal point ABI for native CUDA."""

    method = resolve_method("B3LYP", spin="polarized")
    semilocal = next(
        primitive.functional
        for primitive in method.primitives
        if isinstance(primitive, SemilocalXCPrimitive)
    )
    exchange = [
        primitive
        for primitive in method.primitives
        if isinstance(primitive, ExactExchangePrimitive)
    ]
    if len(exchange) != 1 or exchange[0].operator != "full-range":
        raise RuntimeError("B3LYP MethodIR lost its canonical full-range exchange primitive")
    body = emit_polarized_semilocal(
        semilocal,
        value_type="B3lypDeviceValue",
        function_name="b3lyp_device",
        identity_constant="kB3lypDeviceExpressionIdentity",
        production=True,
        function_qualifier="__device__ inline",
        declarations=(
            f'inline constexpr const char* kB3lypDeviceMethodIdentity = "{method.identity}";',
            f"inline constexpr double kB3lypDeviceExactExchange = {float(exchange[0].coefficient).hex()};",
        ),
    )
    return "\n".join(
        [
            "// Generated from the canonical B3LYP MethodIR semilocal graph.",
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
    write_if_changed(args.output, emit_b3lyp_device())


if __name__ == "__main__":
    main()

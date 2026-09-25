"""Generate native CUDA omegaB97M-V semilocal E/vxc from shared lowering."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.method.spec import SemilocalXCPrimitive, resolve_method
from vibeqc_compiler.xc.semilocal_codegen import emit_polarized_semilocal
from vibeqc_compiler.xc.wb97mv_maple import (
    DENSITY_THRESHOLD,
    SIGMA_THRESHOLD,
    TAU_THRESHOLD,
)


def emit_wb97mv_device() -> str:
    """Preserve one CUDA point ABI while sharing the canonical WB97M-V Graph."""

    method = resolve_method("WB97M-V", spin="polarized")
    semilocal = next(
        primitive.functional
        for primitive in method.primitives
        if isinstance(primitive, SemilocalXCPrimitive)
    )
    body = emit_polarized_semilocal(
        semilocal,
        value_type="Wb97mvDeviceValue",
        function_name="wb97mv_device",
        identity_constant="kWb97mvDeviceExpressionIdentity",
        production=True,
        function_qualifier="__device__ inline",
        declarations=(
            f"inline constexpr double kWb97mvDeviceDensityThreshold = {DENSITY_THRESHOLD.hex()};",
            f"inline constexpr double kWb97mvDeviceSigmaThreshold = {SIGMA_THRESHOLD.hex()};",
            f"inline constexpr double kWb97mvDeviceTauThreshold = {TAU_THRESHOLD.hex()};",
        ),
    )
    return "\n".join(
        [
            "// Generated from pinned Libxc omegaB97M-V semilocal expressions.",
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
    write_if_changed(args.output, emit_wb97mv_device())


if __name__ == "__main__":
    main()

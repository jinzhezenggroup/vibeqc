"""Generate the native CUDA omegaB97M-V semilocal point ABI.

The scientific FunctionalSpec -> Graph -> first-derivative lowering is owned by
vibeqc_compiler.xc.semilocal_codegen, exactly as for the host production
wrapper. This file only selects the CUDA ABI spelling and exposes the pinned
Libxc work_mgga policy constants consumed by the resident grid runtime.
"""

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
    """Emit the CUDA wrapper from the same production FunctionalSpec as CPU."""

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
        declarations=(
            f"inline constexpr double kWb97mvDeviceDensityThreshold = {DENSITY_THRESHOLD.hex()};",
            f"inline constexpr double kWb97mvDeviceSigmaThreshold = {SIGMA_THRESHOLD.hex()};",
            f"inline constexpr double kWb97mvDeviceTauThreshold = {TAU_THRESHOLD.hex()};",
        ),
        function_qualifier="__device__ inline",
    )
    return "\n".join(
        [
            "// Generated from the pinned Libxc 7.0.0 omegaB97M-V Maple expression.",
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

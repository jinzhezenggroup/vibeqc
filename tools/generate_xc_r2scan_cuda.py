"""Generate the native CUDA r2SCAN scalar point evaluator."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc import functional

from tools.generate_xc_cpu import build_roots, write_if_changed


def emit_r2scan_device() -> str:
    spec = functional("R2SCAN", spin="polarized")
    outputs = ((), *((i,) for i in range(len(spec.features))))
    graph, roots, expression_hash = build_roots(spec, outputs, production=True)
    emitter = ScalarCEmitter(graph, {name: name for name in spec.features})
    emitter.emit(roots)
    refs = [emitter.reference(root) for root in roots]
    lines = [
        "// Generated from audited MPL-2.0 r2SCAN expressions.",
        "#pragma once",
        "#include <cmath>",
        "namespace vibeqc::dft::generated {",
        "struct R2scanDeviceValue {",
        "  double energy_density;",
        "  double feature_derivative[7];",
        "};",
        f'inline constexpr const char* kR2scanDeviceExpressionIdentity = "{expression_hash}";',
        "__device__ inline R2scanDeviceValue r2scan_device(",
        "    double rho_a, double rho_b, double sigma_aa, double sigma_ab,",
        "    double sigma_bb, double tau_a, double tau_b) {",
    ]
    lines.extend(emitter.lines)
    lines.append("  return {" + refs[0] + ", {" + ", ".join(refs[1:]) + "}};")
    lines.extend(["}", "}  // namespace vibeqc::dft::generated", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_if_changed(args.output, emit_r2scan_device())


if __name__ == "__main__":
    main()

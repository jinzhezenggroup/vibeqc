"""Generate host FP64 XC expressions from the audited shared scalar DAG."""

import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
from pathlib import Path

from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc import build_program, functional
from vibeqc_compiler.xc.expressions import lda_xc_pw_unpolarized_tail_expression


def write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def emit_lda_xc_pw() -> str:
    program = build_program(
        functional("LDA_XC_PW", spin="unpolarized"),
        order=1,
        outputs=((), (0,)),
    )
    graph, energy, derivative, _ = lda_xc_pw_unpolarized_tail_expression()
    graph, roots = graph.lower_small_integer_powers((energy, derivative))
    emitter = ScalarCEmitter(graph, {"rho_sixth_root": "x"})
    emitter.emit(roots)
    lines = [
        "// Generated from audited MPL-2.0 expressions; see external/libxc-7.0.0/COPYING.",
        "#pragma once",
        "#include <cmath>",
        "namespace vibeqc::dft::generated {",
        "struct LdaXcPwValue { double energy_density; double density_derivative; };",
        f'inline constexpr const char* kLdaXcPwExpressionIdentity = "{program.expression_hash}";',
        'inline constexpr const char* kLdaXcPwTailAlgebra = "sixth-root-v1";',
        "inline LdaXcPwValue lda_xc_pw_unpolarized(double rho) {",
        "  const double x = pow(rho, 1.0 / 6.0);",
    ]
    lines.extend(emitter.lines)
    lines.extend(
        [
            f"  return {{{emitter.reference(roots[0])}, {emitter.reference(roots[1])}}};",
            "}",
        ]
    )
    return "\n".join(lines)


def emit_lda_xc_pw_polarized() -> str:
    spec = functional("LDA_XC_PW", spin="polarized")
    program = build_program(spec, order=1)
    emitter = ScalarCEmitter(
        program.graph,
        {name: name for name in spec.features},
    )
    emitter.emit(program.roots)
    references = [emitter.reference(root) for root in program.roots]
    lines = [
        "struct LdaXcPwPolarizedValue {",
        "  double energy_density;",
        "  double feature_derivative[7];",
        "};",
        f'inline constexpr const char* kLdaXcPwPolarizedExpressionIdentity = "{program.expression_hash}";',
        "inline LdaXcPwPolarizedValue lda_xc_pw_polarized(double rho_a, double rho_b) {",
        "  const double sigma_aa = 0.0;",
        "  const double sigma_ab = 0.0;",
        "  const double sigma_bb = 0.0;",
        "  const double tau_a = 0.0;",
        "  const double tau_b = 0.0;",
    ]
    lines.extend(emitter.lines)
    lines.append(
        "  return {" + references[0] + ", {" + ", ".join(references[1:]) + "}};"
    )
    lines.extend(["}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_if_changed(
        args.output,
        emit_lda_xc_pw() + emit_lda_xc_pw_polarized()
        + "}  // namespace vibeqc::dft::generated\n",
    )


if __name__ == "__main__":
    main()

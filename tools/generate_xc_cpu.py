"""Generate host FP64 XC expressions from the audited shared scalar DAG."""

import sys as _compiler_sys
import types as _compiler_types
from pathlib import Path as _CompilerPath

_compiler_root = (
    _CompilerPath(__file__).resolve().parents[1] / "python" / "vibeqc_compiler"
)
for _name, _path in (
    ("vibeqc_compiler", _compiler_root),
    ("vibeqc_compiler.common", _compiler_root / "common"),
    ("vibeqc_compiler.integral", _compiler_root / "integral"),
    ("vibeqc_compiler.xc", _compiler_root / "xc"),
):
    _module = _compiler_types.ModuleType(_name)
    _module.__path__ = [str(_path)]
    _compiler_sys.modules[_name] = _module

import argparse
from pathlib import Path

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.expr import AlgebraForm
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc.expressions import (
    energy_expression,
    lda_xc_pw_unpolarized_tail_expression,
)
from vibeqc_compiler.xc.spec import functional


def build_roots(spec, outputs):
    """Build first-derivative roots and their ordinary XCProgram identity."""

    graph, energy, variables = energy_expression(spec)
    derivatives = {(): energy}
    for output in outputs:
        for depth in range(1, len(output) + 1):
            key = output[:depth]
            if key not in derivatives:
                derivatives[key] = graph.differentiate(
                    derivatives[key[:-1]], variables[key[-1]]
                )
    roots = tuple(derivatives[output] for output in outputs)
    graph, roots = graph.apply_algebra_form(roots, AlgebraForm.FACTORED_NARY)
    graph, roots = graph.lower_small_integer_powers(roots)
    reachable = graph.topological_order(roots)
    indices = {index: i for i, index in enumerate(reachable)}
    payload = {
        "spec": spec.to_payload(),
        "outputs": outputs,
        "optimization": "after",
        "nodes": [
            (
                graph.nodes[i].operation,
                [indices[j] for j in graph.nodes[i].arguments],
                str(graph.nodes[i].payload),
            )
            for i in reachable
        ],
        "roots": [indices[root.identifier] for root in roots],
    }
    return graph, roots, canonical_hash(payload)


def write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def emit_lda_xc_pw() -> str:
    _graph, _roots, expression_hash = build_roots(
        functional("LDA_XC_PW", spin="unpolarized"), ((), (0,))
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
        f'inline constexpr const char* kLdaXcPwExpressionIdentity = "{expression_hash}";',
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
    outputs = ((), *((i,) for i in range(len(spec.features))))
    graph, roots, expression_hash = build_roots(spec, outputs)
    emitter = ScalarCEmitter(
        graph,
        {name: name for name in spec.features},
    )
    emitter.emit(roots)
    references = [emitter.reference(root) for root in roots]
    lines = [
        "struct LdaXcPwPolarizedValue {",
        "  double energy_density;",
        "  double feature_derivative[7];",
        "};",
        f'inline constexpr const char* kLdaXcPwPolarizedExpressionIdentity = "{expression_hash}";',
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
        emit_lda_xc_pw()
        + emit_lda_xc_pw_polarized()
        + "}  // namespace vibeqc::dft::generated\n",
    )


if __name__ == "__main__":
    main()

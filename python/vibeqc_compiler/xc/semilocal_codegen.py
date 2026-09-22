"""Backend-neutral scalar code generation for semilocal XC point evaluators.

This module owns the shared FunctionalSpec -> scalar Graph -> derivative roots
path used by AOT CPU and CUDA wrappers. Backend entry points may choose ABI
names and function qualifiers, but they must not rebuild or redifferentiate the
scientific expression independently.
"""

import typing

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.expr import AlgebraForm
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter

from .expression_dispatch import build_energy_expression


def build_roots(
    spec: typing.Any, outputs: typing.Any, *, production: bool = False
) -> tuple[typing.Any, typing.Any, str]:
    """Build requested derivative roots and their canonical expression identity."""

    graph, energy, variables = build_energy_expression(spec, production=production)
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


def polarized_feature_count(spec: typing.Any) -> int:
    """Return the native first-derivative ABI width for one polarized functional."""

    if spec.spin != "polarized":
        raise ValueError("native semilocal lowering requires polarized FunctionalSpec")
    if spec.ingredients == ("rho", "sigma"):
        return 5
    if spec.ingredients == ("rho", "sigma", "tau"):
        return 7
    raise ValueError(
        "native semilocal lowering requires rho/sigma or rho/sigma/tau FunctionalSpec"
    )


def emit_polarized_semilocal(
    spec: typing.Any,
    *,
    value_type: str,
    function_name: str,
    identity_constant: str,
    production: bool = False,
    declarations: tuple[str, ...] = (),
    function_qualifier: str = "inline",
) -> str:
    """Emit polarized GGA/MGGA E/vxc from the canonical scalar Graph.

    function_qualifier is a backend ABI choice such as inline on the host or
    __device__ inline for CUDA. It does not participate in the mathematical
    expression identity.
    """

    feature_count = polarized_feature_count(spec)
    if feature_count == 5:
        signature = (
            f"{function_qualifier} {value_type} {function_name}(",
            "    double rho_a, double rho_b, double sigma_aa, double sigma_ab, double sigma_bb) {",
        )
        prelude = (
            "  const double tau_a = 0.0;",
            "  const double tau_b = 0.0;",
        )
    else:
        signature = (
            f"{function_qualifier} {value_type} {function_name}(",
            "    double rho_a, double rho_b, double sigma_aa, double sigma_ab,",
            "    double sigma_bb, double tau_a, double tau_b) {",
        )
        prelude = ()

    outputs = ((), *((i,) for i in range(feature_count)))
    graph, roots, expression_hash = build_roots(spec, outputs, production=production)
    emitter = ScalarCEmitter(graph, {name: name for name in spec.features})
    emitter.emit(roots)
    references = [emitter.reference(root) for root in roots]
    return "\n".join(
        [
            f"struct {value_type} {{",
            "  double energy_density;",
            f"  double feature_derivative[{feature_count}];",
            "};",
            f'inline constexpr const char* {identity_constant} = "{expression_hash}";',
            *declarations,
            *signature,
            *prelude,
            *emitter.lines,
            "  return {" + references[0] + ", {" + ", ".join(references[1:]) + "}};",
            "}",
            "",
        ]
    )

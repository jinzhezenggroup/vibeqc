"""Backend-neutral scalar code generation for semilocal XC point evaluators.

This module owns the shared FunctionalSpec -> scalar Graph -> derivative roots
path used by AOT CPU and CUDA wrappers. Backend entry points may choose ABI
names and function qualifiers, but they must not rebuild or redifferentiate the
scientific expression independently.
"""

import typing
from fractions import Fraction

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.expr import AlgebraForm, ScalarDomain
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter

from .domain import feature_domains
from .expression_dispatch import (
    build_energy_expression,
    build_pointwise_energy_expression,
)
from .scan_maple import scan_runtime_policy
from .spec import FunctionalSpec


def build_roots(
    spec: typing.Any,
    outputs: typing.Any,
    *,
    production: bool = False,
    variable_domains: dict[str, ScalarDomain] | None = None,
    pointwise_bulk: bool = False,
) -> tuple[typing.Any, typing.Any, str]:
    """Build requested derivative roots and their canonical expression identity."""

    if type(pointwise_bulk) is not bool:
        raise TypeError("pointwise_bulk must be bool")
    if pointwise_bulk and production:
        raise ValueError("pointwise bulk lowering cannot claim production semantics")
    if pointwise_bulk:
        graph, energy, variables = build_pointwise_energy_expression(spec)
    else:
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
    violations = (
        graph.domain_violations(roots, variable_domains)
        if variable_domains is not None
        else ()
    )
    if not violations:
        graph, roots = graph.apply_algebra_form(
            roots,
            AlgebraForm.FACTORED_NARY,
            variable_domains=variable_domains,
        )
        graph, roots = graph.lower_small_integer_powers(roots)
    reachable = graph.topological_order(roots)
    indices = {index: i for i, index in enumerate(reachable)}
    payload = {
        "spec": spec.to_payload(),
        "outputs": outputs,
        "representation": "pointwise-bulk" if pointwise_bulk else "admitted",
        "optimization": "domain-preserving-raw" if violations else "after",
        **(
            {
                "variable_domains": {
                    name: domain.value
                    for name, domain in sorted(variable_domains.items())
                }
            }
            if variable_domains is not None
            else {}
        ),
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
    if spec.ingredients == ("rho",):
        return 2
    if spec.ingredients == ("rho", "sigma"):
        return 5
    if spec.ingredients == ("rho", "sigma", "tau"):
        return 7
    raise ValueError(
        "native semilocal lowering requires rho, rho/sigma or rho/sigma/tau FunctionalSpec"
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
    pointwise_bulk: bool = False,
) -> str:
    """Emit polarized LDA/GGA/MGGA E/vxc from the canonical scalar Graph.

    function_qualifier is a backend ABI choice such as inline on the host or
    __device__ inline for CUDA. It does not participate in the mathematical
    expression identity.
    """

    feature_count = polarized_feature_count(spec)
    if feature_count == 2:
        signature = (
            f"{function_qualifier} {value_type} {function_name}(",
            "    double rho_a, double rho_b) {",
        )
        prelude = (
            "  const double sigma_aa = 0.0;",
            "  const double sigma_ab = 0.0;",
            "  const double sigma_bb = 0.0;",
            "  const double tau_a = 0.0;",
            "  const double tau_b = 0.0;",
        )
    elif feature_count == 5:
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
    graph, roots, expression_hash = build_roots(
        spec,
        outputs,
        production=production,
        pointwise_bulk=pointwise_bulk,
    )
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


def _r2scan_component_program(
    component: str,
) -> tuple[typing.Any, typing.Any, str, typing.Any, dict[str, ScalarDomain]]:
    """Build one raw SCAN-family component over Libxc-sanitized work inputs."""

    spec = FunctionalSpec(
        f"{component}_PRODUCTION",
        ((component, Fraction(1)),),
        spin="polarized",
    )
    outputs = ((), *((i,) for i in range(len(spec.features))))
    work_domains = feature_domains(spec, sanitized_mgga=True)
    graph, roots, expression_hash = build_roots(
        spec,
        outputs,
        production=True,
        variable_domains=work_domains,
    )
    violations = graph.domain_violations(roots, work_domains)
    if violations:
        raise RuntimeError(
            f"{component} remains singular on the declared Libxc work domain: "
            f"{violations[:3]!r}"
        )
    return graph, roots, expression_hash, scan_runtime_policy(component), work_domains


def emit_r2scan_program(
    *,
    value_type: str,
    function_name: str,
    identity_constant: str,
    qualifier: str,
) -> str:
    """Emit component-wise r2SCAN with compiler-owned Libxc boundary semantics."""

    component_rows = []
    for tag, component in (
        ("X", "MGGA_X_R2SCAN"),
        ("C", "MGGA_C_R2SCAN"),
    ):
        graph, roots, expression_hash, policy, work_domains = _r2scan_component_program(
            component
        )
        emitter = ScalarCEmitter(graph, {name: name for name in work_domains})
        emitter.emit(roots)
        references = [emitter.reference(root) for root in roots]
        component_rows.append(
            (tag, component, expression_hash, policy, emitter, references)
        )

    identity = canonical_hash(
        {
            "schema": "r2scan-production/libxc-7.0-work-mgga-v1",
            "components": [
                {
                    "name": component,
                    "expression": expression_hash,
                    "runtime_policy": policy.to_payload(),
                }
                for _, component, expression_hash, policy, _, _ in component_rows
            ],
        }
    )
    lines = [
        f"struct {value_type} {{",
        "  double energy_density{};",
        "  double feature_derivative[7]{};",
        "};",
        f'inline constexpr const char* {identity_constant} = "{identity}";',
        'inline constexpr const char* kR2scanProductionPolicy = "libxc-7.0/work-mgga-v1";',
    ]
    for tag, _, expression_hash, policy, emitter, references in component_rows:
        prefix = f"kR2scan{tag}"
        raw_name = f"{function_name}_{tag.lower()}_raw"
        lines.extend(
            [
                f'inline constexpr const char* {prefix}ExpressionIdentity = "{expression_hash}";',
                f"inline constexpr double {prefix}DensityThreshold = {policy.density_threshold.hex()};",
                f"inline constexpr double {prefix}SigmaThreshold = {policy.sigma_threshold.hex()};",
                f"inline constexpr double {prefix}TauThreshold = {policy.tau_threshold.hex()};",
                f"{qualifier} {value_type} {raw_name}(",
                "    double rho_a, double rho_b, double sigma_aa, double sigma_ab,",
                "    double sigma_bb, double tau_a, double tau_b) {",
                *emitter.lines,
                "  return {"
                + references[0]
                + ", {"
                + ", ".join(references[1:])
                + "}};",
                "}",
                "",
            ]
        )

    lines.extend(
        [
            f"{qualifier} {value_type} {function_name}(",
            "    double rho_a, double rho_b, double sigma_aa, double sigma_ab,",
            "    double sigma_bb, double tau_a, double tau_b) {",
            f"  {value_type} out{{}};",
            "  const double total_density = rho_a + rho_b;",
        ]
    )
    for tag, _, _, _, _, _ in component_rows:
        prefix = f"kR2scan{tag}"
        raw_name = f"{function_name}_{tag.lower()}_raw"
        lines.extend(
            [
                f"  if (total_density >= {prefix}DensityThreshold) {{",
                f"    const double work_rho_a = fmax({prefix}DensityThreshold, rho_a);",
                f"    const double work_rho_b = fmax({prefix}DensityThreshold, rho_b);",
                f"    const double sigma_floor = {prefix}SigmaThreshold * {prefix}SigmaThreshold;",
                "    const double work_sigma_aa = fmax(sigma_floor, sigma_aa);",
                "    const double work_sigma_bb = fmax(sigma_floor, sigma_bb);",
                "    const double sigma_average = 0.5 * (work_sigma_aa + work_sigma_bb);",
                "    const double work_sigma_ab = fmax(-sigma_average, fmin(sigma_average, sigma_ab));",
                f"    const double work_tau_a = fmax({prefix}TauThreshold, tau_a);",
                f"    const double work_tau_b = fmax({prefix}TauThreshold, tau_b);",
                f"    const auto raw = {raw_name}(",
                "        work_rho_a, work_rho_b, work_sigma_aa, work_sigma_ab,",
                "        work_sigma_bb, work_tau_a, work_tau_b);",
                "    const double work_density = work_rho_a + work_rho_b;",
                "    out.energy_density += raw.energy_density * total_density / work_density;",
                "    for (unsigned i = 0; i < 7; ++i)",
                "      out.feature_derivative[i] += raw.feature_derivative[i];",
                "  }",
            ]
        )
    lines.extend(["  return out;", "}", ""])
    return "\n".join(lines)

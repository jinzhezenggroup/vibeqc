"""Generate host FP64 XC expressions from the audited shared scalar DAG."""

import sys as _compiler_sys
import types as _compiler_types
from pathlib import Path as _CompilerPath
from typing import Any

_compiler_root = (
    _CompilerPath(__file__).resolve().parents[1] / "python" / "vibeqc_compiler"
)
for _name, _path in (
    ("vibeqc_compiler", _compiler_root),
    ("vibeqc_compiler.common", _compiler_root / "common"),
    ("vibeqc_compiler.integral", _compiler_root / "integral"),
    ("vibeqc_compiler.xc", _compiler_root / "xc"),
    ("vibeqc_compiler.dft", _compiler_root / "dft"),
    ("vibeqc_compiler.method", _compiler_root / "method"),
):
    _module = _compiler_types.ModuleType(_name)
    _module.__path__ = [str(_path)]
    _compiler_sys.modules[_name] = _module

import argparse
from pathlib import Path

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft.feature_policy import emit_feature_policy
from vibeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_THRESHOLD
from vibeqc_compiler.integral.expr import AlgebraForm
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.method.spec import (
    ExactExchangePrimitive,
    NonlocalCorrelationPrimitive,
    RangeSeparatedExchangePrimitive,
    SemilocalXCPrimitive,
    resolve_method,
)
from vibeqc_compiler.xc.expression_dispatch import build_energy_expression
from vibeqc_compiler.xc.production_policy import (
    lda_xc_pw_polarized_tail_expression,
    lda_xc_pw_unpolarized_tail_expression,
    pbe_correlation_scaled_expression,
    pbe_exchange_direct_expression,
    pbe_exchange_reciprocal_expression,
)
from vibeqc_compiler.xc.spec import functional
from vibeqc_compiler.xc.wb97mv_maple import (
    DENSITY_THRESHOLD as WB97MV_DENSITY_THRESHOLD,
)
from vibeqc_compiler.xc.wb97mv_maple import (
    SIGMA_THRESHOLD as WB97MV_SIGMA_THRESHOLD,
)
from vibeqc_compiler.xc.wb97mv_maple import (
    SMOOTH_LR_CUTOFF as WB97MV_SMOOTH_LR_CUTOFF,
)
from vibeqc_compiler.xc.wb97mv_maple import (
    SMOOTH_LR_ORDER as WB97MV_SMOOTH_LR_ORDER,
)
from vibeqc_compiler.xc.wb97mv_maple import (
    TAU_THRESHOLD as WB97MV_TAU_THRESHOLD,
)


def build_roots(
    spec: Any, outputs: Any, *, production: bool = False
) -> tuple[Any, Any, str]:
    """Build derivative roots and the exact emitted-expression identity."""

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


def graph_identity(label: str, graph: Any, roots: Any) -> str:
    """Hash one explicitly constructed production graph and its ordered roots."""

    reachable = graph.topological_order(roots)
    indices = {index: i for i, index in enumerate(reachable)}
    return canonical_hash(
        {
            "schema": label,
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
    )


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
        "// Generated from audited MPL-2.0 expressions; see upstream/libxc/7.0.0/COPYING.",
        "#pragma once",
        "#include <cmath>",
        "#include <cfloat>",
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


def emit_lda_xc_pw_polarized_production() -> str:
    """Emit the exact tail-stable polarized LDA production E/vxc."""

    graph, energy, derivative_a, derivative_b, _variables = (
        lda_xc_pw_polarized_tail_expression()
    )
    roots = (energy, derivative_a, derivative_b)
    graph, roots = graph.apply_algebra_form(roots, AlgebraForm.FACTORED_NARY)
    graph, roots = graph.lower_small_integer_powers(roots)
    names = (
        "normalized_rho_a",
        "normalized_rho_b",
        "rho_scale",
        "rho_scale_sixth_root",
    )
    emitter = ScalarCEmitter(graph, dict(zip(names, names, strict=True)))
    emitter.emit(roots)
    references = [emitter.reference(root) for root in roots]
    identity = graph_identity("lda-xc-pw-polarized-production-v1", graph, roots)
    return "\n".join(
        [
            f'inline constexpr const char* kLdaXcPwPolarizedProductionIdentity = "{identity}";',
            'inline constexpr const char* kLdaXcPwPolarizedProductionPolicy = "scaled-sixth-root-v1";',
            "inline LdaXcPwPolarizedValue lda_xc_pw_polarized_production(double rho_a, double rho_b) {",
            "  const double rho_scale = rho_a + rho_b;",
            "  if (rho_scale == 0.0) return {};",
            "  const double normalized_rho_a = rho_a / rho_scale;",
            "  const double normalized_rho_b = rho_b / rho_scale;",
            "  const double rho_scale_sixth_root = pow(rho_scale, 1.0 / 6.0);",
            *emitter.lines,
            "  return {"
            + references[0]
            + ", {"
            + references[1]
            + ", "
            + references[2]
            + ", 0.0, 0.0, 0.0, 0.0, 0.0}};",
            "}",
            "",
        ]
    )


def emit_wb97mv_polarized() -> str:
    """Emit production B97M semilocal E/vxc from pinned Libxc Maple."""

    method = resolve_method("WB97M-V", spin="polarized")
    semilocal = next(
        primitive.functional
        for primitive in method.primitives
        if isinstance(primitive, SemilocalXCPrimitive)
    )
    ranges = {
        primitive.operator: primitive
        for primitive in method.primitives
        if isinstance(primitive, RangeSeparatedExchangePrimitive)
    }
    nonlocal_term = next(
        primitive
        for primitive in method.primitives
        if isinstance(primitive, NonlocalCorrelationPrimitive)
    )
    short, long = ranges["short-range"], ranges["long-range"]
    if short.omega != long.omega or short.omega != semilocal.range_omega:
        raise RuntimeError("WB97M-V semilocal and exact exchange disagree on omega")
    outputs = ((), *((i,) for i in range(len(semilocal.features))))
    graph, roots, expression_hash = build_roots(semilocal, outputs, production=True)
    emitter = ScalarCEmitter(graph, {name: name for name in semilocal.features})
    emitter.emit(roots)
    references = [emitter.reference(root) for root in roots]
    return "\n".join(
        [
            "struct Wb97mvPolarizedValue {",
            "  double energy_density;",
            "  double feature_derivative[7];",
            "};",
            f'inline constexpr const char* kWb97mvSemilocalExpressionIdentity = "{expression_hash}";',
            f'inline constexpr const char* kWb97mvMethodIdentity = "{method.identity}";',
            f"inline constexpr double kMolecularVv10DensityThreshold = {float(MOLECULAR_VV10_DENSITY_THRESHOLD).hex()};",
            f"inline constexpr double kWb97mvOmega = {float(short.omega).hex()};",
            f"inline constexpr double kWb97mvShortExchange = {float(short.coefficient).hex()};",
            f"inline constexpr double kWb97mvLongExchange = {float(long.coefficient).hex()};",
            f"inline constexpr double kWb97mvNonlocalB = {float(nonlocal_term.spec.b).hex()};",
            f"inline constexpr double kWb97mvNonlocalC = {float(nonlocal_term.spec.c).hex()};",
            f"inline constexpr double kWb97mvNonlocalCoefficient = {float(nonlocal_term.coefficient).hex()};",
            'inline constexpr const char* kWb97mvProductionPolicy = "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16";',
            f"inline constexpr double kWb97mvDensityThreshold = {WB97MV_DENSITY_THRESHOLD.hex()};",
            f"inline constexpr double kWb97mvSigmaThreshold = {WB97MV_SIGMA_THRESHOLD.hex()};",
            f"inline constexpr double kWb97mvTauThreshold = {WB97MV_TAU_THRESHOLD.hex()};",
            f"inline constexpr double kWb97mvSmoothLrCutoff = {WB97MV_SMOOTH_LR_CUTOFF.hex()};",
            f"inline constexpr unsigned kWb97mvSmoothLrOrder = {WB97MV_SMOOTH_LR_ORDER};",
            "inline Wb97mvPolarizedValue wb97mv_polarized(",
            "    double rho_a, double rho_b, double sigma_aa, double sigma_ab,",
            "    double sigma_bb, double tau_a, double tau_b) {",
            *emitter.lines,
            "  return {" + references[0] + ", {" + ", ".join(references[1:]) + "}};",
            "}",
            "",
        ]
    )


def emit_r2scan_polarized() -> str:
    """Emit the production-domain first-feature ABI used by native MGGA KS."""

    spec = functional("R2SCAN", spin="polarized")
    outputs = ((), *((i,) for i in range(len(spec.features))))
    graph, roots, expression_hash = build_roots(spec, outputs, production=True)
    emitter = ScalarCEmitter(graph, {name: name for name in spec.features})
    emitter.emit(roots)
    references = [emitter.reference(root) for root in roots]
    lines = [
        "struct R2scanPolarizedValue {",
        "  double energy_density;",
        "  double feature_derivative[7];",
        "};",
        f'inline constexpr const char* kR2scanPolarizedExpressionIdentity = "{expression_hash}";',
        "inline R2scanPolarizedValue r2scan_polarized(double rho_a, double rho_b,",
        "                                                double sigma_aa, double sigma_ab,",
        "                                                double sigma_bb, double tau_a,",
        "                                                double tau_b) {",
    ]
    lines.extend(emitter.lines)
    lines.append(
        "  return {" + references[0] + ", {" + ", ".join(references[1:]) + "}};"
    )
    lines.extend(["}", ""])
    return "\n".join(lines)


def emit_polarized_gga(
    spec: Any,
    *,
    value_type: str,
    function_name: str,
    identity_constant: str,
    production: bool = False,
    declarations: tuple[str, ...] = (),
) -> str:
    """Emit one polarized GGA energy/feature-gradient evaluator from FunctionalSpec.

    This is the common AOT scalar lowering boundary for GGA semilocal MethodIR
    primitives. Scientific formulas remain owned by FunctionalSpec/XC graphs;
    callers provide only stable ABI names and optional method-owned constants.
    """
    if spec.spin != "polarized" or spec.ingredients != ("rho", "sigma"):
        raise ValueError(
            "generic polarized GGA lowering requires rho/sigma FunctionalSpec"
        )
    outputs = ((), *((i,) for i in range(5)))
    graph, roots, expression_hash = build_roots(spec, outputs, production=production)
    emitter = ScalarCEmitter(graph, {name: name for name in spec.features})
    emitter.emit(roots)
    references = [emitter.reference(root) for root in roots]
    return "\n".join(
        [
            f"struct {value_type} {{",
            "  double energy_density;",
            "  double feature_derivative[5];",
            "};",
            f'inline constexpr const char* {identity_constant} = "{expression_hash}";',
            *declarations,
            f"inline {value_type} {function_name}(",
            "    double rho_a, double rho_b, double sigma_aa, double sigma_ab, double sigma_bb) {",
            "  const double tau_a = 0.0;",
            "  const double tau_b = 0.0;",
            *emitter.lines,
            "  return {" + references[0] + ", {" + ", ".join(references[1:]) + "}};",
            "}",
            "",
        ]
    )


def emit_b3lyp_polarized() -> str:
    """Emit canonical B3LYP semilocal E/vxc and its full-range exchange fraction."""

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
        raise RuntimeError(
            "B3LYP MethodIR lost its canonical full-range exchange primitive"
        )
    return emit_polarized_gga(
        semilocal,
        value_type="B3lypPolarizedValue",
        function_name="b3lyp_polarized",
        identity_constant="kB3lypSemilocalExpressionIdentity",
        production=True,
        declarations=(
            f'inline constexpr const char* kB3lypMethodIdentity = "{method.identity}";',
            f"inline constexpr double kB3lypExactExchange = {float(exchange[0].coefficient).hex()};",
        ),
    )


def emit_cam_b3lyp_polarized() -> str:
    """Emit the semilocal CAM-B3LYP primitive and its MethodIR-owned RSH constants."""

    method = resolve_method("CAM-B3LYP", spin="polarized")
    semilocal = next(
        primitive.functional
        for primitive in method.primitives
        if isinstance(primitive, SemilocalXCPrimitive)
    )
    exchange = [
        primitive
        for primitive in method.primitives
        if isinstance(primitive, RangeSeparatedExchangePrimitive)
    ]
    if len(exchange) != 2 or {primitive.operator for primitive in exchange} != {
        "short-range",
        "long-range",
    }:
        raise RuntimeError("CAM-B3LYP MethodIR lost its canonical SR/LR exchange pair")
    short = next(
        primitive for primitive in exchange if primitive.operator == "short-range"
    )
    long = next(
        primitive for primitive in exchange if primitive.operator == "long-range"
    )
    if short.omega != long.omega:
        raise RuntimeError("CAM-B3LYP MethodIR has inconsistent SR/LR omega")
    return emit_polarized_gga(
        semilocal,
        value_type="CamB3lypPolarizedValue",
        function_name="cam_b3lyp_polarized",
        identity_constant="kCamB3lypSemilocalExpressionIdentity",
        declarations=(
            f'inline constexpr const char* kCamB3lypMethodIdentity = "{method.identity}";',
            f"inline constexpr double kCamB3lypOmega = {float(short.omega).hex()};",
            f"inline constexpr double kCamB3lypShortExchange = {float(short.coefficient).hex()};",
            f"inline constexpr double kCamB3lypLongExchange = {float(long.coefficient).hex()};",
        ),
    )


def emit_pw91_polarized() -> str:
    """Emit the PW91 MethodIR semilocal primitive through the generic GGA lowerer."""

    method = resolve_method("PW91", spin="polarized")
    semilocal = next(
        primitive.functional
        for primitive in method.primitives
        if isinstance(primitive, SemilocalXCPrimitive)
    )
    return emit_polarized_gga(
        semilocal,
        value_type="Pw91PolarizedValue",
        function_name="pw91_polarized",
        identity_constant="kPw91SemilocalExpressionIdentity",
        declarations=(
            f'inline constexpr const char* kPw91MethodIdentity = "{method.identity}";',
            'inline constexpr const char* kPw91Domain = "interior-v1";',
        ),
    )


def emit_pbe_polarized() -> str:
    spec = functional("PBE", spin="polarized")
    outputs = ((), *((i,) for i in range(5)))
    graph, roots, expression_hash = build_roots(spec, outputs)
    emitter = ScalarCEmitter(graph, {name: name for name in spec.features})
    emitter.emit(roots)
    references = [emitter.reference(root) for root in roots]
    lines = [
        "struct PbePolarizedValue {",
        "  double energy_density;",
        "  double feature_derivative[5];",
        "};",
        f'inline constexpr const char* kPbePolarizedExpressionIdentity = "{expression_hash}";',
        "inline PbePolarizedValue pbe_polarized(double rho_a, double rho_b,",
        "                                        double sigma_aa, double sigma_ab,",
        "                                        double sigma_bb) {",
        "  const double tau_a = 0.0;",
        "  const double tau_b = 0.0;",
    ]
    lines.extend(emitter.lines)
    lines.append(
        "  return {" + references[0] + ", {" + ", ".join(references[1:]) + "}};"
    )
    lines.extend(["}", ""])
    return "\n".join(lines)


def emit_pbe_polarized_production() -> str:
    """Emit the tail-stable Cartesian-gradient PBE production E/vxc."""

    correlation_functions = []
    correlation_identities = []
    correlation_names = (
        "normalized_rho_a",
        "normalized_rho_b",
        "rho_scale",
        "rho_scale_sixth_root",
        "rho_scale_cuberoot",
        "gradient_ratio",
        "normalized_gradient_0",
        "normalized_gradient_1",
        "normalized_gradient_2",
    )
    for gradient_correction, suffix in ((False, "zero"), (True, "gradient")):
        graph, roots, _ = pbe_correlation_scaled_expression(
            gradient_correction=gradient_correction
        )
        graph, roots = graph.lower_small_integer_powers(roots)
        emitter = ScalarCEmitter(graph, {name: name for name in correlation_names})
        emitter.emit(roots)
        references = [emitter.reference(root) for root in roots]
        correlation_identities.append(
            graph_identity(f"pbe-correlation-scaled-{suffix}-v1", graph, roots)
        )
        correlation_functions.extend(
            [
                f"inline PbeCorrelationProductionValue pbe_correlation_scaled_{suffix}(",
                "    double normalized_rho_a, double normalized_rho_b, double rho_scale,",
                "    double rho_scale_sixth_root, double rho_scale_cuberoot,",
                "    double gradient_ratio, double normalized_gradient_0,",
                "    double normalized_gradient_1, double normalized_gradient_2) {",
                *emitter.lines,
                "  return {"
                + references[0]
                + ", {"
                + references[1]
                + ", "
                + references[2]
                + "}, {"
                + ", ".join(references[3:])
                + "}};",
                "}",
                "",
            ]
        )

    graph, roots, _ = pbe_exchange_direct_expression()
    graph, roots = graph.lower_small_integer_powers(roots)
    direct_names = (
        "rho_cuberoot",
        "rho_four_thirds",
        "reduced_gradient_0",
        "reduced_gradient_1",
        "reduced_gradient_2",
    )
    direct = ScalarCEmitter(graph, {name: name for name in direct_names})
    direct.emit(roots)
    direct_refs = [direct.reference(root) for root in roots]
    direct_identity = graph_identity("pbe-exchange-direct-v1", graph, roots)

    graph, roots, _ = pbe_exchange_reciprocal_expression()
    graph, roots = graph.lower_small_integer_powers(roots)
    reciprocal_names = (
        "rho_cuberoot",
        "rho_four_thirds",
        "reciprocal_reduced_gradient",
        "gradient_direction_0",
        "gradient_direction_1",
        "gradient_direction_2",
    )
    reciprocal = ScalarCEmitter(graph, {name: name for name in reciprocal_names})
    reciprocal.emit(roots)
    reciprocal_refs = [reciprocal.reference(root) for root in roots]
    reciprocal_identity = graph_identity("pbe-exchange-reciprocal-v1", graph, roots)

    return "\n".join(
        [
            "struct PbeCorrelationProductionValue {",
            "  double energy_density;",
            "  double rho[2];",
            "  double gradient[3];",
            "};",
            "struct PbeExchangeProductionValue {",
            "  double energy_density;",
            "  double rho;",
            "  double gradient[3];",
            "};",
            "struct PbeProductionValue {",
            "  double energy_density;",
            "  double rho[2];",
            "  double gradient[2][3];",
            "};",
            f'inline constexpr const char* kPbeCorrelationZeroProductionIdentity = "{correlation_identities[0]}";',
            f'inline constexpr const char* kPbeCorrelationGradientProductionIdentity = "{correlation_identities[1]}";',
            f'inline constexpr const char* kPbeExchangeDirectProductionIdentity = "{direct_identity}";',
            f'inline constexpr const char* kPbeExchangeReciprocalProductionIdentity = "{reciprocal_identity}";',
            'inline constexpr const char* kPbeProductionPolicy = "semilocal-scaled-v1/pbe-spin-c2-1e-18";',
            *correlation_functions,
            "inline PbeExchangeProductionValue pbe_exchange_direct(",
            "    double rho_cuberoot, double rho_four_thirds,",
            "    double reduced_gradient_0, double reduced_gradient_1,",
            "    double reduced_gradient_2) {",
            *direct.lines,
            "  return {"
            + direct_refs[0]
            + ", "
            + direct_refs[1]
            + ", {"
            + ", ".join(direct_refs[2:])
            + "}};",
            "}",
            "",
            "inline PbeExchangeProductionValue pbe_exchange_reciprocal(",
            "    double rho_cuberoot, double rho_four_thirds,",
            "    double reciprocal_reduced_gradient, double gradient_direction_0,",
            "    double gradient_direction_1, double gradient_direction_2) {",
            *reciprocal.lines,
            "  return {"
            + reciprocal_refs[0]
            + ", "
            + reciprocal_refs[1]
            + ", {"
            + ", ".join(reciprocal_refs[2:])
            + "}};",
            "}",
            "",
            "inline PbeProductionValue pbe_polarized_production(",
            "    double rho_a, double rho_b, const double gradient[2][3],",
            "    double exchange_scale = 1.0, double correlation_scale = 1.0) {",
            "  PbeProductionValue out{};",
            "  const double rho_scale = rho_a + rho_b;",
            "  if (rho_scale == 0.0) return out;",
            "  const double normalized_rho_a = rho_a / rho_scale;",
            "  const double normalized_rho_b = rho_b / rho_scale;",
            "  const double rho_scale_sixth_root = pow(rho_scale, 1.0 / 6.0);",
            "  const double rho_scale_cuberoot = cbrt(rho_scale);",
            "  double total_gradient[3]{};",
            "  double gradient_scale = rho_scale;",
            "  for (unsigned axis = 0; axis < 3; ++axis) {",
            "    total_gradient[axis] = gradient[0][axis] + gradient[1][axis];",
            "    gradient_scale = std::isfinite(total_gradient[axis])",
            "        ? std::fmax(gradient_scale, std::fabs(total_gradient[axis]))",
            "        : DBL_MAX;",
            "  }",
            "  const double gradient_ratio = rho_scale / gradient_scale;",
            "  double normalized_gradient[3]{};",
            "  bool nonzero_gradient = false;",
            "  for (unsigned axis = 0; axis < 3; ++axis) {",
            "    normalized_gradient[axis] = std::isfinite(total_gradient[axis])",
            "        ? total_gradient[axis] / gradient_scale",
            "        : gradient[0][axis] / gradient_scale + gradient[1][axis] / gradient_scale;",
            "    nonzero_gradient = nonzero_gradient || normalized_gradient[axis] != 0.0;",
            "  }",
            "  const auto correlation = nonzero_gradient",
            "      ? pbe_correlation_scaled_gradient(",
            "            normalized_rho_a, normalized_rho_b, rho_scale,",
            "            rho_scale_sixth_root, rho_scale_cuberoot, gradient_ratio,",
            "            normalized_gradient[0], normalized_gradient[1], normalized_gradient[2])",
            "      : pbe_correlation_scaled_zero(",
            "            normalized_rho_a, normalized_rho_b, rho_scale,",
            "            rho_scale_sixth_root, rho_scale_cuberoot, gradient_ratio,",
            "            0.0, 0.0, 0.0);",
            "  out.energy_density = correlation_scale * correlation.energy_density;",
            "  out.rho[0] = correlation_scale * correlation.rho[0];",
            "  out.rho[1] = correlation_scale * correlation.rho[1];",
            "  for (unsigned axis = 0; axis < 3; ++axis)",
            "    out.gradient[0][axis] = out.gradient[1][axis] =",
            "        correlation_scale * correlation.gradient[axis];",
            "  const double rho[2]{rho_a, rho_b};",
            "  for (unsigned spin = 0; spin < 2; ++spin) {",
            "    if (rho[spin] == 0.0) continue;",
            "    const double rho_cuberoot = cbrt(rho[spin]);",
            "    const double rho_four_thirds = rho[spin] * rho_cuberoot;",
            "    const double largest = std::fmax(std::fabs(gradient[spin][0]),",
            "        std::fmax(std::fabs(gradient[spin][1]), std::fabs(gradient[spin][2])));",
            "    PbeExchangeProductionValue exchange{};",
            "    if (largest == 0.0) {",
            "      exchange = pbe_exchange_direct(rho_cuberoot, rho_four_thirds, 0.0, 0.0, 0.0);",
            "    } else {",
            "      double direction[3]{};",
            "      double norm2 = 0.0;",
            "      for (unsigned axis = 0; axis < 3; ++axis) {",
            "        direction[axis] = gradient[spin][axis] / largest;",
            "        norm2 += direction[axis] * direction[axis];",
            "      }",
            "      const double norm = sqrt(norm2);",
            "      if (rho_four_thirds != 0.0 && largest <= rho_four_thirds / norm) {",
            "        exchange = pbe_exchange_direct(",
            "            rho_cuberoot, rho_four_thirds,",
            "            gradient[spin][0] / rho_four_thirds,",
            "            gradient[spin][1] / rho_four_thirds,",
            "            gradient[spin][2] / rho_four_thirds);",
            "      } else {",
            "        const double reciprocal_reduced_gradient =",
            "            (rho[spin] / largest) * (rho_cuberoot / norm);",
            "        exchange = pbe_exchange_reciprocal(",
            "            rho_cuberoot, rho_four_thirds, reciprocal_reduced_gradient,",
            "            direction[0] / norm, direction[1] / norm, direction[2] / norm);",
            "      }",
            "    }",
            "    out.energy_density += exchange_scale * exchange.energy_density;",
            "    out.rho[spin] += exchange_scale * exchange.rho;",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      out.gradient[spin][axis] += exchange_scale * exchange.gradient[axis];",
            "  }",
            "  return out;",
            "}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_if_changed(
        args.output,
        emit_lda_xc_pw()
        + emit_lda_xc_pw_polarized()
        + emit_lda_xc_pw_polarized_production()
        + emit_pbe_polarized()
        + emit_pbe_polarized_production()
        + emit_b3lyp_polarized()
        + emit_cam_b3lyp_polarized()
        + emit_wb97mv_polarized()
        + emit_pw91_polarized()
        + emit_r2scan_polarized()
        + emit_feature_policy()
        + "}  // namespace vibeqc::dft::generated\n",
    )


if __name__ == "__main__":
    main()

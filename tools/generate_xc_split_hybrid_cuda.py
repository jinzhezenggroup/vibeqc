"""Generate CUDA point evaluators for source-derived split global hybrids."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.cuda import CudaEmitter
from vibeqc_compiler.xc.libxc_maple import MapleImportError

from tools.libxc_split_hybrid import LibxcWorkPolicy, build_split_global_hybrid


_GGA_FEATURES = ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb")
_MGGA_FEATURES = (*_GGA_FEATURES, "tau_a", "tau_b")


def _selected_features(program) -> tuple[str, ...]:
    if program.family == "gga":
        if program.features != _GGA_FEATURES:
            raise MapleImportError("split GGA CUDA lowering requires rho/sigma features")
        return _GGA_FEATURES
    if program.family != "mgga":
        raise MapleImportError("split hybrid CUDA lowering requires GGA or MGGA")
    expected = (
        "rho_a",
        "rho_b",
        "sigma_aa",
        "sigma_ab",
        "sigma_bb",
        "lapl_a",
        "lapl_b",
        "tau_a",
        "tau_b",
    )
    if program.features != expected:
        raise MapleImportError("split MGGA CUDA lowering found an unknown feature layout")
    by_name = dict(zip(program.features, program.variables, strict=True))
    for name in ("lapl_a", "lapl_b"):
        derivative = program.graph.differentiate(program.energy, by_name[name])
        node = program.graph.nodes[derivative.identifier]
        if node.operation != "constant" or float(node.payload or 0.0) != 0.0:
            raise MapleImportError(
                f"split MGGA CUDA lowering refuses laplacian-dependent component {program.name}"
            )
    return _MGGA_FEATURES


def _emit_component(program, type_name: str, function_name: str) -> str:
    selected = _selected_features(program)
    by_name = dict(zip(program.features, program.variables, strict=True))
    roots = (program.energy, *(program.graph.differentiate(program.energy, by_name[name]) for name in selected))
    variables = {name: (name if name in selected else "0.0") for name in program.features}
    emitter = CudaEmitter(program.graph, variables)
    emitter.emit(roots)
    signature = ", ".join(f"double {name}" for name in selected)
    return "\n".join(
        [
            f"__device__ inline {type_name} {function_name}({signature}) {{",
            *emitter.lines,
            "  return {"
            + emitter.reference(roots[0])
            + ", {"
            + ", ".join(emitter.reference(root) for root in roots[1:])
            + "}};",
            "}",
            "",
        ]
    )


def _policy_payload(policy: LibxcWorkPolicy) -> dict[str, object]:
    return {
        "density_threshold": policy.density_threshold.hex(),
        "sigma_threshold": policy.sigma_threshold.hex(),
        "tau_threshold": policy.tau_threshold.hex(),
        "needs_tau": policy.needs_tau,
        "enforce_fhc": policy.enforce_fhc,
    }


def _emit_work_component(
    program,
    policy: LibxcWorkPolicy,
    type_name: str,
    function_name: str,
) -> str:
    """Wrap one interior Graph with pinned Libxc work_gga/work_mgga semantics."""

    selected = _selected_features(program)
    raw_name = f"{function_name}_raw"
    raw = _emit_component(program, type_name, raw_name)
    signature = ", ".join(f"double {name}" for name in selected)
    density = policy.density_threshold.hex()
    sigma_floor = (policy.sigma_threshold * policy.sigma_threshold).hex()
    lines = [
        raw.rstrip("\n"),
        f"__device__ inline {type_name} {function_name}({signature}) {{",
        f"  {type_name} out{{}};",
        "  const double total_density = rho_a + rho_b;",
        f"  if (total_density < {density}) return out;",
        f"  const double work_rho_a = fmax({density}, rho_a);",
        f"  const double work_rho_b = fmax({density}, rho_b);",
        f"  const double sigma_floor = {sigma_floor};",
        "  double work_sigma_aa = fmax(sigma_floor, sigma_aa);",
        "  double work_sigma_bb = fmax(sigma_floor, sigma_bb);",
    ]
    if policy.needs_tau:
        tau = policy.tau_threshold.hex()
        lines.extend(
            [
                f"  const double work_tau_a = fmax({tau}, tau_a);",
                f"  const double work_tau_b = fmax({tau}, tau_b);",
            ]
        )
        if policy.enforce_fhc:
            lines.extend(
                [
                    "  work_sigma_aa = fmin(work_sigma_aa, 8.0 * work_rho_a * work_tau_a);",
                    "  work_sigma_bb = fmin(work_sigma_bb, 8.0 * work_rho_b * work_tau_b);",
                ]
            )
    lines.extend(
        [
            "  const double sigma_average = 0.5 * (work_sigma_aa + work_sigma_bb);",
            "  const double work_sigma_ab =",
            "      fmax(-sigma_average, fmin(sigma_average, sigma_ab));",
        ]
    )
    arguments = [
        "work_rho_a",
        "work_rho_b",
        "work_sigma_aa",
        "work_sigma_ab",
        "work_sigma_bb",
    ]
    if len(selected) == 7:
        if not policy.needs_tau:
            raise MapleImportError(
                f"MGGA component {program.name} does not declare XC_FLAGS_NEEDS_TAU"
            )
        arguments.extend(("work_tau_a", "work_tau_b"))
    lines.extend(
        [
            f"  const auto raw = {raw_name}({', '.join(arguments)});",
            "  const double work_density = work_rho_a + work_rho_b;",
            "  out.energy_density = raw.energy_density * total_density / work_density;",
            f"  for (unsigned i = 0; i < {len(selected)}; ++i)",
            "    out.feature_derivative[i] = raw.feature_derivative[i];",
            "  return out;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def emit_split_hybrid_device_body(identifier: str) -> tuple[str, str, tuple[str, ...]]:
    """Return namespace body, type name and selected feature ABI for one method."""

    method = build_split_global_hybrid(identifier)
    if method.exchange.family != method.correlation.family:
        raise MapleImportError("split hybrid CUDA components use different families")
    selected = _selected_features(method.exchange)
    if _selected_features(method.correlation) != selected:
        raise MapleImportError("split hybrid CUDA components use different feature layouts")

    type_stem = re.sub(r"[^A-Za-z0-9]", "", identifier)
    function_stem = re.sub(r"[^a-z0-9]+", "_", identifier.lower()).strip("_")
    type_name = f"{type_stem}DeviceValue"
    function_name = f"{function_stem}_device"
    identity_name = f"k{type_stem}DeviceExpressionIdentity"
    exact_name = f"k{type_stem}ExactExchange"
    expression_identity = canonical_hash(
        {
            "schema": "split-global-hybrid-cuda-point/v2-work-policy",
            "identifier": identifier,
            "exchange": method.exchange.identity,
            "correlation": method.correlation.identity,
            "exchange_work_policy": _policy_payload(method.exchange_policy),
            "correlation_work_policy": _policy_payload(method.correlation_policy),
            "exact_exchange": str(method.exact_exchange),
            "features": selected,
        }
    )
    signature = ", ".join(f"double {name}" for name in selected)
    arguments = ", ".join(selected)
    exchange = _emit_work_component(
        method.exchange, method.exchange_policy, type_name, f"{function_name}_exchange"
    )
    correlation = _emit_work_component(
        method.correlation,
        method.correlation_policy,
        type_name,
        f"{function_name}_correlation",
    )
    body = "\n".join(
        [
            f"struct {type_name} {{",
            "  double energy_density{};",
            f"  double feature_derivative[{len(selected)}]{{}};",
            "};",
            f'inline constexpr const char* {identity_name} = "{expression_identity}";',
            f"inline constexpr double {exact_name} = {float(method.exact_exchange).hex()};",
            exchange.rstrip("\n"),
            correlation.rstrip("\n"),
            f"__device__ inline {type_name} {function_name}({signature}) {{",
            f"  const auto exchange = {function_name}_exchange({arguments});",
            f"  const auto correlation = {function_name}_correlation({arguments});",
            f"  {type_name} out{{}};",
            "  out.energy_density = exchange.energy_density + correlation.energy_density;",
            f"  for (unsigned i = 0; i < {len(selected)}; ++i)",
            "    out.feature_derivative[i] =",
            "        exchange.feature_derivative[i] + correlation.feature_derivative[i];",
            "  return out;",
            "}",
            "",
        ]
    )
    return body, type_name, selected


def emit_split_hybrid_device(identifier: str) -> str:
    body, _, _ = emit_split_hybrid_device_body(identifier)
    return "\n".join(
        [
            "// Generated from pinned Libxc split global-hybrid sources; do not edit.",
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
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_if_changed(args.output, emit_split_hybrid_device(args.method))


if __name__ == "__main__":
    main()

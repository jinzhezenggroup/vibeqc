"""Generate split-hybrid CUDA points with the pinned Libxc MGGA work contract."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.integral.cuda import CudaEmitter
from vibeqc_compiler.xc.libxc_maple import MapleImportError
from vibeqc_compiler.xc.split_hybrid_program import (
    _bound_component,
    build_split_global_hybrid,
)

_GGA_FEATURES = ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb")
_MGGA_FEATURES = (*_GGA_FEATURES, "tau_a", "tau_b")
# PySCF's pinned Libxc 7.0.0 oracle enables this build-wide option. Registration
# flags alone omit the flag added by xc_func_init when that option is compiled.
_PINNED_LIBXC_GLOBAL_FHC = True

if TYPE_CHECKING:
    from vibeqc_compiler.xc.libxc_bulk import BulkProgram


@dataclass(frozen=True)
class SplitHybridWorkPolicy:
    density_threshold: float
    sigma_floor: float
    tau_floor: float
    enforce_fhc: bool
    source_identity: str


def _component_work_policy(program: BulkProgram) -> SplitHybridWorkPolicy:
    """Read component thresholds/flags; pin the worker semantics being adapted."""
    if program.family != "mgga":
        raise MapleImportError(
            "split CUDA production points require an audited MGGA work policy"
        )
    record = _bound_component(
        program.name, allow_hybrid_exchange=program.name.startswith("HYB_")
    )
    flags = set(re.split(r"\s*\|\s*", record["flags"]))
    if "XC_FLAGS_NEEDS_TAU" not in flags:
        raise MapleImportError(
            "split MGGA work policy requires an explicit tau contract"
        )
    density = float(record["bindings"]["p_a_dens_threshold"])
    if not math.isfinite(density) or density <= 0.0:
        raise MapleImportError(
            "split MGGA work policy requires a positive density threshold"
        )
    root = asset_path("upstream/libxc/7.0.0")
    worker = root / "work_mgga_inc.c"
    initialization = root / "functionals.c"
    required = {
        initialization: (
            "func->info->flags = func->info->flags | XC_FLAGS_ENFORCE_FHC;",
            "func->sigma_threshold = pow(func->info->dens_threshold, 4.0/3.0);",
            "func->tau_threshold   = 1e-20;",
        ),
        worker: (
            "if(dens < p->dens_threshold)",
            "my_rho[0] = m_max(p->dens_threshold, VAR(rho, ip, 0));",
            "my_sigma[0] = m_max(p->sigma_threshold * p->sigma_threshold, VAR(sigma, ip, 0));",
            "my_tau[0] = m_max(p->tau_threshold, VAR(tau, ip, 0));",
            "my_sigma[0] = m_min(my_sigma[0], 8.0*my_rho[0]*my_tau[0]);",
            "s_ave = 0.5*(my_sigma[0] + my_sigma[2]);",
            "my_sigma[1] = (my_sigma[1] >= -s_ave ? my_sigma[1] : -s_ave);",
            "my_sigma[1] = (my_sigma[1] <= +s_ave ? my_sigma[1] : +s_ave);",
        ),
    }
    for path, snippets in required.items():
        source = path.read_text(encoding="utf-8")
        if any(snippet not in source for snippet in snippets):
            raise MapleImportError(f"pinned Libxc work policy changed: {path.name}")
    sigma = density ** (4.0 / 3.0)
    return SplitHybridWorkPolicy(
        density,
        sigma * sigma,
        1.0e-20,
        "XC_FLAGS_ENFORCE_FHC" in flags or _PINNED_LIBXC_GLOBAL_FHC,
        canonical_hash({path.name: file_hash(path) for path in required}),
    )


def _selected_features(program: BulkProgram) -> tuple[str, ...]:
    if program.family == "gga":
        if program.features != _GGA_FEATURES:
            raise MapleImportError(
                "split GGA CUDA lowering requires rho/sigma features"
            )
        return _GGA_FEATURES
    if program.family != "mgga":
        raise MapleImportError("split hybrid CUDA lowering requires GGA or MGGA")
    expected = (*_GGA_FEATURES, "lapl_a", "lapl_b", "tau_a", "tau_b")
    if program.features != expected:
        raise MapleImportError(
            "split MGGA CUDA lowering found an unknown feature layout"
        )
    by_name = dict(zip(program.features, program.variables, strict=True))
    for name in ("lapl_a", "lapl_b"):
        derivative = program.graph.differentiate(program.energy, by_name[name])
        node = program.graph.nodes[derivative.identifier]
        # Do not round an arbitrarily small nonzero exact coefficient to FP zero.
        if node.operation != "constant" or node.payload != 0:
            raise MapleImportError(
                f"split MGGA CUDA lowering refuses laplacian-dependent component {program.name}"
            )
    return _MGGA_FEATURES


def _emit_work_wrapper(
    type_name: str, function_name: str, policy: SplitHybridWorkPolicy
) -> str:
    signature = ", ".join(f"double {name}" for name in _MGGA_FEATURES)
    density_threshold = float(policy.density_threshold).hex()
    sigma_floor = float(policy.sigma_floor).hex()
    tau_floor = float(policy.tau_floor).hex()
    lines = [
        f"__device__ inline {type_name} {function_name}({signature}) {{",
        "  const double total = rho_a + rho_b;",
        f"  if (total < {density_threshold}) return {{}};",
        f"  const double work_rho_a = fmax({density_threshold}, rho_a);",
        f"  const double work_rho_b = fmax({density_threshold}, rho_b);",
        f"  double work_sigma_aa = fmax({sigma_floor}, sigma_aa);",
        f"  double work_sigma_bb = fmax({sigma_floor}, sigma_bb);",
        f"  const double work_tau_a = fmax({tau_floor}, tau_a);",
        f"  const double work_tau_b = fmax({tau_floor}, tau_b);",
    ]
    if policy.enforce_fhc:
        lines.extend(
            (
                "  work_sigma_aa = fmin(work_sigma_aa, 8.0 * work_rho_a * work_tau_a);",
                "  work_sigma_bb = fmin(work_sigma_bb, 8.0 * work_rho_b * work_tau_b);",
            )
        )
    lines.extend(
        (
            "  const double average = 0.5 * (work_sigma_aa + work_sigma_bb);",
            "  const double work_sigma_ab = fmax(-average, fmin(average, sigma_ab));",
            f"  auto out = {function_name}_interior(",
            "      work_rho_a, work_rho_b, work_sigma_aa, work_sigma_ab, work_sigma_bb,",
            "      work_tau_a, work_tau_b);",
            "  // Libxc returns epsilon at work inputs. Integrate with the original density.",
            "  out.energy_density *= total / (work_rho_a + work_rho_b);",
            "  // vxc follows work_mgga: derivatives at work inputs, not AD through clipping.",
            "  return out;",
            "}",
            "",
        )
    )
    return "\n".join(lines)


def _emit_component(
    program: BulkProgram,
    type_name: str,
    function_name: str,
    policy: SplitHybridWorkPolicy,
) -> str:
    selected = _selected_features(program)
    by_name = dict(zip(program.features, program.variables, strict=True))
    roots = (
        program.energy,
        *(
            program.graph.differentiate(program.energy, by_name[name])
            for name in selected
        ),
    )
    variables = {name: name if name in selected else "0.0" for name in program.features}
    emitter = CudaEmitter(program.graph, variables)
    emitter.emit(roots)
    signature = ", ".join(f"double {name}" for name in selected)
    interior = "\n".join(
        [
            f"__device__ inline {type_name} {function_name}_interior({signature}) {{",
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
    return interior + _emit_work_wrapper(type_name, function_name, policy)


def emit_split_hybrid_device_body(identifier: str) -> tuple[str, str, tuple[str, ...]]:
    """Return namespace body, type name and selected feature ABI for one method."""
    method = build_split_global_hybrid(identifier)
    if method.exchange.family != method.correlation.family:
        raise MapleImportError("split hybrid CUDA components use different families")
    selected = _selected_features(method.exchange)
    if _selected_features(method.correlation) != selected:
        raise MapleImportError(
            "split hybrid CUDA components use different feature layouts"
        )
    x_policy = _component_work_policy(method.exchange)
    c_policy = _component_work_policy(method.correlation)
    type_stem = re.sub(r"[^A-Za-z0-9]", "", identifier)
    function_stem = re.sub(r"[^a-z0-9]+", "_", identifier.lower()).strip("_")
    type_name = f"{type_stem}DeviceValue"
    function_name = f"{function_stem}_device"
    expression_identity = canonical_hash(
        {
            "schema": "split-global-hybrid-cuda-point/v3",
            "identifier": identifier,
            "exchange": method.exchange.identity,
            "correlation": method.correlation.identity,
            "exact_exchange": str(method.exact_exchange),
            "features": selected,
            "exchange_work_policy": asdict(x_policy),
            "correlation_work_policy": asdict(c_policy),
        }
    )
    signature = ", ".join(f"double {name}" for name in selected)
    arguments = ", ".join(selected)
    exchange = _emit_component(
        method.exchange, type_name, f"{function_name}_exchange", x_policy
    )
    correlation = _emit_component(
        method.correlation, type_name, f"{function_name}_correlation", c_policy
    )
    body = "\n".join(
        [
            f"struct {type_name} {{",
            "  double energy_density{};",
            f"  double feature_derivative[{len(selected)}]{{}};",
            "};",
            f'inline constexpr const char* k{type_stem}DeviceExpressionIdentity = "{expression_identity}";',
            f"inline constexpr double k{type_stem}ExactExchange = {float(method.exact_exchange).hex()};",
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
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_if_changed(args.output, emit_split_hybrid_device(args.method))


if __name__ == "__main__":
    main()

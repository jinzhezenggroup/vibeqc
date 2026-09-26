"""Generate CUDA point evaluators for source-derived split global hybrids."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.cuda import CudaEmitter
from vibeqc_compiler.xc.libxc_maple import MapleImportError

if TYPE_CHECKING:
    from vibeqc_compiler.xc.libxc_bulk import BulkProgram

from tools.libxc_split_hybrid import build_split_global_hybrid

_GGA_FEATURES = ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb")
_MGGA_FEATURES = (*_GGA_FEATURES, "tau_a", "tau_b")


def _selected_features(program: BulkProgram) -> tuple[str, ...]:
    if program.family == "gga":
        if program.features != _GGA_FEATURES:
            raise MapleImportError(
                "split GGA CUDA lowering requires rho/sigma features"
            )
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
        raise MapleImportError(
            "split MGGA CUDA lowering found an unknown feature layout"
        )
    by_name = dict(zip(program.features, program.variables, strict=True))
    for name in ("lapl_a", "lapl_b"):
        derivative = program.graph.differentiate(program.energy, by_name[name])
        node = program.graph.nodes[derivative.identifier]
        if node.operation != "constant" or node.payload != 0:
            raise MapleImportError(
                f"split MGGA CUDA lowering refuses laplacian-dependent component {program.name}"
            )
    return _MGGA_FEATURES


def _emit_component(program: BulkProgram, type_name: str, function_name: str) -> str:
    selected = _selected_features(program)
    by_name = dict(zip(program.features, program.variables, strict=True))
    roots = (
        program.energy,
        *(
            program.graph.differentiate(program.energy, by_name[name])
            for name in selected
        ),
    )
    variables = {
        name: (name if name in selected else "0.0") for name in program.features
    }
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


def emit_split_hybrid_device(identifier: str) -> str:
    method = build_split_global_hybrid(identifier)
    if method.exchange.family != method.correlation.family:
        raise MapleImportError("split hybrid CUDA components use different families")
    selected = _selected_features(method.exchange)
    if _selected_features(method.correlation) != selected:
        raise MapleImportError(
            "split hybrid CUDA components use different feature layouts"
        )

    type_stem = re.sub(r"[^A-Za-z0-9]", "", identifier)
    function_stem = re.sub(r"[^a-z0-9]+", "_", identifier.lower()).strip("_")
    type_name = f"{type_stem}DeviceValue"
    function_name = f"{function_stem}_device"
    identity_name = f"k{type_stem}DeviceExpressionIdentity"
    exact_name = f"k{type_stem}ExactExchange"
    expression_identity = canonical_hash(
        {
            "schema": "split-global-hybrid-cuda-point/v1",
            "identifier": identifier,
            "exchange": method.exchange.identity,
            "correlation": method.correlation.identity,
            "exact_exchange": str(method.exact_exchange),
            "features": selected,
        }
    )
    signature = ", ".join(f"double {name}" for name in selected)
    arguments = ", ".join(selected)
    exchange = _emit_component(method.exchange, type_name, f"{function_name}_exchange")
    correlation = _emit_component(
        method.correlation, type_name, f"{function_name}_correlation"
    )
    return "\n".join(
        [
            "// Generated from pinned Libxc split global-hybrid sources; do not edit.",
            "#pragma once",
            "#include <cmath>",
            "namespace vibeqc::dft::generated {",
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

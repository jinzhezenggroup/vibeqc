"""Host lowering for compiler-owned primitive GFN2 S/D/Q integrals.

The generated artifact owns only primitive Cartesian mathematics. Contracted
coefficients, shell traversal, screening, Cartesian-to-spherical transforms,
matrix packing, and reverse origin translation remain native runtime policy.
"""

from __future__ import annotations

from functools import cache
from itertools import product
from typing import TYPE_CHECKING

from .gfn2_sdq import (
    GFN2_SDQ_COMPONENTS,
    Gfn2SdqPrimitiveKernel,
    build_gfn2_sdq_primitive_kernel,
)
from .scalar_c import ScalarCEmitter
from .shell_spec import AXES, cartesian_components

if TYPE_CHECKING:
    from .expr import Expr

_MODES = (
    ("overlap", 1, False),
    ("overlap_gradient", 1, True),
    ("sdq_values", len(GFN2_SDQ_COMPONENTS), False),
    ("sdq", len(GFN2_SDQ_COMPONENTS), True),
)


def gfn2_sdq_cpu_inventory() -> dict[str, object]:
    """Describe the bounded native primitive contract."""
    component_pairs = sum(
        len(cartesian_components(bra)) * len(cartesian_components(ket))
        for bra, ket in product(range(3), repeat=2)
    )
    return {
        "schema": "vibeqc.gfn2_sdq_cpu",
        "version": 1,
        "precision": "fp64",
        "public_maximum_angular": 2,
        "operator_origin": "ket",
        "components": list(GFN2_SDQ_COMPONENTS),
        "gradient_center": "ket",
        "component_pairs": component_pairs,
        "entry_points": [mode[0] for mode in _MODES],
    }


def _variables() -> dict[str, str]:
    variables = {"alpha": "bra_alpha", "beta": "ket_alpha"}
    for axis in AXES:
        variables[f"a_{axis}"] = "0.0"
    for index, axis in enumerate(AXES):
        variables[f"b_{axis}"] = f"vector[{index}]"
    return variables


@cache
def _kernel(
    bra_angular: int,
    ket_angular: int,
    bra_component: str,
    ket_component: str,
) -> Gfn2SdqPrimitiveKernel:
    return build_gfn2_sdq_primitive_kernel(
        (bra_angular, ket_angular), (bra_component, ket_component)
    )


def _roots(
    kernel: Gfn2SdqPrimitiveKernel, value_count: int, gradient: bool
) -> tuple[Expr, ...]:
    values = tuple(kernel.values[:value_count])
    if not gradient:
        return values
    ket_gradients = tuple(
        root for axis in kernel.gradients[1] for root in axis[:value_count]
    )
    return (*values, *ket_gradients)


def _emit_shell_pair(
    tag: str,
    bra_angular: int,
    ket_angular: int,
    value_count: int,
    gradient: bool,
) -> str:
    bra_components = cartesian_components(bra_angular)
    ket_components = cartesian_components(ket_angular)
    ket_count = len(ket_components)
    name = f"evaluate_gfn2_{tag}_{bra_angular}{ket_angular}"
    lines = [
        f"inline bool {name}(unsigned bra_component, unsigned ket_component,",
        "    double bra_alpha, double ket_alpha, const double* vector,",
        "    Gfn2SdqPrimitive& result) noexcept {",
        f"  switch (bra_component * {ket_count}U + ket_component) {{",
    ]
    for bra_index, bra_component in enumerate(bra_components):
        for ket_index, ket_component in enumerate(ket_components):
            kernel = _kernel(bra_angular, ket_angular, bra_component, ket_component)
            roots = _roots(kernel, value_count, gradient)
            emitter = ScalarCEmitter(kernel.graph, _variables())
            emitter.emit(roots)
            case = bra_index * ket_count + ket_index
            lines.append(f"    case {case}U: {{")
            lines.extend("    " + line for line in emitter.lines)
            for index, root in enumerate(kernel.values[:value_count]):
                lines.append(
                    f"      result.values[{index}] = {emitter.reference(root)};"
                )
            if gradient:
                for axis in range(3):
                    for index, root in enumerate(
                        kernel.gradients[1][axis][:value_count]
                    ):
                        lines.append(
                            f"      result.ket_gradient[{axis}][{index}] = "
                            f"{emitter.reference(root)};"
                        )
            lines += ["      return true;", "    }"]
    lines += ["  }", "  return false;", "}"]
    return "\n".join(lines)


def _emit_dispatch(tag: str) -> str:
    lines = [
        f"inline bool evaluate_gfn2_{tag}_primitive(",
        "    unsigned bra_angular, unsigned ket_angular, unsigned bra_component,",
        "    unsigned ket_component, double bra_alpha, double ket_alpha,",
        "    const double* vector, Gfn2SdqPrimitive& result) noexcept {",
        "  if (bra_angular > 2U || ket_angular > 2U) return false;",
        "  constexpr unsigned components[] = {1U, 3U, 6U};",
        "  if (bra_component >= components[bra_angular] || ket_component >= components[ket_angular]) return false;",
        "  if (!vector || !std::isfinite(bra_alpha) || !std::isfinite(ket_alpha) || bra_alpha <= 0.0 || ket_alpha <= 0.0) return false;",
        "  for (unsigned axis = 0; axis < 3U; ++axis) if (!std::isfinite(vector[axis])) return false;",
        "  switch (bra_angular * 3U + ket_angular) {",
    ]
    for bra, ket in product(range(3), repeat=2):
        lines.append(
            f"    case {bra * 3 + ket}U: return evaluate_gfn2_{tag}_{bra}{ket}("
            "bra_component, ket_component, bra_alpha, ket_alpha, vector, result);"
        )
    lines += ["  }", "  return false;", "}"]
    return "\n".join(lines)


@cache
def emit_gfn2_sdq_cpu() -> str:
    """Emit cost-bounded s/p/d primitive entry points for each runtime use."""
    helpers = []
    dispatches = []
    for tag, value_count, gradient in _MODES:
        helpers.extend(
            _emit_shell_pair(tag, bra, ket, value_count, gradient)
            for bra, ket in product(range(3), repeat=2)
        )
        dispatches.append(_emit_dispatch(tag))
    return "\n".join(
        [
            "// Generated by tools/generate_gfn2_sdq_native.py; do not edit.",
            "#pragma once",
            "#include <cmath>",
            "namespace vibeqc::xtb::generated {",
            "struct Gfn2SdqPrimitive {",
            f"  double values[{len(GFN2_SDQ_COMPONENTS)}]{{}};",
            f"  double ket_gradient[3][{len(GFN2_SDQ_COMPONENTS)}]{{}};",
            "};",
            *helpers,
            *dispatches,
            "}  // namespace vibeqc::xtb::generated",
            "",
        ]
    )

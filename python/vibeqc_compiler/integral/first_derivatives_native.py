"""Raw first-derivative component tiles from the existing S/T/V and ERI DAGs.

This is an explicit CPU lowering, not a new recurrence or method driver. ERI
unit cotangents use the same weighted primitive emitter as native forces.
"""

from itertools import product

from vibeqc_compiler.common.provenance import canonical_hash

from .blocks import RawBlock, WeightedDerivative
from .bounded_component import emit_bounded_component
from .ir import OperatorFamily
from .ir_serialization import integral_to_payload
from .shell_spec import cartesian_components
from .weighted_eri import build_weighted_eri_kernel
from .weighted_eri_native import emit_weighted_eri_primitive_header


def first_component_identity(integral, indices):
    return canonical_hash(
        {
            "schema": "vibeqc.first-components.cpu.v1",
            "integral": integral_to_payload(integral),
            "components": tuple(indices),
        }
    )


def validate_first_components(integral, indices):
    """Reject unsupported semantics rather than relabel another derivative."""
    if integral.derivative is None or integral.derivative.order != 1:
        raise ValueError("first component execution requires derivative order one")
    if any(l > 3 for l in integral.signature.angular):
        raise ValueError("first component execution supports s/p/d/f")
    if integral.requested_derivative_centers != integral.operator.centers:
        raise ValueError("all mathematical derivative centers must be requested")
    if (
        not 1 <= len(indices) <= 64
        or len(set(indices)) != len(indices)
        or any(
            type(i) is not int or not 0 <= i < integral.signature.component_count
            for i in indices
        )
    ):
        raise ValueError("select one to 64 distinct in-range AO components")
    if len(integral.contractions) != 1:
        raise ValueError("first component execution requires one consumer")
    consumer = integral.contractions[0]
    if integral.operator.family == OperatorFamily.FOUR_CENTER_ERI:
        if not isinstance(consumer, WeightedDerivative) or (
            consumer.output_sign,
            consumer.weights.sign,
            consumer.weights.prefactor,
        ) != (1, 1, 1):
            raise ValueError(
                "raw ERI components require unit weighted-primitive factors"
            )
    elif integral.operator.family in (
        OperatorFamily.OVERLAP,
        OperatorFamily.KINETIC,
        OperatorFamily.NUCLEAR_ATTRACTION,
    ):
        if not isinstance(consumer, RawBlock):
            raise ValueError("raw S/T/V component consumer required")
    else:
        raise ValueError("first components support S/T/V and full Coulomb ERIs only")


def emit_first_components(integral, indices):
    """Emit a selected raw tile, retaining normalization/atom mapping outside."""
    indices = tuple(indices)
    validate_first_components(integral, indices)
    ncenter = len(integral.operator.centers)
    nexponent = len(integral.signature.shells)
    if integral.operator.family == OperatorFamily.FOUR_CENTER_ERI:
        kernel = build_weighted_eri_kernel(integral, indices)
        source = emit_weighted_eri_primitive_header(((kernel, "first"),), backend="cpu")
        evaluate = f"""
    double weights[{len(indices)}]{{}};
    weights[component] = 1.0;
    vibeqc::scf::generated_weighted_eri::Gradient result{{}};
    if (!vibeqc::scf::generated_weighted_eri::first_primitive(
            input, input + 4, weights, result)) return false;
    output[0] = result.value;
    for (unsigned c = 0; c < 4; ++c)
      for (unsigned a = 0; a < 3; ++a) output[1 + 3*c + a] = result.center[c][a];
    return true;
"""
    else:
        components = tuple(
            product(*(cartesian_components(l) for l in integral.signature.angular))
        )
        source = "\n".join(
            emit_bounded_component(integral, components[i], backend="cpu").replace(
                'extern "C" void evaluate(', f"static void first_{packed}("
            )
            for packed, i in enumerate(indices)
        )
        cases = "\n".join(
            f"      case {i}: first_{i}(input, output); return true;"
            for i in range(len(indices))
        )
        evaluate = (
            "    switch (component) {\n"
            + cases
            + "\n      default: return false;\n    }"
        )
    identity = first_component_identity(integral, indices)
    return (
        source
        + f"""
#include "integrals/first_component_runtime.hpp"
namespace {{
struct FirstProgram {{
  static constexpr std::size_t exponents = {nexponent};
  static constexpr std::size_t inputs = {nexponent + 3 * ncenter};
  static constexpr std::size_t outputs = {1 + 3 * ncenter};
  static constexpr std::size_t components = {len(indices)};
  static bool evaluate(const double* input, std::size_t component, double* output) {{
{evaluate}
  }}
}};
}}
extern "C" const char* vibeqc_first_identity_v1() {{ return "{identity}"; }}
extern "C" int vibeqc_first_sum_v1(const double* records, std::size_t count,
    std::size_t stride, double* output, std::size_t size) {{
  return vibeqc::integrals::contract_first_components<FirstProgram>(records, count, stride, output, size);
}}
"""
    )

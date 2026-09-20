"""Lower integral/weighted-derivative algebra into shared CUDA fragments.

The existing IR and contraction builders own the mathematical definitions;
consumers reuse these fragments without duplicating their expressions."""

from __future__ import annotations

from ..cuda import CudaEmitter
from ..cuda_schedule import (
    AlgebraForm,
    ScheduleIR,
)
from ..expr import PowerLowering
from ..ir import IntegralIR, KernelConsumer, build_integral_ir
from ..shell_class import (
    build_packed_force_geometry_algebra,
    build_weighted_shell_contraction_kernel,
)
from ..shell_spec import (
    AXES,
    ShellClassSpec,
)


def _emit_triple_pair_matchings(call: str, *, gradients: bool) -> str:
    """Enumerate unordered disjoint triples needed by six-quantum f/f pairs.

    Both value and derivative consumers need all 15 complete Wick pairings.
    Sorting removed-bit masks visits each matching once, including pairs whose
    endpoints interleave. The value consumer omits only gradient arguments.
    """
    source = """  if constexpr (PairOrder >= 6U) {
    for (unsigned first = 0; first < PairOrder; ++first) {
      for (unsigned second = first + 1U; second < PairOrder; ++second) {
        if (axes[first] != axes[second]) continue;
        const unsigned first_removed =
            (1U << first) | (1U << second);
        for (unsigned third = 0; third < PairOrder; ++third) {
          for (unsigned fourth = third + 1U; fourth < PairOrder; ++fourth) {
            if (axes[third] != axes[fourth]) continue;
            const unsigned second_removed =
                (1U << third) | (1U << fourth);
            if (first_removed >= second_removed ||
                (first_removed & second_removed) != 0U) continue;
            for (unsigned fifth = 0; fifth < PairOrder; ++fifth) {
              for (unsigned sixth = fifth + 1U; sixth < PairOrder; ++sixth) {
                if (axes[fifth] != axes[sixth]) continue;
                const unsigned third_removed =
                    (1U << fifth) | (1U << sixth);
                if (second_removed >= third_removed ||
                    ((first_removed | second_removed) & third_removed) != 0U) {
                  continue;
                }
                VIBEQC_PAIR_MATCHING_CALL(
                    term, axes, shifts, shift_gradients,
                    inverse_two_exponent, subset,
                    first_removed | second_removed | third_removed, 3U);
              }
            }
          }
        }
      }
    }
  }
"""
    source = source.replace("VIBEQC_PAIR_MATCHING_CALL", call)
    if not gradients:
        source = source.replace("shifts, shift_gradients,", "shifts,")
    return source


def _emit_packed_force_geometry_algebra_cuda(
    spec: ShellClassSpec,
    *,
    integral: IntegralIR | None = None,
) -> str:
    """Lower backend-neutral packed geometry roots into CUDA field stores.

    Pair-product scales remain execution metadata because orientation chooses
    which input pair coefficient occupies each slot.  Every scalar derived
    from the product centers, exponents, coordinates, and weighted primitive
    coefficients is emitted from the shared mathematical geometry graph.
    """

    selected_integral = integral or build_integral_ir(spec)
    if selected_integral.spec != spec:
        raise ValueError("packed geometry spec does not match its integral IR")
    algebra = build_packed_force_geometry_algebra()
    pair_shift_rows = 4 if spec.angular[3] != 0 else 3
    decay_gradient_rows = (
        4 if 3 in selected_integral.independent_derivative_centers else 3
    )
    variable_code = {
        "p": "p",
        "q": "q",
        "first_reduced_exponent": "first_pair.reduced_exponent",
        "second_reduced_exponent": "second_pair.reduced_exponent",
        "first_weighted_coefficient": "first_pair.weighted_coefficient",
        "second_weighted_coefficient": "second_pair.weighted_coefficient",
    }
    for center in ("first", "second", "third", "fourth"):
        for axis in AXES:
            variable_code[f"{center}_coordinate_{axis}"] = f"{center}.{axis}"
    for axis in AXES:
        variable_code[f"product_p_{axis}"] = f"first_pair.product_center.{axis}"
        variable_code[f"product_q_{axis}"] = f"second_pair.product_center.{axis}"

    field_targets = (
        "geometry.rho",
        "geometry.inverse_two_p",
        "geometry.inverse_two_q",
        *(
            f"geometry.pair_shifts[{center}][{axis}]"
            for center in range(pair_shift_rows)
            for axis in range(3)
        ),
        *(f"geometry.difference[{axis}]" for axis in range(3)),
        *(
            f"geometry.decay_gradients[{center}][{axis}]"
            for center in range(decay_gradient_rows)
            for axis in range(3)
        ),
        "argument_squared_distance",
        None,
        "geometry.prefactor",
        "geometry.primitive_coefficient",
    )
    source_roots = algebra.roots_for_pair_shift_rows(
        pair_shift_rows,
        decay_gradient_rows=decay_gradient_rows,
    )
    root_specs = tuple(zip(source_roots, field_targets, strict=True))
    graph, roots = algebra.graph.apply_algebra_form(
        source_roots,
        AlgebraForm.BINARY,
        PowerLowering.SMALL_INTEGER,
    )
    emitter = CudaEmitter(graph, variable_code)
    emitter.lines.append("  double argument_squared_distance;")
    for (__, target), root in zip(root_specs, roots, strict=True):
        if target is None:
            emitter.emit((root,))
            emitter.lines.append(
                f"  boys_values<{spec.maximum_force_coulomb_order}>"
                f"({emitter.reference(root)}, geometry.boys);"
            )
        else:
            emitter.emit_assignment(root, target)
    return "\n".join(emitter.lines)


def _packed_force_integral(
    spec: ShellClassSpec,
    integral: IntegralIR | None,
) -> IntegralIR:
    """Select force metadata for a packed helper beside value-only plans.

    Production value-only shells still emit the compatibility force symbols
    used by the registry, even though those symbols are never launched.  They
    therefore need the historical default force IR rather than an empty
    FOCK-only derivative basis.
    """

    if integral is None or KernelConsumer.FORCE not in integral.consumers:
        return build_integral_ir(spec)
    return integral


def _emit_weighted_component_gradient_cuda(
    spec: ShellClassSpec,
    schedule: ScheduleIR,
    *,
    integral: IntegralIR | None = None,
) -> str:
    """Emit one shell-wide weighted gradient with horizontal symbolic CSE."""

    selected_integral = _packed_force_integral(spec, integral)
    if selected_integral.spec != spec:
        raise ValueError("weighted gradient spec does not match its integral IR")
    maximum_order = spec.maximum_force_coulomb_order
    side = maximum_order + 1
    pair_shift_rows = 4 if spec.angular[3] != 0 else 3
    decay_gradient_rows = (
        4 if 3 in selected_integral.independent_derivative_centers else 3
    )
    geometry_algebra = _emit_packed_force_geometry_algebra_cuda(
        spec,
        integral=selected_integral,
    )
    compact_geometry = f"""/**
 * Geometry retained by packed force lanes.
 *
 * The generic geometry also materializes Cartesian coordinate powers and
 * (-2 rho) powers for component-at-a-time Coulomb evaluation.  The weighted
 * contraction below expands those expressions directly, so retaining the two
 * tables would waste per-lane shared memory and reduce resident warps.
 */
struct GeneratedDpppPackedForceGeometry {{
  double inverse_two_p;
  double inverse_two_q;
  double rho;
  double product_scales[3];
  // The compact default stores three independent center decay rows.  An
  // explicit IR that differentiates center four requests a fourth row.
  double pair_shifts[{pair_shift_rows}][3];
  double difference[3];
  double decay_gradients[{decay_gradient_rows}][3];
  double boys[{side}];
  double prefactor;
  double primitive_coefficient;
}};

__device__ __forceinline__ void generated_dppp_make_packed_force_geometry(
    const GeneratedDpppPrimitivePairData& first_pair,
    const GeneratedDpppPrimitivePairData& second_pair,
    bool first_pair_reversed,
    bool second_pair_reversed,
    const GeneratedDpppVec3& first,
    const GeneratedDpppVec3& second,
    const GeneratedDpppVec3& third,
    const GeneratedDpppVec3& fourth,
    GeneratedDpppPackedForceGeometry& geometry) {{
  const double p = first_pair.exponent_sum;
  const double q = second_pair.exponent_sum;
  geometry.product_scales[0] = first_pair_reversed
      ? first_pair.second_product_scale : first_pair.first_product_scale;
  geometry.product_scales[1] = first_pair_reversed
      ? first_pair.first_product_scale : first_pair.second_product_scale;
  geometry.product_scales[2] = second_pair_reversed
      ? second_pair.second_product_scale : second_pair.first_product_scale;
{geometry_algebra}
}}
"""
    kernel = build_weighted_shell_contraction_kernel(
        spec,
        integral=selected_integral,
    )
    variable_code = {
        "inverse_two_p": "geometry.inverse_two_p",
        "inverse_two_q": "geometry.inverse_two_q",
        "rho": "geometry.rho",
        "first_product_scale": "geometry.product_scales[0]",
        "second_product_scale": "geometry.product_scales[1]",
        "third_product_scale": "geometry.product_scales[2]",
        # The compact packed record stores one ket product scale.  The other
        # is its exact complement, including reversed shell-pair orientation.
        "fourth_product_scale": "1.0 - geometry.product_scales[2]",
        "prefactor": "geometry.prefactor",
    }
    for axis_index, axis in enumerate(AXES):
        variable_code[f"difference_{axis}"] = f"geometry.difference[{axis_index}]"
        for center, prefix in enumerate(("pa", "pb", "qc", "qd")):
            variable_code[f"{prefix}_{axis}"] = (
                f"geometry.pair_shifts[{center}][{axis_index}]"
            )
        for center_index, center in enumerate(("first", "second", "third", "fourth")):
            if center_index >= decay_gradient_rows:
                continue
            variable_code[f"decay_{center}_{axis}"] = (
                f"geometry.decay_gradients[{center_index}][{axis_index}]"
            )
    for order in range(spec.maximum_force_coulomb_order + 1):
        variable_code[f"boys_{order}"] = f"geometry.boys[{order}]"
    for component in range(spec.component_count):
        variable_code[f"component_weight_{component}"] = (
            f"component_weights[{component}]"
        )

    binary_roots = tuple(
        kernel.gradients[center][coordinate]
        for center in selected_integral.independent_derivative_centers
        for coordinate in range(3)
    )
    graph, roots = kernel.graph.apply_algebra_form(
        binary_roots,
        schedule.algebra_form,
    )
    materialization_plan = graph.materialization_plan(
        roots,
        schedule.algebra_placement.materialization_policy(),
        schedule.algebra_ordering,
        schedule.algebra_fusion,
    )
    emitter = CudaEmitter(
        graph,
        variable_code,
        materialization_plan=materialization_plan,
    )
    emitter.emit(roots)
    lines = [
        compact_geometry.rstrip(),
        "",
        "/** Density-weighted shell gradient with cross-component CSE. */",
        "__device__ __noinline__ void generated_dppp_weighted_component_gradient(",
        "    const GeneratedDpppPackedForceGeometry& geometry,",
        "    const double (&component_weights)[kGeneratedDpppComponentCount],",
        "    double (&gradient)[3][3]) {",
        *emitter.lines,
    ]
    for center in range(len(selected_integral.independent_derivative_centers)):
        for coordinate in range(3):
            lines.append(
                f"  gradient[{center}][{coordinate}] = "
                f"{emitter.reference(roots[center * 3 + coordinate])};"
            )
    lines.append("}")
    return "\n".join(lines) + "\n\n"

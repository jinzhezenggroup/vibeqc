"""Assemble a complete fused shell kernel from schedule-specific consumers.

This module owns the common kernel template and dispatch, while Fock/force
arithmetic families live in their corresponding lowering modules. The large
remaining template preserves byte-identical generated CUDA and ABI layouts."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..capabilities import CAPABILITY_MIXED_FOCK
from ..cuda_schedule import (
    PairOrientation,
    PairStorage,
    ScheduleIR,
    ScheduleKind,
)
from ..fused_schedule import (
    FusedShellPlan,
    build_fused_shell_plan,
)
from ..ir import KernelConsumer
from ..shell_spec import (
    DPPP_SPEC,
    ShellClassSpec,
    cartesian_components,
)
from .algebra import (
    _emit_triple_pair_matchings,
    _emit_weighted_component_gradient_cuda,
    _packed_force_integral,
)
from .common import (
    _emitted_component_names,
    _format_cuda_array,
    _generic_component_gradient_setup,
    _generic_task_component_setup,
    _specialize_dppp_identifiers,
)
from .fock import _emit_shell_class_fock_cuda, _emit_shell_class_mixed_fock_cuda
from .force_packed import (
    _emit_packed_force_consumer_cuda,
    _emit_scalar_thread_force_consumer_cuda,
)
from .force_rys_component import _emit_rys_component_lane_force_consumer_cuda
from .force_rys_thread import _emit_rys_thread_force_consumer_cuda
from .force_rys_uniform import _emit_rys_uniform_warp_force_consumer_cuda
from .force_subgroup import _emit_subgroup_force_consumer_cuda
from .shared import _AXIS_INDEX


def emit_shell_class_fused_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan | None = None,
    *,
    fock_schedule: ScheduleIR | None = None,
    capabilities: Iterable[str] = (),
) -> str:
    """Emit a complete cooperative force kernel from a shell specification.

    The task queue is the outer symmetry/orbit boundary: every task is already
    canonicalized to the requested shell class and retains its slot-to-atom map.
    Consequently the primitive hot loop contains no shell-class, component,
    representative, or coordinate dispatch.
    """

    plan = build_fused_shell_plan(spec) if plan is None else plan
    selected_capabilities = frozenset(capabilities)
    if plan.spec != spec:
        raise ValueError("fused plan and shell specification do not match")
    if plan.schedule.kind not in (
        ScheduleKind.PACKED_TASKS,
        ScheduleKind.THREAD_TASKS,
        ScheduleKind.SUBGROUP_TASKS,
        ScheduleKind.SHELL_TASK,
        ScheduleKind.COMPONENT_LANES,
        ScheduleKind.TILED_COMPONENTS,
    ):
        raise ValueError(
            "current CUDA emitter implements packed, scalar thread-task, subgroup-task, shell-task, component-lane, and tiled schedules"
        )
    if plan.schedule.kind == ScheduleKind.PACKED_TASKS and plan.schedule.shared_coulomb:
        raise ValueError("packed tasks require lane-local Coulomb evaluation")
    if (
        plan.schedule.kind == ScheduleKind.TILED_COMPONENTS
        and plan.schedule.component_tile != plan.schedule.block_threads
    ):
        raise ValueError("tiled CUDA lowering requires one component per block thread")
    if any(order > 6 for order in spec.pair_orders):
        raise ValueError(
            "current fused CUDA candidate supports pair orders zero through six"
        )
    if any(order > 3 for order in spec.angular):
        raise ValueError("current fused CUDA candidate supports s/p/d/f shells")
    maximum_order = spec.maximum_force_coulomb_order
    force_integral = _packed_force_integral(spec, plan.kernel.integral)
    independent_centers = force_integral.independent_derivative_centers
    recovered_centers = force_integral.recovered_derivative_centers
    if len(independent_centers) != 3 or len(recovered_centers) != 1:
        raise ValueError(
            "generic fused force lowering currently requires three independent "
            "and one recovered derivative center"
        )
    decay_gradient_rows = 4 if 3 in force_integral.independent_derivative_centers else 3
    decay_fourth_assignment = (
        "    geometry.decay_gradients[3][axis] =\n"
        "        2.0 * second_pair.reduced_exponent *\n"
        "        (third_coordinate - fourth_coordinate);\n"
        if decay_gradient_rows == 4
        else ""
    )
    state_axis_bits = max(3, maximum_order.bit_length())
    state_mask = (1 << state_axis_bits) - 1
    packed_states = tuple(
        x_order | (y_order << state_axis_bits) | (z_order << (2 * state_axis_bits))
        for x_order, y_order, z_order in plan.coulomb_states
    )
    coulomb_index_type = "signed char" if len(plan.coulomb_states) <= 128 else "short"
    d_axes = tuple(
        _AXIS_INDEX[axis]
        for component in DPPP_SPEC.center_components[0]
        for axis in component
    )
    f_axes = tuple(
        _AXIS_INDEX[axis] for component in cartesian_components(3) for axis in component
    )
    f_axes_declaration = ""
    if any(order == 3 for order in spec.angular):
        f_axes_declaration = f"""
__device__ __constant__ unsigned char generated_dppp_f_axes[10][3] = {{
{_format_cuda_array(f_axes, columns=10)}
}};
"""
    first_pair_order, second_pair_order = spec.pair_orders
    explicit_pair_order = 0 in spec.pair_orders
    first_pair_term = (
        f"generated_dppp_pair_term<{first_pair_order}U>"
        if explicit_pair_order
        else "generated_dppp_pair_term"
    )
    second_pair_term = (
        f"generated_dppp_pair_term<{second_pair_order}U>"
        if explicit_pair_order
        else "generated_dppp_pair_term"
    )
    supported_pair_orders = " || ".join(
        f"PairOrder == {order}U" for order in sorted(set(spec.pair_orders))
    )
    if explicit_pair_order:
        pair_array_parameters = """    const unsigned* axes,
    const double* shifts,
    const double* shift_gradients,"""
        pair_matching_call = "generated_dppp_add_pair_matching<PairOrder>"
    else:
        pair_array_parameters = """    const unsigned (&axes)[PairOrder],
    const double (&shifts)[PairOrder],
    const double (&shift_gradients)[PairOrder],"""
        pair_matching_call = "generated_dppp_add_pair_matching"
    double_pair_matchings = ""
    if max(spec.pair_orders) >= 4:
        double_pair_matchings = """  if constexpr (PairOrder >= 4U) {
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
            VIBEQC_PAIR_MATCHING_CALL(
                term, axes, shifts, shift_gradients, inverse_two_exponent,
                subset, first_removed | second_removed, 2U);
          }
        }
      }
    }
  }
"""
        double_pair_matchings = double_pair_matchings.replace(
            "VIBEQC_PAIR_MATCHING_CALL", pair_matching_call
        )
    triple_pair_matchings = (
        _emit_triple_pair_matchings(pair_matching_call, gradients=True)
        if max(spec.pair_orders) >= 6
        else ""
    )
    component_gradient_setup = _generic_component_gradient_setup(spec)
    task_component_setup = _generic_task_component_setup(spec)
    component_names = _emitted_component_names(spec)
    # The dense state-index table belongs to the requested mathematical IR.
    # Value-only manifests prune the derivative layer even though the common
    # helper declarations retain force-sized scratch storage. Indexing that
    # smaller table with the force stride silently reads unrelated states.
    index_side = plan.kernel.integral.maximum_coulomb_order + 1
    side = maximum_order + 1
    minimum_blocks_per_sm = plan.schedule.minimum_blocks_per_sm or (
        2
        if plan.schedule.kind == ScheduleKind.PACKED_TASKS
        else (384 + plan.block_threads - 1) // plan.block_threads
    )
    shared_coulomb = "true" if plan.schedule.shared_coulomb else "false"
    coulomb_storage_count = (
        "kGeneratedDpppCoulombStateCount" if plan.schedule.shared_coulomb else "1"
    )
    coulomb_setup = ""
    if plan.schedule.shared_coulomb:
        if plan.schedule.kind == ScheduleKind.TILED_COMPONENTS:
            coulomb_setup = """          for (unsigned state = lane;
               state < kGeneratedDpppCoulombStateCount;
               state += kGeneratedDpppBlockThreads) {
            shared.coulomb[state] = generated_dppp_coulomb(
                generated_dppp_coulomb_states[state], shared.primitive);
          }
          __syncthreads();
"""
        else:
            coulomb_setup = """          for (unsigned state = lane;
               state < kGeneratedDpppCoulombStateCount;
               state += kGeneratedDpppBlockThreads) {
            shared.coulomb[state] = generated_dppp_coulomb(
                generated_dppp_coulomb_states[state], shared.primitive);
          }
          __syncthreads();
"""
    if plan.schedule.kind == ScheduleKind.TILED_COMPONENTS:
        component_schedule_setup = f"""  for (unsigned component_tile_begin = 0U;
       component_tile_begin < kGeneratedDpppComponentCount;
       component_tile_begin += {plan.schedule.component_tile}U) {{
  const unsigned tile_component = component_tile_begin + lane;
  const bool component_lane = tile_component < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? tile_component : 0U;
"""
        component_schedule_close = "  __syncthreads();\n  }\n"
    else:
        component_schedule_setup = """  const bool component_lane = lane < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? lane : 0U;
"""
        component_schedule_close = ""
    gradient_updates = []
    for slot, center in enumerate(independent_centers):
        if center == 0:
            expression = (
                "first_coefficient_gradient * state_value + "
                "geometry.product_scales[0] * scaled_derivative"
            )
        elif center == 1:
            expression = (
                "-first_coefficient_gradient * state_value + "
                "geometry.product_scales[1] * scaled_derivative"
            )
        elif center == 2:
            expression = (
                "second_coefficient_gradient * state_value - "
                "geometry.product_scales[2] * scaled_derivative"
            )
        elif center == 3:
            # The fourth ket center is independently differentiated only when
            # the explicit IR recovers a different center. Its product scale
            # is the complement of the stored third-center scale.
            expression = (
                "-second_coefficient_gradient * state_value - "
                "(1.0 - geometry.product_scales[2]) * scaled_derivative"
            )
        else:
            raise ValueError(f"unsupported derivative center {center}")
        gradient_updates.append(
            f"        value_gradient[{slot}][coordinate] += {expression};"
        )
    gradient_update_code = "\n".join(gradient_updates)
    pair_accumulation = f"""      const double sign =
          (generated_dppp_state_total(second_term.derivative_state) & 1U)
          == 0U ? 1.0 : -1.0;
      const unsigned state =
          first_term.derivative_state + second_term.derivative_state;
      const double state_value = generated_dppp_component_coulomb<SharedCoulomb>(
          geometry, coulomb, state);
      const double coefficient =
          sign * first_term.coefficient * second_term.coefficient;
      value += coefficient * state_value;
#pragma unroll
      for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
        const double scaled_derivative = coefficient *
            generated_dppp_component_coulomb<SharedCoulomb>(
                geometry, coulomb,
                state + (1U << ({state_axis_bits}U * coordinate)));
        const double first_coefficient_gradient =
            sign * first_term.first_center[coordinate] *
            second_term.coefficient;
        const double second_coefficient_gradient =
            sign * first_term.coefficient *
            second_term.first_center[coordinate];
{gradient_update_code}
      }}
"""
    if plan.schedule.pair_storage == PairStorage.RECOMPUTED:
        if plan.schedule.pair_orientation == PairOrientation.CANONICAL:
            gradient_contraction = f"""  double value = 0.0;
  double value_gradient[3][3]{{}};
VIBEQC_PAIR_UNROLL
  for (unsigned first_subset = 0;
       first_subset < {1 << first_pair_order}U; ++first_subset) {{
    const GeneratedDpppPairTerm first_term = {first_pair_term}(
        first_axes, first_shifts, first_shift_gradients,
        geometry.inverse_two_p, first_subset);
VIBEQC_PAIR_UNROLL
    for (unsigned second_subset = 0;
         second_subset < {1 << second_pair_order}U; ++second_subset) {{
      const GeneratedDpppPairTerm second_term = {second_pair_term}(
          second_axes, second_shifts, second_shift_gradients,
          geometry.inverse_two_q, second_subset);
{pair_accumulation}    }}
  }}
"""
        else:
            gradient_contraction = f"""  double value = 0.0;
  double value_gradient[3][3]{{}};
VIBEQC_PAIR_UNROLL
  for (unsigned second_subset = 0;
       second_subset < {1 << second_pair_order}U; ++second_subset) {{
    const GeneratedDpppPairTerm second_term = {second_pair_term}(
        second_axes, second_shifts, second_shift_gradients,
        geometry.inverse_two_q, second_subset);
VIBEQC_PAIR_UNROLL
    for (unsigned first_subset = 0;
         first_subset < {1 << first_pair_order}U; ++first_subset) {{
      const GeneratedDpppPairTerm first_term = {first_pair_term}(
          first_axes, first_shifts, first_shift_gradients,
          geometry.inverse_two_p, first_subset);
{pair_accumulation}    }}
  }}
"""
    elif plan.schedule.pair_orientation == PairOrientation.CANONICAL:
        gradient_contraction = f"""  GeneratedDpppPairTerm second_terms[{1 << second_pair_order}];
VIBEQC_PAIR_UNROLL
  for (unsigned subset = 0; subset < {1 << second_pair_order}U; ++subset) {{
    second_terms[subset] = {second_pair_term}(
        second_axes, second_shifts, second_shift_gradients,
        geometry.inverse_two_q, subset);
  }}
  double value = 0.0;
  double value_gradient[3][3]{{}};
VIBEQC_PAIR_UNROLL
  for (unsigned first_subset = 0;
       first_subset < {1 << first_pair_order}U; ++first_subset) {{
    const GeneratedDpppPairTerm first_term = {first_pair_term}(
        first_axes, first_shifts, first_shift_gradients,
        geometry.inverse_two_p, first_subset);
VIBEQC_PAIR_UNROLL
    for (unsigned second_subset = 0;
         second_subset < {1 << second_pair_order}U; ++second_subset) {{
      const GeneratedDpppPairTerm& second_term = second_terms[second_subset];
{pair_accumulation}    }}
  }}
"""
    else:
        gradient_contraction = f"""  GeneratedDpppPairTerm first_terms[{1 << first_pair_order}];
VIBEQC_PAIR_UNROLL
  for (unsigned subset = 0; subset < {1 << first_pair_order}U; ++subset) {{
    first_terms[subset] = {first_pair_term}(
        first_axes, first_shifts, first_shift_gradients,
        geometry.inverse_two_p, subset);
  }}
  double value = 0.0;
  double value_gradient[3][3]{{}};
VIBEQC_PAIR_UNROLL
  for (unsigned second_subset = 0;
       second_subset < {1 << second_pair_order}U; ++second_subset) {{
    const GeneratedDpppPairTerm second_term = {second_pair_term}(
        second_axes, second_shifts, second_shift_gradients,
        geometry.inverse_two_q, second_subset);
VIBEQC_PAIR_UNROLL
    for (unsigned first_subset = 0;
         first_subset < {1 << first_pair_order}U; ++first_subset) {{
      const GeneratedDpppPairTerm& first_term = first_terms[first_subset];
{pair_accumulation}    }}
  }}
"""
    if maximum_order <= 7:
        high_wick_multiplicity = """  return order * (order - 1U) * (order - 2U) * (order - 3U) *
      (order - 4U) * (order - 5U) / 48U;
"""
    else:
        high_wick_multiplicity = """  unsigned numerator = 1U;
  for (unsigned factor = 0U; factor < 2U * pairs; ++factor) {
    numerator *= order - factor;
  }
  unsigned denominator = 1U << pairs;
  for (unsigned factor = 2U; factor <= pairs; ++factor) {
    denominator *= factor;
  }
  return numerator / denominator;
"""
        if maximum_order >= 13:
            # FFFF forces reach order 13: the factorial ratio's numerator can
            # be 13! = 6,227,020,800 even though its final multiplicity fits in
            # unsigned. Keep lower-order code/cache identities unchanged.
            high_wick_multiplicity = (
                high_wick_multiplicity.replace(
                    "unsigned numerator", "std::uint64_t numerator"
                )
                .replace("unsigned denominator", "std::uint64_t denominator")
                .replace(
                    "return numerator / denominator;",
                    "return static_cast<unsigned>(numerator / denominator);",
                )
            )
    # These lowerings replace the generic cooperative force body below, so
    # emitting its warp-count constant would leave a misleading unused symbol.
    replaces_cooperative_force_body = (
        plan.schedule.kind == ScheduleKind.PACKED_TASKS
        or (
            plan.schedule.kind == ScheduleKind.SUBGROUP_TASKS
            and plan.kernel.integral.recurrence in ("rys3", "rys4", "rys5")
        )
    )
    warp_count_declaration = (
        ""
        if replaces_cooperative_force_body
        else f"constexpr unsigned kGeneratedDpppWarpCount = {plan.warp_count}U;\n"
    )
    component_gradient_output_lines = []
    for slot, center in enumerate(independent_centers):
        component_gradient_output_lines.extend(
            [
                f"  // Independent derivative slot {slot} maps to physical center {center}.",
                "#pragma unroll",
                "  for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {",
                f"    gradient[{center}][coordinate] = geometry.prefactor *",
                f"        (value_gradient[{slot}][coordinate] +",
                f"         value * geometry.decay_gradients[{center}][coordinate]);",
                "  }",
            ]
        )
    for center in recovered_centers:
        component_gradient_output_lines.extend(
            [
                "#pragma unroll",
                "  for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {",
                f"    gradient[{center}][coordinate] = -"
                + " - ".join(
                    f"gradient[{independent}][coordinate]"
                    for independent in independent_centers
                )
                + ";",
                "  }",
            ]
        )
    component_gradient_output = "\n".join(component_gradient_output_lines)
    force_slot_count = 3 * len(independent_centers)
    independent_center_table = ", ".join(f"{center}U" for center in independent_centers)
    independent_reduction_code = f"""  if (lane < {force_slot_count}U) {{
    double value = 0.0;
#pragma unroll
    for (unsigned source_warp = 0; source_warp < kGeneratedDpppWarpCount;
         ++source_warp) {{
      value += shared.warp_sums[source_warp][lane];
    }}
    // Keep the dense independent-slot totals available to recovery lanes.
    shared.warp_sums[0][lane] = value;
    if (value != 0.0) {{
      constexpr unsigned derivative_centers[3] = {{{independent_center_table}}};
      const unsigned center = derivative_centers[lane / 3U];
      const unsigned coordinate = lane % 3U;
      atomicAdd(forces + static_cast<std::size_t>(shared.task.atom[center]) * 3U +
                    coordinate,
                value);
    }}
  }}
  __syncthreads();
"""
    recovered_reduction_lines = []
    for recovered_index, center in enumerate(recovered_centers):
        name = "fourth_value" if recovered_index == 0 else f"recovered_value_{center}"
        terms = " - ".join(
            f"shared.warp_sums[0][{slot * 3}U + lane]"
            for slot in range(len(independent_centers))
        )
        recovered_reduction_lines.extend(
            [
                "  if (lane < 3U) {",
                f"    const double {name} = -{terms};",
                f"    if ({name} != 0.0) {{",
                "      atomicAdd(",
                f"          forces + static_cast<std::size_t>(shared.task.atom[{center}]) * 3U + lane,",
                f"          {name});",
                "    }",
                "  }",
            ]
        )
    recovered_reduction_code = "\n".join(recovered_reduction_lines)
    source = f"""/**
 * Generated cooperative AOT candidate for canonical (d p|p p) forces.
 *
 * Launch exactly {plan.block_threads} threads per canonical shell-quartet task.
 * The task builder performs shell-pair/quartet symmetry routing outside this
 * kernel and records the original atom for each canonical center slot.
 */
#include <cstddef>
#include <cstdint>

struct GeneratedDpppVec3 {{ double x; double y; double z; }};

/** Geometry and contraction data reused by every quartet with one shell pair. */
struct GeneratedDpppPrimitivePairData {{
  double exponent_sum;
  double reduced_exponent;
  GeneratedDpppVec3 product_center;
  double weighted_coefficient;
  double first_product_scale;
  double second_product_scale;
}};

/** Canonical task ABI kept independent of the production DeviceBatch layout. */
struct GeneratedDpppShellTask {{
  std::uint64_t primitive_begin[4];
  std::uint64_t primitive_end[4];
  std::uint64_t ao_begin[4];
  std::uint64_t ao_coefficient_begin[4];
  std::uint64_t density_offset;
  std::uint64_t spin_offset;
  std::uint32_t matrix_order;
  std::uint32_t shell_pair[2];
  std::uint32_t reversed_shell_pair_mask;
  std::uint32_t shell[4];
  std::uint32_t atom[4];
}};

struct GeneratedDpppPrimitiveGeometry {{
  double inverse_two_p;
  double inverse_two_q;
  double rho;
  double product_scales[3];
  double pair_shifts[4][3];
  double difference[3];
  double decay_gradients[{decay_gradient_rows}][3];
  double boys[{side}];
  double coordinate_powers[3][{side}];
  double negative_two_rho_powers[{side}];
  double prefactor;
  double primitive_coefficient;
}};

struct GeneratedDpppPairTerm {{
  unsigned derivative_state;
  double coefficient;
  double first_center[3];
}};

constexpr unsigned kGeneratedDpppComponentCount = {spec.component_count}U;
constexpr unsigned kGeneratedDpppBlockThreads = {plan.block_threads}U;
constexpr unsigned kGeneratedDpppCoulombStateCount = {len(plan.coulomb_states)}U;
{warp_count_declaration}

__device__ __constant__ unsigned short generated_dppp_coulomb_states[
    kGeneratedDpppCoulombStateCount] = {{
{_format_cuda_array(packed_states)}
}};

__device__ __constant__ {coulomb_index_type} generated_dppp_coulomb_indices[{index_side**3}] = {{
{_format_cuda_array(plan.coulomb_indices)}
}};

__device__ __constant__ unsigned char generated_dppp_d_axes[6][2] = {{
{_format_cuda_array(d_axes, columns=6)}
}};
{f_axes_declaration}

__device__ __forceinline__ double generated_dppp_axis(
    const GeneratedDpppVec3& value, unsigned axis) {{
  return axis == 0U ? value.x : (axis == 1U ? value.y : value.z);
}}

__device__ __forceinline__ unsigned generated_dppp_state_total(unsigned state) {{
  return (state & {state_mask}U) +
      ((state >> {state_axis_bits}U) & {state_mask}U) +
      ((state >> {2 * state_axis_bits}U) & {state_mask}U);
}}

__device__ __forceinline__ unsigned generated_dppp_state_index(unsigned state) {{
  const unsigned x_order = state & {state_mask}U;
  const unsigned y_order =
      (state >> {state_axis_bits}U) & {state_mask}U;
  const unsigned z_order =
      (state >> {2 * state_axis_bits}U) & {state_mask}U;
  return static_cast<unsigned>(generated_dppp_coulomb_indices[
      (x_order * {index_side}U + y_order) * {index_side}U + z_order]);
}}

__device__ __forceinline__ unsigned generated_dppp_wick_multiplicity(
    unsigned order, unsigned pairs) {{
  if (pairs == 0U) return 1U;
  if (pairs == 1U) return order * (order - 1U) / 2U;
  if (pairs == 2U) {{
    return order * (order - 1U) * (order - 2U) * (order - 3U) / 8U;
  }}
{high_wick_multiplicity.rstrip()}
}}

__device__ __forceinline__ double generated_dppp_coulomb(
    unsigned derivative_state,
    const GeneratedDpppPrimitiveGeometry& geometry) {{
  const unsigned x_order = derivative_state & {state_mask}U;
  const unsigned y_order =
      (derivative_state >> {state_axis_bits}U) & {state_mask}U;
  const unsigned z_order =
      (derivative_state >> {2 * state_axis_bits}U) & {state_mask}U;
  const unsigned total_order = x_order + y_order + z_order;
  double value = 0.0;
  for (unsigned x_pairs = 0; x_pairs <= x_order / 2U; ++x_pairs) {{
    for (unsigned y_pairs = 0; y_pairs <= y_order / 2U; ++y_pairs) {{
      for (unsigned z_pairs = 0; z_pairs <= z_order / 2U; ++z_pairs) {{
        const unsigned contraction_count = x_pairs + y_pairs + z_pairs;
        const unsigned boys_order = total_order - contraction_count;
        const unsigned multiplicity =
            generated_dppp_wick_multiplicity(x_order, x_pairs) *
            generated_dppp_wick_multiplicity(y_order, y_pairs) *
            generated_dppp_wick_multiplicity(z_order, z_pairs);
        value += static_cast<double>(multiplicity) *
            geometry.negative_two_rho_powers[boys_order] *
            geometry.coordinate_powers[0][x_order - 2U * x_pairs] *
            geometry.coordinate_powers[1][y_order - 2U * y_pairs] *
            geometry.coordinate_powers[2][z_order - 2U * z_pairs] *
            geometry.boys[boys_order];
      }}
    }}
  }}
  return value;
}}

template <unsigned PairOrder>
__device__ __forceinline__ void generated_dppp_add_pair_matching(
    GeneratedDpppPairTerm& term,
{pair_array_parameters}
    double inverse_two_exponent,
    unsigned subset,
    unsigned removed,
    unsigned contraction_count) {{
  if ((subset & removed) != 0U) return;
  double inverse_factor = 1.0;
  const unsigned inverse_count = contraction_count + __popc(subset);
  for (unsigned factor = 0; factor < inverse_count; ++factor) {{
    inverse_factor *= inverse_two_exponent;
  }}
  double coefficient = inverse_factor;
  for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {{
    if (((subset | removed) & (1U << quantum)) == 0U) {{
      coefficient *= shifts[quantum];
    }}
  }}
  term.coefficient += coefficient;
  for (unsigned differentiated = 0; differentiated < PairOrder;
       ++differentiated) {{
    if (((subset | removed) & (1U << differentiated)) != 0U) continue;
    double derivative = inverse_factor * shift_gradients[differentiated];
    for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {{
      if (quantum != differentiated &&
          ((subset | removed) & (1U << quantum)) == 0U) {{
        derivative *= shifts[quantum];
      }}
    }}
    term.first_center[axes[differentiated]] += derivative;
  }}
}}

template <unsigned PairOrder>
__device__ __forceinline__ GeneratedDpppPairTerm generated_dppp_pair_term(
{pair_array_parameters}
    double inverse_two_exponent,
    unsigned subset) {{
  static_assert({supported_pair_orders});
  GeneratedDpppPairTerm term{{}};
  for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {{
    if ((subset & (1U << quantum)) != 0U) {{
      term.derivative_state +=
          1U << ({state_axis_bits}U * axes[quantum]);
    }}
  }}
  {pair_matching_call}(
      term, axes, shifts, shift_gradients, inverse_two_exponent,
      subset, 0U, 0U);
  for (unsigned first = 0; first < PairOrder; ++first) {{
    for (unsigned second = first + 1U; second < PairOrder; ++second) {{
      if (axes[first] == axes[second]) {{
        {pair_matching_call}(
            term, axes, shifts, shift_gradients, inverse_two_exponent,
            subset, (1U << first) | (1U << second), 1U);
      }}
    }}
  }}
{double_pair_matchings}{triple_pair_matchings}  return term;
}}

template <bool SharedCoulomb>
__device__ __forceinline__ double generated_dppp_component_coulomb(
    const GeneratedDpppPrimitiveGeometry& geometry,
    const double* values,
    unsigned state) {{
  if constexpr (SharedCoulomb) {{
    return values[generated_dppp_state_index(state)];
  }}
  return generated_dppp_coulomb(state, geometry);
}}

/** Evaluate all centers and all xyz coordinates for one component lane. */
template <bool SharedCoulomb>
__device__ __forceinline__ void generated_dppp_component_gradient(
    unsigned component,
    const GeneratedDpppPrimitiveGeometry& geometry,
    const double* coulomb,
    double (&gradient)[4][3]) {{
{component_gradient_setup}

{gradient_contraction}
{component_gradient_output}
}}

__device__ __forceinline__ std::size_t generated_dppp_matrix_index(
    std::size_t row, std::size_t column, std::size_t order) {{
  return row + column * order;
}}

__device__ __forceinline__ void generated_dppp_eri_permutation(
    unsigned permutation,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    std::size_t& a, std::size_t& b, std::size_t& c, std::size_t& d) {{
  switch (permutation) {{
    case 0: a = i; b = j; c = k; d = l; break;
    case 1: a = j; b = i; c = k; d = l; break;
    case 2: a = i; b = j; c = l; d = k; break;
    case 3: a = j; b = i; c = l; d = k; break;
    case 4: a = k; b = l; c = i; d = j; break;
    case 5: a = l; b = k; c = i; d = j; break;
    case 6: a = k; b = l; c = j; d = i; break;
    default: a = l; b = k; c = j; d = i; break;
  }}
}}

__device__ __forceinline__ bool generated_dppp_unique_permutation(
    unsigned permutation,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    std::size_t a, std::size_t b, std::size_t c, std::size_t d) {{
  for (unsigned previous = 0; previous < permutation; ++previous) {{
    std::size_t pa = 0, pb = 0, pc = 0, pd = 0;
    generated_dppp_eri_permutation(
        previous, i, j, k, l, pa, pb, pc, pd);
    if (a == pa && b == pb && c == pc && d == pd) return false;
  }}
  return true;
}}

template <bool Unrestricted>
__device__ __forceinline__ double generated_dppp_density_coefficient(
    const GeneratedDpppShellTask& task,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    const double* density) {{
  const std::size_t n = static_cast<std::size_t>(task.matrix_order);
  const std::size_t matrix_size = n * n;
  const std::size_t ij = generated_dppp_matrix_index(i, j, n);
  const std::size_t kl = generated_dppp_matrix_index(k, l, n);
  const std::size_t ik = generated_dppp_matrix_index(i, k, n);
  const std::size_t jl = generated_dppp_matrix_index(j, l, n);
  const std::size_t il = generated_dppp_matrix_index(i, l, n);
  const std::size_t jk = generated_dppp_matrix_index(j, k, n);

  // SCF densities are symmetrized before entering direct J/K.  Under that
  // invariant the eight ERI permutations collapse exactly to two exchange
  // products and one Coulomb product.  Degenerate AO pairs reduce the orbit
  // by one half each; retaining these factors preserves the old unique-
  // permutation semantics without executing its nested comparison loop for
  // every Cartesian component.
  double orbit_scale = i == j ? 0.5 : 1.0;
  if (k == l) orbit_scale *= 0.5;
  if ((i == k && j == l) || (i == l && j == k)) orbit_scale *= 0.5;
  if constexpr (Unrestricted) {{
    const double alpha_ij = density[task.spin_offset + ij];
    const double alpha_kl = density[task.spin_offset + kl];
    const double beta_ij =
        density[task.spin_offset + matrix_size + ij];
    const double beta_kl =
        density[task.spin_offset + matrix_size + kl];
    const double coulomb =
        4.0 * (alpha_ij + beta_ij) * (alpha_kl + beta_kl);
    const double exchange = 2.0 * (
        density[task.spin_offset + ik] * density[task.spin_offset + jl] +
        density[task.spin_offset + il] * density[task.spin_offset + jk] +
        density[task.spin_offset + matrix_size + ik] *
            density[task.spin_offset + matrix_size + jl] +
        density[task.spin_offset + matrix_size + il] *
            density[task.spin_offset + matrix_size + jk]);
    return orbit_scale * (coulomb - exchange);
  }} else {{
    const std::size_t offset = task.density_offset;
    return orbit_scale * (
        4.0 * density[offset + ij] * density[offset + kl] -
        density[offset + ik] * density[offset + jl] -
        density[offset + il] * density[offset + jk]);
  }}
}}

/** Combine two reusable shell-pair records into one primitive quartet. */
__device__ __forceinline__ void generated_dppp_make_primitive_geometry(
    const GeneratedDpppPrimitivePairData& first_pair,
    const GeneratedDpppPrimitivePairData& second_pair,
    bool first_pair_reversed,
    bool second_pair_reversed,
    const GeneratedDpppVec3& first,
    const GeneratedDpppVec3& second,
    const GeneratedDpppVec3& third,
    const GeneratedDpppVec3& fourth,
    GeneratedDpppPrimitiveGeometry& geometry) {{
  const double p = first_pair.exponent_sum;
  const double q = second_pair.exponent_sum;
  geometry.rho = p * q / (p + q);
  geometry.inverse_two_p = 0.5 / p;
  geometry.inverse_two_q = 0.5 / q;
  geometry.product_scales[0] = first_pair_reversed
      ? first_pair.second_product_scale : first_pair.first_product_scale;
  geometry.product_scales[1] = first_pair_reversed
      ? first_pair.first_product_scale : first_pair.second_product_scale;
  geometry.product_scales[2] = second_pair_reversed
      ? second_pair.second_product_scale : second_pair.first_product_scale;
  double argument_squared_distance = 0.0;
#pragma unroll
  for (unsigned axis = 0; axis < 3U; ++axis) {{
    const double first_coordinate = generated_dppp_axis(first, axis);
    const double second_coordinate = generated_dppp_axis(second, axis);
    const double third_coordinate = generated_dppp_axis(third, axis);
    const double fourth_coordinate = generated_dppp_axis(fourth, axis);
    const double product_p =
        generated_dppp_axis(first_pair.product_center, axis);
    const double product_q =
        generated_dppp_axis(second_pair.product_center, axis);
    geometry.pair_shifts[0][axis] = product_p - first_coordinate;
    geometry.pair_shifts[1][axis] = product_p - second_coordinate;
    geometry.pair_shifts[2][axis] = product_q - third_coordinate;
    geometry.pair_shifts[3][axis] = product_q - fourth_coordinate;
    geometry.difference[axis] = product_p - product_q;
    geometry.decay_gradients[0][axis] =
        -2.0 * first_pair.reduced_exponent *
        (first_coordinate - second_coordinate);
    geometry.decay_gradients[1][axis] =
        2.0 * first_pair.reduced_exponent *
        (first_coordinate - second_coordinate);
    geometry.decay_gradients[2][axis] =
        -2.0 * second_pair.reduced_exponent *
        (third_coordinate - fourth_coordinate);
{decay_fourth_assignment}
    argument_squared_distance +=
        geometry.difference[axis] * geometry.difference[axis];
    geometry.coordinate_powers[axis][0] = 1.0;
#pragma unroll
    for (unsigned power = 1; power <= {maximum_order}U; ++power) {{
      geometry.coordinate_powers[axis][power] =
          geometry.coordinate_powers[axis][power - 1U] *
          geometry.difference[axis];
    }}
  }}
  boys_values<{maximum_order}>(
      geometry.rho * argument_squared_distance, geometry.boys);
  geometry.negative_two_rho_powers[0] = 1.0;
#pragma unroll
  for (unsigned power = 1; power <= {maximum_order}U; ++power) {{
    geometry.negative_two_rho_powers[power] =
        geometry.negative_two_rho_powers[power - 1U] *
        (-2.0 * geometry.rho);
  }}
  geometry.prefactor =
      34.986836655249725 / (p * q * sqrt(p + q));
  geometry.primitive_coefficient =
      first_pair.weighted_coefficient * second_pair.weighted_coefficient;
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_task(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t task_index) {{
  struct Shared {{
    GeneratedDpppShellTask task;
    GeneratedDpppVec3 positions[4];
    GeneratedDpppPrimitiveGeometry primitive;
    double coulomb[{coulomb_storage_count}];
    // Accumulate only the dense independent derivative slots.  Any declared
    // recovered center is assembled from their translational sum after the
    // block reduction, avoiding extra long-lived FP64 accumulators per lane.
    double warp_sums[kGeneratedDpppWarpCount][{force_slot_count}];
  }};
  __shared__ Shared shared;
  const unsigned lane = threadIdx.x;
  if (blockDim.x != kGeneratedDpppBlockThreads) return;
  if (lane == 0U) {{
    shared.task = tasks[task_index];
#pragma unroll
    for (unsigned center = 0; center < 4U; ++center) {{
      shared.positions[center] = atom_positions[shared.task.atom[center]];
    }}
  }}
  __syncthreads();

{component_schedule_setup}{task_component_setup}
  const std::size_t matrix_order =
      static_cast<std::size_t>(shared.task.matrix_order);
  const double schwarz_product = schwarz_bounds == nullptr
      ? 0.0
      : schwarz_bounds[
          shared.task.density_offset +
          generated_dppp_matrix_index(i, j, matrix_order)] *
        schwarz_bounds[
              shared.task.density_offset +
              generated_dppp_matrix_index(k, l, matrix_order)];
  const bool retained_by_schwarz = schwarz_bounds == nullptr ||
      schwarz_product >= screening_tolerance;
  // Match the Fock consumer's Schwarz-only selection so analytic forces stay
  // variational at both production and deliberately loose test tolerances.
  const double density_coefficient =
      component_lane && unique_ket_component && retained_by_schwarz
      ? generated_dppp_density_coefficient<Unrestricted>(
            shared.task, i, j, k, l, density)
      : 0.0;
  const double angular_coefficient = component_lane
      ? ao_coefficients[shared.task.ao_coefficient_begin[0] + {component_names[0]}] *
        ao_coefficients[shared.task.ao_coefficient_begin[1] + {component_names[1]}] *
        ao_coefficients[shared.task.ao_coefficient_begin[2] + {component_names[2]}] *
        ao_coefficients[shared.task.ao_coefficient_begin[3] + {component_names[3]}]
      : 0.0;
  if (!__syncthreads_or(density_coefficient != 0.0)) return;
  double component_force[{force_slot_count}]{{}};

  const std::int64_t first_pair_begin =
      primitive_pair_offsets[shared.task.shell_pair[0]];
  const std::int64_t first_pair_end =
      primitive_pair_offsets[shared.task.shell_pair[0] + 1U];
  const std::int64_t second_pair_begin =
      primitive_pair_offsets[shared.task.shell_pair[1]];
  const std::int64_t second_pair_end =
      primitive_pair_offsets[shared.task.shell_pair[1] + 1U];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end; ++first_primitive) {{
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end; ++second_primitive) {{
      if (lane == 0U) {{
        generated_dppp_make_primitive_geometry(
            primitive_pairs[first_primitive],
            primitive_pairs[second_primitive],
            (shared.task.reversed_shell_pair_mask & 1U) != 0U,
            (shared.task.reversed_shell_pair_mask & 2U) != 0U,
            shared.positions[0], shared.positions[1],
            shared.positions[2], shared.positions[3], shared.primitive);
      }}
      __syncthreads();
{coulomb_setup}
      if (density_coefficient != 0.0) {{
        double primitive_gradient[4][3];
        generated_dppp_component_gradient<{shared_coulomb}>(
            component, shared.primitive, shared.coulomb,
            primitive_gradient);
        const double scale = -density_coefficient * angular_coefficient *
            shared.primitive.primitive_coefficient;
        constexpr unsigned derivative_centers[3] = {{{independent_center_table}}};
#pragma unroll
        for (unsigned slot = 0; slot < {len(independent_centers)}U; ++slot) {{
#pragma unroll
          for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
            const unsigned center = derivative_centers[slot];
            component_force[slot * 3U + coordinate] +=
                scale * primitive_gradient[center][coordinate];
          }}
        }}
      }}
      __syncthreads();
    }}
  }}

  const unsigned warp = lane / 32U;
  const unsigned warp_lane = lane % 32U;
#pragma unroll
  for (unsigned slot = 0; slot < {force_slot_count}U; ++slot) {{
    double value = component_force[slot];
#pragma unroll
    for (unsigned offset = 16U; offset != 0U; offset /= 2U) {{
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }}
    if (warp_lane == 0U) shared.warp_sums[warp][slot] = value;
  }}
  __syncthreads();
{independent_reduction_code}{recovered_reduction_code}
{component_schedule_close}}}

extern "C" __global__ __launch_bounds__(kGeneratedDpppBlockThreads, {minimum_blocks_per_sm})
void generated_dppp_shell_class_force_rhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t task_count) {{
  if (blockIdx.x >= task_count) return;
  generated_dppp_shell_class_force_task<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      static_cast<std::size_t>(blockIdx.x));
}}

extern "C" __global__ __launch_bounds__(kGeneratedDpppBlockThreads, {minimum_blocks_per_sm})
void generated_dppp_shell_class_force_uhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t task_count) {{
  if (blockIdx.x >= task_count) return;
  generated_dppp_shell_class_force_task<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      static_cast<std::size_t>(blockIdx.x));
}}

/** Persistent workers avoid launching one block per topology-capacity slot. */
template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_persistent(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  const unsigned lane = threadIdx.x;
  __shared__ std::uint32_t shared_task_index;
  while (true) {{
    if (lane == 0U) shared_task_index = atomicAdd(task_head, 1U);
    __syncthreads();
    const std::uint32_t task_index = shared_task_index;
    if (task_index >= *task_count) return;
    generated_dppp_shell_class_force_task<Unrestricted>(
        tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
        atom_positions, screening_tolerance, schwarz_bounds, density, forces,
        static_cast<std::size_t>(*task_offset + task_index));
    __syncthreads();
  }}
}}

extern "C" __global__ __launch_bounds__(kGeneratedDpppBlockThreads, {minimum_blocks_per_sm})
void generated_dppp_shell_class_force_rhf_persistent_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  generated_dppp_shell_class_force_persistent<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}}

extern "C" __global__ __launch_bounds__(kGeneratedDpppBlockThreads, {minimum_blocks_per_sm})
void generated_dppp_shell_class_force_uhf_persistent_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  generated_dppp_shell_class_force_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}}
"""
    if (
        plan.schedule.kind == ScheduleKind.COMPONENT_LANES
        and plan.kernel.integral.recurrence in ("rys3", "rys4", "rys5")
    ):
        force_marker = """template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_task("""
        force_begin = source.find(force_marker)
        if force_begin < 0:
            raise RuntimeError("generated force task marker changed unexpectedly")
        source = source[:force_begin] + _emit_rys_component_lane_force_consumer_cuda(
            spec,
            plan,
            minimum_blocks_per_sm,
        )
    elif plan.schedule.kind == ScheduleKind.PACKED_TASKS:
        force_marker = """template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_task("""
        force_begin = source.find(force_marker)
        if force_begin < 0:
            raise RuntimeError("generated force task marker changed unexpectedly")
        source = (
            source[:force_begin]
            + _emit_weighted_component_gradient_cuda(
                spec,
                plan.schedule,
                integral=plan.kernel.integral,
            )
            + _emit_packed_force_consumer_cuda(
                spec,
                plan,
                minimum_blocks_per_sm,
            )
        )
    elif plan.schedule.kind == ScheduleKind.THREAD_TASKS:
        force_marker = """template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_task("""
        force_begin = source.find(force_marker)
        if force_begin < 0:
            raise RuntimeError("generated force task marker changed unexpectedly")
        if plan.kernel.integral.recurrence in ("rys2", "rys3"):
            force_consumer = _emit_rys_thread_force_consumer_cuda(
                spec,
                plan,
                minimum_blocks_per_sm,
            )
        elif plan.kernel.integral.recurrence in ("rys4", "rys5"):
            raise ValueError(
                "thread-task high-root Rys lowering is unsupported; use "
                "cooperative component lanes"
            )
        else:
            force_consumer = _emit_scalar_thread_force_consumer_cuda(
                spec,
                plan,
                minimum_blocks_per_sm,
            )
        source = source[:force_begin] + force_consumer
    elif plan.schedule.kind == ScheduleKind.SUBGROUP_TASKS:
        force_marker = """template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_task("""
        force_begin = source.find(force_marker)
        if force_begin < 0:
            raise RuntimeError("generated force task marker changed unexpectedly")
        if plan.kernel.integral.recurrence in ("rys3", "rys4", "rys5"):
            force_consumer = _emit_rys_uniform_warp_force_consumer_cuda(
                spec,
                plan,
                minimum_blocks_per_sm,
            )
        else:
            force_consumer = _emit_subgroup_force_consumer_cuda(
                spec,
                plan,
                minimum_blocks_per_sm,
            )
        source = source[:force_begin] + force_consumer
    if KernelConsumer.FOCK in plan.kernel.integral.consumers:
        fock_plan = plan
        if fock_schedule is not None:
            # Force and Fock need not share an execution geometry. In
            # particular, high-component Rys4 force kernels can require a
            # cooperative mapping while the accepted value path remains a
            # compact tiled worker. Keep its subset/Wick recurrence explicit.
            fock_plan = build_fused_shell_plan(
                spec,
                consumers=(KernelConsumer.FOCK,),
                schedule=fock_schedule,
                recurrence="subset_wick",
                target=plan.kernel.target,
            )
        elif plan.schedule.kind == ScheduleKind.SUBGROUP_TASKS and (
            plan.kernel.integral.recurrence in ("rys3", "rys4", "rys5")
        ):
            # Uniform warps are a force-only architecture experiment.  Keep
            # the accepted value path on its original component-lane mapping
            # so an endpoint result isolates the force architecture.
            value_state_count = math.comb(
                plan.kernel.integral.value_coulomb_order + 3, 3
            )
            fock_block_threads = (
                (max(spec.component_count, value_state_count) + 31) // 32 * 32
            )
            fock_schedule = ScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=fock_block_threads,
                component_tile=spec.component_count,
                tasks_per_warp=1,
                shared_coulomb=True,
                pair_orientation=plan.schedule.pair_orientation,
                pair_storage=plan.schedule.pair_storage,
                unroll_pair_terms=plan.schedule.unroll_pair_terms,
                minimum_blocks_per_sm=(
                    2 if plan.kernel.integral.recurrence in ("rys4", "rys5") else 0
                ),
                warp_size=plan.schedule.warp_size,
            )
            fock_plan = build_fused_shell_plan(
                spec,
                consumers=tuple(plan.kernel.integral.consumers),
                schedule=fock_schedule,
                recurrence=plan.kernel.integral.recurrence,
                target=plan.kernel.target,
            )
        elif (
            plan.kernel.integral.recurrence in ("rys2", "rys3")
            and plan.schedule.kind == ScheduleKind.THREAD_TASKS
        ):
            # Scalar Rys3 is a force-only architecture experiment.  Retain the
            # accepted component-lane value recurrence so the real endpoint
            # isolates force performance and does not silently retune Fock.
            value_state_count = math.comb(
                plan.kernel.integral.value_coulomb_order + 3, 3
            )
            fock_block_threads = (
                (max(spec.component_count, value_state_count) + 31) // 32 * 32
            )
            fock_schedule = ScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=fock_block_threads,
                component_tile=spec.component_count,
                tasks_per_warp=1,
                shared_coulomb=True,
                pair_orientation=plan.schedule.pair_orientation,
                pair_storage=plan.schedule.pair_storage,
                unroll_pair_terms=plan.schedule.unroll_pair_terms,
                warp_size=plan.schedule.warp_size,
            )
            fock_plan = build_fused_shell_plan(
                spec,
                consumers=tuple(plan.kernel.integral.consumers),
                schedule=fock_schedule,
                recurrence=plan.kernel.integral.recurrence,
                target=plan.kernel.target,
            )
        source += _emit_shell_class_fock_cuda(
            spec,
            fock_plan,
            honor_schedule_block_threads=fock_schedule is not None,
        )
        if CAPABILITY_MIXED_FOCK in selected_capabilities:
            source += _emit_shell_class_mixed_fock_cuda(spec, fock_plan)
    pair_unroll = (
        "#pragma unroll" if plan.schedule.unroll_pair_terms else "#pragma unroll 1"
    )
    source = source.replace("VIBEQC_PAIR_UNROLL", pair_unroll)
    return _specialize_dppp_identifiers(source, spec)

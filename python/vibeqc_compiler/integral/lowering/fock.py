"""Assemble ordinary and mixed-precision Fock consumer kernels.

Spin coefficients and schedule choices retain their existing semantics; tiled
and component-lane implementations are delegated to their dedicated modules."""

from __future__ import annotations

import re

from ..cuda_schedule import (
    PairOrientation,
    PairStorage,
    ScheduleKind,
)
from ..fused_schedule import (
    FusedShellPlan,
)
from ..shell_spec import (
    ShellClassSpec,
)
from .algebra import _emit_triple_pair_matchings
from .common import (
    _emitted_component_names,
    _generic_component_value_setup,
    _generic_task_component_setup,
)
from .fock_component import _emit_rys_component_lane_fock_consumer_cuda
from .fock_tiled import (
    _emit_packed_fock_consumer_cuda,
    _emit_subgroup_fock_consumer_cuda,
)
from .selection import _supports_rys_component_lane_fock


def _emit_shell_class_fock_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    *,
    honor_schedule_block_threads: bool = False,
) -> str:
    """Emit coefficient-only Fock workers beside an accepted force kernel.

    Fock construction reuses primitive geometry and the Cartesian Coulomb
    table, but deliberately omits shift gradients and the raised derivative
    order required only by analytic forces.  Implicit Rys fallbacks retain the
    historical minimum CTA width; an explicit manifest Fock schedule can opt
    into its tuned width so the ordinary and streaming wrappers agree.
    """

    first_pair_order, second_pair_order = spec.pair_orders
    supported_pair_orders = " || ".join(
        f"PairOrder == {order}U" for order in sorted(set(spec.pair_orders))
    )
    maximum_order = sum(spec.angular)
    # The value consumer reuses the force emitter's packed-state decoder.
    # Keep the per-axis radix identical even though Fock itself needs one
    # lower Coulomb order; otherwise order-eight classes such as dddd encode
    # x/y/z digits with three bits and index outside the shared state table.
    state_axis_bits = max(3, spec.maximum_force_coulomb_order.bit_length())
    coulomb_state_count = (
        (maximum_order + 1) * (maximum_order + 2) * (maximum_order + 3) // 6
    )
    if (
        plan.schedule.kind == ScheduleKind.COMPONENT_LANES
        and not honor_schedule_block_threads
    ):
        # The implicit component-lane fallback uses the smallest block that
        # covers all component and Coulomb states.  Preserve this established
        # code shape for callers that did not provide a separate Fock policy.
        block_threads = (
            (max(spec.component_count, coulomb_state_count) + 31) // 32
        ) * 32
    else:
        # An explicitly tuned Fock mapping owns its CTA width.  Schedule
        # validation guarantees a warp-aligned, target-legal value.
        block_threads = plan.schedule.block_threads
    minimum_blocks_per_sm = plan.schedule.minimum_blocks_per_sm or (
        (384 + block_threads - 1) // block_threads
    )
    barrier = "__syncwarp();" if block_threads == 32 else "__syncthreads();"
    component_setup = _generic_component_value_setup(spec)
    task_component_setup = _generic_task_component_setup(spec)
    component_names = _emitted_component_names(spec)
    shared_coulomb = "true" if plan.schedule.shared_coulomb else "false"
    coulomb_storage_count = (
        "kGeneratedDpppFockCoulombStateCount" if plan.schedule.shared_coulomb else "1"
    )
    coulomb_setup = ""
    if plan.schedule.shared_coulomb:
        if plan.schedule.kind == ScheduleKind.TILED_COMPONENTS:
            coulomb_setup = """          for (unsigned state = lane;
               state < kGeneratedDpppFockCoulombStateCount;
               state += kGeneratedDpppFockBlockThreads) {
            shared.coulomb[state] = generated_dppp_coulomb(
                generated_dppp_coulomb_states[state], shared.primitive);
          }
          __syncthreads();
"""
        else:
            coulomb_setup = (
                "          for (unsigned state = lane;\n"
                "               state < kGeneratedDpppFockCoulombStateCount;\n"
                "               state += kGeneratedDpppFockBlockThreads) {\n"
                "            shared.coulomb[state] = generated_dppp_coulomb(\n"
                "                generated_dppp_coulomb_states[state], "
                "shared.primitive);\n"
                "          }\n"
                f"          {barrier}\n"
            )
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
        component_schedule_setup = """  const bool component_lane =
      lane < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? lane : 0U;
"""
        component_schedule_close = ""
    if plan.schedule.pair_storage == PairStorage.RECOMPUTED:
        if plan.schedule.pair_orientation == PairOrientation.CANONICAL:
            value_contraction = f"""  double value = 0.0;
VIBEQC_PAIR_UNROLL
  for (unsigned first_subset = 0;
       first_subset < {1 << first_pair_order}U; ++first_subset) {{
    const GeneratedDpppValueTerm first_term =
        generated_dppp_pair_value_term<{first_pair_order}U>(
        first_axes, first_shifts, geometry.inverse_two_p, first_subset);
VIBEQC_PAIR_UNROLL
    for (unsigned second_subset = 0;
         second_subset < {1 << second_pair_order}U; ++second_subset) {{
      const GeneratedDpppValueTerm second_term =
          generated_dppp_pair_value_term<{second_pair_order}U>(
          second_axes, second_shifts, geometry.inverse_two_q, second_subset);
      const double sign =
          (generated_dppp_state_total(second_term.derivative_state) & 1U)
          == 0U ? 1.0 : -1.0;
      const unsigned state =
          first_term.derivative_state + second_term.derivative_state;
      value += sign * first_term.coefficient * second_term.coefficient *
          generated_dppp_component_coulomb<SharedCoulomb>(
              geometry, coulomb, state);
    }}
  }}
"""
        else:
            value_contraction = f"""  double value = 0.0;
VIBEQC_PAIR_UNROLL
  for (unsigned second_subset = 0;
       second_subset < {1 << second_pair_order}U; ++second_subset) {{
    const GeneratedDpppValueTerm second_term =
        generated_dppp_pair_value_term<{second_pair_order}U>(
        second_axes, second_shifts, geometry.inverse_two_q, second_subset);
    const double sign =
        (generated_dppp_state_total(second_term.derivative_state) & 1U)
        == 0U ? 1.0 : -1.0;
VIBEQC_PAIR_UNROLL
    for (unsigned first_subset = 0;
         first_subset < {1 << first_pair_order}U; ++first_subset) {{
      const GeneratedDpppValueTerm first_term =
          generated_dppp_pair_value_term<{first_pair_order}U>(
          first_axes, first_shifts, geometry.inverse_two_p, first_subset);
      const unsigned state =
          first_term.derivative_state + second_term.derivative_state;
      value += sign * first_term.coefficient * second_term.coefficient *
          generated_dppp_component_coulomb<SharedCoulomb>(
              geometry, coulomb, state);
    }}
  }}
"""
    elif plan.schedule.pair_orientation == PairOrientation.CANONICAL:
        value_contraction = f"""  GeneratedDpppValueTerm second_terms[{1 << second_pair_order}];
VIBEQC_PAIR_UNROLL
  for (unsigned subset = 0; subset < {1 << second_pair_order}U; ++subset) {{
    second_terms[subset] = generated_dppp_pair_value_term<{second_pair_order}U>(
        second_axes, second_shifts, geometry.inverse_two_q, subset);
  }}
  double value = 0.0;
VIBEQC_PAIR_UNROLL
  for (unsigned first_subset = 0;
       first_subset < {1 << first_pair_order}U; ++first_subset) {{
    const GeneratedDpppValueTerm first_term =
        generated_dppp_pair_value_term<{first_pair_order}U>(
        first_axes, first_shifts, geometry.inverse_two_p, first_subset);
VIBEQC_PAIR_UNROLL
    for (unsigned second_subset = 0;
         second_subset < {1 << second_pair_order}U; ++second_subset) {{
      const GeneratedDpppValueTerm& second_term = second_terms[second_subset];
      const double sign =
          (generated_dppp_state_total(second_term.derivative_state) & 1U)
          == 0U ? 1.0 : -1.0;
      const unsigned state =
          first_term.derivative_state + second_term.derivative_state;
      value += sign * first_term.coefficient * second_term.coefficient *
          generated_dppp_component_coulomb<SharedCoulomb>(
              geometry, coulomb, state);
    }}
  }}
"""
    else:
        value_contraction = f"""  GeneratedDpppValueTerm first_terms[{1 << first_pair_order}];
VIBEQC_PAIR_UNROLL
  for (unsigned subset = 0; subset < {1 << first_pair_order}U; ++subset) {{
    first_terms[subset] = generated_dppp_pair_value_term<{first_pair_order}U>(
        first_axes, first_shifts, geometry.inverse_two_p, subset);
  }}
  double value = 0.0;
VIBEQC_PAIR_UNROLL
  for (unsigned second_subset = 0;
       second_subset < {1 << second_pair_order}U; ++second_subset) {{
    const GeneratedDpppValueTerm second_term =
        generated_dppp_pair_value_term<{second_pair_order}U>(
        second_axes, second_shifts, geometry.inverse_two_q, second_subset);
    const double sign =
        (generated_dppp_state_total(second_term.derivative_state) & 1U)
        == 0U ? 1.0 : -1.0;
VIBEQC_PAIR_UNROLL
    for (unsigned first_subset = 0;
         first_subset < {1 << first_pair_order}U; ++first_subset) {{
      const GeneratedDpppValueTerm& first_term = first_terms[first_subset];
      const unsigned state =
          first_term.derivative_state + second_term.derivative_state;
      value += sign * first_term.coefficient * second_term.coefficient *
          generated_dppp_component_coulomb<SharedCoulomb>(
              geometry, coulomb, state);
    }}
  }}
"""
    double_pair_matchings = ""
    if max(spec.pair_orders) >= 4:
        double_pair_matchings = """  if constexpr (PairOrder >= 4U) {
    for (unsigned first = 0; first < PairOrder; ++first) {
      for (unsigned second = first + 1U; second < PairOrder; ++second) {
        if (axes[first] != axes[second]) continue;
        const unsigned first_removed = (1U << first) | (1U << second);
        for (unsigned third = 0; third < PairOrder; ++third) {
          for (unsigned fourth = third + 1U; fourth < PairOrder; ++fourth) {
            if (axes[third] != axes[fourth]) continue;
            const unsigned second_removed =
                (1U << third) | (1U << fourth);
            if (first_removed >= second_removed ||
                (first_removed & second_removed) != 0U) continue;
            generated_dppp_add_value_matching<PairOrder>(
                term, axes, shifts, inverse_two_exponent, subset,
                first_removed | second_removed, 2U);
          }
        }
      }
    }
  }
"""
    triple_pair_matchings = (
        _emit_triple_pair_matchings(
            "generated_dppp_add_value_matching<PairOrder>", gradients=False
        )
        if max(spec.pair_orders) >= 6
        else ""
    )
    source = f"""

/** Coefficient-only pair term used by the SCF Fock recurrence. */
struct GeneratedDpppValueTerm {{
  unsigned derivative_state;
  double coefficient;
}};

constexpr unsigned kGeneratedDpppFockCoulombStateCount =
    {coulomb_state_count}U;
constexpr unsigned kGeneratedDpppFockBlockThreads = {block_threads}U;

template <unsigned PairOrder>
__device__ __forceinline__ void generated_dppp_add_value_matching(
    GeneratedDpppValueTerm& term,
    const unsigned* axes,
    const double* shifts,
    double inverse_two_exponent,
    unsigned subset,
    unsigned removed,
    unsigned contraction_count) {{
  if ((subset & removed) != 0U) return;
  double coefficient = 1.0;
  const unsigned inverse_count = contraction_count + __popc(subset);
  for (unsigned factor = 0; factor < inverse_count; ++factor) {{
    coefficient *= inverse_two_exponent;
  }}
  for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {{
    if (((subset | removed) & (1U << quantum)) == 0U) {{
      coefficient *= shifts[quantum];
    }}
  }}
  term.coefficient += coefficient;
}}

template <unsigned PairOrder>
__device__ __forceinline__ GeneratedDpppValueTerm
generated_dppp_pair_value_term(
    const unsigned* axes,
    const double* shifts,
    double inverse_two_exponent,
    unsigned subset) {{
  static_assert({supported_pair_orders});
  GeneratedDpppValueTerm term{{}};
  for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {{
    if ((subset & (1U << quantum)) != 0U) {{
      term.derivative_state +=
          1U << ({state_axis_bits}U * axes[quantum]);
    }}
  }}
  generated_dppp_add_value_matching<PairOrder>(
      term, axes, shifts, inverse_two_exponent, subset, 0U, 0U);
  for (unsigned first = 0; first < PairOrder; ++first) {{
    for (unsigned second = first + 1U; second < PairOrder; ++second) {{
      if (axes[first] == axes[second]) {{
        generated_dppp_add_value_matching<PairOrder>(
            term, axes, shifts, inverse_two_exponent, subset,
            (1U << first) | (1U << second), 1U);
      }}
    }}
  }}
{double_pair_matchings}{triple_pair_matchings}  return term;
}}

/** Evaluate one AO component without constructing force-only derivatives. */
template <bool SharedCoulomb>
__device__ __forceinline__ double generated_dppp_component_value(
    unsigned component,
    const GeneratedDpppPrimitiveGeometry& geometry,
    const double* coulomb) {{
{component_setup}

{value_contraction}
  return geometry.prefactor * value;
}}

/** Scatter one canonical integral using VIBEQC's existing RHF/UHF convention. */
template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_accumulate_fock(
    const GeneratedDpppShellTask& task,
    const double* density,
    double* fock,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    double integral) {{
  const std::size_t n = static_cast<std::size_t>(task.matrix_order);
  const std::size_t matrix_size = n * n;
#pragma unroll
  for (unsigned permutation = 0; permutation < 8U; ++permutation) {{
    std::size_t a = 0, b = 0, c = 0, d = 0;
    generated_dppp_eri_permutation(
        permutation, i, j, k, l, a, b, c, d);
    if (!generated_dppp_unique_permutation(
            permutation, i, j, k, l, a, b, c, d)) continue;
    const std::size_t ab = generated_dppp_matrix_index(a, b, n);
    const std::size_t ac = generated_dppp_matrix_index(a, c, n);
    const std::size_t cd = generated_dppp_matrix_index(c, d, n);
    const std::size_t bd = generated_dppp_matrix_index(b, d, n);
    if constexpr (Unrestricted) {{
      const double alpha_cd = density[task.spin_offset + cd];
      const double beta_cd = density[task.spin_offset + matrix_size + cd];
      const double total_cd = alpha_cd + beta_cd;
      if (total_cd != 0.0) {{
        atomicAdd(fock + task.spin_offset + ab, total_cd * integral);
        atomicAdd(
            fock + task.spin_offset + matrix_size + ab,
            total_cd * integral);
      }}
      const double alpha_bd = density[task.spin_offset + bd];
      const double beta_bd = density[task.spin_offset + matrix_size + bd];
      if (alpha_bd != 0.0) {{
        atomicAdd(fock + task.spin_offset + ac, -alpha_bd * integral);
      }}
      if (beta_bd != 0.0) {{
        atomicAdd(
            fock + task.spin_offset + matrix_size + ac,
            -beta_bd * integral);
      }}
    }} else {{
      const double density_cd = density[task.density_offset + cd];
      const double density_bd = density[task.density_offset + bd];
      if (density_cd != 0.0) {{
        atomicAdd(fock + task.density_offset + ab, density_cd * integral);
      }}
      if (density_bd != 0.0) {{
        atomicAdd(
            fock + task.density_offset + ac,
            -0.5 * density_bd * integral);
      }}
    }}
  }}
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_fock_task(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    std::size_t task_index) {{
  struct Shared {{
    GeneratedDpppShellTask task;
    GeneratedDpppVec3 positions[4];
    GeneratedDpppPrimitiveGeometry primitive;
    double coulomb[{coulomb_storage_count}];
  }};
  __shared__ Shared shared;
  const unsigned lane = threadIdx.x;
  if (blockDim.x != kGeneratedDpppFockBlockThreads) return;
  if (lane == 0U) {{
    shared.task = tasks[task_index];
#pragma unroll
    for (unsigned center = 0; center < 4U; ++center) {{
      shared.positions[center] = atom_positions[shared.task.atom[center]];
    }}
  }}
  {barrier}

{component_schedule_setup}{task_component_setup}
  const std::size_t matrix_order =
      static_cast<std::size_t>(shared.task.matrix_order);
  const bool retained_by_schwarz = component_lane && unique_ket_component &&
      (schwarz_bounds == nullptr ||
       schwarz_bounds[
           shared.task.density_offset +
           generated_dppp_matrix_index(i, j, matrix_order)] *
           schwarz_bounds[
               shared.task.density_offset +
               generated_dppp_matrix_index(k, l, matrix_order)] >=
           screening_tolerance);
  const double angular_coefficient = retained_by_schwarz
      ? ao_coefficients[shared.task.ao_coefficient_begin[0] + {component_names[0]}] *
        ao_coefficients[shared.task.ao_coefficient_begin[1] + {component_names[1]}] *
        ao_coefficients[shared.task.ao_coefficient_begin[2] + {component_names[2]}] *
        ao_coefficients[shared.task.ao_coefficient_begin[3] + {component_names[3]}]
      : 0.0;
  double component_integral = 0.0;

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
      {barrier}
{coulomb_setup}
      if (retained_by_schwarz) {{
        component_integral += angular_coefficient *
            shared.primitive.primitive_coefficient *
            generated_dppp_component_value<{shared_coulomb}>(
                component, shared.primitive, shared.coulomb);
      }}
      {barrier}
    }}
  }}
  if (retained_by_schwarz && component_integral != 0.0) {{
    generated_dppp_accumulate_fock<Unrestricted>(
        shared.task, density, fock, i, j, k, l, component_integral);
  }}
{component_schedule_close}}}

extern "C" __global__ __launch_bounds__(
    kGeneratedDpppFockBlockThreads, {minimum_blocks_per_sm})
void generated_dppp_shell_class_fock_rhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    std::size_t task_count) {{
  if (blockIdx.x >= task_count) return;
  generated_dppp_shell_class_fock_task<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      static_cast<std::size_t>(blockIdx.x));
}}

extern "C" __global__ __launch_bounds__(
    kGeneratedDpppFockBlockThreads, {minimum_blocks_per_sm})
void generated_dppp_shell_class_fock_uhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    std::size_t task_count) {{
  if (blockIdx.x >= task_count) return;
  generated_dppp_shell_class_fock_task<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      static_cast<std::size_t>(blockIdx.x));
}}

template <bool Unrestricted>
__device__ __forceinline__ void
generated_dppp_shell_class_fock_persistent(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  const unsigned lane = threadIdx.x;
  __shared__ std::uint32_t shared_task_index;
  while (true) {{
    if (lane == 0U) shared_task_index = atomicAdd(task_head, 1U);
    {barrier}
    const std::uint32_t task_index = shared_task_index;
    if (task_index >= *task_count) return;
    generated_dppp_shell_class_fock_task<Unrestricted>(
        tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
        atom_positions, screening_tolerance, schwarz_bounds, density, fock,
        static_cast<std::size_t>(*task_offset + task_index));
    {barrier}
  }}
}}

extern "C" __global__ __launch_bounds__(
    kGeneratedDpppFockBlockThreads, {minimum_blocks_per_sm})
void generated_dppp_shell_class_fock_rhf_persistent_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  generated_dppp_shell_class_fock_persistent<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      task_offset, task_count, task_head);
}}

extern "C" __global__ __launch_bounds__(
    kGeneratedDpppFockBlockThreads, {minimum_blocks_per_sm})
void generated_dppp_shell_class_fock_uhf_persistent_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  generated_dppp_shell_class_fock_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      task_offset, task_count, task_head);
}}
"""
    if _supports_rys_component_lane_fock(spec, plan):
        worker_marker = """template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_fock_task("""
        worker_begin = source.find(worker_marker)
        if worker_begin < 0:
            raise RuntimeError("generated Fock task marker changed unexpectedly")
        source = source[:worker_begin] + _emit_rys_component_lane_fock_consumer_cuda(
            spec,
            plan,
            minimum_blocks_per_sm,
        )
    elif plan.schedule.kind == ScheduleKind.PACKED_TASKS:
        worker_marker = """template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_fock_task("""
        worker_begin = source.find(worker_marker)
        if worker_begin < 0:
            raise RuntimeError("generated Fock task marker changed unexpectedly")
        source = source[:worker_begin] + _emit_packed_fock_consumer_cuda(
            spec,
            plan,
            minimum_blocks_per_sm,
        )
    elif plan.schedule.kind == ScheduleKind.SUBGROUP_TASKS:
        worker_marker = """template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_fock_task("""
        worker_begin = source.find(worker_marker)
        if worker_begin < 0:
            raise RuntimeError("generated Fock task marker changed unexpectedly")
        source = source[:worker_begin] + _emit_subgroup_fock_consumer_cuda(
            spec,
            plan,
            minimum_blocks_per_sm,
        )
    return source


def _emit_shell_class_mixed_fock_cuda(
    spec: ShellClassSpec, plan: FusedShellPlan
) -> str:
    """Emit an FP32 ERI specialization with FP64 Fock contraction.

    Mixed workers use a compact FP32 primitive-geometry record. Geometry
    differences, powers, and Boys values are evaluated in double precision
    and converted once when stored in the float recurrence record; this keeps
    the accuracy-sensitive recurrence stable while removing repeated
    double-to-float conversions from the Coulomb/component hot loop. Density
    reads and global Fock atomics remain in double precision.
    """

    state_axis_bits = max(3, spec.maximum_force_coulomb_order.bit_length())
    state_mask = (1 << state_axis_bits) - 1
    source = _emit_shell_class_fock_cuda(spec, plan)
    geometry_side = spec.maximum_force_coulomb_order + 1

    mixed_geometry = f"""
/** FP32 geometry used only by the mixed value path. */
struct GeneratedDpppMixedPrimitiveGeometry {{
  float inverse_two_p;
  float inverse_two_q;
  float rho;
  float product_scales[3];
  float pair_shifts[4][3];
  float difference[3];
  float boys[{geometry_side}];
  float coordinate_powers[3][{geometry_side}];
  float negative_two_rho_powers[{geometry_side}];
  float prefactor;
  float primitive_coefficient;
}};

/** Build FP32 primitive geometry without a double-valued hot-loop bridge. */
__device__ __forceinline__ void generated_dppp_make_mixed_primitive_geometry(
    const GeneratedDpppPrimitivePairData& first_pair,
    const GeneratedDpppPrimitivePairData& second_pair,
    bool first_pair_reversed,
    bool second_pair_reversed,
    const GeneratedDpppVec3& first,
    const GeneratedDpppVec3& second,
    const GeneratedDpppVec3& third,
    const GeneratedDpppVec3& fourth,
    GeneratedDpppMixedPrimitiveGeometry& geometry) {{
  const double p = first_pair.exponent_sum;
  const double q = second_pair.exponent_sum;
  const double rho = p * q / (p + q);
  geometry.rho = static_cast<float>(rho);
  geometry.inverse_two_p = static_cast<float>(0.5 / p);
  geometry.inverse_two_q = static_cast<float>(0.5 / q);
  geometry.product_scales[0] = static_cast<float>(first_pair_reversed
      ? first_pair.second_product_scale : first_pair.first_product_scale);
  geometry.product_scales[1] = static_cast<float>(first_pair_reversed
      ? first_pair.first_product_scale : first_pair.second_product_scale);
  geometry.product_scales[2] = static_cast<float>(second_pair_reversed
      ? second_pair.second_product_scale : second_pair.first_product_scale);
  double argument_squared_distance = 0.0;
#pragma unroll
  for (unsigned axis = 0; axis < 3U; ++axis) {{
    const double first_coordinate = generated_dppp_axis(first, axis);
    const double second_coordinate = generated_dppp_axis(second, axis);
    const double third_coordinate = generated_dppp_axis(third, axis);
    const double fourth_coordinate = generated_dppp_axis(fourth, axis);
    const double product_p = generated_dppp_axis(first_pair.product_center, axis);
    const double product_q = generated_dppp_axis(second_pair.product_center, axis);
    geometry.pair_shifts[0][axis] = static_cast<float>(product_p - first_coordinate);
    geometry.pair_shifts[1][axis] = static_cast<float>(product_p - second_coordinate);
    geometry.pair_shifts[2][axis] = static_cast<float>(product_q - third_coordinate);
    geometry.pair_shifts[3][axis] = static_cast<float>(product_q - fourth_coordinate);
    const double difference = product_p - product_q;
    geometry.difference[axis] = static_cast<float>(difference);
    argument_squared_distance +=
        difference * difference;
    double coordinate_power = 1.0;
    geometry.coordinate_powers[axis][0] = 1.0F;
#pragma unroll
  for (unsigned power = 1; power <= {geometry_side - 1}U; ++power) {{
      coordinate_power *= difference;
      geometry.coordinate_powers[axis][power] = static_cast<float>(coordinate_power);
    }}
  }}
  double boys_double[{geometry_side}];
  boys_values<{geometry_side - 1}>(rho * argument_squared_distance, boys_double);
#pragma unroll
  for (unsigned order = 0; order <= {geometry_side - 1}U; ++order) {{
    geometry.boys[order] = static_cast<float>(boys_double[order]);
  }}
  geometry.negative_two_rho_powers[0] = 1.0F;
  double negative_two_rho_power = 1.0;
#pragma unroll
  for (unsigned power = 1; power <= {geometry_side - 1}U; ++power) {{
    negative_two_rho_power *= -2.0 * rho;
    geometry.negative_two_rho_powers[power] =
        static_cast<float>(negative_two_rho_power);
  }}
  geometry.prefactor = static_cast<float>(
      34.986836655249725 / (p * q * sqrt(p + q)));
  geometry.primitive_coefficient = static_cast<float>(
      first_pair.weighted_coefficient * second_pair.weighted_coefficient);
}}

"""

    # Give every emitted Fock symbol an independent mixed-precision sibling.
    # Geometry, lookup tables, and permutation helpers remain shared with the
    # FP64/force fragment that precedes this source.
    symbol_replacements = (
        ("GeneratedDpppValueTerm", "GeneratedDpppMixedValueTerm"),
        (
            "kGeneratedDpppFockCoulombStateCount",
            "kGeneratedDpppMixedFockCoulombStateCount",
        ),
        (
            "kGeneratedDpppFockBlockThreads",
            "kGeneratedDpppMixedFockBlockThreads",
        ),
        (
            "generated_dppp_add_value_matching",
            "generated_dppp_mixed_add_value_matching",
        ),
        (
            "generated_dppp_pair_value_term",
            "generated_dppp_mixed_pair_value_term",
        ),
        (
            "generated_dppp_component_coulomb",
            "generated_dppp_mixed_component_coulomb",
        ),
        (
            "generated_dppp_component_value",
            "generated_dppp_mixed_component_value",
        ),
        (
            "generated_dppp_accumulate_fock",
            "generated_dppp_mixed_accumulate_fock",
        ),
        (
            "generated_dppp_shell_class_fock",
            "generated_dppp_shell_class_mixed_fock",
        ),
        (
            "generated_dppp_packed_fock",
            "generated_dppp_packed_mixed_fock",
        ),
        (
            "GeneratedDpppSubgroupFockStorage",
            "GeneratedDpppMixedSubgroupFockStorage",
        ),
        (
            "generated_dppp_subgroup_fock_task",
            "generated_dppp_mixed_subgroup_fock_task",
        ),
        (
            "generated_dppp_subgroup_fock_persistent",
            "generated_dppp_mixed_subgroup_fock_persistent",
        ),
        # Rys3 component-lane Fock lowering emits a value-axis helper inside
        # the Fock fragment.  The mixed sibling reuses the shared Rys tables
        # but must own a distinct helper symbol because the ordinary Fock
        # fragment is emitted immediately before it in the same CUDA TU.
        (
            "generated_dppp_rys3_value_axis",
            "generated_dppp_mixed_rys3_value_axis",
        ),
    )
    for original, replacement in symbol_replacements:
        source = source.replace(original, replacement)

    # The FP64 geometry helpers are emitted once for the ordinary force/value
    # path. Mixed workers use a compact record and builder so the component
    # recurrence reads native floats instead of converting every geometry
    # array element on every state/component visit.
    source = source.replace(
        "GeneratedDpppPrimitiveGeometry", "GeneratedDpppMixedPrimitiveGeometry"
    )
    source = source.replace(
        "generated_dppp_make_primitive_geometry(\n",
        "generated_dppp_make_mixed_primitive_geometry(\n",
    )

    # Convert only the ERI-evaluation data flow. Density reads, Fock values,
    # and atomics retain their double declarations in the duplicated
    # accumulation helper and kernel ABI.
    source = source.replace(
        "struct GeneratedDpppMixedValueTerm {\n"
        "  unsigned derivative_state;\n"
        "  double coefficient;\n"
        "};",
        "struct GeneratedDpppMixedValueTerm {\n"
        "  unsigned derivative_state;\n"
        "  float coefficient;\n"
        "};",
    )
    source = source.replace("    const double* shifts,", "    const float* shifts,")
    source = source.replace(
        "    double inverse_two_exponent,", "    float inverse_two_exponent,"
    )
    source = source.replace(
        "  double coefficient = 1.0;", "  float coefficient = 1.0F;"
    )
    source = source.replace("  const double first_shifts", "  const float first_shifts")
    source = source.replace(
        "  const double second_shifts", "  const float second_shifts"
    )
    source = re.sub(
        r"(?m)^(\s+)(geometry\.pair_shifts\[[^\n]+?)(,|};)$",
        lambda match: (
            f"{match.group(1)}static_cast<float>({match.group(2)}){match.group(3)}"
        ),
        source,
    )
    source = source.replace("  double value = 0.0;", "  float value = 0.0F;")
    source = source.replace("      const double sign =", "      const float sign =")
    source = source.replace("? 1.0 : -1.0;", "? 1.0F : -1.0F;")
    source = source.replace(
        "__device__ __forceinline__ double generated_dppp_mixed_component_value",
        "__device__ __forceinline__ float generated_dppp_mixed_component_value",
    )
    source = source.replace(
        "    const double* coulomb) {", "    const float* coulomb) {"
    )
    source = source.replace(
        "  return geometry.prefactor * value;",
        "  return static_cast<float>(geometry.prefactor) * value;",
    )
    source = source.replace(
        "    double coulomb[kGeneratedDpppMixedFockCoulombStateCount];",
        "    float coulomb[kGeneratedDpppMixedFockCoulombStateCount];",
    )
    source = source.replace(
        "  double coulomb[kGeneratedDpppMixedFockCoulombStateCount];",
        "  float coulomb[kGeneratedDpppMixedFockCoulombStateCount];",
    )
    # Component-lane schedules may deliberately recompute Coulomb values and
    # therefore use a one-element scratch slot instead of the shared state
    # table.  Keep that slot in FP32 as well; otherwise the generated mixed
    # worker passes a ``double*`` to the FP32 component evaluator and fails
    # only when a resource-valid schedule selects ``shared_coulomb=false``.
    source = source.replace(
        "    double coulomb[1];",
        "    float coulomb[1];",
    )
    source = source.replace(
        "  double angular_coefficients[kGeneratedDpppComponentCount];",
        "  float angular_coefficients[kGeneratedDpppComponentCount];",
    )
    source = re.sub(
        r"(?m)^(\s*)double component_integrals\[([^\]]+)\]\{\};$",
        r"\1float component_integrals[\2]{};",
        source,
    )
    source = source.replace(
        "        const double angular_coefficient =",
        "        const float angular_coefficient =",
    )
    source = source.replace("        : 0.0;\n", "        : 0.0F;\n")
    source = source.replace(
        "if (angular_coefficient == 0.0) continue;",
        "if (angular_coefficient == 0.0F) continue;",
    )
    source = source.replace(
        "  const double angular_coefficient = retained_by_schwarz",
        "  const float angular_coefficient = retained_by_schwarz",
    )
    source = source.replace(
        "      : 0.0;\n  double component_integral = 0.0;",
        "      : 0.0F;\n  float component_integral = 0.0F;",
    )
    source = source.replace(
        "            shared.primitive.primitive_coefficient *\n"
        "            generated_dppp_mixed_component_value",
        "            static_cast<float>(\n"
        "                shared.primitive.primitive_coefficient) *\n"
        "            generated_dppp_mixed_component_value",
    )
    source = source.replace(
        "/** Coefficient-only pair term used by the SCF Fock recurrence. */",
        "/** FP32 coefficient-only pair term used by mixed SCF Fock. */",
        1,
    )

    mixed_coulomb = f"""
/** Evaluate one value-only Coulomb derivative in FP32. */
__device__ __forceinline__ float generated_dppp_mixed_coulomb(
    unsigned derivative_state,
    const GeneratedDpppMixedPrimitiveGeometry& geometry) {{
  const unsigned x_order = derivative_state & {state_mask}U;
  const unsigned y_order =
      (derivative_state >> {state_axis_bits}U) & {state_mask}U;
  const unsigned z_order =
      (derivative_state >> {2 * state_axis_bits}U) & {state_mask}U;
  const unsigned total_order = x_order + y_order + z_order;
  float value = 0.0F;
  for (unsigned x_pairs = 0; x_pairs <= x_order / 2U; ++x_pairs) {{
    for (unsigned y_pairs = 0; y_pairs <= y_order / 2U; ++y_pairs) {{
      for (unsigned z_pairs = 0; z_pairs <= z_order / 2U; ++z_pairs) {{
        const unsigned contraction_count = x_pairs + y_pairs + z_pairs;
        const unsigned boys_order = total_order - contraction_count;
        const unsigned multiplicity =
            generated_dppp_wick_multiplicity(x_order, x_pairs) *
            generated_dppp_wick_multiplicity(y_order, y_pairs) *
            generated_dppp_wick_multiplicity(z_order, z_pairs);
        value += static_cast<float>(multiplicity) *
            static_cast<float>(
                geometry.negative_two_rho_powers[boys_order]) *
            static_cast<float>(geometry.coordinate_powers[0][
                x_order - 2U * x_pairs]) *
            static_cast<float>(geometry.coordinate_powers[1][
                y_order - 2U * y_pairs]) *
            static_cast<float>(geometry.coordinate_powers[2][
                z_order - 2U * z_pairs]) *
            static_cast<float>(geometry.boys[boys_order]);
      }}
    }}
  }}
  return value;
}}

template <bool SharedCoulomb>
__device__ __forceinline__ float generated_dppp_mixed_component_coulomb(
    const GeneratedDpppMixedPrimitiveGeometry& geometry,
    const float* values,
    unsigned state) {{
  if constexpr (SharedCoulomb) {{
    return values[generated_dppp_state_index(state)];
  }}
  return generated_dppp_mixed_coulomb(state, geometry);
}}

"""
    source = source.replace(
        "generated_dppp_coulomb(\n", "generated_dppp_mixed_coulomb(\n"
    )
    return mixed_geometry + mixed_coulomb + source

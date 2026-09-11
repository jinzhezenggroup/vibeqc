"""Emit fixed-root component-lane Fock contractions from the selected shell plan."""

from __future__ import annotations

from ..cuda_schedule import (
    ScheduleKind,
)
from ..fused_schedule import (
    FusedShellPlan,
)
from ..rys import (
    build_rys_force_program,
)
from ..shell_spec import (
    ShellClassSpec,
)
from .common import _emitted_component_names, _generic_task_component_setup


def _emit_rys_component_lane_fock_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit fixed-root Rys value contraction with one lane per component.

    The force emitter immediately preceding the Fock source already owns the
    accepted fixed-root tables, shell-pair geometry, and HRR state helpers.
    Reusing those definitions here avoids the exponential subset/Wick value
    contraction for high-angular-momentum classes.  This mirrors GPU4PySCF's
    fixed-root J/K structure while retaining VibeQC's exact screening and
    canonical Fock scatter conventions.
    """

    program = build_rys_force_program(spec, integral=plan.kernel.integral)
    recurrence = f"rys{program.nroots}"
    if plan.kernel.integral.recurrence != recurrence:
        raise ValueError(
            f"component-lane Rys Fock for {spec.name} requires {recurrence}"
        )
    if (
        plan.schedule.kind != ScheduleKind.COMPONENT_LANES
        or plan.schedule.block_threads < spec.component_count
    ):
        raise ValueError(
            "component-lane Rys Fock requires one lane per Cartesian component"
        )
    if program.nroots not in (3, 4):
        raise ValueError("component-lane Rys Fock supports three or four roots")
    if max(spec.angular) > 2 or spec.angular[3] > 1:
        raise ValueError(
            "component-lane Rys Fock currently supports s/p/d shells with "
            "at most p angular momentum on the fourth center"
        )

    task_component_setup = _generic_task_component_setup(spec)
    component_names = _emitted_component_names(spec)
    symbol_tag = f"rys{program.nroots}"
    class_tag = f"Rys{program.nroots}"
    bra_extent = sum(spec.angular[:2]) + 2
    ket_extent = sum(spec.angular[2:]) + 2
    task_qualifier = "__noinline__" if spec.angular[1] == 2 else "__forceinline__"
    kernel_qualifier = (
        f"__launch_bounds__(kGeneratedDpppFockBlockThreads, {minimum_blocks_per_sm})"
    )

    def axis_count(center: int, axis: int) -> str:
        """Return one runtime Cartesian exponent from a component ordinal."""

        order = spec.angular[center]
        component_name = component_names[center]
        if order == 0:
            return "0U"
        if order == 1:
            return f"({component_name} == {axis}U)"
        return (
            f"(generated_dppp_d_axes[{component_name}][0] == {axis}U) + "
            f"(generated_dppp_d_axes[{component_name}][1] == {axis}U)"
        )

    component_axis_counts = tuple(
        tuple(axis_count(center, axis) for axis in range(3)) for center in range(4)
    )
    return f"""
/** Evaluate only the value needed by Fock, omitting force derivatives. */
__device__ __noinline__ double generated_dppp_{symbol_tag}_value_axis(
    unsigned a, unsigned b, unsigned c, unsigned d,
    double c0, double cp, double ab, double cd,
    double b10, double b00, double b01, double seed) {{
  // Runtime component ordinals require addressed storage.  The exact extents
  // are inherited from the validated force HRR and include no dynamic bounds.
  volatile double trr[{bra_extent}][{ket_extent}];
  trr[0][0] = seed;
#pragma unroll
  for (unsigned bra = 1U; bra < {bra_extent}U; ++bra) {{
    double value = c0 * trr[bra - 1U][0];
    if (bra > 1U) value += (bra - 1U) * b10 * trr[bra - 2U][0];
    trr[bra][0] = value;
  }}
#pragma unroll
  for (unsigned ket = 1U; ket < {ket_extent}U; ++ket) {{
#pragma unroll
    for (unsigned bra = 0U; bra < {bra_extent}U; ++bra) {{
      double value = cp * trr[bra][ket - 1U];
      if (ket > 1U) value +=
          (ket - 1U) * b01 * trr[bra][ket - 2U];
      if (bra > 0U) value +=
          bra * b00 * trr[bra - 1U][ket - 1U];
      trr[bra][ket] = value;
    }}
  }}
  return generated_dppp_{symbol_tag}_state(trr, a, b, c, d, ab, cd);
}}

template <bool Unrestricted>
__device__ {task_qualifier} void generated_dppp_shell_class_fock_task(
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
    GeneratedDppp{class_tag}Primitive primitive;
    double roots_weights[{2 * program.nroots}];
  }};
  __shared__ Shared shared;
  const unsigned lane = threadIdx.x;
  if (blockDim.x != kGeneratedDpppFockBlockThreads) return;
  if (lane == 0U) {{
    shared.task = tasks[task_index];
#pragma unroll
    for (unsigned center = 0U; center < 4U; ++center) {{
      shared.positions[center] = atom_positions[shared.task.atom[center]];
    }}
  }}
  __syncthreads();

  const bool component_lane = lane < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? lane : 0U;
{task_component_setup}
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
      ? ao_coefficients[
            shared.task.ao_coefficient_begin[0] + {component_names[0]}] *
        ao_coefficients[
            shared.task.ao_coefficient_begin[1] + {component_names[1]}] *
        ao_coefficients[
            shared.task.ao_coefficient_begin[2] + {component_names[2]}] *
        ao_coefficients[
            shared.task.ao_coefficient_begin[3] + {component_names[3]}]
      : 0.0;
  if (!__syncthreads_or(retained_by_schwarz)) return;

  const unsigned ax = {component_axis_counts[0][0]};
  const unsigned ay = {component_axis_counts[0][1]};
  const unsigned az = {component_axis_counts[0][2]};
  const unsigned bx = {component_axis_counts[1][0]};
  const unsigned by = {component_axis_counts[1][1]};
  const unsigned bz = {component_axis_counts[1][2]};
  const unsigned cx = {component_axis_counts[2][0]};
  const unsigned cy = {component_axis_counts[2][1]};
  const unsigned cz = {component_axis_counts[2][2]};
  const unsigned dx_order = {component_axis_counts[3][0]};
  const unsigned dy_order = {component_axis_counts[3][1]};
  const unsigned dz_order = {component_axis_counts[3][2]};
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
        const GeneratedDpppPrimitivePairData first_pair =
            primitive_pairs[first_primitive];
        const GeneratedDpppPrimitivePairData second_pair =
            primitive_pairs[second_primitive];
        GeneratedDppp{class_tag}Primitive& primitive = shared.primitive;
        primitive.p = first_pair.exponent_sum;
        primitive.q = second_pair.exponent_sum;
        primitive.pax = first_pair.product_center.x - shared.positions[0].x;
        primitive.pay = first_pair.product_center.y - shared.positions[0].y;
        primitive.paz = first_pair.product_center.z - shared.positions[0].z;
        primitive.qcx = second_pair.product_center.x - shared.positions[2].x;
        primitive.qcy = second_pair.product_center.y - shared.positions[2].y;
        primitive.qcz = second_pair.product_center.z - shared.positions[2].z;
        primitive.abx = shared.positions[1].x - shared.positions[0].x;
        primitive.aby = shared.positions[1].y - shared.positions[0].y;
        primitive.abz = shared.positions[1].z - shared.positions[0].z;
        primitive.cdx = shared.positions[3].x - shared.positions[2].x;
        primitive.cdy = shared.positions[3].y - shared.positions[2].y;
        primitive.cdz = shared.positions[3].z - shared.positions[2].z;
        primitive.dx = first_pair.product_center.x -
            second_pair.product_center.x;
        primitive.dy = first_pair.product_center.y -
            second_pair.product_center.y;
        primitive.dz = first_pair.product_center.z -
            second_pair.product_center.z;
        const double rho =
            primitive.p * primitive.q / (primitive.p + primitive.q);
        generated_dppp_{symbol_tag}_roots(
            rho * (primitive.dx * primitive.dx +
                   primitive.dy * primitive.dy +
                   primitive.dz * primitive.dz),
            shared.roots_weights, 1U);
        primitive.primitive_prefactor =
            34.986836655249725 * first_pair.weighted_coefficient *
            second_pair.weighted_coefficient /
            (primitive.p * primitive.q * sqrt(primitive.p + primitive.q));
      }}
      __syncthreads();
      if (retained_by_schwarz) {{
        const GeneratedDppp{class_tag}Primitive& primitive = shared.primitive;
#pragma unroll 1
        for (unsigned root_index = 0U; root_index < {program.nroots}U;
             ++root_index) {{
          const double root = shared.roots_weights[2U * root_index];
          const double weighted_root =
              shared.roots_weights[2U * root_index + 1U] *
              primitive.primitive_prefactor * angular_coefficient;
          const double root_over_sum = root / (primitive.p + primitive.q);
          const double root_bra = root_over_sum * primitive.q;
          const double root_ket = root_over_sum * primitive.p;
          const double b10 = 0.5 / primitive.p * (1.0 - root_bra);
          const double b00 = 0.5 * root_over_sum;
          const double b01 = 0.5 / primitive.q * (1.0 - root_ket);
          const double x = generated_dppp_{symbol_tag}_value_axis(
              ax, bx, cx, dx_order,
              primitive.pax - primitive.dx * root_bra,
              primitive.qcx + primitive.dx * root_ket,
              primitive.abx, primitive.cdx, b10, b00, b01, 1.0);
          const double y = generated_dppp_{symbol_tag}_value_axis(
              ay, by, cy, dy_order,
              primitive.pay - primitive.dy * root_bra,
              primitive.qcy + primitive.dy * root_ket,
              primitive.aby, primitive.cdy, b10, b00, b01, 1.0);
          const double z = generated_dppp_{symbol_tag}_value_axis(
              az, bz, cz, dz_order,
              primitive.paz - primitive.dz * root_bra,
              primitive.qcz + primitive.dz * root_ket,
              primitive.abz, primitive.cdz, b10, b00, b01, weighted_root);
          component_integral += x * y * z;
        }}
      }}
      __syncthreads();
    }}
  }}
  if (retained_by_schwarz && component_integral != 0.0) {{
    generated_dppp_accumulate_fock<Unrestricted>(
        shared.task, density, fock, i, j, k, l, component_integral);
  }}
}}

extern "C" __global__ {kernel_qualifier}
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

extern "C" __global__ {kernel_qualifier}
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
__device__ __forceinline__ void generated_dppp_shell_class_fock_persistent(
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
  __shared__ std::uint32_t shared_task_index;
  while (true) {{
    if (threadIdx.x == 0U) shared_task_index = atomicAdd(task_head, 1U);
    __syncthreads();
    const std::uint32_t task_index = shared_task_index;
    if (task_index >= *task_count) return;
    generated_dppp_shell_class_fock_task<Unrestricted>(
        tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
        atom_positions, screening_tolerance, schwarz_bounds, density, fock,
        static_cast<std::size_t>(*task_offset + task_index));
    __syncthreads();
  }}
}}

extern "C" __global__ {kernel_qualifier}
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

extern "C" __global__ {kernel_qualifier}
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

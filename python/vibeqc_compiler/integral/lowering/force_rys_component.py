"""Emit component-lane fixed-root force contractions and their shared reductions."""

from __future__ import annotations

from ..cuda_schedule import (
    ScheduleKind,
)
from ..fused_schedule import (
    FusedShellPlan,
)
from ..rys import (
    build_rys_force_program,
    emit_rys3_roots_cuda,
    emit_rys4_roots_cuda,
    emit_rys5_roots_cuda,
)
from ..shell_spec import (
    ShellClassSpec,
)
from .common import _emitted_component_names, _generic_task_component_setup


def _emit_rys_component_lane_force_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit one cooperative fixed-root Rys task across component lanes.

    All scheduled threads execute the same bounded one-dimensional recurrence.
    A component lane selects only the small angular indices used to read its
    fixed-size TRR table, avoiding both the shell-wide scalar DAG and a
    divergent component-function switch.  Lane zero evaluates the fixed roots
    once per primitive quartet; the other lanes retain only nine force
    accumulators and three compact axis results.
    """

    program = build_rys_force_program(spec, integral=plan.kernel.integral)
    recurrence = f"rys{program.nroots}"
    if plan.kernel.integral.recurrence != recurrence:
        raise ValueError(
            f"cooperative fixed-root lowering for {spec.name} requires "
            f"a {recurrence} plan"
        )
    if plan.schedule.kind != ScheduleKind.COMPONENT_LANES:
        raise ValueError("cooperative fixed-root Rys requires component lanes")
    if plan.schedule.block_threads < spec.component_count:
        raise ValueError("cooperative fixed-root Rys requires one lane per component")
    if max(spec.angular) > 2 or spec.angular[3] > 1:
        raise ValueError(
            "runtime-indexed fixed-root lowering currently supports s/p/d "
            "shells with at most p angular momentum on the fourth center"
        )

    # Pack force scalars by the order of independent centers in the
    # mathematical IR.  Center labels are not storage slots: translation
    # recovery may select a non-final dependent center.
    derivative_fields = ("first", "second", "third", "fourth")
    needs_fourth_derivative = 3 in program.independent_derivative_centers
    independent_center_table = ", ".join(
        f"{center}U" for center in program.independent_derivative_centers
    )

    task_component_setup = _generic_task_component_setup(spec)
    component_names = _emitted_component_names(spec)
    nroots = program.nroots
    class_tag = f"Rys{nroots}"
    symbol_tag = f"rys{nroots}"
    root_symbol = f"generated_dppp_{symbol_tag}"
    bra_extent = sum(spec.angular[:2]) + 2
    ket_extent = sum(spec.angular[2:]) + 2

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
    # A raised derivative on a second-center d shell expands the HRR task
    # enough that inlining it into the persistent queue keeps queue state live
    # across the whole recurrence. Keep that new path behind one device-call
    # boundary; existing p-second-center production kernels remain unchanged.
    task_qualifier = "__noinline__" if spec.angular[1] == 2 else "__forceinline__"
    kernel_qualifier = (
        f"__launch_bounds__({plan.schedule.block_threads}, {minimum_blocks_per_sm})"
    )
    roots_emitters = {
        3: emit_rys3_roots_cuda,
        4: emit_rys4_roots_cuda,
        5: emit_rys5_roots_cuda,
    }
    roots_emitter = roots_emitters.get(nroots)
    if roots_emitter is None:
        raise ValueError(
            "cooperative component-lane lowering currently embeds only "
            "three-, four-, and five-root tables"
        )
    roots_cuda = roots_emitter(symbol_prefix=root_symbol)
    component_force_lines = []
    for slot, center in enumerate(program.independent_derivative_centers):
        field = derivative_fields[center]
        component_force_lines.extend(
            (
                (
                    f"          component_force[{slot * 3}] += "
                    f"x.{field} * y.base * z.base;"
                ),
                (
                    f"          component_force[{slot * 3 + 1}] += "
                    f"x.base * y.{field} * z.base;"
                ),
                (
                    f"          component_force[{slot * 3 + 2}] += "
                    f"x.base * y.base * z.{field};"
                ),
            )
        )
    component_force_code = "\n".join(component_force_lines)
    primitive_delta_field = "  double delta2;\n" if needs_fourth_derivative else ""
    primitive_delta_assignment = (
        "        primitive.delta2 = 2.0 * primitive.q * fourth_product_scale;\n"
        if needs_fourth_derivative
        else ""
    )
    pair_fourth_scale_declaration = (
        "        const double fourth_product_scale = second_pair_reversed\n"
        "            ? second_pair.first_product_scale : second_pair.second_product_scale;\n"
        if needs_fourth_derivative
        else ""
    )
    axis_fourth_field = "  double fourth;\n" if needs_fourth_derivative else ""
    axis_extra_parameter = ", double delta2" if needs_fourth_derivative else ""
    axis_delta_call = (
        ",\n              primitive.delta2" if needs_fourth_derivative else ""
    )
    axis_fourth_code = (
        f"""  const double raised_fourth = generated_dppp_{symbol_tag}_state(
      trr, a, b, c, d + 1U, ab, cd);
  const double lowered_fourth = d == 0U ? 0.0 :
      generated_dppp_{symbol_tag}_state(
          trr, a, b, c, d - 1U, ab, cd);
  result.fourth =
      delta2 * raised_fourth - static_cast<double>(d) * lowered_fourth;
"""
        if needs_fourth_derivative
        else ""
    )
    recovered_atomic_blocks = []
    for recovered_index, center in enumerate(program.recovered_derivative_centers):
        name = "fourth_value" if recovered_index == 0 else f"recovered_value_{center}"
        recovered_atomic_blocks.append(
            f"""  if (lane < 3U) {{
    const double {name} =
        -shared.warp_sums[0][lane] -
        shared.warp_sums[0][3U + lane] -
        shared.warp_sums[0][6U + lane];
    if ({name} != 0.0) {{
      atomicAdd(
          forces + static_cast<std::size_t>(shared.task.atom[{center}]) * 3U + lane,
          {name});
    }}
  }}"""
        )
    recovered_atomic_code = "\n".join(recovered_atomic_blocks)
    return (
        roots_cuda
        + f"""
/** Scalars shared by all component lanes for one primitive quartet. */
struct GeneratedDppp{class_tag}Primitive {{
  double p;
  double q;
  double alpha2;
  double beta2;
  double gamma2;
{primitive_delta_field}
  double pax;
  double pay;
  double paz;
  double qcx;
  double qcy;
  double qcz;
  double abx;
  double aby;
  double abz;
  double cdx;
  double cdy;
  double cdz;
  double dx;
  double dy;
  double dz;
  double primitive_prefactor;
}};

/** Base and requested center first derivatives for one Cartesian axis. */
struct GeneratedDppp{class_tag}Axis {{
  double base;
  double first;
  double second;
  double third;
{axis_fourth_field}
}};

__device__ __forceinline__ double generated_dppp_{symbol_tag}_ket_hrr(
    const volatile double (&trr)[{bra_extent}][{ket_extent}], unsigned a,
    unsigned c, unsigned d,
    double cd) {{
  const double base = trr[a][c];
  return d == 0U ? base : trr[a][c + 1U] - cd * base;
}}

__device__ __forceinline__ double generated_dppp_{symbol_tag}_state(
    const volatile double (&trr)[{bra_extent}][{ket_extent}], unsigned a,
    unsigned b, unsigned c,
    unsigned d, double ab, double cd) {{
  const double base = generated_dppp_{symbol_tag}_ket_hrr(
      trr, a, c, d, cd);
  if (b == 0U) return base;
  const double raised = generated_dppp_{symbol_tag}_ket_hrr(
      trr, a + 1U, c, d, cd);
  if (b == 1U) return raised - ab * base;
  const double raised_twice = generated_dppp_{symbol_tag}_ket_hrr(
      trr, a + 2U, c, d, cd);
  if (b == 2U) {{
    return raised_twice - 2.0 * ab * raised + ab * ab * base;
  }}
  // A d shell on the second center needs b=3 only for its raised first
  // derivative. The exact shell bound keeps a+3 inside the addressed TRR
  // table without introducing a runtime HRR loop.
  const double raised_thrice = generated_dppp_{symbol_tag}_ket_hrr(
      trr, a + 3U, c, d, cd);
  return raised_thrice - 3.0 * ab * raised_twice +
      3.0 * ab * ab * raised - ab * ab * ab * base;
}}

/**
 * Evaluate all one-axis values required for the requested center derivatives.
 *
 * {spec.name.upper()} bounds are exact: after one derivative,
 * ``a+b <= {bra_extent - 1}`` and ``c+d <= {ket_extent - 1}``. Keeping this
 * helper noinline makes its bounded addressed
 * table reusable across x/y/z calls instead of tripling caller register
 * pressure.
 */
__device__ __noinline__ GeneratedDppp{class_tag}Axis
generated_dppp_{symbol_tag}_axis(
    unsigned a, unsigned b, unsigned c, unsigned d,
    double c0, double cp, double ab, double cd,
    double b10, double b00, double b01, double seed,
    double alpha2, double beta2, double gamma2{axis_extra_parameter}) {{
  // Runtime component indices would otherwise make PTXAS retain the complete
  // table in registers across every state lookup.  An explicitly addressed
  // local table trades a bounded frame for much higher occupancy.
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

  GeneratedDppp{class_tag}Axis result;
  result.base = generated_dppp_{symbol_tag}_state(
      trr, a, b, c, d, ab, cd);
  const double raised_first = generated_dppp_{symbol_tag}_state(
      trr, a + 1U, b, c, d, ab, cd);
  const double lowered_first = a == 0U ? 0.0 :
      generated_dppp_{symbol_tag}_state(
          trr, a - 1U, b, c, d, ab, cd);
  result.first = alpha2 * raised_first - static_cast<double>(a) * lowered_first;
  const double raised_second = generated_dppp_{symbol_tag}_state(
      trr, a, b + 1U, c, d, ab, cd);
  const double lowered_second = b == 0U ? 0.0 :
      generated_dppp_{symbol_tag}_state(
          trr, a, b - 1U, c, d, ab, cd);
  result.second =
      beta2 * raised_second - static_cast<double>(b) * lowered_second;
  const double raised_third = generated_dppp_{symbol_tag}_state(
      trr, a, b, c + 1U, d, ab, cd);
  const double lowered_third = c == 0U ? 0.0 :
      generated_dppp_{symbol_tag}_state(
          trr, a, b, c - 1U, d, ab, cd);
  result.third =
      gamma2 * raised_third - static_cast<double>(c) * lowered_third;
{axis_fourth_code}
  return result;
}}

template <bool Unrestricted>
__device__ {task_qualifier} void
generated_dppp_{symbol_tag}_component_lane_task(
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
    GeneratedDppp{class_tag}Primitive primitive;
    double roots_weights[{2 * nroots}];
    double warp_sums[kGeneratedDpppWarpCount][9];
  }};
  __shared__ Shared shared;
  const unsigned lane = threadIdx.x;
  if (blockDim.x != kGeneratedDpppBlockThreads) return;
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
  const double density_coefficient =
      component_lane && unique_ket_component && retained_by_schwarz
      ? generated_dppp_density_coefficient<Unrestricted>(
            shared.task, i, j, k, l, density)
      : 0.0;
  const double angular_coefficient = component_lane
      ? ao_coefficients[
            shared.task.ao_coefficient_begin[0] + {component_names[0]}] *
        ao_coefficients[
            shared.task.ao_coefficient_begin[1] + {component_names[1]}] *
        ao_coefficients[
            shared.task.ao_coefficient_begin[2] + {component_names[2]}] *
        ao_coefficients[
            shared.task.ao_coefficient_begin[3] + {component_names[3]}]
      : 0.0;
  const double density_weight = density_coefficient * angular_coefficient;
  if (!__syncthreads_or(density_weight != 0.0)) return;

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
  double component_force[9]{{}};

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
        const bool first_pair_reversed =
            (shared.task.reversed_shell_pair_mask & 1U) != 0U;
        const bool second_pair_reversed =
            (shared.task.reversed_shell_pair_mask & 2U) != 0U;
        const double first_product_scale = first_pair_reversed
            ? first_pair.second_product_scale : first_pair.first_product_scale;
        const double second_product_scale = first_pair_reversed
            ? first_pair.first_product_scale : first_pair.second_product_scale;
        const double third_product_scale = second_pair_reversed
            ? second_pair.second_product_scale : second_pair.first_product_scale;
{pair_fourth_scale_declaration}
        primitive.alpha2 = 2.0 * primitive.p * first_product_scale;
        primitive.beta2 = 2.0 * primitive.p * second_product_scale;
        primitive.gamma2 = 2.0 * primitive.q * third_product_scale;
{primitive_delta_assignment}
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
        {root_symbol}_roots(
            rho * (primitive.dx * primitive.dx +
                   primitive.dy * primitive.dy +
                   primitive.dz * primitive.dz),
            shared.roots_weights, 1U);
        primitive.primitive_prefactor =
            -34.986836655249725 * first_pair.weighted_coefficient *
            second_pair.weighted_coefficient /
            (primitive.p * primitive.q * sqrt(primitive.p + primitive.q));
      }}
      __syncthreads();
      if (density_weight != 0.0) {{
        const GeneratedDppp{class_tag}Primitive& primitive = shared.primitive;
#pragma unroll 1
        for (unsigned root_index = 0U; root_index < {nroots}U; ++root_index) {{
          const double root = shared.roots_weights[2U * root_index];
          const double weighted_root =
              shared.roots_weights[2U * root_index + 1U] *
              primitive.primitive_prefactor * density_weight;
          const double root_over_sum = root / (primitive.p + primitive.q);
          const double root_bra = root_over_sum * primitive.q;
          const double root_ket = root_over_sum * primitive.p;
          const double b10 = 0.5 / primitive.p * (1.0 - root_bra);
          const double b00 = 0.5 * root_over_sum;
          const double b01 = 0.5 / primitive.q * (1.0 - root_ket);
          const GeneratedDppp{class_tag}Axis x =
              generated_dppp_{symbol_tag}_axis(
              ax, bx, cx, dx_order,
              primitive.pax - primitive.dx * root_bra,
              primitive.qcx + primitive.dx * root_ket,
              primitive.abx, primitive.cdx, b10, b00, b01, 1.0,
              primitive.alpha2, primitive.beta2, primitive.gamma2{axis_delta_call});
          const GeneratedDppp{class_tag}Axis y =
              generated_dppp_{symbol_tag}_axis(
              ay, by, cy, dy_order,
              primitive.pay - primitive.dy * root_bra,
              primitive.qcy + primitive.dy * root_ket,
              primitive.aby, primitive.cdy, b10, b00, b01, 1.0,
              primitive.alpha2, primitive.beta2, primitive.gamma2{axis_delta_call});
          const GeneratedDppp{class_tag}Axis z =
              generated_dppp_{symbol_tag}_axis(
              az, bz, cz, dz_order,
              primitive.paz - primitive.dz * root_bra,
              primitive.qcz + primitive.dz * root_ket,
              primitive.abz, primitive.cdz, b10, b00, b01, weighted_root,
              primitive.alpha2, primitive.beta2, primitive.gamma2{axis_delta_call});
{component_force_code}
        }}
      }}
      __syncthreads();
    }}
  }}

  const unsigned warp = lane / 32U;
  const unsigned warp_lane = lane % 32U;
#pragma unroll
  for (unsigned slot = 0U; slot < 9U; ++slot) {{
    double value = component_force[slot];
#pragma unroll
    for (unsigned offset = 16U; offset != 0U; offset /= 2U) {{
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }}
    if (warp_lane == 0U) shared.warp_sums[warp][slot] = value;
  }}
  __syncthreads();
  if (lane < 9U) {{
    double value = 0.0;
#pragma unroll
    for (unsigned source_warp = 0U;
         source_warp < kGeneratedDpppWarpCount;
         ++source_warp) {{
      value += shared.warp_sums[source_warp][lane];
    }}
    shared.warp_sums[0][lane] = value;
    if (value != 0.0) {{
      const unsigned coordinate = lane % 3U;
      constexpr unsigned derivative_centers[3] = {{{independent_center_table}}};
      atomicAdd(
          forces + static_cast<std::size_t>(
              shared.task.atom[derivative_centers[lane / 3U]]) * 3U +
              coordinate,
          value);
    }}
  }}
  __syncthreads();
{recovered_atomic_code}
}}

extern "C" __global__ {kernel_qualifier}
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
  generated_dppp_{symbol_tag}_component_lane_task<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      static_cast<std::size_t>(blockIdx.x));
}}

extern "C" __global__ {kernel_qualifier}
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
  generated_dppp_{symbol_tag}_component_lane_task<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      static_cast<std::size_t>(blockIdx.x));
}}

template <bool Unrestricted>
__device__ __forceinline__ void
generated_dppp_{symbol_tag}_component_lane_persistent(
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
  __shared__ std::uint32_t shared_task_index;
  while (true) {{
    if (threadIdx.x == 0U) shared_task_index = atomicAdd(task_head, 1U);
    __syncthreads();
    const std::uint32_t task_index = shared_task_index;
    if (task_index >= *task_count) return;
    generated_dppp_{symbol_tag}_component_lane_task<Unrestricted>(
        tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
        atom_positions, screening_tolerance, schwarz_bounds, density, forces,
        static_cast<std::size_t>(*task_offset + task_index));
    __syncthreads();
  }}
}}

extern "C" __global__ {kernel_qualifier}
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
  generated_dppp_{symbol_tag}_component_lane_persistent<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}}

extern "C" __global__ {kernel_qualifier}
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
  generated_dppp_{symbol_tag}_component_lane_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}}
"""
    )

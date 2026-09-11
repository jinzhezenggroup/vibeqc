"""Emit uniform-warp fixed-root force contractions with the existing ownership rules."""

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
    emit_rys_force_root_body_cuda,
)
from ..shell_spec import (
    ShellClassSpec,
)
from .common import _emitted_component_names, _generic_task_component_setup


def _emit_rys_uniform_warp_force_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit 32 fixed-root quartets across eight component-uniform warps.

    The logical task coordinate is the hardware lane within each warp and the
    logical component coordinate is the warp ordinal.  Consequently, every
    hardware warp follows one straight-line recurrence slice for 32 independent
    shell quartets.  This reuses the accepted DPPP execution geometry for other
    fixed-root classes whose scalar whole-task program exceeds the register
    file.

    Unlike the older independent-subgroup prototype, all 256 threads advance a
    32-task batch in lockstep.  Primitive-pair counts may differ between VibeQC
    tasks, so the batch loops to the largest count and predicates shorter tasks;
    this preserves block-barrier safety without assuming contraction-uniform
    shell buckets.  The accepted component-lane Fock worker is emitted
    separately and is deliberately unaffected by this force-only schedule.
    """

    program = build_rys_force_program(spec, integral=plan.kernel.integral)
    recurrence = f"rys{program.nroots}"
    if program.nroots not in (3, 4, 5):
        raise ValueError(
            "uniform-warp lowering requires three, four, or five Rys roots"
        )
    if plan.kernel.integral.recurrence != recurrence:
        raise ValueError(f"uniform-warp lowering for {spec.name} requires {recurrence}")
    if plan.schedule.kind != ScheduleKind.SUBGROUP_TASKS:
        raise ValueError("uniform-warp Rys lowering requires subgroup tasks")
    if (
        plan.schedule.block_threads not in (128, 256)
        or plan.schedule.tasks_per_block != 32
        or plan.schedule.subgroup_lanes != plan.schedule.warp_count
    ):
        raise ValueError(
            "uniform-warp Rys lowering requires 32 quartets distributed "
            "across one component lane per hardware warp"
        )

    task_count = plan.schedule.tasks_per_block
    component_lanes = plan.schedule.subgroup_lanes
    components_per_lane = (
        spec.component_count + component_lanes - 1
    ) // component_lanes
    if task_count != 32:
        raise ValueError("uniform-warp Rys lowering requires 32 tasks per block")

    independent_center_table = ", ".join(
        f"{center}U" for center in program.independent_derivative_centers
    )
    recovered_atomic_lines = []
    for recovered_index, center in enumerate(program.recovered_derivative_centers):
        name = "fourth" if recovered_index == 0 else f"recovered_{center}"
        for coordinate in range(3):
            terms = " - ".join(
                f"reduced[{slot * 3 + coordinate}]"
                for slot in range(len(program.independent_derivative_centers))
            )
            recovered_atomic_lines.append(
                f"""    const double {name}_{coordinate} = -{terms};
    if ({name}_{coordinate} != 0.0) {{
      atomicAdd(
          forces +
              static_cast<std::size_t>(shared.tasks[sq].atom[{center}]) * 3U +
              {coordinate}U,
          {name}_{coordinate});
    }}"""
            )
    recovered_atomic_code = "\n".join(recovered_atomic_lines)

    task_component_setup = _generic_task_component_setup(spec).replace(
        "shared.task", "shared.tasks[sq]"
    )
    component_names = _emitted_component_names(spec)
    root_cases: list[str] = []
    for component_lane in range(component_lanes):
        component_indices = tuple(
            range(component_lane, spec.component_count, component_lanes)
        )
        root_body = emit_rys_force_root_body_cuda(
            spec,
            component_weight_expression=(
                f"density_weights[({{component}}U - {component_lane}U) / "
                f"{component_lanes}U]"
            ),
            # Small groups bound live recurrence state while retaining reuse
            # across adjacent Cartesian components owned by one warp.
            component_group=3,
            component_indices=component_indices,
            integral=plan.kernel.integral,
        )
        indented = "\n".join(f"        {line}" for line in root_body.splitlines())
        root_cases.append(
            f"""      case {component_lane}U:
{indented}
        break;"""
        )
    root_switch = "\n".join(root_cases)
    roots_symbol = f"generated_dppp_rys{program.nroots}_uniform_warp"
    roots_cuda = {
        3: emit_rys3_roots_cuda,
        4: emit_rys4_roots_cuda,
        5: emit_rys5_roots_cuda,
    }[program.nroots](symbol_prefix=roots_symbol)
    ket_difference_loads = ""
    if spec.angular[3] > 0:
        ket_difference_loads = """      const double cdx = primitive.cdx;
      const double cdy = primitive.cdy;
      const double cdz = primitive.cdz;
"""
    kernel_qualifier = (
        f"__launch_bounds__(kGeneratedDpppBlockThreads, {minimum_blocks_per_sm})"
    )

    source = (
        roots_cuda
        + f"""
constexpr unsigned kGeneratedDpppRys4TaskCount = {task_count}U;
constexpr unsigned kGeneratedDpppRys4ComponentLanes = {component_lanes}U;
constexpr unsigned kGeneratedDpppRys4ComponentsPerLane =
    {components_per_lane}U;

/** Scalars shared by the eight component warps for one primitive quartet. */
struct GeneratedDpppRys4UniformPrimitive {{
  double p;
  double q;
  double alpha2;
  double beta2;
  double gamma2;
  double delta2;
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

/**
 * State for one 32-quartet batch.
 *
 * The first array coordinate is chosen so each hardware warp accesses 32
 * adjacent doubles.  This is important because ``threadIdx.x & 31`` is the
 * task ordinal while ``threadIdx.x >> 5`` is the component-lane ordinal.
 */
struct GeneratedDpppRys4UniformBatch {{
  GeneratedDpppShellTask tasks[kGeneratedDpppRys4TaskCount];
  GeneratedDpppVec3 positions[kGeneratedDpppRys4TaskCount][4];
  GeneratedDpppRys4UniformPrimitive
      primitive[kGeneratedDpppRys4TaskCount];
  double roots_weights[{2 * program.nroots}][kGeneratedDpppRys4TaskCount];
  double force_partials[9][kGeneratedDpppRys4ComponentLanes]
                       [kGeneratedDpppRys4TaskCount];
  unsigned component_activity[kGeneratedDpppRys4ComponentLanes]
                             [kGeneratedDpppRys4TaskCount];
  std::uint32_t primitive_count[kGeneratedDpppRys4TaskCount];
  std::uint32_t second_pair_count[kGeneratedDpppRys4TaskCount];
  std::int64_t first_pair_begin[kGeneratedDpppRys4TaskCount];
  std::int64_t second_pair_begin[kGeneratedDpppRys4TaskCount];
  std::uint32_t task_base;
  std::uint32_t maximum_primitive_count;
}};

template <bool Unrestricted>
__device__ __noinline__ void generated_dppp_rys4_uniform_warp_batch(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::uint32_t task_base,
    std::uint32_t task_count,
    GeneratedDpppRys4UniformBatch& shared) {{
  const unsigned thread = threadIdx.x;
  const unsigned sq = thread & 31U;
  const unsigned component_lane = thread >> 5U;
  const std::uint32_t local_task = task_base + sq;
  const bool active_task = local_task < task_count;

  // Warp zero owns task metadata; every later warp then reads the same task
  // ordinal through the transposed shared layout.
  if (component_lane == 0U && active_task) {{
    shared.tasks[sq] = tasks[local_task];
#pragma unroll
    for (unsigned center = 0U; center < 4U; ++center) {{
      shared.positions[sq][center] =
          atom_positions[shared.tasks[sq].atom[center]];
    }}
  }}
  __syncthreads();

  // Make the bounded weight table explicitly addressable.  Leaving it as an
  // ordinary local array makes PTXAS first scalarize all 21 entries and then
  // spill an unpredictable subset once the straight-line recurrence reaches
  // the architectural register limit.
  volatile double density_weights[kGeneratedDpppRys4ComponentsPerLane]{{}};
  bool any_component = false;
#pragma unroll
  for (unsigned local_component = 0U;
       local_component < kGeneratedDpppRys4ComponentsPerLane;
       ++local_component) {{
    const unsigned candidate_component =
        component_lane +
        local_component * kGeneratedDpppRys4ComponentLanes;
    if (active_task && candidate_component < kGeneratedDpppComponentCount) {{
      const unsigned component = candidate_component;
{task_component_setup}
      const std::size_t matrix_order =
          static_cast<std::size_t>(shared.tasks[sq].matrix_order);
      const bool retained_by_schwarz = schwarz_bounds == nullptr ||
          schwarz_bounds[
              shared.tasks[sq].density_offset +
              generated_dppp_matrix_index(i, j, matrix_order)] *
              schwarz_bounds[
                  shared.tasks[sq].density_offset +
                  generated_dppp_matrix_index(k, l, matrix_order)] >=
              screening_tolerance;
      const double density_coefficient =
          unique_ket_component && retained_by_schwarz
          ? generated_dppp_density_coefficient<Unrestricted>(
                shared.tasks[sq], i, j, k, l, density)
          : 0.0;
      const double angular_coefficient =
          ao_coefficients[
              shared.tasks[sq].ao_coefficient_begin[0] + {component_names[0]}] *
          ao_coefficients[
              shared.tasks[sq].ao_coefficient_begin[1] + {component_names[1]}] *
          ao_coefficients[
              shared.tasks[sq].ao_coefficient_begin[2] + {component_names[2]}] *
          ao_coefficients[
              shared.tasks[sq].ao_coefficient_begin[3] + {component_names[3]}];
      const double weight = density_coefficient * angular_coefficient;
      density_weights[local_component] = weight;
      any_component = any_component || weight != 0.0;
    }}
  }}
  shared.component_activity[component_lane][sq] = any_component ? 1U : 0U;
  __syncthreads();

  if (component_lane == 0U) {{
    bool task_has_component = false;
#pragma unroll
    for (unsigned source = 0U;
         source < kGeneratedDpppRys4ComponentLanes; ++source) {{
      task_has_component = task_has_component ||
          shared.component_activity[source][sq] != 0U;
    }}
    std::uint32_t primitive_count = 0U;
    if (active_task && task_has_component) {{
      const std::uint32_t first_pair = shared.tasks[sq].shell_pair[0];
      const std::uint32_t second_pair = shared.tasks[sq].shell_pair[1];
      const std::int64_t first_begin = primitive_pair_offsets[first_pair];
      const std::int64_t first_end = primitive_pair_offsets[first_pair + 1U];
      const std::int64_t second_begin = primitive_pair_offsets[second_pair];
      const std::int64_t second_end = primitive_pair_offsets[second_pair + 1U];
      const std::uint32_t first_count =
          static_cast<std::uint32_t>(first_end - first_begin);
      const std::uint32_t second_count =
          static_cast<std::uint32_t>(second_end - second_begin);
      shared.first_pair_begin[sq] = first_begin;
      shared.second_pair_begin[sq] = second_begin;
      shared.second_pair_count[sq] = second_count;
      primitive_count = first_count * second_count;
    }}
    shared.primitive_count[sq] = primitive_count;
    std::uint32_t batch_maximum = primitive_count;
#pragma unroll
    for (unsigned offset = 16U; offset != 0U; offset >>= 1U) {{
      batch_maximum = max(
          batch_maximum,
          __shfl_down_sync(0xffffffffU, batch_maximum, offset));
    }}
    if (sq == 0U) shared.maximum_primitive_count = batch_maximum;
  }}
  __syncthreads();

  double force_0 = 0.0;
  double force_1 = 0.0;
  double force_2 = 0.0;
  double force_3 = 0.0;
  double force_4 = 0.0;
  double force_5 = 0.0;
  double force_6 = 0.0;
  double force_7 = 0.0;
  double force_8 = 0.0;

  for (std::uint32_t primitive_index = 0U;
       primitive_index < shared.maximum_primitive_count;
       ++primitive_index) {{
    if (component_lane == 0U &&
        primitive_index < shared.primitive_count[sq]) {{
      const std::uint32_t second_count = shared.second_pair_count[sq];
      const std::int64_t first_primitive =
          shared.first_pair_begin[sq] + primitive_index / second_count;
      const std::int64_t second_primitive =
          shared.second_pair_begin[sq] + primitive_index % second_count;
      const GeneratedDpppPrimitivePairData first_pair =
          primitive_pairs[first_primitive];
      const GeneratedDpppPrimitivePairData second_pair =
          primitive_pairs[second_primitive];
      GeneratedDpppRys4UniformPrimitive& primitive = shared.primitive[sq];
      primitive.p = first_pair.exponent_sum;
      primitive.q = second_pair.exponent_sum;
      const bool first_pair_reversed =
          (shared.tasks[sq].reversed_shell_pair_mask & 1U) != 0U;
      const bool second_pair_reversed =
          (shared.tasks[sq].reversed_shell_pair_mask & 2U) != 0U;
      const double first_product_scale = first_pair_reversed
          ? first_pair.second_product_scale : first_pair.first_product_scale;
      const double second_product_scale = first_pair_reversed
          ? first_pair.first_product_scale : first_pair.second_product_scale;
      const double third_product_scale = second_pair_reversed
          ? second_pair.second_product_scale : second_pair.first_product_scale;
      const double fourth_product_scale = second_pair_reversed
          ? second_pair.first_product_scale : second_pair.second_product_scale;
      primitive.alpha2 = 2.0 * primitive.p * first_product_scale;
      primitive.beta2 = 2.0 * primitive.p * second_product_scale;
      primitive.gamma2 = 2.0 * primitive.q * third_product_scale;
      primitive.delta2 = 2.0 * primitive.q * fourth_product_scale;
      primitive.pax = first_pair.product_center.x - shared.positions[sq][0].x;
      primitive.pay = first_pair.product_center.y - shared.positions[sq][0].y;
      primitive.paz = first_pair.product_center.z - shared.positions[sq][0].z;
      primitive.qcx = second_pair.product_center.x - shared.positions[sq][2].x;
      primitive.qcy = second_pair.product_center.y - shared.positions[sq][2].y;
      primitive.qcz = second_pair.product_center.z - shared.positions[sq][2].z;
      primitive.abx =
          shared.positions[sq][1].x - shared.positions[sq][0].x;
      primitive.aby =
          shared.positions[sq][1].y - shared.positions[sq][0].y;
      primitive.abz =
          shared.positions[sq][1].z - shared.positions[sq][0].z;
      primitive.cdx =
          shared.positions[sq][3].x - shared.positions[sq][2].x;
      primitive.cdy =
          shared.positions[sq][3].y - shared.positions[sq][2].y;
      primitive.cdz =
          shared.positions[sq][3].z - shared.positions[sq][2].z;
      primitive.dx =
          first_pair.product_center.x - second_pair.product_center.x;
      primitive.dy =
          first_pair.product_center.y - second_pair.product_center.y;
      primitive.dz =
          first_pair.product_center.z - second_pair.product_center.z;
      const double rho = primitive.p * primitive.q /
          (primitive.p + primitive.q);
      {roots_symbol}_roots(
          rho * (primitive.dx * primitive.dx +
                 primitive.dy * primitive.dy +
                 primitive.dz * primitive.dz),
          &shared.roots_weights[0][sq], kGeneratedDpppRys4TaskCount);
      primitive.primitive_prefactor =
          -34.986836655249725 * first_pair.weighted_coefficient *
          second_pair.weighted_coefficient /
          (primitive.p * primitive.q * sqrt(primitive.p + primitive.q));
    }}
    __syncthreads();

    if (primitive_index < shared.primitive_count[sq]) {{
      const GeneratedDpppRys4UniformPrimitive& primitive =
          shared.primitive[sq];
      const double p = primitive.p;
      const double q = primitive.q;
      const double alpha2 = primitive.alpha2;
      const double beta2 = primitive.beta2;
      const double gamma2 = primitive.gamma2;
      const double delta2 = primitive.delta2;
      const double abx = primitive.abx;
      const double aby = primitive.aby;
      const double abz = primitive.abz;
{ket_difference_loads}
#pragma unroll
      for (unsigned root_index = 0U;
           root_index < {program.nroots}U; ++root_index) {{
        const double root = shared.roots_weights[2U * root_index][sq];
        const double weighted_root =
            shared.roots_weights[2U * root_index + 1U][sq] *
            primitive.primitive_prefactor;
        const double root_over_sum = root / (p + q);
        const double root_bra = root_over_sum * q;
        const double root_ket = root_over_sum * p;
        const double b10 = 0.5 / p * (1.0 - root_bra);
        const double b00 = 0.5 * root_over_sum;
        const double b01 = 0.5 / q * (1.0 - root_ket);
        const double c0x = primitive.pax - primitive.dx * root_bra;
        const double c0y = primitive.pay - primitive.dy * root_bra;
        const double c0z = primitive.paz - primitive.dz * root_bra;
        const double cpx = primitive.qcx + primitive.dx * root_ket;
        const double cpy = primitive.qcy + primitive.dy * root_ket;
        const double cpz = primitive.qcz + primitive.dz * root_ket;
        switch (component_lane) {{
{root_switch}
          default:
            break;
        }}
      }}
    }}
    __syncthreads();
  }}

  shared.force_partials[0][component_lane][sq] = force_0;
  shared.force_partials[1][component_lane][sq] = force_1;
  shared.force_partials[2][component_lane][sq] = force_2;
  shared.force_partials[3][component_lane][sq] = force_3;
  shared.force_partials[4][component_lane][sq] = force_4;
  shared.force_partials[5][component_lane][sq] = force_5;
  shared.force_partials[6][component_lane][sq] = force_6;
  shared.force_partials[7][component_lane][sq] = force_7;
  shared.force_partials[8][component_lane][sq] = force_8;
  __syncthreads();

  if (component_lane == 0U && active_task) {{
    double reduced[9]{{}};
#pragma unroll
    for (unsigned slot = 0U; slot < 9U; ++slot) {{
#pragma unroll
      for (unsigned source = 0U;
           source < kGeneratedDpppRys4ComponentLanes; ++source) {{
        reduced[slot] += shared.force_partials[slot][source][sq];
      }}
      if (reduced[slot] != 0.0) {{
        const unsigned coordinate = slot % 3U;
        constexpr unsigned derivative_centers[3] = {{{independent_center_table}}};
        atomicAdd(
            forces +
                static_cast<std::size_t>(
                    shared.tasks[sq].atom[derivative_centers[slot / 3U]]) * 3U +
                coordinate,
            reduced[slot]);
      }}
    }}
#pragma unroll
{recovered_atomic_code}
  }}
  __syncthreads();
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_rys4_uniform_warp_persistent(
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
  __shared__ GeneratedDpppRys4UniformBatch shared;
  while (true) {{
    if (threadIdx.x == 0U) {{
      shared.task_base = atomicAdd(task_head, kGeneratedDpppRys4TaskCount);
    }}
    __syncthreads();
    const std::uint32_t task_base = shared.task_base;
    if (task_base >= *task_count) return;
    generated_dppp_rys4_uniform_warp_batch<Unrestricted>(
        tasks + *task_offset, primitive_pairs, primitive_pair_offsets,
        ao_coefficients, atom_positions, screening_tolerance, schwarz_bounds,
        density, forces, task_base, *task_count, shared);
  }}
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
  __shared__ GeneratedDpppRys4UniformBatch shared;
  const std::size_t task_base =
      static_cast<std::size_t>(blockIdx.x) * kGeneratedDpppRys4TaskCount;
  if (task_base >= task_count) return;
  generated_dppp_rys4_uniform_warp_batch<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      static_cast<std::uint32_t>(task_base),
      static_cast<std::uint32_t>(task_count), shared);
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
  __shared__ GeneratedDpppRys4UniformBatch shared;
  const std::size_t task_base =
      static_cast<std::size_t>(blockIdx.x) * kGeneratedDpppRys4TaskCount;
  if (task_base >= task_count) return;
  generated_dppp_rys4_uniform_warp_batch<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      static_cast<std::uint32_t>(task_base),
      static_cast<std::uint32_t>(task_count), shared);
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
  generated_dppp_rys4_uniform_warp_persistent<false>(
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
  generated_dppp_rys4_uniform_warp_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}}
"""
    )
    if program.nroots != 4:
        # Keep generated identifiers truthful without perturbing the already
        # accepted DPPP Rys4 source or its resource profile.
        replacement = f"Rys{program.nroots}"
        symbol_replacement = f"rys{program.nroots}"
        return source.replace("Rys4", replacement).replace("rys4", symbol_replacement)
    return source

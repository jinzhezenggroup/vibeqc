"""Emit thread-owned fixed-root force contractions for an explicit shell schedule."""

from __future__ import annotations

from ..fused_schedule import (
    FusedShellPlan,
)
from ..rys import (
    build_rys_force_program,
    emit_rys2_roots_cuda,
    emit_rys3_roots_cuda,
    emit_rys_force_root_body_cuda,
)
from ..shell_spec import (
    ShellClassSpec,
)
from .common import _emitted_component_names, _generic_task_component_setup


def _emit_rys_thread_force_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit one complete low-root Rys task per lane.

    The shell-task ABI, screening, density symmetry, and persistent queue are
    identical to the existing generated worker.  Only the primitive hot loop
    changes: fixed-root interpolation feeds a state-on-first-use TRR/HRR
    program, and each component is contracted immediately into nine register
    force accumulators.  Density weights use lane-major shared SoA storage to
    avoid carrying the entire shell's coefficients through the recurrence.

    The mathematical Rys program is built from ``spec`` rather than from a
    PPPS-specific expression.  This makes PPPS, DSPS, DPPS, and other
    two- and three-root catalog classes share one scalar backend while preserving
    shell-specific straight-line component contraction.
    """

    program = build_rys_force_program(spec, integral=plan.kernel.integral)
    recurrence = f"rys{program.nroots}"
    if plan.kernel.integral.recurrence != recurrence or program.nroots not in (2, 3):
        raise ValueError(
            "direct Rys thread lowering requires a two- or three-root plan"
        )
    if plan.schedule.block_threads != 32:
        raise ValueError("direct Rys thread tasks currently use one CUDA warp")

    recovered_atomic_lines = []
    for recovered_index, center in enumerate(program.recovered_derivative_centers):
        name = "fourth_force" if recovered_index == 0 else f"recovered_force_{center}"
        for coordinate in range(3):
            terms = " - ".join(
                f"force_{slot * 3 + coordinate}"
                for slot in range(len(program.independent_derivative_centers))
            )
            recovered_atomic_lines.append(
                f"""  const double {name}_{coordinate} = -{terms};
  if ({name}_{coordinate} != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
            {coordinate}U,
        {name}_{coordinate});
  }}"""
            )
    recovered_atomic_code = "\n".join(recovered_atomic_lines)

    task_component_setup = _generic_task_component_setup(spec).replace(
        "shared.task", "task"
    )
    component_names = _emitted_component_names(spec)
    weight_blocks = []
    for component in range(spec.component_count):
        setup = task_component_setup.replace(
            "  const unsigned component", "    const unsigned component"
        )
        setup = "\n".join(f"  {line}" if line else line for line in setup.splitlines())
        weight_blocks.append(
            f"""  component_weights[{component}U][lane] = 0.0;
  {{
    constexpr unsigned component = {component}U;
{setup}
    const std::size_t matrix_order =
        static_cast<std::size_t>(task.matrix_order);
    const bool retained_by_schwarz = context.schwarz_bounds == nullptr ||
        context.schwarz_bounds[
            task.density_offset +
            generated_dppp_matrix_index(i, j, matrix_order)] *
            context.schwarz_bounds[
                task.density_offset +
                generated_dppp_matrix_index(k, l, matrix_order)] >=
            context.screening_tolerance;
    if (unique_ket_component && retained_by_schwarz) {{
      const double density_coefficient =
          generated_dppp_density_coefficient<Unrestricted>(
              task, i, j, k, l, context.density);
      const double angular_coefficient =
          context.ao_coefficients[
              task.ao_coefficient_begin[0] + {component_names[0]}] *
          context.ao_coefficients[
              task.ao_coefficient_begin[1] + {component_names[1]}] *
          context.ao_coefficients[
              task.ao_coefficient_begin[2] + {component_names[2]}] *
          context.ao_coefficients[
              task.ao_coefficient_begin[3] + {component_names[3]}];
      component_weights[{component}U][lane] =
          density_coefficient * angular_coefficient;
      any_component |= density_coefficient != 0.0;
    }}
  }}"""
        )

    force_declarations = "\n".join(f"  double force_{slot} = 0.0;" for slot in range(9))
    independent_atomics = []
    for slot, center in enumerate(program.independent_derivative_centers):
        for coordinate in range(3):
            force = slot * 3 + coordinate
            independent_atomics.append(
                f"""  if (force_{force} != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
            {coordinate}U,
        force_{force});
  }}"""
            )
    independent_atomic_code = "\n".join(independent_atomics)

    weight_code = "\n".join(weight_blocks)
    root_body = emit_rys_force_root_body_cuda(
        spec,
        component_group=9,
        integral=plan.kernel.integral,
    )
    fourth_atomic_code = recovered_atomic_code
    kernel_qualifier = f"__launch_bounds__(32, {minimum_blocks_per_sm})"
    # Start from the shared DPPP skeleton so shell specialization also renames
    # this helper.  A PPPS-specific global symbol collides as soon as a second
    # scalar Rys shell is emitted into another production shard.
    roots_emitter = (
        emit_rys2_roots_cuda if program.nroots == 2 else emit_rys3_roots_cuda
    )
    roots_cuda = roots_emitter(symbol_prefix="generated_dppp_rys3")
    source = (
        roots_cuda
        + f"""
/** Immutable pointers shared by all lanes in the persistent Rys worker. */
struct GeneratedDpppRysThreadContext {{
  const GeneratedDpppShellTask* tasks;
  const GeneratedDpppPrimitivePairData* primitive_pairs;
  const std::int64_t* primitive_pair_offsets;
  const double* ao_coefficients;
  const GeneratedDpppVec3* atom_positions;
  double screening_tolerance;
  const double* schwarz_bounds;
  const double* density;
  double* forces;
}};

template <bool Unrestricted>
__device__ __noinline__ bool generated_dppp_rys3_fill_weights(
    const GeneratedDpppShellTask& task,
    const GeneratedDpppRysThreadContext& context,
    double (&component_weights)[kGeneratedDpppComponentCount][32]) {{
  const unsigned lane = threadIdx.x;
  bool any_component = false;
{weight_code}
  return any_component;
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_rys3_force_task(
    const GeneratedDpppRysThreadContext& context,
    std::size_t task_index,
    double (&component_weights)[kGeneratedDpppComponentCount][32],
    double (&roots_weights)[{2 * program.nroots}][32]) {{
  const GeneratedDpppShellTask& task = context.tasks[task_index];
  if (!generated_dppp_rys3_fill_weights<Unrestricted>(
          task, context, component_weights)) {{
    return;
  }}
  const unsigned lane = threadIdx.x;
  const GeneratedDpppVec3 first = context.atom_positions[task.atom[0]];
  const GeneratedDpppVec3 second = context.atom_positions[task.atom[1]];
  const GeneratedDpppVec3 third = context.atom_positions[task.atom[2]];
  const bool first_pair_reversed =
      (task.reversed_shell_pair_mask & 1U) != 0U;
  const bool second_pair_reversed =
      (task.reversed_shell_pair_mask & 2U) != 0U;
{force_declarations}

  const std::int64_t first_pair_begin =
      context.primitive_pair_offsets[task.shell_pair[0]];
  const std::int64_t first_pair_end =
      context.primitive_pair_offsets[task.shell_pair[0] + 1U];
  const std::int64_t second_pair_begin =
      context.primitive_pair_offsets[task.shell_pair[1]];
  const std::int64_t second_pair_end =
      context.primitive_pair_offsets[task.shell_pair[1] + 1U];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end; ++first_primitive) {{
    const GeneratedDpppPrimitivePairData first_pair =
        context.primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double first_product_scale = first_pair_reversed
        ? first_pair.second_product_scale : first_pair.first_product_scale;
    const double second_product_scale = first_pair_reversed
        ? first_pair.first_product_scale : first_pair.second_product_scale;
    const double alpha2 = 2.0 * p * first_product_scale;
    const double beta2 = 2.0 * p * second_product_scale;
    const double pax = first_pair.product_center.x - first.x;
    const double pay = first_pair.product_center.y - first.y;
    const double paz = first_pair.product_center.z - first.z;
    const double abx = second.x - first.x;
    const double aby = second.y - first.y;
    const double abz = second.z - first.z;
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end; ++second_primitive) {{
      const GeneratedDpppPrimitivePairData second_pair =
          context.primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double third_product_scale = second_pair_reversed
          ? second_pair.second_product_scale : second_pair.first_product_scale;
      const double fourth_product_scale = second_pair_reversed
          ? second_pair.first_product_scale : second_pair.second_product_scale;
      const double gamma2 = 2.0 * q * third_product_scale;
      const double delta2 = 2.0 * q * fourth_product_scale;
      const double qcx = second_pair.product_center.x - third.x;
      const double qcy = second_pair.product_center.y - third.y;
      const double qcz = second_pair.product_center.z - third.z;
      const double dx = first_pair.product_center.x -
          second_pair.product_center.x;
      const double dy = first_pair.product_center.y -
          second_pair.product_center.y;
      const double dz = first_pair.product_center.z -
          second_pair.product_center.z;
      const double rho = p * q / (p + q);
      generated_dppp_rys3_roots(
          rho * (dx * dx + dy * dy + dz * dz),
          &roots_weights[0][lane], 32U);
      const double primitive_prefactor =
          -34.986836655249725 * first_pair.weighted_coefficient *
          second_pair.weighted_coefficient / (p * q * sqrt(p + q));
#pragma unroll
      for (unsigned root_index = 0; root_index < {program.nroots}U; ++root_index) {{
        const double root = roots_weights[2U * root_index][lane];
        const double weighted_root =
            roots_weights[2U * root_index + 1U][lane] * primitive_prefactor;
        const double root_over_sum = root / (p + q);
        const double root_bra = root_over_sum * q;
        const double root_ket = root_over_sum * p;
        const double b10 = 0.5 / p * (1.0 - root_bra);
        const double b00 = 0.5 * root_over_sum;
        const double b01 = 0.5 / q * (1.0 - root_ket);
        const double c0x = pax - dx * root_bra;
        const double c0y = pay - dy * root_bra;
        const double c0z = paz - dz * root_bra;
        const double cpx = qcx + dx * root_ket;
        const double cpy = qcy + dy * root_ket;
        const double cpz = qcz + dz * root_ket;
{root_body}
      }}
    }}
  }}
{independent_atomic_code}
{fourth_atomic_code}
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_rys3_force_persistent(
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
  __shared__ std::uint32_t task_base;
  __shared__ GeneratedDpppRysThreadContext context;
  __shared__ double component_weights[kGeneratedDpppComponentCount][32];
  __shared__ double roots_weights[{2 * program.nroots}][32];
  if (threadIdx.x == 0U) {{
    context.tasks = tasks;
    context.primitive_pairs = primitive_pairs;
    context.primitive_pair_offsets = primitive_pair_offsets;
    context.ao_coefficients = ao_coefficients;
    context.atom_positions = atom_positions;
    context.screening_tolerance = screening_tolerance;
    context.schwarz_bounds = schwarz_bounds;
    context.density = density;
    context.forces = forces;
  }}
  __syncthreads();
  while (true) {{
    if (threadIdx.x == 0U) task_base = atomicAdd(task_head, 32U);
    __syncthreads();
    if (task_base >= *task_count) return;
    const std::uint32_t task_index = task_base + threadIdx.x;
    if (task_index < *task_count) {{
      generated_dppp_rys3_force_task<Unrestricted>(
          context,
          static_cast<std::size_t>(*task_offset + task_index),
          component_weights, roots_weights);
    }}
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
  generated_dppp_rys3_force_persistent<false>(
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
  generated_dppp_rys3_force_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}}
"""
    )
    # Retain one source template while keeping every generated symbol honest
    # about the fixed-root evaluator embedded in its translation unit.
    return source.replace("rys3", recurrence)

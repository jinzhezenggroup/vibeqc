"""Emit subgroup force consumers while preserving primitive and component reductions."""

from __future__ import annotations

from ..fused_schedule import (
    FusedShellPlan,
)
from ..shell_spec import (
    ShellClassSpec,
)
from .common import _emitted_component_names, _generic_task_component_setup


def _emit_subgroup_force_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit force workers with several independently progressing tasks per warp."""

    subgroup_lanes = plan.schedule.subgroup_lanes
    subgroup_count = plan.schedule.tasks_per_block
    components_per_lane = (spec.component_count + subgroup_lanes - 1) // subgroup_lanes
    subgroup_mask = (1 << subgroup_lanes) - 1
    task_component_setup = _generic_task_component_setup(spec)
    component_names = _emitted_component_names(spec)
    kernel_qualifier = (
        f"__launch_bounds__({plan.block_threads}, {minimum_blocks_per_sm})"
    )

    def ordinary_wrapper(name: str, unrestricted: str) -> str:
        return f"""
extern "C" __global__ {kernel_qualifier}
void {name}(
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
  __shared__ GeneratedDpppSubgroupForceStorage
      subgroup_storage[{subgroup_count}];
  const unsigned subgroup = threadIdx.x / {subgroup_lanes}U;
  const unsigned lane = threadIdx.x % {subgroup_lanes}U;
  const unsigned subgroup_in_warp =
      (threadIdx.x & 31U) / {subgroup_lanes}U;
  const unsigned subgroup_mask =
      0x{subgroup_mask:08x}U << (subgroup_in_warp * {subgroup_lanes}U);
  const std::size_t task_index =
      static_cast<std::size_t>(blockIdx.x) * {subgroup_count}U + subgroup;
  if (task_index >= task_count) return;
  generated_dppp_subgroup_force_task<{unrestricted}>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_index, subgroup_storage[subgroup], lane, subgroup_mask);
}}
"""

    def persistent_wrapper(name: str, unrestricted: str) -> str:
        return f"""
extern "C" __global__ {kernel_qualifier}
void {name}(
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
  generated_dppp_subgroup_force_persistent<{unrestricted}>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}}
"""

    return (
        f"""

/** Task-local state for one independently progressing CUDA lane subgroup. */
struct GeneratedDpppSubgroupForceStorage {{
  GeneratedDpppShellTask task;
  GeneratedDpppVec3 positions[4];
  GeneratedDpppPrimitiveGeometry primitive;
  double coulomb[kGeneratedDpppCoulombStateCount];
  std::uint32_t task_index;
}};

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_subgroup_force_task(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t task_index,
    GeneratedDpppSubgroupForceStorage& shared,
    unsigned lane,
    unsigned subgroup_mask) {{
  if (lane == 0U) {{
    shared.task = tasks[task_index];
#pragma unroll
    for (unsigned center = 0; center < 4U; ++center) {{
      shared.positions[center] = atom_positions[shared.task.atom[center]];
    }}
  }}
  __syncwarp(subgroup_mask);

  double density_coefficients[{components_per_lane}]{{}};
  double angular_coefficients[{components_per_lane}]{{}};
#pragma unroll
  for (unsigned local_component = 0U;
       local_component < {components_per_lane}U; ++local_component) {{
    const unsigned candidate_component =
        lane + local_component * {subgroup_lanes}U;
    const bool component_lane =
        candidate_component < kGeneratedDpppComponentCount;
    const unsigned component = component_lane ? candidate_component : 0U;
{task_component_setup}
    const std::size_t matrix_order =
        static_cast<std::size_t>(shared.task.matrix_order);
    const bool retained_by_schwarz = schwarz_bounds == nullptr ||
        schwarz_bounds[
            shared.task.density_offset +
            generated_dppp_matrix_index(i, j, matrix_order)] *
            schwarz_bounds[
                shared.task.density_offset +
                generated_dppp_matrix_index(k, l, matrix_order)] >=
            screening_tolerance;
    density_coefficients[local_component] =
        component_lane && unique_ket_component && retained_by_schwarz
        ? generated_dppp_density_coefficient<Unrestricted>(
              shared.task, i, j, k, l, density)
        : 0.0;
    angular_coefficients[local_component] = component_lane
        ? ao_coefficients[
              shared.task.ao_coefficient_begin[0] + {component_names[0]}] *
          ao_coefficients[
              shared.task.ao_coefficient_begin[1] + {component_names[1]}] *
          ao_coefficients[
              shared.task.ao_coefficient_begin[2] + {component_names[2]}] *
          ao_coefficients[
              shared.task.ao_coefficient_begin[3] + {component_names[3]}]
        : 0.0;
  }}

  double subgroup_force[12]{{}};
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
      __syncwarp(subgroup_mask);
      for (unsigned state = lane;
           state < kGeneratedDpppCoulombStateCount;
           state += {subgroup_lanes}U) {{
        shared.coulomb[state] = generated_dppp_coulomb(
            generated_dppp_coulomb_states[state], shared.primitive);
      }}
      __syncwarp(subgroup_mask);
#pragma unroll
      for (unsigned local_component = 0U;
           local_component < {components_per_lane}U; ++local_component) {{
        const double density_coefficient =
            density_coefficients[local_component];
        if (density_coefficient == 0.0) continue;
        const unsigned component =
            lane + local_component * {subgroup_lanes}U;
        double primitive_gradient[4][3];
        generated_dppp_component_gradient<true>(
            component, shared.primitive, shared.coulomb,
            primitive_gradient);
        const double scale =
            -density_coefficient * angular_coefficients[local_component] *
            shared.primitive.primitive_coefficient;
#pragma unroll
        for (unsigned center = 0; center < 4U; ++center) {{
#pragma unroll
          for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
            subgroup_force[center * 3U + coordinate] +=
                scale * primitive_gradient[center][coordinate];
          }}
        }}
      }}
      __syncwarp(subgroup_mask);
    }}
  }}

#pragma unroll
  for (unsigned slot = 0U; slot < 12U; ++slot) {{
    double value = subgroup_force[slot];
#pragma unroll
    for (unsigned offset = {subgroup_lanes // 2}U; offset != 0U;
         offset /= 2U) {{
      value += __shfl_down_sync(
          subgroup_mask, value, offset, {subgroup_lanes});
    }}
    if (lane == 0U && value != 0.0) {{
      const unsigned center = slot / 3U;
      const unsigned coordinate = slot % 3U;
      atomicAdd(
          forces + static_cast<std::size_t>(shared.task.atom[center]) * 3U +
              coordinate,
          value);
    }}
  }}
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_subgroup_force_persistent(
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
  __shared__ GeneratedDpppSubgroupForceStorage
      subgroup_storage[{subgroup_count}];
  const unsigned subgroup = threadIdx.x / {subgroup_lanes}U;
  const unsigned lane = threadIdx.x % {subgroup_lanes}U;
  const unsigned subgroup_in_warp =
      (threadIdx.x & 31U) / {subgroup_lanes}U;
  const unsigned subgroup_mask =
      0x{subgroup_mask:08x}U << (subgroup_in_warp * {subgroup_lanes}U);
  GeneratedDpppSubgroupForceStorage& shared = subgroup_storage[subgroup];
  while (true) {{
    if (lane == 0U) shared.task_index = atomicAdd(task_head, 1U);
    __syncwarp(subgroup_mask);
    const std::uint32_t task_index = shared.task_index;
    if (task_index >= *task_count) return;
    generated_dppp_subgroup_force_task<Unrestricted>(
        tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
        atom_positions, screening_tolerance, schwarz_bounds, density, forces,
        static_cast<std::size_t>(*task_offset + task_index), shared, lane,
        subgroup_mask);
    __syncwarp(subgroup_mask);
  }}
}}

"""
        + ordinary_wrapper("generated_dppp_shell_class_force_rhf_kernel", "false")
        + ordinary_wrapper("generated_dppp_shell_class_force_uhf_kernel", "true")
        + persistent_wrapper(
            "generated_dppp_shell_class_force_rhf_persistent_kernel", "false"
        )
        + persistent_wrapper(
            "generated_dppp_shell_class_force_uhf_persistent_kernel", "true"
        )
    )

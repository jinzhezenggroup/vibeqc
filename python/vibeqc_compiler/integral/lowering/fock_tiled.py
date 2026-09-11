"""Emit packed and subgroup Fock consumers with shared component/index conventions."""

from __future__ import annotations

from ..fused_schedule import (
    FusedShellPlan,
)
from ..shell_spec import (
    ShellClassSpec,
)
from .common import _emitted_component_names, _generic_task_component_setup


def _emit_packed_fock_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit packed low-order Fock kernels using the shared value recurrence."""

    task_component_setup = _generic_task_component_setup(spec).replace(
        "shared.task", "task"
    )
    component_names = _emitted_component_names(spec)
    kernel_qualifier = (
        f"__maxnreg__({plan.schedule.maximum_registers})"
        if plan.schedule.maximum_registers
        else f"__launch_bounds__(32, {minimum_blocks_per_sm})"
    )
    return f"""struct GeneratedDpppPackedFockLaneStorage {{
  GeneratedDpppVec3 positions[4];
  GeneratedDpppPrimitiveGeometry primitive;
}};

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_packed_fock_lane(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    std::size_t task_index,
    GeneratedDpppPackedFockLaneStorage& storage) {{
  const GeneratedDpppShellTask& task = tasks[task_index];
#pragma unroll
  for (unsigned center = 0; center < 4U; ++center) {{
    storage.positions[center] = atom_positions[task.atom[center]];
  }}
  bool evaluate_components[kGeneratedDpppComponentCount]{{}};
  double angular_coefficients[kGeneratedDpppComponentCount]{{}};
#pragma unroll
  for (unsigned component = 0U;
       component < kGeneratedDpppComponentCount; ++component) {{
{task_component_setup}
    const std::size_t matrix_order =
        static_cast<std::size_t>(task.matrix_order);
    const bool retained_by_schwarz = schwarz_bounds == nullptr ||
        schwarz_bounds[
            task.density_offset +
            generated_dppp_matrix_index(i, j, matrix_order)] *
            schwarz_bounds[
                task.density_offset +
                generated_dppp_matrix_index(k, l, matrix_order)] >=
            screening_tolerance;
    evaluate_components[component] =
        unique_ket_component && retained_by_schwarz;
    angular_coefficients[component] =
        ao_coefficients[task.ao_coefficient_begin[0] + {component_names[0]}] *
        ao_coefficients[task.ao_coefficient_begin[1] + {component_names[1]}] *
        ao_coefficients[task.ao_coefficient_begin[2] + {component_names[2]}] *
        ao_coefficients[task.ao_coefficient_begin[3] + {component_names[3]}];
  }}
  double component_integrals[kGeneratedDpppComponentCount]{{}};
  const std::int64_t first_pair_begin =
      primitive_pair_offsets[task.shell_pair[0]];
  const std::int64_t first_pair_end =
      primitive_pair_offsets[task.shell_pair[0] + 1U];
  const std::int64_t second_pair_begin =
      primitive_pair_offsets[task.shell_pair[1]];
  const std::int64_t second_pair_end =
      primitive_pair_offsets[task.shell_pair[1] + 1U];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end; ++first_primitive) {{
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end; ++second_primitive) {{
      generated_dppp_make_primitive_geometry(
          primitive_pairs[first_primitive],
          primitive_pairs[second_primitive],
          (task.reversed_shell_pair_mask & 1U) != 0U,
          (task.reversed_shell_pair_mask & 2U) != 0U,
          storage.positions[0], storage.positions[1],
          storage.positions[2], storage.positions[3],
          storage.primitive);
#pragma unroll
      for (unsigned component = 0U;
           component < kGeneratedDpppComponentCount; ++component) {{
        if (!evaluate_components[component]) continue;
        component_integrals[component] +=
            angular_coefficients[component] *
            storage.primitive.primitive_coefficient *
            generated_dppp_component_value<false>(
                component, storage.primitive, nullptr);
      }}
    }}
  }}
#pragma unroll
  for (unsigned component = 0U;
       component < kGeneratedDpppComponentCount; ++component) {{
    const double component_integral = component_integrals[component];
    if (component_integral != 0.0) {{
{task_component_setup}
      generated_dppp_accumulate_fock<Unrestricted>(
          task, density, fock, i, j, k, l, component_integral);
    }}
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
  __shared__ GeneratedDpppPackedFockLaneStorage lane_storage[32];
  const std::size_t task_index =
      static_cast<std::size_t>(blockIdx.x) * 32U + threadIdx.x;
  if (task_index >= task_count) return;
  generated_dppp_packed_fock_lane<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      task_index, lane_storage[threadIdx.x]);
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
  __shared__ GeneratedDpppPackedFockLaneStorage lane_storage[32];
  const std::size_t task_index =
      static_cast<std::size_t>(blockIdx.x) * 32U + threadIdx.x;
  if (task_index >= task_count) return;
  generated_dppp_packed_fock_lane<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      task_index, lane_storage[threadIdx.x]);
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_packed_fock_persistent(
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
  __shared__ std::uint32_t task_base;
  __shared__ GeneratedDpppPackedFockLaneStorage lane_storage[32];
  while (true) {{
    if (threadIdx.x == 0U) task_base = atomicAdd(task_head, 32U);
    __syncthreads();
    if (task_base >= *task_count) return;
    const std::uint32_t task_index = task_base + threadIdx.x;
    if (task_index < *task_count) {{
      generated_dppp_packed_fock_lane<Unrestricted>(
          tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, fock,
          static_cast<std::size_t>(*task_offset + task_index),
          lane_storage[threadIdx.x]);
    }}
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
  generated_dppp_packed_fock_persistent<false>(
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
  generated_dppp_packed_fock_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      task_offset, task_count, task_head);
}}
"""


def _emit_subgroup_fock_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit value-only Fock workers sharing recurrence state per subgroup."""

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
    double* fock,
    std::size_t task_count) {{
  __shared__ GeneratedDpppSubgroupFockStorage
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
  generated_dppp_subgroup_fock_task<{unrestricted}>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
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
    double* fock,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  generated_dppp_subgroup_fock_persistent<{unrestricted}>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      task_offset, task_count, task_head);
}}
"""

    return (
        f"""

/** Value-path state private to one independently progressing subgroup. */
struct GeneratedDpppSubgroupFockStorage {{
  GeneratedDpppShellTask task;
  GeneratedDpppVec3 positions[4];
  GeneratedDpppPrimitiveGeometry primitive;
  double coulomb[kGeneratedDpppFockCoulombStateCount];
  // Screening and AO normalization are invariant across primitive quartets.
  // Retaining one coefficient per component in shared memory avoids carrying
  // a second large per-thread register array for high-component value paths.
  double angular_coefficients[kGeneratedDpppComponentCount];
  std::uint32_t task_index;
}};

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_subgroup_fock_task(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    std::size_t task_index,
    GeneratedDpppSubgroupFockStorage& shared,
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

  double component_integrals[{components_per_lane}]{{}};
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
    const bool retained_by_schwarz = component_lane &&
        unique_ket_component &&
        (schwarz_bounds == nullptr ||
         schwarz_bounds[
             shared.task.density_offset +
             generated_dppp_matrix_index(i, j, matrix_order)] *
             schwarz_bounds[
                 shared.task.density_offset +
                 generated_dppp_matrix_index(k, l, matrix_order)] >=
             screening_tolerance);
    if (component_lane) {{
      shared.angular_coefficients[candidate_component] = retained_by_schwarz
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
  }}

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
           state < kGeneratedDpppFockCoulombStateCount;
           state += {subgroup_lanes}U) {{
        shared.coulomb[state] = generated_dppp_coulomb(
            generated_dppp_coulomb_states[state], shared.primitive);
      }}
      __syncwarp(subgroup_mask);
#pragma unroll
      for (unsigned local_component = 0U;
           local_component < {components_per_lane}U; ++local_component) {{
        const unsigned component =
            lane + local_component * {subgroup_lanes}U;
        if (component >= kGeneratedDpppComponentCount) continue;
        const double angular_coefficient =
            shared.angular_coefficients[component];
        if (angular_coefficient == 0.0) continue;
        component_integrals[local_component] +=
            angular_coefficient *
            shared.primitive.primitive_coefficient *
            generated_dppp_component_value<true>(
                component, shared.primitive, shared.coulomb);
      }}
      __syncwarp(subgroup_mask);
    }}
  }}

#pragma unroll
  for (unsigned local_component = 0U;
       local_component < {components_per_lane}U; ++local_component) {{
    const double component_integral = component_integrals[local_component];
    if (component_integral == 0.0) continue;
    const unsigned component =
        lane + local_component * {subgroup_lanes}U;
{task_component_setup}
    generated_dppp_accumulate_fock<Unrestricted>(
        shared.task, density, fock, i, j, k, l, component_integral);
  }}
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_subgroup_fock_persistent(
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
  __shared__ GeneratedDpppSubgroupFockStorage
      subgroup_storage[{subgroup_count}];
  const unsigned subgroup = threadIdx.x / {subgroup_lanes}U;
  const unsigned lane = threadIdx.x % {subgroup_lanes}U;
  const unsigned subgroup_in_warp =
      (threadIdx.x & 31U) / {subgroup_lanes}U;
  const unsigned subgroup_mask =
      0x{subgroup_mask:08x}U << (subgroup_in_warp * {subgroup_lanes}U);
  GeneratedDpppSubgroupFockStorage& shared = subgroup_storage[subgroup];
  while (true) {{
    if (lane == 0U) shared.task_index = atomicAdd(task_head, 1U);
    __syncwarp(subgroup_mask);
    const std::uint32_t task_index = shared.task_index;
    if (task_index >= *task_count) return;
    generated_dppp_subgroup_fock_task<Unrestricted>(
        tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
        atom_positions, screening_tolerance, schwarz_bounds, density, fock,
        static_cast<std::size_t>(*task_offset + task_index), shared, lane,
        subgroup_mask);
    __syncwarp(subgroup_mask);
  }}
}}
"""
        + ordinary_wrapper("generated_dppp_shell_class_fock_rhf_kernel", "false")
        + ordinary_wrapper("generated_dppp_shell_class_fock_uhf_kernel", "true")
        + persistent_wrapper(
            "generated_dppp_shell_class_fock_rhf_persistent_kernel", "false"
        )
        + persistent_wrapper(
            "generated_dppp_shell_class_fock_uhf_persistent_kernel", "true"
        )
    )

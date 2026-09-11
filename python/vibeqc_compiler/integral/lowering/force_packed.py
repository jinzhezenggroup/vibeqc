"""Emit packed and scalar-thread weighted force consumers using common derivative algebra."""

from __future__ import annotations

from ..cuda import CudaEmitter
from ..fused_schedule import (
    FusedShellPlan,
)
from ..shell_class import (
    build_weighted_shell_contraction_kernel,
)
from ..shell_spec import (
    AXES,
    ShellClassSpec,
)
from .algebra import _packed_force_integral
from .common import _emitted_component_names, _generic_task_component_setup


def _emit_packed_force_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit one-independent-task-per-lane force kernels for small shells.

    Packed lowering intentionally reuses the generic component recurrence and
    exact density permutation helpers emitted above it.  Only task ownership
    changes: a 32-thread block processes up to 32 unrelated shell quartets.
    """

    selected_integral = _packed_force_integral(spec, plan.kernel.integral)
    independent_centers = selected_integral.independent_derivative_centers
    recovered_centers = selected_integral.recovered_derivative_centers
    if len(independent_centers) != 3 or len(recovered_centers) != 1:
        raise ValueError(
            "packed force lowering currently requires three independent and "
            "one recovered derivative center"
        )
    independent_center_table = ", ".join(f"{center}U" for center in independent_centers)
    independent_atomic_code = f"""  constexpr unsigned derivative_centers[3] = {{
      {independent_center_table}}};
#pragma unroll
  for (unsigned slot = 0; slot < 9U; ++slot) {{
    const double value = task_force[slot];
    if (value == 0.0) continue;
    const unsigned center = derivative_centers[slot / 3U];
    const unsigned coordinate = slot % 3U;
    atomicAdd(
        forces + static_cast<std::size_t>(task.atom[center]) * 3U + coordinate,
        value);
  }}"""
    recovered_atomic_blocks = []
    for recovered_index, center in enumerate(recovered_centers):
        name = "recovered_force" if recovered_index else "fourth_force"
        terms = " - ".join(
            f"task_force[{slot * 3} + coordinate]"
            for slot in range(len(independent_centers))
        )
        recovered_atomic_blocks.append(
            f"""#pragma unroll
  for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
    const double {name} = -{terms};
    if ({name} != 0.0) {{
      atomicAdd(
          forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
              coordinate,
          {name});
    }}
  }}"""
        )
    recovered_atomic_code = "\n".join(recovered_atomic_blocks)

    task_component_setup = _generic_task_component_setup(spec).replace(
        "shared.task", "task"
    )
    component_names = _emitted_component_names(spec)
    kernel_qualifier = (
        f"__maxnreg__({plan.schedule.maximum_registers})"
        if plan.schedule.maximum_registers
        else f"__launch_bounds__(32, {minimum_blocks_per_sm})"
    )
    return f"""struct GeneratedDpppPackedForceLaneStorage {{
  GeneratedDpppVec3 positions[4];
  GeneratedDpppPackedForceGeometry primitive;
}};

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_packed_force_lane(
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
    GeneratedDpppPackedForceLaneStorage& storage) {{
  const GeneratedDpppShellTask& task = tasks[task_index];
#pragma unroll
  for (unsigned center = 0; center < 4U; ++center) {{
    storage.positions[center] = atom_positions[task.atom[center]];
  }}
  double component_weights[kGeneratedDpppComponentCount]{{}};
  bool any_component = false;
#pragma unroll
  for (unsigned component = 0U;
       component < kGeneratedDpppComponentCount; ++component) {{
{task_component_setup}
    const std::size_t matrix_order =
        static_cast<std::size_t>(task.matrix_order);
    const double schwarz_product = schwarz_bounds == nullptr
        ? 0.0
        : schwarz_bounds[
            task.density_offset +
            generated_dppp_matrix_index(i, j, matrix_order)] *
          schwarz_bounds[
                task.density_offset +
                generated_dppp_matrix_index(k, l, matrix_order)];
    const bool retained_by_schwarz = schwarz_bounds == nullptr ||
        schwarz_product >= screening_tolerance;
    // Force and Fock must retain the same AO quartet.  The density coefficient
    // weights the derivative but must not introduce a second, non-variational
    // screening decision that is absent from the energy path.
    const double density_coefficient =
        unique_ket_component && retained_by_schwarz
        ? generated_dppp_density_coefficient<Unrestricted>(
              task, i, j, k, l, density)
        : 0.0;
    const double angular_coefficient =
        ao_coefficients[task.ao_coefficient_begin[0] + {component_names[0]}] *
        ao_coefficients[task.ao_coefficient_begin[1] + {component_names[1]}] *
        ao_coefficients[task.ao_coefficient_begin[2] + {component_names[2]}] *
        ao_coefficients[task.ao_coefficient_begin[3] + {component_names[3]}];
    component_weights[component] =
        density_coefficient * angular_coefficient;
    any_component = any_component || density_coefficient != 0.0;
  }}
  if (!any_component) return;
  double task_force[9]{{}};
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
      generated_dppp_make_packed_force_geometry(
          primitive_pairs[first_primitive],
          primitive_pairs[second_primitive],
          (task.reversed_shell_pair_mask & 1U) != 0U,
          (task.reversed_shell_pair_mask & 2U) != 0U,
          storage.positions[0], storage.positions[1],
          storage.positions[2], storage.positions[3],
          storage.primitive);
      double primitive_gradient[3][3];
      generated_dppp_weighted_component_gradient(
          storage.primitive, component_weights, primitive_gradient);
      const double scale = -storage.primitive.primitive_coefficient;
#pragma unroll
      for (unsigned center = 0; center < 3U; ++center) {{
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
          task_force[center * 3U + coordinate] +=
              scale * primitive_gradient[center][coordinate];
        }}
      }}
    }}
  }}
{independent_atomic_code}
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
  __shared__ GeneratedDpppPackedForceLaneStorage lane_storage[32];
  const std::size_t task_index =
      static_cast<std::size_t>(blockIdx.x) * 32U + threadIdx.x;
  if (task_index >= task_count) return;
  generated_dppp_packed_force_lane<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_index, lane_storage[threadIdx.x]);
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
  __shared__ GeneratedDpppPackedForceLaneStorage lane_storage[32];
  const std::size_t task_index =
      static_cast<std::size_t>(blockIdx.x) * 32U + threadIdx.x;
  if (task_index >= task_count) return;
  generated_dppp_packed_force_lane<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_index, lane_storage[threadIdx.x]);
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_packed_force_persistent(
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
  __shared__ GeneratedDpppPackedForceLaneStorage lane_storage[32];
  while (true) {{
    if (threadIdx.x == 0U) task_base = atomicAdd(task_head, 32U);
    __syncthreads();
    if (task_base >= *task_count) return;
    const std::uint32_t task_index = task_base + threadIdx.x;
    if (task_index < *task_count) {{
      generated_dppp_packed_force_lane<Unrestricted>(
          tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density, forces,
          static_cast<std::size_t>(*task_offset + task_index),
          lane_storage[threadIdx.x]);
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
  generated_dppp_packed_force_persistent<false>(
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
  generated_dppp_packed_force_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}}
"""


def _emit_scalar_thread_force_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit a spill-resistant one-complete-task-per-thread force worker.

    The generic packed prototype passes component and gradient arrays through
    one noinline shell-wide function.  A scalar task owns one complete shell
    quartet, so this lowering instead names every long-lived value explicitly
    and emits one coordinate scope at a time.  Geometry still resides in
    lane-private shared storage so the thread does not materialize the large
    primitive structure on its stack.  The component and derivative extents
    come from ``spec`` and ``IntegralIR``; no shell-name dispatch is required.
    """

    if plan.schedule.block_threads != 32:
        raise ValueError("scalar thread tasks currently use one CUDA warp")

    selected_integral = _packed_force_integral(spec, plan.kernel.integral)
    independent_centers = selected_integral.independent_derivative_centers
    recovered_centers = selected_integral.recovered_derivative_centers
    decay_gradient_rows = 4 if 3 in independent_centers else 3
    if len(independent_centers) != 3 or len(recovered_centers) != 1:
        raise ValueError(
            "scalar thread force lowering currently requires three independent "
            "and one recovered derivative center"
        )

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
            f"""  storage.component_weights[{component}] = 0.0;
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
      storage.component_weights[{component}] =
          density_coefficient * angular_coefficient;
    }}
  }}"""
        )

    kernel = build_weighted_shell_contraction_kernel(
        spec,
        integral=selected_integral,
    )
    variable_code = {
        "inverse_two_p": "storage.primitive.inverse_two_p",
        "inverse_two_q": "storage.primitive.inverse_two_q",
        "rho": "storage.primitive.rho",
        "first_product_scale": "storage.primitive.product_scales[0]",
        "second_product_scale": "storage.primitive.product_scales[1]",
        "third_product_scale": "storage.primitive.product_scales[2]",
        "fourth_product_scale": "1.0 - storage.primitive.product_scales[2]",
        "prefactor": "storage.primitive.prefactor",
    }
    for axis_index, axis in enumerate(AXES):
        variable_code[f"difference_{axis}"] = (
            f"storage.primitive.difference[{axis_index}]"
        )
        for center, prefix in enumerate(("pa", "pb", "qc", "qd")):
            variable_code[f"{prefix}_{axis}"] = (
                f"storage.primitive.pair_shifts[{center}][{axis_index}]"
            )
        for center_index, center in enumerate(("first", "second", "third", "fourth")):
            if center_index >= decay_gradient_rows:
                continue
            variable_code[f"decay_{center}_{axis}"] = (
                f"storage.primitive.decay_gradients[{center_index}][{axis_index}]"
            )
    for order in range(spec.maximum_force_coulomb_order + 1):
        variable_code[f"boys_{order}"] = f"storage.primitive.boys[{order}]"
    for component in range(spec.component_count):
        variable_code[f"component_weight_{component}"] = (
            f"storage.component_weights[{component}]"
        )

    # Bound CSE to one force root at a time.  The first component-major pilot
    # shared all nine roots, but PTXAS still needed more than 1 KiB of spill
    # storage because values common to later centers stayed live across each
    # coordinate result.  A root is therefore generated and consumed in its
    # own lexical region.  Once the zero-spill floor is established, adjacent
    # roots can be fused selectively where the resource gate shows headroom.
    component_scopes = []
    for component in range(spec.component_count):
        kernel = build_weighted_shell_contraction_kernel(
            spec,
            (component,),
            integral=selected_integral,
        )
        roots = [
            kernel.gradients[center][coordinate]
            for center in selected_integral.independent_derivative_centers
            for coordinate in range(3)
        ]
        statements = []
        for slot, root in enumerate(roots):
            emitter = CudaEmitter(kernel.graph, variable_code)
            emitter.emit((root,))
            statements.append("      {")
            statements.extend("      " + line for line in emitter.lines)
            statements.append(
                f"        force_{slot} += primitive_scale * {emitter.reference(root)};"
            )
            statements.append("      }")
        component_scopes.append("\n".join(statements))

    force_declarations = "\n".join(
        f"  double force_{slot} = storage.task_force[{slot}];" for slot in range(9)
    )
    force_stores = "\n".join(
        f"  storage.task_force[{slot}] = force_{slot};" for slot in range(9)
    )
    # Three components are the largest stable helper region that compiles with
    # no stack or local-memory spills on sm_120.  ``component_tile`` still
    # describes schedule coverage; this smaller value is a lowering detail.
    component_group_size = 3
    primitive_helpers = []
    primitive_calls = []
    for group_begin in range(0, spec.component_count, component_group_size):
        group_end = min(
            group_begin + component_group_size,
            spec.component_count,
        )
        helper_name = (
            "generated_dppp_scalar_thread_accumulate_components_"
            f"{group_begin}_{group_end}"
        )
        group_component_code = "\n".join(component_scopes[group_begin:group_end])
        primitive_helpers.append(
            f"""__device__ __noinline__ void {helper_name}(
    double primitive_scale,
    GeneratedDpppScalarThreadStorage& storage) {{
{force_declarations}
{group_component_code}
{force_stores}
}}"""
        )
        primitive_calls.append(f"      {helper_name}(primitive_scale, storage);")
    independent_atomics = []
    for slot, center in enumerate(independent_centers):
        for coordinate in range(3):
            force_slot = slot * 3 + coordinate
            independent_atomics.append(
                f"""  if (storage.task_force[{force_slot}] != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
            {coordinate}U,
        storage.task_force[{force_slot}]);
  }}"""
            )
    recovered_atomics = []
    for recovered_index, center in enumerate(recovered_centers):
        name = "recovered_force" if recovered_index else "fourth_force"
        terms = " - ".join(
            f"storage.task_force[{slot * 3} + coordinate]"
            for slot in range(len(independent_centers))
        )
        recovered_atomics.append(
            f"""#pragma unroll
  for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
    const double {name} = -{terms};
    if ({name} != 0.0) {{
      atomicAdd(
          context.forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
              coordinate,
          {name});
    }}
  }}"""
        )

    weight_code = "\n".join(weight_blocks)
    primitive_helper_code = "\n\n".join(primitive_helpers)
    primitive_call_code = "\n".join(primitive_calls)
    independent_atomic_code = "\n".join(independent_atomics)
    fourth_atomic_code = "\n".join(recovered_atomics)
    kernel_qualifier = f"__launch_bounds__(32, {minimum_blocks_per_sm})"
    return f"""struct GeneratedDpppScalarThreadStorage {{
  GeneratedDpppVec3 positions[4];
  GeneratedDpppPrimitiveGeometry primitive;
  double component_weights[kGeneratedDpppComponentCount];
  double task_force[9];
}};

/** Kernel-wide immutable arguments shared by every scalar task lane. */
struct GeneratedDpppScalarThreadContext {{
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
__device__ __noinline__ void generated_dppp_scalar_thread_fill_weights(
    const GeneratedDpppShellTask& task,
    const GeneratedDpppScalarThreadContext& context,
    GeneratedDpppScalarThreadStorage& storage) {{
{weight_code}
}}

/**
 * Accumulate one primitive quartet behind a device-call register boundary.
 *
 * Keeping this recurrence separate from task decoding and the persistent
 * queue lets PTXAS reuse scalar temporaries without simultaneously carrying
 * every queue pointer, shell index, and cross-primitive force accumulator.
 */
{primitive_helper_code}

template <bool Unrestricted>
__device__ __noinline__ void generated_dppp_scalar_thread_force_task(
    const GeneratedDpppScalarThreadContext& context,
    std::size_t task_index,
    GeneratedDpppScalarThreadStorage& storage) {{
  const GeneratedDpppShellTask& task = context.tasks[task_index];
#pragma unroll
  for (unsigned center = 0; center < 4U; ++center) {{
    storage.positions[center] = context.atom_positions[task.atom[center]];
  }}
  // Keep the 27 density weights behind a device-call boundary.  Otherwise
  // PTXAS scalarizes the constant-index shared writes and carries all weights
  // through the primitive recurrence as long-lived registers.
  generated_dppp_scalar_thread_fill_weights<Unrestricted>(
      task, context, storage);
#pragma unroll
  for (unsigned slot = 0; slot < 9U; ++slot) {{
    storage.task_force[slot] = 0.0;
  }}
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
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end; ++second_primitive) {{
      generated_dppp_make_primitive_geometry(
          context.primitive_pairs[first_primitive],
          context.primitive_pairs[second_primitive],
          (task.reversed_shell_pair_mask & 1U) != 0U,
          (task.reversed_shell_pair_mask & 2U) != 0U,
          storage.positions[0], storage.positions[1],
          storage.positions[2], storage.positions[3], storage.primitive);
      const double primitive_scale =
          -storage.primitive.primitive_coefficient;
{primitive_call_code}
    }}
  }}
{independent_atomic_code}
{fourth_atomic_code}
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_scalar_thread_force_persistent(
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
  __shared__ GeneratedDpppScalarThreadContext context;
  __shared__ GeneratedDpppScalarThreadStorage lane_storage[32];
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
      generated_dppp_scalar_thread_force_task<Unrestricted>(
          context,
          static_cast<std::size_t>(*task_offset + task_index),
          lane_storage[threadIdx.x]);
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
  generated_dppp_scalar_thread_force_persistent<false>(
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
  generated_dppp_scalar_thread_force_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}}
"""

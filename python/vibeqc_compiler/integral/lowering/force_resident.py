"""Emit the measured resident-bra PPPS fixed-root force schedule.

The historical specialization remains explicit; eligibility and general fallback
dispatch belong to the compatibility adapter, not this arithmetic body."""

from __future__ import annotations

from ..ir import IntegralIR, build_integral_ir
from ..rys import (
    emit_rys3_roots_cuda,
    emit_rys_force_root_body_cuda,
)
from ..shell_spec import (
    FUSED_SHELL_SPEC_BY_NAME,
)
from .common import _emitted_component_names, _generic_task_component_setup


def _emit_ppps_resident_bra_rys3_force_consumer_cuda(
    *, include_rys3_roots: bool = True, integral: IntegralIR | None = None
) -> str:
    """Emit the isolated canonical ``(p p|p s)`` resident-bra prototype.

    A resident descriptor maps one block to one ``p p`` shell pair and a
    contiguous range of existing ``GeneratedDpppShellTask`` ket records.  The
    descriptor itself is deliberately small and remains ABI-compatible with
    the host-side definition in ``generated_shell_task.hpp``; the generated
    type is checked by the production C wrapper before it is passed to CUDA.

    Each lane walks a strided subset of the grouped ``p s`` ket tasks.  The
    block first stages the common bra primitive-pair records in shared
    memory, then each lane evaluates its ket primitive pairs with 27
    explicitly named density/Cartesian weights.  This keeps the bra cache
    resident even when one descriptor covers more than 256 ket tasks.
    Rys roots remain in shared SoA storage to avoid an addressable six-double
    local array; the straight-line TRR/HRR body immediately folds each
    component into nine scalar force accumulators and recovers center D by
    translation invariance.  When ``integral`` is supplied, its explicit
    derivative centers and recovery relation drive every force slot and
    atomic destination in this topology.
    """

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    selected_integral = integral or build_integral_ir(spec, recurrence="rys3")
    if selected_integral.spec != spec:
        raise ValueError("resident ppps integral spec does not match ppps")
    if selected_integral.recurrence != "rys3":
        raise ValueError("resident ppps lowering requires an rys3 integral")
    independent_centers = selected_integral.independent_derivative_centers
    recovered_centers = selected_integral.recovered_derivative_centers
    if len(independent_centers) != 3 or len(recovered_centers) != 1:
        raise ValueError(
            "resident ppps lowering currently requires three independent "
            "and one recovered derivative center"
        )
    # The canonical resident topology stages a ``p p`` bra and streams a
    # ``p s`` ket.  Center D is normally recovered, so its pair displacement
    # and exponent are only materialized when an explicit IR makes D
    # independent (for example, when center B is recovered instead).
    needs_fourth_derivative = 3 in independent_centers
    fourth_position_setup = (
        "  const GeneratedDpppVec3 fourth = context.atom_positions[task.atom[3]];\n"
        "  const double cdx = fourth.x - third.x;\n"
        "  const double cdy = fourth.y - third.y;\n"
        "  const double cdz = fourth.z - third.z;"
        if needs_fourth_derivative
        else ""
    )
    fourth_scale_setup = (
        "      const double fourth_product_scale = second_pair_reversed\n"
        "          ? second_pair.first_product_scale : "
        "second_pair.second_product_scale;\n"
        "      const double delta2 = 2.0 * q * fourth_product_scale;"
        if needs_fourth_derivative
        else ""
    )
    task_component_setup = _generic_task_component_setup(spec).replace(
        "shared.task", "task"
    )
    component_names = _emitted_component_names(spec)
    component_weight_names = tuple(
        f"component_weight_{component}" for component in range(spec.component_count)
    )
    weight_blocks: list[str] = []
    for component in range(spec.component_count):
        setup = "\n".join(f"    {line}" for line in task_component_setup.splitlines())
        weight_blocks.append(
            f"""  {component_weight_names[component]} = 0.0;
  {{
    constexpr unsigned component = {component}U;
{setup}
    const std::size_t matrix_order =
        static_cast<std::size_t>(task.matrix_order);
    const double schwarz_product = context.schwarz_bounds == nullptr
        ? 0.0
        : context.schwarz_bounds[
            task.density_offset +
            generated_dppp_matrix_index(i, j, matrix_order)] *
          context.schwarz_bounds[
            task.density_offset +
            generated_dppp_matrix_index(k, l, matrix_order)];
    const bool retained_by_schwarz = context.schwarz_bounds == nullptr ||
        schwarz_product >= context.screening_tolerance;
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
      {component_weight_names[component]} =
          density_coefficient * angular_coefficient;
      any_component = any_component || density_coefficient != 0.0;
    }}
  }}"""
        )

    root_body = emit_rys_force_root_body_cuda(
        spec,
        component_weight_expression="component_weight_{component}",
        integral=selected_integral,
    )
    weight_code = "\n".join(weight_blocks)
    independent_atomics = []
    # Contributions for independent ket centers vary by lane and can be
    # committed immediately.  Independent bra centers are reduced uniformly
    # below because every lane shares the same staged bra pair.
    for slot, center in enumerate(independent_centers):
        if center < 2:
            continue
        for coordinate in range(3):
            force_slot = slot * 3 + coordinate
            independent_atomics.append(
                f"""  if (force_{force_slot} != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
            {coordinate}U,
        force_{force_slot});
  }}"""
            )
    fourth_atomics = []
    for recovered_index, center in enumerate(recovered_centers):
        name = "fourth_force" if recovered_index == 0 else f"recovered_force_{center}"
        for coordinate in range(3):
            slots = [slot * 3 + coordinate for slot in range(len(independent_centers))]
            terms = " - ".join(f"force_{slot}" for slot in slots)
            fourth_atomics.append(
                f"""  const double {name}_{coordinate} = -{terms};
  if ({name}_{coordinate} != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
            {coordinate}U,
        {name}_{coordinate});
  }}"""
            )
    force_declarations = "\n".join(
        f"  double force_{slot} = 0.0;" for slot in range(3 * len(independent_centers))
    )
    bra_force_slots = tuple(
        slot for slot, center in enumerate(independent_centers) if center < 2
    )
    bra_force_reductions = "\n".join(
        f"    force_{slot * 3 + coordinate} += "
        f"__shfl_down_sync(0xffffffffU, force_{slot * 3 + coordinate}, delta);"
        for slot in bra_force_slots
        for coordinate in range(3)
    )
    bra_force_atomics = "\n".join(
        f"""    if (force_{slot} != 0.0) {{
      atomicAdd(
          context.forces +
              static_cast<std::size_t>(
                  context.ket_tasks[resident.ket_begin].atom[{center}]) * 3U +
              {slot % 3}U,
          force_{slot});
    }}"""
        for bra_slot in bra_force_slots
        for center in (independent_centers[bra_slot],)
        for slot in range(bra_slot * 3, bra_slot * 3 + 3)
    )
    independent_atomic_code = "\n".join(independent_atomics)
    fourth_atomic_code = "\n".join(fourth_atomics)

    weight_declarations = "\n".join(
        f"  double {name} = 0.0;" for name in component_weight_names
    )

    roots = emit_rys3_roots_cuda() if include_rys3_roots else ""
    return (
        roots
        + f"""/*
 * Canonical ppps (1110) resident-bra force worker.
 *
 * The descriptor remains profile-scoped in generated CUDA and is checked
 * against the host ABI by the production C wrapper.  The runtime may keep
 * using the ordinary shell-task route as a fallback.
 */
struct GeneratedPppsResidentTask {{
  std::uint32_t bra_pair;
  std::uint32_t ket_begin;
  std::uint32_t ket_count;
}};

// Maximum launch width and shared Rys-root pitch. The production wrapper may
// launch 32, 64, 128, or 256 scalar task lanes for same-binary CTA sweeps.
constexpr unsigned kGeneratedPppsResidentBlockThreads = 256U;
constexpr unsigned kGeneratedPppsResidentMaximumBraPrimitivePairs = 64U;

/** Pointers and immutable screening inputs shared by all resident lanes. */
struct GeneratedPppsResidentContext {{
  const GeneratedDpppShellTask* ket_tasks;
  const GeneratedDpppPrimitivePairData* primitive_pairs;
  const std::int64_t* primitive_pair_offsets;
  const double* ao_coefficients;
  const GeneratedDpppVec3* atom_positions;
  double screening_tolerance;
  const double* schwarz_bounds;
  const double* density;
  double* forces;
}};

/**
 * Evaluate one ket lane against every staged bra primitive pair.
 *
 * The recurrence body is straight-line code generated from the unique Rys
 * state DAG.  Each component is contracted at its last-use point, so only
 * the nine force scalars and the lane's 27 weights cross primitive loops.
 */
template <bool Unrestricted>
__device__ __forceinline__ void generated_ppps_resident_force_task(
    const GeneratedPppsResidentContext& context,
    const GeneratedPppsResidentTask& resident,
    const GeneratedDpppPrimitivePairData* resident_bra_pairs,
    unsigned resident_bra_pair_count,
    double (&roots_weights)[6][kGeneratedPppsResidentBlockThreads]) {{
  const unsigned lane = threadIdx.x;
  // Advance in uniform CTA-sized rounds. Ragged tail lanes still execute the
  // warp reduction with zero force, keeping the full-warp shuffle mask valid.
  for (unsigned ket_base = 0U; ket_base < resident.ket_count;
       ket_base += blockDim.x) {{
    const unsigned local_ket = ket_base + lane;
{force_declarations}
    if (local_ket < resident.ket_count) {{
    const std::size_t task_index =
        static_cast<std::size_t>(resident.ket_begin) + local_ket;
    const GeneratedDpppShellTask& task = context.ket_tasks[task_index];
    if (task.shell_pair[0] == resident.bra_pair) {{

{weight_declarations}
  bool any_component = false;
{weight_code}
  if (any_component) {{
  const GeneratedDpppVec3 first = context.atom_positions[task.atom[0]];
  const GeneratedDpppVec3 second = context.atom_positions[task.atom[1]];
  const GeneratedDpppVec3 third = context.atom_positions[task.atom[2]];
{fourth_position_setup}
  const bool first_pair_reversed =
      (task.reversed_shell_pair_mask & 1U) != 0U;
  const bool second_pair_reversed =
      (task.reversed_shell_pair_mask & 2U) != 0U;

  const std::int64_t second_pair_begin =
      context.primitive_pair_offsets[task.shell_pair[1]];
  const std::int64_t second_pair_end =
      context.primitive_pair_offsets[task.shell_pair[1] + 1U];
  for (unsigned bra_primitive = 0U;
       bra_primitive < resident_bra_pair_count; ++bra_primitive) {{
    const GeneratedDpppPrimitivePairData first_pair =
        resident_bra_pairs[bra_primitive];
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
      const double gamma2 = 2.0 * q * third_product_scale;
{fourth_scale_setup}
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
      generated_ppps_rys3_roots(
          rho * (dx * dx + dy * dy + dz * dz),
          &roots_weights[0][lane],
          kGeneratedPppsResidentBlockThreads);
      const double primitive_prefactor =
          -34.986836655249725 * first_pair.weighted_coefficient *
          second_pair.weighted_coefficient / (p * q * sqrt(p + q));
#pragma unroll
      for (unsigned root_index = 0U; root_index < 3U; ++root_index) {{
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
  }}
{independent_atomic_code}
{fourth_atomic_code}
    }}
    }}
    // Reduce one round's common-bra contribution before the next round can
    // overwrite the task-local force scalars.
    for (unsigned delta = 16U; delta != 0U; delta >>= 1U) {{
{bra_force_reductions}
    }}
    if ((lane & 31U) == 0U) {{
{bra_force_atomics}
    }}
  }}
}}

/**
 * One block owns one pp bra pair and a contiguous p-s ket-task chunk.
 *
 * All lanes participate in the cache barrier, including tail lanes whose
 * ket-task slot is outside ``ket_count``.  This is required because the
 * resident descriptor is intentionally ragged at the end of a chunk.  Tail
 * lanes still reach the cache barrier above, while each lane then walks all
 * of its assigned ket records in 256-thread strides.
 */
template <bool Unrestricted>
__device__ __forceinline__ void generated_ppps_resident_bra_worker(
    const GeneratedPppsResidentTask* resident_tasks,
    const GeneratedDpppShellTask* ket_tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t resident_task_count) {{
  __shared__ GeneratedPppsResidentContext context;
  __shared__ GeneratedDpppPrimitivePairData resident_bra_pairs[
      kGeneratedPppsResidentMaximumBraPrimitivePairs];
  __shared__ double roots_weights[6][kGeneratedPppsResidentBlockThreads];
  __shared__ GeneratedPppsResidentTask resident;
  if (threadIdx.x == 0U) {{
    context.ket_tasks = ket_tasks;
    context.primitive_pairs = primitive_pairs;
    context.primitive_pair_offsets = primitive_pair_offsets;
    context.ao_coefficients = ao_coefficients;
    context.atom_positions = atom_positions;
    context.screening_tolerance = screening_tolerance;
    context.schwarz_bounds = schwarz_bounds;
    context.density = density;
    context.forces = forces;
    if (blockIdx.x < resident_task_count) {{
      resident = resident_tasks[blockIdx.x];
    }}
  }}
  __syncthreads();
  if (blockIdx.x >= resident_task_count) return;
  // The runtime deliberately preserves one descriptor slot per shell-pair
  // ordinal to avoid a host readback and a second device compaction.  Most
  // ordinals are holes, so reject them before reading primitive offsets or
  // touching the shared bra cache.
  if (resident.ket_count == 0U) return;
  const std::int64_t bra_pair_begin =
      primitive_pair_offsets[resident.bra_pair];
  const std::int64_t bra_pair_end =
      primitive_pair_offsets[resident.bra_pair + 1U];
  const std::int64_t bra_pair_count = bra_pair_end - bra_pair_begin;
  if (bra_pair_count <= 0 ||
      bra_pair_count > static_cast<std::int64_t>(
          kGeneratedPppsResidentMaximumBraPrimitivePairs)) return;
  for (std::int64_t primitive = threadIdx.x;
       primitive < bra_pair_count; primitive +=
           blockDim.x) {{
    resident_bra_pairs[primitive] =
        primitive_pairs[bra_pair_begin + primitive];
  }}
  __syncthreads();
  generated_ppps_resident_force_task<Unrestricted>(
      context, resident, resident_bra_pairs,
      static_cast<unsigned>(bra_pair_count), roots_weights);
}}

extern "C" __global__ __launch_bounds__(
    kGeneratedPppsResidentBlockThreads, 1)
void generated_ppps_resident_bra_force_rhf_kernel(
    const GeneratedPppsResidentTask* resident_tasks,
    const GeneratedDpppShellTask* ket_tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t resident_task_count) {{
  generated_ppps_resident_bra_worker<false>(
      resident_tasks, ket_tasks, primitive_pairs, primitive_pair_offsets,
      ao_coefficients, atom_positions, screening_tolerance, schwarz_bounds,
      density, forces, resident_task_count);
}}

extern "C" __global__ __launch_bounds__(
    kGeneratedPppsResidentBlockThreads, 1)
void generated_ppps_resident_bra_force_uhf_kernel(
    const GeneratedPppsResidentTask* resident_tasks,
    const GeneratedDpppShellTask* ket_tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t resident_task_count) {{
  generated_ppps_resident_bra_worker<true>(
      resident_tasks, ket_tasks, primitive_pairs, primitive_pair_offsets,
      ao_coefficients, atom_positions, screening_tolerance, schwarz_bounds,
      density, forces, resident_task_count);
}}
"""
    )

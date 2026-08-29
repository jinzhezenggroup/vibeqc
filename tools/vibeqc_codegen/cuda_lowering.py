"""Lower shell-class integral plans to deterministic CUDA source.

The implementation grew from the first cooperative ``dppp`` AOT experiment,
but now owns the shared packed, scalar-thread, subgroup, component-lane, tiled,
and fixed-root lowering families used throughout the canonical shell catalog.
Historical DPPP and resident-PPPS entry points are re-exported by the narrow
``dppp_dispatch`` compatibility adapter; generic compiler users depend on this
backend-named module through ``cuda_emitter``.

The emitted kernels perform density, primitive, and force contraction directly,
so generated hot loops contain ordinary scalar CUDA arithmetic rather than
runtime differentiation objects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .cuda import CudaEmitter
from .cuda_schedule import (
    AlgebraForm,
    PairOrientation,
    PairStorage,
    ScheduleIR,
    ScheduleKind,
)
from .expr import Expr, PowerLowering
from .fused_schedule import (
    CoulombState,
    FusedShellPlan,
    build_fused_shell_plan,
    evaluate_fused_shell_component,
)
from .ir import KernelConsumer
from .rys import (
    build_rys_force_program,
    emit_ppps_rys3_root_body_cuda,
    emit_rys2_roots_cuda,
    emit_rys3_roots_cuda,
    emit_rys4_roots_cuda,
    emit_rys5_roots_cuda,
    emit_rys_force_root_body_cuda,
)
from .shell_class import (
    build_packed_force_geometry_algebra,
    build_weighted_shell_contraction_kernel,
)
from .shell_spec import (
    AXES,
    DPPP_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    ShellClassSpec,
    cartesian_components,
)

DpppComponent = tuple[str, str, str, str]
_AXIS_INDEX = {axis: index for index, axis in enumerate(AXES)}
_COMPONENT_COUNT = DPPP_SPEC.component_count


@dataclass(frozen=True, slots=True)
class DpppFusedPlan:
    """Static schedule and lookup tables for one fused ``dppp`` kernel."""

    components: tuple[DpppComponent, ...]
    coulomb_states: tuple[CoulombState, ...]
    coulomb_indices: tuple[int, ...]
    block_threads: int

    @property
    def warp_count(self) -> int:
        """Return the number of full warps used by one generated block."""

        return self.block_threads // 32


def dppp_components() -> tuple[DpppComponent, ...]:
    """Return all Cartesian components in production CCA ordering."""

    return DPPP_SPEC.components


def build_dppp_fused_plan() -> DpppFusedPlan:
    """Build the deterministic component and shared-Coulomb schedule.

    Cartesian derivative states are ordered by total degree and then by
    ``x/y/z`` degree. The dense 7x7x7 lookup table makes the generated hot
    loop a few integer operations plus one shared-memory load; invalid states
    retain ``-1`` so generator tests can audit the complete domain.
    """

    generic = build_fused_shell_plan(DPPP_SPEC)
    if len(generic.components) != _COMPONENT_COUNT:
        raise RuntimeError("dppp component schedule has an unexpected size")
    if len(generic.coulomb_states) != 84:
        raise RuntimeError("order-six Cartesian Coulomb schedule must have 84 states")
    return DpppFusedPlan(
        components=generic.components,
        coulomb_states=generic.coulomb_states,
        coulomb_indices=generic.coulomb_indices,
        block_threads=generic.block_threads,
    )


def evaluate_dppp_fused_component(
    component: DpppComponent,
    variables: Mapping[str, float],
) -> tuple[tuple[float, float, float], ...]:
    """Evaluate one component using the fused kernel's recurrence schedule.

    This host-side oracle deliberately mirrors the emitted CUDA loops rather
    than calling the symbolic expression graph. Comparing both independent
    lowerings for all 162 components catches table ordering, sign, and center
    mapping mistakes before a generated kernel is considered for production.
    """

    return evaluate_fused_shell_component(DPPP_SPEC, component, variables)


def _format_cuda_array(values: Sequence[int], columns: int = 12) -> str:
    """Format a deterministic wrapped CUDA initializer."""

    rows = []
    for start in range(0, len(values), columns):
        rows.append(
            "    " + ", ".join(str(value) for value in values[start : start + columns])
        )
    return ",\n".join(rows)


def _shell_letter(angular_momentum: int) -> str:
    """Return conventional shell notation for the supported AOT range."""

    labels = "spdfgh"
    if not 0 <= angular_momentum < len(labels):
        raise ValueError("fused CUDA emitter supports shell labels through h")
    return labels[angular_momentum]


def _component_names(spec: ShellClassSpec) -> tuple[str, str, str, str]:
    """Create readable, unique CUDA names for decoded center components."""

    ordinals = ("first", "second", "third", "fourth")
    return tuple(
        f"{ordinal}_{_shell_letter(order)}"
        for ordinal, order in zip(ordinals, spec.angular, strict=True)
    )


def _emitted_component_names(
    spec: ShellClassSpec,
) -> tuple[str, str, str, str]:
    """Return the actual scalar names present in specialized CUDA source."""

    if spec == DPPP_SPEC:
        return ("d_component", "first_p", "third_p", "fourth_p")
    return _component_names(spec)


def _specialize_dppp_identifiers(source: str, spec: ShellClassSpec) -> str:
    """Rename the shared CUDA skeleton for a non-dppp shell class."""

    if spec == DPPP_SPEC:
        return source
    notation = (
        f"({_shell_letter(spec.angular[0])} "
        f"{_shell_letter(spec.angular[1])}|"
        f"{_shell_letter(spec.angular[2])} "
        f"{_shell_letter(spec.angular[3])})"
    )
    class_name = spec.name[0].upper() + spec.name[1:]
    source = source.replace("(d p|p p)", notation)
    source = source.replace("Dppp", class_name)
    source = source.replace("DPPP", spec.name.upper())
    return source.replace("dppp", spec.name)


def _generic_component_decode(
    spec: ShellClassSpec, *, include_s: bool = True
) -> tuple[str, ...]:
    """Emit compile-time division/modulo lane decoding from spec strides."""

    names = _component_names(spec)
    counts = tuple(map(len, spec.center_components))
    lines = []
    for angular, name, count, stride in zip(
        spec.angular, names, counts, spec.component_strides, strict=True
    ):
        if angular == 0 and not include_s:
            continue
        if count == 1:
            expression = "0U"
        else:
            expression = "component"
            if stride != 1:
                expression = f"({expression} / {stride}U)"
            expression = f"{expression} % {count}U"
        lines.append(f"  const unsigned {name} = {expression};")
    return tuple(lines)


def _component_axis_expression(
    spec: ShellClassSpec,
    center: int,
    quantum: int,
    component_name: str,
) -> str:
    """Lower one component quantum without a runtime shell-class branch."""

    angular_momentum = spec.angular[center]
    if angular_momentum == 1:
        return component_name
    if angular_momentum == 2:
        return f"generated_dppp_d_axes[{component_name}][{quantum}]"
    if angular_momentum == 3:
        return f"generated_dppp_f_axes[{component_name}][{quantum}]"
    raise ValueError(
        "current fused CUDA candidate supports s, p, d, and f centers only"
    )


def _cuda_array_declaration(declaration: str, values: Sequence[str]) -> list[str]:
    """Format a small local CUDA initializer with stable indentation."""

    if not values:
        raise ValueError("CUDA local arrays cannot be empty")
    lines = [f"  {declaration} = {{"]
    lines.extend(
        f"      {value}{',' if index + 1 < len(values) else ''}"
        for index, value in enumerate(values)
    )
    lines[-1] += "};"
    return lines


def _generic_component_gradient_setup(spec: ShellClassSpec) -> str:
    """Generate lane decoding and both Gaussian-pair recurrence inputs."""

    if spec == DPPP_SPEC:
        # Preserve the production golden source byte-for-byte while other
        # shell classes use the fully generated center naming below.
        return """  const unsigned d_component = component / 27U;
  const unsigned p_components = component % 27U;
  const unsigned first_p = p_components / 9U;
  const unsigned third_p = (p_components / 3U) % 3U;
  const unsigned fourth_p = p_components % 3U;
  const unsigned first_axes[3] = {
      generated_dppp_d_axes[d_component][0],
      generated_dppp_d_axes[d_component][1],
      first_p};
  const double first_shifts[3] = {
      geometry.pair_shifts[0][first_axes[0]],
      geometry.pair_shifts[0][first_axes[1]],
      geometry.pair_shifts[1][first_axes[2]]};
  const double first_shift_gradients[3] = {
      geometry.product_scales[0] - 1.0,
      geometry.product_scales[0] - 1.0,
      geometry.product_scales[0]};
  const unsigned second_axes[2] = {third_p, fourth_p};
  const double second_shifts[2] = {
      geometry.pair_shifts[2][third_p],
      geometry.pair_shifts[3][fourth_p]};
  const double second_shift_gradients[2] = {
      geometry.product_scales[2] - 1.0,
      geometry.product_scales[2]};"""

    names = _component_names(spec)
    lines = list(_generic_component_decode(spec, include_s=False))
    pair_definitions = (
        ((0, 1), "first", "geometry.product_scales[0]"),
        ((2, 3), "second", "geometry.product_scales[2]"),
    )
    for centers, pair_name, scale in pair_definitions:
        axes = []
        shifts = []
        gradients = []
        for pair_center, center in enumerate(centers):
            for quantum in range(spec.angular[center]):
                axis = _component_axis_expression(spec, center, quantum, names[center])
                axis_position = len(axes)
                axes.append(axis)
                shifts.append(
                    f"geometry.pair_shifts[{center}][{pair_name}_axes[{axis_position}]]"
                )
                gradients.append(f"{scale} - 1.0" if pair_center == 0 else scale)
        order = sum(spec.angular[center] for center in centers)
        storage_order = max(order, 1)
        lines.extend(
            _cuda_array_declaration(
                f"const unsigned {pair_name}_axes[{storage_order}]",
                axes or ["0U"],
            )
        )
        lines.extend(
            _cuda_array_declaration(
                f"const double {pair_name}_shifts[{storage_order}]",
                shifts or ["0.0"],
            )
        )
        lines.extend(
            _cuda_array_declaration(
                f"const double {pair_name}_shift_gradients[{storage_order}]",
                gradients or ["0.0"],
            )
        )
    return "\n".join(lines)


def _generic_component_value_setup(spec: ShellClassSpec) -> str:
    """Generate lane decoding and coefficient-only pair recurrence inputs."""

    names = _component_names(spec)
    lines = list(_generic_component_decode(spec, include_s=False))
    for centers, pair_name in (((0, 1), "first"), ((2, 3), "second")):
        axes = []
        shifts = []
        for center in centers:
            for quantum in range(spec.angular[center]):
                axis = _component_axis_expression(spec, center, quantum, names[center])
                axis_position = len(axes)
                axes.append(axis)
                shifts.append(
                    f"geometry.pair_shifts[{center}][{pair_name}_axes[{axis_position}]]"
                )
        order = sum(spec.angular[center] for center in centers)
        storage_order = max(order, 1)
        lines.extend(
            _cuda_array_declaration(
                f"const unsigned {pair_name}_axes[{storage_order}]",
                axes or ["0U"],
            )
        )
        lines.extend(
            _cuda_array_declaration(
                f"const double {pair_name}_shifts[{storage_order}]",
                shifts or ["0.0"],
            )
        )
    return "\n".join(lines)


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
        "__launch_bounds__(kGeneratedDpppFockBlockThreads, "
        f"{minimum_blocks_per_sm})"
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


def _emit_shell_class_fock_cuda(spec: ShellClassSpec, plan: FusedShellPlan) -> str:
    """Emit coefficient-only Fock workers beside an accepted force kernel.

    Fock construction reuses primitive geometry and the Cartesian Coulomb
    table, but deliberately omits shift gradients and the raised derivative
    order required only by analytic forces.
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
    if plan.schedule.kind == ScheduleKind.COMPONENT_LANES:
        block_threads = (
            (max(spec.component_count, coulomb_state_count) + 31) // 32
        ) * 32
    else:
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
{double_pair_matchings}  return term;
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
    if (
        spec.name in ("dpdp", "ddpp", "ddds")
        and plan.schedule.kind == ScheduleKind.COMPONENT_LANES
        and plan.kernel.integral.recurrence in ("rys3", "rys4", "rys5")
        and plan.schedule.block_threads >= spec.component_count
    ):
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


def _generic_task_component_setup(spec: ShellClassSpec) -> str:
    """Generate AO/density routing for one automatically decoded lane."""

    if spec == DPPP_SPEC:
        return """  const unsigned d_component = component / 27U;
  const unsigned p_components = component % 27U;
  const unsigned first_p = p_components / 9U;
  const unsigned third_p = (p_components / 3U) % 3U;
  const unsigned fourth_p = p_components % 3U;
  const bool unique_ket_component =
      shared.task.shell[2] != shared.task.shell[3] || third_p >= fourth_p;
  const std::size_t i = shared.task.ao_begin[0] + d_component;
  const std::size_t j = shared.task.ao_begin[1] + first_p;
  const std::size_t k = shared.task.ao_begin[2] + third_p;
  const std::size_t l = shared.task.ao_begin[3] + fourth_p;"""

    names = _component_names(spec)
    lines = list(_generic_component_decode(spec))
    symmetry_conditions = []
    for first, second in ((0, 1), (2, 3)):
        if spec.angular[first] == spec.angular[second]:
            symmetry_conditions.append(
                f"shared.task.shell[{first}] != shared.task.shell[{second}] || "
                f"{names[first]} >= {names[second]}"
            )
    if spec.angular[:2] == spec.angular[2:]:
        # The active-tile builder triangularizes AO-pair quartets when the
        # bra and ket refer to the same shell pair.  Generated shell-wide
        # workers must reproduce that domain; density permutation de-dup only
        # removes equal AO-index permutations and cannot prevent evaluating
        # both (ij|kl) and (kl|ij) component lanes.
        second_component_count = len(spec.center_components[1])
        fourth_component_count = len(spec.center_components[3])
        symmetry_conditions.append(
            "shared.task.shell_pair[0] != shared.task.shell_pair[1] || "
            f"({names[0]} * {second_component_count}U + {names[1]}) >= "
            f"({names[2]} * {fourth_component_count}U + {names[3]})"
        )
    if symmetry_conditions:
        expression = ") && (".join(symmetry_conditions)
        lines.extend(
            [
                "  const bool unique_ket_component =",
                f"      ({expression});",
            ]
        )
    else:
        lines.append("  constexpr bool unique_ket_component = true;")
    for center, (ao_name, component_name) in enumerate(
        zip(("i", "j", "k", "l"), names, strict=True)
    ):
        lines.append(
            f"  const std::size_t {ao_name} = "
            f"shared.task.ao_begin[{center}] + {component_name};"
        )
    return "\n".join(lines)


def _emit_packed_force_geometry_algebra_cuda(spec: ShellClassSpec) -> str:
    """Lower backend-neutral packed geometry roots into CUDA field stores.

    Pair-product scales remain execution metadata because orientation chooses
    which input pair coefficient occupies each slot.  Every scalar derived
    from the product centers, exponents, coordinates, and weighted primitive
    coefficients is emitted from the shared mathematical geometry graph.
    """

    algebra = build_packed_force_geometry_algebra()
    pair_shift_rows = 4 if spec.angular[3] != 0 else 3
    variable_code = {
        "p": "p",
        "q": "q",
        "first_reduced_exponent": "first_pair.reduced_exponent",
        "second_reduced_exponent": "second_pair.reduced_exponent",
        "first_weighted_coefficient": "first_pair.weighted_coefficient",
        "second_weighted_coefficient": "second_pair.weighted_coefficient",
    }
    for center in ("first", "second", "third", "fourth"):
        for axis in AXES:
            variable_code[f"{center}_coordinate_{axis}"] = f"{center}.{axis}"
    for axis in AXES:
        variable_code[f"product_p_{axis}"] = f"first_pair.product_center.{axis}"
        variable_code[f"product_q_{axis}"] = f"second_pair.product_center.{axis}"

    root_specs: list[tuple[Expr, str | None]] = [
        (algebra.rho, "geometry.rho"),
        (algebra.inverse_two_p, "geometry.inverse_two_p"),
        (algebra.inverse_two_q, "geometry.inverse_two_q"),
    ]
    root_specs.extend(
        (
            expression,
            f"geometry.pair_shifts[{center}][{axis}]",
        )
        for center, row in enumerate(algebra.pair_shifts[:pair_shift_rows])
        for axis, expression in enumerate(row)
    )
    root_specs.extend(
        (expression, f"geometry.difference[{axis}]")
        for axis, expression in enumerate(algebra.difference)
    )
    root_specs.extend(
        (
            expression,
            f"geometry.decay_gradients[{center}][{axis}]",
        )
        for center, row in enumerate(algebra.decay_gradients)
        for axis, expression in enumerate(row)
    )
    root_specs.extend(
        (
            (algebra.argument_squared_distance, "argument_squared_distance"),
            (algebra.boys_argument, None),
            (algebra.prefactor, "geometry.prefactor"),
            (algebra.primitive_coefficient, "geometry.primitive_coefficient"),
        )
    )
    source_roots = tuple(expression for expression, _ in root_specs)
    graph, roots = algebra.graph.apply_algebra_form(
        source_roots,
        AlgebraForm.BINARY,
        PowerLowering.SMALL_INTEGER,
    )
    emitter = CudaEmitter(graph, variable_code)
    emitter.lines.append("  double argument_squared_distance;")
    for (__, target), root in zip(root_specs, roots, strict=True):
        if target is None:
            emitter.emit((root,))
            emitter.lines.append(
                f"  boys_values<{spec.maximum_force_coulomb_order}>"
                f"({emitter.reference(root)}, geometry.boys);"
            )
        else:
            emitter.emit_assignment(root, target)
    return "\n".join(emitter.lines)


def _emit_weighted_component_gradient_cuda(
    spec: ShellClassSpec,
    schedule: ScheduleIR,
) -> str:
    """Emit one shell-wide weighted gradient with horizontal symbolic CSE."""

    maximum_order = spec.maximum_force_coulomb_order
    side = maximum_order + 1
    pair_shift_rows = 4 if spec.angular[3] != 0 else 3
    geometry_algebra = _emit_packed_force_geometry_algebra_cuda(spec)
    compact_geometry = f"""/**
 * Geometry retained by packed force lanes.
 *
 * The generic geometry also materializes Cartesian coordinate powers and
 * (-2 rho) powers for component-at-a-time Coulomb evaluation.  The weighted
 * contraction below expands those expressions directly, so retaining the two
 * tables would waste per-lane shared memory and reduce resident warps.
 */
struct GeneratedDpppPackedForceGeometry {{
  double inverse_two_p;
  double inverse_two_q;
  double rho;
  double product_scales[3];
  // Only the first three center derivatives are evaluated explicitly; the
  // fourth follows from translational invariance and needs no stored shift.
  double pair_shifts[{pair_shift_rows}][3];
  double difference[3];
  double decay_gradients[3][3];
  double boys[{side}];
  double prefactor;
  double primitive_coefficient;
}};

__device__ __forceinline__ void generated_dppp_make_packed_force_geometry(
    const GeneratedDpppPrimitivePairData& first_pair,
    const GeneratedDpppPrimitivePairData& second_pair,
    bool first_pair_reversed,
    bool second_pair_reversed,
    const GeneratedDpppVec3& first,
    const GeneratedDpppVec3& second,
    const GeneratedDpppVec3& third,
    const GeneratedDpppVec3& fourth,
    GeneratedDpppPackedForceGeometry& geometry) {{
  const double p = first_pair.exponent_sum;
  const double q = second_pair.exponent_sum;
  geometry.product_scales[0] = first_pair_reversed
      ? first_pair.second_product_scale : first_pair.first_product_scale;
  geometry.product_scales[1] = first_pair_reversed
      ? first_pair.first_product_scale : first_pair.second_product_scale;
  geometry.product_scales[2] = second_pair_reversed
      ? second_pair.second_product_scale : second_pair.first_product_scale;
{geometry_algebra}
}}
"""
    kernel = build_weighted_shell_contraction_kernel(spec)
    variable_code = {
        "inverse_two_p": "geometry.inverse_two_p",
        "inverse_two_q": "geometry.inverse_two_q",
        "rho": "geometry.rho",
        "first_product_scale": "geometry.product_scales[0]",
        "second_product_scale": "geometry.product_scales[1]",
        "third_product_scale": "geometry.product_scales[2]",
        "prefactor": "geometry.prefactor",
    }
    for axis_index, axis in enumerate(AXES):
        variable_code[f"difference_{axis}"] = f"geometry.difference[{axis_index}]"
        for center, prefix in enumerate(("pa", "pb", "qc", "qd")):
            variable_code[f"{prefix}_{axis}"] = (
                f"geometry.pair_shifts[{center}][{axis_index}]"
            )
        for center_index, center in enumerate(("first", "second", "third")):
            variable_code[f"decay_{center}_{axis}"] = (
                f"geometry.decay_gradients[{center_index}][{axis_index}]"
            )
    for order in range(spec.maximum_force_coulomb_order + 1):
        variable_code[f"boys_{order}"] = f"geometry.boys[{order}]"
    for component in range(spec.component_count):
        variable_code[f"component_weight_{component}"] = (
            f"component_weights[{component}]"
        )

    binary_roots = tuple(
        kernel.gradients[center][coordinate]
        for center in range(3)
        for coordinate in range(3)
    )
    graph, roots = kernel.graph.apply_algebra_form(
        binary_roots,
        schedule.algebra_form,
    )
    materialization_plan = graph.materialization_plan(
        roots,
        schedule.algebra_placement.materialization_policy(),
        schedule.algebra_ordering,
        schedule.algebra_fusion,
    )
    emitter = CudaEmitter(
        graph,
        variable_code,
        materialization_plan=materialization_plan,
    )
    emitter.emit(roots)
    lines = [
        compact_geometry.rstrip(),
        "",
        "/** Density-weighted shell gradient with cross-component CSE. */",
        "__device__ __noinline__ void generated_dppp_weighted_component_gradient(",
        "    const GeneratedDpppPackedForceGeometry& geometry,",
        "    const double (&component_weights)[kGeneratedDpppComponentCount],",
        "    double (&gradient)[3][3]) {",
        *emitter.lines,
    ]
    for center in range(3):
        for coordinate in range(3):
            lines.append(
                f"  gradient[{center}][{coordinate}] = "
                f"{emitter.reference(roots[center * 3 + coordinate])};"
            )
    lines.append("}")
    return "\n".join(lines) + "\n\n"


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
#pragma unroll
  for (unsigned slot = 0; slot < 9U; ++slot) {{
    const double value = task_force[slot];
    if (value == 0.0) continue;
    const unsigned center = slot / 3U;
    const unsigned coordinate = slot % 3U;
    atomicAdd(
        forces + static_cast<std::size_t>(task.atom[center]) * 3U + coordinate,
        value);
  }}
#pragma unroll
  for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
    const double value = -task_force[coordinate] -
        task_force[3U + coordinate] - task_force[6U + coordinate];
    if (value != 0.0) {{
      atomicAdd(
          forces + static_cast<std::size_t>(task.atom[3]) * 3U + coordinate,
          value);
    }}
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
    one noinline shell-wide function.  For ``ppps`` that shape gives NVCC an
    addressable 27-double weight table and keeps cross-coordinate CSE values
    live until all nine independent force roots are consumed.  This lowering
    instead names every long-lived value explicitly and emits one coordinate
    scope at a time.  Geometry still resides in lane-private shared storage so
    the thread does not materialize the large primitive structure on its stack.
    """

    if spec.name != "ppps":
        raise ValueError(
            "scalar thread-task force lowering is currently specialized for ppps"
        )
    if plan.schedule.block_threads != 32:
        raise ValueError("scalar ppps thread tasks currently use one CUDA warp")

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

    kernel = build_weighted_shell_contraction_kernel(spec)
    variable_code = {
        "inverse_two_p": "storage.primitive.inverse_two_p",
        "inverse_two_q": "storage.primitive.inverse_two_q",
        "rho": "storage.primitive.rho",
        "first_product_scale": "storage.primitive.product_scales[0]",
        "second_product_scale": "storage.primitive.product_scales[1]",
        "third_product_scale": "storage.primitive.product_scales[2]",
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
        for center_index, center in enumerate(("first", "second", "third")):
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
        kernel = build_weighted_shell_contraction_kernel(spec, (component,))
        roots = [
            kernel.gradients[center][coordinate]
            for center in range(3)
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
    for center in range(3):
        for coordinate in range(3):
            slot = center * 3 + coordinate
            independent_atomics.append(
                f"""  if (storage.task_force[{slot}] != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
            {coordinate}U,
        storage.task_force[{slot}]);
  }}"""
            )
    fourth_atomics = []
    for coordinate in range(3):
        slots = [center * 3 + coordinate for center in range(3)]
        fourth_atomics.append(
            f"""  const double fourth_force_{coordinate} =
      -storage.task_force[{slots[0]}] - storage.task_force[{slots[1]}] -
      storage.task_force[{slots[2]}];
  if (fourth_force_{coordinate} != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[3]) * 3U +
            {coordinate}U,
        fourth_force_{coordinate});
  }}"""
        )

    weight_code = "\n".join(weight_blocks)
    primitive_helper_code = "\n\n".join(primitive_helpers)
    primitive_call_code = "\n".join(primitive_calls)
    independent_atomic_code = "\n".join(independent_atomics)
    fourth_atomic_code = "\n".join(fourth_atomics)
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
    for center in range(3):
        for coordinate in range(3):
            slot = center * 3 + coordinate
            independent_atomics.append(
                f"""  if (force_{slot} != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
            {coordinate}U,
        force_{slot});
  }}"""
            )
    fourth_atomics = []
    for coordinate in range(3):
        slots = [center * 3 + coordinate for center in range(3)]
        fourth_atomics.append(
            f"""  const double fourth_force_{coordinate} =
      -force_{slots[0]} - force_{slots[1]} - force_{slots[2]};
  if (fourth_force_{coordinate} != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[3]) * 3U +
            {coordinate}U,
        fourth_force_{coordinate});
  }}"""
        )

    weight_code = "\n".join(weight_blocks)
    root_body = emit_rys_force_root_body_cuda(
        spec,
        component_group=9,
    )
    independent_atomic_code = "\n".join(independent_atomics)
    fourth_atomic_code = "\n".join(fourth_atomics)
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
      const double gamma2 = 2.0 * q * third_product_scale;
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

/** Base and three independent first derivatives for one Cartesian axis. */
struct GeneratedDppp{class_tag}Axis {{
  double base;
  double first;
  double second;
  double third;
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
 * Evaluate all one-axis values required for A/B/C first derivatives.
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
    double alpha2, double beta2, double gamma2) {{
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
        primitive.alpha2 = 2.0 * primitive.p * first_product_scale;
        primitive.beta2 = 2.0 * primitive.p * second_product_scale;
        primitive.gamma2 = 2.0 * primitive.q * third_product_scale;
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
              primitive.alpha2, primitive.beta2, primitive.gamma2);
          const GeneratedDppp{class_tag}Axis y =
              generated_dppp_{symbol_tag}_axis(
              ay, by, cy, dy_order,
              primitive.pay - primitive.dy * root_bra,
              primitive.qcy + primitive.dy * root_ket,
              primitive.aby, primitive.cdy, b10, b00, b01, 1.0,
              primitive.alpha2, primitive.beta2, primitive.gamma2);
          const GeneratedDppp{class_tag}Axis z =
              generated_dppp_{symbol_tag}_axis(
              az, bz, cz, dz_order,
              primitive.paz - primitive.dz * root_bra,
              primitive.qcz + primitive.dz * root_ket,
              primitive.abz, primitive.cdz, b10, b00, b01, weighted_root,
              primitive.alpha2, primitive.beta2, primitive.gamma2);
          component_force[0] += x.first * y.base * z.base;
          component_force[1] += x.base * y.first * z.base;
          component_force[2] += x.base * y.base * z.first;
          component_force[3] += x.second * y.base * z.base;
          component_force[4] += x.base * y.second * z.base;
          component_force[5] += x.base * y.base * z.second;
          component_force[6] += x.third * y.base * z.base;
          component_force[7] += x.base * y.third * z.base;
          component_force[8] += x.base * y.base * z.third;
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
      const unsigned center = lane / 3U;
      const unsigned coordinate = lane % 3U;
      atomicAdd(
          forces + static_cast<std::size_t>(shared.task.atom[center]) * 3U +
              coordinate,
          value);
    }}
  }}
  __syncthreads();
  if (lane < 3U) {{
    const double fourth_value =
        -shared.warp_sums[0][lane] - shared.warp_sums[0][3U + lane] -
        shared.warp_sums[0][6U + lane];
    if (fourth_value != 0.0) {{
      atomicAdd(
          forces + static_cast<std::size_t>(shared.task.atom[3]) * 3U + lane,
          fourth_value);
    }}
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
      primitive.alpha2 = 2.0 * primitive.p * first_product_scale;
      primitive.beta2 = 2.0 * primitive.p * second_product_scale;
      primitive.gamma2 = 2.0 * primitive.q * third_product_scale;
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
        const unsigned center = slot / 3U;
        const unsigned coordinate = slot % 3U;
        atomicAdd(
            forces +
                static_cast<std::size_t>(shared.tasks[sq].atom[center]) * 3U +
                coordinate,
            reduced[slot]);
      }}
    }}
#pragma unroll
    for (unsigned coordinate = 0U; coordinate < 3U; ++coordinate) {{
      const double fourth =
          -reduced[coordinate] - reduced[3U + coordinate] -
          reduced[6U + coordinate];
      if (fourth != 0.0) {{
        atomicAdd(
            forces +
                static_cast<std::size_t>(shared.tasks[sq].atom[3]) * 3U +
                coordinate,
            fourth);
      }}
    }}
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
        return source.replace("Rys4", replacement).replace(
            "rys4", symbol_replacement
        )
    return source


def _emit_ppps_resident_bra_rys3_force_consumer_cuda(
    *, include_rys3_roots: bool = True
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
    translation invariance.
    """

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
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

    root_body = emit_ppps_rys3_root_body_cuda(
        component_weight_expression="component_weight_{component}"
    )
    weight_code = "\n".join(weight_blocks)
    independent_atomics = []
    # The two pp-bra centers are reduced uniformly across each warp below.
    # Only the task-varying ket center remains an immediate global update.
    for center in range(2, 3):
        for coordinate in range(3):
            slot = center * 3 + coordinate
            independent_atomics.append(
                f"""  if (force_{slot} != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[{center}]) * 3U +
            {coordinate}U,
        force_{slot});
  }}"""
            )
    fourth_atomics = []
    for coordinate in range(3):
        slots = [center * 3 + coordinate for center in range(3)]
        fourth_atomics.append(
            f"""  const double fourth_force_{coordinate} =
      -force_{slots[0]} - force_{slots[1]} - force_{slots[2]};
  if (fourth_force_{coordinate} != 0.0) {{
    atomicAdd(
        context.forces + static_cast<std::size_t>(task.atom[3]) * 3U +
            {coordinate}U,
        fourth_force_{coordinate});
  }}"""
        )
    force_declarations = "\n".join(f"  double force_{slot} = 0.0;" for slot in range(9))
    bra_force_reductions = "\n".join(
        f"    force_{slot} += __shfl_down_sync(0xffffffffU, force_{slot}, delta);"
        for slot in range(6)
    )
    bra_force_atomics = "\n".join(
        f"""    if (force_{slot} != 0.0) {{
      atomicAdd(
          context.forces +
              static_cast<std::size_t>(
                  context.ket_tasks[resident.ket_begin].atom[{slot // 3}]) * 3U +
              {slot % 3}U,
          force_{slot});
    }}"""
        for slot in range(6)
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


def emit_ppps_resident_bra_rys3_cuda(
    *,
    include_shared_definitions: bool = True,
    include_rys3_roots: bool = True,
) -> str:
    """Emit the scalar ``ppps`` resident-bra Rys3 worker.

    The generated kernel has a 256-thread launch bound and shared-memory
    capacity, while its task and bra-staging strides use the actual block
    dimension. Production can therefore compare 32/64/128/256-thread CTAs
    from one binary without changing scalar quartet ownership.

    ``include_shared_definitions`` keeps the standalone correctness harness
    self-contained.  Production AOT shards already contain the ordinary ppps
    shell definitions, so they request only the resident-specific tail to
    avoid duplicate CUDA type/function definitions.  A subset/Wick ordinary
    ppps row also lacks the Rys evaluator; ``include_rys3_roots`` therefore
    controls that one additional shared dependency independently.
    """

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
    )
    plan = build_fused_shell_plan(spec, schedule=schedule, recurrence="rys3")
    resident_tail = _emit_ppps_resident_bra_rys3_force_consumer_cuda(
        include_rys3_roots=include_rys3_roots
    )
    if include_shared_definitions:
        source = emit_shell_class_fused_cuda(spec, plan)
        # The existing Rys thread consumer starts with the attributed roots
        # table; cut before that table so the standalone source does not
        # retain an unused 27-entry shared weight helper from the one-task
        # prototype.
        marker = """/*
 * Three-root interpolation adapted from GPU4PySCF."""
        marker_index = source.find(marker)
        if marker_index < 0:
            raise RuntimeError("generated force task marker changed unexpectedly")
        prefix = source[:marker_index]
    else:
        prefix = ""
        # The resident tail references the normal generated ppps task/cache
        # helpers.  The production shard places it directly after the normal
        # ppps emitter, where those definitions are already available.
    return prefix + _specialize_dppp_identifiers(resident_tail, spec)


# Keep the numbering used in the GPU4PySCF literature discoverable to callers
# while retaining the descriptive name in generated source/tests.
emit_ppps_1110_resident_bra_cuda = emit_ppps_resident_bra_rys3_cuda


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


def emit_shell_class_fused_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan | None = None,
    *,
    fock_schedule: ScheduleIR | None = None,
) -> str:
    """Emit a complete cooperative force kernel from a shell specification.

    The task queue is the outer symmetry/orbit boundary: every task is already
    canonicalized to the requested shell class and retains its slot-to-atom map.
    Consequently the primitive hot loop contains no shell-class, component,
    representative, or coordinate dispatch.
    """

    plan = build_fused_shell_plan(spec) if plan is None else plan
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
    triple_pair_matchings = ""
    if max(spec.pair_orders) >= 6:
        triple_pair_matchings = """  if constexpr (PairOrder >= 6U) {
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
            for (unsigned fifth = 0; fifth < PairOrder; ++fifth) {
              for (unsigned sixth = fifth + 1U; sixth < PairOrder; ++sixth) {
                if (axes[fifth] != axes[sixth]) continue;
                const unsigned third_removed =
                    (1U << fifth) | (1U << sixth);
                if (second_removed >= third_removed ||
                    ((first_removed | second_removed) & third_removed) != 0U) {
                  continue;
                }
                VIBEQC_PAIR_MATCHING_CALL(
                    term, axes, shifts, shift_gradients,
                    inverse_two_exponent, subset,
                    first_removed | second_removed | third_removed, 3U);
              }
            }
          }
        }
      }
    }
  }
"""
        triple_pair_matchings = triple_pair_matchings.replace(
            "VIBEQC_PAIR_MATCHING_CALL", pair_matching_call
        )
    component_gradient_setup = _generic_component_gradient_setup(spec)
    task_component_setup = _generic_task_component_setup(spec)
    component_names = _emitted_component_names(spec)
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
        value_gradient[0][coordinate] +=
            first_coefficient_gradient * state_value +
            geometry.product_scales[0] * scaled_derivative;
        value_gradient[1][coordinate] +=
            -first_coefficient_gradient * state_value +
            geometry.product_scales[1] * scaled_derivative;
        value_gradient[2][coordinate] +=
            second_coefficient_gradient * state_value -
            geometry.product_scales[2] * scaled_derivative;
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
  double decay_gradients[3][3];
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

__device__ __constant__ {coulomb_index_type} generated_dppp_coulomb_indices[{side**3}] = {{
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
      (x_order * {side}U + y_order) * {side}U + z_order]);
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
#pragma unroll
  for (unsigned center = 0; center < 3U; ++center) {{
#pragma unroll
    for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
      gradient[center][coordinate] = geometry.prefactor *
          (value_gradient[center][coordinate] +
           value * geometry.decay_gradients[center][coordinate]);
    }}
  }}
#pragma unroll
  for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] -
        gradient[2][coordinate];
  }}
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
    // Accumulate only the three independent center derivatives.  The fourth
    // center follows from translational invariance after the block reduction,
    // avoiding three long-lived FP64 accumulators in every component lane.
    double warp_sums[kGeneratedDpppWarpCount][9];
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
#pragma unroll
        for (unsigned center = 0; center < 3U; ++center) {{
#pragma unroll
          for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {{
            component_force[center * 3U + coordinate] +=
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
  for (unsigned slot = 0; slot < 9U; ++slot) {{
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
    for (unsigned source_warp = 0; source_warp < kGeneratedDpppWarpCount;
         ++source_warp) {{
      value += shared.warp_sums[source_warp][lane];
    }}
    // Preserve the independent totals for the translation-recovered center.
    // Each lane reads and overwrites only its own slot before the next barrier.
    shared.warp_sums[0][lane] = value;
    if (value != 0.0) {{
      const unsigned center = lane / 3U;
      const unsigned coordinate = lane % 3U;
      atomicAdd(forces + static_cast<std::size_t>(shared.task.atom[center]) * 3U +
                    coordinate,
                value);
    }}
  }}
  __syncthreads();
  if (lane < 3U) {{
    const double fourth_value =
        -shared.warp_sums[0][lane] - shared.warp_sums[0][3U + lane] -
        shared.warp_sums[0][6U + lane];
    if (fourth_value != 0.0) {{
      atomicAdd(
          forces + static_cast<std::size_t>(shared.task.atom[3]) * 3U + lane,
          fourth_value);
    }}
  }}
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
            + _emit_weighted_component_gradient_cuda(spec, plan.schedule)
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
                    2
                    if plan.kernel.integral.recurrence in ("rys4", "rys5")
                    else 0
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
        source += _emit_shell_class_fock_cuda(spec, fock_plan)
    pair_unroll = (
        "#pragma unroll" if plan.schedule.unroll_pair_terms else "#pragma unroll 1"
    )
    source = source.replace("VIBEQC_PAIR_UNROLL", pair_unroll)
    return _specialize_dppp_identifiers(source, spec)


def emit_uncached_primitive_geometry_cuda(spec: ShellClassSpec) -> str:
    """Emit the primitive-quartet setup retained by standalone baselines."""

    maximum_order = spec.maximum_force_coulomb_order
    source = f"""
__device__ __forceinline__ void generated_dppp_make_primitive_geometry_uncached(
    double alpha, const GeneratedDpppVec3& first,
    double beta, const GeneratedDpppVec3& second,
    double gamma, const GeneratedDpppVec3& third,
    double delta, const GeneratedDpppVec3& fourth,
    double primitive_coefficient,
    GeneratedDpppPrimitiveGeometry& geometry) {{
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  geometry.rho = p * q / (p + q);
  geometry.inverse_two_p = 0.5 / p;
  geometry.inverse_two_q = 0.5 / q;
  geometry.product_scales[0] = alpha / p;
  geometry.product_scales[1] = beta / p;
  geometry.product_scales[2] = gamma / q;
  double pair_decay_exponent = 0.0;
  double argument_squared_distance = 0.0;
#pragma unroll
  for (unsigned axis = 0; axis < 3U; ++axis) {{
    const double first_coordinate = generated_dppp_axis(first, axis);
    const double second_coordinate = generated_dppp_axis(second, axis);
    const double third_coordinate = generated_dppp_axis(third, axis);
    const double fourth_coordinate = generated_dppp_axis(fourth, axis);
    const double product_p =
        (alpha * first_coordinate + beta * second_coordinate) / p;
    const double product_q =
        (gamma * third_coordinate + delta * fourth_coordinate) / q;
    geometry.pair_shifts[0][axis] = product_p - first_coordinate;
    geometry.pair_shifts[1][axis] = product_p - second_coordinate;
    geometry.pair_shifts[2][axis] = product_q - third_coordinate;
    geometry.pair_shifts[3][axis] = product_q - fourth_coordinate;
    geometry.difference[axis] = product_p - product_q;
    geometry.decay_gradients[0][axis] =
        -2.0 * mu * (first_coordinate - second_coordinate);
    geometry.decay_gradients[1][axis] =
        2.0 * mu * (first_coordinate - second_coordinate);
    geometry.decay_gradients[2][axis] =
        -2.0 * nu * (third_coordinate - fourth_coordinate);
    pair_decay_exponent +=
        -mu * (first_coordinate - second_coordinate) *
            (first_coordinate - second_coordinate) -
        nu * (third_coordinate - fourth_coordinate) *
            (third_coordinate - fourth_coordinate);
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
      34.986836655249725 / (p * q * sqrt(p + q)) *
      exp(pair_decay_exponent);
  geometry.primitive_coefficient = primitive_coefficient;
}}
"""
    return _specialize_dppp_identifiers(source, spec)


def emit_dppp_fused_cuda(plan: DpppFusedPlan | None = None) -> str:
    """Emit the production-golden dppp specialization of the generic emitter."""

    if plan is None:
        generic = build_fused_shell_plan(DPPP_SPEC)
    else:
        generic = FusedShellPlan(
            kernel=build_fused_shell_plan(DPPP_SPEC).kernel,
            spec=DPPP_SPEC,
            components=plan.components,
            coulomb_states=plan.coulomb_states,
            coulomb_indices=plan.coulomb_indices,
            block_threads=plan.block_threads,
        )
    return emit_shell_class_fused_cuda(DPPP_SPEC, generic)

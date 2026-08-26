"""Generate a cooperative, shell-class-wide ``(d p|p p)`` CUDA kernel.

The first ``dppp`` AOT experiment emitted scalar component functions and
dispatched them inside the primitive/component/coordinate loops. That shape
duplicated Cartesian algebra, prevented primitive reuse, and made register
pressure part of the already-large generic force kernel.

This module instead specializes the execution schedule. One 192-thread block
owns one canonical ``dppp`` shell quartet, its first 162 lanes own the
Cartesian AO components, and all lanes reuse one primitive geometry, Boys
sequence, and compact table of 84 Cartesian Coulomb derivatives. The emitted
kernel also performs density, primitive, and force contraction, so no
generated function is called from a hot inner loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .cuda import CudaEmitter
from .fused_schedule import (
    CoulombState,
    FusedShellPlan,
    build_fused_shell_plan,
    evaluate_fused_shell_component,
)
from .ir import KernelConsumer, PairOrientation, PairStorage, ScheduleKind
from .shell_class import build_weighted_shell_contraction_kernel
from .shell_spec import AXES, DPPP_SPEC, ShellClassSpec, cartesian_components

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


def _cuda_array_declaration(
    declaration: str, values: Sequence[str]
) -> list[str]:
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
                axis = _component_axis_expression(
                    spec, center, quantum, names[center]
                )
                axis_position = len(axes)
                axes.append(axis)
                shifts.append(
                    f"geometry.pair_shifts[{center}]"
                    f"[{pair_name}_axes[{axis_position}]]"
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
                axis = _component_axis_expression(
                    spec, center, quantum, names[center]
                )
                axis_position = len(axes)
                axes.append(axis)
                shifts.append(
                    f"geometry.pair_shifts[{center}]"
                    f"[{pair_name}_axes[{axis_position}]]"
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


def _emit_shell_class_fock_cuda(
    spec: ShellClassSpec, plan: FusedShellPlan
) -> str:
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
    coulomb_state_count = (
        (maximum_order + 1) * (maximum_order + 2) * (maximum_order + 3) // 6
    )
    if plan.schedule.kind == ScheduleKind.COMPONENT_LANES:
        block_threads = (
            (max(spec.component_count, coulomb_state_count) + 31) // 32
        ) * 32
    else:
        block_threads = plan.schedule.block_threads
    minimum_blocks_per_sm = (384 + block_threads - 1) // block_threads
    barrier = "__syncwarp();" if block_threads == 32 else "__syncthreads();"
    component_setup = _generic_component_value_setup(spec)
    task_component_setup = _generic_task_component_setup(spec)
    component_names = _emitted_component_names(spec)
    shared_coulomb = "true" if plan.schedule.shared_coulomb else "false"
    coulomb_storage_count = (
        "kGeneratedDpppFockCoulombStateCount"
        if plan.schedule.shared_coulomb
        else "1"
    )
    coulomb_setup = ""
    if plan.schedule.shared_coulomb:
        if plan.schedule.kind == ScheduleKind.TILED_COMPONENTS:
            coulomb_setup = """          for (unsigned state = lane;
               state < kGeneratedDpppFockCoulombStateCount;
               state += kGeneratedDpppBlockThreads) {
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
QCE_PAIR_UNROLL
  for (unsigned first_subset = 0;
       first_subset < {1 << first_pair_order}U; ++first_subset) {{
    const GeneratedDpppValueTerm first_term =
        generated_dppp_pair_value_term<{first_pair_order}U>(
        first_axes, first_shifts, geometry.inverse_two_p, first_subset);
QCE_PAIR_UNROLL
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
QCE_PAIR_UNROLL
  for (unsigned second_subset = 0;
       second_subset < {1 << second_pair_order}U; ++second_subset) {{
    const GeneratedDpppValueTerm second_term =
        generated_dppp_pair_value_term<{second_pair_order}U>(
        second_axes, second_shifts, geometry.inverse_two_q, second_subset);
    const double sign =
        (generated_dppp_state_total(second_term.derivative_state) & 1U)
        == 0U ? 1.0 : -1.0;
QCE_PAIR_UNROLL
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
QCE_PAIR_UNROLL
  for (unsigned subset = 0; subset < {1 << second_pair_order}U; ++subset) {{
    second_terms[subset] = generated_dppp_pair_value_term<{second_pair_order}U>(
        second_axes, second_shifts, geometry.inverse_two_q, subset);
  }}
  double value = 0.0;
QCE_PAIR_UNROLL
  for (unsigned first_subset = 0;
       first_subset < {1 << first_pair_order}U; ++first_subset) {{
    const GeneratedDpppValueTerm first_term =
        generated_dppp_pair_value_term<{first_pair_order}U>(
        first_axes, first_shifts, geometry.inverse_two_p, first_subset);
QCE_PAIR_UNROLL
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
QCE_PAIR_UNROLL
  for (unsigned subset = 0; subset < {1 << first_pair_order}U; ++subset) {{
    first_terms[subset] = generated_dppp_pair_value_term<{first_pair_order}U>(
        first_axes, first_shifts, geometry.inverse_two_p, subset);
  }}
  double value = 0.0;
QCE_PAIR_UNROLL
  for (unsigned second_subset = 0;
       second_subset < {1 << second_pair_order}U; ++second_subset) {{
    const GeneratedDpppValueTerm second_term =
        generated_dppp_pair_value_term<{second_pair_order}U>(
        second_axes, second_shifts, geometry.inverse_two_q, second_subset);
    const double sign =
        (generated_dppp_state_total(second_term.derivative_state) & 1U)
        == 0U ? 1.0 : -1.0;
QCE_PAIR_UNROLL
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
      term.derivative_state += 1U << (3U * axes[quantum]);
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

/** Scatter one canonical integral using QCE's existing RHF/UHF convention. */
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
    if plan.schedule.kind == ScheduleKind.PACKED_TASKS:
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


def _emit_weighted_component_gradient_cuda(spec: ShellClassSpec) -> str:
    """Emit one shell-wide weighted gradient with horizontal symbolic CSE."""

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
        variable_code[f"difference_{axis}"] = (
            f"geometry.difference[{axis_index}]"
        )
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

    roots = [
        kernel.gradients[center][coordinate]
        for center in range(3)
        for coordinate in range(3)
    ]
    emitter = CudaEmitter(kernel.graph, variable_code)
    emitter.emit(roots)
    lines = [
        "/** Density-weighted shell gradient with cross-component CSE. */",
        "__device__ __noinline__ void generated_dppp_weighted_component_gradient(",
        "    const GeneratedDpppPrimitiveGeometry& geometry,",
        "    const double (&component_weights)[kGeneratedDpppComponentCount],",
        "    double (&gradient)[3][3]) {",
        *emitter.lines,
    ]
    for center in range(3):
        for coordinate in range(3):
            lines.append(
                f"  gradient[{center}][{coordinate}] = "
                f"{emitter.reference(kernel.gradients[center][coordinate])};"
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
    return f"""struct GeneratedDpppPackedLaneStorage {{
  GeneratedDpppVec3 positions[4];
  GeneratedDpppPrimitiveGeometry primitive;
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
    GeneratedDpppPackedLaneStorage& storage) {{
  const GeneratedDpppShellTask& task = tasks[task_index];
#pragma unroll
  for (unsigned center = 0; center < 4U; ++center) {{
    storage.positions[center] = atom_positions[task.atom[center]];
  }}
  double component_weights[kGeneratedDpppComponentCount]{{}};
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
  }}
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
      generated_dppp_make_primitive_geometry(
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
  __shared__ GeneratedDpppPackedLaneStorage lane_storage[32];
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
  __shared__ GeneratedDpppPackedLaneStorage lane_storage[32];
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
  __shared__ GeneratedDpppPackedLaneStorage lane_storage[32];
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


def _emit_subgroup_force_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
) -> str:
    """Emit force workers with several independently progressing tasks per warp."""

    subgroup_lanes = plan.schedule.subgroup_lanes
    subgroup_count = plan.schedule.tasks_per_block
    components_per_lane = (
        spec.component_count + subgroup_lanes - 1
    ) // subgroup_lanes
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
    return f"""

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

""" + ordinary_wrapper(
        "generated_dppp_shell_class_force_rhf_kernel", "false"
    ) + ordinary_wrapper(
        "generated_dppp_shell_class_force_uhf_kernel", "true"
    ) + persistent_wrapper(
        "generated_dppp_shell_class_force_rhf_persistent_kernel", "false"
    ) + persistent_wrapper(
        "generated_dppp_shell_class_force_uhf_persistent_kernel", "true"
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
    return f"""

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
    GeneratedDpppPackedLaneStorage& storage) {{
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
  __shared__ GeneratedDpppPackedLaneStorage lane_storage[32];
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
  __shared__ GeneratedDpppPackedLaneStorage lane_storage[32];
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
  __shared__ GeneratedDpppPackedLaneStorage lane_storage[32];
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
    components_per_lane = (
        spec.component_count + subgroup_lanes - 1
    ) // subgroup_lanes
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

    return f"""

/** Value-path state private to one independently progressing subgroup. */
struct GeneratedDpppSubgroupFockStorage {{
  GeneratedDpppShellTask task;
  GeneratedDpppVec3 positions[4];
  GeneratedDpppPrimitiveGeometry primitive;
  double coulomb[kGeneratedDpppFockCoulombStateCount];
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

  bool evaluate_components[{components_per_lane}]{{}};
  double angular_coefficients[{components_per_lane}]{{}};
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
    const bool retained_by_schwarz = schwarz_bounds == nullptr ||
        schwarz_bounds[
            shared.task.density_offset +
            generated_dppp_matrix_index(i, j, matrix_order)] *
            schwarz_bounds[
                shared.task.density_offset +
                generated_dppp_matrix_index(k, l, matrix_order)] >=
            screening_tolerance;
    evaluate_components[local_component] =
        component_lane && unique_ket_component && retained_by_schwarz;
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
        if (!evaluate_components[local_component]) continue;
        const unsigned component =
            lane + local_component * {subgroup_lanes}U;
        component_integrals[local_component] +=
            angular_coefficients[local_component] *
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
""" + ordinary_wrapper(
        "generated_dppp_shell_class_fock_rhf_kernel", "false"
    ) + ordinary_wrapper(
        "generated_dppp_shell_class_fock_uhf_kernel", "true"
    ) + persistent_wrapper(
        "generated_dppp_shell_class_fock_rhf_persistent_kernel", "false"
    ) + persistent_wrapper(
        "generated_dppp_shell_class_fock_uhf_persistent_kernel", "true"
    )


def emit_shell_class_fused_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan | None = None,
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
        ScheduleKind.SUBGROUP_TASKS,
        ScheduleKind.SHELL_TASK,
        ScheduleKind.COMPONENT_LANES,
        ScheduleKind.TILED_COMPONENTS,
    ):
        raise ValueError(
            "current CUDA emitter implements packed, subgroup-task, shell-task, component-lane, and tiled schedules"
        )
    if (
        plan.schedule.kind == ScheduleKind.PACKED_TASKS
        and plan.schedule.shared_coulomb
    ):
        raise ValueError("packed tasks require lane-local Coulomb evaluation")
    if (
        plan.schedule.kind == ScheduleKind.TILED_COMPONENTS
        and plan.schedule.component_tile != plan.schedule.block_threads
    ):
        raise ValueError(
            "tiled CUDA lowering requires one component per block thread"
        )
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
        x_order
        | (y_order << state_axis_bits)
        | (z_order << (2 * state_axis_bits))
        for x_order, y_order, z_order in plan.coulomb_states
    )
    coulomb_index_type = (
        "signed char" if len(plan.coulomb_states) <= 128 else "short"
    )
    d_axes = tuple(
        _AXIS_INDEX[axis]
        for component in DPPP_SPEC.center_components[0]
        for axis in component
    )
    f_axes = tuple(
        _AXIS_INDEX[axis]
        for component in cartesian_components(3)
        for axis in component
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
            QCE_PAIR_MATCHING_CALL(
                term, axes, shifts, shift_gradients, inverse_two_exponent,
                subset, first_removed | second_removed, 2U);
          }
        }
      }
    }
  }
"""
        double_pair_matchings = double_pair_matchings.replace(
            "QCE_PAIR_MATCHING_CALL", pair_matching_call
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
                QCE_PAIR_MATCHING_CALL(
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
            "QCE_PAIR_MATCHING_CALL", pair_matching_call
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
        "kGeneratedDpppCoulombStateCount"
        if plan.schedule.shared_coulomb
        else "1"
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
QCE_PAIR_UNROLL
  for (unsigned first_subset = 0;
       first_subset < {1 << first_pair_order}U; ++first_subset) {{
    const GeneratedDpppPairTerm first_term = {first_pair_term}(
        first_axes, first_shifts, first_shift_gradients,
        geometry.inverse_two_p, first_subset);
QCE_PAIR_UNROLL
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
QCE_PAIR_UNROLL
  for (unsigned second_subset = 0;
       second_subset < {1 << second_pair_order}U; ++second_subset) {{
    const GeneratedDpppPairTerm second_term = {second_pair_term}(
        second_axes, second_shifts, second_shift_gradients,
        geometry.inverse_two_q, second_subset);
QCE_PAIR_UNROLL
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
QCE_PAIR_UNROLL
  for (unsigned subset = 0; subset < {1 << second_pair_order}U; ++subset) {{
    second_terms[subset] = {second_pair_term}(
        second_axes, second_shifts, second_shift_gradients,
        geometry.inverse_two_q, subset);
  }}
  double value = 0.0;
  double value_gradient[3][3]{{}};
QCE_PAIR_UNROLL
  for (unsigned first_subset = 0;
       first_subset < {1 << first_pair_order}U; ++first_subset) {{
    const GeneratedDpppPairTerm first_term = {first_pair_term}(
        first_axes, first_shifts, first_shift_gradients,
        geometry.inverse_two_p, first_subset);
QCE_PAIR_UNROLL
    for (unsigned second_subset = 0;
         second_subset < {1 << second_pair_order}U; ++second_subset) {{
      const GeneratedDpppPairTerm& second_term = second_terms[second_subset];
{pair_accumulation}    }}
  }}
"""
    else:
        gradient_contraction = f"""  GeneratedDpppPairTerm first_terms[{1 << first_pair_order}];
QCE_PAIR_UNROLL
  for (unsigned subset = 0; subset < {1 << first_pair_order}U; ++subset) {{
    first_terms[subset] = {first_pair_term}(
        first_axes, first_shifts, first_shift_gradients,
        geometry.inverse_two_p, subset);
  }}
  double value = 0.0;
  double value_gradient[3][3]{{}};
QCE_PAIR_UNROLL
  for (unsigned second_subset = 0;
       second_subset < {1 << second_pair_order}U; ++second_subset) {{
    const GeneratedDpppPairTerm second_term = {second_pair_term}(
        second_axes, second_shifts, second_shift_gradients,
        geometry.inverse_two_q, second_subset);
QCE_PAIR_UNROLL
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
    warp_count_declaration = (
        ""
        if plan.schedule.kind == ScheduleKind.PACKED_TASKS
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
  double coefficient = 0.0;
#pragma unroll
  for (unsigned permutation = 0; permutation < 8U; ++permutation) {{
    std::size_t a = 0, b = 0, c = 0, d = 0;
    generated_dppp_eri_permutation(permutation, i, j, k, l, a, b, c, d);
    if (!generated_dppp_unique_permutation(
            permutation, i, j, k, l, a, b, c, d)) continue;
    const std::size_t ab = generated_dppp_matrix_index(a, b, n);
    const std::size_t ac = generated_dppp_matrix_index(a, c, n);
    const std::size_t cd = generated_dppp_matrix_index(c, d, n);
    const std::size_t bd = generated_dppp_matrix_index(b, d, n);
    if constexpr (Unrestricted) {{
      const double total_ab = density[task.spin_offset + ab] +
          density[task.spin_offset + matrix_size + ab];
      const double total_cd = density[task.spin_offset + cd] +
          density[task.spin_offset + matrix_size + cd];
      coefficient += 0.5 * total_ab * total_cd;
      coefficient -= 0.5 *
          (density[task.spin_offset + ac] * density[task.spin_offset + bd] +
           density[task.spin_offset + matrix_size + ac] *
               density[task.spin_offset + matrix_size + bd]);
    }} else {{
      coefficient += 0.5 * density[task.density_offset + ab] *
          density[task.density_offset + cd] -
          0.25 * density[task.density_offset + ac] *
          density[task.density_offset + bd];
    }}
  }}
  return coefficient;
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
    double warp_sums[kGeneratedDpppWarpCount][12];
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
  const bool retained_by_schwarz = schwarz_bounds == nullptr ||
      schwarz_bounds[
          shared.task.density_offset +
          generated_dppp_matrix_index(i, j, matrix_order)] *
          schwarz_bounds[
              shared.task.density_offset +
              generated_dppp_matrix_index(k, l, matrix_order)] >=
          screening_tolerance;
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
  double component_force[12]{{}};

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
        for (unsigned center = 0; center < 4U; ++center) {{
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
  for (unsigned slot = 0; slot < 12U; ++slot) {{
    double value = component_force[slot];
#pragma unroll
    for (unsigned offset = 16U; offset != 0U; offset /= 2U) {{
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }}
    if (warp_lane == 0U) shared.warp_sums[warp][slot] = value;
  }}
  __syncthreads();
  if (lane < 12U) {{
    double value = 0.0;
#pragma unroll
    for (unsigned source_warp = 0; source_warp < kGeneratedDpppWarpCount;
         ++source_warp) {{
      value += shared.warp_sums[source_warp][lane];
    }}
    if (value != 0.0) {{
      const unsigned center = lane / 3U;
      const unsigned coordinate = lane % 3U;
      atomicAdd(forces + static_cast<std::size_t>(shared.task.atom[center]) * 3U +
                    coordinate,
                value);
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
    if plan.schedule.kind == ScheduleKind.PACKED_TASKS:
        force_marker = """template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_task("""
        force_begin = source.find(force_marker)
        if force_begin < 0:
            raise RuntimeError("generated force task marker changed unexpectedly")
        source = (
            source[:force_begin]
            + _emit_weighted_component_gradient_cuda(spec)
            + _emit_packed_force_consumer_cuda(
                spec,
                plan,
                minimum_blocks_per_sm,
            )
        )
    elif plan.schedule.kind == ScheduleKind.SUBGROUP_TASKS:
        force_marker = """template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_task("""
        force_begin = source.find(force_marker)
        if force_begin < 0:
            raise RuntimeError("generated force task marker changed unexpectedly")
        source = source[:force_begin] + _emit_subgroup_force_consumer_cuda(
            spec,
            plan,
            minimum_blocks_per_sm,
        )
    if KernelConsumer.FOCK in plan.kernel.integral.consumers:
        source += _emit_shell_class_fock_cuda(spec, plan)
    pair_unroll = (
        "#pragma unroll"
        if plan.schedule.unroll_pair_terms
        else "#pragma unroll 1"
    )
    source = source.replace("QCE_PAIR_UNROLL", pair_unroll)
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

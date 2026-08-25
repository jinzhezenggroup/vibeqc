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

from .fused_schedule import (
    CoulombState,
    build_fused_shell_plan,
    evaluate_fused_shell_component,
)
from .shell_spec import AXES, DPPP_SPEC

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


def emit_dppp_fused_cuda(plan: DpppFusedPlan | None = None) -> str:
    """Emit a complete cooperative ``dppp`` force-contraction kernel.

    The task queue is the outer symmetry/orbit boundary: every task is already
    canonicalized to ``(d p|p p)`` and retains its canonical-slot-to-atom map.
    Consequently the primitive hot loop contains no shell-class, component,
    representative, or coordinate dispatch.
    """

    plan = build_dppp_fused_plan() if plan is None else plan
    packed_states = tuple(
        x_order | (y_order << 3) | (z_order << 6)
        for x_order, y_order, z_order in plan.coulomb_states
    )
    d_axes = tuple(
        _AXIS_INDEX[axis]
        for component in DPPP_SPEC.center_components[0]
        for axis in component
    )
    first_pair_order, second_pair_order = DPPP_SPEC.pair_orders
    component_strides = DPPP_SPEC.component_strides
    component_counts = tuple(map(len, DPPP_SPEC.center_components))
    supported_pair_orders = " || ".join(
        f"PairOrder == {order}U" for order in sorted(DPPP_SPEC.pair_orders)
    )
    return f"""/**
 * Generated cooperative AOT candidate for canonical (d p|p p) forces.
 *
 * Launch exactly {plan.block_threads} threads per canonical shell-quartet task.
 * The task builder performs shell-pair/quartet symmetry routing outside this
 * kernel and records the original atom for each canonical center slot.
 */
#include <cstddef>
#include <cstdint>

struct GeneratedDpppVec3 {{ double x; double y; double z; }};

/** Canonical task ABI kept independent of the production DeviceBatch layout. */
struct GeneratedDpppShellTask {{
  std::uint64_t primitive_begin[4];
  std::uint64_t primitive_end[4];
  std::uint64_t ao_begin[4];
  std::uint64_t ao_coefficient_begin[4];
  std::uint64_t density_offset;
  std::uint64_t spin_offset;
  std::uint32_t matrix_order;
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
  double boys[7];
  double coordinate_powers[3][7];
  double negative_two_rho_powers[7];
  double prefactor;
  double primitive_coefficient;
}};

struct GeneratedDpppPairTerm {{
  unsigned derivative_state;
  double coefficient;
  double first_center[3];
}};

constexpr unsigned kGeneratedDpppComponentCount = {_COMPONENT_COUNT}U;
constexpr unsigned kGeneratedDpppBlockThreads = {plan.block_threads}U;
constexpr unsigned kGeneratedDpppCoulombStateCount = {len(plan.coulomb_states)}U;
constexpr unsigned kGeneratedDpppWarpCount = {plan.warp_count}U;

__device__ __constant__ unsigned short generated_dppp_coulomb_states[
    kGeneratedDpppCoulombStateCount] = {{
{_format_cuda_array(packed_states)}
}};

__device__ __constant__ signed char generated_dppp_coulomb_indices[343] = {{
{_format_cuda_array(plan.coulomb_indices)}
}};

__device__ __constant__ unsigned char generated_dppp_d_axes[{component_counts[0]}][{DPPP_SPEC.angular[0]}] = {{
{_format_cuda_array(d_axes, columns=6)}
}};

__device__ __forceinline__ double generated_dppp_axis(
    const GeneratedDpppVec3& value, unsigned axis) {{
  return axis == 0U ? value.x : (axis == 1U ? value.y : value.z);
}}

__device__ __forceinline__ unsigned generated_dppp_state_total(unsigned state) {{
  return (state & 7U) + ((state >> 3U) & 7U) + ((state >> 6U) & 7U);
}}

__device__ __forceinline__ unsigned generated_dppp_state_index(unsigned state) {{
  const unsigned x_order = state & 7U;
  const unsigned y_order = (state >> 3U) & 7U;
  const unsigned z_order = (state >> 6U) & 7U;
  return static_cast<unsigned>(generated_dppp_coulomb_indices[
      (x_order * 7U + y_order) * 7U + z_order]);
}}

__device__ __forceinline__ unsigned generated_dppp_wick_multiplicity(
    unsigned order, unsigned pairs) {{
  if (pairs == 0U) return 1U;
  if (pairs == 1U) return order * (order - 1U) / 2U;
  if (pairs == 2U) {{
    return order * (order - 1U) * (order - 2U) * (order - 3U) / 8U;
  }}
  return order * (order - 1U) * (order - 2U) * (order - 3U) *
      (order - 4U) * (order - 5U) / 48U;
}}

__device__ __forceinline__ double generated_dppp_coulomb(
    unsigned derivative_state,
    const GeneratedDpppPrimitiveGeometry& geometry) {{
  const unsigned x_order = derivative_state & 7U;
  const unsigned y_order = (derivative_state >> 3U) & 7U;
  const unsigned z_order = (derivative_state >> 6U) & 7U;
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
    const unsigned (&axes)[PairOrder],
    const double (&shifts)[PairOrder],
    const double (&shift_gradients)[PairOrder],
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
    const unsigned (&axes)[PairOrder],
    const double (&shifts)[PairOrder],
    const double (&shift_gradients)[PairOrder],
    double inverse_two_exponent,
    unsigned subset) {{
  static_assert({supported_pair_orders});
  GeneratedDpppPairTerm term{{}};
  for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {{
    if ((subset & (1U << quantum)) != 0U) {{
      term.derivative_state += 1U << (3U * axes[quantum]);
    }}
  }}
  generated_dppp_add_pair_matching(
      term, axes, shifts, shift_gradients, inverse_two_exponent,
      subset, 0U, 0U);
  for (unsigned first = 0; first < PairOrder; ++first) {{
    for (unsigned second = first + 1U; second < PairOrder; ++second) {{
      if (axes[first] == axes[second]) {{
        generated_dppp_add_pair_matching(
            term, axes, shifts, shift_gradients, inverse_two_exponent,
            subset, (1U << first) | (1U << second), 1U);
      }}
    }}
  }}
  return term;
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
  const unsigned d_component = component / {component_strides[0]}U;
  const unsigned p_components = component % {component_strides[0]}U;
  const unsigned first_p = p_components / {component_strides[1]}U;
  const unsigned third_p = (p_components / {component_strides[2]}U) % {component_counts[2]}U;
  const unsigned fourth_p = p_components % {component_counts[3]}U;
  const unsigned first_axes[{first_pair_order}] = {{
      generated_dppp_d_axes[d_component][0],
      generated_dppp_d_axes[d_component][1],
      first_p}};
  const double first_shifts[{first_pair_order}] = {{
      geometry.pair_shifts[0][first_axes[0]],
      geometry.pair_shifts[0][first_axes[1]],
      geometry.pair_shifts[1][first_axes[2]]}};
  const double first_shift_gradients[{first_pair_order}] = {{
      geometry.product_scales[0] - 1.0,
      geometry.product_scales[0] - 1.0,
      geometry.product_scales[0]}};
  const unsigned second_axes[{second_pair_order}] = {{third_p, fourth_p}};
  const double second_shifts[{second_pair_order}] = {{
      geometry.pair_shifts[2][third_p],
      geometry.pair_shifts[3][fourth_p]}};
  const double second_shift_gradients[{second_pair_order}] = {{
      geometry.product_scales[2] - 1.0,
      geometry.product_scales[2]}};

  GeneratedDpppPairTerm second_terms[4];
#pragma unroll
  for (unsigned subset = 0; subset < {1 << second_pair_order}U; ++subset) {{
    second_terms[subset] = generated_dppp_pair_term(
        second_axes, second_shifts, second_shift_gradients,
        geometry.inverse_two_q, subset);
  }}
  double value = 0.0;
  double value_gradient[3][3]{{}};
#pragma unroll
  for (unsigned first_subset = 0; first_subset < {1 << first_pair_order}U; ++first_subset) {{
    const GeneratedDpppPairTerm first_term = generated_dppp_pair_term(
        first_axes, first_shifts, first_shift_gradients,
        geometry.inverse_two_p, first_subset);
#pragma unroll
    for (unsigned second_subset = 0; second_subset < {1 << second_pair_order}U; ++second_subset) {{
      const GeneratedDpppPairTerm& second_term = second_terms[second_subset];
      const double sign =
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
                geometry, coulomb, state + (1U << (3U * coordinate)));
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
    }}
  }}
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

__device__ __forceinline__ void generated_dppp_make_primitive_geometry(
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
    for (unsigned power = 1; power <= 6U; ++power) {{
      geometry.coordinate_powers[axis][power] =
          geometry.coordinate_powers[axis][power - 1U] *
          geometry.difference[axis];
    }}
  }}
  boys_values<6>(geometry.rho * argument_squared_distance, geometry.boys);
  geometry.negative_two_rho_powers[0] = 1.0;
#pragma unroll
  for (unsigned power = 1; power <= 6U; ++power) {{
    geometry.negative_two_rho_powers[power] =
        geometry.negative_two_rho_powers[power - 1U] *
        (-2.0 * geometry.rho);
  }}
  geometry.prefactor =
      34.986836655249725 / (p * q * sqrt(p + q)) *
      exp(pair_decay_exponent);
  geometry.primitive_coefficient = primitive_coefficient;
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_task(
    const GeneratedDpppShellTask* tasks,
    const double* primitive_exponents,
    const double* primitive_coefficients,
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
    double coulomb[kGeneratedDpppCoulombStateCount];
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

  const bool component_lane = lane < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? lane : 0U;
  const unsigned d_component = component / {component_strides[0]}U;
  const unsigned p_components = component % {component_strides[0]}U;
  const unsigned first_p = p_components / {component_strides[1]}U;
  const unsigned third_p = (p_components / {component_strides[2]}U) % {component_counts[2]}U;
  const unsigned fourth_p = p_components % {component_counts[3]}U;
  const bool unique_ket_component =
      shared.task.shell[2] != shared.task.shell[3] || third_p >= fourth_p;
  const std::size_t i = shared.task.ao_begin[0] + d_component;
  const std::size_t j = shared.task.ao_begin[1] + first_p;
  const std::size_t k = shared.task.ao_begin[2] + third_p;
  const std::size_t l = shared.task.ao_begin[3] + fourth_p;
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
      ? ao_coefficients[shared.task.ao_coefficient_begin[0] + d_component] *
        ao_coefficients[shared.task.ao_coefficient_begin[1] + first_p] *
        ao_coefficients[shared.task.ao_coefficient_begin[2] + third_p] *
        ao_coefficients[shared.task.ao_coefficient_begin[3] + fourth_p]
      : 0.0;
  double component_force[12]{{}};

  for (std::uint64_t a = shared.task.primitive_begin[0];
       a < shared.task.primitive_end[0]; ++a) {{
    for (std::uint64_t b = shared.task.primitive_begin[1];
         b < shared.task.primitive_end[1]; ++b) {{
      for (std::uint64_t c = shared.task.primitive_begin[2];
           c < shared.task.primitive_end[2]; ++c) {{
        for (std::uint64_t d = shared.task.primitive_begin[3];
             d < shared.task.primitive_end[3]; ++d) {{
          if (lane == 0U) {{
            generated_dppp_make_primitive_geometry(
                primitive_exponents[a], shared.positions[0],
                primitive_exponents[b], shared.positions[1],
                primitive_exponents[c], shared.positions[2],
                primitive_exponents[d], shared.positions[3],
                primitive_coefficients[a] * primitive_coefficients[b] *
                    primitive_coefficients[c] * primitive_coefficients[d],
                shared.primitive);
          }}
          __syncthreads();
          if (lane < kGeneratedDpppCoulombStateCount) {{
            shared.coulomb[lane] = generated_dppp_coulomb(
                generated_dppp_coulomb_states[lane], shared.primitive);
          }}
          __syncthreads();
          if (density_coefficient != 0.0) {{
            double primitive_gradient[4][3];
            generated_dppp_component_gradient<true>(
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
}}

extern "C" __global__ __launch_bounds__(kGeneratedDpppBlockThreads, 2)
void generated_dppp_shell_class_force_rhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const double* primitive_exponents,
    const double* primitive_coefficients,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t task_count) {{
  if (blockIdx.x >= task_count) return;
  generated_dppp_shell_class_force_task<false>(
      tasks, primitive_exponents, primitive_coefficients, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      static_cast<std::size_t>(blockIdx.x));
}}

extern "C" __global__ __launch_bounds__(kGeneratedDpppBlockThreads, 2)
void generated_dppp_shell_class_force_uhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const double* primitive_exponents,
    const double* primitive_coefficients,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t task_count) {{
  if (blockIdx.x >= task_count) return;
  generated_dppp_shell_class_force_task<true>(
      tasks, primitive_exponents, primitive_coefficients, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      static_cast<std::size_t>(blockIdx.x));
}}

/** Persistent workers avoid launching one block per topology-capacity slot. */
template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_force_persistent(
    const GeneratedDpppShellTask* tasks,
    const double* primitive_exponents,
    const double* primitive_coefficients,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
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
        tasks, primitive_exponents, primitive_coefficients, ao_coefficients,
        atom_positions, screening_tolerance, schwarz_bounds, density, forces,
        static_cast<std::size_t>(task_index));
    __syncthreads();
  }}
}}

extern "C" __global__ __launch_bounds__(kGeneratedDpppBlockThreads, 2)
void generated_dppp_shell_class_force_rhf_persistent_kernel(
    const GeneratedDpppShellTask* tasks,
    const double* primitive_exponents,
    const double* primitive_coefficients,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  generated_dppp_shell_class_force_persistent<false>(
      tasks, primitive_exponents, primitive_coefficients, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_count, task_head);
}}

extern "C" __global__ __launch_bounds__(kGeneratedDpppBlockThreads, 2)
void generated_dppp_shell_class_force_uhf_persistent_kernel(
    const GeneratedDpppShellTask* tasks,
    const double* primitive_exponents,
    const double* primitive_coefficients,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  generated_dppp_shell_class_force_persistent<true>(
      tasks, primitive_exponents, primitive_coefficients, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_count, task_head);
}}
"""

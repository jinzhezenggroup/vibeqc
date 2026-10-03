#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <weighted_eri.cuh>

#include "integrals/range_moments.hpp"
#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_force_scatter.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

// A constant keeps the failure marker usable by both NVCC and CuMetal without
// invoking a host-only nan()/numeric_limits function from a device function.
inline constexpr double kInvalidDirectForceMoment = std::numeric_limits<double>::quiet_NaN();

/** Bind one low-order class to radial-moment-parametric compiler force roots. */
template <unsigned ShellClass>
struct LowOrderSourceRoots {
  static constexpr unsigned angular_order = direct_shell_class_angular_order(ShellClass);
  static_assert(angular_order <= 3U);
  static constexpr unsigned component_count =
      ShellClass == kSsssShellClass                                    ? 1U
      : ShellClass == kPsssShellClass                                  ? 3U
      : ShellClass == kDsssShellClass                                  ? 6U
      : ShellClass == kFsssShellClass                                  ? 10U
      : ShellClass == kPppsShellClass                                  ? 27U
      : ShellClass == kDspsShellClass || ShellClass == kDpssShellClass ? 18U
                                                                       : 9U;

  __device__ static __forceinline__ generated_weighted_eri::IndependentGradient evaluate(
      const generated_weighted_eri::Geometry& geometry, const double* weights) {
    if constexpr (ShellClass == kSsssShellClass) {
      return generated_weighted_eri::ssss_force(geometry, weights);
    } else if constexpr (ShellClass == kPsssShellClass) {
      return generated_weighted_eri::psss_force(geometry, weights);
    } else if constexpr (ShellClass == kPspsShellClass) {
      return generated_weighted_eri::psps_force(geometry, weights);
    } else if constexpr (ShellClass == kPpssShellClass) {
      return generated_weighted_eri::ppss_force(geometry, weights);
    } else if constexpr (ShellClass == kDsssShellClass) {
      return generated_weighted_eri::dsss_force(geometry, weights);
    } else if constexpr (ShellClass == kPppsShellClass) {
      return generated_weighted_eri::ppps_force(geometry, weights);
    } else if constexpr (ShellClass == kDspsShellClass) {
      return generated_weighted_eri::dsps_force(geometry, weights);
    } else if constexpr (ShellClass == kDpssShellClass) {
      return generated_weighted_eri::dpss_force(geometry, weights);
    } else {
      static_assert(ShellClass == kFsssShellClass);
      return generated_weighted_eri::fsss_force(geometry, weights);
    }
  }
};

/**
 * Consume one screened shell task into full-range J/K or one LR exchange force.
 *
 * Only immutable shell geometry, AO traversal and radial moments are shared. Each
 * source retains its own density weights, generated force evaluation and
 * primitive reduction order. This is not a combined HF-force specialization.
 * The caller owns screening/queue policy and selects only orders zero to three;
 * no resident primitive storage, new cutoff or additional allocation is needed.
 *
 * LR uses the same generated roots: with rho and omega fixed under nuclear
 * displacement, its radial moments satisfy dM_n/dT = -M_(n+1), just as Boys
 * values do. One moment ladder therefore supplies every AO component and
 * independent center derivative, instead of repeating a Dual3 recurrence for
 * each AO quartet and atom. Higher angular orders keep their existing owner.
 */
template <bool Unrestricted, unsigned ShellClass, bool LongRange = false>
__device__ inline __noinline__ void contract_two_electron_force_low_order_sources_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    double coulomb_coefficient, double exchange_coefficient, double omega = 0.0) {
  using Roots = LowOrderSourceRoots<ShellClass>;
  constexpr unsigned source_count = LongRange ? 1U : 2U;
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;
  if ((LongRange || coulomb_coefficient == 0.0) && exchange_coefficient == 0.0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair], batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair], batch.shell_pair_second[second_pair]};
  unsigned canonical_raw_slot[4];
  generated_weighted_eri::canonicalize_direct_shell_slots(batch.shell_angular, raw_shell,
                                                          canonical_raw_slot);
  std::int32_t canonical_shell[4], center_atoms[4], unique_center_atoms[4];
  std::int32_t unique_order_atoms[4];
  std::size_t component_begin[4];
  unsigned component_count[4];
  const std::size_t dimension = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = dimension * dimension;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * dimension;
#pragma unroll
  for (unsigned center = 0; center < 4; ++center) {
    canonical_shell[center] = raw_shell[canonical_raw_slot[center]];
    center_atoms[center] = batch.shell_atoms[canonical_shell[center]];
    unique_order_atoms[center] =
        ShellClass == kPsssShellClass ? batch.shell_atoms[raw_shell[center]] : center_atoms[center];
    component_begin[center] =
        static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[center]]) -
        system_ao_begin;
    const unsigned angular = batch.shell_angular[canonical_shell[center]];
    component_count[center] = (angular + 1U) * (angular + 2U) / 2U;
  }
  // PSSS's existing worker restores the last unique atom in raw shell order.
  // Retain that order even when the canonical p shell moves to the first slot.
  const unsigned unique_center_count =
      direct_force_unique_center_atoms(unique_order_atoms, unique_center_atoms);
  if (unique_center_count == 1U) return;

  const DirectShellAoQuartetLayout layout =
      direct_shell_ao_quartet_layout(batch, first_pair, second_pair);
  double weights[source_count][Roots::component_count]{};
  bool source_active[source_count]{};
  for (std::size_t ordinal = 0; ordinal < layout.quartet_count; ++ordinal) {
    std::size_t raw_ao[4];
    decode_shell_ao_quartet(batch, first_pair, second_pair, layout, ordinal, system_ao_begin,
                            raw_ao);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], dimension)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], dimension)] <
        screening_tolerance) {
      continue;
    }
    unsigned output = 0U;
#pragma unroll
    for (unsigned center = 0; center < 4; ++center) {
      const std::size_t canonical_ao = raw_ao[canonical_raw_slot[center]];
      if (canonical_ao < component_begin[center]) return;
      const auto component = static_cast<unsigned>(canonical_ao - component_begin[center]);
      if (component >= component_count[center]) return;
      output = output * component_count[center] + component;
    }
    if (output >= Roots::component_count) return;
#pragma unroll
    for (unsigned source = 0; source < source_count; ++source) {
      const double coulomb = !LongRange && source == 0U ? coulomb_coefficient : 0.0;
      const double exchange = LongRange || source == 1U ? exchange_coefficient : 0.0;
      if (coulomb == 0.0 && exchange == 0.0) continue;
      const double coefficient = direct_force_density_coefficient_scaled<Unrestricted>(
          dimension, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2],
          raw_ao[3], coulomb, exchange);
      if constexpr (ShellClass != kPsssShellClass) {
        if (coefficient == 0.0) continue;
      }
      const double weight =
          direct_force_component_weight(batch.direct_ao_coefficients, system_ao_begin, raw_ao[0],
                                        raw_ao[1], raw_ao[2], raw_ao[3], coefficient);
      if constexpr (Roots::angular_order <= 1U) {
        weights[source][output] = weight;
      } else {
        weights[source][output] += weight;
      }
      source_active[source] = true;
    }
  }
  if constexpr (ShellClass == kPsssShellClass) {
#pragma unroll
    for (unsigned source = 0; source < source_count; ++source) {
      source_active[source] =
          weights[source][0] != 0.0 || weights[source][1] != 0.0 || weights[source][2] != 0.0;
    }
  }
  if constexpr (LongRange) {
    if (!source_active[0]) return;
  } else {
    if (!source_active[0] && !source_active[1]) return;
  }

  const std::size_t canonical_pair[2] = {canonical_raw_slot[0] < 2U ? first_pair : second_pair,
                                         canonical_raw_slot[2] < 2U ? first_pair : second_pair};
  const Vec3<double> position[4] = {atom_position<double>(batch, center_atoms[0], -1),
                                    atom_position<double>(batch, center_atoms[1], -1),
                                    atom_position<double>(batch, center_atoms[2], -1),
                                    atom_position<double>(batch, center_atoms[3], -1)};
  const bool reverse_first = batch.shell_pair_first[canonical_pair[0]] != canonical_shell[0];
  const bool reverse_second = batch.shell_pair_first[canonical_pair[1]] != canonical_shell[2];
  const std::int64_t first_begin = batch.shell_pair_primitive_offsets[canonical_pair[0]];
  const std::int64_t first_end = batch.shell_pair_primitive_offsets[canonical_pair[0] + 1];
  const std::int64_t second_begin = batch.shell_pair_primitive_offsets[canonical_pair[1]];
  const std::int64_t second_end = batch.shell_pair_primitive_offsets[canonical_pair[1] + 1];
  generated_weighted_eri::IndependentGradient result[source_count]{};
  for (std::int64_t first_primitive = first_begin; first_primitive < first_end; ++first_primitive) {
    const PrimitivePairData first_data = batch.shell_primitive_pairs[first_primitive];
    for (std::int64_t second_primitive = second_begin; second_primitive < second_end;
         ++second_primitive) {
      const PrimitivePairData second_data = batch.shell_primitive_pairs[second_primitive];
      generated_weighted_eri::Geometry geometry;
      const double boys_argument = generated_weighted_eri::make_direct_cached_geometry(
          first_data, second_data, reverse_first, reverse_second, position[0], position[1],
          position[2], position[3], geometry);
      if constexpr (LongRange) {
        const bool valid = generativeqc::integrals::range_moments(
            Roots::angular_order + 1U, boys_argument, geometry.rho,
            generativeqc::integrals::CoulombRange::Long, omega, geometry.boys);
        // Preserve the existing nonfinite-result failure contract. An invalid
        // ladder must not leave uninitialized moments or silently omit work.
        if (!valid) {
          for (unsigned order = 0; order <= Roots::angular_order + 1U; ++order)
            geometry.boys[order] = kInvalidDirectForceMoment;
        }
      } else {
        boys_values<Roots::angular_order + 1U>(boys_argument, geometry.boys);
      }
#pragma unroll
      for (unsigned source = 0; source < source_count; ++source) {
        if (!source_active[source]) continue;
        const auto primitive = Roots::evaluate(geometry, weights[source]);
#pragma unroll
        for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
          for (unsigned axis = 0; axis < 3; ++axis) {
            result[source].center[center][axis] += primitive.center[center][axis];
          }
        }
      }
    }
  }
#pragma unroll
  for (unsigned source = 0; source < source_count; ++source) {
    if (!source_active[source]) continue;
    scatter_direct_force_independent_gradient(
        center_atoms, unique_center_atoms, unique_center_count, result[source],
        forces + source * static_cast<std::size_t>(batch.total_atoms) * 3U);
  }
}

/** Dispatch only the closed low-order domain for the selected radial owner. */
template <bool Unrestricted, bool LongRange = false>
__device__ inline void contract_two_electron_force_low_order_sources(
    unsigned shell_class, const DeviceBatch& batch, ActiveShellQuartetTile task,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, double coulomb_coefficient,
    double exchange_coefficient, double omega = 0.0) {
#define GENERATIVEQC_LOW_ORDER_SOURCES_CASE(ShellClass)                                      \
  case ShellClass:                                                                           \
    contract_two_electron_force_low_order_sources_task<Unrestricted, ShellClass, LongRange>( \
        batch, task, screening_tolerance, schwarz_bounds, density, active, forces,           \
        coulomb_coefficient, exchange_coefficient, omega);                                   \
    break
  switch (shell_class) {
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kSsssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kPsssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kPspsShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kPpssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kDsssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kPppsShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kDspsShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kDpssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kFsssShellClass);
    default:
      break;
  }
#undef GENERATIVEQC_LOW_ORDER_SOURCES_CASE
}

}  // namespace generativeqc::scf::cuda_execution

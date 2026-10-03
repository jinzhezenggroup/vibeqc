#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "generated_direct_high_order_pair_gradient.cuh"
#include "generated_direct_source_contraction.cuh"
#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_force_order2.cuh"
#include "scf/cuda/direct_force_scatter.cuh"
#include "scf/cuda/direct_gradient_types.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct force quartet contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace generativeqc::scf::cuda_execution {

template <bool Unrestricted, unsigned AngularOrder, bool SeparateSources = false>
__device__ __forceinline__ void contract_two_electron_force_quartet_subtile_scaled(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask, double coulomb_coefficient,
    double exchange_coefficient, std::size_t active_subtile, unsigned ao_quartet_lane) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  static_assert(AngularOrder >= 2U, "order-0/1 Direct force uses generated exact shell tasks");
  static_assert(AngularOrder != 3U,
                "order-three Direct force is owned by generated shell-task workers");
  constexpr std::size_t subtiles_per_tile = detail::direct_quartet_subtiles_per_tile(AngularOrder);
  const std::size_t active_tile = active_subtile / subtiles_per_tile;
  // Consume the identical compact tile list as direct Fock so energy and
  // derivative screening cover precisely the same AO-quartet domain.
  if (active_tile >= static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const std::size_t subtile = active_subtile % subtiles_per_tile;
  const ActiveShellQuartetTile task = active_shell_quartet_tiles[active_tile];
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[first_shell], batch.shell_atoms[second_shell],
      batch.shell_atoms[third_shell], batch.shell_atoms[fourth_shell]};
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  // Generated consumers contract the complete exact class independently.
  // The host-selected bit mask keeps the generic fallback active for classes
  // disabled during production bisection.
  if (shell_class < 64U && (generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    return;
  }

  const std::size_t ordinal = static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize +
                              subtile * detail::kDirectQuartetThreads + ao_quartet_lane;
  if (ordinal < ao_quartet_count) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    std::size_t l = 0;
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, i, j);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, k, l);
    if (schwarz_bounds[physical_offset + matrix_index(i, j, n)] *
            schwarz_bounds[physical_offset + matrix_index(k, l, n)] <
        screening_tolerance) {
      return;
    }

    // Full-range J/K share one derivative; keep their independently observable
    // components separate instead of repeating the invariant ERI recurrence.
    const double coefficient = direct_force_density_coefficient_scaled<Unrestricted>(
        n, physical_offset, spin_offset, density, i, j, k, l, coulomb_coefficient,
        SeparateSources ? 0.0 : exchange_coefficient);
    double exchange_weight = 0.0;
    if constexpr (SeparateSources) {
      exchange_weight = direct_force_density_coefficient_scaled<Unrestricted>(
          n, physical_offset, spin_offset, density, i, j, k, l, 0.0, exchange_coefficient);
    }
    if (coefficient == 0.0 && exchange_weight == 0.0) return;
    const double source_coefficients[2] = {coefficient, exchange_weight};
    constexpr unsigned source_count = SeparateSources ? 2U : 1U;
    const std::size_t source_stride = static_cast<std::size_t>(batch.total_atoms) * 3U;

    // An ERI is invariant when all four basis centers translate together, so
    // its derivatives over the unique participating atoms sum to zero. Build
    // that unique list, evaluate only N-1 centers, and recover the final one
    // from the negative sum. This halves two-center work and removes one third
    // of three-center work without changing the analytic-gradient contract.
    std::int32_t unique_center_atoms[4];
    const unsigned unique_center_count =
        direct_force_unique_center_atoms(center_atoms, unique_center_atoms);
    double explicit_unique_gradient[4][3]{};
    if constexpr (AngularOrder == 2U || (AngularOrder >= 4U && AngularOrder <= 6U)) {
      CartesianQuartetGradient explicit_gradient{};
      if constexpr (AngularOrder == 2) {
        explicit_gradient = contracted_eri_cartesian_source_order2_generated_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 4) {
        explicit_gradient = contracted_eri_cartesian_source_order4_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 5) {
        explicit_gradient = contracted_eri_cartesian_source_order5_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else {
        explicit_gradient = contracted_eri_cartesian_source_order6_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      }
      for (unsigned center = 0; center < 4; ++center) {
        unsigned unique_center = 0;
        while (unique_center_atoms[unique_center] != center_atoms[center]) {
          ++unique_center;
        }
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          explicit_unique_gradient[unique_center][coordinate] +=
              explicit_gradient.center[center][coordinate];
        }
      }
    }
    double derivative_sum_x = 0.0;
    double derivative_sum_y = 0.0;
    double derivative_sum_z = 0.0;
    for (unsigned center = 0; center + 1 < unique_center_count; ++center) {
      const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[center]) * 3;
      double derivative_x = 0.0;
      double derivative_y = 0.0;
      double derivative_z = 0.0;
      if constexpr (AngularOrder == 2U || (AngularOrder >= 4U && AngularOrder <= 6U)) {
        derivative_x = explicit_unique_gradient[center][0];
        derivative_y = explicit_unique_gradient[center][1];
        derivative_z = explicit_unique_gradient[center][2];
      } else {
        const Dual3 derivative =
            dispatch_contracted_eri_cartesian_source_shell_class<AngularOrder, Dual3>(
                shell_class, batch, system, static_cast<std::int32_t>(i),
                static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
                static_cast<std::int32_t>(l), coordinate);
        derivative_x = derivative.derivative_x;
        derivative_y = derivative.derivative_y;
        derivative_z = derivative.derivative_z;
      }
      derivative_sum_x += derivative_x;
      derivative_sum_y += derivative_y;
      derivative_sum_z += derivative_z;
      for (unsigned source = 0; source < source_count; ++source) {
        const double weight = source_coefficients[source];
        double* output = forces + source * source_stride + coordinate;
        if (weight != 0.0 && derivative_x != 0.0) {
          atomicAdd(output, -weight * derivative_x);
        }
        if (weight != 0.0 && derivative_y != 0.0) {
          atomicAdd(output + 1, -weight * derivative_y);
        }
        if (weight != 0.0 && derivative_z != 0.0) {
          atomicAdd(output + 2, -weight * derivative_z);
        }
      }
    }
    if (unique_center_count > 1) {
      const std::int64_t final_coordinate =
          static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
      for (unsigned source = 0; source < source_count; ++source) {
        const double weight = source_coefficients[source];
        double* output = forces + source * source_stride + final_coordinate;
        if (weight != 0.0 && derivative_sum_x != 0.0) {
          atomicAdd(output, weight * derivative_sum_x);
        }
        if (weight != 0.0 && derivative_sum_y != 0.0) {
          atomicAdd(output + 1, weight * derivative_sum_y);
        }
        if (weight != 0.0 && derivative_sum_z != 0.0) {
          atomicAdd(output + 2, weight * derivative_sum_z);
        }
      }
    }
  }
}

/**
 * Generic Cartesian source fallback for range-separated exchange derivatives.
 *
 * Full-range Direct force keeps its qualified generated/specialized kernels.
 * SR/LR work reuses the same compact shell tasks and density contraction but
 * evaluates the explicit radial operator through the shared range-moment
 * Cartesian source recurrence.
 */
template <bool Unrestricted, unsigned AngularOrder, int PackagedShellClass = -1,
          generativeqc::integrals::CoulombRange PackagedRange =
              generativeqc::integrals::CoulombRange::Full,
          int PackagedOmegaMilli = 0>
__device__ __forceinline__ void contract_two_electron_force_quartet_subtile_range_impl(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    double exchange_coefficient, generativeqc::integrals::CoulombRange range, double omega,
    std::size_t active_subtile, unsigned ao_quartet_lane) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  static_assert(PackagedShellClass == -1 ||
                (PackagedShellClass >= 0 &&
                 PackagedShellClass < static_cast<int>(detail::kDirectQuartetShellClassCount)));
  if constexpr (PackagedShellClass >= 0) {
    static_assert(PackagedRange != generativeqc::integrals::CoulombRange::Full);
    static_assert(PackagedOmegaMilli > 0);
    range = PackagedRange;
    omega = static_cast<double>(PackagedOmegaMilli) / 1000.0;
  }
  constexpr std::size_t subtiles_per_tile = detail::direct_quartet_subtiles_per_tile(AngularOrder);
  const std::size_t active_tile = active_subtile / subtiles_per_tile;
  if (active_tile >= static_cast<std::size_t>(*active_shell_quartet_tile_count)) return;
  if (range == generativeqc::integrals::CoulombRange::Full) return;

  const std::size_t subtile = active_subtile % subtiles_per_tile;
  const ActiveShellQuartetTile task = active_shell_quartet_tiles[active_tile];
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;

  const std::size_t ordinal = static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize +
                              subtile * detail::kDirectQuartetThreads + ao_quartet_lane;
  if (ordinal >= ao_quartet_count) return;

  std::size_t first_ao_pair = 0;
  std::size_t second_ao_pair = 0;
  if (same_shell_pair) {
    decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
  } else {
    first_ao_pair = ordinal / second_ao_pair_count;
    second_ao_pair = ordinal % second_ao_pair_count;
  }
  std::size_t i = 0, j = 0, k = 0, l = 0;
  decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, i, j);
  decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, k, l);
  if (schwarz_bounds[physical_offset + matrix_index(i, j, n)] *
          schwarz_bounds[physical_offset + matrix_index(k, l, n)] <
      screening_tolerance)
    return;

  const double coefficient = direct_force_density_coefficient_scaled<Unrestricted>(
      n, physical_offset, spin_offset, density, i, j, k, l, 0.0, exchange_coefficient);
  if (coefficient == 0.0) return;

  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[first_shell], batch.shell_atoms[second_shell],
      batch.shell_atoms[third_shell], batch.shell_atoms[fourth_shell]};
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if constexpr (PackagedShellClass >= 0) {
    if (shell_class != static_cast<unsigned>(PackagedShellClass)) return;
  }
  std::int32_t unique_center_atoms[4];
  const unsigned unique_center_count =
      direct_force_unique_center_atoms(center_atoms, unique_center_atoms);
  if (unique_center_count <= 1U) return;

  // Orders 4--6 share the compiler's all-center recurrence with one LR
  // moment ladder per primitive/AO quartet, instead of repeating Dual3 for
  // each unique atom. Short range and other orders keep their current owner.
  double explicit_unique_gradient[4][3]{};
  bool shared_long_range = false;
  if constexpr (AngularOrder >= 4U && AngularOrder <= 6U &&
                (PackagedShellClass < 0 ||
                 PackagedRange == generativeqc::integrals::CoulombRange::Long)) {
    if (range == generativeqc::integrals::CoulombRange::Long) {
      CartesianQuartetGradient gradient{};
      if constexpr (AngularOrder == 4U) {
        gradient = contracted_eri_cartesian_source_order4_gradient<true>(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), omega);
      } else if constexpr (AngularOrder == 5U) {
        gradient = contracted_eri_cartesian_source_order5_gradient<true>(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), omega);
      } else {
        gradient = contracted_eri_cartesian_source_order6_gradient<true>(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), omega);
      }
      // The generated consumer restores raw shell order. Combine repeated
      // atoms before restoring the last unique atom by translation invariance.
      for (unsigned shell_center = 0; shell_center < 4; ++shell_center) {
        unsigned atom = 0;
        while (unique_center_atoms[atom] != center_atoms[shell_center]) ++atom;
        for (unsigned axis = 0; axis < 3; ++axis)
          explicit_unique_gradient[atom][axis] += gradient.center[shell_center][axis];
      }
      shared_long_range = true;
    }
  }

  double derivative_sum[3]{};
  for (unsigned center = 0; center + 1U < unique_center_count; ++center) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[center]) * 3;
    Dual3 derivative{};
    if (shared_long_range) {
      derivative.derivative_x = explicit_unique_gradient[center][0];
      derivative.derivative_y = explicit_unique_gradient[center][1];
      derivative.derivative_z = explicit_unique_gradient[center][2];
    } else if constexpr (PackagedShellClass >= 0) {
      constexpr double packaged_omega = static_cast<double>(PackagedOmegaMilli) / 1000.0;
      derivative =
          contracted_eri_cartesian_source_shell_class<static_cast<unsigned>(PackagedShellClass),
                                                      Dual3>(
              batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
              static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate, PackagedRange,
              packaged_omega);
    } else {
      derivative = dispatch_contracted_eri_cartesian_source_shell_class<AngularOrder, Dual3>(
          shell_class, batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
          static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate, range, omega);
    }
    const double value[3] = {
        derivative.derivative_x,
        derivative.derivative_y,
        derivative.derivative_z,
    };
#pragma unroll
    for (unsigned axis = 0; axis < 3; ++axis) {
      derivative_sum[axis] += value[axis];
      if (value[axis] != 0.0)
        atomicAdd(forces + static_cast<std::size_t>(coordinate) + axis, -coefficient * value[axis]);
    }
  }

  const std::size_t final_coordinate =
      static_cast<std::size_t>(unique_center_atoms[unique_center_count - 1U]) * 3U;
#pragma unroll
  for (unsigned axis = 0; axis < 3; ++axis)
    if (derivative_sum[axis] != 0.0)
      atomicAdd(forces + final_coordinate + axis, coefficient * derivative_sum[axis]);
}

/**
 * Fused RSH stationary sources over one screened shell quartet.
 *
 * Full-range Coulomb/exchange share one derivative evaluation and LR exchange
 * adds one range-moment evaluation. SR exchange is reconstructed as Full-LR,
 * so J', SR-K' and LR-K' share one compact shell traversal.
 */
template <bool Unrestricted, unsigned AngularOrder>
__device__ __forceinline__ void contract_two_electron_force_quartet_subtile_range_scaled(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    double exchange_coefficient, generativeqc::integrals::CoulombRange range, double omega,
    std::size_t active_subtile, unsigned ao_quartet_lane) {
  contract_two_electron_force_quartet_subtile_range_impl<Unrestricted, AngularOrder>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
      schwarz_bounds, density, active, forces, exchange_coefficient, range, omega, active_subtile,
      ao_quartet_lane);
}

template <bool Unrestricted, unsigned ShellClass, generativeqc::integrals::CoulombRange Range,
          int OmegaMilli>
__device__ inline __noinline__ void
contract_two_electron_force_quartet_subtile_range_shell_aot_scaled(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    double exchange_coefficient, std::size_t active_subtile, unsigned ao_quartet_lane) {
  constexpr unsigned angular_order = direct_shell_class_angular_order(ShellClass);
  constexpr double omega = static_cast<double>(OmegaMilli) / 1000.0;
  contract_two_electron_force_quartet_subtile_range_impl<
      Unrestricted, angular_order, static_cast<int>(ShellClass), Range, OmegaMilli>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
      schwarz_bounds, density, active, forces, exchange_coefficient, Range, omega, active_subtile,
      ao_quartet_lane);
}

template <bool Unrestricted, unsigned AngularOrder>
__device__ __forceinline__ void contract_two_electron_force_quartet_subtile_rsh_scaled(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active,
    double* source_forces, double coulomb_coefficient, double short_exchange_coefficient,
    double long_exchange_coefficient, double omega, std::size_t active_subtile,
    unsigned ao_quartet_lane) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  constexpr std::size_t subtiles_per_tile = detail::direct_quartet_subtiles_per_tile(AngularOrder);
  const std::size_t active_tile = active_subtile / subtiles_per_tile;
  if (active_tile >= static_cast<std::size_t>(*active_shell_quartet_tile_count)) return;

  const std::size_t subtile = active_subtile % subtiles_per_tile;
  const ActiveShellQuartetTile task = active_shell_quartet_tiles[active_tile];
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;
  const std::size_t ordinal = static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize +
                              subtile * detail::kDirectQuartetThreads + ao_quartet_lane;
  if (ordinal >= ao_quartet_count) return;

  std::size_t first_ao_pair = 0;
  std::size_t second_ao_pair = 0;
  if (same_shell_pair) {
    decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
  } else {
    first_ao_pair = ordinal / second_ao_pair_count;
    second_ao_pair = ordinal % second_ao_pair_count;
  }
  std::size_t i = 0, j = 0, k = 0, l = 0;
  decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, i, j);
  decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, k, l);
  if (schwarz_bounds[physical_offset + matrix_index(i, j, n)] *
          schwarz_bounds[physical_offset + matrix_index(k, l, n)] <
      screening_tolerance)
    return;

  double coulomb_density = 0.0;
  if (coulomb_coefficient != 0.0)
    coulomb_density = direct_force_density_coefficient_scaled<Unrestricted>(
        n, physical_offset, spin_offset, density, i, j, k, l, 1.0, 0.0);
  double exchange_density = 0.0;
  if (short_exchange_coefficient != 0.0 || long_exchange_coefficient != 0.0)
    exchange_density = direct_force_density_coefficient_scaled<Unrestricted>(
        n, physical_offset, spin_offset, density, i, j, k, l, 0.0, 1.0);
  const double source_coefficient[3] = {
      coulomb_coefficient * coulomb_density,
      short_exchange_coefficient * exchange_density,
      long_exchange_coefficient * exchange_density,
  };
  if (source_coefficient[0] == 0.0 && source_coefficient[1] == 0.0 && source_coefficient[2] == 0.0)
    return;

  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[first_shell], batch.shell_atoms[second_shell],
      batch.shell_atoms[third_shell], batch.shell_atoms[fourth_shell]};
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  std::int32_t unique_center_atoms[4];
  const unsigned unique_center_count =
      direct_force_unique_center_atoms(center_atoms, unique_center_atoms);
  if (unique_center_count <= 1U) return;

  const bool want_full = source_coefficient[0] != 0.0 || source_coefficient[1] != 0.0;
  const bool want_long = source_coefficient[1] != 0.0 || source_coefficient[2] != 0.0;
  double explicit_unique_gradient[4][3]{};
  if constexpr (AngularOrder == 2U || (AngularOrder >= 4U && AngularOrder <= 6U)) {
    if (want_full) {
      CartesianQuartetGradient explicit_gradient{};
      if constexpr (AngularOrder == 2U) {
        explicit_gradient = contracted_eri_cartesian_source_order2_generated_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 4U) {
        explicit_gradient = contracted_eri_cartesian_source_order4_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 5U) {
        explicit_gradient = contracted_eri_cartesian_source_order5_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else {
        explicit_gradient = contracted_eri_cartesian_source_order6_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      }
      for (unsigned shell_center = 0; shell_center < 4; ++shell_center) {
        unsigned unique_center = 0;
        while (unique_center_atoms[unique_center] != center_atoms[shell_center]) ++unique_center;
        for (unsigned axis = 0; axis < 3; ++axis)
          explicit_unique_gradient[unique_center][axis] +=
              explicit_gradient.center[shell_center][axis];
      }
    }
  }

  const std::size_t source_stride = static_cast<std::size_t>(batch.total_atoms) * 3U;
  double reconstructed[3][3]{};
  for (unsigned center = 0; center + 1U < unique_center_count; ++center) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[center]) * 3;
    Dual3 full{};
    Dual3 long_range{};
    if (want_full) {
      if constexpr (AngularOrder == 2U || (AngularOrder >= 4U && AngularOrder <= 6U)) {
        full.derivative_x = explicit_unique_gradient[center][0];
        full.derivative_y = explicit_unique_gradient[center][1];
        full.derivative_z = explicit_unique_gradient[center][2];
      } else {
        full = dispatch_contracted_eri_cartesian_source_shell_class<AngularOrder, Dual3>(
            shell_class, batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate);
      }
    }
    if (want_long) {
      long_range = dispatch_contracted_eri_cartesian_source_shell_class<AngularOrder, Dual3>(
          shell_class, batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
          static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate,
          generativeqc::integrals::CoulombRange::Long, omega);
    }
    const double full_value[3] = {
        full.derivative_x,
        full.derivative_y,
        full.derivative_z,
    };
    const double long_value[3] = {
        long_range.derivative_x,
        long_range.derivative_y,
        long_range.derivative_z,
    };
#pragma unroll
    for (unsigned axis = 0; axis < 3; ++axis) {
      const double contribution[3] = {
          source_coefficient[0] * full_value[axis],
          source_coefficient[1] * (full_value[axis] - long_value[axis]),
          source_coefficient[2] * long_value[axis],
      };
#pragma unroll
      for (unsigned source = 0; source < 3; ++source) {
        reconstructed[source][axis] += contribution[source];
        if (contribution[source] != 0.0)
          atomicAdd(
              source_forces + source * source_stride + static_cast<std::size_t>(coordinate) + axis,
              -contribution[source]);
      }
    }
  }

  const std::size_t final_coordinate =
      static_cast<std::size_t>(unique_center_atoms[unique_center_count - 1U]) * 3U;
#pragma unroll
  for (unsigned source = 0; source < 3; ++source)
#pragma unroll
    for (unsigned axis = 0; axis < 3; ++axis)
      if (reconstructed[source][axis] != 0.0)
        atomicAdd(source_forces + source * source_stride + final_coordinate + axis,
                  reconstructed[source][axis]);
}

template <bool Unrestricted, unsigned AngularOrder>
__device__ __forceinline__ void contract_two_electron_force_quartet_subtile(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask, std::size_t active_subtile,
    unsigned ao_quartet_lane) {
  constexpr double exchange_coefficient = Unrestricted ? -1.0 : -0.5;
  contract_two_electron_force_quartet_subtile_scaled<Unrestricted, AngularOrder>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
      schwarz_bounds, density, active, forces, generated_shell_class_mask, 1.0,
      exchange_coefficient, active_subtile, ao_quartet_lane);
}

}  // namespace generativeqc::scf::cuda_execution

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_native_gradient_types.cuh"
#include "scf/cuda/direct_native_order01_gradient.cuh"
#include "scf/cuda/direct_native_order2_gradient.cuh"
#include "scf/cuda/direct_native_order3_gradient.cuh"
#include "scf/cuda/direct_native_order456_gradient.cuh"
#include "scf/cuda/direct_native_source_contraction.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct force quartet contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

template <bool Unrestricted, unsigned AngularOrder>
__device__ __forceinline__ void contract_two_electron_force_quartet_subtile(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask, std::size_t active_subtile,
    unsigned ao_quartet_lane) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
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

    const double coefficient = direct_force_density_coefficient<Unrestricted>(
        n, physical_offset, spin_offset, density, i, j, k, l);
    if (coefficient == 0.0) return;

    // An ERI is invariant when all four basis centers translate together, so
    // its derivatives over the unique participating atoms sum to zero. Build
    // that unique list, evaluate only N-1 centers, and recover the final one
    // from the negative sum. This halves two-center work and removes one third
    // of three-center work without changing the analytic-gradient contract.
    std::int32_t unique_center_atoms[4];
    unsigned unique_center_count = 0;
    for (unsigned center = 0; center < 4; ++center) {
      bool duplicate_center = false;
      for (unsigned previous = 0; previous < unique_center_count; ++previous) {
        duplicate_center =
            duplicate_center || center_atoms[center] == unique_center_atoms[previous];
      }
      if (!duplicate_center) {
        unique_center_atoms[unique_center_count++] = center_atoms[center];
      }
    }
    double explicit_unique_gradient[4][3]{};
    if constexpr (AngularOrder <= 6) {
      CartesianQuartetGradient explicit_gradient{};
      if constexpr (AngularOrder <= 1) {
        explicit_gradient = contracted_eri_cartesian_source_order01_gradient<AngularOrder>(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 2) {
        explicit_gradient = contracted_eri_cartesian_source_order2_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 3) {
        explicit_gradient = contracted_eri_cartesian_source_order3_gradient(
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
      if constexpr (AngularOrder <= 6) {
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
      if (derivative_x != 0.0) {
        atomicAdd(forces + coordinate, -coefficient * derivative_x);
      }
      if (derivative_y != 0.0) {
        atomicAdd(forces + coordinate + 1, -coefficient * derivative_y);
      }
      if (derivative_z != 0.0) {
        atomicAdd(forces + coordinate + 2, -coefficient * derivative_z);
      }
    }
    if (unique_center_count > 1) {
      const std::int64_t final_coordinate =
          static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
      if (derivative_sum_x != 0.0) {
        atomicAdd(forces + final_coordinate, coefficient * derivative_sum_x);
      }
      if (derivative_sum_y != 0.0) {
        atomicAdd(forces + final_coordinate + 1, coefficient * derivative_sum_y);
      }
      if (derivative_sum_z != 0.0) {
        atomicAdd(forces + final_coordinate + 2, coefficient * derivative_sum_z);
      }
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution

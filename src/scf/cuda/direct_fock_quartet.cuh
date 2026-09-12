#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_fock_accumulation.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_native_source_contraction.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct fock quartet contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

template <bool Unrestricted, unsigned AngularOrder, typename EvalScalar = double>
__device__ __forceinline__ void contract_fock_direct_quartet_subtile(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask, std::size_t active_subtile,
    unsigned ao_quartet_lane) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  constexpr std::size_t subtiles_per_tile = detail::direct_quartet_subtiles_per_tile(AngularOrder);
  const std::size_t active_tile = active_subtile / subtiles_per_tile;
  if (active_tile >= static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const std::size_t subtile = active_subtile % subtiles_per_tile;
  const ActiveShellQuartetTile task = active_shell_quartet_tiles[active_tile];
  if (task.first_pair >= static_cast<std::uint32_t>(batch.total_shell_pairs) ||
      task.second_pair >= static_cast<std::uint32_t>(batch.total_shell_pairs)) {
    return;
  }
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (system < 0 || system >= batch.batch_size || batch.shell_pair_systems[second_pair] != system) {
    return;
  }
  if (active != nullptr && active[system] == 0) return;
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (generated_fock_shell_class_mask != nullptr &&
      ((*generated_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U)) {
    return;
  }

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
  if (ordinal < ao_quartet_count) {
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    std::size_t l = 0;
    if (!decode_direct_tile_ao_ordinal(batch, task, ordinal, first_ao_pair_count,
                                       second_ao_pair_count, system_ao_begin, n, i, j, k, l)) {
      return;
    }
    if (schwarz_bounds[physical_offset + matrix_index(i, j, n)] *
            schwarz_bounds[physical_offset + matrix_index(k, l, n)] <
        screening_tolerance) {
      return;
    }
    const EvalScalar evaluated_integral =
        dispatch_contracted_eri_cartesian_source_shell_class<AngularOrder, EvalScalar>(
            shell_class, batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), -1);
    const double integral = scalar_value(evaluated_integral);
    if (integral == 0.0) return;
    accumulate_direct_fock_integral<Unrestricted>(n, physical_offset, spin_offset, density, fock, i,
                                                  j, k, l, integral);
  }
}

}  // namespace vibeqc::scf::cuda_execution

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_native_dsss_gradient.cuh"
#include "scf/cuda/direct_native_gradient_types.cuh"
#include "scf/cuda/direct_native_ppss_gradient.cuh"
#include "scf/cuda/direct_native_psps_gradient.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct force order2 contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

/** Evaluate and write one complete density-weighted psps force shell task. */
template <bool Unrestricted>
__device__ inline __noinline__ void contract_two_electron_force_psps_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  // A psps shell quartet has at most nine Cartesian AO quartets and therefore
  // always fits in the first compact tile.
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class != kPspsShellClass) return;
  if ((generated_shell_class_mask & (std::uint64_t{1} << kPspsShellClass)) != 0U) {
    return;
  }

  unsigned first_p_slot = 4;
  unsigned second_p_slot = 4;
  for (unsigned slot = 0; slot < 2; ++slot) {
    if (batch.shell_angular[raw_shell[slot]] == 1U) first_p_slot = slot;
  }
  for (unsigned slot = 2; slot < 4; ++slot) {
    if (batch.shell_angular[raw_shell[slot]] == 1U) second_p_slot = slot;
  }
  if (first_p_slot >= 2 || second_p_slot < 2 || second_p_slot >= 4) return;
  const unsigned canonical_raw_slot[4] = {
      first_p_slot,
      1U - first_p_slot,
      second_p_slot,
      5U - second_p_slot,
  };
  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]],
      raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]],
      raw_shell[canonical_raw_slot[3]],
  };

  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    const std::int32_t atom = batch.shell_atoms[canonical_shell[center]];
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center || atom == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = atom;
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_p_ao_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[0]]) - system_ao_begin;
  const std::size_t second_p_ao_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[2]]) - system_ao_begin;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, raw_ao[2], raw_ao[3]);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    const double density_coefficient = direct_force_density_coefficient<Unrestricted>(
        n, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3]);
    if (density_coefficient == 0.0) continue;
    const unsigned first_axis =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[0]] - first_p_ao_begin);
    const unsigned second_axis =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[2]] - second_p_ao_begin);
    if (first_axis >= 3 || second_axis >= 3) return;
    const double angular_coefficient = batch.direct_ao_coefficients[system_ao_begin + raw_ao[0]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[1]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[2]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[3]];
    component_weight[first_axis * 3 + second_axis] += density_coefficient * angular_coefficient;
    any_component = true;
  }
  if (!any_component) return;

  const PspsWeightedGradient gradient = contracted_eri_cartesian_source_psps_weighted_gradient(
      batch, first_pair, second_pair, canonical_shell[0], canonical_shell[1], canonical_shell[2],
      canonical_shell[3], component_weight);
  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned canonical = 0; canonical < 3; ++canonical) {
        const double value = gradient.center[canonical][axis];
        fourth_derivative -= value;
        if (batch.shell_atoms[canonical_shell[canonical]] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (batch.shell_atoms[canonical_shell[3]] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

/** Evaluate one closed ppss or dsss shell task over its exact AO domain. */
template <bool Unrestricted, unsigned TargetShellClass>
__device__ inline __noinline__ void contract_two_electron_force_pair_order2_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  static_assert(TargetShellClass == kPpssShellClass || TargetShellClass == kDsssShellClass);
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class != TargetShellClass) return;
  if ((generated_shell_class_mask & (std::uint64_t{1} << TargetShellClass)) != 0U) {
    return;
  }

  unsigned canonical_raw_slot[4];
  if constexpr (TargetShellClass == kPpssShellClass) {
    const bool first_pair_is_pp =
        batch.shell_angular[raw_shell[0]] == 1U && batch.shell_angular[raw_shell[1]] == 1U;
    const unsigned pair_begin = first_pair_is_pp ? 0U : 2U;
    const unsigned other_pair_begin = first_pair_is_pp ? 2U : 0U;
    canonical_raw_slot[0] = pair_begin;
    canonical_raw_slot[1] = pair_begin + 1U;
    canonical_raw_slot[2] = other_pair_begin;
    canonical_raw_slot[3] = other_pair_begin + 1U;
  } else {
    unsigned d_slot = 4U;
    for (unsigned slot = 0; slot < 4; ++slot) {
      if (batch.shell_angular[raw_shell[slot]] == 2U) d_slot = slot;
    }
    if (d_slot >= 4U) return;
    const unsigned pair_begin = d_slot < 2U ? 0U : 2U;
    const unsigned other_pair_begin = pair_begin == 0U ? 2U : 0U;
    canonical_raw_slot[0] = d_slot;
    canonical_raw_slot[1] = pair_begin + (d_slot == pair_begin ? 1U : 0U);
    canonical_raw_slot[2] = other_pair_begin;
    canonical_raw_slot[3] = other_pair_begin + 1U;
  }
  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]],
      raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]],
      raw_shell[canonical_raw_slot[3]],
  };
  const std::size_t canonical_pair[2] = {
      canonical_raw_slot[0] < 2U ? first_pair : second_pair,
      canonical_raw_slot[2] < 2U ? first_pair : second_pair,
  };

  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    const std::int32_t atom = batch.shell_atoms[canonical_shell[center]];
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center || atom == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = atom;
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_component_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[0]]) - system_ao_begin;
  const std::size_t second_component_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[1]]) - system_ao_begin;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, raw_ao[2], raw_ao[3]);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    const double density_coefficient = direct_force_density_coefficient<Unrestricted>(
        n, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3]);
    if (density_coefficient == 0.0) continue;
    const unsigned first_component =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[0]] - first_component_begin);
    unsigned output = first_component;
    if constexpr (TargetShellClass == kPpssShellClass) {
      const unsigned second_component =
          static_cast<unsigned>(raw_ao[canonical_raw_slot[1]] - second_component_begin);
      if (first_component >= 3U || second_component >= 3U) return;
      output = first_component * 3U + second_component;
    } else if (first_component >= 6U) {
      return;
    }
    const double angular_coefficient = batch.direct_ao_coefficients[system_ao_begin + raw_ao[0]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[1]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[2]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[3]];
    component_weight[output] += density_coefficient * angular_coefficient;
    any_component = true;
  }
  if (!any_component) return;

  PspsWeightedGradient gradient{};
  if constexpr (TargetShellClass == kPpssShellClass) {
    gradient = contracted_eri_cartesian_source_ppss_weighted_gradient(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  } else {
    gradient = contracted_eri_cartesian_source_dsss_weighted_gradient(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  }

  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned canonical = 0; canonical < 3; ++canonical) {
        const double value = gradient.center[canonical][axis];
        fourth_derivative -= value;
        if (batch.shell_atoms[canonical_shell[canonical]] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (batch.shell_atoms[canonical_shell[3]] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution

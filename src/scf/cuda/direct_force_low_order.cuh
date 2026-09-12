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
#include "scf/cuda/direct_native_psss.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct force low order contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

/** Evaluate and write one complete density-weighted ssss force shell task. */
template <bool Unrestricted>
__device__ __forceinline__ void contract_two_electron_force_ssss_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  // Every s shell contains one Cartesian AO, so a valid ssss shell quartet
  // occupies exactly the first compact tile and needs no AO-pair decoding.
  if (task.tile != 0U) return;
  if ((generated_shell_class_mask & std::uint64_t{1}) != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t shells[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  for (unsigned slot = 0; slot < 4; ++slot) {
    if (batch.shell_angular[shells[slot]] != 0U) return;
  }

  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[shells[0]],
      batch.shell_atoms[shells[1]],
      batch.shell_atoms[shells[2]],
      batch.shell_atoms[shells[3]],
  };
  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center || center_atoms[center] == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = center_atoms[center];
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t ao[4] = {
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[0]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[1]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[2]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[3]]) - system_ao_begin,
  };
  if (schwarz_bounds[physical_offset + matrix_index(ao[0], ao[1], n)] *
          schwarz_bounds[physical_offset + matrix_index(ao[2], ao[3], n)] <
      screening_tolerance) {
    return;
  }
  const double density_coefficient = direct_force_density_coefficient<Unrestricted>(
      n, physical_offset, spin_offset, density, ao[0], ao[1], ao[2], ao[3]);
  if (density_coefficient == 0.0) return;
  const double component_weight = density_coefficient *
                                  batch.direct_ao_coefficients[system_ao_begin + ao[0]] *
                                  batch.direct_ao_coefficients[system_ao_begin + ao[1]] *
                                  batch.direct_ao_coefficients[system_ao_begin + ao[2]] *
                                  batch.direct_ao_coefficients[system_ao_begin + ao[3]];
  const SsssWeightedGradient gradient = contracted_eri_cartesian_source_ssss_weighted_gradient(
      batch, first_pair, second_pair, shells[0], shells[1], shells[2], shells[3], component_weight);

  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned center = 0; center < 3; ++center) {
        const double value = gradient.center[center][axis];
        fourth_derivative -= value;
        if (center_atoms[center] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (center_atoms[3] == unique_center_atoms[atom]) {
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

/** Evaluate and write one complete density-weighted psss force shell task. */
template <bool Unrestricted, bool ResidentBra = false>
__device__ inline __noinline__ void contract_two_electron_force_psss_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask,
    const PrimitivePairData* resident_first_pairs = nullptr,
    std::int64_t resident_first_pair_count = 0) {
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
  if (shell_class < 64U && (generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    return;
  }
  unsigned p_slot = 4;
  unsigned p_count = 0;
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[raw_shell[slot]];
    if (angular == 1) {
      p_slot = slot;
      ++p_count;
    } else if (angular != 0) {
      return;
    }
  }
  if (p_count != 1) return;

  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[raw_shell[0]],
      batch.shell_atoms[raw_shell[1]],
      batch.shell_atoms[raw_shell[2]],
      batch.shell_atoms[raw_shell[3]],
  };
  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center || center_atoms[center] == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = center_atoms[center];
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  std::size_t raw_ao[4] = {
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[0]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[1]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[2]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[3]]) - system_ao_begin,
  };
  const std::size_t p_ao_begin = raw_ao[p_slot];
  double density_coefficient[3]{};
  for (unsigned axis = 0; axis < 3; ++axis) {
    raw_ao[p_slot] = p_ao_begin + axis;
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    density_coefficient[axis] = direct_force_density_coefficient<Unrestricted>(
        n, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3]);
  }
  if (density_coefficient[0] == 0.0 && density_coefficient[1] == 0.0 &&
      density_coefficient[2] == 0.0) {
    return;
  }

  std::int32_t slots[4] = {raw_shell[0], raw_shell[1], raw_shell[2], raw_shell[3]};
  std::size_t canonical_pair[2] = {first_pair, second_pair};
  if (p_slot == 1) {
    const std::int32_t swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  } else if (p_slot >= 2) {
    if (p_slot == 3) {
      const std::int32_t swap = slots[2];
      slots[2] = slots[3];
      slots[3] = swap;
    }
    const std::int32_t first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const std::int32_t second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
    const std::size_t pair_swap = canonical_pair[0];
    canonical_pair[0] = canonical_pair[1];
    canonical_pair[1] = pair_swap;
  }

  const PsssWeightedGradient gradient =
      contracted_eri_cartesian_source_psss_weighted_gradient<ResidentBra>(
          batch, canonical_pair[0], canonical_pair[1], slots[0], slots[1], slots[2], slots[3],
          density_coefficient, resident_first_pairs, resident_first_pair_count);
  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned canonical = 0; canonical < 3; ++canonical) {
        const double value = gradient.center[canonical][axis];
        fourth_derivative -= value;
        if (batch.shell_atoms[slots[canonical]] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (batch.shell_atoms[slots[3]] == unique_center_atoms[atom]) {
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

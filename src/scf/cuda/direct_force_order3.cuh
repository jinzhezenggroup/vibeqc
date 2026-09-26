#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <weighted_eri.cuh>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

namespace vibeqc::scf::cuda_execution {

namespace order3_detail {

__device__ inline unsigned cartesian_component_count(unsigned angular) {
  return (angular + 1U) * (angular + 2U) / 2U;
}

__device__ inline unsigned pair_class(unsigned first, unsigned second) {
  if (first < second) {
    const unsigned swap = first;
    first = second;
    second = swap;
  }
  return first * (first + 1U) / 2U + second;
}

__device__ inline void canonicalize_shell_slots(const DeviceBatch& batch,
                                                const std::int32_t (&raw_shell)[4],
                                                unsigned (&canonical_raw_slot)[4]) {
  canonical_raw_slot[0] = 0U;
  canonical_raw_slot[1] = 1U;
  canonical_raw_slot[2] = 2U;
  canonical_raw_slot[3] = 3U;
  if (batch.shell_angular[raw_shell[canonical_raw_slot[0]]] <
      batch.shell_angular[raw_shell[canonical_raw_slot[1]]]) {
    const unsigned swap = canonical_raw_slot[0];
    canonical_raw_slot[0] = canonical_raw_slot[1];
    canonical_raw_slot[1] = swap;
  }
  if (batch.shell_angular[raw_shell[canonical_raw_slot[2]]] <
      batch.shell_angular[raw_shell[canonical_raw_slot[3]]]) {
    const unsigned swap = canonical_raw_slot[2];
    canonical_raw_slot[2] = canonical_raw_slot[3];
    canonical_raw_slot[3] = swap;
  }
  const unsigned first_pair_class =
      pair_class(batch.shell_angular[raw_shell[canonical_raw_slot[0]]],
                 batch.shell_angular[raw_shell[canonical_raw_slot[1]]]);
  const unsigned second_pair_class =
      pair_class(batch.shell_angular[raw_shell[canonical_raw_slot[2]]],
                 batch.shell_angular[raw_shell[canonical_raw_slot[3]]]);
  if (first_pair_class < second_pair_class) {
    const unsigned first_swap = canonical_raw_slot[0];
    canonical_raw_slot[0] = canonical_raw_slot[2];
    canonical_raw_slot[2] = first_swap;
    const unsigned second_swap = canonical_raw_slot[1];
    canonical_raw_slot[1] = canonical_raw_slot[3];
    canonical_raw_slot[3] = second_swap;
  }
}

}  // namespace order3_detail

template <unsigned TargetShellClass>
__device__ inline __noinline__ generated_weighted_eri::IndependentGradient
contracted_eri_cartesian_source_order3_generated_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t first_shell, std::int32_t second_shell, std::int32_t third_shell,
    std::int32_t fourth_shell, const double* component_weight) {
  static_assert(TargetShellClass == kPppsShellClass || TargetShellClass == kDspsShellClass ||
                TargetShellClass == kDpssShellClass || TargetShellClass == kFsssShellClass);

  const Vec3<double> position[4] = {
      atom_position<double>(batch, batch.shell_atoms[first_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[second_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[third_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1),
  };
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == first_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_shell;
  const std::int64_t first_pair_begin = batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];

  generated_weighted_eri::IndependentGradient result{};
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_data = batch.shell_primitive_pairs[first_primitive];
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_data = batch.shell_primitive_pairs[second_primitive];
      generated_weighted_eri::Geometry geometry;
      const double boys_argument = generated_weighted_eri::make_direct_cached_geometry(
          first_data, second_data, !first_pair_matches_canonical_order,
          !second_pair_matches_canonical_order, position[0], position[1], position[2], position[3],
          geometry);
      boys_values<4>(boys_argument, geometry.boys);

      generated_weighted_eri::IndependentGradient primitive{};
      if constexpr (TargetShellClass == kPppsShellClass) {
        primitive = generated_weighted_eri::ppps_force(geometry, component_weight);
      } else if constexpr (TargetShellClass == kDspsShellClass) {
        primitive = generated_weighted_eri::dsps_force(geometry, component_weight);
      } else if constexpr (TargetShellClass == kDpssShellClass) {
        primitive = generated_weighted_eri::dpss_force(geometry, component_weight);
      } else {
        primitive = generated_weighted_eri::fsss_force(geometry, component_weight);
      }
#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned axis = 0; axis < 3; ++axis) {
          result.center[center][axis] += primitive.center[center][axis];
        }
      }
    }
  }
  return result;
}

template <bool Unrestricted, unsigned TargetShellClass>
__device__ inline __noinline__ void contract_two_electron_force_order3_class_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active,
    double* forces) {
  static_assert(TargetShellClass == kPppsShellClass || TargetShellClass == kDspsShellClass ||
                TargetShellClass == kDpssShellClass || TargetShellClass == kFsssShellClass);
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
  unsigned canonical_raw_slot[4];
  order3_detail::canonicalize_shell_slots(batch, raw_shell, canonical_raw_slot);
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
    if (!duplicate_center) unique_center_atoms[unique_center_count++] = atom;
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  std::size_t canonical_component_begin[4];
  unsigned canonical_component_count[4];
#pragma unroll
  for (unsigned center = 0; center < 4; ++center) {
    canonical_component_begin[center] =
        static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[center]]) -
        system_ao_begin;
    canonical_component_count[center] =
        order3_detail::cartesian_component_count(batch.shell_angular[canonical_shell[center]]);
  }

  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;

  double component_weight[27]{};
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

    unsigned component[4];
#pragma unroll
    for (unsigned center = 0; center < 4; ++center) {
      const std::size_t canonical_ao = raw_ao[canonical_raw_slot[center]];
      if (canonical_ao < canonical_component_begin[center]) return;
      component[center] = static_cast<unsigned>(canonical_ao - canonical_component_begin[center]);
      if (component[center] >= canonical_component_count[center]) return;
    }
    const unsigned output = ((component[0] * canonical_component_count[1] + component[1]) *
                                 canonical_component_count[2] +
                             component[2]) *
                                canonical_component_count[3] +
                            component[3];
    if (output >= 27U) return;

    const double angular_coefficient = batch.direct_ao_coefficients[system_ao_begin + raw_ao[0]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[1]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[2]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[3]];
    component_weight[output] += density_coefficient * angular_coefficient;
    any_component = true;
  }
  if (!any_component) return;

  const auto gradient =
      contracted_eri_cartesian_source_order3_generated_weighted_gradient<TargetShellClass>(
          batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
          canonical_shell[2], canonical_shell[3], component_weight);

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
      if (derivative != 0.0) atomicAdd(forces + coordinate + axis, -derivative);
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

template <bool Unrestricted>
__device__ inline __noinline__ void contract_two_electron_force_order3_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if ((generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) return;

  switch (shell_class) {
    case kPppsShellClass:
      contract_two_electron_force_order3_class_task<Unrestricted, kPppsShellClass>(
          batch, task, screening_tolerance, schwarz_bounds, density, active, forces);
      break;
    case kDspsShellClass:
      contract_two_electron_force_order3_class_task<Unrestricted, kDspsShellClass>(
          batch, task, screening_tolerance, schwarz_bounds, density, active, forces);
      break;
    case kDpssShellClass:
      contract_two_electron_force_order3_class_task<Unrestricted, kDpssShellClass>(
          batch, task, screening_tolerance, schwarz_bounds, density, active, forces);
      break;
    case kFsssShellClass:
      contract_two_electron_force_order3_class_task<Unrestricted, kFsssShellClass>(
          batch, task, screening_tolerance, schwarz_bounds, density, active, forces);
      break;
    default:
      break;
  }
}

}  // namespace vibeqc::scf::cuda_execution

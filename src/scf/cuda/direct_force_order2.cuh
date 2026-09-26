#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <weighted_eri.cuh>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_native_gradient_types.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct force order2 contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

/**
 * Contract one canonical order-two shell class through compiler-owned Weighted IntegralIR math.
 *
 * Queueing, screening, density folding and atom accumulation remain native scheduler concerns.
 * This adapter only binds the existing primitive-pair representation to the generated geometry
 * vocabulary; no independent ERI/gradient recurrence is retained here.
 */
template <unsigned TargetShellClass>
__device__ inline __noinline__ generated_weighted_eri::IndependentGradient
contracted_eri_cartesian_source_order2_generated_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t first_shell, std::int32_t second_shell, std::int32_t third_shell,
    std::int32_t fourth_shell, const double* component_weight) {
  static_assert(TargetShellClass == kPspsShellClass || TargetShellClass == kPpssShellClass ||
                TargetShellClass == kDsssShellClass);

  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[first_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[second_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[third_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1);
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
    const PrimitivePairData first_pair = batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const Vec3<double> product_p = first_pair.product_center;
    const double first_product_scale = first_pair_matches_canonical_order
                                           ? first_pair.first_product_scale
                                           : first_pair.second_product_scale;
    const double second_product_scale = first_pair_matches_canonical_order
                                            ? first_pair.second_product_scale
                                            : first_pair.first_product_scale;
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const Vec3<double> product_q = second_pair.product_center;
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;

      generated_weighted_eri::Geometry geometry;
      geometry.inverse_two_p = 0.5 / p;
      if constexpr (TargetShellClass == kPspsShellClass) {
        geometry.inverse_two_q = 0.5 / q;
      }
      geometry.rho = p * q / (p + q);
      geometry.prefactor = first_pair.weighted_coefficient * second_pair.weighted_coefficient *
                           2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));
      geometry.product_scales[0] = first_product_scale;
      geometry.product_scales[1] = second_product_scale;
      geometry.product_scales[2] = third_product_scale;

      const Vec3<double> difference{
          product_p.x - product_q.x,
          product_p.y - product_q.y,
          product_p.z - product_q.z,
      };
      const Vec3<double> pa{
          product_p.x - first.x,
          product_p.y - first.y,
          product_p.z - first.z,
      };
      Vec3<double> pb{};
      Vec3<double> qc{};
      if constexpr (TargetShellClass == kPpssShellClass) {
        pb = {
            product_p.x - second.x,
            product_p.y - second.y,
            product_p.z - second.z,
        };
      } else if constexpr (TargetShellClass == kPspsShellClass) {
        qc = {
            product_q.x - third.x,
            product_q.y - third.y,
            product_q.z - third.z,
        };
      }

      boys_values<3>(
          geometry.rho * distance_squared(first_pair.product_center, second_pair.product_center),
          geometry.boys);
#pragma unroll
      for (unsigned axis = 0; axis < 3; ++axis) {
        geometry.difference[axis] = vec_axis(difference, axis);
        geometry.shifts[0][axis] = vec_axis(pa, axis);
        if constexpr (TargetShellClass == kPpssShellClass) {
          geometry.shifts[1][axis] = vec_axis(pb, axis);
        } else if constexpr (TargetShellClass == kPspsShellClass) {
          geometry.shifts[2][axis] = vec_axis(qc, axis);
        }
        const double first_separation = vec_axis(first, axis) - vec_axis(second, axis);
        const double second_separation = vec_axis(third, axis) - vec_axis(fourth, axis);
        geometry.decay[0][axis] = -2.0 * mu * first_separation;
        geometry.decay[1][axis] = -geometry.decay[0][axis];
        geometry.decay[2][axis] = -2.0 * nu * second_separation;
      }

      generated_weighted_eri::IndependentGradient primitive{};
      if constexpr (TargetShellClass == kPspsShellClass) {
        primitive = generated_weighted_eri::psps_force(geometry, component_weight);
      } else if constexpr (TargetShellClass == kPpssShellClass) {
        primitive = generated_weighted_eri::ppss_force(geometry, component_weight);
      } else {
        primitive = generated_weighted_eri::dsss_force(geometry, component_weight);
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

/**
 * Evaluate one exact order-two AO-quartet gradient through compiler-owned force roots.
 *
 * The generic force fallback supplies a one-hot Cartesian component weight. Shell/pair
 * canonicalization remains runtime plumbing; all ERI/derivative algebra is shared with the
 * generated PSPS/PPSS/DSSS production consumers above.
 */
__device__ inline CartesianQuartetGradient
contracted_eri_cartesian_source_order2_generated_gradient(const DeviceBatch& batch,
                                                          std::int32_t system, std::int32_t i,
                                                          std::int32_t j, std::int32_t k,
                                                          std::int32_t l) {
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t raw_ao[4] = {
      static_cast<std::size_t>(i),
      static_cast<std::size_t>(j),
      static_cast<std::size_t>(k),
      static_cast<std::size_t>(l),
  };
  const std::int32_t raw_shell[4] = {
      batch.direct_ao_shells[system_ao_begin + raw_ao[0]],
      batch.direct_ao_shells[system_ao_begin + raw_ao[1]],
      batch.direct_ao_shells[system_ao_begin + raw_ao[2]],
      batch.direct_ao_shells[system_ao_begin + raw_ao[3]],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);

  unsigned canonical_raw_slot[4]{};
  if (shell_class == kPspsShellClass) {
    unsigned first_p_slot = 4U;
    unsigned second_p_slot = 4U;
    for (unsigned slot = 0; slot < 2; ++slot) {
      if (batch.shell_angular[raw_shell[slot]] == 1U) first_p_slot = slot;
    }
    for (unsigned slot = 2; slot < 4; ++slot) {
      if (batch.shell_angular[raw_shell[slot]] == 1U) second_p_slot = slot;
    }
    if (first_p_slot >= 2U || second_p_slot < 2U || second_p_slot >= 4U) return {};
    canonical_raw_slot[0] = first_p_slot;
    canonical_raw_slot[1] = 1U - first_p_slot;
    canonical_raw_slot[2] = second_p_slot;
    canonical_raw_slot[3] = 5U - second_p_slot;
  } else if (shell_class == kPpssShellClass) {
    const bool first_pair_is_pp =
        batch.shell_angular[raw_shell[0]] == 1U && batch.shell_angular[raw_shell[1]] == 1U;
    const unsigned pair_begin = first_pair_is_pp ? 0U : 2U;
    const unsigned other_pair_begin = first_pair_is_pp ? 2U : 0U;
    canonical_raw_slot[0] = pair_begin;
    canonical_raw_slot[1] = pair_begin + 1U;
    canonical_raw_slot[2] = other_pair_begin;
    canonical_raw_slot[3] = other_pair_begin + 1U;
  } else if (shell_class == kDsssShellClass) {
    unsigned d_slot = 4U;
    for (unsigned slot = 0; slot < 4; ++slot) {
      if (batch.shell_angular[raw_shell[slot]] == 2U) d_slot = slot;
    }
    if (d_slot >= 4U) return {};
    const unsigned pair_begin = d_slot < 2U ? 0U : 2U;
    const unsigned other_pair_begin = pair_begin == 0U ? 2U : 0U;
    canonical_raw_slot[0] = d_slot;
    canonical_raw_slot[1] = pair_begin + (d_slot == pair_begin ? 1U : 0U);
    canonical_raw_slot[2] = other_pair_begin;
    canonical_raw_slot[3] = other_pair_begin + 1U;
  } else {
    return {};
  }

  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]],
      raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]],
      raw_shell[canonical_raw_slot[3]],
  };
  const std::size_t canonical_pair[2] = {
      system_shell_pair_index(batch, system, canonical_shell[0], canonical_shell[1]),
      system_shell_pair_index(batch, system, canonical_shell[2], canonical_shell[3]),
  };

  const std::size_t component_begin[3] = {
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[0]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[1]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[2]]) - system_ao_begin,
  };
  const unsigned first_component =
      static_cast<unsigned>(raw_ao[canonical_raw_slot[0]] - component_begin[0]);
  unsigned output = first_component;
  if (shell_class == kPspsShellClass) {
    const unsigned third_component =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[2]] - component_begin[2]);
    if (first_component >= 3U || third_component >= 3U) return {};
    output = first_component * 3U + third_component;
  } else if (shell_class == kPpssShellClass) {
    const unsigned second_component =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[1]] - component_begin[1]);
    if (first_component >= 3U || second_component >= 3U) return {};
    output = first_component * 3U + second_component;
  } else if (first_component >= 6U) {
    return {};
  }

  const double angular_coefficient = batch.direct_ao_coefficients[system_ao_begin + raw_ao[0]] *
                                     batch.direct_ao_coefficients[system_ao_begin + raw_ao[1]] *
                                     batch.direct_ao_coefficients[system_ao_begin + raw_ao[2]] *
                                     batch.direct_ao_coefficients[system_ao_begin + raw_ao[3]];
  double component_weight[9]{};
  component_weight[output] = angular_coefficient;

  generated_weighted_eri::IndependentGradient gradient{};
  if (shell_class == kPspsShellClass) {
    gradient = contracted_eri_cartesian_source_order2_generated_weighted_gradient<kPspsShellClass>(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  } else if (shell_class == kPpssShellClass) {
    gradient = contracted_eri_cartesian_source_order2_generated_weighted_gradient<kPpssShellClass>(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  } else {
    gradient = contracted_eri_cartesian_source_order2_generated_weighted_gradient<kDsssShellClass>(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  }

  CartesianQuartetGradient result{};
#pragma unroll
  for (unsigned axis = 0; axis < 3; ++axis) {
    double fourth = 0.0;
#pragma unroll
    for (unsigned canonical = 0; canonical < 3; ++canonical) {
      const double value = gradient.center[canonical][axis];
      result.center[canonical_raw_slot[canonical]][axis] = value;
      fourth -= value;
    }
    result.center[canonical_raw_slot[3]][axis] = fourth;
  }
  return result;
}

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
  const DirectShellAoQuartetLayout ao_quartet_layout =
      direct_shell_ao_quartet_layout(batch, first_pair, second_pair);

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_layout.quartet_count; ++ordinal) {
    std::size_t raw_ao[4];
    decode_shell_ao_quartet(batch, first_pair, second_pair, ao_quartet_layout, ordinal,
                            system_ao_begin, raw_ao);
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

  const auto gradient =
      contracted_eri_cartesian_source_order2_generated_weighted_gradient<kPspsShellClass>(
          batch, first_pair, second_pair, canonical_shell[0], canonical_shell[1],
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
  const DirectShellAoQuartetLayout ao_quartet_layout =
      direct_shell_ao_quartet_layout(batch, first_pair, second_pair);

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_layout.quartet_count; ++ordinal) {
    std::size_t raw_ao[4];
    decode_shell_ao_quartet(batch, first_pair, second_pair, ao_quartet_layout, ordinal,
                            system_ao_begin, raw_ao);
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

  const auto gradient =
      contracted_eri_cartesian_source_order2_generated_weighted_gradient<TargetShellClass>(
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

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
// Resolve the generated callable through CMake's generated-header include path.
#include <weighted_eri.cuh>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_gradient_types.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for psss.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/**
 * Contract all three psss component gradients in one primitive traversal.
 *
 * `density_coefficient` already contains the exact eightfold RHF/UHF density
 * contraction for each p axis. Linearity lets the three component gradients
 * be combined before the primitive loops: PA, P-Q, and their coordinate
 * derivatives become short weighted dot products, while product centers,
 * decay, and Boys values are evaluated only once. The fourth-center gradient
 * is intentionally omitted and restored from translation by the force task.
 */
template <bool ResidentBra>
__device__ inline __noinline__ PsssWeightedGradient
contracted_eri_cartesian_source_psss_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t p_shell, std::int32_t paired_s_shell, std::int32_t third_shell,
    std::int32_t fourth_shell, const double (&density_coefficient)[3],
    const PrimitivePairData* resident_first_pairs, std::int64_t resident_first_pair_count) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[p_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[paired_s_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[third_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1);

  const std::int64_t p_ao_begin = batch.shell_direct_ao_offsets[p_shell];
  const double s_angular_coefficient =
      batch.direct_ao_coefficients[batch.shell_direct_ao_offsets[paired_s_shell]] *
      batch.direct_ao_coefficients[batch.shell_direct_ao_offsets[third_shell]] *
      batch.direct_ao_coefficients[batch.shell_direct_ao_offsets[fourth_shell]];
  const double axis_weight[3] = {
      density_coefficient[0] * s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin],
      density_coefficient[1] * s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin + 1],
      density_coefficient[2] * s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin + 2],
  };

  PsssWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == p_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_shell;
  const std::int64_t first_pair_begin =
      ResidentBra ? 0 : batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end =
      ResidentBra ? resident_first_pair_count
                  : batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = ResidentBra ? resident_first_pairs[first_primitive]
                                                     : batch.shell_primitive_pairs[first_primitive];
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      generated_weighted_eri::Geometry geometry;
      const double boys_argument = generated_weighted_eri::make_direct_cached_geometry(
          first_pair, second_pair, !first_pair_matches_canonical_order,
          !second_pair_matches_canonical_order, first, second, third, fourth, geometry);
      boys_values<2>(boys_argument, geometry.boys);
      const auto generated = generated_weighted_eri::psss_force(geometry, axis_weight);
#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          result.center[center][coordinate] += generated.center[center][coordinate];
        }
      }
    }
  }
  return result;
}

}  // namespace vibeqc::scf::cuda_execution

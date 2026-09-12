#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_eri_order3.cuh"
#include "scf/cuda/direct_native_gradient_types.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for psps gradient.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/**
 * Contract all nine canonical (p s | p s) components through one closed DAG.
 *
 * `component_weight` contains the screened density contraction and all four
 * Cartesian AO normalizations for the exact decoded AO-quartet domain. In
 * particular, missing entries from an identical-shell-pair triangular domain
 * remain zero; mirroring them would double-count the established ERI
 * multiplicity. The recurrence below forms the ten unique Coulomb states
 * through order two and their ten order-three raises exactly once per
 * primitive quartet, then contracts value and three coordinate derivatives.
 */
__device__ inline __noinline__ PspsWeightedGradient
contracted_eri_cartesian_source_psps_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t first_p_shell, std::int32_t first_s_shell, std::int32_t second_p_shell,
    std::int32_t second_s_shell, const double (&component_weight)[9]) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[first_p_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[first_s_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[second_p_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[second_s_shell], -1);

  PspsWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == first_p_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == second_p_shell;
  const std::int64_t first_pair_begin = batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const double inverse_two_p = 0.5 / p;
    const Vec3<double> product_p = first_pair.product_center;
    const Vec3<double> pa{
        product_p.x - first.x,
        product_p.y - first.y,
        product_p.z - first.z,
    };
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
      const double inverse_two_q = 0.5 / q;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const Vec3<double> qc{
          product_q.x - third.x,
          product_q.y - third.y,
          product_q.z - third.z,
      };
      const double x = product_p.x - product_q.x;
      const double y = product_p.y - product_q.y;
      const double z = product_p.z - product_q.z;
      double boys[4];
      boys_values<3>(rho * (x * x + y * y + z * z), boys);

      const double row_x =
          component_weight[0] * qc.x + component_weight[1] * qc.y + component_weight[2] * qc.z;
      const double row_y =
          component_weight[3] * qc.x + component_weight[4] * qc.y + component_weight[5] * qc.z;
      const double row_z =
          component_weight[6] * qc.x + component_weight[7] * qc.y + component_weight[8] * qc.z;
      const double column_x =
          component_weight[0] * pa.x + component_weight[3] * pa.y + component_weight[6] * pa.z;
      const double column_y =
          component_weight[1] * pa.x + component_weight[4] * pa.y + component_weight[7] * pa.z;
      const double column_z =
          component_weight[2] * pa.x + component_weight[5] * pa.y + component_weight[8] * pa.z;
      const double h0 = pa.x * row_x + pa.y * row_y + pa.z * row_z;
      const double hx = inverse_two_p * row_x - inverse_two_q * column_x;
      const double hy = inverse_two_p * row_y - inverse_two_q * column_y;
      const double hz = inverse_two_p * row_z - inverse_two_q * column_z;
      const double second_scale = -inverse_two_p * inverse_two_q;
      const double hxx = second_scale * component_weight[0];
      const double hxy = second_scale * (component_weight[1] + component_weight[3]);
      const double hxz = second_scale * (component_weight[2] + component_weight[6]);
      const double hyy = second_scale * component_weight[4];
      const double hyz = second_scale * (component_weight[5] + component_weight[7]);
      const double hzz = second_scale * component_weight[8];

      const WeightedOrder2Coulomb coulomb = contract_weighted_order2_coulomb(
          rho, x, y, z, boys, h0, hx, hy, hz, hxx, hxy, hxz, hyy, hyz, hzz);

      const double first_explicit[3] = {
          row_x * coulomb.c0 -
              inverse_two_q * (component_weight[0] * coulomb.cx + component_weight[1] * coulomb.cy +
                               component_weight[2] * coulomb.cz),
          row_y * coulomb.c0 -
              inverse_two_q * (component_weight[3] * coulomb.cx + component_weight[4] * coulomb.cy +
                               component_weight[5] * coulomb.cz),
          row_z * coulomb.c0 -
              inverse_two_q * (component_weight[6] * coulomb.cx + component_weight[7] * coulomb.cy +
                               component_weight[8] * coulomb.cz),
      };
      const double second_explicit[3] = {
          column_x * coulomb.c0 +
              inverse_two_p * (component_weight[0] * coulomb.cx + component_weight[3] * coulomb.cy +
                               component_weight[6] * coulomb.cz),
          column_y * coulomb.c0 +
              inverse_two_p * (component_weight[1] * coulomb.cx + component_weight[4] * coulomb.cy +
                               component_weight[7] * coulomb.cz),
          column_z * coulomb.c0 +
              inverse_two_p * (component_weight[2] * coulomb.cx + component_weight[5] * coulomb.cy +
                               component_weight[8] * coulomb.cz),
      };
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;
      const double product_scale[3] = {first_product_scale, second_product_scale,
                                       -third_product_scale};
      const double shift_scale[3] = {first_product_scale - 1.0, second_product_scale,
                                     third_product_scale - 1.0};
      const double prefactor = first_pair.weighted_coefficient * second_pair.weighted_coefficient *
                               2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          const double pair_coefficient_derivative =
              shift_scale[center] *
              (center < 2 ? first_explicit[coordinate] : second_explicit[coordinate]);
          double decay_derivative = 0.0;
          if (center < 2) {
            const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
            decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
          } else {
            const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
            decay_derivative = -2.0 * nu * difference;
          }
          result.center[center][coordinate] +=
              prefactor *
              (pair_coefficient_derivative + product_scale[center] * coulomb.chain[coordinate] +
               coulomb.value * decay_derivative);
        }
      }
    }
  }
  return result;
}

}  // namespace vibeqc::scf::cuda_execution

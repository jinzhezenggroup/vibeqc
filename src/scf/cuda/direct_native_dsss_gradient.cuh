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

// Retained direct integral arithmetic for dsss gradient.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/**
 * Contract canonical (d s | s s) in CCA xx,xy,xz,yy,yz,zz order.
 */
__device__ inline __noinline__ PspsWeightedGradient
contracted_eri_cartesian_source_dsss_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t d_shell, std::int32_t paired_s_shell, std::int32_t third_s_shell,
    std::int32_t fourth_s_shell, const double* component_weight) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[d_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[paired_s_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[third_s_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[fourth_s_shell], -1);

  PspsWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == d_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_s_shell;
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
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const double x = product_p.x - product_q.x;
      const double y = product_p.y - product_q.y;
      const double z = product_p.z - product_q.z;
      double boys[4];
      boys_values<3>(rho * (x * x + y * y + z * z), boys);

      const double gx = 2.0 * component_weight[0] * pa.x + component_weight[1] * pa.y +
                        component_weight[2] * pa.z;
      const double gy = component_weight[1] * pa.x + 2.0 * component_weight[3] * pa.y +
                        component_weight[4] * pa.z;
      const double gz = component_weight[2] * pa.x + component_weight[4] * pa.y +
                        2.0 * component_weight[5] * pa.z;
      const double h0 =
          component_weight[0] * (pa.x * pa.x + inverse_two_p) + component_weight[1] * pa.x * pa.y +
          component_weight[2] * pa.x * pa.z + component_weight[3] * (pa.y * pa.y + inverse_two_p) +
          component_weight[4] * pa.y * pa.z + component_weight[5] * (pa.z * pa.z + inverse_two_p);
      const double hx = inverse_two_p * gx;
      const double hy = inverse_two_p * gy;
      const double hz = inverse_two_p * gz;
      const double second_scale = inverse_two_p * inverse_two_p;
      const WeightedOrder2Coulomb coulomb = contract_weighted_order2_coulomb(
          rho, x, y, z, boys, h0, hx, hy, hz, second_scale * component_weight[0],
          second_scale * component_weight[1], second_scale * component_weight[2],
          second_scale * component_weight[3], second_scale * component_weight[4],
          second_scale * component_weight[5]);
      const double shift_scale = first_product_scale - 1.0;
      const double explicit_first[3] = {
          shift_scale * (gx * coulomb.c0 + inverse_two_p * (2.0 * component_weight[0] * coulomb.cx +
                                                            component_weight[1] * coulomb.cy +
                                                            component_weight[2] * coulomb.cz)),
          shift_scale * (gy * coulomb.c0 + inverse_two_p * (component_weight[1] * coulomb.cx +
                                                            2.0 * component_weight[3] * coulomb.cy +
                                                            component_weight[4] * coulomb.cz)),
          shift_scale *
              (gz * coulomb.c0 + inverse_two_p * (component_weight[2] * coulomb.cx +
                                                  component_weight[4] * coulomb.cy +
                                                  2.0 * component_weight[5] * coulomb.cz)),
      };
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;
      const double product_scale[3] = {first_product_scale, second_product_scale,
                                       -third_product_scale};
      const double prefactor = first_pair.weighted_coefficient * second_pair.weighted_coefficient *
                               2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          const double pair_coefficient_derivative =
              center == 0 ? explicit_first[coordinate]
                          : (center == 1 ? -explicit_first[coordinate] : 0.0);
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

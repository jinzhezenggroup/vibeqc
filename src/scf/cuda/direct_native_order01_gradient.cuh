#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_gradient_types.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for order01 gradient.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/**
 * Contract an ssss shell quartet from the reusable primitive-pair cache.
 *
 * The generic order-zero path rebuilds both primitive pairs for every
 * primitive quartet, including product centers, Gaussian pair decay, and
 * coefficient products. Those quantities already live in PrimitivePairData
 * and are shared with direct Fock. Only the inter-pair Boys argument and the
 * derivative chain therefore remain here. The fourth center is omitted by
 * translational invariance and reconstructed by the force-task consumer.
 */
__device__ __forceinline__ SsssWeightedGradient
contracted_eri_cartesian_source_ssss_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t first_shell, std::int32_t second_shell, std::int32_t third_shell,
    std::int32_t fourth_shell, double component_weight) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[first_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[second_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[third_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1);
  const Vec3<double> first_difference{
      first.x - second.x,
      first.y - second.y,
      first.z - second.z,
  };
  const Vec3<double> second_difference{
      third.x - fourth.x,
      third.y - fourth.y,
      third.z - fourth.z,
  };

  SsssWeightedGradient result{};
  const std::int64_t first_pair_begin = batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double rho = p * q / (p + q);
      const Vec3<double> product_difference{
          first_pair.product_center.x - second_pair.product_center.x,
          first_pair.product_center.y - second_pair.product_center.y,
          first_pair.product_center.z - second_pair.product_center.z,
      };
      double boys[2];
      boys_values<1>(rho * distance_squared(first_pair.product_center, second_pair.product_center),
                     boys);
      const double prefactor = component_weight * first_pair.weighted_coefficient *
                               second_pair.weighted_coefficient * 2.0 * pow(kPi, 2.5) /
                               (p * q * sqrt(p + q));
      const double product_chain[3] = {
          -2.0 * rho * product_difference.x * boys[1],
          -2.0 * rho * product_difference.y * boys[1],
          -2.0 * rho * product_difference.z * boys[1],
      };
      const double first_decay[3] = {
          -2.0 * first_pair.reduced_exponent * first_difference.x * boys[0],
          -2.0 * first_pair.reduced_exponent * first_difference.y * boys[0],
          -2.0 * first_pair.reduced_exponent * first_difference.z * boys[0],
      };
      const double third_decay[3] = {
          -2.0 * second_pair.reduced_exponent * second_difference.x * boys[0],
          -2.0 * second_pair.reduced_exponent * second_difference.y * boys[0],
          -2.0 * second_pair.reduced_exponent * second_difference.z * boys[0],
      };

#pragma unroll
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        result.center[0][coordinate] +=
            prefactor *
            (first_pair.first_product_scale * product_chain[coordinate] + first_decay[coordinate]);
        result.center[1][coordinate] +=
            prefactor *
            (first_pair.second_product_scale * product_chain[coordinate] - first_decay[coordinate]);
        result.center[2][coordinate] +=
            prefactor * (-second_pair.first_product_scale * product_chain[coordinate] +
                         third_decay[coordinate]);
      }
    }
  }
  return result;
}

/**
 * Evaluate the independent center derivatives of an ssss or canonical psss
 * primitive.
 *
 * Coordinate differentiation changes only the Gaussian pair decay, product
 * centers, and Boys argument. Computing those shared values once is much
 * cheaper than replaying the complete primitive with one Dual3 seed per
 * independent atom. `p_axis` is ignored for the order-zero specialization.
 */
template <unsigned AngularOrder>
__device__ inline void primitive_eri_order01_gradient(int p_axis, double alpha,
                                                      const Vec3<double>& first, double beta,
                                                      const Vec3<double>& second, double gamma,
                                                      const Vec3<double>& third, double delta,
                                                      const Vec3<double>& fourth,
                                                      double (&gradient)[4][3]) {
  static_assert(AngularOrder <= 1);
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<double> product_p = product_center(alpha, first, beta, second);
  const Vec3<double> product_q = product_center(gamma, third, delta, fourth);
  const Vec3<double> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };
  double boys[AngularOrder + 2];
  boys_values<AngularOrder + 1>(rho * distance_squared(product_p, product_q), boys);
  const double pair_decay =
      exp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  const double prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;

  // d(P-Q)/d(A,B,C,D); the same scalar applies independently to x/y/z.
  const double product_scales[4] = {alpha / p, beta / p, -gamma / q, -delta / q};
  double value = boys[0];
  double pa = 0.0;
  double pq_axis = 0.0;
  const double coulomb_scale = rho / p;
  if constexpr (AngularOrder == 1) {
    pa = vec_axis(product_p, p_axis) - vec_axis(first, p_axis);
    pq_axis = vec_axis(product_difference, p_axis);
    value = pa * boys[0] - coulomb_scale * pq_axis * boys[1];
  }

  // A simultaneous translation of all four Gaussian centers leaves the ERI
  // unchanged. Evaluate only three centers and recover the fourth exactly;
  // the force consumer already relies on this invariant across unique atoms.
  for (unsigned center = 0; center < 3; ++center) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double decay_derivative = 0.0;
      if (center < 2) {
        const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
        decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
      } else {
        const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
        decay_derivative = (center == 2 ? -2.0 * nu : 2.0 * nu) * difference;
      }
      const double argument_derivative =
          2.0 * rho * product_scales[center] * vec_axis(product_difference, coordinate);
      double value_derivative = -boys[1] * argument_derivative;
      if constexpr (AngularOrder == 1) {
        double pa_derivative = 0.0;
        if (coordinate == p_axis) {
          if (center == 0) {
            pa_derivative = alpha / p - 1.0;
          } else if (center == 1) {
            pa_derivative = beta / p;
          }
        }
        const double pq_derivative = coordinate == p_axis ? product_scales[center] : 0.0;
        value_derivative = pa_derivative * boys[0] - pa * boys[1] * argument_derivative -
                           coulomb_scale * pq_derivative * boys[1] +
                           coulomb_scale * pq_axis * boys[2] * argument_derivative;
      }
      gradient[center][coordinate] = prefactor * (value_derivative + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] - gradient[2][coordinate];
  }
}

/**
 * Contract explicit order-zero/one primitive gradients into input AO slots.
 *
 * The order-one integral is canonicalized to (p s|s s), while `original`
 * preserves the caller's slot-to-atom mapping for force accumulation.
 */
template <unsigned AngularOrder>
__device__ inline CartesianQuartetGradient contracted_eri_cartesian_source_order01_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  static_assert(AngularOrder <= 1);
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if constexpr (AngularOrder == 1) {
    unsigned p_slot = 0;
    for (unsigned slot = 0; slot < 4; ++slot) {
      if (batch.shell_angular[slots[slot].shell] == 1) p_slot = slot;
    }
    if (p_slot == 1) {
      const SourceSlot swap = slots[0];
      slots[0] = slots[1];
      slots[1] = swap;
    } else if (p_slot >= 2) {
      if (p_slot == 3) {
        const SourceSlot swap = slots[2];
        slots[2] = slots[3];
        slots[3] = swap;
      }
      const SourceSlot first_swap = slots[0];
      slots[0] = slots[2];
      slots[2] = first_swap;
      const SourceSlot second_swap = slots[1];
      slots[1] = slots[3];
      slots[3] = second_swap;
    }
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  int p_axis = 0;
  if constexpr (AngularOrder == 1) {
    const Angular angular = direct_ao_angular(batch, slots[0].ao);
    p_axis = angular.x == 1 ? 0 : (angular.y == 1 ? 1 : 2);
  }
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          primitive_eri_order01_gradient<AngularOrder>(
              p_axis, batch.primitive_exponents[a], positions[0], batch.primitive_exponents[b],
              positions[1], batch.primitive_exponents[c], positions[2],
              batch.primitive_exponents[d], positions[3], primitive_gradient);
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

}  // namespace vibeqc::scf::cuda_execution

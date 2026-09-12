#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_eri_order4.cuh"
#include "scf/cuda/direct_native_gradient_types.cuh"
#include "scf/cuda/direct_native_pair_order3_gradient.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for order3 gradient.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** Evaluate all-center derivatives of one canonical order-three primitive. */
template <unsigned FirstPairOrder, unsigned SecondPairOrder>
__device__ inline void primitive_eri_order3_gradient(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second, double gamma,
    const Vec3<double>& third, const Angular& angular_third, double delta,
    const Vec3<double>& fourth, const Angular& angular_fourth, double (&gradient)[4][3]) {
  static_assert(FirstPairOrder + SecondPairOrder == 3);
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
  const ThirdOrderPairGradientExpansion<FirstPairOrder> first_expansion =
      make_third_order_pair_gradient_expansion<FirstPairOrder>(alpha, first, angular_first, beta,
                                                               second, angular_second);
  const ThirdOrderPairGradientExpansion<SecondPairOrder> second_expansion =
      make_third_order_pair_gradient_expansion<SecondPairOrder>(gamma, third, angular_third, delta,
                                                                fourth, angular_fourth);
  double boys[5];
  boys_values<4>(rho * distance_squared(product_p, product_q), boys);
  const double first_product_scale = alpha / p;
  const double second_product_scale = beta / p;
  const double third_product_scale = -gamma / q;
  double value = 0.0;
  // The fourth center is restored from translational invariance. Besides
  // removing one quarter of the center updates, this keeps the hot primitive
  // path from spilling another three-component accumulator to local memory.
  double value_gradient[3][3]{};
  for (unsigned first_term = 0; first_term < (1U << FirstPairOrder); ++first_term) {
    for (unsigned second_term = 0; second_term < (1U << SecondPairOrder); ++second_term) {
      const ThirdOrderPairGradientTerm& first_item = first_expansion.terms[first_term];
      const ThirdOrderPairGradientTerm& second_item = second_expansion.terms[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      const unsigned derivative_state = first_item.derivative_state + second_item.derivative_state;
      const double coulomb = fourth_order_coulomb(derivative_state, rho, product_difference, boys);
      const double coefficient = sign * first_item.coefficient * second_item.coefficient;
      value += coefficient * coulomb;
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        const double first_pair_gradient =
            third_order_pair_first_center_gradient(first_expansion, first_term, coordinate);
        const double second_pair_gradient =
            third_order_pair_first_center_gradient(second_expansion, second_term, coordinate);
        const double first_coefficient_gradient =
            sign * first_pair_gradient * second_item.coefficient;
        const double second_coefficient_gradient =
            sign * first_item.coefficient * second_pair_gradient;
        // The Cartesian Coulomb derivative is center independent; only the
        // product-center chain-rule scale changes. Compute it once instead of
        // repeating the fourth-order closed form for all four centers.
        const double scaled_coulomb_derivative =
            coefficient *
            fourth_order_coulomb(derivative_state + fourth_order_derivative_state(coordinate), rho,
                                 product_difference, boys);
        value_gradient[0][coordinate] +=
            first_coefficient_gradient * coulomb + first_product_scale * scaled_coulomb_derivative;
        value_gradient[1][coordinate] += -first_coefficient_gradient * coulomb +
                                         second_product_scale * scaled_coulomb_derivative;
        value_gradient[2][coordinate] +=
            second_coefficient_gradient * coulomb + third_product_scale * scaled_coulomb_derivative;
      }
    }
  }

  const double pair_decay =
      exp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  const double prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;
  for (unsigned center = 0; center < 3; ++center) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double decay_derivative = 0.0;
      if (center < 2) {
        const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
        decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
      } else {
        const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
        decay_derivative = -2.0 * nu * difference;
      }
      gradient[center][coordinate] =
          prefactor * (value_gradient[center][coordinate] + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] - gradient[2][coordinate];
  }
}

/** Canonicalize and contract all-center gradients for total angular order 3. */
__device__ inline CartesianQuartetGradient contracted_eri_cartesian_source_order3_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
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
  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] + batch.shell_angular[slots[1].shell];
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
          if (first_pair_order == 3) {
            primitive_eri_order3_gradient<3, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else {
            primitive_eri_order3_gradient<2, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          }
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

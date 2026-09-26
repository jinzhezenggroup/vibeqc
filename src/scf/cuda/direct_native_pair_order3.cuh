#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_eri_order2.cuh"
#include "scf/cuda/direct_native_pair_order2.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for pair order3.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** Exact-sized sparse pair expansion used only by total-order-3 quartets. */
template <unsigned PairOrder, typename Scalar>
struct ThirdOrderPairExpansion {
  static_assert(PairOrder <= 3);
  LowOrderHermiteTerm<Scalar> terms[1U << PairOrder];
};

/**
 * Generate a shell pair through order three from its angular quanta.
 *
 * The base expansion is the product of one first-order factor per quantum.
 * Two quanta on the same Cartesian axis additionally have one Gaussian Wick
 * contraction, 1/(2p). Through order three, adding that contraction to the
 * surviving base terms produces the complete Hermite expansion while keeping
 * the exact 1/2/4/8-term bound.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, typename Scalar>
__device__ inline ThirdOrderPairExpansion<FirstShellAngular + SecondShellAngular, Scalar>
make_third_order_pair_expansion(double exponent, const Vec3<Scalar>& product,
                                const Vec3<Scalar>& first, const Angular& angular_first,
                                const Vec3<Scalar>& second, const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 3);
  constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  ThirdOrderPairExpansion<PairOrder, Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[QuantumStorage];
    Scalar shifts[QuantumStorage];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = low_order_derivative_state(axis);
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        ++quantum_count;
      }
    }

    for (unsigned subset = 0; subset < (1U << PairOrder); ++subset) {
      unsigned derivative_state = 0;
      Scalar coefficient = scalar<Scalar>(1.0);
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if ((subset & (1U << quantum)) != 0) {
          derivative_state += derivative_states[quantum];
          coefficient = inverse_two_exponent * coefficient;
        } else {
          coefficient = coefficient * shifts[quantum];
        }
      }
      expansion.terms[subset] = {derivative_state, coefficient};
    }

    if constexpr (PairOrder == 2) {
      if (derivative_states[0] == derivative_states[1]) {
        expansion.terms[0].coefficient =
            expansion.terms[0].coefficient + scalar<Scalar>(inverse_two_exponent);
      }
    } else if constexpr (PairOrder == 3) {
      for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 3; ++second_quantum) {
          if (derivative_states[first_quantum] != derivative_states[second_quantum]) {
            continue;
          }
          const unsigned remaining_quantum = 3U - first_quantum - second_quantum;
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient + inverse_two_exponent * shifts[remaining_quantum];
          const unsigned surviving_derivative = 1U << remaining_quantum;
          expansion.terms[surviving_derivative].coefficient =
              expansion.terms[surviving_derivative].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most three. */
template <typename Scalar>
__device__ inline Scalar third_order_coulomb(unsigned derivative_state, double rho,
                                             const Vec3<Scalar>& product_difference,
                                             const Scalar* boys) {
  const unsigned x_order = derivative_state & 3U;
  const unsigned y_order = (derivative_state >> 2U) & 3U;
  const unsigned z_order = (derivative_state >> 4U) & 3U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 3) {
    return low_order_coulomb(derivative_state, rho, product_difference, boys);
  }

  const double third_order_factor = -8.0 * rho * rho * rho;
  if (x_order == 3 || y_order == 3 || z_order == 3) {
    const Scalar coordinate = x_order == 3
                                  ? product_difference.x
                                  : (y_order == 3 ? product_difference.y : product_difference.z);
    return third_order_factor * coordinate * coordinate * coordinate * boys[3] +
           (12.0 * rho * rho) * coordinate * boys[2];
  }

  if (x_order == 2 || y_order == 2 || z_order == 2) {
    const Scalar repeated_coordinate =
        x_order == 2 ? product_difference.x
                     : (y_order == 2 ? product_difference.y : product_difference.z);
    const Scalar single_coordinate =
        x_order == 1 ? product_difference.x
                     : (y_order == 1 ? product_difference.y : product_difference.z);
    return third_order_factor * repeated_coordinate * repeated_coordinate * single_coordinate *
               boys[3] +
           (4.0 * rho * rho) * single_coordinate * boys[2];
  }

  return third_order_factor * product_difference.x * product_difference.y * product_difference.z *
         boys[3];
}

}  // namespace vibeqc::scf::cuda_execution

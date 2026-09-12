#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_pair_order3.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for eri order4.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** One sparse Hermite coefficient with three bits per Cartesian derivative. */
template <typename Scalar>
struct FourthOrderHermiteTerm {
  // Order four needs values 0--4 on one axis, so the two-bit encoding used by
  // lower orders is deliberately widened only for this specialization.
  unsigned derivative_state;
  Scalar coefficient;
};

/** Exact-sized sparse shell-pair expansion through total order four. */
template <unsigned PairOrder, typename Scalar>
struct FourthOrderPairExpansion {
  static_assert(PairOrder <= 4);
  FourthOrderHermiteTerm<Scalar> terms[1U << PairOrder];
};

__device__ inline unsigned fourth_order_derivative_state(int axis) { return 1U << (3 * axis); }

__device__ inline unsigned fourth_order_derivative_total(unsigned state) {
  return (state & 7U) + ((state >> 3U) & 7U) + ((state >> 6U) & 7U);
}

/**
 * Generate the exact Wick expansion of one shell pair through order four.
 *
 * Base subset terms represent uncontracted angular quanta. Every same-axis
 * pair adds one 1/(2p) contraction times the uncontracted remaining factors;
 * order four additionally admits the three possible disjoint pairings. All
 * contributions merge into the existing 2^N subset slots, so no generic
 * recurrence workspace is required.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, typename Scalar>
__device__ inline FourthOrderPairExpansion<FirstShellAngular + SecondShellAngular, Scalar>
make_fourth_order_pair_expansion(double exponent, const Vec3<Scalar>& product,
                                 const Vec3<Scalar>& first, const Angular& angular_first,
                                 const Vec3<Scalar>& second, const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 4);
  constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  FourthOrderPairExpansion<PairOrder, Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[QuantumStorage];
    Scalar shifts[QuantumStorage];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = fourth_order_derivative_state(axis);
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
    } else if constexpr (PairOrder == 4) {
      for (unsigned first_quantum = 0; first_quantum < 4; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 4; ++second_quantum) {
          if (derivative_states[first_quantum] != derivative_states[second_quantum]) {
            continue;
          }
          unsigned remaining[2];
          unsigned remaining_count = 0;
          for (unsigned quantum = 0; quantum < 4; ++quantum) {
            if (quantum != first_quantum && quantum != second_quantum) {
              remaining[remaining_count++] = quantum;
            }
          }
          const unsigned first_remaining = remaining[0];
          const unsigned second_remaining = remaining[1];
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              inverse_two_exponent * shifts[first_remaining] * shifts[second_remaining];
          expansion.terms[1U << first_remaining].coefficient =
              expansion.terms[1U << first_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent) *
                  shifts[second_remaining];
          expansion.terms[1U << second_remaining].coefficient =
              expansion.terms[1U << second_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent) * shifts[first_remaining];
          const unsigned both_remaining = (1U << first_remaining) | (1U << second_remaining);
          expansion.terms[both_remaining].coefficient =
              expansion.terms[both_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent * inverse_two_exponent);
        }
      }

      constexpr unsigned Pairings[3][4] = {
          {0, 1, 2, 3},
          {0, 2, 1, 3},
          {0, 3, 1, 2},
      };
      for (unsigned pairing = 0; pairing < 3; ++pairing) {
        if (derivative_states[Pairings[pairing][0]] == derivative_states[Pairings[pairing][1]] &&
            derivative_states[Pairings[pairing][2]] == derivative_states[Pairings[pairing][3]]) {
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most four. */
template <typename Scalar>
__device__ inline Scalar fourth_order_coulomb(unsigned derivative_state, double rho,
                                              const Vec3<Scalar>& product_difference,
                                              const Scalar* boys) {
  const unsigned x_order = derivative_state & 7U;
  const unsigned y_order = (derivative_state >> 3U) & 7U;
  const unsigned z_order = (derivative_state >> 6U) & 7U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 4) {
    const unsigned lower_order_state = x_order | (y_order << 2U) | (z_order << 4U);
    return third_order_coulomb(lower_order_state, rho, product_difference, boys);
  }

  const double fourth_order_factor = 16.0 * rho * rho * rho * rho;
  const double third_order_factor = -8.0 * rho * rho * rho;
  const double second_order_factor = 4.0 * rho * rho;
  if (x_order == 4 || y_order == 4 || z_order == 4) {
    const Scalar coordinate = x_order == 4
                                  ? product_difference.x
                                  : (y_order == 4 ? product_difference.y : product_difference.z);
    const Scalar coordinate_squared = coordinate * coordinate;
    return fourth_order_factor * coordinate_squared * coordinate_squared * boys[4] +
           (6.0 * third_order_factor) * coordinate_squared * boys[3] +
           (3.0 * second_order_factor) * boys[2];
  }

  if (x_order == 3 || y_order == 3 || z_order == 3) {
    const Scalar repeated_coordinate =
        x_order == 3 ? product_difference.x
                     : (y_order == 3 ? product_difference.y : product_difference.z);
    const Scalar single_coordinate =
        x_order == 1 ? product_difference.x
                     : (y_order == 1 ? product_difference.y : product_difference.z);
    return fourth_order_factor * repeated_coordinate * repeated_coordinate * repeated_coordinate *
               single_coordinate * boys[4] +
           (3.0 * third_order_factor) * repeated_coordinate * single_coordinate * boys[3];
  }

  if ((x_order == 2 && y_order == 2) || (x_order == 2 && z_order == 2) ||
      (y_order == 2 && z_order == 2)) {
    Scalar first_coordinate = product_difference.x;
    Scalar second_coordinate = product_difference.y;
    if (x_order == 0) {
      first_coordinate = product_difference.y;
      second_coordinate = product_difference.z;
    } else if (y_order == 0) {
      second_coordinate = product_difference.z;
    }
    const Scalar first_squared = first_coordinate * first_coordinate;
    const Scalar second_squared = second_coordinate * second_coordinate;
    return fourth_order_factor * first_squared * second_squared * boys[4] +
           third_order_factor * (first_squared + second_squared) * boys[3] +
           second_order_factor * boys[2];
  }

  const Scalar repeated_coordinate =
      x_order == 2 ? product_difference.x
                   : (y_order == 2 ? product_difference.y : product_difference.z);
  Scalar single_product = scalar<Scalar>(1.0);
  if (x_order == 1) single_product = single_product * product_difference.x;
  if (y_order == 1) single_product = single_product * product_difference.y;
  if (z_order == 1) single_product = single_product * product_difference.z;
  return fourth_order_factor * repeated_coordinate * repeated_coordinate * single_product *
             boys[4] +
         third_order_factor * single_product * boys[3];
}

/**
 * Closed order-4 contraction for canonical (f p|s s), (d d|s s),
 * (f s|p s), (d p|p s), (d s|d s), (d s|p p), and (p p|p p) quartets.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ inline Scalar primitive_eri_order4(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder = FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder = ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 4);
  static_assert(FirstPairOrder <= 4 && SecondPairOrder <= 4);
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const FourthOrderPairExpansion<FirstPairOrder, Scalar> first_expansion =
      make_fourth_order_pair_expansion<FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const FourthOrderPairExpansion<SecondPairOrder, Scalar> second_expansion =
      make_fourth_order_pair_expansion<ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[5];
  boys_values<4>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
      const FourthOrderHermiteTerm<Scalar>& first_item = first_expansion.terms[first_term];
      const FourthOrderHermiteTerm<Scalar>& second_item = second_expansion.terms[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      value = value +
              sign * first_item.coefficient * second_item.coefficient *
                  fourth_order_coulomb(first_item.derivative_state + second_item.derivative_state,
                                       rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

}  // namespace vibeqc::scf::cuda_execution

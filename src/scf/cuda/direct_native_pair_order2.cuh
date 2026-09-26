#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for pair order2.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** One nonzero three-dimensional Hermite coefficient through order two. */
template <typename Scalar>
struct LowOrderHermiteTerm {
  // Each Cartesian derivative occupies two bits. Adding two states therefore
  // combines pair derivatives without carrying between x, y, and z.
  unsigned derivative_state;
  Scalar coefficient;
};

/** Compact shell-pair expansion; total order two reaches at most four terms. */
template <typename Scalar>
struct LowOrderPairExpansion {
  LowOrderHermiteTerm<Scalar> terms[4];
};

__device__ inline unsigned low_order_derivative_state(int axis) { return 1U << (2 * axis); }

__device__ inline unsigned low_order_derivative_total(unsigned state) {
  return (state & 3U) + ((state >> 2U) & 3U) + ((state >> 4U) & 3U);
}

/**
 * Generate only the nonzero Hermite terms of one order-0/1/2 shell pair.
 *
 * The Gaussian pair decay is deliberately excluded and applied once by the
 * primitive quartet. At order two, retaining duplicate first-derivative terms
 * for a repeated axis keeps one runtime component path for d and p-p AOs while
 * still bounding the expansion at four entries.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, typename Scalar>
__device__ inline LowOrderPairExpansion<Scalar> make_low_order_pair_expansion(
    double exponent, const Vec3<Scalar>& product, const Vec3<Scalar>& first,
    const Angular& angular_first, const Vec3<Scalar>& second, const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 2);
  LowOrderPairExpansion<Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[2];
    Scalar shifts[2];
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

    expansion.terms[0] = {0U, shifts[0]};
    expansion.terms[1] = {derivative_states[0], scalar<Scalar>(inverse_two_exponent)};
    if constexpr (PairOrder == 2) {
      // A repeated Cartesian axis contributes the recurrence's +1/(2p)
      // correction. The two first-derivative entries then share a state and
      // sum to the exact E1 coefficient during contraction.
      const double repeated_axis_correction =
          derivative_states[0] == derivative_states[1] ? inverse_two_exponent : 0.0;
      expansion.terms[0].coefficient =
          shifts[0] * shifts[1] + scalar<Scalar>(repeated_axis_correction);
      expansion.terms[1].coefficient = inverse_two_exponent * shifts[1];
      expansion.terms[2] = {derivative_states[1], inverse_two_exponent * shifts[0]};
      expansion.terms[3] = {derivative_states[0] + derivative_states[1],
                            scalar<Scalar>(inverse_two_exponent * inverse_two_exponent)};
    }
  }
  return expansion;
}

}  // namespace vibeqc::scf::cuda_execution

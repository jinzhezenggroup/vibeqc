#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_pair_order2.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct integral arithmetic for pair order2 gradient.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** One order-two shell-pair term and its first-center coefficient gradient. */
struct LowOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
  double first_center[3];
};

struct LowOrderPairGradientExpansion {
  LowOrderPairGradientTerm terms[4];
  unsigned count;
};

/**
 * Build the exact order-0/1/2 Hermite pair expansion and center gradients.
 *
 * Pair decay is handled once by the primitive quartet. The only
 * coordinate-dependent pair coefficients are products of P-A/P-B shifts;
 * 1/(2p) contraction corrections are coordinate independent.
 */
__device__ inline LowOrderPairGradientExpansion make_low_order_pair_gradient_expansion(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second) {
  const double exponent = alpha + beta;
  const Vec3<double> product = product_center(alpha, first, beta, second);
  const double inverse_two_exponent = 0.5 / exponent;
  unsigned derivative_states[2]{};
  double shifts[2]{};
  double first_center_shift_gradients[2][3]{};
  unsigned quantum_count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
      derivative_states[quantum_count] = low_order_derivative_state(axis);
      shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
      first_center_shift_gradients[quantum_count][axis] = alpha / exponent - 1.0;
      ++quantum_count;
    }
    for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
      derivative_states[quantum_count] = low_order_derivative_state(axis);
      shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
      first_center_shift_gradients[quantum_count][axis] = alpha / exponent;
      ++quantum_count;
    }
  }

  LowOrderPairGradientExpansion expansion{};
  if (quantum_count == 0) {
    expansion.count = 1;
    expansion.terms[0].coefficient = 1.0;
    return expansion;
  }
  if (quantum_count == 1) {
    expansion.count = 2;
    expansion.terms[0].coefficient = shifts[0];
    expansion.terms[1].derivative_state = derivative_states[0];
    expansion.terms[1].coefficient = inverse_two_exponent;
    for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
      expansion.terms[0].first_center[coordinate] = first_center_shift_gradients[0][coordinate];
    }
    return expansion;
  }

  expansion.count = 4;
  expansion.terms[0].coefficient =
      shifts[0] * shifts[1] +
      (derivative_states[0] == derivative_states[1] ? inverse_two_exponent : 0.0);
  expansion.terms[1].derivative_state = derivative_states[0];
  expansion.terms[1].coefficient = inverse_two_exponent * shifts[1];
  expansion.terms[2].derivative_state = derivative_states[1];
  expansion.terms[2].coefficient = inverse_two_exponent * shifts[0];
  expansion.terms[3].derivative_state = derivative_states[0] + derivative_states[1];
  expansion.terms[3].coefficient = inverse_two_exponent * inverse_two_exponent;
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    expansion.terms[0].first_center[coordinate] =
        first_center_shift_gradients[0][coordinate] * shifts[1] +
        shifts[0] * first_center_shift_gradients[1][coordinate];
    expansion.terms[1].first_center[coordinate] =
        inverse_two_exponent * first_center_shift_gradients[1][coordinate];
    expansion.terms[2].first_center[coordinate] =
        inverse_two_exponent * first_center_shift_gradients[0][coordinate];
  }
  return expansion;
}

}  // namespace vibeqc::scf::cuda_execution

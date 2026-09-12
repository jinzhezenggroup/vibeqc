#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_eri_order4.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct integral arithmetic for pair order3 gradient.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** One sparse coefficient term in an order-three differentiated pair. */
struct ThirdOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
};

template <unsigned PairOrder>
struct ThirdOrderPairGradientExpansion {
  static_assert(PairOrder <= 3);
  static constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  ThirdOrderPairGradientTerm terms[1U << PairOrder];
  unsigned axes[QuantumStorage];
  double shifts[QuantumStorage];
  double first_center_shift_gradients[QuantumStorage];
  double inverse_two_exponent;
};

/**
 * Differentiate the exact subset/Wick pair expansion through order three.
 *
 * Three-bit Cartesian derivative fields are used because differentiating an
 * order-three Coulomb state can raise one axis to order four. Exponents are
 * fixed nuclear-coordinate parameters, so only P-A/P-B shift products carry
 * coefficient derivatives; Wick factors 1/(2p) remain constant.
 */
template <unsigned PairOrder>
__device__ inline ThirdOrderPairGradientExpansion<PairOrder>
make_third_order_pair_gradient_expansion(double alpha, const Vec3<double>& first,
                                         const Angular& angular_first, double beta,
                                         const Vec3<double>& second,
                                         const Angular& angular_second) {
  static_assert(PairOrder <= 3);
  ThirdOrderPairGradientExpansion<PairOrder> expansion{};
  if constexpr (PairOrder == 0) {
    expansion.terms[0].coefficient = 1.0;
  } else {
    const double exponent = alpha + beta;
    const Vec3<double> product = product_center(alpha, first, beta, second);
    expansion.inverse_two_exponent = 0.5 / exponent;
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        expansion.axes[quantum_count] = static_cast<unsigned>(axis);
        expansion.shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        expansion.first_center_shift_gradients[quantum_count] = alpha / exponent - 1.0;
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        expansion.axes[quantum_count] = static_cast<unsigned>(axis);
        expansion.shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        expansion.first_center_shift_gradients[quantum_count] = alpha / exponent;
        ++quantum_count;
      }
    }

    for (unsigned subset = 0; subset < (1U << PairOrder); ++subset) {
      ThirdOrderPairGradientTerm& term = expansion.terms[subset];
      term.coefficient = 1.0;
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if ((subset & (1U << quantum)) != 0) {
          term.derivative_state += fourth_order_derivative_state(expansion.axes[quantum]);
          term.coefficient *= expansion.inverse_two_exponent;
        } else {
          term.coefficient *= expansion.shifts[quantum];
        }
      }
    }

    if constexpr (PairOrder == 2) {
      if (expansion.axes[0] == expansion.axes[1]) {
        expansion.terms[0].coefficient += expansion.inverse_two_exponent;
      }
    } else if constexpr (PairOrder == 3) {
      for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 3; ++second_quantum) {
          if (expansion.axes[first_quantum] != expansion.axes[second_quantum]) {
            continue;
          }
          const unsigned remaining_quantum = 3U - first_quantum - second_quantum;
          ThirdOrderPairGradientTerm& value_term = expansion.terms[0];
          value_term.coefficient +=
              expansion.inverse_two_exponent * expansion.shifts[remaining_quantum];
          expansion.terms[1U << remaining_quantum].coefficient +=
              expansion.inverse_two_exponent * expansion.inverse_two_exponent;
        }
      }
    }
  }
  return expansion;
}

/** Differentiate one pair coefficient with respect to its first center. */
template <unsigned PairOrder>
__device__ inline double third_order_pair_first_center_gradient(
    const ThirdOrderPairGradientExpansion<PairOrder>& expansion, unsigned subset,
    unsigned coordinate) {
  if constexpr (PairOrder == 0) {
    return 0.0;
  } else {
    double gradient = 0.0;
    for (unsigned differentiated = 0; differentiated < PairOrder; ++differentiated) {
      if ((subset & (1U << differentiated)) != 0 || expansion.axes[differentiated] != coordinate) {
        continue;
      }
      double derivative = expansion.first_center_shift_gradients[differentiated];
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if (quantum == differentiated) continue;
        derivative *= (subset & (1U << quantum)) != 0 ? expansion.inverse_two_exponent
                                                      : expansion.shifts[quantum];
      }
      gradient += derivative;
    }
    if constexpr (PairOrder == 3) {
      if (subset == 0) {
        for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
          for (unsigned second_quantum = first_quantum + 1; second_quantum < 3; ++second_quantum) {
            if (expansion.axes[first_quantum] != expansion.axes[second_quantum]) {
              continue;
            }
            const unsigned remaining_quantum = 3U - first_quantum - second_quantum;
            if (expansion.axes[remaining_quantum] == coordinate) {
              gradient += expansion.inverse_two_exponent *
                          expansion.first_center_shift_gradients[remaining_quantum];
            }
          }
        }
      }
    }
    return gradient;
  }
}

}  // namespace vibeqc::scf::cuda_execution

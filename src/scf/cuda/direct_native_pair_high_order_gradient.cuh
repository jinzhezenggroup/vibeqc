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

// Retained direct integral arithmetic for pair high order gradient.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** One sparse coefficient term and its first-center gradient. */
struct HighOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
  double first_center[3];
};

template <unsigned PairOrder>
struct HighOrderPairGradientGeometry {
  static_assert(PairOrder <= 6);
  static constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  unsigned axes[QuantumStorage];
  double shifts[QuantumStorage];
  double first_center_shift_gradients[QuantumStorage];
  double inverse_two_exponent;
};

/** Add one Wick matching to one derivative subset and its gradient. */
template <unsigned PairOrder>
__device__ inline void add_high_order_wick_matching_term(
    HighOrderPairGradientTerm& term, const HighOrderPairGradientGeometry<PairOrder>& geometry,
    unsigned subset, unsigned removed, unsigned contraction_count) {
  static_assert(PairOrder <= 6);
  if ((subset & removed) != 0) return;
  double inverse_factor = 1.0;
  const unsigned inverse_count = contraction_count + static_cast<unsigned>(__popc(subset));
  for (unsigned factor = 0; factor < inverse_count; ++factor) {
    inverse_factor *= geometry.inverse_two_exponent;
  }

  double coefficient = inverse_factor;
  for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
    const unsigned bit = 1U << quantum;
    if (((subset | removed) & bit) == 0) {
      coefficient *= geometry.shifts[quantum];
    }
  }
  term.coefficient += coefficient;

  // Differentiate the same surviving product while its factors are hot.
  // The old path revisited every matching once for coefficients and three
  // more times for Cartesian gradients. Accumulating by the differentiated
  // quantum avoids that repeated combinatorial walk and keeps the gradient
  // sparse in its quantum's Cartesian axis.
  for (unsigned differentiated = 0; differentiated < PairOrder; ++differentiated) {
    const unsigned differentiated_bit = 1U << differentiated;
    if (((subset | removed) & differentiated_bit) != 0) continue;
    double derivative = inverse_factor * geometry.first_center_shift_gradients[differentiated];
    for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
      const unsigned bit = 1U << quantum;
      if (quantum == differentiated || ((subset | removed) & bit) != 0) {
        continue;
      }
      derivative *= geometry.shifts[quantum];
    }
    term.first_center[geometry.axes[differentiated]] += derivative;
  }
}

/**
 * Generate the exact subset/Wick pair expansion through angular order six.
 *
 * One contraction removes a same-axis quantum pair. Three disjoint
 * contractions are the highest possible matching at order six. Expanding
 * every surviving quantum into either its center shift or Hermite derivative
 * covers the complete Gaussian product recurrence without a dense
 * coefficient workspace. Pair masks are ordered to visit every disjoint Wick
 * matching exactly once.
 */
template <unsigned PairOrder>
__device__ inline HighOrderPairGradientGeometry<PairOrder> make_high_order_pair_gradient_geometry(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second) {
  static_assert(PairOrder <= 6);
  HighOrderPairGradientGeometry<PairOrder> geometry{};
  if constexpr (PairOrder != 0) {
    const double exponent = alpha + beta;
    const Vec3<double> product = product_center(alpha, first, beta, second);
    geometry.inverse_two_exponent = 0.5 / exponent;
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        geometry.axes[quantum_count] = static_cast<unsigned>(axis);
        geometry.shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        geometry.first_center_shift_gradients[quantum_count] = alpha / exponent - 1.0;
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        geometry.axes[quantum_count] = static_cast<unsigned>(axis);
        geometry.shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        geometry.first_center_shift_gradients[quantum_count] = alpha / exponent;
        ++quantum_count;
      }
    }
  }
  return geometry;
}

/** Generate one exact subset/Wick coefficient and its center gradient. */
template <unsigned PairOrder>
__device__ inline HighOrderPairGradientTerm make_high_order_pair_gradient_term(
    const HighOrderPairGradientGeometry<PairOrder>& geometry, unsigned subset) {
  HighOrderPairGradientTerm term{};
  if constexpr (PairOrder == 0) {
    // The scalar pair has one unit term and no center derivative. Keeping it
    // out of the combinatorial loops also avoids zero-trip unsigned-loop
    // diagnostics in CUDA's template instantiation.
    term.coefficient = 1.0;
  } else {
    for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
      if ((subset & (1U << quantum)) != 0) {
        term.derivative_state += fourth_order_derivative_state(geometry.axes[quantum]);
      }
    }
    add_high_order_wick_matching_term(term, geometry, subset, 0U, 0U);
    for (unsigned first_quantum = 0; first_quantum < PairOrder; ++first_quantum) {
      for (unsigned second_quantum = first_quantum + 1; second_quantum < PairOrder;
           ++second_quantum) {
        if (geometry.axes[first_quantum] != geometry.axes[second_quantum]) {
          continue;
        }
        const unsigned first_pair = (1U << first_quantum) | (1U << second_quantum);
        add_high_order_wick_matching_term(term, geometry, subset, first_pair, 1U);
        for (unsigned third_quantum = 0; third_quantum < PairOrder; ++third_quantum) {
          for (unsigned fourth_quantum = third_quantum + 1; fourth_quantum < PairOrder;
               ++fourth_quantum) {
            const unsigned second_pair = (1U << third_quantum) | (1U << fourth_quantum);
            if ((first_pair & second_pair) != 0 || first_pair >= second_pair ||
                geometry.axes[third_quantum] != geometry.axes[fourth_quantum]) {
              continue;
            }
            add_high_order_wick_matching_term(term, geometry, subset, first_pair | second_pair, 2U);
            if constexpr (PairOrder == 6) {
              for (unsigned fifth_quantum = 0; fifth_quantum < PairOrder; ++fifth_quantum) {
                for (unsigned sixth_quantum = fifth_quantum + 1; sixth_quantum < PairOrder;
                     ++sixth_quantum) {
                  const unsigned third_pair = (1U << fifth_quantum) | (1U << sixth_quantum);
                  if (((first_pair | second_pair) & third_pair) != 0 || second_pair >= third_pair ||
                      geometry.axes[fifth_quantum] != geometry.axes[sixth_quantum]) {
                    continue;
                  }
                  add_high_order_wick_matching_term(term, geometry, subset,
                                                    first_pair | second_pair | third_pair, 3U);
                }
              }
            }
          }
        }
      }
    }
  }
  return term;
}

}  // namespace vibeqc::scf::cuda_execution

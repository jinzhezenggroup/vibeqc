#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_eri_order4.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct integral arithmetic for high order coulomb.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** Primitive-local powers reused by one bounded high-order Coulomb recurrence. */
template <unsigned MaximumOrder>
struct HighOrderCoulombWorkspace {
  static_assert(MaximumOrder >= 5 && MaximumOrder <= 7);
  Vec3<double> difference;
  double coordinate_powers[3][MaximumOrder + 1];
  double negative_two_rho_powers[MaximumOrder + 1];
};

template <unsigned MaximumOrder>
__device__ inline HighOrderCoulombWorkspace<MaximumOrder> make_high_order_coulomb_workspace(
    double rho, const Vec3<double>& difference) {
  HighOrderCoulombWorkspace<MaximumOrder> workspace{};
  workspace.difference = difference;
  for (unsigned axis = 0; axis < 3; ++axis) {
    workspace.coordinate_powers[axis][0] = 1.0;
    for (unsigned power = 1; power <= MaximumOrder; ++power) {
      workspace.coordinate_powers[axis][power] = workspace.coordinate_powers[axis][power - 1] *
                                                 vec_axis(difference, static_cast<int>(axis));
    }
  }
  workspace.negative_two_rho_powers[0] = 1.0;
  for (unsigned power = 1; power <= MaximumOrder; ++power) {
    workspace.negative_two_rho_powers[power] =
        workspace.negative_two_rho_powers[power - 1] * (-2.0 * rho);
  }
  return workspace;
}

/** Number of ways to form `pairs` disjoint contractions from one axis. */
__device__ inline unsigned axis_wick_multiplicity(unsigned order, unsigned pairs) {
  if (pairs == 0) return 1U;
  if (pairs == 1) return order * (order - 1U) / 2U;
  if (pairs == 2) {
    return order * (order - 1U) * (order - 2U) * (order - 3U) / 8U;
  }
  return order * (order - 1U) * (order - 2U) * (order - 3U) * (order - 4U) * (order - 5U) / 48U;
}

/** Evaluate one Cartesian Coulomb derivative through `MaximumOrder`. */
template <unsigned MaximumOrder>
__device__ inline double high_order_coulomb(
    unsigned derivative_state, double rho, const HighOrderCoulombWorkspace<MaximumOrder>& workspace,
    const double* boys) {
  const unsigned x_order = derivative_state & 7U;
  const unsigned y_order = (derivative_state >> 3U) & 7U;
  const unsigned z_order = (derivative_state >> 6U) & 7U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 5) {
    return fourth_order_coulomb(derivative_state, rho, workspace.difference, boys);
  }

  double value = 0.0;
  for (unsigned x_pairs = 0; x_pairs <= x_order / 2U; ++x_pairs) {
    for (unsigned y_pairs = 0; y_pairs <= y_order / 2U; ++y_pairs) {
      for (unsigned z_pairs = 0; z_pairs <= z_order / 2U; ++z_pairs) {
        const unsigned contraction_count = x_pairs + y_pairs + z_pairs;
        const unsigned boys_order = total_order - contraction_count;
        const unsigned multiplicity = axis_wick_multiplicity(x_order, x_pairs) *
                                      axis_wick_multiplicity(y_order, y_pairs) *
                                      axis_wick_multiplicity(z_order, z_pairs);
        value += static_cast<double>(multiplicity) * workspace.negative_two_rho_powers[boys_order] *
                 workspace.coordinate_powers[0][x_order - 2U * x_pairs] *
                 workspace.coordinate_powers[1][y_order - 2U * y_pairs] *
                 workspace.coordinate_powers[2][z_order - 2U * z_pairs] * boys[boys_order];
      }
    }
  }
  return value;
}

}  // namespace vibeqc::scf::cuda_execution

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_cartesian.cuh"
#include "scf/cuda/direct_native_eri_order2.cuh"
#include "scf/cuda/direct_native_eri_order3.cuh"
#include "scf/cuda/direct_native_eri_order4.cuh"
#include "scf/cuda/direct_native_shell_pair_hermite.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for shell class.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/**
 * Evaluate one Cartesian primitive quartet with exact shell-pair workspaces.
 *
 * Axis powers remain AO-component data, while the enclosing shell angular
 * momenta bound every Hermite dimension at compile time. This avoids charging
 * an s/p/d task for the generic f/f pair workspace.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ inline Scalar primitive_eri_cartesian_shell_class(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned MaximumAngular =
      FirstShellAngular + SecondShellAngular + ThirdShellAngular + FourthShellAngular;
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  if constexpr (FirstShellAngular == 1 && SecondShellAngular == 0 && ThirdShellAngular == 0 &&
                FourthShellAngular == 0) {
    const int axis = angular_first.x == 1 ? 0 : (angular_first.y == 1 ? 1 : 2);
    return primitive_eri_psss(axis, alpha, first, beta, second, gamma, third, delta, fourth);
  } else if constexpr (MaximumAngular == 2) {
    return primitive_eri_order2<FirstShellAngular, SecondShellAngular, ThirdShellAngular,
                                FourthShellAngular>(alpha, first, angular_first, beta, second,
                                                    angular_second, gamma, third, angular_third,
                                                    delta, fourth, angular_fourth);
  } else if constexpr (MaximumAngular == 3) {
    return primitive_eri_order3<FirstShellAngular, SecondShellAngular, ThirdShellAngular,
                                FourthShellAngular>(alpha, first, angular_first, beta, second,
                                                    angular_second, gamma, third, angular_third,
                                                    delta, fourth, angular_fourth);
  } else if constexpr (MaximumAngular == 4) {
    return primitive_eri_order4<FirstShellAngular, SecondShellAngular, ThirdShellAngular,
                                FourthShellAngular>(alpha, first, angular_first, beta, second,
                                                    angular_second, gamma, third, angular_third,
                                                    delta, fourth, angular_fourth);
  } else {
    using Real = EvaluationReal<Scalar>;
    const Real alpha_value{alpha};
    const Real beta_value{beta};
    const Real gamma_value{gamma};
    const Real delta_value{delta};
    const Real p = alpha_value + beta_value;
    const Real q = gamma_value + delta_value;
    const Real rho = p * q / (p + q);
    const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
    const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
    ShellPairHermiteCoefficients<Scalar, FirstShellAngular, SecondShellAngular>
        first_coefficients[3];
    ShellPairHermiteCoefficients<Scalar, ThirdShellAngular, FourthShellAngular>
        second_coefficients[3];
    for (int axis = 0; axis < 3; ++axis) {
      fill_shell_pair_hermite<FirstShellAngular, SecondShellAngular>(
          angular_axis(angular_first, axis), angular_axis(angular_second, axis),
          vec_axis(product_p, axis), vec_axis(first, axis), vec_axis(second, axis), alpha, beta,
          first_coefficients[axis]);
      fill_shell_pair_hermite<ThirdShellAngular, FourthShellAngular>(
          angular_axis(angular_third, axis), angular_axis(angular_fourth, axis),
          vec_axis(product_q, axis), vec_axis(third, axis), vec_axis(fourth, axis), gamma, delta,
          second_coefficients[axis]);
    }
    return eri_cartesian_value<MaximumAngular>(p, q, rho, product_p, product_q, angular_first,
                                               angular_second, angular_third, angular_fourth,
                                               first_coefficients, second_coefficients);
  }
}

}  // namespace vibeqc::scf::cuda_execution

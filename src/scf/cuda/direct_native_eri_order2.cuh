#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_pair_order2.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for eri order2.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** Evaluate a Cartesian Coulomb derivative of total order at most two. */
template <typename Scalar>
__device__ inline Scalar low_order_coulomb(unsigned derivative_state, double rho,
                                           const Vec3<Scalar>& product_difference,
                                           const Scalar* boys) {
  const unsigned x_order = derivative_state & 3U;
  const unsigned y_order = (derivative_state >> 2U) & 3U;
  const unsigned z_order = (derivative_state >> 4U) & 3U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order == 0) return boys[0];

  if (total_order == 1) {
    const Scalar coordinate = x_order != 0
                                  ? product_difference.x
                                  : (y_order != 0 ? product_difference.y : product_difference.z);
    return (-2.0 * rho) * coordinate * boys[1];
  }

  const double second_order_factor = 4.0 * rho * rho;
  if (x_order == 2 || y_order == 2 || z_order == 2) {
    const Scalar coordinate = x_order == 2
                                  ? product_difference.x
                                  : (y_order == 2 ? product_difference.y : product_difference.z);
    return second_order_factor * coordinate * coordinate * boys[2] - (2.0 * rho) * boys[1];
  }

  Scalar coordinate_product = scalar<Scalar>(1.0);
  if (x_order != 0) coordinate_product = coordinate_product * product_difference.x;
  if (y_order != 0) coordinate_product = coordinate_product * product_difference.y;
  if (z_order != 0) coordinate_product = coordinate_product * product_difference.z;
  return second_order_factor * coordinate_product * boys[2];
}

/** Ten unique Cartesian Coulomb states through total order two. */
struct Order2CoulombValues {
  double c0;
  double cx;
  double cy;
  double cz;
  double cxx;
  double cxy;
  double cxz;
  double cyy;
  double cyz;
  double czz;
};

/** Build the complete order-two Coulomb tensor once per primitive quartet. */
__device__ __forceinline__ Order2CoulombValues order2_coulomb_values(double rho, double x, double y,
                                                                     double z,
                                                                     const double (&boys)[3]) {
  const double twice_rho = 2.0 * rho;
  const double twice_rho_squared = twice_rho * twice_rho;
  return {
      boys[0],
      -twice_rho * x * boys[1],
      -twice_rho * y * boys[1],
      -twice_rho * z * boys[1],
      twice_rho_squared * x * x * boys[2] - twice_rho * boys[1],
      twice_rho_squared * x * y * boys[2],
      twice_rho_squared * x * z * boys[2],
      twice_rho_squared * y * y * boys[2] - twice_rho * boys[1],
      twice_rho_squared * y * z * boys[2],
      twice_rho_squared * z * z * boys[2] - twice_rho * boys[1],
  };
}

/**
 * Closed order-2 contraction for canonical (d s|s s), (p p|s s), and
 * (p s|p s) primitive quartets.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ inline Scalar primitive_eri_order2(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder = FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder = ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 2);
  static_assert(FirstPairOrder <= 2 && SecondPairOrder <= 2);
  constexpr unsigned FirstTermCount = FirstPairOrder == 0 ? 1 : (FirstPairOrder == 1 ? 2 : 4);
  constexpr unsigned SecondTermCount = SecondPairOrder == 0 ? 1 : (SecondPairOrder == 1 ? 2 : 4);

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const LowOrderPairExpansion<Scalar> first_expansion =
      make_low_order_pair_expansion<FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const LowOrderPairExpansion<Scalar> second_expansion =
      make_low_order_pair_expansion<ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[3];
  boys_values<2>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
      const LowOrderHermiteTerm<Scalar>& first_item = first_expansion.terms[first_term];
      const LowOrderHermiteTerm<Scalar>& second_item = second_expansion.terms[second_term];
      const double sign =
          (low_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      value =
          value + sign * first_item.coefficient * second_item.coefficient *
                      low_order_coulomb(first_item.derivative_state + second_item.derivative_state,
                                        rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

/** Cartesian source component count for one s, p, or d shell. */
template <unsigned ShellAngular>
__host__ __device__ constexpr unsigned order2_shell_component_count() {
  static_assert(ShellAngular <= 2);
  return (ShellAngular + 1) * (ShellAngular + 2) / 2;
}

/** Return one CCA-ordered Cartesian component through d angular momentum. */
template <unsigned ShellAngular>
__device__ __forceinline__ Angular order2_shell_component(unsigned component) {
  static_assert(ShellAngular <= 2);
  if constexpr (ShellAngular == 0) {
    (void)component;
    return {0, 0, 0};
  } else if constexpr (ShellAngular == 1) {
    return component == 0 ? Angular{1, 0, 0} : component == 1 ? Angular{0, 1, 0} : Angular{0, 0, 1};
  } else {
    switch (component) {
      case 0:
        return {2, 0, 0};
      case 1:
        return {1, 1, 0};
      case 2:
        return {1, 0, 1};
      case 3:
        return {0, 2, 0};
      case 4:
        return {0, 1, 1};
      default:
        return {0, 0, 2};
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution

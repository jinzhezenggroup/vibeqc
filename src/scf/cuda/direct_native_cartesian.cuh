#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/hermite_recurrence.cuh"
#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for cartesian.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

template <typename Scalar>
__device__ inline Scalar primitive_eri(double alpha, const Vec3<Scalar>& first, double beta,
                                       const Vec3<Scalar>& second, double gamma,
                                       const Vec3<Scalar>& third, double delta,
                                       const Vec3<Scalar>& fourth) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const Vec3<Scalar> center_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> center_q = product_center(gamma, third, delta, fourth);
  const double rho = p * q / (p + q);
  const double prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));
  return prefactor *
         qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth)) *
         boys0(rho * distance_squared(center_p, center_q));
}

template <unsigned MaximumAngular, typename Scalar, typename FirstCoefficients,
          typename SecondCoefficients>
__device__ inline __noinline__ Scalar eri_cartesian_value(
    EvaluationReal<Scalar> p, EvaluationReal<Scalar> q, EvaluationReal<Scalar> rho,
    const Vec3<Scalar>& product_p, const Vec3<Scalar>& product_q, const Angular& angular_first,
    const Angular& angular_second, const Angular& angular_third, const Angular& angular_fourth,
    const FirstCoefficients* first_coefficients, const SecondCoefficients* second_coefficients) {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  CoulombAuxiliary<Scalar, MaximumAngular> auxiliary;
  fill_coulomb<MaximumAngular>(rho, product_p, product_q, auxiliary);

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned t = 0; t <= angular_first.x + angular_second.x; ++t) {
    for (unsigned u = 0; u <= angular_first.y + angular_second.y; ++u) {
      for (unsigned v = 0; v <= angular_first.z + angular_second.z; ++v) {
        const Scalar first_value = first_coefficients[0].at(angular_first.x, angular_second.x, t) *
                                   first_coefficients[1].at(angular_first.y, angular_second.y, u) *
                                   first_coefficients[2].at(angular_first.z, angular_second.z, v);
        for (unsigned tau = 0; tau <= angular_third.x + angular_fourth.x; ++tau) {
          for (unsigned nu = 0; nu <= angular_third.y + angular_fourth.y; ++nu) {
            for (unsigned phi = 0; phi <= angular_third.z + angular_fourth.z; ++phi) {
              const double sign = ((tau + nu + phi) & 1U) == 0 ? 1.0 : -1.0;
              value =
                  value + sign * first_value *
                              second_coefficients[0].at(angular_third.x, angular_fourth.x, tau) *
                              second_coefficients[1].at(angular_third.y, angular_fourth.y, nu) *
                              second_coefficients[2].at(angular_third.z, angular_fourth.z, phi) *
                              auxiliary.at(0, t + tau, u + nu, v + phi);
            }
          }
        }
      }
    }
  }
  const EvaluationReal<Scalar> prefactor =
      EvaluationReal<Scalar>{2.0 * pow(kPi, 2.5)} / (p * q * qsqrt(p + q));
  return prefactor * value;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ inline Scalar primitive_eri_cartesian(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  HermiteCoefficients<Scalar> first_coefficients[3];
  HermiteCoefficients<Scalar> second_coefficients[3];
  for (int axis = 0; axis < 3; ++axis) {
    fill_hermite(angular_axis(angular_first, axis), angular_axis(angular_second, axis),
                 vec_axis(product_p, axis), vec_axis(first, axis), vec_axis(second, axis), alpha,
                 beta, first_coefficients[axis]);
    fill_hermite(angular_axis(angular_third, axis), angular_axis(angular_fourth, axis),
                 vec_axis(product_q, axis), vec_axis(third, axis), vec_axis(fourth, axis), gamma,
                 delta, second_coefficients[axis]);
  }
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  return eri_cartesian_value<MaximumAngular>(p, q, rho, product_p, product_q, angular_first,
                                             angular_second, angular_third, angular_fourth,
                                             first_coefficients, second_coefficients);
}

/**
 * Closed first-order Hermite contraction for canonical (p s | s s).
 *
 * The exact order-1 shell class has only one Cartesian component on the first
 * center. Generating its two reachable Hermite terms directly avoids all six
 * coefficient workspaces and the generic six-deep component contraction.
 */
template <typename Scalar>
__device__ inline Scalar primitive_eri_psss(int axis, double alpha, const Vec3<Scalar>& first,
                                            double beta, const Vec3<Scalar>& second, double gamma,
                                            const Vec3<Scalar>& third, double delta,
                                            const Vec3<Scalar>& fourth) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  Scalar boys[2];
  boys_values<1>(rho * distance_squared(product_p, product_q), boys);
  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  const Scalar pa = vec_axis(product_p, axis) - vec_axis(first, axis);
  const Scalar pq = vec_axis(product_p, axis) - vec_axis(product_q, axis);
  const Scalar value = pa * boys[0] - (rho / p) * pq * boys[1];
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

}  // namespace vibeqc::scf::cuda_execution

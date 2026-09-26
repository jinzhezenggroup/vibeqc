#pragma once

#include <cuda_runtime.h>

#include <cmath>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/hermite_recurrence.cuh"

// Retained overlap/kinetic primitives used by independent derivative
// and force paths; generated value evaluation has a separate owner.
namespace vibeqc::scf::cuda_execution {

template <typename Scalar>
__device__ inline Scalar primitive_overlap(double alpha, const Vec3<Scalar>& first, double beta,
                                           const Vec3<Scalar>& second) {
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  return pow(kPi / exponent, 1.5) * qexp(-reduced * distance_squared(first, second));
}

template <typename Scalar>
__device__ inline Scalar primitive_kinetic(double alpha, const Vec3<Scalar>& first, double beta,
                                           const Vec3<Scalar>& second) {
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  const Scalar squared_distance = distance_squared(first, second);
  return reduced * (scalar<Scalar>(3.0) - 2.0 * reduced * squared_distance) *
         primitive_overlap(alpha, first, beta, second);
}

template <typename Scalar>
__device__ inline Scalar primitive_overlap_cartesian(double alpha, const Vec3<Scalar>& first,
                                                     const Angular& angular_first, double beta,
                                                     const Vec3<Scalar>& second,
                                                     const Angular& angular_second) {
  const double p = alpha + beta;
  const Vec3<Scalar> product = product_center(alpha, first, beta, second);
  Scalar result = scalar<Scalar>(pow(kPi / p, 1.5));
  for (int axis = 0; axis < 3; ++axis) {
    HermiteCoefficients<Scalar> coefficients;
    const unsigned first_power = angular_axis(angular_first, axis);
    const unsigned second_power = angular_axis(angular_second, axis);
    fill_hermite(first_power, second_power, vec_axis(product, axis), vec_axis(first, axis),
                 vec_axis(second, axis), alpha, beta, coefficients);
    result = result * coefficients.at(first_power, second_power, 0);
  }
  return result;
}

template <typename Scalar>
__device__ inline Scalar primitive_kinetic_cartesian(double alpha, const Vec3<Scalar>& first,
                                                     const Angular& angular_first, double beta,
                                                     const Vec3<Scalar>& second,
                                                     const Angular& angular_second) {
  Scalar result =
      beta * (2.0 * static_cast<double>(angular_total(angular_second)) + 3.0) *
      primitive_overlap_cartesian(alpha, first, angular_first, beta, second, angular_second);
  for (int axis = 0; axis < 3; ++axis) {
    Angular raised = angular_second;
    add_angular_axis(raised, axis, 2);
    result =
        result - 2.0 * beta * beta *
                     primitive_overlap_cartesian(alpha, first, angular_first, beta, second, raised);
    const unsigned power = angular_axis(angular_second, axis);
    if (power >= 2) {
      Angular lowered = angular_second;
      add_angular_axis(lowered, axis, -2);
      result = result -
               0.5 * static_cast<double>(power * (power - 1)) *
                   primitive_overlap_cartesian(alpha, first, angular_first, beta, second, lowered);
    }
  }
  return result;
}

/**
 * Compact value-only overlap used by analytic one-electron derivatives.
 *
 * Center differentiation raises one basis function by one quantum, while the
 * kinetic operator may raise that result by two more. The generic Hermite
 * workspace is intentionally not enlarged for this force-only requirement;
 * the separable Obara-Saika overlap recurrence needs only the final t=0
 * integral for first-center order <=3 and second-center order <=6.
 */
__device__ inline double primitive_overlap_cartesian_compact(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second) {
  constexpr unsigned FirstDimension = kMaximumAngularMomentum + 1;
  constexpr unsigned SecondDimension = kMaximumAngularMomentum + 4;
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  const double inverse_two_exponent = 0.5 / exponent;
  const Vec3<double> product = product_center(alpha, first, beta, second);
  double result = 1.0;
  for (int axis = 0; axis < 3; ++axis) {
    const unsigned first_power = angular_axis(angular_first, axis);
    const unsigned second_power = angular_axis(angular_second, axis);
    double values[FirstDimension][SecondDimension]{};
    const double first_coordinate = vec_axis(first, axis);
    const double second_coordinate = vec_axis(second, axis);
    const double difference = first_coordinate - second_coordinate;
    values[0][0] = sqrt(kPi / exponent) * exp(-reduced * difference * difference);
    const double product_first = vec_axis(product, axis) - first_coordinate;
    const double product_second = vec_axis(product, axis) - second_coordinate;
    for (unsigned i = 1; i <= first_power; ++i) {
      values[i][0] = product_first * values[i - 1][0];
      if (i > 1) {
        values[i][0] += static_cast<double>(i - 1) * inverse_two_exponent * values[i - 2][0];
      }
    }
    for (unsigned j = 1; j <= second_power; ++j) {
      values[0][j] = product_second * values[0][j - 1];
      if (j > 1) {
        values[0][j] += static_cast<double>(j - 1) * inverse_two_exponent * values[0][j - 2];
      }
      for (unsigned i = 1; i <= first_power; ++i) {
        values[i][j] = product_first * values[i - 1][j] +
                       static_cast<double>(j) * inverse_two_exponent * values[i - 1][j - 1];
        if (i > 1) {
          values[i][j] += static_cast<double>(i - 1) * inverse_two_exponent * values[i - 2][j];
        }
      }
    }
    result *= values[first_power][second_power];
  }
  return result;
}

/** Value-only kinetic integral backed by the compact overlap recurrence. */
__device__ inline double primitive_kinetic_cartesian_compact(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second) {
  double result = beta * (2.0 * static_cast<double>(angular_total(angular_second)) + 3.0) *
                  primitive_overlap_cartesian_compact(alpha, first, angular_first, beta, second,
                                                      angular_second);
  for (int axis = 0; axis < 3; ++axis) {
    Angular raised = angular_second;
    add_angular_axis(raised, axis, 2);
    result -=
        2.0 * beta * beta *
        primitive_overlap_cartesian_compact(alpha, first, angular_first, beta, second, raised);
    const unsigned power = angular_axis(angular_second, axis);
    if (power >= 2) {
      Angular lowered = angular_second;
      add_angular_axis(lowered, axis, -2);
      result -=
          0.5 * static_cast<double>(power * (power - 1)) *
          primitive_overlap_cartesian_compact(alpha, first, angular_first, beta, second, lowered);
    }
  }
  return result;
}

/** Differentiate an overlap integral at its second Gaussian center. */
__device__ inline void primitive_overlap_second_center_gradient(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second, double (&gradient)[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    Angular raised = angular_second;
    add_angular_axis(raised, axis, 1);
    double value =
        2.0 * beta *
        primitive_overlap_cartesian_compact(alpha, first, angular_first, beta, second, raised);
    const unsigned power = angular_axis(angular_second, axis);
    if (power > 0) {
      Angular lowered = angular_second;
      add_angular_axis(lowered, axis, -1);
      value -= static_cast<double>(power) * primitive_overlap_cartesian_compact(
                                                alpha, first, angular_first, beta, second, lowered);
    }
    gradient[axis] = value;
  }
}

/** Differentiate a kinetic integral at its second Gaussian center. */
__device__ inline void primitive_kinetic_second_center_gradient(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second, double (&gradient)[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    Angular raised = angular_second;
    add_angular_axis(raised, axis, 1);
    double value =
        2.0 * beta *
        primitive_kinetic_cartesian_compact(alpha, first, angular_first, beta, second, raised);
    const unsigned power = angular_axis(angular_second, axis);
    if (power > 0) {
      Angular lowered = angular_second;
      add_angular_axis(lowered, axis, -1);
      value -= static_cast<double>(power) * primitive_kinetic_cartesian_compact(
                                                alpha, first, angular_first, beta, second, lowered);
    }
    gradient[axis] = value;
  }
}

}  // namespace vibeqc::scf::cuda_execution

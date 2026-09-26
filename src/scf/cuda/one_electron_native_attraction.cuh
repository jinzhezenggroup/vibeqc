#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/hermite_recurrence.cuh"

// Retained nuclear-attraction primitives for the Dual response path.
// Keep this operator arithmetic separate from host export and launch policy.
namespace vibeqc::scf::cuda_execution {

template <typename Scalar>
__device__ inline Scalar primitive_nuclear_attraction(const DeviceBatch& batch, std::int32_t system,
                                                      double alpha, const Vec3<Scalar>& first,
                                                      double beta, const Vec3<Scalar>& second,
                                                      std::int64_t derivative_coordinate) {
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  const Vec3<Scalar> center = product_center(alpha, first, beta, second);
  const Scalar prefactor =
      (2.0 * kPi / exponent) * qexp(-reduced * distance_squared(first, second));
  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t atom = batch.atom_offsets[system]; atom < batch.atom_offsets[system + 1];
       ++atom) {
    const Scalar argument =
        exponent *
        distance_squared(center, atom_position<Scalar>(batch, atom, derivative_coordinate));
    result = result - static_cast<double>(batch.atomic_numbers[atom]) * prefactor * boys0(argument);
  }
  return result;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ inline __noinline__ Scalar nuclear_attraction_cartesian_value(
    const DeviceBatch& batch, std::int32_t system, double exponent, const Vec3<Scalar>& product,
    const Angular& angular_first, const Angular& angular_second,
    const HermiteCoefficients<Scalar>* coefficients, std::int64_t derivative_coordinate) {
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum);
  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t atom = batch.atom_offsets[system]; atom < batch.atom_offsets[system + 1];
       ++atom) {
    CoulombAuxiliary<Scalar, MaximumAngular> auxiliary;
    fill_coulomb<MaximumAngular>(
        exponent, product, atom_position<Scalar>(batch, atom, derivative_coordinate), auxiliary);
    Scalar value = scalar<Scalar>(0.0);
    for (unsigned t = 0; t <= angular_first.x + angular_second.x; ++t) {
      for (unsigned u = 0; u <= angular_first.y + angular_second.y; ++u) {
        for (unsigned v = 0; v <= angular_first.z + angular_second.z; ++v) {
          value = value + coefficients[0].at(angular_first.x, angular_second.x, t) *
                              coefficients[1].at(angular_first.y, angular_second.y, u) *
                              coefficients[2].at(angular_first.z, angular_second.z, v) *
                              auxiliary.at(0, t, u, v);
        }
      }
    }
    result =
        result - static_cast<double>(batch.atomic_numbers[atom]) * (2.0 * kPi / exponent) * value;
  }
  return result;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ inline Scalar primitive_nuclear_attraction_cartesian(
    const DeviceBatch& batch, std::int32_t system, double alpha, const Vec3<Scalar>& first,
    const Angular& angular_first, double beta, const Vec3<Scalar>& second,
    const Angular& angular_second, std::int64_t derivative_coordinate) {
  const double exponent = alpha + beta;
  const Vec3<Scalar> product = product_center(alpha, first, beta, second);
  HermiteCoefficients<Scalar> coefficients[3];
  for (int axis = 0; axis < 3; ++axis) {
    fill_hermite(angular_axis(angular_first, axis), angular_axis(angular_second, axis),
                 vec_axis(product, axis), vec_axis(first, axis), vec_axis(second, axis), alpha,
                 beta, coefficients[axis]);
  }

  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum);
  return nuclear_attraction_cartesian_value<MaximumAngular>(batch, system, exponent, product,
                                                            angular_first, angular_second,
                                                            coefficients, derivative_coordinate);
}

}  // namespace vibeqc::scf::cuda_execution

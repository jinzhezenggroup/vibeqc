#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "scf/cuda/one_electron_native_attraction.cuh"
#include "scf/cuda/one_electron_native_overlap.cuh"

// Retained normalized AO contraction for the Dual one-electron response
// path, including public spherical expansion weights.
namespace vibeqc::scf::cuda_execution {

template <typename Scalar>
__device__ inline Scalar contracted_overlap(const DeviceBatch& batch, std::int32_t system,
                                            std::int32_t i, std::int32_t j,
                                            std::int64_t derivative_coordinate) {
  const std::int64_t ao_i = static_cast<std::int64_t>(system) * batch.nbf + i;
  const std::int64_t ao_j = static_cast<std::int64_t>(system) * batch.nbf + j;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const Vec3<Scalar> first =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_i], derivative_coordinate);
  const Vec3<Scalar> second =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_j], derivative_coordinate);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const Angular angular_first = ao_angular(batch, ao_i, 0);
  const Angular angular_second = ao_angular(batch, ao_j, 0);
  const bool all_s = first_terms == 1 && second_terms == 1 && is_s_function(angular_first) &&
                     is_s_function(angular_second);
  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      const double primitive_weight =
          batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
      if (all_s) {
        result = result + primitive_weight * ao_term_coefficient(batch, ao_i, 0) *
                              ao_term_coefficient(batch, ao_j, 0) *
                              primitive_overlap(batch.primitive_exponents[a], first,
                                                batch.primitive_exponents[b], second);
      } else {
        for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
          const Angular first_angular = ao_angular(batch, ao_i, first_term);
          const double first_coefficient = ao_term_coefficient(batch, ao_i, first_term);
          for (unsigned second_term = 0; second_term < second_terms; ++second_term) {
            result = result + primitive_weight * first_coefficient *
                                  ao_term_coefficient(batch, ao_j, second_term) *
                                  primitive_overlap_cartesian(batch.primitive_exponents[a], first,
                                                              first_angular,
                                                              batch.primitive_exponents[b], second,
                                                              ao_angular(batch, ao_j, second_term));
          }
        }
      }
    }
  }
  return result;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ inline __noinline__ Scalar contracted_hcore_cartesian(
    const DeviceBatch& batch, std::int32_t system, std::int64_t ao_i, std::int64_t ao_j,
    std::int32_t shell_i, std::int32_t shell_j, const Vec3<Scalar>& first,
    const Vec3<Scalar>& second, unsigned first_terms, unsigned second_terms,
    std::int64_t derivative_coordinate) {
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum);
  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      const double weight = batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
      for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
        const Angular first_angular = ao_angular(batch, ao_i, first_term);
        const double first_coefficient = ao_term_coefficient(batch, ao_i, first_term);
        for (unsigned second_term = 0; second_term < second_terms; ++second_term) {
          const Angular second_angular = ao_angular(batch, ao_j, second_term);
          result =
              result + weight * first_coefficient * ao_term_coefficient(batch, ao_j, second_term) *
                           (primitive_kinetic_cartesian(batch.primitive_exponents[a], first,
                                                        first_angular, batch.primitive_exponents[b],
                                                        second, second_angular) +
                            primitive_nuclear_attraction_cartesian<MaximumAngular>(
                                batch, system, batch.primitive_exponents[a], first, first_angular,
                                batch.primitive_exponents[b], second, second_angular,
                                derivative_coordinate));
        }
      }
    }
  }
  return result;
}

template <typename Scalar>
__device__ inline Scalar contracted_hcore(const DeviceBatch& batch, std::int32_t system,
                                          std::int32_t i, std::int32_t j,
                                          std::int64_t derivative_coordinate) {
  const std::int64_t ao_i = static_cast<std::int64_t>(system) * batch.nbf + i;
  const std::int64_t ao_j = static_cast<std::int64_t>(system) * batch.nbf + j;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const Vec3<Scalar> first =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_i], derivative_coordinate);
  const Vec3<Scalar> second =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_j], derivative_coordinate);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const Angular angular_first = ao_angular(batch, ao_i, 0);
  const Angular angular_second = ao_angular(batch, ao_j, 0);
  const bool all_s = first_terms == 1 && second_terms == 1 && is_s_function(angular_first) &&
                     is_s_function(angular_second);
  if (all_s) {
    Scalar result = scalar<Scalar>(0.0);
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        const double weight = batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
        result = result + weight * ao_term_coefficient(batch, ao_i, 0) *
                              ao_term_coefficient(batch, ao_j, 0) *
                              (primitive_kinetic(batch.primitive_exponents[a], first,
                                                 batch.primitive_exponents[b], second) +
                               primitive_nuclear_attraction(
                                   batch, system, batch.primitive_exponents[a], first,
                                   batch.primitive_exponents[b], second, derivative_coordinate));
      }
    }
    return result;
  }

  const unsigned maximum = batch.shell_angular[shell_i] + batch.shell_angular[shell_j];
  switch (maximum) {
    case 1:
      return contracted_hcore_cartesian<1>(batch, system, ao_i, ao_j, shell_i, shell_j, first,
                                           second, first_terms, second_terms,
                                           derivative_coordinate);
    case 2:
      return contracted_hcore_cartesian<2>(batch, system, ao_i, ao_j, shell_i, shell_j, first,
                                           second, first_terms, second_terms,
                                           derivative_coordinate);
    case 3:
      return contracted_hcore_cartesian<3>(batch, system, ao_i, ao_j, shell_i, shell_j, first,
                                           second, first_terms, second_terms,
                                           derivative_coordinate);
    case 4:
      return contracted_hcore_cartesian<4>(batch, system, ao_i, ao_j, shell_i, shell_j, first,
                                           second, first_terms, second_terms,
                                           derivative_coordinate);
    case 5:
      return contracted_hcore_cartesian<5>(batch, system, ao_i, ao_j, shell_i, shell_j, first,
                                           second, first_terms, second_terms,
                                           derivative_coordinate);
    case 6:
      return contracted_hcore_cartesian<6>(batch, system, ao_i, ao_j, shell_i, shell_j, first,
                                           second, first_terms, second_terms,
                                           derivative_coordinate);
  }
  return scalar<Scalar>(0.0);
}

}  // namespace vibeqc::scf::cuda_execution

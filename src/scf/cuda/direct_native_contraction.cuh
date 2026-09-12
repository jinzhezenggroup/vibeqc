#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_cartesian.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for contraction.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

template <unsigned MaximumAngular, typename Scalar>
__device__ inline __noinline__ Scalar contracted_eri_cartesian(
    const DeviceBatch& batch, std::int64_t ao_i, std::int64_t ao_j, std::int64_t ao_k,
    std::int64_t ao_l, std::int32_t shell_i, std::int32_t shell_j, std::int32_t shell_k,
    std::int32_t shell_l, std::int64_t derivative_coordinate) {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  const Vec3<Scalar> first =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_i], derivative_coordinate);
  const Vec3<Scalar> second =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_j], derivative_coordinate);
  const Vec3<Scalar> third =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_k], derivative_coordinate);
  const Vec3<Scalar> fourth =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_l], derivative_coordinate);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const unsigned third_terms = batch.ao_term_counts[ao_k];
  const unsigned fourth_terms = batch.ao_term_counts[ao_l];

  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[shell_k];
           c < batch.shell_primitive_offsets[shell_k + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[shell_l];
             d < batch.shell_primitive_offsets[shell_l + 1]; ++d) {
          const double weight = batch.primitive_coefficients[a] * batch.primitive_coefficients[b] *
                                batch.primitive_coefficients[c] * batch.primitive_coefficients[d];
          for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
            const Angular first_angular = ao_angular(batch, ao_i, first_term);
            const double first_coefficient = ao_term_coefficient(batch, ao_i, first_term);
            for (unsigned second_term = 0; second_term < second_terms; ++second_term) {
              const Angular second_angular = ao_angular(batch, ao_j, second_term);
              const double second_coefficient = ao_term_coefficient(batch, ao_j, second_term);
              for (unsigned third_term = 0; third_term < third_terms; ++third_term) {
                const Angular third_angular = ao_angular(batch, ao_k, third_term);
                const double third_coefficient = ao_term_coefficient(batch, ao_k, third_term);
                for (unsigned fourth_term = 0; fourth_term < fourth_terms; ++fourth_term) {
                  result = result + weight * first_coefficient * second_coefficient *
                                        third_coefficient *
                                        ao_term_coefficient(batch, ao_l, fourth_term) *
                                        primitive_eri_cartesian<MaximumAngular>(
                                            batch.primitive_exponents[a], first, first_angular,
                                            batch.primitive_exponents[b], second, second_angular,
                                            batch.primitive_exponents[c], third, third_angular,
                                            batch.primitive_exponents[d], fourth,
                                            ao_angular(batch, ao_l, fourth_term));
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ inline Scalar contracted_eri_order(const DeviceBatch& batch, std::int32_t system,
                                              std::int32_t i, std::int32_t j, std::int32_t k,
                                              std::int32_t l, std::int64_t derivative_coordinate) {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.nbf;
  const std::int64_t ao_i = base + i;
  const std::int64_t ao_j = base + j;
  const std::int64_t ao_k = base + k;
  const std::int64_t ao_l = base + l;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const std::int32_t shell_k = batch.ao_shells[ao_k];
  const std::int32_t shell_l = batch.ao_shells[ao_l];
  if constexpr (MaximumAngular == 0) {
    const Vec3<Scalar> first =
        atom_position<Scalar>(batch, batch.shell_atoms[shell_i], derivative_coordinate);
    const Vec3<Scalar> second =
        atom_position<Scalar>(batch, batch.shell_atoms[shell_j], derivative_coordinate);
    const Vec3<Scalar> third =
        atom_position<Scalar>(batch, batch.shell_atoms[shell_k], derivative_coordinate);
    const Vec3<Scalar> fourth =
        atom_position<Scalar>(batch, batch.shell_atoms[shell_l], derivative_coordinate);
    Scalar result = scalar<Scalar>(0.0);
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        for (std::int64_t c = batch.shell_primitive_offsets[shell_k];
             c < batch.shell_primitive_offsets[shell_k + 1]; ++c) {
          for (std::int64_t d = batch.shell_primitive_offsets[shell_l];
               d < batch.shell_primitive_offsets[shell_l + 1]; ++d) {
            const double weight = batch.primitive_coefficients[a] *
                                  batch.primitive_coefficients[b] *
                                  batch.primitive_coefficients[c] * batch.primitive_coefficients[d];
            result =
                result +
                weight * ao_term_coefficient(batch, ao_i, 0) * ao_term_coefficient(batch, ao_j, 0) *
                    ao_term_coefficient(batch, ao_k, 0) * ao_term_coefficient(batch, ao_l, 0) *
                    primitive_eri(batch.primitive_exponents[a], first, batch.primitive_exponents[b],
                                  second, batch.primitive_exponents[c], third,
                                  batch.primitive_exponents[d], fourth);
          }
        }
      }
    }
    return result;
  } else {
    return contracted_eri_cartesian<MaximumAngular, Scalar>(
        batch, ao_i, ao_j, ao_k, ao_l, shell_i, shell_j, shell_k, shell_l, derivative_coordinate);
  }
}

template <typename Scalar>
__device__ inline Scalar contracted_eri(const DeviceBatch& batch, std::int32_t system,
                                        std::int32_t i, std::int32_t j, std::int32_t k,
                                        std::int32_t l, std::int64_t derivative_coordinate) {
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.nbf;
  const std::int32_t shell_i = batch.ao_shells[base + i];
  const std::int32_t shell_j = batch.ao_shells[base + j];
  const std::int32_t shell_k = batch.ao_shells[base + k];
  const std::int32_t shell_l = batch.ao_shells[base + l];
  // Shell angular momentum is invariant across Cartesian expansion terms, so
  // one contracted-quartet dispatch covers every primitive and sparse
  // spherical term below it.
  const unsigned maximum = batch.shell_angular[shell_i] + batch.shell_angular[shell_j] +
                           batch.shell_angular[shell_k] + batch.shell_angular[shell_l];
  switch (maximum) {
    case 0:
      return contracted_eri_order<0, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 1:
      return contracted_eri_order<1, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 2:
      return contracted_eri_order<2, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 3:
      return contracted_eri_order<3, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 4:
      return contracted_eri_order<4, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 5:
      return contracted_eri_order<5, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 6:
      return contracted_eri_order<6, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 7:
      return contracted_eri_order<7, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 8:
      return contracted_eri_order<8, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 9:
      return contracted_eri_order<9, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 10:
      return contracted_eri_order<10, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 11:
      return contracted_eri_order<11, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
    case 12:
      return contracted_eri_order<12, Scalar>(batch, system, i, j, k, l, derivative_coordinate);
  }
  return scalar<Scalar>(0.0);
}

}  // namespace vibeqc::scf::cuda_execution

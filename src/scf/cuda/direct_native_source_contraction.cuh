#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_cartesian.cuh"
#include "scf/cuda/direct_native_order2_shell.cuh"
#include "scf/cuda/direct_native_shell_class.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for source contraction.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/**
 * Contract one quartet of normalized Cartesian source AOs.
 *
 * Public spherical AOs are handled by transforming their density before this
 * evaluator and their Fock matrix afterwards. Each source AO therefore has
 * exactly one angular component, eliminating the sparse term-product loops
 * from the dominant direct Fock and force recurrences.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ inline Scalar contracted_eri_cartesian_source_shell_class(
    const DeviceBatch& batch, std::int64_t ao_i, std::int64_t ao_j, std::int64_t ao_k,
    std::int64_t ao_l, std::int32_t shell_i, std::int32_t shell_j, std::int32_t shell_k,
    std::int32_t shell_l, std::int64_t derivative_coordinate) {
  constexpr unsigned MaximumAngular =
      FirstShellAngular + SecondShellAngular + ThirdShellAngular + FourthShellAngular;
  const Vec3<Scalar> first =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_i], derivative_coordinate);
  const Vec3<Scalar> second =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_j], derivative_coordinate);
  const Vec3<Scalar> third =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_k], derivative_coordinate);
  const Vec3<Scalar> fourth =
      atom_position<Scalar>(batch, batch.shell_atoms[shell_l], derivative_coordinate);
  const Angular angular_first = direct_ao_angular(batch, ao_i);
  const Angular angular_second = direct_ao_angular(batch, ao_j);
  const Angular angular_third = direct_ao_angular(batch, ao_k);
  const Angular angular_fourth = direct_ao_angular(batch, ao_l);
  using CoefficientScalar =
      std::conditional_t<std::is_same_v<Scalar, MixedPrecisionFloat>, Scalar, double>;
  const CoefficientScalar angular_coefficient =
      CoefficientScalar{batch.direct_ao_coefficients[ao_i] * batch.direct_ao_coefficients[ao_j] *
                        batch.direct_ao_coefficients[ao_k] * batch.direct_ao_coefficients[ao_l]};

  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[shell_k];
           c < batch.shell_primitive_offsets[shell_k + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[shell_l];
             d < batch.shell_primitive_offsets[shell_l + 1]; ++d) {
          const CoefficientScalar weight = angular_coefficient * batch.primitive_coefficients[a] *
                                           batch.primitive_coefficients[b] *
                                           batch.primitive_coefficients[c] *
                                           batch.primitive_coefficients[d];
          if constexpr (MaximumAngular == 0) {
            result = result + weight * primitive_eri(batch.primitive_exponents[a], first,
                                                     batch.primitive_exponents[b], second,
                                                     batch.primitive_exponents[c], third,
                                                     batch.primitive_exponents[d], fourth);
          } else {
            result =
                result +
                weight * primitive_eri_cartesian_shell_class<FirstShellAngular, SecondShellAngular,
                                                             ThirdShellAngular, FourthShellAngular>(
                             batch.primitive_exponents[a], first, angular_first,
                             batch.primitive_exponents[b], second, angular_second,
                             batch.primitive_exponents[c], third, angular_third,
                             batch.primitive_exponents[d], fourth, angular_fourth);
          }
        }
      }
    }
  }
  return result;
}

/** Canonicalize one Cartesian source quartet to its exact shell class. */
template <unsigned ShellClass, typename Scalar>
__device__ inline Scalar contracted_eri_cartesian_source_shell_class(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l, std::int64_t derivative_coordinate) {
  static_assert(ShellClass < detail::kDirectQuartetShellClassCount);
  constexpr unsigned FirstPairClass = direct_triangular_class_high(ShellClass);
  constexpr unsigned SecondPairClass = ShellClass - FirstPairClass * (FirstPairClass + 1) / 2;
  constexpr unsigned FirstShellAngular = direct_triangular_class_high(FirstPairClass);
  constexpr unsigned SecondShellAngular =
      FirstPairClass - FirstShellAngular * (FirstShellAngular + 1) / 2;
  constexpr unsigned ThirdShellAngular = direct_triangular_class_high(SecondPairClass);
  constexpr unsigned FourthShellAngular =
      SecondPairClass - ThirdShellAngular * (ThirdShellAngular + 1) / 2;

  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  std::int32_t shell_i = batch.direct_ao_shells[base + i];
  std::int32_t shell_j = batch.direct_ao_shells[base + j];
  std::int32_t shell_k = batch.direct_ao_shells[base + k];
  std::int32_t shell_l = batch.direct_ao_shells[base + l];
  unsigned angular_i = batch.shell_angular[shell_i];
  unsigned angular_j = batch.shell_angular[shell_j];
  unsigned angular_k = batch.shell_angular[shell_k];
  unsigned angular_l = batch.shell_angular[shell_l];

  if (angular_i < angular_j) {
    const std::int32_t ao = i;
    i = j;
    j = ao;
    const std::int32_t shell = shell_i;
    shell_i = shell_j;
    shell_j = shell;
    const unsigned angular = angular_i;
    angular_i = angular_j;
    angular_j = angular;
  }
  if (angular_k < angular_l) {
    const std::int32_t ao = k;
    k = l;
    l = ao;
    const std::int32_t shell = shell_k;
    shell_k = shell_l;
    shell_l = shell;
    const unsigned angular = angular_k;
    angular_k = angular_l;
    angular_l = angular;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(angular_i, angular_j);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(angular_k, angular_l);
  if (first_pair_class < second_pair_class) {
    const std::int32_t first_ao = i;
    const std::int32_t second_ao = j;
    i = k;
    j = l;
    k = first_ao;
    l = second_ao;
    const std::int32_t first_shell = shell_i;
    const std::int32_t second_shell = shell_j;
    shell_i = shell_k;
    shell_j = shell_l;
    shell_k = first_shell;
    shell_l = second_shell;
  }

  // The order-two shell classes have closed-form Cartesian contractions below
  // (the same routines used by the handwritten Fock consumer).  Reuse those
  // routines for Schwarz diagonals as well: the generic source evaluator keeps
  // a much larger recurrence frame alive and can exceed CUDA's per-thread local
  // stack on p/s and d/s quartets even though the order-two result is tiny.
  if constexpr (ShellClass == 2 || ShellClass == 3 || ShellClass == 6) {
    const unsigned first_count = (static_cast<unsigned>(FirstShellAngular) + 1U) *
                                 (static_cast<unsigned>(FirstShellAngular) + 2U) / 2U;
    const unsigned second_count = (static_cast<unsigned>(SecondShellAngular) + 1U) *
                                  (static_cast<unsigned>(SecondShellAngular) + 2U) / 2U;
    const unsigned third_count = (static_cast<unsigned>(ThirdShellAngular) + 1U) *
                                 (static_cast<unsigned>(ThirdShellAngular) + 2U) / 2U;
    const unsigned fourth_count = (static_cast<unsigned>(FourthShellAngular) + 1U) *
                                  (static_cast<unsigned>(FourthShellAngular) + 2U) / 2U;
    unsigned first_component = 0U;
    unsigned second_component = 0U;
    unsigned third_component = 0U;
    unsigned fourth_component = 0U;
    if (!direct_shell_component_index(batch, base, shell_i, i, first_component) ||
        !direct_shell_component_index(batch, base, shell_j, j, second_component) ||
        !direct_shell_component_index(batch, base, shell_k, k, third_component) ||
        !direct_shell_component_index(batch, base, shell_l, l, fourth_component)) {
      return scalar<Scalar>(0.0);
    }
    if (first_component >= first_count || second_component >= second_count ||
        third_component >= third_count || fourth_component >= fourth_count) {
      return scalar<Scalar>(0.0);
    }
    const unsigned component =
        (((first_component * second_count + second_component) * third_count + third_component) *
             fourth_count +
         fourth_component);
    const unsigned active_component_mask = 1U << component;
    Order2IntegralVector integral{};
    if constexpr (ShellClass == 2) {
      integral = contracted_eri_cartesian_source_order2_shell<1, 0, 1, 0>(
          batch, shell_i, shell_j, shell_k, shell_l, active_component_mask);
    } else if constexpr (ShellClass == 3) {
      integral = contracted_eri_cartesian_source_order2_shell<1, 1, 0, 0>(
          batch, shell_i, shell_j, shell_k, shell_l, active_component_mask);
    } else {
      integral = contracted_eri_cartesian_source_order2_shell<2, 0, 0, 0>(
          batch, shell_i, shell_j, shell_k, shell_l, active_component_mask);
    }
    return static_cast<Scalar>(integral.component[component]);
  }

  return contracted_eri_cartesian_source_shell_class<FirstShellAngular, SecondShellAngular,
                                                     ThirdShellAngular, FourthShellAngular, Scalar>(
      batch, base + i, base + j, base + k, base + l, shell_i, shell_j, shell_k, shell_l,
      derivative_coordinate);
}

/** Dispatch one angular-order task to its Cartesian source evaluator. */
template <unsigned AngularOrder, typename Scalar>
__device__ inline Scalar dispatch_contracted_eri_cartesian_source_shell_class(
    unsigned runtime_shell_class, const DeviceBatch& batch, std::int32_t system, std::int32_t i,
    std::int32_t j, std::int32_t k, std::int32_t l, std::int64_t derivative_coordinate) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
#define VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(ShellClass)                         \
  case ShellClass:                                                                \
    if constexpr (direct_shell_class_angular_order(ShellClass) == AngularOrder) { \
      return contracted_eri_cartesian_source_shell_class<ShellClass, Scalar>(     \
          batch, system, i, j, k, l, derivative_coordinate);                      \
    }                                                                             \
    break
  switch (runtime_shell_class) {
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(0);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(1);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(2);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(3);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(4);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(5);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(6);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(7);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(8);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(9);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(10);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(11);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(12);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(13);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(14);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(15);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(16);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(17);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(18);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(19);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(20);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(21);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(22);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(23);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(24);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(25);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(26);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(27);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(28);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(29);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(30);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(31);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(32);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(33);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(34);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(35);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(36);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(37);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(38);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(39);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(40);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(41);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(42);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(43);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(44);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(45);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(46);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(47);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(48);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(49);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(50);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(51);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(52);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(53);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(54);
  }
#undef VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE
  return scalar<Scalar>(0.0);
}

template <typename Scalar>
__device__ inline Scalar contracted_eri_cartesian_source(const DeviceBatch& batch,
                                                         std::int32_t system, std::int32_t i,
                                                         std::int32_t j, std::int32_t k,
                                                         std::int32_t l,
                                                         std::int64_t derivative_coordinate) {
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  const std::int32_t shell_i = batch.direct_ao_shells[base + i];
  const std::int32_t shell_j = batch.direct_ao_shells[base + j];
  const std::int32_t shell_k = batch.direct_ao_shells[base + k];
  const std::int32_t shell_l = batch.direct_ao_shells[base + l];
  const unsigned angular_order = batch.shell_angular[shell_i] + batch.shell_angular[shell_j] +
                                 batch.shell_angular[shell_k] + batch.shell_angular[shell_l];
  const unsigned shell_class =
      direct_quartet_shell_class_device(batch.shell_angular[shell_i], batch.shell_angular[shell_j],
                                        batch.shell_angular[shell_k], batch.shell_angular[shell_l]);
#define VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(Order)                                \
  case Order:                                                                   \
    return dispatch_contracted_eri_cartesian_source_shell_class<Order, Scalar>( \
        shell_class, batch, system, i, j, k, l, derivative_coordinate)
  switch (angular_order) {
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(0);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(1);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(2);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(3);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(4);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(5);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(6);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(7);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(8);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(9);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(10);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(11);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(12);
  }
#undef VIBEQC_DIRECT_SOURCE_ANGULAR_CASE
  return scalar<Scalar>(0.0);
}

}  // namespace vibeqc::scf::cuda_execution

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/allocation_measurement.hpp"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#include "scf/aot_shell_registry.hpp"
#include "scf/cuda/arena.hpp"
#include "scf/cuda/basis_transform_kernels.hpp"
#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/checked_layout.hpp"
#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/device_timer.cuh"
#include "scf/cuda/direct_bounded_pages.hpp"
#include "scf/cuda/direct_bounded_tasks.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_density_bounds.hpp"
#include "scf/cuda/direct_generated_tasks.hpp"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_page_screening.cuh"
#include "scf/cuda/direct_pair_cache.hpp"
#include "scf/cuda/direct_queue_diagnostics.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/direct_queue_profile.cuh"
#include "scf/cuda/direct_queue_scan.hpp"
#include "scf/cuda/direct_resident_tasks.hpp"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/direct_task_encoding.cuh"
#include "scf/cuda/direct_tile_compaction.hpp"
#include "scf/cuda/direct_tile_validation.hpp"
#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/hermite_recurrence.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/matrix_library.hpp"
#include "scf/cuda/metadata_upload.hpp"
#include "scf/cuda/nuclear_kernels.hpp"
#include "scf/cuda/one_electron_derivatives.cuh"
#include "scf/cuda/one_electron_export_kernels.hpp"
#include "scf/cuda/one_electron_force_reference.hpp"
#include "scf/cuda/one_electron_force_workspace.hpp"
#include "scf/cuda/one_electron_values.cuh"
#include "scf/cuda/one_electron_view.hpp"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/queue_plan.hpp"
#include "scf/cuda/reference_export.cuh"
#include "scf/cuda/resources.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda/runtime_support.hpp"
#include "scf/cuda/scalar_math.cuh"
#include "scf/cuda/scf_convergence_kernels.hpp"
#include "scf/cuda/scf_density_kernels.hpp"
#include "scf/cuda/scf_diis_kernels.hpp"
#include "scf/cuda/scf_matrix_kernels.hpp"
#include "scf/cuda/scf_state_kernels.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/cuda_direct_jk.hpp"
#include "scf/cuda_eigensolver_policy.hpp"
#include "scf/cuda_weighted_eri.hpp"
#include "scf/direct_task_layout.hpp"
#include "scf/generated_shell_task.hpp"
#include "scf/mean_field.hpp"
#include "scf/rhf.hpp"
#include "tensor/metrics.hpp"
#include "weighted_eri.cuh"

namespace vibeqc::scf {

namespace {

using namespace cuda_execution;

// Runtime policy parsing lives in a host-only C++ TU. Keep the numerical CUDA
// source's call sites unchanged while making policy-only edits incremental.
// The legacy spellings below remain documented here for source-level tooling;
// their actual ``std::getenv`` calls and thresholds are implemented by
// ``scf/cuda/rhf_policy.cpp`` so editing policy does not rebuild this TU:
//   std::getenv("VIBEQC_FINAL_FOCK_REBUILD")
//   std::getenv("VIBEQC_PPPS_SIGNATURE_BUCKETING")
//   std::getenv("VIBEQC_PPPS_BLOCK_THREADS")
//   std::getenv("VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING")
//   std::getenv("VIBEQC_ONE_ELECTRON_FORCE_SCALAR")
//   std::getenv("VIBEQC_PSSS_RESIDENT_BRA")
//   scalar_one_electron_force_environment == nullptr
//   kTightConvergedFockReuseDensityRms = 1.0e-12
//   kExpandedConvergedFockReuseDensityTolerance = 1.0e-9
//   kExpandedConvergedFockReuseDensityRms = 2.0e-9
//   kAutoMixedPrecisionErrorBudgetFraction = 6.25e-02
//   kFloat32UnitRoundoff = 5.9604644775390625e-08
using cuda_policy::bounded_direct_aot_only_diagnostic_requested;
using cuda_policy::bounded_direct_count_diagnostic_requested;
using cuda_policy::bounded_direct_fock_only_diagnostic_requested;
using cuda_policy::bounded_direct_streaming_override_requested;
using cuda_policy::bounded_fock_class_timing_requested;
using cuda_policy::configured_mixed_precision_fock_threshold;
using cuda_policy::converged_fock_reuse_density_rms;
using cuda_policy::direct_tile_validation_requested;
using cuda_policy::force_density_product_screening_requested;
using cuda_policy::graph_native_eigensolver_override_requested;
using cuda_policy::MixedPrecisionFockPolicy;
using cuda_policy::MixedPrecisionItemPolicy;
using cuda_policy::one_electron_force_scalar_requested;
using cuda_policy::ppps_resident_block_threads_requested;
using cuda_policy::ppps_signature_bucketing_requested;
using cuda_policy::ppss_signature_bucketing_requested;
using cuda_policy::psps_signature_bucketing_requested;
using cuda_policy::resident_ppps_bra_requested;
using cuda_policy::resident_psss_bra_requested;
using cuda_policy::resolve_mixed_precision_fock_policy;
using cuda_policy::resolve_mixed_precision_item;
using cuda_policy::reuse_converged_fock_requested;
using cuda_policy::xsyev_probe_skip_diagnostic_requested;

__device__ std::size_t eri_index(std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                                 std::size_t n) {
  return ((i * n + j) * n + k) * n + l;
}

template <typename Scalar>
__device__ Scalar primitive_eri(double alpha, const Vec3<Scalar>& first, double beta,
                                const Vec3<Scalar>& second, double gamma, const Vec3<Scalar>& third,
                                double delta, const Vec3<Scalar>& fourth) {
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

/** Hermite workspace bounded by one exact shell-pair class. */
template <typename Scalar, unsigned FirstAngular, unsigned SecondAngular>
struct ShellPairHermiteCoefficients {
  static constexpr unsigned kIDimension = FirstAngular + 1;
  static constexpr unsigned kJDimension = SecondAngular + 1;
  // One zero boundary element is required because the recurrence reads t+1.
  static constexpr unsigned kTDimension = FirstAngular + SecondAngular + 2;
  Scalar data[kIDimension * kJDimension * kTDimension];

  __device__ Scalar& at(unsigned i, unsigned j, unsigned t) {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
  __device__ const Scalar& at(unsigned i, unsigned j, unsigned t) const {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
};

template <unsigned FirstAngular, unsigned SecondAngular, typename Scalar>
__device__ void fill_shell_pair_hermite(
    unsigned maximum_i, unsigned maximum_j, Scalar product, Scalar center_a, Scalar center_b,
    double alpha, double beta,
    ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>& coefficients) {
  static_assert(FirstAngular <= kMaximumAngularMomentum);
  static_assert(SecondAngular <= kMaximumAngularMomentum);
  for (unsigned item = 0;
       item < ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>::kIDimension *
                  ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>::kJDimension *
                  ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>::kTDimension;
       ++item) {
    coefficients.data[item] = scalar<Scalar>(0.0);
  }
  const double p = alpha + beta;
  const double mu = alpha * beta / p;
  const Scalar ab = center_a - center_b;
  coefficients.at(0, 0, 0) = qexp(-mu * ab * ab);
  const Scalar pa = product - center_a;
  const Scalar pb = product - center_b;
  const double inverse_two_p = 0.5 / p;

  for (unsigned i = 0; i <= maximum_i; ++i) {
    for (unsigned j = 0; j <= maximum_j; ++j) {
      if (i == 0 && j == 0) continue;
      if (i > 0) {
        coefficients.at(i, j, 0) = pa * coefficients.at(i - 1, j, 0) + coefficients.at(i - 1, j, 1);
      } else {
        coefficients.at(i, j, 0) = pb * coefficients.at(i, j - 1, 0) + coefficients.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coefficients.at(i, j, t) = pa * coefficients.at(i - 1, j, t) +
                                     inverse_two_p * coefficients.at(i - 1, j, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i - 1, j, t + 1);
        } else {
          coefficients.at(i, j, t) = pb * coefficients.at(i, j - 1, t) +
                                     inverse_two_p * coefficients.at(i, j - 1, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i, j - 1, t + 1);
        }
      }
    }
  }
}

template <unsigned MaximumAngular, typename Scalar, typename FirstCoefficients,
          typename SecondCoefficients>
__device__ __noinline__ Scalar eri_cartesian_value(
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
__device__ Scalar primitive_eri_cartesian(double alpha, const Vec3<Scalar>& first,
                                          const Angular& angular_first, double beta,
                                          const Vec3<Scalar>& second, const Angular& angular_second,
                                          double gamma, const Vec3<Scalar>& third,
                                          const Angular& angular_third, double delta,
                                          const Vec3<Scalar>& fourth,
                                          const Angular& angular_fourth) {
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
__device__ Scalar primitive_eri_psss(int axis, double alpha, const Vec3<Scalar>& first, double beta,
                                     const Vec3<Scalar>& second, double gamma,
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

/** One nonzero three-dimensional Hermite coefficient through order two. */
template <typename Scalar>
struct LowOrderHermiteTerm {
  // Each Cartesian derivative occupies two bits. Adding two states therefore
  // combines pair derivatives without carrying between x, y, and z.
  unsigned derivative_state;
  Scalar coefficient;
};

/** Compact shell-pair expansion; total order two reaches at most four terms. */
template <typename Scalar>
struct LowOrderPairExpansion {
  LowOrderHermiteTerm<Scalar> terms[4];
};

__device__ unsigned low_order_derivative_state(int axis) { return 1U << (2 * axis); }

__device__ unsigned low_order_derivative_total(unsigned state) {
  return (state & 3U) + ((state >> 2U) & 3U) + ((state >> 4U) & 3U);
}

/**
 * Generate only the nonzero Hermite terms of one order-0/1/2 shell pair.
 *
 * The Gaussian pair decay is deliberately excluded and applied once by the
 * primitive quartet. At order two, retaining duplicate first-derivative terms
 * for a repeated axis keeps one runtime component path for d and p-p AOs while
 * still bounding the expansion at four entries.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, typename Scalar>
__device__ LowOrderPairExpansion<Scalar> make_low_order_pair_expansion(
    double exponent, const Vec3<Scalar>& product, const Vec3<Scalar>& first,
    const Angular& angular_first, const Vec3<Scalar>& second, const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 2);
  LowOrderPairExpansion<Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[2];
    Scalar shifts[2];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = low_order_derivative_state(axis);
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        ++quantum_count;
      }
    }

    expansion.terms[0] = {0U, shifts[0]};
    expansion.terms[1] = {derivative_states[0], scalar<Scalar>(inverse_two_exponent)};
    if constexpr (PairOrder == 2) {
      // A repeated Cartesian axis contributes the recurrence's +1/(2p)
      // correction. The two first-derivative entries then share a state and
      // sum to the exact E1 coefficient during contraction.
      const double repeated_axis_correction =
          derivative_states[0] == derivative_states[1] ? inverse_two_exponent : 0.0;
      expansion.terms[0].coefficient =
          shifts[0] * shifts[1] + scalar<Scalar>(repeated_axis_correction);
      expansion.terms[1].coefficient = inverse_two_exponent * shifts[1];
      expansion.terms[2] = {derivative_states[1], inverse_two_exponent * shifts[0]};
      expansion.terms[3] = {derivative_states[0] + derivative_states[1],
                            scalar<Scalar>(inverse_two_exponent * inverse_two_exponent)};
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most two. */
template <typename Scalar>
__device__ Scalar low_order_coulomb(unsigned derivative_state, double rho,
                                    const Vec3<Scalar>& product_difference, const Scalar* boys) {
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
__device__ Scalar primitive_eri_order2(double alpha, const Vec3<Scalar>& first,
                                       const Angular& angular_first, double beta,
                                       const Vec3<Scalar>& second, const Angular& angular_second,
                                       double gamma, const Vec3<Scalar>& third,
                                       const Angular& angular_third, double delta,
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

/** Maximum full Cartesian output count among the three order-two classes. */
struct Order2IntegralVector {
  double component[9];
};

/**
 * Contract one canonical order-two shell quartet into all requested outputs.
 *
 * `active_component_mask` uses the canonical full Cartesian product order.
 * Symmetry-diagonal tasks therefore avoid computing product entries that are
 * absent from their lower-triangular AO-quartet domain.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular>
__device__ __noinline__ Order2IntegralVector contracted_eri_cartesian_source_order2_shell(
    const DeviceBatch& batch, std::int32_t first_shell, std::int32_t second_shell,
    std::int32_t third_shell, std::int32_t fourth_shell, unsigned active_component_mask) {
  static_assert(FirstShellAngular + SecondShellAngular + ThirdShellAngular + FourthShellAngular ==
                2);
  constexpr unsigned FirstCount = order2_shell_component_count<FirstShellAngular>();
  constexpr unsigned SecondCount = order2_shell_component_count<SecondShellAngular>();
  constexpr unsigned ThirdCount = order2_shell_component_count<ThirdShellAngular>();
  constexpr unsigned FourthCount = order2_shell_component_count<FourthShellAngular>();
  constexpr unsigned OutputCount = FirstCount * SecondCount * ThirdCount * FourthCount;
  static_assert(OutputCount <= 9);

  const std::int32_t shells[4] = {first_shell, second_shell, third_shell, fourth_shell};
  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[first_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[second_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[third_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1),
  };
  const std::int64_t ao_begin[4] = {
      batch.shell_direct_ao_offsets[first_shell],
      batch.shell_direct_ao_offsets[second_shell],
      batch.shell_direct_ao_offsets[third_shell],
      batch.shell_direct_ao_offsets[fourth_shell],
  };
  const double first_pair_distance = distance_squared(positions[0], positions[1]);
  const double second_pair_distance = distance_squared(positions[2], positions[3]);

  Order2IntegralVector result{};
  for (std::int64_t a = batch.shell_primitive_offsets[shells[0]];
       a < batch.shell_primitive_offsets[shells[0] + 1]; ++a) {
    const double alpha = batch.primitive_exponents[a];
    const double coefficient_a = batch.primitive_coefficients[a];
    for (std::int64_t b = batch.shell_primitive_offsets[shells[1]];
         b < batch.shell_primitive_offsets[shells[1] + 1]; ++b) {
      const double beta = batch.primitive_exponents[b];
      const double p = alpha + beta;
      const double mu = alpha * beta / p;
      const Vec3<double> product_p = product_center(alpha, positions[0], beta, positions[1]);
      const double first_pair_coefficient = coefficient_a * batch.primitive_coefficients[b];
      for (std::int64_t c = batch.shell_primitive_offsets[shells[2]];
           c < batch.shell_primitive_offsets[shells[2] + 1]; ++c) {
        const double gamma = batch.primitive_exponents[c];
        const double first_three_coefficient =
            first_pair_coefficient * batch.primitive_coefficients[c];
        for (std::int64_t d = batch.shell_primitive_offsets[shells[3]];
             d < batch.shell_primitive_offsets[shells[3] + 1]; ++d) {
          const double delta = batch.primitive_exponents[d];
          const double q = gamma + delta;
          const double nu = gamma * delta / q;
          const double rho = p * q / (p + q);
          const Vec3<double> product_q = product_center(gamma, positions[2], delta, positions[3]);
          const double x = product_p.x - product_q.x;
          const double y = product_p.y - product_q.y;
          const double z = product_p.z - product_q.z;
          double boys[3];
          boys_values<2>(rho * (x * x + y * y + z * z), boys);
          const Order2CoulombValues coulomb = order2_coulomb_values(rho, x, y, z, boys);
          const double pair_decay = exp(-mu * first_pair_distance - nu * second_pair_distance);
          const double primitive_coefficient =
              first_three_coefficient * batch.primitive_coefficients[d];
          const double common =
              primitive_coefficient * 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;

          if constexpr (FirstShellAngular == 1 && SecondShellAngular == 0 &&
                        ThirdShellAngular == 1 && FourthShellAngular == 0) {
            const double hp = 0.5 / p;
            const double hq = 0.5 / q;
            const double hpq = hp * hq;
            const Vec3<double> pa{product_p.x - positions[0].x, product_p.y - positions[0].y,
                                  product_p.z - positions[0].z};
            const Vec3<double> qc{product_q.x - positions[2].x, product_q.y - positions[2].y,
                                  product_q.z - positions[2].z};
            if ((active_component_mask & (1U << 0)) != 0U) {
              result.component[0] += common * (pa.x * qc.x * coulomb.c0 + hp * qc.x * coulomb.cx -
                                               hq * pa.x * coulomb.cx - hpq * coulomb.cxx);
            }
            if ((active_component_mask & (1U << 1)) != 0U) {
              result.component[1] += common * (pa.x * qc.y * coulomb.c0 + hp * qc.y * coulomb.cx -
                                               hq * pa.x * coulomb.cy - hpq * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 2)) != 0U) {
              result.component[2] += common * (pa.x * qc.z * coulomb.c0 + hp * qc.z * coulomb.cx -
                                               hq * pa.x * coulomb.cz - hpq * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 3)) != 0U) {
              result.component[3] += common * (pa.y * qc.x * coulomb.c0 + hp * qc.x * coulomb.cy -
                                               hq * pa.y * coulomb.cx - hpq * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 4)) != 0U) {
              result.component[4] += common * (pa.y * qc.y * coulomb.c0 + hp * qc.y * coulomb.cy -
                                               hq * pa.y * coulomb.cy - hpq * coulomb.cyy);
            }
            if ((active_component_mask & (1U << 5)) != 0U) {
              result.component[5] += common * (pa.y * qc.z * coulomb.c0 + hp * qc.z * coulomb.cy -
                                               hq * pa.y * coulomb.cz - hpq * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 6)) != 0U) {
              result.component[6] += common * (pa.z * qc.x * coulomb.c0 + hp * qc.x * coulomb.cz -
                                               hq * pa.z * coulomb.cx - hpq * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 7)) != 0U) {
              result.component[7] += common * (pa.z * qc.y * coulomb.c0 + hp * qc.y * coulomb.cz -
                                               hq * pa.z * coulomb.cy - hpq * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 8)) != 0U) {
              result.component[8] += common * (pa.z * qc.z * coulomb.c0 + hp * qc.z * coulomb.cz -
                                               hq * pa.z * coulomb.cz - hpq * coulomb.czz);
            }
          } else if constexpr (FirstShellAngular == 1 && SecondShellAngular == 1) {
            const double h = 0.5 / p;
            const double h2 = h * h;
            const Vec3<double> pa{product_p.x - positions[0].x, product_p.y - positions[0].y,
                                  product_p.z - positions[0].z};
            const Vec3<double> pb{product_p.x - positions[1].x, product_p.y - positions[1].y,
                                  product_p.z - positions[1].z};
            if ((active_component_mask & (1U << 0)) != 0U) {
              result.component[0] += common * ((pa.x * pb.x + h) * coulomb.c0 +
                                               h * (pb.x + pa.x) * coulomb.cx + h2 * coulomb.cxx);
            }
            if ((active_component_mask & (1U << 1)) != 0U) {
              result.component[1] += common * (pa.x * pb.y * coulomb.c0 + h * pb.y * coulomb.cx +
                                               h * pa.x * coulomb.cy + h2 * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 2)) != 0U) {
              result.component[2] += common * (pa.x * pb.z * coulomb.c0 + h * pb.z * coulomb.cx +
                                               h * pa.x * coulomb.cz + h2 * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 3)) != 0U) {
              result.component[3] += common * (pa.y * pb.x * coulomb.c0 + h * pb.x * coulomb.cy +
                                               h * pa.y * coulomb.cx + h2 * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 4)) != 0U) {
              result.component[4] += common * ((pa.y * pb.y + h) * coulomb.c0 +
                                               h * (pb.y + pa.y) * coulomb.cy + h2 * coulomb.cyy);
            }
            if ((active_component_mask & (1U << 5)) != 0U) {
              result.component[5] += common * (pa.y * pb.z * coulomb.c0 + h * pb.z * coulomb.cy +
                                               h * pa.y * coulomb.cz + h2 * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 6)) != 0U) {
              result.component[6] += common * (pa.z * pb.x * coulomb.c0 + h * pb.x * coulomb.cz +
                                               h * pa.z * coulomb.cx + h2 * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 7)) != 0U) {
              result.component[7] += common * (pa.z * pb.y * coulomb.c0 + h * pb.y * coulomb.cz +
                                               h * pa.z * coulomb.cy + h2 * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 8)) != 0U) {
              result.component[8] += common * ((pa.z * pb.z + h) * coulomb.c0 +
                                               h * (pb.z + pa.z) * coulomb.cz + h2 * coulomb.czz);
            }
          } else {
            static_assert(FirstShellAngular == 2 && SecondShellAngular == 0 &&
                          ThirdShellAngular == 0 && FourthShellAngular == 0);
            const double h = 0.5 / p;
            const double h2 = h * h;
            const Vec3<double> pa{product_p.x - positions[0].x, product_p.y - positions[0].y,
                                  product_p.z - positions[0].z};
            if ((active_component_mask & (1U << 0)) != 0U) {
              result.component[0] += common * ((pa.x * pa.x + h) * coulomb.c0 +
                                               2.0 * h * pa.x * coulomb.cx + h2 * coulomb.cxx);
            }
            if ((active_component_mask & (1U << 1)) != 0U) {
              result.component[1] += common * (pa.x * pa.y * coulomb.c0 + h * pa.y * coulomb.cx +
                                               h * pa.x * coulomb.cy + h2 * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 2)) != 0U) {
              result.component[2] += common * (pa.x * pa.z * coulomb.c0 + h * pa.z * coulomb.cx +
                                               h * pa.x * coulomb.cz + h2 * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 3)) != 0U) {
              result.component[3] += common * ((pa.y * pa.y + h) * coulomb.c0 +
                                               2.0 * h * pa.y * coulomb.cy + h2 * coulomb.cyy);
            }
            if ((active_component_mask & (1U << 4)) != 0U) {
              result.component[4] += common * (pa.y * pa.z * coulomb.c0 + h * pa.z * coulomb.cy +
                                               h * pa.y * coulomb.cz + h2 * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 5)) != 0U) {
              result.component[5] += common * ((pa.z * pa.z + h) * coulomb.c0 +
                                               2.0 * h * pa.z * coulomb.cz + h2 * coulomb.czz);
            }
          }
        }
      }
    }
  }

  // Cartesian normalization is primitive-independent. Applying it once after
  // contraction avoids four coefficient loads for every component of every
  // primitive quartet.
  unsigned output = 0;
#pragma unroll
  for (unsigned first_component = 0; first_component < FirstCount; ++first_component) {
#pragma unroll
    for (unsigned second_component = 0; second_component < SecondCount; ++second_component) {
#pragma unroll
      for (unsigned third_component = 0; third_component < ThirdCount; ++third_component) {
#pragma unroll
        for (unsigned fourth_component = 0; fourth_component < FourthCount;
             ++fourth_component, ++output) {
          if ((active_component_mask & (1U << output)) == 0U) continue;
          result.component[output] *= batch.direct_ao_coefficients[ao_begin[0] + first_component] *
                                      batch.direct_ao_coefficients[ao_begin[1] + second_component] *
                                      batch.direct_ao_coefficients[ao_begin[2] + third_component] *
                                      batch.direct_ao_coefficients[ao_begin[3] + fourth_component];
        }
      }
    }
  }
  return result;
}

/** Exact-sized sparse pair expansion used only by total-order-3 quartets. */
template <unsigned PairOrder, typename Scalar>
struct ThirdOrderPairExpansion {
  static_assert(PairOrder <= 3);
  LowOrderHermiteTerm<Scalar> terms[1U << PairOrder];
};

/**
 * Generate a shell pair through order three from its angular quanta.
 *
 * The base expansion is the product of one first-order factor per quantum.
 * Two quanta on the same Cartesian axis additionally have one Gaussian Wick
 * contraction, 1/(2p). Through order three, adding that contraction to the
 * surviving base terms produces the complete Hermite expansion while keeping
 * the exact 1/2/4/8-term bound.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, typename Scalar>
__device__ ThirdOrderPairExpansion<FirstShellAngular + SecondShellAngular, Scalar>
make_third_order_pair_expansion(double exponent, const Vec3<Scalar>& product,
                                const Vec3<Scalar>& first, const Angular& angular_first,
                                const Vec3<Scalar>& second, const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 3);
  constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  ThirdOrderPairExpansion<PairOrder, Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[QuantumStorage];
    Scalar shifts[QuantumStorage];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = low_order_derivative_state(axis);
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        ++quantum_count;
      }
    }

    for (unsigned subset = 0; subset < (1U << PairOrder); ++subset) {
      unsigned derivative_state = 0;
      Scalar coefficient = scalar<Scalar>(1.0);
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if ((subset & (1U << quantum)) != 0) {
          derivative_state += derivative_states[quantum];
          coefficient = inverse_two_exponent * coefficient;
        } else {
          coefficient = coefficient * shifts[quantum];
        }
      }
      expansion.terms[subset] = {derivative_state, coefficient};
    }

    if constexpr (PairOrder == 2) {
      if (derivative_states[0] == derivative_states[1]) {
        expansion.terms[0].coefficient =
            expansion.terms[0].coefficient + scalar<Scalar>(inverse_two_exponent);
      }
    } else if constexpr (PairOrder == 3) {
      for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 3; ++second_quantum) {
          if (derivative_states[first_quantum] != derivative_states[second_quantum]) {
            continue;
          }
          const unsigned remaining_quantum = 3U - first_quantum - second_quantum;
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient + inverse_two_exponent * shifts[remaining_quantum];
          const unsigned surviving_derivative = 1U << remaining_quantum;
          expansion.terms[surviving_derivative].coefficient =
              expansion.terms[surviving_derivative].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most three. */
template <typename Scalar>
__device__ Scalar third_order_coulomb(unsigned derivative_state, double rho,
                                      const Vec3<Scalar>& product_difference, const Scalar* boys) {
  const unsigned x_order = derivative_state & 3U;
  const unsigned y_order = (derivative_state >> 2U) & 3U;
  const unsigned z_order = (derivative_state >> 4U) & 3U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 3) {
    return low_order_coulomb(derivative_state, rho, product_difference, boys);
  }

  const double third_order_factor = -8.0 * rho * rho * rho;
  if (x_order == 3 || y_order == 3 || z_order == 3) {
    const Scalar coordinate = x_order == 3
                                  ? product_difference.x
                                  : (y_order == 3 ? product_difference.y : product_difference.z);
    return third_order_factor * coordinate * coordinate * coordinate * boys[3] +
           (12.0 * rho * rho) * coordinate * boys[2];
  }

  if (x_order == 2 || y_order == 2 || z_order == 2) {
    const Scalar repeated_coordinate =
        x_order == 2 ? product_difference.x
                     : (y_order == 2 ? product_difference.y : product_difference.z);
    const Scalar single_coordinate =
        x_order == 1 ? product_difference.x
                     : (y_order == 1 ? product_difference.y : product_difference.z);
    return third_order_factor * repeated_coordinate * repeated_coordinate * single_coordinate *
               boys[3] +
           (4.0 * rho * rho) * single_coordinate * boys[2];
  }

  return third_order_factor * product_difference.x * product_difference.y * product_difference.z *
         boys[3];
}

/** Value and P-Q chain derivatives of a weighted order-two Hermite DAG. */
struct WeightedOrder2Coulomb {
  double c0;
  double cx;
  double cy;
  double cz;
  double value;
  double chain[3];
};

/**
 * Contract the ten order-zero-through-two Hermite coefficients with exactly
 * twenty unique Cartesian Coulomb states. The returned chain is only the
 * derivative through P-Q; shell-class helpers add their explicit PA/PB/QC
 * coefficient derivatives and Gaussian-pair decay separately.
 */
__device__ __forceinline__ WeightedOrder2Coulomb contract_weighted_order2_coulomb(
    double rho, double x, double y, double z, const double (&boys)[4], double h0, double hx,
    double hy, double hz, double hxx, double hxy, double hxz, double hyy, double hyz, double hzz) {
  const double twice_rho = 2.0 * rho;
  const double twice_rho_squared = twice_rho * twice_rho;
  const double twice_rho_cubed = twice_rho_squared * twice_rho;
  const double c0 = boys[0];
  const double cx = -twice_rho * x * boys[1];
  const double cy = -twice_rho * y * boys[1];
  const double cz = -twice_rho * z * boys[1];
  const double cxx = twice_rho_squared * x * x * boys[2] - twice_rho * boys[1];
  const double cxy = twice_rho_squared * x * y * boys[2];
  const double cxz = twice_rho_squared * x * z * boys[2];
  const double cyy = twice_rho_squared * y * y * boys[2] - twice_rho * boys[1];
  const double cyz = twice_rho_squared * y * z * boys[2];
  const double czz = twice_rho_squared * z * z * boys[2] - twice_rho * boys[1];
  const double cxxx =
      -twice_rho_cubed * x * x * x * boys[3] + 3.0 * twice_rho_squared * x * boys[2];
  const double cxxy = -twice_rho_cubed * x * x * y * boys[3] + twice_rho_squared * y * boys[2];
  const double cxxz = -twice_rho_cubed * x * x * z * boys[3] + twice_rho_squared * z * boys[2];
  const double cxyy = -twice_rho_cubed * x * y * y * boys[3] + twice_rho_squared * x * boys[2];
  const double cxyz = -twice_rho_cubed * x * y * z * boys[3];
  const double cxzz = -twice_rho_cubed * x * z * z * boys[3] + twice_rho_squared * x * boys[2];
  const double cyyy =
      -twice_rho_cubed * y * y * y * boys[3] + 3.0 * twice_rho_squared * y * boys[2];
  const double cyyz = -twice_rho_cubed * y * y * z * boys[3] + twice_rho_squared * z * boys[2];
  const double cyzz = -twice_rho_cubed * y * z * z * boys[3] + twice_rho_squared * y * boys[2];
  const double czzz =
      -twice_rho_cubed * z * z * z * boys[3] + 3.0 * twice_rho_squared * z * boys[2];

  WeightedOrder2Coulomb result{};
  result.c0 = c0;
  result.cx = cx;
  result.cy = cy;
  result.cz = cz;
  result.value = h0 * c0 + hx * cx + hy * cy + hz * cz + hxx * cxx + hxy * cxy + hxz * cxz +
                 hyy * cyy + hyz * cyz + hzz * czz;
  result.chain[0] = h0 * cx + hx * cxx + hy * cxy + hz * cxz + hxx * cxxx + hxy * cxxy +
                    hxz * cxxz + hyy * cxyy + hyz * cxyz + hzz * cxzz;
  result.chain[1] = h0 * cy + hx * cxy + hy * cyy + hz * cyz + hxx * cxxy + hxy * cxyy +
                    hxz * cxyz + hyy * cyyy + hyz * cyyz + hzz * cyzz;
  result.chain[2] = h0 * cz + hx * cxz + hy * cyz + hz * czz + hxx * cxxz + hxy * cxyz +
                    hxz * cxzz + hyy * cyyz + hyz * cyzz + hzz * czzz;
  return result;
}

/**
 * Closed order-3 contraction for canonical (f s|s s), (d p|s s),
 * (d s|p s), and (p p|p s) primitive quartets.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ Scalar primitive_eri_order3(double alpha, const Vec3<Scalar>& first,
                                       const Angular& angular_first, double beta,
                                       const Vec3<Scalar>& second, const Angular& angular_second,
                                       double gamma, const Vec3<Scalar>& third,
                                       const Angular& angular_third, double delta,
                                       const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder = FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder = ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 3);
  static_assert(FirstPairOrder <= 3 && SecondPairOrder <= 3);
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const ThirdOrderPairExpansion<FirstPairOrder, Scalar> first_expansion =
      make_third_order_pair_expansion<FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const ThirdOrderPairExpansion<SecondPairOrder, Scalar> second_expansion =
      make_third_order_pair_expansion<ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[4];
  boys_values<3>(rho * distance_squared(product_p, product_q), boys);
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
      value = value +
              sign * first_item.coefficient * second_item.coefficient *
                  third_order_coulomb(first_item.derivative_state + second_item.derivative_state,
                                      rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

/** One sparse Hermite coefficient with three bits per Cartesian derivative. */
template <typename Scalar>
struct FourthOrderHermiteTerm {
  // Order four needs values 0--4 on one axis, so the two-bit encoding used by
  // lower orders is deliberately widened only for this specialization.
  unsigned derivative_state;
  Scalar coefficient;
};

/** Exact-sized sparse shell-pair expansion through total order four. */
template <unsigned PairOrder, typename Scalar>
struct FourthOrderPairExpansion {
  static_assert(PairOrder <= 4);
  FourthOrderHermiteTerm<Scalar> terms[1U << PairOrder];
};

__device__ unsigned fourth_order_derivative_state(int axis) { return 1U << (3 * axis); }

__device__ unsigned fourth_order_derivative_total(unsigned state) {
  return (state & 7U) + ((state >> 3U) & 7U) + ((state >> 6U) & 7U);
}

/**
 * Generate the exact Wick expansion of one shell pair through order four.
 *
 * Base subset terms represent uncontracted angular quanta. Every same-axis
 * pair adds one 1/(2p) contraction times the uncontracted remaining factors;
 * order four additionally admits the three possible disjoint pairings. All
 * contributions merge into the existing 2^N subset slots, so no generic
 * recurrence workspace is required.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, typename Scalar>
__device__ FourthOrderPairExpansion<FirstShellAngular + SecondShellAngular, Scalar>
make_fourth_order_pair_expansion(double exponent, const Vec3<Scalar>& product,
                                 const Vec3<Scalar>& first, const Angular& angular_first,
                                 const Vec3<Scalar>& second, const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 4);
  constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  FourthOrderPairExpansion<PairOrder, Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[QuantumStorage];
    Scalar shifts[QuantumStorage];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = fourth_order_derivative_state(axis);
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        ++quantum_count;
      }
    }

    for (unsigned subset = 0; subset < (1U << PairOrder); ++subset) {
      unsigned derivative_state = 0;
      Scalar coefficient = scalar<Scalar>(1.0);
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if ((subset & (1U << quantum)) != 0) {
          derivative_state += derivative_states[quantum];
          coefficient = inverse_two_exponent * coefficient;
        } else {
          coefficient = coefficient * shifts[quantum];
        }
      }
      expansion.terms[subset] = {derivative_state, coefficient};
    }

    if constexpr (PairOrder == 2) {
      if (derivative_states[0] == derivative_states[1]) {
        expansion.terms[0].coefficient =
            expansion.terms[0].coefficient + scalar<Scalar>(inverse_two_exponent);
      }
    } else if constexpr (PairOrder == 3) {
      for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 3; ++second_quantum) {
          if (derivative_states[first_quantum] != derivative_states[second_quantum]) {
            continue;
          }
          const unsigned remaining_quantum = 3U - first_quantum - second_quantum;
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient + inverse_two_exponent * shifts[remaining_quantum];
          const unsigned surviving_derivative = 1U << remaining_quantum;
          expansion.terms[surviving_derivative].coefficient =
              expansion.terms[surviving_derivative].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    } else if constexpr (PairOrder == 4) {
      for (unsigned first_quantum = 0; first_quantum < 4; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 4; ++second_quantum) {
          if (derivative_states[first_quantum] != derivative_states[second_quantum]) {
            continue;
          }
          unsigned remaining[2];
          unsigned remaining_count = 0;
          for (unsigned quantum = 0; quantum < 4; ++quantum) {
            if (quantum != first_quantum && quantum != second_quantum) {
              remaining[remaining_count++] = quantum;
            }
          }
          const unsigned first_remaining = remaining[0];
          const unsigned second_remaining = remaining[1];
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              inverse_two_exponent * shifts[first_remaining] * shifts[second_remaining];
          expansion.terms[1U << first_remaining].coefficient =
              expansion.terms[1U << first_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent) *
                  shifts[second_remaining];
          expansion.terms[1U << second_remaining].coefficient =
              expansion.terms[1U << second_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent) * shifts[first_remaining];
          const unsigned both_remaining = (1U << first_remaining) | (1U << second_remaining);
          expansion.terms[both_remaining].coefficient =
              expansion.terms[both_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent * inverse_two_exponent);
        }
      }

      constexpr unsigned Pairings[3][4] = {
          {0, 1, 2, 3},
          {0, 2, 1, 3},
          {0, 3, 1, 2},
      };
      for (unsigned pairing = 0; pairing < 3; ++pairing) {
        if (derivative_states[Pairings[pairing][0]] == derivative_states[Pairings[pairing][1]] &&
            derivative_states[Pairings[pairing][2]] == derivative_states[Pairings[pairing][3]]) {
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most four. */
template <typename Scalar>
__device__ Scalar fourth_order_coulomb(unsigned derivative_state, double rho,
                                       const Vec3<Scalar>& product_difference, const Scalar* boys) {
  const unsigned x_order = derivative_state & 7U;
  const unsigned y_order = (derivative_state >> 3U) & 7U;
  const unsigned z_order = (derivative_state >> 6U) & 7U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 4) {
    const unsigned lower_order_state = x_order | (y_order << 2U) | (z_order << 4U);
    return third_order_coulomb(lower_order_state, rho, product_difference, boys);
  }

  const double fourth_order_factor = 16.0 * rho * rho * rho * rho;
  const double third_order_factor = -8.0 * rho * rho * rho;
  const double second_order_factor = 4.0 * rho * rho;
  if (x_order == 4 || y_order == 4 || z_order == 4) {
    const Scalar coordinate = x_order == 4
                                  ? product_difference.x
                                  : (y_order == 4 ? product_difference.y : product_difference.z);
    const Scalar coordinate_squared = coordinate * coordinate;
    return fourth_order_factor * coordinate_squared * coordinate_squared * boys[4] +
           (6.0 * third_order_factor) * coordinate_squared * boys[3] +
           (3.0 * second_order_factor) * boys[2];
  }

  if (x_order == 3 || y_order == 3 || z_order == 3) {
    const Scalar repeated_coordinate =
        x_order == 3 ? product_difference.x
                     : (y_order == 3 ? product_difference.y : product_difference.z);
    const Scalar single_coordinate =
        x_order == 1 ? product_difference.x
                     : (y_order == 1 ? product_difference.y : product_difference.z);
    return fourth_order_factor * repeated_coordinate * repeated_coordinate * repeated_coordinate *
               single_coordinate * boys[4] +
           (3.0 * third_order_factor) * repeated_coordinate * single_coordinate * boys[3];
  }

  if ((x_order == 2 && y_order == 2) || (x_order == 2 && z_order == 2) ||
      (y_order == 2 && z_order == 2)) {
    Scalar first_coordinate = product_difference.x;
    Scalar second_coordinate = product_difference.y;
    if (x_order == 0) {
      first_coordinate = product_difference.y;
      second_coordinate = product_difference.z;
    } else if (y_order == 0) {
      second_coordinate = product_difference.z;
    }
    const Scalar first_squared = first_coordinate * first_coordinate;
    const Scalar second_squared = second_coordinate * second_coordinate;
    return fourth_order_factor * first_squared * second_squared * boys[4] +
           third_order_factor * (first_squared + second_squared) * boys[3] +
           second_order_factor * boys[2];
  }

  const Scalar repeated_coordinate =
      x_order == 2 ? product_difference.x
                   : (y_order == 2 ? product_difference.y : product_difference.z);
  Scalar single_product = scalar<Scalar>(1.0);
  if (x_order == 1) single_product = single_product * product_difference.x;
  if (y_order == 1) single_product = single_product * product_difference.y;
  if (z_order == 1) single_product = single_product * product_difference.z;
  return fourth_order_factor * repeated_coordinate * repeated_coordinate * single_product *
             boys[4] +
         third_order_factor * single_product * boys[3];
}

/**
 * Closed order-4 contraction for canonical (f p|s s), (d d|s s),
 * (f s|p s), (d p|p s), (d s|d s), (d s|p p), and (p p|p p) quartets.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ Scalar primitive_eri_order4(double alpha, const Vec3<Scalar>& first,
                                       const Angular& angular_first, double beta,
                                       const Vec3<Scalar>& second, const Angular& angular_second,
                                       double gamma, const Vec3<Scalar>& third,
                                       const Angular& angular_third, double delta,
                                       const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder = FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder = ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 4);
  static_assert(FirstPairOrder <= 4 && SecondPairOrder <= 4);
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const FourthOrderPairExpansion<FirstPairOrder, Scalar> first_expansion =
      make_fourth_order_pair_expansion<FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const FourthOrderPairExpansion<SecondPairOrder, Scalar> second_expansion =
      make_fourth_order_pair_expansion<ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[5];
  boys_values<4>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
      const FourthOrderHermiteTerm<Scalar>& first_item = first_expansion.terms[first_term];
      const FourthOrderHermiteTerm<Scalar>& second_item = second_expansion.terms[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      value = value +
              sign * first_item.coefficient * second_item.coefficient *
                  fourth_order_coulomb(first_item.derivative_state + second_item.derivative_state,
                                       rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

/**
 * Evaluate one Cartesian primitive quartet with exact shell-pair workspaces.
 *
 * Axis powers remain AO-component data, while the enclosing shell angular
 * momenta bound every Hermite dimension at compile time. This avoids charging
 * an s/p/d task for the generic f/f pair workspace.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ Scalar primitive_eri_cartesian_shell_class(
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

template <unsigned MaximumAngular, typename Scalar>
__device__ __noinline__ Scalar contracted_eri_cartesian(const DeviceBatch& batch, std::int64_t ao_i,
                                                        std::int64_t ao_j, std::int64_t ao_k,
                                                        std::int64_t ao_l, std::int32_t shell_i,
                                                        std::int32_t shell_j, std::int32_t shell_k,
                                                        std::int32_t shell_l,
                                                        std::int64_t derivative_coordinate) {
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
__device__ Scalar contracted_eri_order(const DeviceBatch& batch, std::int32_t system,
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
__device__ Scalar contracted_eri(const DeviceBatch& batch, std::int32_t system, std::int32_t i,
                                 std::int32_t j, std::int32_t k, std::int32_t l,
                                 std::int64_t derivative_coordinate) {
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
__device__ Scalar contracted_eri_cartesian_source_shell_class(
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

/** Three contracted Cartesian components of canonical (p s | s s). */
struct PsssIntegralVector {
  double axis[3];
};

/** Density-weighted psss derivatives for the first three canonical centers. */
struct PsssWeightedGradient {
  double center[3][3];
};

/**
 * Contract all psss Cartesian outputs with one shared primitive traversal.
 *
 * The p_x, p_y, and p_z values differ only in the final PA/PQ component of
 * the closed order-one expression. Keeping the complete shell task in one
 * lane removes threefold repetition of product centers, pair decay, and Boys
 * values while retaining the established CCA x/y/z component order.
 */
__device__ __noinline__ PsssIntegralVector contracted_eri_cartesian_source_psss(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t p_shell, std::int32_t paired_s_shell, std::int32_t third_shell,
    std::int32_t fourth_shell) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[p_shell], -1);

  const std::int64_t p_ao_begin = batch.shell_direct_ao_offsets[p_shell];
  const double s_angular_coefficient =
      batch.direct_ao_coefficients[batch.shell_direct_ao_offsets[paired_s_shell]] *
      batch.direct_ao_coefficients[batch.shell_direct_ao_offsets[third_shell]] *
      batch.direct_ao_coefficients[batch.shell_direct_ao_offsets[fourth_shell]];
  const double angular_coefficient[3] = {
      s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin],
      s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin + 1],
      s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin + 2],
  };

  PsssIntegralVector result{};
  const std::int64_t first_pair_begin = batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const Vec3<double> product_p = first_pair.product_center;
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      double boys[2];
      boys_values<1>(rho * distance_squared(product_p, product_q), boys);
      const double prefactor = first_pair.weighted_coefficient * second_pair.weighted_coefficient *
                               2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));
      const double coulomb_scale = rho / p;
      result.axis[0] += angular_coefficient[0] * prefactor *
                        ((product_p.x - first.x) * boys[0] -
                         coulomb_scale * (product_p.x - product_q.x) * boys[1]);
      result.axis[1] += angular_coefficient[1] * prefactor *
                        ((product_p.y - first.y) * boys[0] -
                         coulomb_scale * (product_p.y - product_q.y) * boys[1]);
      result.axis[2] += angular_coefficient[2] * prefactor *
                        ((product_p.z - first.z) * boys[0] -
                         coulomb_scale * (product_p.z - product_q.z) * boys[1]);
    }
  }
  return result;
}

/**
 * Contract all three psss component gradients in one primitive traversal.
 *
 * `density_coefficient` already contains the exact eightfold RHF/UHF density
 * contraction for each p axis. Linearity lets the three component gradients
 * be combined before the primitive loops: PA, P-Q, and their coordinate
 * derivatives become short weighted dot products, while product centers,
 * decay, and Boys values are evaluated only once. The fourth-center gradient
 * is intentionally omitted and restored from translation by the force task.
 */
template <bool ResidentBra>
__device__ __noinline__ PsssWeightedGradient contracted_eri_cartesian_source_psss_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t p_shell, std::int32_t paired_s_shell, std::int32_t third_shell,
    std::int32_t fourth_shell, const double (&density_coefficient)[3],
    const PrimitivePairData* resident_first_pairs, std::int64_t resident_first_pair_count) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[p_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[paired_s_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[third_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1);

  const std::int64_t p_ao_begin = batch.shell_direct_ao_offsets[p_shell];
  const double s_angular_coefficient =
      batch.direct_ao_coefficients[batch.shell_direct_ao_offsets[paired_s_shell]] *
      batch.direct_ao_coefficients[batch.shell_direct_ao_offsets[third_shell]] *
      batch.direct_ao_coefficients[batch.shell_direct_ao_offsets[fourth_shell]];
  const double axis_weight[3] = {
      density_coefficient[0] * s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin],
      density_coefficient[1] * s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin + 1],
      density_coefficient[2] * s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin + 2],
  };

  PsssWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == p_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_shell;
  const std::int64_t first_pair_begin =
      ResidentBra ? 0 : batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end =
      ResidentBra ? resident_first_pair_count
                  : batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = ResidentBra ? resident_first_pairs[first_primitive]
                                                     : batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const Vec3<double> product_p = first_pair.product_center;
    const double first_product_scale = first_pair_matches_canonical_order
                                           ? first_pair.first_product_scale
                                           : first_pair.second_product_scale;
    const double second_product_scale = first_pair_matches_canonical_order
                                            ? first_pair.second_product_scale
                                            : first_pair.first_product_scale;
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const Vec3<double> product_difference{
          product_p.x - product_q.x,
          product_p.y - product_q.y,
          product_p.z - product_q.z,
      };
      const Vec3<double> pa{
          product_p.x - first.x,
          product_p.y - first.y,
          product_p.z - first.z,
      };
      double boys[3];
      boys_values<2>(rho * distance_squared(product_p, product_q), boys);
      const double prefactor = first_pair.weighted_coefficient * second_pair.weighted_coefficient *
                               2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));
      if (batch.generated_psss_weighted) {
        // Retain resident-bra reuse, primitive orientation, normalization, and
        // one traversal across all p outputs. Only the scalar weighted
        // expression comes from the generic external-weight DAG lowering.
        generated_weighted_eri::Geometry geometry{};
        geometry.inverse_two_p = 0.5 / p;
        geometry.rho = rho;
        geometry.prefactor = prefactor;
        geometry.product_scales[0] = first_product_scale;
        geometry.product_scales[1] = second_product_scale;
        geometry.product_scales[2] = second_pair_matches_canonical_order
                                         ? second_pair.first_product_scale
                                         : second_pair.second_product_scale;
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          geometry.shifts[0][coordinate] = vec_axis(pa, coordinate);
          geometry.difference[coordinate] = vec_axis(product_difference, coordinate);
          geometry.boys[coordinate] = boys[coordinate];
          geometry.decay[0][coordinate] =
              -2.0 * mu * (vec_axis(first, coordinate) - vec_axis(second, coordinate));
          geometry.decay[1][coordinate] = -geometry.decay[0][coordinate];
          geometry.decay[2][coordinate] =
              -2.0 * nu * (vec_axis(third, coordinate) - vec_axis(fourth, coordinate));
        }
        const auto generated = generated_weighted_eri::psss(geometry, axis_weight);
#pragma unroll
        for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
          for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
            result.center[center][coordinate] += generated.center[center][coordinate];
          }
        }
        continue;
      }
      const double coulomb_scale = rho / p;
      const double weighted_pa =
          axis_weight[0] * pa.x + axis_weight[1] * pa.y + axis_weight[2] * pa.z;
      const double weighted_pq = axis_weight[0] * product_difference.x +
                                 axis_weight[1] * product_difference.y +
                                 axis_weight[2] * product_difference.z;
      const double weighted_value = weighted_pa * boys[0] - coulomb_scale * weighted_pq * boys[1];
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;
      const double product_scales[3] = {first_product_scale, second_product_scale,
                                        -third_product_scale};

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          double decay_derivative = 0.0;
          if (center < 2) {
            const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
            decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
          } else {
            const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
            decay_derivative = -2.0 * nu * difference;
          }
          const double argument_derivative =
              2.0 * rho * product_scales[center] * vec_axis(product_difference, coordinate);
          const double pa_derivative =
              center == 0 ? first_product_scale - 1.0 : (center == 1 ? second_product_scale : 0.0);
          const double weighted_value_derivative =
              axis_weight[coordinate] * pa_derivative * boys[0] -
              weighted_pa * boys[1] * argument_derivative -
              coulomb_scale * axis_weight[coordinate] * product_scales[center] * boys[1] +
              coulomb_scale * weighted_pq * boys[2] * argument_derivative;
          result.center[center][coordinate] +=
              prefactor * (weighted_value_derivative + weighted_value * decay_derivative);
        }
      }
    }
  }
  return result;
}

/** Density-weighted psps derivatives for the first three canonical centers. */
struct PspsWeightedGradient {
  double center[3][3];
};

/**
 * Contract all nine canonical (p s | p s) components through one closed DAG.
 *
 * `component_weight` contains the screened density contraction and all four
 * Cartesian AO normalizations for the exact decoded AO-quartet domain. In
 * particular, missing entries from an identical-shell-pair triangular domain
 * remain zero; mirroring them would double-count the established ERI
 * multiplicity. The recurrence below forms the ten unique Coulomb states
 * through order two and their ten order-three raises exactly once per
 * primitive quartet, then contracts value and three coordinate derivatives.
 */
__device__ __noinline__ PspsWeightedGradient contracted_eri_cartesian_source_psps_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t first_p_shell, std::int32_t first_s_shell, std::int32_t second_p_shell,
    std::int32_t second_s_shell, const double (&component_weight)[9]) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[first_p_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[first_s_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[second_p_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[second_s_shell], -1);

  PspsWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == first_p_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == second_p_shell;
  const std::int64_t first_pair_begin = batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const double inverse_two_p = 0.5 / p;
    const Vec3<double> product_p = first_pair.product_center;
    const Vec3<double> pa{
        product_p.x - first.x,
        product_p.y - first.y,
        product_p.z - first.z,
    };
    const double first_product_scale = first_pair_matches_canonical_order
                                           ? first_pair.first_product_scale
                                           : first_pair.second_product_scale;
    const double second_product_scale = first_pair_matches_canonical_order
                                            ? first_pair.second_product_scale
                                            : first_pair.first_product_scale;
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const double inverse_two_q = 0.5 / q;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const Vec3<double> qc{
          product_q.x - third.x,
          product_q.y - third.y,
          product_q.z - third.z,
      };
      const double x = product_p.x - product_q.x;
      const double y = product_p.y - product_q.y;
      const double z = product_p.z - product_q.z;
      double boys[4];
      boys_values<3>(rho * (x * x + y * y + z * z), boys);

      const double row_x =
          component_weight[0] * qc.x + component_weight[1] * qc.y + component_weight[2] * qc.z;
      const double row_y =
          component_weight[3] * qc.x + component_weight[4] * qc.y + component_weight[5] * qc.z;
      const double row_z =
          component_weight[6] * qc.x + component_weight[7] * qc.y + component_weight[8] * qc.z;
      const double column_x =
          component_weight[0] * pa.x + component_weight[3] * pa.y + component_weight[6] * pa.z;
      const double column_y =
          component_weight[1] * pa.x + component_weight[4] * pa.y + component_weight[7] * pa.z;
      const double column_z =
          component_weight[2] * pa.x + component_weight[5] * pa.y + component_weight[8] * pa.z;
      const double h0 = pa.x * row_x + pa.y * row_y + pa.z * row_z;
      const double hx = inverse_two_p * row_x - inverse_two_q * column_x;
      const double hy = inverse_two_p * row_y - inverse_two_q * column_y;
      const double hz = inverse_two_p * row_z - inverse_two_q * column_z;
      const double second_scale = -inverse_two_p * inverse_two_q;
      const double hxx = second_scale * component_weight[0];
      const double hxy = second_scale * (component_weight[1] + component_weight[3]);
      const double hxz = second_scale * (component_weight[2] + component_weight[6]);
      const double hyy = second_scale * component_weight[4];
      const double hyz = second_scale * (component_weight[5] + component_weight[7]);
      const double hzz = second_scale * component_weight[8];

      const WeightedOrder2Coulomb coulomb = contract_weighted_order2_coulomb(
          rho, x, y, z, boys, h0, hx, hy, hz, hxx, hxy, hxz, hyy, hyz, hzz);

      const double first_explicit[3] = {
          row_x * coulomb.c0 -
              inverse_two_q * (component_weight[0] * coulomb.cx + component_weight[1] * coulomb.cy +
                               component_weight[2] * coulomb.cz),
          row_y * coulomb.c0 -
              inverse_two_q * (component_weight[3] * coulomb.cx + component_weight[4] * coulomb.cy +
                               component_weight[5] * coulomb.cz),
          row_z * coulomb.c0 -
              inverse_two_q * (component_weight[6] * coulomb.cx + component_weight[7] * coulomb.cy +
                               component_weight[8] * coulomb.cz),
      };
      const double second_explicit[3] = {
          column_x * coulomb.c0 +
              inverse_two_p * (component_weight[0] * coulomb.cx + component_weight[3] * coulomb.cy +
                               component_weight[6] * coulomb.cz),
          column_y * coulomb.c0 +
              inverse_two_p * (component_weight[1] * coulomb.cx + component_weight[4] * coulomb.cy +
                               component_weight[7] * coulomb.cz),
          column_z * coulomb.c0 +
              inverse_two_p * (component_weight[2] * coulomb.cx + component_weight[5] * coulomb.cy +
                               component_weight[8] * coulomb.cz),
      };
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;
      const double product_scale[3] = {first_product_scale, second_product_scale,
                                       -third_product_scale};
      const double shift_scale[3] = {first_product_scale - 1.0, second_product_scale,
                                     third_product_scale - 1.0};
      const double prefactor = first_pair.weighted_coefficient * second_pair.weighted_coefficient *
                               2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          const double pair_coefficient_derivative =
              shift_scale[center] *
              (center < 2 ? first_explicit[coordinate] : second_explicit[coordinate]);
          double decay_derivative = 0.0;
          if (center < 2) {
            const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
            decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
          } else {
            const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
            decay_derivative = -2.0 * nu * difference;
          }
          result.center[center][coordinate] +=
              prefactor *
              (pair_coefficient_derivative + product_scale[center] * coulomb.chain[coordinate] +
               coulomb.value * decay_derivative);
        }
      }
    }
  }
  return result;
}

/**
 * Contract canonical (p p | s s) through the shared order-two Coulomb DAG.
 */
__device__ __noinline__ PspsWeightedGradient contracted_eri_cartesian_source_ppss_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t first_p_shell, std::int32_t second_p_shell, std::int32_t third_s_shell,
    std::int32_t fourth_s_shell, const double (&component_weight)[9]) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[first_p_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[second_p_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[third_s_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[fourth_s_shell], -1);

  PspsWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == first_p_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_s_shell;
  const std::int64_t first_pair_begin = batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const double inverse_two_p = 0.5 / p;
    const Vec3<double> product_p = first_pair.product_center;
    const Vec3<double> pa{
        product_p.x - first.x,
        product_p.y - first.y,
        product_p.z - first.z,
    };
    const Vec3<double> pb{
        product_p.x - second.x,
        product_p.y - second.y,
        product_p.z - second.z,
    };
    const double first_product_scale = first_pair_matches_canonical_order
                                           ? first_pair.first_product_scale
                                           : first_pair.second_product_scale;
    const double second_product_scale = first_pair_matches_canonical_order
                                            ? first_pair.second_product_scale
                                            : first_pair.first_product_scale;
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const double x = product_p.x - product_q.x;
      const double y = product_p.y - product_q.y;
      const double z = product_p.z - product_q.z;
      double boys[4];
      boys_values<3>(rho * (x * x + y * y + z * z), boys);

      const double row_x =
          component_weight[0] * pb.x + component_weight[1] * pb.y + component_weight[2] * pb.z;
      const double row_y =
          component_weight[3] * pb.x + component_weight[4] * pb.y + component_weight[5] * pb.z;
      const double row_z =
          component_weight[6] * pb.x + component_weight[7] * pb.y + component_weight[8] * pb.z;
      const double column_x =
          component_weight[0] * pa.x + component_weight[3] * pa.y + component_weight[6] * pa.z;
      const double column_y =
          component_weight[1] * pa.x + component_weight[4] * pa.y + component_weight[7] * pa.z;
      const double column_z =
          component_weight[2] * pa.x + component_weight[5] * pa.y + component_weight[8] * pa.z;
      const double h0 =
          pa.x * row_x + pa.y * row_y + pa.z * row_z +
          inverse_two_p * (component_weight[0] + component_weight[4] + component_weight[8]);
      const double hx = inverse_two_p * (row_x + column_x);
      const double hy = inverse_two_p * (row_y + column_y);
      const double hz = inverse_two_p * (row_z + column_z);
      const double second_scale = inverse_two_p * inverse_two_p;
      const double hxx = second_scale * component_weight[0];
      const double hxy = second_scale * (component_weight[1] + component_weight[3]);
      const double hxz = second_scale * (component_weight[2] + component_weight[6]);
      const double hyy = second_scale * component_weight[4];
      const double hyz = second_scale * (component_weight[5] + component_weight[7]);
      const double hzz = second_scale * component_weight[8];
      const WeightedOrder2Coulomb coulomb = contract_weighted_order2_coulomb(
          rho, x, y, z, boys, h0, hx, hy, hz, hxx, hxy, hxz, hyy, hyz, hzz);

      const double first_shift_scale = first_product_scale - 1.0;
      const double second_shift_scale = first_product_scale;
      const double explicit_first[3] = {
          (first_shift_scale * row_x + second_shift_scale * column_x) * coulomb.c0 +
              inverse_two_p * (first_shift_scale * (component_weight[0] * coulomb.cx +
                                                    component_weight[1] * coulomb.cy +
                                                    component_weight[2] * coulomb.cz) +
                               second_shift_scale * (component_weight[0] * coulomb.cx +
                                                     component_weight[3] * coulomb.cy +
                                                     component_weight[6] * coulomb.cz)),
          (first_shift_scale * row_y + second_shift_scale * column_y) * coulomb.c0 +
              inverse_two_p * (first_shift_scale * (component_weight[3] * coulomb.cx +
                                                    component_weight[4] * coulomb.cy +
                                                    component_weight[5] * coulomb.cz) +
                               second_shift_scale * (component_weight[1] * coulomb.cx +
                                                     component_weight[4] * coulomb.cy +
                                                     component_weight[7] * coulomb.cz)),
          (first_shift_scale * row_z + second_shift_scale * column_z) * coulomb.c0 +
              inverse_two_p * (first_shift_scale * (component_weight[6] * coulomb.cx +
                                                    component_weight[7] * coulomb.cy +
                                                    component_weight[8] * coulomb.cz) +
                               second_shift_scale * (component_weight[2] * coulomb.cx +
                                                     component_weight[5] * coulomb.cy +
                                                     component_weight[8] * coulomb.cz)),
      };
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;
      const double product_scale[3] = {first_product_scale, second_product_scale,
                                       -third_product_scale};
      const double prefactor = first_pair.weighted_coefficient * second_pair.weighted_coefficient *
                               2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          const double pair_coefficient_derivative =
              center == 0 ? explicit_first[coordinate]
                          : (center == 1 ? -explicit_first[coordinate] : 0.0);
          double decay_derivative = 0.0;
          if (center < 2) {
            const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
            decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
          } else {
            const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
            decay_derivative = -2.0 * nu * difference;
          }
          result.center[center][coordinate] +=
              prefactor *
              (pair_coefficient_derivative + product_scale[center] * coulomb.chain[coordinate] +
               coulomb.value * decay_derivative);
        }
      }
    }
  }
  return result;
}

/**
 * Contract canonical (d s | s s) in CCA xx,xy,xz,yy,yz,zz order.
 */
__device__ __noinline__ PspsWeightedGradient contracted_eri_cartesian_source_dsss_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t d_shell, std::int32_t paired_s_shell, std::int32_t third_s_shell,
    std::int32_t fourth_s_shell, const double* component_weight) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[d_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[paired_s_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[third_s_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[fourth_s_shell], -1);

  PspsWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == d_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_s_shell;
  const std::int64_t first_pair_begin = batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const double inverse_two_p = 0.5 / p;
    const Vec3<double> product_p = first_pair.product_center;
    const Vec3<double> pa{
        product_p.x - first.x,
        product_p.y - first.y,
        product_p.z - first.z,
    };
    const double first_product_scale = first_pair_matches_canonical_order
                                           ? first_pair.first_product_scale
                                           : first_pair.second_product_scale;
    const double second_product_scale = first_pair_matches_canonical_order
                                            ? first_pair.second_product_scale
                                            : first_pair.first_product_scale;
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const double x = product_p.x - product_q.x;
      const double y = product_p.y - product_q.y;
      const double z = product_p.z - product_q.z;
      double boys[4];
      boys_values<3>(rho * (x * x + y * y + z * z), boys);

      const double gx = 2.0 * component_weight[0] * pa.x + component_weight[1] * pa.y +
                        component_weight[2] * pa.z;
      const double gy = component_weight[1] * pa.x + 2.0 * component_weight[3] * pa.y +
                        component_weight[4] * pa.z;
      const double gz = component_weight[2] * pa.x + component_weight[4] * pa.y +
                        2.0 * component_weight[5] * pa.z;
      const double h0 =
          component_weight[0] * (pa.x * pa.x + inverse_two_p) + component_weight[1] * pa.x * pa.y +
          component_weight[2] * pa.x * pa.z + component_weight[3] * (pa.y * pa.y + inverse_two_p) +
          component_weight[4] * pa.y * pa.z + component_weight[5] * (pa.z * pa.z + inverse_two_p);
      const double hx = inverse_two_p * gx;
      const double hy = inverse_two_p * gy;
      const double hz = inverse_two_p * gz;
      const double second_scale = inverse_two_p * inverse_two_p;
      const WeightedOrder2Coulomb coulomb = contract_weighted_order2_coulomb(
          rho, x, y, z, boys, h0, hx, hy, hz, second_scale * component_weight[0],
          second_scale * component_weight[1], second_scale * component_weight[2],
          second_scale * component_weight[3], second_scale * component_weight[4],
          second_scale * component_weight[5]);
      const double shift_scale = first_product_scale - 1.0;
      const double explicit_first[3] = {
          shift_scale * (gx * coulomb.c0 + inverse_two_p * (2.0 * component_weight[0] * coulomb.cx +
                                                            component_weight[1] * coulomb.cy +
                                                            component_weight[2] * coulomb.cz)),
          shift_scale * (gy * coulomb.c0 + inverse_two_p * (component_weight[1] * coulomb.cx +
                                                            2.0 * component_weight[3] * coulomb.cy +
                                                            component_weight[4] * coulomb.cz)),
          shift_scale *
              (gz * coulomb.c0 + inverse_two_p * (component_weight[2] * coulomb.cx +
                                                  component_weight[4] * coulomb.cy +
                                                  2.0 * component_weight[5] * coulomb.cz)),
      };
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;
      const double product_scale[3] = {first_product_scale, second_product_scale,
                                       -third_product_scale};
      const double prefactor = first_pair.weighted_coefficient * second_pair.weighted_coefficient *
                               2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          const double pair_coefficient_derivative =
              center == 0 ? explicit_first[coordinate]
                          : (center == 1 ? -explicit_first[coordinate] : 0.0);
          double decay_derivative = 0.0;
          if (center < 2) {
            const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
            decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
          } else {
            const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
            decay_derivative = -2.0 * nu * difference;
          }
          result.center[center][coordinate] +=
              prefactor *
              (pair_coefficient_derivative + product_scale[center] * coulomb.chain[coordinate] +
               coulomb.value * decay_derivative);
        }
      }
    }
  }
  return result;
}

/** Cartesian derivatives of one contracted quartet, indexed by input slot. */
struct CartesianQuartetGradient {
  double center[4][3];
};

/** Density-weighted ssss derivatives for the first three input centers. */
struct SsssWeightedGradient {
  double center[3][3];
};

/**
 * Contract an ssss shell quartet from the reusable primitive-pair cache.
 *
 * The generic order-zero path rebuilds both primitive pairs for every
 * primitive quartet, including product centers, Gaussian pair decay, and
 * coefficient products. Those quantities already live in PrimitivePairData
 * and are shared with direct Fock. Only the inter-pair Boys argument and the
 * derivative chain therefore remain here. The fourth center is omitted by
 * translational invariance and reconstructed by the force-task consumer.
 */
__device__ __forceinline__ SsssWeightedGradient
contracted_eri_cartesian_source_ssss_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t first_shell, std::int32_t second_shell, std::int32_t third_shell,
    std::int32_t fourth_shell, double component_weight) {
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[first_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[second_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[third_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1);
  const Vec3<double> first_difference{
      first.x - second.x,
      first.y - second.y,
      first.z - second.z,
  };
  const Vec3<double> second_difference{
      third.x - fourth.x,
      third.y - fourth.y,
      third.z - fourth.z,
  };

  SsssWeightedGradient result{};
  const std::int64_t first_pair_begin = batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double rho = p * q / (p + q);
      const Vec3<double> product_difference{
          first_pair.product_center.x - second_pair.product_center.x,
          first_pair.product_center.y - second_pair.product_center.y,
          first_pair.product_center.z - second_pair.product_center.z,
      };
      double boys[2];
      boys_values<1>(rho * distance_squared(first_pair.product_center, second_pair.product_center),
                     boys);
      const double prefactor = component_weight * first_pair.weighted_coefficient *
                               second_pair.weighted_coefficient * 2.0 * pow(kPi, 2.5) /
                               (p * q * sqrt(p + q));
      const double product_chain[3] = {
          -2.0 * rho * product_difference.x * boys[1],
          -2.0 * rho * product_difference.y * boys[1],
          -2.0 * rho * product_difference.z * boys[1],
      };
      const double first_decay[3] = {
          -2.0 * first_pair.reduced_exponent * first_difference.x * boys[0],
          -2.0 * first_pair.reduced_exponent * first_difference.y * boys[0],
          -2.0 * first_pair.reduced_exponent * first_difference.z * boys[0],
      };
      const double third_decay[3] = {
          -2.0 * second_pair.reduced_exponent * second_difference.x * boys[0],
          -2.0 * second_pair.reduced_exponent * second_difference.y * boys[0],
          -2.0 * second_pair.reduced_exponent * second_difference.z * boys[0],
      };

#pragma unroll
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        result.center[0][coordinate] +=
            prefactor *
            (first_pair.first_product_scale * product_chain[coordinate] + first_decay[coordinate]);
        result.center[1][coordinate] +=
            prefactor *
            (first_pair.second_product_scale * product_chain[coordinate] - first_decay[coordinate]);
        result.center[2][coordinate] +=
            prefactor * (-second_pair.first_product_scale * product_chain[coordinate] +
                         third_decay[coordinate]);
      }
    }
  }
  return result;
}

/**
 * Evaluate the independent center derivatives of an ssss or canonical psss
 * primitive.
 *
 * Coordinate differentiation changes only the Gaussian pair decay, product
 * centers, and Boys argument. Computing those shared values once is much
 * cheaper than replaying the complete primitive with one Dual3 seed per
 * independent atom. `p_axis` is ignored for the order-zero specialization.
 */
template <unsigned AngularOrder>
__device__ void primitive_eri_order01_gradient(int p_axis, double alpha, const Vec3<double>& first,
                                               double beta, const Vec3<double>& second,
                                               double gamma, const Vec3<double>& third,
                                               double delta, const Vec3<double>& fourth,
                                               double (&gradient)[4][3]) {
  static_assert(AngularOrder <= 1);
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<double> product_p = product_center(alpha, first, beta, second);
  const Vec3<double> product_q = product_center(gamma, third, delta, fourth);
  const Vec3<double> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };
  double boys[AngularOrder + 2];
  boys_values<AngularOrder + 1>(rho * distance_squared(product_p, product_q), boys);
  const double pair_decay =
      exp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  const double prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;

  // d(P-Q)/d(A,B,C,D); the same scalar applies independently to x/y/z.
  const double product_scales[4] = {alpha / p, beta / p, -gamma / q, -delta / q};
  double value = boys[0];
  double pa = 0.0;
  double pq_axis = 0.0;
  const double coulomb_scale = rho / p;
  if constexpr (AngularOrder == 1) {
    pa = vec_axis(product_p, p_axis) - vec_axis(first, p_axis);
    pq_axis = vec_axis(product_difference, p_axis);
    value = pa * boys[0] - coulomb_scale * pq_axis * boys[1];
  }

  // A simultaneous translation of all four Gaussian centers leaves the ERI
  // unchanged. Evaluate only three centers and recover the fourth exactly;
  // the force consumer already relies on this invariant across unique atoms.
  for (unsigned center = 0; center < 3; ++center) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double decay_derivative = 0.0;
      if (center < 2) {
        const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
        decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
      } else {
        const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
        decay_derivative = (center == 2 ? -2.0 * nu : 2.0 * nu) * difference;
      }
      const double argument_derivative =
          2.0 * rho * product_scales[center] * vec_axis(product_difference, coordinate);
      double value_derivative = -boys[1] * argument_derivative;
      if constexpr (AngularOrder == 1) {
        double pa_derivative = 0.0;
        if (coordinate == p_axis) {
          if (center == 0) {
            pa_derivative = alpha / p - 1.0;
          } else if (center == 1) {
            pa_derivative = beta / p;
          }
        }
        const double pq_derivative = coordinate == p_axis ? product_scales[center] : 0.0;
        value_derivative = pa_derivative * boys[0] - pa * boys[1] * argument_derivative -
                           coulomb_scale * pq_derivative * boys[1] +
                           coulomb_scale * pq_axis * boys[2] * argument_derivative;
      }
      gradient[center][coordinate] = prefactor * (value_derivative + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] - gradient[2][coordinate];
  }
}

/**
 * Contract explicit order-zero/one primitive gradients into input AO slots.
 *
 * The order-one integral is canonicalized to (p s|s s), while `original`
 * preserves the caller's slot-to-atom mapping for force accumulation.
 */
template <unsigned AngularOrder>
__device__ CartesianQuartetGradient contracted_eri_cartesian_source_order01_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  static_assert(AngularOrder <= 1);
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if constexpr (AngularOrder == 1) {
    unsigned p_slot = 0;
    for (unsigned slot = 0; slot < 4; ++slot) {
      if (batch.shell_angular[slots[slot].shell] == 1) p_slot = slot;
    }
    if (p_slot == 1) {
      const SourceSlot swap = slots[0];
      slots[0] = slots[1];
      slots[1] = swap;
    } else if (p_slot >= 2) {
      if (p_slot == 3) {
        const SourceSlot swap = slots[2];
        slots[2] = slots[3];
        slots[3] = swap;
      }
      const SourceSlot first_swap = slots[0];
      slots[0] = slots[2];
      slots[2] = first_swap;
      const SourceSlot second_swap = slots[1];
      slots[1] = slots[3];
      slots[3] = second_swap;
    }
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  int p_axis = 0;
  if constexpr (AngularOrder == 1) {
    const Angular angular = direct_ao_angular(batch, slots[0].ao);
    p_axis = angular.x == 1 ? 0 : (angular.y == 1 ? 1 : 2);
  }
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          primitive_eri_order01_gradient<AngularOrder>(
              p_axis, batch.primitive_exponents[a], positions[0], batch.primitive_exponents[b],
              positions[1], batch.primitive_exponents[c], positions[2],
              batch.primitive_exponents[d], positions[3], primitive_gradient);
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

/** One order-two shell-pair term and its first-center coefficient gradient. */
struct LowOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
  double first_center[3];
};

struct LowOrderPairGradientExpansion {
  LowOrderPairGradientTerm terms[4];
  unsigned count;
};

/**
 * Build the exact order-0/1/2 Hermite pair expansion and center gradients.
 *
 * Pair decay is handled once by the primitive quartet. The only
 * coordinate-dependent pair coefficients are products of P-A/P-B shifts;
 * 1/(2p) contraction corrections are coordinate independent.
 */
__device__ LowOrderPairGradientExpansion make_low_order_pair_gradient_expansion(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second) {
  const double exponent = alpha + beta;
  const Vec3<double> product = product_center(alpha, first, beta, second);
  const double inverse_two_exponent = 0.5 / exponent;
  unsigned derivative_states[2]{};
  double shifts[2]{};
  double first_center_shift_gradients[2][3]{};
  unsigned quantum_count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
      derivative_states[quantum_count] = low_order_derivative_state(axis);
      shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
      first_center_shift_gradients[quantum_count][axis] = alpha / exponent - 1.0;
      ++quantum_count;
    }
    for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
      derivative_states[quantum_count] = low_order_derivative_state(axis);
      shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
      first_center_shift_gradients[quantum_count][axis] = alpha / exponent;
      ++quantum_count;
    }
  }

  LowOrderPairGradientExpansion expansion{};
  if (quantum_count == 0) {
    expansion.count = 1;
    expansion.terms[0].coefficient = 1.0;
    return expansion;
  }
  if (quantum_count == 1) {
    expansion.count = 2;
    expansion.terms[0].coefficient = shifts[0];
    expansion.terms[1].derivative_state = derivative_states[0];
    expansion.terms[1].coefficient = inverse_two_exponent;
    for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
      expansion.terms[0].first_center[coordinate] = first_center_shift_gradients[0][coordinate];
    }
    return expansion;
  }

  expansion.count = 4;
  expansion.terms[0].coefficient =
      shifts[0] * shifts[1] +
      (derivative_states[0] == derivative_states[1] ? inverse_two_exponent : 0.0);
  expansion.terms[1].derivative_state = derivative_states[0];
  expansion.terms[1].coefficient = inverse_two_exponent * shifts[1];
  expansion.terms[2].derivative_state = derivative_states[1];
  expansion.terms[2].coefficient = inverse_two_exponent * shifts[0];
  expansion.terms[3].derivative_state = derivative_states[0] + derivative_states[1];
  expansion.terms[3].coefficient = inverse_two_exponent * inverse_two_exponent;
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    expansion.terms[0].first_center[coordinate] =
        first_center_shift_gradients[0][coordinate] * shifts[1] +
        shifts[0] * first_center_shift_gradients[1][coordinate];
    expansion.terms[1].first_center[coordinate] =
        inverse_two_exponent * first_center_shift_gradients[1][coordinate];
    expansion.terms[2].first_center[coordinate] =
        inverse_two_exponent * first_center_shift_gradients[0][coordinate];
  }
  return expansion;
}

/** Evaluate all center derivatives of one total-order-two primitive quartet. */
__device__ void primitive_eri_order2_gradient(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second, double gamma,
    const Vec3<double>& third, const Angular& angular_third, double delta,
    const Vec3<double>& fourth, const Angular& angular_fourth, double (&gradient)[4][3]) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<double> product_p = product_center(alpha, first, beta, second);
  const Vec3<double> product_q = product_center(gamma, third, delta, fourth);
  const Vec3<double> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };
  const LowOrderPairGradientExpansion first_expansion = make_low_order_pair_gradient_expansion(
      alpha, first, angular_first, beta, second, angular_second);
  const LowOrderPairGradientExpansion second_expansion = make_low_order_pair_gradient_expansion(
      gamma, third, angular_third, delta, fourth, angular_fourth);
  double boys[4];
  boys_values<3>(rho * distance_squared(product_p, product_q), boys);
  const double first_product_scale = alpha / p;
  const double second_product_scale = beta / p;
  const double third_product_scale = -gamma / q;
  double value = 0.0;
  // Pair coefficients are translation invariant within each pair, so the
  // second-center coefficient derivative is the negative of the first. The
  // fourth full-center derivative is restored below from total translation.
  // Keeping only three accumulators also avoids spilling another vector in
  // this order-two kernel, whose remaining gap is dominated by local state.
  double value_gradient[3][3]{};
  for (unsigned first_term = 0; first_term < first_expansion.count; ++first_term) {
    for (unsigned second_term = 0; second_term < second_expansion.count; ++second_term) {
      const LowOrderPairGradientTerm& first_item = first_expansion.terms[first_term];
      const LowOrderPairGradientTerm& second_item = second_expansion.terms[second_term];
      const double sign =
          (low_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      const unsigned derivative_state = first_item.derivative_state + second_item.derivative_state;
      const double coulomb = low_order_coulomb(derivative_state, rho, product_difference, boys);
      const double coefficient = sign * first_item.coefficient * second_item.coefficient;
      value += coefficient * coulomb;
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        const double first_pair_gradient =
            sign * first_item.first_center[coordinate] * second_item.coefficient;
        const double second_pair_gradient =
            sign * first_item.coefficient * second_item.first_center[coordinate];
        // The raised Coulomb state is shared by every center; only the
        // product-center chain-rule scale differs.
        const double scaled_coulomb_derivative =
            coefficient *
            third_order_coulomb(derivative_state + low_order_derivative_state(coordinate), rho,
                                product_difference, boys);
        value_gradient[0][coordinate] +=
            first_pair_gradient * coulomb + first_product_scale * scaled_coulomb_derivative;
        value_gradient[1][coordinate] +=
            -first_pair_gradient * coulomb + second_product_scale * scaled_coulomb_derivative;
        value_gradient[2][coordinate] +=
            second_pair_gradient * coulomb + third_product_scale * scaled_coulomb_derivative;
      }
    }
  }

  const double pair_decay =
      exp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  const double prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;
  for (unsigned center = 0; center < 3; ++center) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double decay_derivative = 0.0;
      if (center < 2) {
        const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
        decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
      } else {
        const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
        decay_derivative = -2.0 * nu * difference;
      }
      gradient[center][coordinate] =
          prefactor * (value_gradient[center][coordinate] + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] - gradient[2][coordinate];
  }
}

/** Canonicalize and contract all-center gradients for total angular order 2. */
__device__ CartesianQuartetGradient contracted_eri_cartesian_source_order2_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          primitive_eri_order2_gradient(
              batch.primitive_exponents[a], positions[0], angular[0], batch.primitive_exponents[b],
              positions[1], angular[1], batch.primitive_exponents[c], positions[2], angular[2],
              batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

/** One sparse coefficient term in an order-three differentiated pair. */
struct ThirdOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
};

template <unsigned PairOrder>
struct ThirdOrderPairGradientExpansion {
  static_assert(PairOrder <= 3);
  static constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  ThirdOrderPairGradientTerm terms[1U << PairOrder];
  unsigned axes[QuantumStorage];
  double shifts[QuantumStorage];
  double first_center_shift_gradients[QuantumStorage];
  double inverse_two_exponent;
};

/**
 * Differentiate the exact subset/Wick pair expansion through order three.
 *
 * Three-bit Cartesian derivative fields are used because differentiating an
 * order-three Coulomb state can raise one axis to order four. Exponents are
 * fixed nuclear-coordinate parameters, so only P-A/P-B shift products carry
 * coefficient derivatives; Wick factors 1/(2p) remain constant.
 */
template <unsigned PairOrder>
__device__ ThirdOrderPairGradientExpansion<PairOrder> make_third_order_pair_gradient_expansion(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second) {
  static_assert(PairOrder <= 3);
  ThirdOrderPairGradientExpansion<PairOrder> expansion{};
  if constexpr (PairOrder == 0) {
    expansion.terms[0].coefficient = 1.0;
  } else {
    const double exponent = alpha + beta;
    const Vec3<double> product = product_center(alpha, first, beta, second);
    expansion.inverse_two_exponent = 0.5 / exponent;
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        expansion.axes[quantum_count] = static_cast<unsigned>(axis);
        expansion.shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        expansion.first_center_shift_gradients[quantum_count] = alpha / exponent - 1.0;
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        expansion.axes[quantum_count] = static_cast<unsigned>(axis);
        expansion.shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        expansion.first_center_shift_gradients[quantum_count] = alpha / exponent;
        ++quantum_count;
      }
    }

    for (unsigned subset = 0; subset < (1U << PairOrder); ++subset) {
      ThirdOrderPairGradientTerm& term = expansion.terms[subset];
      term.coefficient = 1.0;
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if ((subset & (1U << quantum)) != 0) {
          term.derivative_state += fourth_order_derivative_state(expansion.axes[quantum]);
          term.coefficient *= expansion.inverse_two_exponent;
        } else {
          term.coefficient *= expansion.shifts[quantum];
        }
      }
    }

    if constexpr (PairOrder == 2) {
      if (expansion.axes[0] == expansion.axes[1]) {
        expansion.terms[0].coefficient += expansion.inverse_two_exponent;
      }
    } else if constexpr (PairOrder == 3) {
      for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 3; ++second_quantum) {
          if (expansion.axes[first_quantum] != expansion.axes[second_quantum]) {
            continue;
          }
          const unsigned remaining_quantum = 3U - first_quantum - second_quantum;
          ThirdOrderPairGradientTerm& value_term = expansion.terms[0];
          value_term.coefficient +=
              expansion.inverse_two_exponent * expansion.shifts[remaining_quantum];
          expansion.terms[1U << remaining_quantum].coefficient +=
              expansion.inverse_two_exponent * expansion.inverse_two_exponent;
        }
      }
    }
  }
  return expansion;
}

/** Differentiate one pair coefficient with respect to its first center. */
template <unsigned PairOrder>
__device__ double third_order_pair_first_center_gradient(
    const ThirdOrderPairGradientExpansion<PairOrder>& expansion, unsigned subset,
    unsigned coordinate) {
  if constexpr (PairOrder == 0) {
    return 0.0;
  } else {
    double gradient = 0.0;
    for (unsigned differentiated = 0; differentiated < PairOrder; ++differentiated) {
      if ((subset & (1U << differentiated)) != 0 || expansion.axes[differentiated] != coordinate) {
        continue;
      }
      double derivative = expansion.first_center_shift_gradients[differentiated];
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if (quantum == differentiated) continue;
        derivative *= (subset & (1U << quantum)) != 0 ? expansion.inverse_two_exponent
                                                      : expansion.shifts[quantum];
      }
      gradient += derivative;
    }
    if constexpr (PairOrder == 3) {
      if (subset == 0) {
        for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
          for (unsigned second_quantum = first_quantum + 1; second_quantum < 3; ++second_quantum) {
            if (expansion.axes[first_quantum] != expansion.axes[second_quantum]) {
              continue;
            }
            const unsigned remaining_quantum = 3U - first_quantum - second_quantum;
            if (expansion.axes[remaining_quantum] == coordinate) {
              gradient += expansion.inverse_two_exponent *
                          expansion.first_center_shift_gradients[remaining_quantum];
            }
          }
        }
      }
    }
    return gradient;
  }
}

/** Evaluate all-center derivatives of one canonical order-three primitive. */
template <unsigned FirstPairOrder, unsigned SecondPairOrder>
__device__ void primitive_eri_order3_gradient(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second, double gamma,
    const Vec3<double>& third, const Angular& angular_third, double delta,
    const Vec3<double>& fourth, const Angular& angular_fourth, double (&gradient)[4][3]) {
  static_assert(FirstPairOrder + SecondPairOrder == 3);
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<double> product_p = product_center(alpha, first, beta, second);
  const Vec3<double> product_q = product_center(gamma, third, delta, fourth);
  const Vec3<double> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };
  const ThirdOrderPairGradientExpansion<FirstPairOrder> first_expansion =
      make_third_order_pair_gradient_expansion<FirstPairOrder>(alpha, first, angular_first, beta,
                                                               second, angular_second);
  const ThirdOrderPairGradientExpansion<SecondPairOrder> second_expansion =
      make_third_order_pair_gradient_expansion<SecondPairOrder>(gamma, third, angular_third, delta,
                                                                fourth, angular_fourth);
  double boys[5];
  boys_values<4>(rho * distance_squared(product_p, product_q), boys);
  const double first_product_scale = alpha / p;
  const double second_product_scale = beta / p;
  const double third_product_scale = -gamma / q;
  double value = 0.0;
  // The fourth center is restored from translational invariance. Besides
  // removing one quarter of the center updates, this keeps the hot primitive
  // path from spilling another three-component accumulator to local memory.
  double value_gradient[3][3]{};
  for (unsigned first_term = 0; first_term < (1U << FirstPairOrder); ++first_term) {
    for (unsigned second_term = 0; second_term < (1U << SecondPairOrder); ++second_term) {
      const ThirdOrderPairGradientTerm& first_item = first_expansion.terms[first_term];
      const ThirdOrderPairGradientTerm& second_item = second_expansion.terms[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      const unsigned derivative_state = first_item.derivative_state + second_item.derivative_state;
      const double coulomb = fourth_order_coulomb(derivative_state, rho, product_difference, boys);
      const double coefficient = sign * first_item.coefficient * second_item.coefficient;
      value += coefficient * coulomb;
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        const double first_pair_gradient =
            third_order_pair_first_center_gradient(first_expansion, first_term, coordinate);
        const double second_pair_gradient =
            third_order_pair_first_center_gradient(second_expansion, second_term, coordinate);
        const double first_coefficient_gradient =
            sign * first_pair_gradient * second_item.coefficient;
        const double second_coefficient_gradient =
            sign * first_item.coefficient * second_pair_gradient;
        // The Cartesian Coulomb derivative is center independent; only the
        // product-center chain-rule scale changes. Compute it once instead of
        // repeating the fourth-order closed form for all four centers.
        const double scaled_coulomb_derivative =
            coefficient *
            fourth_order_coulomb(derivative_state + fourth_order_derivative_state(coordinate), rho,
                                 product_difference, boys);
        value_gradient[0][coordinate] +=
            first_coefficient_gradient * coulomb + first_product_scale * scaled_coulomb_derivative;
        value_gradient[1][coordinate] += -first_coefficient_gradient * coulomb +
                                         second_product_scale * scaled_coulomb_derivative;
        value_gradient[2][coordinate] +=
            second_coefficient_gradient * coulomb + third_product_scale * scaled_coulomb_derivative;
      }
    }
  }

  const double pair_decay =
      exp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  const double prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;
  for (unsigned center = 0; center < 3; ++center) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double decay_derivative = 0.0;
      if (center < 2) {
        const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
        decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
      } else {
        const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
        decay_derivative = -2.0 * nu * difference;
      }
      gradient[center][coordinate] =
          prefactor * (value_gradient[center][coordinate] + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] - gradient[2][coordinate];
  }
}

/** Canonicalize and contract all-center gradients for total angular order 3. */
__device__ CartesianQuartetGradient contracted_eri_cartesian_source_order3_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] + batch.shell_angular[slots[1].shell];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          if (first_pair_order == 3) {
            primitive_eri_order3_gradient<3, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else {
            primitive_eri_order3_gradient<2, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          }
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

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
__device__ void add_high_order_wick_matching_term(
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
__device__ HighOrderPairGradientGeometry<PairOrder> make_high_order_pair_gradient_geometry(
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
__device__ HighOrderPairGradientTerm make_high_order_pair_gradient_term(
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

/** Primitive-local powers reused by one bounded high-order Coulomb recurrence. */
template <unsigned MaximumOrder>
struct HighOrderCoulombWorkspace {
  static_assert(MaximumOrder >= 5 && MaximumOrder <= 7);
  Vec3<double> difference;
  double coordinate_powers[3][MaximumOrder + 1];
  double negative_two_rho_powers[MaximumOrder + 1];
};

template <unsigned MaximumOrder>
__device__ HighOrderCoulombWorkspace<MaximumOrder> make_high_order_coulomb_workspace(
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
__device__ unsigned axis_wick_multiplicity(unsigned order, unsigned pairs) {
  if (pairs == 0) return 1U;
  if (pairs == 1) return order * (order - 1U) / 2U;
  if (pairs == 2) {
    return order * (order - 1U) * (order - 2U) * (order - 3U) / 8U;
  }
  return order * (order - 1U) * (order - 2U) * (order - 3U) * (order - 4U) * (order - 5U) / 48U;
}

/** Evaluate one Cartesian Coulomb derivative through `MaximumOrder`. */
template <unsigned MaximumOrder>
__device__ double high_order_coulomb(unsigned derivative_state, double rho,
                                     const HighOrderCoulombWorkspace<MaximumOrder>& workspace,
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

/** Evaluate all-center derivatives of one canonical order-four to-six primitive. */
template <unsigned FirstPairOrder, unsigned SecondPairOrder>
__device__ void primitive_eri_order456_gradient(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second, double gamma,
    const Vec3<double>& third, const Angular& angular_third, double delta,
    const Vec3<double>& fourth, const Angular& angular_fourth, double (&gradient)[4][3]) {
  constexpr unsigned AngularOrder = FirstPairOrder + SecondPairOrder;
  constexpr unsigned CoulombOrder = AngularOrder + 1;
  static_assert(AngularOrder == 4 || AngularOrder == 5 || AngularOrder == 6);
  static_assert(FirstPairOrder >= SecondPairOrder);
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<double> product_p = product_center(alpha, first, beta, second);
  const Vec3<double> product_q = product_center(gamma, third, delta, fourth);
  const Vec3<double> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };
  const HighOrderPairGradientGeometry<FirstPairOrder> first_geometry =
      make_high_order_pair_gradient_geometry<FirstPairOrder>(alpha, first, angular_first, beta,
                                                             second, angular_second);
  const HighOrderPairGradientGeometry<SecondPairOrder> second_geometry =
      make_high_order_pair_gradient_geometry<SecondPairOrder>(gamma, third, angular_third, delta,
                                                              fourth, angular_fourth);
  double boys[AngularOrder + 2];
  boys_values<AngularOrder + 1>(rho * distance_squared(product_p, product_q), boys);
  const HighOrderCoulombWorkspace<CoulombOrder> coulomb_workspace =
      make_high_order_coulomb_workspace<CoulombOrder>(rho, product_difference);
  const double first_product_scale = alpha / p;
  const double second_product_scale = beta / p;
  const double third_product_scale = -gamma / q;
  double value = 0.0;
  double value_gradient[3][3]{};
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;
  // Canonical pair ordering keeps the second expansion small (at most eight
  // terms through total order six). Materialize those terms once; generate
  // each larger first-pair term immediately before consuming it so its
  // coefficient and gradient do not create another full local array.
  HighOrderPairGradientTerm second_items[SecondTermCount];
  for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
    second_items[second_term] = make_high_order_pair_gradient_term(second_geometry, second_term);
  }
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    const HighOrderPairGradientTerm first_item =
        make_high_order_pair_gradient_term(first_geometry, first_term);
    for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
      const HighOrderPairGradientTerm& second_item = second_items[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      const unsigned derivative_state = first_item.derivative_state + second_item.derivative_state;
      const double coulomb =
          high_order_coulomb<CoulombOrder>(derivative_state, rho, coulomb_workspace, boys);
      const double coefficient = sign * first_item.coefficient * second_item.coefficient;
      value += coefficient * coulomb;
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        const double first_coefficient_gradient =
            sign * first_item.first_center[coordinate] * second_item.coefficient;
        const double second_coefficient_gradient =
            sign * first_item.coefficient * second_item.first_center[coordinate];
        const double scaled_coulomb_derivative =
            coefficient * high_order_coulomb<CoulombOrder>(
                              derivative_state + fourth_order_derivative_state(coordinate), rho,
                              coulomb_workspace, boys);
        value_gradient[0][coordinate] +=
            first_coefficient_gradient * coulomb + first_product_scale * scaled_coulomb_derivative;
        value_gradient[1][coordinate] += -first_coefficient_gradient * coulomb +
                                         second_product_scale * scaled_coulomb_derivative;
        value_gradient[2][coordinate] +=
            second_coefficient_gradient * coulomb + third_product_scale * scaled_coulomb_derivative;
      }
    }
  }

  const double pair_decay =
      exp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  const double prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;
  for (unsigned center = 0; center < 3; ++center) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double decay_derivative = 0.0;
      if (center < 2) {
        const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
        decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
      } else {
        const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
        decay_derivative = -2.0 * nu * difference;
      }
      gradient[center][coordinate] =
          prefactor * (value_gradient[center][coordinate] + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] - gradient[2][coordinate];
  }
}

/** Canonicalize and contract all-center gradients for total angular order 4. */
__device__ CartesianQuartetGradient contracted_eri_cartesian_source_order4_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] + batch.shell_angular[slots[1].shell];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          if (first_pair_order == 4) {
            primitive_eri_order456_gradient<4, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else if (first_pair_order == 3) {
            primitive_eri_order456_gradient<3, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else {
            primitive_eri_order456_gradient<2, 2>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          }
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

/** Canonicalize and contract all-center gradients for total angular order 5. */
__device__ CartesianQuartetGradient contracted_eri_cartesian_source_order5_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] + batch.shell_angular[slots[1].shell];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          if (first_pair_order == 5) {
            primitive_eri_order456_gradient<5, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else if (first_pair_order == 4) {
            primitive_eri_order456_gradient<4, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else {
            primitive_eri_order456_gradient<3, 2>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          }
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

/** Canonicalize and contract all-center gradients for total angular order 6. */
__device__ CartesianQuartetGradient contracted_eri_cartesian_source_order6_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] + batch.shell_angular[slots[1].shell];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          if (first_pair_order == 6) {
            primitive_eri_order456_gradient<6, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else if (first_pair_order == 5) {
            primitive_eri_order456_gradient<5, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else if (first_pair_order == 4) {
            primitive_eri_order456_gradient<4, 2>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else {
            primitive_eri_order456_gradient<3, 3>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          }
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

/** Canonicalize one Cartesian source quartet to its exact shell class. */
template <unsigned ShellClass, typename Scalar>
__device__ Scalar contracted_eri_cartesian_source_shell_class(const DeviceBatch& batch,
                                                              std::int32_t system, std::int32_t i,
                                                              std::int32_t j, std::int32_t k,
                                                              std::int32_t l,
                                                              std::int64_t derivative_coordinate) {
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
__device__ Scalar dispatch_contracted_eri_cartesian_source_shell_class(
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
__device__ Scalar contracted_eri_cartesian_source(const DeviceBatch& batch, std::int32_t system,
                                                  std::int32_t i, std::int32_t j, std::int32_t k,
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

__global__ void build_eri_kernel(DeviceBatch batch, double* eri) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t eri_size = n * n * n * n;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * eri_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / eri_size);
  std::size_t local = element % eri_size;
  const std::int32_t l = static_cast<std::int32_t>(local % n);
  local /= n;
  const std::int32_t k = static_cast<std::int32_t>(local % n);
  local /= n;
  const std::int32_t j = static_cast<std::int32_t>(local % n);
  const std::int32_t i = static_cast<std::int32_t>(local / n);
  eri[element] = contracted_eri<double>(batch, system, i, j, k, l, -1);
}

#include "scf/cuda/weighted_eri.cuh"

__global__ void build_fock_kernel(std::int32_t batch_size, std::int32_t nbf, const double* hcore,
                                  const double* eri, const double* density,
                                  const std::uint8_t* active, double* fock) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local % n;
  const std::size_t j = local / n;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t eri_offset = static_cast<std::size_t>(system) * eri_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const double pkl = density[matrix_offset + matrix_index(k, l, n)];
      coulomb += pkl * eri[eri_offset + eri_index(i, j, k, l, n)];
      exchange += pkl * eri[eri_offset + eri_index(i, k, j, l, n)];
    }
  }
  fock[element] = hcore[element] + coulomb - 0.5 * exchange;
}

__global__ void build_uhf_fock_kernel(std::int32_t batch_size, std::int32_t nbf,
                                      const double* hcore, const double* eri, const double* density,
                                      const std::uint8_t* active, double* fock) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * 2 * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / 2;
  if (active != nullptr && active[system] == 0) return;
  const std::size_t spin = state % 2;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local % n;
  const std::size_t j = local / n;
  const std::size_t physical_matrix_offset = system * matrix_size;
  const std::size_t eri_offset = system * eri_size;
  const std::size_t alpha_offset = system * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t spin_offset = alpha_offset + spin * matrix_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const std::size_t kl = matrix_index(k, l, n);
      const double total = density[alpha_offset + kl] + density[beta_offset + kl];
      coulomb += total * eri[eri_offset + eri_index(i, j, k, l, n)];
      exchange += density[spin_offset + kl] * eri[eri_offset + eri_index(i, k, j, l, n)];
    }
  }
  fock[element] = hcore[physical_matrix_offset + local] + coulomb - exchange;
}

__global__ void build_schwarz_bounds_packed_kernel(DeviceBatch batch, std::size_t pair_count,
                                                   double* schwarz_bounds) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t pair = element % pair_count;
  std::size_t i = 0;
  std::size_t j = 0;
  decode_lower_triangle(pair, i, j);
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * n * n;
  const double diagonal = contracted_eri_cartesian_source<double>(
      batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
      static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), -1);
  // fabs is conservative when roundoff makes a non-negative diagonal
  // slightly negative; it never converts that noise into a false zero.
  const double bound = sqrt(fabs(diagonal));
  schwarz_bounds[matrix_offset + matrix_index(i, j, n)] = bound;
  if (i != j) {
    schwarz_bounds[matrix_offset + matrix_index(j, i, n)] = bound;
  }
}

/** Atomically retain the maximum non-negative IEEE-754 double. */
__device__ void atomic_max_double(double* address, double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (__longlong_as_double(old) < value) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, __double_as_longlong(value));
    if (old == assumed) return;
  }
}

/**
 * Build AO-pair Schwarz bounds and shell-pair maxima in one packed pass.
 *
 * AO-pair bounds are naturally a dense packed workload: a 768-AO system has
 * only 295,296 canonical AO pairs but 73,920 shell pairs.  Assigning one
 * 256-thread block to every shell pair leaves nearly all lanes idle for the
 * common s/p shells.  This kernel keeps the original packed grid and uses a
 * low-contention atomic maximum for the shell-pair reduction while each
 * contracted diagonal ERI is still in registers.
 */
__global__ void build_schwarz_and_shell_pair_bounds_packed_kernel(DeviceBatch batch,
                                                                  std::size_t pair_count,
                                                                  double* schwarz_bounds,
                                                                  double* shell_pair_bounds) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(batch.batch_size) * pair_count;
  if (element >= total) return;

  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t pair = element % pair_count;
  std::size_t i = 0;
  std::size_t j = 0;
  decode_lower_triangle(pair, i, j);
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * n * n;
  const double diagonal = contracted_eri_cartesian_source<double>(
      batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
      static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), -1);
  // fabs is conservative when roundoff makes a non-negative diagonal
  // slightly negative; it never converts that noise into a false zero.
  const double bound = sqrt(fabs(diagonal));
  schwarz_bounds[matrix_offset + matrix_index(i, j, n)] = bound;
  if (i != j) {
    schwarz_bounds[matrix_offset + matrix_index(j, i, n)] = bound;
  }

  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::int32_t first_shell = batch.direct_ao_shells[system_ao_begin + i];
  const std::int32_t second_shell = batch.direct_ao_shells[system_ao_begin + j];
  // The direct-AO-to-shell map is topology metadata.  Keep malformed or
  // stale entries from turning the unsigned local-index arithmetic below
  // into an arena write outside this system's shell-pair segment.
  const std::int64_t shell_begin = batch.system_shell_offsets[system];
  const std::int64_t shell_end = batch.system_shell_offsets[system + 1];
  if (first_shell < shell_begin || second_shell < shell_begin || first_shell >= shell_end ||
      second_shell >= shell_end) {
    return;
  }
  const std::size_t shell_pair = system_shell_pair_index(batch, system, first_shell, second_shell);
  const std::size_t shell_pair_begin =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
  const std::size_t shell_pair_end =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[system + 1]);
  if (shell_pair < shell_pair_begin || shell_pair >= shell_pair_end ||
      shell_pair >= static_cast<std::size_t>(batch.total_shell_pairs)) {
    return;
  }
  atomic_max_double(shell_pair_bounds + shell_pair, bound);
}

/** Prepare the resident ppps histogram, prefix, descriptors, and ket records. */
cudaError_t prepare_ppps_resident_tasks(
    cudaStream_t stream, std::size_t active_tile_capacity, std::size_t total_shell_pairs,
    DeviceBatch batch, const std::uint32_t* active_tile_count,
    const ActiveShellQuartetTile* active_tiles, GeneratedPppsResidentTask* resident_tasks,
    GeneratedShellTask* resident_ket_tasks, std::uint32_t* resident_bra_counts,
    std::uint32_t* resident_bra_offsets, std::uint32_t* resident_bra_write_counts,
    std::uint32_t* resident_signature_counts, std::uint32_t* resident_signature_offsets,
    std::uint32_t* resident_ket_signatures, std::uint64_t enabled_mask) {
  if (active_tile_capacity == 0 || total_shell_pairs == 0 || resident_tasks == nullptr ||
      resident_ket_tasks == nullptr || resident_bra_counts == nullptr ||
      resident_bra_offsets == nullptr || resident_bra_write_counts == nullptr ||
      (enabled_mask & (std::uint64_t{1} << kPppsShellClass)) == 0U) {
    return cudaSuccess;
  }
  cudaError_t error =
      cudaMemsetAsync(resident_bra_counts, 0, total_shell_pairs * sizeof(std::uint32_t), stream);
  if (error != cudaSuccess) return error;
  if (resident_signature_counts != nullptr) {
    error = cudaMemsetAsync(resident_signature_counts, 0,
                            total_shell_pairs * kPppsSignatureBucketCount * sizeof(std::uint32_t),
                            stream);
    if (error != cudaSuccess) return error;
  }
  constexpr unsigned preparation_threads = kCaptureSafeKernelThreads;
  const unsigned preparation_blocks =
      static_cast<unsigned>((active_tile_capacity + preparation_threads - 1) / preparation_threads);
  launch_count_ppps_resident_bra_tasks_kernel(preparation_blocks, preparation_threads, 0, stream,
                                              batch, active_tile_capacity, active_tile_count,
                                              active_tiles, enabled_mask, resident_bra_counts,
                                              resident_signature_counts);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  launch_prefix_ppps_resident_bra_tasks_kernel(1, 1, 0, stream, total_shell_pairs,
                                               resident_bra_counts, resident_bra_offsets,
                                               resident_bra_write_counts, resident_tasks);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  if (resident_signature_counts != nullptr && resident_signature_offsets != nullptr) {
    launch_prefix_ppps_resident_signature_buckets_kernel(
        static_cast<unsigned>(total_shell_pairs), 1, 0, stream, total_shell_pairs,
        resident_bra_offsets, resident_signature_counts, resident_signature_offsets);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
  }
  launch_materialize_ppps_resident_bra_tasks_kernel(
      preparation_blocks, preparation_threads, 0, stream, batch, active_tile_capacity,
      active_tile_count, active_tiles, resident_bra_offsets, resident_bra_write_counts,
      resident_signature_offsets, resident_signature_counts, resident_ket_tasks,
      resident_ket_signatures);
  return cudaPeekAtLastError();
}

__global__ void build_fock_direct_packed_kernel(
    DeviceBatch batch, double screening_tolerance, const double* hcore,
    const std::int32_t* ao_pair_first, const std::int32_t* ao_pair_second, std::size_t pair_count,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock) {
  extern __shared__ double pair_sums[];
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t matrix_element = static_cast<std::size_t>(blockIdx.x);
  if (matrix_element >= static_cast<std::size_t>(batch.batch_size) * matrix_size) {
    return;
  }
  const std::int32_t system = static_cast<std::int32_t>(matrix_element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local_matrix = matrix_element % matrix_size;
  const std::size_t i = local_matrix % n;
  const std::size_t j = local_matrix / n;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double bound_ij = schwarz_bounds[matrix_offset + matrix_index(i, j, n)];

  double contribution = 0.0;
  for (std::size_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
    const std::size_t k = static_cast<std::size_t>(ao_pair_first[pair]);
    const std::size_t l = static_cast<std::size_t>(ao_pair_second[pair]);
    const double pkl = density[matrix_offset + matrix_index(k, l, n)];
    if (pkl == 0.0) continue;

    if (bound_ij * schwarz_bounds[matrix_offset + matrix_index(k, l, n)] >= screening_tolerance) {
      const double pair_weight = k == l ? pkl : 2.0 * pkl;
      contribution += pair_weight * contracted_eri<double>(
                                        batch, system, static_cast<std::int32_t>(i),
                                        static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
                                        static_cast<std::int32_t>(l), -1);
    }
    if (schwarz_bounds[matrix_offset + matrix_index(i, k, n)] *
            schwarz_bounds[matrix_offset + matrix_index(j, l, n)] >=
        screening_tolerance) {
      contribution -= 0.5 * pkl *
                      contracted_eri<double>(
                          batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(k),
                          static_cast<std::int32_t>(j), static_cast<std::int32_t>(l), -1);
    }
    if (k != l && schwarz_bounds[matrix_offset + matrix_index(i, l, n)] *
                          schwarz_bounds[matrix_offset + matrix_index(j, k, n)] >=
                      screening_tolerance) {
      contribution -= 0.5 * pkl *
                      contracted_eri<double>(
                          batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(l),
                          static_cast<std::int32_t>(j), static_cast<std::int32_t>(k), -1);
    }
  }

  pair_sums[threadIdx.x] = contribution;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      pair_sums[threadIdx.x] += pair_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    fock[matrix_element] = hcore[matrix_element] + pair_sums[0];
  }
}

__global__ void build_uhf_fock_direct_packed_kernel(
    DeviceBatch batch, double screening_tolerance, const double* hcore,
    const std::int32_t* ao_pair_first, const std::int32_t* ao_pair_second, std::size_t pair_count,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock) {
  extern __shared__ double pair_sums[];
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t matrix_element = static_cast<std::size_t>(blockIdx.x);
  if (matrix_element >= static_cast<std::size_t>(batch.batch_size) * 2 * matrix_size) {
    return;
  }
  const std::size_t state = matrix_element / matrix_size;
  const std::size_t system = state / 2;
  if (active != nullptr && active[system] == 0) return;
  const std::size_t spin = state % 2;
  const std::size_t local_matrix = matrix_element % matrix_size;
  const std::size_t i = local_matrix % n;
  const std::size_t j = local_matrix / n;
  const std::size_t physical_offset = system * matrix_size;
  const std::size_t alpha_offset = system * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t spin_offset = alpha_offset + spin * matrix_size;
  const double bound_ij = schwarz_bounds[physical_offset + matrix_index(i, j, n)];

  double contribution = 0.0;
  for (std::size_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
    const std::size_t k = static_cast<std::size_t>(ao_pair_first[pair]);
    const std::size_t l = static_cast<std::size_t>(ao_pair_second[pair]);
    const std::size_t kl = matrix_index(k, l, n);
    const double alpha = density[alpha_offset + kl];
    const double beta = density[beta_offset + kl];
    const double same_spin = density[spin_offset + kl];
    const double total = alpha + beta;
    if (total == 0.0 && same_spin == 0.0) continue;

    if (total != 0.0 && bound_ij * schwarz_bounds[physical_offset + kl] >= screening_tolerance) {
      const double pair_weight = k == l ? total : 2.0 * total;
      contribution += pair_weight * contracted_eri<double>(batch, static_cast<std::int32_t>(system),
                                                           static_cast<std::int32_t>(i),
                                                           static_cast<std::int32_t>(j),
                                                           static_cast<std::int32_t>(k),
                                                           static_cast<std::int32_t>(l), -1);
    }
    if (same_spin != 0.0 && schwarz_bounds[physical_offset + matrix_index(i, k, n)] *
                                    schwarz_bounds[physical_offset + matrix_index(j, l, n)] >=
                                screening_tolerance) {
      contribution -= same_spin * contracted_eri<double>(batch, static_cast<std::int32_t>(system),
                                                         static_cast<std::int32_t>(i),
                                                         static_cast<std::int32_t>(k),
                                                         static_cast<std::int32_t>(j),
                                                         static_cast<std::int32_t>(l), -1);
    }
    if (k != l && same_spin != 0.0 &&
        schwarz_bounds[physical_offset + matrix_index(i, l, n)] *
                schwarz_bounds[physical_offset + matrix_index(j, k, n)] >=
            screening_tolerance) {
      contribution -= same_spin * contracted_eri<double>(batch, static_cast<std::int32_t>(system),
                                                         static_cast<std::int32_t>(i),
                                                         static_cast<std::int32_t>(l),
                                                         static_cast<std::int32_t>(j),
                                                         static_cast<std::int32_t>(k), -1);
    }
  }

  pair_sums[threadIdx.x] = contribution;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      pair_sums[threadIdx.x] += pair_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    fock[matrix_element] = hcore[physical_offset + local_matrix] + pair_sums[0];
  }
}

__device__ void eri_symmetry_permutation(unsigned permutation, std::size_t i, std::size_t j,
                                         std::size_t k, std::size_t l, std::size_t& a,
                                         std::size_t& b, std::size_t& c, std::size_t& d) {
  switch (permutation) {
    case 0:
      a = i;
      b = j;
      c = k;
      d = l;
      break;
    case 1:
      a = j;
      b = i;
      c = k;
      d = l;
      break;
    case 2:
      a = i;
      b = j;
      c = l;
      d = k;
      break;
    case 3:
      a = j;
      b = i;
      c = l;
      d = k;
      break;
    case 4:
      a = k;
      b = l;
      c = i;
      d = j;
      break;
    case 5:
      a = l;
      b = k;
      c = i;
      d = j;
      break;
    case 6:
      a = k;
      b = l;
      c = j;
      d = i;
      break;
    default:
      a = l;
      b = k;
      c = j;
      d = i;
      break;
  }
}

/** Test uniqueness directly from the canonical pair symmetries. */
__device__ bool unique_eri_symmetry_permutation(unsigned permutation, std::size_t i, std::size_t j,
                                                std::size_t k, std::size_t l) {
  const bool pair_swapped = permutation >= 4;
  const bool first_pair_diagonal = pair_swapped ? k == l : i == j;
  const bool second_pair_diagonal = pair_swapped ? i == j : k == l;
  if ((permutation & 1U) != 0 && first_pair_diagonal) return false;
  if ((permutation & 2U) != 0 && second_pair_diagonal) return false;
  return !pair_swapped || i != k || j != l;
}

/** Scatter one symmetry-canonical ERI into the direct RHF/UHF Fock matrix. */
template <bool Unrestricted>
__device__ __forceinline__ void accumulate_direct_fock_integral(
    std::size_t n, std::size_t physical_offset, std::size_t spin_offset, const double* density,
    double* fock, std::size_t i, std::size_t j, std::size_t k, std::size_t l, double integral) {
  const std::size_t matrix_size = n * n;
  for (unsigned permutation = 0; permutation < 8; ++permutation) {
    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) {
      continue;
    }
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::size_t d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);
    const std::size_t ab = matrix_index(a, b, n);
    const std::size_t ac = matrix_index(a, c, n);
    const std::size_t cd = matrix_index(c, d, n);
    const std::size_t bd = matrix_index(b, d, n);
    if constexpr (Unrestricted) {
      const double alpha_cd = density[spin_offset + cd];
      const double beta_cd = density[spin_offset + matrix_size + cd];
      const double total_cd = alpha_cd + beta_cd;
      if (total_cd != 0.0) {
        atomicAdd(fock + spin_offset + ab, total_cd * integral);
        atomicAdd(fock + spin_offset + matrix_size + ab, total_cd * integral);
      }
      const double alpha_bd = density[spin_offset + bd];
      const double beta_bd = density[spin_offset + matrix_size + bd];
      if (alpha_bd != 0.0) {
        atomicAdd(fock + spin_offset + ac, -alpha_bd * integral);
      }
      if (beta_bd != 0.0) {
        atomicAdd(fock + spin_offset + matrix_size + ac, -beta_bd * integral);
      }
    } else {
      const double density_cd = density[physical_offset + cd];
      const double density_bd = density[physical_offset + bd];
      if (density_cd != 0.0) {
        atomicAdd(fock + physical_offset + ab, density_cd * integral);
      }
      if (density_bd != 0.0) {
        atomicAdd(fock + physical_offset + ac, -0.5 * density_bd * integral);
      }
    }
  }
}

template <bool Unrestricted, unsigned AngularOrder, typename EvalScalar = double>
__device__ __forceinline__ void contract_fock_direct_quartet_subtile(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask, std::size_t active_subtile,
    unsigned ao_quartet_lane) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  constexpr std::size_t subtiles_per_tile = detail::direct_quartet_subtiles_per_tile(AngularOrder);
  const std::size_t active_tile = active_subtile / subtiles_per_tile;
  if (active_tile >= static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const std::size_t subtile = active_subtile % subtiles_per_tile;
  const ActiveShellQuartetTile task = active_shell_quartet_tiles[active_tile];
  if (task.first_pair >= static_cast<std::uint32_t>(batch.total_shell_pairs) ||
      task.second_pair >= static_cast<std::uint32_t>(batch.total_shell_pairs)) {
    return;
  }
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (system < 0 || system >= batch.batch_size || batch.shell_pair_systems[second_pair] != system) {
    return;
  }
  if (active != nullptr && active[system] == 0) return;
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (generated_fock_shell_class_mask != nullptr &&
      ((*generated_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U)) {
    return;
  }

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;
  const std::size_t ordinal = static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize +
                              subtile * detail::kDirectQuartetThreads + ao_quartet_lane;
  if (ordinal < ao_quartet_count) {
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    std::size_t l = 0;
    if (!decode_direct_tile_ao_ordinal(batch, task, ordinal, first_ao_pair_count,
                                       second_ao_pair_count, system_ao_begin, n, i, j, k, l)) {
      return;
    }
    if (schwarz_bounds[physical_offset + matrix_index(i, j, n)] *
            schwarz_bounds[physical_offset + matrix_index(k, l, n)] <
        screening_tolerance) {
      return;
    }
    const EvalScalar evaluated_integral =
        dispatch_contracted_eri_cartesian_source_shell_class<AngularOrder, EvalScalar>(
            shell_class, batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), -1);
    const double integral = scalar_value(evaluated_integral);
    if (integral == 0.0) return;
    accumulate_direct_fock_integral<Unrestricted>(n, physical_offset, spin_offset, density, fock, i,
                                                  j, k, l, integral);
  }
}

/**
 * Evaluate and scatter one complete order-one shell task.
 *
 * Shell-pair topology is index-canonical rather than angular-canonical, so
 * the p shell can occupy any input slot. Integral evaluation is reordered to
 * canonical (p s|s s), while screening and Fock scatter keep the original AO
 * slots to preserve the existing eightfold symmetry semantics.
 */
template <bool Unrestricted>
__device__ __noinline__ void contract_fock_direct_psss_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock) {
  // A psss shell quartet has three AO outputs and therefore exactly one tile.
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active != nullptr && active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  unsigned p_slot = 4;
  unsigned p_count = 0;
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[raw_shell[slot]];
    if (angular == 1) {
      p_slot = slot;
      ++p_count;
    } else if (angular != 0) {
      return;
    }
  }
  if (p_count != 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  std::size_t raw_ao[4] = {
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[0]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[1]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[2]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[3]]) - system_ao_begin,
  };
  const std::size_t p_ao_begin = raw_ao[p_slot];
  unsigned active_axis_mask = 0;
  for (unsigned axis = 0; axis < 3; ++axis) {
    raw_ao[p_slot] = p_ao_begin + axis;
    const double first_bound =
        schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)];
    const double second_bound =
        schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)];
    if (first_bound * second_bound >= screening_tolerance) {
      active_axis_mask |= 1U << axis;
    }
  }
  if (active_axis_mask == 0) return;

  std::int32_t canonical_shell[4] = {raw_shell[0], raw_shell[1], raw_shell[2], raw_shell[3]};
  std::size_t canonical_pair[2] = {first_pair, second_pair};
  if (p_slot == 1) {
    const std::int32_t swap = canonical_shell[0];
    canonical_shell[0] = canonical_shell[1];
    canonical_shell[1] = swap;
  } else if (p_slot >= 2) {
    if (p_slot == 3) {
      const std::int32_t swap = canonical_shell[2];
      canonical_shell[2] = canonical_shell[3];
      canonical_shell[3] = swap;
    }
    const std::int32_t first_swap = canonical_shell[0];
    canonical_shell[0] = canonical_shell[2];
    canonical_shell[2] = first_swap;
    const std::int32_t second_swap = canonical_shell[1];
    canonical_shell[1] = canonical_shell[3];
    canonical_shell[3] = second_swap;
    const std::size_t pair_swap = canonical_pair[0];
    canonical_pair[0] = canonical_pair[1];
    canonical_pair[1] = pair_swap;
  }

  const PsssIntegralVector integral = contracted_eri_cartesian_source_psss(
      batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
      canonical_shell[2], canonical_shell[3]);
  for (unsigned axis = 0; axis < 3; ++axis) {
    if ((active_axis_mask & (1U << axis)) == 0 || integral.axis[axis] == 0.0) {
      continue;
    }
    raw_ao[p_slot] = p_ao_begin + axis;
    accumulate_direct_fock_integral<Unrestricted>(n, physical_offset, spin_offset, density, fock,
                                                  raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3],
                                                  integral.axis[axis]);
  }
}

/** One canonical shell slot and its position in the original quartet. */
struct Order2SourceSlot {
  std::int32_t shell;
  unsigned original;
};

/** Map raw AO slots to the fused vector's canonical Cartesian product. */
__device__ __forceinline__ unsigned order2_component_index(const DeviceBatch& batch,
                                                           const Order2SourceSlot (&slots)[4],
                                                           const std::size_t (&raw_ao)[4],
                                                           std::size_t system_ao_begin) {
  unsigned output = 0;
#pragma unroll
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[slots[slot].shell];
    const unsigned component_count = (angular + 1) * (angular + 2) / 2;
    const std::size_t local_begin =
        static_cast<std::size_t>(batch.shell_direct_ao_offsets[slots[slot].shell]) -
        system_ao_begin;
    const unsigned component = static_cast<unsigned>(raw_ao[slots[slot].original] - local_begin);
    output = output * component_count + component;
  }
  return output;
}

/** Evaluate and scatter one complete psps, ppss, or dsss shell task. */
template <bool Unrestricted>
__device__ __noinline__ void contract_fock_direct_order2_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active != nullptr && active[system] == 0) return;

  Order2SourceSlot slots[4] = {
      {batch.shell_pair_first[first_pair], 0},
      {batch.shell_pair_second[first_pair], 1},
      {batch.shell_pair_first[second_pair], 2},
      {batch.shell_pair_second[second_pair], 3},
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell],
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (shell_class != 2U && shell_class != 3U && shell_class != 6U) return;
  // Generated order-two workers execute before this handwritten fallback.
  // Honor the exact-class mask here as the generic subtile path does, or the
  // same shell quartet is scattered into the Fock matrix twice.
  if (generated_fock_shell_class_mask != nullptr &&
      ((*generated_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U)) {
    return;
  }

  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const Order2SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const Order2SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const Order2SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const Order2SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;

  unsigned active_component_mask = 0;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, raw_ao[2], raw_ao[3]);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    active_component_mask |= 1U << order2_component_index(batch, slots, raw_ao, system_ao_begin);
  }
  if (active_component_mask == 0) return;

  Order2IntegralVector integral{};
  switch (shell_class) {
    case 2:
      integral = contracted_eri_cartesian_source_order2_shell<1, 0, 1, 0>(
          batch, slots[0].shell, slots[1].shell, slots[2].shell, slots[3].shell,
          active_component_mask);
      break;
    case 3:
      integral = contracted_eri_cartesian_source_order2_shell<1, 1, 0, 0>(
          batch, slots[0].shell, slots[1].shell, slots[2].shell, slots[3].shell,
          active_component_mask);
      break;
    default:
      integral = contracted_eri_cartesian_source_order2_shell<2, 0, 0, 0>(
          batch, slots[0].shell, slots[1].shell, slots[2].shell, slots[3].shell,
          active_component_mask);
      break;
  }

  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, raw_ao[2], raw_ao[3]);
    const unsigned component = order2_component_index(batch, slots, raw_ao, system_ao_begin);
    if ((active_component_mask & (1U << component)) == 0 || integral.component[component] == 0.0) {
      continue;
    }
    accumulate_direct_fock_integral<Unrestricted>(n, physical_offset, spin_offset, density, fock,
                                                  raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3],
                                                  integral.component[component]);
  }
}

/** Fixed-capacity wrapper retained for high-register angular orders. */
template <bool Unrestricted, unsigned AngularOrder, typename EvalScalar = double>
__global__ void build_fock_direct_quartet_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder, EvalScalar>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
      schwarz_bounds, density, active, fock, generated_fock_shell_class_mask,
      static_cast<std::size_t>(blockIdx.x), threadIdx.x);
}

/** Pack exact ssss shell tasks across all lanes of one worker warp. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void build_fock_direct_quartet_packed_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock) {
  static_assert(AngularOrder < kPackedSsssAngularOrderCount);
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      // Order zero has one subtile and one exact logical tile per shell
      // quartet, so packed_item is also its compact subtile index.
      contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder>(
          batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
          schwarz_bounds, density, active, fock, nullptr, packed_item, 0U);
    }
  }
}

/** Consume complete psss shell tasks, one independent task per lane. */
template <bool Unrestricted>
__global__ void build_fock_direct_psss_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock) {
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_fock_direct_psss_task<Unrestricted>(batch, active_shell_quartet_tiles[packed_item],
                                                   screening_tolerance, schwarz_bounds, density,
                                                   active, fock);
    }
    // Tail lanes must remain live until the next warp-uniform queue exit so
    // the full-mask shuffle above is valid on every persistent iteration.
  }
}

/** Consume complete order-two shell tasks, one independent task per lane. */
template <bool Unrestricted>
__global__ void build_fock_direct_order2_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_fock_direct_order2_task<Unrestricted>(batch, active_shell_quartet_tiles[packed_item],
                                                     screening_tolerance, schwarz_bounds, density,
                                                     active, fock, generated_fock_shell_class_mask);
    }
  }
}

/** Consume only the active compacted Fock domain from a device queue. */
template <bool Unrestricted, unsigned AngularOrder, typename EvalScalar = double>
__global__ void build_fock_direct_quartet_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  const unsigned lane = threadIdx.x % warpSize;
  constexpr std::uint32_t subtiles_per_tile =
      static_cast<std::uint32_t>(detail::direct_quartet_subtiles_per_tile(AngularOrder));
  const std::uint32_t work_count = *active_shell_quartet_tile_count * subtiles_per_tile;
  while (true) {
    std::uint32_t active_subtile = 0;
    if (lane == 0) active_subtile = atomicAdd(task_head, 1U);
    active_subtile = __shfl_sync(0xffffffffU, active_subtile, 0);
    if (active_subtile >= work_count) return;
    contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder, EvalScalar>(
        batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
        schwarz_bounds, density, active, fock, generated_fock_shell_class_mask, active_subtile,
        threadIdx.x);
  }
}

__global__ void two_electron_force_kernel(DeviceBatch batch, const double* density,
                                          const std::uint8_t* active, double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t quartet_count = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count = static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * quartet_count) return;
  const std::int64_t coordinate = static_cast<std::int64_t>(element / quartet_count);
  std::size_t local = element % quartet_count;
  const std::size_t l = local % n;
  local /= n;
  const std::size_t k = local % n;
  local /= n;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double coefficient = 0.5 * density[matrix_offset + matrix_index(i, j, n)] *
                                 density[matrix_offset + matrix_index(k, l, n)] -
                             0.25 * density[matrix_offset + matrix_index(i, k, n)] *
                                 density[matrix_offset + matrix_index(j, l, n)];
  if (coefficient == 0.0) return;
  const Dual derivative = contracted_eri<Dual>(
      batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
      static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate);
  atomicAdd(forces + coordinate, -coefficient * derivative.derivative);
}

__global__ void two_electron_uhf_force_kernel(DeviceBatch batch, const double* spin_density,
                                              const std::uint8_t* active, double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t quartet_count = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count = static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * quartet_count) return;
  const std::int64_t coordinate = static_cast<std::int64_t>(element / quartet_count);
  std::size_t local = element % quartet_count;
  const std::size_t l = local % n;
  local /= n;
  const std::size_t k = local % n;
  local /= n;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t alpha_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t ij = matrix_index(i, j, n);
  const std::size_t kl = matrix_index(k, l, n);
  const double alpha_ij = spin_density[alpha_offset + ij];
  const double beta_ij = spin_density[beta_offset + ij];
  const double alpha_kl = spin_density[alpha_offset + kl];
  const double beta_kl = spin_density[beta_offset + kl];
  const double total_ij = alpha_ij + beta_ij;
  const double total_kl = alpha_kl + beta_kl;
  const double coefficient = 0.5 * total_ij * total_kl -
                             0.5 * spin_density[alpha_offset + matrix_index(i, k, n)] *
                                 spin_density[alpha_offset + matrix_index(j, l, n)] -
                             0.5 * spin_density[beta_offset + matrix_index(i, k, n)] *
                                 spin_density[beta_offset + matrix_index(j, l, n)];
  if (coefficient == 0.0) return;
  const Dual derivative = contracted_eri<Dual>(
      batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
      static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate);
  atomicAdd(forces + coordinate, -coefficient * derivative.derivative);
}

__global__ void two_electron_force_direct_kernel(
    DeviceBatch batch, double screening_tolerance, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t work_per_coordinate = matrix_size * pair_count;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count = static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * work_per_coordinate) return;
  const std::int64_t coordinate = static_cast<std::int64_t>(element / work_per_coordinate);
  std::size_t local = element % work_per_coordinate;
  const std::size_t packed_kl = local % pair_count;
  local /= pair_count;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::size_t k = static_cast<std::size_t>(pair_first[packed_kl]);
  const std::size_t l = static_cast<std::size_t>(pair_second[packed_kl]);
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double pij = density[matrix_offset + matrix_index(i, j, n)];
  const double pkl = density[matrix_offset + matrix_index(k, l, n)];
  if (pij == 0.0 || pkl == 0.0) return;

  double energy_derivative = 0.0;
  const double coulomb_bound = schwarz_bounds[matrix_offset + matrix_index(i, j, n)] *
                               schwarz_bounds[matrix_offset + matrix_index(k, l, n)];
  if (coulomb_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
        static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate);
    // The packed density pair represents both (k,l) and (l,k) when k != l.
    const double coulomb_coefficient = k == l ? 0.5 * pij * pkl : pij * pkl;
    energy_derivative += coulomb_coefficient * derivative.derivative;
  }

  const double exchange_bound = schwarz_bounds[matrix_offset + matrix_index(i, k, n)] *
                                schwarz_bounds[matrix_offset + matrix_index(j, l, n)];
  if (exchange_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(k),
        static_cast<std::int32_t>(j), static_cast<std::int32_t>(l), coordinate);
    energy_derivative -= 0.25 * pij * pkl * derivative.derivative;
  }
  if (k != l) {
    // Packing (k,l) also represents the swapped density pair. Unlike the
    // Coulomb term, exchange maps it to a distinct integral permutation.
    const double transposed_exchange_bound = schwarz_bounds[matrix_offset + matrix_index(i, l, n)] *
                                             schwarz_bounds[matrix_offset + matrix_index(j, k, n)];
    if (transposed_exchange_bound >= screening_tolerance) {
      const Dual derivative = contracted_eri<Dual>(
          batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(l),
          static_cast<std::int32_t>(j), static_cast<std::int32_t>(k), coordinate);
      energy_derivative -= 0.25 * pij * pkl * derivative.derivative;
    }
  }
  if (energy_derivative != 0.0) {
    atomicAdd(forces + coordinate, -energy_derivative);
  }
}

__global__ void two_electron_uhf_force_direct_kernel(
    DeviceBatch batch, double screening_tolerance, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, const double* schwarz_bounds,
    const double* spin_density, const std::uint8_t* active, double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t work_per_coordinate = matrix_size * pair_count;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count = static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * work_per_coordinate) return;
  const std::int64_t coordinate = static_cast<std::int64_t>(element / work_per_coordinate);
  std::size_t local = element % work_per_coordinate;
  const std::size_t packed_kl = local % pair_count;
  local /= pair_count;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::size_t k = static_cast<std::size_t>(pair_first[packed_kl]);
  const std::size_t l = static_cast<std::size_t>(pair_second[packed_kl]);
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t alpha_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t ij = matrix_index(i, j, n);
  const std::size_t kl = matrix_index(k, l, n);
  const double alpha_ij = spin_density[alpha_offset + ij];
  const double beta_ij = spin_density[beta_offset + ij];
  const double alpha_kl = spin_density[alpha_offset + kl];
  const double beta_kl = spin_density[beta_offset + kl];
  const double total_ij = alpha_ij + beta_ij;
  const double total_kl = alpha_kl + beta_kl;

  double energy_derivative = 0.0;
  const double coulomb_bound =
      schwarz_bounds[physical_offset + ij] * schwarz_bounds[physical_offset + kl];
  if (total_ij != 0.0 && total_kl != 0.0 && coulomb_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
        static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate);
    const double coefficient = k == l ? 0.5 * total_ij * total_kl : total_ij * total_kl;
    energy_derivative += coefficient * derivative.derivative;
  }

  const double exchange_pair = alpha_ij * alpha_kl + beta_ij * beta_kl;
  const double exchange_bound = schwarz_bounds[physical_offset + matrix_index(i, k, n)] *
                                schwarz_bounds[physical_offset + matrix_index(j, l, n)];
  if (exchange_pair != 0.0 && exchange_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(k),
        static_cast<std::int32_t>(j), static_cast<std::int32_t>(l), coordinate);
    energy_derivative -= 0.5 * exchange_pair * derivative.derivative;
  }
  if (k != l && exchange_pair != 0.0) {
    const double transposed_exchange_bound =
        schwarz_bounds[physical_offset + matrix_index(i, l, n)] *
        schwarz_bounds[physical_offset + matrix_index(j, k, n)];
    if (transposed_exchange_bound >= screening_tolerance) {
      const Dual derivative = contracted_eri<Dual>(
          batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(l),
          static_cast<std::int32_t>(j), static_cast<std::int32_t>(k), coordinate);
      energy_derivative -= 0.5 * exchange_pair * derivative.derivative;
    }
  }
  if (energy_derivative != 0.0) {
    atomicAdd(forces + coordinate, -energy_derivative);
  }
}

/** Exact symmetry-reduced density coefficient for one force AO quartet. */
template <bool Unrestricted>
__device__ __forceinline__ double direct_force_density_coefficient(
    std::size_t n, std::size_t physical_offset, std::size_t spin_offset, const double* density,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
  const std::size_t matrix_size = n * n;
  double coefficient = 0.0;
  for (unsigned permutation = 0; permutation < 8; ++permutation) {
    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) {
      continue;
    }
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::size_t d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);
    const std::size_t ab = matrix_index(a, b, n);
    const std::size_t ac = matrix_index(a, c, n);
    const std::size_t cd = matrix_index(c, d, n);
    const std::size_t bd = matrix_index(b, d, n);
    if constexpr (Unrestricted) {
      const double total_ab = density[spin_offset + ab] + density[spin_offset + matrix_size + ab];
      const double total_cd = density[spin_offset + cd] + density[spin_offset + matrix_size + cd];
      coefficient += 0.5 * total_ab * total_cd;
      coefficient -=
          0.5 * (density[spin_offset + ac] * density[spin_offset + bd] +
                 density[spin_offset + matrix_size + ac] * density[spin_offset + matrix_size + bd]);
    } else {
      coefficient += 0.5 * density[physical_offset + ab] * density[physical_offset + cd] -
                     0.25 * density[physical_offset + ac] * density[physical_offset + bd];
    }
  }
  return coefficient;
}

/** Evaluate and write one complete density-weighted ssss force shell task. */
template <bool Unrestricted>
__device__ __forceinline__ void contract_two_electron_force_ssss_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  // Every s shell contains one Cartesian AO, so a valid ssss shell quartet
  // occupies exactly the first compact tile and needs no AO-pair decoding.
  if (task.tile != 0U) return;
  if ((generated_shell_class_mask & std::uint64_t{1}) != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t shells[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  for (unsigned slot = 0; slot < 4; ++slot) {
    if (batch.shell_angular[shells[slot]] != 0U) return;
  }

  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[shells[0]],
      batch.shell_atoms[shells[1]],
      batch.shell_atoms[shells[2]],
      batch.shell_atoms[shells[3]],
  };
  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center || center_atoms[center] == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = center_atoms[center];
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t ao[4] = {
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[0]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[1]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[2]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[3]]) - system_ao_begin,
  };
  if (schwarz_bounds[physical_offset + matrix_index(ao[0], ao[1], n)] *
          schwarz_bounds[physical_offset + matrix_index(ao[2], ao[3], n)] <
      screening_tolerance) {
    return;
  }
  const double density_coefficient = direct_force_density_coefficient<Unrestricted>(
      n, physical_offset, spin_offset, density, ao[0], ao[1], ao[2], ao[3]);
  if (density_coefficient == 0.0) return;
  const double component_weight = density_coefficient *
                                  batch.direct_ao_coefficients[system_ao_begin + ao[0]] *
                                  batch.direct_ao_coefficients[system_ao_begin + ao[1]] *
                                  batch.direct_ao_coefficients[system_ao_begin + ao[2]] *
                                  batch.direct_ao_coefficients[system_ao_begin + ao[3]];
  const SsssWeightedGradient gradient = contracted_eri_cartesian_source_ssss_weighted_gradient(
      batch, first_pair, second_pair, shells[0], shells[1], shells[2], shells[3], component_weight);

  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned center = 0; center < 3; ++center) {
        const double value = gradient.center[center][axis];
        fourth_derivative -= value;
        if (center_atoms[center] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (center_atoms[3] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

/** Evaluate and write one complete density-weighted psss force shell task. */
template <bool Unrestricted, bool ResidentBra = false>
__device__ __noinline__ void contract_two_electron_force_psss_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask,
    const PrimitivePairData* resident_first_pairs = nullptr,
    std::int64_t resident_first_pair_count = 0) {
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class < 64U && (generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    return;
  }
  unsigned p_slot = 4;
  unsigned p_count = 0;
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[raw_shell[slot]];
    if (angular == 1) {
      p_slot = slot;
      ++p_count;
    } else if (angular != 0) {
      return;
    }
  }
  if (p_count != 1) return;

  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[raw_shell[0]],
      batch.shell_atoms[raw_shell[1]],
      batch.shell_atoms[raw_shell[2]],
      batch.shell_atoms[raw_shell[3]],
  };
  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center || center_atoms[center] == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = center_atoms[center];
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  std::size_t raw_ao[4] = {
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[0]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[1]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[2]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[raw_shell[3]]) - system_ao_begin,
  };
  const std::size_t p_ao_begin = raw_ao[p_slot];
  double density_coefficient[3]{};
  for (unsigned axis = 0; axis < 3; ++axis) {
    raw_ao[p_slot] = p_ao_begin + axis;
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    density_coefficient[axis] = direct_force_density_coefficient<Unrestricted>(
        n, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3]);
  }
  if (density_coefficient[0] == 0.0 && density_coefficient[1] == 0.0 &&
      density_coefficient[2] == 0.0) {
    return;
  }

  std::int32_t slots[4] = {raw_shell[0], raw_shell[1], raw_shell[2], raw_shell[3]};
  std::size_t canonical_pair[2] = {first_pair, second_pair};
  if (p_slot == 1) {
    const std::int32_t swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  } else if (p_slot >= 2) {
    if (p_slot == 3) {
      const std::int32_t swap = slots[2];
      slots[2] = slots[3];
      slots[3] = swap;
    }
    const std::int32_t first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const std::int32_t second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
    const std::size_t pair_swap = canonical_pair[0];
    canonical_pair[0] = canonical_pair[1];
    canonical_pair[1] = pair_swap;
  }

  const PsssWeightedGradient gradient =
      contracted_eri_cartesian_source_psss_weighted_gradient<ResidentBra>(
          batch, canonical_pair[0], canonical_pair[1], slots[0], slots[1], slots[2], slots[3],
          density_coefficient, resident_first_pairs, resident_first_pair_count);
  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned canonical = 0; canonical < 3; ++canonical) {
        const double value = gradient.center[canonical][axis];
        fourth_derivative -= value;
        if (batch.shell_atoms[slots[canonical]] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (batch.shell_atoms[slots[3]] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

/** Evaluate and write one complete density-weighted psps force shell task. */
template <bool Unrestricted>
__device__ __noinline__ void contract_two_electron_force_psps_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  // A psps shell quartet has at most nine Cartesian AO quartets and therefore
  // always fits in the first compact tile.
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class != kPspsShellClass) return;
  if ((generated_shell_class_mask & (std::uint64_t{1} << kPspsShellClass)) != 0U) {
    return;
  }

  unsigned first_p_slot = 4;
  unsigned second_p_slot = 4;
  for (unsigned slot = 0; slot < 2; ++slot) {
    if (batch.shell_angular[raw_shell[slot]] == 1U) first_p_slot = slot;
  }
  for (unsigned slot = 2; slot < 4; ++slot) {
    if (batch.shell_angular[raw_shell[slot]] == 1U) second_p_slot = slot;
  }
  if (first_p_slot >= 2 || second_p_slot < 2 || second_p_slot >= 4) return;
  const unsigned canonical_raw_slot[4] = {
      first_p_slot,
      1U - first_p_slot,
      second_p_slot,
      5U - second_p_slot,
  };
  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]],
      raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]],
      raw_shell[canonical_raw_slot[3]],
  };

  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    const std::int32_t atom = batch.shell_atoms[canonical_shell[center]];
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center || atom == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = atom;
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_p_ao_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[0]]) - system_ao_begin;
  const std::size_t second_p_ao_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[2]]) - system_ao_begin;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, raw_ao[2], raw_ao[3]);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    const double density_coefficient = direct_force_density_coefficient<Unrestricted>(
        n, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3]);
    if (density_coefficient == 0.0) continue;
    const unsigned first_axis =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[0]] - first_p_ao_begin);
    const unsigned second_axis =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[2]] - second_p_ao_begin);
    if (first_axis >= 3 || second_axis >= 3) return;
    const double angular_coefficient = batch.direct_ao_coefficients[system_ao_begin + raw_ao[0]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[1]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[2]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[3]];
    component_weight[first_axis * 3 + second_axis] += density_coefficient * angular_coefficient;
    any_component = true;
  }
  if (!any_component) return;

  const PspsWeightedGradient gradient = contracted_eri_cartesian_source_psps_weighted_gradient(
      batch, first_pair, second_pair, canonical_shell[0], canonical_shell[1], canonical_shell[2],
      canonical_shell[3], component_weight);
  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned canonical = 0; canonical < 3; ++canonical) {
        const double value = gradient.center[canonical][axis];
        fourth_derivative -= value;
        if (batch.shell_atoms[canonical_shell[canonical]] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (batch.shell_atoms[canonical_shell[3]] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

/** Evaluate one closed ppss or dsss shell task over its exact AO domain. */
template <bool Unrestricted, unsigned TargetShellClass>
__device__ __noinline__ void contract_two_electron_force_pair_order2_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  static_assert(TargetShellClass == kPpssShellClass || TargetShellClass == kDsssShellClass);
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class != TargetShellClass) return;
  if ((generated_shell_class_mask & (std::uint64_t{1} << TargetShellClass)) != 0U) {
    return;
  }

  unsigned canonical_raw_slot[4];
  if constexpr (TargetShellClass == kPpssShellClass) {
    const bool first_pair_is_pp =
        batch.shell_angular[raw_shell[0]] == 1U && batch.shell_angular[raw_shell[1]] == 1U;
    const unsigned pair_begin = first_pair_is_pp ? 0U : 2U;
    const unsigned other_pair_begin = first_pair_is_pp ? 2U : 0U;
    canonical_raw_slot[0] = pair_begin;
    canonical_raw_slot[1] = pair_begin + 1U;
    canonical_raw_slot[2] = other_pair_begin;
    canonical_raw_slot[3] = other_pair_begin + 1U;
  } else {
    unsigned d_slot = 4U;
    for (unsigned slot = 0; slot < 4; ++slot) {
      if (batch.shell_angular[raw_shell[slot]] == 2U) d_slot = slot;
    }
    if (d_slot >= 4U) return;
    const unsigned pair_begin = d_slot < 2U ? 0U : 2U;
    const unsigned other_pair_begin = pair_begin == 0U ? 2U : 0U;
    canonical_raw_slot[0] = d_slot;
    canonical_raw_slot[1] = pair_begin + (d_slot == pair_begin ? 1U : 0U);
    canonical_raw_slot[2] = other_pair_begin;
    canonical_raw_slot[3] = other_pair_begin + 1U;
  }
  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]],
      raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]],
      raw_shell[canonical_raw_slot[3]],
  };
  const std::size_t canonical_pair[2] = {
      canonical_raw_slot[0] < 2U ? first_pair : second_pair,
      canonical_raw_slot[2] < 2U ? first_pair : second_pair,
  };

  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    const std::int32_t atom = batch.shell_atoms[canonical_shell[center]];
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center || atom == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = atom;
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_component_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[0]]) - system_ao_begin;
  const std::size_t second_component_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[1]]) - system_ao_begin;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, raw_ao[2], raw_ao[3]);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    const double density_coefficient = direct_force_density_coefficient<Unrestricted>(
        n, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3]);
    if (density_coefficient == 0.0) continue;
    const unsigned first_component =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[0]] - first_component_begin);
    unsigned output = first_component;
    if constexpr (TargetShellClass == kPpssShellClass) {
      const unsigned second_component =
          static_cast<unsigned>(raw_ao[canonical_raw_slot[1]] - second_component_begin);
      if (first_component >= 3U || second_component >= 3U) return;
      output = first_component * 3U + second_component;
    } else if (first_component >= 6U) {
      return;
    }
    const double angular_coefficient = batch.direct_ao_coefficients[system_ao_begin + raw_ao[0]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[1]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[2]] *
                                       batch.direct_ao_coefficients[system_ao_begin + raw_ao[3]];
    component_weight[output] += density_coefficient * angular_coefficient;
    any_component = true;
  }
  if (!any_component) return;

  PspsWeightedGradient gradient{};
  if constexpr (TargetShellClass == kPpssShellClass) {
    gradient = contracted_eri_cartesian_source_ppss_weighted_gradient(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  } else {
    gradient = contracted_eri_cartesian_source_dsss_weighted_gradient(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  }

  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned canonical = 0; canonical < 3; ++canonical) {
        const double value = gradient.center[canonical][axis];
        fourth_derivative -= value;
        if (batch.shell_atoms[canonical_shell[canonical]] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (batch.shell_atoms[canonical_shell[3]] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

template <bool Unrestricted, unsigned AngularOrder>
__device__ __forceinline__ void contract_two_electron_force_quartet_subtile(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask, std::size_t active_subtile,
    unsigned ao_quartet_lane) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  constexpr std::size_t subtiles_per_tile = detail::direct_quartet_subtiles_per_tile(AngularOrder);
  const std::size_t active_tile = active_subtile / subtiles_per_tile;
  // Consume the identical compact tile list as direct Fock so energy and
  // derivative screening cover precisely the same AO-quartet domain.
  if (active_tile >= static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const std::size_t subtile = active_subtile % subtiles_per_tile;
  const ActiveShellQuartetTile task = active_shell_quartet_tiles[active_tile];
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_ao_pair_count = shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count = shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
                                           ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                                           : first_ao_pair_count * second_ao_pair_count;
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[first_shell], batch.shell_atoms[second_shell],
      batch.shell_atoms[third_shell], batch.shell_atoms[fourth_shell]};
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  // Generated consumers contract the complete exact class independently.
  // The host-selected bit mask keeps the generic fallback active for classes
  // disabled during production bisection.
  if (shell_class < 64U && (generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    return;
  }

  const std::size_t ordinal = static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize +
                              subtile * detail::kDirectQuartetThreads + ao_quartet_lane;
  if (ordinal < ao_quartet_count) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    std::size_t l = 0;
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin, i, j);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin, k, l);
    if (schwarz_bounds[physical_offset + matrix_index(i, j, n)] *
            schwarz_bounds[physical_offset + matrix_index(k, l, n)] <
        screening_tolerance) {
      return;
    }

    const double coefficient = direct_force_density_coefficient<Unrestricted>(
        n, physical_offset, spin_offset, density, i, j, k, l);
    if (coefficient == 0.0) return;

    // An ERI is invariant when all four basis centers translate together, so
    // its derivatives over the unique participating atoms sum to zero. Build
    // that unique list, evaluate only N-1 centers, and recover the final one
    // from the negative sum. This halves two-center work and removes one third
    // of three-center work without changing the analytic-gradient contract.
    std::int32_t unique_center_atoms[4];
    unsigned unique_center_count = 0;
    for (unsigned center = 0; center < 4; ++center) {
      bool duplicate_center = false;
      for (unsigned previous = 0; previous < unique_center_count; ++previous) {
        duplicate_center =
            duplicate_center || center_atoms[center] == unique_center_atoms[previous];
      }
      if (!duplicate_center) {
        unique_center_atoms[unique_center_count++] = center_atoms[center];
      }
    }
    double explicit_unique_gradient[4][3]{};
    if constexpr (AngularOrder <= 6) {
      CartesianQuartetGradient explicit_gradient{};
      if constexpr (AngularOrder <= 1) {
        explicit_gradient = contracted_eri_cartesian_source_order01_gradient<AngularOrder>(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 2) {
        explicit_gradient = contracted_eri_cartesian_source_order2_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 3) {
        explicit_gradient = contracted_eri_cartesian_source_order3_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 4) {
        explicit_gradient = contracted_eri_cartesian_source_order4_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 5) {
        explicit_gradient = contracted_eri_cartesian_source_order5_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      } else {
        explicit_gradient = contracted_eri_cartesian_source_order6_gradient(
            batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
            static_cast<std::int32_t>(k), static_cast<std::int32_t>(l));
      }
      for (unsigned center = 0; center < 4; ++center) {
        unsigned unique_center = 0;
        while (unique_center_atoms[unique_center] != center_atoms[center]) {
          ++unique_center;
        }
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          explicit_unique_gradient[unique_center][coordinate] +=
              explicit_gradient.center[center][coordinate];
        }
      }
    }
    double derivative_sum_x = 0.0;
    double derivative_sum_y = 0.0;
    double derivative_sum_z = 0.0;
    for (unsigned center = 0; center + 1 < unique_center_count; ++center) {
      const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[center]) * 3;
      double derivative_x = 0.0;
      double derivative_y = 0.0;
      double derivative_z = 0.0;
      if constexpr (AngularOrder <= 6) {
        derivative_x = explicit_unique_gradient[center][0];
        derivative_y = explicit_unique_gradient[center][1];
        derivative_z = explicit_unique_gradient[center][2];
      } else {
        const Dual3 derivative =
            dispatch_contracted_eri_cartesian_source_shell_class<AngularOrder, Dual3>(
                shell_class, batch, system, static_cast<std::int32_t>(i),
                static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
                static_cast<std::int32_t>(l), coordinate);
        derivative_x = derivative.derivative_x;
        derivative_y = derivative.derivative_y;
        derivative_z = derivative.derivative_z;
      }
      derivative_sum_x += derivative_x;
      derivative_sum_y += derivative_y;
      derivative_sum_z += derivative_z;
      if (derivative_x != 0.0) {
        atomicAdd(forces + coordinate, -coefficient * derivative_x);
      }
      if (derivative_y != 0.0) {
        atomicAdd(forces + coordinate + 1, -coefficient * derivative_y);
      }
      if (derivative_z != 0.0) {
        atomicAdd(forces + coordinate + 2, -coefficient * derivative_z);
      }
    }
    if (unique_center_count > 1) {
      const std::int64_t final_coordinate =
          static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
      if (derivative_sum_x != 0.0) {
        atomicAdd(forces + final_coordinate, coefficient * derivative_sum_x);
      }
      if (derivative_sum_y != 0.0) {
        atomicAdd(forces + final_coordinate + 1, coefficient * derivative_sum_y);
      }
      if (derivative_sum_z != 0.0) {
        atomicAdd(forces + final_coordinate + 2, coefficient * derivative_sum_z);
      }
    }
  }
}

/** Runtime angular dispatch used only by the generic large-topology path. */
template <bool Unrestricted>
__device__ __noinline__ void contract_bounded_direct_fock_subtile(
    DeviceBatch batch, unsigned angular_order, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock, std::size_t subtile,
    unsigned lane) {
#define VIBEQC_BOUNDED_FOCK_CASE(order)                                                       \
  case order:                                                                                 \
    contract_fock_direct_quartet_subtile<Unrestricted, order>(                                \
        batch, queue_count, task, screening_tolerance, schwarz_bounds, density, active, fock, \
        nullptr, subtile, lane);                                                              \
    break
  switch (angular_order) {
    VIBEQC_BOUNDED_FOCK_CASE(0);
    VIBEQC_BOUNDED_FOCK_CASE(1);
    VIBEQC_BOUNDED_FOCK_CASE(2);
    VIBEQC_BOUNDED_FOCK_CASE(3);
    VIBEQC_BOUNDED_FOCK_CASE(4);
    VIBEQC_BOUNDED_FOCK_CASE(5);
    VIBEQC_BOUNDED_FOCK_CASE(6);
    VIBEQC_BOUNDED_FOCK_CASE(7);
    VIBEQC_BOUNDED_FOCK_CASE(8);
    VIBEQC_BOUNDED_FOCK_CASE(9);
    VIBEQC_BOUNDED_FOCK_CASE(10);
    VIBEQC_BOUNDED_FOCK_CASE(11);
    VIBEQC_BOUNDED_FOCK_CASE(12);
    default:
      break;
  }
#undef VIBEQC_BOUNDED_FOCK_CASE
}

template <bool Unrestricted>
__device__ __noinline__ void contract_bounded_direct_force_subtile(
    DeviceBatch batch, unsigned angular_order, const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* forces, std::size_t subtile,
    unsigned lane) {
#define VIBEQC_BOUNDED_FORCE_CASE(order)                                                        \
  case order:                                                                                   \
    contract_two_electron_force_quartet_subtile<Unrestricted, order>(                           \
        batch, queue_count, task, screening_tolerance, schwarz_bounds, density, active, forces, \
        0U, subtile, lane);                                                                     \
    break
  switch (angular_order) {
    VIBEQC_BOUNDED_FORCE_CASE(0);
    VIBEQC_BOUNDED_FORCE_CASE(1);
    VIBEQC_BOUNDED_FORCE_CASE(2);
    VIBEQC_BOUNDED_FORCE_CASE(3);
    VIBEQC_BOUNDED_FORCE_CASE(4);
    VIBEQC_BOUNDED_FORCE_CASE(5);
    VIBEQC_BOUNDED_FORCE_CASE(6);
    VIBEQC_BOUNDED_FORCE_CASE(7);
    VIBEQC_BOUNDED_FORCE_CASE(8);
    VIBEQC_BOUNDED_FORCE_CASE(9);
    VIBEQC_BOUNDED_FORCE_CASE(10);
    VIBEQC_BOUNDED_FORCE_CASE(11);
    VIBEQC_BOUNDED_FORCE_CASE(12);
    default:
      break;
  }
#undef VIBEQC_BOUNDED_FORCE_CASE
}

/**
 * Stream only canonical dddd work from class-major shell-pair segments.
 *
 * This is an exact class-specific route, not the bounded generic fallback:
 * its outer domain is the dd-pair triangle and every accepted quartet is
 * consumed immediately.  It replaces the currently unreliable generated
 * dddd value/gradient consumer while retaining O(N_shell^2) topology storage
 * and zero whole-topology scan when all production classes are covered.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose, bool Force>
__global__
__launch_bounds__(detail::kDirectQuartetThreads) void bounded_direct_dddd_streaming_kernel(
    DeviceBatch batch, const GeneratedShellPairStream* topology_pointer, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* output,
    std::uint32_t* bra_head, DeviceShellClassProfileEntry* profile,
    unsigned long long* fp64_work_count) {
  static_assert(detail::kDirectQuartetThreads == 32);
  constexpr std::uint32_t kSkip = 0U;
  constexpr std::uint32_t kConsume = 1U;
  constexpr std::uint32_t kFinished = 2U;
  constexpr std::size_t kSubtilesPerTile =
      detail::direct_quartet_subtiles_per_tile(kDdddAngularOrder);

  __shared__ ActiveShellQuartetTile task;
  __shared__ std::uint32_t queue_count;
  __shared__ std::uint32_t candidate_ordinal;
  __shared__ std::uint32_t stream_state;
  __shared__ std::uint32_t tile_count;

  const unsigned lane = threadIdx.x;
  const GeneratedShellPairStream& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::size_t dd_pair_class = 5U;
  const auto* density_bounds =
      reinterpret_cast<const ShellPairDensityBounds*>(topology.shell_pair_density_bounds);

  if (lane == 0U) queue_count = 1U;
  __syncwarp();
  while (true) {
    if (lane == 0U) {
      candidate_ordinal = atomicAdd(bra_head, 1U);
      std::uint64_t remaining = candidate_ordinal;
      stream_state = kFinished;
      for (std::int32_t system = 0; system < topology.batch_size; ++system) {
        const std::uint32_t pair_begin =
            topology.pair_class_offsets[dd_pair_class * stride + static_cast<std::size_t>(system)];
        const std::uint32_t pair_end =
            topology
                .pair_class_offsets[dd_pair_class * stride + static_cast<std::size_t>(system) + 1U];
        const std::uint64_t pair_count = pair_end - pair_begin;
        const std::uint64_t system_candidates = pair_count * (pair_count + 1U) / 2U;
        if (remaining >= system_candidates) {
          remaining -= system_candidates;
          continue;
        }

        std::size_t bra_local = 0U;
        std::size_t ket_local = 0U;
        decode_lower_triangle(static_cast<std::size_t>(remaining), bra_local, ket_local);
        const std::uint32_t bra_pair = topology.pair_order[pair_begin + bra_local];
        const std::uint32_t ket_pair = topology.pair_order[pair_begin + ket_local];
        bool keep = (active == nullptr || active[system] != 0U) &&
                    topology.shell_pair_bounds[bra_pair] * topology.shell_pair_bounds[ket_pair] >=
                        screening_tolerance;
        if (keep) {
          keep = direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
              batch, bra_pair, ket_pair, screening_tolerance, topology.shell_pair_bounds,
              density_bounds);
        }
        stream_state = keep ? kConsume : kSkip;
        if (keep) {
          if (fp64_work_count != nullptr) atomicAdd(fp64_work_count, 1ULL);
          task = {bra_pair, ket_pair, 0U};
          const std::size_t first_count = shell_ao_pair_count(batch, bra_pair);
          const std::size_t second_count = shell_ao_pair_count(batch, ket_pair);
          const std::size_t ao_quartets = bra_pair == ket_pair
                                              ? first_count * (first_count + 1U) / 2U
                                              : first_count * second_count;
          tile_count = static_cast<std::uint32_t>(
              (ao_quartets + detail::kDirectQuartetTileSize - 1U) / detail::kDirectQuartetTileSize);
        }
        break;
      }
    }
    __syncwarp();
    if (stream_state == kFinished) return;
    if (stream_state != kConsume) continue;

    if constexpr (Force) {
      if (lane == 0U) {
        profile_bounded_direct_shell_quartet(batch, task, profile);
      }
      __syncwarp();
    }
    for (std::uint32_t tile = 0U; tile < tile_count; ++tile) {
      if (lane == 0U) task.tile = tile;
      __syncwarp();
      for (std::size_t subtile = 0U; subtile < kSubtilesPerTile; ++subtile) {
        if constexpr (Force) {
          contract_two_electron_force_quartet_subtile<Unrestricted, kDdddAngularOrder>(
              batch, &queue_count, &task, screening_tolerance, schwarz_bounds, density, active,
              output, 0U, subtile, lane);
        } else {
          contract_fock_direct_quartet_subtile<Unrestricted, kDdddAngularOrder>(
              batch, &queue_count, &task, screening_tolerance, schwarz_bounds, density, active,
              output, nullptr, subtile, lane);
        }
      }
      __syncwarp();
    }
  }
}

/**
 * Consume one paged exact class through the handwritten low-order force path.
 *
 * The production AOT bundle deliberately does not advertise generated
 * ``ssss``/``psss`` force consumers; their validated implementations are the
 * scalar handwritten contractions below.  Bounded streaming still needs to
 * avoid the topology-wide fallback, so enumerate only the page's exact
 * shell-pair rectangle and contract each surviving low-order quartet in place.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose>
__global__ void contract_bounded_exact_low_order_force_page_kernel(
    DeviceBatch batch, const GeneratedShellPairStream* topology_pointer, unsigned shell_class,
    unsigned high_pair_class, unsigned low_pair_class, double screening_tolerance,
    std::uint64_t page_begin, std::uint32_t page_capacity, std::uint32_t bra_ordinal_begin,
    std::uint32_t bra_ordinal_end, bool same_pair_class, const double* schwarz_bounds,
    const double* density, double* forces, std::uint32_t* bra_head) {
  __shared__ std::uint32_t bra_ordinal;
  if (shell_class != kSsssShellClass && shell_class != kPsssShellClass) {
    return;
  }
  const GeneratedShellPairStream& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::uint32_t bra_begin = topology.pair_class_offsets[high_pair_class * stride];
  const auto* density_bounds =
      reinterpret_cast<const ShellPairDensityBounds*>(topology.shell_pair_density_bounds);

  while (true) {
    if (threadIdx.x == 0U) {
      bra_ordinal = bra_ordinal_begin + atomicAdd(bra_head, 1U);
    }
    __syncthreads();
    if (bra_ordinal >= bra_ordinal_end) return;

    const std::uint32_t bra_pair = topology.pair_order[bra_begin + bra_ordinal];
    const std::int32_t system = topology.shell_pair_systems[bra_pair];
    if (topology.active != nullptr && topology.active[system] == 0U) {
      continue;
    }
    const std::uint32_t ket_begin =
        topology.pair_class_offsets[low_pair_class * stride + static_cast<std::size_t>(system)];
    const std::uint32_t ket_end =
        topology
            .pair_class_offsets[low_pair_class * stride + static_cast<std::size_t>(system) + 1U];
    const std::uint32_t system_bra_begin =
        topology.pair_class_offsets[high_pair_class * stride + static_cast<std::size_t>(system)];
    const std::uint64_t bra_local = bra_begin + bra_ordinal - system_bra_begin;
    const std::uint64_t ket_count = ket_end - ket_begin;
    std::uint64_t system_candidate_begin = 0U;
    for (std::int32_t previous = 0; previous < system; ++previous) {
      const std::uint32_t previous_bra_begin =
          topology
              .pair_class_offsets[high_pair_class * stride + static_cast<std::size_t>(previous)];
      const std::uint32_t previous_bra_end =
          topology.pair_class_offsets[high_pair_class * stride +
                                      static_cast<std::size_t>(previous) + 1U];
      const std::uint32_t previous_ket_begin =
          topology.pair_class_offsets[low_pair_class * stride + static_cast<std::size_t>(previous)];
      const std::uint32_t previous_ket_end =
          topology.pair_class_offsets[low_pair_class * stride + static_cast<std::size_t>(previous) +
                                      1U];
      const std::uint64_t previous_bra_count = previous_bra_end - previous_bra_begin;
      const std::uint64_t previous_ket_count = previous_ket_end - previous_ket_begin;
      system_candidate_begin += same_pair_class
                                    ? previous_bra_count * (previous_bra_count + 1U) / 2U
                                    : previous_bra_count * previous_ket_count;
    }
    const std::uint64_t bra_candidate_begin =
        system_candidate_begin +
        (same_pair_class ? bra_local * (bra_local + 1U) / 2U : bra_local * ket_count);
    const std::uint64_t row_candidate_count = same_pair_class ? bra_local + 1U : ket_count;
    const std::uint64_t bra_candidate_end = bra_candidate_begin + row_candidate_count;
    const std::uint64_t page_end = page_begin + page_capacity;
    if (bra_candidate_end <= page_begin) continue;
    if (bra_candidate_begin >= page_end) return;
    const std::uint64_t first_page_offset =
        page_begin > bra_candidate_begin ? page_begin - bra_candidate_begin : 0U;
    const std::uint64_t last_page_offset =
        page_end < bra_candidate_end ? page_end - bra_candidate_begin : row_candidate_count;
    const std::uint32_t ket_first = ket_begin + static_cast<std::uint32_t>(first_page_offset);
    const std::uint32_t ket_last = ket_begin + static_cast<std::uint32_t>(last_page_offset);
    const BoundedPageDensityTails page_density_tails = bounded_page_density_tails(
        batch, topology, system, bra_pair, high_pair_class, low_pair_class);
    const bool has_density_bound =
        topology.system_pair_density_bounds != nullptr || topology.system_density_bounds != nullptr;
    for (std::uint32_t ket_ordinal = ket_first + threadIdx.x; ket_ordinal < ket_last;
         ket_ordinal += blockDim.x) {
      const std::uint32_t ket_pair = topology.pair_order[ket_ordinal];
      const double quartet_bound =
          topology.shell_pair_bounds[bra_pair] * topology.shell_pair_bounds[ket_pair];
      if constexpr (Purpose == DirectScreeningPurpose::Force) {
        const double force_tolerance =
            fmin(screening_tolerance, kForceDensityProductScreeningTolerance);
        if (has_density_bound && quartet_bound * page_density_tails.force < force_tolerance) {
          continue;
        }
      } else if (has_density_bound &&
                 quartet_bound * page_density_tails.fock < screening_tolerance) {
        continue;
      }
      if (!direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
              batch, bra_pair, ket_pair, screening_tolerance, topology.shell_pair_bounds,
              density_bounds)) {
        continue;
      }
      const ActiveShellQuartetTile task{bra_pair, ket_pair, 0U};
      if (shell_class == kSsssShellClass) {
        contract_two_electron_force_ssss_task<Unrestricted>(
            batch, task, screening_tolerance, schwarz_bounds, density, topology.active, forces, 0U);
      } else {
        contract_two_electron_force_psss_task<Unrestricted>(
            batch, task, screening_tolerance, schwarz_bounds, density, topology.active, forces, 0U);
      }
    }
    __syncthreads();
  }
}

/**
 * Enumerate, screen, queue, and drain shell pair-of-pairs hierarchically.
 *
 * A persistent CTA first claims one pair-block product. Conservative Schwarz
 * and density maxima reject the complete block without visiting its members;
 * surviving blocks are expanded in fixed 256-candidate chunks and retain the
 * exact shell-quartet predicate. This keeps storage bounded while replacing
 * the former unconditional O(N_shell^4) scan with a small O(N_shell^4/B^2)
 * outer domain plus exact work only in surviving blocks.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose, bool Force>
__global__ __launch_bounds__(kBoundedDirectThreads, 1) void bounded_direct_shell_quartet_kernel(
    DeviceBatch batch, double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds, const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer, std::uint64_t enabled_mask,
    const std::uint32_t* bounded_generated_overflow, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* output,
    unsigned long long* global_cursor, DeviceShellClassProfileEntry* profile) {
  __shared__ ActiveShellQuartetTile queue[detail::kBoundedDirectQueueCapacity];
  __shared__ std::uint32_t queue_count;
  __shared__ unsigned long long block_quartet;
  const unsigned lane = threadIdx.x % detail::kDirectQuartetThreads;
  const unsigned warp = threadIdx.x / detail::kDirectQuartetThreads;
  const std::size_t total = static_cast<std::size_t>(batch.total_shell_pair_block_quartets);

  while (true) {
    if (threadIdx.x == 0) {
      block_quartet = atomicAdd(global_cursor, 1ULL);
    }
    __syncthreads();
    if (block_quartet >= total) return;

    const std::size_t packed_block_quartet = static_cast<std::size_t>(block_quartet);
    const std::int32_t system = shell_pair_block_quartet_system(batch, packed_block_quartet);
    if (active != nullptr && active[system] == 0) continue;
    const std::size_t local_block_quartet =
        packed_block_quartet -
        static_cast<std::size_t>(batch.system_shell_pair_block_quartet_offsets[system]);
    std::size_t first_block_local = 0;
    std::size_t second_block_local = 0;
    decode_lower_triangle(local_block_quartet, first_block_local, second_block_local);
    const std::size_t system_block_begin =
        static_cast<std::size_t>(batch.system_shell_pair_block_offsets[system]);
    const std::size_t first_block = system_block_begin + first_block_local;
    const std::size_t second_block = system_block_begin + second_block_local;
    if (!bounded_direct_block_pair_survives_screening<Purpose>(
            first_block, second_block, system, screening_tolerance, shell_pair_block_bounds,
            system_density_bounds)) {
      continue;
    }

    const std::size_t system_pair_begin =
        static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
    const std::size_t system_pair_end =
        static_cast<std::size_t>(batch.system_shell_pair_offsets[system + 1]);
    const std::size_t first_ordered_begin =
        system_pair_begin + first_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t second_ordered_begin =
        system_pair_begin + second_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t first_count =
        min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - first_ordered_begin);
    const std::size_t second_count =
        min(detail::kBoundedDirectShellPairBlockSize, system_pair_end - second_ordered_begin);
    const bool same_block = first_block == second_block;
    const std::size_t candidate_count =
        same_block ? first_count * (first_count + 1) / 2 : first_count * second_count;

    for (std::size_t candidate_begin = 0; candidate_begin < candidate_count;
         candidate_begin += detail::kBoundedDirectQueueCapacity) {
      if (threadIdx.x == 0) queue_count = 0;
      __syncthreads();
      const std::size_t candidate = candidate_begin + threadIdx.x;
      if (candidate < candidate_count) {
        std::size_t first_local = 0;
        std::size_t second_local = 0;
        if (same_block) {
          decode_lower_triangle(candidate, first_local, second_local);
        } else {
          first_local = candidate / second_count;
          second_local = candidate % second_count;
        }
        const std::size_t first_pair = shell_pair_order[first_ordered_begin + first_local];
        const std::size_t second_pair = shell_pair_order[second_ordered_begin + second_local];
        if (direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
                batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
                shell_pair_density_bounds)) {
          const std::int32_t first_shell = batch.shell_pair_first[first_pair];
          const std::int32_t second_shell = batch.shell_pair_second[first_pair];
          const std::int32_t third_shell = batch.shell_pair_first[second_pair];
          const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
          const unsigned shell_class = direct_quartet_shell_class_device(
              batch.shell_angular[first_shell], batch.shell_angular[second_shell],
              batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
          const bool generated_class =
              bounded_generated_overflow[shell_class] == 0U &&
              bounded_generated_class_enabled(shell_class, enabled_mask_pointer, enabled_mask);
          if (!generated_class) {
            const std::uint32_t slot = atomicAdd(&queue_count, 1U);
            queue[slot] = {static_cast<std::uint32_t>(first_pair),
                           static_cast<std::uint32_t>(second_pair), 0U};
            if constexpr (Force) {
              profile_bounded_direct_shell_quartet(batch, queue[slot], profile);
            }
          }
        }
      }
      __syncthreads();

      // Low-order shell tasks fit in one scalar lane. Drain up to 256 of
      // them concurrently before assigning the larger classes one warp each;
      // the former generic path spent 31 idle lanes on every ssss/psss/order2
      // task and dominates molecular systems built from s/p/d basis shells.
      for (std::uint32_t slot = threadIdx.x; slot < queue_count; slot += blockDim.x) {
        const ActiveShellQuartetTile task = queue[slot];
        const std::int32_t first_shell = batch.shell_pair_first[task.first_pair];
        const std::int32_t second_shell = batch.shell_pair_second[task.first_pair];
        const std::int32_t third_shell = batch.shell_pair_first[task.second_pair];
        const std::int32_t fourth_shell = batch.shell_pair_second[task.second_pair];
        const unsigned angular_order =
            batch.shell_angular[first_shell] + batch.shell_angular[second_shell] +
            batch.shell_angular[third_shell] + batch.shell_angular[fourth_shell];
        if constexpr (Force) {
          if (angular_order == 0U) {
            contract_two_electron_force_ssss_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
          } else if (angular_order == 1U) {
            contract_two_electron_force_psss_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
          } else if (angular_order == 2U) {
            contract_two_electron_force_psps_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
            contract_two_electron_force_pair_order2_task<Unrestricted, kPpssShellClass>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
            contract_two_electron_force_pair_order2_task<Unrestricted, kDsssShellClass>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, 0U);
          }
        } else {
          if (angular_order == 0U) {
            contract_fock_direct_quartet_subtile<Unrestricted, 0U>(
                batch, &queue_count, queue + slot, screening_tolerance, schwarz_bounds, density,
                active, output, nullptr, 0U, 0U);
          } else if (angular_order == 1U) {
            contract_fock_direct_psss_task<Unrestricted>(batch, task, screening_tolerance,
                                                         schwarz_bounds, density, active, output);
          } else if (angular_order == 2U) {
            contract_fock_direct_order2_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, output, nullptr);
          }
        }
      }
      __syncthreads();

      for (std::uint32_t slot = warp; slot < queue_count;
           slot += kBoundedDirectThreads / detail::kDirectQuartetThreads) {
        const ActiveShellQuartetTile base = queue[slot];
        const std::int32_t first_shell = batch.shell_pair_first[base.first_pair];
        const std::int32_t second_shell = batch.shell_pair_second[base.first_pair];
        const std::int32_t third_shell = batch.shell_pair_first[base.second_pair];
        const std::int32_t fourth_shell = batch.shell_pair_second[base.second_pair];
        const unsigned angular_order =
            batch.shell_angular[first_shell] + batch.shell_angular[second_shell] +
            batch.shell_angular[third_shell] + batch.shell_angular[fourth_shell];
        if (angular_order <= 2U) continue;
        const std::size_t first_ao_count = shell_ao_pair_count(batch, base.first_pair);
        const std::size_t second_ao_count = shell_ao_pair_count(batch, base.second_pair);
        const std::size_t ao_quartets = base.first_pair == base.second_pair
                                            ? first_ao_count * (first_ao_count + 1) / 2
                                            : first_ao_count * second_ao_count;
        const std::uint32_t tile_count = static_cast<std::uint32_t>(
            (ao_quartets + detail::kDirectQuartetTileSize - 1) / detail::kDirectQuartetTileSize);
        const std::size_t subtile_count = detail::direct_quartet_subtiles_per_tile(angular_order);
        for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
          if (lane == 0) queue[slot].tile = tile;
          __syncwarp();
          for (std::size_t subtile = 0; subtile < subtile_count; ++subtile) {
            if constexpr (Force) {
              contract_bounded_direct_force_subtile<Unrestricted>(
                  batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                  schwarz_bounds, density, active, output, subtile, lane);
            } else {
              contract_bounded_direct_fock_subtile<Unrestricted>(
                  batch, angular_order, &queue_count, queue + slot, screening_tolerance,
                  schwarz_bounds, density, active, output, subtile, lane);
            }
          }
          __syncwarp();
        }
      }
      __syncthreads();
    }
  }
}

/** Fixed-capacity wrapper for the small generic high-order force grids. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  contract_two_electron_force_quartet_subtile<Unrestricted, AngularOrder>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
      schwarz_bounds, density, active, forces, generated_shell_class_mask,
      static_cast<std::size_t>(blockIdx.x), threadIdx.x);
}

/** Pack independent ssss derivative shell tasks across one worker warp. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_packed_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask) {
  static_assert(AngularOrder < kPackedSsssAngularOrderCount);
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_two_electron_force_ssss_task<Unrestricted>(
          batch, active_shell_quartet_tiles[packed_item], screening_tolerance, schwarz_bounds,
          density, active, forces, generated_shell_class_mask);
    }
  }
}

/**
 * Keep one p-s bra resident while block threads traverse all s-s ket pairs.
 *
 * Total angular order one contains only psss quartets, so enumerating each
 * p-s shell pair once and each s-s shell pair in its system once preserves
 * unique quartet ownership without consulting the unordered compact queue.
 * The shared primitive-pair records remove the remaining repeated bra loads
 * across the hundreds of ket tasks normally associated with one 192-AO bra.
 */
template <bool Unrestricted>
// Four resident 128-thread blocks cap this register-heavy contraction at
// 128 registers/thread on sm_120.  The extra occupancy hides the long
// primitive-pair dependency chain without changing the resident-bra schedule.
__global__
__launch_bounds__(kResidentPsssThreads, 4) void two_electron_force_psss_resident_bra_kernel(
    DeviceBatch batch, const PsssResidentTask* resident_tasks,
    const std::uint32_t* resident_ket_pairs, std::size_t resident_task_count,
    double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds, bool force_density_product_screening,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  extern __shared__ PrimitivePairData resident_first_pairs[];
  const std::size_t task_index = static_cast<std::size_t>(blockIdx.x);
  if (task_index >= resident_task_count) return;
  const PsssResidentTask task = resident_tasks[task_index];
  const std::size_t bra_pair = task.bra_pair;

  const std::int32_t system = batch.shell_pair_systems[bra_pair];
  if (active[system] == 0) return;
  const std::int32_t bra_first_shell = batch.shell_pair_first[bra_pair];
  const std::int32_t bra_second_shell = batch.shell_pair_second[bra_pair];
  const unsigned bra_first_angular = batch.shell_angular[bra_first_shell];
  const unsigned bra_second_angular = batch.shell_angular[bra_second_shell];
  if (bra_first_angular + bra_second_angular != 1U) return;

  const std::int64_t bra_primitive_begin = batch.shell_pair_primitive_offsets[bra_pair];
  const std::int64_t bra_primitive_count =
      batch.shell_pair_primitive_offsets[bra_pair + 1U] - bra_primitive_begin;
  if (bra_primitive_count <= 0 ||
      bra_primitive_count > static_cast<std::int64_t>(kResidentPsssMaximumBraPrimitivePairs))
    return;
  for (std::int64_t primitive = threadIdx.x; primitive < bra_primitive_count;
       primitive += blockDim.x) {
    resident_first_pairs[primitive] = batch.shell_primitive_pairs[bra_primitive_begin + primitive];
  }
  __syncthreads();

  for (std::size_t local_ket = threadIdx.x; local_ket < task.ket_count; local_ket += blockDim.x) {
    const std::size_t ket_pair = resident_ket_pairs[task.ket_begin + local_ket];
    const std::size_t first_pair = bra_pair > ket_pair ? bra_pair : ket_pair;
    const std::size_t second_pair = bra_pair > ket_pair ? ket_pair : bra_pair;
    const bool survives_screening =
        force_density_product_screening
            ? direct_shell_quartet_survives_screening<Unrestricted, DirectScreeningPurpose::Force>(
                  batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
                  shell_pair_density_bounds)
            : direct_shell_quartet_survives_screening<Unrestricted, DirectScreeningPurpose::Fock>(
                  batch, first_pair, second_pair, screening_tolerance, shell_pair_bounds,
                  shell_pair_density_bounds);
    if (!survives_screening) {
      continue;
    }
    contract_two_electron_force_psss_task<Unrestricted, true>(
        batch,
        {static_cast<std::uint32_t>(first_pair), static_cast<std::uint32_t>(second_pair), 0U},
        screening_tolerance, schwarz_bounds, density, active, forces, generated_shell_class_mask,
        resident_first_pairs, bra_primitive_count);
  }
}

/** Consume complete density-weighted psss force tasks, one task per lane. */
template <bool Unrestricted>
__global__ void two_electron_force_psss_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask) {
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_two_electron_force_psss_task<Unrestricted>(
          batch, active_shell_quartet_tiles[packed_item], screening_tolerance, schwarz_bounds,
          density, active, forces, generated_shell_class_mask);
    }
    // Keep tail lanes live through the next full-mask queue broadcast.
  }
}

/** Scan compact order-two tiles and consume complete psps shell tasks. */
template <bool Unrestricted>
__global__ void two_electron_force_psps_grid_stride_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t task_index = blockIdx.x * blockDim.x + threadIdx.x; task_index < work_count;
       task_index += stride) {
    contract_two_electron_force_psps_task<Unrestricted>(
        batch, active_shell_quartet_tiles[task_index], screening_tolerance, schwarz_bounds, density,
        active, forces, generated_shell_class_mask);
  }
}

/** Scan compact order-two tiles for one exact ppss or dsss class. */
template <bool Unrestricted, unsigned TargetShellClass>
__global__ void two_electron_force_pair_order2_grid_stride_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  static_assert(TargetShellClass == kPpssShellClass || TargetShellClass == kDsssShellClass);
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t task_index = blockIdx.x * blockDim.x + threadIdx.x; task_index < work_count;
       task_index += stride) {
    contract_two_electron_force_pair_order2_task<Unrestricted, TargetShellClass>(
        batch, active_shell_quartet_tiles[task_index], screening_tolerance, schwarz_bounds, density,
        active, forces, generated_shell_class_mask);
  }
}

/**
 * Persistent one-warp workers dynamically consume only compacted force work.
 *
 * The topology-capacity launch remains useful for CUDA Graph Fock replay, but
 * the final force executes outside that iterative Graph. A device task head
 * therefore removes empty capacity blocks and balances irregular AO-quartet
 * derivative cost without introducing a host readback of compacted counts.
 */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  const unsigned lane = threadIdx.x % warpSize;
  constexpr std::uint32_t subtiles_per_tile =
      static_cast<std::uint32_t>(detail::direct_quartet_subtiles_per_tile(AngularOrder));
  const std::uint32_t work_count = *active_shell_quartet_tile_count * subtiles_per_tile;
  while (true) {
    std::uint32_t active_subtile = 0;
    if (lane == 0) active_subtile = atomicAdd(task_head, 1U);
    active_subtile = __shfl_sync(0xffffffffU, active_subtile, 0);
    if (active_subtile >= work_count) return;
    contract_two_electron_force_quartet_subtile<Unrestricted, AngularOrder>(
        batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
        schwarz_bounds, density, active, forces, generated_shell_class_mask, active_subtile,
        threadIdx.x);
  }
}

/** Prepare compact exact-class slices shared by generated Fock and force code. */
cudaError_t prepare_generated_shell_tasks(
    cudaStream_t stream, std::size_t total_tile_capacity, std::size_t generated_task_capacity,
    const std::uint32_t* active_tile_offsets, DeviceBatch batch,
    const std::uint32_t* active_tile_counts, const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks, std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets, std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts, std::uint32_t* generated_task_heads,
    std::uint64_t low_order_signature_mask, std::uint32_t* low_order_signature_counts,
    std::uint32_t* low_order_signature_offsets, std::uint64_t enabled_mask,
    const std::uint64_t* enabled_mask_pointer, bool exclude_resident_ppps) {
  if ((enabled_mask == 0U && enabled_mask_pointer == nullptr) || generated_task_capacity == 0 ||
      total_tile_capacity == 0) {
    return cudaSuccess;
  }
  cudaError_t error =
      cudaMemsetAsync(generated_task_counts, 0,
                      detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), stream);
  if (error != cudaSuccess) return error;
  if (low_order_signature_counts != nullptr) {
    error = cudaMemsetAsync(low_order_signature_counts, 0,
                            kLowOrderSignatureElementCount * sizeof(std::uint32_t), stream);
    if (error != cudaSuccess) return error;
  }
  constexpr unsigned preparation_threads = kCaptureSafeKernelThreads;
  const unsigned preparation_blocks =
      static_cast<unsigned>((total_tile_capacity + preparation_threads - 1) / preparation_threads);
  launch_classify_generated_shell_tasks_kernel(
      preparation_blocks, preparation_threads, 0, stream, batch, total_tile_capacity,
      active_tile_offsets, active_tile_counts, active_tiles, enabled_mask, enabled_mask_pointer,
      exclude_resident_ppps, generated_task_counts, generated_shell_classes,
      low_order_signature_mask, low_order_signature_counts);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  launch_prefix_generated_shell_task_counts_kernel(
      1, 1, 0, stream, generated_task_counts, generated_task_offsets, generated_task_write_counts,
      generated_task_heads);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  if (low_order_signature_counts != nullptr && low_order_signature_offsets != nullptr) {
    launch_prefix_low_order_signature_counts_kernel(
        1, 1, 0, stream, generated_task_offsets, low_order_signature_mask,
        low_order_signature_counts, low_order_signature_offsets);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
  }
  launch_materialize_generated_shell_tasks_kernel(
      preparation_blocks, preparation_threads, 0, stream, batch, total_tile_capacity, active_tiles,
      generated_shell_classes, generated_task_offsets, generated_task_write_counts, generated_tasks,
      low_order_signature_mask, low_order_signature_offsets, low_order_signature_counts);
  return cudaPeekAtLastError();
}

/**
 * Bucket all enabled generated classes once, then launch their force slices.
 *
 * Counts, offsets, and worker heads remain device-resident, so adding an AOT
 * class does not add a host synchronization or another scan of every active
 * quartet. The generated persistent kernels apply their class offset when
 * loading tasks from the shared compact allocation.
 */
cudaError_t launch_generated_shell_class_forces(
    cudaStream_t stream, std::size_t total_tile_capacity, std::size_t generated_task_capacity,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::uint32_t* active_tile_offsets, DeviceBatch batch,
    const std::uint32_t* active_tile_counts, const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks, std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets, std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts, std::uint32_t* generated_task_heads,
    std::uint32_t* low_order_signature_counts, std::uint32_t* low_order_signature_offsets,
    GeneratedPppsResidentTask* resident_ppps_tasks, GeneratedShellTask* resident_ppps_ket_tasks,
    std::uint32_t* resident_ppps_bra_counts, std::uint32_t* resident_ppps_bra_offsets,
    std::uint32_t* resident_ppps_bra_write_counts, std::uint32_t* resident_ppps_signature_counts,
    std::uint32_t* resident_ppps_signature_offsets, std::uint32_t* resident_ppps_signatures,
    std::size_t total_shell_pairs, bool resident_ppps_enabled,
    bool resident_ppps_signature_bucketing, bool psps_signature_bucketing,
    bool ppss_signature_bucketing, unsigned resident_ppps_block_threads,
    unsigned persistent_worker_blocks, bool unrestricted, std::uint64_t enabled_mask,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    double* forces) {
  bool use_resident_ppps =
      resident_ppps_enabled && (enabled_mask & (std::uint64_t{1} << kPppsShellClass)) != 0U;
  if (total_tile_capacity == 0 || (enabled_mask == 0U && !use_resident_ppps) ||
      (generated_task_capacity == 0 && !use_resident_ppps)) {
    return cudaSuccess;
  }
  cudaError_t error = cudaSuccess;
  if (use_resident_ppps) {
    std::size_t ppps_tile_offset = 0;
    for (unsigned order = 0; order < kPppsAngularOrder; ++order) {
      ppps_tile_offset += capacities[order];
    }
    // ppps has total angular order three.  Restrict both grouping scans to
    // that fixed partition instead of rereading every active shell quartet.
    error = prepare_ppps_resident_tasks(
        stream, capacities[kPppsAngularOrder], total_shell_pairs, batch,
        active_tile_counts + kPppsAngularOrder, active_tiles + ppps_tile_offset,
        resident_ppps_tasks, resident_ppps_ket_tasks, resident_ppps_bra_counts,
        resident_ppps_bra_offsets, resident_ppps_bra_write_counts,
        resident_ppps_signature_bucketing ? resident_ppps_signature_counts : nullptr,
        resident_ppps_signature_bucketing ? resident_ppps_signature_offsets : nullptr,
        resident_ppps_signatures, enabled_mask);
    if (error != cudaSuccess) return error;
  }
  if (use_resident_ppps) {
    // Probe the selected AOT profile before excluding eligible ppps records
    // from the ordinary queue.  A portable profile may contain the ordinary
    // ppps class without its optional resident route; in that case fall back
    // losslessly instead of dropping the resident-eligible quartets.
    error = generated::launch_ppps_resident(
        stream, unrestricted, resident_ppps_tasks, resident_ppps_ket_tasks,
        batch.shell_pair_primitive_offsets, batch.shell_primitive_pairs,
        batch.direct_ao_coefficients, batch.positions, screening_tolerance, schwarz_bounds, density,
        forces, resident_ppps_block_threads, total_shell_pairs);
    if (error == cudaErrorNotSupported) {
      use_resident_ppps = false;
    } else if (error != cudaSuccess) {
      return error;
    }
  }
  const std::uint64_t low_order_signature_mask =
      (psps_signature_bucketing ? (std::uint64_t{1} << kPspsShellClass) : 0U) |
      (ppss_signature_bucketing ? (std::uint64_t{1} << kPpssShellClass) : 0U);
  error = prepare_generated_shell_tasks(
      stream, total_tile_capacity, generated_task_capacity, active_tile_offsets, batch,
      active_tile_counts, active_tiles, generated_tasks, generated_shell_classes,
      generated_task_offsets, generated_task_counts, generated_task_write_counts,
      generated_task_heads, low_order_signature_mask,
      low_order_signature_mask != 0U ? low_order_signature_counts : nullptr,
      low_order_signature_mask != 0U ? low_order_signature_offsets : nullptr, enabled_mask, nullptr,
      use_resident_ppps);
  if (error != cudaSuccess) return error;

  std::size_t kernel_count = 0;
  const generated::ShellKernelMetadata* kernels = generated::selected_shell_kernels(kernel_count);
  for (std::size_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
    const generated::ShellKernelMetadata& kernel = kernels[kernel_index];
    if ((enabled_mask & (std::uint64_t{1} << kernel.shell_class)) == 0U) {
      continue;
    }
    const unsigned worker_blocks =
        std::min(static_cast<unsigned>(capacities[kernel.angular_order]), persistent_worker_blocks);
    error = generated::launch_shell_class(
        kernel.shell_class, stream, unrestricted, worker_blocks, generated_tasks,
        generated_task_offsets + kernel.shell_class, batch.shell_pair_primitive_offsets,
        batch.shell_primitive_pairs, batch.direct_ao_coefficients, batch.positions,
        screening_tolerance, schwarz_bounds, density, forces,
        generated_task_counts + kernel.shell_class, generated_task_heads + kernel.shell_class);
    if (error != cudaSuccess) return error;
  }
  // The resident launch precedes ordinary preparation so an unsupported
  // optional route can select the complete fallback queue without dropping
  // any eligible ppps records.
  return cudaSuccess;
}

/**
 * Bucket all enabled generated classes once, then launch their Fock slices.
 *
 * The mask stays device-resident so graph replay can change the environment
 * selection without changing its fixed preparation and worker launch nodes.
 */
cudaError_t launch_generated_shell_class_focks(
    cudaStream_t stream, std::size_t total_tile_capacity, std::size_t generated_task_capacity,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::uint32_t* active_tile_offsets, DeviceBatch batch,
    const std::uint32_t* active_tile_counts, const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks, std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets, std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts, std::uint32_t* generated_task_heads,
    const std::uint64_t* enabled_mask, unsigned persistent_worker_blocks, bool unrestricted,
    double screening_tolerance, const double* schwarz_bounds, const double* density, double* fock) {
  cudaError_t error = prepare_generated_shell_tasks(
      stream, total_tile_capacity, generated_task_capacity, active_tile_offsets, batch,
      active_tile_counts, active_tiles, generated_tasks, generated_shell_classes,
      generated_task_offsets, generated_task_counts, generated_task_write_counts,
      generated_task_heads, 0U, nullptr, nullptr, 0U, enabled_mask, false);
  if (error != cudaSuccess) return error;

  std::size_t kernel_count = 0;
  const generated::ShellKernelMetadata* kernels =
      generated::selected_fock_shell_kernels(kernel_count);
  for (std::size_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
    const generated::ShellKernelMetadata& kernel = kernels[kernel_index];
    const unsigned worker_blocks =
        std::min(static_cast<unsigned>(capacities[kernel.angular_order]), persistent_worker_blocks);
    error = generated::launch_shell_class_fock(
        kernel.shell_class, stream, unrestricted, worker_blocks, generated_tasks,
        generated_task_offsets + kernel.shell_class, batch.shell_pair_primitive_offsets,
        batch.shell_primitive_pairs, batch.direct_ao_coefficients, batch.positions,
        screening_tolerance, schwarz_bounds, density, fock,
        generated_task_counts + kernel.shell_class, generated_task_heads + kernel.shell_class);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

/** Bucket the FP32 queue and launch only generated mixed-Fock capabilities. */
cudaError_t launch_generated_shell_class_mixed_focks(
    cudaStream_t stream, std::size_t total_tile_capacity, std::size_t generated_task_capacity,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::uint32_t* active_tile_offsets, DeviceBatch batch,
    const std::uint32_t* active_tile_counts, const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks, std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets, std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts, std::uint32_t* generated_task_heads,
    const std::uint64_t* enabled_mask, unsigned persistent_worker_blocks, bool unrestricted,
    double screening_tolerance, const double* schwarz_bounds, const double* density, double* fock) {
  const std::uint64_t capability_mask = generated::enabled_mixed_fock_shell_class_mask();
  if (capability_mask == 0U) return cudaSuccess;
  cudaError_t error = prepare_generated_shell_tasks(
      stream, total_tile_capacity, generated_task_capacity, active_tile_offsets, batch,
      active_tile_counts, active_tiles, generated_tasks, generated_shell_classes,
      generated_task_offsets, generated_task_counts, generated_task_write_counts,
      generated_task_heads, 0U, nullptr, nullptr, 0U, enabled_mask, false);
  if (error != cudaSuccess) return error;

  std::size_t kernel_count = 0;
  const generated::ShellKernelMetadata* kernels =
      generated::selected_fock_shell_kernels(kernel_count);
  for (std::size_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
    const generated::ShellKernelMetadata& kernel = kernels[kernel_index];
    if ((capability_mask & (std::uint64_t{1} << kernel.shell_class)) == 0U) {
      continue;
    }
    const unsigned worker_blocks =
        std::min(static_cast<unsigned>(capacities[kernel.angular_order]), persistent_worker_blocks);
    error = generated::launch_shell_class_mixed_fock(
        kernel.shell_class, stream, unrestricted, worker_blocks, generated_tasks,
        generated_task_offsets + kernel.shell_class, batch.shell_pair_primitive_offsets,
        batch.shell_primitive_pairs, batch.direct_ao_coefficients, batch.positions,
        screening_tolerance, schwarz_bounds, density, fock,
        generated_task_counts + kernel.shell_class, generated_task_heads + kernel.shell_class);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

template <bool Unrestricted, typename EvalScalar = double, unsigned AngularOrder = 0>
void launch_angular_fock_quartets(
    cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if constexpr (AngularOrder < detail::kDirectQuartetAngularOrderCount) {
    if (capacities[AngularOrder] != 0) {
      const std::uint32_t* order_tile_count = active_tile_counts + AngularOrder;
      const ActiveShellQuartetTile* order_tiles = active_tiles + offsets[AngularOrder];
      if constexpr (AngularOrder == kGenericOrderFiveAngularOrder &&
                    std::is_same_v<EvalScalar, double>) {
        // Every order-five class currently enabled for generated Fock owns an
        // exact queue. Avoid making the generic persistent worker claim all
        // six subtiles only to decode the class and return.
        order_tile_count = generic_order5_tile_count;
        order_tiles = generic_order5_tiles;
      }
      if constexpr (std::is_same_v<EvalScalar, MixedPrecisionFloat>) {
        // Mixed work begins at order three.  It keeps an independent queue and
        // persistent head so the ERI recurrence contains no per-warp precision
        // branch and low-order shell-fused workers remain unchanged.
        if constexpr (AngularOrder >= kMixedFockMinimumAngularOrder &&
                      AngularOrder < kPersistentFockAngularOrderCount) {
          const unsigned capacity_blocks = static_cast<unsigned>(
              capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
          build_fock_direct_quartet_persistent_kernel<Unrestricted, AngularOrder, EvalScalar>
              <<<std::min(capacity_blocks, persistent_worker_blocks), detail::kDirectQuartetThreads,
                 0, stream>>>(batch, order_tile_count, order_tiles,
                              persistent_task_heads + AngularOrder, screening_tolerance,
                              schwarz_bounds, density, active, fock,
                              generated_fock_shell_class_mask);
        } else if constexpr (AngularOrder >= kPersistentFockAngularOrderCount) {
          build_fock_direct_quartet_kernel<Unrestricted, AngularOrder, EvalScalar>
              <<<static_cast<unsigned>(capacities[AngularOrder] *
                                       detail::direct_quartet_subtiles_per_tile(AngularOrder)),
                 detail::kDirectQuartetThreads, 0, stream>>>(
                  batch, order_tile_count, order_tiles, screening_tolerance, schwarz_bounds,
                  density, active, fock, generated_fock_shell_class_mask);
        }
      } else if constexpr (AngularOrder < kPackedSsssAngularOrderCount) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        build_fock_direct_quartet_packed_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, fock);
      } else if constexpr (AngularOrder == kFusedPsssAngularOrder) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        build_fock_direct_psss_persistent_kernel<Unrestricted>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, fock);
      } else if constexpr (AngularOrder == kFusedOrderTwoAngularOrder) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        build_fock_direct_order2_persistent_kernel<Unrestricted>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, fock, generated_fock_shell_class_mask);
      } else if constexpr (AngularOrder < kPersistentFockAngularOrderCount) {
        const unsigned capacity_blocks = static_cast<unsigned>(
            capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
        build_fock_direct_quartet_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_blocks, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, fock, generated_fock_shell_class_mask);
      } else {
        build_fock_direct_quartet_kernel<Unrestricted, AngularOrder>
            <<<static_cast<unsigned>(capacities[AngularOrder] *
                                     detail::direct_quartet_subtiles_per_tile(AngularOrder)),
               detail::kDirectQuartetThreads, 0, stream>>>(
                batch, order_tile_count, order_tiles, screening_tolerance, schwarz_bounds, density,
                active, fock, generated_fock_shell_class_mask);
      }
    }
    launch_angular_fock_quartets<Unrestricted, EvalScalar, AngularOrder + 1>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
        persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
        generated_fock_shell_class_mask);
  }
}

template <bool Unrestricted, unsigned AngularOrder = 0>
void launch_angular_force_quartets(
    cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, const PsssResidentTask* psss_resident_tasks,
    const std::uint32_t* psss_resident_ket_pairs, std::size_t psss_resident_task_count,
    std::size_t resident_psss_bra_primitive_pairs, double screening_tolerance,
    const double* shell_pair_bounds, const ShellPairDensityBounds* shell_pair_density_bounds,
    bool force_density_product_screening, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, std::uint64_t generated_shell_class_mask) {
  if constexpr (AngularOrder < detail::kDirectQuartetAngularOrderCount) {
    if (capacities[AngularOrder] != 0) {
      const std::uint32_t* order_tile_count = active_tile_counts + AngularOrder;
      const ActiveShellQuartetTile* order_tiles = active_tiles + offsets[AngularOrder];
      if constexpr (AngularOrder == kGenericOrderFiveAngularOrder) {
        // Generated classes have exact queues. The generic order-five worker
        // consumes only classes not selected by the current runtime mask.
        order_tile_count = generic_order5_tile_count;
        order_tiles = generic_order5_tiles;
      }
      if constexpr (AngularOrder < kPackedSsssAngularOrderCount) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        two_electron_force_quartet_packed_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);
      } else if constexpr (AngularOrder == kFusedPsssAngularOrder) {
        if (psss_resident_task_count != 0 && resident_psss_bra_primitive_pairs != 0 &&
            resident_psss_bra_primitive_pairs <= kResidentPsssMaximumBraPrimitivePairs) {
          two_electron_force_psss_resident_bra_kernel<Unrestricted>
              <<<static_cast<unsigned>(psss_resident_task_count), kResidentPsssThreads,
                 resident_psss_bra_primitive_pairs * sizeof(PrimitivePairData), stream>>>(
                  batch, psss_resident_tasks, psss_resident_ket_pairs, psss_resident_task_count,
                  screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
                  force_density_product_screening, schwarz_bounds, density, active, forces,
                  generated_shell_class_mask);
        } else {
          const unsigned capacity_workers =
              static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                    detail::kDirectQuartetThreads);
          two_electron_force_psss_persistent_kernel<Unrestricted>
              <<<std::min(capacity_workers, persistent_worker_blocks),
                 detail::kDirectQuartetThreads, 0, stream>>>(
                  batch, order_tile_count, order_tiles, persistent_task_heads + AngularOrder,
                  screening_tolerance, schwarz_bounds, density, active, forces,
                  generated_shell_class_mask);
        }
      } else if constexpr (AngularOrder == kFusedOrderTwoAngularOrder) {
        const unsigned capacity_workers =
            static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                  detail::kDirectQuartetThreads);
        two_electron_force_psps_grid_stride_kernel<Unrestricted>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);
        two_electron_force_pair_order2_grid_stride_kernel<Unrestricted, kPpssShellClass>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);
        two_electron_force_pair_order2_grid_stride_kernel<Unrestricted, kDsssShellClass>
            <<<std::min(capacity_workers, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);

        // Exact shell workers own all three order-two classes unless an
        // enabled generated kernel already consumed one. The generic launch
        // is retained as a guarded safety net for future class additions.
        const std::uint64_t generic_shell_class_mask =
            generated_shell_class_mask | (std::uint64_t{1} << kPspsShellClass) |
            (std::uint64_t{1} << kPpssShellClass) | (std::uint64_t{1} << kDsssShellClass);
        const unsigned capacity_blocks = static_cast<unsigned>(
            capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
        two_electron_force_quartet_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_blocks, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, forces, generic_shell_class_mask);
      } else if constexpr (AngularOrder < kPersistentForceAngularOrderCount) {
        const unsigned capacity_blocks = static_cast<unsigned>(
            capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
        two_electron_force_quartet_persistent_kernel<Unrestricted, AngularOrder>
            <<<std::min(capacity_blocks, persistent_worker_blocks), detail::kDirectQuartetThreads,
               0, stream>>>(batch, order_tile_count, order_tiles,
                            persistent_task_heads + AngularOrder, screening_tolerance,
                            schwarz_bounds, density, active, forces, generated_shell_class_mask);
      } else {
        two_electron_force_quartet_kernel<Unrestricted, AngularOrder>
            <<<static_cast<unsigned>(capacities[AngularOrder] *
                                     detail::direct_quartet_subtiles_per_tile(AngularOrder)),
               detail::kDirectQuartetThreads, 0, stream>>>(
                batch, order_tile_count, order_tiles, screening_tolerance, schwarz_bounds, density,
                active, forces, generated_shell_class_mask);
      }
    }
    launch_angular_force_quartets<Unrestricted, AngularOrder + 1>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
        persistent_worker_blocks, psss_resident_tasks, psss_resident_ket_pairs,
        psss_resident_task_count, resident_psss_bra_primitive_pairs, screening_tolerance,
        shell_pair_bounds, shell_pair_density_bounds, force_density_product_screening,
        schwarz_bounds, density, active, forces, generated_shell_class_mask);
  }
}

void fill_global_failure(std::vector<RhfBucketItem>& outputs, vibeqc_status status) {
  for (RhfBucketItem& output : outputs) output.status = status;
}

}  // namespace

struct CudaRhfBucketPlan {
  CudaResources resources;
  ArenaLayout layout;
  HostBatch topology;
  // Geometry-derived arena state is reusable until coordinates change.
  std::vector<double> cached_positions;
  // The current device density and its associated convergence seed are one
  // cache, while a fixed benchmark dm0 and seed are a separate cache. The
  // distinction matters because finalization advances the returned density
  // after evaluating the final energy, so repeated fixed-dm0 replays cease to
  // be resident hits even though they must retain the original energy seed.
  std::vector<double> resident_warm_positions;
  std::vector<double> resident_warm_density;
  std::vector<double> resident_previous_energy;
  std::vector<double> frozen_warm_positions;
  std::vector<double> frozen_warm_density;
  std::vector<double> frozen_previous_energy;
  std::optional<CudaRhfShellClassProfile> last_shell_class_profile;
  std::optional<CudaPppsQueueProfile> last_ppps_queue_profile;
  std::optional<CudaInactiveEigensolverProfile> last_inactive_eigensolver_profile;
  CudaEigensolverDiagnostic eigensolver_diagnostic;
  ScfOptions options;
  std::size_t batch_size{};
  std::size_t nbf{};
  std::size_t direct_nbf{};
  std::size_t total_atoms{};
  std::size_t total_shells{};
  std::size_t total_shell_pairs{};
  std::size_t total_shell_quartets{};
  std::size_t total_shell_pair_blocks{};
  std::size_t total_shell_pair_block_quartets{};
  std::size_t total_shell_quartet_tiles{};
  std::vector<std::uint32_t> bounded_direct_shell_pair_order;
  std::vector<std::uint32_t> bounded_stream_shell_pair_order;
  std::vector<std::uint32_t> bounded_stream_pair_class_offsets;
  std::size_t bounded_generated_task_capacity{};
  std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1>
      bounded_generated_task_offsets{};
  std::array<std::uint64_t, detail::kDirectQuartetShellClassCount>
      bounded_generated_task_upper_bounds{};
  std::size_t generated_shell_task_capacity{};
  std::size_t resident_ppps_ket_task_capacity{};
  std::array<std::size_t, detail::kDirectQuartetAngularOrderCount> shell_quartet_tile_capacities{};
  std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>
      shell_quartet_tile_offsets{};
  std::size_t fp32_shell_quartet_tile_capacity{};
  std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>
      fp32_shell_quartet_tile_offsets{};
  unsigned persistent_quartet_worker_blocks{};
  std::size_t resident_psss_bra_primitive_pairs{};
  std::size_t resident_psss_task_count{};
  bool generated_psss_weighted{};
  unsigned one_electron_value_mapping{};
  std::size_t primitive_count{};
  std::size_t diis_history{};
  int lwork{};
  bool persistent_eri{};
  bool quartet_direct{};
  bool transformed_direct{};
  bool bounded_direct_streaming{};
  bool unrestricted{};
  bool shell_class_profiling{};
  bool inactive_eigensolver_profiling{};
  bool bounded_fock_class_timing{};
  // These switches change captured work even when topology and arithmetic match.
  bool bounded_streaming_override{};
  bool fock_only_diagnostic{};
  bool graph_native_eigensolver_override{};
  bool reuse_converged_fock{};
  bool mixed_precision_fock{};
  double mixed_precision_fock_threshold{};
  /** Largest item census the batch admission ceiling was bound to. */
  std::size_t mixed_precision_eligible_tile_count{};
  /** Exact per-system mixed-capable tile census the per-item budget divides. */
  std::vector<std::uint32_t> mixed_precision_system_census;
  bool warm_start_updates_enabled{true};
  bool cublas_enabled{true};
  bool retry_without_cublas{};
  bool initialized{};
};

namespace {

bool same_options(const ScfOptions& first, const ScfOptions& second) {
  return first.max_iterations == second.max_iterations &&
         first.diis_history == second.diis_history &&
         first.energy_tolerance == second.energy_tolerance &&
         first.density_tolerance == second.density_tolerance &&
         first.screening_tolerance == second.screening_tolerance &&
         first.compute_forces == second.compute_forces &&
         first.export_physical_reference == second.export_physical_reference &&
         first.reference_memory_budget_bytes == second.reference_memory_budget_bytes &&
         first.precision_mode == second.precision_mode &&
         first.resolved_fock_build == second.resolved_fock_build;
}

std::vector<RhfBucketItem> execute_hf_cuda_bucket(CudaRhfBucketPlan& plan, const HostBatch& host,
                                                  const ScfOptions& options, int device_id,
                                                  bool unrestricted, bool shell_class_profiling,
                                                  bool inactive_eigensolver_profiling) {
  const std::size_t batch_size = host.warm_mask.size();
  std::vector<RhfBucketItem> outputs(batch_size);
  if (options.export_physical_reference &&
      (unrestricted || options.screening_tolerance != 0 || batch_size != 1)) {
    fill_global_failure(outputs, VIBEQC_STATUS_NOT_IMPLEMENTED);
    return outputs;
  }
  plan.last_shell_class_profile.reset();
  plan.last_ppps_queue_profile.reset();
  plan.last_inactive_eigensolver_profile.reset();

  const std::size_t nbf = host.nbf;
  const std::size_t direct_nbf = host.direct_nbf;
  const std::size_t spin_count = host.spin_count;
  std::size_t spin_batch_size = 0;
  std::size_t matrix_size = 0;
  std::size_t eri_size = 0;
  std::size_t matrix_elements = 0;
  std::size_t spin_matrix_elements = 0;
  std::size_t eri_elements = 0;
  std::size_t nbf_plus_one = 0;
  std::size_t pair_product = 0;
  std::size_t direct_matrix_size = 0;
  std::size_t direct_matrix_elements = 0;
  std::size_t direct_spin_matrix_elements = 0;
  std::size_t direct_nbf_plus_one = 0;
  std::size_t direct_pair_product = 0;
  std::size_t public_ao_elements = 0;
  std::size_t rectangular_matrix_elements = 0;
  std::size_t spin_rectangular_matrix_elements = 0;
  if (!checked_multiply(nbf, nbf, matrix_size) ||
      !checked_multiply(matrix_size, matrix_size, eri_size) ||
      !checked_multiply(batch_size, matrix_size, matrix_elements) ||
      !checked_multiply(batch_size, spin_count, spin_batch_size) ||
      !checked_multiply(matrix_elements, spin_count, spin_matrix_elements) ||
      !checked_multiply(batch_size, eri_size, eri_elements) || !checked_add(nbf, 1, nbf_plus_one) ||
      !checked_multiply(nbf, nbf_plus_one, pair_product) ||
      !checked_multiply(direct_nbf, direct_nbf, direct_matrix_size) ||
      !checked_multiply(batch_size, direct_matrix_size, direct_matrix_elements) ||
      !checked_multiply(direct_matrix_elements, spin_count, direct_spin_matrix_elements) ||
      !checked_add(direct_nbf, 1, direct_nbf_plus_one) ||
      !checked_multiply(direct_nbf, direct_nbf_plus_one, direct_pair_product) ||
      !checked_multiply(batch_size, nbf, public_ao_elements) ||
      !checked_multiply(public_ao_elements, direct_nbf, rectangular_matrix_elements) ||
      !checked_multiply(rectangular_matrix_elements, spin_count,
                        spin_rectangular_matrix_elements)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t pair_count = pair_product / 2;
  const std::size_t direct_pair_count = direct_pair_product / 2;
  std::size_t pair_elements = 0;
  std::size_t direct_pair_elements = 0;
  if (!checked_multiply(batch_size, pair_count, pair_elements) ||
      !checked_multiply(batch_size, direct_pair_count, direct_pair_elements)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (matrix_elements > std::numeric_limits<unsigned>::max() ||
      spin_matrix_elements > std::numeric_limits<unsigned>::max() ||
      direct_matrix_elements > std::numeric_limits<unsigned>::max() ||
      direct_spin_matrix_elements > std::numeric_limits<unsigned>::max() ||
      rectangular_matrix_elements > std::numeric_limits<unsigned>::max() ||
      spin_rectangular_matrix_elements > std::numeric_limits<unsigned>::max() ||
      direct_pair_elements > std::numeric_limits<unsigned>::max() ||
      spin_batch_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_atoms = host.atomic_numbers.size();
  const std::size_t total_shells = host.shell_atoms.size();
  const std::size_t total_shell_pairs = host.shell_pair_first.size();
  if (host.shell_pair_primitive_offsets.size() != total_shell_pairs + 1 ||
      host.shell_pair_primitive_offsets.back() < 0) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_pair_primitives =
      static_cast<std::size_t>(host.shell_pair_primitive_offsets.back());
  if (host.system_shell_quartet_offsets.size() != batch_size + 1 ||
      host.system_shell_quartet_offsets.back() < 0) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_quartets =
      static_cast<std::size_t>(host.system_shell_quartet_offsets.back());
  if (host.system_shell_pair_block_offsets.size() != batch_size + 1 ||
      host.system_shell_pair_block_quartet_offsets.size() != batch_size + 1 ||
      host.system_shell_pair_block_offsets.back() < 0 ||
      host.system_shell_pair_block_quartet_offsets.back() < 0) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_pair_blocks =
      static_cast<std::size_t>(host.system_shell_pair_block_offsets.back());
  const std::size_t total_shell_pair_block_quartets =
      static_cast<std::size_t>(host.system_shell_pair_block_quartet_offsets.back());
  const bool requested_persistent_eri =
      !options.export_physical_reference && nbf <= kPersistentEriAoLimit;
  const bool requested_quartet_direct =
      !options.export_physical_reference && !requested_persistent_eri &&
      std::all_of(host.shell_angular.begin(), host.shell_angular.end(),
                  [](std::uint8_t angular) { return angular <= 3; });
  const bool requested_transformed_direct = requested_quartet_direct && direct_nbf != nbf;
  // Reference export selects the bounded matrix-direct evaluator; optimized
  // quartet dispatch retains its generated-class coverage gate.
  bool requested_bounded_direct_streaming =
      requested_quartet_direct &&
      (detail::direct_topology_requires_bounded_streaming(total_shell_quartets) ||
       bounded_direct_streaming_override_requested());
  const bool cooperative_one_electron_force = one_electron_force_scalar_requested();
  const bool requested_graph_native_eigensolver_override =
      !options.export_physical_reference && graph_native_eigensolver_override_requested();
  const bool xsyev_probe_skip_diagnostic = xsyev_probe_skip_diagnostic_requested();
  // Read this on every cached execution so one prepared batch can provide a
  // fixed-dm0 old/new A/B without rebuilding its immutable topology plan.
  const bool force_density_product_screening = force_density_product_screening_requested();
  const bool bounded_direct_count_diagnostic = bounded_direct_count_diagnostic_requested();
  const bool bounded_direct_aot_only_diagnostic = bounded_direct_aot_only_diagnostic_requested();
  const bool bounded_direct_fock_only_diagnostic = bounded_direct_fock_only_diagnostic_requested();
  const bool bounded_fock_class_timing = bounded_fock_class_timing_requested();
  const bool direct_tile_validation = direct_tile_validation_requested();
  // Read this per execution so one prepared topology can compare the new
  // route with the complete ordinary ppps queue in the same binary.
  const bool resident_ppps_bra = resident_ppps_bra_requested();
  const bool resident_ppps_signature_bucketing = ppps_signature_bucketing_requested();
  const bool psps_signature_bucketing = psps_signature_bucketing_requested();
  const bool ppss_signature_bucketing = ppss_signature_bucketing_requested();
  const unsigned resident_ppps_block_threads = ppps_resident_block_threads_requested();
  const bool first_setup = !plan.initialized;
  detail::DirectQuartetTaskLayout direct_task_layout{};
  std::size_t total_shell_quartet_tiles = 0;
  if (requested_quartet_direct && first_setup && !requested_bounded_direct_streaming) {
    // Exact topology capacities are immutable for a prepared bucket. Their
    // pair-of-pairs enumeration is O(n_shell_pairs^2), so recomputing it on
    // every warm replay adds substantial host latency at large AO counts.
    if (!detail::make_direct_quartet_task_layout(
            host.shell_direct_ao_offsets, host.shell_angular, host.system_shell_pair_offsets,
            host.shell_pair_first, host.shell_pair_second, kMixedFockMinimumAngularOrder,
            direct_task_layout) ||
        direct_task_layout.shell_quartet_count != total_shell_quartets) {
      fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
    total_shell_quartet_tiles = direct_task_layout.exact_tile_count;
    if (total_shell_quartet_tiles > detail::kDirectFixedTopologyTileLimit) {
      // A high-angular topology can exceed the fixed grid before its quartet
      // count alone proves that fact. Discard the setup-only exact counts and
      // use the same bounded device enumerator as obviously large systems.
      requested_bounded_direct_streaming = true;
      direct_task_layout = {};
      total_shell_quartet_tiles = 0;
    } else if (direct_task_layout.exact_tile_count >
               kFixedGeneratedTaskArenaMaximumBytes / sizeof(GeneratedShellTask)) {
      // The uint32 grid limit is much larger than a practical descriptor
      // arena on a 32 GiB device.  Route large-but-grid-addressable buckets
      // through bounded streaming before make_layout() reserves the complete
      // generated task array.
      requested_bounded_direct_streaming = true;
      direct_task_layout = {};
      total_shell_quartet_tiles = 0;
    }
  } else if (requested_quartet_direct) {
    requested_bounded_direct_streaming =
        first_setup ? requested_bounded_direct_streaming : plan.bounded_direct_streaming;
    total_shell_quartet_tiles =
        requested_bounded_direct_streaming ? 0 : plan.total_shell_quartet_tiles;
  }
  // Per-item mixed-capable tile census: the FP32-error budget is evaluated for
  // every system on its own count. Bounded streaming keeps zeros, which the
  // policy refuses rather than guesses, and the largest census is the batch
  // ceiling that decides whether the plan allocates the route at all.
  std::vector<std::size_t> mixed_precision_system_census(batch_size, 0U);
  if (requested_quartet_direct && !requested_bounded_direct_streaming) {
    if (first_setup) {
      mixed_precision_system_census = direct_task_layout.system_mixed_capable_tile_counts;
      if (mixed_precision_system_census.size() != batch_size) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
    } else if (plan.mixed_precision_system_census.size() == batch_size) {
      for (std::size_t system = 0; system < batch_size; ++system) {
        mixed_precision_system_census[system] = plan.mixed_precision_system_census[system];
      }
    }
  }
  const std::size_t mixed_precision_eligible_tile_count =
      mixed_precision_system_census.empty()
          ? 0U
          : *std::max_element(mixed_precision_system_census.begin(),
                              mixed_precision_system_census.end());
  const MixedPrecisionFockPolicy requested_precision_policy =
      requested_quartet_direct
          ? resolve_mixed_precision_fock_policy(
                options.precision_mode, options.energy_tolerance, options.screening_tolerance,
                static_cast<double>(mixed_precision_eligible_tile_count))
          : MixedPrecisionFockPolicy{};
  const std::optional<double> requested_mixed_precision_fock_threshold =
      requested_precision_policy.threshold;
  const bool requested_mixed_precision_fock = requested_mixed_precision_fock_threshold.has_value();
  // A mixed item is promoted to exact FP64 by the target refinement before any
  // consumer runs, so the matrix it retains is target precision. The density
  // criterion and convergence check below still decide each item's reuse.
  const bool requested_reuse_converged_fock =
      reuse_converged_fock_requested() && !options.export_physical_reference;
  // Direct consumers expand each compact logical tile into one-warp blocks;
  // validate the resulting fixed Graph grid before narrowing it to unsigned.
  if (total_shell_pairs > std::numeric_limits<unsigned>::max() ||
      (!requested_bounded_direct_streaming &&
       total_shell_quartets > std::numeric_limits<unsigned>::max()) ||
      host.psss_resident_tasks.size() > std::numeric_limits<unsigned>::max() ||
      (!requested_bounded_direct_streaming &&
       total_shell_quartet_tiles > detail::kDirectFixedTopologyTileLimit)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  std::size_t force_coordinate_count = 0;
  std::size_t one_electron_force_elements = 0;
  std::size_t force_matrix_elements = 0;
  std::size_t persistent_force_elements = 0;
  std::size_t direct_force_elements = 0;
  if (!checked_multiply(total_atoms, 3, force_coordinate_count) ||
      !checked_multiply(batch_size, pair_count, one_electron_force_elements) ||
      !checked_multiply(force_coordinate_count, matrix_size, force_matrix_elements) ||
      !checked_multiply(force_coordinate_count, eri_size, persistent_force_elements) ||
      !checked_multiply(force_matrix_elements, pair_count, direct_force_elements)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t diis_history = std::max<std::size_t>(1, options.diis_history);
  if (diis_history > 64) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (!first_setup &&
      (plan.resources.device_id_ != device_id || !same_topology(plan.topology, host) ||
       !same_options(plan.options, options) || plan.unrestricted != unrestricted ||
       plan.bounded_direct_streaming != requested_bounded_direct_streaming ||
       plan.shell_class_profiling != shell_class_profiling ||
       plan.inactive_eigensolver_profiling != inactive_eigensolver_profiling ||
       plan.bounded_fock_class_timing != bounded_fock_class_timing ||
       plan.bounded_streaming_override != bounded_direct_streaming_override_requested() ||
       plan.fock_only_diagnostic != bounded_direct_fock_only_diagnostic ||
       plan.graph_native_eigensolver_override != requested_graph_native_eigensolver_override ||
       plan.reuse_converged_fock != requested_reuse_converged_fock ||
       plan.one_electron_value_mapping != cuda_policy::one_electron_value_mapping_requested() ||
       plan.mixed_precision_fock != requested_mixed_precision_fock ||
       plan.mixed_precision_fock_threshold !=
           requested_mixed_precision_fock_threshold.value_or(0.0))) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  std::size_t generated_shell_task_capacity = first_setup ? 0 : plan.generated_shell_task_capacity;
  std::size_t resident_ppps_ket_task_capacity =
      first_setup ? 0 : plan.resident_ppps_ket_task_capacity;
  std::size_t fp32_shell_quartet_tile_capacity =
      first_setup ? 0 : plan.fp32_shell_quartet_tile_capacity;
  if (first_setup && requested_mixed_precision_fock) {
    for (std::size_t order = kMixedFockMinimumAngularOrder;
         order < detail::kDirectQuartetAngularOrderCount; ++order) {
      if (!checked_add(fp32_shell_quartet_tile_capacity,
                       direct_task_layout.angular_order_tile_counts[order],
                       fp32_shell_quartet_tile_capacity)) {
        fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
        return outputs;
      }
    }
    if (fp32_shell_quartet_tile_capacity >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
  }
  const std::size_t generic_order5_tile_capacity =
      requested_quartet_direct && !requested_bounded_direct_streaming
          ? (first_setup
                 ? direct_task_layout.angular_order_tile_counts[kGenericOrderFiveAngularOrder]
                 : plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder])
          : 0;
  std::size_t bounded_generated_task_capacity =
      first_setup ? 0 : plan.bounded_generated_task_capacity;
  if (first_setup && requested_bounded_direct_streaming) {
    if (!checked_multiply(total_shell_pairs, kBoundedGeneratedTasksPerShellPair,
                          bounded_generated_task_capacity)) {
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    bounded_generated_task_capacity =
        std::min(bounded_generated_task_capacity, kBoundedGeneratedMaximumTaskCapacity);
  }
  if (requested_quartet_direct && first_setup && !requested_bounded_direct_streaming) {
    // The shared generated-task arena serves both exact Fock and force
    // consumers.  Their registries are intentionally not identical: for
    // example, `ssss` has a generated Fock consumer but remains on the
    // handwritten force path.  Build the capacity from their union so a
    // Fock-only class cannot leave its persistent kernel with a zero-sized
    // task arena.
    std::array<bool, detail::kDirectQuartetShellClassCount> generated_task_classes{};
    const auto include_generated_task_classes = [&](const generated::ShellKernelMetadata* kernels,
                                                    std::size_t kernel_count) {
      for (std::size_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
        const unsigned shell_class = kernels[kernel_index].shell_class;
        if (shell_class < generated_task_classes.size()) {
          generated_task_classes[shell_class] = true;
        }
      }
    };
    std::size_t force_kernel_count = 0;
    const generated::ShellKernelMetadata* force_kernels =
        generated::selected_shell_kernels(force_kernel_count);
    include_generated_task_classes(force_kernels, force_kernel_count);
    std::size_t fock_kernel_count = 0;
    const generated::ShellKernelMetadata* fock_kernels =
        generated::selected_fock_shell_kernels(fock_kernel_count);
    include_generated_task_classes(fock_kernels, fock_kernel_count);
    for (std::size_t shell_class = 0; shell_class < generated_task_classes.size(); ++shell_class) {
      if (!generated_task_classes[shell_class]) continue;
      if (!checked_add(generated_shell_task_capacity,
                       direct_task_layout.shell_class_tile_counts[shell_class],
                       generated_shell_task_capacity)) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
    }
    // Every active canonical ppps shell quartet occupies one tile at angular
    // order three, so this exact class count is the maximum resident ket
    // record count.  Reserve a reusable tail of the existing generated-task
    // arena only when the selected AOT bundle contains a ppps force consumer;
    // portable/stub builds then retain their original arena footprint.
    bool ppps_force_available = false;
    for (std::size_t kernel_index = 0; kernel_index < force_kernel_count; ++kernel_index) {
      if (force_kernels[kernel_index].shell_class == kPppsShellClass) {
        ppps_force_available = true;
        break;
      }
    }
    resident_ppps_ket_task_capacity =
        ppps_force_available ? direct_task_layout.shell_class_tile_counts[kPppsShellClass] : 0;
  }
  if (resident_ppps_ket_task_capacity > generated_shell_task_capacity ||
      resident_ppps_ket_task_capacity >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (first_setup) {
    if (!make_layout(
            batch_size, nbf, direct_nbf, total_atoms, total_shells, total_shell_pairs,
            total_shell_pair_blocks, bounded_generated_task_capacity, total_shell_pair_primitives,
            requested_quartet_direct ? host.psss_resident_tasks.size() : 0,
            requested_quartet_direct ? host.psss_resident_ket_pairs.size() : 0,
            total_shell_quartet_tiles, fp32_shell_quartet_tile_capacity,
            generated_shell_task_capacity, resident_ppps_ket_task_capacity,
            generic_order5_tile_capacity, host.primitive_exponents.size(), diis_history,
            options.max_iterations, host.spin_count, requested_persistent_eri,
            requested_transformed_direct, shell_class_profiling, inactive_eigensolver_profiling,
            bounded_fock_class_timing, requested_bounded_direct_streaming,
            requested_mixed_precision_fock, plan.layout)) {
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    plan.batch_size = batch_size;
    plan.nbf = nbf;
    plan.direct_nbf = direct_nbf;
    plan.total_atoms = total_atoms;
    plan.total_shells = total_shells;
    plan.total_shell_pairs = total_shell_pairs;
    plan.total_shell_quartets = total_shell_quartets;
    plan.total_shell_pair_blocks = total_shell_pair_blocks;
    plan.total_shell_pair_block_quartets = total_shell_pair_block_quartets;
    plan.total_shell_quartet_tiles = total_shell_quartet_tiles;
    plan.bounded_generated_task_capacity = bounded_generated_task_capacity;
    plan.bounded_generated_task_offsets =
        requested_bounded_direct_streaming
            ? make_bounded_generated_task_offsets(host, bounded_generated_task_capacity,
                                                  &plan.bounded_generated_task_upper_bounds)
            : std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1>{};
    if (!requested_bounded_direct_streaming) {
      plan.bounded_generated_task_upper_bounds.fill(0U);
    }
    if (requested_bounded_direct_streaming) {
      plan.bounded_direct_shell_pair_order.resize(total_shell_pairs);
      std::iota(plan.bounded_direct_shell_pair_order.begin(),
                plan.bounded_direct_shell_pair_order.end(), 0U);
      if (!make_bounded_stream_shell_pair_order(host, plan.bounded_stream_shell_pair_order,
                                                plan.bounded_stream_pair_class_offsets)) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
    }
    plan.resident_psss_bra_primitive_pairs = 0;
    plan.generated_psss_weighted = cuda_policy::generated_psss_weighted_requested();
    plan.one_electron_value_mapping = cuda_policy::one_electron_value_mapping_requested();
    const bool resident_psss_enabled = resident_psss_bra_requested();
    // The bounded direct force path has its own exact page consumer for psss.
    // Keep the resident-bra optimization on the fixed-queue path only until
    // its bounded scheduling and force accumulation are independently gated.
    plan.resident_psss_task_count =
        requested_quartet_direct && !requested_bounded_direct_streaming && resident_psss_enabled
            ? host.psss_resident_tasks.size()
            : 0;
    for (std::size_t pair = 0; pair < total_shell_pairs; ++pair) {
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      if (host.shell_angular[first_shell] + host.shell_angular[second_shell] != 1U) {
        continue;
      }
      const std::size_t primitive_pairs = static_cast<std::size_t>(
          host.shell_pair_primitive_offsets[pair + 1] - host.shell_pair_primitive_offsets[pair]);
      plan.resident_psss_bra_primitive_pairs =
          std::max(plan.resident_psss_bra_primitive_pairs, primitive_pairs);
    }
    plan.generated_shell_task_capacity = generated_shell_task_capacity;
    plan.resident_ppps_ket_task_capacity = resident_ppps_ket_task_capacity;
    plan.shell_quartet_tile_capacities = direct_task_layout.angular_order_tile_counts;
    for (std::size_t order = 0; order < direct_task_layout.angular_order_tile_offsets.size();
         ++order) {
      plan.shell_quartet_tile_offsets[order] =
          static_cast<std::uint32_t>(direct_task_layout.angular_order_tile_offsets[order]);
    }
    plan.fp32_shell_quartet_tile_capacity = fp32_shell_quartet_tile_capacity;
    std::size_t fp32_tile_offset = 0;
    for (std::size_t order = 0; order < detail::kDirectQuartetAngularOrderCount; ++order) {
      plan.fp32_shell_quartet_tile_offsets[order] = static_cast<std::uint32_t>(fp32_tile_offset);
      if (requested_mixed_precision_fock && order >= kMixedFockMinimumAngularOrder) {
        fp32_tile_offset += direct_task_layout.angular_order_tile_counts[order];
      }
    }
    plan.fp32_shell_quartet_tile_offsets[detail::kDirectQuartetAngularOrderCount] =
        static_cast<std::uint32_t>(fp32_tile_offset);
    plan.primitive_count = host.primitive_exponents.size();
    plan.diis_history = diis_history;
    plan.persistent_eri = requested_persistent_eri;
    plan.quartet_direct = requested_quartet_direct;
    plan.transformed_direct = requested_transformed_direct;
    plan.bounded_direct_streaming = requested_bounded_direct_streaming;
    plan.unrestricted = unrestricted;
    plan.shell_class_profiling = shell_class_profiling;
    plan.inactive_eigensolver_profiling = inactive_eigensolver_profiling;
    plan.bounded_fock_class_timing = bounded_fock_class_timing;
    plan.bounded_streaming_override = bounded_direct_streaming_override_requested();
    plan.fock_only_diagnostic = bounded_direct_fock_only_diagnostic;
    plan.graph_native_eigensolver_override = requested_graph_native_eigensolver_override;
    plan.reuse_converged_fock = requested_reuse_converged_fock;
    plan.mixed_precision_fock = requested_mixed_precision_fock;
    plan.mixed_precision_fock_threshold = requested_mixed_precision_fock_threshold.value_or(0.0);
    plan.mixed_precision_eligible_tile_count = mixed_precision_eligible_tile_count;
    plan.mixed_precision_system_census.assign(mixed_precision_system_census.size(), 0U);
    for (std::size_t system = 0; system < mixed_precision_system_census.size(); ++system) {
      if (mixed_precision_system_census[system] >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
      plan.mixed_precision_system_census[system] =
          static_cast<std::uint32_t>(mixed_precision_system_census[system]);
    }
    plan.options = options;
    plan.topology = host;
    // Positions and warm guesses are dynamic execution inputs, not part of
    // the immutable fixed-topology cache identity.
    plan.topology.positions.clear();
    plan.topology.warm_mask.clear();
    plan.topology.warm_density.clear();
    plan.resources.device_id_ = device_id;
  }
  ArenaLayout& layout = plan.layout;
  CudaResources& resources = plan.resources;
  const bool persistent_eri = plan.persistent_eri;
  const bool quartet_direct = plan.quartet_direct;
  const bool transformed_direct = plan.transformed_direct;
  if (transformed_direct != !host.ao_to_direct_transform.empty()) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const bool bounded_direct_streaming = plan.bounded_direct_streaming;
  const bool reuse_converged_fock = plan.reuse_converged_fock;
  const bool mixed_precision_fock = plan.mixed_precision_fock;
  const double mixed_precision_fock_threshold = plan.mixed_precision_fock_threshold;
  if (first_setup) {
    plan.eigensolver_diagnostic = {};
    plan.eigensolver_diagnostic.matrix_dimension = nbf;
    plan.eigensolver_diagnostic.physical_system_count = batch_size;
    plan.eigensolver_diagnostic.solver_batch_count = spin_batch_size;
    // A forced graph-native selection is an explicit escape hatch for large
    // matrices where cuSOLVER XsyevBatched capture is known to be unusable.
    // Do not probe that provider first: the probe allocates and executes a
    // full-size eigensystem and can itself spend minutes in host-side setup.
    if (requested_graph_native_eigensolver_override) {
      plan.eigensolver_diagnostic.family = CudaEigensolverFamily::graph_native;
      plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::graph_native;
      plan.eigensolver_diagnostic.selection_source =
          CudaEigensolverSelectionSource::benchmark_override;
    } else if (nbf <= static_cast<std::size_t>(kSmallEigensolverLimit)) {
      plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::small_native;
      plan.eigensolver_diagnostic.family = CudaEigensolverFamily::small_native;
    } else if (nbf <= static_cast<std::size_t>(kBatchedEigensolverLimit)) {
      plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::jacobi_batched;
      plan.eigensolver_diagnostic.family = CudaEigensolverFamily::jacobi_batched;
    } else if (options.export_physical_reference) {
      // The existing ordinary-stream Xsyevd path avoids an unbudgeted full
      // provider/capture probe. Query and bound its real workspaces below.
      plan.eigensolver_diagnostic.family = CudaEigensolverFamily::graph_native;
      plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::xsyevd;
      plan.eigensolver_diagnostic.selection_source =
          CudaEigensolverSelectionSource::dimension_policy;
    } else {
      plan.eigensolver_diagnostic.xsyev_probe =
          probe_xsyev_batched_device_launch_graph(device_id, nbf, spin_batch_size);
      // A rejected capture intentionally leaves CUDA's per-thread last-error
      // slot set to cudaErrorStreamCaptureUnsupported (901) on CUDA 12.9.
      // The probe has already recorded that evidence; clear the sticky slot
      // before the real plan allocates, captures, and launches its fallback.
      // Otherwise the unrelated final cudaGetLastError() would report the
      // old probe rejection as a calculation failure.
      (void)cudaGetLastError();
      const XsyevBatchedDispatch dispatch =
          select_xsyev_batched_dispatch(plan.eigensolver_diagnostic.xsyev_probe);
      plan.eigensolver_diagnostic.family = dispatch.device_launch_graph_provider
                                               ? CudaEigensolverFamily::xsyev_batched
                                               : CudaEigensolverFamily::graph_native;
      if (dispatch.device_launch_graph_provider) {
        plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::xsyev_batched;
      } else if (dispatch.ordinary_stream_provider) {
        // Match GPU4PySCF's robust large-matrix strategy when the generic
        // batched provider works on a stream but rejects Graph capture.
        plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::xsyevd;
      } else {
        plan.eigensolver_diagnostic.ordinary_family = CudaEigensolverFamily::graph_native;
      }
      plan.eigensolver_diagnostic.selection_source =
          xsyev_probe_skip_diagnostic ? CudaEigensolverSelectionSource::benchmark_override
          : plan.eigensolver_diagnostic.xsyev_probe.graph_eligible
              ? CudaEigensolverSelectionSource::exact_probe
              : CudaEigensolverSelectionSource::exact_probe_fallback;
    }
  }
  const CudaEigensolverFamily graph_eigensolver_family = plan.eigensolver_diagnostic.family;
  const CudaEigensolverFamily ordinary_eigensolver_family =
      plan.eigensolver_diagnostic.ordinary_family;
  const bool use_jacobi = graph_eigensolver_family == CudaEigensolverFamily::jacobi_batched ||
                          ordinary_eigensolver_family == CudaEigensolverFamily::jacobi_batched;
  const bool use_cusolver = use_jacobi ||
                            ordinary_eigensolver_family == CudaEigensolverFamily::xsyev_batched ||
                            ordinary_eigensolver_family == CudaEigensolverFamily::xsyevd;
  const bool geometry_changed = first_setup || plan.cached_positions != host.positions;
  const bool all_systems_warm = std::all_of(host.warm_mask.begin(), host.warm_mask.end(),
                                            [](std::uint8_t value) { return value != 0; });
  const bool any_system_warm = std::any_of(host.warm_mask.begin(), host.warm_mask.end(),
                                           [](std::uint8_t value) { return value != 0; });
  // Density residency and the previous-energy baseline are deliberately
  // independent. A fixed warm start can reuse its frozen energy seed after a
  // prior replay advanced the device density. Geometry remains part of a
  // resident-density hit because applying an external dm0 also renormalizes
  // its electron trace against the geometry-dependent overlap matrix.
  const bool device_resident_density_hit = all_systems_warm &&
                                           plan.resident_warm_positions == host.positions &&
                                           plan.resident_warm_density == host.warm_density;
  const bool resident_energy_baseline_hit = device_resident_density_hit &&
                                            plan.resident_warm_positions == host.positions &&
                                            plan.resident_previous_energy.size() == batch_size;
  const bool frozen_energy_baseline_hit = all_systems_warm && !plan.warm_start_updates_enabled &&
                                          plan.frozen_warm_density == host.warm_density &&
                                          plan.frozen_warm_positions == host.positions &&
                                          plan.frozen_previous_energy.size() == batch_size;
  const bool cached_energy_baseline_hit =
      frozen_energy_baseline_hit || resident_energy_baseline_hit;
  // Copy the tiny seed vector locally before invalidating residency. Any
  // early CUDA or validation failure below may have partially changed the
  // device density; only a fully successful execution republishes it.
  std::vector<double> host_previous_energy_seed;
  if (frozen_energy_baseline_hit) {
    host_previous_energy_seed = plan.frozen_previous_energy;
  } else if (resident_energy_baseline_hit) {
    host_previous_energy_seed = plan.resident_previous_energy;
  }
  plan.resident_warm_positions.clear();
  plan.resident_warm_density.clear();
  plan.resident_previous_energy.clear();
  const bool use_cublas = plan.cublas_enabled && nbf >= kCublasMatrixProductAoThreshold;
  std::size_t reference_base_bytes = 0;
  const std::size_t reference_provider_allowance =
      (use_cublas ? 96ULL << 20 : 0) + (use_cusolver ? 96ULL << 20 : 0);
  if (options.export_physical_reference) {
    reference_base_bytes = reference_detail::base_capacity(layout.bytes, host, matrix_elements,
                                                           reference_provider_allowance,
                                                           options.reference_memory_budget_bytes);
    resources.reference_peak_bytes_ = reference_base_bytes;
  }
  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  if (quartet_direct || options.export_physical_reference) {
    // High-order direct ERI recurrences use a bounded per-thread local
    // workspace.  CUDA's default stack limit is only 1 KiB, which is enough
    // for s/p/d low-order tiles but lets d/f quartets fault with an apparent
    // local-memory out-of-bounds access.  Reserve a generous fixed ceiling
    // once per device; low-order kernels do not consume it.
    std::size_t stack_limit = 0;
    cuda_error = cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize);
    if (cuda_error == cudaSuccess && stack_limit < kDirectCudaStackLimitBytes) {
      cuda_error = cudaDeviceSetLimit(cudaLimitStackSize, kDirectCudaStackLimitBytes);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (first_setup && quartet_direct) {
    int multiprocessor_count = 0;
    cuda_error =
        cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device_id);
    if (cuda_error != cudaSuccess || multiprocessor_count <= 0 ||
        static_cast<unsigned>(multiprocessor_count) >
            std::numeric_limits<unsigned>::max() / kPersistentQuartetWarpsPerMultiprocessor) {
      fill_global_failure(outputs, cuda_error == cudaSuccess ? VIBEQC_STATUS_INVALID_ARGUMENT
                                                             : cuda_status(cuda_error));
      return outputs;
    }
    plan.persistent_quartet_worker_blocks =
        static_cast<unsigned>(multiprocessor_count) * kPersistentQuartetWarpsPerMultiprocessor;
  }
  cublasStatus_t blas_error = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t solver_error = CUSOLVER_STATUS_SUCCESS;
  if (first_setup) {
    std::unique_lock<std::mutex> allocation_lock(runtime::allocation_measurement_mutex);
    if ((cuda_error = cudaStreamCreateWithFlags(&resources.stream_, cudaStreamNonBlocking)) !=
            cudaSuccess ||
        (cuda_error = runtime::resource_cuda_malloc_async(&resources.arena_, layout.bytes,
                                                          resources.stream_)) != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (quartet_direct &&
        (cuda_error = runtime::resource_cuda_malloc_async(&resources.direct_tile_validation_,
                                                          sizeof(DirectTileValidationRecord),
                                                          resources.stream_)) != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    runtime::sample_cuda_arena_capacity(layout.bytes);
    const auto provider_before =
        options.export_physical_reference ? reference_detail::free_bytes(resources.stream_) : 0;
    if (use_cublas) {
      blas_error = cublasCreate(&resources.blas_);
      if (blas_error == CUBLAS_STATUS_SUCCESS) {
        blas_error = cublasSetStream(resources.blas_, resources.stream_);
      }
      if (blas_error == CUBLAS_STATUS_SUCCESS) {
        blas_error = cublasSetPointerMode(resources.blas_, CUBLAS_POINTER_MODE_HOST);
      }
      if (blas_error != CUBLAS_STATUS_SUCCESS) {
        plan.retry_without_cublas = true;
        fill_global_failure(outputs, blas_status(blas_error));
        return outputs;
      }
    }
    if (use_cusolver) {
      solver_error = cusolverDnCreate(&resources.solver_);
      if (solver_error != CUSOLVER_STATUS_SUCCESS ||
          (solver_error = cusolverDnSetStream(resources.solver_, resources.stream_)) !=
              CUSOLVER_STATUS_SUCCESS) {
        fill_global_failure(outputs, solver_status(solver_error));
        return outputs;
      }
      if (use_jacobi) {
        if ((solver_error = cusolverDnCreateSyevjInfo(&resources.jacobi_)) !=
                CUSOLVER_STATUS_SUCCESS ||
            (solver_error = cusolverDnXsyevjSetTolerance(resources.jacobi_, 1.0e-13)) !=
                CUSOLVER_STATUS_SUCCESS ||
            (solver_error = cusolverDnXsyevjSetMaxSweeps(resources.jacobi_, 100)) !=
                CUSOLVER_STATUS_SUCCESS ||
            (solver_error = cusolverDnXsyevjSetSortEig(resources.jacobi_, 1)) !=
                CUSOLVER_STATUS_SUCCESS) {
          fill_global_failure(outputs, solver_status(solver_error));
          return outputs;
        }
      } else if ((solver_error = cusolverDnCreateParams(&resources.solver_parameters_)) !=
                 CUSOLVER_STATUS_SUCCESS) {
        fill_global_failure(outputs, solver_status(solver_error));
        return outputs;
      }
    }
    if (options.export_physical_reference) {
      resources.provider_retained_bytes_ = reference_detail::retained_bytes(
          resources.stream_, provider_before, reference_provider_allowance);
    }
  }

  auto atom_offsets = arena_pointer<std::int64_t>(resources.arena_, layout.atom_offsets);
  auto atom_systems = arena_pointer<std::int32_t>(resources.arena_, layout.atom_systems);
  auto atomic_numbers = arena_pointer<std::int32_t>(resources.arena_, layout.atomic_numbers);
  auto positions = arena_pointer<double>(resources.arena_, layout.positions);
  auto system_shell_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_offsets);
  auto shell_atoms = arena_pointer<std::int32_t>(resources.arena_, layout.shell_atoms);
  auto shell_angular = arena_pointer<std::uint8_t>(resources.arena_, layout.shell_angular);
  auto shell_ao_offsets = arena_pointer<std::int64_t>(resources.arena_, layout.shell_ao_offsets);
  auto shell_direct_ao_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.shell_direct_ao_offsets);
  auto shell_primitive_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.shell_primitive_offsets);
  auto system_shell_pair_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_pair_offsets);
  auto system_shell_quartet_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_quartet_offsets);
  auto system_shell_pair_block_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_pair_block_offsets);
  auto system_shell_pair_block_quartet_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.system_shell_pair_block_quartet_offsets);
  auto shell_pair_systems =
      arena_pointer<std::int32_t>(resources.arena_, layout.shell_pair_systems);
  auto shell_pair_first = arena_pointer<std::int32_t>(resources.arena_, layout.shell_pair_first);
  auto shell_pair_second = arena_pointer<std::int32_t>(resources.arena_, layout.shell_pair_second);
  auto shell_pair_primitive_offsets =
      arena_pointer<std::int64_t>(resources.arena_, layout.shell_pair_primitive_offsets);
  auto shell_primitive_pairs =
      arena_pointer<PrimitivePairData>(resources.arena_, layout.shell_primitive_pairs);
  auto psss_resident_tasks =
      arena_pointer<PsssResidentTask>(resources.arena_, layout.psss_resident_tasks);
  auto psss_resident_ket_pairs =
      arena_pointer<std::uint32_t>(resources.arena_, layout.psss_resident_ket_pairs);
  auto ao_shells = arena_pointer<std::int32_t>(resources.arena_, layout.ao_shells);
  auto ao_term_counts = arena_pointer<std::uint8_t>(resources.arena_, layout.ao_term_counts);
  auto ao_term_angular = arena_pointer<std::uint8_t>(resources.arena_, layout.ao_term_angular);
  auto ao_term_coefficients = arena_pointer<double>(resources.arena_, layout.ao_term_coefficients);
  auto direct_ao_shells = arena_pointer<std::int32_t>(resources.arena_, layout.direct_ao_shells);
  auto direct_ao_angular = arena_pointer<std::uint8_t>(resources.arena_, layout.direct_ao_angular);
  auto direct_ao_coefficients =
      arena_pointer<double>(resources.arena_, layout.direct_ao_coefficients);
  auto ao_to_direct_transform =
      arena_pointer<double>(resources.arena_, layout.ao_to_direct_transform);
  auto primitive_exponents = arena_pointer<double>(resources.arena_, layout.primitive_exponents);
  auto primitive_coefficients =
      arena_pointer<double>(resources.arena_, layout.primitive_coefficients);
  auto occupied = arena_pointer<std::int32_t>(resources.arena_, layout.occupied);
  auto warm_mask = arena_pointer<std::uint8_t>(resources.arena_, layout.warm_mask);
  // Per-item admission gate consumed by the tile compaction and the target
  // refinement. It is only allocated when the plan may run the mixed route.
  std::uint32_t* mixed_precision_item_census =
      mixed_precision_fock
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.mixed_item_census)
          : nullptr;
  auto warm_density = arena_pointer<double>(resources.arena_, layout.warm_density);
  auto warm_invalid = arena_pointer<std::uint8_t>(resources.arena_, layout.warm_invalid);
  auto overlap = arena_pointer<double>(resources.arena_, layout.overlap);
  auto hcore = arena_pointer<double>(resources.arena_, layout.hcore);
  auto eri = arena_pointer<double>(resources.arena_, layout.eri);
  auto schwarz_bounds = arena_pointer<double>(resources.arena_, layout.schwarz_bounds);
  auto direct_density = arena_pointer<double>(resources.arena_, layout.direct_density);
  auto direct_fock = arena_pointer<double>(resources.arena_, layout.direct_fock);
  auto direct_transform_temporary =
      arena_pointer<double>(resources.arena_, layout.direct_transform_temporary);
  auto shell_pair_bounds = arena_pointer<double>(resources.arena_, layout.shell_pair_bounds);
  auto shell_pair_density_bounds =
      arena_pointer<ShellPairDensityBounds>(resources.arena_, layout.shell_pair_density_bounds);
  auto bounded_direct_shell_pair_order =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_shell_pair_order);
  auto bounded_stream_shell_pair_order =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_stream_shell_pair_order);
  auto bounded_stream_pair_class_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_stream_pair_class_offsets);
  auto bounded_stream_topology =
      arena_pointer<GeneratedShellPairStream>(resources.arena_, layout.bounded_stream_topology);
  auto bounded_direct_shell_pair_block_bounds =
      arena_pointer<double>(resources.arena_, layout.bounded_direct_shell_pair_block_bounds);
  auto bounded_direct_system_density_bounds =
      arena_pointer<double>(resources.arena_, layout.bounded_direct_system_density_bounds);
  auto bounded_direct_system_pair_density_bounds =
      arena_pointer<double>(resources.arena_, layout.bounded_direct_system_pair_density_bounds);
  auto bounded_direct_generated_overflow =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_overflow);
  auto bounded_direct_generated_tasks =
      arena_pointer<GeneratedShellTask>(resources.arena_, layout.bounded_direct_generated_tasks);
  auto bounded_direct_generated_task_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_task_counts);
  auto bounded_direct_generated_task_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_task_offsets);
  auto bounded_direct_generated_retry_task_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_direct_generated_retry_task_offsets);
  auto bounded_direct_generated_task_heads =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_task_heads);
  auto bounded_direct_generated_retry_mask =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_retry_mask);
  auto bounded_direct_generated_retry_any =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_direct_generated_retry_any);
  auto bounded_force_signature_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_force_signature_counts);
  auto bounded_force_signature_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_force_signature_offsets);
  auto bounded_force_signature_block_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_force_signature_block_offsets);
  std::uint64_t* bounded_fock_class_timer_starts =
      bounded_fock_class_timing
          ? arena_pointer<std::uint64_t>(resources.arena_, layout.bounded_fock_class_timer_starts)
          : nullptr;
  std::uint64_t* bounded_fock_class_timer_elapsed =
      bounded_fock_class_timing
          ? arena_pointer<std::uint64_t>(resources.arena_, layout.bounded_fock_class_timer_elapsed)
          : nullptr;
  std::uint32_t* bounded_fock_class_timer_launches =
      bounded_fock_class_timing
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.bounded_fock_class_timer_launches)
          : nullptr;
  unsigned long long* bounded_fock_fp64_work_counts =
      bounded_fock_class_timing ? arena_pointer<unsigned long long>(
                                      resources.arena_, layout.bounded_fock_fp64_work_counts)
                                : nullptr;
  unsigned long long* bounded_fock_fp32_work_counts =
      bounded_fock_class_timing ? arena_pointer<unsigned long long>(
                                      resources.arena_, layout.bounded_fock_fp32_work_counts)
                                : nullptr;
  auto active_shell_quartet_tile_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.active_shell_quartet_tile_offsets);
  auto active_shell_quartet_tile_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.active_shell_quartet_tile_counts);
  auto active_shell_quartet_tiles =
      arena_pointer<ActiveShellQuartetTile>(resources.arena_, layout.active_shell_quartet_tiles);
  std::uint32_t* fp32_shell_quartet_tile_offsets =
      mixed_precision_fock
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.fp32_shell_quartet_tile_offsets)
          : nullptr;
  std::uint32_t* fp32_shell_quartet_tile_counts =
      mixed_precision_fock
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.fp32_shell_quartet_tile_counts)
          : nullptr;
  ActiveShellQuartetTile* fp32_shell_quartet_tiles =
      mixed_precision_fock
          ? arena_pointer<ActiveShellQuartetTile>(resources.arena_, layout.fp32_shell_quartet_tiles)
          : nullptr;
  auto shell_class_profile =
      arena_pointer<DeviceShellClassProfileEntry>(resources.arena_, layout.shell_class_profile);
  auto persistent_fock_task_heads =
      arena_pointer<std::uint32_t>(resources.arena_, layout.persistent_fock_task_heads);
  std::uint32_t* fp32_persistent_fock_task_heads =
      mixed_precision_fock
          ? arena_pointer<std::uint32_t>(resources.arena_, layout.fp32_persistent_fock_task_heads)
          : nullptr;
  auto persistent_force_task_heads =
      arena_pointer<std::uint32_t>(resources.arena_, layout.persistent_force_task_heads);
  auto generated_shell_tasks =
      arena_pointer<GeneratedShellTask>(resources.arena_, layout.generated_shell_tasks);
  auto generated_shell_classes =
      arena_pointer<std::uint8_t>(resources.arena_, layout.generated_shell_classes);
  auto generated_shell_task_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_shell_task_offsets);
  auto generated_shell_task_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_shell_task_counts);
  auto generated_shell_task_write_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_shell_task_write_counts);
  auto generated_shell_task_heads =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_shell_task_heads);
  auto generated_low_order_signature_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_low_order_signature_counts);
  auto generated_low_order_signature_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_low_order_signature_offsets);
  auto generated_ppps_resident_tasks = arena_pointer<GeneratedPppsResidentTask>(
      resources.arena_, layout.generated_ppps_resident_tasks);
  // Final force preparation is ordered after the last Fock consumer on the
  // same stream.  Reuse the ppps-sized tail of the ordinary generated-task
  // arena for resident ket records, then let ordinary force preparation
  // overwrite it only after the resident launch completes.  This avoids a
  // multi-gigabyte duplicate queue at the 384-AO endpoint.
  GeneratedShellTask* generated_ppps_resident_ket_tasks = nullptr;
  if (resident_ppps_ket_task_capacity != 0) {
    generated_ppps_resident_ket_tasks =
        generated_shell_tasks + (generated_shell_task_capacity - resident_ppps_ket_task_capacity);
  }
  auto generated_ppps_resident_bra_counts =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_ppps_resident_bra_counts);
  auto generated_ppps_resident_bra_offsets =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generated_ppps_resident_bra_offsets);
  auto generated_ppps_resident_bra_write_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_ppps_resident_bra_write_counts);
  auto generated_ppps_resident_signature_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_ppps_resident_signature_counts);
  auto generated_ppps_resident_signature_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_ppps_resident_signature_offsets);
  // A zero-sized arena slice still has an offset, so do not turn it into a
  // writable pointer when profiling did not allocate per-task signatures.
  std::uint32_t* generated_ppps_resident_signatures =
      shell_class_profiling ? arena_pointer<std::uint32_t>(
                                  resources.arena_, layout.generated_ppps_resident_signatures)
                            : nullptr;
  auto generated_fock_shell_class_mask =
      arena_pointer<std::uint64_t>(resources.arena_, layout.generated_fock_shell_class_mask);
  std::uint64_t* generated_mixed_fock_shell_class_mask =
      mixed_precision_fock ? arena_pointer<std::uint64_t>(
                                 resources.arena_, layout.generated_mixed_fock_shell_class_mask)
                           : nullptr;
  auto generic_order5_tiles =
      arena_pointer<ActiveShellQuartetTile>(resources.arena_, layout.generic_order5_tiles);
  auto generic_order5_tile_count =
      arena_pointer<std::uint32_t>(resources.arena_, layout.generic_order5_tile_count);
  auto ao_pair_first = arena_pointer<std::int32_t>(resources.arena_, layout.ao_pair_first);
  auto ao_pair_second = arena_pointer<std::int32_t>(resources.arena_, layout.ao_pair_second);
  auto nuclear_repulsion = arena_pointer<double>(resources.arena_, layout.nuclear_repulsion);
  auto orthogonalizer = arena_pointer<double>(resources.arena_, layout.orthogonalizer);
  auto temporary = arena_pointer<double>(resources.arena_, layout.temporary);
  auto eigensystem = arena_pointer<double>(resources.arena_, layout.eigensystem);
  auto coefficients = arena_pointer<double>(resources.arena_, layout.coefficients);
  auto eigenvalues = arena_pointer<double>(resources.arena_, layout.eigenvalues);
  auto density = arena_pointer<double>(resources.arena_, layout.density);
  auto next_density = arena_pointer<double>(resources.arena_, layout.next_density);
  auto fock = arena_pointer<double>(resources.arena_, layout.fock);
  auto residual = arena_pointer<double>(resources.arena_, layout.residual);
  auto weighted_density = arena_pointer<double>(resources.arena_, layout.weighted_density);
  auto total_density = arena_pointer<double>(resources.arena_, layout.total_density);
  auto total_weighted_density =
      arena_pointer<double>(resources.arena_, layout.total_weighted_density);
  auto fock_history = arena_pointer<double>(resources.arena_, layout.fock_history);
  auto residual_history = arena_pointer<double>(resources.arena_, layout.residual_history);
  auto diis_linear_system = arena_pointer<double>(resources.arena_, layout.diis_linear_system);
  auto diis_coefficients = arena_pointer<double>(resources.arena_, layout.diis_coefficients);
  auto diis_count = arena_pointer<std::uint32_t>(resources.arena_, layout.diis_count);
  auto diis_head = arena_pointer<std::uint32_t>(resources.arena_, layout.diis_head);
  auto energy = arena_pointer<double>(resources.arena_, layout.energy);
  auto previous_energy = arena_pointer<double>(resources.arena_, layout.previous_energy);
  auto energy_change = arena_pointer<double>(resources.arena_, layout.energy_change);
  auto density_rms = arena_pointer<double>(resources.arena_, layout.density_rms);
  auto forces = arena_pointer<double>(resources.arena_, layout.forces);
  auto active = arena_pointer<std::uint8_t>(resources.arena_, layout.active);
  auto converged = arena_pointer<std::uint8_t>(resources.arena_, layout.converged);
  auto failed = arena_pointer<std::uint8_t>(resources.arena_, layout.failed);
  auto final_fock_reuse_mask =
      arena_pointer<std::uint8_t>(resources.arena_, layout.final_fock_reuse_mask);
  auto final_fock_rebuild_count =
      arena_pointer<std::uint32_t>(resources.arena_, layout.final_fock_rebuild_count);
  auto spin_active = arena_pointer<std::uint8_t>(resources.arena_, layout.spin_active);
  auto iterations = arena_pointer<std::uint32_t>(resources.arena_, layout.iterations);
  auto solver_info = arena_pointer<int>(resources.arena_, layout.solver_info);
  std::uint32_t* inactive_eigensolver_profile_count =
      inactive_eigensolver_profiling
          ? arena_pointer<std::uint32_t>(resources.arena_,
                                         layout.inactive_eigensolver_profile_count)
          : nullptr;
  DeviceInactiveEigensolverProfileEntry* inactive_eigensolver_profile =
      inactive_eigensolver_profiling ? arena_pointer<DeviceInactiveEigensolverProfileEntry>(
                                           resources.arena_, layout.inactive_eigensolver_profile)
                                     : nullptr;
  auto bounded_direct_cursor =
      arena_pointer<unsigned long long>(resources.arena_, layout.bounded_direct_cursor);

  std::vector<std::int32_t> host_pair_first;
  std::vector<std::int32_t> host_pair_second;
  if (first_setup) {
    host_pair_first.reserve(pair_count);
    host_pair_second.reserve(pair_count);
    // Canonical lower-triangle order is stable for the lifetime of a topology
    // plan. Upload it once so one-electron and direct-J/K consumers reuse the
    // same device metadata without rebuilding or decoding pair indices.
    for (std::size_t first = 0; first < nbf; ++first) {
      for (std::size_t second = 0; second <= first; ++second) {
        host_pair_first.push_back(static_cast<std::int32_t>(first));
        host_pair_second.push_back(static_cast<std::int32_t>(second));
      }
    }
  }

  const std::size_t shell_quartet_offset_bytes =
      quartet_direct && !bounded_direct_streaming
          ? plan.shell_quartet_tile_offsets.size() * sizeof(std::uint32_t)
          : 0;
  const std::size_t fp32_shell_quartet_offset_bytes =
      mixed_precision_fock ? plan.fp32_shell_quartet_tile_offsets.size() * sizeof(std::uint32_t)
                           : 0;
  const GeneratedShellPairStream host_bounded_stream_topology{
      static_cast<std::int32_t>(batch_size),
      static_cast<std::uint32_t>(direct_nbf),
      system_shell_offsets,
      system_shell_pair_offsets,
      shell_atoms,
      shell_angular,
      shell_direct_ao_offsets,
      shell_primitive_offsets,
      shell_pair_systems,
      shell_pair_first,
      shell_pair_second,
      bounded_stream_shell_pair_order,
      bounded_stream_pair_class_offsets,
      shell_pair_bounds,
      reinterpret_cast<const detail::GeneratedShellPairDensityBounds*>(shell_pair_density_bounds),
      bounded_direct_system_density_bounds,
      bounded_direct_system_pair_density_bounds,
      bounded_direct_generated_overflow,
      active};
  const std::pair<const void*, std::pair<void*, std::size_t>> static_uploads[] = {
      {host.atom_offsets.data(), {atom_offsets, host.atom_offsets.size() * sizeof(std::int64_t)}},
      {host.atom_systems.data(), {atom_systems, host.atom_systems.size() * sizeof(std::int32_t)}},
      {host.atomic_numbers.data(),
       {atomic_numbers, host.atomic_numbers.size() * sizeof(std::int32_t)}},
      {host.system_shell_offsets.data(),
       {system_shell_offsets, host.system_shell_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_atoms.data(), {shell_atoms, host.shell_atoms.size() * sizeof(std::int32_t)}},
      {host.shell_angular.data(),
       {shell_angular, host.shell_angular.size() * sizeof(std::uint8_t)}},
      {host.shell_ao_offsets.data(),
       {shell_ao_offsets, host.shell_ao_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_direct_ao_offsets.data(),
       {shell_direct_ao_offsets, host.shell_direct_ao_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_primitive_offsets.data(),
       {shell_primitive_offsets, host.shell_primitive_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_pair_offsets.data(),
       {system_shell_pair_offsets, host.system_shell_pair_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_quartet_offsets.data(),
       {system_shell_quartet_offsets,
        host.system_shell_quartet_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_pair_block_offsets.data(),
       {system_shell_pair_block_offsets,
        host.system_shell_pair_block_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_pair_block_quartet_offsets.data(),
       {system_shell_pair_block_quartet_offsets,
        host.system_shell_pair_block_quartet_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_pair_systems.data(),
       {shell_pair_systems, host.shell_pair_systems.size() * sizeof(std::int32_t)}},
      {host.shell_pair_first.data(),
       {shell_pair_first, host.shell_pair_first.size() * sizeof(std::int32_t)}},
      {host.shell_pair_second.data(),
       {shell_pair_second, host.shell_pair_second.size() * sizeof(std::int32_t)}},
      {host.shell_pair_primitive_offsets.data(),
       {shell_pair_primitive_offsets,
        quartet_direct ? host.shell_pair_primitive_offsets.size() * sizeof(std::int64_t) : 0}},
      {host.psss_resident_tasks.data(),
       {psss_resident_tasks, quartet_direct && !bounded_direct_streaming
                                 ? host.psss_resident_tasks.size() * sizeof(PsssResidentTask)
                                 : 0}},
      {host.psss_resident_ket_pairs.data(),
       {psss_resident_ket_pairs, quartet_direct && !bounded_direct_streaming
                                     ? host.psss_resident_ket_pairs.size() * sizeof(std::uint32_t)
                                     : 0}},
      {host.ao_shells.data(), {ao_shells, host.ao_shells.size() * sizeof(std::int32_t)}},
      {host.ao_term_counts.data(),
       {ao_term_counts, host.ao_term_counts.size() * sizeof(std::uint8_t)}},
      {host.ao_term_angular.data(),
       {ao_term_angular, host.ao_term_angular.size() * sizeof(std::uint8_t)}},
      {host.ao_term_coefficients.data(),
       {ao_term_coefficients, host.ao_term_coefficients.size() * sizeof(double)}},
      {host.direct_ao_shells.data(),
       {direct_ao_shells, host.direct_ao_shells.size() * sizeof(std::int32_t)}},
      {host.direct_ao_angular.data(),
       {direct_ao_angular, host.direct_ao_angular.size() * sizeof(std::uint8_t)}},
      {host.direct_ao_coefficients.data(),
       {direct_ao_coefficients, host.direct_ao_coefficients.size() * sizeof(double)}},
      {host.ao_to_direct_transform.data(),
       {ao_to_direct_transform, host.ao_to_direct_transform.size() * sizeof(double)}},
      {host.primitive_exponents.data(),
       {primitive_exponents, host.primitive_exponents.size() * sizeof(double)}},
      {host.primitive_coefficients.data(),
       {primitive_coefficients, host.primitive_coefficients.size() * sizeof(double)}},
      {host.occupied.data(), {occupied, host.occupied.size() * sizeof(std::int32_t)}},
      {plan.shell_quartet_tile_offsets.data(),
       {active_shell_quartet_tile_offsets, shell_quartet_offset_bytes}},
      {plan.fp32_shell_quartet_tile_offsets.data(),
       {fp32_shell_quartet_tile_offsets, fp32_shell_quartet_offset_bytes}},
      {plan.bounded_generated_task_offsets.data(),
       {bounded_direct_generated_task_offsets,
        bounded_direct_streaming
            ? plan.bounded_generated_task_offsets.size() * sizeof(std::uint32_t)
            : 0}},
      {plan.bounded_direct_shell_pair_order.data(),
       {bounded_direct_shell_pair_order,
        bounded_direct_streaming
            ? plan.bounded_direct_shell_pair_order.size() * sizeof(std::uint32_t)
            : 0}},
      {plan.bounded_stream_shell_pair_order.data(),
       {bounded_stream_shell_pair_order,
        bounded_direct_streaming
            ? plan.bounded_stream_shell_pair_order.size() * sizeof(std::uint32_t)
            : 0}},
      {plan.bounded_stream_pair_class_offsets.data(),
       {bounded_stream_pair_class_offsets,
        bounded_direct_streaming
            ? plan.bounded_stream_pair_class_offsets.size() * sizeof(std::uint32_t)
            : 0}},
      {&host_bounded_stream_topology,
       {bounded_stream_topology, bounded_direct_streaming ? sizeof(GeneratedShellPairStream) : 0}},
      {host_pair_first.data(), {ao_pair_first, host_pair_first.size() * sizeof(std::int32_t)}},
      {host_pair_second.data(), {ao_pair_second, host_pair_second.size() * sizeof(std::int32_t)}},
  };
  // Registry selections may include f-shell kernels for a batch containing
  // only s/p/d shells.  Intersect with the topology before deciding whether
  // the streaming set is complete; otherwise an impossible class forces an
  // O(N_shell^4) generic bounded scan.
  const std::uint64_t host_present_shell_class_mask =
      quartet_direct ? present_direct_shell_class_mask(host) : 0U;
  const std::uint64_t host_generated_fock_shell_class_mask =
      (generated::enabled_fock_shell_class_mask() & host_present_shell_class_mask) &
      (bounded_direct_streaming ? std::numeric_limits<std::uint64_t>::max()
                                : ~kFixedTopologyGeneratedFockExclusionMask);
  const std::uint64_t host_generated_mixed_fock_shell_class_mask =
      mixed_precision_fock
          ? generated::enabled_mixed_fock_shell_class_mask() &
                host_generated_fock_shell_class_mask & host_present_shell_class_mask
          : 0U;
  const std::uint64_t host_generated_streaming_fock_shell_class_mask =
      host_generated_fock_shell_class_mask & kGeneratedStreamingFockShellClassMask;
  const std::uint64_t host_native_streaming_fock_shell_class_mask =
      host_generated_fock_shell_class_mask & kNativeStreamingFockShellClassMask;
  const std::uint64_t host_uncovered_fock_shell_class_mask =
      host_present_shell_class_mask & ~host_generated_fock_shell_class_mask;
  // Per-item admission: each item divides the certified batch budget with its
  // own mixed-capable census and only enters the mixed route from a validated
  // warm state, because the reserved error bounds the perturbation of a known
  // state rather than of a cold guess. A refused item keeps a zero census, so
  // its tiles stay in the FP64 lists while its neighbors may still use mixed.
  // An explicit diagnostic cutoff stays item agnostic.
  std::vector<std::uint32_t> host_mixed_item_census(batch_size, 0U);
  std::vector<double> host_mixed_item_threshold(batch_size, 0.0);
  if (mixed_precision_fock) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      const MixedPrecisionItemPolicy item = resolve_mixed_precision_item(
          requested_precision_policy, host.warm_mask[system] != 0,
          mixed_precision_system_census[system], options.screening_tolerance);
      if (!item.admitted) continue;
      host_mixed_item_census[system] = item.census;
      host_mixed_item_threshold[system] = item.threshold;
    }
  }
  const std::pair<const void*, std::pair<void*, std::size_t>> dynamic_uploads[] = {
      {host_mixed_item_census.data(),
       {mixed_precision_item_census,
        mixed_precision_fock ? host_mixed_item_census.size() * sizeof(std::uint32_t) : 0}},
      {host.warm_mask.data(),
       {warm_mask, device_resident_density_hit ? 0 : host.warm_mask.size() * sizeof(std::uint8_t)}},
      {host.warm_density.data(),
       {warm_density, device_resident_density_hit ? 0 : host.warm_density.size() * sizeof(double)}},
      {&host_generated_fock_shell_class_mask,
       {generated_fock_shell_class_mask, quartet_direct ? sizeof(std::uint64_t) : 0}},
      {&host_generated_mixed_fock_shell_class_mask,
       {generated_mixed_fock_shell_class_mask, mixed_precision_fock ? sizeof(std::uint64_t) : 0}},
  };
  if (first_setup) {
    for (const auto& upload : static_uploads) {
      const vibeqc_status status = copy_to_device(upload.second.first, upload.first,
                                                  upload.second.second, resources.stream_);
      if (status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
    }
  }
  if (geometry_changed) {
    const vibeqc_status position_status =
        copy_to_device(positions, host.positions.data(), host.positions.size() * sizeof(double),
                       resources.stream_);
    if (position_status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, position_status);
      return outputs;
    }
  }
  for (const auto& upload : dynamic_uploads) {
    const vibeqc_status status =
        copy_to_device(upload.second.first, upload.first, upload.second.second, resources.stream_);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  }
  if (cached_energy_baseline_hit) {
    // The SCF graph initializes previous_energy from this device buffer. A
    // frozen replay therefore restores its original seed explicitly instead
    // of relying on whatever energy the most recent resident density left.
    const vibeqc_status status =
        copy_to_device(energy, host_previous_energy_seed.data(),
                       host_previous_energy_seed.size() * sizeof(double), resources.stream_);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  }

  DeviceBatch device_batch{static_cast<std::int32_t>(batch_size),
                           static_cast<std::int32_t>(nbf),
                           static_cast<std::int32_t>(direct_nbf),
                           static_cast<std::int64_t>(total_atoms),
                           static_cast<std::int64_t>(total_shells),
                           static_cast<std::int64_t>(total_shell_pairs),
                           static_cast<std::int64_t>(total_shell_quartets),
                           static_cast<std::int64_t>(total_shell_pair_blocks),
                           static_cast<std::int64_t>(total_shell_pair_block_quartets),
                           atom_offsets,
                           atom_systems,
                           atomic_numbers,
                           positions,
                           system_shell_offsets,
                           shell_atoms,
                           shell_angular,
                           shell_ao_offsets,
                           shell_direct_ao_offsets,
                           shell_primitive_offsets,
                           system_shell_pair_offsets,
                           system_shell_quartet_offsets,
                           system_shell_pair_block_offsets,
                           system_shell_pair_block_quartet_offsets,
                           shell_pair_systems,
                           shell_pair_first,
                           shell_pair_second,
                           shell_pair_primitive_offsets,
                           shell_primitive_pairs,
                           ao_shells,
                           ao_term_counts,
                           ao_term_angular,
                           ao_term_coefficients,
                           direct_ao_shells,
                           direct_ao_angular,
                           direct_ao_coefficients,
                           ao_to_direct_transform,
                           primitive_exponents,
                           primitive_coefficients,
                           occupied};
  device_batch.generated_psss_weighted = plan.generated_psss_weighted;

  if (quartet_direct && geometry_changed) {
    launch_build_shell_primitive_pair_cache_kernel(
        static_cast<unsigned>(total_shell_pairs), detail::kDirectQuartetThreads, 0,
        resources.stream_, device_batch, shell_primitive_pairs);
    cuda_error = cudaPeekAtLastError();
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }

  if (first_setup && use_cusolver) {
    if (use_jacobi) {
      solver_error = cusolverDnDsyevjBatched_bufferSize(
          resources.solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
          static_cast<int>(nbf), eigensystem, static_cast<int>(nbf), eigenvalues, &plan.lwork,
          resources.jacobi_, static_cast<int>(spin_batch_size));
      resources.solver_workspace_bytes_ = static_cast<std::size_t>(plan.lwork) * sizeof(double);
    } else if (ordinary_eigensolver_family == CudaEigensolverFamily::xsyevd) {
      // Xsyevd is the non-batched counterpart used by GPU4PySCF for large
      // matrices.  Its workspace is independent of the number of systems;
      // launch_solver serializes one call per matrix on the ordinary stream.
      std::size_t device_bytes = 0;
      std::size_t host_bytes = 0;
      solver_error = cusolverDnXsyevd_bufferSize(
          resources.solver_, resources.solver_parameters_, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER, static_cast<std::int64_t>(nbf), CUDA_R_64F, eigensystem,
          static_cast<std::int64_t>(nbf), CUDA_R_64F, eigenvalues, CUDA_R_64F, &device_bytes,
          &host_bytes);
      resources.solver_workspace_bytes_ = device_bytes;
      resources.solver_host_workspace_bytes_ = host_bytes;
    } else {
      // RHF submits batch_size matrices; UHF additionally submits the doubled
      // spin batch. Query both actual capacities because cuSOLVER does not
      // guarantee workspace sizes are monotonic in batch count.
      const std::array<int, 2> capacities{static_cast<int>(batch_size),
                                          static_cast<int>(spin_batch_size)};
      for (const int capacity : capacities) {
        std::size_t device_bytes = 0;
        std::size_t host_bytes = 0;
        solver_error = cusolverDnXsyevBatched_bufferSize(
            resources.solver_, resources.solver_parameters_, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, static_cast<int>(nbf), CUDA_R_64F, eigensystem,
            static_cast<int>(nbf), CUDA_R_64F, eigenvalues, CUDA_R_64F, &device_bytes, &host_bytes,
            capacity);
        if (solver_error != CUSOLVER_STATUS_SUCCESS) break;
        resources.solver_workspace_bytes_ =
            std::max(resources.solver_workspace_bytes_, device_bytes);
        resources.solver_host_workspace_bytes_ =
            std::max(resources.solver_host_workspace_bytes_, host_bytes);
      }
      plan.lwork = 0;
    }
    if (solver_error != CUSOLVER_STATUS_SUCCESS) {
      fill_global_failure(outputs, solver_status(solver_error));
      return outputs;
    }
    if (plan.lwork < 0 || resources.solver_workspace_bytes_ == 0) {
      fill_global_failure(outputs, VIBEQC_STATUS_CUDA_ERROR);
      return outputs;
    }
    if (options.export_physical_reference) {
      resources.reference_peak_bytes_ = reference_detail::check_capacity(
          reference_base_bytes,
          posthf::checked_add(resources.solver_workspace_bytes_,
                              resources.solver_host_workspace_bytes_),
          options.reference_memory_budget_bytes);
    }
    if ((cuda_error = runtime::resource_cuda_malloc_async(
             &resources.solver_workspace_, resources.solver_workspace_bytes_, resources.stream_)) !=
        cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (resources.solver_host_workspace_bytes_ != 0) {
      resources.solver_host_workspace_ = std::malloc(resources.solver_host_workspace_bytes_);
      if (resources.solver_host_workspace_ == nullptr) {
        fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
        return outputs;
      }
    }
    if (use_cublas) {
      blas_error = cublasSetWorkspace(resources.blas_, resources.solver_workspace_,
                                      resources.solver_workspace_bytes_);
      if (blas_error != CUBLAS_STATUS_SUCCESS) {
        plan.retry_without_cublas = true;
        fill_global_failure(outputs, blas_status(blas_error));
        return outputs;
      }
    }
  } else if (first_setup) {
    plan.lwork = 0;
  }
  const int lwork = plan.lwork;

  constexpr unsigned threads = kCaptureSafeKernelThreads;
  constexpr unsigned matrix_reduction_threads = kMatrixReductionThreads;
  const auto blocks_for = [](std::size_t elements) {
    return static_cast<unsigned>((elements + threads - 1) / threads);
  };
  const auto multiply_matrices = [&](const double* left, bool transpose_left, const double* right,
                                     double* output) {
    const vibeqc_status product_status = launch_matrix_product(
        resources.matrix_view(), static_cast<int>(batch_size), static_cast<int>(nbf), left,
        transpose_left, right, active, output, use_cublas);
    if (use_cublas && product_status != VIBEQC_STATUS_SUCCESS) {
      plan.retry_without_cublas = true;
    }
    return product_status;
  };
  const auto multiply_spin_matrices = [&](const double* left, bool left_is_spin,
                                          bool transpose_left, const double* right,
                                          bool right_is_spin, double* output) {
    const vibeqc_status product_status = launch_spin_matrix_product(
        resources.matrix_view(), static_cast<int>(batch_size), 2, static_cast<int>(nbf), left,
        left_is_spin, transpose_left, right, right_is_spin, active, output, use_cublas);
    if (use_cublas && product_status != VIBEQC_STATUS_SUCCESS) {
      plan.retry_without_cublas = true;
    }
    return product_status;
  };
  const auto build_commutator_residual = [&]() -> vibeqc_status {
    // [F, P]S is evaluated as four O(N^3) products.  `temporary` and
    // `eigensystem` are iteration scratch at this point: the former holds the
    // first product until it is folded into `residual`, while the latter is
    // overwritten before DIIS uses it as its effective-Fock output.  Keeping
    // the products in separate launches also lets the existing cuBLAS
    // strided-batched wrapper handle RHF and interleaved UHF layouts alike.
    vibeqc_status product_status = VIBEQC_STATUS_SUCCESS;
    if (unrestricted) {
      product_status = multiply_spin_matrices(fock, true, false, density, true, temporary);
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_spin_matrices(temporary, true, false, overlap, false, residual);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_spin_matrices(overlap, false, false, density, true, eigensystem);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_spin_matrices(eigensystem, true, false, fock, true, temporary);
      }
    } else {
      product_status = multiply_matrices(fock, false, density, temporary);
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_matrices(temporary, false, overlap, residual);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_matrices(overlap, false, density, eigensystem);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_matrices(eigensystem, false, fock, temporary);
      }
    }
    if (product_status != VIBEQC_STATUS_SUCCESS) return product_status;
    launch_subtract_matrix_batches_kernel(
        blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
        static_cast<std::int32_t>(nbf), temporary, active, residual);
    return cuda_status(cudaPeekAtLastError());
  };
  const auto launch_direct_quartet_metadata = [&](const double* density_input,
                                                  bool allow_mixed_precision) -> cudaError_t {
    if (!quartet_direct) return cudaSuccess;
    if (direct_tile_validation && resources.direct_tile_validation_ != nullptr) {
      cudaError_t validation_error =
          cudaMemsetAsync(resources.direct_tile_validation_, 0xff,
                          sizeof(DirectTileValidationRecord), resources.stream_);
      if (validation_error != cudaSuccess) return validation_error;
    }
    const double* quartet_density = density_input;
    if (transformed_direct) {
      launch_transform_density_to_direct_right_kernel(
          blocks_for(spin_rectangular_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), static_cast<std::int32_t>(direct_nbf),
          ao_to_direct_transform, density_input, active, direct_transform_temporary);
      launch_transform_density_to_direct_left_kernel(
          blocks_for(direct_spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), static_cast<std::int32_t>(direct_nbf),
          ao_to_direct_transform, direct_transform_temporary, active, direct_density);
      quartet_density = direct_density;
    }
    if (!bounded_direct_streaming) {
      launch_clear_active_shell_quartet_tile_counts_kernel(
          blocks_for(detail::kDirectQuartetAngularOrderCount), threads, 0, resources.stream_,
          active_shell_quartet_tile_counts, persistent_fock_task_heads,
          fp32_shell_quartet_tile_counts, fp32_persistent_fock_task_heads);
    }
    if (unrestricted) {
      launch_reduce_shell_pair_density_bounds_kernel(
          true, static_cast<unsigned>(total_shell_pairs), threads, 3 * threads * sizeof(double),
          resources.stream_, device_batch, quartet_density, active, shell_pair_density_bounds);
      if (bounded_direct_streaming) {
        launch_reduce_bounded_system_density_bounds_kernel(
            static_cast<unsigned>(batch_size), threads,
            detail::kDirectShellPairClassCount * threads * sizeof(double), resources.stream_,
            device_batch, shell_pair_density_bounds, bounded_direct_system_density_bounds,
            bounded_direct_system_pair_density_bounds);
        return cudaPeekAtLastError();
      }
      launch_compact_active_shell_quartet_tiles_kernel(
          true, DirectScreeningPurpose::Fock, blocks_for(total_shell_quartets), threads, 0,
          resources.stream_, device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          allow_mixed_precision && mixed_precision_fock,
          requested_precision_policy.item_cutoff_ceiling,
          requested_precision_policy.item_budget_error, mixed_precision_item_census,
          fp32_shell_quartet_tile_offsets, fp32_shell_quartet_tile_counts,
          fp32_shell_quartet_tiles);
    } else {
      launch_reduce_shell_pair_density_bounds_kernel(
          false, static_cast<unsigned>(total_shell_pairs), threads, 3 * threads * sizeof(double),
          resources.stream_, device_batch, quartet_density, active, shell_pair_density_bounds);
      if (bounded_direct_streaming) {
        launch_reduce_bounded_system_density_bounds_kernel(
            static_cast<unsigned>(batch_size), threads,
            detail::kDirectShellPairClassCount * threads * sizeof(double), resources.stream_,
            device_batch, shell_pair_density_bounds, bounded_direct_system_density_bounds,
            bounded_direct_system_pair_density_bounds);
        return cudaPeekAtLastError();
      }
      launch_compact_active_shell_quartet_tiles_kernel(
          false, DirectScreeningPurpose::Fock, blocks_for(total_shell_quartets), threads, 0,
          resources.stream_, device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          allow_mixed_precision && mixed_precision_fock,
          requested_precision_policy.item_cutoff_ceiling,
          requested_precision_policy.item_budget_error, mixed_precision_item_census,
          fp32_shell_quartet_tile_offsets, fp32_shell_quartet_tile_counts,
          fp32_shell_quartet_tiles);
    }
    if (direct_tile_validation && resources.direct_tile_validation_ != nullptr) {
      launch_validate_direct_tile_descriptors_kernel(
          blocks_for(plan.total_shell_quartet_tiles), threads, 0, resources.stream_, device_batch,
          active_shell_quartet_tile_offsets, active_shell_quartet_tile_counts,
          active_shell_quartet_tiles, plan.total_shell_quartet_tiles,
          resources.direct_tile_validation_);
    }
    return cudaPeekAtLastError();
  };
  const auto launch_direct_force_compaction = [&]() -> cudaError_t {
    if (!quartet_direct || !force_density_product_screening) {
      return cudaSuccess;
    }
    if (bounded_direct_streaming) {
      // The force kernel reapplies the stronger density-product predicate as
      // it enumerates pair-of-pairs, so no intermediate force queue exists.
      return cudaSuccess;
    }
    // The final Fock/metadata path above has already reduced the selected
    // density in the direct Cartesian AO domain. Reuse those bounds and
    // overwrite the no-longer-needed Fock queue with its force-only subset.
    launch_clear_active_shell_quartet_tile_counts_kernel(
        blocks_for(detail::kDirectQuartetAngularOrderCount), threads, 0, resources.stream_,
        active_shell_quartet_tile_counts, persistent_fock_task_heads,
        fp32_shell_quartet_tile_counts, fp32_persistent_fock_task_heads);
    if (unrestricted) {
      launch_compact_active_shell_quartet_tiles_kernel(
          true, DirectScreeningPurpose::Force, blocks_for(total_shell_quartets), threads, 0,
          resources.stream_, device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, false, 0.0, 0.0, nullptr,
          nullptr, nullptr, nullptr);
    } else {
      launch_compact_active_shell_quartet_tiles_kernel(
          false, DirectScreeningPurpose::Force, blocks_for(total_shell_quartets), threads, 0,
          resources.stream_, device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, false, 0.0, 0.0, nullptr,
          nullptr, nullptr, nullptr);
    }
    if (direct_tile_validation && resources.direct_tile_validation_ != nullptr) {
      cudaError_t validation_error =
          cudaMemsetAsync(resources.direct_tile_validation_, 0xff,
                          sizeof(DirectTileValidationRecord), resources.stream_);
      if (validation_error != cudaSuccess) return validation_error;
      launch_validate_direct_tile_descriptors_kernel(
          blocks_for(plan.total_shell_quartet_tiles), threads, 0, resources.stream_, device_batch,
          active_shell_quartet_tile_offsets, active_shell_quartet_tile_counts,
          active_shell_quartet_tiles, plan.total_shell_quartet_tiles,
          resources.direct_tile_validation_);
    }
    return cudaPeekAtLastError();
  };
  std::size_t bounded_fock_kernel_count = 0;
  const generated::ShellKernelMetadata* bounded_fock_kernels =
      generated::selected_fock_shell_kernels(bounded_fock_kernel_count);
  const auto launch_bounded_streaming_fock =
      [&](bool is_unrestricted, const double* quartet_density, double* quartet_fock,
          bool allow_mixed_precision) -> cudaError_t {
    // Every selected class owns one independent queue head.  Reset the
    // complete fixed-size head array in one asynchronous memset before the
    // class-major launches instead of issuing one host API call per class.
    // The heads are disjoint, so this preserves launch ordering and atomic
    // accumulation semantics while removing serial dispatch overhead from
    // the bounded warm path.
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_heads, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess) return error;
    for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
      const unsigned shell_class = bounded_fock_kernels[kernel_index].shell_class;
      if ((host_generated_streaming_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) ==
          0U) {
        continue;
      }
      if (bounded_fock_class_timing) {
        launch_start_bounded_fock_class_timer_kernel(1, 1, 0, resources.stream_, shell_class,
                                                     bounded_fock_class_timer_starts);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
      }
      error = generated::launch_shell_class_streaming_fock(
          shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
          bounded_stream_topology, device_batch.shell_pair_primitive_offsets,
          device_batch.shell_primitive_pairs, device_batch.direct_ao_coefficients,
          device_batch.positions, options.screening_tolerance,
          allow_mixed_precision && mixed_precision_fock &&
              (host_generated_mixed_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) !=
                  0U,
          mixed_precision_fock_threshold, schwarz_bounds, quartet_density, quartet_fock,
          bounded_direct_generated_task_heads + shell_class,
          bounded_fock_class_timing ? bounded_fock_fp64_work_counts + shell_class : nullptr,
          bounded_fock_class_timing ? bounded_fock_fp32_work_counts + shell_class : nullptr);
      if (error != cudaSuccess) return error;
      if (bounded_fock_class_timing) {
        launch_finish_bounded_fock_class_timer_kernel(
            1, 1, 0, resources.stream_, shell_class, bounded_fock_class_timer_starts,
            bounded_fock_class_timer_elapsed, bounded_fock_class_timer_launches);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
      }
    }
    if ((host_native_streaming_fock_shell_class_mask & kDdddShellClassMask) == 0U) {
      return cudaSuccess;
    }
    // The complete head-array reset above also covers native DDDD.  Do not
    // issue a second class-specific memset here: the native fallback uses the
    // same disjoint head slot as generated classes.
    if (bounded_fock_class_timing) {
      launch_start_bounded_fock_class_timer_kernel(1, 1, 0, resources.stream_, kDdddShellClass,
                                                   bounded_fock_class_timer_starts);
      error = cudaPeekAtLastError();
      if (error != cudaSuccess) return error;
    }
    if (is_unrestricted) {
      bounded_direct_dddd_streaming_kernel<true, DirectScreeningPurpose::Fock, false>
          <<<plan.persistent_quartet_worker_blocks, detail::kDirectQuartetThreads, 0,
             resources.stream_>>>(
              device_batch, bounded_stream_topology, options.screening_tolerance, schwarz_bounds,
              quartet_density, active, quartet_fock,
              bounded_direct_generated_task_heads + kDdddShellClass, nullptr,
              bounded_fock_class_timing ? bounded_fock_fp64_work_counts + kDdddShellClass
                                        : nullptr);
    } else {
      bounded_direct_dddd_streaming_kernel<false, DirectScreeningPurpose::Fock, false>
          <<<plan.persistent_quartet_worker_blocks, detail::kDirectQuartetThreads, 0,
             resources.stream_>>>(
              device_batch, bounded_stream_topology, options.screening_tolerance, schwarz_bounds,
              quartet_density, active, quartet_fock,
              bounded_direct_generated_task_heads + kDdddShellClass, nullptr,
              bounded_fock_class_timing ? bounded_fock_fp64_work_counts + kDdddShellClass
                                        : nullptr);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (bounded_fock_class_timing) {
      launch_finish_bounded_fock_class_timer_kernel(
          1, 1, 0, resources.stream_, kDdddShellClass, bounded_fock_class_timer_starts,
          bounded_fock_class_timer_elapsed, bounded_fock_class_timer_launches);
      error = cudaPeekAtLastError();
    }
    return error;
  };
  const auto launch_bounded_paged_generated_fock = [&](bool is_unrestricted,
                                                       const double* quartet_density,
                                                       double* quartet_fock) -> cudaError_t {
    // Generated classes use a fixed descriptor arena.  Enumerate their full
    // candidate ordinal domain in disjoint pages and consume each page before
    // reusing the arena.  This is both bounded in memory and exact: unlike the
    // old overflow tail, no first-wave task is revisited by a second scanner.
    const std::uint32_t page_capacity =
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity);
    if (page_capacity == 0U) return cudaErrorInvalidValue;
    for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
      const generated::ShellKernelMetadata& kernel = bounded_fock_kernels[kernel_index];
      const unsigned shell_class = kernel.shell_class;
      if ((host_generated_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) == 0U ||
          (host_native_streaming_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
        continue;
      }
      unsigned high_pair_class = 0U;
      while ((high_pair_class + 1U) * (high_pair_class + 2U) / 2U <= shell_class) {
        ++high_pair_class;
      }
      const unsigned low_pair_class = shell_class - high_pair_class * (high_pair_class + 1U) / 2U;
      const std::uint64_t page_domain =
          bounded_generated_page_range(plan.bounded_stream_pair_class_offsets, plan.batch_size,
                                       high_pair_class, low_pair_class, 0U, page_capacity)
              .candidate_count;
      for (std::uint64_t page_begin = 0U; page_begin < page_domain; page_begin += page_capacity) {
        const BoundedGeneratedPageRange page_range = bounded_generated_page_range(
            plan.bounded_stream_pair_class_offsets, plan.batch_size, high_pair_class,
            low_pair_class, page_begin, page_capacity);
        if (page_range.bra_begin >= page_range.bra_end) continue;
        cudaError_t error = cudaMemsetAsync(bounded_direct_generated_task_counts + shell_class, 0,
                                            sizeof(std::uint32_t), resources.stream_);
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_retry_task_offsets + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error != cudaSuccess) return error;

#define VIBEQC_COMPACT_BOUNDED_FOCK_PAGE(unrestricted_value)                                      \
  launch_compact_bounded_exact_class_force_wave_kernel(                                           \
      unrestricted_value, DirectScreeningPurpose::Fock, plan.persistent_quartet_worker_blocks,    \
      kBoundedDirectThreads, 0, resources.stream_, device_batch, bounded_stream_topology,         \
      shell_class, high_pair_class, low_pair_class, options.screening_tolerance, page_begin,      \
      page_capacity, page_range.bra_begin, page_range.bra_end, high_pair_class == low_pair_class, \
      bounded_direct_generated_tasks, bounded_direct_generated_task_counts + shell_class,         \
      bounded_direct_generated_task_heads + shell_class, nullptr, true, nullptr, nullptr)
        if (is_unrestricted) {
          VIBEQC_COMPACT_BOUNDED_FOCK_PAGE(true);
        } else {
          VIBEQC_COMPACT_BOUNDED_FOCK_PAGE(false);
        }
#undef VIBEQC_COMPACT_BOUNDED_FOCK_PAGE
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
        if (bounded_fock_class_timing) {
          launch_accumulate_fock_precision_work_kernel(
              1, 1, 0, resources.stream_, bounded_direct_generated_task_counts + shell_class,
              bounded_fock_fp64_work_counts + shell_class);
          error = cudaPeekAtLastError();
          if (error != cudaSuccess) return error;
        }
        // The compactor uses the head as a persistent bra scheduler; generated
        // consumers use the same slot as their task scheduler, so reset it
        // after compaction and before launching the page.
        error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                sizeof(std::uint32_t), resources.stream_);
        if (error != cudaSuccess) return error;
        error = generated::launch_shell_class_fock(
            shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks,
            bounded_direct_generated_retry_task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets, device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock,
            bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (error != cudaSuccess) return error;
      }
    }
    return cudaSuccess;
  };
  const auto launch_bounded_generated_fock =
      [&](bool is_unrestricted, const double* quartet_density, double* quartet_fock,
          bool allow_mixed_precision) -> cudaError_t {
    // The bounded Fock path follows the same hard routing invariant as force:
    // every present class must have a generated or native exact consumer.
    // Missing classes are unsupported instead of silently invoking the
    // whole-topology generic evaluator.
    if (host_uncovered_fock_shell_class_mask != 0U && !bounded_direct_aot_only_diagnostic) {
      return cudaErrorNotSupported;
    }
    if (bounded_direct_fock_only_diagnostic) {
      // The fixed-density measurement uses one uniform streaming schedule.
      // Mark every generated class for that consumer so an all-FP64 page does
      // not hide the arithmetic selected by the diagnostic threshold.
      cudaError_t diagnostic_error = cudaMemsetAsync(
          bounded_direct_generated_overflow, 1,
          detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
      if (diagnostic_error != cudaSuccess) return diagnostic_error;
      return launch_bounded_streaming_fock(is_unrestricted, quartet_density, quartet_fock,
                                           allow_mixed_precision);
    }
    if (!bounded_direct_count_diagnostic) {
      // Normal bounded execution uses disjoint exact pages for every
      // generated class.  Keep the legacy count/first-retry machinery below
      // exclusively for diagnostics, where its device readback is useful but
      // no Fock contribution is consumed.
      cudaError_t reset_error = cudaMemsetAsync(
          bounded_direct_generated_overflow, 0,
          detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
      if (reset_error != cudaSuccess) return reset_error;
      cudaError_t paged_error =
          launch_bounded_paged_generated_fock(is_unrestricted, quartet_density, quartet_fock);
      if (paged_error != cudaSuccess) return paged_error;
      return launch_bounded_streaming_fock(is_unrestricted, quartet_density, quartet_fock,
                                           allow_mixed_precision);
    }
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_counts, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(
        bounded_direct_generated_overflow, bounded_fock_kernel_count == 0 ? 1 : 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess || bounded_fock_kernel_count == 0) return error;
    // Even when every selected generated class has a streaming consumer,
    // compact through the shell-pair block gate first.  Directly scanning all
    // shell-pair products is quadratic in the 73,920-pair 768-AO case; the
    // compact route reduces the candidate domain by the fixed 256-pair block
    // factor and leaves streaming only as an overflow fallback.
    error =
        cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(unrestricted_value, task_offsets, selected_classes, \
                                             selected_any)                                       \
  launch_compact_bounded_generated_tasks_kernel(                                                 \
      unrestricted_value, DirectScreeningPurpose::Fock, plan.persistent_quartet_worker_blocks,   \
      kBoundedDirectThreads, 0, resources.stream_, device_batch, options.screening_tolerance,    \
      shell_pair_bounds, shell_pair_density_bounds, bounded_direct_shell_pair_order,             \
      bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, active,      \
      generated_fock_shell_class_mask, 0U, host_native_streaming_fock_shell_class_mask,          \
      selected_classes, selected_any, bounded_direct_cursor, bounded_direct_generated_tasks,     \
      bounded_direct_generated_task_counts, task_offsets, bounded_direct_generated_overflow)
    if (is_unrestricted) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(true, bounded_direct_generated_task_offsets, nullptr,
                                           nullptr);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(false, bounded_direct_generated_task_offsets, nullptr,
                                           nullptr);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    launch_prepare_bounded_generated_retry_kernel(
        1, 1, 0, resources.stream_,
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity),
        bounded_direct_generated_task_offsets, bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads, bounded_direct_generated_overflow,
        bounded_direct_generated_retry_mask, bounded_direct_generated_retry_task_offsets,
        bounded_direct_generated_retry_any, bounded_direct_count_diagnostic);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (bounded_direct_count_diagnostic) return cudaSuccess;
    const auto consume_generated_wave = [&](const std::uint32_t* task_offsets) -> cudaError_t {
      for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const unsigned shell_class = bounded_fock_kernels[kernel_index].shell_class;
        if ((host_generated_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) == 0U) {
          continue;
        }
        cudaError_t launch_error = generated::launch_shell_class_fock(
            shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks, task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets, device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock,
            bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (launch_error != cudaSuccess) return launch_error;
      }
      return cudaSuccess;
    };
    error = consume_generated_wave(bounded_direct_generated_task_offsets);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(bounded_direct_generated_task_counts, 0,
                            detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                            resources.stream_);
    if (error == cudaSuccess) {
      error = cudaMemsetAsync(bounded_direct_generated_overflow, 0,
                              detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                              resources.stream_);
    }
    if (error == cudaSuccess) {
      error =
          cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    }
    if (error != cudaSuccess) return error;
    if (is_unrestricted) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(true, bounded_direct_generated_retry_task_offsets,
                                           bounded_direct_generated_retry_mask,
                                           bounded_direct_generated_retry_any);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(false, bounded_direct_generated_retry_task_offsets,
                                           bounded_direct_generated_retry_mask,
                                           bounded_direct_generated_retry_any);
    }
#undef VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    launch_normalize_bounded_generated_task_counts_kernel(
        blocks_for(detail::kDirectQuartetShellClassCount), threads, 0, resources.stream_,
        bounded_direct_generated_retry_task_offsets, bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads, bounded_direct_generated_overflow, false);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    error = consume_generated_wave(bounded_direct_generated_retry_task_offsets);
    if (error != cudaSuccess) return error;

    return launch_bounded_streaming_fock(is_unrestricted, quartet_density, quartet_fock,
                                         allow_mixed_precision);
  };
  // The exact provider is resolved/validated by run_hf_cuda_bucket_cached.
  // Dense, packed, generated and streamed paths below are execution schedules
  // of that same operator; retain their fused standard-HF kernel ownership.
  const auto launch_fock_builder = [&](const double* density_input,
                                       bool allow_mixed_precision) -> cudaError_t {
    const double* quartet_density = transformed_direct ? direct_density : density_input;
    double* quartet_fock = transformed_direct ? direct_fock : fock;
    if (quartet_direct) {
      cudaError_t metadata_error =
          launch_direct_quartet_metadata(density_input, allow_mixed_precision);
      if (metadata_error != cudaSuccess) return metadata_error;
      // Validation mode intentionally stops after compaction.  Continuing
      // into a consumer would turn a descriptor report into a secondary
      // illegal access and would obscure whether the queue itself is valid.
      if (direct_tile_validation && resources.direct_tile_validation_ != nullptr) {
        return cudaSuccess;
      }
    }
    if (quartet_direct && transformed_direct) {
      launch_clear_active_matrices_kernel(
          blocks_for(direct_spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(direct_nbf), active, direct_fock);
    }
    if (quartet_direct) {
      if (plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder] != 0) {
        cudaError_t compact_error =
            cudaMemsetAsync(generic_order5_tile_count, 0, sizeof(std::uint32_t), resources.stream_);
        if (compact_error != cudaSuccess) return compact_error;
        launch_compact_generic_order5_tiles_kernel(
            blocks_for(plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder]), threads,
            0, resources.stream_, device_batch,
            active_shell_quartet_tile_counts + kGenericOrderFiveAngularOrder,
            active_shell_quartet_tiles +
                plan.shell_quartet_tile_offsets[kGenericOrderFiveAngularOrder],
            0U, generated_fock_shell_class_mask, generic_order5_tile_count, generic_order5_tiles);
        compact_error = cudaPeekAtLastError();
        if (compact_error != cudaSuccess) return compact_error;
      }
    }
    if (unrestricted && persistent_eri) {
      build_uhf_fock_kernel<<<blocks_for(spin_matrix_elements), threads, 0, resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf), hcore, eri,
          density_input, active, fock);
    } else if (unrestricted && quartet_direct) {
      if (!transformed_direct) {
        launch_initialize_direct_fock_kernel(blocks_for(spin_matrix_elements), threads, 0,
                                             resources.stream_,
                                             static_cast<std::int32_t>(batch_size), 2,
                                             static_cast<std::int32_t>(nbf), hcore, active, fock);
      }
      if (bounded_direct_streaming) {
        cudaError_t streaming_error = launch_bounded_generated_fock(
            true, quartet_density, quartet_fock, allow_mixed_precision);
        if (streaming_error != cudaSuccess || bounded_direct_count_diagnostic) {
          return streaming_error;
        }
        if (bounded_direct_aot_only_diagnostic) return cudaSuccess;
      } else {
        cudaError_t generated_error = launch_generated_shell_class_focks(
            resources.stream_, plan.total_shell_quartet_tiles, plan.generated_shell_task_capacity,
            plan.shell_quartet_tile_capacities, active_shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles, generated_shell_tasks,
            generated_shell_classes, generated_shell_task_offsets, generated_shell_task_counts,
            generated_shell_task_write_counts, generated_shell_task_heads,
            generated_fock_shell_class_mask, plan.persistent_quartet_worker_blocks, true,
            options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock);
        if (generated_error != cudaSuccess) return generated_error;
        launch_angular_fock_quartets<true>(
            resources.stream_, plan.shell_quartet_tile_capacities, plan.shell_quartet_tile_offsets,
            device_batch, active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generic_order5_tile_count, generic_order5_tiles, persistent_fock_task_heads,
            plan.persistent_quartet_worker_blocks, options.screening_tolerance, schwarz_bounds,
            quartet_density, active, quartet_fock, generated_fock_shell_class_mask);
        if (allow_mixed_precision && mixed_precision_fock) {
          generated_error = launch_generated_shell_class_mixed_focks(
              resources.stream_, plan.fp32_shell_quartet_tile_capacity,
              plan.generated_shell_task_capacity, plan.shell_quartet_tile_capacities,
              fp32_shell_quartet_tile_offsets, device_batch, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles, generated_shell_tasks, generated_shell_classes,
              generated_shell_task_offsets, generated_shell_task_counts,
              generated_shell_task_write_counts, generated_shell_task_heads,
              generated_mixed_fock_shell_class_mask, plan.persistent_quartet_worker_blocks, true,
              options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock);
          if (generated_error != cudaSuccess) return generated_error;
          launch_angular_fock_quartets<true, MixedPrecisionFloat>(
              resources.stream_, plan.shell_quartet_tile_capacities,
              plan.fp32_shell_quartet_tile_offsets, device_batch, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles, nullptr, nullptr, fp32_persistent_fock_task_heads,
              plan.persistent_quartet_worker_blocks, options.screening_tolerance, schwarz_bounds,
              quartet_density, active, quartet_fock, generated_mixed_fock_shell_class_mask);
        }
      }
    } else if (unrestricted) {
      build_uhf_fock_direct_packed_kernel<<<static_cast<unsigned>(spin_matrix_elements), threads,
                                            threads * sizeof(double), resources.stream_>>>(
          device_batch, options.screening_tolerance, hcore, ao_pair_first, ao_pair_second,
          pair_count, schwarz_bounds, density_input, active, fock);
    } else if (persistent_eri) {
      build_fock_kernel<<<blocks_for(matrix_elements), threads, 0, resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf), hcore, eri,
          density_input, active, fock);
    } else if (quartet_direct) {
      if (!transformed_direct) {
        launch_initialize_direct_fock_kernel(blocks_for(matrix_elements), threads, 0,
                                             resources.stream_,
                                             static_cast<std::int32_t>(batch_size), 1,
                                             static_cast<std::int32_t>(nbf), hcore, active, fock);
      }
      if (bounded_direct_streaming) {
        cudaError_t streaming_error = launch_bounded_generated_fock(
            false, quartet_density, quartet_fock, allow_mixed_precision);
        if (streaming_error != cudaSuccess || bounded_direct_count_diagnostic) {
          return streaming_error;
        }
        if (bounded_direct_aot_only_diagnostic) return cudaSuccess;
      } else {
        cudaError_t generated_error = launch_generated_shell_class_focks(
            resources.stream_, plan.total_shell_quartet_tiles, plan.generated_shell_task_capacity,
            plan.shell_quartet_tile_capacities, active_shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles, generated_shell_tasks,
            generated_shell_classes, generated_shell_task_offsets, generated_shell_task_counts,
            generated_shell_task_write_counts, generated_shell_task_heads,
            generated_fock_shell_class_mask, plan.persistent_quartet_worker_blocks, false,
            options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock);
        if (generated_error != cudaSuccess) return generated_error;
        launch_angular_fock_quartets<false>(
            resources.stream_, plan.shell_quartet_tile_capacities, plan.shell_quartet_tile_offsets,
            device_batch, active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generic_order5_tile_count, generic_order5_tiles, persistent_fock_task_heads,
            plan.persistent_quartet_worker_blocks, options.screening_tolerance, schwarz_bounds,
            quartet_density, active, quartet_fock, generated_fock_shell_class_mask);
        if (allow_mixed_precision && mixed_precision_fock) {
          generated_error = launch_generated_shell_class_mixed_focks(
              resources.stream_, plan.fp32_shell_quartet_tile_capacity,
              plan.generated_shell_task_capacity, plan.shell_quartet_tile_capacities,
              fp32_shell_quartet_tile_offsets, device_batch, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles, generated_shell_tasks, generated_shell_classes,
              generated_shell_task_offsets, generated_shell_task_counts,
              generated_shell_task_write_counts, generated_shell_task_heads,
              generated_mixed_fock_shell_class_mask, plan.persistent_quartet_worker_blocks, false,
              options.screening_tolerance, schwarz_bounds, quartet_density, quartet_fock);
          if (generated_error != cudaSuccess) return generated_error;
          launch_angular_fock_quartets<false, MixedPrecisionFloat>(
              resources.stream_, plan.shell_quartet_tile_capacities,
              plan.fp32_shell_quartet_tile_offsets, device_batch, fp32_shell_quartet_tile_counts,
              fp32_shell_quartet_tiles, nullptr, nullptr, fp32_persistent_fock_task_heads,
              plan.persistent_quartet_worker_blocks, options.screening_tolerance, schwarz_bounds,
              quartet_density, active, quartet_fock, generated_mixed_fock_shell_class_mask);
        }
      }
    } else {
      build_fock_direct_packed_kernel<<<static_cast<unsigned>(matrix_elements), threads,
                                        threads * sizeof(double), resources.stream_>>>(
          device_batch, options.screening_tolerance, hcore, ao_pair_first, ao_pair_second,
          pair_count, schwarz_bounds, density_input, active, fock);
    }
    if (quartet_direct && transformed_direct) {
      launch_transform_direct_fock_left_kernel(
          blocks_for(spin_rectangular_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), static_cast<std::int32_t>(direct_nbf),
          ao_to_direct_transform, direct_fock, active, direct_transform_temporary);
      launch_transform_direct_fock_right_kernel(
          blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), static_cast<std::int32_t>(direct_nbf),
          ao_to_direct_transform, direct_transform_temporary, hcore, active, fock);
    }
    return cudaPeekAtLastError();
  };
  launch_initialize_state_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                 static_cast<std::int32_t>(batch_size), cached_energy_baseline_hit,
                                 energy, active, converged, failed, iterations, previous_energy,
                                 energy_change, density_rms, diis_count, diis_head);
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  if (geometry_changed) {
    cuda_error = launch_generated_one_electron_values(
        one_electron_view(device_batch), ao_pair_first, ao_pair_second, pair_count,
        plan.one_electron_value_mapping, overlap, hcore, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (persistent_eri) {
      build_eri_kernel<<<blocks_for(eri_elements), threads, 0, resources.stream_>>>(device_batch,
                                                                                    eri);
    } else {
      if (quartet_direct) {
        // The bounded route needs both AO-pair and shell-pair bounds.  Keep
        // the dense AO-pair grid (rather than one block per shell pair) and
        // reduce shell maxima atomically while each diagonal ERI is live.
        cuda_error = cudaMemsetAsync(shell_pair_bounds, 0, total_shell_pairs * sizeof(double),
                                     resources.stream_);
        if (cuda_error == cudaSuccess) {
          build_schwarz_and_shell_pair_bounds_packed_kernel<<<
              static_cast<unsigned>(direct_pair_elements), kSchwarzThreads, 0, resources.stream_>>>(
              device_batch, direct_pair_count, schwarz_bounds, shell_pair_bounds);
          cuda_error = cudaPeekAtLastError();
        }
        if (cuda_error != cudaSuccess) {
          fill_global_failure(outputs, cuda_status(cuda_error));
          return outputs;
        }
        if (bounded_direct_streaming) {
          // Keep every system and class segment Schwarz-descending for the
          // current geometry.  Generic blocks then group similar work, while
          // generated resident-bra streams may safely stop at the first full
          // ket chunk below the geometry-only gate.
          std::vector<double> host_shell_pair_bounds(total_shell_pairs);
          cuda_error = cudaMemcpyAsync(host_shell_pair_bounds.data(), shell_pair_bounds,
                                       total_shell_pairs * sizeof(double), cudaMemcpyDeviceToHost,
                                       resources.stream_);
          if (cuda_error == cudaSuccess) {
            cuda_error = cudaStreamSynchronize(resources.stream_);
          }
          if (cuda_error != cudaSuccess) {
            fill_global_failure(outputs, cuda_status(cuda_error));
            return outputs;
          }
          for (std::size_t system = 0; system < batch_size; ++system) {
            const std::size_t pair_begin =
                static_cast<std::size_t>(host.system_shell_pair_offsets[system]);
            const std::size_t pair_end =
                static_cast<std::size_t>(host.system_shell_pair_offsets[system + 1]);
            std::stable_sort(plan.bounded_direct_shell_pair_order.begin() + pair_begin,
                             plan.bounded_direct_shell_pair_order.begin() + pair_end,
                             [&](std::uint32_t first, std::uint32_t second) {
                               return host_shell_pair_bounds[first] >
                                      host_shell_pair_bounds[second];
                             });
          }
          const std::size_t class_stride = batch_size + 1U;
          for (std::size_t pair_class = 0; pair_class < detail::kDirectShellPairClassCount;
               ++pair_class) {
            for (std::size_t system = 0; system < batch_size; ++system) {
              const std::size_t segment_begin =
                  plan.bounded_stream_pair_class_offsets[pair_class * class_stride + system];
              const std::size_t segment_end =
                  plan.bounded_stream_pair_class_offsets[pair_class * class_stride + system + 1U];
              std::stable_sort(plan.bounded_stream_shell_pair_order.begin() + segment_begin,
                               plan.bounded_stream_shell_pair_order.begin() + segment_end,
                               [&](std::uint32_t first, std::uint32_t second) {
                                 return host_shell_pair_bounds[first] >
                                        host_shell_pair_bounds[second];
                               });
            }
          }
          const vibeqc_status order_upload_status = copy_to_device(
              bounded_direct_shell_pair_order, plan.bounded_direct_shell_pair_order.data(),
              total_shell_pairs * sizeof(std::uint32_t), resources.stream_);
          if (order_upload_status != VIBEQC_STATUS_SUCCESS) {
            fill_global_failure(outputs, order_upload_status);
            return outputs;
          }
          const vibeqc_status stream_order_upload_status = copy_to_device(
              bounded_stream_shell_pair_order, plan.bounded_stream_shell_pair_order.data(),
              total_shell_pairs * sizeof(std::uint32_t), resources.stream_);
          if (stream_order_upload_status != VIBEQC_STATUS_SUCCESS) {
            fill_global_failure(outputs, stream_order_upload_status);
            return outputs;
          }
        }
        if (bounded_direct_streaming) {
          launch_reduce_bounded_shell_pair_block_bounds_kernel(
              static_cast<unsigned>(total_shell_pair_blocks), threads, threads * sizeof(double),
              resources.stream_, device_batch, bounded_direct_shell_pair_order, shell_pair_bounds,
              bounded_direct_shell_pair_block_bounds);
        }
      } else if (options.export_physical_reference) {
        // Every unscreened AO pair satisfies 0*0 >= 0. No Cartesian Schwarz
        // array is needed by this bounded public-AO matrix-direct path.
        cuda_error =
            cudaMemsetAsync(schwarz_bounds, 0, matrix_elements * sizeof(double), resources.stream_);
        if (cuda_error != cudaSuccess) {
          fill_global_failure(outputs, cuda_status(cuda_error));
          return outputs;
        }
      } else {
        build_schwarz_bounds_packed_kernel<<<blocks_for(direct_pair_elements), threads, 0,
                                             resources.stream_>>>(device_batch, direct_pair_count,
                                                                  schwarz_bounds);
      }
    }
    launch_build_nuclear_repulsion_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                          device_batch, nuclear_repulsion);

    launch_copy_matrix_kernel(blocks_for(matrix_elements), threads, 0, resources.stream_,
                              matrix_elements, overlap, eigensystem);
    status = launch_solver(resources.eigensolver_view(), ordinary_eigensolver_family,
                           static_cast<int>(nbf), static_cast<int>(batch_size), eigensystem,
                           temporary, eigenvalues, lwork, solver_info, active);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    launch_inspect_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                 static_cast<std::int32_t>(batch_size), solver_info, active, failed,
                                 converged);
    launch_build_orthogonalizer_kernel(blocks_for(matrix_elements), threads, 0, resources.stream_,
                                       static_cast<std::int32_t>(batch_size),
                                       static_cast<std::int32_t>(nbf), eigensystem, eigenvalues,
                                       active, orthogonalizer, failed);
  }

  // A valid warm density supersedes the core-Hamiltonian guess. Homogeneous
  // warm replay can therefore skip its transforms, eigensolve, and density
  // construction without changing mixed warm/cold bucket semantics.
  if (!all_systems_warm) {
    status = multiply_matrices(hcore, false, orthogonalizer, temporary);
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = multiply_matrices(orthogonalizer, true, temporary, eigensystem);
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    status = launch_solver(resources.eigensolver_view(), ordinary_eigensolver_family,
                           static_cast<int>(nbf), static_cast<int>(batch_size), eigensystem,
                           temporary, eigenvalues, lwork, solver_info, active);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    launch_inspect_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                 static_cast<std::int32_t>(batch_size), solver_info, active, failed,
                                 converged);
    if (unrestricted) {
      status = multiply_matrices(orthogonalizer, false, eigensystem, temporary);
      if (status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
      launch_broadcast_spin_matrix_kernel(blocks_for(spin_matrix_elements), threads, 0,
                                          resources.stream_, static_cast<std::int32_t>(batch_size),
                                          2, static_cast<std::int32_t>(nbf), temporary, active,
                                          coefficients);
      launch_mix_open_shell_guess_kernel(blocks_for(batch_size * nbf), threads, 0,
                                         resources.stream_, static_cast<std::int32_t>(batch_size),
                                         static_cast<std::int32_t>(nbf), occupied, active,
                                         coefficients);
      launch_build_spin_density_kernel(blocks_for(spin_matrix_elements), threads, 0,
                                       resources.stream_, static_cast<std::int32_t>(batch_size), 2,
                                       static_cast<std::int32_t>(nbf), occupied, coefficients,
                                       active, density);
    } else {
      status = multiply_matrices(orthogonalizer, false, eigensystem, coefficients);
      if (status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
      launch_build_density_kernel(blocks_for(matrix_elements), threads, 0, resources.stream_,
                                  static_cast<std::int32_t>(batch_size),
                                  static_cast<std::int32_t>(nbf), occupied, coefficients, active,
                                  density);
    }
  }
  std::vector<std::uint8_t> host_warm_invalid;
  if (!device_resident_density_hit && any_system_warm) {
    // The normalization kernel also performs the CPU-equivalent metric trace
    // check.  Fence only this exceptional input-validation path; a resident
    // replay skips both the host upload and this O(N^2) setup scan.
    cuda_error =
        cudaMemsetAsync(warm_invalid, 0, batch_size * sizeof(std::uint8_t), resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (unrestricted) {
      launch_apply_uhf_warm_density_kernel(
          static_cast<unsigned>(batch_size), kWarmDensityThreads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf), occupied,
          warm_mask, warm_density, overlap, density, warm_invalid);
    } else {
      launch_apply_warm_density_kernel(static_cast<unsigned>(batch_size), kWarmDensityThreads, 0,
                                       resources.stream_, static_cast<std::int32_t>(batch_size),
                                       static_cast<std::int32_t>(nbf), occupied, warm_mask,
                                       warm_density, overlap, density, warm_invalid);
    }
    host_warm_invalid.resize(batch_size, 0);
    cuda_error =
        cudaMemcpyAsync(host_warm_invalid.data(), warm_invalid, batch_size * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (std::any_of(host_warm_invalid.begin(), host_warm_invalid.end(),
                    [](std::uint8_t value) { return value != 0; })) {
      // The validation kernel has already symmetrized the candidate in the
      // resident density buffer. Residency was invalidated before execution,
      // and a rejected trace must not publish a replacement cache entry. A
      // separately frozen valid dm0/energy pair remains safe to replay.
      fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
  }

  const EigensolverProfileLaunch graph_eigensolver_profile{
      static_cast<std::int32_t>(batch_size),
      active,
      use_cublas,
      static_cast<std::uint32_t>(options.max_iterations),
      inactive_eigensolver_profile_count,
      inactive_eigensolver_profile,
  };
  const EigensolverProfileLaunch* graph_eigensolver_profile_pointer =
      inactive_eigensolver_profiling ? &graph_eigensolver_profile : nullptr;

  const bool fock_only_iteration = bounded_direct_fock_only_diagnostic && bounded_direct_streaming;
  // CUDA 12.9 rejects XsyevBatched capture for matrices above 512 AOs.  Keep
  // the expensive Fock and matrix work in two reusable Graphs while the host
  // inserts a GPU4PySCF-style ordinary eigensolver call between them and
  // checks the tiny physical active mask once per SCF iteration.
  const bool split_provider_iteration =
      !fock_only_iteration &&
      (ordinary_eigensolver_family == CudaEigensolverFamily::xsyev_batched ||
       ordinary_eigensolver_family == CudaEigensolverFamily::xsyevd) &&
      graph_eigensolver_family != ordinary_eigensolver_family;

  const auto launch_iteration_pre_eigensolver = [&](bool allow_mixed_precision) -> vibeqc_status {
    const cudaError_t fock_error = launch_fock_builder(density, allow_mixed_precision);
    if (fock_error != cudaSuccess) return cuda_status(fock_error);
    if (fock_only_iteration) return VIBEQC_STATUS_SUCCESS;

    vibeqc_status iteration_status = VIBEQC_STATUS_SUCCESS;
    if (unrestricted) {
      launch_compute_uhf_energy_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads,
                                       0, resources.stream_, static_cast<std::int32_t>(batch_size),
                                       static_cast<std::int32_t>(nbf), density, hcore, fock,
                                       nuclear_repulsion, active, energy);
      iteration_status = build_commutator_residual();
    } else {
      launch_compute_energy_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
                                   resources.stream_, static_cast<std::int32_t>(batch_size),
                                   static_cast<std::int32_t>(nbf), density, hcore, fock,
                                   nuclear_repulsion, active, energy);
      iteration_status = build_commutator_residual();
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;

    launch_update_diis_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
                              resources.stream_, static_cast<std::int32_t>(batch_size),
                              static_cast<std::int32_t>(nbf), unrestricted ? 2 : 1,
                              static_cast<std::uint32_t>(diis_history), fock, residual, active,
                              fock_history, residual_history, diis_linear_system, diis_coefficients,
                              diis_count, diis_head, eigensystem);
    if (unrestricted) {
      iteration_status =
          multiply_spin_matrices(eigensystem, true, false, orthogonalizer, false, temporary);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        iteration_status =
            multiply_spin_matrices(orthogonalizer, false, true, temporary, true, eigensystem);
      }
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        launch_expand_spin_active_kernel(blocks_for(spin_batch_size), threads, 0, resources.stream_,
                                         static_cast<std::int32_t>(batch_size), 2, active,
                                         spin_active);
      }
    } else {
      iteration_status = multiply_matrices(eigensystem, false, orthogonalizer, temporary);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        iteration_status = multiply_matrices(orthogonalizer, true, temporary, eigensystem);
      }
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    return cuda_status(cudaPeekAtLastError());
  };

  const auto launch_iteration_eigensolver = [&](CudaEigensolverFamily family) -> vibeqc_status {
    return launch_solver(resources.eigensolver_view(), family, static_cast<int>(nbf),
                         static_cast<int>(unrestricted ? spin_batch_size : batch_size), eigensystem,
                         temporary, eigenvalues, lwork, solver_info,
                         unrestricted ? spin_active : active, graph_eigensolver_profile_pointer);
  };

  const auto launch_iteration_post_eigensolver = [&](bool append_device_tail) -> vibeqc_status {
    vibeqc_status iteration_status = VIBEQC_STATUS_SUCCESS;
    if (unrestricted) {
      launch_inspect_spin_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                        static_cast<std::int32_t>(batch_size), 2, solver_info,
                                        active, failed, converged);
      iteration_status =
          multiply_spin_matrices(orthogonalizer, false, false, eigensystem, true, coefficients);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        launch_build_spin_density_kernel(blocks_for(spin_matrix_elements), threads, 0,
                                         resources.stream_, static_cast<std::int32_t>(batch_size),
                                         2, static_cast<std::int32_t>(nbf), occupied, coefficients,
                                         active, next_density);
        if (reuse_converged_fock) {
          launch_update_uhf_convergence_kernel(
              true, static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_, static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance, options.density_tolerance,
              quartet_direct, energy, previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        } else {
          launch_update_uhf_convergence_kernel(
              false, static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_, static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance, options.density_tolerance,
              quartet_direct, energy, previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        }
      }
    } else {
      launch_inspect_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                   static_cast<std::int32_t>(batch_size), solver_info, active,
                                   failed, converged);
      iteration_status = multiply_matrices(orthogonalizer, false, eigensystem, coefficients);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        launch_build_density_kernel(blocks_for(matrix_elements), threads, 0, resources.stream_,
                                    static_cast<std::int32_t>(batch_size),
                                    static_cast<std::int32_t>(nbf), occupied, coefficients, active,
                                    next_density);
        if (reuse_converged_fock) {
          launch_update_convergence_kernel(
              true, static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_, static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance, options.density_tolerance,
              quartet_direct, energy, previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        } else {
          launch_update_convergence_kernel(
              false, static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_, static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance, options.density_tolerance,
              quartet_direct, energy, previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        }
      }
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    if (append_device_tail) {
      launch_tail_rhf_loop_kernel(1, 1, 0, resources.stream_, static_cast<std::int32_t>(batch_size),
                                  options.max_iterations, active, iterations);
    }
    return cuda_status(cudaPeekAtLastError());
  };

  if (first_setup) {
    // Graph construction is allocation-permitted setup work. Synchronize once
    // so capture cannot race the initial guess; fixed-topology replays reuse
    // this executable and do not repeat the fence or provider setup.
    cuda_error = cudaStreamSynchronize(resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamBeginCapture(resources.stream_, cudaStreamCaptureModeThreadLocal);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    status = launch_iteration_pre_eigensolver(true);
    if (status == VIBEQC_STATUS_SUCCESS && !fock_only_iteration && !split_provider_iteration) {
      status = launch_iteration_eigensolver(graph_eigensolver_family);
    }
    if (status == VIBEQC_STATUS_SUCCESS && !fock_only_iteration && !split_provider_iteration) {
      status = launch_iteration_post_eigensolver(true);
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      cudaGraph_t abandoned_graph = nullptr;
      (void)cudaStreamEndCapture(resources.stream_, &abandoned_graph);
      if (abandoned_graph != nullptr) (void)cudaGraphDestroy(abandoned_graph);
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs, status);
      return outputs;
    }
    cuda_error = cudaStreamEndCapture(resources.stream_, &resources.iteration_graph_);
    if (status != VIBEQC_STATUS_SUCCESS || cuda_error != cudaSuccess ||
        resources.iteration_graph_ == nullptr) {
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs,
                          status != VIBEQC_STATUS_SUCCESS ? status : cuda_status(cuda_error));
      return outputs;
    }
    cuda_error =
        cudaGraphInstantiate(&resources.iteration_graph_exec_, resources.iteration_graph_,
                             split_provider_iteration ? 0U : cudaGraphInstantiateFlagDeviceLaunch);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaGraphUpload(resources.iteration_graph_exec_, resources.stream_);
    }
    if (cuda_error == cudaSuccess && split_provider_iteration) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaStreamBeginCapture(resources.stream_, cudaStreamCaptureModeThreadLocal);
      }
      if (cuda_error == cudaSuccess) {
        status = launch_iteration_post_eigensolver(false);
      }
      if (cuda_error == cudaSuccess && status == VIBEQC_STATUS_SUCCESS) {
        cuda_error = cudaStreamEndCapture(resources.stream_, &resources.post_eigensolver_graph_);
      } else {
        cudaGraph_t abandoned_graph = nullptr;
        (void)cudaStreamEndCapture(resources.stream_, &abandoned_graph);
        if (abandoned_graph != nullptr) {
          (void)cudaGraphDestroy(abandoned_graph);
        }
      }
      if (cuda_error == cudaSuccess && status == VIBEQC_STATUS_SUCCESS &&
          resources.post_eigensolver_graph_ != nullptr) {
        cuda_error = cudaGraphInstantiate(&resources.post_eigensolver_graph_exec_,
                                          resources.post_eigensolver_graph_, 0U);
      }
      if (cuda_error == cudaSuccess && status == VIBEQC_STATUS_SUCCESS) {
        cuda_error = cudaGraphUpload(resources.post_eigensolver_graph_exec_, resources.stream_);
      }
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (status != VIBEQC_STATUS_SUCCESS || cuda_error != cudaSuccess) {
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs,
                          status != VIBEQC_STATUS_SUCCESS ? status : cuda_status(cuda_error));
      return outputs;
    }
    plan.initialized = true;
  }
  if (inactive_eigensolver_profiling) {
    cuda_error = cudaMemsetAsync(inactive_eigensolver_profile_count, 0, sizeof(std::uint32_t),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(bounded_fock_class_timer_elapsed, 0,
                                 detail::kDirectQuartetShellClassCount * sizeof(std::uint64_t),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(bounded_fock_class_timer_launches, 0,
                                 detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(bounded_fock_fp64_work_counts, 0,
                                 detail::kDirectQuartetShellClassCount * sizeof(unsigned long long),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(bounded_fock_fp32_work_counts, 0,
                                 detail::kDirectQuartetShellClassCount * sizeof(unsigned long long),
                                 resources.stream_);
  }
  if (cuda_error == cudaSuccess && split_provider_iteration) {
    std::vector<std::uint8_t> host_active(batch_size, 1U);
    for (std::uint32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
      cuda_error = cudaGraphLaunch(resources.iteration_graph_exec_, resources.stream_);
      if (cuda_error != cudaSuccess) break;
      status = launch_iteration_eigensolver(ordinary_eigensolver_family);
      if (status != VIBEQC_STATUS_SUCCESS) break;
      cuda_error = cudaGraphLaunch(resources.post_eigensolver_graph_exec_, resources.stream_);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpyAsync(host_active.data(), active, batch_size * sizeof(std::uint8_t),
                                     cudaMemcpyDeviceToHost, resources.stream_);
      }
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaStreamSynchronize(resources.stream_);
      }
      if (cuda_error != cudaSuccess ||
          std::none_of(host_active.begin(), host_active.end(),
                       [](std::uint8_t value) { return value != 0; })) {
        break;
      }
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  } else if (cuda_error == cudaSuccess) {
    cuda_error = cudaGraphLaunch(resources.iteration_graph_exec_, resources.stream_);
  }
  if (cuda_error == cudaSuccess && direct_tile_validation &&
      resources.direct_tile_validation_ != nullptr) {
    cuda_error = cudaStreamSynchronize(resources.stream_);
    DirectTileValidationRecord host_validation{};
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(&host_validation, resources.direct_tile_validation_,
                              sizeof(host_validation), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      if (host_validation.error == kDirectTileValidationNoError) {
        std::fprintf(stderr, "direct-tile-validation error=none\n");
        std::fflush(stderr);
      } else {
        const char* error_name = "none";
        switch (static_cast<DirectTileValidationError>(host_validation.error)) {
          case DirectTileValidationError::count_exceeds_capacity:
            error_name = "count-exceeds-capacity";
            break;
          case DirectTileValidationError::pair_out_of_bounds:
            error_name = "pair-out-of-bounds";
            break;
          case DirectTileValidationError::shell_out_of_bounds:
            error_name = "shell-out-of-bounds";
            break;
          case DirectTileValidationError::tile_out_of_bounds:
            error_name = "tile-out-of-bounds";
            break;
          case DirectTileValidationError::ao_range_invalid:
            error_name = "ao-range-invalid";
            break;
          default:
            break;
        }
        std::fprintf(stderr,
                     "direct-tile-validation error=%s order=%u slot=%u tile=%u "
                     "pairs=(%u,%u) shells=(%d,%d,%d,%d) direct_nbf=%u "
                     "pair_counts=(%u,%u) ao=(%u,%u,%u,%u) count=%u capacity=%u "
                     "partition_begin=%u\n",
                     error_name, host_validation.angular_order, host_validation.slot,
                     host_validation.tile, host_validation.first_pair, host_validation.second_pair,
                     host_validation.shell[0], host_validation.shell[1], host_validation.shell[2],
                     host_validation.shell[3], host_validation.direct_nbf,
                     host_validation.first_pair_count, host_validation.second_pair_count,
                     host_validation.i, host_validation.j, host_validation.k, host_validation.l,
                     host_validation.active_tile_count, host_validation.partition_capacity,
                     host_validation.partition_begin);
        std::fflush(stderr);
      }
    }
  }
  if (cuda_error == cudaSuccess && bounded_direct_count_diagnostic && bounded_direct_streaming) {
    cuda_error = cudaStreamSynchronize(resources.stream_);
    std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_counts{};
    std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_overflow{};
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(host_counts.data(), bounded_direct_generated_task_counts,
                              host_counts.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(host_overflow.data(), bounded_direct_generated_overflow,
                              host_overflow.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      std::fprintf(stderr, "bounded-direct-count purpose=scf-fock capacity=%zu\n",
                   plan.bounded_generated_task_capacity);
      for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const generated::ShellKernelMetadata& kernel = bounded_fock_kernels[kernel_index];
        const std::uint32_t class_capacity =
            plan.bounded_generated_task_offsets[kernel.shell_class + 1U] -
            plan.bounded_generated_task_offsets[kernel.shell_class];
        std::fprintf(stderr, "  %-4s count=%u capacity=%u overflow=%u\n", kernel.name,
                     host_counts[kernel.shell_class], class_capacity,
                     host_overflow[kernel.shell_class]);
      }
      std::fflush(stderr);
    }
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing && bounded_direct_streaming) {
    std::array<std::uint64_t, detail::kDirectQuartetShellClassCount> host_elapsed{};
    std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_launches{};
    std::array<unsigned long long, detail::kDirectQuartetShellClassCount> host_fp64_work{};
    std::array<unsigned long long, detail::kDirectQuartetShellClassCount> host_fp32_work{};
    cuda_error = cudaMemcpyAsync(host_elapsed.data(), bounded_fock_class_timer_elapsed,
                                 host_elapsed.size() * sizeof(std::uint64_t),
                                 cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(host_launches.data(), bounded_fock_class_timer_launches,
                                   host_launches.size() * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(host_fp64_work.data(), bounded_fock_fp64_work_counts,
                                   host_fp64_work.size() * sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(host_fp32_work.data(), bounded_fock_fp32_work_counts,
                                   host_fp32_work.size() * sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      struct HostFockClassTiming {
        const char* name;
        unsigned shell_class;
        std::uint64_t elapsed_nanoseconds;
        std::uint32_t launches;
      };
      std::vector<HostFockClassTiming> timings;
      std::uint64_t total_elapsed_nanoseconds = 0U;
      timings.reserve(bounded_fock_kernel_count);
      for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const generated::ShellKernelMetadata& kernel = bounded_fock_kernels[kernel_index];
        const unsigned shell_class = kernel.shell_class;
        if (shell_class >= host_launches.size() || host_launches[shell_class] == 0U) {
          continue;
        }
        timings.push_back(
            {kernel.name, shell_class, host_elapsed[shell_class], host_launches[shell_class]});
        total_elapsed_nanoseconds += host_elapsed[shell_class];
      }
      std::sort(timings.begin(), timings.end(),
                [](const HostFockClassTiming& first, const HostFockClassTiming& second) {
                  return first.elapsed_nanoseconds > second.elapsed_nanoseconds;
                });
      std::fprintf(stderr, "bounded-direct-fock-class-profile total_gpu_ms=%.6f classes=%zu\n",
                   static_cast<double>(total_elapsed_nanoseconds) * 1.0e-6, timings.size());
      for (const HostFockClassTiming& timing : timings) {
        const double share = total_elapsed_nanoseconds == 0U
                                 ? 0.0
                                 : 100.0 * static_cast<double>(timing.elapsed_nanoseconds) /
                                       static_cast<double>(total_elapsed_nanoseconds);
        std::fprintf(stderr, "  %-4s class=%u launches=%u gpu_ms=%.6f share=%.2f%%\n", timing.name,
                     timing.shell_class, timing.launches,
                     static_cast<double>(timing.elapsed_nanoseconds) * 1.0e-6, share);
      }
      std::fprintf(stderr, "bounded-direct-fock-precision-profile enabled=%u threshold=%.17g\n",
                   mixed_precision_fock ? 1U : 0U, mixed_precision_fock_threshold);
      for (std::size_t kernel_index = 0; kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const generated::ShellKernelMetadata& kernel = bounded_fock_kernels[kernel_index];
        const unsigned shell_class = kernel.shell_class;
        if (shell_class >= host_fp64_work.size() ||
            (host_fp64_work[shell_class] == 0ULL && host_fp32_work[shell_class] == 0ULL)) {
          continue;
        }
        const unsigned mixed_capable =
            (host_generated_mixed_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U
                ? 1U
                : 0U;
        std::fprintf(stderr,
                     "  %-4s class=%u fp64_quartets=%llu fp32_quartets=%llu mixed_capable=%u\n",
                     kernel.name, shell_class, host_fp64_work[shell_class],
                     host_fp32_work[shell_class], mixed_capable);
      }
      std::fflush(stderr);
    }
  }
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  if (bounded_direct_fock_only_diagnostic && bounded_direct_streaming) {
    // The isolated profile intentionally captures one Fock construction and
    // omits the eigensolve/convergence tail. Return after device timings are
    // copied so no final-Fock rebuild or analytic-force work contaminates it.
    fill_global_failure(outputs, VIBEQC_STATUS_NOT_CONVERGED);
    for (auto& output : outputs) output.fock_only_diagnostic = true;
    return outputs;
  }

  // ---------------------------------------------------------------------------
  // Target-precision refinement.
  //
  // The mixed Fock is only the iterative operator, so a density converged under
  // it is not a converged solution of the requested FP64 equations. Promote the
  // items that used mixed precision to exact FP64 from their mixed density and
  // continue until the same criteria are met. An item that exhausts the bound
  // stays unconverged, so a noisy state is never reported as a success, and the
  // refinement cost is counted in the reported iterations.
  // ---------------------------------------------------------------------------
  std::vector<std::uint32_t> host_mixed_iterations(batch_size, 0U);
  if (mixed_precision_fock) {
    cuda_error = cudaMemcpyAsync(host_mixed_iterations.data(), iterations,
                                 batch_size * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                                 resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    // Only the items that actually ran the mixed operator re-enter the loop.
    launch_enter_target_refinement_kernel(
        blocks_for(batch_size), threads, 0, resources.stream_,
        static_cast<std::int32_t>(batch_size), mixed_precision_item_census, active, converged,
        failed, iterations, previous_energy, energy_change, density_rms, diis_count, diis_head);
    std::vector<std::uint8_t> host_refinement_active(batch_size, 0U);
    cuda_error =
        cudaMemcpyAsync(host_refinement_active.data(), active, batch_size * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    // A mixed density within the reserved budget needs only a few exact
    // iterations; the full iteration bound applies so a pathological state
    // reports an honest non-convergence instead of a clamped success. Each item
    // leaves the loop on its own convergence, so a stagnating item is promoted
    // without holding back or dictating the precision of its neighbors.
    for (std::uint32_t refinement = 0; refinement < options.max_iterations; ++refinement) {
      if (std::none_of(host_refinement_active.begin(), host_refinement_active.end(),
                       [](std::uint8_t value) { return value != 0; })) {
        break;
      }
      status = launch_iteration_pre_eigensolver(false);
      if (status == VIBEQC_STATUS_SUCCESS) {
        status = launch_iteration_eigensolver(ordinary_eigensolver_family);
      }
      if (status == VIBEQC_STATUS_SUCCESS) {
        status = launch_iteration_post_eigensolver(false);
      }
      if (status != VIBEQC_STATUS_SUCCESS) break;
      cuda_error =
          cudaMemcpyAsync(host_refinement_active.data(), active, batch_size * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost, resources.stream_);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaStreamSynchronize(resources.stream_);
      }
      if (cuda_error != cudaSuccess) break;
    }
    if (status != VIBEQC_STATUS_SUCCESS || cuda_error != cudaSuccess) {
      fill_global_failure(outputs,
                          status != VIBEQC_STATUS_SUCCESS ? status : cuda_status(cuda_error));
      return outputs;
    }
  }
  std::uint32_t host_final_fock_rebuild_count = static_cast<std::uint32_t>(batch_size);
  if (reuse_converged_fock) {
    // Partition on the device because density RMS is already per-system. This
    // permits a mixed bucket: tight systems retain P_n/F(P_n), while only
    // looser systems restore P_{n+1} and execute the exact legacy rebuild.
    cuda_error =
        cudaMemsetAsync(final_fock_rebuild_count, 0, sizeof(std::uint32_t), resources.stream_);
    if (cuda_error == cudaSuccess) {
      launch_select_final_fock_rebuild_kernel(
          blocks_for(batch_size), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size),
          converged_fock_reuse_density_rms(options.density_tolerance), density_rms, converged,
          failed, final_fock_reuse_mask, active, final_fock_rebuild_count);
      launch_copy_selected_matrices_kernel(
          blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), active, next_density, density);
      cuda_error =
          cudaMemcpyAsync(&host_final_fock_rebuild_count, final_fock_rebuild_count,
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      // One post-Graph scalar fence avoids launching the expensive Fock
      // worker family when every system can reuse its retained matrix.
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (host_final_fock_rebuild_count != 0) {
      cuda_error = launch_fock_builder(density, false);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
    launch_select_converged_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                   static_cast<std::int32_t>(batch_size), converged, failed,
                                   active);
    if (quartet_direct && batch_size > 1 && host_final_fock_rebuild_count != batch_size) {
      // Later device-tail launches overwrite the shared compact quartet list
      // after an early peer converges. Recreate only density transforms,
      // shell-pair bounds, and task metadata for all final snapshots; do not
      // evaluate any two-electron integrals or modify retained Fock matrices.
      cuda_error = launch_direct_quartet_metadata(density, false);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
  } else {
    launch_select_converged_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                   static_cast<std::int32_t>(batch_size), converged, failed,
                                   active);
    cuda_error = launch_fock_builder(density, false);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }

  // Diagonalize each un-extrapolated final Fock. Tight systems consume their
  // retained P_n/F(P_n); rebuilt systems consume P_{n+1}/F(P_{n+1}). The
  // active mask now contains every converged system for common finalization.
  if (unrestricted) {
    status = multiply_spin_matrices(fock, true, false, orthogonalizer, false, temporary);
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = multiply_spin_matrices(orthogonalizer, false, true, temporary, true, eigensystem);
    }
    if (status == VIBEQC_STATUS_SUCCESS) {
      launch_expand_spin_active_kernel(blocks_for(spin_batch_size), threads, 0, resources.stream_,
                                       static_cast<std::int32_t>(batch_size), 2, active,
                                       spin_active);
      status = launch_solver(resources.eigensolver_view(), ordinary_eigensolver_family,
                             static_cast<int>(nbf), static_cast<int>(spin_batch_size), eigensystem,
                             temporary, eigenvalues, lwork, solver_info, spin_active);
    }
  } else {
    status = multiply_matrices(fock, false, orthogonalizer, temporary);
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = multiply_matrices(orthogonalizer, true, temporary, eigensystem);
    }
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = launch_solver(resources.eigensolver_view(), ordinary_eigensolver_family,
                             static_cast<int>(nbf), static_cast<int>(batch_size), eigensystem,
                             temporary, eigenvalues, lwork, solver_info, active);
    }
  }
  if (status != VIBEQC_STATUS_SUCCESS) {
    fill_global_failure(outputs, status);
    return outputs;
  }
  if (unrestricted) {
    launch_inspect_spin_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                      static_cast<std::int32_t>(batch_size), 2, solver_info, active,
                                      failed, converged);
    status = multiply_spin_matrices(orthogonalizer, false, false, eigensystem, true, coefficients);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  } else {
    launch_inspect_solver_kernel(blocks_for(batch_size), threads, 0, resources.stream_,
                                 static_cast<std::int32_t>(batch_size), solver_info, active, failed,
                                 converged);
    status = multiply_matrices(orthogonalizer, false, eigensystem, coefficients);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  }
  // Keep the converged density paired with the un-extrapolated F(P) that was
  // just diagonalized. The canonical coefficients and eigenvalues are needed
  // for the Pulay weighted density, but replacing P with C_occ C_occ^T would
  // require a second complete J/K rebuild before energy and force evaluation.
  // The accepted density update has already passed the requested SCF density
  // tolerance, so retaining P keeps all final energy/two-electron force terms
  // evaluated consistently at the same P and F(P).
  if (options.compute_forces) {
    cuda_error = launch_direct_force_compaction();
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (shell_class_profiling && quartet_direct) {
      cuda_error = cudaMemsetAsync(
          shell_class_profile, 0,
          detail::kDirectQuartetShellClassCount * sizeof(DeviceShellClassProfileEntry),
          resources.stream_);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
    if (shell_class_profiling && quartet_direct && total_shell_quartet_tiles != 0) {
      launch_profile_active_shell_quartet_tiles_kernel(
          blocks_for(total_shell_quartet_tiles), threads, 0, resources.stream_, device_batch,
          total_shell_quartet_tiles, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, shell_class_profile);
    }
  }
  if (unrestricted) {
    launch_compute_uhf_energy_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
                                     resources.stream_, static_cast<std::int32_t>(batch_size),
                                     static_cast<std::int32_t>(nbf), density, hcore, fock,
                                     nuclear_repulsion, active, energy);
    if (options.compute_forces) {
      launch_build_spin_weighted_density_kernel(
          blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf), occupied,
          coefficients, eigenvalues, active, weighted_density);
      launch_sum_uhf_spin_matrices_kernel(blocks_for(matrix_elements), threads, 0,
                                          resources.stream_, static_cast<std::int32_t>(batch_size),
                                          static_cast<std::int32_t>(nbf), density, active,
                                          total_density);
      launch_sum_uhf_spin_matrices_kernel(blocks_for(matrix_elements), threads, 0,
                                          resources.stream_, static_cast<std::int32_t>(batch_size),
                                          static_cast<std::int32_t>(nbf), weighted_density, active,
                                          total_weighted_density);
    }
  } else {
    launch_compute_energy_kernel(static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
                                 resources.stream_, static_cast<std::int32_t>(batch_size),
                                 static_cast<std::int32_t>(nbf), density, hcore, fock,
                                 nuclear_repulsion, active, energy);
    if (options.compute_forces) {
      launch_build_weighted_density_kernel(blocks_for(matrix_elements), threads, 0,
                                           resources.stream_, static_cast<std::int32_t>(batch_size),
                                           static_cast<std::int32_t>(nbf), occupied, coefficients,
                                           eigenvalues, active, weighted_density);
    }
  }
  if (options.export_physical_reference) {
    outputs[0].status = reference_detail::download(
        resources.stream_, nbf, host.occupied[0], resources.reference_peak_bytes_,
        {overlap, hcore, fock, coefficients, density}, eigenvalues,
        {energy, energy_change, density_rms}, converged, failed, iterations, outputs[0].scf);
    return outputs;
  }
  if (options.compute_forces) {
    cuda_error = cudaMemsetAsync(forces, 0, total_atoms * 3 * sizeof(double), resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (quartet_direct && !bounded_direct_streaming) {
      cuda_error = cudaMemsetAsync(persistent_force_task_heads, 0,
                                   kPersistentForceAngularOrderCount * sizeof(std::uint32_t),
                                   resources.stream_);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
    launch_nuclear_force_kernel(blocks_for(force_coordinate_count), threads, 0, resources.stream_,
                                device_batch, active, forces);
    // Derivative selection is read on every force execution; it retains no
    // candidate-specific geometry or plan buffers that could become stale.
    if (cuda_policy::generated_one_electron_derivatives_requested()) {
      const OneElectronWeightView weights{unrestricted ? total_weighted_density : weighted_density,
                                          unrestricted ? total_density : density,
                                          unrestricted ? total_density : density,
                                          -1.0,
                                          1.0,
                                          1.0};
      cuda_error = launch_generated_one_electron_gradient(
          one_electron_view(device_batch), ao_pair_first, ao_pair_second, pair_count, weights,
          active, cuda_policy::one_electron_derivative_mapping_requested(), -1.0, forces,
          resources.stream_);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    } else if (cooperative_one_electron_force) {
      constexpr std::size_t shared_bytes = 3 * sizeof(OneElectronDerivativeHermiteCoefficients);
      launch_one_electron_force_cooperative_kernel(
          static_cast<unsigned>(one_electron_force_elements), threads, shared_bytes,
          resources.stream_, device_batch, ao_pair_first, ao_pair_second, pair_count,
          unrestricted ? total_density : density,
          unrestricted ? total_weighted_density : weighted_density, active, forces);
    } else {
      launch_one_electron_force_scalar_kernel(
          blocks_for(one_electron_force_elements), threads, 0, resources.stream_, device_batch,
          ao_pair_first, ao_pair_second, pair_count, unrestricted ? total_density : density,
          unrestricted ? total_weighted_density : weighted_density, active, forces);
    }
  }
  const std::uint64_t explicit_generated_force_shell_class_mask =
      generated::enabled_shell_class_mask() & host_present_shell_class_mask;
  // Fock-only AOT entries (currently ssss/psss) are deliberately not added
  // to the force queue.  The force dispatcher is a separate registry and
  // returns ``cudaErrorNotSupported`` for classes without a validated force
  // consumer.  Keep these classes on the exact handwritten low-order page
  // kernel below until an independently validated generated force entry is
  // promoted.
  const bool bounded_resident_psss_force_enabled =
      bounded_direct_streaming && plan.resident_psss_task_count != 0U &&
      plan.resident_psss_bra_primitive_pairs != 0U &&
      plan.resident_psss_bra_primitive_pairs <= kResidentPsssMaximumBraPrimitivePairs;
  const std::uint64_t bounded_native_paged_force_shell_class_mask =
      (bounded_direct_streaming
           ? host_present_shell_class_mask & kBoundedNativePagedForceShellClassMask
           : 0U) &
      ~(bounded_resident_psss_force_enabled ? (std::uint64_t{1} << kPsssShellClass) : 0U);
  const std::uint64_t selected_force_shell_class_mask =
      (explicit_generated_force_shell_class_mask |
       (bounded_direct_streaming
            ? generated::enabled_fock_shell_class_mask() & kStreamingFockShellClassMask &
                  ~kBoundedNativePagedForceShellClassMask
            : 0U)) &
      host_present_shell_class_mask;
  const std::uint64_t native_streaming_force_shell_class_mask =
      bounded_direct_streaming ? selected_force_shell_class_mask & kDdddShellClassMask &
                                     ~explicit_generated_force_shell_class_mask
                               : 0U;
  const std::uint64_t generated_shell_class_mask =
      selected_force_shell_class_mask & ~native_streaming_force_shell_class_mask;
  const std::uint64_t generated_queued_force_shell_class_mask = generated_shell_class_mask;
  // Whole-task and subgroup-task workers use page-local primitive signatures
  // before advancing independent quartets in warp lockstep. Keep those exact
  // classes out of the unsorted first/retry arenas and route them through the
  // same bounded page stream used by their generated consumers.
  const std::uint64_t bounded_paged_force_shell_class_mask =
      bounded_direct_streaming
          ? generated_queued_force_shell_class_mask & kBoundedForceSignatureShellClassMask
          : 0U;
  const std::uint64_t bounded_first_wave_force_shell_class_mask =
      generated_queued_force_shell_class_mask & ~bounded_paged_force_shell_class_mask;
  // Count diagnostics intentionally materialize every class through the
  // pre-paging queue. Production keeps only classes outside the page
  // mask there; the remaining classes use disjoint exact pages.
  const std::uint64_t bounded_force_legacy_queue_shell_class_mask =
      bounded_direct_count_diagnostic ? generated_queued_force_shell_class_mask
                                      : bounded_first_wave_force_shell_class_mask;
  const std::uint64_t covered_force_shell_class_mask = generated_shell_class_mask |
                                                       native_streaming_force_shell_class_mask |
                                                       bounded_native_paged_force_shell_class_mask;
  const std::uint64_t uncovered_force_shell_class_mask =
      host_present_shell_class_mask & ~covered_force_shell_class_mask;
  std::size_t bounded_force_kernel_count = 0;
  const generated::ShellKernelMetadata* bounded_force_kernels =
      generated::selected_shell_kernels(bounded_force_kernel_count);
  const auto launch_bounded_generated_force = [&](bool is_unrestricted,
                                                  DirectScreeningPurpose purpose,
                                                  const double* quartet_density) -> cudaError_t {
    // The count diagnostic is also used to compare the pre-paging generated
    // queue against the exact page consumer.  In that mode every selected
    // generated class must enter the legacy queue, including classes that
    // production would route through the signature-paged stream; otherwise
    // the diagnostic silently omits the very class under investigation.
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_counts, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(
        bounded_direct_generated_overflow, bounded_force_kernel_count == 0 ? 1 : 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess || bounded_force_kernel_count == 0) return error;
    if (bounded_force_legacy_queue_shell_class_mask == 0U) return cudaSuccess;
    error =
        cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(unrestricted_value, purpose_value, task_offsets,     \
                                              selected_classes, selected_any)                      \
  launch_compact_bounded_generated_tasks_kernel(                                                   \
      unrestricted_value, purpose_value, plan.persistent_quartet_worker_blocks,                    \
      kBoundedDirectThreads, 0, resources.stream_, device_batch, options.screening_tolerance,      \
      shell_pair_bounds, shell_pair_density_bounds, bounded_direct_shell_pair_order,               \
      bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, active,        \
      nullptr, bounded_force_legacy_queue_shell_class_mask, 0U, selected_classes, selected_any,    \
      bounded_direct_cursor, bounded_direct_generated_tasks, bounded_direct_generated_task_counts, \
      task_offsets, bounded_direct_generated_overflow)
    if (is_unrestricted) {
      if (purpose == DirectScreeningPurpose::Force) {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(true, DirectScreeningPurpose::Force,
                                              bounded_direct_generated_task_offsets, nullptr,
                                              nullptr);
      } else {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(true, DirectScreeningPurpose::Fock,
                                              bounded_direct_generated_task_offsets, nullptr,
                                              nullptr);
      }
    } else if (purpose == DirectScreeningPurpose::Force) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(false, DirectScreeningPurpose::Force,
                                            bounded_direct_generated_task_offsets, nullptr,
                                            nullptr);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(false, DirectScreeningPurpose::Fock,
                                            bounded_direct_generated_task_offsets, nullptr,
                                            nullptr);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    launch_prepare_bounded_generated_retry_kernel(
        1, 1, 0, resources.stream_,
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity),
        bounded_direct_generated_task_offsets, bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads, bounded_direct_generated_overflow,
        bounded_direct_generated_retry_mask, bounded_direct_generated_retry_task_offsets,
        bounded_direct_generated_retry_any, bounded_direct_count_diagnostic);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (bounded_direct_count_diagnostic) {
      std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_counts{};
      std::array<std::uint32_t, detail::kDirectQuartetShellClassCount> host_overflow{};
      error = cudaMemcpyAsync(host_counts.data(), bounded_direct_generated_task_counts,
                              host_counts.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                              resources.stream_);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(host_overflow.data(), bounded_direct_generated_overflow,
                                host_overflow.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, resources.stream_);
      }
      if (error == cudaSuccess) {
        error = cudaStreamSynchronize(resources.stream_);
      }
      if (error != cudaSuccess) return error;
      std::fprintf(stderr, "bounded-direct-count purpose=%s capacity=%zu\n",
                   purpose == DirectScreeningPurpose::Force ? "force" : "fock",
                   plan.bounded_generated_task_capacity);
      for (std::size_t kernel_index = 0; kernel_index < bounded_force_kernel_count;
           ++kernel_index) {
        const generated::ShellKernelMetadata& kernel = bounded_force_kernels[kernel_index];
        if ((bounded_force_legacy_queue_shell_class_mask &
             (std::uint64_t{1} << kernel.shell_class)) == 0U) {
          continue;
        }
        const std::uint32_t class_capacity =
            plan.bounded_generated_task_offsets[kernel.shell_class + 1U] -
            plan.bounded_generated_task_offsets[kernel.shell_class];
        std::fprintf(stderr, "  %-4s count=%u capacity=%u overflow=%u\n", kernel.name,
                     host_counts[kernel.shell_class], class_capacity,
                     host_overflow[kernel.shell_class]);
      }
      std::fflush(stderr);
      return cudaSuccess;
    }
    const auto consume_generated_wave = [&](const std::uint32_t* task_offsets) -> cudaError_t {
      for (std::size_t kernel_index = 0; kernel_index < bounded_force_kernel_count;
           ++kernel_index) {
        const unsigned shell_class = bounded_force_kernels[kernel_index].shell_class;
        if ((bounded_force_legacy_queue_shell_class_mask & (std::uint64_t{1} << shell_class)) ==
            0U) {
          continue;
        }
        if (shell_class_profiling) {
          launch_profile_bounded_generated_tasks_kernel(
              plan.persistent_quartet_worker_blocks, threads, 0, resources.stream_, device_batch,
              bounded_direct_generated_tasks, task_offsets + shell_class,
              bounded_direct_generated_task_counts + shell_class, shell_class_profile);
          cudaError_t profile_error = cudaPeekAtLastError();
          if (profile_error != cudaSuccess) return profile_error;
        }
        cudaError_t launch_error = generated::launch_shell_class(
            shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks, task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets, device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density, forces,
            bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (launch_error != cudaSuccess) return launch_error;
      }
      return cudaSuccess;
    };
    error = consume_generated_wave(bounded_direct_generated_task_offsets);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(bounded_direct_generated_task_counts, 0,
                            detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                            resources.stream_);
    if (error == cudaSuccess) {
      error = cudaMemsetAsync(bounded_direct_generated_overflow, 0,
                              detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                              resources.stream_);
    }
    if (error == cudaSuccess) {
      error =
          cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    }
    if (error != cudaSuccess) return error;
    if (is_unrestricted) {
      if (purpose == DirectScreeningPurpose::Force) {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
            true, DirectScreeningPurpose::Force, bounded_direct_generated_retry_task_offsets,
            bounded_direct_generated_retry_mask, bounded_direct_generated_retry_any);
      } else {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
            true, DirectScreeningPurpose::Fock, bounded_direct_generated_retry_task_offsets,
            bounded_direct_generated_retry_mask, bounded_direct_generated_retry_any);
      }
    } else if (purpose == DirectScreeningPurpose::Force) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
          false, DirectScreeningPurpose::Force, bounded_direct_generated_retry_task_offsets,
          bounded_direct_generated_retry_mask, bounded_direct_generated_retry_any);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
          false, DirectScreeningPurpose::Fock, bounded_direct_generated_retry_task_offsets,
          bounded_direct_generated_retry_mask, bounded_direct_generated_retry_any);
    }
#undef VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    launch_normalize_bounded_generated_task_counts_kernel(
        blocks_for(detail::kDirectQuartetShellClassCount), threads, 0, resources.stream_,
        bounded_direct_generated_retry_task_offsets, bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads, bounded_direct_generated_overflow, false);
    error = cudaPeekAtLastError();
    return error == cudaSuccess
               ? consume_generated_wave(bounded_direct_generated_retry_task_offsets)
               : error;
  };
  const auto launch_bounded_overflow_force = [&](bool is_unrestricted,
                                                 DirectScreeningPurpose purpose,
                                                 const double* quartet_density) -> cudaError_t {
    // This routine is the normal force queue, despite the historical
    // ``overflow`` name.  Every generated force class is enumerated exactly
    // once in disjoint candidate pages; the optional signature pass only
    // changes task order within a page for lockstep consumers.
    for (std::size_t kernel_index = 0; kernel_index < bounded_force_kernel_count; ++kernel_index) {
      const unsigned shell_class = bounded_force_kernels[kernel_index].shell_class;
      if ((generated_queued_force_shell_class_mask & (std::uint64_t{1} << shell_class)) == 0U) {
        continue;
      }
      if ((bounded_force_legacy_queue_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
        // The pre-paging queue already consumed this class; keep the paged
        // stream disjoint so a force quartet is never evaluated twice.
        continue;
      }
      unsigned high_pair_class = 0U;
      while ((high_pair_class + 1U) * (high_pair_class + 2U) / 2U <= shell_class) {
        ++high_pair_class;
      }
      const unsigned low_pair_class = shell_class - high_pair_class * (high_pair_class + 1U) / 2U;
      const std::uint32_t page_capacity =
          static_cast<std::uint32_t>(plan.bounded_generated_task_capacity);
      if (page_capacity == 0U) return cudaErrorInvalidValue;
      const bool signature_paged =
          (bounded_paged_force_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U;
      const std::uint64_t page_domain =
          bounded_generated_page_range(plan.bounded_stream_pair_class_offsets, plan.batch_size,
                                       high_pair_class, low_pair_class, 0U, page_capacity)
              .candidate_count;
      for (std::uint64_t page_begin = 0U; page_begin < page_domain; page_begin += page_capacity) {
        const BoundedGeneratedPageRange page_range = bounded_generated_page_range(
            plan.bounded_stream_pair_class_offsets, plan.batch_size, high_pair_class,
            low_pair_class, page_begin, page_capacity);
        if (page_range.bra_begin >= page_range.bra_end) continue;
        cudaError_t error = cudaMemsetAsync(bounded_direct_generated_task_counts + shell_class, 0,
                                            sizeof(std::uint32_t), resources.stream_);
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_retry_task_offsets + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error == cudaSuccess && signature_paged) {
          error = cudaMemsetAsync(bounded_force_signature_counts, 0,
                                  kBoundedForceSignatureBucketCount * sizeof(std::uint32_t),
                                  resources.stream_);
        }
        if (error != cudaSuccess) return error;
        const auto compact_page = [&](std::uint32_t* signature_counts,
                                      const std::uint32_t* signature_offsets,
                                      bool force_execution) -> cudaError_t {
#define VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(unrestricted_value, purpose_value)                    \
  launch_compact_bounded_exact_class_force_wave_kernel(                                           \
      unrestricted_value, purpose_value, plan.persistent_quartet_worker_blocks,                   \
      kBoundedDirectThreads, 0, resources.stream_, device_batch, bounded_stream_topology,         \
      shell_class, high_pair_class, low_pair_class, options.screening_tolerance, page_begin,      \
      page_capacity, page_range.bra_begin, page_range.bra_end, high_pair_class == low_pair_class, \
      bounded_direct_generated_tasks, bounded_direct_generated_task_counts + shell_class,         \
      bounded_direct_generated_task_heads + shell_class, bounded_direct_generated_overflow,       \
      force_execution, signature_counts, signature_offsets)
          if (is_unrestricted) {
            if (purpose == DirectScreeningPurpose::Force) {
              VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(true, DirectScreeningPurpose::Force);
            } else {
              VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(true, DirectScreeningPurpose::Fock);
            }
          } else if (purpose == DirectScreeningPurpose::Force) {
            VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(false, DirectScreeningPurpose::Force);
          } else {
            VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(false, DirectScreeningPurpose::Fock);
          }
#undef VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE
          return cudaPeekAtLastError();
        };
        if (signature_paged) {
          // Count and scatter the same exact candidate page without host
          // readback. The second scan trades a small compaction cost for
          // primitive-uniform batches across all lockstep force workers.
          error = compact_page(bounded_force_signature_counts, nullptr, true);
          if (error == cudaSuccess) {
            launch_scan_bounded_force_signature_counts_kernel(
                kBoundedForceSignatureScanBlockCount, kBoundedForceSignatureScanThreads, 0,
                resources.stream_, bounded_force_signature_counts, bounded_force_signature_offsets,
                bounded_force_signature_block_offsets);
            error = cudaPeekAtLastError();
          }
          if (error == cudaSuccess) {
            launch_prefix_bounded_force_signature_blocks_kernel(
                1, kBoundedForceSignatureScanThreads, 0, resources.stream_,
                bounded_force_signature_offsets, bounded_force_signature_block_offsets,
                bounded_direct_generated_task_counts + shell_class);
            error = cudaPeekAtLastError();
          }
          if (error == cudaSuccess) {
            error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                    sizeof(std::uint32_t), resources.stream_);
          }
          if (error == cudaSuccess) {
            error =
                compact_page(bounded_force_signature_counts, bounded_force_signature_offsets, true);
          }
        } else {
          error = compact_page(nullptr, nullptr, false);
        }
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                  sizeof(std::uint32_t), resources.stream_);
        }
        if (error != cudaSuccess) return error;
        if (shell_class_profiling) {
          launch_profile_bounded_generated_tasks_kernel(
              plan.persistent_quartet_worker_blocks, threads, 0, resources.stream_, device_batch,
              bounded_direct_generated_tasks,
              bounded_direct_generated_retry_task_offsets + shell_class,
              bounded_direct_generated_task_counts + shell_class, shell_class_profile);
          error = cudaPeekAtLastError();
          if (error != cudaSuccess) return error;
        }
        error = generated::launch_shell_class(
            shell_class, resources.stream_, is_unrestricted, plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks,
            bounded_direct_generated_retry_task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets, device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density, forces,
            bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (error != cudaSuccess) return error;
      }
    }
    return cudaSuccess;
  };
  const auto launch_bounded_native_force = [&](bool is_unrestricted, DirectScreeningPurpose purpose,
                                               const double* quartet_density) -> cudaError_t {
    if ((native_streaming_force_shell_class_mask & kDdddShellClassMask) == 0U) {
      return cudaSuccess;
    }
    cudaError_t error = cudaMemsetAsync(bounded_direct_generated_task_heads + kDdddShellClass, 0,
                                        sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(unrestricted_value, purpose_value)                        \
  bounded_direct_dddd_streaming_kernel<unrestricted_value, purpose_value, true>                   \
      <<<plan.persistent_quartet_worker_blocks, detail::kDirectQuartetThreads, 0,                 \
         resources.stream_>>>(device_batch, bounded_stream_topology, options.screening_tolerance, \
                              schwarz_bounds, quartet_density, active, forces,                    \
                              bounded_direct_generated_task_heads + kDdddShellClass,              \
                              shell_class_profiling ? shell_class_profile : nullptr, nullptr)
    if (is_unrestricted) {
      if (purpose == DirectScreeningPurpose::Force) {
        VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(true, DirectScreeningPurpose::Force);
      } else {
        VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(true, DirectScreeningPurpose::Fock);
      }
    } else if (purpose == DirectScreeningPurpose::Force) {
      VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(false, DirectScreeningPurpose::Force);
    } else {
      VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(false, DirectScreeningPurpose::Fock);
    }
#undef VIBEQC_LAUNCH_NATIVE_DDDD_FORCE
    return cudaPeekAtLastError();
  };
  const auto launch_bounded_generic_force = [&](bool is_unrestricted,
                                                DirectScreeningPurpose purpose,
                                                const double* quartet_density) -> cudaError_t {
    if (uncovered_force_shell_class_mask == 0U) return cudaSuccess;
    // The generated/native routes above cover the common classes.  Keep the
    // remaining exact classes correct with the bounded runtime dispatcher,
    // rather than rejecting an otherwise valid large-AO force calculation.
    // Zero the generated-overflow state so an earlier diagnostic page cannot
    // make a covered class look like an uncovered fallback candidate.
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_overflow, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), resources.stream_);
    if (error == cudaSuccess) {
      error =
          cudaMemsetAsync(bounded_direct_cursor, 0, sizeof(unsigned long long), resources.stream_);
    }
    if (error != cudaSuccess) return error;
    if (is_unrestricted && purpose == DirectScreeningPurpose::Force) {
      bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Force, true>
          <<<plan.persistent_quartet_worker_blocks, kBoundedDirectThreads, 0, resources.stream_>>>(
              device_batch, options.screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds, bounded_direct_shell_pair_order,
              bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, nullptr,
              covered_force_shell_class_mask, bounded_direct_generated_overflow, schwarz_bounds,
              quartet_density, active, forces, bounded_direct_cursor,
              shell_class_profiling ? shell_class_profile : nullptr);
    } else if (!is_unrestricted && purpose == DirectScreeningPurpose::Force) {
      bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Force, true>
          <<<plan.persistent_quartet_worker_blocks, kBoundedDirectThreads, 0, resources.stream_>>>(
              device_batch, options.screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds, bounded_direct_shell_pair_order,
              bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, nullptr,
              covered_force_shell_class_mask, bounded_direct_generated_overflow, schwarz_bounds,
              quartet_density, active, forces, bounded_direct_cursor,
              shell_class_profiling ? shell_class_profile : nullptr);
    } else if (is_unrestricted) {
      bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Fock, true>
          <<<plan.persistent_quartet_worker_blocks, kBoundedDirectThreads, 0, resources.stream_>>>(
              device_batch, options.screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds, bounded_direct_shell_pair_order,
              bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, nullptr,
              covered_force_shell_class_mask, bounded_direct_generated_overflow, schwarz_bounds,
              quartet_density, active, forces, bounded_direct_cursor,
              shell_class_profiling ? shell_class_profile : nullptr);
    } else {
      bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Fock, true>
          <<<plan.persistent_quartet_worker_blocks, kBoundedDirectThreads, 0, resources.stream_>>>(
              device_batch, options.screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds, bounded_direct_shell_pair_order,
              bounded_direct_shell_pair_block_bounds, bounded_direct_system_density_bounds, nullptr,
              covered_force_shell_class_mask, bounded_direct_generated_overflow, schwarz_bounds,
              quartet_density, active, forces, bounded_direct_cursor,
              shell_class_profiling ? shell_class_profile : nullptr);
    }
    return cudaPeekAtLastError();
  };
  const auto launch_bounded_paged_native_force = [&](bool is_unrestricted,
                                                     DirectScreeningPurpose purpose,
                                                     const double* quartet_density) -> cudaError_t {
    const std::uint64_t native_mask = bounded_native_paged_force_shell_class_mask;
    if (native_mask == 0U) return cudaSuccess;
    const std::uint32_t page_capacity =
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity);
    if (page_capacity == 0U) return cudaErrorInvalidValue;
    for (unsigned shell_class = kSsssShellClass; shell_class <= kPsssShellClass; ++shell_class) {
      if ((native_mask & (std::uint64_t{1} << shell_class)) == 0U) {
        continue;
      }
      const unsigned high_pair_class = shell_class == kSsssShellClass ? 0U : 1U;
      const unsigned low_pair_class = shell_class == kSsssShellClass ? 0U : 0U;
      const std::uint64_t page_domain =
          bounded_generated_page_range(plan.bounded_stream_pair_class_offsets, plan.batch_size,
                                       high_pair_class, low_pair_class, 0U, page_capacity)
              .candidate_count;
      for (std::uint64_t page_begin = 0U; page_begin < page_domain; page_begin += page_capacity) {
        const BoundedGeneratedPageRange page_range = bounded_generated_page_range(
            plan.bounded_stream_pair_class_offsets, plan.batch_size, high_pair_class,
            low_pair_class, page_begin, page_capacity);
        if (page_range.bra_begin >= page_range.bra_end) continue;
        cudaError_t error = cudaMemsetAsync(bounded_direct_generated_task_heads + shell_class, 0,
                                            sizeof(std::uint32_t), resources.stream_);
        if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(unrestricted_value, purpose_value)                      \
  contract_bounded_exact_low_order_force_page_kernel<unrestricted_value, purpose_value>           \
      <<<plan.persistent_quartet_worker_blocks, kBoundedDirectThreads, 0, resources.stream_>>>(   \
          device_batch, bounded_stream_topology, shell_class, high_pair_class, low_pair_class,    \
          options.screening_tolerance, page_begin, page_capacity, page_range.bra_begin,           \
          page_range.bra_end, high_pair_class == low_pair_class, schwarz_bounds, quartet_density, \
          forces, bounded_direct_generated_task_heads + shell_class)
        if (purpose == DirectScreeningPurpose::Force) {
          if (is_unrestricted) {
            VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(true, DirectScreeningPurpose::Force);
          } else {
            VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(false, DirectScreeningPurpose::Force);
          }
        } else {
          if (is_unrestricted) {
            VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(true, DirectScreeningPurpose::Fock);
          } else {
            VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE(false, DirectScreeningPurpose::Fock);
          }
        }
#undef VIBEQC_LAUNCH_BOUNDED_NATIVE_PAGE
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
      }
    }
    return cudaSuccess;
  };
  const auto launch_bounded_resident_psss_force =
      [&](bool is_unrestricted, DirectScreeningPurpose purpose,
          const double* quartet_density) -> cudaError_t {
    if (!bounded_resident_psss_force_enabled) return cudaSuccess;
    const bool use_force_screening = purpose == DirectScreeningPurpose::Force;
    if (is_unrestricted) {
      two_electron_force_psss_resident_bra_kernel<true>
          <<<static_cast<unsigned>(plan.resident_psss_task_count), kResidentPsssThreads,
             plan.resident_psss_bra_primitive_pairs * sizeof(PrimitivePairData),
             resources.stream_>>>(device_batch, psss_resident_tasks, psss_resident_ket_pairs,
                                  plan.resident_psss_task_count, options.screening_tolerance,
                                  shell_pair_bounds, shell_pair_density_bounds, use_force_screening,
                                  schwarz_bounds, quartet_density, active, forces,
                                  generated_shell_class_mask);
    } else {
      two_electron_force_psss_resident_bra_kernel<false>
          <<<static_cast<unsigned>(plan.resident_psss_task_count), kResidentPsssThreads,
             plan.resident_psss_bra_primitive_pairs * sizeof(PrimitivePairData),
             resources.stream_>>>(device_batch, psss_resident_tasks, psss_resident_ket_pairs,
                                  plan.resident_psss_task_count, options.screening_tolerance,
                                  shell_pair_bounds, shell_pair_density_bounds, use_force_screening,
                                  schwarz_bounds, quartet_density, active, forces,
                                  generated_shell_class_mask);
    }
    return cudaPeekAtLastError();
  };
  const auto launch_bounded_force = [&](bool is_unrestricted, DirectScreeningPurpose purpose,
                                        const double* quartet_density) -> cudaError_t {
    // Bounded production execution is class-specific by construction. The
    // generated/native routes handle the common classes; any registry gap is
    // sent to the exact bounded runtime dispatcher below instead of rejecting
    // a valid topology or silently scanning an unbounded AO-space fallback.
    // Generated classes selected by the page mask use exact, disjoint
    // pages. The pre-paging queue is retained only for diagnostics and
    // classes that do not use the paged route.
    cudaError_t error = cudaSuccess;
    if (bounded_direct_count_diagnostic) {
      return launch_bounded_generated_force(is_unrestricted, purpose, quartet_density);
    }
    if ((bounded_force_legacy_queue_shell_class_mask & generated_queued_force_shell_class_mask) !=
        0U) {
      // Consume non-paged generated classes first, then keep their exact
      // class mask out of the page stream to avoid duplicate quartets.
      error = launch_bounded_generated_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      error = launch_bounded_overflow_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      error = launch_bounded_resident_psss_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      error = launch_bounded_paged_native_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      error = launch_bounded_native_force(is_unrestricted, purpose, quartet_density);
      if (error != cudaSuccess) return error;
      return launch_bounded_generic_force(is_unrestricted, purpose, quartet_density);
    }
    error = launch_bounded_overflow_force(is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess) return error;
    error = launch_bounded_resident_psss_force(is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess) return error;
    error = launch_bounded_paged_native_force(is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess) return error;
    error = launch_bounded_native_force(is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess) return error;
    error = launch_bounded_generic_force(is_unrestricted, purpose, quartet_density);
    return error;
  };
  if (options.compute_forces && quartet_direct &&
      plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder] != 0) {
    cuda_error =
        cudaMemsetAsync(generic_order5_tile_count, 0, sizeof(std::uint32_t), resources.stream_);
    if (cuda_error == cudaSuccess) {
      launch_compact_generic_order5_tiles_kernel(
          blocks_for(plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder]), threads, 0,
          resources.stream_, device_batch,
          active_shell_quartet_tile_counts + kGenericOrderFiveAngularOrder,
          active_shell_quartet_tiles +
              plan.shell_quartet_tile_offsets[kGenericOrderFiveAngularOrder],
          generated_shell_class_mask, nullptr, generic_order5_tile_count, generic_order5_tiles);
      cuda_error = cudaPeekAtLastError();
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (!options.compute_forces) {
    // Energy-only execution intentionally omits every analytic-force kernel.
  } else if (bounded_direct_fock_only_diagnostic && bounded_direct_streaming) {
    // Nuclear and one-electron forces above remain in the timing so this
    // diagnostic isolates only the bounded two-electron force tail.
  } else if (unrestricted && persistent_eri) {
    two_electron_uhf_force_kernel<<<blocks_for(persistent_force_elements), threads, 0,
                                    resources.stream_>>>(device_batch, density, active, forces);
  } else if (unrestricted && quartet_direct) {
    if (bounded_direct_streaming) {
      const DirectScreeningPurpose bounded_force_purpose = force_density_product_screening
                                                               ? DirectScreeningPurpose::Force
                                                               : DirectScreeningPurpose::Fock;
      cuda_error = launch_bounded_force(true, bounded_force_purpose,
                                        transformed_direct ? direct_density : density);
    } else {
      cuda_error = launch_generated_shell_class_forces(
          resources.stream_, plan.total_shell_quartet_tiles, plan.generated_shell_task_capacity,
          plan.shell_quartet_tile_capacities, active_shell_quartet_tile_offsets, device_batch,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, generated_shell_tasks,
          generated_shell_classes, generated_shell_task_offsets, generated_shell_task_counts,
          generated_shell_task_write_counts, generated_shell_task_heads,
          generated_low_order_signature_counts, generated_low_order_signature_offsets,
          generated_ppps_resident_tasks, generated_ppps_resident_ket_tasks,
          generated_ppps_resident_bra_counts, generated_ppps_resident_bra_offsets,
          generated_ppps_resident_bra_write_counts, generated_ppps_resident_signature_counts,
          generated_ppps_resident_signature_offsets, generated_ppps_resident_signatures,
          total_shell_pairs, resident_ppps_bra && resident_ppps_ket_task_capacity != 0,
          resident_ppps_signature_bucketing, psps_signature_bucketing, ppss_signature_bucketing,
          resident_ppps_block_threads, plan.persistent_quartet_worker_blocks, true,
          generated_shell_class_mask, options.screening_tolerance, schwarz_bounds,
          transformed_direct ? direct_density : density, forces);
      if (cuda_error == cudaSuccess) {
        launch_angular_force_quartets<true>(
            resources.stream_, plan.shell_quartet_tile_capacities, plan.shell_quartet_tile_offsets,
            device_batch, active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generic_order5_tile_count, generic_order5_tiles, persistent_force_task_heads,
            plan.persistent_quartet_worker_blocks, psss_resident_tasks, psss_resident_ket_pairs,
            plan.resident_psss_task_count, plan.resident_psss_bra_primitive_pairs,
            options.screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            force_density_product_screening, schwarz_bounds,
            transformed_direct ? direct_density : density, active, forces,
            generated_shell_class_mask);
        cuda_error = cudaPeekAtLastError();
      }
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  } else if (unrestricted) {
    two_electron_uhf_force_direct_kernel<<<blocks_for(direct_force_elements), threads, 0,
                                           resources.stream_>>>(
        device_batch, options.screening_tolerance, ao_pair_first, ao_pair_second, pair_count,
        schwarz_bounds, density, active, forces);
  } else if (persistent_eri) {
    two_electron_force_kernel<<<blocks_for(persistent_force_elements), threads, 0,
                                resources.stream_>>>(device_batch, density, active, forces);
  } else if (quartet_direct) {
    if (bounded_direct_streaming) {
      const DirectScreeningPurpose bounded_force_purpose = force_density_product_screening
                                                               ? DirectScreeningPurpose::Force
                                                               : DirectScreeningPurpose::Fock;
      cuda_error = launch_bounded_force(false, bounded_force_purpose,
                                        transformed_direct ? direct_density : density);
    } else {
      cuda_error = launch_generated_shell_class_forces(
          resources.stream_, plan.total_shell_quartet_tiles, plan.generated_shell_task_capacity,
          plan.shell_quartet_tile_capacities, active_shell_quartet_tile_offsets, device_batch,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles, generated_shell_tasks,
          generated_shell_classes, generated_shell_task_offsets, generated_shell_task_counts,
          generated_shell_task_write_counts, generated_shell_task_heads,
          generated_low_order_signature_counts, generated_low_order_signature_offsets,
          generated_ppps_resident_tasks, generated_ppps_resident_ket_tasks,
          generated_ppps_resident_bra_counts, generated_ppps_resident_bra_offsets,
          generated_ppps_resident_bra_write_counts, generated_ppps_resident_signature_counts,
          generated_ppps_resident_signature_offsets, generated_ppps_resident_signatures,
          total_shell_pairs, resident_ppps_bra && resident_ppps_ket_task_capacity != 0,
          resident_ppps_signature_bucketing, psps_signature_bucketing, ppss_signature_bucketing,
          resident_ppps_block_threads, plan.persistent_quartet_worker_blocks, false,
          generated_shell_class_mask, options.screening_tolerance, schwarz_bounds,
          transformed_direct ? direct_density : density, forces);
      if (cuda_error == cudaSuccess) {
        launch_angular_force_quartets<false>(
            resources.stream_, plan.shell_quartet_tile_capacities, plan.shell_quartet_tile_offsets,
            device_batch, active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generic_order5_tile_count, generic_order5_tiles, persistent_force_task_heads,
            plan.persistent_quartet_worker_blocks, psss_resident_tasks, psss_resident_ket_pairs,
            plan.resident_psss_task_count, plan.resident_psss_bra_primitive_pairs,
            options.screening_tolerance, shell_pair_bounds, shell_pair_density_bounds,
            force_density_product_screening, schwarz_bounds,
            transformed_direct ? direct_density : density, active, forces,
            generated_shell_class_mask);
        cuda_error = cudaPeekAtLastError();
      }
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  } else {
    two_electron_force_direct_kernel<<<blocks_for(direct_force_elements), threads, 0,
                                       resources.stream_>>>(
        device_batch, options.screening_tolerance, ao_pair_first, ao_pair_second, pair_count,
        schwarz_bounds, density, active, forces);
  }

  if (reuse_converged_fock) {
    // The requested outputs above consumed each system's selected consistent
    // snapshot. Advance only reused systems to the already accepted P_{n+1}
    // for their returned warm state; rebuilt systems already contain it.
    launch_copy_selected_matrices_kernel(
        blocks_for(spin_matrix_elements), threads, 0, resources.stream_,
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(spin_count),
        static_cast<std::int32_t>(nbf), final_fock_reuse_mask, next_density, density);
  }

  cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }

  std::vector<double> host_energy(batch_size);
  std::vector<double> host_energy_change(batch_size);
  std::vector<double> host_density_rms(batch_size);
  std::vector<double> host_density(spin_matrix_elements);
  std::vector<double> host_forces(options.compute_forces ? total_atoms * 3 : 0U);
  std::vector<std::uint8_t> host_converged(batch_size);
  std::vector<std::uint8_t> host_failed(batch_size);
  std::vector<std::uint32_t> host_iterations(batch_size);
  std::uint32_t host_inactive_eigensolver_profile_count = 0U;
  std::vector<DeviceInactiveEigensolverProfileEntry> host_inactive_eigensolver_profile(
      inactive_eigensolver_profiling ? options.max_iterations : 0U);
  CudaRhfShellClassProfile host_shell_class_profile{};
  const bool collect_ppps_queue_profile =
      shell_class_profiling && quartet_direct && resident_ppps_bra &&
      resident_ppps_ket_task_capacity != 0U &&
      (generated_shell_class_mask & (std::uint64_t{1} << kPppsShellClass)) != 0U;
  std::vector<std::uint32_t> host_ppps_descriptor_counts(
      collect_ppps_queue_profile ? total_shell_pairs : 0U);
  std::vector<std::uint32_t> host_ppps_signatures(
      collect_ppps_queue_profile ? resident_ppps_ket_task_capacity : 0U);
  const struct Download {
    void* host;
    const void* device;
    std::size_t bytes;
  } downloads[] = {
      {host_energy.data(), energy, batch_size * sizeof(double)},
      {host_energy_change.data(), energy_change, batch_size * sizeof(double)},
      {host_density_rms.data(), density_rms, batch_size * sizeof(double)},
      {host_density.data(), density, spin_matrix_elements * sizeof(double)},
      {host_converged.data(), converged, batch_size * sizeof(std::uint8_t)},
      {host_failed.data(), failed, batch_size * sizeof(std::uint8_t)},
      {host_iterations.data(), iterations, batch_size * sizeof(std::uint32_t)},
  };
  for (const Download& download : downloads) {
    cuda_error = cudaMemcpyAsync(download.host, download.device, download.bytes,
                                 cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (options.compute_forces) {
    cuda_error = cudaMemcpyAsync(host_forces.data(), forces, total_atoms * 3 * sizeof(double),
                                 cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (inactive_eigensolver_profiling) {
    cuda_error = cudaMemcpyAsync(&host_inactive_eigensolver_profile_count,
                                 inactive_eigensolver_profile_count, sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess && !host_inactive_eigensolver_profile.empty()) {
      cuda_error = cudaMemcpyAsync(
          host_inactive_eigensolver_profile.data(), inactive_eigensolver_profile,
          host_inactive_eigensolver_profile.size() * sizeof(DeviceInactiveEigensolverProfileEntry),
          cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (shell_class_profiling && quartet_direct) {
    cuda_error =
        cudaMemcpyAsync(host_shell_class_profile.data(), shell_class_profile,
                        host_shell_class_profile.size() * sizeof(CudaRhfShellClassProfileEntry),
                        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (collect_ppps_queue_profile) {
    cuda_error =
        cudaMemcpyAsync(host_ppps_descriptor_counts.data(), generated_ppps_resident_bra_counts,
                        host_ppps_descriptor_counts.size() * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(host_ppps_signatures.data(), generated_ppps_resident_signatures,
                                   host_ppps_signatures.size() * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  cuda_error = cudaStreamSynchronize(resources.stream_);
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  if (shell_class_profiling && quartet_direct) {
    plan.last_shell_class_profile = host_shell_class_profile;
  }
  if (inactive_eigensolver_profiling) {
    if (host_inactive_eigensolver_profile_count > host_inactive_eigensolver_profile.size()) {
      fill_global_failure(outputs, VIBEQC_STATUS_INTERNAL_ERROR);
      return outputs;
    }
    CudaInactiveEigensolverProfile profile;
    profile.reserve(host_inactive_eigensolver_profile_count);
    for (std::uint32_t index = 0; index < host_inactive_eigensolver_profile_count; ++index) {
      const DeviceInactiveEigensolverProfileEntry& input = host_inactive_eigensolver_profile[index];
      CudaInactiveEigensolverProfileEntry output;
      output.iteration = input.iteration;
      output.family = static_cast<CudaEigensolverFamily>(input.family);
      output.physical_system_count = input.physical_system_count;
      output.solver_batch_count = input.solver_batch_count;
      output.active_physical_count = input.active_physical_count;
      output.active_solver_count = input.active_solver_count;
      output.solver_elapsed_nanoseconds = input.solver_elapsed_nanoseconds;
      output.inactive_input_nonfinite_count = input.inactive_input_nonfinite_count;
      output.inactive_submission_nonfinite_count = input.inactive_submission_nonfinite_count;
      output.inactive_info_nonzero_count = input.inactive_info_nonzero_count;
      output.inactive_touch_flags = input.inactive_touch_flags;
      output.provider_invoked = input.provider_invoked != 0U;
      profile.push_back(output);
    }
    plan.last_inactive_eigensolver_profile = std::move(profile);
  }
  if (collect_ppps_queue_profile) {
    const unsigned multiprocessor_count = std::max(
        1U, plan.persistent_quartet_worker_blocks / kPersistentQuartetWarpsPerMultiprocessor);
    CudaPppsQueueProfile ppps_profile = build_ppps_queue_profile(
        host, host_ppps_descriptor_counts, host_ppps_signatures, multiprocessor_count);
    if (ppps_profile.descriptor_slots != 0U) {
      plan.last_ppps_queue_profile = std::move(ppps_profile);
    }
  }
  const bool no_system_failed = std::none_of(host_failed.begin(), host_failed.end(),
                                             [](std::uint8_t value) { return value != 0; });
  if (no_system_failed) {
    plan.cached_positions = host.positions;
  } else if (geometry_changed) {
    // Never reuse an orthogonalizer from a calculation that reported a
    // numerical failure; retry the full geometry path on the next execution.
    plan.cached_positions.clear();
  }
  const bool all_systems_converged =
      no_system_failed && std::all_of(host_converged.begin(), host_converged.end(),
                                      [](std::uint8_t value) { return value != 0; });
  if (all_systems_converged) {
    plan.resident_warm_positions = host.positions;
    plan.resident_warm_density = host_density;
    plan.resident_previous_energy = host_energy;
  } else {
    // A failed or incomplete execution cannot provide an energy baseline for
    // the next warm density, even if the fleet retains another system's state.
    plan.resident_warm_positions.clear();
    plan.resident_warm_density.clear();
    plan.resident_previous_energy.clear();
  }

  // The mixed-precision Fock decision (already gated on quartet-direct) and its
  // forced final FP64 rebuild are fixed for the plan; report what actually ran.
  const int32_t requested_precision_mode = options.precision_mode.value_or(VIBEQC_PRECISION_FP64);
  const bool precision_route_enabled = plan.mixed_precision_fock;
  for (std::size_t system = 0; system < batch_size; ++system) {
    RhfBucketItem& output = outputs[system];
    ScfResult& result = output.scf;
    // The mixed operator belongs to this item alone: an item that stayed on the
    // exact operator reports it, with its own cutoff, budget and refinement
    // cost, regardless of what its batch neighbors resolved.
    const bool precision_item_mixed =
        precision_route_enabled && host_mixed_item_census[system] != 0U;
    result.energy = host_energy[system];
    // The refinement iterations are reported as part of the complete solve.
    result.iterations = precision_item_mixed
                            ? host_mixed_iterations[system] + host_iterations[system]
                            : host_iterations[system];
    result.energy_change = host_energy_change[system];
    result.density_rms = host_density_rms[system];
    result.converged = host_converged[system] != 0 && host_failed[system] == 0;
    result.initial_density_used = host.warm_mask[system] != 0;
    result.precision.requested_mode = requested_precision_mode;
    result.precision.effective_bits = precision_item_mixed ? 32U : 64U;
    result.precision.mixed_precision_fock_threshold =
        precision_item_mixed ? host_mixed_item_threshold[system] : 0.0;
    result.precision.strict_refinement_applied = precision_item_mixed;
    result.precision.mixed_precision_reserved_error =
        precision_item_mixed ? requested_precision_policy.item_budget_error : 0.0;
    result.precision.refinement_iterations = precision_item_mixed ? host_iterations[system] : 0U;
    const std::size_t density_stride = spin_count * matrix_size;
    result.density.assign(host_density.begin() + system * density_stride,
                          host_density.begin() + (system + 1) * density_stride);
    if (options.compute_forces) {
      const std::size_t atom_begin = static_cast<std::size_t>(host.atom_offsets[system]);
      const std::size_t atom_end = static_cast<std::size_t>(host.atom_offsets[system + 1]);
      result.forces.assign(host_forces.begin() + atom_begin * 3,
                           host_forces.begin() + atom_end * 3);
    }
    output.status = host_failed[system] != 0 ? VIBEQC_STATUS_NUMERICAL_FAILURE
                                             : (result.converged ? VIBEQC_STATUS_SUCCESS
                                                                 : VIBEQC_STATUS_SCF_NOT_CONVERGED);
  }
  return outputs;
}

}  // namespace

bool small_hf_cuda_resource_layout(std::size_t nbf, std::size_t direct_nbf, std::size_t atoms,
                                   std::size_t shells, std::size_t primitives,
                                   std::size_t diis_history, std::size_t spins,
                                   std::size_t& arena_bytes, std::size_t& plan_object_bytes) {
  // This contract intentionally covers the provider with no cuBLAS/cuSOLVER
  // workspace or exact-quartet descriptor table. Other routes need their own
  // compact provider-workspace query before they can advertise a global bound.
  if (nbf == 0 || nbf > kPersistentEriAoLimit || nbf > kSmallEigensolverLimit ||
      nbf >= kCublasMatrixProductAoThreshold || direct_nbf < nbf || direct_nbf > 2 * nbf ||
      atoms == 0 || shells == 0 || shells > nbf || primitives == 0 || diis_history > 64 ||
      (spins != 1 && spins != 2))
    return false;
  const std::size_t pairs = shells * (shells + 1) / 2;
  const std::size_t blocks =
      detail::bounded_direct_queue_refill_count(pairs, detail::kBoundedDirectShellPairBlockSize);
  ArenaLayout layout{};
  if (!make_layout(1, nbf, direct_nbf, atoms, shells, pairs, blocks, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   primitives, std::max<std::size_t>(1, diis_history), 0, spins, true, false, false,
                   false, false, false, false, layout))
    return false;
  arena_bytes = layout.bytes;
  plan_object_bytes = sizeof(CudaRhfBucketPlan);
  return true;
}

std::size_t hf_cuda_owned_device_bytes(const CudaRhfBucketPlan* plan) noexcept {
  if (plan == nullptr) return 0;
  const auto& resources = plan->resources;
  auto bytes = resources.arena_ == nullptr ? 0 : plan->layout.bytes;
  if (resources.solver_workspace_ != nullptr)
    bytes = runtime::add_capacity(bytes, resources.solver_workspace_bytes_);
  if (resources.direct_tile_validation_ != nullptr)
    bytes = runtime::add_capacity(bytes, sizeof(DirectTileValidationRecord));
  return bytes;
}

CudaRhfBasisLayoutStats inspect_rhf_cuda_basis_layout(const std::vector<core::System>& systems) {
  std::vector<const std::vector<double>*> initial_densities(systems.size(), nullptr);
  HostBatch host;
  if (!pack_host_batch(systems, initial_densities, host)) {
    throw std::invalid_argument("systems cannot be represented by one CUDA RHF bucket");
  }

  std::size_t expanded_primitive_references = 0;
  for (const core::System& system : systems) {
    for (const core::Shell& shell : system.shells) {
      std::size_t shell_references = 0;
      if (!checked_multiply(molecule::cartesian_count(shell.angular_momentum),
                            shell.primitives.size(), shell_references) ||
          !checked_add(expanded_primitive_references, shell_references,
                       expanded_primitive_references)) {
        throw std::overflow_error("expanded CUDA primitive reference count overflowed");
      }
    }
  }

  const std::size_t device_basis_bytes =
      host.system_shell_offsets.size() * sizeof(std::int64_t) +
      host.shell_atoms.size() * sizeof(std::int32_t) +
      host.shell_angular.size() * sizeof(std::uint8_t) +
      host.shell_ao_offsets.size() * sizeof(std::int64_t) +
      host.shell_direct_ao_offsets.size() * sizeof(std::int64_t) +
      host.shell_primitive_offsets.size() * sizeof(std::int64_t) +
      host.system_shell_pair_offsets.size() * sizeof(std::int64_t) +
      host.system_shell_quartet_offsets.size() * sizeof(std::int64_t) +
      host.shell_pair_systems.size() * sizeof(std::int32_t) +
      host.shell_pair_first.size() * sizeof(std::int32_t) +
      host.shell_pair_second.size() * sizeof(std::int32_t) +
      host.ao_shells.size() * sizeof(std::int32_t) +
      host.ao_term_counts.size() * sizeof(std::uint8_t) +
      host.ao_term_angular.size() * sizeof(std::uint8_t) +
      host.ao_term_coefficients.size() * sizeof(double) +
      host.direct_ao_shells.size() * sizeof(std::int32_t) +
      host.direct_ao_angular.size() * sizeof(std::uint8_t) +
      host.direct_ao_coefficients.size() * sizeof(double) +
      host.ao_to_direct_transform.size() * sizeof(double) +
      host.primitive_exponents.size() * sizeof(double) +
      host.primitive_coefficients.size() * sizeof(double);
  return {
      systems.size(),
      host.shell_atoms.size(),
      host.shell_pair_first.size(),
      static_cast<std::size_t>(host.system_shell_quartet_offsets.back()),
      host.ao_shells.size(),
      host.primitive_exponents.size(),
      expanded_primitive_references,
      device_basis_bytes,
      detail::direct_topology_requires_bounded_streaming(
          static_cast<std::size_t>(host.system_shell_quartet_offsets.back())),
      detail::direct_topology_requires_bounded_streaming(
          static_cast<std::size_t>(host.system_shell_quartet_offsets.back()))
          ? detail::kBoundedDirectQueueCapacity
          : 0,
  };
}

namespace {

std::vector<RhfBucketItem> run_hf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan, const std::vector<core::System>& systems,
    const ScfOptions& requested_options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool unrestricted, bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  if (requested_options.hooks || requested_options.strict_initial_density) {
    // Host callbacks are an explicit CPU capability, never a device fallback.
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_NOT_IMPLEMENTED);
    return outputs;
  }

  // Resolve legacy internal callers once per prepared execution, before any
  // device setup. Fock kernels and final exact force assembly share this guard.
  ScfOptions execution_options = requested_options;
  try {
    const FockSpin spin = unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted;
    if (!execution_options.resolved_fock_build.has_value()) {
      execution_options.resolved_fock_build = resolve_fock_build(
          make_hf_fock_spec(spin), FockBackend::Cuda, execution_options.screening_tolerance);
    }
    require_exact_direct_strategy(*execution_options.resolved_fock_build, spin, FockBackend::Cuda);
    if (execution_options.resolved_fock_build->screening_tolerance !=
        execution_options.screening_tolerance) {
      throw std::invalid_argument("CUDA screening differs from its resolved Fock strategy");
    }
  } catch (const std::invalid_argument&) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const ScfOptions& options = execution_options;

  if (plan == nullptr) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  HostBatch candidate;
  if (!pack_host_batch(systems, initial_densities, candidate, unrestricted,
                       options.export_physical_reference)) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::optional<double> mixed_precision_fock_threshold =
      *plan != nullptr && (*plan)->quartet_direct
          ? resolve_mixed_precision_fock_policy(
                options.precision_mode, options.energy_tolerance, options.screening_tolerance,
                static_cast<double>((*plan)->mixed_precision_eligible_tile_count))
                .threshold
          : std::nullopt;
  const bool mixed_precision_fock = mixed_precision_fock_threshold.has_value();
  const bool reuse_converged_fock = reuse_converged_fock_requested();
  const bool graph_native_eigensolver_override = graph_native_eigensolver_override_requested();
  if (*plan != nullptr && (*plan)->initialized &&
      ((*plan)->resources.device_id_ != device_id || !same_topology((*plan)->topology, candidate) ||
       !same_options((*plan)->options, options) || (*plan)->unrestricted != unrestricted ||
       (*plan)->shell_class_profiling != shell_class_profiling ||
       (*plan)->inactive_eigensolver_profiling != inactive_eigensolver_profiling ||
       (*plan)->bounded_fock_class_timing != bounded_fock_class_timing_requested() ||
       (*plan)->bounded_streaming_override != bounded_direct_streaming_override_requested() ||
       (*plan)->fock_only_diagnostic != bounded_direct_fock_only_diagnostic_requested() ||
       (*plan)->graph_native_eigensolver_override != graph_native_eigensolver_override ||
       (*plan)->reuse_converged_fock != reuse_converged_fock ||
       (*plan)->one_electron_value_mapping != cuda_policy::one_electron_value_mapping_requested() ||
       (*plan)->mixed_precision_fock != mixed_precision_fock ||
       (*plan)->mixed_precision_fock_threshold != mixed_precision_fock_threshold.value_or(0.0))) {
    delete *plan;
    *plan = nullptr;
  }
  if (*plan == nullptr) {
    *plan = new (std::nothrow) CudaRhfBucketPlan{};
    if (*plan == nullptr) {
      std::vector<RhfBucketItem> outputs(systems.size());
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
  }
  std::vector<RhfBucketItem> outputs =
      execute_hf_cuda_bucket(**plan, candidate, options, device_id, unrestricted,
                             shell_class_profiling, inactive_eigensolver_profiling);
  const bool retry_without_cublas = !(*plan)->initialized && (*plan)->retry_without_cublas;
  if (!(*plan)->initialized) {
    delete *plan;
    *plan = nullptr;
  }
  if (retry_without_cublas) {
    // Provider setup or graph capture can reject a cuBLAS implementation on a
    // particular CUDA release. Rebuild once with the numerically identical
    // native kernel so public CUDA execution remains available.
    *plan = new (std::nothrow) CudaRhfBucketPlan{};
    if (*plan == nullptr) {
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    (*plan)->cublas_enabled = false;
    outputs = execute_hf_cuda_bucket(**plan, candidate, options, device_id, unrestricted,
                                     shell_class_profiling, inactive_eigensolver_profiling);
    if (!(*plan)->initialized) {
      delete *plan;
      *plan = nullptr;
    }
  }
  return outputs;
}

}  // namespace

#include "scf/cuda/direct_jk_kernels.cuh"

std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan, const std::vector<core::System>& systems, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  return run_hf_cuda_bucket_cached(plan, systems, options, initial_densities, device_id, false,
                                   shell_class_profiling, inactive_eigensolver_profiling);
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan, const std::vector<core::System>& systems, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  return run_hf_cuda_bucket_cached(plan, systems, options, initial_densities, device_id, true,
                                   shell_class_profiling, inactive_eigensolver_profiling);
}

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept { delete plan; }

void set_rhf_cuda_bucket_warm_start_updates(CudaRhfBucketPlan* plan, bool enabled) noexcept {
  if (plan == nullptr || plan->warm_start_updates_enabled == enabled) return;
  if (!enabled) {
    // Freeze only on the policy transition. Repeating the setter while fixed
    // must not replace the original post-cold dm0/seed with a later replay's
    // advanced resident state.
    plan->frozen_warm_positions = plan->resident_warm_positions;
    plan->frozen_warm_density = plan->resident_warm_density;
    plan->frozen_previous_energy = plan->resident_previous_energy;
  } else {
    plan->frozen_warm_positions.clear();
    plan->frozen_warm_density.clear();
    plan->frozen_previous_energy.clear();
  }
  plan->warm_start_updates_enabled = enabled;
}

void clear_rhf_cuda_bucket_warm_starts(CudaRhfBucketPlan* plan) noexcept {
  if (plan == nullptr) return;
  plan->resident_warm_positions.clear();
  plan->resident_warm_density.clear();
  plan->resident_previous_energy.clear();
  plan->frozen_warm_positions.clear();
  plan->frozen_warm_density.clear();
  plan->frozen_previous_energy.clear();
}

bool get_rhf_cuda_shell_class_profile(const CudaRhfBucketPlan* plan,
                                      CudaRhfShellClassProfile& profile) noexcept {
  if (plan == nullptr || !plan->last_shell_class_profile.has_value()) {
    return false;
  }
  profile = *plan->last_shell_class_profile;
  return true;
}

bool get_rhf_cuda_ppps_queue_profile(const CudaRhfBucketPlan* plan,
                                     CudaPppsQueueProfile& profile) noexcept {
  if (plan == nullptr || !plan->last_ppps_queue_profile.has_value()) {
    return false;
  }
  profile = *plan->last_ppps_queue_profile;
  return true;
}

bool get_rhf_cuda_eigensolver_diagnostic(const CudaRhfBucketPlan* plan,
                                         CudaEigensolverDiagnostic& diagnostic) noexcept {
  if (plan == nullptr || !plan->initialized) return false;
  diagnostic = plan->eigensolver_diagnostic;
  return true;
}

bool get_rhf_cuda_inactive_eigensolver_profile(const CudaRhfBucketPlan* plan,
                                               CudaInactiveEigensolverProfile& profile) noexcept {
  if (plan == nullptr || !plan->last_inactive_eigensolver_profile.has_value()) {
    return false;
  }
  profile = *plan->last_inactive_eigensolver_profile;
  return true;
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket(
    const std::vector<core::System>& systems, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  CudaRhfBucketPlan* plan = nullptr;
  try {
    auto outputs =
        run_rhf_cuda_bucket_cached(&plan, systems, options, initial_densities, device_id,
                                   shell_class_profiling, inactive_eigensolver_profiling);
    destroy_rhf_cuda_bucket_plan(plan);
    return outputs;
  } catch (...) {
    destroy_rhf_cuda_bucket_plan(plan);
    throw;
  }
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket(
    const std::vector<core::System>& systems, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    bool shell_class_profiling, bool inactive_eigensolver_profiling) {
  CudaRhfBucketPlan* plan = nullptr;
  try {
    auto outputs =
        run_uhf_cuda_bucket_cached(&plan, systems, options, initial_densities, device_id,
                                   shell_class_profiling, inactive_eigensolver_profiling);
    destroy_rhf_cuda_bucket_plan(plan);
    return outputs;
  } catch (...) {
    destroy_rhf_cuda_bucket_plan(plan);
    throw;
  }
}

vibeqc_status contract_cuda_weighted_eri_primitives(
    int device_id, const CudaWeightedEriPrimitive* records, std::size_t record_count,
    std::size_t tile_count, std::size_t memory_budget_bytes, bool generated,
    std::vector<CudaWeightedEriResult>& output, CudaWeightedEriDiagnostic& diagnostic,
    std::string& detail) {
  // Discard previous output capacity so a small-budget call cannot retain an
  // old larger allocation while reporting only its new logical result size.
  std::vector<CudaWeightedEriResult>{}.swap(output);
  diagnostic = {};
  detail.clear();
  std::size_t output_bytes = 0, result_peak = 0, input_bytes = 0;
  if ((record_count != 0U && records == nullptr) ||
      tile_count > std::numeric_limits<std::uint32_t>::max() ||
      !checked_multiply(record_count, sizeof(CudaWeightedEriPrimitive), input_bytes) ||
      !checked_multiply(tile_count, sizeof(CudaWeightedEriResult), output_bytes) ||
      !checked_multiply(output_bytes, 2U, result_peak) || result_peak > memory_budget_bytes) {
    detail = "weighted ERI dimensions or numeric memory budget are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t index = 0; index < record_count; ++index) {
    const auto& record = records[index];
    if (record.kind > 1U || record.output_tile >= tile_count) {
      detail = "weighted ERI record kind or output tile is invalid";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    for (unsigned slot = 0; slot < 4; ++slot) {
      if (!(record.exponents[slot] > 0.0) || !std::isfinite(record.exponents[slot])) {
        detail = "weighted ERI primitive exponents must be finite and positive";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      unsigned total = 0;
      for (unsigned axis = 0; axis < 3; ++axis) {
        const auto angular = record.angular[slot][axis];
        if (angular > 3U || !std::isfinite(record.centers[slot][axis]) ||
            (record.kind == 1U && angular != static_cast<unsigned>(slot == 0U && axis == 0U))) {
          detail = "weighted ERI angular components or positions are invalid";
          return VIBEQC_STATUS_INVALID_ARGUMENT;
        }
        total += angular;
      }
      if (total > 3U) {
        detail = "weighted ERI primitive shells beyond f are unsupported";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
    }
    for (double weight : record.weights) {
      if (!std::isfinite(weight)) {
        detail = "external ERI weights must be finite";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
    }
    if (generated && record.kind == 1U)
      ++diagnostic.generated_records;
    else
      ++diagnostic.reference_records;
  }
  try {
    if (record_count == 0U) {
      output.resize(tile_count);
      diagnostic.host_peak_bytes = output_bytes;
      return VIBEQC_STATUS_SUCCESS;
    }
    const std::size_t capacity =
        std::min({record_count, std::size_t{65536},
                  (memory_budget_bytes - result_peak) / sizeof(CudaWeightedEriPrimitive)});
    if (capacity == 0U) {
      detail = "weighted ERI budget cannot hold one primitive and its outputs";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    diagnostic.primitive_capacity = capacity;
    diagnostic.host_peak_bytes = output_bytes;
    diagnostic.device_peak_bytes = output_bytes + capacity * sizeof(CudaWeightedEriPrimitive);
    auto error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
      detail = "weighted ERI CUDA device selection failed";
      return cuda_status(error);
    }
    struct Buffers {
      CudaWeightedEriPrimitive* records{};
      CudaWeightedEriResult* results{};
      cudaStream_t stream{};
      ~Buffers() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (records) (void)runtime::resource_cuda_free(records);
        if (results) (void)runtime::resource_cuda_free(results);
        if (stream) (void)cudaStreamDestroy(stream);
      }
    } buffers;
    error = cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking);
    if (error == cudaSuccess)
      error = runtime::resource_cuda_malloc(&buffers.records, capacity * sizeof(*buffers.records));
    if (error == cudaSuccess) error = runtime::resource_cuda_malloc(&buffers.results, output_bytes);
    if (error == cudaSuccess)
      error = cudaMemsetAsync(buffers.results, 0, output_bytes, buffers.stream);
    constexpr unsigned threads = 64U;
    for (std::size_t begin = 0; begin < record_count && error == cudaSuccess;) {
      const std::size_t count = std::min(capacity, record_count - begin);
      bool has_generated = false, has_reference = false;
      for (std::size_t i = begin; i < begin + count; ++i) {
        if (generated && records[i].kind == 1U)
          has_generated = true;
        else
          has_reference = true;
      }
      error = cudaMemcpyAsync(buffers.records, records + begin, count * sizeof(*records),
                              cudaMemcpyHostToDevice, buffers.stream);
      const unsigned blocks = static_cast<unsigned>((count + threads - 1U) / threads);
      if (error == cudaSuccess && has_reference) {
        weighted_eri_reference_kernel<<<blocks, threads, 0, buffers.stream>>>(
            buffers.records, count, generated, buffers.results);
        error = cudaGetLastError();
      }
      if (error == cudaSuccess && has_generated) {
        weighted_eri_generated_psss_kernel<<<blocks, threads, 0, buffers.stream>>>(
            buffers.records, count, buffers.results);
        error = cudaGetLastError();
      }
      begin += count;
    }
    if (error == cudaSuccess) {
      output.resize(tile_count);
      error = cudaMemcpyAsync(output.data(), buffers.results, output_bytes, cudaMemcpyDeviceToHost,
                              buffers.stream);
    }
    if (error == cudaSuccess) error = cudaStreamSynchronize(buffers.stream);
    if (error != cudaSuccess) {
      output.clear();
      detail = "CUDA external-weight ERI contraction failed";
      return cuda_status(error);
    }
    for (const auto& result : output) {
      bool finite = std::isfinite(result.value);
      for (const auto& center : result.center) {
        for (double value : center) finite = finite && std::isfinite(value);
      }
      if (!finite) {
        output.clear();
        detail = "weighted ERI contraction overflowed or produced nonfinite values";
        return VIBEQC_STATUS_NUMERICAL_FAILURE;
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    output.clear();
    detail = "weighted ERI host allocation failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    output.clear();
    detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
}

}  // namespace vibeqc::scf

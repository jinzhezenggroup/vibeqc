#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/matrix_index.cuh"

namespace generativeqc::scf::cuda_execution {

/** Apply the AO-level Schwarz gate after task/ordinal decoding. */
__device__ __forceinline__ bool direct_ao_quartet_survives_schwarz(
    const double* schwarz_bounds, std::size_t physical_offset, std::size_t n, std::size_t i,
    std::size_t j, std::size_t k, std::size_t l, double screening_tolerance) {
  const double quartet_bound = schwarz_bounds[physical_offset + matrix_index(i, j, n)] *
                               schwarz_bounds[physical_offset + matrix_index(k, l, n)];
  // Reject only an ordered bound below the threshold, as the original consumers
  // did. NaN (including 0 * infinity) must not silently screen away invalid data.
  return !(quartet_bound < screening_tolerance);
}

/** Experimental AO refinement of the existing independent force-product gate.
 * Preserve both density orientations and same-spin exchange products. In
 * particular, neither J/K cancellation nor opposite-spin exchange can justify
 * dropping a derivative. Nonfinite input must reach the numerical-failure path.
 * This is an additional screening approximation, not an ERI-derivative bound;
 * callers must keep the qualified unscreened-AO fallback available.
 */
template <bool Unrestricted>
__device__ __forceinline__ bool direct_ao_force_survives_density_products(
    double quartet_bound, double screening_tolerance, std::size_t dimension,
    std::size_t physical_offset, std::size_t spin_offset, const double* density, std::size_t first,
    std::size_t second, std::size_t third, std::size_t fourth) {
  if (!isfinite(quartet_bound)) return true;
  const std::size_t pair_first[6] = {first, third, first, second, first, second};
  const std::size_t pair_second[6] = {second, fourth, third, fourth, fourth, third};
  ShellPairDensityBounds pairs[6];
  const std::size_t matrix_size = dimension * dimension;
#pragma unroll
  for (unsigned pair = 0; pair < 6; ++pair) {
    const auto forward = matrix_index(pair_first[pair], pair_second[pair], dimension);
    const auto reverse = matrix_index(pair_second[pair], pair_first[pair], dimension);
    if constexpr (Unrestricted) {
      const double alpha_forward = density[spin_offset + forward];
      const double alpha_reverse = density[spin_offset + reverse];
      const double beta_forward = density[spin_offset + matrix_size + forward];
      const double beta_reverse = density[spin_offset + matrix_size + reverse];
      if (!isfinite(alpha_forward) || !isfinite(alpha_reverse) || !isfinite(beta_forward) ||
          !isfinite(beta_reverse))
        return true;
      const double total_forward = alpha_forward + beta_forward;
      const double total_reverse = alpha_reverse + beta_reverse;
      if (!isfinite(total_forward) || !isfinite(total_reverse)) return true;
      pairs[pair] = {fmax(fabs(total_forward), fabs(total_reverse)),
                     fmax(fabs(alpha_forward), fabs(alpha_reverse)),
                     fmax(fabs(beta_forward), fabs(beta_reverse))};
    } else {
      const double forward_value = density[physical_offset + forward];
      const double reverse_value = density[physical_offset + reverse];
      if (!isfinite(forward_value) || !isfinite(reverse_value)) return true;
      const double magnitude = fmax(fabs(forward_value), fabs(reverse_value));
      pairs[pair] = {magnitude, magnitude, 0.0};
    }
  }
  const double tolerance = fmin(screening_tolerance, kForceDensityProductScreeningTolerance);
  if (!(quartet_bound * pairs[0].coulomb * pairs[1].coulomb < tolerance)) return true;
  if (!(quartet_bound * pairs[2].exchange_alpha * pairs[3].exchange_alpha < tolerance) ||
      !(quartet_bound * pairs[4].exchange_alpha * pairs[5].exchange_alpha < tolerance))
    return true;
  if constexpr (Unrestricted) {
    if (!(quartet_bound * pairs[2].exchange_beta * pairs[3].exchange_beta < tolerance) ||
        !(quartet_bound * pairs[4].exchange_beta * pairs[5].exchange_beta < tolerance))
      return true;
  }
  return false;
}

/** Apply the shell-level Schwarz and density gate for one direct consumer. */
template <bool Unrestricted, DirectScreeningPurpose Purpose>
__device__ __forceinline__ bool direct_shell_quartet_survives_screening(
    const DeviceBatch& batch, std::size_t first_pair, std::size_t second_pair,
    double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds,
    double* fock_contribution_bound = nullptr, bool exchange_only = false,
    bool coulomb_only = false) {
  const double quartet_bound = shell_pair_bounds[first_pair] * shell_pair_bounds[second_pair];
  if (quartet_bound < screening_tolerance) return false;

  const std::int32_t system = batch.shell_pair_systems[first_pair];
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const std::size_t ac_pair = system_shell_pair_index(batch, system, first_shell, third_shell);
  const std::size_t ad_pair = system_shell_pair_index(batch, system, first_shell, fourth_shell);
  const std::size_t bc_pair = system_shell_pair_index(batch, system, second_shell, third_shell);
  const std::size_t bd_pair = system_shell_pair_index(batch, system, second_shell, fourth_shell);
  const ShellPairDensityBounds ab = shell_pair_density_bounds[first_pair];
  const ShellPairDensityBounds cd = shell_pair_density_bounds[second_pair];
  const ShellPairDensityBounds ac = shell_pair_density_bounds[ac_pair];
  const ShellPairDensityBounds ad = shell_pair_density_bounds[ad_pair];
  const ShellPairDensityBounds bc = shell_pair_density_bounds[bc_pair];
  const ShellPairDensityBounds bd = shell_pair_density_bounds[bd_pair];

  double fock_density_bound = exchange_only ? 0.0 : fmax(ab.coulomb, cd.coulomb);
  if (!coulomb_only) {
    if constexpr (Unrestricted) {
      const double exchange_bound = fmax(fmax(fmax(ac.exchange_alpha, ac.exchange_beta),
                                              fmax(ad.exchange_alpha, ad.exchange_beta)),
                                         fmax(fmax(bc.exchange_alpha, bc.exchange_beta),
                                              fmax(bd.exchange_alpha, bd.exchange_beta)));
      fock_density_bound =
          exchange_only ? exchange_bound : fmax(fock_density_bound, exchange_bound);
    } else {
      const double exchange_bound = fmax(fmax(ac.exchange_alpha, ad.exchange_alpha),
                                         fmax(bc.exchange_alpha, bd.exchange_alpha));
      // Ordinary RHF Fock uses J-K/2; the raw-K provider uses the full bound.
      fock_density_bound =
          exchange_only ? exchange_bound : fmax(fock_density_bound, 0.5 * exchange_bound);
    }
  }
  const double contribution_bound = quartet_bound * fock_density_bound;
  if (fock_contribution_bound != nullptr) {
    *fock_contribution_bound = contribution_bound;
  }
  if (contribution_bound < screening_tolerance) return false;
  if constexpr (Purpose == DirectScreeningPurpose::Fock) return true;

  // Screen J and each same-spin K contraction independently. Combining the
  // exact symmetry-reduced coefficient here would exploit cancellation and
  // can make loose-screening analytic forces disagree with finite differences.
  const double force_screening_tolerance =
      fmin(screening_tolerance, kForceDensityProductScreeningTolerance);
  if (quartet_bound * ab.coulomb * cd.coulomb >= force_screening_tolerance) {
    return true;
  }
  if constexpr (Unrestricted) {
    return quartet_bound * ac.exchange_alpha * bd.exchange_alpha >= force_screening_tolerance ||
           quartet_bound * ac.exchange_beta * bd.exchange_beta >= force_screening_tolerance ||
           quartet_bound * ad.exchange_alpha * bc.exchange_alpha >= force_screening_tolerance ||
           quartet_bound * ad.exchange_beta * bc.exchange_beta >= force_screening_tolerance;
  } else {
    return quartet_bound * ac.exchange_alpha * bd.exchange_alpha >= force_screening_tolerance ||
           quartet_bound * ad.exchange_alpha * bc.exchange_alpha >= force_screening_tolerance;
  }
}

/** Safely reject a complete shell-pair-block product before exact screening. */
template <DirectScreeningPurpose Purpose>
__device__ __forceinline__ bool bounded_direct_block_pair_survives_screening(
    std::size_t first_block, std::size_t second_block, std::int32_t system,
    double screening_tolerance, const double* shell_pair_block_bounds,
    const double* system_density_bounds) {
  const double quartet_bound =
      shell_pair_block_bounds[first_block] * shell_pair_block_bounds[second_block];
  if (quartet_bound < screening_tolerance) return false;
  const double density_bound = system_density_bounds[system];
  if (quartet_bound * density_bound < screening_tolerance) return false;
  if constexpr (Purpose == DirectScreeningPurpose::Force) {
    const double force_tolerance =
        fmin(screening_tolerance, kForceDensityProductScreeningTolerance);
    if (quartet_bound * density_bound * density_bound < force_tolerance) {
      return false;
    }
  }
  return true;
}

}  // namespace generativeqc::scf::cuda_execution

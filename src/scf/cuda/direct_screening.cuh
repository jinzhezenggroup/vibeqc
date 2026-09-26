#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_queue_index.cuh"

namespace vibeqc::scf::cuda_execution {

/** Apply the shell-level Schwarz and density gate for one direct consumer. */
template <bool Unrestricted, DirectScreeningPurpose Purpose>
__device__ __forceinline__ bool direct_shell_quartet_survives_screening(
    const DeviceBatch& batch, std::size_t first_pair, std::size_t second_pair,
    double screening_tolerance, const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds,
    double* fock_contribution_bound = nullptr) {
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

  double fock_density_bound = fmax(ab.coulomb, cd.coulomb);
  if constexpr (Unrestricted) {
    fock_density_bound = fmax(fock_density_bound, fmax(fmax(ac.exchange_alpha, ac.exchange_beta),
                                                       fmax(ad.exchange_alpha, ad.exchange_beta)));
    fock_density_bound = fmax(fock_density_bound, fmax(fmax(bc.exchange_alpha, bc.exchange_beta),
                                                       fmax(bd.exchange_alpha, bd.exchange_beta)));
  } else {
    // Preserve the established RHF Fock gate exactly: F = J - K/2.
    const double exchange_bound = fmax(fmax(ac.exchange_alpha, ad.exchange_alpha),
                                       fmax(bc.exchange_alpha, bd.exchange_alpha));
    fock_density_bound = fmax(fock_density_bound, 0.5 * exchange_bound);
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

}  // namespace vibeqc::scf::cuda_execution

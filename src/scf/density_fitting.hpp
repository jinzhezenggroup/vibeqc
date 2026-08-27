#ifndef VIBEQC_SCF_DENSITY_FITTING_HPP
#define VIBEQC_SCF_DENSITY_FITTING_HPP

#include <cstddef>
#include <vector>

namespace vibeqc::scf {

/** Conditioning diagnostics and symmetric inverse square root of (P|Q). */
struct DensityFittingMetricFactor {
  std::size_t dimension{};
  std::size_t effective_rank{};
  double absolute_threshold{};
  double condition_number{};
  std::vector<double> inverse_square_root;
};

/**
 * Remove linearly dependent metric eigenvectors and form J^(-1/2).
 *
 * Eigenvalues not exceeding `relative_threshold * largest_eigenvalue` are
 * omitted. The returned matrix remains square so later blocked contractions
 * can retain fixed auxiliary indexing while the effective rank is diagnosed.
 */
[[nodiscard]] DensityFittingMetricFactor factor_density_fitting_metric(
    const std::vector<double>& metric,
    std::size_t dimension,
    double relative_threshold = 1.0e-10);

/** Metric-orthonormalized three-center tensor B(mu, nu, Q). */
struct DensityFittingThreeCenter {
  std::size_t nbf{};
  std::size_t naux{};
  std::size_t effective_rank{};
  // Row-major values use ((mu * nbf + nu) * naux + Q).
  // The auxiliary dimension remains uncompressed so fixed-topology device
  // plans can preserve their indexing after dependent directions are removed.
  std::vector<double> values;
};

/**
 * Apply the symmetric Coulomb-metric inverse square root to (mu nu|P).
 *
 * The returned tensor is B(mu,nu,Q) = sum_P (mu nu|P) J^(-1/2)(P,Q).
 * It is the reusable input shared by the RI-J and RI-K contractions below.
 */
[[nodiscard]] DensityFittingThreeCenter
orthonormalize_density_fitting_three_center(
    const std::vector<double>& three_center, std::size_t nbf,
    const DensityFittingMetricFactor& metric_factor);

/** Coulomb and exchange matrices built from one RHF AO density. */
struct DensityFittingRhfJk {
  std::size_t nbf{};
  std::vector<double> coulomb;
  std::vector<double> exchange;
};

/**
 * Build the host-reference RHF RI-J/K matrices.
 *
 * `density` uses VibeQC's existing closed-shell convention and includes the
 * factor of two for doubly occupied orbitals. The caller therefore assembles
 * the two-electron Fock contribution as J - 0.5 K.
 */
[[nodiscard]] DensityFittingRhfJk build_density_fitting_rhf_jk(
    const DensityFittingThreeCenter& three_center,
    const std::vector<double>& density);

/** Shared Coulomb and matching-spin exchange matrices for UHF. */
struct DensityFittingUhfJk {
  std::size_t nbf{};
  std::vector<double> coulomb;
  std::vector<double> alpha_exchange;
  std::vector<double> beta_exchange;
};

/**
 * Build the host-reference UHF RI-J/K matrices.
 *
 * Coulomb uses alpha + beta density, while each exchange matrix uses only its
 * matching spin density. The caller assembles F_sigma = H + J - K_sigma.
 */
[[nodiscard]] DensityFittingUhfJk build_density_fitting_uhf_jk(
    const DensityFittingThreeCenter& three_center,
    const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density);

/** Deterministic memory-bounded tile policy for future RI-J/K contractions. */
struct DensityFittingTilePlan {
  std::size_t batch_tile{};
  std::size_t ao_pair_tile{};
  std::size_t auxiliary_tile{};
  std::size_t occupied_tile{};
  std::size_t peak_workspace_bytes{};
  bool stores_full_three_center{};
};

/**
 * Select batch, AO-pair, auxiliary, and occupied tiles under a byte budget.
 *
 * The permanent metric inverse square root is included in the budget. The
 * planner never requires the full `(mu nu|P)` tensor when one minimal tile
 * fits, and throws when even the metric plus a one-element tile cannot fit.
 */
[[nodiscard]] DensityFittingTilePlan plan_density_fitting_tiles(
    std::size_t batch_size,
    std::size_t nbf,
    std::size_t naux,
    std::size_t occupied,
    std::size_t memory_budget_bytes);

}  // namespace vibeqc::scf

#endif

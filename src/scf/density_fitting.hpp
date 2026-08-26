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

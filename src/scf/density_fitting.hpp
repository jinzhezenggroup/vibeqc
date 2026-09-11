#ifndef VIBEQC_SCF_DENSITY_FITTING_HPP
#define VIBEQC_SCF_DENSITY_FITTING_HPP

#include <cstddef>
#include <optional>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "scf/fock_build.hpp"

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
    const std::vector<double>& metric, std::size_t dimension, double relative_threshold = 1.0e-10);

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

/** Immutable integral state shared by all iterations of a DF SCF solve. */
struct DensityFittingScfData {
  integrals::IntegralData one_electron;
  integrals::DensityFittingIntegralData raw;
  DensityFittingThreeCenter three_center;
  // The metric cutoff selects the retained Hamiltonian as well as its response.
  // Cached plans must be rebuilt when callers change this numerical control.
  double metric_relative_threshold{};
  // Geometry and execution policy replace complete dA/dM tensors when the
  // generated two-electron response is selected. Both bases keep real owners.
  std::optional<core::System> df_gradient_orbital, df_gradient_auxiliary;
  unsigned df_gradient_mapping{};
  std::size_t df_gradient_budget{};

  /** A fused response retains topology instead of AO derivative tensors.
   * Geometry and mapping belong to this immutable per-geometry SCF data. */
  std::optional<core::System> one_electron_gradient_system;
  int one_electron_gradient_device{-1};
  unsigned one_electron_gradient_mapping{};
  std::size_t one_electron_gradient_budget{};
};

/**
 * Apply the symmetric Coulomb-metric inverse square root to (mu nu|P).
 *
 * The returned tensor is B(mu,nu,Q) = sum_P (mu nu|P) J^(-1/2)(P,Q).
 * It is the reusable input shared by the RI-J and RI-K contractions below.
 */
[[nodiscard]] DensityFittingThreeCenter orthonormalize_density_fitting_three_center(
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
 * the standard HF contribution as J - 0.5 K. Unselected raw outputs are empty
 * and their contractions are skipped, including exchange scratch allocation.
 */
[[nodiscard]] DensityFittingRhfJk build_density_fitting_rhf_jk(
    const DensityFittingThreeCenter& three_center, const std::vector<double>& density,
    JkTermSelection terms = {});

/** Shared Coulomb and matching-spin exchange matrices for UHF. */
struct DensityFittingUhfJk {
  std::size_t nbf{};
  std::vector<double> coulomb;
  std::vector<double> alpha_exchange;
  std::vector<double> beta_exchange;
};

/**
 * Two-electron RHF density-fitting energy derivative.
 *
 * `derivative` is ordered by nuclear coordinate and contains dE2/dR. The
 * matching `forces` vector contains -dE2/dR.  These are the DF two-electron
 * contributions only; one-electron, nuclear-repulsion, and orbital-Pulay
 * terms belong to the surrounding SCF gradient assembly.  The contraction is
 * evaluated from the raw metric and three-center derivatives, so it includes
 * the auxiliary-metric response (the derivative of the metric pseudoinverse)
 * and is independent of the particular inverse-square-root eigenvector gauge.
 */
struct DensityFittingRhfGradient {
  std::size_t ncoord{};
  std::vector<double> derivative;
  std::vector<double> forces;
};

/** Two-electron UHF density-fitting energy derivative for both spins. */
struct DensityFittingUhfGradient {
  std::size_t ncoord{};
  std::vector<double> derivative;
  std::vector<double> forces;
};

/**
 * Build the RHF DF two-electron analytic gradient from raw integral data.
 *
 * The density uses the same closed-shell, doubly occupied convention as
 * `build_density_fitting_rhf_jk`.  `relative_threshold` is applied to every
 * metric before forming its pseudoinverse, matching the value contraction.
 * Coefficients are signed Fock weights; zero skips that response contraction.
 */
[[nodiscard]] DensityFittingRhfGradient build_density_fitting_rhf_gradient(
    const integrals::DensityFittingIntegralData& integrals, const std::vector<double>& density,
    double relative_threshold = 1.0e-10, JkCoefficients coefficients = {});

/**
 * Build the UHF DF two-electron analytic gradient from raw integral data.
 *
 * Coulomb uses alpha + beta density, while exchange response is evaluated
 * independently for each matching-spin density.
 */
[[nodiscard]] DensityFittingUhfGradient build_density_fitting_uhf_gradient(
    const integrals::DensityFittingIntegralData& integrals,
    const std::vector<double>& alpha_density, const std::vector<double>& beta_density,
    double relative_threshold = 1.0e-10, JkCoefficients coefficients = {1.0, -1.0});

/**
 * Construct the thresholded Moore--Penrose inverse of a Coulomb metric.
 *
 * CUDA force-response consumers use this explicit factor so the expensive
 * metric algebra is prepared once on the host while all density/tensor
 * contractions remain device-resident.
 */
[[nodiscard]] std::vector<double> density_fitting_metric_pseudoinverse(
    const integrals::DensityFittingIntegralData& integrals, double relative_threshold = 1.0e-10);

/** Apply the self-adjoint Frechet derivative of the spectrally truncated inverse.
 * For a forward response, response is dM and the result is d(M+). In reverse,
 * response is an external bar_(M+) and the result is bar_M. Matrices are full
 * row-major; the symmetric metric uses the symmetric part of response.
 * Retained/discarded mixing uses (f(lambda_i)-f(lambda_j))/(lambda_i-lambda_j),
 * including nonzero discarded eigenvalues. No eigenvector gauge is differentiated.
 * A positive relative_threshold verifies the supplied inverse's active subspace
 * and rejects numerically unresolved rank crossings at its scaled cutoff.
 * Zero preserves legacy callers' inferred active mask; it cannot diagnose the
 * distance to an unknown threshold. Neither mode changes the value-side rank.
 */
[[nodiscard]] std::vector<double> density_fitting_metric_inverse_response(
    const std::vector<double>& metric, const std::vector<double>& inverse,
    const std::vector<double>& response, std::size_t dimension, double relative_threshold = 0.0);

/** Construct d(M+) for one coordinate, with optional explicit rank-crossing checks. */
[[nodiscard]] std::vector<double> density_fitting_metric_pseudoinverse_derivative(
    const integrals::DensityFittingIntegralData& integrals, const std::vector<double>& inverse,
    std::size_t coordinate, double relative_threshold = 0.0);

/**
 * Assemble a complete RHF analytic force vector for a DF two-electron
 * energy. This combines the DF response above with orbital one-electron
 * derivatives, the orbital-basis Pulay overlap term, and nuclear repulsion.
 */
[[nodiscard]] std::vector<double> build_density_fitting_rhf_forces(
    const integrals::IntegralData& one_electron,
    const integrals::DensityFittingIntegralData& density_fitting,
    const std::vector<double>& density, const std::vector<double>& weighted_density,
    double relative_threshold = 1.0e-10);

/** Complete UHF analytic forces including one-electron and Pulay terms. */
[[nodiscard]] std::vector<double> build_density_fitting_uhf_forces(
    const integrals::IntegralData& one_electron,
    const integrals::DensityFittingIntegralData& density_fitting,
    const std::vector<double>& alpha_density, const std::vector<double>& beta_density,
    const std::vector<double>& alpha_weighted_density,
    const std::vector<double>& beta_weighted_density, double relative_threshold = 1.0e-10);

/**
 * Build the host-reference UHF RI-J/K matrices.
 *
 * Coulomb uses alpha + beta density, while each exchange matrix uses only its
 * matching spin density. The caller assembles F_sigma = H + J - K_sigma.
 */
[[nodiscard]] DensityFittingUhfJk build_density_fitting_uhf_jk(
    const DensityFittingThreeCenter& three_center, const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, JkTermSelection terms = {});

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
 * A zero budget selects the deterministic implementation defaults. Positive
 * budgets are hard limits and fail when the metric plus one minimal tile does
 * not fit.
 *
 * The permanent metric inverse square root is included in the budget. The
 * planner never requires the full `(mu nu|P)` tensor when one minimal tile
 * fits, and throws when even the metric plus a one-element tile cannot fit.
 */
[[nodiscard]] DensityFittingTilePlan plan_density_fitting_tiles(std::size_t batch_size,
                                                                std::size_t nbf, std::size_t naux,
                                                                std::size_t occupied,
                                                                std::size_t memory_budget_bytes,
                                                                std::size_t fixed_device_bytes = 0);

/** Shape-only capacity of the current bounded DF source's owned uploads.
 * Counts include the combined orbital/auxiliary/dummy basis across the batch;
 * transform_elements counts both public-to-Cartesian transform matrices.
 */
[[nodiscard]] std::size_t density_fitting_source_metadata_bytes(
    std::size_t batch, std::size_t atoms, std::size_t shells, std::size_t cartesian_aos,
    std::size_t primitives, std::size_t transform_elements);

}  // namespace vibeqc::scf

#endif

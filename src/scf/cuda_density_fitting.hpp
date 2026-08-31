#ifndef VIBEQC_SCF_CUDA_DENSITY_FITTING_HPP
#define VIBEQC_SCF_CUDA_DENSITY_FITTING_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vibeqc/vibeqc.h"

namespace vibeqc::scf {

struct CudaDensityFittingJkPlan;

/** Scalar state returned by the device-resident DF SCF loop. */
struct CudaDensityFittingDeviceScfItem {
  vibeqc_status status{VIBEQC_STATUS_INTERNAL_ERROR};
  bool converged{};
  unsigned iterations{};
  double energy{};
  double energy_change{};
  double density_rms{};
};

/** Metric conditioning, workspace, and device-allocation diagnostics. */
struct CudaDensityFittingMetricDiagnostic {
  /** Zero-based item within the prepared CUDA DF plan. */
  std::size_t system_index{};
  /** Fleet bucket that owns this plan; zero for direct plan construction. */
  std::size_t bucket_id{};
  std::size_t effective_rank{};
  double absolute_threshold{};
  double condition_number{};
  std::size_t solver_device_workspace_bytes{};
  std::size_t solver_host_workspace_bytes{};
  /** Persistent device bytes retained after setup (all batch systems). */
  std::size_t device_resident_bytes{};
  /** Conservative peak device bytes during plan construction. */
  std::size_t peak_device_bytes{};
  /** Host bytes retained for streamed raw values and inverse metrics. */
  std::size_t host_resident_bytes{};
  /** Conservative host peak while preparing the streamed plan. */
  std::size_t peak_host_bytes{};
  /** Auxiliary tile selected by the planner/backend. */
  std::size_t auxiliary_tile{};
  /** True when transformed three-center values use host-backed tile streaming. */
  bool streamed{};
};

/**
 * Prepare a persistent homogeneous CUDA DF J/K bucket.
 *
 * Metrics use [system][P][Q] row-major storage and three-center integrals use
 * [system][mu][nu][P]. Metric eigendecomposition, inverse-square-root
 * construction, and the three-center metric transform execute on the selected
 * device. When both tile dimensions cover the full tensor, the transformed
 * B(mu,nu,Q) tensor remains resident afterward; smaller auxiliary or AO-pair
 * tiles select a host-backed streaming mode that keeps only active tiles on
 * the device.
 *
 * `auxiliary_tile` bounds the two RI-K GEMM intermediates. Passing zero selects
 * a conservative default capped at 32 auxiliary functions.
 */
vibeqc_status create_cuda_density_fitting_jk_plan(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile,
    CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail);

/**
 * Planner-aware variant that also bounds the staged AO-pair tile.  The
 * compatibility entry point above uses the complete AO matrix as its pair
 * tile; CUDA fleet callers pass the planner's smaller value here when a
 * memory budget requires raw/AO-pair streaming.
 */
vibeqc_status create_cuda_density_fitting_jk_plan_tiled(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile,
    std::size_t ao_pair_tile, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail);

/**
 * Build batched RHF RI-J/K matrices on the plan's non-blocking CUDA stream.
 *
 * The closed-shell density includes double occupation. Returned matrices are
 * row-major and the caller forms the two-electron Fock term as J - 0.5 K.
 */
vibeqc_status execute_cuda_density_fitting_rhf_jk(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& density,
    std::vector<double>& coulomb, std::vector<double>& exchange,
    std::string& detail);

/**
 * Build batched UHF RI-J/K matrices on the plan's non-blocking CUDA stream.
 *
 * Coulomb uses alpha + beta density. Each exchange matrix uses only its
 * matching spin density, so F_sigma = H + J - K_sigma.
 */
vibeqc_status execute_cuda_density_fitting_uhf_jk(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, std::vector<double>& coulomb,
    std::vector<double>& alpha_exchange, std::vector<double>& beta_exchange,
    std::string& detail);

/**
 * Execute one RHF DF J/K contraction directly from device-resident density
 * matrices.  No host transfer is performed; callers own all device pointers
 * and must keep them valid until the plan stream has completed.
 */
vibeqc_status execute_cuda_density_fitting_rhf_jk_device(
    CudaDensityFittingJkPlan* plan, const double* density, double* coulomb,
    double* exchange, std::string& detail);

/** Device-pointer counterpart for unrestricted DF J/K. */
vibeqc_status execute_cuda_density_fitting_uhf_jk_device(
    CudaDensityFittingJkPlan* plan, const double* alpha_density,
    const double* beta_density, double* coulomb, double* alpha_exchange,
    double* beta_exchange, std::string& detail);

/**
 * Evaluate the complete raw-tensor RHF DF two-electron force response on the
 * plan stream.  Inputs are packed by system, with derivative tensors packed
 * by system then coordinate.  The metric pseudoinverse and its derivative are
 * supplied explicitly so thresholding and rank-deficiency policy exactly
 * match the validated host oracle.  Only the compact derivative vector is
 * copied back to the host.
 */
vibeqc_status execute_cuda_density_fitting_rhf_force_response(
    CudaDensityFittingJkPlan* plan,
    const std::vector<double>& raw_three_center,
    const std::vector<double>& metric_inverse,
    const std::vector<double>& three_center_derivative,
    const std::vector<double>& metric_inverse_derivative,
    std::size_t coordinate_count, const std::vector<double>& density,
    std::vector<double>& derivative, std::string& detail);

/** Device counterpart for unrestricted spin-resolved DF force response. */
vibeqc_status execute_cuda_density_fitting_uhf_force_response(
    CudaDensityFittingJkPlan* plan,
    const std::vector<double>& raw_three_center,
    const std::vector<double>& metric_inverse,
    const std::vector<double>& three_center_derivative,
    const std::vector<double>& metric_inverse_derivative,
    std::size_t coordinate_count, const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, std::vector<double>& derivative,
    std::string& detail);

/**
 * Run batched RHF DF SCF with densities, Fock assembly, eigensolves, and
 * convergence reductions resident on the selected CUDA device.  The host
 * supplies immutable one-electron matrices and an initial density once; only
 * the final density and compact scalar records are copied back.
 * The compact device loop uses direct fixed-point updates; callers may fall
 * back to the DIIS reference path when a provider does not converge within
 * the requested iteration budget.
 */
vibeqc_status run_cuda_density_fitting_rhf_device_scf(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& hcore,
    const std::vector<double>& orthogonalizer,
    const std::vector<double>& initial_density,
    const std::vector<std::int32_t>& occupied,
    const std::vector<double>& nuclear_repulsion, unsigned max_iterations,
    double energy_tolerance, double density_tolerance,
    std::vector<double>& final_density,
    std::vector<CudaDensityFittingDeviceScfItem>& results,
    std::string& detail);

/** UHF counterpart of the device-resident DF SCF loop. */
vibeqc_status run_cuda_density_fitting_uhf_device_scf(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& hcore,
    const std::vector<double>& orthogonalizer,
    const std::vector<double>& initial_alpha_density,
    const std::vector<double>& initial_beta_density,
    const std::vector<std::int32_t>& alpha_occupied,
    const std::vector<std::int32_t>& beta_occupied,
    const std::vector<double>& nuclear_repulsion, unsigned max_iterations,
    double energy_tolerance, double density_tolerance,
    std::vector<double>& final_alpha_density,
    std::vector<double>& final_beta_density,
    std::vector<CudaDensityFittingDeviceScfItem>& results,
    std::string& detail);

void destroy_cuda_density_fitting_jk_plan(
    CudaDensityFittingJkPlan* plan) noexcept;

}  // namespace vibeqc::scf

#endif

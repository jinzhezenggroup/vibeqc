#ifndef VIBEQC_SCF_CUDA_DENSITY_FITTING_HPP
#define VIBEQC_SCF_CUDA_DENSITY_FITTING_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "scf/fock_build.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::scf {

struct CudaDensityFittingJkPlan;
struct CudaDensityFittingIntegralSource;
struct CudaDensityFittingMetricDiagnostic;
struct DensityFittingDensityResponse;
struct DfGradientResources;

/** Integral-source placement, distinct from metric rank and J/K plan storage. */
struct CudaDensityFittingSourceDiagnostic {
  const char* value_backend{"unavailable"};
  const char* value_mapping{"unavailable"};
  bool public_transform_on_device{};
  /** The returned metric crosses D2H, then H2D for cuSOLVER factorization. */
  bool metric_staged_on_host{};
};

/** Describe the choices frozen into a source; a null handle is unavailable. */
CudaDensityFittingSourceDiagnostic cuda_density_fitting_integral_source_diagnostic(
    const CudaDensityFittingIntegralSource* source) noexcept;

/** Contract generated DF derivatives using resident raw values or this plan's source.
 * Positive gradients contain only the DF two-electron response. The owning
 * stream is shared with value regeneration; weights and metric reverse work
 * use explicitly bounded host staging. No failure authorizes an oracle retry.
 */
vibeqc_status execute_cuda_density_fitting_generated_force_response(
    CudaDensityFittingJkPlan* plan, std::size_t system, const core::System& orbital,
    const core::System& auxiliary, std::span<const double> raw_a, const std::vector<double>& metric,
    std::span<const DensityFittingDensityResponse> terms, unsigned schedule,
    std::size_t maximum_bytes, std::size_t maximum_auxiliary_tile, std::vector<double>& derivative,
    std::string& detail, DfGradientResources* resources = nullptr);

/**
 * Prepare a device-resident source for bounded DF tile generation.
 *
 * The source owns only packed basis metadata and public-basis transforms. It
 * intentionally does not allocate or retain the O(nbf^2*naux) three-center
 * tensor; callers request individual transformed tiles on demand.
 */
vibeqc_status create_cuda_density_fitting_integral_source(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems, CudaDensityFittingIntegralSource** source,
    std::vector<double>& metrics, std::size_t& nbf, std::size_t& naux, std::string& detail);

void destroy_cuda_density_fitting_integral_source(
    CudaDensityFittingIntegralSource* source) noexcept;

/** Device bytes retained by an opaque bounded DF integral source. */
std::size_t cuda_density_fitting_integral_source_device_bytes(
    const CudaDensityFittingIntegralSource* source) noexcept;

/** Host bytes retained by an opaque bounded DF integral source. */
std::size_t cuda_density_fitting_integral_source_host_bytes(
    const CudaDensityFittingIntegralSource* source) noexcept;

/** Host allocation peak while constructing an opaque bounded DF source. */
std::size_t cuda_density_fitting_integral_source_host_peak_bytes(
    const CudaDensityFittingIntegralSource* source) noexcept;

/** Maximum per-system nuclear-coordinate count represented by a source. */
std::size_t cuda_density_fitting_integral_source_coordinate_count(
    const CudaDensityFittingIntegralSource* source) noexcept;

/** Validate the fixed dimensions/device associated with a source handle. */
bool cuda_density_fitting_integral_source_matches(const CudaDensityFittingIntegralSource* source,
                                                  int device_id, std::size_t batch_size,
                                                  std::size_t nbf, std::size_t naux) noexcept;

/** Prepare a streamed J/K plan that regenerates tiles from `source`. */
vibeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail);

/** Generate one public-basis transformed three-center tile on `stream`. */
vibeqc_status generate_cuda_density_fitting_transformed_tile(
    CudaDensityFittingIntegralSource* source, std::size_t system, std::size_t pair_begin,
    std::size_t pair_count, std::size_t auxiliary_begin, std::size_t auxiliary_count,
    std::int64_t derivative_coordinate, const double* inverse_square_root, void* stream,
    double* output, std::string& detail);

/**
 * Generate raw A[mu,nu,P] in row-major [pair][auxiliary] order on `stream`.
 * Pairs use pair=mu*nbf+nu with unit weight (no symmetry compression).
 * Empty extents at valid offsets are no-ops; stream/output must remain valid.
 */
vibeqc_status generate_cuda_density_fitting_raw_tile(
    CudaDensityFittingIntegralSource* source, std::size_t system, std::size_t pair_begin,
    std::size_t pair_count, std::size_t auxiliary_begin, std::size_t auxiliary_count,
    std::int64_t derivative_coordinate, void* stream, double* output, std::string& detail);

/** Generate one auxiliary-metric derivative row tile on `stream`. */
vibeqc_status generate_cuda_density_fitting_metric_derivative_tile(
    CudaDensityFittingIntegralSource* source, std::size_t system, std::size_t auxiliary_row_begin,
    std::size_t auxiliary_row_count, std::int64_t derivative_coordinate, void* stream,
    double* output, std::string& detail);

/** Return the fixed batch cardinality owned by a prepared plan. */
std::size_t cuda_density_fitting_jk_plan_batch_size(const CudaDensityFittingJkPlan* plan) noexcept;
/** Verify a borrowed item's dimensions and the value-side metric cutoff before
 * binding an independent Fock/response view. */
bool cuda_density_fitting_jk_plan_matches(const CudaDensityFittingJkPlan* plan, std::size_t item,
                                          std::size_t nbf, std::size_t naux,
                                          double relative_threshold) noexcept;

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
  /** Conservative value/SCF plan setup peak; force bridge storage is separate. */
  std::size_t peak_device_bytes{};
  /** Host bytes retained for streamed raw values and inverse metrics. */
  std::size_t host_resident_bytes{};
  /** Conservative value-plan host setup peak; force bridge storage is separate. */
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
 * the resident full-auxiliary default; budgeted callers should use the tiled
 * entry point to request a smaller streamed tile.
 */
vibeqc_status create_cuda_density_fitting_jk_plan(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail);

/**
 * Planner-aware variant that also bounds the staged AO-pair tile.  The
 * compatibility entry point above uses the complete AO matrix as its pair
 * tile; CUDA fleet callers pass the planner's smaller value here when a
 * memory budget requires raw/AO-pair streaming.
 */
vibeqc_status create_cuda_density_fitting_jk_plan_tiled(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail);

/**
 * Build batched RHF RI-J/K matrices on the plan's non-blocking CUDA stream.
 *
 * The closed-shell density includes double occupation. Returned matrices are
 * row-major and the caller forms the standard HF term as J - 0.5 K. `terms`
 * skips unrequested contractions and downloads; absent host outputs are empty.
 * The prepared plan retains its accounted scratch capacity for later replays.
 */
vibeqc_status execute_cuda_density_fitting_rhf_jk(CudaDensityFittingJkPlan* plan,
                                                  const std::vector<double>& density,
                                                  std::vector<double>& coulomb,
                                                  std::vector<double>& exchange,
                                                  std::string& detail, JkTermSelection terms = {});

/**
 * Build batched UHF RI-J/K matrices on the plan's non-blocking CUDA stream.
 *
 * Coulomb uses alpha + beta density. Each exchange matrix uses only its
 * matching spin density, so F_sigma = H + J - K_sigma.
 */
vibeqc_status execute_cuda_density_fitting_uhf_jk(CudaDensityFittingJkPlan* plan,
                                                  const std::vector<double>& alpha_density,
                                                  const std::vector<double>& beta_density,
                                                  std::vector<double>& coulomb,
                                                  std::vector<double>& alpha_exchange,
                                                  std::vector<double>& beta_exchange,
                                                  std::string& detail, JkTermSelection terms = {});

/**
 * Build one RHF J/K item without packing a complete batch on the host.
 *
 * The plan still owns the fixed batch-stride device buffers, but only the
 * selected item's density and outputs cross the host/device boundary. This is
 * used by bucket finalization under a positive memory budget.
 */
vibeqc_status execute_cuda_density_fitting_rhf_jk_item(
    CudaDensityFittingJkPlan* plan, std::size_t system, const std::vector<double>& density,
    std::vector<double>& coulomb, std::vector<double>& exchange, std::string& detail,
    JkTermSelection terms = {});

/** UHF counterpart of the bounded item-level J/K helper. */
vibeqc_status execute_cuda_density_fitting_uhf_jk_item(
    CudaDensityFittingJkPlan* plan, std::size_t system, const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, std::vector<double>& coulomb,
    std::vector<double>& alpha_exchange, std::vector<double>& beta_exchange, std::string& detail,
    JkTermSelection terms = {});

/**
 * Execute one RHF DF J/K contraction directly from device-resident density
 * matrices.  No host transfer is performed; callers own all device pointers
 * and must keep them valid until the plan stream has completed. Unselected
 * output pointers may be null and are never accessed. Selected outputs are raw
 * and unscaled, in row-major order; external assembly applies coefficients
 * exactly once. Density layout is explicit: the default preserves the legacy
 * column-major SCF buffer convention. Row-major callers reuse the existing
 * transpose staging; nonsymmetric densities retain their orientation.
 */
vibeqc_status execute_cuda_density_fitting_rhf_jk_device(
    CudaDensityFittingJkPlan* plan, const double* density, double* coulomb, double* exchange,
    std::string& detail, JkTermSelection terms = {},
    FockMatrixLayout density_layout = FockMatrixLayout::ColumnMajor);

/** Device-pointer counterpart for unrestricted DF J/K. */
vibeqc_status execute_cuda_density_fitting_uhf_jk_device(
    CudaDensityFittingJkPlan* plan, const double* alpha_density, const double* beta_density,
    double* coulomb, double* alpha_exchange, double* beta_exchange, std::string& detail,
    JkTermSelection terms = {}, FockMatrixLayout density_layout = FockMatrixLayout::ColumnMajor);

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
    const std::vector<double>& orthogonalizer, const std::vector<double>& initial_density,
    const std::vector<std::int32_t>& occupied, const std::vector<double>& nuclear_repulsion,
    unsigned max_iterations, double energy_tolerance, double density_tolerance,
    std::vector<double>& final_density, std::vector<CudaDensityFittingDeviceScfItem>& results,
    std::string& detail);

/** UHF counterpart of the device-resident DF SCF loop. */
vibeqc_status run_cuda_density_fitting_uhf_device_scf(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& hcore,
    const std::vector<double>& orthogonalizer, const std::vector<double>& initial_alpha_density,
    const std::vector<double>& initial_beta_density,
    const std::vector<std::int32_t>& alpha_occupied, const std::vector<std::int32_t>& beta_occupied,
    const std::vector<double>& nuclear_repulsion, unsigned max_iterations, double energy_tolerance,
    double density_tolerance, std::vector<double>& final_alpha_density,
    std::vector<double>& final_beta_density, std::vector<CudaDensityFittingDeviceScfItem>& results,
    std::string& detail);

void destroy_cuda_density_fitting_jk_plan(CudaDensityFittingJkPlan* plan) noexcept;

}  // namespace vibeqc::scf

#endif

#ifndef VIBEQC_SCF_CUDA_DENSITY_FITTING_HPP
#define VIBEQC_SCF_CUDA_DENSITY_FITTING_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "vibeqc/vibeqc.h"

namespace vibeqc::scf {

struct CudaDensityFittingJkPlan;

/** Metric conditioning reported after device-side eigendecomposition. */
struct CudaDensityFittingMetricDiagnostic {
  std::size_t effective_rank{};
  double absolute_threshold{};
  double condition_number{};
};

/**
 * Prepare a persistent homogeneous CUDA DF J/K bucket.
 *
 * Metrics use [system][P][Q] row-major storage and three-center integrals use
 * [system][mu][nu][P]. Metric eigendecomposition, inverse-square-root
 * construction, and the three-center metric transform execute on the selected
 * device. Only the transformed B(mu,nu,Q) tensor remains resident afterward.
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

void destroy_cuda_density_fitting_jk_plan(
    CudaDensityFittingJkPlan* plan) noexcept;

}  // namespace vibeqc::scf

#endif

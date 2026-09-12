#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_plan_setup.hpp"
#include "scf/df_exchange_policy.hpp"

namespace vibeqc::scf {
using namespace cuda_df;

// Public plan adapters retain their opaque ABI and source-transfer contract.
vibeqc_status create_cuda_density_fitting_jk_plan_tiled(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail) {
  return create_cuda_density_fitting_jk_plan_tiled_impl(
      device_id, batch_size, nbf, naux, metrics, three_center, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail, nullptr);
}

vibeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail) {
  if (plan != nullptr) *plan = nullptr;
  if (source == nullptr || *source == nullptr) {
    detail = "source-backed CUDA DF plan requires a source";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  // The implementation owns the handle for the duration of this call,
  // including validation/allocation failures.  It destroys the handle on
  // every failure path; clear the caller slot unconditionally below.
  vibeqc_status status = create_cuda_density_fitting_jk_plan_tiled_impl(
      device_id, batch_size, nbf, naux, metrics, {}, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail, *source);
  if (status == VIBEQC_STATUS_SUCCESS) {
    *source = nullptr;  // ownership transfers to the prepared plan
  } else {
    // The implementation has already destroyed the transferred source on
    // all failure paths.  Keep the caller handle null to make cleanup safe.
    *source = nullptr;
  }
  return status;
}

vibeqc_status create_cuda_density_fitting_jk_plan(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail) {
  // The compatibility API is the resident/default path: keep the complete
  // auxiliary dimension so large-AO warm replays can capture the SCF Graph.
  // Budgeted callers use the tiled entry point directly and may select host
  // streaming when the full transformed tensor cannot fit.
  const std::size_t resident_auxiliary_tile = auxiliary_tile == 0U ? naux : auxiliary_tile;
  return create_cuda_density_fitting_jk_plan_tiled(
      device_id, batch_size, nbf, naux, metrics, three_center, relative_threshold,
      resident_auxiliary_tile, 0, plan, diagnostics, detail);
}

void destroy_cuda_density_fitting_jk_plan(CudaDensityFittingJkPlan* plan) noexcept {
  if (plan == nullptr) return;
  release(*plan);
  delete plan;
}

std::size_t cuda_density_fitting_jk_plan_batch_size(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan == nullptr ? 0U : plan->batch_size;
}

bool cuda_density_fitting_scf_policy_matches(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan != nullptr && plan->occupied_scf_reserved == df_occupied_exchange_requested();
}

bool cuda_density_fitting_jk_plan_matches(const CudaDensityFittingJkPlan* plan, std::size_t item,
                                          std::size_t nbf, std::size_t naux,
                                          double relative_threshold) noexcept {
  return plan && item < plan->batch_size && plan->nbf == nbf && plan->naux == naux &&
         plan->metric_relative_threshold == relative_threshold;
}

}  // namespace vibeqc::scf

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
    std::string& detail, std::size_t automatic_rhf_rank) {
  return create_cuda_density_fitting_jk_plan_tiled_impl(
      device_id, batch_size, nbf, naux, metrics, three_center, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail, nullptr, false, {}, automatic_rhf_rank);
}

vibeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail) {
  return create_cuda_density_fitting_jk_plan_from_source(
      device_id, source, batch_size, nbf, naux, metrics, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail, false);
}

vibeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail, bool retain_three_center) {
  return create_cuda_density_fitting_jk_plan_from_source(
      device_id, source, batch_size, nbf, naux, metrics, relative_threshold, auxiliary_tile,
      ao_pair_tile, plan, diagnostics, detail, retain_three_center, {});
}

vibeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int device_id, CudaDensityFittingIntegralSource** source, std::size_t batch_size,
    std::size_t nbf, std::size_t naux, const std::vector<double>& metrics,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail, bool retain_three_center, DfValueStorageOptions storage,
    std::size_t automatic_rhf_rank) {
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
      ao_pair_tile, plan, diagnostics, detail, *source, retain_three_center, storage,
      automatic_rhf_rank);
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
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail,
    std::size_t automatic_rhf_rank) {
  // The compatibility API is the resident/default path: keep the complete
  // auxiliary dimension so large-AO warm replays can capture the SCF Graph.
  // Budgeted callers use the tiled entry point directly and may select host
  // streaming when the full transformed tensor cannot fit.
  const std::size_t resident_auxiliary_tile = auxiliary_tile == 0U ? naux : auxiliary_tile;
  return create_cuda_density_fitting_jk_plan_tiled(
      device_id, batch_size, nbf, naux, metrics, three_center, relative_threshold,
      resident_auxiliary_tile, 0, plan, diagnostics, detail, automatic_rhf_rank);
}

void destroy_cuda_density_fitting_jk_plan(CudaDensityFittingJkPlan* plan) noexcept {
  if (plan == nullptr) return;
  release(*plan);
  delete plan;
}

std::size_t cuda_density_fitting_jk_plan_batch_size(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan == nullptr ? 0U : plan->batch_size;
}

void set_cuda_density_fitting_scf_value_budget(CudaDensityFittingJkPlan* plan,
                                               std::size_t budget) noexcept {
  if (plan) plan->scf_value_budget_bytes = budget;
}

std::size_t cuda_density_fitting_scf_value_budget(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan ? plan->scf_value_budget_bytes : 0;
}

void set_cuda_density_fitting_scf_diis_history(CudaDensityFittingJkPlan* plan,
                                               unsigned history) noexcept {
  if (plan) plan->scf_diis_history = history;
}

unsigned cuda_density_fitting_scf_diis_history(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan ? plan->scf_diis_history : 0;
}

bool cuda_density_fitting_scf_policy_matches(const CudaDensityFittingJkPlan* plan) noexcept {
  const char* resident = std::getenv("VIBEQC_DF_RESIDENT_EXCHANGE");
  const char* diis = std::getenv("VIBEQC_DF_DIIS_DOTS");
  return plan != nullptr &&
         (!resident || std::strcmp(resident, "auto") == 0 || std::strcmp(resident, "full") == 0 ||
          std::strcmp(resident, "flat") == 0 || std::strcmp(resident, "legacy") == 0) &&
         (!diis || std::strcmp(diis, "auto") == 0 || std::strcmp(diis, "serial") == 0) &&
         plan->resident_exchange_enabled == df_resident_exchange_requested() &&
         plan->triangular_exchange == df_triangular_exchange_requested() &&
         plan->flat_dense_exchange == df_flat_dense_exchange_requested() &&
         plan->cooperative_diis == df_cooperative_diis_requested() &&
         plan->occupied_scf_reserved == df_occupied_exchange_requested(plan->nbf, plan->naux,
                                                                       plan->batch_size,
                                                                       plan->automatic_rhf_rank);
}

DfPairStorage cuda_density_fitting_pair_storage(const CudaDensityFittingJkPlan* plan) noexcept {
  return plan ? plan->value_storage.pairs : DfPairStorage::Dense;
}

bool cuda_density_fitting_jk_plan_matches(const CudaDensityFittingJkPlan* plan, std::size_t item,
                                          std::size_t nbf, std::size_t naux,
                                          double relative_threshold) noexcept {
  return plan && item < plan->batch_size && plan->nbf == nbf && plan->naux == naux &&
         plan->metric_relative_threshold == relative_threshold;
}

}  // namespace vibeqc::scf

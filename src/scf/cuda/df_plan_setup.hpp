#pragma once

#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf::cuda_df {

/** Own source transfer on entry; publish a plan only after setup succeeds. */
vibeqc_status create_cuda_density_fitting_jk_plan_tiled_impl(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail, CudaDensityFittingIntegralSource* integral_source,
    bool retain_three_center = false, DfValueStorageOptions storage = {},
    std::size_t automatic_rhf_rank = 0);

/** Destroy the sole plan owner, including retained SCF state. */
void release(CudaDensityFittingJkPlan& plan) noexcept;
vibeqc_status fail_plan(CudaDensityFittingJkPlan* plan, vibeqc_status status);

}  // namespace vibeqc::scf::cuda_df

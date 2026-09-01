#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf {

vibeqc_status create_cuda_density_fitting_integral_source(int, const std::vector<core::System>&,
                                                          const std::vector<core::System>&,
                                                          CudaDensityFittingIntegralSource** source,
                                                          std::vector<double>& metrics,
                                                          std::size_t& nbf, std::size_t& naux,
                                                          std::string& detail) {
  if (source != nullptr) *source = nullptr;
  metrics.clear();
  nbf = 0;
  naux = 0;
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

void destroy_cuda_density_fitting_integral_source(CudaDensityFittingIntegralSource*) noexcept {}

std::size_t cuda_density_fitting_integral_source_device_bytes(
    const CudaDensityFittingIntegralSource*) noexcept {
  return 0U;
}

std::size_t cuda_density_fitting_integral_source_host_bytes(
    const CudaDensityFittingIntegralSource*) noexcept {
  return 0U;
}

std::size_t cuda_density_fitting_integral_source_host_peak_bytes(
    const CudaDensityFittingIntegralSource*) noexcept {
  return 0U;
}

std::size_t cuda_density_fitting_integral_source_coordinate_count(
    const CudaDensityFittingIntegralSource*) noexcept {
  return 0U;
}

bool cuda_density_fitting_integral_source_matches(const CudaDensityFittingIntegralSource*, int,
                                                  std::size_t, std::size_t, std::size_t) noexcept {
  return false;
}

vibeqc_status create_cuda_density_fitting_jk_plan_from_source(
    int, CudaDensityFittingIntegralSource**, std::size_t, std::size_t, std::size_t,
    const std::vector<double>&, double, std::size_t, std::size_t, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail) {
  if (plan != nullptr) *plan = nullptr;
  diagnostics.clear();
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status generate_cuda_density_fitting_transformed_tile(CudaDensityFittingIntegralSource*,
                                                             std::size_t, std::size_t, std::size_t,
                                                             std::size_t, std::size_t, std::int64_t,
                                                             const double*, void*, double*,
                                                             std::string& detail) {
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status generate_cuda_density_fitting_raw_tile(CudaDensityFittingIntegralSource*, std::size_t,
                                                     std::size_t, std::size_t, std::size_t,
                                                     std::size_t, std::int64_t, void*, double*,
                                                     std::string& detail) {
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status generate_cuda_density_fitting_metric_derivative_tile(
    CudaDensityFittingIntegralSource*, std::size_t, std::size_t, std::size_t, std::int64_t, void*,
    double*, std::string& detail) {
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status execute_cuda_density_fitting_source_rhf_force_response(
    CudaDensityFittingJkPlan*, std::size_t, const std::vector<double>&, std::size_t,
    std::vector<double>&, std::string& detail) {
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status execute_cuda_density_fitting_source_uhf_force_response(
    CudaDensityFittingJkPlan*, std::size_t, const std::vector<double>&, const std::vector<double>&,
    std::size_t, std::vector<double>&, std::string& detail) {
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

std::size_t cuda_density_fitting_jk_plan_batch_size(const CudaDensityFittingJkPlan*) noexcept {
  return 0U;
}

namespace {

vibeqc_status unavailable(CudaDensityFittingJkPlan** plan, std::string& detail) {
  if (plan != nullptr) *plan = nullptr;
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

}  // namespace

vibeqc_status create_cuda_density_fitting_jk_plan(
    int, std::size_t, std::size_t, std::size_t, const std::vector<double>&,
    const std::vector<double>&, double, std::size_t, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail) {
  diagnostics.clear();
  return unavailable(plan, detail);
}

vibeqc_status create_cuda_density_fitting_jk_plan_tiled(
    int, std::size_t, std::size_t, std::size_t, const std::vector<double>&,
    const std::vector<double>&, double, std::size_t, std::size_t, CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics, std::string& detail) {
  diagnostics.clear();
  return unavailable(plan, detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk(CudaDensityFittingJkPlan*,
                                                  const std::vector<double>&, std::vector<double>&,
                                                  std::vector<double>&, std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk(CudaDensityFittingJkPlan*,
                                                  const std::vector<double>&,
                                                  const std::vector<double>&, std::vector<double>&,
                                                  std::vector<double>&, std::vector<double>&,
                                                  std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk_item(CudaDensityFittingJkPlan*, std::size_t,
                                                       const std::vector<double>&,
                                                       std::vector<double>&, std::vector<double>&,
                                                       std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk_item(CudaDensityFittingJkPlan*, std::size_t,
                                                       const std::vector<double>&,
                                                       const std::vector<double>&,
                                                       std::vector<double>&, std::vector<double>&,
                                                       std::vector<double>&, std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk_device(CudaDensityFittingJkPlan*, const double*,
                                                         double*, double*, std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk_device(CudaDensityFittingJkPlan*, const double*,
                                                         const double*, double*, double*, double*,
                                                         std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_force_response(
    CudaDensityFittingJkPlan*, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, std::size_t, const std::vector<double>&,
    std::vector<double>&, std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_force_response(
    CudaDensityFittingJkPlan*, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, std::size_t, const std::vector<double>&,
    const std::vector<double>&, std::vector<double>&, std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status run_cuda_density_fitting_rhf_device_scf(
    CudaDensityFittingJkPlan*, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<std::int32_t>&, const std::vector<double>&,
    unsigned, double, double, std::vector<double>&, std::vector<CudaDensityFittingDeviceScfItem>&,
    std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status run_cuda_density_fitting_uhf_device_scf(
    CudaDensityFittingJkPlan*, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<std::int32_t>&,
    const std::vector<std::int32_t>&, const std::vector<double>&, unsigned, double, double,
    std::vector<double>&, std::vector<double>&, std::vector<CudaDensityFittingDeviceScfItem>&,
    std::string& detail) {
  return unavailable(nullptr, detail);
}

void destroy_cuda_density_fitting_jk_plan(CudaDensityFittingJkPlan*) noexcept {}

}  // namespace vibeqc::scf

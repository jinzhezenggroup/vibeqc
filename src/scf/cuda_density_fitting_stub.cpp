#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf {
namespace {

vibeqc_status unavailable(CudaDensityFittingJkPlan** plan,
                          std::string& detail) {
  if (plan != nullptr) *plan = nullptr;
  detail = "CUDA density-fitting support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

}  // namespace

vibeqc_status create_cuda_density_fitting_jk_plan(
    int, std::size_t, std::size_t, std::size_t, const std::vector<double>&,
    const std::vector<double>&, double, std::size_t,
    CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail) {
  diagnostics.clear();
  return unavailable(plan, detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk(CudaDensityFittingJkPlan*,
                                                  const std::vector<double>&,
                                                  std::vector<double>&,
                                                  std::vector<double>&,
                                                  std::string& detail) {
  return unavailable(nullptr, detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk(
    CudaDensityFittingJkPlan*, const std::vector<double>&,
    const std::vector<double>&, std::vector<double>&, std::vector<double>&,
    std::vector<double>&, std::string& detail) {
  return unavailable(nullptr, detail);
}

void destroy_cuda_density_fitting_jk_plan(CudaDensityFittingJkPlan*) noexcept {}

}  // namespace vibeqc::scf

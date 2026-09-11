#include "scf/cuda_direct_jk.hpp"

namespace vibeqc::scf {
namespace {
vibeqc_status unavailable(std::string& detail) {
  detail = "CUDA direct J/K support is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}
}  // namespace
vibeqc_status create_cuda_direct_jk_plan(int, const std::vector<core::System>&, unsigned, double,
                                         std::size_t, CudaDirectJkPlan** output,
                                         CudaDirectJkDiagnostic& diagnostic, std::string& detail) {
  if (output) *output = nullptr;
  diagnostic = {};
  return unavailable(detail);
}
void destroy_cuda_direct_jk_plan(CudaDirectJkPlan*) noexcept {}
vibeqc_status execute_cuda_direct_jk(CudaDirectJkPlan*, FockBuildSpec, const std::vector<double>&,
                                     const std::vector<double>&, std::vector<double>&,
                                     std::vector<double>&, std::vector<double>&,
                                     std::string& detail) {
  return unavailable(detail);
}
vibeqc_status execute_cuda_direct_energy_derivative(CudaDirectJkPlan*, FockBuildSpec,
                                                    const std::vector<double>&,
                                                    const std::vector<double>&,
                                                    std::vector<double>&, std::string& detail) {
  return unavailable(detail);
}
CudaDirectJkDiagnostic cuda_direct_jk_plan_diagnostic(const CudaDirectJkPlan*) noexcept {
  return {};
}
vibeqc_status execute_cuda_direct_jk_item(CudaDirectJkPlan*, std::size_t, FockBuildSpec,
                                          const std::vector<double>&, const std::vector<double>&,
                                          std::vector<double>&, std::vector<double>&,
                                          std::vector<double>&, std::string& detail) {
  return unavailable(detail);
}
vibeqc_status execute_cuda_direct_energy_derivative_item(CudaDirectJkPlan*, std::size_t,
                                                         FockBuildSpec, const std::vector<double>&,
                                                         const std::vector<double>&,
                                                         std::vector<double>&,
                                                         std::string& detail) {
  return unavailable(detail);
}
}  // namespace vibeqc::scf

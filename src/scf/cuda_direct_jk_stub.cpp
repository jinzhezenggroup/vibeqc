#include <stdexcept>

#include "runtime/provider_registry.hpp"
#include "scf/cuda_direct_jk.hpp"

namespace vibeqc::scf {
namespace {
vibeqc_status unavailable(std::string& detail) {
  return runtime::provider_not_implemented(
      fock_provider_registration(FockApproximation::Exact, FockBackend::Cuda), detail,
      "CUDA direct J/K");
}
}  // namespace
std::size_t cuda_direct_jk_device_bytes(std::size_t, std::size_t, std::size_t, std::size_t,
                                        std::size_t, unsigned) {
  throw std::runtime_error(runtime::provider_diagnostic(
      fock_provider_registration(FockApproximation::Exact, FockBackend::Cuda),
      "CUDA direct J/K allocation inventory"));
}
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

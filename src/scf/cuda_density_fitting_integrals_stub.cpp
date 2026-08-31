#include "scf/cuda_density_fitting_integrals.hpp"

namespace vibeqc::scf {

vibeqc_status build_cuda_density_fitting_integrals(
    int, const core::System&, const core::System&,
    integrals::DensityFittingIntegralData&, std::string& detail) {
  detail = "CUDA density-fitting integral generation is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status build_cuda_density_fitting_integrals_batch(
    int, const std::vector<core::System>&,
    const std::vector<core::System>&,
    std::vector<integrals::DensityFittingIntegralData>& outputs,
    std::string& detail, std::size_t) {
  outputs.clear();
  detail = "CUDA density-fitting integral generation is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status build_cuda_one_electron_integrals_batch(
    int, const std::vector<core::System>&,
    std::vector<integrals::IntegralData>& outputs, std::string& detail) {
  outputs.clear();
  detail = "CUDA one-electron integral generation is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

vibeqc_status build_cuda_one_electron_integrals(
    int, const core::System&, integrals::IntegralData&, std::string& detail) {
  detail = "CUDA one-electron integral generation is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

}  // namespace vibeqc::scf

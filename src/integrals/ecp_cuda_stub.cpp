#include "integrals/ecp_cuda.hpp"
namespace vibeqc::integrals {
vibeqc_status ecp_integrals_cuda(int, const core::System&, unsigned, unsigned, bool, EcpData&,
                                 std::string& detail, bool) {
  detail = "ECP CUDA was not compiled";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}
vibeqc_status add_ecp_cuda(int, const core::System&, void*, double*, const double*, double*,
                           std::string& detail) {
  detail = "ECP CUDA was not compiled";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}
}  // namespace vibeqc::integrals

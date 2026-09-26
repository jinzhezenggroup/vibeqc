#include "integrals/ecp_cuda.hpp"
#include "runtime/provider_registry.hpp"
namespace vibeqc::integrals {
namespace {
struct EcpCudaDomain {
  std::uint32_t maximum_derivative_order{1};
};
constexpr runtime::ProviderDescriptor<EcpCudaDomain> kEcpCudaProvider{
    {"integrals.ecp", "cuda.scalar", 1, runtime::ProviderBackend::Cuda},
    {},
    runtime::ProviderAvailability::NotBuilt,
    runtime::ProviderFallback::None,
    runtime::ProviderRequirement::Resources,
    "CUDA support was not compiled into this build",
    "src/integrals/ecp_cuda_stub.cpp"};

vibeqc_status unavailable(std::string& detail, std::string_view operation) {
  return runtime::provider_not_implemented(kEcpCudaProvider, detail, operation);
}
}  // namespace

vibeqc_status ecp_integrals_cuda(int, const core::System&, unsigned, unsigned, bool, EcpData&,
                                 std::string& detail, bool) {
  return unavailable(detail, "ECP integral evaluation");
}
vibeqc_status add_ecp_cuda(int, const core::System&, void*, double*, const double*, double*,
                           std::string& detail) {
  return unavailable(detail, "ECP operator application");
}
}  // namespace vibeqc::integrals

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>

#include "integrals/ecp.hpp"
#include "integrals/ecp_cuda.hpp"
#include "vibeqc/vibeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check_grid_wrapper() {
  std::vector<vibeqc::integrals::EcpRadialPoint> radial;
  std::vector<vibeqc::integrals::EcpSpherePoint> sphere;
  vibeqc::integrals::ecp_quadrature(16, 8, radial, sphere);
  require(radial.size() == 16 && sphere.size() == 128, "native grid wrapper shape");
  double area = 0;
  for (const auto& p : sphere) area += p.weight;
  require(std::abs(area - 4 * std::numbers::pi) < 1e-12, "native grid wrapper weights");
  vibeqc::integrals::ecp_quadrature(16, 8, radial, sphere);
  require(radial.size() == 32 && sphere.size() == 256, "native grid wrapper append contract");
}

void check_c_api() {
  const vibeqc_context_descriptor context_desc{sizeof(context_desc), VIBEQC_ABI_VERSION, 0,
                                               VIBEQC_BACKEND_CPU_REFERENCE};
  vibeqc_context* raw{};
  require(vibeqc_context_create(&context_desc, &raw) == VIBEQC_STATUS_SUCCESS, "context create");
  std::unique_ptr<vibeqc_context, decltype(&vibeqc_context_destroy)> context(
      raw, vibeqc_context_destroy);
  const std::array<vibeqc_atom, 2> atoms{{{11, 0, 0, 0}, {1, 0.3, 0.1, 3.0}}};
  const std::array<vibeqc_primitive, 2> primitives{{{0.55, 1}, {0.7, 1}}};
  const std::array<int32_t, 2> cores{10, 0};
  for (unsigned center : {0U, 1U})
    for (auto representation : {VIBEQC_BASIS_CARTESIAN, VIBEQC_BASIS_SPHERICAL})
      for (unsigned angular : {3U, 4U}) {
        std::array<vibeqc_shell, 2> shells{{{0, 0, 0, 1}, {1, 0, 1, 1}}};
        shells[center].angular_momentum = angular;
        const vibeqc_system_descriptor descriptor{sizeof(descriptor),
                                                  VIBEQC_ABI_VERSION,
                                                  atoms.data(),
                                                  atoms.size(),
                                                  shells.data(),
                                                  shells.size(),
                                                  primitives.data(),
                                                  primitives.size(),
                                                  0,
                                                  1,
                                                  representation};
        const vibeqc_ecp_term term{0, -1, 2, 0.8, -2};
        vibeqc_system* system{};
        const auto status =
            vibeqc_system_create_ecp(context.get(), &descriptor, cores.data(), &term, 1, &system);
        std::unique_ptr<vibeqc_system, decltype(&vibeqc_system_destroy)> owner(
            system, vibeqc_system_destroy);
        require(status == (angular == 3 ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_INVALID_ARGUMENT),
                "native ECP orbital-f/g capability mismatch");
        require((system != nullptr) == (angular == 3), "failed ECP constructor published output");
        if (angular == 3) {
          const std::array<vibeqc_ecp_term, 2> supported{{term, {0, 3, 2, 0.63, 0.74}}};
          vibeqc_system* extended{};
          require(
              vibeqc_system_create_ecp(context.get(), &descriptor, cores.data(), supported.data(),
                                       supported.size(), &extended) == VIBEQC_STATUS_SUCCESS &&
                  extended,
              "native f projector rejected");
          vibeqc_system_destroy(extended);
          const std::array<vibeqc_ecp_term, 2> unsupported{{term, {0, 4, 2, 0.8, -2}}};
          vibeqc_system* bad{};
          require(vibeqc_system_create_ecp(context.get(), &descriptor, cores.data(),
                                           unsupported.data(), unsupported.size(),
                                           &bad) == VIBEQC_STATUS_INVALID_ARGUMENT &&
                      !bad,
                  "projector f support silently enabled projector g");
        }
      }
}
#if !VIBEQC_HAS_CUDA
void check_cuda_not_built_stub() {
  vibeqc::core::System system;
  vibeqc::integrals::EcpData output;
  std::string detail;
  require(vibeqc::integrals::ecp_integrals_cuda(0, system, 160, 32, false, output, detail) ==
              VIBEQC_STATUS_NOT_IMPLEMENTED,
          "CPU-only ECP CUDA stub lost not-implemented status");
  require(detail.find("cuda.scalar") != std::string::npos &&
              detail.find("not built") != std::string::npos,
          "CPU-only ECP CUDA stub lost provider-specific diagnostics");
}
#endif

}  // namespace

int main() {
  try {
    check_grid_wrapper();
    check_c_api();
#if !VIBEQC_HAS_CUDA
    check_cuda_not_built_stub();
#endif
    std::cout << "ECP native orbital-f boundary and host-grid wrapper passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

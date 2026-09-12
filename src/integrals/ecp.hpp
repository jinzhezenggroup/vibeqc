#pragma once

#include <vector>

#include "core/types.hpp"

namespace vibeqc::integrals {

// Residual Gaussian terms: c r^(power-2) exp(-exponent r^2).
// The Coulomb tail -Zeff/r is evaluated by nuclear attraction separately.
struct EcpData {
  std::size_t nbf{}, ncoord{};
  std::vector<double> local, nonlocal, local_derivative, nonlocal_derivative;
};

struct EcpSpherePoint {
  double x, y, z, weight;
  double harmonics[9];
};
struct EcpRadialPoint {
  double r, weight;
};
void ecp_quadrature(unsigned radial, unsigned angular, std::vector<EcpRadialPoint>& radii,
                    std::vector<EcpSpherePoint>& sphere);

EcpData ecp_integrals(const core::System& system, unsigned radial = 160, unsigned angular = 32,
                      bool derivatives = true);
EcpData checked_ecp_integrals(const core::System& system, bool derivatives);
void add_ecp(const EcpData& ecp, std::vector<double>& hcore, std::vector<double>& derivative);

}  // namespace vibeqc::integrals

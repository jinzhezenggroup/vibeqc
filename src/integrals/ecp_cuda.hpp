#pragma once
#include <string>

#include "integrals/ecp.hpp"

namespace vibeqc::integrals {
// Host export for independent validation and existing host-visible consumers.
vibeqc_status ecp_integrals_cuda(int device, const core::System& system, unsigned radial,
                                 unsigned polar, bool derivatives, EcpData& output,
                                 std::string& detail, bool convergence = false);
// Production device consumer: add values to hcore and/or -D:dV to forces.
// Pointers and stream are borrowed; no host integral or density staging.
vibeqc_status add_ecp_cuda(int device, const core::System& system, void* stream, double* hcore,
                           const double* density, double* forces, std::string& detail);
}  // namespace vibeqc::integrals

#pragma once

#include <vector>

#include "posthf/mp2_gradient.hpp"

namespace vibeqc::core {
struct System;
}
namespace vibeqc::scf {
struct PhysicalReference;
}

namespace vibeqc::mp2 {

/** Contract relaxed canonical-MO Lagrangian weights with CPU derivatives.
 *
 * Returns positive total-energy derivatives. One-electron weights are pulled
 * back into public AO matrices. Four-index weights are transformed one shell
 * at a time and immediately consumed, so no molecular AO-rank-four cotangent
 * or coordinate-major derivative tensor exists.
 */
std::vector<double> conventional_derivative_cpu(const core::System& system,
                                                const scf::PhysicalReference& reference,
                                                const LagrangianWeights& weights);

std::vector<double> conventional_derivative_cuda(const core::System& system,
                                                 const scf::PhysicalReference& reference,
                                                 const LagrangianWeights& weights, int device_id,
                                                 std::size_t stage_budget);

}  // namespace vibeqc::mp2

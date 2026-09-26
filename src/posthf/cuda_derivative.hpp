#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string>

#include "core/types.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::posthf {

/** Contract one public-AO shell-quartet cotangent with CUDA ERI derivatives.
 *
 * The four center derivatives remain independent; the caller owns physical-
 * atom scatter. Caller weights, output, and system storage are outside
 * stage_budget. Output is replaced only after complete success.
 */
vibeqc_status contract_weighted_eri_shell_derivative_cuda(
    int device_id, const core::System& system, const std::array<std::size_t, 4>& shell_indices,
    std::span<const double> weights, std::size_t stage_budget,
    std::array<double, 12>& center_gradient, std::string& detail);

}  // namespace vibeqc::posthf

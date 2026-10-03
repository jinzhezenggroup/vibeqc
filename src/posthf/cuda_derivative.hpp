#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "core/types.hpp"
#include "generativeqc/generativeqc.h"

namespace generativeqc::posthf {

/** Successful consumer work for one shell; timings include its existing fence.
 * Caller record construction is excluded from consumer_nanoseconds. No extra
 * CUDA synchronization is introduced; a null diagnostic avoids clock reads.
 */
struct CudaShellDerivativeDiagnostic {
  std::size_t primitive_records{};
  std::size_t consumer_calls{};
  std::uint64_t consumer_nanoseconds{};
};

/** Contract one public-AO shell-quartet cotangent with CUDA ERI derivatives.
 *
 * The four center derivatives remain independent; the caller owns physical-
 * atom scatter. Caller weights, output, and system storage are outside
 * stage_budget. Output is replaced only after complete success.
 */
generativeqc_status contract_weighted_eri_shell_derivative_cuda(
    int device_id, const core::System& system, const std::array<std::size_t, 4>& shell_indices,
    std::span<const double> weights, std::size_t stage_budget,
    std::array<double, 12>& center_gradient, std::string& detail,
    CudaShellDerivativeDiagnostic* diagnostic = nullptr);

}  // namespace generativeqc::posthf

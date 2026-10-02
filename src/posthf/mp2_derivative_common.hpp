#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "posthf/mp2_gradient.hpp"

namespace generativeqc::core {
struct System;
}
namespace generativeqc::hf {
struct PhysicalReference;
}

namespace generativeqc::mp2::detail {

using OneElectronDerivativeContract =
    std::function<std::vector<double>(std::span<const double>, std::span<const double>)>;
using EriShellDerivativeContract = std::function<std::array<double, 12>(
    const std::array<std::size_t, 4>&, std::span<const double>)>;

/** Consume each local cotangent immediately, accumulating into an atom gradient.
 * A bounded consumer may defer part of its work until finalize; no weights or
 * gradient spans may escape the call. The gradient is published only after
 * successful finalization and a final finiteness check.
 */
using EriShellDerivativeAccumulate = std::function<void(
    const std::array<std::size_t, 4>&, std::span<const double>, std::span<double>)>;
std::vector<double> conventional_derivative_accumulate(
    const core::System& system, const hf::PhysicalReference& reference,
    const LagrangianWeights& weights, const OneElectronDerivativeContract& one_electron,
    const EriShellDerivativeAccumulate& eri_shell,
    const std::function<void(std::span<double>)>& finalize = {});

std::vector<double> conventional_derivative(const core::System& system,
                                            const hf::PhysicalReference& reference,
                                            const LagrangianWeights& weights,
                                            const OneElectronDerivativeContract& one_electron,
                                            const EriShellDerivativeContract& eri_shell);

}  // namespace generativeqc::mp2::detail

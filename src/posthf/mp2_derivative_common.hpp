#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

#include "posthf/mp2_gradient.hpp"

namespace vibeqc::core {
struct System;
}
namespace vibeqc::scf {
struct PhysicalReference;
}

namespace vibeqc::mp2::detail {

using OneElectronDerivativeContract =
    std::function<std::vector<double>(std::span<const double>, std::span<const double>)>;
using EriShellDerivativeContract = std::function<std::array<double, 12>(
    const std::array<std::size_t, 4>&, std::span<const double>)>;

std::vector<double> conventional_derivative(const core::System& system,
                                            const scf::PhysicalReference& reference,
                                            const LagrangianWeights& weights,
                                            const OneElectronDerivativeContract& one_electron,
                                            const EriShellDerivativeContract& eri_shell);

}  // namespace vibeqc::mp2::detail

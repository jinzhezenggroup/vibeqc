#pragma once
#include "methods/method.hpp"
namespace vibeqc::methods::detail {
vibeqc_status validate_mp2_system(vibeqc_method, const core::System&, std::string&);
std::unique_ptr<PreparedCalculation> prepare_mp2_calculation(const Capabilities&,
                                                             core::ContextState&,
                                                             const core::System&,
                                                             const vibeqc_method_descriptor&);
}  // namespace vibeqc::methods::detail

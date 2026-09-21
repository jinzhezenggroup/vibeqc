#pragma once

#include "methods/method.hpp"

namespace vibeqc::methods::detail {
vibeqc_status validate_rccsdt_system(vibeqc_method, const core::System&, std::string&);
std::unique_ptr<PreparedCalculation> prepare_rccsdt_calculation(const Capabilities&,
                                                                core::ContextState&,
                                                                const core::System&,
                                                                const vibeqc_method_descriptor&);
std::unique_ptr<PreparedBatch> prepare_rccsdt_batch(const Capabilities&, core::ContextState&,
                                                    std::vector<core::System>,
                                                    const vibeqc_method_descriptor&,
                                                    vibeqc_batch_flags);
}  // namespace vibeqc::methods::detail

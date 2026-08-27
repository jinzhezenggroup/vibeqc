#ifndef VIBEQC_METHODS_HF_METHOD_HPP
#define VIBEQC_METHODS_HF_METHOD_HPP

#include "methods/method.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vibeqc::methods::detail {

vibeqc_status validate_hf_system(vibeqc_method method,
                                 const core::System& system,
                                 std::string& detail);

std::unique_ptr<PreparedCalculation> prepare_hf_calculation(
    const Capabilities& capabilities,
    core::ContextState& context,
    const core::System& system,
    const vibeqc_method_descriptor& descriptor);

std::unique_ptr<PreparedBatch> prepare_hf_batch(
    const Capabilities& capabilities,
    core::ContextState& context,
    std::vector<core::System> systems,
    const vibeqc_method_descriptor& descriptor,
    vibeqc_batch_flags flags);

}  // namespace vibeqc::methods::detail

#endif

#ifndef VIBEQC_METHODS_XTB_METHOD_HPP
#define VIBEQC_METHODS_XTB_METHOD_HPP

#include <memory>
#include <string>

#include "core/types.hpp"
#include "methods/method.hpp"

namespace vibeqc::methods::detail {

vibeqc_status validate_xtb_system(vibeqc_method method, const core::System& system,
                                  std::string& detail);

std::unique_ptr<PreparedCalculation> prepare_xtb_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor);

}  // namespace vibeqc::methods::detail

#endif

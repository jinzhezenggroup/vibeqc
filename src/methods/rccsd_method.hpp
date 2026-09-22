#pragma once
#include "cc/solver.hpp"
#include "methods/method.hpp"

namespace vibeqc::methods::detail {
struct RccsdNativeState {
  cc::Problem problem;
  cc::SolverResult solved;
  std::vector<double> eps_o, eps_v;
  vibeqc_correlation_diagnostic diagnostic{};
  Result result;
  std::size_t budget{};
};

RccsdNativeState run_rccsd_native_state(core::ContextState&, const core::System&,
                                        const vibeqc_method_descriptor&);
vibeqc_status validate_rccsd_system(vibeqc_method, const core::System&, std::string&);
std::unique_ptr<PreparedCalculation> prepare_rccsd_calculation(const Capabilities&,
                                                               core::ContextState&,
                                                               const core::System&,
                                                               const vibeqc_method_descriptor&);
std::unique_ptr<PreparedBatch> prepare_rccsd_batch(const Capabilities&, core::ContextState&,
                                                   std::vector<core::System>,
                                                   const vibeqc_method_descriptor&,
                                                   vibeqc_batch_flags);
}  // namespace vibeqc::methods::detail

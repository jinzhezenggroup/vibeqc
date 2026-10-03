#pragma once
#include <memory>

#include "cc/solver.hpp"
#include "hf/reference.hpp"
#include "methods/method.hpp"

namespace generativeqc::scf {
class PreparedFockPlan;
}
namespace generativeqc::methods::detail {
struct RccsdNativeState {
  std::shared_ptr<const hf::PhysicalReference> reference;
  cc::Problem problem;
  cc::SolverResult solved;
  std::vector<double> eps_o, eps_v;
  generativeqc_correlation_diagnostic diagnostic{};
  CcPerformanceDiagnostic performance{};
  Result result;
  std::size_t budget{};
  std::size_t external_reservation_bytes{};
  double reference_energy_change{};
  double reference_density_rms{};
  int reference_iterations{};
};

// The optional auxiliary is an internal energy-only correlation source. The
// descriptor/reference remain conventional FP64 RHF; public DF-CC(T)/force
// registration is separate and remains unsupported until response qualifies.
RccsdNativeState run_rccsd_native_state(
    runtime::ExecutionContext&, const core::System&, const generativeqc_method_descriptor&,
    std::unique_ptr<scf::PreparedFockPlan>* prepared_exact_cache = nullptr,
    const std::vector<double>* initial_density = nullptr, bool* warm_start_fallback = nullptr,
    std::size_t external_reservation_bytes = 0,
    const core::System* correlation_auxiliary = nullptr);
generativeqc_status validate_rccsd_system(generativeqc_method, const core::System&, std::string&);
std::unique_ptr<PreparedCalculation> prepare_rccsd_calculation(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&);
std::unique_ptr<PreparedBatch> prepare_rccsd_batch(const Capabilities&, core::ContextState&,
                                                   std::vector<core::System>,
                                                   const generativeqc_method_descriptor&,
                                                   generativeqc_batch_flags);
}  // namespace generativeqc::methods::detail

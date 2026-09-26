#ifndef VIBEQC_SCF_SOLVER_MEAN_FIELD_DRIVER_HPP
#define VIBEQC_SCF_SOLVER_MEAN_FIELD_DRIVER_HPP
#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "scf/types.hpp"
namespace vibeqc::scf {
class PreparedFockPlan;
namespace initial_guess {
class OverlapOrthogonalizer;
}
namespace solver {
/** Conventional restricted mean-field iteration against one immutable #202
 * provider. Preparation, mathematical identity and resource ownership belong
 * to the plan; this driver owns DIIS/proposal/convergence state for one solve.
 * Failed solves retain their last iterate and never run converged finalization.
 */
ScfResult run_rhf_host_plan(const core::System& system, const ScfOptions& options,
                            const integrals::IntegralData& ints, const PreparedFockPlan& plan,
                            const std::vector<double>* initial_density,
                            initial_guess::OverlapOrthogonalizer* overlap_cache = nullptr);
/** Unrestricted counterpart; joined DIIS/proposals retain the alpha/beta order. */
ScfResult run_uhf_host_plan(const core::System& system, const ScfOptions& options,
                            const integrals::IntegralData& ints, const PreparedFockPlan& plan,
                            const std::vector<double>* initial_density,
                            initial_guess::OverlapOrthogonalizer* overlap_cache = nullptr);
}  // namespace solver
}  // namespace vibeqc::scf
#endif

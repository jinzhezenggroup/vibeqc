#pragma once

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_scf_state.hpp"

namespace vibeqc::scf::cuda_df {
/** Borrow an immutable qualified entry before begin_scf_final_state_solve
 * revokes public readiness. Exact numerical inputs, not pointer equality or
 * a density norm, bind this optimization to the previous physical state. */
std::shared_ptr<const RhfWarmState> find_rhf_warm_state(const CudaDensityFittingJkPlan* plan,
                                                        const std::vector<double>& density,
                                                        const std::vector<double>& hcore,
                                                        const std::vector<double>& overlap,
                                                        const std::vector<double>& orthogonalizer,
                                                        std::size_t occupied, double nuclear);
}  // namespace vibeqc::scf::cuda_df

#pragma once

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_scf_state.hpp"

namespace vibeqc::scf::cuda_df {

/** Explicit opt-in until fixed-K and complete endpoint evidence selects policy. */
vibeqc_status occupied_scf_policy(bool& enabled, std::string& detail);
vibeqc_status allocate_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                   const std::vector<std::int32_t>& alpha,
                                   const std::vector<std::int32_t>& beta, std::string& detail);
vibeqc_status reset_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                std::string& detail);
vibeqc_status build_scf_occupied_jk(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                    const double* alpha, const double* beta, bool ready,
                                    std::string& detail);
void store_scf_factor(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                      const double* coefficients, bool beta);
vibeqc_status verify_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                 const std::vector<std::uint32_t>& iterations, std::string& detail);

}  // namespace vibeqc::scf::cuda_df

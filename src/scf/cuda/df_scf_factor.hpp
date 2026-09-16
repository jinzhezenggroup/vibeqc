#pragma once

#include <span>

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_scf_state.hpp"

namespace vibeqc::scf::cuda_df {

/** Exact measured resident RHF domain shared by occupied, seed and final selectors.
 * Shape, occupation, storage/provider and actual device identity must all match.
 */
vibeqc_status qualified_resident_rhf_exchange(const CudaDensityFittingJkPlan& plan,
                                              std::span<const std::int32_t> alpha,
                                              std::span<const std::int32_t> beta, bool& qualified,
                                              std::string& detail);

/** Select explicit occupied exchange or the exact qualified automatic domain. */
vibeqc_status occupied_scf_policy(const CudaDensityFittingJkPlan& plan, bool& enabled,
                                  std::string& detail, std::span<const std::int32_t> alpha = {},
                                  std::span<const std::int32_t> beta = {});
vibeqc_status allocate_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                   const std::vector<std::int32_t>& alpha,
                                   const std::vector<std::int32_t>& beta, std::string& detail);
vibeqc_status reset_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                std::string& detail);
/** Factor a singleton resident RHF seed using the existing solver/scratch.
 * Numerical or capacity rejection returns success with accepted=false. Runtime
 * errors propagate. The factor has occupation absorbed and no orbital identity.
 */
vibeqc_status factor_density_for_exchange(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                          const double* density, bool& accepted, std::size_t& rank,
                                          std::string& detail);
vibeqc_status build_scf_occupied_jk(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                    const double* alpha, const double* beta, bool ready,
                                    std::string& detail);
void store_scf_factor(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                      const double* coefficients, bool beta);
vibeqc_status verify_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                 const std::vector<std::uint32_t>& iterations, std::string& detail);

}  // namespace vibeqc::scf::cuda_df

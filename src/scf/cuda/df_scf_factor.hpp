#pragma once

#include <span>

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_scf_state.hpp"

namespace vibeqc::scf::cuda_df {

/** Shared resident capacity/work gate for SCF, seed, final K and force response.
 * Callers must establish singleton RHF and validate their density/factor owner;
 * a profitable rank and sufficient storage never establish provenance.
 */
bool qualified_resident_rhf_exchange(const CudaDensityFittingJkPlan& plan,
                                     std::size_t rank) noexcept;

/** SCF value-only gate, additionally allowing profitable streamed projection.
 * Does not authorize a resident final projection or force-response borrowing.
 */
bool qualified_value_rhf_exchange(const CudaDensityFittingJkPlan& plan, std::size_t rank) noexcept;

/** Select explicit occupied exchange or the qualified RHF value work policy. */
vibeqc_status occupied_scf_policy(const CudaDensityFittingJkPlan& plan, bool& enabled,
                                  std::string& detail, std::span<const std::int32_t> alpha = {},
                                  std::span<const std::int32_t> beta = {});
vibeqc_status allocate_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                   const std::vector<std::int32_t>& alpha,
                                   const std::vector<std::int32_t>& beta, std::string& detail);
vibeqc_status reset_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                std::string& detail);
/** Factor a qualified singleton RHF seed using the existing solver/scratch.
 * Numerical or capacity rejection returns success with accepted=false. Runtime
 * errors propagate. The factor has occupation absorbed and no orbital identity.
 */
vibeqc_status factor_density_for_exchange(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                          const double* density, bool& accepted, std::size_t& rank,
                                          std::string& detail);
vibeqc_status build_scf_occupied_jk(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                    const double* alpha, const double* beta, bool ready,
                                    std::string& detail, bool retained_seed = false);
void store_scf_factor(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                      const double* coefficients, bool beta);
vibeqc_status verify_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                 const std::vector<std::uint32_t>& iterations, std::string& detail);

}  // namespace vibeqc::scf::cuda_df

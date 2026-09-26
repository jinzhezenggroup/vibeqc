#pragma once

#include "scf/cuda_density_fitting_final_state.hpp"

namespace vibeqc::scf::cuda_df {
/** One serialized workspace per prepared plan; never aliases captured state. */
void destroy_final_validation(void*& opaque) noexcept;
/** Exact current physical occupied-J/K lease; queried before force scratch
 * consumers run. It does not grant a raw response projection lease. */
bool final_occupied_fock_matches(const CudaDensityFittingJkPlan& plan,
                                 const CudaDfFinalStateToken& token) noexcept;
}  // namespace vibeqc::scf::cuda_df

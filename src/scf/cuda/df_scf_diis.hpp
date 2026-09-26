#pragma once

#include <string>
#include <vector>

#include "vibeqc/vibeqc.h"

namespace vibeqc::scf {
struct CudaDensityFittingJkPlan;
namespace cuda_df {
struct PersistentScfState;

/** Allocate the requested history within the planner's explicit reservation.
 * RHF and joined-spin UHF reuse the existing shared device DIIS arithmetic. */
vibeqc_status allocate_scf_diis(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                unsigned history, std::string& detail);

/** Refresh the current overlap and clear history generations on every solve. */
vibeqc_status reset_scf_diis(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                             const std::vector<double>& overlap, std::string& detail);

/** Build the physical FDS-SDF residual and replace F only with a DIIS proposal.
 * Call after physical energy evaluation and before eigen/density generation.
 * UHF shares one coefficient vector across its two spin residuals. Inactive
 * items preserve their histories; strict final selection still checks F[D]. */
vibeqc_status apply_scf_diis(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                             std::string& detail);
}  // namespace cuda_df
}  // namespace vibeqc::scf

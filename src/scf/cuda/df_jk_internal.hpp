#pragma once

#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf::cuda_df {

/** Build mathematical J/K into plan-owned device buffers on plan.stream.
 * The same interfaces implement resident, host-backed and source-backed tiles.
 */
vibeqc_status build_coulomb(CudaDensityFittingJkPlan& plan, const double* density,
                            std::string& detail, bool raw_charge_ready = false);

vibeqc_status build_exchange(CudaDensityFittingJkPlan& plan, const double* density,
                             double* exchange, std::string& detail,
                             bool density_is_column_major = false, std::size_t system_begin = 0,
                             std::size_t system_end = SIZE_MAX);

/** One compatible spin factor on the plan stream: K = weight*(L C)(L C)^T.
 * Host snapshots supply row-major sqrt(occupation)*C and weight=1; device
 * SCF supplies column-major C and its canonical occupation. The caller owns
 * provenance validation. All T/panel/output scratch reuses existing tiles.
 */
vibeqc_status build_occupied_exchange(CudaDensityFittingJkPlan& plan, std::size_t system,
                                      const double* coefficients, std::size_t rank,
                                      bool column_major, double weight, double* exchange,
                                      std::string& detail);

/** Use the same exact raw traversal for a qualified RHF charge and occupied K.
 * The supplied factor must already be an admitted witness for density; this
 * function neither creates a final-projection lease nor changes J/K resources.
 * Wider-than-two-block schedules are accepted only for an explicit
 * profitability experiment selected by the caller.
 */
vibeqc_status build_shared_coulomb_occupied_exchange(CudaDensityFittingJkPlan& plan,
                                                     const double* density,
                                                     const double* coefficients, std::size_t rank,
                                                     double weight, bool allow_multiblock,
                                                     std::string& detail);

}  // namespace vibeqc::scf::cuda_df

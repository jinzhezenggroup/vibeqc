#pragma once

#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf::cuda_df {

/** Build mathematical J/K into plan-owned device buffers on plan.stream.
 * The same interfaces implement resident, host-backed and source-backed tiles.
 */
vibeqc_status build_coulomb(CudaDensityFittingJkPlan& plan, const double* density,
                            std::string& detail);

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

}  // namespace vibeqc::scf::cuda_df

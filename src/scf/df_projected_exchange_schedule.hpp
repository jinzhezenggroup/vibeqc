#pragma once

#include "generated_df_exchange_schedule.hpp"
#include "scf/df_streamed_k_policy.hpp"

namespace vibeqc::scf {

/** Bind the existing dense fallback to the compiler's source-work comparison.
 * Capacity counts doubles in EACH of four existing disjoint buffers. This
 * shape decision grants no orbital provenance or resident response lease.
 */
inline generated::ProjectedExchangeSchedule df_projected_exchange_schedule(
    std::size_t nbf, std::size_t naux, std::size_t rank, std::size_t capacity,
    bool triangular) noexcept {
  const auto dense = df_streamed_k_panel(nbf, naux, capacity);
  return generated::projected_exchange_schedule(nbf, naux, rank, capacity, dense.row_tiles,
                                                dense.output_tiles, triangular);
}

/** Keep automatic shared J/K on its qualified <=2-block domain while allowing
 * an explicit profitability experiment to exercise any compiler-valid
 * triangular projected schedule. This changes no capacity or source-work
 * admission by itself.
 */
inline bool df_shared_projected_exchange_schedule_admitted(
    const generated::ProjectedExchangeSchedule& schedule, bool explicit_multiblock) noexcept {
  return schedule.rows && schedule.blocks && (schedule.blocks <= 2 || explicit_multiblock);
}

}  // namespace vibeqc::scf

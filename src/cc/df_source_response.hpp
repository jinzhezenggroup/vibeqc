#pragma once

#include "cc/df_source.hpp"
#include "posthf/df_mo_response.hpp"

namespace generativeqc::cc {

struct DFSourceResponseDiagnostic {
  posthf::CudaDFMOSourceResponseDiagnostic transform;
  std::size_t numeric_capacity_bytes{}, retained_source_bytes{}, h2d_bytes{};
  std::size_t embedding_arena_bytes{}, metric_workspace_bytes{};
};

/** Pull physical factor cotangents back through their original molecular source.
 * Requires an opt-in state from build_df_source_cuda and its matching nonzero
 * source_identity in seeds. No replacement source, orbital frame or metric
 * factorization is constructed. Calls sharing this state are serialized.
 *
 * Consume receives provisional raw three-center cotangents in full, unit-weight
 * [mu,nu,P] coordinates: contract every pair once, with no triangular doubling.
 * Finish receives bar_C and the symmetric full Frobenius bar_metric, on the
 * owner's stream. Both callbacks only borrow their pointers until return and
 * must enqueue on that stream. Outputs become valid only on successful return;
 * callbacks must not recursively invoke response on the same state.
 *
 * The shared spectral rule retains discarded/retained subspace motion at fixed
 * rank and refuses unresolved cutoff crossings. This is the physical integral
 * source response, not orbital stationarity or a complete nuclear gradient.
 * Admission includes source ownership, seed vector capacities and all scratch.
 * caller_bytes covers callback destinations and other live numeric owners.
 */
DFSourceResponseDiagnostic pullback_df_source_cuda(std::shared_ptr<DFSourceState> source,
                                                   const DFFactorResponseResult& seeds,
                                                   const posthf::CudaDFSourceConsume& consume,
                                                   const posthf::CudaDFSourceFinish& finish,
                                                   std::size_t maximum_bytes,
                                                   std::size_t caller_bytes = 0);

}  // namespace generativeqc::cc

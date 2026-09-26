#pragma once

#include <algorithm>

#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_metric_kernels.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/df_streamed_k_policy.hpp"

namespace vibeqc::scf::cuda_df {

/** Transform a bounded raw panel with GEMM instead of repeating recurrences
 * for every output auxiliary direction. Both buffers borrow one existing plan
 * tile and must be disjoint. Raw input blocks are reduced in ascending order;
 * no allocation, host tensor, extra stream or synchronization is introduced.
 */
inline vibeqc_status generate_metric_panel(CudaDensityFittingJkPlan& plan, std::size_t system,
                                           std::size_t pair_begin, std::size_t pairs,
                                           std::size_t auxiliary_begin, std::size_t auxiliaries,
                                           double* output, double* scratch, std::string& detail) {
  const auto capacity = plan.row_tile * plan.nbf * plan.auxiliary_tile;
  if (!pairs || !auxiliaries || pairs > capacity / auxiliaries || output == scratch) {
    detail = "generated DF panel exceeds its borrowed tile capacity";
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
  const auto raw_tile = std::min(plan.naux, capacity / pairs);
  // These private panels feed only streamed K, which sums over its entire
  // whitened auxiliary axis. E=A Q diag(lambda^-1/2) and the symmetric
  // C=E Q^T therefore give exactly the same K. Keep E here: rotating back
  // would project every eigendirection again for each tiny output panel.
  // Streamed plans publish no resident C view and cannot lend their final
  // projection to response; resident/packed owners keep their symmetric C.
  const bool eigen_basis = plan.streamed && plan.metric_full_rank[system];
  // Only fuse when neither buffer can hold two auxiliary directions. Here
  // both routes evaluate each source integral once for this Q; fusion avoids
  // naux singleton launches without sacrificing any possible raw-panel reuse.
  // A short output tail alone must never select repeated fused recurrences.
  if (!eigen_basis && auxiliaries == 1 && raw_tile == 1) {
    runtime::cuda_trace::trace_counter("fused_metric_panel_productions", 1);
    // Logical (pair,Q,P) work, not an assertion about instruction count or
    // elapsed-time amplification. Capture counters describe construction only.
    runtime::cuda_trace::trace_counter("fused_source_auxiliary_evaluations",
                                       pairs * auxiliaries * plan.naux);
    return generate_cuda_density_fitting_transformed_tile(
        plan.integral_source, system, pair_begin, pairs, auxiliary_begin, auxiliaries, -1,
        plan.inverse_square_roots + system * plan.naux * plan.naux,
        reinterpret_cast<void*>(plan.stream), output, detail);
  }
  runtime::cuda_trace::trace_counter("raw_panel_source_auxiliary_evaluations", pairs * plan.naux);
  const double one = 1, zero = 0;
  runtime::cuda_trace::TraceRegion transform("transformed_three_center_generation", plan.stream);
  for (std::size_t begin = 0; begin < plan.naux; begin += raw_tile) {
    const auto count = std::min(raw_tile, plan.naux - begin);
    auto status = generate_cuda_density_fitting_raw_tile(
        plan.integral_source, system, pair_begin, pairs, begin, count, -1,
        reinterpret_cast<void*>(plan.stream), scratch, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    // Accumulate Q^T A before scaling weak eigendirections. Raw splitting
    // changes only the reduction order, never the number of source integrals.
    // The truncated-rank compatibility path retains its symmetric factor.
    const auto* factor = eigen_basis ? plan.metric_eigenvectors + system * plan.naux * plan.naux +
                                           auxiliary_begin * plan.naux + begin
                                     : plan.inverse_square_roots + system * plan.naux * plan.naux +
                                           begin * plan.naux + auxiliary_begin;
    const auto blas_status = runtime::cuda_trace::trace_call("tile_metric_gemm", plan.stream, [&] {
      return cublasDgemm(plan.blas, eigen_basis ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
                         static_cast<int>(auxiliaries), static_cast<int>(pairs),
                         static_cast<int>(count), &one, factor, static_cast<int>(plan.naux),
                         scratch, static_cast<int>(count), begin ? &one : &zero, output,
                         static_cast<int>(auxiliaries));
    });
    if (blas_status != CUBLAS_STATUS_SUCCESS)
      return blas_failure(blas_status, "transform bounded raw DF panel", detail);
    if (eigen_basis) runtime::cuda_trace::trace_counter("streamed_whitening_factor_gemms", 1);
  }
  if (eigen_basis) {
    launch_scale_metric_projection(plan.stream, auxiliaries, pairs,
                                   plan.metric_eigenvalues + system * plan.naux + auxiliary_begin,
                                   true, output);
    const auto error = cudaPeekAtLastError();
    if (error != cudaSuccess)
      return cuda_failure(error, "scale streamed DF eigenbasis panel", detail);
    runtime::cuda_trace::trace_counter("streamed_whitening_eigen_basis", 1);
    runtime::cuda_trace::trace_counter("streamed_whitening_factor_panels", 1);
    runtime::cuda_trace::trace_counter("streamed_whitening_factor_flops",
                                       2 * pairs * plan.naux * auxiliaries);
  }
  runtime::cuda_trace::trace_tile(system, pair_begin, pairs, auxiliary_begin, auxiliaries, -1,
                                  true);
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_df

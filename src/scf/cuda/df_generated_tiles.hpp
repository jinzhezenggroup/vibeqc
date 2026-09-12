#pragma once

#include <algorithm>

#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"

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
  // A skinny output panel cannot amortize one raw launch per metric block.
  // Keep the existing fused recurrence/transform for this tight-budget case;
  // it uses the same output capacity and avoids a cascade of tiny GEMMs.
  if (auxiliaries <= 4) {
    runtime::cuda_trace::trace_counter("fused_metric_panel_productions", 1);
    return generate_cuda_density_fitting_transformed_tile(
        plan.integral_source, system, pair_begin, pairs, auxiliary_begin, auxiliaries, -1,
        plan.inverse_square_roots + system * plan.naux * plan.naux,
        reinterpret_cast<void*>(plan.stream), output, detail);
  }
  const double one = 1, zero = 0;
  runtime::cuda_trace::TraceRegion transform("transformed_three_center_generation", plan.stream);
  for (std::size_t begin = 0; begin < plan.naux; begin += raw_tile) {
    const auto count = std::min(raw_tile, plan.naux - begin);
    auto status = generate_cuda_density_fitting_raw_tile(
        plan.integral_source, system, pair_begin, pairs, begin, count, -1,
        reinterpret_cast<void*>(plan.stream), scratch, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    // Raw tiles are [pair, raw auxiliary]; the retained symmetric metric
    // factor uses cuBLAS column-major storage. Produce [pair, output auxiliary].
    const auto blas_status = runtime::cuda_trace::trace_call("tile_metric_gemm", plan.stream, [&] {
      return cublasDgemm(plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(auxiliaries),
                         static_cast<int>(pairs), static_cast<int>(count), &one,
                         plan.inverse_square_roots + system * plan.naux * plan.naux +
                             begin * plan.naux + auxiliary_begin,
                         static_cast<int>(plan.naux), scratch, static_cast<int>(count),
                         begin ? &one : &zero, output, static_cast<int>(auxiliaries));
    });
    if (blas_status != CUBLAS_STATUS_SUCCESS)
      return blas_failure(blas_status, "transform bounded raw DF panel", detail);
  }
  runtime::cuda_trace::trace_tile(system, pair_begin, pairs, auxiliary_begin, auxiliaries, -1,
                                  true);
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_df

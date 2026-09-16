#include <algorithm>
#include <limits>
#include <new>

#include "runtime/cuda_component_trace.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda/df_generated_tiles.hpp"
#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_jk_kernels.hpp"
#include "scf/cuda/df_packed_values.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"

namespace vibeqc::scf::cuda_df {

vibeqc_status build_occupied_exchange(CudaDensityFittingJkPlan& plan, std::size_t system,
                                      const double* coefficients, std::size_t rank,
                                      bool column_major, double weight, double* exchange,
                                      std::string& detail) {
  plan.final_projection_token.reset();
  using namespace runtime::cuda_trace;
  TraceOperation trace(
      "ri_k_occupied", plan.stream,
      {1, plan.nbf, plan.naux, plan.integral_source != nullptr, plan.streamed, system});
  if (system >= plan.batch_size || rank > plan.nbf || (rank && !coefficients)) {
    detail = "invalid occupied DF exchange dimensions";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  auto error = cudaMemsetAsync(exchange + system * plan.matrix_elements, 0,
                               plan.matrix_elements * sizeof(double), plan.stream);
  if (error != cudaSuccess) return cuda_failure(error, "zero occupied DF exchange", detail);
  trace_counter("occupied_rank", rank);
  trace_counter("occupied_factor_bytes", plan.nbf * rank * sizeof(double));
  if (!rank) return VIBEQC_STATUS_SUCCESS;

  const bool packed = plan.value_storage.pairs == DfPairStorage::SymmetricLower;
  const bool complete_projection =
      packed ? rank <= plan.value_storage.rank_capacity : plan.auxiliary_tile == plan.naux;
  if (plan.resident_exchange_enabled && !plan.streamed && plan.row_tile == plan.nbf &&
      complete_projection &&
      plan.naux * rank <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    // B[mu,nu,Q] is a batch of column-major (Q,nu) matrices. Project the
    // contracted AO index directly, retaining U[mu,i,Q] in one existing
    // buffer. Flattening (i,Q) then sums every auxiliary contribution inside
    // GEMM: K = w U U^T. This also works for nonsymmetric diagnostic B.
    const auto n = static_cast<int>(plan.nbf), a = static_cast<int>(plan.naux);
    const auto r = static_cast<int>(rank), ar = a * r;
    const double one = 1, zero = 0;
    const auto* b = plan.three_center + system * plan.stored_tensor_elements_per_system;
    auto* u = plan.auxiliary_tile_values;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    if (packed) {
      runtime::cuda_trace::TraceRegion projection("ri_k_occupied_packed_projection", plan.stream);
      launch_project_packed_df(plan.stream, plan.nbf, plan.naux, rank, column_major, b,
                               coefficients, u);
      error = cudaPeekAtLastError();
      if (error != cudaSuccess) return cuda_failure(error, "packed occupied DF projection", detail);
      trace_counter("value_packed_pairs", plan.stored_pair_count);
      trace_counter("packed_projection_launches", 1);
    } else {
      status = trace_call("ri_k_occupied_projection_gemm", plan.stream, [&] {
        return cublasDgemmStridedBatched(plan.blas, CUBLAS_OP_N,
                                         column_major ? CUBLAS_OP_N : CUBLAS_OP_T, a, r, n, &one, b,
                                         a, plan.naux * plan.nbf, coefficients,
                                         column_major ? n : r, 0, &zero, u, a, plan.naux * rank, n);
      });
      if (status != CUBLAS_STATUS_SUCCESS)
        return blas_failure(status, "resident occupied DF projection", detail);
    }
    bool triangular = false;
    auto* output = exchange + system * plan.matrix_elements;
    status = trace_call("ri_k_occupied_exchange_gemm", plan.stream, [&] {
      // K is a Gram matrix even when the three-center fixture is not AO
      // symmetric. Reduce its product domain before BLAS, then mirror the
      // result. Providers without SYRK retain the exact full GEMM route.
      const auto product = [&](auto handle) {
        if constexpr (requires {
                        cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, ar, &weight, u,
                                    ar, &zero, output, n);
                      }) {
          if (plan.triangular_exchange) {
            triangular = true;
            return cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, ar, &weight, u, ar,
                               &zero, output, n);
          }
        }
        return cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, ar, &weight, u, ar, u, ar, &zero,
                           output, n);
      };
      return product(plan.blas);
    });
    if (status != CUBLAS_STATUS_SUCCESS)
      return blas_failure(status, "resident occupied DF K product", detail);
    if (triangular) {
      runtime::cuda_trace::TraceRegion mirror("ri_k_occupied_mirror", plan.stream);
      launch_mirror_exchange_triangle(blocks_for(plan.matrix_elements), kThreads, plan.stream,
                                      plan.nbf, output);
      error = cudaPeekAtLastError();
      if (error != cudaSuccess) return cuda_failure(error, "mirror occupied DF K", detail);
    }
    trace_counter("occupied_projection_products", plan.nbf);
    trace_counter("occupied_projection_flops", 2 * plan.naux * plan.nbf * plan.nbf * rank);
    trace_counter("occupied_exchange_products", 1);
    trace_counter("occupied_exchange_flops",
                  plan.naux * rank * plan.nbf * (triangular ? plan.nbf + 1 : 2 * plan.nbf));
    trace_counter("occupied_exchange_triangular", triangular);
    trace_counter("occupied_projection_m", plan.naux);
    trace_counter("occupied_projection_n", rank);
    trace_counter("occupied_projection_k", plan.nbf);
    trace_counter("occupied_projection_batch", plan.nbf);
    trace_counter("occupied_exchange_m", plan.nbf);
    trace_counter("occupied_exchange_n", plan.nbf);
    trace_counter("occupied_exchange_k", plan.naux * rank);
    trace_counter("occupied_intermediate_bytes", plan.nbf * plan.naux * rank * sizeof(double));
    return status == CUBLAS_STATUS_SUCCESS
               ? VIBEQC_STATUS_SUCCESS
               : blas_failure(status, "resident occupied DF K product", detail);
  }

  // Rank never exceeds nbf, so both T panels fit the dense plan's buffers.
  // Source-backed execution uses exactly the dense raw-work policy; host
  // compatibility panels retain their existing layout and upload lifetime.
  const auto capacity = plan.row_tile * plan.nbf * plan.auxiliary_tile;
  const bool full_pairs = !plan.streamed || capacity >= plan.matrix_elements;
  auto rows = full_pairs ? plan.nbf : plan.row_tile;
  auto auxiliary_tile = full_pairs && plan.streamed
                            ? std::min(plan.auxiliary_tile, capacity / plan.matrix_elements)
                            : plan.auxiliary_tile;
  if (plan.streamed && plan.integral_source) {
    const auto panels = df_streamed_k_panel(plan.nbf, plan.naux, capacity);
    rows = panels.rows;
    auxiliary_tile = panels.output_auxiliaries;
    runtime::df_progress::number("executed_raw_auxiliary_tile", panels.raw_auxiliaries);
    runtime::df_progress::number("raw_tensor_passes_per_k", panels.row_tiles * panels.output_tiles);
  }
  runtime::df_progress::number("planner_ao_pair_tile", plan.ao_pair_tile);
  runtime::df_progress::number("planner_auxiliary_tile", plan.auxiliary_tile);
  runtime::df_progress::number("executed_ao_rows", rows);
  runtime::df_progress::number("executed_auxiliary_tile", auxiliary_tile);
  const auto row_tiles = (plan.nbf + rows - 1) / rows;
  std::vector<double> host_tile;
  if (plan.streamed && !plan.integral_source) {
    try {
      host_tile.resize(rows * plan.nbf * auxiliary_tile);
    } catch (const std::bad_alloc&) {
      detail = "host allocation for occupied DF streamed panel failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  // Produce [auxiliary][row][AO] into exchange_intermediate. As a cuBLAS
  // column-major matrix each slice is L_row^T (nbf x row_count).
  const auto panel = [&](std::size_t begin, std::size_t count, std::size_t qbegin,
                         std::size_t qcount) -> vibeqc_status {
    const auto pairs = count * plan.nbf;
    if (packed) {
      runtime::cuda_trace::TraceRegion unpack("ri_k_packed_unpack", plan.stream);
      launch_unpack_df_values(plan.stream, plan.nbf, plan.naux, begin, count, qbegin, qcount, true,
                              plan.three_center + system * plan.stored_tensor_elements_per_system,
                              plan.exchange_intermediate);
      trace_counter("packed_unpack_elements", pairs * qcount);
    } else if (!plan.streamed) {
      launch_gather_auxiliary_tile_kernel(blocks_for(pairs * qcount), kThreads, 0, plan.stream,
                                          plan.matrix_elements, plan.naux, system, qbegin, qcount,
                                          plan.three_center, plan.exchange_intermediate);
    } else if (plan.integral_source) {
      auto status =
          generate_metric_panel(plan, system, begin * plan.nbf, pairs, qbegin, qcount,
                                plan.auxiliary_tile_values, plan.exchange_tile_output, detail);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
      launch_transpose_streamed_df_tile_kernel(blocks_for(pairs * qcount), kThreads, 0, plan.stream,
                                               pairs, qcount, plan.auxiliary_tile_values,
                                               plan.exchange_intermediate);
    } else {
      const auto* raw =
          plan.streamed_raw_three_center.data() + system * plan.tensor_elements_per_system;
      const auto* inverse =
          plan.streamed_inverse_square_roots.data() + system * plan.naux * plan.naux;
      for (std::size_t q = 0; q < qcount; ++q)
        for (std::size_t pair = 0; pair < pairs; ++pair) {
          double value = 0;
          for (std::size_t a = 0; a < plan.naux; ++a)
            value += raw[((begin * plan.nbf) + pair) * plan.naux + a] *
                     inverse[(qbegin + q) * plan.naux + a];
          host_tile[q * pairs + pair] = value;
        }
      error = cudaMemcpyAsync(plan.exchange_intermediate, host_tile.data(),
                              pairs * qcount * sizeof(double), cudaMemcpyHostToDevice, plan.stream);
      // This compatibility provider owns pageable staging. Drain its upload
      // before the next panel overwrites the same host buffer; source plans
      // never take this branch and remain graph-capture safe.
      if (error == cudaSuccess) error = cudaStreamSynchronize(plan.stream);
      if (error != cudaSuccess) return cuda_failure(error, "upload occupied DF panel", detail);
      trace_counter("host_to_device_bytes", pairs * qcount * sizeof(double));
    }
    error = cudaPeekAtLastError();
    return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                                : cuda_failure(error, "prepare occupied DF panel", detail);
  };
  const double one = 1, zero = 0;
  const auto transform = [&](std::size_t count, std::size_t qcount,
                             double* output) -> vibeqc_status {
    // U = C^T L_row^T has [rank,row_count] column-major layout. Host B is
    // already C^T in that convention and includes sqrt(occupation).
    auto status = trace_call("ri_k_occupied_gemm", plan.stream, [&] {
      return cublasDgemmStridedBatched(
          plan.blas, column_major ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(rank),
          static_cast<int>(count), static_cast<int>(plan.nbf), &one, coefficients,
          static_cast<int>(column_major ? plan.nbf : rank), 0, plan.exchange_intermediate,
          static_cast<int>(plan.nbf), count * plan.nbf, &zero, output, static_cast<int>(rank),
          rank * count, static_cast<int>(qcount));
    });
    trace_counter("occupied_projection_products", qcount);
    return status == CUBLAS_STATUS_SUCCESS ? VIBEQC_STATUS_SUCCESS
                                           : blas_failure(status, "occupied DF projection", detail);
  };
  for (std::size_t qbegin = 0; qbegin < plan.naux; qbegin += auxiliary_tile) {
    const auto qcount = std::min(auxiliary_tile, plan.naux - qbegin);
    for (std::size_t rbegin = 0; rbegin < plan.nbf; rbegin += rows) {
      const auto rcount = std::min(rows, plan.nbf - rbegin);
      auto status = panel(rbegin, rcount, qbegin, qcount);
      if (status == VIBEQC_STATUS_SUCCESS)
        status = transform(rcount, qcount, plan.exchange_contributions);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
      for (std::size_t tile = 0; tile < row_tiles; ++tile) {
        const auto cbegin = ((rbegin / rows + tile) % row_tiles) * rows;
        const auto ccount = std::min(rows, plan.nbf - cbegin);
        const double* column = plan.exchange_contributions;
        if (tile) {
          status = panel(cbegin, ccount, qbegin, qcount);
          if (status == VIBEQC_STATUS_SUCCESS)
            status = transform(ccount, qcount, plan.auxiliary_tile_values);
          if (status != VIBEQC_STATUS_SUCCESS) return status;
          column = plan.auxiliary_tile_values;
        } else {
          trace_counter("occupied_panel_cache_hits", 1);
        }
        // Consume the diagonal's T twice before replacing any column panel.
        // Output reuses L's buffer after its projection has completed on this
        // stream. No full three-center temporary or rank-dependent allocation.
        const auto stride = rcount * ccount;
        auto blas = trace_call("ri_k_occupied_gemm", plan.stream, [&] {
          return cublasDgemmStridedBatched(
              plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rcount),
              static_cast<int>(ccount), static_cast<int>(rank), &weight,
              plan.exchange_contributions, static_cast<int>(rank), rank * rcount, column,
              static_cast<int>(rank), rank * ccount, &zero, plan.exchange_intermediate,
              static_cast<int>(rcount), stride, static_cast<int>(qcount));
        });
        if (blas != CUBLAS_STATUS_SUCCESS)
          return blas_failure(blas, "occupied DF K product", detail);
        trace_counter("occupied_exchange_products", qcount);
        launch_reduce_exchange_row_tile_kernel(blocks_for(stride), kThreads, 0, plan.stream,
                                               plan.nbf, rbegin, rcount, cbegin, ccount, qcount,
                                               system, plan.exchange_intermediate, exchange);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure(error, "reduce occupied DF K", detail);
      }
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_df

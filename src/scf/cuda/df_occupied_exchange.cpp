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
#include "scf/df_projected_exchange_schedule.hpp"

namespace vibeqc::scf::cuda_df {
namespace {

/** Project raw source slices before whitening when all auxiliary directions of
 * a bounded AO-row block fit. The four existing, disjoint plan buffers hold
 * left/right occupied projections, raw AO input and one eigenprojection.
 * This changes source work from auxiliary-output-panel regeneration to AO-row
 * regeneration; it allocates nothing and remains safe inside SCF capture.
 */
vibeqc_status build_streamed_projected_exchange(CudaDensityFittingJkPlan& plan, std::size_t system,
                                                const double* coefficients, std::size_t rank,
                                                bool column_major, double weight, std::size_t rows,
                                                double* exchange, std::string& detail,
                                                const double* charge_density = nullptr) {
  using namespace runtime::cuda_trace;
  const auto n = plan.nbf, a = plan.naux, capacity = plan.panel_capacity;
  const auto ar = a * rank;
  const double one = 1, zero = 0;
  double* projections[] = {plan.auxiliary_tile_values, plan.exchange_contributions};
  auto* raw = plan.exchange_intermediate;
  auto* transformed = plan.exchange_tile_output;
  auto* output = exchange + system * plan.matrix_elements;
  const auto project = [&](std::size_t begin, std::size_t count, double* target,
                           bool charge) -> vibeqc_status {
    const auto raw_tile = std::min(a, capacity / (count * n));
    trace_counter("raw_panel_source_auxiliary_evaluations", count * n * a);
    trace_counter("streamed_occupied_raw_generation_rows", count);
    for (std::size_t p = 0; p < a; p += raw_tile) {
      const auto q = std::min(raw_tile, a - p);
      const auto status = generate_cuda_density_fitting_raw_tile(
          plan.integral_source, system, begin * n, count * n, p, q, -1,
          reinterpret_cast<void*>(plan.stream), raw, detail);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
      if (charge) {
        launch_accumulate_streamed_auxiliary_density_kernel(
            blocks_for(q), kThreads, 0, plan.stream, count * n, q, raw, charge_density + begin * n,
            plan.auxiliary_density + p);
        const auto error = cudaPeekAtLastError();
        if (error != cudaSuccess)
          return cuda_failure(error, "shared raw DF charge contraction", detail);
      }
      // Raw [mu,nu,P] has contiguous P. Each mu projects its contracted nu
      // directly into U[mu,i,P]; lda=a preserves prior raw-auxiliary blocks.
      const auto blas = trace_call("streamed_occupied_raw_projection", plan.stream, [&] {
        return cublasDgemmStridedBatched(
            plan.blas, CUBLAS_OP_N, column_major ? CUBLAS_OP_N : CUBLAS_OP_T, static_cast<int>(q),
            static_cast<int>(rank), static_cast<int>(n), &one, raw, static_cast<int>(q), q * n,
            coefficients, static_cast<int>(column_major ? n : rank), 0, &zero, target + p,
            static_cast<int>(a), ar, static_cast<int>(count));
      });
      if (blas != CUBLAS_STATUS_SUCCESS)
        return blas_failure(blas, "project streamed raw DF factors", detail);
      trace_counter("occupied_projection_products", count);
      trace_counter("occupied_projection_flops", 2 * q * count * n * rank);
    }
    if (charge) trace_counter("shared_coulomb_charge_rows", count);
    // Keep eigendirections separate until after division. Orthogonal rotation
    // back to symmetric whitening cancels in the complete K Gram. These
    // private factors are never published as a final symmetric-C projection.
    const auto blas = trace_call("streamed_occupied_metric_projection", plan.stream, [&] {
      return cublasDgemm(plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(a),
                         static_cast<int>(count * rank), static_cast<int>(a), &one,
                         plan.metric_eigenvectors + system * a * a, static_cast<int>(a), target,
                         static_cast<int>(a), &zero, transformed, static_cast<int>(a));
    });
    if (blas != CUBLAS_STATUS_SUCCESS)
      return blas_failure(blas, "whiten streamed occupied DF factors", detail);
    launch_scale_metric_projection(plan.stream, a, count * rank,
                                   plan.metric_eigenvalues + system * a, true, transformed);
    auto error = cudaPeekAtLastError();
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(target, transformed, count * ar * sizeof(double),
                              cudaMemcpyDeviceToDevice, plan.stream);
    if (error != cudaSuccess)
      return cuda_failure(error, "retain streamed occupied DF factors", detail);
    trace_counter("streamed_whitening_factor_gemms", 1);
    trace_counter("streamed_whitening_factor_flops", 2 * a * a * count * rank);
    trace_counter("streamed_occupied_projection_copy_bytes", count * ar * sizeof(double));
    return VIBEQC_STATUS_SUCCESS;
  };
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  // The compiler owns visit order and the two-slot lifetime. Raw input and
  // metric scratch remain disjoint from both retained projections; native
  // callbacks bind the existing buffers and enqueue all work on plan.stream.
  const auto project_visit = [&](std::size_t begin, std::size_t count, std::size_t slot,
                                 bool charge) {
    status = project(begin, count, projections[slot], charge);
    return status == VIBEQC_STATUS_SUCCESS;
  };
  const auto contract_visit = [&](std::size_t r, std::size_t nr, std::size_t c, std::size_t nc,
                                  std::size_t left, std::size_t right, bool retained) {
    if (retained) trace_counter("occupied_panel_cache_hits", 1);
    // Columns of each (a*rank,rows) panel are output AO rows. Every matrix
    // block is produced exactly once, including partial row/column tails.
    const auto blas = trace_call("streamed_occupied_exchange_gemm", plan.stream, [&] {
      return cublasDgemm(plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(nr),
                         static_cast<int>(nc), static_cast<int>(ar), &weight, projections[left],
                         static_cast<int>(ar), projections[right], static_cast<int>(ar), &zero,
                         output + r + c * n, static_cast<int>(n));
    });
    if (blas != CUBLAS_STATUS_SUCCESS) {
      status = blas_failure(blas, "contract streamed occupied DF factors", detail);
      return false;
    }
    trace_counter("occupied_exchange_products", 1);
    trace_counter("occupied_exchange_flops", 2 * nr * nc * ar);
    return true;
  };
  const auto completed = charge_density
                             ? generated::visit_shared_projected_exchange(
                                   n, rows, plan.triangular_exchange, project_visit, contract_visit)
                             : generated::visit_projected_exchange(
                                   n, rows, plan.triangular_exchange,
                                   [&](std::size_t begin, std::size_t count, std::size_t slot) {
                                     return project_visit(begin, count, slot, false);
                                   },
                                   contract_visit);
  if (!completed) {
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    detail = "invalid compiler projected exchange traversal";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (plan.triangular_exchange) {
    launch_mirror_exchange_triangle(blocks_for(plan.matrix_elements), kThreads, plan.stream, n,
                                    output);
    const auto error = cudaPeekAtLastError();
    if (error != cudaSuccess) return cuda_failure(error, "mirror streamed occupied DF K", detail);
  }
  trace_counter("streamed_occupied_source_first", 1);
  trace_counter("streamed_occupied_row_blocks", (n + rows - 1) / rows);
  trace_counter("streamed_occupied_retained_projection_capacity_bytes",
                2 * rows * ar * sizeof(double));
  trace_counter("streamed_occupied_metric_projection_capacity_bytes", rows * ar * sizeof(double));
  trace_counter("streamed_occupied_scratch_capacity_bytes", 4 * capacity * sizeof(double));
  trace_counter("occupied_exchange_triangular", plan.triangular_exchange);
  return VIBEQC_STATUS_SUCCESS;
}
}  // namespace

vibeqc_status build_shared_coulomb_occupied_exchange(CudaDensityFittingJkPlan& plan,
                                                     const double* density,
                                                     const double* coefficients, std::size_t rank,
                                                     double weight, std::string& detail) {
  plan.final_projection_token.reset();
  if (plan.batch_size != 1 || !plan.streamed || !plan.integral_source ||
      !plan.triangular_exchange || plan.metric_full_rank.size() != 1 || !plan.metric_full_rank[0] ||
      !density || !coefficients || !rank ||
      rank > static_cast<std::size_t>(std::numeric_limits<int>::max()) / plan.naux ||
      plan.row_tile * plan.nbf * plan.auxiliary_tile < plan.naux) {
    detail = "shared DF J/K source requires admitted streamed RHF factor-first traversal";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto schedule =
      df_projected_exchange_schedule(plan.nbf, plan.naux, rank, plan.panel_capacity, true);
  if (!schedule.rows || schedule.blocks > 2) {
    detail = "shared DF J/K source lacks bounded projection capacity";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  auto error = cudaMemsetAsync(plan.auxiliary_density, 0, plan.naux * sizeof(double), plan.stream);
  if (error == cudaSuccess)
    error =
        cudaMemsetAsync(plan.alpha_exchange, 0, plan.matrix_elements * sizeof(double), plan.stream);
  if (error != cudaSuccess) return cuda_failure(error, "zero shared DF J/K outputs", detail);
  runtime::cuda_trace::TraceOperation trace("ri_jk_shared", plan.stream,
                                            {1, plan.nbf, plan.naux, true, true});
  const auto status =
      build_streamed_projected_exchange(plan, 0, coefficients, rank, true, weight, schedule.rows,
                                        plan.alpha_exchange, detail, density);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  runtime::cuda_trace::trace_counter("shared_raw_source_values",
                                     schedule.generated_rows * plan.nbf * plan.naux);
  return build_coulomb(plan, density, detail, true);
}

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
  trace_counter("occupied_rank", rank);
  trace_counter("occupied_factor_bytes", plan.nbf * rank * sizeof(double));
  if (!rank) {
    const auto error = cudaMemsetAsync(exchange + system * plan.matrix_elements, 0,
                                       plan.matrix_elements * sizeof(double), plan.stream);
    return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                                : cuda_failure(error, "zero occupied DF exchange", detail);
  }
  cudaError_t error = cudaSuccess;

  if (plan.integral_source && plan.streamed && plan.metric_full_rank[system] &&
      rank <= static_cast<std::size_t>(std::numeric_limits<int>::max()) / plan.naux) {
    // Allocation, automatic SCF qualification and execution share this compiler
    // schedule. Balanced row blocks avoid repeatedly generating an oversized
    // prefix when the last triangular block is short.
    const auto schedule = df_projected_exchange_schedule(
        plan.nbf, plan.naux, rank, plan.panel_capacity, plan.triangular_exchange);
    if (schedule.rows)
      return build_streamed_projected_exchange(plan, system, coefficients, rank, column_major,
                                               weight, schedule.rows, exchange, detail);
  }

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

  // Only the tiled fallback accumulates partial auxiliary products into K.
  // Fast projected and resident Gram routes use beta=0 for every output block,
  // so pre-zeroing the whole matrix is pure device traffic on each Fock build.
  error = cudaMemsetAsync(exchange + system * plan.matrix_elements, 0,
                          plan.matrix_elements * sizeof(double), plan.stream);
  if (error != cudaSuccess) return cuda_failure(error, "zero occupied DF exchange", detail);

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

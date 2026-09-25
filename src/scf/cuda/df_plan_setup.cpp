#include "scf/cuda/df_plan_setup.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cuda_component_trace.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda/df_jk_kernels.hpp"
#include "scf/cuda/df_metric_kernels.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_setup_internal.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"
#include "scf/df_exchange_policy.hpp"
#include "scf/df_projected_exchange_schedule.hpp"

namespace vibeqc::scf::cuda_df {
namespace {
/** Apply Q diag(lambda^-1/2) Q^T to one public raw panel in factor order.
 * Forming the explicit inverse root first loses weak-direction cancellation
 * when it subsequently contracts raw A. The two products below preserve the
 * same symmetric whitening convention, cutoff and eigensystem. Setup borrows
 * the already charged exchange panel before any SCF consumer exists; raw
 * values and their immutable caches remain intact.
 */
vibeqc_status whiten_factor_panel(CudaDensityFittingJkPlan& plan, std::size_t system,
                                  std::size_t pairs, const double* raw, double* output,
                                  const double* eigenvectors, const double* scaled_eigenvectors,
                                  std::string& detail) {
  const auto a = static_cast<int>(plan.naux), rows = static_cast<int>(pairs);
  const auto offset = system * plan.naux * plan.naux;
  const double one = 1, zero = 0;
  auto status =
      runtime::cuda_trace::trace_call("resident_metric_eigen_projection", plan.stream, [&] {
        return cublasDgemm(plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, a, rows, a, &one,
                           eigenvectors + offset, a, raw, a, &zero, plan.exchange_intermediate, a);
      });
  if (status != CUBLAS_STATUS_SUCCESS)
    return blas_failure(status, "project raw CUDA DF metric factors", detail);
  status = runtime::cuda_trace::trace_call("resident_metric_scaled_rotation", plan.stream, [&] {
    return cublasDgemm(plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, a, rows, a, &one,
                       scaled_eigenvectors + offset, a, plan.exchange_intermediate, a, &zero,
                       output, a);
  });
  if (status != CUBLAS_STATUS_SUCCESS)
    return blas_failure(status, "rotate scaled CUDA DF metric factors", detail);
  runtime::cuda_trace::trace_counter("resident_whitening_factor_panels", 1);
  runtime::cuda_trace::trace_counter("resident_whitening_factor_gemms", 2);
  runtime::cuda_trace::trace_counter("resident_whitening_factor_flops",
                                     4 * pairs * plan.naux * plan.naux);
  runtime::cuda_trace::trace_counter("resident_whitening_projection_elements", pairs * plan.naux);
  return VIBEQC_STATUS_SUCCESS;
}

/** Materialize a fixed-geometry source once, reusing bounded resident K staging.
 * Every raw (pair,P) is generated once and feeds ALL Q directly into retained
 * B. The explicit single-factor variant transforms each lower-pair panel
 * immediately, freeing raw scratch before the next panel. Force response must
 * regenerate raw values from the immutable physical source in that variant.
 * The caller handles failure after the plan's stream is drained.
 */
vibeqc_status materialize_generated_tensor(CudaDensityFittingJkPlan& plan, const double* inverse,
                                           const double* eigenvectors,
                                           const double* scaled_eigenvectors, std::string& detail) {
  runtime::cuda_trace::TraceOperation trace("resident_three_center_materialization", plan.stream,
                                            {plan.batch_size, plan.nbf, plan.naux, true, false});
  const double one = 1.0, zero = 0.0;
  if (df_packed_pairs(plan.value_storage.pairs)) {
    // The retained raw owner accepts whole lower rows independently of the
    // bounded scratch. Only single-owner transformation needs pair panels.
    std::size_t generated_panels = 0;
    for (std::size_t system = 0; system < plan.batch_size; ++system) {
      auto* raw = plan.packed_raw
                      ? plan.packed_raw + system * plan.stored_tensor_elements_per_system
                      : nullptr;
      for (std::size_t mu = 0; mu < plan.nbf; ++mu) {
        if (raw) {
          const auto status = generate_cuda_density_fitting_raw_tile(
              plan.integral_source, system, mu * plan.nbf, mu + 1, 0, plan.naux, -1,
              reinterpret_cast<void*>(plan.stream), raw + mu * (mu + 1) / 2 * plan.naux,
              detail);
          if (status != VIBEQC_STATUS_SUCCESS) return status;
          ++generated_panels;
          continue;
        }
        const auto pair_capacity = std::min(plan.projection_capacity, plan.panel_capacity) /
                                   plan.naux;
        if (!pair_capacity) return VIBEQC_STATUS_OUT_OF_MEMORY;
        for (std::size_t nu = 0; nu <= mu; nu += pair_capacity) {
          const auto count = std::min(pair_capacity, mu + 1 - nu);
          auto* panel = plan.auxiliary_tile_values;
          const auto status = generate_cuda_density_fitting_raw_tile(
              plan.integral_source, system, mu * plan.nbf + nu, count, 0, plan.naux, -1,
              reinterpret_cast<void*>(plan.stream), panel, detail);
          if (status != VIBEQC_STATUS_SUCCESS) return status;
          ++generated_panels;
          auto* output = plan.three_center +
                         (system * plan.stored_pair_count + mu * (mu + 1) / 2 + nu) * plan.naux;
          const auto transform = plan.metric_full_rank[system]
                                     ? whiten_factor_panel(plan, system, count, panel, output,
                                                           eigenvectors, scaled_eigenvectors,
                                                           detail)
                                     : VIBEQC_STATUS_SUCCESS;
          if (transform != VIBEQC_STATUS_SUCCESS) return transform;
          if (!plan.metric_full_rank[system]) {
            const auto blas_status = runtime::cuda_trace::trace_call(
                "resident_metric_transform", plan.stream, [&] {
                  return cublasDgemm(plan.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     static_cast<int>(plan.naux), static_cast<int>(count),
                                     static_cast<int>(plan.naux), &one,
                                     inverse + system * plan.naux * plan.naux,
                                     static_cast<int>(plan.naux), panel,
                                     static_cast<int>(plan.naux), &zero,
                                     output, static_cast<int>(plan.naux));
                });
            if (blas_status != CUBLAS_STATUS_SUCCESS)
              return blas_failure(blas_status, "whiten single packed DF values", detail);
          }
        }
      }
      if (!raw) continue;
      if (plan.metric_full_rank[system] && plan.panel_capacity >= plan.naux) {
        const auto tile = plan.panel_capacity / plan.naux;
        for (std::size_t pair = 0; pair < plan.stored_pair_count; pair += tile) {
          const auto count = std::min(tile, plan.stored_pair_count - pair);
          const auto status = whiten_factor_panel(
              plan, system, count, raw + pair * plan.naux,
              plan.three_center + system * plan.stored_tensor_elements_per_system +
                  pair * plan.naux,
              eigenvectors, scaled_eigenvectors, detail);
          if (status != VIBEQC_STATUS_SUCCESS) return status;
        }
        continue;
      }
      const auto status =
          runtime::cuda_trace::trace_call("resident_metric_transform", plan.stream, [&] {
            return cublasDgemm(plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(plan.naux),
                               static_cast<int>(plan.stored_pair_count),
                               static_cast<int>(plan.naux), &one,
                               inverse + system * plan.naux * plan.naux,
                               static_cast<int>(plan.naux), raw, static_cast<int>(plan.naux), &zero,
                               plan.three_center + system * plan.stored_tensor_elements_per_system,
                               static_cast<int>(plan.naux));
          });
      if (status != CUBLAS_STATUS_SUCCESS)
        return blas_failure(status, "whiten packed CUDA DF values", detail);
    }
    plan.resident_raw_valid = plan.packed_raw != nullptr;
    runtime::cuda_trace::trace_counter("value_packed_pairs", plan.stored_pair_count);
    runtime::cuda_trace::trace_counter("packed_raw_generation_calls", generated_panels);
    runtime::cuda_trace::trace_counter(
        "resident_raw_bytes",
        plan.packed_raw ? plan.batch_size * plan.stored_tensor_elements_per_system * sizeof(double)
                        : 0);
    runtime::cuda_trace::trace_counter(
        "resident_transformed_bytes",
        plan.batch_size * plan.stored_tensor_elements_per_system * sizeof(double));
    return VIBEQC_STATUS_SUCCESS;
  }
  const auto capacity = plan.row_tile * plan.nbf * plan.auxiliary_tile;
  const auto pair_tile =
      std::min(plan.matrix_elements, std::max<std::size_t>(1, capacity / plan.naux));
  const bool retain_raw =
      plan.resident_exchange_enabled && plan.batch_size == 1 && plan.row_tile == plan.nbf &&
      plan.auxiliary_tile == plan.naux &&
      plan.nbf * plan.naux <= static_cast<std::size_t>(std::numeric_limits<int>::max());
  // Under retain_raw, capacity is exactly matrix_elements*naux, so both
  // whitening branches generate one complete pair-major tensor (system=0,
  // pair=0, begin=0). The gather converts that layout to A[Q,mu,nu]; bounded
  // panels must never use this full-tensor stride. Preserve discarded metric
  // directions as well: whitening B cannot recover them for the derivative.
  for (std::size_t system = 0; system < plan.batch_size; ++system) {
    for (std::size_t pair = 0; pair < plan.matrix_elements; pair += pair_tile) {
      const auto pairs = std::min(pair_tile, plan.matrix_elements - pair);
      if (plan.metric_full_rank[system] && capacity >= plan.naux) {
        // The existing pair cap fits all auxiliary directions in each of two
        // disjoint panels. Generate every raw pair once, then use both GEMMs;
        // the extra projection never amplifies integral-source work.
        auto status = generate_cuda_density_fitting_raw_tile(
            plan.integral_source, system, pair, pairs, 0, plan.naux, -1,
            reinterpret_cast<void*>(plan.stream), plan.auxiliary_tile_values, detail);
        if (status != VIBEQC_STATUS_SUCCESS) return status;
        if (retain_raw) {
          launch_gather_auxiliary_tile_kernel(blocks_for(pairs * plan.naux), kThreads, 0,
                                              plan.stream, plan.matrix_elements, plan.naux, system,
                                              0, plan.naux, plan.auxiliary_tile_values,
                                              plan.exchange_contributions);
          const auto cuda_error = cudaGetLastError();
          if (cuda_error != cudaSuccess)
            return cuda_failure(cuda_error, "retain generated raw DF tensor", detail);
        }
        status = whiten_factor_panel(
            plan, system, pairs, plan.auxiliary_tile_values,
            plan.three_center + system * plan.tensor_elements_per_system + pair * plan.naux,
            eigenvectors, scaled_eigenvectors, detail);
        if (status != VIBEQC_STATUS_SUCCESS) return status;
        runtime::cuda_trace::trace_tile(system, pair, pairs, 0, plan.naux, -1, true);
        continue;
      }
      const auto raw_tile = std::min(plan.naux, capacity / pairs);
      for (std::size_t begin = 0; begin < plan.naux; begin += raw_tile) {
        const auto count = std::min(raw_tile, plan.naux - begin);
        const auto status = generate_cuda_density_fitting_raw_tile(
            plan.integral_source, system, pair, pairs, begin, count, -1,
            reinterpret_cast<void*>(plan.stream), plan.auxiliary_tile_values, detail);
        if (status != VIBEQC_STATUS_SUCCESS) return status;
        if (retain_raw) {
          launch_gather_auxiliary_tile_kernel(blocks_for(pairs * plan.naux), kThreads, 0,
                                              plan.stream, plan.matrix_elements, plan.naux, system,
                                              0, plan.naux, plan.auxiliary_tile_values,
                                              plan.exchange_contributions);
          const auto cuda_error = cudaGetLastError();
          if (cuda_error != cudaSuccess)
            return cuda_failure(cuda_error, "retain generated raw DF tensor", detail);
        }
        const auto blas_status =
            runtime::cuda_trace::trace_call("resident_metric_transform", plan.stream, [&] {
              return cublasDgemm(
                  plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(plan.naux),
                  static_cast<int>(pairs), static_cast<int>(count), &one,
                  inverse + system * plan.naux * plan.naux + begin * plan.naux,
                  static_cast<int>(plan.naux), plan.auxiliary_tile_values, static_cast<int>(count),
                  begin ? &one : &zero,
                  plan.three_center + system * plan.tensor_elements_per_system + pair * plan.naux,
                  static_cast<int>(plan.naux));
            });
        if (blas_status != CUBLAS_STATUS_SUCCESS)
          return blas_failure(blas_status, "materialize generated CUDA DF tensor", detail);
      }
      runtime::cuda_trace::trace_tile(system, pair, pairs, 0, plan.naux, -1, true);
    }
  }
  if (retain_raw) {
    plan.resident_raw_valid = true;
    runtime::cuda_trace::trace_counter("resident_raw_bytes",
                                       plan.tensor_elements_per_system * sizeof(double));
  }
  runtime::cuda_trace::trace_counter(
      "resident_transformed_bytes",
      plan.batch_size * plan.tensor_elements_per_system * sizeof(double));
  return VIBEQC_STATUS_SUCCESS;
}
}  // namespace

// Keep setup as one transaction: validate, allocate, factor, account, then
// publish. Error exits and final stream drain preserve the original lifetime.

vibeqc_status create_cuda_density_fitting_jk_plan_tiled_impl(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile, std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan, std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail, CudaDensityFittingIntegralSource* integral_source,
    bool retain_three_center, DfValueStorageOptions storage, std::size_t automatic_rhf_rank) {
  runtime::df_progress::Scope preparation("df_plan_setup");
  runtime::df_progress::number("planner_ao_pair_tile", ao_pair_tile);
  runtime::df_progress::number("planner_auxiliary_tile", auxiliary_tile);
  // `integral_source` is transferred into this routine by the source-backed
  // wrapper.  Dispose of it on every pre-plan failure as well as failures
  // after `candidate` has taken ownership; this makes the transfer atomic
  // from the caller's perspective and prevents a double free in callers that
  // unconditionally clean up their local handle.
  const auto fail_before_plan = [&](vibeqc_status status) {
    destroy_cuda_density_fitting_integral_source(integral_source);
    return status;
  };
  detail.clear();
  diagnostics.clear();
  if (plan == nullptr) return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  if (integral_source != nullptr && !cuda_density_fitting_integral_source_matches(
                                        integral_source, device_id, batch_size, nbf, naux)) {
    detail = "CUDA DF source dimensions or device do not match the plan";
    return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  }
  *plan = nullptr;
  const bool packed = df_packed_pairs(storage.pairs);
  if ((storage.pairs != DfPairStorage::Dense && !packed) || storage.rank_capacity > nbf ||
      (!packed && storage.rank_capacity) ||
      (packed && (!integral_source || !retain_three_center))) {
    detail =
        "packed DF values require an explicit retained physical source and valid rank capacity";
    return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  }
  if (device_id < 0 || batch_size == 0 || nbf == 0 || naux == 0 || !(relative_threshold > 0.0) ||
      !(relative_threshold < 1.0) ||
      batch_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      batch_size > 65535 || nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      naux > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    detail = "CUDA DF plan dimensions or metric threshold are invalid";
    return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  }

  std::size_t matrix_elements = 0;
  std::size_t metric_elements = 0;
  std::size_t tensor_elements_per_system = 0;
  std::size_t all_matrix_elements = 0;
  std::size_t all_metric_elements = 0;
  std::size_t all_tensor_elements = 0;
  if (!checked_multiply(nbf, nbf, matrix_elements) ||
      !checked_multiply(naux, naux, metric_elements) ||
      !checked_multiply(matrix_elements, naux, tensor_elements_per_system) ||
      !checked_multiply(batch_size, matrix_elements, all_matrix_elements) ||
      !checked_multiply(batch_size, metric_elements, all_metric_elements) ||
      !checked_multiply(batch_size, tensor_elements_per_system, all_tensor_elements) ||
      matrix_elements > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      metrics.size() != all_metric_elements || !finite_values(metrics) ||
      ((integral_source == nullptr) &&
       (three_center.size() != all_tensor_elements || !finite_values(three_center)))) {
    detail = "CUDA DF plan buffers have invalid dimensions or values";
    return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  }
  auxiliary_tile = auxiliary_tile == 0 ? std::min<std::size_t>(naux, 32) : auxiliary_tile;
  if (auxiliary_tile > naux ||
      auxiliary_tile > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    detail = "CUDA DF auxiliary tile is invalid";
    return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  }
  ao_pair_tile = ao_pair_tile == 0 ? matrix_elements : ao_pair_tile;
  if (ao_pair_tile > matrix_elements || ao_pair_tile == 0) {
    detail = "CUDA DF AO-pair tile is invalid";
    return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  }
  if (retain_three_center && (!integral_source || ao_pair_tile != matrix_elements)) {
    detail = "retained generated DF storage requires a source and complete AO rows";
    return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  }

  std::size_t matrix_bytes = 0;
  std::size_t metric_bytes = 0;
  std::size_t tensor_bytes = 0;
  std::size_t auxiliary_bytes = 0;
  std::size_t tile_elements = 0;
  std::size_t tile_bytes = 0;
  // Explicit generated retention decouples B from contraction auxiliary width.
  // Compatibility partial dimensions still request regeneration/host staging.
  const bool streamed =
      !retain_three_center && (auxiliary_tile < naux || ao_pair_tile < matrix_elements);
  const std::size_t staged_row_tile =
      streamed ? std::min<std::size_t>(nbf, std::max<std::size_t>(1, ao_pair_tile / nbf)) : nbf;
  const bool complete_resident_layout =
      !streamed && ao_pair_tile == matrix_elements && auxiliary_tile == naux;
  std::size_t staged_pair_capacity = 0;
  std::size_t auxiliary_vector_elements = 0;
  std::size_t auxiliary_vector_bytes = 0;
  std::size_t solver_info_bytes = 0;
  if (!checked_bytes(all_matrix_elements, matrix_bytes) ||
      !checked_bytes(all_metric_elements, metric_bytes) ||
      !checked_bytes(all_tensor_elements, tensor_bytes) ||
      !checked_multiply(batch_size, naux, tile_elements) ||
      !checked_bytes(tile_elements, auxiliary_bytes) ||
      !checked_multiply(batch_size, naux, auxiliary_vector_elements) ||
      !checked_bytes(auxiliary_vector_elements, auxiliary_vector_bytes) ||
      !checked_multiply(batch_size, sizeof(int), solver_info_bytes) ||
      !checked_multiply(staged_row_tile, nbf, staged_pair_capacity) ||
      !checked_multiply(auxiliary_tile, staged_pair_capacity, tile_elements) ||
      !checked_bytes(tile_elements, tile_bytes)) {
    detail = "CUDA DF plan storage overflows size_t";
    return fail_before_plan(VIBEQC_STATUS_OUT_OF_MEMORY);
  }

  DfPackedValueCapacity packed_capacity;
  std::size_t projection_bytes = tile_bytes;
  if (packed) {
    try {
      packed_capacity =
          df_packed_value_capacity(batch_size, nbf, naux, storage.rank_capacity, auxiliary_tile,
                                   df_retains_packed_raw(storage.pairs));
    } catch (const std::exception& error) {
      detail = error.what();
      return fail_before_plan(VIBEQC_STATUS_OUT_OF_MEMORY);
    }
    tensor_bytes = packed_capacity.factor_bytes;
    if (!checked_bytes(packed_capacity.projection_elements, projection_bytes)) {
      detail = "packed DF projection byte capacity overflows size_t";
      return fail_before_plan(VIBEQC_STATUS_OUT_OF_MEMORY);
    }
  }

  // Generic tensor/source callers cannot infer a reference from dimensions.
  // Even an authorized RHF hint cannot reserve for an ineligible layout. A
  // Resident generated plans need the complete raw/scratch lease. Streamed
  // plans may instead reserve a private source-first value projection; metric
  // rank and exact density provenance are checked later, before execution.
  const bool streamed_projection =
      streamed && integral_source &&
      df_projected_exchange_schedule(nbf, naux, automatic_rhf_rank, tile_elements,
                                     df_triangular_exchange_requested())
          .rows;
  if ((streamed && !streamed_projection) ||
      (integral_source && !packed && !complete_resident_layout && !streamed_projection) ||
      (packed && automatic_rhf_rank > storage.rank_capacity))
    automatic_rhf_rank = 0;
  const bool occupied_scf_reserved =
      df_occupied_exchange_requested(nbf, naux, batch_size, automatic_rhf_rank);
  const char* diis_policy = std::getenv("VIBEQC_DF_DIIS_DOTS");
  if (diis_policy && std::strcmp(diis_policy, "auto") != 0 &&
      std::strcmp(diis_policy, "serial") != 0) {
    detail = "VIBEQC_DF_DIIS_DOTS must be auto or serial";
    return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  }
  const char* resident_policy = std::getenv("VIBEQC_DF_RESIDENT_EXCHANGE");
  if (resident_policy && std::strcmp(resident_policy, "auto") != 0 &&
      std::strcmp(resident_policy, "full") != 0 && std::strcmp(resident_policy, "flat") != 0 &&
      std::strcmp(resident_policy, "legacy") != 0) {
    detail = "VIBEQC_DF_RESIDENT_EXCHANGE must be auto, full, flat or legacy";
    return fail_before_plan(VIBEQC_STATUS_INVALID_ARGUMENT);
  }
  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    return fail_before_plan(cuda_failure(cuda_error, "select CUDA DF device", detail));
  }
  auto* candidate = new (std::nothrow) CudaDensityFittingJkPlan{};
  if (candidate == nullptr) return fail_before_plan(VIBEQC_STATUS_OUT_OF_MEMORY);
  candidate->device_id = device_id;
  candidate->occupied_scf_reserved = occupied_scf_reserved;
  candidate->automatic_rhf_rank = automatic_rhf_rank;
  candidate->resident_exchange_enabled = df_resident_exchange_requested();
  candidate->triangular_exchange = df_triangular_exchange_requested();
  candidate->flat_dense_exchange = df_flat_dense_exchange_requested();
  candidate->cooperative_diis = df_cooperative_diis_requested();
  candidate->metric_relative_threshold = relative_threshold;
  candidate->batch_size = batch_size;
  candidate->nbf = nbf;
  candidate->naux = naux;
  candidate->matrix_elements = matrix_elements;
  candidate->tensor_elements_per_system = tensor_elements_per_system;
  candidate->value_storage = storage;
  candidate->stored_pair_count = packed ? packed_capacity.pairs : matrix_elements;
  candidate->stored_tensor_elements_per_system =
      packed ? packed_capacity.tensor_per_system : tensor_elements_per_system;
  candidate->projection_capacity = packed ? packed_capacity.projection_elements : tile_elements;
  candidate->panel_capacity = tile_elements;
  candidate->auxiliary_tile = auxiliary_tile;
  candidate->ao_pair_tile = ao_pair_tile;
  candidate->row_tile = staged_row_tile;
  candidate->streamed = streamed;
  candidate->integral_source = integral_source;
  if (candidate->streamed && integral_source == nullptr) {
    try {
      candidate->streamed_raw_three_center = three_center;
    } catch (const std::bad_alloc&) {
      return fail_plan(candidate, VIBEQC_STATUS_OUT_OF_MEMORY);
    }
  }

  cuda_error = cudaStreamCreateWithFlags(&candidate->stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate, cuda_failure(cuda_error, "create CUDA DF stream", detail));
  }
  cublasStatus_t blas_status = cublasCreate(&candidate->blas);
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasSetStream(candidate->blas, candidate->stream);
  }
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasSetPointerMode(candidate->blas, CUBLAS_POINTER_MODE_HOST);
  }
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return fail_plan(candidate, blas_failure(blas_status, "initialize CUDA DF cuBLAS", detail));
  }
  cusolverStatus_t solver_status = cusolverDnCreate(&candidate->solver);
  if (solver_status == CUSOLVER_STATUS_SUCCESS) {
    solver_status = cusolverDnSetStream(candidate->solver, candidate->stream);
  }
  if (solver_status == CUSOLVER_STATUS_SUCCESS) {
    solver_status = cusolverDnCreateParams(&candidate->solver_parameters);
  }
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    return fail_plan(candidate,
                     solver_failure(solver_status, "initialize CUDA DF cuSOLVER", detail));
  }

  auto allocate_permanent = [&](double** pointer, std::size_t bytes, const char* description) {
    return allocate_device(reinterpret_cast<void**>(pointer), bytes, description, detail);
  };
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  if (!candidate->streamed) {
    status = allocate_permanent(&candidate->three_center, tensor_bytes,
                                "allocate transformed CUDA DF tensor");
  }
  if (status == VIBEQC_STATUS_SUCCESS && df_retains_packed_raw(storage.pairs)) {
    status = allocate_permanent(&candidate->packed_raw, tensor_bytes,
                                "allocate immutable packed raw CUDA DF tensor");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->primary_density, matrix_bytes,
                                "allocate primary CUDA DF density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->secondary_density, matrix_bytes,
                                "allocate secondary CUDA DF density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->total_density, matrix_bytes,
                                "allocate total CUDA DF density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->auxiliary_density, auxiliary_bytes,
                                "allocate CUDA DF auxiliary density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status =
        allocate_permanent(&candidate->coulomb, matrix_bytes, "allocate CUDA DF Coulomb matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->alpha_exchange, matrix_bytes,
                                "allocate CUDA DF alpha exchange matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->beta_exchange, matrix_bytes,
                                "allocate CUDA DF beta exchange matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->auxiliary_tile_values, projection_bytes,
                                "allocate CUDA DF auxiliary tile");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->exchange_intermediate, tile_bytes,
                                "allocate CUDA DF exchange intermediate");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->exchange_contributions, tile_bytes,
                                "allocate CUDA DF exchange contributions");
  }
  if (status == VIBEQC_STATUS_SUCCESS && candidate->streamed) {
    status = allocate_permanent(&candidate->exchange_tile_output, tile_bytes,
                                "allocate CUDA DF exchange tile output");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->exchange_density_column_major, matrix_bytes,
                                "allocate CUDA DF exchange density transpose");
  }
  if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);

  SetupBuffers setup;
  auto allocate_setup = [&](void** pointer, std::size_t bytes, const char* description) {
    return allocate_device(pointer, bytes, description, detail);
  };
  status = allocate_setup(reinterpret_cast<void**>(&setup.metrics), metric_bytes,
                          "allocate CUDA DF metric eigensystem");
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.eigenvalues), auxiliary_vector_bytes,
                            "allocate CUDA DF metric eigenvalues");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.scales), auxiliary_vector_bytes,
                            "allocate CUDA DF metric eigenvalue scales");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.scaled_eigenvectors), metric_bytes,
                            "allocate scaled CUDA DF metric eigenvectors");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.inverse_square_roots), metric_bytes,
                            "allocate CUDA DF metric inverse square roots");
  }
  if (status == VIBEQC_STATUS_SUCCESS && !candidate->streamed && !candidate->integral_source) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.raw_three_center), tensor_bytes,
                            "allocate raw CUDA DF three-center tensor");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.solver_info), solver_info_bytes,
                            "allocate CUDA DF solver status");
  }
  if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);

  cuda_error = cudaMemcpyAsync(setup.metrics, metrics.data(), metric_bytes, cudaMemcpyHostToDevice,
                               candidate->stream);
  if (cuda_error == cudaSuccess && !candidate->streamed && !candidate->integral_source) {
    cuda_error = cudaMemcpyAsync(setup.raw_three_center, three_center.data(), tensor_bytes,
                                 cudaMemcpyHostToDevice, candidate->stream);
  }
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate, cuda_failure(cuda_error, "upload CUDA DF setup tensors", detail));
  }
  const dim3 symmetric_threads(16, 16, 1);
  const dim3 symmetric_blocks(
      static_cast<unsigned>((naux + symmetric_threads.x - 1) / symmetric_threads.x),
      static_cast<unsigned>((naux + symmetric_threads.y - 1) / symmetric_threads.y),
      static_cast<unsigned>(batch_size));
  launch_symmetrize_metrics_kernel(symmetric_blocks, symmetric_threads, 0, candidate->stream, naux,
                                   setup.metrics);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate, cuda_failure(cuda_error, "symmetrize CUDA DF metrics", detail));
  }

  runtime::df_progress::Scope metric_progress("metric_factorization");
  runtime::df_progress::label("provider", "cusolverDnXsyevd");
  std::size_t solver_device_workspace_bytes = 0;
  std::size_t solver_host_workspace_bytes = 0;
  solver_status = cusolverDnXsyevd_bufferSize(
      candidate->solver, candidate->solver_parameters, CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER, static_cast<std::int64_t>(naux), CUDA_R_64F, setup.metrics,
      static_cast<std::int64_t>(naux), CUDA_R_64F, setup.eigenvalues, CUDA_R_64F,
      &solver_device_workspace_bytes, &solver_host_workspace_bytes);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    return fail_plan(candidate,
                     solver_failure(solver_status, "size CUDA DF metric eigensolver", detail));
  }
  if (solver_device_workspace_bytes > df_eigen_workspace_allowance(naux)) {
    detail = "CUDA DF metric eigensolver query exceeds its planned workspace";
    return fail_plan(candidate, VIBEQC_STATUS_OUT_OF_MEMORY);
  }
  if (solver_device_workspace_bytes != 0) {
    status = allocate_setup(&setup.solver_workspace, solver_device_workspace_bytes,
                            "allocate CUDA DF metric solver workspace");
    if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);
  }
  try {
    setup.solver_host_workspace.resize(solver_host_workspace_bytes);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF metric solver workspace failed";
    return fail_plan(candidate, VIBEQC_STATUS_OUT_OF_MEMORY);
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    solver_status = cusolverDnXsyevd(
        candidate->solver, candidate->solver_parameters, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, static_cast<std::int64_t>(naux), CUDA_R_64F,
        setup.metrics + system * metric_elements, static_cast<std::int64_t>(naux), CUDA_R_64F,
        setup.eigenvalues + system * naux, CUDA_R_64F, setup.solver_workspace,
        solver_device_workspace_bytes,
        setup.solver_host_workspace.empty() ? nullptr : setup.solver_host_workspace.data(),
        solver_host_workspace_bytes, setup.solver_info + system);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      return fail_plan(candidate,
                       solver_failure(solver_status, "diagonalize CUDA DF metric", detail));
    }
  }

  std::vector<double> eigenvalues;
  std::vector<double> scales;
  std::vector<int> solver_info;
  try {
    eigenvalues.resize(batch_size * naux);
    scales.assign(batch_size * naux, 0.0);
    solver_info.resize(batch_size);
    diagnostics.resize(batch_size);
    candidate->metric_response_valid.assign(batch_size, 1);
    candidate->metric_full_rank.assign(batch_size, 0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF metric diagnostics failed";
    return fail_plan(candidate, VIBEQC_STATUS_OUT_OF_MEMORY);
  }
  cuda_error =
      cudaMemcpyAsync(eigenvalues.data(), setup.eigenvalues, eigenvalues.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, candidate->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error =
        cudaMemcpyAsync(solver_info.data(), setup.solver_info, solver_info.size() * sizeof(int),
                        cudaMemcpyDeviceToHost, candidate->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaStreamSynchronize(candidate->stream);
  }
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate,
                     cuda_failure(cuda_error, "read CUDA DF metric eigensystem", detail));
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    if (solver_info[system] != 0) {
      detail = "CUDA DF metric eigensolver did not converge for system " + std::to_string(system);
      return fail_plan(candidate, VIBEQC_STATUS_CUDA_ERROR);
    }
    const std::size_t offset = system * naux;
    const double largest = eigenvalues[offset + naux - 1];
    if (!(largest > 0.0) || !std::isfinite(largest)) {
      detail =
          "CUDA DF metric has no finite positive eigenspace for system " + std::to_string(system);
      return fail_plan(candidate, VIBEQC_STATUS_INVALID_ARGUMENT);
    }
    auto& diagnostic = diagnostics[system];
    diagnostic.system_index = system;
    diagnostic.solver_device_workspace_bytes = solver_device_workspace_bytes;
    diagnostic.solver_host_workspace_bytes = solver_host_workspace_bytes;
    diagnostic.absolute_threshold = relative_threshold * largest;
    double smallest_retained = largest;
    for (std::size_t item = 0; item < naux; ++item) {
      const double value = eigenvalues[offset + item];
      if (!std::isfinite(value)) {
        detail = "CUDA DF metric eigensolver returned a non-finite eigenvalue";
        return fail_plan(candidate, VIBEQC_STATUS_CUDA_ERROR);
      }
      if (std::abs(value - diagnostic.absolute_threshold) <=
          128 * std::numeric_limits<double>::epsilon() * largest)
        candidate->metric_response_valid[system] = 0;
      if (value <= diagnostic.absolute_threshold) continue;
      scales[offset + item] = 1.0 / std::sqrt(value);
      ++diagnostic.effective_rank;
      smallest_retained = std::min(smallest_retained, value);
    }
    if (diagnostic.effective_rank == 0) {
      detail = "CUDA DF metric threshold removed every auxiliary direction";
      return fail_plan(candidate, VIBEQC_STATUS_INVALID_ARGUMENT);
    }
    candidate->metric_full_rank[system] = diagnostic.effective_rank == naux;
    diagnostic.condition_number = largest / smallest_retained;
  }

  cuda_error = cudaMemcpyAsync(setup.scales, scales.data(), scales.size() * sizeof(double),
                               cudaMemcpyHostToDevice, candidate->stream);
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate, cuda_failure(cuda_error, "upload CUDA DF metric scales", detail));
  }
  launch_scale_eigenvectors_kernel(blocks_for(all_metric_elements), kThreads, 0, candidate->stream,
                                   all_metric_elements, naux, setup.metrics, setup.scales,
                                   setup.scaled_eigenvectors);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate,
                     cuda_failure(cuda_error, "scale CUDA DF metric eigenvectors", detail));
  }

  const double one = 1.0;
  const double zero = 0.0;
  blas_status = cublasDgemmStridedBatched(
      candidate->blas, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(naux), static_cast<int>(naux),
      static_cast<int>(naux), &one, setup.scaled_eigenvectors, static_cast<int>(naux),
      static_cast<long long>(metric_elements), setup.metrics, static_cast<int>(naux),
      static_cast<long long>(metric_elements), &zero, setup.inverse_square_roots,
      static_cast<int>(naux), static_cast<long long>(metric_elements),
      static_cast<int>(batch_size));
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return fail_plan(
        candidate,
        blas_failure(blas_status, "construct CUDA DF metric inverse square root", detail));
  }
  if (metric_progress.enabled()) {
    // The journal needs a completed boundary before raw generation starts.
    // The ordinary, unprofiled path retains its existing stream ordering.
    cuda_error = cudaStreamSynchronize(candidate->stream);
    if (cuda_error != cudaSuccess)
      return fail_plan(candidate, cuda_failure(cuda_error, "trace metric completion", detail));
  }
  metric_progress.finish("stream_complete");
  if (!candidate->streamed && candidate->integral_source) {
    status = materialize_generated_tensor(*candidate, setup.inverse_square_roots, setup.metrics,
                                          setup.scaled_eigenvectors, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);
  } else if (candidate->streamed) {
    if (!candidate->integral_source) {
      try {
        candidate->streamed_inverse_square_roots.resize(batch_size * metric_elements);
      } catch (const std::bad_alloc&) {
        detail = "host allocation for streamed CUDA DF metric inverse failed";
        return fail_plan(candidate, VIBEQC_STATUS_OUT_OF_MEMORY);
      }
      cuda_error = cudaMemcpyAsync(candidate->streamed_inverse_square_roots.data(),
                                   setup.inverse_square_roots,
                                   candidate->streamed_inverse_square_roots.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, candidate->stream);
      if (cuda_error != cudaSuccess) {
        return fail_plan(candidate,
                         cuda_failure(cuda_error, "read streamed CUDA DF metric inverse", detail));
      }
    }
  } else {
    // End the trace (and its pending stream reads) before failure destroys
    // the plan's stream and allocations.
    const auto materialization_status = [&]() -> vibeqc_status {
      runtime::cuda_trace::TraceOperation trace("resident_three_center_materialization",
                                                candidate->stream,
                                                {batch_size, nbf, naux, false, false});
      for (std::size_t system = 0; system < batch_size; ++system) {
        if (candidate->metric_full_rank[system]) {
          status =
              whiten_factor_panel(*candidate, system, matrix_elements,
                                  setup.raw_three_center + system * tensor_elements_per_system,
                                  candidate->three_center + system * tensor_elements_per_system,
                                  setup.metrics, setup.scaled_eigenvectors, detail);
          if (status != VIBEQC_STATUS_SUCCESS) return status;
        } else {
          // Preserve the original truncated-space preparation and its response.
          blas_status = cublasDgemm(
              candidate->blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(naux),
              static_cast<int>(matrix_elements), static_cast<int>(naux), &one,
              setup.inverse_square_roots + system * metric_elements, static_cast<int>(naux),
              setup.raw_three_center + system * tensor_elements_per_system, static_cast<int>(naux),
              &zero, candidate->three_center + system * tensor_elements_per_system,
              static_cast<int>(naux));
          if (blas_status != CUBLAS_STATUS_SUCCESS)
            return blas_failure(blas_status, "transform CUDA DF three-center tensor", detail);
        }
      }
      if (candidate->resident_exchange_enabled && batch_size == 1 && candidate->row_tile == nbf &&
          auxiliary_tile == naux &&
          nbf * naux <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        // The resident exchange contractions no longer write this third full
        // scratch tensor. Preserve the original, untruncated raw values here
        // while setup still owns them; B alone cannot reconstruct discarded
        // metric directions needed by the exact Frechet derivative.
        launch_gather_auxiliary_tile_kernel(
            blocks_for(tensor_elements_per_system), kThreads, 0, candidate->stream, matrix_elements,
            naux, 0, 0, naux, setup.raw_three_center, candidate->exchange_contributions);
        cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess)
          return cuda_failure(cuda_error, "retain raw DF tensor", detail);
        candidate->resident_raw_valid = true;
      }
      return VIBEQC_STATUS_SUCCESS;
    }();
    if (materialization_status != VIBEQC_STATUS_SUCCESS)
      return fail_plan(candidate, materialization_status);
  }
  // Source force replay borrows these original device factors. Transfer them
  // only after the final stream drain below, so setup still owns error cleanup.
  // Publish a conservative allocation accounting record.  Setup buffers are
  // still live at this point, so the peak includes both permanent contraction
  // storage and metric-factorization workspace; host-side solver workspace is
  // intentionally excluded from the device-byte figures.
  // Device SCF state is allocated lazily on the first solve. Reserve a
  // conservative upper bound here so diagnostics remain valid before and
  // after that allocation (RHF/UHF share this plan type).
  const long double persistent_scf_estimate =
      (candidate->occupied_scf_reserved ? 22.0L : 20.0L) * static_cast<long double>(matrix_bytes) +
      static_cast<long double>(batch_size) *
          (16.0L * sizeof(double) + 2.0L * sizeof(std::int32_t) + 2.0L * sizeof(std::uint8_t) +
           sizeof(std::uint32_t) +
           (candidate->occupied_scf_reserved ? 2 * sizeof(std::uint32_t) + sizeof(int) : 0)) +
      df_scf_workspace_allowance(nbf, batch_size) + matrix_bytes +  // graph bookkeeping
      df_eigen_device_reservation(nbf) + df_final_snapshot_device_reservation(nbf, batch_size) +
      df_final_validation_device_reservation(nbf);
  const std::size_t persistent_scf_bytes =
      persistent_scf_estimate >= static_cast<long double>(std::numeric_limits<std::size_t>::max())
          ? std::numeric_limits<std::size_t>::max()
          : static_cast<std::size_t>(persistent_scf_estimate);
  const std::size_t persistent_device_bytes =
      6 * matrix_bytes + auxiliary_bytes + (candidate->streamed ? 0 : tensor_bytes) +
      projection_bytes + 2 * tile_bytes + (candidate->packed_raw ? tensor_bytes : 0) + matrix_bytes +
      (candidate->streamed ? tile_bytes : 0) + persistent_scf_bytes + 2 * metric_bytes +
      auxiliary_vector_bytes +
      (candidate->integral_source != nullptr
           ? cuda_density_fitting_integral_source_device_bytes(candidate->integral_source)
           : 0);
  const std::size_t setup_device_bytes =
      metric_bytes + auxiliary_vector_bytes +
      (candidate->streamed || candidate->integral_source ? 0 : tensor_bytes) +
      solver_device_workspace_bytes + solver_info_bytes;
  // This record covers the value/SCF plan and its setup. Generated force
  // staging is owned by the separately budgeted bridge and reported through
  // DfGradientResources and the whole-HF allocation ledger.
  const long double peak_estimate =
      static_cast<long double>(persistent_device_bytes) + setup_device_bytes;
  const std::size_t peak_device_bytes =
      peak_estimate >= static_cast<long double>(std::numeric_limits<std::size_t>::max())
          ? std::numeric_limits<std::size_t>::max()
          : static_cast<std::size_t>(peak_estimate);
  const long double host_resident_estimate =
      static_cast<long double>(sizeof(*candidate)) + df_eigen_workspace_allowance(nbf) +
      64.0L * batch_size +  // lazy final-frame occupation/eligibility metadata
      vector_capacity_bytes(candidate->metric_response_valid) +
      vector_capacity_bytes(candidate->metric_full_rank) +

      (candidate->integral_source
           ? static_cast<long double>(
                 cuda_density_fitting_integral_source_host_bytes(candidate->integral_source))
       : candidate->streamed
           ? static_cast<long double>(vector_capacity_bytes(candidate->streamed_raw_three_center)) +
                 vector_capacity_bytes(candidate->streamed_inverse_square_roots)
           : 0.0L);
  const std::size_t host_resident_bytes = saturating_bytes(host_resident_estimate);

  // Setup vectors coexist with source metadata. Force staging is
  // accounted by the owning bridge, independently of this value-plan record.
  const long double setup_host_estimate =
      static_cast<long double>(vector_capacity_bytes(metrics)) +
      static_cast<long double>(vector_capacity_bytes(three_center)) +
      static_cast<long double>(vector_capacity_bytes(eigenvalues)) +
      static_cast<long double>(vector_capacity_bytes(scales)) +
      static_cast<long double>(vector_capacity_bytes(solver_info)) +
      static_cast<long double>(solver_host_workspace_bytes);
  const std::size_t host_peak_bytes = saturating_bytes(
      host_resident_estimate + setup_host_estimate +
      (candidate->integral_source != nullptr
           ? static_cast<long double>(
                 cuda_density_fitting_integral_source_host_peak_bytes(candidate->integral_source))
           : 0.0L));
  for (auto& diagnostic : diagnostics) {
    diagnostic.device_resident_bytes = persistent_device_bytes;
    diagnostic.peak_device_bytes = peak_device_bytes;
    diagnostic.host_resident_bytes = host_resident_bytes;
    diagnostic.peak_host_bytes = host_peak_bytes;
    diagnostic.auxiliary_tile = auxiliary_tile;
    diagnostic.streamed = candidate->streamed;
    diagnostic.pair_storage = storage.pairs;
    diagnostic.stored_factor_bytes = candidate->streamed ? 0 : tensor_bytes;
    diagnostic.raw_factor_bytes = candidate->packed_raw           ? tensor_bytes
                                  : candidate->resident_raw_valid ? tile_bytes
                                                                  : 0;
    diagnostic.contraction_scratch_bytes =
        projection_bytes + 2 * tile_bytes + (candidate->streamed ? tile_bytes : 0);
    diagnostic.occupied_rank_capacity = storage.rank_capacity;
  }
  cuda_error = cudaStreamSynchronize(candidate->stream);
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate,
                     cuda_failure(cuda_error, "finish CUDA DF plan preparation", detail));
  }

  // Keep the metric provider's handles for ordinary AO setup/final solves.
  // SetupBuffers still drops the numeric metric scratch; release() owns the
  // handle/parameter teardown after the retained ordinary workspace is freed.
  // Every value provider reuses this same spectral map for forces. Transfer
  // the completed setup buffers instead of copying or refactoring the metric;
  // setup still owns all failure exits before this transaction commits.
  candidate->inverse_square_roots = std::exchange(setup.inverse_square_roots, nullptr);
  candidate->metric_eigenvectors = std::exchange(setup.metrics, nullptr);
  candidate->metric_eigenvalues = std::exchange(setup.eigenvalues, nullptr);
  *plan = candidate;
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_df

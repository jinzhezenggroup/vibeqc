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
#include "scf/cuda/df_metric_kernels.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_setup_internal.hpp"

namespace vibeqc::scf::cuda_df {
namespace {
/** Materialize a fixed-geometry source once, reusing the resident K staging.
 * Raw public-AO values are generated once per system and transformed by GEMM.
 * The scratch already belongs to the resolved full-tile plan; neither a host
 * full tensor nor an additional raw device allocation is introduced. The
 * caller handles failure only after this trace has drained its borrowed stream.
 */
vibeqc_status materialize_generated_tensor(CudaDensityFittingJkPlan& plan, const double* inverse,
                                           std::string& detail) {
  runtime::cuda_trace::TraceOperation trace("resident_three_center_materialization", plan.stream,
                                            {plan.batch_size, plan.nbf, plan.naux, true, false});
  const double one = 1.0, zero = 0.0;
  for (std::size_t system = 0; system < plan.batch_size; ++system) {
    const auto status = generate_cuda_density_fitting_raw_tile(
        plan.integral_source, system, 0, plan.matrix_elements, 0, plan.naux, -1,
        reinterpret_cast<void*>(plan.stream), plan.auxiliary_tile_values, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    runtime::cuda_trace::trace_tile(system, 0, plan.matrix_elements, 0, plan.naux, -1, true);
    const auto blas_status =
        runtime::cuda_trace::trace_call("resident_metric_transform", plan.stream, [&] {
          return cublasDgemm(plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(plan.naux),
                             static_cast<int>(plan.matrix_elements), static_cast<int>(plan.naux),
                             &one, inverse + system * plan.naux * plan.naux,
                             static_cast<int>(plan.naux), plan.auxiliary_tile_values,
                             static_cast<int>(plan.naux), &zero,
                             plan.three_center + system * plan.tensor_elements_per_system,
                             static_cast<int>(plan.naux));
        });
    if (blas_status != CUBLAS_STATUS_SUCCESS)
      return blas_failure(blas_status, "materialize generated CUDA DF tensor", detail);
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
    std::string& detail, CudaDensityFittingIntegralSource* integral_source) {
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

  std::size_t matrix_bytes = 0;
  std::size_t metric_bytes = 0;
  std::size_t tensor_bytes = 0;
  std::size_t auxiliary_bytes = 0;
  std::size_t tile_elements = 0;
  std::size_t tile_bytes = 0;
  // Stream whenever either planner dimension is smaller than the full
  // transformed tensor.  This matters for small auxiliary bases where all
  // Q directions fit but the AO-pair budget still requires row staging.
  const bool streamed = auxiliary_tile < naux || ao_pair_tile < matrix_elements;
  const std::size_t staged_row_tile =
      streamed ? std::min<std::size_t>(nbf, std::max<std::size_t>(1, ao_pair_tile / nbf)) : nbf;
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

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    return fail_before_plan(cuda_failure(cuda_error, "select CUDA DF device", detail));
  }
  auto* candidate = new (std::nothrow) CudaDensityFittingJkPlan{};
  if (candidate == nullptr) return fail_before_plan(VIBEQC_STATUS_OUT_OF_MEMORY);
  candidate->device_id = device_id;
  candidate->metric_relative_threshold = relative_threshold;
  candidate->batch_size = batch_size;
  candidate->nbf = nbf;
  candidate->naux = naux;
  candidate->matrix_elements = matrix_elements;
  candidate->tensor_elements_per_system = tensor_elements_per_system;
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
    status = allocate_permanent(&candidate->auxiliary_tile_values, tile_bytes,
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
  if (status == VIBEQC_STATUS_SUCCESS && candidate->integral_source != nullptr) {
    status = allocate_permanent(&candidate->inverse_square_roots, metric_bytes,
                                "allocate source-backed CUDA DF metric inverse");
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
    if (candidate->integral_source) candidate->metric_response_valid.assign(batch_size, 1);
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
      if (candidate->integral_source && std::abs(value - diagnostic.absolute_threshold) <=
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
  if (candidate->integral_source != nullptr) {
    cuda_error = cudaMemcpyAsync(candidate->inverse_square_roots, setup.inverse_square_roots,
                                 metric_bytes, cudaMemcpyDeviceToDevice, candidate->stream);
    if (cuda_error != cudaSuccess) {
      return fail_plan(
          candidate,
          cuda_failure(cuda_error, "retain source-backed CUDA DF metric inverse", detail));
    }
  }
  if (!candidate->streamed && candidate->integral_source) {
    status = materialize_generated_tensor(*candidate, setup.inverse_square_roots, detail);
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
    blas_status = cublasDgemmStridedBatched(
        candidate->blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(naux),
        static_cast<int>(matrix_elements), static_cast<int>(naux), &one, setup.inverse_square_roots,
        static_cast<int>(naux), static_cast<long long>(metric_elements), setup.raw_three_center,
        static_cast<int>(naux), static_cast<long long>(tensor_elements_per_system), &zero,
        candidate->three_center, static_cast<int>(naux),
        static_cast<long long>(tensor_elements_per_system), static_cast<int>(batch_size));
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      return fail_plan(candidate,
                       blas_failure(blas_status, "transform CUDA DF three-center tensor", detail));
    }
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
      20.0L * static_cast<long double>(matrix_bytes) +
      static_cast<long double>(batch_size) * (16.0L * sizeof(double) + 2.0L * sizeof(std::int32_t) +
                                              2.0L * sizeof(std::uint8_t) + sizeof(std::uint32_t)) +
      solver_device_workspace_bytes + matrix_bytes;  // graph bookkeeping
  const std::size_t persistent_scf_bytes =
      persistent_scf_estimate >= static_cast<long double>(std::numeric_limits<std::size_t>::max())
          ? std::numeric_limits<std::size_t>::max()
          : static_cast<std::size_t>(persistent_scf_estimate);
  const std::size_t persistent_device_bytes =
      6 * matrix_bytes + auxiliary_bytes + (candidate->streamed ? 0 : tensor_bytes) +
      3 * tile_bytes + matrix_bytes + (candidate->streamed ? tile_bytes : 0) +
      persistent_scf_bytes +
      (candidate->integral_source != nullptr
           ? 2 * metric_bytes + auxiliary_vector_bytes +
                 cuda_density_fitting_integral_source_device_bytes(candidate->integral_source)
           : 0);
  const std::size_t setup_device_bytes =
      (candidate->integral_source ? 2 * metric_bytes + auxiliary_vector_bytes
                                  : 3 * metric_bytes + 2 * auxiliary_vector_bytes) +
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
      static_cast<long double>(sizeof(*candidate)) +
      (candidate->integral_source
           ? static_cast<long double>(
                 cuda_density_fitting_integral_source_host_bytes(candidate->integral_source)) +
                 vector_capacity_bytes(candidate->metric_response_valid)
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
  }
  cuda_error = cudaStreamSynchronize(candidate->stream);
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate,
                     cuda_failure(cuda_error, "finish CUDA DF plan preparation", detail));
  }

  (void)cusolverDnDestroyParams(candidate->solver_parameters);
  candidate->solver_parameters = nullptr;
  (void)cusolverDnDestroy(candidate->solver);
  candidate->solver = nullptr;
  if (candidate->integral_source) {
    candidate->metric_eigenvectors = std::exchange(setup.metrics, nullptr);
    candidate->metric_eigenvalues = std::exchange(setup.eigenvalues, nullptr);
  }
  *plan = candidate;
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_df

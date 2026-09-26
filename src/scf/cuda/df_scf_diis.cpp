#include "scf/cuda/df_scf_diis.hpp"

#include <limits>
#include <new>

#include "runtime/cuda_component_trace.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda/scf_diis_kernels.hpp"
#include "scf/density_fitting.hpp"

namespace vibeqc::scf::cuda_df {

vibeqc_status allocate_scf_diis(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                unsigned history, std::string& detail) {
  if (history < 2) return VIBEQC_STATUS_SUCCESS;
  if (history > plan.scf_diis_history) {
    detail = "CUDA DF DIIS history exceeds its planned reservation";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  const auto bound = density_fitting_scf_diis_device_bytes(plan.batch_size, plan.nbf, history);
  if (bound == std::numeric_limits<std::size_t>::max()) {
    detail = "CUDA DF DIIS storage overflows size_t";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  const auto spins = state.unrestricted ? 2U : 1U;
  const auto elements = spins * state.expected;
  const auto dimension = static_cast<std::size_t>(history) + 1;
  const auto allocate = [&](void** pointer, std::size_t bytes) -> vibeqc_status {
    const auto status = allocate_device(pointer, bytes, "allocate CUDA DF DIIS state", detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    try {
      state.allocations.push_back(*pointer);
    } catch (const std::bad_alloc&) {
      (void)runtime::resource_cuda_free(*pointer);
      *pointer = nullptr;
      detail = "host allocation for CUDA DF DIIS handles failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    return VIBEQC_STATUS_SUCCESS;
  };
  // The conservative UHF bound above checks every product before these sizes
  // are formed. No matrix storage is borrowed from a final-state candidate.
  struct Buffer {
    void** pointer;
    std::size_t bytes;
  };
  const Buffer buffers[]{
      {reinterpret_cast<void**>(&state.d_diis_overlap), state.expected * sizeof(double)},
      {reinterpret_cast<void**>(&state.d_diis_residual), elements * sizeof(double)},
      {reinterpret_cast<void**>(&state.d_diis_temporary), state.expected * sizeof(double)},
      {reinterpret_cast<void**>(&state.d_diis_fock_history), elements * history * sizeof(double)},
      {reinterpret_cast<void**>(&state.d_diis_residual_history),
       elements * history * sizeof(double)},
      {reinterpret_cast<void**>(&state.d_diis_gram),
       plan.batch_size * dimension * dimension * sizeof(double)},
      {reinterpret_cast<void**>(&state.d_diis_coefficients),
       plan.batch_size * dimension * sizeof(double)},
      {reinterpret_cast<void**>(&state.d_diis_count), plan.batch_size * sizeof(std::uint32_t)},
      {reinterpret_cast<void**>(&state.d_diis_head), plan.batch_size * sizeof(std::uint32_t)},
  };
  for (const auto& buffer : buffers) {
    const auto status = allocate(buffer.pointer, buffer.bytes);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  if (state.unrestricted) {
    const auto status =
        allocate(reinterpret_cast<void**>(&state.d_diis_fock), elements * sizeof(double));
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  state.diis_history = history;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status reset_scf_diis(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                             const std::vector<double>& overlap, std::string& detail) {
  if (!state.diis_history) return VIBEQC_STATUS_SUCCESS;
  runtime::cuda_trace::TraceOperation trace(
      "compact_diis_reset", plan.stream,
      {plan.batch_size, plan.nbf, plan.naux, plan.integral_source != nullptr, plan.streamed});
  if (overlap.size() != state.expected || !finite_values(overlap)) {
    detail = "CUDA DF DIIS overlap has invalid dimensions or values";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  auto status =
      cudaMemcpyAsync(state.d_diis_overlap, overlap.data(), state.expected * sizeof(double),
                      cudaMemcpyHostToDevice, plan.stream);
  if (status == cudaSuccess)
    status = cudaMemsetAsync(state.d_diis_count, 0, plan.batch_size * sizeof(std::uint32_t),
                             plan.stream);
  if (status == cudaSuccess)
    status =
        cudaMemsetAsync(state.d_diis_head, 0, plan.batch_size * sizeof(std::uint32_t), plan.stream);
  return status == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                               : cuda_failure(status, "reset CUDA DF DIIS history", detail);
}

vibeqc_status apply_scf_diis(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                             std::string& detail) {
  if (!state.diis_history) return VIBEQC_STATUS_SUCCESS;
  runtime::cuda_trace::TraceOperation trace(
      "compact_diis", plan.stream,
      {plan.batch_size, plan.nbf, plan.naux, plan.integral_source != nullptr, plan.streamed});
  runtime::df_progress::Scope progress("compact_diis_update");
  runtime::df_progress::number("history_capacity", state.diis_history);
  runtime::cuda_trace::trace_counter("diis_cooperative_dots", plan.cooperative_diis);
  const auto spins = state.unrestricted ? 2U : 1U;
  const auto matrix = plan.matrix_elements;
  const auto bytes = matrix * sizeof(double);
  const double* densities[]{state.unrestricted ? state.d_alpha_density : state.d_density,
                            state.d_beta_density};
  double* focks[]{state.unrestricted ? state.d_alpha_fock : state.d_fock, state.d_beta_fock};
  // Device frames are column-major views of symmetric public AO matrices.
  // Reverse factors when forming a row-major FDS-SDF residual. The output
  // stride interleaves alpha/beta per system for the shared DIIS kernel.
  const auto product = [&](const double* left, const double* right, double* output,
                           std::size_t stride, double alpha = 1.0, double beta = 0.0) {
    const auto status = cublasDgemmStridedBatched(
        plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(plan.nbf), static_cast<int>(plan.nbf),
        static_cast<int>(plan.nbf), &alpha, left, static_cast<int>(plan.nbf), matrix, right,
        static_cast<int>(plan.nbf), matrix, &beta, output, static_cast<int>(plan.nbf), stride,
        static_cast<int>(plan.batch_size));
    return status == CUBLAS_STATUS_SUCCESS
               ? VIBEQC_STATUS_SUCCESS
               : blas_failure(status, "CUDA DF DIIS physical residual", detail);
  };
  runtime::cuda_trace::TraceRegion residual_products("diis_residual_products", plan.stream);
  for (unsigned spin = 0; spin < spins; ++spin) {
    auto* residual = state.d_diis_residual + spin * matrix;
    auto status = product(densities[spin], focks[spin], state.d_diis_temporary, matrix);
    if (status == VIBEQC_STATUS_SUCCESS)
      status = product(state.d_diis_overlap, state.d_diis_temporary, residual, spins * matrix);
    if (status == VIBEQC_STATUS_SUCCESS)
      status = product(densities[spin], state.d_diis_overlap, state.d_diis_temporary, matrix);
    if (status == VIBEQC_STATUS_SUCCESS)
      status = product(focks[spin], state.d_diis_temporary, residual, spins * matrix, -1.0, 1.0);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  residual_products.finish();
  double* effective = state.unrestricted ? state.d_diis_fock : state.d_fock;
  if (state.unrestricted)
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto status =
          cudaMemcpy2DAsync(effective + spin * matrix, spins * bytes, focks[spin], bytes, bytes,
                            plan.batch_size, cudaMemcpyDeviceToDevice, plan.stream);
      if (status != cudaSuccess)
        return cuda_failure(status, "pack CUDA DF spin DIIS Focks", detail);
    }
  // Common normalization and oldest-dependent-history retirement match the
  // existing CPU DIIS policy and the qualified KS device implementation.
  runtime::cuda_trace::TraceRegion update("diis_history_update", plan.stream);
  const auto parts = (spins * matrix + 4095) / 4096;
  const auto history = std::size_t{state.diis_history};
  const bool parallel_dots = plan.cooperative_diis && plan.batch_size <= 65535 &&
                             history * history <= 65535 && parts <= matrix / (history * history);
  const double* partials = nullptr;
  if (parallel_dots) {
    // All residual GEMMs have consumed this AO temporary. Lend it to a
    // deterministic two-level reduction; no extra allocation or host wait
    // is needed, and dependent-history retries reuse the same dot products.
    partials = state.d_diis_temporary;
    cuda_execution::launch_diis_dot_partials(
        plan.stream, static_cast<std::int32_t>(plan.batch_size),
        static_cast<std::int32_t>(plan.nbf), spins, state.diis_history, state.d_diis_residual,
        state.d_diis_residual_history, state.d_active, state.d_diis_count, state.d_diis_head, parts,
        state.d_diis_temporary);
    const auto error = cudaPeekAtLastError();
    if (error != cudaSuccess) return cuda_failure(error, "CUDA DF DIIS dot partials", detail);
    runtime::cuda_trace::trace_counter(
        "diis_dot_partial_bytes", plan.batch_size * history * history * parts * sizeof(double));
  }
  cuda_execution::launch_update_diis_kernel(
      static_cast<unsigned>(plan.batch_size), 32, 0, plan.stream,
      static_cast<std::int32_t>(plan.batch_size), static_cast<std::int32_t>(plan.nbf), spins,
      state.diis_history, effective, state.d_diis_residual, state.d_active,
      state.d_diis_fock_history, state.d_diis_residual_history, state.d_diis_gram,
      state.d_diis_coefficients, state.d_diis_count, state.d_diis_head, effective, true,
      plan.cooperative_diis, partials, parallel_dots ? parts : 0);
  auto status = cudaPeekAtLastError();
  update.finish();
  if (status != cudaSuccess) return cuda_failure(status, "update CUDA DF DIIS proposal", detail);
  if (state.unrestricted)
    for (unsigned spin = 0; spin < spins; ++spin) {
      status = cudaMemcpy2DAsync(focks[spin], bytes, effective + spin * matrix, spins * bytes,
                                 bytes, plan.batch_size, cudaMemcpyDeviceToDevice, plan.stream);
      if (status != cudaSuccess)
        return cuda_failure(status, "unpack CUDA DF spin DIIS proposals", detail);
    }
  return VIBEQC_STATUS_SUCCESS;
}
}  // namespace vibeqc::scf::cuda_df

#include "scf/cuda/df_scf_factor.hpp"

#include <algorithm>
#include <cstdlib>
#include <new>

#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_kernels.hpp"

namespace vibeqc::scf::cuda_df {

vibeqc_status occupied_scf_policy(bool& enabled, std::string& detail) {
  const char* value = std::getenv("VIBEQC_DF_EXCHANGE");
  enabled = value && std::string(value) == "occupied";
  if (value && !enabled && std::string(value) != "dense") {
    detail = "VIBEQC_DF_EXCHANGE must be dense or occupied";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status allocate_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                   const std::vector<std::int32_t>& alpha,
                                   const std::vector<std::int32_t>& beta, std::string& detail) {
  try {
    state.factor_alpha_ranks = alpha;
    state.factor_beta_ranks = beta;
    state.alpha_factor_rank = *std::max_element(alpha.begin(), alpha.end());
    state.beta_factor_rank = beta.empty() ? 0 : *std::max_element(beta.begin(), beta.end());
    const auto allocate = [&](auto** pointer, std::size_t bytes) -> vibeqc_status {
      if (!bytes) return VIBEQC_STATUS_SUCCESS;
      const auto status = allocate_device(reinterpret_cast<void**>(pointer), bytes,
                                          "allocate CUDA DF occupied SCF state", detail);
      if (status == VIBEQC_STATUS_SUCCESS) {
        try {
          state.allocations.push_back(*pointer);
        } catch (const std::bad_alloc&) {
          (void)runtime::resource_cuda_free(*pointer);
          *pointer = nullptr;
          throw;
        }
      }
      return status;
    };
    auto status = allocate(&state.d_alpha_factor,
                           plan.batch_size * plan.nbf * state.alpha_factor_rank * sizeof(double));
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(&state.d_alpha_factor_generation, plan.batch_size * sizeof(std::uint32_t));
    if (status == VIBEQC_STATUS_SUCCESS && !beta.empty())
      status = allocate(&state.d_beta_factor,
                        plan.batch_size * plan.nbf * state.beta_factor_rank * sizeof(double));
    if (status == VIBEQC_STATUS_SUCCESS && !beta.empty())
      status = allocate(&state.d_beta_factor_generation, plan.batch_size * sizeof(std::uint32_t));
    if (status == VIBEQC_STATUS_SUCCESS) status = allocate(&state.d_factor_error, sizeof(int));
    return status;
  } catch (const std::bad_alloc&) {
    detail = "host allocation for occupied SCF ownership failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
}

vibeqc_status reset_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                std::string& detail) {
  // Every imported/warm density is untrusted until the first dense iteration
  // builds C and D together. Old factors are unreachable before that seed.
  auto error = cudaMemsetAsync(state.d_factor_error, 0, sizeof(int), plan.stream);
  if (error == cudaSuccess)
    error = cudaMemsetAsync(state.d_alpha_factor_generation, 0,
                            plan.batch_size * sizeof(std::uint32_t), plan.stream);
  if (error == cudaSuccess && state.unrestricted)
    error = cudaMemsetAsync(state.d_beta_factor_generation, 0,
                            plan.batch_size * sizeof(std::uint32_t), plan.stream);
  return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                              : cuda_failure(error, "reset occupied SCF generations", detail);
}

vibeqc_status build_scf_occupied_jk(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                    const double* alpha, const double* beta, bool ready,
                                    std::string& detail) {
  const JkTermSelection terms{true, !ready};
  auto status = beta ? execute_cuda_density_fitting_uhf_jk_device(&plan, alpha, beta, plan.coulomb,
                                                                  plan.alpha_exchange,
                                                                  plan.beta_exchange, detail, terms)
                     : execute_cuda_density_fitting_rhf_jk_device(
                           &plan, alpha, plan.coulomb, plan.alpha_exchange, detail, terms);
  if (status != VIBEQC_STATUS_SUCCESS || !ready) return status;
  launch_validate_device_occupied_kernel(
      blocks_for(plan.batch_size), kThreads, 0, plan.stream, plan.batch_size, state.d_iterations,
      state.d_alpha_factor_generation, state.d_beta_factor_generation, state.d_factor_error);
  for (std::size_t item = 0; item < plan.batch_size; ++item) {
    status = build_occupied_exchange(
        plan, item,
        state.d_alpha_factor ? state.d_alpha_factor + item * plan.nbf * state.alpha_factor_rank
                             : nullptr,
        state.factor_alpha_ranks[item], true, beta ? 1 : 2, plan.alpha_exchange, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    if (beta) {
      status = build_occupied_exchange(
          plan, item,
          state.d_beta_factor ? state.d_beta_factor + item * plan.nbf * state.beta_factor_rank
                              : nullptr,
          state.factor_beta_ranks[item], true, 1, plan.beta_exchange, detail);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

void store_scf_factor(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                      const double* coefficients, bool beta) {
  launch_store_device_occupied_kernel(
      static_cast<unsigned>(plan.batch_size), kThreads, 0, plan.stream, plan.nbf,
      beta ? state.beta_factor_rank : state.alpha_factor_rank,
      beta                 ? state.d_beta_occupied
      : state.unrestricted ? state.d_alpha_occupied
                           : state.d_occupied,
      coefficients, state.d_active, state.d_iterations,
      beta ? state.d_beta_factor : state.d_alpha_factor,
      beta ? state.d_beta_factor_generation : state.d_alpha_factor_generation);
}

vibeqc_status verify_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                 const std::vector<std::uint32_t>& iterations,
                                 std::string& detail) {
  using namespace runtime::cuda_trace;
  TraceOperation trace(
      "occupied_scf_provenance", plan.stream,
      {plan.batch_size, plan.nbf, plan.naux, plan.integral_source != nullptr, plan.streamed});
  launch_validate_device_occupied_kernel(
      blocks_for(plan.batch_size), kThreads, 0, plan.stream, plan.batch_size, state.d_iterations,
      state.d_alpha_factor_generation, state.d_beta_factor_generation, state.d_factor_error);
  int failed = 0;
  auto error = cudaMemcpyAsync(&failed, state.d_factor_error, sizeof(int), cudaMemcpyDeviceToHost,
                               plan.stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(plan.stream);
  if (error != cudaSuccess) return cuda_failure(error, "validate occupied SCF provenance", detail);
  if (failed) {
    detail = "CUDA DF occupied factor generation differs from its canonical density";
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  trace_counter("validated_density_generations", plan.batch_size);
  trace_counter("dense_seed_iterations", 1);
  trace_counter("occupied_iterations", *std::max_element(iterations.begin(), iterations.end()) - 1);
  trace_counter("occupied_state_bytes",
                plan.batch_size * plan.nbf * (state.alpha_factor_rank + state.beta_factor_rank) *
                        sizeof(double) +
                    plan.batch_size * (state.unrestricted ? 2 : 1) * sizeof(std::uint32_t) +
                    sizeof(int));
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_df

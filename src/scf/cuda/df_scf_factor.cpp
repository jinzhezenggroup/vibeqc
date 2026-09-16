#include "scf/cuda/df_scf_factor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "runtime/cuda_component_trace.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_kernels.hpp"
#include "scf/cuda/df_scf_library.hpp"
#include "scf/df_exchange_policy.hpp"

namespace vibeqc::scf::cuda_df {

vibeqc_status qualified_resident_rhf_exchange(const CudaDensityFittingJkPlan& plan,
                                              std::span<const std::int32_t> alpha,
                                              std::span<const std::int32_t> beta, bool& qualified,
                                              std::string& detail) {
  qualified = false;
  const bool packed_resident = plan.value_storage.pairs == DfPairStorage::SymmetricLower &&
                               plan.integral_source && plan.packed_raw &&
                               plan.value_storage.rank_capacity >= 160;
  if (plan.occupied_scf_reserved && plan.nbf == 768 && plan.naux == 768 && plan.batch_size == 1 &&
      alpha.size() == 1 && alpha[0] == 160 && beta.empty() &&
      (!plan.integral_source || packed_resident) && !plan.streamed && plan.row_tile == plan.nbf &&
      (plan.auxiliary_tile == plan.naux || packed_resident)) {
    // Explicit packed experiments keep the established occupied/seed/final
    // algorithm in the same domain. This does not automatically select packing.
    cudaDeviceProp properties{};
    const auto error = cudaGetDeviceProperties(&properties, plan.device_id);
    if (error != cudaSuccess) return cuda_failure(error, "DF exchange device identity", detail);
    qualified = properties.major == 12 && properties.minor == 0 &&
                std::strcmp(properties.name, "NVIDIA GeForce RTX 5090") == 0;
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status factor_density_for_exchange(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                          const double* density, bool& accepted, std::size_t& rank,
                                          std::string& detail) {
  using namespace runtime::cuda_trace;
  TraceOperation trace(
      "density_exchange_seed", plan.stream,
      {plan.batch_size, plan.nbf, plan.naux, plan.integral_source != nullptr, plan.streamed});
  accepted = false;
  rank = 0;
  // This first qualification uses existing singleton resident RHF workspace.
  // Unsupported plans retain dense exchange without allocating another solver.
  const bool packed = plan.value_storage.pairs == DfPairStorage::SymmetricLower &&
                      plan.integral_source && plan.packed_raw;
  if (state.unrestricted || plan.batch_size != 1 || !state.occupied_exchange || plan.streamed ||
      (plan.integral_source && !packed) || plan.row_tile != plan.nbf ||
      (!packed && plan.auxiliary_tile != plan.naux) || plan.nbf < 2) {
    trace_counter("unsupported", 1);
    return VIBEQC_STATUS_SUCCESS;
  }
  constexpr double spectral_threshold = 1e-13;
  constexpr double discarded_frobenius_tolerance = 1e-12;
  constexpr double maximum_tolerance = 1e-12;
  constexpr double rms_tolerance = 1e-13;
  auto error = cudaMemcpyAsync(state.d_fock, density, plan.matrix_elements * sizeof(double),
                               cudaMemcpyDeviceToDevice, plan.stream);
  if (error != cudaSuccess) return cuda_failure(error, "copy density seed for eigensolve", detail);
  auto status = solve_device_batch(plan, state.solver, plan.nbf, 1, state.d_fock,
                                   state.d_eigenvalues, state.d_info, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  std::vector<double> values(plan.nbf);
  int info = 0;
  error = cudaMemcpyAsync(values.data(), state.d_eigenvalues, plan.nbf * sizeof(double),
                          cudaMemcpyDeviceToHost, plan.stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&info, state.d_info, sizeof(info), cudaMemcpyDeviceToHost, plan.stream);
  const auto spectrum_drained = cudaStreamSynchronize(plan.stream);
  if (error == cudaSuccess) error = spectrum_drained;
  if (error != cudaSuccess) return cuda_failure(error, "read density seed spectrum", detail);
  trace_counter("d2h_bytes", plan.nbf * sizeof(double) + sizeof(info));
  trace_counter("explicit_synchronizations", 1);
  if (info) {
    trace_counter("solver_rejected", 1);
    return VIBEQC_STATUS_SUCCESS;
  }
  double discarded_square = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double value = values[i];
    if (!std::isfinite(value) || value < -spectral_threshold || (i && value < values[i - 1])) {
      trace_counter("spectrum_rejected", 1);
      return VIBEQC_STATUS_SUCCESS;
    }
    if (value > spectral_threshold)
      ++rank;
    else
      discarded_square += value * value;
  }
  trace_counter("candidate_rank", rank);
  if (rank > state.alpha_factor_rank ||
      discarded_square > discarded_frobenius_tolerance * discarded_frobenius_tolerance) {
    trace_counter("rank_rejected", 1);
    return VIBEQC_STATUS_SUCCESS;
  }
  // d_fock/d_eigenvalues are scratch before the first physical SCF iteration.
  // No canonical generation is published for this algebraic factor.
  launch_density_exchange_factor(plan.stream, plan.nbf, rank, state.d_fock, state.d_eigenvalues,
                                 state.d_alpha_factor);
  if (rank) {
    const double one = 1, zero = 0;
    const auto n = static_cast<int>(plan.nbf);
    const auto product = trace_call("density_seed_reconstruction", plan.stream, [&] {
      return cublasDgemm(plan.blas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, static_cast<int>(rank), &one,
                         state.d_alpha_factor, n, state.d_alpha_factor, n, &zero, state.d_temporary,
                         n);
    });
    if (product != CUBLAS_STATUS_SUCCESS)
      return blas_failure(product, "reconstruct density seed", detail);
  } else {
    error =
        cudaMemsetAsync(state.d_temporary, 0, plan.matrix_elements * sizeof(double), plan.stream);
    if (error != cudaSuccess) return cuda_failure(error, "zero density reconstruction", detail);
  }
  // Checking the full original matrix also rejects nonsymmetric input: the
  // eigensolver reads just one triangle, but promotion must reproduce both.
  launch_density_exchange_error(plan.stream, plan.matrix_elements, density, state.d_temporary,
                                state.d_next_density);
  double errors[2]{};
  error = cudaMemcpyAsync(errors, state.d_next_density, sizeof(errors), cudaMemcpyDeviceToHost,
                          plan.stream);
  const auto reconstruction_drained = cudaStreamSynchronize(plan.stream);
  if (error == cudaSuccess) error = reconstruction_drained;
  if (error != cudaSuccess) return cuda_failure(error, "verify density reconstruction", detail);
  trace_counter("d2h_bytes", sizeof(errors));
  trace_counter("explicit_synchronizations", 1);
  const auto record = [](const char* name, double value) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.17g", value);
    runtime::df_progress::label(name, text);
  };
  record("seed_reconstruction_maximum", errors[0]);
  record("seed_reconstruction_rms", errors[1]);
  record("seed_discarded_frobenius", std::sqrt(discarded_square));
  accepted = std::isfinite(errors[0]) && std::isfinite(errors[1]) &&
             errors[0] <= maximum_tolerance && errors[1] <= rms_tolerance;
  trace_counter(accepted ? "accepted" : "reconstruction_rejected", 1);
  runtime::df_progress::number("density_seed_rank", rank);
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status occupied_scf_policy(const CudaDensityFittingJkPlan& plan, bool& enabled,
                                  std::string& detail, std::span<const std::int32_t> alpha,
                                  std::span<const std::int32_t> beta) {
  const char* value = std::getenv("VIBEQC_DF_EXCHANGE");
  // No dimensions are supplied here: the helper's zero defaults keep auto
  // disabled until every runtime qualification below succeeds.
  enabled = df_occupied_exchange_requested();
  if (value && !enabled && std::string(value) != "dense" && std::string(value) != "auto") {
    detail = "VIBEQC_DF_EXCHANGE must be auto, dense or occupied";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (df_occupied_exchange_auto_requested()) {
    // This qualification is deliberately an exact measured domain, including
    // rank/reference and backend identity, rather than an AO-only threshold.
    const auto status = qualified_resident_rhf_exchange(plan, alpha, beta, enabled, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  if (enabled && !plan.occupied_scf_reserved) {
    detail = "CUDA DF plan did not reserve occupied SCF storage; recreate the plan";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status allocate_scf_factors(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                   const std::vector<std::int32_t>& alpha,
                                   const std::vector<std::int32_t>& beta, std::string& detail) {
  if (!plan.occupied_scf_reserved) {
    detail = "CUDA DF plan did not reserve occupied SCF storage; recreate the plan";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
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
  // Every solve discards old provenance before attempting an algebraic seed.
  // Only the first canonical SCF update may publish a new orbital generation.
  state.density_seed_used = false;
  state.density_seed_rank = 0;
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
  bool seed = false;
  std::size_t seed_rank = 0;
  if (!ready && !beta && state.occupied_exchange) {
    const char* policy = std::getenv("VIBEQC_DF_SEED_EXCHANGE");
    if (policy && std::string(policy) != "auto" && std::string(policy) != "dense" &&
        std::string(policy) != "factor") {
      detail = "VIBEQC_DF_SEED_EXCHANGE must be auto, dense or factor";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    bool selected = policy && std::string(policy) == "factor";
    if (!policy || std::string(policy) == "auto") {
      // #399's matched-density endpoint qualifies the existing 768/768/160
      // resident RTX 5090 domain only; other shapes keep their dense seeds.
      const auto status = qualified_resident_rhf_exchange(
          plan, state.factor_alpha_ranks, state.factor_beta_ranks, selected, detail);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
    }
    if (selected) {
      const auto status = factor_density_for_exchange(plan, state, alpha, seed, seed_rank, detail);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
    }
  }
  const JkTermSelection terms{true, !ready && !seed};
  auto status = beta ? execute_cuda_density_fitting_uhf_jk_device(&plan, alpha, beta, plan.coulomb,
                                                                  plan.alpha_exchange,
                                                                  plan.beta_exchange, detail, terms)
                     : execute_cuda_density_fitting_rhf_jk_device(
                           &plan, alpha, plan.coulomb, plan.alpha_exchange, detail, terms);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (seed) {
    state.density_seed_used = true;
    state.density_seed_rank = seed_rank;
    status = build_occupied_exchange(plan, 0, state.d_alpha_factor, seed_rank, true, 1,
                                     plan.alpha_exchange, detail);
    const char* verify = std::getenv("VIBEQC_DF_SEED_VERIFY");
    if (status != VIBEQC_STATUS_SUCCESS || !verify || std::string(verify) != "1") return status;
    // Intrusive qualification only: compute both K matrices for the identical
    // supplied density. Restore candidate K so endpoint gates test its actual
    // arithmetic, and never include this pass in clean performance claims.
    runtime::cuda_trace::TraceOperation trace("density_seed_k_validation", plan.stream,
                                              {1, plan.nbf, plan.naux, false, false});
    auto error =
        cudaMemcpyAsync(state.d_fock, plan.alpha_exchange, plan.matrix_elements * sizeof(double),
                        cudaMemcpyDeviceToDevice, plan.stream);
    if (error != cudaSuccess) return cuda_failure(error, "retain candidate seed K", detail);
    status = build_exchange(plan, alpha, plan.alpha_exchange, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    launch_density_exchange_error(plan.stream, plan.matrix_elements, state.d_fock,
                                  plan.alpha_exchange, state.d_next_density);
    double errors[2]{};
    error = cudaMemcpyAsync(errors, state.d_next_density, sizeof(errors), cudaMemcpyDeviceToHost,
                            plan.stream);
    const auto drained = cudaStreamSynchronize(plan.stream);
    if (error == cudaSuccess) error = drained;
    if (error != cudaSuccess) return cuda_failure(error, "read seed K validation", detail);
    for (unsigned i = 0; i < 2; ++i) {
      char text[64];
      std::snprintf(text, sizeof(text), "%.17g", errors[i]);
      runtime::df_progress::label(i ? "seed_k_rms_error" : "seed_k_maximum_error", text);
      std::snprintf(text, sizeof(text), "%.17g", 0.5 * errors[i]);
      runtime::df_progress::label(i ? "seed_fock_rms_error" : "seed_fock_maximum_error", text);
    }
    if (!std::isfinite(errors[0]) || !std::isfinite(errors[1]) || errors[0] > 1e-10 ||
        errors[1] > 1e-11) {
      detail = "density seed K failed dense reference qualification";
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    error =
        cudaMemcpyAsync(plan.alpha_exchange, state.d_fock, plan.matrix_elements * sizeof(double),
                        cudaMemcpyDeviceToDevice, plan.stream);
    return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                                : cuda_failure(error, "restore candidate seed K", detail);
  }
  if (!ready) return status;
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
  trace_counter("dense_seed_iterations", !state.density_seed_used);
  trace_counter("factor_seed_iterations", state.density_seed_used);
  trace_counter("density_seed_rank", state.density_seed_rank);
  trace_counter("occupied_iterations", *std::max_element(iterations.begin(), iterations.end()) - 1);
  trace_counter("occupied_state_bytes",
                plan.batch_size * plan.nbf * (state.alpha_factor_rank + state.beta_factor_rank) *
                        sizeof(double) +
                    plan.batch_size * (state.unrestricted ? 2 : 1) * sizeof(std::uint32_t) +
                    sizeof(int));
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_df

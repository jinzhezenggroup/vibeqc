#include "scf/cuda/df_scf_final_state.hpp"

#include <algorithm>
#include <limits>
#include <new>

#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_factor.hpp"
#include "scf/cuda/df_scf_kernels.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"

namespace vibeqc::scf::cuda_df {
vibeqc_status begin_scf_final_state_solve(CudaDensityFittingJkPlan& plan, std::string& detail) {
  if (auto* state = static_cast<PersistentScfState*>(plan.persistent_scf_state))
    state->final_frames_available = false;
  if (!plan.factor_basis_identity ||
      plan.final_state_solve_epoch == std::numeric_limits<std::uint64_t>::max()) {
    detail = "CUDA DF final-state owner or solve epoch exhausted";
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  ++plan.final_state_solve_epoch;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status allocate_scf_final_frames(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                        std::string& detail) {
  try {
    (void)df_final_snapshot_device_reservation(plan.nbf, plan.batch_size);
    const auto allocate = [&](auto** pointer, std::size_t bytes) {
      const auto status = allocate_device(reinterpret_cast<void**>(pointer), bytes,
                                          "allocate CUDA DF final frame", detail);
      if (status == VIBEQC_STATUS_SUCCESS) {
        try {
          state.allocations.push_back(*pointer);
        } catch (...) {
          (void)runtime::resource_cuda_free(*pointer);
          *pointer = nullptr;
          throw;
        }
      }
      return status;
    };
    for (unsigned spin = 0; spin < (state.unrestricted ? 2U : 1U); ++spin) {
      auto status =
          allocate(spin ? &state.d_final_beta_coefficients : &state.d_final_alpha_coefficients,
                   state.expected * sizeof(double));
      if (status == VIBEQC_STATUS_SUCCESS)
        status = allocate(spin ? &state.d_final_beta_values : &state.d_final_alpha_values,
                          plan.batch_size * plan.nbf * sizeof(double));
      if (status == VIBEQC_STATUS_SUCCESS)
        status = allocate(spin ? &state.d_final_beta_generation : &state.d_final_alpha_generation,
                          plan.batch_size * sizeof(std::uint64_t));
      if (status == VIBEQC_STATUS_SUCCESS)
        status = allocate(spin ? &state.d_final_beta_info : &state.d_final_alpha_info,
                          plan.batch_size * sizeof(int));
      if (status != VIBEQC_STATUS_SUCCESS) return status;
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF final-frame ownership failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::overflow_error& error) {
    detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
}

vibeqc_status reset_scf_final_frames(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                                     const std::vector<std::int32_t>& alpha,
                                     const std::vector<std::int32_t>& beta, std::string& detail) {
  state.final_frames_available = false;
  state.final_alpha_occupied = alpha;
  state.final_beta_occupied = beta;
  state.final_iterations.assign(plan.batch_size, 0);
  auto error = cudaMemsetAsync(state.d_final_alpha_generation, 0,
                               plan.batch_size * sizeof(std::uint64_t), plan.stream);
  if (error == cudaSuccess && state.unrestricted)
    error = cudaMemsetAsync(state.d_final_beta_generation, 0,
                            plan.batch_size * sizeof(std::uint64_t), plan.stream);
  return error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(error, "reset CUDA DF final-frame generations", detail);
}

void store_scf_final_frame(CudaDensityFittingJkPlan& plan, PersistentScfState& state,
                           const double* coefficients, bool beta) {
  launch_store_device_final_frame_kernel(
      static_cast<unsigned>(plan.batch_size), kThreads, 0, plan.stream, plan.nbf, coefficients,
      beta                 ? state.d_beta_eigenvalues
      : state.unrestricted ? state.d_alpha_eigenvalues
                           : state.d_eigenvalues,
      beta                 ? state.d_beta_info
      : state.unrestricted ? state.d_alpha_info
                           : state.d_info,
      state.d_active, state.d_iterations,
      beta ? state.d_final_beta_coefficients : state.d_final_alpha_coefficients,
      beta ? state.d_final_beta_values : state.d_final_alpha_values,
      beta ? state.d_final_beta_generation : state.d_final_alpha_generation,
      beta ? state.d_final_beta_info : state.d_final_alpha_info);
}

void publish_scf_final_frames(PersistentScfState& state,
                              const std::vector<CudaDensityFittingDeviceScfItem>& results) {
  for (std::size_t item = 0; item < results.size(); ++item)
    state.final_iterations[item] =
        results[item].status == VIBEQC_STATUS_SUCCESS && results[item].converged
            ? results[item].iterations
            : 0;
  state.final_frames_available = true;
}
}  // namespace vibeqc::scf::cuda_df

namespace vibeqc::scf {
using namespace cuda_df;
std::uint64_t cuda_density_fitting_solve_epoch(const CudaDensityFittingJkPlan* plan) noexcept {
  // Recovery cannot manufacture another valid solve after epoch exhaustion.
  // Conservatively exclude the saturated value from the host-recovery path.
  return plan && plan->final_state_solve_epoch != std::numeric_limits<std::uint64_t>::max()
             ? plan->final_state_solve_epoch
             : 0;
}
vibeqc_status cuda_density_fitting_final_state_token(const CudaDensityFittingJkPlan* plan,
                                                     std::size_t item, CudaDfFinalStateToken& token,
                                                     std::string& detail) {
  token = {};
  detail.clear();
  const auto* state =
      plan ? static_cast<const PersistentScfState*>(plan->persistent_scf_state) : nullptr;
  if (!state || !state->final_frames_available || !plan->final_state_solve_epoch ||
      item >= plan->batch_size || state->final_iterations.size() != plan->batch_size ||
      !state->final_iterations[item]) {
    detail = "CUDA DF item has no successful current final frame";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  bool occupied_exchange = false;
  const auto policy = occupied_scf_policy(*plan, occupied_exchange, detail,
                                          state->final_alpha_occupied, state->final_beta_occupied);
  if (policy != VIBEQC_STATUS_SUCCESS) return policy;
  if (occupied_exchange != state->occupied_exchange) {
    detail = "CUDA DF exchange policy changed after the retained solve";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    const auto generation = static_cast<std::uint64_t>(state->final_iterations[item]) + 1;
    token.identity.factor =
        cuda_density_fitting_factor_identity(plan, item, generation, generation);
    token.identity.solve_epoch = plan->final_state_solve_epoch;
    // The compact loop implements exactly standard full-range DF HF. The
    // source and its generation/metric policy are immutable within this plan.
    token.identity.model = resolve_fock_build(
        make_hf_fock_spec(state->unrestricted ? FockSpin::Unrestricted : FockSpin::Restricted,
                          FockApproximation::DensityFitted),
        FockBackend::Cuda, 1e-12, plan->metric_relative_threshold);
    token.identity.occupied = {static_cast<std::size_t>(state->final_alpha_occupied[item])};
    if (state->unrestricted)
      token.identity.occupied.push_back(static_cast<std::size_t>(state->final_beta_occupied[item]));
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    token = {};
    detail = "host allocation for CUDA DF final-state token failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
}

vibeqc_status read_cuda_density_fitting_final_state(CudaDensityFittingJkPlan* plan,
                                                    const CudaDfFinalStateToken& expected,
                                                    CudaDfFinalStateSnapshot& snapshot,
                                                    std::string& detail) {
  snapshot = {};
  try {
    CudaDfFinalStateToken current;
    const auto item = expected.identity.factor.reference - 1;
    const auto status = cuda_density_fitting_final_state_token(plan, item, current, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    if (expected != current) {
      detail = "CUDA DF final-state token has stale owner, epoch, generation, model or occupations";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    auto error = cudaSetDevice(plan->device_id);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (error == cudaSuccess) error = cudaStreamIsCapturing(plan->stream, &capture);
    if (error != cudaSuccess)
      return cuda_failure(error, "inspect CUDA DF final-state stream", detail);
    if (capture != cudaStreamCaptureStatusNone) {
      detail = "CUDA DF final-state read requires an ordinary noncapturing stream";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto& state = *static_cast<const PersistentScfState*>(plan->persistent_scf_state);
    const std::size_t spins = state.unrestricted ? 2 : 1, n = plan->nbf, count = n * n;
    CudaDfFinalStateSnapshot local;
    local.candidate.identity = current.identity;
    local.candidate.physical_origin = true;
    local.candidate.fock_density_generation = current.identity.factor.density_generation - 1;
    local.candidate.spins.resize(spins);
    local.density.resize(spins);
    // Allocate all pageable destinations before the first asynchronous copy,
    // then always drain the stream before any local storage can be released.
    for (std::size_t spin = 0; spin < spins; ++spin) {
      local.candidate.spins[spin].vectors.resize(count);
      local.candidate.spins[spin].values.resize(n);
      local.density[spin].resize(count);
    }
    std::uint64_t generations[2]{};
    int info[2]{};
    runtime::cuda_trace::TraceOperation trace(
        "final_state_download", plan->stream,
        {1, n, plan->naux, plan->integral_source != nullptr, plan->streamed});
    const auto copy = [&](void* target, const void* source, std::size_t bytes) {
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToHost, plan->stream);
    };
    for (std::size_t spin = 0; spin < spins; ++spin) {
      copy(local.candidate.spins[spin].vectors.data(),
           (spin ? state.d_final_beta_coefficients : state.d_final_alpha_coefficients) +
               item * count,
           count * sizeof(double));
      copy(local.candidate.spins[spin].values.data(),
           (spin ? state.d_final_beta_values : state.d_final_alpha_values) + item * n,
           n * sizeof(double));
      copy(local.density[spin].data(),
           (spin                 ? state.d_beta_density
            : state.unrestricted ? state.d_alpha_density
                                 : state.d_density) +
               item * count,
           count * sizeof(double));
      copy(&generations[spin],
           (spin ? state.d_final_beta_generation : state.d_final_alpha_generation) + item,
           sizeof(std::uint64_t));
      copy(&info[spin], (spin ? state.d_final_beta_info : state.d_final_alpha_info) + item,
           sizeof(int));
    }
    const auto drained = cudaStreamSynchronize(plan->stream);
    if (error == cudaSuccess) error = drained;
    if (error != cudaSuccess) return cuda_failure(error, "read CUDA DF final frame", detail);
    for (std::size_t spin = 0; spin < spins; ++spin) {
      auto& frame = local.candidate.spins[spin];
      if (generations[spin] != current.identity.factor.density_generation || info[spin] != 0 ||
          !finite_values(frame.values) || !finite_values(frame.vectors) ||
          !finite_values(local.density[spin])) {
        detail = "CUDA DF retained frame has stale generation, solver failure or nonfinite data";
        return VIBEQC_STATUS_NUMERICAL_FAILURE;
      }
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = row + 1; column < n; ++column)
          std::swap(frame.vectors[row * n + column], frame.vectors[column * n + row]);
    }
    runtime::cuda_trace::trace_counter(
        "final_state_download_bytes",
        spins * ((2 * count + n) * sizeof(double) + sizeof(std::uint64_t) + sizeof(int)));
    snapshot = std::move(local);
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "host allocation for detached CUDA DF final frame failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
}
}  // namespace vibeqc::scf

#include "scf/cuda/df_scf_final_state.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <new>

#include "runtime/cuda_component_trace.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_factor.hpp"
#include "scf/cuda/df_scf_kernels.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"

namespace vibeqc::scf::cuda_df {
vibeqc_status begin_scf_final_state_solve(CudaDensityFittingJkPlan& plan, std::string& detail) {
  plan.final_projection_token.reset();
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

namespace {
bool bounded_corrected_final_rhf_identity(const CudaDfFinalStateToken& retained,
                                          const CudaDfFinalStateToken& requested) {
  const auto& old = retained.identity;
  const auto& next = requested.identity;
  if (requested.version != retained.version || next.factor.basis != old.factor.basis ||
      next.factor.reference != old.factor.reference || next.solve_epoch != old.solve_epoch ||
      next.model != old.model || next.occupied != old.occupied || next.occupied.size() != 1 ||
      !next.occupied[0] || next.factor.density_generation <= old.factor.density_generation ||
      next.factor.orbital_generation <= old.factor.orbital_generation)
    return false;
  const auto density_delta = next.factor.density_generation - old.factor.density_generation;
  const auto orbital_delta = next.factor.orbital_generation - old.factor.orbital_generation;
  // Mirror the corrected-response provenance fence: strict finalization may
  // advance generations, but an arbitrarily stale detached identity may not.
  return density_delta == orbital_delta && density_delta <= 16;
}
}  // namespace

vibeqc_status try_cuda_density_fitting_final_rhf_jk(CudaDensityFittingJkPlan* plan,
                                                    const CudaDfFinalStateToken& expected,
                                                    const std::vector<double>& density,
                                                    std::vector<double>& coulomb,
                                                    std::vector<double>& exchange, bool& used,
                                                    std::string& detail, bool download) {
  using namespace runtime::cuda_trace;
  used = false;
  if (plan) plan->final_projection_token.reset();
  const char* policy = std::getenv("VIBEQC_DF_FINAL_EXCHANGE");
  if (policy && std::string(policy) != "auto" && std::string(policy) != "dense" &&
      std::string(policy) != "occupied") {
    detail = "VIBEQC_DF_FINAL_EXCHANGE must be auto, dense or occupied";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  runtime::df_progress::label("final_exchange_policy", policy ? policy : "auto");
  const auto fallback = [](const char* reason) {
    runtime::df_progress::label("final_exchange_fallback", reason);
    return VIBEQC_STATUS_SUCCESS;
  };
  // Keep the final-state ablation independent of the seed control.
  const bool packed = plan && plan->value_storage.pairs == DfPairStorage::SymmetricLower &&
                      plan->integral_source && plan->packed_raw;
  const bool source_dense_resident = plan && plan->integral_source && !packed &&
                                     plan->resident_raw_valid && plan->row_tile == plan->nbf &&
                                     plan->auxiliary_tile == plan->naux;
  if (policy && std::string(policy) == "dense") return fallback("policy_dense");
  if (!plan) return fallback("missing_plan");
  if (plan->batch_size != 1) return fallback("non_singleton");
  const bool automatic = !policy || std::string(policy) == "auto";
  const bool explicit_occupied = policy && std::string(policy) == "occupied";
  const bool streamed_occupied = plan->streamed && (automatic || explicit_occupied);
  if (plan->streamed && !streamed_occupied) return fallback("streamed");
  if (plan->streamed && !plan->integral_source) return fallback("streamed_without_source");
  if (!plan->streamed && plan->integral_source && !packed && !source_dense_resident)
    return fallback("source_without_complete_resident_values");
  if (!plan->streamed && plan->row_tile != plan->nbf) return fallback("partial_ao_rows");
  if (!plan->streamed && !packed && plan->auxiliary_tile != plan->naux)
    return fallback("partial_auxiliary");
  if (plan->nbf < 2) return fallback("trivial_dimension");
  auto* state = static_cast<PersistentScfState*>(plan->persistent_scf_state);
  if (!state) return fallback("missing_persistent_state");
  if (state->unrestricted) return fallback("unrestricted");
  if (!state->occupied_exchange) return fallback("occupied_exchange_disabled");
  if (density.size() != plan->matrix_elements || !finite_values(density))
    return fallback("invalid_density");
  if (automatic || streamed_occupied) {
    if (state->final_alpha_occupied.size() != 1 || !state->final_beta_occupied.empty() ||
        state->final_alpha_occupied[0] <= 0)
      return fallback("invalid_final_occupation");
    // The streamed projection is private to this K build. Exact final C/D
    // identity is checked below, but it grants no response projection lease.
    if (streamed_occupied ? !qualified_value_rhf_exchange(*plan, state->final_alpha_occupied[0])
                          : !qualified_resident_rhf_exchange(*plan, state->final_alpha_occupied[0]))
      return fallback("work_or_capacity_policy");
  }
  CudaDfFinalStateToken current;
  auto status = cuda_density_fitting_final_state_token(plan, 0, current, detail);
  if (status != VIBEQC_STATUS_SUCCESS && status != VIBEQC_STATUS_INVALID_ARGUMENT) return status;
  const bool exact_retained = status == VIBEQC_STATUS_SUCCESS && expected == current;
  const bool corrected_streamed = status == VIBEQC_STATUS_SUCCESS && !exact_retained &&
                                  streamed_occupied &&
                                  bounded_corrected_final_rhf_identity(current, expected);
  if (!exact_retained && !corrected_streamed) {
    detail.clear();
    return fallback("stale_final_state_token");
  }
  const auto bytes = plan->matrix_elements * sizeof(double);
  if (corrected_streamed) {
    TraceOperation trace(
        "final_state_corrected_jk", plan->stream,
        {1, plan->nbf, plan->naux, plan->integral_source != nullptr, plan->streamed});
    auto error = cudaSetDevice(plan->device_id);
    if (error != cudaSuccess) return cuda_failure(error, "select corrected final device", detail);
    // Early failures must release the caller's density and the J/K stream lease
    // only after outstanding uploads and partial contractions have drained.
    struct CorrectedFinalDrain {
      cudaStream_t stream;
      bool active{true};
      ~CorrectedFinalDrain() {
        if (active) (void)cudaStreamSynchronize(stream);
      }
    } corrected_drain{plan->stream};
    error = cudaMemcpyAsync(plan->primary_density, density.data(), bytes, cudaMemcpyHostToDevice,
                            plan->stream);
    // This algebraic factor represents the corrected D, not the preceding
    // canonical SCF frame. Revoke that generation before overwriting scratch.
    if (error == cudaSuccess)
      error =
          cudaMemsetAsync(state->d_alpha_factor_generation, 0, sizeof(std::uint32_t), plan->stream);
    if (error != cudaSuccess)
      return cuda_failure(error, "upload corrected final density and revoke SCF factor", detail);
    trace_counter("h2d_bytes", bytes);

    bool accepted = false;
    std::size_t rank = 0;
    status =
        factor_density_for_exchange(*plan, *state, plan->primary_density, accepted, rank, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    if (!accepted || rank != expected.identity.occupied[0]) {
      trace_counter("factor_rejected", 1);
      detail.clear();
      return fallback("corrected_factor_rejected");
    }

    coulomb.resize(download ? plan->matrix_elements : 0);
    exchange.resize(download ? plan->matrix_elements : 0);
    status = build_coulomb(*plan, plan->primary_density, detail);
    if (status == VIBEQC_STATUS_SUCCESS)
      // factor_density_for_exchange reconstructs D = L L^T, so occupation is
      // already absorbed. Canonical RHF C uses weight two; algebraic L uses one.
      status = build_occupied_exchange(*plan, 0, state->d_alpha_factor, rank, true, 1,
                                       plan->alpha_exchange, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    if (download) {
      error = cudaMemcpyAsync(coulomb.data(), plan->coulomb, bytes, cudaMemcpyDeviceToHost,
                              plan->stream);
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(exchange.data(), plan->alpha_exchange, bytes,
                                cudaMemcpyDeviceToHost, plan->stream);
    }
    const auto finished = cudaStreamSynchronize(plan->stream);
    if (error == cudaSuccess) error = finished;
    if (error != cudaSuccess)
      return cuda_failure(error, "download corrected final occupied J/K", detail);
    corrected_drain.active = false;  // The successful final drain already completed.
    trace_counter("d2h_bytes", download ? 2 * bytes : 0);
    trace_counter("explicit_synchronizations", 1);
    trace_counter("correction_generations", expected.identity.factor.density_generation -
                                                current.identity.factor.density_generation);
    trace_counter("accepted", 1);
    runtime::df_progress::label("final_exchange_fallback", "none");
    used = true;
    return VIBEQC_STATUS_SUCCESS;
  }

  TraceOperation trace(
      "final_state_retained_jk", plan->stream,
      {1, plan->nbf, plan->naux, plan->integral_source != nullptr, plan->streamed});
  auto error = cudaSetDevice(plan->device_id);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(plan->primary_density, density.data(), bytes, cudaMemcpyHostToDevice,
                            plan->stream);
  if (error != cudaSuccess)
    return cuda_failure(error, "upload final density for identity check", detail);
  launch_density_exchange_error(plan->stream, plan->matrix_elements, plan->primary_density,
                                state->d_density, state->d_next_density);
  double errors[2]{};
  std::uint64_t generation = 0;
  int info = 0;
  error = cudaMemcpyAsync(errors, state->d_next_density, sizeof(errors), cudaMemcpyDeviceToHost,
                          plan->stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&generation, state->d_final_alpha_generation, sizeof(generation),
                            cudaMemcpyDeviceToHost, plan->stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&info, state->d_final_alpha_info, sizeof(info), cudaMemcpyDeviceToHost,
                            plan->stream);
  const auto drained = cudaStreamSynchronize(plan->stream);
  if (error == cudaSuccess) error = drained;
  if (error != cudaSuccess) return cuda_failure(error, "validate final retained density", detail);
  trace_counter("h2d_bytes", bytes);
  trace_counter("d2h_bytes", sizeof(errors) + sizeof(generation) + sizeof(info));
  trace_counter("explicit_synchronizations", 1);
  if (errors[0] != 0 || errors[1] != 0 || info ||
      generation != current.identity.factor.density_generation) {
    trace_counter("identity_rejected", 1);
    return fallback("density_or_generation_mismatch");
  }
  // Both D and the full retained C were committed under this generation. The
  // strict selector still checks the resulting physical F[D] and eigenframe;
  // if it changes D or generation, this exact identity gate refuses reuse.
  coulomb.resize(download ? plan->matrix_elements : 0);
  exchange.resize(download ? plan->matrix_elements : 0);
  status = build_coulomb(*plan, state->d_density, detail);
  if (status == VIBEQC_STATUS_SUCCESS)
    status = build_occupied_exchange(*plan, 0, state->d_final_alpha_coefficients,
                                     current.identity.occupied[0], true, 2, plan->alpha_exchange,
                                     detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (download) {
    error =
        cudaMemcpyAsync(coulomb.data(), plan->coulomb, bytes, cudaMemcpyDeviceToHost, plan->stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(exchange.data(), plan->alpha_exchange, bytes, cudaMemcpyDeviceToHost,
                              plan->stream);
  }
  const auto finished = cudaStreamSynchronize(plan->stream);
  if (error == cudaSuccess) error = finished;
  if (error != cudaSuccess) return cuda_failure(error, "download final retained J/K", detail);
  trace_counter("d2h_bytes", download ? 2 * bytes : 0);
  trace_counter("explicit_synchronizations", 1);
  // The projection and final coefficients refer to precisely this density
  // generation. Publishing after the successful drain excludes partial K.
  if (!plan->streamed && plan->resident_exchange_enabled && (plan->resident_raw_valid || packed) &&
      (!packed || current.identity.occupied[0] <= plan->value_storage.rank_capacity) &&
      plan->metric_full_rank[0] && current.identity.occupied[0] &&
      plan->naux * current.identity.occupied[0] <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
    plan->final_projection_token = current;
  trace_counter("accepted", 1);
  runtime::df_progress::label("final_exchange_fallback", "none");
  used = true;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status read_cuda_density_fitting_final_state(CudaDensityFittingJkPlan* plan,
                                                    const CudaDfFinalStateToken& expected,
                                                    CudaDfFinalStateSnapshot& snapshot,
                                                    std::string& detail, bool include_density) {
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
    if (include_density) local.density.resize(spins);
    // Allocate all pageable destinations before the first asynchronous copy,
    // then always drain the stream before any local storage can be released.
    for (std::size_t spin = 0; spin < spins; ++spin) {
      local.candidate.spins[spin].vectors.resize(count);
      local.candidate.spins[spin].values.resize(n);
      if (include_density) local.density[spin].resize(count);
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
      if (include_density) {
        copy(local.density[spin].data(),
             (spin                 ? state.d_beta_density
              : state.unrestricted ? state.d_alpha_density
                                   : state.d_density) +
                 item * count,
             count * sizeof(double));
      }
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
          (include_density && !finite_values(local.density[spin]))) {
        detail = "CUDA DF retained frame has stale generation, solver failure or nonfinite data";
        return VIBEQC_STATUS_NUMERICAL_FAILURE;
      }
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = row + 1; column < n; ++column)
          std::swap(frame.vectors[row * n + column], frame.vectors[column * n + row]);
    }
    runtime::cuda_trace::trace_counter(
        "final_state_download_bytes",
        spins * (((include_density ? 2 : 1) * count + n) * sizeof(double) + sizeof(std::uint64_t) +
                 sizeof(int)));
    snapshot = std::move(local);
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "host allocation for detached CUDA DF final frame failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
}
}  // namespace vibeqc::scf

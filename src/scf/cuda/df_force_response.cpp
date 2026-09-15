#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "molecule/basis.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"
#include "scf/cuda_df_gradient.hpp"
#include "scf/density_fitting.hpp"

namespace vibeqc::scf {
using namespace cuda_df;

namespace {
/** Validate a method-authorized canonical density before lending SCF factors.
 * Tokens bind owner/geometry, epoch, system, model and occupations. Exact
 * comparison with the actual device density excludes corrected determinants;
 * dimension equality or a nearby physical residual cannot authorize reuse.
 */
vibeqc_status select_occupied_response_factors(CudaDensityFittingJkPlan& plan, std::size_t system,
                                               const CudaDfFinalStateToken* requested,
                                               std::span<const DensityFittingDensityResponse> terms,
                                               std::size_t maximum_bytes,
                                               CudaDfResponseBuffers& buffers,
                                               std::string& detail) {
  auto* state = static_cast<PersistentScfState*>(plan.persistent_scf_state);
  if (!requested || !state || !state->occupied_exchange || !plan.occupied_scf_reserved ||
      !state->final_frames_available)
    return VIBEQC_STATUS_SUCCESS;
  CudaDfFinalStateToken current;
  const auto status = cuda_density_fitting_final_state_token(&plan, system, current, detail);
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) return status;
  if (status != VIBEQC_STATUS_SUCCESS || current != *requested) {
    detail.clear();
    return VIBEQC_STATUS_SUCCESS;
  }
  const auto spins = state->unrestricted ? 2U : 1U, first = state->unrestricted ? 1U : 0U;
  const auto matrix = plan.nbf * plan.nbf;
  if (terms.size() != (state->unrestricted ? 3U : 1U) || terms[0].density.size() != matrix ||
      terms[0].coulomb_coefficient != 1.0 ||
      (state->unrestricted && terms[0].exchange_coefficient != 0.0) ||
      spins * matrix * sizeof(double) > maximum_bytes)
    return VIBEQC_STATUS_SUCCESS;
  const auto rank = [&](unsigned spin) {
    return static_cast<std::size_t>(spin ? state->factor_beta_ranks[system]
                                         : state->factor_alpha_ranks[system]);
  };
  std::size_t projected = 0;
  for (unsigned spin = 0; spin < spins; ++spin) {
    const auto t = first + spin;
    const auto r = rank(spin);
    if (r > plan.nbf || terms[t].density.size() != matrix ||
        terms[t].exchange_coefficient != (state->unrestricted ? .5 : .25) ||
        (state->unrestricted && terms[t].coulomb_coefficient != 0.0) ||
        current.identity.occupied[spin] != r)
      return VIBEQC_STATUS_SUCCESS;
    projected += r * r;
  }
  // Both spin projections coexist in one existing tensor. Saturated UHF
  // ranks that exceed this capacity keep the dense reference route.
  if (projected > buffers.elements_per_buffer / plan.naux) return VIBEQC_STATUS_SUCCESS;
  try {
    std::vector<double> canonical(spins * matrix);
    std::uint32_t generations[2]{};
    auto error = cudaSetDevice(plan.device_id);
    runtime::cuda_trace::TraceOperation trace("occupied_response_provenance", plan.stream,
                                              {1, plan.nbf, plan.naux, false, false, system});
    const auto copy = [&](void* output, const void* input, std::size_t bytes) {
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToHost, plan.stream);
    };
    for (unsigned spin = 0; spin < spins; ++spin) {
      copy(canonical.data() + spin * matrix,
           (spin                  ? state->d_beta_density
            : state->unrestricted ? state->d_alpha_density
                                  : state->d_density) +
               system * matrix,
           matrix * sizeof(double));
      copy(&generations[spin],
           (spin ? state->d_beta_factor_generation : state->d_alpha_factor_generation) + system,
           sizeof(std::uint32_t));
    }
    const auto drained = cudaStreamSynchronize(plan.stream);
    if (error == cudaSuccess) error = drained;
    if (error != cudaSuccess) return cuda_failure(error, "validate DF response factors", detail);
    runtime::cuda_trace::trace_counter("factor_validation_download_bytes",
                                       spins * (matrix * sizeof(double) + sizeof(std::uint32_t)));
    for (unsigned spin = 0; spin < spins; ++spin) {
      // SCF factor generations count committed iterations from zero; the
      // final-state protocol counts the imported density as generation one.
      if (generations[spin] != state->final_iterations[system] ||
          static_cast<std::uint64_t>(generations[spin]) + 1 !=
              current.identity.factor.density_generation ||
          !std::equal(terms[first + spin].density.begin(), terms[first + spin].density.end(),
                      canonical.begin() + spin * matrix))
        return VIBEQC_STATUS_SUCCESS;
    }
    if (state->unrestricted)
      for (std::size_t k = 0; k < matrix; ++k)
        if (terms[0].density[k] != canonical[k] + canonical[matrix + k])
          return VIBEQC_STATUS_SUCCESS;
    for (unsigned spin = 0; spin < spins; ++spin) {
      const auto r = rank(spin);
      const auto* coefficients = spin ? state->d_beta_factor : state->d_alpha_factor;
      if (r && !coefficients) return VIBEQC_STATUS_SUCCESS;
      buffers.occupied_factors[first + spin] = {
          coefficients
              ? coefficients +
                    system * plan.nbf * (spin ? state->beta_factor_rank : state->alpha_factor_rank)
              : nullptr,
          r, state->unrestricted ? 1.0 : 2.0};
    }
    buffers.occupied_response = true;
    runtime::cuda_trace::trace_counter("validated_density_generations", spins);
    runtime::cuda_trace::trace_counter("density_generation",
                                       current.identity.factor.density_generation);
    runtime::cuda_trace::trace_counter("solve_epoch", current.identity.solve_epoch);
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "DF response factor validation exceeded host storage";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
}
}  // namespace

// Both value providers borrow forward device factors and bounded bridge
// scratch. The explicit host adapter remains available for diagnostic ablation.
vibeqc_status execute_cuda_density_fitting_generated_force_response(
    CudaDensityFittingJkPlan* plan, std::size_t system, const core::System& orbital,
    const core::System& auxiliary, std::span<const double> raw_a,
    const std::vector<double>& raw_metric, std::span<const DensityFittingDensityResponse> terms,
    unsigned schedule, std::size_t maximum_bytes, std::size_t maximum_auxiliary_tile,
    std::vector<double>& derivative, std::string& detail, DfGradientResources* resources,
    const CudaDfFinalStateToken* final_state) {
  if (resources) *resources = {};
  if (!plan || system >= plan->batch_size || molecule::ao_count(orbital) != plan->nbf ||
      molecule::ao_count(auxiliary) != plan->naux) {
    detail = "invalid generated DF force plan or batch index";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto elements = plan->naux * plan->naux, offset = system * elements;
  const char* host_policy = std::getenv("VIBEQC_DF_HOST_RESPONSE_WEIGHTS");
  const bool host_weights = host_policy && host_policy[0] == '1' && host_policy[1] == '\0';
  const char* storage_control = std::getenv("VIBEQC_DF_RESPONSE_STORAGE");
  const std::string_view storage = storage_control ? storage_control : "auto";
  if (storage != "auto" && storage != "panel" && storage != "jk-scratch") {
    detail = "unknown DF response storage (use auto, panel or jk-scratch)";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  // Retained B alone does not establish scratch capacity: generated resident
  // plans can retain B while their J/K temporaries cover only a small tile.
  const bool full_scratch = !host_weights && !plan->integral_source && !plan->streamed &&
                            plan->row_tile == plan->nbf && plan->auxiliary_tile == plan->naux &&
                            plan->auxiliary_tile_values && plan->exchange_intermediate &&
                            plan->exchange_contributions;
  const char* space_control = std::getenv("VIBEQC_DF_RESPONSE_SPACE");
  const std::string_view space = space_control ? space_control : "auto";
  if (space != "auto" && space != "dense" && space != "occupied") {
    detail = "unknown DF response space (use auto, dense or occupied)";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  bool borrow =
      storage == "jk-scratch" || (space == "occupied" && full_scratch && storage != "panel");
  bool automatic_occupied = false;
  // An explicit occupied request already chose compatible borrowed storage;
  // automatic device filtering must not replace that comparison override.
  if (space != "occupied" && storage == "auto" && full_scratch && schedule == 0 &&
      plan->nbf == 768 && plan->naux == 768 && plan->batch_size == 1 && terms.size() == 1) {
    // Promote only the measured large RHF endpoint. Explicit comparison
    // schedules and attribution probes keep their panel execution; other
    // shapes/backends remain available through the checked opt-in selector.
    const auto compatible = [](const char* name, std::string_view expected) {
      const char* value = std::getenv(name);
      return !value || std::string_view(value) == expected;
    };
    const auto absent = [](const char* name) {
      const char* value = std::getenv(name);
      return !value || !*value;
    };
    const char* serial = std::getenv("VIBEQC_DF_SERIAL_RESPONSE_DOT");
    if (compatible("VIBEQC_DF_WEIGHTED_EXECUTION", "shell") &&
        compatible("VIBEQC_DF_SHELL_SCHEDULE", "compact") &&
        compatible("VIBEQC_DF_RESPONSE_ALGEBRA", "blas") &&
        absent("VIBEQC_DF_RESPONSE_UPLOAD_PROBE") && absent("VIBEQC_DF_RESPONSE_SCATTER_PROBE") &&
        !(serial && std::string_view(serial) == "1")) {
      cudaDeviceProp properties{};
      const auto error = cudaGetDeviceProperties(&properties, plan->device_id);
      if (error != cudaSuccess) return cuda_failure(error, "DF response device properties", detail);
      borrow = properties.major == 12 && properties.minor == 0;
      // The low-rank endpoint is qualified only for this exact RHF rank and
      // device. The token is only a selection hint here; full owner, model,
      // density and device-generation validation below authorizes execution.
      automatic_occupied =
          borrow && std::string_view(properties.name) == "NVIDIA GeForce RTX 5090" && final_state &&
          final_state->identity.occupied.size() == 1 && final_state->identity.occupied[0] == 160;
    }
  }
  CudaDfResponseBuffers buffers;
  if (borrow) {
    // Dense resident plans reserved three full tensor temporaries for J/K.
    // Force executes after SCF on this same stream, and lends them back before
    // the next replay. Never infer capacity from retained B alone: generated
    // resident plans can retain B while their K scratch is only a small tile.
    if (!full_scratch) {
      detail = "JK-scratch response requires a resident host-raw plan with full J/K tensors";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    buffers = {plan->auxiliary_tile_values, plan->exchange_contributions,
               plan->exchange_intermediate, plan->tensor_elements_per_system};
  }
  if (borrow && (space == "occupied" || (space == "auto" && automatic_occupied))) {
    const auto selected = select_occupied_response_factors(*plan, system, final_state, terms,
                                                           maximum_bytes, buffers, detail);
    if (selected != VIBEQC_STATUS_SUCCESS) return selected;
  }
  if (plan->integral_source || !host_weights) {
    if (!plan->metric_response_valid[system]) {
      detail = "DF metric rank crossing: retained/discarded subspaces are unresolved";
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    const CudaDfMetricView metric{
        plan->inverse_square_roots + offset, plan->metric_eigenvectors + offset,
        plan->metric_eigenvalues + system * plan->naux, plan->metric_relative_threshold};
    return execute_cuda_df_hf_gradient(
        plan->device_id, reinterpret_cast<void*>(plan->stream), plan->integral_source, system,
        orbital, auxiliary, raw_a, {}, {}, terms, plan->metric_relative_threshold, schedule,
        maximum_bytes, maximum_auxiliary_tile, derivative, detail, resources, &metric,
        reinterpret_cast<void*>(plan->blas), borrow ? &buffers : nullptr);
  }
  // Copies isolate one system's spectral reverse map from the packed batch.
  // Charge them while the bounded HF/derivative bridge is also alive.
  const auto copies_bytes = 2 * elements * sizeof(double);
  if (maximum_bytes <= copies_bytes || 12.0L * elements * sizeof(double) > maximum_bytes) {
    detail = "generated DF metric reverse staging exceeds maximum_bytes";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  try {
    std::vector<double> metric, inverse;
    if (raw_metric.size() != elements) {
      detail = "generated resident DF response needs its original metric";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    metric = raw_metric;
    // The resident value path already stages raw A/M on the host. Reuse
    // that same Hamiltonian and cutoff, without inventing derivative arrays
    // to satisfy the legacy complete-tensor oracle interface.
    const auto factor =
        factor_density_fitting_metric(metric, plan->naux, plan->metric_relative_threshold);
    inverse.assign(elements, 0.0);
    for (std::size_t i = 0; i < plan->naux; ++i)
      for (std::size_t j = 0; j < plan->naux; ++j)
        for (std::size_t k = 0; k < plan->naux; ++k)
          inverse[i * plan->naux + j] += factor.inverse_square_root[i * plan->naux + k] *
                                         factor.inverse_square_root[j * plan->naux + k];
    DfGradientResources measured;
    const auto status = execute_cuda_df_hf_gradient(
        plan->device_id, reinterpret_cast<void*>(plan->stream), plan->integral_source, system,
        orbital, auxiliary, raw_a, metric, inverse, terms, plan->metric_relative_threshold,
        schedule, maximum_bytes - copies_bytes, maximum_auxiliary_tile, derivative, detail,
        &measured);
    if (status == VIBEQC_STATUS_SUCCESS && resources) {
      measured.host_bytes += copies_bytes;
      *resources = measured;
    }
    return status;
  } catch (const std::bad_alloc&) {
    detail = "generated DF metric reverse staging exceeded its allocation budget";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::exception& error) {
    detail = error.what();
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
}

}  // namespace vibeqc::scf

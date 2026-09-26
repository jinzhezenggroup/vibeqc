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
#include "scf/cuda/df_scf_factor.hpp"
#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"
#include "scf/cuda_df_gradient.hpp"
#include "scf/density_fitting.hpp"

namespace vibeqc::scf {
using namespace cuda_df;

void bind_cuda_density_fitting_response_source(CudaDensityFittingJkPlan* plan,
                                               const core::System& orbital,
                                               const core::System& auxiliary,
                                               std::span<const double> raw) noexcept {
  if (!plan || plan->response_host_raw || !plan->resident_raw_valid || plan->batch_size != 1 ||
      raw.size() != plan->tensor_elements_per_system)
    return;
  plan->response_host_raw = raw.data();
  plan->response_orbital_atoms = orbital.atoms.data();
  plan->response_auxiliary_atoms = auxiliary.atoms.data();
  plan->response_orbital_shells = orbital.shells.data();
  plan->response_auxiliary_shells = auxiliary.shells.data();
  plan->response_orbital_representation = orbital.basis_representation;
  plan->response_auxiliary_representation = auxiliary.basis_representation;
}

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
                                               CudaDfOccupiedResponseView& view,
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
  for (unsigned spin = 0; spin < spins; ++spin) {
    const auto t = first + spin;
    const auto r = rank(spin);
    if (r > plan.nbf || terms[t].density.size() != matrix ||
        terms[t].exchange_coefficient != (state->unrestricted ? .5 : .25) ||
        (state->unrestricted && terms[t].coulomb_coefficient != 0.0) ||
        current.identity.occupied[spin] != r)
      return VIBEQC_STATUS_SUCCESS;
  }
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
      view.factors[first + spin] = {
          coefficients
              ? coefficients +
                    system * plan.nbf * (spin ? state->beta_factor_rank : state->alpha_factor_rank)
              : nullptr,
          r, state->unrestricted ? 1.0 : 2.0};
    }
    view.nbf = plan.nbf;
    view.naux = plan.naux;
    view.owner_identity = plan.factor_basis_identity;
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

/** A strict final-state correction has no SCF factor of its own. Reconstruct
 * its exact density independently before lending the value plan's reserved
 * factor scratch; revoke the previous SCF generation before overwriting it.
 */
vibeqc_status select_corrected_occupied_response_factor(
    CudaDensityFittingJkPlan& plan, std::size_t system, const CudaDfFinalStateToken* requested,
    std::span<const DensityFittingDensityResponse> terms, std::size_t maximum_bytes,
    CudaDfOccupiedResponseView& view, std::string& detail) {
  auto* state = static_cast<PersistentScfState*>(plan.persistent_scf_state);
  if (!requested || !state || state->unrestricted || !state->occupied_exchange ||
      !state->final_frames_available || !plan.streamed || !plan.integral_source ||
      plan.batch_size != 1 || system != 0 || terms.size() != 1 ||
      terms[0].density.size() != plan.matrix_elements || terms[0].coulomb_coefficient != 1.0 ||
      terms[0].exchange_coefficient != .25 || requested->identity.occupied.size() != 1 ||
      !requested->identity.occupied[0] ||
      !qualified_value_rhf_exchange(plan, requested->identity.occupied[0]))
    return VIBEQC_STATUS_SUCCESS;
  CudaDfFinalStateToken original;
  const auto token_status = cuda_density_fitting_final_state_token(&plan, system, original, detail);
  if (token_status == VIBEQC_STATUS_OUT_OF_MEMORY) return token_status;
  if (token_status != VIBEQC_STATUS_SUCCESS) {
    detail.clear();
    return VIBEQC_STATUS_SUCCESS;
  }
  const auto& current = original.identity;
  const auto& corrected = requested->identity;
  const auto density_generation = corrected.factor.density_generation;
  const auto orbital_generation = corrected.factor.orbital_generation;
  if (requested->version != original.version || corrected.factor.basis != current.factor.basis ||
      corrected.factor.reference != current.factor.reference ||
      corrected.solve_epoch != current.solve_epoch || corrected.model != current.model ||
      corrected.occupied != current.occupied ||
      density_generation <= current.factor.density_generation ||
      orbital_generation <= current.factor.orbital_generation ||
      density_generation - current.factor.density_generation > 16 ||
      density_generation - current.factor.density_generation !=
          orbital_generation - current.factor.orbital_generation ||
      !std::all_of(terms[0].density.begin(), terms[0].density.end(),
                   [](double value) { return std::isfinite(value); }))
    return VIBEQC_STATUS_SUCCESS;
  runtime::cuda_trace::TraceOperation trace("corrected_response_factor", plan.stream,
                                            {1, plan.nbf, plan.naux, true, true, system});
  const auto bytes = plan.matrix_elements * sizeof(double);
  auto error = cudaSetDevice(plan.device_id);
  if (error != cudaSuccess) return cuda_failure(error, "select corrected response device", detail);
  // A later enqueue, eigensolver or host allocation may fail after the H2D
  // upload borrows terms[0].density. Drain before returning/rethrowing so the
  // caller can release its density even when no response bridge is entered.
  struct UploadDrain {
    cudaStream_t stream;
    bool active{true};
    ~UploadDrain() {
      if (active) (void)cudaStreamSynchronize(stream);
    }
  } upload_drain{plan.stream};
  error = cudaMemcpyAsync(plan.primary_density, terms[0].density.data(), bytes,
                          cudaMemcpyHostToDevice, plan.stream);
  if (error == cudaSuccess)
    error =
        cudaMemsetAsync(state->d_alpha_factor_generation, 0, sizeof(std::uint32_t), plan.stream);
  if (error != cudaSuccess)
    return cuda_failure(error, "upload corrected response density and revoke SCF factor", detail);
  bool accepted = false;
  std::size_t rank = 0;
  const auto status =
      factor_density_for_exchange(plan, *state, plan.primary_density, accepted, rank, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (!accepted || rank != corrected.occupied[0] ||
      rank * rank > maximum_bytes / plan.naux / view.factors.size()) {
    runtime::cuda_trace::trace_counter("reconstruction_rejected", 1);
    return VIBEQC_STATUS_SUCCESS;
  }
  // Accepted reconstruction has already drained its spectrum and full-density
  // checks. Do not add a synchronization to the successful occupied path.
  upload_drain.active = false;
  view.factors[0] = {state->d_alpha_factor, rank, 1.0};
  view.nbf = plan.nbf;
  view.naux = plan.naux;
  view.owner_identity = plan.factor_basis_identity;
  runtime::cuda_trace::trace_counter("accepted", 1);
  runtime::cuda_trace::trace_counter("correction_generations",
                                     density_generation - current.factor.density_generation);
  runtime::cuda_trace::trace_counter("borrowed_factor_bytes", plan.nbf * rank * sizeof(double));
  return VIBEQC_STATUS_SUCCESS;
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
  if (plan->integral_source && !cuda_density_fitting_integral_source_geometry_matches(
                                   plan->integral_source, system, orbital, auxiliary)) {
    detail = "generated DF response source geometry or basis does not match";
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
  // Prepared generated-source metadata deliberately releases host A. In that
  // case the integral source itself is the immutable geometry/metric owner;
  // an empty raw span is therefore valid only after the exact per-item check
  // above; it is not independently a geometry identity. This is not a
  // missing owner. Materialized host routes still require every allocation
  // identity below to match exactly.
  const bool matching_source =
      (plan->integral_source && !plan->response_host_raw && raw_a.empty()) ||
      (plan->response_host_raw && plan->response_host_raw == raw_a.data() &&
       plan->response_orbital_atoms == orbital.atoms.data() &&
       plan->response_auxiliary_atoms == auxiliary.atoms.data() &&
       plan->response_orbital_shells == orbital.shells.data() &&
       plan->response_auxiliary_shells == auxiliary.shells.data() &&
       plan->response_orbital_representation == orbital.basis_representation &&
       plan->response_auxiliary_representation == auxiliary.basis_representation);
  const bool source_dense_resident =
      plan->integral_source && plan->value_storage.pairs == DfPairStorage::Dense && matching_source;
  // Retained B alone does not establish scratch capacity: generated resident
  // plans can retain B while their J/K temporaries cover only a small tile.
  const bool full_scratch =
      !host_weights && !plan->streamed && (!plan->integral_source || source_dense_resident) &&
      plan->row_tile == plan->nbf && plan->auxiliary_tile == plan->naux &&
      plan->auxiliary_tile_values && plan->exchange_intermediate && plan->exchange_contributions;
  const bool packed_resident = !host_weights && plan->integral_source && !plan->streamed &&
                               plan->value_storage.pairs == DfPairStorage::SymmetricLower &&
                               plan->packed_raw && plan->row_tile == plan->nbf;
  const char* space_control = std::getenv("VIBEQC_DF_RESPONSE_SPACE");
  const std::string_view space = space_control ? space_control : "auto";
  if (space != "auto" && space != "dense" && space != "occupied") {
    detail = "unknown DF response space (use auto, dense or occupied)";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  bool borrow =
      storage == "jk-scratch" || (space == "occupied" && full_scratch && storage != "panel");
  bool automatic_occupied = false;
  if (packed_resident && storage != "panel" && space == "occupied") borrow = true;
  // An explicit occupied request already chose compatible borrowed storage;
  // automatic work selection must not replace that comparison override.
  if (space != "occupied" && storage != "panel" && (full_scratch || packed_resident) &&
      schedule == 0 && plan->batch_size == 1 && terms.size() == 1) {
    // Share SCF's work/capacity policy. The token is only a selection hint:
    // exact owner, model, density and device generations are validated below.
    // Diagnostic schedules and attribution probes retain their panel path.
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
        (compatible("VIBEQC_DF_SHELL_SCHEDULE", "compact") ||
         compatible("VIBEQC_DF_SHELL_SCHEDULE", "auto")) &&
        compatible("VIBEQC_DF_RESPONSE_ALGEBRA", "blas") &&
        absent("VIBEQC_DF_RESPONSE_UPLOAD_PROBE") && absent("VIBEQC_DF_RESPONSE_SCATTER_PROBE") &&
        !(serial && std::string_view(serial) == "1")) {
      automatic_occupied =
          space == "auto" && final_state && final_state->identity.occupied.size() == 1 &&
          qualified_resident_rhf_exchange(*plan, final_state->identity.occupied[0]);
      // Dense response can also reuse existing full J/K storage: projecting
      // each Q once avoids repeating work across response panels. Its storage
      // policy needs no occupied reservation or factor token. Keep this path
      // when occupied factors are unavailable, without any new allocation or
      // inferring full capacity from retained B alone.
      if ((full_scratch && storage == "auto" &&
           (!plan->integral_source || plan->resident_raw_valid)) ||
          automatic_occupied)
        borrow = true;
    }
  }
  CudaDfResponseBuffers buffers;
  CudaDfPackedRawTensorView packed_raw;
  const char* raw_reuse_control = std::getenv("VIBEQC_DF_RAW_REUSE");
  if (packed_resident && (!raw_reuse_control || std::string_view(raw_reuse_control) == "auto")) {
    packed_raw = {plan->packed_raw + system * plan->stored_tensor_elements_per_system,
                  plan->nbf,
                  plan->naux,
                  plan->stored_pair_count,
                  plan->factor_basis_identity,
                  {plan->inverse_square_roots + offset, plan->metric_eigenvectors + offset,
                   plan->metric_eigenvalues + system * plan->naux, plan->metric_relative_threshold,
                   plan->metric_full_rank[system] != 0, plan->factor_basis_identity}};
  }
  const auto enabled = [](const char* name) {
    const char* value = std::getenv(name);
    return !value || std::string_view(value) == "auto";
  };
  for (const char* name : {"VIBEQC_DF_RAW_REUSE", "VIBEQC_DF_RESPONSE_BATCHING"}) {
    const char* value = std::getenv(name);
    if (value && std::string_view(value) != "auto" && std::string_view(value) != "off") {
      detail = std::string(name) + " must be auto or off";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }
  if (borrow) {
    // Dense resident plans reserved three full tensor temporaries for J/K.
    // Force executes after SCF on this same stream, and lends them back before
    // the next replay. Never infer capacity from retained B alone: generated
    // resident plans can retain B while their K scratch is only a small tile.
    if (!full_scratch && !packed_resident) {
      detail = "JK-scratch response requires a resident plan with full J/K tensors";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    buffers = {plan->auxiliary_tile_values, plan->exchange_contributions,
               plan->exchange_intermediate, plan->tensor_elements_per_system};
    if (packed_resident) {
      buffers.resident_packed_raw = packed_raw;
      buffers.elements_per_buffer = 0;
      buffers.staging_elements = plan->projection_capacity;
      buffers.raw_elements = plan->panel_capacity;
      buffers.exchange_elements = plan->panel_capacity;
    }
    buffers.batch_products = enabled("VIBEQC_DF_RESPONSE_BATCHING");
    if (!packed_resident && plan->resident_raw_valid && matching_source &&
        enabled("VIBEQC_DF_RAW_REUSE")) {
      buffers.resident_raw = {
          plan->exchange_contributions,
          plan->nbf,
          plan->naux,
          plan->nbf * plan->nbf,
          plan->nbf,
          1,
          plan->factor_basis_identity,
          {plan->inverse_square_roots + offset, plan->metric_eigenvectors + offset,
           plan->metric_eigenvalues + system * plan->naux, plan->metric_relative_threshold,
           plan->metric_full_rank[system] != 0, plan->factor_basis_identity}};
    }
  }
  if (borrow && (space == "occupied" || (space == "auto" && automatic_occupied))) {
    CudaDfOccupiedResponseView factors;
    const auto selected = select_occupied_response_factors(*plan, system, final_state, terms,
                                                           maximum_bytes, factors, detail);
    if (selected != VIBEQC_STATUS_SUCCESS) return selected;
    // Validation authorizes immutable factors, not arbitrary mutable capacity.
    // Both spin projections must still fit the actual resident scratch lease.
    std::size_t projected = 0;
    bool fits = factors.owner_identity != 0;
    for (const auto& factor : factors.factors) {
      const auto rr = factor.rank * factor.rank;
      fits = fits && rr <= buffers.exchange_capacity() / plan->naux &&
             rr <= buffers.staging_capacity() / plan->naux - projected;
      if (!fits) break;
      projected += rr;
    }
    if (fits) {
      buffers.occupied_factors = factors.factors;
      buffers.occupied_response = true;
    }
  }
  // Packed scratch is not a dense all-Q allocation. Invalid/stale/corrected
  // factors retain the exact bounded raw loader instead of widening storage.
  if (packed_resident && (!buffers.occupied_response || !packed_raw.data)) borrow = false;
  const char* projection_control = std::getenv("VIBEQC_DF_FINAL_PROJECTION");
  const std::string_view projection = projection_control ? projection_control : "auto";
  if (projection != "auto" && projection != "off" && projection != "reuse") {
    detail = "VIBEQC_DF_FINAL_PROJECTION must be auto, off or reuse";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  // Automatic reuse shares the resident work/capacity policy above.
  // Full-rank M gives
  // G_raw = G_whitened M^(1/2); discarded directions cannot be recovered and
  // therefore keep the raw projection path, even under an explicit request.
  if ((projection == "reuse" || (projection == "auto" && automatic_occupied)) &&
      buffers.occupied_response && (buffers.resident_raw.data || packed_raw.data) &&
      plan->batch_size == 1 && terms.size() == 1 && final_state && plan->final_projection_token &&
      *plan->final_projection_token == *final_state && plan->metric_full_rank[0] &&
      plan->metric_response_valid[0]) {
    auto* state = static_cast<PersistentScfState*>(plan->persistent_scf_state);
    const auto rank = buffers.occupied_factors[0].rank;
    if (!state->unrestricted && rank && rank == final_state->identity.occupied[0] &&
        2 * rank * rank <= buffers.exchange_capacity() / plan->naux &&
        plan->nbf * rank <= buffers.staging_capacity() / plan->naux) {
      buffers.final_occupied_projection = plan->auxiliary_tile_values;
      buffers.occupied_factors[0].coefficients = state->d_final_alpha_coefficients;
    }
  }
  // A force attempt consumes the exclusive scratch lease. Repeated forces
  // without another final K, errors, and incompatible consumers all fall back.
  plan->final_projection_token.reset();
  if (plan->integral_source || !host_weights) {
    if (!plan->metric_response_valid[system]) {
      detail = "DF metric rank crossing: retained/discarded subspaces are unresolved";
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    const CudaDfMetricView metric{plan->inverse_square_roots + offset,
                                  plan->metric_eigenvectors + offset,
                                  plan->metric_eigenvalues + system * plan->naux,
                                  plan->metric_relative_threshold,
                                  plan->metric_full_rank[system] != 0,
                                  plan->factor_basis_identity};
    CudaDfWhitenedTensorView whitened;
    if (!plan->streamed && plan->three_center && metric.full_rank) {
      // SCF and force share the plan stream; this resident forward tensor is
      // immutable until the entire prepared geometry is replaced. Borrowing it
      // lets a smaller response panel avoid repeated raw integral generation.
      whitened = {plan->three_center + system * plan->stored_tensor_elements_per_system,
                  plan->nbf,
                  plan->naux,
                  plan->stored_pair_count,
                  plan->value_storage.pairs == DfPairStorage::SymmetricLower,
                  plan->factor_basis_identity,
                  metric};
    }
    CudaDfOccupiedResponseView owned_factors;
    if (!borrow && plan->integral_source && metric.full_rank && space != "dense" &&
        (plan->streamed || space == "occupied")) {
      // Canonical occupied factors are an immutable owner view, independent of
      // mutable J/K tensor storage. Retained-whitened and streamed source-backed
      // plans can lend C while the response bridge owns the bounded projection
      // and raw-slice scratch. Token, density and generation checks remain the
      // authority; a response-space label alone never establishes ownership.
      const auto selected = select_occupied_response_factors(*plan, system, final_state, terms,
                                                             maximum_bytes, owned_factors, detail);
      if (selected != VIBEQC_STATUS_SUCCESS) return selected;
      if (!owned_factors.owner_identity && plan->streamed && space == "occupied") {
        const auto corrected = select_corrected_occupied_response_factor(
            *plan, system, final_state, terms, maximum_bytes, owned_factors, detail);
        if (corrected != VIBEQC_STATUS_SUCCESS) return corrected;
      }
      if (owned_factors.owner_identity) {
        // The bridge's source-owned occupied path regenerates bounded raw A
        // panels. Do not simultaneously lend its mutually exclusive tensor
        // views. Retained plans keep their fitted-B route in automatic mode;
        // switching to source projection remains an explicit occupied request.
        whitened = {};
        packed_raw = {};
      }
    }
    // The diagnostic upload route writes the former raw scratch buffer.
    // Revoke its immutable view before submission, so an interrupted copy
    // cannot leave a previously valid cache available to the next force.
    if (borrow && !buffers.resident_raw.data && !packed_raw.data) plan->resident_raw_valid = false;
    const auto status = execute_cuda_df_hf_gradient(
        plan->device_id, reinterpret_cast<void*>(plan->stream), plan->integral_source, system,
        orbital, auxiliary, raw_a, {}, {}, terms, plan->metric_relative_threshold, schedule,
        maximum_bytes, maximum_auxiliary_tile, derivative, detail, resources, &metric,
        reinterpret_cast<void*>(plan->blas), borrow ? &buffers : nullptr,
        packed_raw.data ? &packed_raw : nullptr, whitened.data ? &whitened : nullptr,
        owned_factors.owner_identity ? &owned_factors : nullptr);
    if (status == VIBEQC_STATUS_SUCCESS && borrow && matching_source &&
        plan->resident_exchange_enabled && plan->batch_size == 1 &&
        plan->nbf * plan->naux <= static_cast<std::size_t>(std::numeric_limits<int>::max()))
      plan->resident_raw_valid = true;
    return status;
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

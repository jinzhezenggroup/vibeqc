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
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda_df_gradient.hpp"
#include "scf/density_fitting.hpp"

namespace vibeqc::scf {
using namespace cuda_df;

// Both value providers borrow forward device factors and bounded bridge
// scratch. The explicit host adapter remains available for diagnostic ablation.
vibeqc_status execute_cuda_density_fitting_generated_force_response(
    CudaDensityFittingJkPlan* plan, std::size_t system, const core::System& orbital,
    const core::System& auxiliary, std::span<const double> raw_a,
    const std::vector<double>& raw_metric, std::span<const DensityFittingDensityResponse> terms,
    unsigned schedule, std::size_t maximum_bytes, std::size_t maximum_auxiliary_tile,
    std::vector<double>& derivative, std::string& detail, DfGradientResources* resources) {
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
  bool borrow = storage == "jk-scratch";
  if (storage == "auto" && full_scratch && schedule == 0 && plan->nbf == 768 && plan->naux == 768 &&
      plan->batch_size == 1 && terms.size() == 1) {
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

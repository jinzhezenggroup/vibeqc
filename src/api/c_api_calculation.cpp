#include <algorithm>
#include <cstring>
#include <memory>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "api/ks_diagnostic.hpp"
#include "api/precision.hpp"
#include "methods/method.hpp"
#include "runtime/host_component_trace.hpp"
#include "vibeqc/vibeqc.h"

extern "C" {

vibeqc_status vibeqc_calculation_prepare(vibeqc_context* context, const vibeqc_system* system,
                                         const vibeqc_method_descriptor* descriptor,
                                         vibeqc_calculation** calculation) {
  vibeqc::runtime::host_trace::Region trace("calculation_prepare");
  if (context == nullptr || system == nullptr || descriptor == nullptr || calculation == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *calculation = nullptr;
  if (!vibeqc::api::valid_descriptor(descriptor)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
  try {
    auto candidate = std::make_unique<vibeqc_calculation>();
    candidate->context = context;
    candidate->plan =
        vibeqc::methods::prepare_calculation(context->state, system->data, *descriptor);
    *calculation = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}

void vibeqc_calculation_destroy(vibeqc_calculation* calculation) {
  vibeqc::runtime::host_trace::Region trace("calculation_destroy");
  delete calculation;
}

vibeqc_status vibeqc_calculation_execute(vibeqc_calculation* calculation,
                                         vibeqc_result_descriptor* output) {
  vibeqc::runtime::host_trace::Region trace("calculation_execute");
  if (calculation == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::recursive_mutex> context_lock(calculation->context->mutex);
  // An attempted execution revokes any internal final-state token even if
  // the output descriptor is rejected before the method can run.
  try {
    calculation->plan->invalidate_result();
  } catch (...) {
    return vibeqc::api::map_exception(&calculation->context->last_detail);
  }
  if (output == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(output)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
  const bool omit_forces = output->forces == nullptr && output->force_count == 0;
  if (output->forces == nullptr && !omit_forces) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!omit_forces && output->force_count < calculation->plan->atom_count() * 3) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  // Reset to the conservative FP64 record before the run so a failed or
  // fallback execution can never expose the previous successful mixed run.
  calculation->precision = {};
  calculation->precision_available = false;
  calculation->scf_diagnostic.reset();
  calculation->ks_diagnostic.reset();
  try {
    // NULL/zero is an execution request, not merely a copy-out choice: the
    // backend must not launch or assemble analytic-force work in this mode.
    vibeqc::methods::Result native = calculation->plan->execute(!omit_forces);
    // A normal return (converged or not) is a completed run: record what ran.
    calculation->precision = native.precision;
    calculation->precision_available = true;
    calculation->ks_diagnostic = std::move(native.ks_diagnostic);
    if (native.physical_residual_rms) {
      calculation->scf_diagnostic =
          vibeqc_scf_diagnostic{sizeof(vibeqc_scf_diagnostic), VIBEQC_ABI_VERSION,
                                native.convergence.residual_rms, *native.physical_residual_rms};
    }
    output->energy = native.energy;
    output->iterations = native.convergence.iterations;
    output->energy_change = native.convergence.energy_change;
    output->density_rms = native.convergence.residual_rms;
    output->converged = native.convergence.converged ? 1 : 0;
    output->executed_backend = native.executed_backend;
    if (!native.convergence.converged) {
      return VIBEQC_STATUS_NOT_CONVERGED;
    }
    if (!omit_forces) {
      if (native.forces.size() > output->force_count) {
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      std::copy(native.forces.begin(), native.forces.end(), output->forces);
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    calculation->plan->invalidate_result();
    return vibeqc::api::map_exception(&calculation->context->last_detail);
  }
}

vibeqc_status vibeqc_calculation_get_scf_diagnostic(const vibeqc_calculation* calculation,
                                                    vibeqc_scf_diagnostic* out) {
  if (!calculation) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (out && !vibeqc::api::valid_descriptor(out)) return VIBEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  if (!calculation->scf_diagnostic) return VIBEQC_STATUS_NOT_IMPLEMENTED;
  if (out) *out = *calculation->scf_diagnostic;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_calculation_get_ks_diagnostic(const vibeqc_calculation* calculation,
                                                   vibeqc_ks_diagnostic* out,
                                                   vibeqc_ks_iteration* history,
                                                   uint32_t history_capacity) {
  if (!calculation) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  return vibeqc::api::copy_ks_diagnostic(calculation->ks_diagnostic, out, history,
                                         history_capacity);
}

vibeqc_status vibeqc_calculation_get_ks_transport_diagnostic(const vibeqc_calculation* calculation,
                                                             vibeqc_ks_transport_diagnostic* out) {
  if (!calculation) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (out && !vibeqc::api::valid_descriptor(out)) return VIBEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  try {
    const auto source = calculation->plan->ks_transport_diagnostic();
    if (!source) return VIBEQC_STATUS_NOT_IMPLEMENTED;
    if (out)
      *out = {sizeof(*out),
              VIBEQC_ABI_VERSION,
              source->setup_h2d_bytes,
              source->density_h2d_bytes,
              source->scalar_d2h_bytes,
              source->matrix_d2h_bytes,
              source->synchronizations,
              source->iterations,
              source->occupation_stabilized_proposals};
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&calculation->context->last_detail);
  }
}

vibeqc_status vibeqc_calculation_get_precision_provenance(const vibeqc_calculation* calculation,
                                                          vibeqc_precision_provenance* out) {
  if (calculation == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  // A completed run (converged or not) populates \p precision and sets
  // \p precision_available; execute() resets it to false before the run so a
  // failed or not-yet-run execution never exposes a stale record. Gate both the
  // availability query (a NULL \p out) and the copy-out on it so callers see
  // an honest non-success result until a run has actually resolved.
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  if (!calculation->precision_available) {
    return VIBEQC_STATUS_PRECISION_UNAVAILABLE;
  }
  return vibeqc::api::copy_precision_provenance(calculation->precision, out);
}

vibeqc_status vibeqc_calculation_get_correlation_diagnostic(
    const vibeqc_calculation* calculation, vibeqc_correlation_diagnostic* diagnostic) {
  if (!calculation || !diagnostic) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(calculation->context->mutex);
  const auto caller_size = diagnostic->struct_size;
  constexpr auto legacy_size = offsetof(vibeqc_correlation_diagnostic, response_iterations);
  if (caller_size < legacy_size || diagnostic->abi_version != VIBEQC_ABI_VERSION)
    return VIBEQC_STATUS_ABI_MISMATCH;
  const auto value = calculation->plan->correlation_diagnostic();
  if (!value) return VIBEQC_STATUS_NOT_IMPLEMENTED;
  std::memcpy(diagnostic, &*value, std::min<std::size_t>(caller_size, sizeof(*diagnostic)));
  // Keep the caller's actual capacity. Publishing the producer's larger size
  // would make a repeated query overrun a legacy caller's unchanged buffer.
  diagnostic->struct_size = caller_size;
  return VIBEQC_STATUS_SUCCESS;
}

}  // extern "C"

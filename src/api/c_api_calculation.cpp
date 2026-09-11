#include <algorithm>
#include <memory>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "api/precision.hpp"
#include "methods/method.hpp"
#include "vibeqc/vibeqc.h"

extern "C" {

vibeqc_status vibeqc_calculation_prepare(vibeqc_context* context, const vibeqc_system* system,
                                         const vibeqc_method_descriptor* descriptor,
                                         vibeqc_calculation** calculation) {
  if (context == nullptr || system == nullptr || descriptor == nullptr || calculation == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *calculation = nullptr;
  if (!vibeqc::api::valid_method_descriptor(descriptor)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
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

void vibeqc_calculation_destroy(vibeqc_calculation* calculation) { delete calculation; }

vibeqc_status vibeqc_calculation_execute(vibeqc_calculation* calculation,
                                         vibeqc_result_descriptor* output) {
  if (calculation == nullptr || output == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
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
  try {
    // NULL/zero is an execution request, not merely a copy-out choice: the
    // backend must not launch or assemble analytic-force work in this mode.
    vibeqc::methods::Result native = calculation->plan->execute(!omit_forces);
    // A normal return (converged or not) is a completed run: record what ran.
    calculation->precision = native.precision;
    calculation->precision_available = true;
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
  if (!calculation->precision_available) {
    return VIBEQC_STATUS_PRECISION_UNAVAILABLE;
  }
  return vibeqc::api::copy_precision_provenance(calculation->precision, out);
}

}  // extern "C"

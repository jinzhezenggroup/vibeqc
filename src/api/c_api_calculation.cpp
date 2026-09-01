#include <algorithm>
#include <memory>

#include "api/error.hpp"
#include "api/handles.hpp"
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

  try {
    vibeqc::methods::Result native = calculation->plan->execute();
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

}  // extern "C"

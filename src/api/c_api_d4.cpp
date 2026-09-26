#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "dft/dispersion/d4_runtime.hpp"
#include "generated_d4_derivative.hpp"
#include "vibeqc/vibeqc.h"

struct vibeqc_d4_batch {
  vibeqc_context* context{};
  std::unique_ptr<vibeqc::dft::dispersion::D4Plan> plan;
};

namespace {

vibeqc_status public_status(vibeqc::dft::dispersion::D4Status status) {
  using vibeqc::dft::dispersion::D4Status;
  switch (status) {
    case D4Status::success:
      return VIBEQC_STATUS_SUCCESS;
    case D4Status::invalid_argument:
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    case D4Status::unsupported:
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    case D4Status::numerical_failure:
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  return VIBEQC_STATUS_INTERNAL_ERROR;
}

bool profile_from_public(vibeqc_d4_profile value, vibeqc::dft::dispersion::D4EEQProfile& profile) {
  using vibeqc::dft::dispersion::D4EEQProfile;
  if (value == VIBEQC_D4_PROFILE_STANDARD_EEQ) {
    profile = D4EEQProfile::standard;
    return true;
  }
  if (value == VIBEQC_D4_PROFILE_R2SCAN3C_EEQ) {
    profile = D4EEQProfile::r2scan3c;
    return true;
  }
  return false;
}

vibeqc_d4_profile public_profile(vibeqc::dft::dispersion::D4EEQProfile profile) {
  return profile == vibeqc::dft::dispersion::D4EEQProfile::r2scan3c
             ? VIBEQC_D4_PROFILE_R2SCAN3C_EEQ
             : VIBEQC_D4_PROFILE_STANDARD_EEQ;
}

}  // namespace

extern "C" {

const char* vibeqc_d4_table_sha256(void) { return vibeqc::dft::dispersion::kD4EEQTableSha256; }

const char* vibeqc_d4_charge_parameter_sha256(void) {
  return vibeqc::dft::dispersion::kD4EEQChargeParameterSha256;
}

const char* vibeqc_d4_derivative_identity(void) {
  return ::vibeqc::generated::d4::kDerivativeLoweringIdentity;
}

const char* vibeqc_d4_provider_identity(void) {
  return vibeqc::dft::dispersion::kD4ProductionProviderIdentity;
}

const char* vibeqc_d4_scheduler_identity(void) {
  return vibeqc::dft::dispersion::kD4ProductionSchedulerIdentity;
}

vibeqc_status vibeqc_d4_batch_prepare(vibeqc_context* context,
                                      const vibeqc_d4_system_descriptor* systems,
                                      uint32_t system_count,
                                      const vibeqc_d4_bj_eeq_descriptor* model,
                                      vibeqc_d4_batch** batch) {
  if (!context || !systems || !system_count || !model || !batch)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  *batch = nullptr;
  if (!vibeqc::api::valid_descriptor(model)) return VIBEQC_STATUS_ABI_MISMATCH;
  vibeqc::dft::dispersion::D4EEQProfile profile{};
  if (!profile_from_public(model->profile, profile)) {
    context->last_detail = "production D4 requires an explicit supported EEQ profile";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }

  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    if (model->maximum_bytes == 0) {
      context->last_detail = "D4 maximum_bytes must be positive";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto minimum_metadata = (std::uint64_t{system_count} + 1) * sizeof(std::uint32_t) +
                                  std::uint64_t{system_count} * sizeof(double);
    if (minimum_metadata > model->maximum_bytes) {
      context->last_detail = "D4 descriptor metadata exceeds maximum_bytes";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }

    std::uint64_t total_atoms = 0;
    for (std::uint32_t system = 0; system < system_count; ++system) {
      const auto& input = systems[system];
      if (!vibeqc::api::valid_descriptor(&input)) return VIBEQC_STATUS_ABI_MISMATCH;
      if (!input.atom_count || !input.atomic_numbers || !input.coordinates ||
          !std::isfinite(input.total_charge)) {
        context->last_detail =
            "D4 systems require nonempty atomic numbers/coordinates and finite charge";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      if (input.atom_count > static_cast<std::uint32_t>(vibeqc::dft::dispersion::kD4MaximumAtoms)) {
        context->last_detail = "D4 system exceeds the production per-system atom bound";
        return VIBEQC_STATUS_NOT_IMPLEMENTED;
      }
      total_atoms += input.atom_count;
      if (total_atoms > std::numeric_limits<std::uint32_t>::max() ||
          total_atoms > std::numeric_limits<std::size_t>::max() / 3u) {
        context->last_detail = "D4 ragged fleet exceeds the public offset domain";
        return VIBEQC_STATUS_OUT_OF_MEMORY;
      }
    }
    std::vector<std::uint32_t> offsets;
    std::vector<std::int32_t> atomic_numbers;
    std::vector<double> total_charges;
    std::vector<double> coordinates;
    offsets.reserve(static_cast<std::size_t>(system_count) + 1);
    atomic_numbers.reserve(static_cast<std::size_t>(total_atoms));
    total_charges.reserve(system_count);
    coordinates.reserve(3u * static_cast<std::size_t>(total_atoms));
    offsets.push_back(0);
    std::uint64_t copied_atoms = 0;
    for (std::uint32_t system = 0; system < system_count; ++system) {
      const auto& input = systems[system];
      copied_atoms += input.atom_count;
      const auto atom_count = static_cast<std::size_t>(input.atom_count);
      const auto coordinate_count = 3u * atom_count;
      atomic_numbers.insert(atomic_numbers.end(), input.atomic_numbers,
                            input.atomic_numbers + atom_count);
      coordinates.insert(coordinates.end(), input.coordinates,
                         input.coordinates + coordinate_count);
      total_charges.push_back(input.total_charge);
      offsets.push_back(static_cast<std::uint32_t>(copied_atoms));
    }
    vibeqc::dft::dispersion::D4Parameters parameters{
        vibeqc::dft::dispersion::D4ReferenceModel::eeq,
        model->s6,
        model->s8,
        model->s9,
        model->a1,
        model->a2,
        model->cn_cutoff,
        model->pair_cutoff,
        model->atm_cutoff,
        model->ga,
        model->gc,
    };
    vibeqc_status status = VIBEQC_STATUS_INTERNAL_ERROR;
    auto plan = vibeqc::dft::dispersion::D4Plan::prepare(
        context->state.executed_backend, context->state.device_id, std::move(offsets),
        std::move(atomic_numbers), std::move(total_charges), std::move(coordinates), parameters,
        profile, model->maximum_bytes, context->last_detail, status);
    if (!plan) return status;
    auto candidate = std::make_unique<vibeqc_d4_batch>();
    candidate->context = context;
    candidate->plan = std::move(plan);
    *batch = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}

void vibeqc_d4_batch_destroy(vibeqc_d4_batch* batch) { delete batch; }

vibeqc_status vibeqc_d4_batch_get_diagnostic(const vibeqc_d4_batch* batch,
                                             vibeqc_d4_runtime_diagnostic* diagnostic) {
  if (!batch || !diagnostic) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(diagnostic)) return VIBEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  const auto& resources = batch->plan->resources();
  const auto& counters = batch->plan->counters();
  diagnostic->backend = batch->plan->backend();
  diagnostic->profile = public_profile(batch->plan->profile());
  diagnostic->plan_host_bytes = resources.plan_host_bytes;
  diagnostic->execution_host_bytes = resources.execution_host_bytes;
  diagnostic->device_bytes = resources.device_bytes;
  diagnostic->table_bytes = resources.table_bytes;
  diagnostic->workspace_bytes = resources.workspace_bytes;
  diagnostic->maximum_bytes = resources.maximum_bytes;
  diagnostic->total_atoms = resources.total_atoms;
  diagnostic->system_count = resources.system_count;
  diagnostic->maximum_atoms = resources.maximum_atoms;
  diagnostic->worker_blocks = resources.worker_blocks;
  diagnostic->workspace_slots = resources.workspace_slots;
  diagnostic->execution_count = counters.execution_count;
  diagnostic->unchanged_geometry_replays = counters.unchanged_geometry_replays;
  diagnostic->changed_geometry_replays = counters.changed_geometry_replays;
  diagnostic->coordinate_h2d_bytes = counters.coordinate_h2d_bytes;
  diagnostic->kernel_launches = counters.kernel_launches;
  diagnostic->atm_enabled = batch->plan->parameters().s9 != 0.0 ? 1 : 0;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_d4_batch_execute(vibeqc_d4_batch* batch,
                                      const vibeqc_d4_batch_input_descriptor* inputs,
                                      uint32_t input_count,
                                      vibeqc_d4_batch_item_result_descriptor* results,
                                      uint32_t result_count) {
  if (!batch || !results) return VIBEQC_STATUS_INVALID_ARGUMENT;
  const auto systems = batch->plan->system_count();
  if (result_count != systems) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if ((!inputs && input_count != 0) || (inputs && input_count != systems))
    return VIBEQC_STATUS_INVALID_ARGUMENT;

  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    std::vector<double> coordinates;
    coordinates.reserve(3u * batch->plan->resources().total_atoms);
    std::vector<std::uint8_t> active(systems, 1);
    std::vector<std::uint8_t> want_gradient(systems, 0);
    std::vector<vibeqc_status> input_status(systems, VIBEQC_STATUS_SUCCESS);
    for (std::uint32_t system = 0; system < systems; ++system) {
      auto& output = results[system];
      if (!vibeqc::api::valid_descriptor(&output)) return VIBEQC_STATUS_ABI_MISMATCH;
      const auto atoms = batch->plan->atom_count(system);
      if ((!output.gradient && output.gradient_count != 0) ||
          (output.gradient && output.gradient_count != 3u * atoms) ||
          (!output.charges && output.charge_count != 0) ||
          (output.charges && output.charge_count != atoms)) {
        batch->context->last_detail =
            "D4 result gradient/charge shape does not match prepared system";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      want_gradient[system] = output.gradient ? 1 : 0;
      const auto prepared = batch->plan->default_coordinates(system);
      if (!inputs) {
        coordinates.insert(coordinates.end(), prepared.begin(), prepared.end());
        continue;
      }
      const auto& input = inputs[system];
      if (!vibeqc::api::valid_descriptor(&input)) return VIBEQC_STATUS_ABI_MISMATCH;
      if (!input.coordinates && input.coordinate_count == 0) {
        coordinates.insert(coordinates.end(), prepared.begin(), prepared.end());
        continue;
      }
      if (!input.coordinates || input.coordinate_count != 3u * atoms) {
        input_status[system] = VIBEQC_STATUS_INVALID_ARGUMENT;
        active[system] = 0;
        coordinates.insert(coordinates.end(), prepared.begin(), prepared.end());
        continue;
      }
      coordinates.insert(coordinates.end(), input.coordinates,
                         input.coordinates + input.coordinate_count);
    }

    std::vector<vibeqc::dft::dispersion::D4Status> statuses;
    std::vector<double> energy_components;
    std::vector<double> gradients;
    std::vector<double> charges;
    const auto status =
        batch->plan->execute(coordinates, active, want_gradient, statuses, energy_components,
                             gradients, charges, batch->context->last_detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;

    std::size_t atom_cursor = 0;
    for (std::uint32_t system = 0; system < systems; ++system) {
      auto& output = results[system];
      const auto atoms = batch->plan->atom_count(system);
      output.executed_backend = batch->plan->backend();
      output.status = input_status[system] == VIBEQC_STATUS_SUCCESS
                          ? public_status(statuses[system])
                          : input_status[system];
      if (output.status == VIBEQC_STATUS_SUCCESS) {
        output.two_body_energy = energy_components[2 * system];
        output.atm_energy = energy_components[2 * system + 1];
        output.energy = output.two_body_energy + output.atm_energy;
        if (output.gradient)
          std::copy_n(gradients.data() + 3 * atom_cursor, 3u * atoms, output.gradient);
        if (output.charges) std::copy_n(charges.data() + atom_cursor, atoms, output.charges);
      } else {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        output.energy = nan;
        output.two_body_energy = nan;
        output.atm_energy = nan;
      }
      atom_cursor += atoms;
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

}  // extern "C"

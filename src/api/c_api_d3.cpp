#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "d3_data.hpp"
#include "dft/dispersion/d3_runtime.hpp"
#include "vibeqc/vibeqc.h"

struct vibeqc_d3_batch {
  vibeqc_context* context{};
  std::unique_ptr<vibeqc::dft::dispersion::D3Plan> plan;
};

namespace {

vibeqc_status public_status(vibeqc::dft::dispersion::D3Status status) {
  using vibeqc::dft::dispersion::D3Status;
  switch (status) {
    case D3Status::success:
      return VIBEQC_STATUS_SUCCESS;
    case D3Status::invalid_argument:
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    case D3Status::unsupported:
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    case D3Status::numerical_failure:
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  return VIBEQC_STATUS_INTERNAL_ERROR;
}

}  // namespace

extern "C" {

const char* vibeqc_d3_table_sha256(void) { return vibeqc::dft::dispersion::d3_data::kTableSha256; }

const char* vibeqc_d3_radii_sha256(void) { return vibeqc::dft::dispersion::d3_data::kRadiiSha256; }

const char* vibeqc_d3_provider_identity(void) {
  return vibeqc::dft::dispersion::kD3ProductionProviderIdentity;
}

const char* vibeqc_d3_scheduler_identity(void) {
  return vibeqc::dft::dispersion::kD3ProductionSchedulerIdentity;
}

const char* vibeqc_d3_batch_variant_identity(const vibeqc_d3_batch* batch) {
  return batch ? vibeqc::dft::dispersion::d3_variant_identity(batch->plan->parameters()) : nullptr;
}

vibeqc_status vibeqc_d3_batch_prepare(vibeqc_context* context,
                                      const vibeqc_d3_system_descriptor* systems,
                                      uint32_t system_count, const vibeqc_d3_bj_descriptor* model,
                                      vibeqc_d3_batch** batch) {
  if (!context || !systems || !system_count || !model || !batch)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  *batch = nullptr;
  if (!vibeqc::api::valid_descriptor(model)) return VIBEQC_STATUS_ABI_MISMATCH;

  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    std::vector<std::uint32_t> offsets;
    std::vector<std::int32_t> atomic_numbers;
    std::vector<double> coordinates;
    offsets.reserve(static_cast<std::size_t>(system_count) + 1);
    offsets.push_back(0);
    std::uint64_t total_atoms = 0;

    for (std::uint32_t system = 0; system < system_count; ++system) {
      const auto& input = systems[system];
      if (!vibeqc::api::valid_descriptor(&input)) return VIBEQC_STATUS_ABI_MISMATCH;
      if (!input.atom_count || !input.atomic_numbers || !input.coordinates) {
        context->last_detail = "D3 systems require nonempty atomic numbers and coordinates";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      total_atoms += input.atom_count;
      if (total_atoms > std::numeric_limits<std::uint32_t>::max()) {
        context->last_detail = "D3 ragged fleet exceeds the public offset domain";
        return VIBEQC_STATUS_OUT_OF_MEMORY;
      }
      atomic_numbers.insert(atomic_numbers.end(), input.atomic_numbers,
                            input.atomic_numbers + input.atom_count);
      coordinates.insert(coordinates.end(), input.coordinates,
                         input.coordinates + 3u * input.atom_count);
      offsets.push_back(static_cast<std::uint32_t>(total_atoms));
    }

    using vibeqc::dft::dispersion::D3Damping;
    using vibeqc::dft::dispersion::D3ModelParameters;
    D3ModelParameters parameters{};
    if (model->damping == VIBEQC_D3_DAMPING_BJ) {
      parameters.damping = D3Damping::bj;
      parameters.bj = {model->s6, model->s8,        model->a1,          model->a2,
                       0.0,       model->cn_cutoff, model->pair_cutoff, model->pair_switch_width};
      if (model->s9 != 0.0) {
        parameters.atm_enabled = true;
        parameters.atm = {model->s9, model->cn_cutoff, model->atm_cutoff, model->atm_switch_width};
      } else {
        parameters.atm = {0.0, model->cn_cutoff, 0.0, 0.0};
      }
    } else if (model->damping == VIBEQC_D3_DAMPING_ZERO) {
      if (model->s9 != 0.0) {
        context->last_detail = "zero-damping D3 plus ATM is not a separately qualified capability";
        return VIBEQC_STATUS_NOT_IMPLEMENTED;
      }
      parameters.damping = D3Damping::zero;
      parameters.zero = {
          model->s6,  model->s8,        model->rs6,         model->rs8,
          model->alp, model->cn_cutoff, model->pair_cutoff, model->pair_switch_width};
      parameters.atm = {0.0, model->cn_cutoff, 0.0, 0.0};
    } else {
      context->last_detail = "unsupported D3 damping variant";
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    }

    vibeqc_status status = VIBEQC_STATUS_INTERNAL_ERROR;
    auto plan = vibeqc::dft::dispersion::D3Plan::prepare(
        context->state.executed_backend, context->state.device_id, std::move(offsets),
        std::move(atomic_numbers), std::move(coordinates), parameters, model->maximum_bytes,
        context->last_detail, status);
    if (!plan) return status;

    auto candidate = std::make_unique<vibeqc_d3_batch>();
    candidate->context = context;
    candidate->plan = std::move(plan);
    *batch = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}

void vibeqc_d3_batch_destroy(vibeqc_d3_batch* batch) { delete batch; }

vibeqc_status vibeqc_d3_batch_get_diagnostic(const vibeqc_d3_batch* batch,
                                             vibeqc_d3_runtime_diagnostic* diagnostic) {
  if (!batch || !diagnostic) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(diagnostic)) return VIBEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  const auto& resources = batch->plan->resources();
  diagnostic->backend = batch->plan->backend();
  diagnostic->plan_host_bytes = resources.plan_host_bytes;
  diagnostic->execution_host_bytes = resources.execution_host_bytes;
  diagnostic->device_bytes = resources.device_bytes;
  diagnostic->table_bytes = resources.table_bytes;
  diagnostic->workspace_bytes = resources.workspace_bytes;
  diagnostic->maximum_bytes = resources.maximum_bytes;
  diagnostic->total_atoms = resources.total_atoms;
  diagnostic->system_count = resources.system_count;
  diagnostic->maximum_atoms = resources.maximum_atoms;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_d3_batch_execute(vibeqc_d3_batch* batch,
                                      const vibeqc_d3_batch_input_descriptor* inputs,
                                      uint32_t input_count,
                                      vibeqc_d3_batch_item_result_descriptor* results,
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
          (output.gradient && output.gradient_count != 3u * atoms)) {
        batch->context->last_detail = "D3 result gradient shape does not match prepared system";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      want_gradient[system] = output.gradient ? 1 : 0;

      auto prepared = batch->plan->default_coordinates(system);
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

    std::vector<vibeqc::dft::dispersion::D3Status> statuses;
    std::vector<double> energies;
    std::vector<double> gradients;
    const auto status = batch->plan->execute(coordinates, active, want_gradient, statuses, energies,
                                             gradients, batch->context->last_detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;

    std::size_t atom_cursor = 0;
    for (std::uint32_t system = 0; system < systems; ++system) {
      auto& output = results[system];
      const auto atoms = batch->plan->atom_count(system);
      output.executed_backend = batch->plan->backend();
      output.status = input_status[system] == VIBEQC_STATUS_SUCCESS
                          ? public_status(statuses[system])
                          : input_status[system];
      output.energy = output.status == VIBEQC_STATUS_SUCCESS
                          ? energies[system]
                          : std::numeric_limits<double>::quiet_NaN();
      if (output.status == VIBEQC_STATUS_SUCCESS && output.gradient) {
        std::copy_n(gradients.data() + 3 * atom_cursor, 3u * atoms, output.gradient);
      }
      atom_cursor += atoms;
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

}  // extern "C"

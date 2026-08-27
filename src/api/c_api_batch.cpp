#include "vibeqc/vibeqc.h"

#include "api/error.hpp"
#include "api/handles.hpp"
#include "methods/method.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

extern "C" {

vibeqc_status vibeqc_batch_prepare(
    vibeqc_context* context,
    const vibeqc_system* const* systems,
    uint32_t system_count,
    const vibeqc_method_descriptor* descriptor,
    vibeqc_batch_flags flags,
    vibeqc_batch** batch) {
  if (context == nullptr || systems == nullptr || system_count == 0 ||
      descriptor == nullptr || batch == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *batch = nullptr;
  if (!vibeqc::api::valid_descriptor(descriptor)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }

  try {
    std::vector<vibeqc::core::System> native_systems;
    native_systems.reserve(system_count);
    std::vector<std::uint32_t> atom_counts;
    atom_counts.reserve(system_count);
    for (std::uint32_t i = 0; i < system_count; ++i) {
      if (systems[i] == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
      native_systems.push_back(systems[i]->data);
      atom_counts.push_back(
          static_cast<std::uint32_t>(systems[i]->data.atoms.size()));
    }
    auto candidate = std::make_unique<vibeqc_batch>();
    candidate->context = context;
    candidate->atom_counts = std::move(atom_counts);
    candidate->plan = vibeqc::methods::prepare_batch(
        context->state, std::move(native_systems), *descriptor, flags);
    *batch = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}

void vibeqc_batch_destroy(vibeqc_batch* batch) { delete batch; }

uint32_t vibeqc_batch_get_system_count(const vibeqc_batch* batch) {
  return batch == nullptr ? 0
                          : static_cast<std::uint32_t>(batch->plan->size());
}

vibeqc_status vibeqc_batch_get_last_shell_class_profile(
    const vibeqc_batch* batch,
    vibeqc_shell_class_profile_entry* entries,
    uint32_t entry_count) {
  if (batch == nullptr || entries == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    const auto profile = batch->plan->last_direct_shell_class_profile();
    if (!profile.has_value()) return VIBEQC_STATUS_NOT_IMPLEMENTED;
    if (entry_count < profile->size()) return VIBEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < profile->size(); ++index) {
      const vibeqc::methods::DirectShellClassProfileEntry& source =
          (*profile)[index];
      entries[index] = {source.shell_quartets, source.tiles, source.ao_quartets,
                        source.primitive_quartets};
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_batch_clear_warm_starts(vibeqc_batch* batch) {
  if (batch == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  try {
    batch->plan->clear_warm_starts();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_batch_set_warm_start_updates(vibeqc_batch* batch,
                                                   int32_t enabled) {
  if (batch == nullptr || (enabled != 0 && enabled != 1)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    batch->plan->set_warm_start_updates(enabled != 0);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_batch_execute(
    vibeqc_batch* batch,
    const vibeqc_batch_input_descriptor* inputs,
    uint32_t input_count,
    vibeqc_batch_item_result_descriptor* results,
    uint32_t result_count) {
  if (batch == nullptr || results == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::uint32_t system_count = vibeqc_batch_get_system_count(batch);
  if (result_count != system_count ||
      ((inputs == nullptr) != (input_count == 0)) ||
      (inputs != nullptr && input_count != system_count)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::uint32_t i = 0; i < result_count; ++i) {
    if (!vibeqc::api::valid_descriptor(&results[i])) {
      return VIBEQC_STATUS_ABI_MISMATCH;
    }
  }
  if (inputs != nullptr) {
    for (std::uint32_t i = 0; i < input_count; ++i) {
      if (!vibeqc::api::valid_descriptor(&inputs[i])) {
        return VIBEQC_STATUS_ABI_MISMATCH;
      }
    }
  }

  try {
    vibeqc::methods::Coordinates coordinates;
    if (inputs != nullptr) {
      coordinates.resize(system_count);
      for (std::uint32_t i = 0; i < system_count; ++i) {
        if (inputs[i].coordinates == nullptr && inputs[i].coordinate_count == 0) {
          continue;
        }
        if (inputs[i].coordinates == nullptr) {
          // Preserve item-level failure isolation for a malformed coordinate
          // payload without rejecting structurally valid neighboring systems.
          coordinates[i] = std::vector<double>{
              std::numeric_limits<double>::quiet_NaN()};
          continue;
        }
        coordinates[i] = std::vector<double>(
            inputs[i].coordinates,
            inputs[i].coordinates + inputs[i].coordinate_count);
      }
    }

    std::vector<vibeqc::methods::BatchItemResult> native =
        batch->plan->execute(coordinates);
    if (native.size() != system_count) {
      return VIBEQC_STATUS_INTERNAL_ERROR;
    }
    for (std::uint32_t i = 0; i < system_count; ++i) {
      vibeqc_batch_item_result_descriptor& output = results[i];
      const vibeqc::methods::BatchItemResult& item = native[i];
      const std::uint32_t required_forces = batch->atom_counts[i] * 3;
      const bool omit_forces =
          output.forces == nullptr && output.force_count == 0;
      const bool valid_force_buffer = omit_forces ||
          (output.forces != nullptr && output.force_count >= required_forces);
      output.status = valid_force_buffer
          ? item.status : VIBEQC_STATUS_INVALID_ARGUMENT;
      output.energy = item.calculation.energy;
      output.iterations = item.calculation.convergence.iterations;
      output.energy_change = item.calculation.convergence.energy_change;
      output.density_rms = item.calculation.convergence.residual_rms;
      output.converged = item.calculation.convergence.converged ? 1 : 0;
      output.executed_backend = item.calculation.executed_backend;
      output.bucket_id = static_cast<std::uint32_t>(item.bucket_id);
      output.warm_start_used = item.warm_start_used ? 1 : 0;
      output.warm_start_fallback = item.warm_start_fallback ? 1 : 0;
      if (valid_force_buffer && !omit_forces &&
          item.status == VIBEQC_STATUS_SUCCESS) {
        std::copy(item.calculation.forces.begin(),
                  item.calculation.forces.end(), output.forces);
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

}  // extern "C"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "methods/method.hpp"
#include "vibeqc/vibeqc.h"

extern "C" {

vibeqc_status vibeqc_batch_prepare(vibeqc_context* context, const vibeqc_system* const* systems,
                                   uint32_t system_count,
                                   const vibeqc_method_descriptor* descriptor,
                                   vibeqc_batch_flags flags, vibeqc_batch** batch) {
  if (context == nullptr || systems == nullptr || system_count == 0 || descriptor == nullptr ||
      batch == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *batch = nullptr;
  if (!vibeqc::api::valid_method_descriptor(descriptor)) {
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
      atom_counts.push_back(static_cast<std::uint32_t>(systems[i]->data.atoms.size()));
    }
    auto candidate = std::make_unique<vibeqc_batch>();
    candidate->context = context;
    candidate->atom_counts = std::move(atom_counts);
    candidate->plan = vibeqc::methods::prepare_batch(context->state, std::move(native_systems),
                                                     *descriptor, flags);
    *batch = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}

void vibeqc_batch_destroy(vibeqc_batch* batch) { delete batch; }

uint32_t vibeqc_batch_get_system_count(const vibeqc_batch* batch) {
  return batch == nullptr ? 0 : static_cast<std::uint32_t>(batch->plan->size());
}

vibeqc_status vibeqc_batch_get_last_shell_class_profile(const vibeqc_batch* batch,
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
      const vibeqc::methods::DirectShellClassProfileEntry& source = (*profile)[index];
      entries[index] = {source.shell_quartets, source.tiles, source.ao_quartets,
                        source.primitive_quartets};
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_batch_get_last_ppps_queue_profile(const vibeqc_batch* batch,
                                                       vibeqc_ppps_queue_profile* profile) {
  if (batch == nullptr || profile == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    const auto source = batch->plan->last_direct_ppps_queue_profile();
    if (!source.has_value()) return VIBEQC_STATUS_NOT_IMPLEMENTED;
    *profile = {};
    profile->descriptor_slots = source->descriptor_slots;
    profile->non_empty_descriptors = source->non_empty_descriptors;
    profile->empty_descriptors = source->empty_descriptors;
    profile->tasks = source->tasks;
    profile->primitive_work = source->primitive_work;
    profile->ket_count_min = source->ket_count_min;
    profile->ket_count_median = source->ket_count_median;
    profile->ket_count_p90 = source->ket_count_p90;
    profile->ket_count_p99 = source->ket_count_p99;
    profile->ket_count_max = source->ket_count_max;
    std::copy(source->lane_efficiency.begin(), source->lane_efficiency.end(),
              profile->lane_efficiency);
    profile->primitive_warp_efficiency = source->primitive_warp_efficiency;
    std::copy(source->task_tail_imbalance.begin(), source->task_tail_imbalance.end(),
              profile->task_tail_imbalance);
    std::copy(source->primitive_tail_imbalance.begin(), source->primitive_tail_imbalance.end(),
              profile->primitive_tail_imbalance);
    std::copy(source->orientation_tasks.begin(), source->orientation_tasks.end(),
              profile->orientation_tasks);
    std::copy(source->orientation_primitive_work.begin(), source->orientation_primitive_work.end(),
              profile->orientation_primitive_work);
    std::copy(source->bra_primitive_tasks.begin(), source->bra_primitive_tasks.end(),
              profile->bra_primitive_tasks);
    std::copy(source->bra_primitive_work.begin(), source->bra_primitive_work.end(),
              profile->bra_primitive_work);
    std::copy(source->ket_primitive_tasks.begin(), source->ket_primitive_tasks.end(),
              profile->ket_primitive_tasks);
    std::copy(source->ket_primitive_work.begin(), source->ket_primitive_work.end(),
              profile->ket_primitive_work);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_batch_get_last_eigensolver_diagnostics(const vibeqc_batch* batch,
                                                            vibeqc_eigensolver_diagnostic* entries,
                                                            uint32_t entry_count,
                                                            uint32_t* written_count) {
  if (batch == nullptr || written_count == nullptr || (entries == nullptr && entry_count != 0U)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *written_count = 0U;
  try {
    const auto source = batch->plan->last_eigensolver_diagnostics();
    if (source.empty()) return VIBEQC_STATUS_NOT_IMPLEMENTED;
    if (source.size() > std::numeric_limits<std::uint32_t>::max()) {
      return VIBEQC_STATUS_INTERNAL_ERROR;
    }
    *written_count = static_cast<std::uint32_t>(source.size());
    if (entries == nullptr) return VIBEQC_STATUS_SUCCESS;
    if (entry_count < source.size()) return VIBEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.size(); ++index) {
      const vibeqc::methods::EigensolverDiagnostic& input = source[index];
      vibeqc_eigensolver_diagnostic& output = entries[index];
      output = {};
      output.bucket_id = input.bucket_id;
      output.ordinary_family = static_cast<std::int32_t>(input.ordinary_family);
      output.graph_family = static_cast<std::int32_t>(input.graph_family);
      output.selection_source = static_cast<std::int32_t>(input.selection_source);
      output.matrix_dimension = input.matrix_dimension;
      output.physical_system_count = input.physical_system_count;
      output.solver_batch_count = input.solver_batch_count;
      output.api_eligible = input.api_eligible ? 1 : 0;
      output.api_reason = static_cast<std::int32_t>(input.api_reason);
      output.matrix_batch_product = input.matrix_batch_product;
      output.probe_failure_stage = static_cast<std::int32_t>(input.probe_failure_stage);
      output.device_workspace_bytes = input.device_workspace_bytes;
      output.host_workspace_bytes = input.host_workspace_bytes;
      output.available_device_bytes = input.available_device_bytes;
      output.device_id = input.device_id;
      std::copy(input.device_uuid.begin(), input.device_uuid.end(), output.device_uuid);
      std::copy(input.device_name.begin(), input.device_name.end(), output.device_name);
      output.compute_capability_major = input.compute_capability_major;
      output.compute_capability_minor = input.compute_capability_minor;
      output.cuda_runtime_version = input.cuda_runtime_version;
      output.cuda_driver_version = input.cuda_driver_version;
      output.cusolver_version = input.cusolver_version;
      output.cuda_error = input.cuda_error;
      output.cusolver_error = input.cusolver_error;
      output.ordinary_execution_passed = input.ordinary_execution_passed ? 1 : 0;
      output.graph_capture_passed = input.graph_capture_passed ? 1 : 0;
      output.host_graph_replay_passed = input.host_graph_replay_passed ? 1 : 0;
      output.device_tail_replay_passed = input.device_tail_replay_passed ? 1 : 0;
      output.graph_eligible = input.graph_eligible ? 1 : 0;
      output.maximum_eigenvalue_error = input.maximum_eigenvalue_error;
      output.maximum_residual = input.maximum_residual;
      output.maximum_orthogonality_error = input.maximum_orthogonality_error;
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_batch_get_last_density_fitting_metric_diagnostics(
    const vibeqc_batch* batch, vibeqc_density_fitting_metric_diagnostic* entries,
    uint32_t entry_count, uint32_t* written_count) {
  if (batch == nullptr || written_count == nullptr || (entries == nullptr && entry_count != 0U)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *written_count = 0U;
  try {
    const auto& source = batch->plan->last_density_fitting_metric_diagnostics();
    if (source.empty()) return VIBEQC_STATUS_NOT_IMPLEMENTED;
    if (source.size() > std::numeric_limits<uint32_t>::max()) {
      return VIBEQC_STATUS_INTERNAL_ERROR;
    }
    *written_count = static_cast<uint32_t>(source.size());
    if (entries == nullptr) return VIBEQC_STATUS_SUCCESS;
    if (entry_count < source.size()) return VIBEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.size(); ++index) {
      const auto& input = source[index];
      auto& output = entries[index];
      output = {};
      output.bucket_id = static_cast<uint32_t>(input.bucket_id);
      output.system_index = static_cast<uint32_t>(input.system_index);
      output.effective_rank = input.effective_rank;
      output.absolute_threshold = input.absolute_threshold;
      output.condition_number = input.condition_number;
      output.solver_device_workspace_bytes = input.solver_device_workspace_bytes;
      output.solver_host_workspace_bytes = input.solver_host_workspace_bytes;
      output.device_resident_bytes = input.device_resident_bytes;
      output.peak_device_bytes = input.peak_device_bytes;
      output.host_resident_bytes = input.host_resident_bytes;
      output.peak_host_bytes = input.peak_host_bytes;
      output.auxiliary_tile = input.auxiliary_tile;
      output.streamed = input.streamed ? 1 : 0;
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_batch_get_last_inactive_eigensolver_profile(
    const vibeqc_batch* batch, vibeqc_inactive_eigensolver_profile_entry* entries,
    uint32_t entry_count, uint32_t* written_count) {
  if (batch == nullptr || written_count == nullptr || (entries == nullptr && entry_count != 0U)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *written_count = 0U;
  try {
    const auto source = batch->plan->last_inactive_eigensolver_profile();
    if (source.empty()) return VIBEQC_STATUS_NOT_IMPLEMENTED;
    if (source.size() > std::numeric_limits<std::uint32_t>::max()) {
      return VIBEQC_STATUS_INTERNAL_ERROR;
    }
    *written_count = static_cast<std::uint32_t>(source.size());
    if (entries == nullptr) return VIBEQC_STATUS_SUCCESS;
    if (entry_count < source.size()) return VIBEQC_STATUS_INVALID_ARGUMENT;
    for (std::size_t index = 0; index < source.size(); ++index) {
      const vibeqc::methods::InactiveEigensolverProfileEntry& input = source[index];
      vibeqc_inactive_eigensolver_profile_entry& output = entries[index];
      output = {};
      output.bucket_id = input.bucket_id;
      output.iteration = input.iteration;
      output.family = static_cast<std::int32_t>(input.family);
      output.physical_system_count = input.physical_system_count;
      output.solver_batch_count = input.solver_batch_count;
      output.active_physical_count = input.active_physical_count;
      output.active_solver_count = input.active_solver_count;
      output.solver_elapsed_nanoseconds = input.solver_elapsed_nanoseconds;
      output.inactive_input_nonfinite_count = input.inactive_input_nonfinite_count;
      output.inactive_submission_nonfinite_count = input.inactive_submission_nonfinite_count;
      output.inactive_info_nonzero_count = input.inactive_info_nonzero_count;
      output.inactive_touch_flags = input.inactive_touch_flags;
      output.provider_invoked = input.provider_invoked ? 1 : 0;
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

vibeqc_status vibeqc_batch_set_warm_start_updates(vibeqc_batch* batch, int32_t enabled) {
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

vibeqc_status vibeqc_batch_execute(vibeqc_batch* batch, const vibeqc_batch_input_descriptor* inputs,
                                   uint32_t input_count,
                                   vibeqc_batch_item_result_descriptor* results,
                                   uint32_t result_count) {
  if (batch == nullptr || results == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::uint32_t system_count = vibeqc_batch_get_system_count(batch);
  if (result_count != system_count || ((inputs == nullptr) != (input_count == 0)) ||
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
          coordinates[i] = std::vector<double>{std::numeric_limits<double>::quiet_NaN()};
          continue;
        }
        coordinates[i] = std::vector<double>(inputs[i].coordinates,
                                             inputs[i].coordinates + inputs[i].coordinate_count);
      }
    }

    std::vector<vibeqc::methods::BatchItemResult> native = batch->plan->execute(coordinates);
    if (native.size() != system_count) {
      return VIBEQC_STATUS_INTERNAL_ERROR;
    }
    for (std::uint32_t i = 0; i < system_count; ++i) {
      vibeqc_batch_item_result_descriptor& output = results[i];
      const vibeqc::methods::BatchItemResult& item = native[i];
      const std::uint32_t required_forces = batch->atom_counts[i] * 3;
      const bool omit_forces = output.forces == nullptr && output.force_count == 0;
      const bool valid_force_buffer =
          omit_forces || (output.forces != nullptr && output.force_count >= required_forces);
      output.status = valid_force_buffer ? item.status : VIBEQC_STATUS_INVALID_ARGUMENT;
      output.energy = item.calculation.energy;
      output.iterations = item.calculation.convergence.iterations;
      output.energy_change = item.calculation.convergence.energy_change;
      output.density_rms = item.calculation.convergence.residual_rms;
      output.converged = item.calculation.convergence.converged ? 1 : 0;
      output.executed_backend = item.calculation.executed_backend;
      output.bucket_id = static_cast<std::uint32_t>(item.bucket_id);
      output.warm_start_used = item.warm_start_used ? 1 : 0;
      output.warm_start_fallback = item.warm_start_fallback ? 1 : 0;
      if (valid_force_buffer && !omit_forces && item.status == VIBEQC_STATUS_SUCCESS) {
        std::copy(item.calculation.forces.begin(), item.calculation.forces.end(), output.forces);
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

}  // extern "C"

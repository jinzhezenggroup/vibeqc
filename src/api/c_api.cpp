#include "vibeqc/vibeqc.h"

#include "core/types.hpp"
#include "molecule/basis.hpp"
#include "runtime/context.hpp"
#include "scf/fleet.hpp"
#include "scf/rhf.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

struct vibeqc_context {
  vibeqc::core::ContextState state;
  std::string last_detail;
};

struct vibeqc_system {
  vibeqc::core::System data;
};

struct vibeqc_calculation {
  vibeqc_context* context{};
  vibeqc::core::System system;
  vibeqc::core::ScfOptions options;
  vibeqc_method method{VIBEQC_METHOD_RHF};
};

struct vibeqc_batch {
  vibeqc_context* context{};
  std::unique_ptr<vibeqc::scf::FleetPlan> plan;
  std::vector<std::uint32_t> atom_counts;
};

namespace {

template <typename T>
bool valid_descriptor(const T* descriptor) {
  return descriptor != nullptr && descriptor->struct_size >= sizeof(T) &&
         descriptor->abi_version == VIBEQC_ABI_VERSION;
}

vibeqc_status map_exception() {
  try {
    throw;
  } catch (const std::bad_alloc&) {
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

vibeqc::core::ScfOptions scf_options(const vibeqc_method_descriptor& descriptor) {
  vibeqc::core::ScfOptions options;
  options.max_iterations =
      descriptor.max_iterations == 0 ? 100 : descriptor.max_iterations;
  options.diis_history = descriptor.diis_history == 0 ? 8 : descriptor.diis_history;
  options.energy_tolerance = descriptor.energy_tolerance > 0.0
      ? descriptor.energy_tolerance : 1.0e-10;
  options.density_tolerance = descriptor.density_tolerance > 0.0
      ? descriptor.density_tolerance : 1.0e-8;
  options.screening_tolerance = descriptor.screening_tolerance > 0.0
      ? descriptor.screening_tolerance : 1.0e-12;
  return options;
}

vibeqc_status validate_rhf_system(const vibeqc::core::System& system) {
  if (system.electron_count % 2 != 0 || system.multiplicity != 1) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status validate_uhf_system(const vibeqc::core::System& system) {
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (spin_excess < 0 || spin_excess > system.electron_count ||
      ((system.electron_count + spin_excess) & 1) != 0) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status validate_hf_system(vibeqc_method method,
                              const vibeqc::core::System& system) {
  return method == VIBEQC_METHOD_UHF ? validate_uhf_system(system)
                                  : validate_rhf_system(system);
}

}  // namespace

extern "C" {

uint32_t vibeqc_get_abi_version(void) { return VIBEQC_ABI_VERSION; }

const char* vibeqc_status_message(vibeqc_status status) {
  switch (status) {
    case VIBEQC_STATUS_SUCCESS: return "success";
    case VIBEQC_STATUS_INVALID_ARGUMENT: return "invalid argument";
    case VIBEQC_STATUS_ABI_MISMATCH: return "ABI mismatch";
    case VIBEQC_STATUS_NOT_IMPLEMENTED: return "requested capability is not implemented";
    case VIBEQC_STATUS_SCF_NOT_CONVERGED: return "SCF did not converge";
    case VIBEQC_STATUS_NUMERICAL_FAILURE: return "numerical failure";
    case VIBEQC_STATUS_CUDA_ERROR: return "CUDA runtime error";
    case VIBEQC_STATUS_OUT_OF_MEMORY: return "out of memory";
    case VIBEQC_STATUS_INTERNAL_ERROR: return "internal error";
  }
  return "unknown status";
}

vibeqc_status vibeqc_method_available(vibeqc_method method, int32_t* available) {
  if (available == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  switch (method) {
    case VIBEQC_METHOD_RHF:
    case VIBEQC_METHOD_UHF:
      *available = 1;
      return VIBEQC_STATUS_SUCCESS;
    case VIBEQC_METHOD_WB97M_V:
    case VIBEQC_METHOD_RCCSD_T:
      *available = 0;
      return VIBEQC_STATUS_SUCCESS;
  }
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status vibeqc_context_create(const vibeqc_context_descriptor* descriptor,
                              vibeqc_context** context) {
  if (context == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  *context = nullptr;
  if (descriptor == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (descriptor->abi_version != VIBEQC_ABI_VERSION ||
      descriptor->struct_size < sizeof(vibeqc_context_descriptor)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
  try {
    auto candidate = std::make_unique<vibeqc_context>();
    candidate->state.device_id = descriptor->device_id;
    candidate->state.requested_backend = descriptor->backend;
    const vibeqc_status status =
        vibeqc::runtime::initialize_context(candidate->state, candidate->last_detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    *context = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

void vibeqc_context_destroy(vibeqc_context* context) { delete context; }

vibeqc_status vibeqc_system_create(vibeqc_context* context,
                             const vibeqc_system_descriptor* descriptor,
                             vibeqc_system** system) {
  if (context == nullptr || system == nullptr || descriptor == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *system = nullptr;
  if (descriptor->abi_version != VIBEQC_ABI_VERSION ||
      descriptor->struct_size <
          offsetof(vibeqc_system_descriptor, basis_representation)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
  if (descriptor->atoms == nullptr || descriptor->shells == nullptr ||
      descriptor->primitives == nullptr || descriptor->atom_count == 0 ||
      descriptor->shell_count == 0) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto candidate = std::make_unique<vibeqc_system>();
    candidate->data.charge = descriptor->charge;
    candidate->data.multiplicity = descriptor->multiplicity;
    candidate->data.basis_representation =
        descriptor->struct_size >= sizeof(vibeqc_system_descriptor)
        ? descriptor->basis_representation
        : VIBEQC_BASIS_CARTESIAN;
    candidate->data.atoms.reserve(descriptor->atom_count);
    for (std::uint32_t i = 0; i < descriptor->atom_count; ++i) {
      const vibeqc_atom& atom = descriptor->atoms[i];
      candidate->data.atoms.push_back(
          {atom.atomic_number, {atom.x, atom.y, atom.z}});
    }
    candidate->data.shells.reserve(descriptor->shell_count);
    for (std::uint32_t i = 0; i < descriptor->shell_count; ++i) {
      const vibeqc_shell& shell = descriptor->shells[i];
      if (shell.primitive_count == 0 ||
          shell.primitive_offset > descriptor->primitive_count ||
          shell.primitive_count >
              descriptor->primitive_count - shell.primitive_offset) {
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      vibeqc::core::Shell native_shell;
      native_shell.atom_index = shell.atom_index;
      native_shell.angular_momentum = shell.angular_momentum;
      native_shell.primitives.reserve(shell.primitive_count);
      for (std::uint32_t p = 0; p < shell.primitive_count; ++p) {
        const vibeqc_primitive& primitive =
            descriptor->primitives[shell.primitive_offset + p];
        native_shell.primitives.push_back({primitive.exponent, primitive.coefficient});
      }
      candidate->data.shells.push_back(std::move(native_shell));
    }
    const vibeqc_status status =
        vibeqc::molecule::validate_and_normalize(candidate->data, context->last_detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    *system = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

void vibeqc_system_destroy(vibeqc_system* system) { delete system; }

vibeqc_status vibeqc_calculation_prepare(vibeqc_context* context,
                                   const vibeqc_system* system,
                                   const vibeqc_method_descriptor* descriptor,
                                   vibeqc_calculation** calculation) {
  if (context == nullptr || system == nullptr || descriptor == nullptr ||
      calculation == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *calculation = nullptr;
  if (descriptor->abi_version != VIBEQC_ABI_VERSION ||
      descriptor->struct_size < sizeof(vibeqc_method_descriptor)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
  if (descriptor->method != VIBEQC_METHOD_RHF &&
      descriptor->method != VIBEQC_METHOD_UHF) {
    int available = 0;
    if (vibeqc_method_available(descriptor->method, &available) != VIBEQC_STATUS_SUCCESS) {
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  if (validate_hf_system(descriptor->method, system->data) !=
      VIBEQC_STATUS_SUCCESS) {
    context->last_detail = descriptor->method == VIBEQC_METHOD_UHF
        ? "UHF requires electron count and multiplicity to define integral spin occupations"
        : "RHF requires an even electron count and spin multiplicity 1";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto candidate = std::make_unique<vibeqc_calculation>();
    candidate->context = context;
    candidate->system = system->data;
    candidate->method = descriptor->method;
    candidate->options = scf_options(*descriptor);
    *calculation = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

void vibeqc_calculation_destroy(vibeqc_calculation* calculation) { delete calculation; }

vibeqc_status vibeqc_calculation_execute(vibeqc_calculation* calculation,
                                   vibeqc_result_descriptor* result) {
  if (calculation == nullptr || result == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (result->abi_version != VIBEQC_ABI_VERSION ||
      result->struct_size < sizeof(vibeqc_result_descriptor)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
  const std::size_t required_forces = calculation->system.atoms.size() * 3;
  if (result->forces == nullptr || result->force_count < required_forces) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    const bool use_cuda =
        calculation->context->state.requested_backend == VIBEQC_BACKEND_CUDA;
    const bool unrestricted = calculation->method == VIBEQC_METHOD_UHF;
    const vibeqc::core::ScfResult native = use_cuda
        ? (unrestricted
               ? vibeqc::scf::run_uhf_cuda(
                     calculation->system, calculation->options,
                     calculation->context->state.device_id)
               : vibeqc::scf::run_rhf_cuda(
                     calculation->system, calculation->options,
                     calculation->context->state.device_id))
        : (unrestricted
               ? vibeqc::scf::run_uhf(calculation->system, calculation->options)
               : vibeqc::scf::run_rhf(calculation->system, calculation->options));
    result->energy = native.energy;
    result->iterations = native.iterations;
    result->energy_change = native.energy_change;
    result->density_rms = native.density_rms;
    result->converged = native.converged ? 1 : 0;
    result->executed_backend = use_cuda ? VIBEQC_BACKEND_CUDA
                                        : VIBEQC_BACKEND_CPU_REFERENCE;
    if (!native.converged) return VIBEQC_STATUS_SCF_NOT_CONVERGED;
    std::copy(native.forces.begin(), native.forces.end(), result->forces);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

vibeqc_status vibeqc_batch_prepare(vibeqc_context* context,
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
  if (!valid_descriptor(descriptor)) return VIBEQC_STATUS_ABI_MISMATCH;
  if (descriptor->method != VIBEQC_METHOD_RHF &&
      descriptor->method != VIBEQC_METHOD_UHF) {
    int available = 0;
    if (vibeqc_method_available(descriptor->method, &available) != VIBEQC_STATUS_SUCCESS) {
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  constexpr vibeqc_batch_flags supported_flags =
      VIBEQC_BATCH_ENABLE_WARM_STARTS |
      VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING;
  if ((flags & ~supported_flags) != 0) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    std::vector<vibeqc::core::System> native_systems;
    native_systems.reserve(system_count);
    std::vector<std::uint32_t> atom_counts;
    atom_counts.reserve(system_count);
    for (std::uint32_t i = 0; i < system_count; ++i) {
      if (systems[i] == nullptr ||
          validate_hf_system(descriptor->method, systems[i]->data) !=
              VIBEQC_STATUS_SUCCESS) {
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      native_systems.push_back(systems[i]->data);
      atom_counts.push_back(static_cast<std::uint32_t>(systems[i]->data.atoms.size()));
    }
    auto candidate = std::make_unique<vibeqc_batch>();
    candidate->context = context;
    candidate->atom_counts = std::move(atom_counts);
    candidate->plan = std::make_unique<vibeqc::scf::FleetPlan>(
        std::move(native_systems), descriptor->method, scf_options(*descriptor),
        (flags & VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0,
        context->state.requested_backend == VIBEQC_BACKEND_CUDA,
        (flags & VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING) != 0,
        context->state.device_id);
    *batch = candidate.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

void vibeqc_batch_destroy(vibeqc_batch* batch) { delete batch; }

uint32_t vibeqc_batch_get_system_count(const vibeqc_batch* batch) {
  if (batch == nullptr || batch->plan == nullptr) return 0;
  return static_cast<std::uint32_t>(batch->plan->size());
}

vibeqc_status vibeqc_batch_get_last_shell_class_profile(
    const vibeqc_batch* batch,
    vibeqc_shell_class_profile_entry* entries,
    uint32_t entry_count) {
  if (batch == nullptr || batch->plan == nullptr || entries == nullptr ||
      entry_count < VIBEQC_DIRECT_SHELL_CLASS_COUNT) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto& profile = batch->plan->last_shell_class_profile();
  if (!profile.has_value()) return VIBEQC_STATUS_NOT_IMPLEMENTED;
  static_assert(vibeqc::scf::detail::kDirectQuartetShellClassCount ==
                VIBEQC_DIRECT_SHELL_CLASS_COUNT);
  for (std::size_t shell_class = 0; shell_class < profile->size();
       ++shell_class) {
    entries[shell_class] = {
        (*profile)[shell_class].shell_quartets,
        (*profile)[shell_class].tiles,
        (*profile)[shell_class].ao_quartets,
        (*profile)[shell_class].primitive_quartets,
    };
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_batch_clear_warm_starts(vibeqc_batch* batch) {
  if (batch == nullptr || batch->plan == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  batch->plan->clear_warm_starts();
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_batch_execute(vibeqc_batch* batch,
                             const vibeqc_batch_input_descriptor* inputs,
                             uint32_t input_count,
                             vibeqc_batch_item_result_descriptor* results,
                             uint32_t result_count) {
  if (batch == nullptr || batch->plan == nullptr || results == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::uint32_t system_count = vibeqc_batch_get_system_count(batch);
  if (result_count != system_count ||
      ((inputs == nullptr) != (input_count == 0)) ||
      (inputs != nullptr && input_count != system_count)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::uint32_t i = 0; i < result_count; ++i) {
    if (!valid_descriptor(&results[i])) return VIBEQC_STATUS_ABI_MISMATCH;
  }
  if (inputs != nullptr) {
    for (std::uint32_t i = 0; i < input_count; ++i) {
      if (!valid_descriptor(&inputs[i])) return VIBEQC_STATUS_ABI_MISMATCH;
    }
  }

  try {
    std::vector<std::optional<std::vector<double>>> coordinates;
    if (inputs != nullptr) {
      coordinates.resize(system_count);
      for (std::uint32_t i = 0; i < system_count; ++i) {
        if (inputs[i].coordinates == nullptr && inputs[i].coordinate_count == 0) {
          continue;
        }
        if (inputs[i].coordinates == nullptr) {
          // A deliberately invalid finite-coordinate payload lets FleetPlan
          // isolate this item without aborting structurally valid neighbors.
          coordinates[i] = std::vector<double>{
              std::numeric_limits<double>::quiet_NaN()};
          continue;
        }
        coordinates[i] = std::vector<double>(
            inputs[i].coordinates,
            inputs[i].coordinates + inputs[i].coordinate_count);
      }
    }

    const std::vector<vibeqc::scf::FleetItemResult> native =
        batch->plan->execute(coordinates);
    for (std::uint32_t i = 0; i < system_count; ++i) {
      vibeqc_batch_item_result_descriptor& output = results[i];
      const vibeqc::scf::FleetItemResult& item = native[i];
      const std::uint32_t required_forces = batch->atom_counts[i] * 3;
      const bool valid_force_buffer =
          output.forces != nullptr && output.force_count >= required_forces;
      output.status = valid_force_buffer ? item.status : VIBEQC_STATUS_INVALID_ARGUMENT;
      output.energy = item.scf.energy;
      output.iterations = item.scf.iterations;
      output.energy_change = item.scf.energy_change;
      output.density_rms = item.scf.density_rms;
      output.converged = item.scf.converged ? 1 : 0;
      output.executed_backend = item.executed_backend;
      output.bucket_id = static_cast<std::uint32_t>(item.bucket_id);
      output.warm_start_used = item.warm_start_used ? 1 : 0;
      output.warm_start_fallback = item.warm_start_fallback ? 1 : 0;
      if (valid_force_buffer && item.status == VIBEQC_STATUS_SUCCESS) {
        std::copy(item.scf.forces.begin(), item.scf.forces.end(), output.forces);
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

}  // extern "C"

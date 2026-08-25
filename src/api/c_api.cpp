#include "qce/qce.h"

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

struct qce_context {
  qce::core::ContextState state;
  std::string last_detail;
};

struct qce_system {
  qce::core::System data;
};

struct qce_calculation {
  qce_context* context{};
  qce::core::System system;
  qce::core::ScfOptions options;
  qce_method method{QCE_METHOD_RHF};
};

struct qce_batch {
  qce_context* context{};
  std::unique_ptr<qce::scf::FleetPlan> plan;
  std::vector<std::uint32_t> atom_counts;
};

namespace {

template <typename T>
bool valid_descriptor(const T* descriptor) {
  return descriptor != nullptr && descriptor->struct_size >= sizeof(T) &&
         descriptor->abi_version == QCE_ABI_VERSION;
}

qce_status map_exception() {
  try {
    throw;
  } catch (const std::bad_alloc&) {
    return QCE_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return QCE_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return QCE_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return QCE_STATUS_INTERNAL_ERROR;
  }
}

qce::core::ScfOptions scf_options(const qce_method_descriptor& descriptor) {
  qce::core::ScfOptions options;
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

qce_status validate_rhf_system(const qce::core::System& system) {
  if (system.electron_count % 2 != 0 || system.multiplicity != 1) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  return QCE_STATUS_SUCCESS;
}

qce_status validate_uhf_system(const qce::core::System& system) {
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (spin_excess < 0 || spin_excess > system.electron_count ||
      ((system.electron_count + spin_excess) & 1) != 0) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  return QCE_STATUS_SUCCESS;
}

qce_status validate_hf_system(qce_method method,
                              const qce::core::System& system) {
  return method == QCE_METHOD_UHF ? validate_uhf_system(system)
                                  : validate_rhf_system(system);
}

}  // namespace

extern "C" {

uint32_t qce_get_abi_version(void) { return QCE_ABI_VERSION; }

const char* qce_status_message(qce_status status) {
  switch (status) {
    case QCE_STATUS_SUCCESS: return "success";
    case QCE_STATUS_INVALID_ARGUMENT: return "invalid argument";
    case QCE_STATUS_ABI_MISMATCH: return "ABI mismatch";
    case QCE_STATUS_NOT_IMPLEMENTED: return "requested capability is not implemented";
    case QCE_STATUS_SCF_NOT_CONVERGED: return "SCF did not converge";
    case QCE_STATUS_NUMERICAL_FAILURE: return "numerical failure";
    case QCE_STATUS_CUDA_ERROR: return "CUDA runtime error";
    case QCE_STATUS_OUT_OF_MEMORY: return "out of memory";
    case QCE_STATUS_INTERNAL_ERROR: return "internal error";
  }
  return "unknown status";
}

qce_status qce_method_available(qce_method method, int32_t* available) {
  if (available == nullptr) return QCE_STATUS_INVALID_ARGUMENT;
  switch (method) {
    case QCE_METHOD_RHF:
    case QCE_METHOD_UHF:
      *available = 1;
      return QCE_STATUS_SUCCESS;
    case QCE_METHOD_WB97M_V:
    case QCE_METHOD_RCCSD_T:
      *available = 0;
      return QCE_STATUS_SUCCESS;
  }
  return QCE_STATUS_INVALID_ARGUMENT;
}

qce_status qce_context_create(const qce_context_descriptor* descriptor,
                              qce_context** context) {
  if (context == nullptr) return QCE_STATUS_INVALID_ARGUMENT;
  *context = nullptr;
  if (descriptor == nullptr) return QCE_STATUS_INVALID_ARGUMENT;
  if (descriptor->abi_version != QCE_ABI_VERSION ||
      descriptor->struct_size < sizeof(qce_context_descriptor)) {
    return QCE_STATUS_ABI_MISMATCH;
  }
  try {
    auto candidate = std::make_unique<qce_context>();
    candidate->state.device_id = descriptor->device_id;
    candidate->state.requested_backend = descriptor->backend;
    const qce_status status =
        qce::runtime::initialize_context(candidate->state, candidate->last_detail);
    if (status != QCE_STATUS_SUCCESS) return status;
    *context = candidate.release();
    return QCE_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

void qce_context_destroy(qce_context* context) { delete context; }

qce_status qce_system_create(qce_context* context,
                             const qce_system_descriptor* descriptor,
                             qce_system** system) {
  if (context == nullptr || system == nullptr || descriptor == nullptr) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  *system = nullptr;
  if (descriptor->abi_version != QCE_ABI_VERSION ||
      descriptor->struct_size <
          offsetof(qce_system_descriptor, basis_representation)) {
    return QCE_STATUS_ABI_MISMATCH;
  }
  if (descriptor->atoms == nullptr || descriptor->shells == nullptr ||
      descriptor->primitives == nullptr || descriptor->atom_count == 0 ||
      descriptor->shell_count == 0) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto candidate = std::make_unique<qce_system>();
    candidate->data.charge = descriptor->charge;
    candidate->data.multiplicity = descriptor->multiplicity;
    candidate->data.basis_representation =
        descriptor->struct_size >= sizeof(qce_system_descriptor)
        ? descriptor->basis_representation
        : QCE_BASIS_CARTESIAN;
    candidate->data.atoms.reserve(descriptor->atom_count);
    for (std::uint32_t i = 0; i < descriptor->atom_count; ++i) {
      const qce_atom& atom = descriptor->atoms[i];
      candidate->data.atoms.push_back(
          {atom.atomic_number, {atom.x, atom.y, atom.z}});
    }
    candidate->data.shells.reserve(descriptor->shell_count);
    for (std::uint32_t i = 0; i < descriptor->shell_count; ++i) {
      const qce_shell& shell = descriptor->shells[i];
      if (shell.primitive_count == 0 ||
          shell.primitive_offset > descriptor->primitive_count ||
          shell.primitive_count >
              descriptor->primitive_count - shell.primitive_offset) {
        return QCE_STATUS_INVALID_ARGUMENT;
      }
      qce::core::Shell native_shell;
      native_shell.atom_index = shell.atom_index;
      native_shell.angular_momentum = shell.angular_momentum;
      native_shell.primitives.reserve(shell.primitive_count);
      for (std::uint32_t p = 0; p < shell.primitive_count; ++p) {
        const qce_primitive& primitive =
            descriptor->primitives[shell.primitive_offset + p];
        native_shell.primitives.push_back({primitive.exponent, primitive.coefficient});
      }
      candidate->data.shells.push_back(std::move(native_shell));
    }
    const qce_status status =
        qce::molecule::validate_and_normalize(candidate->data, context->last_detail);
    if (status != QCE_STATUS_SUCCESS) return status;
    *system = candidate.release();
    return QCE_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

void qce_system_destroy(qce_system* system) { delete system; }

qce_status qce_calculation_prepare(qce_context* context,
                                   const qce_system* system,
                                   const qce_method_descriptor* descriptor,
                                   qce_calculation** calculation) {
  if (context == nullptr || system == nullptr || descriptor == nullptr ||
      calculation == nullptr) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  *calculation = nullptr;
  if (descriptor->abi_version != QCE_ABI_VERSION ||
      descriptor->struct_size < sizeof(qce_method_descriptor)) {
    return QCE_STATUS_ABI_MISMATCH;
  }
  if (descriptor->method != QCE_METHOD_RHF &&
      descriptor->method != QCE_METHOD_UHF) {
    int available = 0;
    if (qce_method_available(descriptor->method, &available) != QCE_STATUS_SUCCESS) {
      return QCE_STATUS_INVALID_ARGUMENT;
    }
    return QCE_STATUS_NOT_IMPLEMENTED;
  }
  if (validate_hf_system(descriptor->method, system->data) !=
      QCE_STATUS_SUCCESS) {
    context->last_detail = descriptor->method == QCE_METHOD_UHF
        ? "UHF requires electron count and multiplicity to define integral spin occupations"
        : "RHF requires an even electron count and spin multiplicity 1";
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto candidate = std::make_unique<qce_calculation>();
    candidate->context = context;
    candidate->system = system->data;
    candidate->method = descriptor->method;
    candidate->options = scf_options(*descriptor);
    *calculation = candidate.release();
    return QCE_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

void qce_calculation_destroy(qce_calculation* calculation) { delete calculation; }

qce_status qce_calculation_execute(qce_calculation* calculation,
                                   qce_result_descriptor* result) {
  if (calculation == nullptr || result == nullptr) return QCE_STATUS_INVALID_ARGUMENT;
  if (result->abi_version != QCE_ABI_VERSION ||
      result->struct_size < sizeof(qce_result_descriptor)) {
    return QCE_STATUS_ABI_MISMATCH;
  }
  const std::size_t required_forces = calculation->system.atoms.size() * 3;
  if (result->forces == nullptr || result->force_count < required_forces) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  try {
    const bool use_cuda =
        calculation->context->state.requested_backend == QCE_BACKEND_CUDA;
    const bool unrestricted = calculation->method == QCE_METHOD_UHF;
    const qce::core::ScfResult native = use_cuda
        ? (unrestricted
               ? qce::scf::run_uhf_cuda(
                     calculation->system, calculation->options,
                     calculation->context->state.device_id)
               : qce::scf::run_rhf_cuda(
                     calculation->system, calculation->options,
                     calculation->context->state.device_id))
        : (unrestricted
               ? qce::scf::run_uhf(calculation->system, calculation->options)
               : qce::scf::run_rhf(calculation->system, calculation->options));
    result->energy = native.energy;
    result->iterations = native.iterations;
    result->energy_change = native.energy_change;
    result->density_rms = native.density_rms;
    result->converged = native.converged ? 1 : 0;
    result->executed_backend = use_cuda ? QCE_BACKEND_CUDA
                                        : QCE_BACKEND_CPU_REFERENCE;
    if (!native.converged) return QCE_STATUS_SCF_NOT_CONVERGED;
    std::copy(native.forces.begin(), native.forces.end(), result->forces);
    return QCE_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

qce_status qce_batch_prepare(qce_context* context,
                             const qce_system* const* systems,
                             uint32_t system_count,
                             const qce_method_descriptor* descriptor,
                             qce_batch_flags flags,
                             qce_batch** batch) {
  if (context == nullptr || systems == nullptr || system_count == 0 ||
      descriptor == nullptr || batch == nullptr) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  *batch = nullptr;
  if (!valid_descriptor(descriptor)) return QCE_STATUS_ABI_MISMATCH;
  if (descriptor->method != QCE_METHOD_RHF &&
      descriptor->method != QCE_METHOD_UHF) {
    int available = 0;
    if (qce_method_available(descriptor->method, &available) != QCE_STATUS_SUCCESS) {
      return QCE_STATUS_INVALID_ARGUMENT;
    }
    return QCE_STATUS_NOT_IMPLEMENTED;
  }
  constexpr qce_batch_flags supported_flags =
      QCE_BATCH_ENABLE_WARM_STARTS |
      QCE_BATCH_ENABLE_SHELL_CLASS_PROFILING;
  if ((flags & ~supported_flags) != 0) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  try {
    std::vector<qce::core::System> native_systems;
    native_systems.reserve(system_count);
    std::vector<std::uint32_t> atom_counts;
    atom_counts.reserve(system_count);
    for (std::uint32_t i = 0; i < system_count; ++i) {
      if (systems[i] == nullptr ||
          validate_hf_system(descriptor->method, systems[i]->data) !=
              QCE_STATUS_SUCCESS) {
        return QCE_STATUS_INVALID_ARGUMENT;
      }
      native_systems.push_back(systems[i]->data);
      atom_counts.push_back(static_cast<std::uint32_t>(systems[i]->data.atoms.size()));
    }
    auto candidate = std::make_unique<qce_batch>();
    candidate->context = context;
    candidate->atom_counts = std::move(atom_counts);
    candidate->plan = std::make_unique<qce::scf::FleetPlan>(
        std::move(native_systems), descriptor->method, scf_options(*descriptor),
        (flags & QCE_BATCH_ENABLE_WARM_STARTS) != 0,
        context->state.requested_backend == QCE_BACKEND_CUDA,
        (flags & QCE_BATCH_ENABLE_SHELL_CLASS_PROFILING) != 0,
        context->state.device_id);
    *batch = candidate.release();
    return QCE_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

void qce_batch_destroy(qce_batch* batch) { delete batch; }

uint32_t qce_batch_get_system_count(const qce_batch* batch) {
  if (batch == nullptr || batch->plan == nullptr) return 0;
  return static_cast<std::uint32_t>(batch->plan->size());
}

qce_status qce_batch_get_last_shell_class_profile(
    const qce_batch* batch,
    qce_shell_class_profile_entry* entries,
    uint32_t entry_count) {
  if (batch == nullptr || batch->plan == nullptr || entries == nullptr ||
      entry_count < QCE_DIRECT_SHELL_CLASS_COUNT) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  const auto& profile = batch->plan->last_shell_class_profile();
  if (!profile.has_value()) return QCE_STATUS_NOT_IMPLEMENTED;
  static_assert(qce::scf::detail::kDirectQuartetShellClassCount ==
                QCE_DIRECT_SHELL_CLASS_COUNT);
  for (std::size_t shell_class = 0; shell_class < profile->size();
       ++shell_class) {
    entries[shell_class] = {
        (*profile)[shell_class].shell_quartets,
        (*profile)[shell_class].tiles,
        (*profile)[shell_class].ao_quartets,
        (*profile)[shell_class].primitive_quartets,
    };
  }
  return QCE_STATUS_SUCCESS;
}

qce_status qce_batch_clear_warm_starts(qce_batch* batch) {
  if (batch == nullptr || batch->plan == nullptr) return QCE_STATUS_INVALID_ARGUMENT;
  batch->plan->clear_warm_starts();
  return QCE_STATUS_SUCCESS;
}

qce_status qce_batch_execute(qce_batch* batch,
                             const qce_batch_input_descriptor* inputs,
                             uint32_t input_count,
                             qce_batch_item_result_descriptor* results,
                             uint32_t result_count) {
  if (batch == nullptr || batch->plan == nullptr || results == nullptr) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  const std::uint32_t system_count = qce_batch_get_system_count(batch);
  if (result_count != system_count ||
      ((inputs == nullptr) != (input_count == 0)) ||
      (inputs != nullptr && input_count != system_count)) {
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  for (std::uint32_t i = 0; i < result_count; ++i) {
    if (!valid_descriptor(&results[i])) return QCE_STATUS_ABI_MISMATCH;
  }
  if (inputs != nullptr) {
    for (std::uint32_t i = 0; i < input_count; ++i) {
      if (!valid_descriptor(&inputs[i])) return QCE_STATUS_ABI_MISMATCH;
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

    const std::vector<qce::scf::FleetItemResult> native =
        batch->plan->execute(coordinates);
    for (std::uint32_t i = 0; i < system_count; ++i) {
      qce_batch_item_result_descriptor& output = results[i];
      const qce::scf::FleetItemResult& item = native[i];
      const std::uint32_t required_forces = batch->atom_counts[i] * 3;
      const bool valid_force_buffer =
          output.forces != nullptr && output.force_count >= required_forces;
      output.status = valid_force_buffer ? item.status : QCE_STATUS_INVALID_ARGUMENT;
      output.energy = item.scf.energy;
      output.iterations = item.scf.iterations;
      output.energy_change = item.scf.energy_change;
      output.density_rms = item.scf.density_rms;
      output.converged = item.scf.converged ? 1 : 0;
      output.executed_backend = item.executed_backend;
      output.bucket_id = static_cast<std::uint32_t>(item.bucket_id);
      output.warm_start_used = item.warm_start_used ? 1 : 0;
      output.warm_start_fallback = item.warm_start_fallback ? 1 : 0;
      if (valid_force_buffer && item.status == QCE_STATUS_SUCCESS) {
        std::copy(item.scf.forces.begin(), item.scf.forces.end(), output.forces);
      }
    }
    return QCE_STATUS_SUCCESS;
  } catch (...) {
    return map_exception();
  }
}

}  // extern "C"

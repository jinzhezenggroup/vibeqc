#include <cstddef>
#include <memory>
#include <utility>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "molecule/basis.hpp"
#include "runtime/context.hpp"
#include "vibeqc/vibeqc.h"

extern "C" {

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
    return vibeqc::api::map_exception();
  }
}

void vibeqc_context_destroy(vibeqc_context* context) { delete context; }

const char* vibeqc_context_get_last_detail(const vibeqc_context* context) {
  if (context == nullptr) return "invalid context";
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  return context->last_detail.c_str();
}

const char* vibeqc_context_last_error(const vibeqc_context* context) {
  return vibeqc_context_get_last_detail(context);
}

vibeqc_status vibeqc_system_create(vibeqc_context* context,
                                   const vibeqc_system_descriptor* descriptor,
                                   vibeqc_system** system) {
  if (context == nullptr || system == nullptr || descriptor == nullptr) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  *system = nullptr;
  if (descriptor->abi_version != VIBEQC_ABI_VERSION ||
      descriptor->struct_size < offsetof(vibeqc_system_descriptor, basis_representation)) {
    return VIBEQC_STATUS_ABI_MISMATCH;
  }
  if (descriptor->atoms == nullptr || descriptor->shells == nullptr ||
      descriptor->primitives == nullptr || descriptor->atom_count == 0 ||
      descriptor->shell_count == 0) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  std::lock_guard<std::recursive_mutex> context_lock(context->mutex);
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
      candidate->data.atoms.push_back({atom.atomic_number, {atom.x, atom.y, atom.z}});
    }
    candidate->data.shells.reserve(descriptor->shell_count);
    for (std::uint32_t i = 0; i < descriptor->shell_count; ++i) {
      const vibeqc_shell& shell = descriptor->shells[i];
      if (context->state.executed_backend == VIBEQC_BACKEND_CUDA && shell.angular_momentum > 3) {
        context->last_detail = "CUDA basis execution supports l<=3; g shells require CPU reference";
        return VIBEQC_STATUS_NOT_IMPLEMENTED;
      }
      if (shell.primitive_count == 0 || shell.primitive_offset > descriptor->primitive_count ||
          shell.primitive_count > descriptor->primitive_count - shell.primitive_offset) {
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      vibeqc::core::Shell native_shell;
      native_shell.atom_index = shell.atom_index;
      native_shell.angular_momentum = shell.angular_momentum;
      native_shell.primitives.reserve(shell.primitive_count);
      for (std::uint32_t p = 0; p < shell.primitive_count; ++p) {
        const vibeqc_primitive& primitive = descriptor->primitives[shell.primitive_offset + p];
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
    return vibeqc::api::map_exception(&context->last_detail);
  }
}

void vibeqc_system_destroy(vibeqc_system* system) { delete system; }

}  // extern "C"

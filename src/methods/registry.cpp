#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "methods/hf_method.hpp"
#include "methods/method.hpp"

namespace vibeqc::methods {
namespace {

using ValidateSystem = vibeqc_status (*)(vibeqc_method, const core::System&, std::string&);
using PrepareCalculation = std::unique_ptr<PreparedCalculation> (*)(
    const Capabilities&, core::ContextState&, const core::System&, const vibeqc_method_descriptor&);
using PrepareBatch = std::unique_ptr<PreparedBatch> (*)(const Capabilities&, core::ContextState&,
                                                        std::vector<core::System>,
                                                        const vibeqc_method_descriptor&,
                                                        vibeqc_batch_flags);

struct MethodDefinition {
  Capabilities capabilities;
  ValidateSystem validate_system{};
  PrepareCalculation prepare_calculation{};
  PrepareBatch prepare_batch{};
};

constexpr vibeqc_property_flags kEnergyAndForces = VIBEQC_PROPERTY_ENERGY | VIBEQC_PROPERTY_FORCES;

const std::array<MethodDefinition, 4> kMethods{{
    {{VIBEQC_METHOD_RHF, VIBEQC_METHOD_FAMILY_HARTREE_FOCK, kEnergyAndForces, true, true},
     detail::validate_hf_system,
     detail::prepare_hf_calculation,
     detail::prepare_hf_batch},
    {{VIBEQC_METHOD_UHF, VIBEQC_METHOD_FAMILY_HARTREE_FOCK, kEnergyAndForces, true, true},
     detail::validate_hf_system,
     detail::prepare_hf_calculation,
     detail::prepare_hf_batch},
    {{VIBEQC_METHOD_WB97M_V, VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL, 0, false, false},
     nullptr,
     nullptr,
     nullptr},
    {{VIBEQC_METHOD_RCCSD_T, VIBEQC_METHOD_FAMILY_COUPLED_CLUSTER, 0, false, false},
     nullptr,
     nullptr,
     nullptr},
}};

const MethodDefinition* find_definition(vibeqc_method method) noexcept {
  const auto found = std::find_if(
      kMethods.begin(), kMethods.end(),
      [method](const MethodDefinition& item) { return item.capabilities.method == method; });
  return found == kMethods.end() ? nullptr : &*found;
}

const MethodDefinition& require_available(vibeqc_method method) {
  const MethodDefinition* definition = find_definition(method);
  if (definition == nullptr) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown method identifier");
  }
  if (!definition->capabilities.available) {
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "requested method is reserved but not implemented");
  }
  if (definition->validate_system == nullptr || definition->prepare_calculation == nullptr) {
    throw MethodError(VIBEQC_STATUS_INTERNAL_ERROR,
                      "available method has an incomplete registry definition");
  }
  return *definition;
}

void validate_system(const MethodDefinition& definition, const core::System& system) {
  std::string detail;
  const vibeqc_status status =
      definition.validate_system(definition.capabilities.method, system, detail);
  if (status != VIBEQC_STATUS_SUCCESS) {
    throw MethodError(status, detail.empty() ? "method rejected the system" : detail);
  }
}

}  // namespace

const Capabilities* find_capabilities(vibeqc_method method) noexcept {
  const MethodDefinition* definition = find_definition(method);
  return definition == nullptr ? nullptr : &definition->capabilities;
}

std::unique_ptr<PreparedCalculation> prepare_calculation(
    core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
  const MethodDefinition& definition = require_available(descriptor.method);
  validate_system(definition, system);
  return definition.prepare_calculation(definition.capabilities, context, system, descriptor);
}

std::unique_ptr<PreparedBatch> prepare_batch(core::ContextState& context,
                                             std::vector<core::System> systems,
                                             const vibeqc_method_descriptor& descriptor,
                                             vibeqc_batch_flags flags) {
  const MethodDefinition& definition = require_available(descriptor.method);
  if (!definition.capabilities.supports_batch || definition.prepare_batch == nullptr) {
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "requested method does not support prepared batches");
  }
  for (const core::System& system : systems) validate_system(definition, system);
  return definition.prepare_batch(definition.capabilities, context, std::move(systems), descriptor,
                                  flags);
}

}  // namespace vibeqc::methods

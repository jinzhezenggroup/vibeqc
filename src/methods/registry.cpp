#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "methods/dft_method.hpp"
#include "methods/hf_method.hpp"
#include "methods/method.hpp"
#include "methods/mp2_method.hpp"
#include "runtime/provider_registry.hpp"

namespace vibeqc::methods {
namespace {

using ValidateSystem = vibeqc_status (*)(vibeqc_method, const core::System&, std::string&);
using PrepareCalculation = std::unique_ptr<PreparedCalculation> (*)(
    const Capabilities&, core::ContextState&, const core::System&, const vibeqc_method_descriptor&);
using PrepareBatch = std::unique_ptr<PreparedBatch> (*)(const Capabilities&, core::ContextState&,
                                                        std::vector<core::System>,
                                                        const vibeqc_method_descriptor&,
                                                        vibeqc_batch_flags);

using MethodProviderRegistration = runtime::ProviderDescriptor<Capabilities>;

struct MethodDefinition {
  MethodProviderRegistration provider;
  ValidateSystem validate_system{};
  PrepareCalculation prepare_calculation{};
  PrepareBatch prepare_batch{};
};

constexpr vibeqc_property_flags kEnergyAndForces = VIBEQC_PROPERTY_ENERGY | VIBEQC_PROPERTY_FORCES;

constexpr MethodDefinition register_method(std::string_view name, vibeqc_method method,
                                           vibeqc_method_family family,
                                           vibeqc_property_flags properties,
                                           ValidateSystem validate, PrepareCalculation prepare,
                                           PrepareBatch batch,
                                           std::string_view unavailable_reason = {}) {
  const bool executable = validate != nullptr && prepare != nullptr;
  const auto availability = executable ? runtime::ProviderAvailability::Executable
                                       : runtime::ProviderAvailability::Reserved;
  const auto registered_properties = executable ? properties : vibeqc_property_flags{};
  return {{{"methods", name, 1, runtime::ProviderBackend::Any},
           {method, family, registered_properties, executable, executable && batch != nullptr},
           availability,
           runtime::ProviderFallback::None,
           runtime::ProviderRequirement::PreparedState,
           unavailable_reason,
           "src/methods/registry.cpp"},
          validate,
          prepare,
          batch};
}

constexpr std::array<MethodDefinition, 9> kMethods{{
    register_method("mp2", VIBEQC_METHOD_MP2, VIBEQC_METHOD_FAMILY_PERTURBATION, kEnergyAndForces,
                    detail::validate_mp2_system, detail::prepare_mp2_calculation,
                    detail::prepare_mp2_batch),
    register_method("rhf", VIBEQC_METHOD_RHF, VIBEQC_METHOD_FAMILY_HARTREE_FOCK, kEnergyAndForces,
                    detail::validate_hf_system, detail::prepare_hf_calculation,
                    detail::prepare_hf_batch),
    register_method("uhf", VIBEQC_METHOD_UHF, VIBEQC_METHOD_FAMILY_HARTREE_FOCK, kEnergyAndForces,
                    detail::validate_hf_system, detail::prepare_hf_calculation,
                    detail::prepare_hf_batch),
    register_method("wb97m-v", VIBEQC_METHOD_WB97M_V, VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL, 0,
                    nullptr, nullptr, nullptr, "method has no executable provider registration"),
    register_method("rccsd(t)", VIBEQC_METHOD_RCCSD_T, VIBEQC_METHOD_FAMILY_COUPLED_CLUSTER, 0,
                    nullptr, nullptr, nullptr, "method has no executable provider registration"),
    register_method("lda-rks", VIBEQC_METHOD_LDA_RKS, VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL,
                    VIBEQC_PROPERTY_ENERGY, detail::validate_dft_system,
                    detail::prepare_dft_calculation, detail::prepare_dft_batch),
    register_method("pbe-rks", VIBEQC_METHOD_PBE_RKS, VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL,
                    VIBEQC_PROPERTY_ENERGY, detail::validate_dft_system,
                    detail::prepare_dft_calculation, detail::prepare_dft_batch),
    register_method("lda-uks", VIBEQC_METHOD_LDA_UKS, VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL,
                    VIBEQC_PROPERTY_ENERGY, detail::validate_dft_system,
                    detail::prepare_dft_calculation, detail::prepare_dft_batch),
    register_method("pbe-uks", VIBEQC_METHOD_PBE_UKS, VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL,
                    VIBEQC_PROPERTY_ENERGY, detail::validate_dft_system,
                    detail::prepare_dft_calculation, detail::prepare_dft_batch),
}};

const MethodDefinition* find_definition(vibeqc_method method) noexcept {
  const auto found = std::find_if(
      kMethods.begin(), kMethods.end(),
      [method](const MethodDefinition& item) { return item.provider.domain.method == method; });
  return found == kMethods.end() ? nullptr : &*found;
}

const MethodDefinition& require_available(vibeqc_method method) {
  const MethodDefinition* definition = find_definition(method);
  if (definition == nullptr) {
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown method identifier");
  }
  if (!runtime::provider_executable(definition->provider)) {
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      runtime::provider_diagnostic(definition->provider, "method preparation"));
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
      definition.validate_system(definition.provider.domain.method, system, detail);
  if (status != VIBEQC_STATUS_SUCCESS) {
    throw MethodError(status, detail.empty() ? "method rejected the system" : detail);
  }
}

void validate_option_family(const MethodDefinition& definition,
                            const vibeqc_method_descriptor& descriptor) {
  if (descriptor.struct_size >=
          offsetof(vibeqc_method_descriptor, ks_options) + sizeof(descriptor.ks_options) &&
      descriptor.ks_options &&
      definition.provider.domain.family != VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "KS options require the DFT method family");
}

}  // namespace

const Capabilities* find_capabilities(vibeqc_method method) noexcept {
  const MethodDefinition* definition = find_definition(method);
  return definition == nullptr ? nullptr : &definition->provider.domain;
}

std::unique_ptr<PreparedCalculation> prepare_calculation(
    core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
  const MethodDefinition& definition = require_available(descriptor.method);
  validate_option_family(definition, descriptor);
  validate_system(definition, system);
  return definition.prepare_calculation(definition.provider.domain, context, system, descriptor);
}

std::unique_ptr<PreparedBatch> prepare_batch(core::ContextState& context,
                                             std::vector<core::System> systems,
                                             const vibeqc_method_descriptor& descriptor,
                                             vibeqc_batch_flags flags) {
  const MethodDefinition& definition = require_available(descriptor.method);
  validate_option_family(definition, descriptor);
  if (!definition.provider.domain.supports_batch || definition.prepare_batch == nullptr) {
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "requested method does not support prepared batches");
  }
  for (const core::System& system : systems) validate_system(definition, system);
  return definition.prepare_batch(definition.provider.domain, context, std::move(systems),
                                  descriptor, flags);
}

}  // namespace vibeqc::methods

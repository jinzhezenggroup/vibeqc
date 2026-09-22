#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "methods/dft_method.hpp"
#include "methods/generated_method_manifest.hpp"
#include "methods/hf_method.hpp"
#include "methods/method.hpp"
#include "methods/mp2_method.hpp"
#include "methods/rccsd_method.hpp"
#include "methods/rccsdt_method.hpp"
#include "methods/xtb_method.hpp"
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

constexpr MethodDefinition register_method(const generated::MethodManifestEntry& manifest) {
  ValidateSystem validate = nullptr;
  PrepareCalculation prepare = nullptr;
  PrepareBatch batch = nullptr;
  switch (manifest.provider) {
    case generated::PublicProvider::Hf:
      validate = detail::validate_hf_system;
      prepare = detail::prepare_hf_calculation;
      batch = detail::prepare_hf_batch;
      break;
    case generated::PublicProvider::Mp2:
      validate = detail::validate_mp2_system;
      prepare = detail::prepare_mp2_calculation;
      batch = detail::prepare_mp2_batch;
      break;
    case generated::PublicProvider::Rccsd:
      validate = detail::validate_rccsd_system;
      prepare = detail::prepare_rccsd_calculation;
      batch = detail::prepare_rccsd_batch;
      break;
    case generated::PublicProvider::Rccsdt:
      validate = detail::validate_rccsdt_system;
      prepare = detail::prepare_rccsdt_calculation;
      batch = detail::prepare_rccsdt_batch;
      break;
    case generated::PublicProvider::Dft:
      validate = detail::validate_dft_system;
      prepare = detail::prepare_dft_calculation;
      batch = detail::prepare_dft_batch;
      break;
    case generated::PublicProvider::Xtb:
      validate = detail::validate_xtb_system;
      prepare = detail::prepare_xtb_calculation;
      batch = nullptr;
      break;
    case generated::PublicProvider::Reserved:
      break;
  }

  const bool executable = validate != nullptr && prepare != nullptr;
  const bool supports_batch = executable && batch != nullptr;
  if (manifest.supports_batch != supports_batch)
    throw "public method manifest batch capability disagrees with native provider";
  if ((manifest.properties != 0) != executable)
    throw "public method manifest properties disagree with native provider";
  const auto availability = executable ? runtime::ProviderAvailability::Executable
                                       : runtime::ProviderAvailability::Reserved;
  const auto registered_properties = executable ? manifest.properties : vibeqc_property_flags{};
  return {{{"methods", manifest.name, 1, runtime::ProviderBackend::Any},
           {manifest.method, manifest.family, registered_properties, executable, supports_batch},
           availability,
           runtime::ProviderFallback::None,
           runtime::ProviderRequirement::PreparedState,
           manifest.unavailable_reason,
           "manifests/public_methods.json"},
          validate,
          prepare,
          batch};
}

constexpr auto build_methods() {
  std::array<MethodDefinition, generated::kMethodManifest.size()> result{};
  for (std::size_t i = 0; i < generated::kMethodManifest.size(); ++i)
    result[i] = register_method(generated::kMethodManifest[i]);
  return result;
}

constexpr auto kMethods = build_methods();

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

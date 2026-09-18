#ifndef VIBEQC_RUNTIME_PROVIDER_REGISTRY_HPP
#define VIBEQC_RUNTIME_PROVIDER_REGISTRY_HPP

#include <cstdint>
#include <string>
#include <string_view>

#include "vibeqc/vibeqc.h"

namespace vibeqc::runtime {

/** Execution identity shared by provider registries without owning a method's
 * scientific selection. `Any` is for method-level registrations whose concrete
 * backend is selected later by that method's own policy. */
enum class ProviderBackend : std::uint8_t { Any, Cpu, Cuda };
enum class ProviderAvailability : std::uint8_t { Executable, NotBuilt, Reserved, Unavailable };
enum class ProviderFallback : std::uint8_t { None, ExplicitReference, PerformanceException };
enum class ProviderRequirement : std::uint32_t {
  None = 0,
  PreparedState = 1U << 0,
  Resources = 1U << 1,
};

constexpr ProviderRequirement operator|(ProviderRequirement a, ProviderRequirement b) noexcept {
  return static_cast<ProviderRequirement>(static_cast<std::uint32_t>(a) |
                                          static_cast<std::uint32_t>(b));
}
constexpr bool has_requirement(ProviderRequirement set, ProviderRequirement value) noexcept {
  return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(value)) != 0;
}

struct ProviderIdentity {
  std::string_view subsystem;
  std::string_view name;
  std::uint32_t version{1};
  ProviderBackend backend{ProviderBackend::Any};
  bool operator==(const ProviderIdentity&) const = default;
};

/** Common execution metadata wrapped around a subsystem-owned typed domain.
 * Domain remains method-specific: the runtime registry never interprets
 * operators, approximations, spin/reference conventions, or numerical policy. */
template <class Domain>
struct ProviderDescriptor {
  ProviderIdentity identity;
  Domain domain;
  ProviderAvailability availability{ProviderAvailability::Unavailable};
  ProviderFallback fallback{ProviderFallback::None};
  ProviderRequirement requirements{ProviderRequirement::None};
  std::string_view reason;
  std::string_view provenance;
};

template <class Domain>
constexpr bool provider_executable(const ProviderDescriptor<Domain>& provider) noexcept {
  return provider.availability == ProviderAvailability::Executable;
}

inline constexpr std::string_view provider_backend_name(ProviderBackend backend) noexcept {
  switch (backend) {
    case ProviderBackend::Any:
      return "any";
    case ProviderBackend::Cpu:
      return "cpu";
    case ProviderBackend::Cuda:
      return "cuda";
  }
  return "unknown";
}
inline constexpr std::string_view provider_availability_name(
    ProviderAvailability availability) noexcept {
  switch (availability) {
    case ProviderAvailability::Executable:
      return "executable";
    case ProviderAvailability::NotBuilt:
      return "not built";
    case ProviderAvailability::Reserved:
      return "reserved";
    case ProviderAvailability::Unavailable:
      return "unavailable";
  }
  return "unknown";
}

template <class Domain>
std::string provider_diagnostic(const ProviderDescriptor<Domain>& provider,
                                std::string_view operation = {}) {
  std::string detail;
  if (!operation.empty()) {
    detail.append(operation);
    detail.append(": ");
  }
  detail.append(provider.identity.subsystem);
  detail.append(" provider '");
  detail.append(provider.identity.name);
  detail.append("' v");
  detail.append(std::to_string(provider.identity.version));
  detail.append(" [");
  detail.append(provider_backend_name(provider.identity.backend));
  detail.append("] is ");
  detail.append(provider_availability_name(provider.availability));
  if (!provider.reason.empty()) {
    detail.append(": ");
    detail.append(provider.reason);
  }
  return detail;
}

/** Shared ABI-stable not-built stub result. Entry points retain their explicit
 * signatures; only status/detail formatting is centralized. */
template <class Domain>
vibeqc_status provider_not_implemented(const ProviderDescriptor<Domain>& provider,
                                       std::string& detail, std::string_view operation = {}) {
  detail = provider_diagnostic(provider, operation);
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}

}  // namespace vibeqc::runtime

#endif

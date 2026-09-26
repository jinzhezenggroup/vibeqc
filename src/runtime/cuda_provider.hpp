#ifndef VIBEQC_RUNTIME_CUDA_PROVIDER_HPP
#define VIBEQC_RUNTIME_CUDA_PROVIDER_HPP

#include <cstdint>
#include <string_view>

namespace vibeqc::runtime {

enum class CudaProviderKind : std::uint8_t { None, Nvidia, CuMetal };

struct CudaProviderCapabilities {
  CudaProviderKind kind{CudaProviderKind::None};
  std::string_view identity{"none"};
  bool templated_shell_warp_one_electron{};
  bool operator==(const CudaProviderCapabilities&) const = default;
};

constexpr CudaProviderCapabilities cuda_provider_capabilities(CudaProviderKind kind) noexcept {
  switch (kind) {
    case CudaProviderKind::Nvidia:
      return {kind, "nvidia", true};
    case CudaProviderKind::CuMetal:
      return {kind, "cumetal", false};
    case CudaProviderKind::None:
      return {kind, "none", false};
  }
  return {};
}

constexpr std::string_view cuda_provider_name(CudaProviderKind kind) noexcept {
  return cuda_provider_capabilities(kind).identity;
}

/** Provider selected by the configured execution image, never by environment setup. */
const CudaProviderCapabilities& active_cuda_provider() noexcept;

}  // namespace vibeqc::runtime

#endif

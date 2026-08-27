#include "scf/aot_shell_registry.hpp"

namespace vibeqc::scf::generated {
namespace {

constexpr ProfileInfo kGenericProfile{
    "generic_cuda", "portable_cuda", 0, 0, false, true, false};

}  // namespace

void select_profile_for_device(int, int, int) noexcept {}

const ProfileInfo& selected_profile() noexcept { return kGenericProfile; }

const ShellKernelMetadata* selected_shell_kernels(
    std::size_t& count) noexcept {
  count = 0;
  return nullptr;
}

const ShellKernelMetadata* selected_fock_shell_kernels(
    std::size_t& count) noexcept {
  count = 0;
  return nullptr;
}

std::uint64_t enabled_shell_class_mask() noexcept { return 0; }

std::uint64_t enabled_fock_shell_class_mask() noexcept { return 0; }

cudaError_t launch_shell_class(
    unsigned, cudaStream_t, bool, unsigned, const void*, const std::uint32_t*,
    const std::int64_t*, const void*, const double*, const void*, double,
    const double*, const double*, double*, const std::uint32_t*,
    std::uint32_t*) noexcept {
  return cudaErrorInvalidValue;
}

cudaError_t launch_shell_class_fock(
    unsigned, cudaStream_t, bool, unsigned, const void*, const std::uint32_t*,
    const std::int64_t*, const void*, const double*, const void*, double,
    const double*, const double*, double*, const std::uint32_t*,
    std::uint32_t*) noexcept {
  return cudaErrorInvalidValue;
}

cudaError_t launch_ppps_resident(
    cudaStream_t, bool, const void*, const void*, const std::int64_t*,
    const void*, const double*, const void*, double, const double*,
    const double*, double*, std::size_t) noexcept {
  return cudaErrorNotSupported;
}

}  // namespace vibeqc::scf::generated

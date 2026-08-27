#include "scf/aot_shell_registry.hpp"

#include <cuda_runtime_api.h>

#include <cstring>
#include <iostream>

int main() {
  if (cudaSetDevice(0) != cudaSuccess) return 1;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 2;

  // Context initialization uses the same one-time per-device selector. Calling
  // it explicitly here makes profile identity and fallback behavior observable.
  vibeqc::scf::generated::select_profile_for_device(
      0, properties.major, properties.minor);
  const auto& profile = vibeqc::scf::generated::selected_profile();
  std::cout << profile.name << ' ' << profile.target_architecture << '\n';

  if (properties.major == 12 && properties.minor == 0) {
    if (std::strcmp(profile.name, "sm_120") != 0 || !profile.tuned ||
        profile.portable || profile.compatible) {
      return 3;
    }
    if (vibeqc::scf::generated::enabled_shell_class_mask() == 0) return 4;
  } else if (profile.tuned) {
    // A binary must never select the RTX 5090 schedule on another SM.
    return 5;
  }
  return 0;
}

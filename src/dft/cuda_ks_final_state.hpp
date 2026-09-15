#pragma once

#include <cstdint>

#include "dft/ks_final_state.hpp"

namespace vibeqc::dft {

/** Detached authorization for exactly one successful solve on one immutable
 * resident CUDA KS owner. A later begin invalidates it before CUDA work. */
struct CudaKsFinalStateToken {
  std::uint32_t version{1};
  KsFinalStateIdentity identity;
  bool operator==(const CudaKsFinalStateToken&) const = default;
};

}  // namespace vibeqc::dft

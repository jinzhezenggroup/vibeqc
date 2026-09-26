#pragma once

#include <cstdint>

#include "dft/ks_final_state.hpp"

namespace vibeqc::dft {

/** Detached authorization for exactly one successful solve on one immutable
 * native KS owner. A later begin invalidates it before scientific work.
 * The historical type name is retained for CUDA source compatibility; backend
 * and device semantics are carried by the complete identity. */
struct CudaKsFinalStateToken {
  std::uint32_t version{1};
  KsFinalStateIdentity identity;
  bool operator==(const CudaKsFinalStateToken&) const = default;
};

}  // namespace vibeqc::dft

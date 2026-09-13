#pragma once

#include <cstdlib>
#include <cstring>

namespace vibeqc::scf {

/** CPU-safe reservation policy shared by the shape planner and CUDA owner.
 * Only the explicit opt-in reserves occupied SCF state. The execution entry
 * point separately rejects invalid values; an unset policy retains dense
 * capacity. Callers snapshot this once for each planning transaction.
 */
inline bool df_occupied_exchange_requested() noexcept {
  const char* value = std::getenv("VIBEQC_DF_EXCHANGE");
  return value != nullptr && std::strcmp(value, "occupied") == 0;
}

}  // namespace vibeqc::scf

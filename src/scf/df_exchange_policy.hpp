#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace vibeqc::scf {

/** An unset policy is auto; explicit dense/occupied remain comparison controls. */
inline bool df_occupied_exchange_auto_requested() noexcept {
  const char* value = std::getenv("VIBEQC_DF_EXCHANGE");
  return !value || std::strcmp(value, "auto") == 0;
}

/** CPU-safe conservative reservation before device, rank and plan admission.
 * Only the measured potential 768/768 batch-one domain reserves in auto mode.
 * Runtime additionally checks reference, occupied rank, full resident shape
 * and the exact measured device. Reservation alone never authorizes factors.
 */
inline bool df_occupied_exchange_requested(std::size_t nbf = 0, std::size_t naux = 0,
                                           std::size_t batch = 0) noexcept {
  const char* value = std::getenv("VIBEQC_DF_EXCHANGE");
  return (value && std::strcmp(value, "occupied") == 0) ||
         (df_occupied_exchange_auto_requested() && nbf == 768 && naux == 768 && batch == 1);
}

}  // namespace vibeqc::scf

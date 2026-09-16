#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace vibeqc::scf {

/** Diagnostic legacy contraction requires a new owner: captured J/K nodes
 * and the immutable raw-buffer lifetime must follow the same frozen policy. */
inline bool df_resident_exchange_requested() noexcept {
  const char* value = std::getenv("VIBEQC_DF_RESIDENT_EXCHANGE");
  return !value || std::strcmp(value, "auto") == 0 || std::strcmp(value, "full") == 0 ||
         std::strcmp(value, "flat") == 0;
}

inline bool df_triangular_exchange_requested() noexcept {
  const char* value = std::getenv("VIBEQC_DF_RESIDENT_EXCHANGE");
  return !value || std::strcmp(value, "auto") == 0 || std::strcmp(value, "flat") == 0;
}

/** Keep the original flattened dense contraction as an isolated ablation. */
inline bool df_flat_dense_exchange_requested() noexcept {
  const char* value = std::getenv("VIBEQC_DF_RESIDENT_EXCHANGE");
  return value && std::strcmp(value, "flat") == 0;
}

/** Freeze the DIIS reduction policy with captured SCF work. */
inline bool df_cooperative_diis_requested() noexcept {
  const char* value = std::getenv("VIBEQC_DF_DIIS_DOTS");
  return !value || std::strcmp(value, "auto") == 0;
}

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

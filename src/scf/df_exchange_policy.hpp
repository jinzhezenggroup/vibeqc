#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>

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

/** Backend-independent work policy for the resident FP64 occupied algorithm.
 * Dense K costs 4*a*n^3 FLOPs; occupied projection plus a full Gram costs
 * at most 4*a*n^2*r (SYRK can reduce this further). Require a twofold arithmetic
 * reduction to leave headroom for factor validation and smaller BLAS shapes.
 * This is a conservative work heuristic, not a device latency prediction.
 * Division keeps both the work comparison and native BLAS bounds overflow-safe.
 * Residency, storage, RHF provenance and density validity are separate gates.
 */
inline bool df_occupied_exchange_preferred(std::size_t nbf, std::size_t naux, std::size_t batch,
                                           std::size_t rank) noexcept {
  constexpr auto blas_limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  return batch == 1 && nbf >= 2 && nbf <= blas_limit / nbf && naux > 0 &&
         naux <= blas_limit / nbf && rank > 0 && rank <= nbf / 2;
}

/** Reserve automatic factors only with a method-authorized RHF occupation.
 * Zero means unknown reference/rank or UHF: those plans keep dense capacity.
 * Explicit occupied retains its conservative reservation for all references.
 */
inline bool df_occupied_exchange_requested(std::size_t nbf = 0, std::size_t naux = 0,
                                           std::size_t batch = 0,
                                           std::size_t automatic_rhf_rank = 0) noexcept {
  const char* value = std::getenv("VIBEQC_DF_EXCHANGE");
  return (value && std::strcmp(value, "occupied") == 0) ||
         (df_occupied_exchange_auto_requested() && df_resident_exchange_requested() &&
          df_occupied_exchange_preferred(nbf, naux, batch, automatic_rhf_rank));
}

}  // namespace vibeqc::scf

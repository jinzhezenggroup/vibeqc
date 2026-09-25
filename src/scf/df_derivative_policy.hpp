#pragma once

#include <cstddef>

namespace vibeqc::scf {

/** Conservative schedule profile, not a correctness/capability declaration.
 * Existing device-metric, source, representation, state and allocation gates
 * remain with their owners. Unknown architectures retain the generic path.
 */
struct DfDerivativeProfile {
  std::size_t minimum_public_weights;
  std::size_t minimum_ordered_primitive_products;
  std::size_t minimum_packed_response_weights;
};

constexpr DfDerivativeProfile df_derivative_profile(unsigned architecture) noexcept {
  // Small molecular endpoints lose to shell-launch/metadata overhead. Retain
  // headroom below the measured medium-size wins; these are work thresholds,
  // not AO/rank pairs, molecule fingerprints or GPU marketing names.
  return architecture == 120 ? DfDerivativeProfile{1U << 18, 1U << 22, 1U << 28}
                             : DfDerivativeProfile{0, 0, 0};
}

/** Compare a*b*c with a positive bound without ever forming the product. */
constexpr bool df_derivative_work_at_least(std::size_t a, std::size_t b, std::size_t c,
                                           std::size_t bound) noexcept {
  if (!a || !b || !c || !bound) return false;
  const auto divide_up = [](std::size_t x, std::size_t y) { return x / y + (x % y != 0); };
  return c >= divide_up(divide_up(bound, a), b);
}

constexpr bool df_shell_execution_preferred(std::size_t nbf, std::size_t naux,
                                            unsigned architecture) noexcept {
  return df_derivative_work_at_least(nbf, nbf, naux,
                                     df_derivative_profile(architecture).minimum_public_weights);
}

constexpr bool df_signature_packets_preferred(std::size_t orbital_primitives,
                                              std::size_t auxiliary_primitives,
                                              bool heterogeneous_contractions,
                                              unsigned architecture) noexcept {
  // Homogeneous contraction lengths already execute without divergent loop
  // trip counts. Do not pay signature metadata costs when grouping cannot help.
  return heterogeneous_contractions &&
         df_derivative_work_at_least(
             orbital_primitives, orbital_primitives, auxiliary_primitives,
             df_derivative_profile(architecture).minimum_ordered_primitive_products);
}

/** Promote folded packed occupied-response weights by general work and rank
 * features, never by a benchmark AO/rank tuple or GPU marketing name.
 *
 * The first sm_120 profile deliberately keeps the measured smaller-domain
 * dense/symmetric default: 2^28 lies above the 384^3 negative/default domain
 * and below the qualified 768^3 packed-response endpoint.  Rank admission is
 * expressed as an occupied fraction so nearby RHF shapes can reuse the same
 * policy.  Correctness, provenance, resident storage and one-term RHF gates
 * remain with the response owner; unknown architectures stay unpromoted.
 */
constexpr bool df_packed_response_preferred(std::size_t nbf, std::size_t naux, std::size_t rank,
                                            unsigned architecture) noexcept {
  const auto profile = df_derivative_profile(architecture);
  return rank > 0 && nbf >= 4 && rank <= nbf / 4 &&
         df_derivative_work_at_least(nbf, nbf, naux, profile.minimum_packed_response_weights);
}

/** Explicit qualification can decouple source-only occupied response from
 * derivative scheduling. This is NOT automatic production promotion. The
 * bridge must have validated the occupied/source/metric identities first.
 * Unknown targets and all existing diagnostic exclusions remain unchanged.
 */
constexpr bool df_response_shell_source_eligible(bool has_source, bool has_raw,
                                                 bool has_fitted,
                                                 bool validated_full_rank_occupied,
                                                 bool qualify_source) noexcept {
  return !has_source || has_raw || has_fitted ||
         (qualify_source && validated_full_rank_occupied);
}

}  // namespace vibeqc::scf

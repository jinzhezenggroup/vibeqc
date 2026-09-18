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
};

constexpr DfDerivativeProfile df_derivative_profile(unsigned architecture) noexcept {
  // Small molecular endpoints lose to shell-launch/metadata overhead. Retain
  // headroom below the measured medium-size wins; these are work thresholds,
  // not AO/rank pairs, molecule fingerprints or GPU marketing names.
  return architecture == 120 ? DfDerivativeProfile{1U << 18, 1U << 22} : DfDerivativeProfile{0, 0};
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

}  // namespace vibeqc::scf

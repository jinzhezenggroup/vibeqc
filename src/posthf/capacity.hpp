#ifndef VIBEQC_POSTHF_CAPACITY_HPP
#define VIBEQC_POSTHF_CAPACITY_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "core/types.hpp"
#include "molecule/basis.hpp"

namespace vibeqc::posthf {
inline std::size_t checked_add(std::size_t a, std::size_t b) {
  constexpr auto limit = static_cast<std::size_t>(INT64_MAX);
  if (a > limit || b > limit - a) throw std::overflow_error("post-HF byte count overflow");
  return a + b;
}
inline std::size_t checked_mul(std::size_t a, std::size_t b) {
  constexpr auto limit = static_cast<std::size_t>(INT64_MAX);
  if (a && b > limit / a) throw std::overflow_error("post-HF byte count overflow");
  return a * b;
}

/** CG10 numeric source capacity, including normalized/original coefficients
 * and a conservative complete public-to-Cartesian expansion. Object headers
 * and allocator rounding are excluded, as in the existing source contract. */
inline std::size_t source_capacity(const core::System& system) {
  auto bytes = checked_mul(128, system.atoms.size());
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 3)
      throw std::invalid_argument("post-HF source supports through f");
    const auto l = static_cast<std::size_t>(shell.angular_momentum);
    const auto cart = (l + 1) * (l + 2) / 2;
    const auto pub = system.basis_representation == VIBEQC_BASIS_SPHERICAL ? 2 * l + 1 : cart;
    bytes = checked_add(bytes, 64 + 32 * cart + 16 * pub * cart);
    bytes = checked_add(bytes, checked_mul(32, shell.primitives.size()));
  }
  return bytes;
}
inline constexpr std::size_t source_scratch_bytes = 8U << 20;

inline std::size_t rhf_reference_capacity(const core::System& system, unsigned diis_history,
                                          bool include_dense_eri = false) {
  const auto n = molecule::ao_count(system);
  const auto matrix_elements = checked_mul(n, n);
  // Original and canonical states, eigensolver/matrix temporaries, all DIIS
  // history, exported snapshot, residual validation and raw tile capacity.
  const auto matrices = checked_add(64, checked_mul(2, diis_history));
  auto bytes = checked_add(checked_add(source_capacity(system), source_scratch_bytes + 4096),
                           checked_mul(8, checked_mul(matrix_elements, matrices)));
  // CPU preparation holds both Cartesian Jet values and the unpacked FP64
  // tensor. A spherical conversion adds the public tensor while both Cartesian
  // copies are still live. Count scalar payloads here; Jet vector headers and
  // allocator rounding remain outside the documented numeric-buffer budget.
  if (include_dense_eri) {
    const auto cart = molecule::cartesian_ao_count(system);
    const auto cart2 = checked_mul(cart, cart);
    auto eri_elements = checked_mul(2, checked_mul(cart2, cart2));
    if (system.basis_representation == VIBEQC_BASIS_SPHERICAL)
      eri_elements = checked_add(eri_elements, checked_mul(matrix_elements, matrix_elements));
    bytes = checked_add(bytes, checked_mul(8, eri_elements));
  }
  return bytes;
}
}  // namespace vibeqc::posthf
#endif

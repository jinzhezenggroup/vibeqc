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

/** Conservative peak for one energy-only RI-MP2 phase.
 *
 * Count the Cartesian generator output
 * and device/export staging, the public
 * transformed raw tensors, the Jacobi metric
 * factorization, whitened
 * three-center values, occupied-virtual coefficients, both normalized
 * source
 * owners and the detached physical reference. Object headers and allocator
 * rounding
 * follow the same numeric-buffer convention as the other capacity
 * helpers.
 */
inline std::size_t ri_mp2_capacity(const core::System& orbital, const core::System& auxiliary,
                                   std::size_t occupied, std::size_t kernel_bytes = 0) {
  const auto n = molecule::ao_count(orbital);
  const auto na = molecule::ao_count(auxiliary);
  const auto cart = molecule::cartesian_ao_count(orbital);
  const auto cart_aux = molecule::cartesian_ao_count(auxiliary);
  if (!occupied || occupied >= n || !na) throw std::invalid_argument("invalid RI-MP2 dimensions");
  const auto matrix = checked_mul(n, n);
  const auto cart_matrix = checked_mul(cart, cart);
  const auto metric = checked_mul(na, na);
  const auto cart_metric = checked_mul(cart_aux, cart_aux);
  const auto three_center = checked_mul(matrix, na);
  const auto cart_three_center = checked_mul(cart_matrix, cart_aux);
  const auto bia = checked_mul(checked_mul(occupied, n - occupied), na);
  const auto reference = checked_add(checked_mul(5, matrix), n);
  auto elements = reference;
  // The CUDA batch exporter simultaneously owns packed device output, host
  // chunk staging and the returned per-system arrays. Three copies also
  // conservatively cover CPU Jet evaluation and unpacked values.
  elements = checked_add(elements, checked_mul(3, checked_add(cart_metric, cart_three_center)));
  elements = checked_add(elements, checked_mul(4, metric));
  elements = checked_add(elements, checked_mul(2, three_center));
  elements = checked_add(elements, bia);
  auto bytes = checked_mul(elements, sizeof(double));
  bytes = checked_add(bytes, source_capacity(orbital));
  bytes = checked_add(bytes, source_capacity(auxiliary));
  // CUDA's combined orbital/auxiliary HostBatch and device topology mirror
  // the normalized source metadata while the RawSource owners remain live.
  bytes = checked_add(bytes, source_capacity(orbital));
  bytes = checked_add(bytes, source_capacity(auxiliary));
  bytes = checked_add(bytes, checked_mul(2, source_scratch_bytes));
  return checked_add(bytes, kernel_bytes);
}

/** CPU DF-SCF peak while its prepared metric/three-center tensors coexist
 * with the
 * iterative/reference matrices. The orbital source and one-electron
 * matrices are already
 * included by rhf_reference_capacity; add only the
 * auxiliary owner plus raw metric, raw
 * three-center and whitened three-center.
 */
inline std::size_t ri_mp2_reference_capacity(const core::System& orbital,
                                             const core::System& auxiliary, unsigned diis_history) {
  const auto n = molecule::ao_count(orbital);
  const auto na = molecule::ao_count(auxiliary);
  const auto metric = checked_mul(na, na);
  const auto three_center = checked_mul(checked_mul(n, n), na);
  auto retained = rhf_reference_capacity(orbital, diis_history, false);
  retained = checked_add(retained, source_capacity(auxiliary));
  retained = checked_add(retained, checked_mul(sizeof(double), metric));
  retained = checked_add(retained, checked_mul(sizeof(double), checked_mul(2, three_center)));
  // Before DIIS exists, CPU preparation can instead peak while Cartesian
  // Jet/unpacked tensors, the metric eigensolver and the whitened tensor
  // coexist. Reuse the stricter checked preparation/correlation bound so the
  // reference admission and its public diagnostic cover both phases.
  const auto occupied = static_cast<std::size_t>(orbital.electron_count / 2);
  return std::max(retained, ri_mp2_capacity(orbital, auxiliary, occupied));
}
}  // namespace vibeqc::posthf
#endif

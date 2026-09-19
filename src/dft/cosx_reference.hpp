#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/types.hpp"

namespace vibeqc::dft {

/** Density convention for the exchange-only COSX reference energy. */
enum class CosxDensityConvention : std::uint8_t {
  /** One alpha or beta density block: E_x = -1/2 Tr(D K[D]). */
  spin_resolved = 0,
  /** RHF spin-summed density: E_x = -1/4 Tr(D K[D]). */
  rhf_spin_summed = 1,
};

/** Versioned semantics of the first auditable COSX reference model.
 *
 * Version 1 is deliberately unscreened and unfitted. The one-sided
 * seminumerical K build is explicitly symmetrized after quadrature.
 */
struct CosxReferenceSpec {
  std::uint32_t version{1};
  bool symmetrize{true};
  bool overlap_fitting{false};
  bool screening{false};

  bool operator==(const CosxReferenceSpec&) const = default;
};

struct CosxReferenceResult {
  std::size_t nbf{};
  std::size_t npoint{};
  CosxReferenceSpec spec{};
  CosxDensityConvention convention{CosxDensityConvention::spin_resolved};
  /** One-sided seminumerical contraction before matrix symmetrization. */
  std::vector<double> raw_exchange;
  /** Symmetric positive exchange matrix K[D]. */
  std::vector<double> exchange;
  /** Exchange-only electronic energy under the selected density convention. */
  double exchange_energy{};
};

/** Small CPU oracle for the discrete COSX exchange model.
 *
 * points_xyz contains explicit Bohr xyz triples and weights contains the
 * matching volume weights. The exact point set and weights are therefore part
 * of the mathematical model. This reference intentionally materializes all
 * AO values and ESP matrices; production COSX must use bounded tiles instead.
 */
CosxReferenceResult build_cosx_reference(
    const core::System& system, std::span<const double> points_xyz, std::span<const double> weights,
    std::span<const double> density, CosxDensityConvention convention, CosxReferenceSpec spec = {});

}  // namespace vibeqc::dft

#ifndef QCE_MOLECULE_BASIS_HPP
#define QCE_MOLECULE_BASIS_HPP

#include "core/types.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace qce::molecule {

/** Cartesian exponent triple in CCA order (xx, xy, xz, yy, yz, zz for d). */
using CartesianComponent = std::array<unsigned, 3>;

/** Number of Cartesian functions in a shell of angular momentum `l`. */
[[nodiscard]] constexpr std::size_t cartesian_count(unsigned l) noexcept {
  return static_cast<std::size_t>((l + 1) * (l + 2) / 2);
}

/** Generate Cartesian functions in the CCA ordering used by libcint/PySCF. */
[[nodiscard]] std::vector<CartesianComponent> cartesian_components(unsigned l);

/** Total Cartesian AO count represented by a system's shells. */
[[nodiscard]] std::size_t ao_count(const core::System& system) noexcept;

/**
 * Cartesian normalization factor not contained in a shell's radial factor.
 *
 * The validator folds `(2a/pi)^(3/4) (4a)^(l/2)` and contraction
 * normalization into each primitive coefficient.  This factor supplies the
 * remaining `1/sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!)` for an AO component.
 */
[[nodiscard]] double cartesian_component_normalization(
    const CartesianComponent& component) noexcept;

/** Validate and radially normalize contracted Cartesian Gaussian shells. */
qce_status validate_and_normalize(core::System& system, std::string& detail);

}  // namespace qce::molecule

#endif

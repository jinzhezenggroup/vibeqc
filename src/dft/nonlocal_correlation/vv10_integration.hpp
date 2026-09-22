#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "dft/density_source.hpp"

namespace vibeqc::dft {
class AoBasis;
class MolecularGrid;
namespace nlc {
class Vv10Plan;

/** StrictPositive preserves the raw fixed-grid contract. MolecularV1 removes
 * rho < 1e-8 from BOTH pair domains; it never floors an active density. */
enum class Vv10DensityDomain : unsigned { StrictPositive = 0, MolecularV1 = 1 };

struct Vv10Integral {
  double energy{};
  std::vector<double> potential;
  std::size_t points{};
  std::size_t owned_numeric_bytes{};
};

struct SpinVv10Integral {
  double energy{};
  std::array<std::vector<double>, 2> potential;
  std::size_t points{};
  std::size_t owned_numeric_bytes{};
};

/** Assemble the self-consistent AO contribution of one prepared VV10/rVV10
 * primitive. The pair plan owns only nonlocal pair mathematics; this bridge
 * owns total-density AO features and the weak-form vrho/vsigma contraction. */
Vv10Integral integrate_vv10_rks(const AoBasis& basis, const MolecularGrid& grid,
                                const std::vector<double>& density, Vv10Plan& plan,
                                std::size_t tile_points = 256, XcDensitySource source = {},
                                Vv10DensityDomain domain = Vv10DensityDomain::StrictPositive);

SpinVv10Integral integrate_vv10_uks(const AoBasis& basis, const MolecularGrid& grid,
                                    const std::vector<double>& alpha_density,
                                    const std::vector<double>& beta_density, Vv10Plan& plan,
                                    std::size_t tile_points = 256,
                                    Vv10DensityDomain domain = Vv10DensityDomain::StrictPositive);

}  // namespace nlc
}  // namespace vibeqc::dft

#ifndef VIBEQC_DFT_XC_HPP
#define VIBEQC_DFT_XC_HPP

#include <array>
#include <cstddef>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/density_source.hpp"
#include "dft/grid.hpp"

namespace vibeqc::dft {

/** LDA tail-v1: exact positive-density formula, analytic zero-density limit,
 * and explicit rejection of negative or non-finite density. No clipping or
 * density floor changes either the energy or its first derivative. */
inline constexpr const char* kLdaTailPolicy = "lda-tail-v1";
/** Spin-polarized LDA tail-v2: the exact PW92 spin interpolation is evaluated
 * through a
 * total-density sixth-root algebra, including analytic vacuum and
 * complete-polarization limits
 * without clipping either spin density. */
inline constexpr const char* kLdaSpinTailPolicy = "lda-spin-tail-v2-sixth-root";
/** PBE tail-v1 keeps the exact vacuum limit and requires all non-vacuum
 * features to remain in the audited interior-v1 domain. It never clips a
 * density or reduced gradient; unsupported tail points are rejected. */
inline constexpr const char* kPbeTailPolicy = "pbe-tail-v1";
/** PBE production tail-v2 keeps exact PBE in interior-v1 and uses the stable
 * LDA_XC_PW positive-density expression outside that domain. The fallback has
 * zero sigma derivative and reaches the exact zero-density limit. */
inline constexpr const char* kPbeProductionTailPolicy = "pbe-tail-v2-lda-fallback";
/** Polarized PBE uses stable scaled algebra without an LDA fallback. Only
 * phi's u^(2/3) spin interpolation has a C2 extension below u=1e-18; energy
 * and both spin potentials derive from this same versioned expression.
 * See docs/xc_scf_domain.md for its finite empty-spin derivative. */
inline constexpr const char* kPbeSpinProductionTailPolicy = "semilocal-scaled-v1/pbe-spin-c2-1e-18";

struct XcIntegral {
  double energy{};
  double electrons{};
  std::vector<double> potential;
  std::size_t points{};
  XcDensityDiagnostic density_diagnostic;
};

struct SpinXcIntegral {
  double energy{};
  std::array<double, 2> electrons{};
  std::array<std::vector<double>, 2> potential;
  std::size_t points{};
};

/** Integrate unpolarized PBE for an RHF total AO density. */
XcIntegral integrate_pbe_rks(const AoBasis& basis, const MolecularGrid& grid,
                             const std::vector<double>& density, std::size_t tile_points = 256,
                             XcDensitySource source = {});

/** Integrate PBE with the explicit production tail-v2 policy. */
XcIntegral integrate_pbe_rks_with_tail(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& density,
                                       std::size_t tile_points = 256, XcDensitySource source = {});

/** Integrate unpolarized LDA_XC_PW for an RHF total AO density. */
XcIntegral integrate_lda_xc_pw_rks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& density,
                                   std::size_t tile_points = 256, XcDensitySource source = {});

/** Integrate spin-polarized LDA_XC_PW for separate alpha/beta AO densities
 * using
 * kLdaSpinTailPolicy. */
SpinXcIntegral integrate_lda_xc_pw_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       std::size_t tile_points = 256);

/** Integrate spin-polarized PBE with kPbeSpinProductionTailPolicy. */
SpinXcIntegral integrate_pbe_uks_with_tail(const AoBasis& basis, const MolecularGrid& grid,
                                           const std::vector<double>& alpha_density,
                                           const std::vector<double>& beta_density,
                                           std::size_t tile_points = 256);

}  // namespace vibeqc::dft

#endif

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
/** PBE tail-v1 keeps the exact vacuum limit and requires all non-vacuum
 * features to remain in the audited interior-v1 domain. It never clips a
 * density or reduced gradient; unsupported tail points are rejected. */
inline constexpr const char* kPbeTailPolicy = "pbe-tail-v1";
/** Exact low-density PBE algebra with a declared C2 spin-endpoint extension;
 * no PBE-to-LDA fallback. The historical with_tail entry names are retained
 * for internal callers; the prepared numerical identity is versioned here. */
inline constexpr const char* kPbeProductionTailPolicy = "semilocal-scaled-v1/pbe-spin-c2-1e-18";

/** Compatibility names for the already registered CPU spin compositions. */
inline constexpr const char* kLdaSpinTailPolicy = "lda-spin-tail-v2-sixth-root";
inline constexpr const char* kPbeSpinProductionTailPolicy = kPbeProductionTailPolicy;

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

/** Integrate PBE with the explicit scaled-v1 domain policy. */
XcIntegral integrate_pbe_rks_with_tail(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& density,
                                       std::size_t tile_points = 256, XcDensitySource source = {});

XcIntegral integrate_pbe_rks_with_tail_scaled(const AoBasis& basis, const MolecularGrid& grid,
                                              const std::vector<double>& density,
                                              std::size_t tile_points, XcDensitySource source,
                                              double exchange_scale, double correlation_scale);

/** Integrate unpolarized LDA_XC_PW for an RHF total AO density. */
XcIntegral integrate_lda_xc_pw_rks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& density,
                                   std::size_t tile_points = 256, XcDensitySource source = {});

/** Integrate spin-polarized LDA_XC_PW for separate alpha/beta AO densities.
 * Uses the analytic LDA spin limit, including an empty spin channel. */
SpinXcIntegral integrate_lda_xc_pw_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       std::size_t tile_points = 256);

/** PBE with independent spin densities and the versioned point-domain policy. */
SpinXcIntegral integrate_pbe_uks(const AoBasis& basis, const MolecularGrid& grid,
                                 const std::vector<double>& alpha_density,
                                 const std::vector<double>& beta_density,
                                 std::size_t tile_points = 256);

SpinXcIntegral integrate_pbe_uks_scaled(const AoBasis& basis, const MolecularGrid& grid,
                                        const std::vector<double>& alpha_density,
                                        const std::vector<double>& beta_density,
                                        std::size_t tile_points, double exchange_scale,
                                        double correlation_scale);

/** Integrate spin-polarized PBE with kPbeSpinProductionTailPolicy. */
SpinXcIntegral integrate_pbe_uks_with_tail(const AoBasis& basis, const MolecularGrid& grid,
                                           const std::vector<double>& alpha_density,
                                           const std::vector<double>& beta_density,
                                           std::size_t tile_points = 256);

}  // namespace vibeqc::dft

#endif

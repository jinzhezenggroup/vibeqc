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
/** Only the numerically null far tail is modified; the quintic switch is C2
 * and its density derivative is included in the generalized-KS potential. */
inline constexpr const char* kR2scanProductionTailPolicy = "r2scan-tail-c2-v1/n=1e-56:1e-52";

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
/** CAM-B3LYP semilocal MethodIR primitive on its audited interior-v1 domain.
 * SR/LR exact exchange is owned by the common Fock providers, not this object.
 */
struct CamB3lypPointValue {
  double energy{};
  double rho[2]{};
  double gradient[2][3]{};
};

CamB3lypPointValue evaluate_cam_b3lyp_point(const double rho[2], const double (&gradient)[2][3]);

XcIntegral integrate_cam_b3lyp_rks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& density,
                                   std::size_t tile_points = 256, XcDensitySource source = {});

SpinXcIntegral integrate_cam_b3lyp_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       std::size_t tile_points = 256);

struct R2scanPointValue {
  double energy{};
  double rho[2]{};
  double gradient[2][3]{};
  /** Coefficient of grad(phi_mu).grad(phi_nu), i.e. vtau/2. */
  double kinetic[2]{};
};

R2scanPointValue evaluate_r2scan_point(const double rho[2], const double (&gradient)[2][3],
                                       const double tau[2]);

/** r2SCAN meta-GGA using rho/sigma/tau and the generated vtau weak-form term. */
XcIntegral integrate_r2scan_rks(const AoBasis& basis, const MolecularGrid& grid,
                                const std::vector<double>& density, std::size_t tile_points = 256,
                                XcDensitySource source = {});

SpinXcIntegral integrate_r2scan_uks(const AoBasis& basis, const MolecularGrid& grid,
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

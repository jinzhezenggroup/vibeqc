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

/** Bounded prepared CPU AO-grid values for repeated fixed-geometry RKS builds.
 * Layout matches AoBasis::evaluate over the complete grid: jet-major, then
 * point, then AO. Scientific grid/basis identity remains owned by the caller. */
struct RksAoCache {
  unsigned order{};
  std::size_t points{};
  std::size_t nao{};
  std::vector<double> jets;

  [[nodiscard]] std::size_t numeric_capacity_bytes() const noexcept;
};

[[nodiscard]] std::size_t rks_ao_cache_bytes(const AoBasis& basis, const MolecularGrid& grid,
                                             unsigned order);
RksAoCache prepare_rks_ao_cache(const AoBasis& basis, const MolecularGrid& grid, unsigned order);

struct SpinXcIntegral {
  double energy{};
  std::array<double, 2> electrons{};
  std::array<std::vector<double>, 2> potential;
  std::size_t points{};
};

/** Fixed-model exact incremental PBE prototype for #237.
 *
 * anchor_density is the accepted reference state and delta_density is a signed
 * AO-matrix increment. Linear grid features are contracted independently from
 * D0 and delta-D, then added before any nonlinear invariant/functional
 * evaluation. potential_difference is Vxc[D0+delta-D] - Vxc[D0], assembled
 * from exact coefficient differences; no fxc linearization or local skipping
 * is used.
 */
struct ExactIncrementalXcIntegral {
  XcIntegral total;
  double anchor_energy{};
  double energy_difference{};
  std::vector<double> potential_difference;
};

ExactIncrementalXcIntegral integrate_pbe_rks_incremental_exact(
    const AoBasis& basis, const MolecularGrid& grid, const std::vector<double>& anchor_density,
    const std::vector<double>& delta_density, std::size_t tile_points = 256,
    double exchange_scale = 1.0, double correlation_scale = 1.0);

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

/** Same PBE integration using an immutable prepared order-1 AO grid. */
XcIntegral integrate_pbe_rks_with_tail_scaled_cached(
    const AoBasis& basis, const MolecularGrid& grid, const std::vector<double>& density,
    std::size_t tile_points, XcDensitySource source, double exchange_scale,
    double correlation_scale, const RksAoCache& cache);

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
/** Generated B3-family GGA point result.  B3LYP uses the production-tail-v1
 * analytic endpoint continuation; CAM-B3LYP remains on the audited interior-v1
 * domain. Full-/range-separated exact exchange stays with common Fock providers.
 */
inline constexpr const char* kB3lypProductionTailPolicy =
    "b3lyp-vwn-rpa-tail-v1/density-vacuum-1e-18";
struct SemilocalPointValue {
  double energy{};
  double rho[2]{};
  double gradient[2][3]{};
  /** Coefficient of grad(phi_mu).grad(phi_nu), i.e. vtau/2 when tau is active. */
  double kinetic[2]{};
};

using SemilocalPointEvaluator = SemilocalPointValue (*)(const double rho[2],
                                                        const double (&gradient)[2][3],
                                                        const double tau[2]);

/** One already-compiled semilocal point program. The identifier is diagnostic
 * only; expression_identity binds the generated mathematics. ingredient_mask
 * follows the native rho/sigma/tau feature bits (1, 7, or 15). This descriptor
 * is executable plumbing and does not grant production capability by itself. */
struct SemilocalPointProgram {
  const char* identifier{};
  const char* expression_identity{};
  unsigned ingredient_mask{};
  unsigned domain_version{};
  SemilocalPointEvaluator evaluate{};
};

void validate_semilocal_point_program(const SemilocalPointProgram& program);

XcIntegral integrate_semilocal_rks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& density,
                                   const SemilocalPointProgram& program,
                                   std::size_t tile_points = 256, XcDensitySource source = {});
SpinXcIntegral integrate_semilocal_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       const SemilocalPointProgram& program,
                                       std::size_t tile_points = 256);

using GgaPointValue = SemilocalPointValue;
using B3GgaPointValue = SemilocalPointValue;
using B3lypPointValue = SemilocalPointValue;
using CamB3lypPointValue = SemilocalPointValue;
using Pw91PointValue = SemilocalPointValue;
using R2scanPointValue = SemilocalPointValue;

B3lypPointValue evaluate_b3lyp_point(const double rho[2], const double (&gradient)[2][3]);
CamB3lypPointValue evaluate_cam_b3lyp_point(const double rho[2], const double (&gradient)[2][3]);
Pw91PointValue evaluate_pw91_point(const double rho[2], const double (&gradient)[2][3]);

XcIntegral integrate_b3lyp_rks(const AoBasis& basis, const MolecularGrid& grid,
                               const std::vector<double>& density, std::size_t tile_points = 256,
                               XcDensitySource source = {});
SpinXcIntegral integrate_b3lyp_uks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& alpha_density,
                                   const std::vector<double>& beta_density,
                                   std::size_t tile_points = 256);

XcIntegral integrate_cam_b3lyp_rks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& density,
                                   std::size_t tile_points = 256, XcDensitySource source = {});
SpinXcIntegral integrate_cam_b3lyp_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       std::size_t tile_points = 256);

/** Interior-v1 PW91 native fixed-density qualification path.
 * This is a generic-GGA lowerer proof and does not register a public KS method.
 */
XcIntegral integrate_pw91_rks(const AoBasis& basis, const MolecularGrid& grid,
                              const std::vector<double>& density, std::size_t tile_points = 256,
                              XcDensitySource source = {});
SpinXcIntegral integrate_pw91_uks(const AoBasis& basis, const MolecularGrid& grid,
                                  const std::vector<double>& alpha_density,
                                  const std::vector<double>& beta_density,
                                  std::size_t tile_points = 256);

using Wb97mvPointValue = SemilocalPointValue;

inline constexpr const char* kWb97mvProductionTailPolicy =
    "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16";

R2scanPointValue evaluate_r2scan_point(const double rho[2], const double (&gradient)[2][3],
                                       const double tau[2]);
Wb97mvPointValue evaluate_wb97mv_point(const double rho[2], const double (&gradient)[2][3],
                                       const double tau[2]);

XcIntegral integrate_wb97mv_rks(const AoBasis& basis, const MolecularGrid& grid,
                                const std::vector<double>& density, std::size_t tile_points = 256,
                                XcDensitySource source = {});
SpinXcIntegral integrate_wb97mv_uks(const AoBasis& basis, const MolecularGrid& grid,
                                    const std::vector<double>& alpha_density,
                                    const std::vector<double>& beta_density,
                                    std::size_t tile_points = 256);

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

#ifndef VIBEQC_INTEGRALS_S_INTEGRALS_HPP
#define VIBEQC_INTEGRALS_S_INTEGRALS_HPP

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "core/types.hpp"
#include "integrals/range_moments.hpp"

namespace vibeqc::integrals {

struct IntegralData {
  std::size_t nbf{};
  std::size_t ncoord{};
  std::vector<double> overlap;
  std::vector<double> hcore;
  std::vector<double> eri;
  // Derivative arrays are coordinate-major: derivative[coord * size + item].
  std::vector<double> overlap_derivative;
  std::vector<double> hcore_derivative;
  std::vector<double> eri_derivative;
  double nuclear_repulsion{};
  std::vector<double> nuclear_repulsion_derivative;
};

/** Independent two-/three-center Coulomb integral oracle for density fitting. */
struct DensityFittingIntegralData {
  std::size_t nbf{};
  std::size_t naux{};
  std::size_t ncoord{};
  // Row-major metric (P|Q) and three-center tensor (mu nu|P).
  std::vector<double> metric;
  std::vector<double> three_center;
  // Coordinate-major derivatives use the same item order as their values.
  std::vector<double> metric_derivative;
  std::vector<double> three_center_derivative;
};

/** Analytic AO electrostatic-potential matrices at explicit probe points.
 *
 * Values are point-major row-major matrices:
 *   values[(point * nbf + mu) * nbf + nu]
 *     = <mu | 1 / |r - R_point| | nu>.
 *
 * Probe coordinates are supplied as flat xyz triples in Bohr. They are fixed
 * external points: this value-only reference does not attach nuclear
 * derivatives to the probe coordinates.
 */
struct EspIntegralData {
  std::size_t nbf{};
  std::size_t npoint{};
  std::vector<double> values;
};

/** AO ESP matrices and analytic derivatives with respect to each probe point.
 *
 * probe_derivative is point-major, then Cartesian-axis, then row-major AO
 * matrix:
 *   probe_derivative[((point * 3 + axis) * nbf + mu) * nbf + nu].
 *
 * Only the explicit probe coordinate moves. Gaussian centers, contractions,
 * and all other probes are held fixed.
 */
struct EspProbeDerivativeData {
  std::size_t nbf{};
  std::size_t npoint{};
  std::vector<double> values;
  std::vector<double> probe_derivative;
};

/**
 * Evaluate normalized, contracted Cartesian or real-spherical integrals.
 *
 * The production CPU s/p/d/f overlap/kinetic family is emitted at build time
 * from the compiler-owned one-electron DAG shared with CUDA.  RawSource keeps
 * a structurally independent host recurrence for reference validation; g+ S/T
 * explicitly falls back to that reference until its generated domain expands.
 * Other host integral families retain their existing implementation here.
 */
IntegralData build_integrals(const core::System& system, bool include_derivatives = true,
                             bool include_eri = true);

/** Evaluate value-only short-/long-range two-electron integrals in the public AO basis.
 *
 * This CPU reference path reuses the #166 positive-interval radial moments, so
 * short range is evaluated directly rather than as Coulomb-minus-long-range.
 * It deliberately exposes values only; analytic range-separated derivatives
 * remain owned by the weighted derivative provider.
 */
std::vector<double> build_range_eri(const core::System& system, CoulombRange range, double omega);

/** Evaluate analytic AO ESP matrices on explicit probe points. */
EspIntegralData build_esp_integrals(const core::System& system, std::span<const double> points_xyz);

/** Evaluate AO ESP matrices and analytic explicit-probe coordinate derivatives. */
EspProbeDerivativeData build_esp_integrals_with_probe_derivatives(
    const core::System& system, std::span<const double> points_xyz);

/** Contract one ordered public-AO shell quartet with arbitrary weights.
 *
 * The twelve returned entries are positive integral derivatives in independent
 * shell-slot order (Axyz, Bxyz, Cxyz, Dxyz). The caller scatters slots to
 * physical atoms, so repeated atoms remain distinct through differentiation.
 * Only this quartet's public-basis expansions and twelve derivative scalars
 * are materialized; no molecular AO-rank-four or coordinate derivative tensor
 * is formed.
 */
std::array<double, 12> contract_weighted_eri_shell_derivative(
    const core::System& system, const std::array<std::size_t, 4>& shell_indices,
    std::span<const double> weights);

/** Directly contract arbitrary public-AO overlap and hcore weights.
 *
 * The coordinate-sized result contains positive energy derivatives. When
 * include_nuclear_repulsion is true the nuclear term is added exactly once.
 * The implementation accumulates one differentiated scalar and never forms
 * coordinate-major AO matrices.
 */
std::vector<double> contract_weighted_one_electron_derivative(
    const core::System& system, std::span<const double> overlap_weights,
    std::span<const double> hcore_weights, bool include_nuclear_repulsion);

/**
 * Write a row-major rectangular <target AO | source AO> overlap on the CPU.
 * Systems already own validated normalized shells and may independently use
 * Cartesian or real-spherical AOs. Reuses the value recurrence with no nuclear
 * derivative storage, ERIs, or combined-basis square matrix allocation.
 */
void cross_overlap(const core::System& target, const core::System& source,
                   std::span<double> output);

/**
 * Evaluate normalized two- and three-center density-fitting integrals.
 *
 * Orbital and auxiliary systems must describe the same atoms and geometry but
 * may use independent Cartesian or real-spherical shell sets. This host-only
 * implementation remains the independent correctness oracle for CUDA DF
 * integral-generation kernels.
 */
DensityFittingIntegralData build_density_fitting_integrals(const core::System& orbital_system,
                                                           const core::System& auxiliary_system,
                                                           bool include_derivatives = true);

/** Contract public-basis DF derivative weights directly into nuclear coordinates.
 *
 * metric_weights and three_center_weights use the same public AO layouts as
 * DensityFittingIntegralData::metric and three_center. Generated s/p/d/f
 * derivatives are consumed immediately, so no coordinate-resolved dM/dR or
 * d(mu nu|P)/dR tensor is materialized. Higher angular momentum retains the
 * full-tensor oracle as an explicit correctness fallback.
 */
std::vector<double> contract_weighted_density_fitting_derivative(
    const core::System& orbital_system, const core::System& auxiliary_system,
    std::span<const double> metric_weights, std::span<const double> three_center_weights);
/**
 * Transform Cartesian density-fitting tensors into the public AO
 * representations selected by the two systems.
 *
 * This is intentionally separate from integral evaluation so accelerator
 * backends can generate the Cartesian tensor on device and reuse the
 * independent, numerically stable spherical transformation here.  Derivative
 * tensors are transformed coordinate-by-coordinate with the same contractions.
 */
DensityFittingIntegralData transform_density_fitting_integrals(
    const DensityFittingIntegralData& cartesian, const core::System& orbital_system,
    const core::System& auxiliary_system);

/** Transform Cartesian one-electron tensors into a system's public AO basis. */
IntegralData transform_integrals(const IntegralData& cartesian, const core::System& system);

/** Compatibility name retained for callers that explicitly request Cartesian. */
inline IntegralData build_cartesian_integrals(const core::System& system,
                                              bool include_derivatives = true) {
  return build_integrals(system, include_derivatives);
}

/** Backward-compatible name retained for the original s-shell test helpers. */
inline IntegralData build_s_integrals(const core::System& system, bool include_derivatives = true) {
  return build_integrals(system, include_derivatives);
}

}  // namespace vibeqc::integrals

#endif

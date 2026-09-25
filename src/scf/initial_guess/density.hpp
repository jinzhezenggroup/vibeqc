#ifndef VIBEQC_SCF_INITIAL_GUESS_DENSITY_HPP
#define VIBEQC_SCF_INITIAL_GUESS_DENSITY_HPP

#include <functional>
#include <optional>
#include <span>
#include <utility>

#include "scf/initial_guess/eigen_operation.hpp"
#include "scf/reference/linalg.hpp"

namespace vibeqc::core {
struct System;
}
namespace vibeqc::integrals {
struct IntegralData;
}

namespace vibeqc::scf::initial_guess {

using reference::EigenResult;
using reference::Matrix;

/** Immutable context for an optional restricted cold-start density provider.
 *
 * Providers run exactly once before the physical SCF loop. They may propose a
 * target-basis AO density derived from a cheaper model, but do not own the
 * target Hamiltonian, convergence policy, or final state. Returning nullopt,
 * failing proposal validation, or throwing a non-allocation standard exception
 * requests the canonical core-density fallback.
 */
struct RestrictedInitialDensityRequest {
  const core::System& system;
  const integrals::IntegralData& integrals;
  const Matrix& orthogonalizer;
  const Matrix& core_density;
  std::size_t occupied{};
};
using RestrictedInitialDensityProvider =
    std::function<std::optional<Matrix>(const RestrictedInitialDensityRequest&)>;

/** Iterative callers need a core frame only to construct a cold density.
 * A noniterative consumer must explicitly request a frame for supplied D. */
enum class InitialOrbitalRequest { ColdDensityOnly, RequireCoreFrame };

/** Resolve integral alpha/beta occupations from a validated electron/spin state. */
std::pair<std::size_t, std::size_t> spin_occupations(const core::System& system);
/** Apply the historical UHF frontier perturbation only to an open-shell seed.
 * This orthogonal rotation avoids symmetry-locked excited-state core guesses;
 * it does not constrain the final physical orbitals or electronic state.
 */
void mix_open_shell_frontier_orbitals(Matrix& beta_coefficients, std::size_t n,
                                      std::size_t alpha_occupied, std::size_t beta_occupied);
/** Restore a warm spin density's symmetry and Tr(D S), clearing empty spins. */
void normalize_spin_density(Matrix& density, const Matrix& overlap, std::size_t n,
                            std::size_t target_electrons);
/** Validate and normalize supplied D independently of X/hcore or a provider.
 * Batch owners may reject malformed warm inputs before allocating shared CUDA
 * resources. These functions have no eigensolve or tracing side effects. */
Matrix normalized_warm_density(const core::System& system, const integrals::IntegralData& ints,
                               const Matrix& input);
std::pair<Matrix, Matrix> normalized_warm_uhf_density(const integrals::IntegralData& ints,
                                                      std::size_t alpha_occupied,
                                                      std::size_t beta_occupied,
                                                      const Matrix& input);
/** Redistribute a finite restricted seed toward requested per-atom electron populations.
 *
 * atomic_charges follow the usual partial-charge convention q_A = Z_A^eff - N_A, where
 * Z_A^eff is the explicit ionic charge after any ECP core removal. The redistribution is
 * performed in the Loewdin-orthogonal AO basis by atom-owned AO blocks, then transformed
 * back to the public AO metric. This is an initial-guess heuristic, not a population-analysis
 * claim about the converged state.
 */
Matrix charge_guided_lowdin_density(const core::System& system, const integrals::IntegralData& ints,
                                    const Matrix& orthogonalizer, const Matrix& input,
                                    std::span<const double> atomic_charges);
/** Prepare a restricted core guess or a finite, normalized warm density.
 * Supplied density is validated/normalized without reading hcore. Its optional
 * core frame is empty unless explicitly requested; a cold density always has
 * a frame. The output is cleared even when validation fails, so a retry cannot
 * expose a previous call's orbitals. Neither frame represents a physical final
 * state; the solver must evaluate its target Fock before publishing results.
 */
Matrix prepare_initial_density(
    const core::System& system, const integrals::IntegralData& ints, const Matrix& orthogonalizer,
    std::size_t occupied, const std::vector<double>* initial_density,
    std::optional<EigenResult>& orbitals,
    InitialOrbitalRequest request = InitialOrbitalRequest::ColdDensityOnly,
    const EigenOperation& eigen = {}, const RestrictedInitialDensityProvider& provider = {});
/** Prepare independent unit-occupation spin guesses with the same warm-state contract. */
std::pair<Matrix, Matrix> prepare_initial_uhf_density(
    const integrals::IntegralData& ints, const Matrix& orthogonalizer, std::size_t alpha_occupied,
    std::size_t beta_occupied, const std::vector<double>* initial_density,
    std::optional<EigenResult>& alpha_orbitals, std::optional<EigenResult>& beta_orbitals,
    InitialOrbitalRequest request = InitialOrbitalRequest::ColdDensityOnly,
    const EigenOperation& eigen = {});

}  // namespace vibeqc::scf::initial_guess
#endif

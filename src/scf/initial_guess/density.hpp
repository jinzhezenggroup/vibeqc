#ifndef VIBEQC_SCF_INITIAL_GUESS_DENSITY_HPP
#define VIBEQC_SCF_INITIAL_GUESS_DENSITY_HPP

#include <utility>

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
/** Prepare a restricted core guess or a finite, normalized warm density.
 * Returned orbitals are the core-Hamiltonian frame, including for warm starts;
 * the solver must recompute the target Fock/orbitals before publishing results.
 */
Matrix prepare_initial_density(const core::System& system, const integrals::IntegralData& ints,
                               const Matrix& orthogonalizer, std::size_t occupied,
                               const std::vector<double>* initial_density, EigenResult& orbitals);
/** Prepare independent unit-occupation spin guesses with the same warm-state contract. */
std::pair<Matrix, Matrix> prepare_initial_uhf_density(
    const integrals::IntegralData& ints, const Matrix& orthogonalizer, std::size_t alpha_occupied,
    std::size_t beta_occupied, const std::vector<double>* initial_density,
    EigenResult& alpha_orbitals, EigenResult& beta_orbitals);

}  // namespace vibeqc::scf::initial_guess
#endif

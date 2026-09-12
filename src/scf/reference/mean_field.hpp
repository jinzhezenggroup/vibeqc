#ifndef VIBEQC_SCF_REFERENCE_MEAN_FIELD_HPP
#define VIBEQC_SCF_REFERENCE_MEAN_FIELD_HPP

#include <utility>

#include "scf/reference/linalg.hpp"

namespace vibeqc::scf::reference {

/** Form a density from the first occupied columns of S-orthonormal orbitals.
 * occupation_weight is two for a restricted closed shell and one per UHF spin.
 * The caller has already validated occupied <= n and the physical occupations.
 */
Matrix density_from_orbitals(const Matrix& coefficients, std::size_t n, std::size_t occupied,
                             double occupation_weight = 2.0);
/** Form the energy-weighted density used in the stationary overlap/Pulay term. */
Matrix energy_weighted_density(const Matrix& coefficients, const std::vector<double>& energies,
                               std::size_t n, std::size_t occupied, double occupation_weight = 2.0);
/** Restricted electronic energy, excluding nuclear repulsion. */
double electronic_energy(const Matrix& density, const Matrix& hcore, const Matrix& fock);
/** Unrestricted electronic energy with separate unit-occupation spin densities. */
double uhf_electronic_energy(const Matrix& alpha_density, const Matrix& beta_density,
                             const Matrix& hcore, const Matrix& alpha_fock,
                             const Matrix& beta_fock);
/** Concatenate independent spin blocks for shared DIIS/proposal bookkeeping. */
Matrix concatenate(const Matrix& first, const Matrix& second);
/** Recover two spin blocks, rejecting an inconsistent joined extent. */
std::pair<Matrix, Matrix> split_spin_matrices(const Matrix& joined, std::size_t matrix_size);
/** Physical nonorthogonal-AO stationarity residual F D S - S D F. */
Matrix commutator_residual(const Matrix& fock, const Matrix& density, const Matrix& overlap,
                           std::size_t n);
/** RMS difference over equally sized nonempty density arrays. */
double density_rms(const Matrix& a, const Matrix& b);
/** RMS physical residual, distinct from an update or DIIS extrapolation error. */
double residual_rms(const Matrix& residual);

}  // namespace vibeqc::scf::reference
#endif

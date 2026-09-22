#ifndef VIBEQC_SCF_SOLVER_WARM_SUBSPACE_HPP
#define VIBEQC_SCF_SOLVER_WARM_SUBSPACE_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace vibeqc::scf::solver {

/**
 * Evidence that a previous orthonormal orbital frame still spans the occupied
 * invariant subspace of a new orthonormal-basis Fock matrix.
 *
 * A Rayleigh-Ritz rotation restricted to the occupied columns cannot change
 * the occupied projector and therefore cannot update an RHF density.  The
 * useful quantity is the component of F*C_occ outside span(C_occ):
 *
 *   R = F C_occ - C_occ (C_occ^T F C_occ).
 *
 * Its Frobenius norm equals the occupied/virtual coupling norm when the full
 * previous frame is orthogonal.  A future warm eigensolver may use this gate
 * to decide whether to form correction directions or fall back to the dense
 * provider; passing this diagnostic alone never certifies a final eigenframe.
 */
struct WarmSubspaceResidual {
  double maximum_residual{};
  double frobenius_residual{};
  double scaled_residual{};
  bool finite{};
};

/**
 * Inspect a previous orthogonal orbital frame against a new symmetric Fock
 * matrix, both in the same orthonormal AO basis and row-major storage.
 *
 * previous_orbitals is an n-by-n matrix whose columns are the previous orbital
 * frame. Only the first occupied columns contribute to the residual.
 *
 * Throws std::invalid_argument for inconsistent shapes or occupied > n.
 * Non-finite numerical input is returned as finite=false so callers can take
 * the strict dense fallback without relying on exception handling.
 */
WarmSubspaceResidual inspect_warm_occupied_subspace(const std::vector<double>& orthonormal_fock,
                                                    const std::vector<double>& previous_orbitals,
                                                    std::size_t n, std::size_t occupied);

/**
 * Apply an explicit pre-solver gate to finite residual evidence.
 *
 * This gate only authorizes trying a warm correction/subspace route. The
 * resulting orbitals must still pass the ordinary eigenframe and SCF
 * convergence checks, and finalization retains the strict dense fallback.
 */
bool accept_warm_occupied_subspace(const WarmSubspaceResidual& diagnostic, double maximum_tolerance,
                                   double scaled_tolerance, std::string& detail);

}  // namespace vibeqc::scf::solver

#endif

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "scf/solver/final_state.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::scf {
struct CudaDensityFittingJkPlan;

/** Versioned, detached eligibility token for one successful device solve item.
 * The immutable plan binds the full DF source; the loop fixes standard FP64 HF
 * coefficients. Imported D has generation one, and each committed projection
 * advances it. Epochs never restart when persistent storage is rebuilt. */
struct CudaDfFinalStateToken {
  std::uint32_t version{1};
  solver::FinalStateIdentity identity;
  bool operator==(const CudaDfFinalStateToken&) const = default;
};

/** Detached candidate, not an authorized final physical state. The orbitals
 * diagonalize the preceding physical F[D]; the selector must evaluate the
 * supplied returned D and pass its strict checks before producing energy/W. */
struct CudaDfFinalStateSnapshot {
  solver::FinalFrameCandidate candidate;
  std::vector<reference::Matrix> density;
};

/** Current attempted compact-solve epoch, including a nonconverged attempt
 * followed by host DIIS recovery. Zero cannot authorize finalization. This
 * read does not invalidate another item's candidate in the same bucket. */
std::uint64_t cuda_density_fitting_solve_epoch(const CudaDensityFittingJkPlan* plan) noexcept;

/** Read only host eligibility metadata. Failed/nonconverged items and every
 * new attempted solve invalidate previous tokens before any device submission. */
vibeqc_status cuda_density_fitting_final_state_token(const CudaDensityFittingJkPlan* plan,
                                                     std::size_t item, CudaDfFinalStateToken& token,
                                                     std::string& detail);

/** Copy only the requested item's active-masked full C/epsilon and actual D.
 * Requires exact current token identity before any transfer. Calls use the
 * plan stream synchronously outside capture and never borrow graph scratch.
 * Failure clears the snapshot. Reference export may omit the already-public
 * density after device validation. No C/Python result record is extended. */
vibeqc_status read_cuda_density_fitting_final_state(CudaDensityFittingJkPlan* plan,
                                                    const CudaDfFinalStateToken& expected,
                                                    CudaDfFinalStateSnapshot& snapshot,
                                                    std::string& detail,
                                                    bool include_density = true);

/** Try physical J/K from an exact current singleton RHF retained density/frame.
 * Token, device generation and every supplied density entry must match.
 * Qualified streamed auto/occupied K may also reconstruct an algebraic factor
 * for a bounded strict-finalization correction after matching owner, solve,
 * model, occupations and generation advance. Rejected/stale/unsupported
 * requests retain used=false and the dense fallback. Corrected factors never publish a
 * response projection lease. Successful reuse still evaluates physical F[D].
 * With download=false the serialized final validator consumes the plan's J/K
 * directly; completion and exact projection-lease publication retain ordering.
 */
vibeqc_status try_cuda_density_fitting_final_rhf_jk(CudaDensityFittingJkPlan* plan,
                                                    const CudaDfFinalStateToken& expected,
                                                    const std::vector<double>& density,
                                                    std::vector<double>& coulomb,
                                                    std::vector<double>& exchange, bool& used,
                                                    std::string& detail, bool download = true);

/** Device algebra for the shared final-state selector. Matrices absent from
 * a candidate resolve only through its exact current retained-state identity.
 * Host outputs remain available to intentional force/reference consumers. */
solver::FinalStateOperations cuda_density_fitting_final_state_operations(
    CudaDensityFittingJkPlan* plan);

/** Evaluate physical J/K into the existing plan, borrowing them for exactly
 * this serialized validation request. Empty host F matrices are materialized
 * only if a correction or reference export needs them. */
solver::PhysicalFockFrame evaluate_cuda_density_fitting_final_fock(
    CudaDensityFittingJkPlan* plan, const solver::FinalStateIdentity& current,
    const std::vector<reference::Matrix>& density, const reference::Matrix& hcore);

/** Ordinary eigen-provider validation uses the same device products/gates,
 * including during bounded final corrections and host DIIS recovery. */
bool validate_cuda_density_fitting_eigen_frame(CudaDensityFittingJkPlan* plan,
                                               const reference::Matrix& matrix,
                                               const reference::Matrix* overlap,
                                               const reference::Matrix& values,
                                               const reference::Matrix& coefficients,
                                               solver::EigenFrameDiagnostic& diagnostic,
                                               std::string& detail);

/** Nine serialized matrices plus one spectrum and bounded reduction storage.
 * The conservative scalar allowance is checked against actual allocations. */
inline std::size_t df_final_validation_device_reservation(std::size_t n) {
  // Match one partial per 128 matrix elements, capped at 128 blocks. A
  // 128-byte payload bound includes padding without exposing CUDA structs.
  const long double elements = static_cast<long double>(n) * n;
  const auto blocks = static_cast<std::size_t>(std::min(128.0L, std::ceil(elements / 128)));
  const long double bytes = (9.0L * elements + n) * sizeof(double) + (3 * blocks + 1) * 128;
  if (!n || bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("DF final validation storage size overflows");
  return static_cast<std::size_t>(bytes);
}

/** Two-spin upper bound shared by native planning and the prepared owner.
 * Full C/epsilon snapshots and generation/info words are per item, unlike
 * the one serialized ordinary correction eigensystem. */
inline std::size_t df_final_snapshot_device_reservation(std::size_t n, std::size_t batch) {
  const long double bytes = static_cast<long double>(batch) *
                            (2.0L * (static_cast<long double>(n) * n + n) * sizeof(double) +
                             2 * (sizeof(std::uint64_t) + sizeof(int)));
  if (!n || !batch || bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("DF final snapshot storage size overflows");
  return static_cast<std::size_t>(bytes);
}
}  // namespace vibeqc::scf

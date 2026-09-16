#pragma once

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
 * Failure clears the snapshot. No C/Python result record is extended. */
vibeqc_status read_cuda_density_fitting_final_state(CudaDensityFittingJkPlan* plan,
                                                    const CudaDfFinalStateToken& expected,
                                                    CudaDfFinalStateSnapshot& snapshot,
                                                    std::string& detail);

/** Try physical J/K from an exact current singleton RHF retained density/frame.
 * Token, device generation and every supplied density entry must match. A
 * correction step, stale token or unsupported plan returns used=false so the
 * caller evaluates dense J/K. Successful reuse still evaluates physical F[D].
 */
vibeqc_status try_cuda_density_fitting_final_rhf_jk(CudaDensityFittingJkPlan* plan,
                                                    const CudaDfFinalStateToken& expected,
                                                    const std::vector<double>& density,
                                                    std::vector<double>& coulomb,
                                                    std::vector<double>& exchange, bool& used,
                                                    std::string& detail);

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

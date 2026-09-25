#ifndef VIBEQC_SCF_SOLVER_DIIS_HPP
#define VIBEQC_SCF_SOLVER_DIIS_HPP
#include "scf/reference/linalg.hpp"
#include "solver/diis_history.hpp"

namespace vibeqc::scf::solver {
using reference::Matrix;
/** Bounded conventional DIIS history for one trajectory, shared by both spins.
 * Singular augmented solves return the unextrapolated Fock. A proposal/reset
 * clears trajectory history; history never belongs to another batch item.
 */
class Diis {
 public:
  /** KS can normalize the residual Gram block before the pivot test. A common
   * scale preserves DIIS coefficients while allowing physical residuals well
   * below 1e-7. The default preserves historical HF iteration behavior. */
  explicit Diis(std::size_t capacity, bool normalize_metric = false);
  /** Retained numerical capacity, excluding temporary extrapolation storage. */
  std::size_t numeric_capacity() const noexcept;
  void clear();
  Matrix update(const Matrix& fock, const Matrix& residual);

 private:
  ::vibeqc::solver::detail::DiisHistory history_;
  bool normalize_metric_{};
};
}  // namespace vibeqc::scf::solver
#endif

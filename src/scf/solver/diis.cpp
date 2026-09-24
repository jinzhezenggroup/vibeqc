#include "scf/solver/diis.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "generated_scf_array_native.hpp"
#include "runtime/resource_usage.hpp"

namespace vibeqc::scf::solver {
using reference::index;
using reference::solve_linear;
Diis::Diis(std::size_t capacity, bool normalize_metric)
    : history_(capacity), normalize_metric_(normalize_metric) {}

/** Actual retained numerical capacity; no temporary extrapolation work. */
std::size_t Diis::numeric_capacity() const noexcept { return history_.numeric_capacity_bytes(); }

void Diis::clear() { history_.clear(); }

Matrix Diis::update(const Matrix& fock, const Matrix& residual) {
  if (history_.capacity() < 2) return fock;
  history_.push(fock, residual);
  if (history_.size() < 2) return fock;

  for (;;) {
    const std::size_t m = history_.size();
    const std::size_t dim = m + 1;
    Matrix b(dim * dim, 0.0);
    std::vector<double> rhs(dim, 0.0);
    rhs[m] = -1.0;
    generated::diis_gram(b.data(), dim, history_.errors(), m, residual.size());
    for (std::size_t i = 0; i < m; ++i) {
      b[index(i, m, dim)] = -1.0;
      b[index(m, i, dim)] = -1.0;
    }
    if (normalize_metric_) {
      double scale = 0.0;
      for (std::size_t i = 0; i < m; ++i) scale = std::max(scale, std::abs(b[index(i, i, dim)]));
      if (!(scale > 0.0) || !std::isfinite(scale)) return fock;
      for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) b[index(i, j, dim)] /= scale;
    }
    std::vector<double> coefficients;
    if (!solve_linear(std::move(b), std::move(rhs), coefficients, dim)) {
      if (!normalize_metric_ || m <= 2) return fock;
      // Keep the most recent physical states when old, nearly dependent errors
      // make the augmented solve singular. Both spin blocks retire together.
      history_.retire_oldest();
      continue;
    }

    Matrix extrapolated(fock.size());
    generated::diis_extrapolate(extrapolated.data(), history_.vectors(), coefficients.data(), m,
                                fock.size());
    return extrapolated;
  }
}

}  // namespace vibeqc::scf::solver

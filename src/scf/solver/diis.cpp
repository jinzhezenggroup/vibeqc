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
    : capacity_(capacity), normalize_metric_(normalize_metric) {}

/** Actual retained numerical capacity; no temporary extrapolation work. */
std::size_t Diis::numeric_capacity() const noexcept {
  std::size_t bytes = 0;
  for (const auto& value : focks_)
    bytes = runtime::add_capacity(bytes, runtime::vector_bytes(value));
  for (const auto& value : residuals_)
    bytes = runtime::add_capacity(bytes, runtime::vector_bytes(value));
  return bytes;
}

void Diis::clear() {
  focks_.clear();
  residuals_.clear();
}

Matrix Diis::update(const Matrix& fock, const Matrix& residual) {
  if (capacity_ < 2) return fock;
  focks_.push_back(fock);
  residuals_.push_back(residual);
  if (focks_.size() > capacity_) {
    focks_.erase(focks_.begin());
    residuals_.erase(residuals_.begin());
  }
  if (focks_.size() < 2) return fock;

  for (;;) {
    const std::size_t m = focks_.size();
    const std::size_t dim = m + 1;
    Matrix b(dim * dim, 0.0);
    std::vector<double> rhs(dim, 0.0);
    rhs[m] = -1.0;
    generated::diis_gram(b.data(), dim, residuals_, m, residual.size());
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
      focks_.erase(focks_.begin());
      residuals_.erase(residuals_.begin());
      continue;
    }

    Matrix extrapolated(fock.size());
    generated::diis_extrapolate(extrapolated.data(), focks_, coefficients.data(), m, fock.size());
    return extrapolated;
  }
}

}  // namespace vibeqc::scf::solver

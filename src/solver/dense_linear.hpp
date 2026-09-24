#ifndef VIBEQC_SOLVER_DENSE_LINEAR_HPP
#define VIBEQC_SOLVER_DENSE_LINEAR_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace vibeqc::solver {

/** Small pivoted dense FP64 solve used by host iterative-solver policies.
 *
 * The input matrix is row-major and square with dimension rhs.size(). A pivot
 * below 1e-14 or any nonfinite pivot fails closed and leaves x unchanged.
 */
inline bool solve_dense_linear(std::vector<double> matrix, std::vector<double> rhs,
                               std::vector<double>& x) {
  const std::size_t n = rhs.size();
  if (n && n > std::numeric_limits<std::size_t>::max() / n) return false;
  if (matrix.size() != n * n) return false;

  for (std::size_t column = 0; column < n; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < n; ++row) {
      if (std::abs(matrix[row * n + column]) > std::abs(matrix[pivot * n + column])) pivot = row;
    }
    const double diagonal = matrix[pivot * n + column];
    if (!std::isfinite(diagonal) || std::abs(diagonal) < 1.0e-14) return false;
    if (pivot != column) {
      for (std::size_t j = 0; j < n; ++j) std::swap(matrix[column * n + j], matrix[pivot * n + j]);
      std::swap(rhs[column], rhs[pivot]);
    }
    for (std::size_t j = column; j < n; ++j) matrix[column * n + j] /= diagonal;
    rhs[column] /= diagonal;
    for (std::size_t row = 0; row < n; ++row) {
      if (row == column) continue;
      const double factor = matrix[row * n + column];
      for (std::size_t j = column; j < n; ++j)
        matrix[row * n + j] -= factor * matrix[column * n + j];
      rhs[row] -= factor * rhs[column];
    }
  }

  if (!std::all_of(rhs.begin(), rhs.end(), [](double value) { return std::isfinite(value); }))
    return false;
  x = std::move(rhs);
  return true;
}

}  // namespace vibeqc::solver

#endif

#ifndef VIBEQC_SOLVER_DIIS_HPP
#define VIBEQC_SOLVER_DIIS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "solver/diis_history.hpp"

namespace vibeqc::solver {

/** Method-neutral host Pulay DIIS for flattened FP64 state/error vectors.
 *
 * The implementation preserves a bounded chronological history, normalizes
 * the residual Gram block before the augmented solve, rejects pathological
 * coefficients, and retires the oldest dependent history entry before retrying.
 * Scientific meaning and residual construction remain method-owned.
 */
class Diis {
 public:
  Diis(unsigned capacity, std::size_t elements) : history_(capacity, elements) {}

  [[nodiscard]] unsigned restarts() const noexcept { return restarts_; }

  void clear() { history_.clear(); }

  std::vector<double> update(std::vector<double> vector, std::vector<double> error) {
    history_.validate(vector, error);
    if (!history_.capacity()) return vector;
    history_.push(vector, std::move(error));
    while (history_.size() > 1) {
      const auto n = history_.size();
      std::vector<double> gram(n * n);
      double scale = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j) {
          const double dot =
              std::inner_product(history_.errors()[i].begin(), history_.errors()[i].end(),
                                 history_.errors()[j].begin(), 0.0);
          gram[i * n + j] = dot;
          gram[j * n + i] = dot;
          scale = std::max(scale, std::abs(dot));
        }
      if (scale == 0.0) return vector;

      std::vector<double> system((n + 1) * (n + 1)), rhs(n + 1), solution;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) system[i * (n + 1) + j] = gram[i * n + j] / scale;
      for (std::size_t i = 0; i < n; ++i) system[i * (n + 1) + n] = system[n * (n + 1) + i] = -1.0;
      rhs[n] = -1.0;

      if (solve_linear(system, rhs, solution)) {
        solution.resize(n);
        if (std::all_of(solution.begin(), solution.end(),
                        [](double c) { return std::isfinite(c) && std::abs(c) <= 1e6; })) {
          std::vector<double> result(history_.elements());
          for (std::size_t row = 0; row < n; ++row)
            for (std::size_t i = 0; i < history_.elements(); ++i)
              result[i] += solution[row] * history_.vectors()[row][i];
          return result;
        }
      }

      history_.retire_oldest();
      ++restarts_;
    }
    return vector;
  }

 private:
  static bool solve_linear(std::vector<double> matrix, std::vector<double> rhs,
                           std::vector<double>& x) {
    const auto n = rhs.size();
    for (std::size_t col = 0; col < n; ++col) {
      auto pivot = col;
      for (std::size_t row = col + 1; row < n; ++row)
        if (std::abs(matrix[row * n + col]) > std::abs(matrix[pivot * n + col])) pivot = row;
      const double divisor = matrix[pivot * n + col];
      if (!std::isfinite(divisor) || std::abs(divisor) < 1e-14) return false;
      if (pivot != col) {
        for (std::size_t j = 0; j < n; ++j) std::swap(matrix[col * n + j], matrix[pivot * n + j]);
        std::swap(rhs[col], rhs[pivot]);
      }
      for (std::size_t j = col; j < n; ++j) matrix[col * n + j] /= divisor;
      rhs[col] /= divisor;
      for (std::size_t row = 0; row < n; ++row) {
        if (row == col) continue;
        const double factor = matrix[row * n + col];
        for (std::size_t j = col; j < n; ++j) matrix[row * n + j] -= factor * matrix[col * n + j];
        rhs[row] -= factor * rhs[col];
      }
    }
    x = std::move(rhs);
    return std::all_of(x.begin(), x.end(), [](double y) { return std::isfinite(y); });
  }

  unsigned restarts_{};
  detail::DiisHistory history_;
};

}  // namespace vibeqc::solver

#endif

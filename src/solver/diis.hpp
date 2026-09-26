#ifndef VIBEQC_SOLVER_DIIS_HPP
#define VIBEQC_SOLVER_DIIS_HPP

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "solver/diis_coefficients.hpp"
#include "solver/diis_history.hpp"

namespace vibeqc::solver {

/** Method-neutral host Pulay DIIS for flattened FP64 state/error vectors.
 *
 * The implementation preserves a bounded chronological history, uses the
 * shared augmented coefficient policy, and retains its historical normalized
 * metric, coefficient guard and dependent-history retirement behavior.
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
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j) {
          const double dot =
              std::inner_product(history_.errors()[i].begin(), history_.errors()[i].end(),
                                 history_.errors()[j].begin(), 0.0);
          gram[i * n + j] = dot;
          gram[j * n + i] = dot;
        }

      detail::DiisCoefficientPolicy policy;
      policy.metric_scaling = detail::DiisMetricScaling::MaximumAbsoluteEntry;
      policy.failure_retirement_floor = 1;
      policy.maximum_abs_coefficient = 1.0e6;
      std::vector<double> coefficients;
      const auto action = detail::solve_diis_coefficients(gram, n, policy, coefficients);
      if (action == detail::DiisCoefficientAction::Extrapolate) {
        std::vector<double> result(history_.elements());
        for (std::size_t row = 0; row < n; ++row)
          for (std::size_t i = 0; i < history_.elements(); ++i)
            result[i] += coefficients[row] * history_.vectors()[row][i];
        return result;
      }
      if (action == detail::DiisCoefficientAction::RetainCurrent) return vector;
      history_.retire_oldest();
      ++restarts_;
    }
    return vector;
  }

 private:
  unsigned restarts_{};
  detail::DiisHistory history_;
};

}  // namespace vibeqc::solver

#endif

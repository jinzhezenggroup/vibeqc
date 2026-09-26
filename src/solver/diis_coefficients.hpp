#ifndef VIBEQC_SOLVER_DIIS_COEFFICIENTS_HPP
#define VIBEQC_SOLVER_DIIS_COEFFICIENTS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "solver/dense_linear.hpp"

namespace vibeqc::solver::detail {

enum class DiisMetricScaling { None, MaximumAbsoluteEntry, MaximumDiagonal };

enum class DiisCoefficientAction { Extrapolate, RetainCurrent, RetireOldest };

/** Method-neutral policy for the small augmented Pulay coefficient solve.
 *
 * Scientific residual construction and Gram/extrapolation kernels remain with
 * their callers. This policy only owns metric scaling, coefficient acceptance,
 * and whether a failed solve retires the oldest history entry before retrying.
 */
struct DiisCoefficientPolicy {
  DiisMetricScaling metric_scaling{DiisMetricScaling::None};
  /** Retire after a failed/guard-rejected solve only while history exceeds this floor. */
  std::size_t failure_retirement_floor{std::numeric_limits<std::size_t>::max()};
  /** Infinite preserves callers that historically accepted every finite coefficient. */
  double maximum_abs_coefficient{std::numeric_limits<double>::infinity()};
};

inline DiisCoefficientAction solve_diis_coefficients(const std::vector<double>& gram,
                                                     std::size_t history_size,
                                                     const DiisCoefficientPolicy& policy,
                                                     std::vector<double>& coefficients) {
  coefficients.clear();
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (!history_size || history_size > maximum / history_size ||
      gram.size() != history_size * history_size || history_size == maximum)
    return DiisCoefficientAction::RetainCurrent;

  double scale = 1.0;
  if (policy.metric_scaling == DiisMetricScaling::MaximumAbsoluteEntry) {
    scale = 0.0;
    for (double value : gram) scale = std::max(scale, std::abs(value));
    if (scale == 0.0) return DiisCoefficientAction::RetainCurrent;
  } else if (policy.metric_scaling == DiisMetricScaling::MaximumDiagonal) {
    scale = 0.0;
    for (std::size_t i = 0; i < history_size; ++i)
      scale = std::max(scale, std::abs(gram[i * history_size + i]));
    if (!(scale > 0.0) || !std::isfinite(scale)) return DiisCoefficientAction::RetainCurrent;
  }

  const std::size_t dimension = history_size + 1;
  if (dimension > maximum / dimension) return DiisCoefficientAction::RetainCurrent;
  std::vector<double> system(dimension * dimension, 0.0);
  std::vector<double> rhs(dimension, 0.0);
  for (std::size_t i = 0; i < history_size; ++i) {
    for (std::size_t j = 0; j < history_size; ++j) {
      const double value = gram[i * history_size + j];
      system[i * dimension + j] =
          policy.metric_scaling == DiisMetricScaling::None ? value : value / scale;
    }
    system[i * dimension + history_size] = -1.0;
    system[history_size * dimension + i] = -1.0;
  }
  rhs[history_size] = -1.0;

  if (solve_dense_linear(std::move(system), std::move(rhs), coefficients)) {
    const bool accepted = std::all_of(coefficients.begin(), coefficients.end(), [&](double value) {
      return std::isfinite(value) && std::abs(value) <= policy.maximum_abs_coefficient;
    });
    if (accepted) {
      coefficients.resize(history_size);
      return DiisCoefficientAction::Extrapolate;
    }
  }
  coefficients.clear();
  return history_size > policy.failure_retirement_floor ? DiisCoefficientAction::RetireOldest
                                                        : DiisCoefficientAction::RetainCurrent;
}

}  // namespace vibeqc::solver::detail

#endif

#include "response/native_gmres.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc::response {
namespace {
std::size_t checked_add(std::size_t first, std::size_t second) {
  if (second > std::numeric_limits<std::size_t>::max() - first)
    throw std::overflow_error("GMRES workspace size overflow");
  return first + second;
}

std::size_t checked_multiply(std::size_t first, std::size_t second) {
  if (first && second > std::numeric_limits<std::size_t>::max() / first)
    throw std::overflow_error("GMRES workspace size overflow");
  return first * second;
}

bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}

double relative_residual(double residual, double rhs) {
  if (rhs > 0.0) return residual / rhs;
  return residual == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
}

GmresResult result_for(const GmresPlan& plan, std::vector<double> solution,
                       GmresStatus status, double residual, double rhs_norm,
                       std::size_t iterations, std::size_t restarts,
                       std::size_t operator_actions,
                       std::size_t preconditioner_actions) {
  GmresResult result;
  result.solution = std::move(solution);
  result.status = status;
  result.residual_norm = residual;
  result.relative_residual = relative_residual(residual, rhs_norm);
  result.iterations = iterations;
  result.restarts = restarts;
  result.operator_actions = operator_actions;
  result.preconditioner_actions = preconditioner_actions;
  result.workspace_bytes = plan.workspace_bytes;
  return result;
}

bool solve_upper(std::span<const double> hessenberg, std::size_t stride,
                 std::span<const double> transformed_rhs, std::size_t count,
                 double tolerance, std::span<double> solution) {
  std::fill(solution.begin(), solution.end(), 0.0);
  for (std::size_t reverse = count; reverse > 0; --reverse) {
    const auto row = reverse - 1;
    double value = transformed_rhs[row];
    for (std::size_t column = row + 1; column < count; ++column)
      value -= hessenberg[row * stride + column] * solution[column];
    const double diagonal = hessenberg[row * stride + row];
    if (!std::isfinite(diagonal) || std::abs(diagonal) <= tolerance) return false;
    solution[row] = value / diagonal;
    if (!std::isfinite(solution[row])) return false;
  }
  return true;
}
}  // namespace

double stable_norm(std::span<const double> values) {
  double scale = 0.0;
  for (double value : values) {
    if (!std::isfinite(value))
      throw std::invalid_argument("GMRES norm requires finite vector values");
    scale = std::max(scale, std::abs(value));
  }
  if (scale == 0.0) return 0.0;
  double squares = 0.0;
  for (double value : values) {
    const double scaled = value / scale;
    squares += scaled * scaled;
  }
  const double result = scale * std::sqrt(squares);
  if (!std::isfinite(result)) throw std::overflow_error("GMRES vector norm overflows FP64");
  return result;
}

GmresPlan prepare_gmres(std::size_t dimension, const GmresOptions& options) {
  if (!dimension || !options.restart || !options.max_iterations ||
      !options.max_workspace_bytes || !options.reorthogonalize ||
      !options.true_residual_every || !options.stagnation_window ||
      !std::isfinite(options.relative_tolerance) || options.relative_tolerance < 0.0 ||
      options.relative_tolerance >= 1.0 || !std::isfinite(options.absolute_tolerance) ||
      options.absolute_tolerance < 0.0 || !std::isfinite(options.breakdown_tolerance) ||
      options.breakdown_tolerance < 0.0 || !std::isfinite(options.stagnation_tolerance) ||
      options.stagnation_tolerance < 0.0)
    throw std::invalid_argument("invalid bounded GMRES options");
  const auto restart = std::min({dimension, options.restart, options.max_iterations});
  const auto restart_plus_one = checked_add(restart, 1);
  const auto iterations_plus_two = checked_add(options.max_iterations, 2);
  std::size_t elements = checked_multiply(restart_plus_one, dimension);
  elements = checked_add(elements, checked_multiply(4, checked_multiply(dimension, restart)));
  elements = checked_add(elements, checked_multiply(20, dimension));
  elements = checked_add(
      elements, checked_multiply(8, checked_multiply(restart_plus_one, restart_plus_one)));
  elements = checked_add(elements, checked_multiply(4, iterations_plus_two));
  return {dimension, restart, checked_multiply(elements, sizeof(double)), options};
}

GmresResult solve_gmres(const GmresPlan& plan, const LinearOperator& apply,
                        std::span<const double> rhs, std::span<const double> initial_guess,
                        std::span<const double> diagonal_preconditioner) {
  const auto n = plan.dimension;
  std::vector<double> x(n, 0.0);
  if (rhs.size() != n || !finite(rhs) ||
      (!initial_guess.empty() && (initial_guess.size() != n || !finite(initial_guess))) ||
      (!diagonal_preconditioner.empty() &&
       (diagonal_preconditioner.size() != n || !finite(diagonal_preconditioner))))
    return result_for(plan, std::move(x), GmresStatus::nonfinite_input,
                      std::numeric_limits<double>::infinity(), 0.0, 0, 0, 0, 0);
  if (!apply) throw std::invalid_argument("GMRES operator callback is empty");
  for (double value : diagonal_preconditioner)
    if (std::abs(value) <= plan.options.breakdown_tolerance)
      throw std::invalid_argument("GMRES diagonal preconditioner contains a zero entry");
  if (!initial_guess.empty()) std::copy(initial_guess.begin(), initial_guess.end(), x.begin());
  if (plan.options.max_workspace_bytes < plan.workspace_bytes)
    return result_for(plan, std::move(x), GmresStatus::workspace_limit,
                      std::numeric_limits<double>::infinity(), 0.0, 0, 0, 0, 0);

  std::size_t operator_actions = 0;
  std::size_t preconditioner_actions = 0;
  auto apply_checked = [&](std::span<const double> input, std::span<double> output) {
    std::fill(output.begin(), output.end(), 0.0);
    apply(input, output);
    ++operator_actions;
    return finite(output);
  };
  auto precondition = [&](std::span<const double> input, std::span<double> output) {
    if (diagonal_preconditioner.empty()) {
      std::copy(input.begin(), input.end(), output.begin());
    } else {
      ++preconditioner_actions;
      for (std::size_t index = 0; index < n; ++index)
        output[index] = input[index] / diagonal_preconditioner[index];
    }
  };

  double rhs_norm = 0.0;
  try {
    rhs_norm = stable_norm(rhs);
  } catch (const std::exception&) {
    return result_for(plan, std::move(x), GmresStatus::nonfinite_input,
                      std::numeric_limits<double>::infinity(), 0.0, 0, 0, 0, 0);
  }
  const double target =
      std::max(plan.options.absolute_tolerance, plan.options.relative_tolerance * rhs_norm);
  std::vector<double> image(n), residual(rhs.begin(), rhs.end());
  if (!initial_guess.empty()) {
    if (!apply_checked(x, image))
      return result_for(plan, std::move(x), GmresStatus::nonfinite_operator,
                        std::numeric_limits<double>::infinity(), rhs_norm, 0, 0,
                        operator_actions, preconditioner_actions);
    for (std::size_t index = 0; index < n; ++index) residual[index] -= image[index];
  }
  double beta = 0.0;
  try {
    beta = stable_norm(residual);
  } catch (const std::exception&) {
    return result_for(plan, std::move(x), GmresStatus::nonfinite_input,
                      std::numeric_limits<double>::infinity(), rhs_norm, 0, 0,
                      operator_actions, preconditioner_actions);
  }
  if (beta <= target)
    return result_for(plan, std::move(x), GmresStatus::initial_residual, beta, rhs_norm, 0, 0,
                      operator_actions, preconditioner_actions);

  const auto restart = plan.restart;
  std::vector<double> basis((restart + 1) * n), preconditioned(restart * n);
  std::vector<double> hessenberg((restart + 1) * restart);
  std::vector<double> cosine(restart), sine(restart), transformed(restart + 1);
  std::vector<double> work(n), candidate(n), candidate_image(n), candidate_residual(n);
  std::vector<double> best_x(x), best_residual_vector(residual), coefficients(restart);
  std::size_t iterations = 0, restarts = 0, stagnation = 0;

  while (iterations < plan.options.max_iterations) {
    std::fill(basis.begin(), basis.end(), 0.0);
    std::fill(preconditioned.begin(), preconditioned.end(), 0.0);
    std::fill(hessenberg.begin(), hessenberg.end(), 0.0);
    std::fill(transformed.begin(), transformed.end(), 0.0);
    transformed[0] = beta;
    for (std::size_t index = 0; index < n; ++index) basis[index] = residual[index] / beta;
    const auto base_x = x;
    double best_norm = beta;
    bool completed_cycle = false;

    for (std::size_t column = 0;
         column < restart && iterations < plan.options.max_iterations; ++column) {
      const auto basis_column =
          std::span<const double>(basis.data() + column * n, n);
      auto z_column = std::span<double>(preconditioned.data() + column * n, n);
      precondition(basis_column, z_column);
      if (!apply_checked(z_column, work))
        return result_for(plan, std::move(best_x), GmresStatus::nonfinite_operator,
                          best_norm, rhs_norm, iterations, restarts, operator_actions,
                          preconditioner_actions);
      for (unsigned pass = 0; pass < plan.options.reorthogonalize; ++pass) {
        for (std::size_t row = 0; row <= column; ++row) {
          const auto prior = std::span<const double>(basis.data() + row * n, n);
          double projection = 0.0;
          for (std::size_t index = 0; index < n; ++index)
            projection += prior[index] * work[index];
          hessenberg[row * restart + column] += projection;
          for (std::size_t index = 0; index < n; ++index)
            work[index] -= projection * prior[index];
        }
      }
      double next_norm = 0.0;
      try {
        next_norm = stable_norm(work);
      } catch (const std::exception&) {
        return result_for(plan, std::move(best_x), GmresStatus::nonfinite_operator,
                          best_norm, rhs_norm, iterations, restarts, operator_actions,
                          preconditioner_actions);
      }
      hessenberg[(column + 1) * restart + column] = next_norm;
      const bool broke_down = next_norm <= plan.options.breakdown_tolerance;
      if (!broke_down)
        for (std::size_t index = 0; index < n; ++index)
          basis[(column + 1) * n + index] = work[index] / next_norm;

      for (std::size_t row = 0; row < column; ++row) {
        const double upper = hessenberg[row * restart + column];
        const double lower = hessenberg[(row + 1) * restart + column];
        hessenberg[row * restart + column] = cosine[row] * upper + sine[row] * lower;
        hessenberg[(row + 1) * restart + column] =
            -sine[row] * upper + cosine[row] * lower;
      }
      const double upper = hessenberg[column * restart + column];
      const double lower = hessenberg[(column + 1) * restart + column];
      const double rotation_norm = std::hypot(upper, lower);
      if (rotation_norm > plan.options.breakdown_tolerance) {
        cosine[column] = upper / rotation_norm;
        sine[column] = lower / rotation_norm;
      } else {
        cosine[column] = 1.0;
        sine[column] = 0.0;
      }
      hessenberg[column * restart + column] =
          cosine[column] * upper + sine[column] * lower;
      hessenberg[(column + 1) * restart + column] = 0.0;
      const double transformed_upper = transformed[column];
      transformed[column] = cosine[column] * transformed_upper;
      transformed[column + 1] = -sine[column] * transformed_upper;
      ++iterations;

      const bool checkpoint = broke_down || column + 1 == restart ||
                              iterations == plan.options.max_iterations ||
                              iterations % plan.options.true_residual_every == 0;
      if (!checkpoint) continue;
      const std::size_t columns = column + 1;
      const bool solvable = solve_upper(hessenberg, restart, transformed, columns,
                                        plan.options.breakdown_tolerance, coefficients);
      candidate = base_x;
      if (solvable)
        for (std::size_t vector = 0; vector < columns; ++vector)
          for (std::size_t index = 0; index < n; ++index)
            candidate[index] += preconditioned[vector * n + index] * coefficients[vector];
      if (!apply_checked(candidate, candidate_image))
        return result_for(plan, std::move(best_x), GmresStatus::nonfinite_operator,
                          best_norm, rhs_norm, iterations, restarts, operator_actions,
                          preconditioner_actions);
      for (std::size_t index = 0; index < n; ++index)
        candidate_residual[index] = rhs[index] - candidate_image[index];
      double candidate_norm = 0.0;
      try {
        candidate_norm = stable_norm(candidate_residual);
      } catch (const std::exception&) {
        return result_for(plan, std::move(best_x), GmresStatus::nonfinite_operator,
                          best_norm, rhs_norm, iterations, restarts, operator_actions,
                          preconditioner_actions);
      }
      if (candidate_norm < best_norm) {
        best_norm = candidate_norm;
        best_x = candidate;
        best_residual_vector = candidate_residual;
        stagnation = 0;
      } else if (candidate_norm >=
                 best_norm * (1.0 - plan.options.stagnation_tolerance)) {
        ++stagnation;
      }
      if (candidate_norm <= target)
        return result_for(plan, std::move(candidate), GmresStatus::converged,
                          candidate_norm, rhs_norm, iterations, restarts, operator_actions,
                          preconditioner_actions);
      if (broke_down || !solvable)
        return result_for(plan, std::move(best_x), GmresStatus::breakdown, best_norm,
                          rhs_norm, iterations, restarts, operator_actions,
                          preconditioner_actions);
      if (stagnation >= plan.options.stagnation_window)
        return result_for(plan, std::move(best_x), GmresStatus::stagnation, best_norm,
                          rhs_norm, iterations, restarts, operator_actions,
                          preconditioner_actions);
      if (column + 1 == restart || iterations == plan.options.max_iterations) {
        completed_cycle = true;
        break;
      }
    }
    x = best_x;
    residual = best_residual_vector;
    beta = best_norm;
    if (iterations >= plan.options.max_iterations) break;
    if (!completed_cycle || beta == 0.0) break;
    ++restarts;
  }
  return result_for(plan, std::move(best_x), GmresStatus::max_iterations, beta, rhs_norm,
                    iterations, restarts, operator_actions, preconditioner_actions);
}

}  // namespace vibeqc::response

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "response/native_gmres.hpp"

namespace {
using vibeqc::response::GmresOptions;
using vibeqc::response::GmresStatus;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct DenseOperator {
  std::size_t dimension{};
  std::vector<double> values;
  std::size_t actions{};
  bool emit_nonfinite{};

  void operator()(std::span<const double> input, std::span<double> output) {
    ++actions;
    require(input.size() == dimension && output.size() == dimension,
            "operator received the wrong dimension");
    if (emit_nonfinite) {
      std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
      return;
    }
    for (std::size_t row = 0; row < dimension; ++row) {
      output[row] = 0.0;
      for (std::size_t column = 0; column < dimension; ++column)
        output[row] += values[row * dimension + column] * input[column];
    }
  }
};

double explicit_residual(const DenseOperator& matrix, std::span<const double> x,
                         std::span<const double> rhs) {
  std::vector<double> residual(rhs.begin(), rhs.end());
  for (std::size_t row = 0; row < matrix.dimension; ++row)
    for (std::size_t column = 0; column < matrix.dimension; ++column)
      residual[row] -= matrix.values[row * matrix.dimension + column] * x[column];
  return vibeqc::response::stable_norm(residual);
}

void exact_solve_and_true_residual() {
  DenseOperator matrix{2, {4.0, 1.0, 2.0, 3.0}};
  const std::array<double, 2> rhs{1.0, 2.0};
  GmresOptions options;
  options.relative_tolerance = 1e-13;
  options.restart = 2;
  options.max_iterations = 4;
  const auto plan = vibeqc::response::prepare_gmres(2, options);
  const auto result = vibeqc::response::solve_gmres(
      plan, [&](auto input, auto output) { matrix(input, output); }, rhs);
  require(result.status == GmresStatus::converged, "2x2 solve did not converge");
  require(std::abs(result.solution[0] - 0.1) < 1e-12 && std::abs(result.solution[1] - 0.6) < 1e-12,
          "2x2 solution is wrong");
  const auto residual = explicit_residual(matrix, result.solution, rhs);
  require(std::abs(result.residual_norm - residual) < 1e-15,
          "reported residual is not the recomputed true residual");
  require(result.relative_residual < 1e-12 && result.operator_actions == matrix.actions,
          "2x2 diagnostics are inconsistent");
}

void zero_rhs_is_transactional() {
  DenseOperator identity{2, {1.0, 0.0, 0.0, 1.0}};
  const std::array<double, 2> rhs{};
  const auto plan = vibeqc::response::prepare_gmres(2, {});
  const auto result = vibeqc::response::solve_gmres(
      plan, [&](auto input, auto output) { identity(input, output); }, rhs);
  require(result.status == GmresStatus::initial_residual && result.converged(),
          "zero RHS did not use the initial-residual path");
  require(result.iterations == 0 && result.operator_actions == 0 && identity.actions == 0,
          "zero RHS applied the operator");
  require(result.residual_norm == 0.0 && result.relative_residual == 0.0,
          "zero RHS residual convention is wrong");
}

void restarted_and_exhausted_paths() {
  DenseOperator diagonal{3, {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0}};
  const std::array<double, 3> rhs{1.0, -2.0, 3.0};
  GmresOptions restarted;
  restarted.relative_tolerance = 1e-11;
  restarted.restart = 1;
  restarted.max_iterations = 80;
  const auto converged = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(3, restarted),
      [&](auto input, auto output) { diagonal(input, output); }, rhs);
  require(converged.status == GmresStatus::converged && converged.restarts > 0,
          "restart-one solve did not exercise a restart");
  require(explicit_residual(diagonal, converged.solution, rhs) < 1e-10,
          "restarted solution has a large true residual");

  DenseOperator limited{3, diagonal.values};
  GmresOptions exhausted = restarted;
  exhausted.relative_tolerance = 1e-15;
  exhausted.max_iterations = 1;
  const auto failed = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(3, exhausted),
      [&](auto input, auto output) { limited(input, output); }, rhs);
  require(
      failed.status == GmresStatus::max_iterations && !failed.converged() && failed.iterations == 1,
      "iteration exhaustion claimed success or returned the wrong status");
  require(std::abs(failed.residual_norm - explicit_residual(limited, failed.solution, rhs)) < 1e-14,
          "exhausted solve did not publish its true residual");
}

void breakdown_and_nonfinite_paths() {
  DenseOperator zero{2, {0.0, 0.0, 0.0, 0.0}};
  const std::array<double, 2> rhs{1.0, 1.0};
  const auto broken = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, {}), [&](auto input, auto output) { zero(input, output); },
      rhs);
  require(broken.status == GmresStatus::breakdown && !broken.converged(),
          "singular operator did not report breakdown");
  require(std::abs(broken.relative_residual - 1.0) < 1e-15,
          "breakdown residual diagnostic is wrong");

  DenseOperator identity{2, {1.0, 0.0, 0.0, 1.0}};
  const std::array<double, 2> bad_rhs{1.0, std::numeric_limits<double>::infinity()};
  const auto invalid = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, {}),
      [&](auto input, auto output) { identity(input, output); }, bad_rhs);
  require(invalid.status == GmresStatus::nonfinite_input && identity.actions == 0,
          "nonfinite RHS was not rejected before the operator");

  DenseOperator bad_operator{2, {1.0, 0.0, 0.0, 1.0}};
  bad_operator.emit_nonfinite = true;
  const auto emitted = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, {}),
      [&](auto input, auto output) { bad_operator(input, output); }, rhs);
  require(emitted.status == GmresStatus::nonfinite_operator && !emitted.converged(),
          "nonfinite operator output was not diagnosed");
}

void options_and_workspace_boundaries() {
  bool rejected = false;
  try {
    auto options = GmresOptions{};
    options.restart = 0;
    (void)vibeqc::response::prepare_gmres(2, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "zero restart was accepted");
  rejected = false;
  try {
    auto options = GmresOptions{};
    options.relative_tolerance = std::numeric_limits<double>::quiet_NaN();
    (void)vibeqc::response::prepare_gmres(2, options);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "nonfinite tolerance was accepted");
  rejected = false;
  try {
    auto options = GmresOptions{};
    options.restart = std::numeric_limits<std::size_t>::max();
    options.max_iterations = std::numeric_limits<std::size_t>::max();
    options.max_workspace_bytes = std::numeric_limits<std::size_t>::max();
    (void)vibeqc::response::prepare_gmres(std::numeric_limits<std::size_t>::max(), options);
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected, "GMRES workspace control overflow was accepted");

  GmresOptions probe_options;
  probe_options.restart = 2;
  probe_options.max_iterations = 4;
  const auto probe = vibeqc::response::prepare_gmres(2, probe_options);
  require(probe.workspace_bytes > 1, "workspace plan is empty");
  const std::array<double, 2> rhs{1.0, 2.0};
  DenseOperator exact_matrix{2, {2.0, 0.0, 0.0, 3.0}};
  auto exact_options = probe_options;
  exact_options.max_workspace_bytes = probe.workspace_bytes;
  const auto exact = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, exact_options),
      [&](auto input, auto output) { exact_matrix(input, output); }, rhs);
  require(exact.converged(), "exact workspace budget was rejected");

  DenseOperator short_matrix{2, exact_matrix.values};
  auto short_options = exact_options;
  short_options.max_workspace_bytes = probe.workspace_bytes - 1;
  const auto short_result = vibeqc::response::solve_gmres(
      vibeqc::response::prepare_gmres(2, short_options),
      [&](auto input, auto output) { short_matrix(input, output); }, rhs);
  require(short_result.status == GmresStatus::workspace_limit &&
              short_result.operator_actions == 0 && short_matrix.actions == 0,
          "one-byte-short workspace was not rejected before the operator");
  require(short_result.workspace_bytes == probe.workspace_bytes,
          "workspace rejection lost the exact requirement");
  require(short_result.solution.empty(),
          "workspace refusal allocated a dimension-sized solution");
}

void stable_norm_extremes() {
  const std::array<double, 2> tiny{1e-200, 0.0};
  const std::array<double, 2> large{1e200, 0.0};
  require(vibeqc::response::stable_norm(tiny) == 1e-200, "stable norm underflowed a finite vector");
  require(vibeqc::response::stable_norm(large) == 1e200, "stable norm overflowed a finite vector");
  bool rejected = false;
  try {
    const std::array<double, 2> impossible{std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max()};
    (void)vibeqc::response::stable_norm(impossible);
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected, "unrepresentable finite norm was accepted");
}
}  // namespace

int main() {
  try {
    exact_solve_and_true_residual();
    zero_rhs_is_transactional();
    restarted_and_exhausted_paths();
    breakdown_and_nonfinite_paths();
    options_and_workspace_boundaries();
    stable_norm_extremes();
    std::cout << "Native GMRES contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

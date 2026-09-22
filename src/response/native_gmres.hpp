#pragma once

#include <cstddef>
#include <span>

#include "response/linear_problem.hpp"
#include "runtime/tracked_allocator.hpp"

namespace vibeqc::response {

enum class GmresStatus {
  initial_residual,
  converged,
  workspace_limit,
  max_iterations,
  breakdown,
  stagnation,
  nonfinite_input,
  nonfinite_operator,
};

struct GmresOptions {
  double relative_tolerance{1e-10};
  double absolute_tolerance{};
  std::size_t restart{30};
  std::size_t max_iterations{200};
  std::size_t max_workspace_bytes{64ULL << 20};
  unsigned reorthogonalize{2};
  double breakdown_tolerance{1e-14};
  std::size_t true_residual_every{1};
  std::size_t stagnation_window{25};
  double stagnation_tolerance{1e-14};
};

struct GmresPlan {
  std::size_t dimension{};
  std::size_t restart{};
  std::size_t workspace_bytes{};
  GmresOptions options;
};

struct GmresResult {
  /** Empty on workspace refusal; no dimension-sized output is allocated. */
  runtime::TrackedVector<double> solution;
  GmresStatus status{GmresStatus::nonfinite_input};
  double residual_norm{};
  double relative_residual{};
  std::size_t iterations{};
  std::size_t restarts{};
  std::size_t operator_actions{};
  std::size_t preconditioner_actions{};
  std::size_t workspace_bytes{};
  /** Actual high-water mark of GMRES-owned array payload, including its result.
   * Excludes input spans, operator callback allocations and allocator overhead;
   * this is not a complete MP2 endpoint measurement. Zero on admission refusal.
   */
  std::size_t measured_workspace_peak_bytes{};
  std::size_t workspace_allocation_count{};

  [[nodiscard]] bool converged() const noexcept {
    return status == GmresStatus::initial_residual || status == GmresStatus::converged;
  }
};

double stable_norm(std::span<const double> values);
GmresPlan prepare_gmres(std::size_t dimension, const GmresOptions& options);
GmresResult solve_gmres(const GmresPlan& plan, const LinearOperator& apply,
                        std::span<const double> rhs, std::span<const double> initial_guess = {},
                        std::span<const double> diagonal_preconditioner = {});

}  // namespace vibeqc::response

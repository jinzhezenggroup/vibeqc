#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "response/linear_problem.hpp"
#include "response/native_gmres.hpp"

namespace vibeqc::response {

struct ResponseSolveRequest {
  std::span<const double> rhs;
  std::span<const double> initial_guess{};
  std::span<const double> diagonal_preconditioner{};
};

enum class ResponseBatchStatus { complete, workspace_limit };

/** Numeric-payload admission for a sequential multi-RHS response batch.
 *
 * Each solve may use the full GMRES workspace while solutions from earlier
 * requests remain live. `peak_numeric_bytes` therefore includes one active
 * solver workspace plus all previously retained solution vectors. It excludes
 * `std::vector`/allocator bookkeeping and any storage owned by the operator or
 * method adapter.
 */
struct ResponseBatchPlan {
  GmresPlan solve_plan;
  std::size_t request_count{};
  std::size_t retained_solution_bytes{};
  std::size_t peak_numeric_bytes{};
  std::size_t max_numeric_bytes{};
  bool admitted{};
};

struct ResponseBatchResult {
  std::vector<GmresResult> responses;
  ResponseBatchStatus status{ResponseBatchStatus::complete};
  std::size_t planned_peak_numeric_bytes{};
  std::size_t retained_solution_bytes{};

  [[nodiscard]] bool complete() const noexcept { return status == ResponseBatchStatus::complete; }
};

namespace detail {
inline std::size_t response_checked_add(std::size_t first, std::size_t second) {
  if (second > std::numeric_limits<std::size_t>::max() - first)
    throw std::overflow_error("response batch workspace size overflow");
  return first + second;
}

inline std::size_t response_checked_multiply(std::size_t first, std::size_t second) {
  if (first && second > std::numeric_limits<std::size_t>::max() / first)
    throw std::overflow_error("response batch workspace size overflow");
  return first * second;
}
}  // namespace detail

inline GmresPlan prepare_response(const LinearResponseProblem& problem,
                                  const GmresOptions& options) {
  return prepare_gmres(problem.dimension(), options);
}

inline GmresResult solve_response(const GmresPlan& plan, const LinearResponseProblem& problem,
                                  ResponseSolveRequest request) {
  if (plan.dimension != problem.dimension())
    throw std::invalid_argument("response plan and problem dimensions do not match");
  return solve_gmres(plan, problem.apply(), request.rhs, request.initial_guess,
                     request.diagonal_preconditioner);
}

inline GmresResult solve_response(const GmresPlan& plan, const LinearResponseProblem& problem,
                                  std::span<const double> rhs,
                                  std::span<const double> initial_guess = {},
                                  std::span<const double> diagonal_preconditioner = {}) {
  return solve_response(plan, problem, {rhs, initial_guess, diagonal_preconditioner});
}

inline ResponseBatchPlan prepare_response_batch(const LinearResponseProblem& problem,
                                                const GmresOptions& options,
                                                std::size_t request_count,
                                                std::size_t max_numeric_bytes) {
  const auto solve_plan = prepare_response(problem, options);
  if (request_count == 0) return {solve_plan, 0, 0, 0, max_numeric_bytes, true};

  const auto solution_bytes =
      detail::response_checked_multiply(problem.dimension(), sizeof(double));
  const auto retained_solution_bytes =
      detail::response_checked_multiply(request_count, solution_bytes);
  const auto prior_solution_bytes =
      detail::response_checked_multiply(request_count - 1, solution_bytes);
  const auto peak_numeric_bytes =
      detail::response_checked_add(solve_plan.workspace_bytes, prior_solution_bytes);
  const bool admitted = solve_plan.options.max_workspace_bytes >= solve_plan.workspace_bytes &&
                        peak_numeric_bytes <= max_numeric_bytes;
  return {solve_plan,         request_count,     retained_solution_bytes,
          peak_numeric_bytes, max_numeric_bytes, admitted};
}

inline ResponseBatchResult solve_response_batch(const ResponseBatchPlan& plan,
                                                const LinearResponseProblem& problem,
                                                std::span<const ResponseSolveRequest> requests) {
  const auto expected = prepare_response_batch(problem, plan.solve_plan.options, plan.request_count,
                                               plan.max_numeric_bytes);
  if (plan.solve_plan.dimension != expected.solve_plan.dimension ||
      plan.solve_plan.restart != expected.solve_plan.restart ||
      plan.solve_plan.workspace_bytes != expected.solve_plan.workspace_bytes ||
      plan.retained_solution_bytes != expected.retained_solution_bytes ||
      plan.peak_numeric_bytes != expected.peak_numeric_bytes || plan.admitted != expected.admitted)
    throw std::invalid_argument("response batch plan does not match its problem and options");
  if (requests.size() != plan.request_count)
    throw std::invalid_argument("response batch request count does not match its plan");
  if (!plan.admitted) return {{}, ResponseBatchStatus::workspace_limit, plan.peak_numeric_bytes, 0};

  ResponseBatchResult result;
  result.responses.reserve(requests.size());
  result.status = ResponseBatchStatus::complete;
  result.planned_peak_numeric_bytes = plan.peak_numeric_bytes;
  for (const auto& request : requests)
    result.responses.push_back(solve_response(plan.solve_plan, problem, request));
  result.retained_solution_bytes = plan.retained_solution_bytes;
  return result;
}

}  // namespace vibeqc::response

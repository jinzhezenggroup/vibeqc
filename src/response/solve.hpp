#pragma once

#include <span>
#include <stdexcept>

#include "response/linear_problem.hpp"
#include "response/native_gmres.hpp"

namespace vibeqc::response {

inline GmresPlan prepare_response(const LinearResponseProblem& problem,
                                  const GmresOptions& options) {
  return prepare_gmres(problem.dimension(), options);
}

inline GmresResult solve_response(const GmresPlan& plan, const LinearResponseProblem& problem,
                                  std::span<const double> rhs,
                                  std::span<const double> initial_guess = {},
                                  std::span<const double> diagonal_preconditioner = {}) {
  if (plan.dimension != problem.dimension())
    throw std::invalid_argument("response plan and problem dimensions do not match");
  return solve_gmres(plan, problem.apply(), rhs, initial_guess, diagonal_preconditioner);
}

}  // namespace vibeqc::response

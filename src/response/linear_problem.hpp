#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>

namespace vibeqc::response {

using LinearOperator = std::function<void(std::span<const double> input, std::span<double> output)>;

/** Method-neutral matrix-free linear-response problem.
 *
 * The problem owns only the response-space dimension and operator action.
 * Right-hand sides, guesses, preconditioners and Krylov policy remain solve
 * inputs so one prepared physical operator can serve multiple perturbations.
 */
class LinearResponseProblem {
 public:
  LinearResponseProblem(std::size_t dimension, LinearOperator apply)
      : dimension_(dimension), apply_(std::move(apply)) {
    if (!dimension_) throw std::invalid_argument("response problem dimension must be positive");
    if (!apply_) throw std::invalid_argument("response problem operator callback is empty");
  }

  [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] const LinearOperator& apply() const noexcept { return apply_; }

 private:
  std::size_t dimension_{};
  LinearOperator apply_;
};

}  // namespace vibeqc::response

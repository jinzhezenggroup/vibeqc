#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>

namespace vibeqc::response {

using LinearOperator = std::function<void(std::span<const double> input, std::span<double> output)>;

/** Symmetry of one logical response operator in its declared response-vector metric.
 *
 * `General` is deliberately the fail-closed default: matching dimensions do not
 * authorize callers to reuse the primal action as a transpose/JVP-VJP partner.
 * `Symmetric` certifies that the primal action is its Euclidean transpose in the
 * problem's declared vector representation. Non-Euclidean or nonsymmetric
 * problems must provide an explicit transpose action when a consumer needs one.
 */
enum class LinearResponseSymmetry { General, Symmetric };

/** Method-neutral matrix-free linear-response problem.
 *
 * The problem owns only the response-space dimension and operator actions.
 * Right-hand sides, guesses, preconditioners and Krylov policy remain solve
 * inputs so one prepared physical operator can serve multiple perturbations.
 * Symmetry is explicit: a future nonsymmetric CC Jacobian cannot inherit the
 * primal action as its transpose merely because its vector shape matches.
 */
class LinearResponseProblem {
 public:
  LinearResponseProblem(std::size_t dimension, LinearOperator apply,
                        LinearResponseSymmetry symmetry = LinearResponseSymmetry::General,
                        LinearOperator apply_transpose = {})
      : dimension_(dimension),
        apply_(std::move(apply)),
        symmetry_(symmetry),
        apply_transpose_(std::move(apply_transpose)) {
    if (!dimension_) throw std::invalid_argument("response problem dimension must be positive");
    if (!apply_) throw std::invalid_argument("response problem operator callback is empty");
    if (symmetry_ == LinearResponseSymmetry::Symmetric && apply_transpose_)
      throw std::invalid_argument(
          "symmetric response problem cannot provide a distinct transpose operator");
  }

  [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
  [[nodiscard]] const LinearOperator& apply() const noexcept { return apply_; }
  [[nodiscard]] LinearResponseSymmetry symmetry() const noexcept { return symmetry_; }
  [[nodiscard]] bool has_transpose() const noexcept {
    return symmetry_ == LinearResponseSymmetry::Symmetric || static_cast<bool>(apply_transpose_);
  }
  [[nodiscard]] const LinearOperator& apply_transpose() const {
    if (apply_transpose_) return apply_transpose_;
    if (symmetry_ == LinearResponseSymmetry::Symmetric) return apply_;
    throw std::logic_error(
        "general response problem has no transpose operator; method adapter must provide one");
  }

 private:
  std::size_t dimension_{};
  LinearOperator apply_;
  LinearResponseSymmetry symmetry_{LinearResponseSymmetry::General};
  LinearOperator apply_transpose_;
};

}  // namespace vibeqc::response

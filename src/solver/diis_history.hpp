#ifndef VIBEQC_SOLVER_DIIS_HISTORY_HPP
#define VIBEQC_SOLVER_DIIS_HISTORY_HPP

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vibeqc::solver::detail {

/** Method-neutral bounded host DIIS history storage.
 *
 * The owner knows only flattened FP64 vectors/errors and chronological
 * retention. Gram construction, coefficient policy, scientific residual
 * semantics and extrapolation remain backend/caller owned.
 */
class DiisHistory {
 public:
  explicit DiisHistory(std::size_t capacity, std::size_t elements = 0)
      : capacity_(capacity), elements_(elements) {}

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t size() const noexcept { return vectors_.size(); }
  [[nodiscard]] std::size_t elements() const noexcept { return elements_; }

  [[nodiscard]] const std::vector<std::vector<double>>& vectors() const noexcept {
    return vectors_;
  }
  [[nodiscard]] const std::vector<std::vector<double>>& errors() const noexcept {
    return errors_;
  }

  void validate(const std::vector<double>& vector, const std::vector<double>& error) const {
    const std::size_t expected = elements_ ? elements_ : vector.size();
    if (vector.size() != expected || error.size() != expected)
      throw std::invalid_argument("DIIS vector/error dimensions do not match the state size");
  }

  void push(const std::vector<double>& vector, std::vector<double> error) {
    validate(vector, error);
    if (!elements_) elements_ = vector.size();
    vectors_.push_back(vector);
    errors_.push_back(std::move(error));
    if (vectors_.size() > capacity_) retire_oldest();
  }

  void retire_oldest() {
    if (vectors_.empty()) return;
    vectors_.erase(vectors_.begin());
    errors_.erase(errors_.begin());
  }

  void clear() {
    vectors_.clear();
    errors_.clear();
  }

  /** Retained numeric vector capacity only; object metadata is excluded. */
  [[nodiscard]] std::size_t numeric_capacity_bytes() const noexcept {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    std::size_t bytes = 0;
    const auto add = [&](const std::vector<double>& values) {
      const std::size_t current =
          values.capacity() > maximum / sizeof(double) ? maximum : values.capacity() * sizeof(double);
      bytes = current > maximum - bytes ? maximum : bytes + current;
    };
    for (const auto& value : vectors_) add(value);
    for (const auto& value : errors_) add(value);
    return bytes;
  }

 private:
  std::size_t capacity_{};
  std::size_t elements_{};
  std::vector<std::vector<double>> vectors_;
  std::vector<std::vector<double>> errors_;
};

}  // namespace vibeqc::solver::detail

#endif

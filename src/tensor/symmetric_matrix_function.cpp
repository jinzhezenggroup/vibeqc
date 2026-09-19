#include "tensor/symmetric_matrix_function.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc::tensor {
namespace {
std::size_t index(std::size_t row, std::size_t column, std::size_t n) { return row * n + column; }

double function_value(double value, bool retained, SymmetricMatrixFunction function) {
  if (!retained) return 0.0;
  if (!(value > 0.0)) throw std::invalid_argument("retained spectral value must be positive");
  return function == SymmetricMatrixFunction::pseudoinverse ? 1.0 / value : 1.0 / std::sqrt(value);
}

double divided_difference(double left, double right, bool keep_left, bool keep_right,
                          SymmetricMatrixFunction function, double resolution) {
  if (!keep_left && !keep_right) return 0.0;
  if (keep_left && keep_right) {
    if (!(left > 0.0) || !(right > 0.0))
      throw std::invalid_argument("retained spectral value must be positive");
    if (function == SymmetricMatrixFunction::pseudoinverse) return -1.0 / left / right;
    const double sl = std::sqrt(left), sr = std::sqrt(right);
    return -1.0 / sl / sr / (sl + sr);
  }
  const double gap = left - right;
  if (std::abs(gap) <= resolution)
    throw std::runtime_error("retained/discarded spectral subspaces are unresolved");
  return (function_value(left, keep_left, function) - function_value(right, keep_right, function)) /
         gap;
}
}  // namespace

std::vector<double> symmetric_matrix_function_vjp(std::span<const double> eigenvalues,
                                                  std::span<const double> eigenvectors,
                                                  std::span<const std::uint8_t> retained,
                                                  std::span<const double> response,
                                                  SymmetricMatrixFunction function,
                                                  double resolution) {
  const auto n = eigenvalues.size();
  if (!n || n > std::numeric_limits<std::size_t>::max() / n || eigenvectors.size() != n * n ||
      response.size() != n * n || retained.size() != n || !std::isfinite(resolution) ||
      resolution < 0.0)
    throw std::invalid_argument("invalid symmetric matrix-function response contract");
  if (!std::all_of(eigenvalues.begin(), eigenvalues.end(),
                   [](double x) { return std::isfinite(x); }) ||
      !std::all_of(eigenvectors.begin(), eigenvectors.end(),
                   [](double x) { return std::isfinite(x); }) ||
      !std::all_of(response.begin(), response.end(), [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("symmetric matrix-function response must be finite");
  if (std::none_of(retained.begin(), retained.end(), [](std::uint8_t x) { return x != 0; }))
    throw std::invalid_argument("symmetric matrix-function retains no spectral subspace");

  const auto elements = n * n;
  std::vector<double> symmetric(elements), temp(elements, 0.0), transformed(elements, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      symmetric[index(i, j, n)] = 0.5 * response[index(i, j, n)] + 0.5 * response[index(j, i, n)];

  // temp = sym(E) Q; transformed = Q^T temp.
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k)
        temp[index(i, j, n)] += symmetric[index(i, k, n)] * eigenvectors[index(k, j, n)];
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < n; ++k)
        transformed[index(i, j, n)] += eigenvectors[index(k, i, n)] * temp[index(k, j, n)];
      transformed[index(i, j, n)] *= divided_difference(
          eigenvalues[i], eigenvalues[j], retained[i] != 0, retained[j] != 0, function, resolution);
    }

  std::fill(temp.begin(), temp.end(), 0.0);
  std::vector<double> result(elements, 0.0);
  // result = Q transformed Q^T.
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k)
        temp[index(i, j, n)] += eigenvectors[index(i, k, n)] * transformed[index(k, j, n)];
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k)
        result[index(i, j, n)] += temp[index(i, k, n)] * eigenvectors[index(j, k, n)];
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j)
      result[index(i, j, n)] = result[index(j, i, n)] =
          0.5 * result[index(i, j, n)] + 0.5 * result[index(j, i, n)];
  if (!std::all_of(result.begin(), result.end(), [](double x) { return std::isfinite(x); }))
    throw std::runtime_error("symmetric matrix-function response is non-finite");
  return result;
}
}  // namespace vibeqc::tensor

#include "integrals/density_fitting_metric.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "tensor/cpu_linalg.hpp"

namespace vibeqc::integrals {
namespace {
std::size_t index(std::size_t row, std::size_t column, std::size_t n) { return row * n + column; }

using EigenResult = tensor::CpuSymmetricEigenResult;

EigenResult symmetric_eigen(std::vector<double> matrix, std::size_t n) {
  // Current endpoint evidence keeps the metric eigensolve on the deterministic
  // scalar schedule; dense response products below may still use the external provider.
  const tensor::CpuLinalgPlan plan{tensor::CpuLinalgProvider::scalar,
                                   tensor::CpuLinalgThreadOwnership::task_parallel, 1};
  return tensor::cpu_symmetric_eigen(std::move(matrix), n, plan);
}

bool checked_multiply(std::size_t first, std::size_t second, std::size_t& product) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  product = first * second;
  return true;
}

std::size_t checked_matrix_elements(std::size_t dimension, const char* description) {
  std::size_t elements = 0;
  if (dimension == 0 || !checked_multiply(dimension, dimension, elements)) {
    throw std::invalid_argument(description);
  }
  return elements;
}

void require_finite(const std::vector<double>& values, const char* description) {
  if (!std::all_of(values.begin(), values.end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(description);
  }
}

}  // namespace

std::vector<double> density_fitting_metric_response(
    const std::vector<double>& metric, const std::vector<double>& function_value_matrix,
    const std::vector<double>& response, std::size_t n, double relative_threshold,
    tensor::SymmetricMatrixFunction function) {
  const auto elements = checked_matrix_elements(n, "DF metric response dimension is invalid");
  if (metric.size() != elements || function_value_matrix.size() != elements ||
      response.size() != elements || !std::isfinite(relative_threshold) ||
      relative_threshold < 0.0 || relative_threshold >= 1.0)
    throw std::invalid_argument("DF metric response dimensions or threshold are inconsistent");
  require_finite(metric, "DF metric response requires a finite metric");
  require_finite(function_value_matrix, "DF metric response requires a finite matrix function");
  require_finite(response, "DF metric response requires finite weights");

  std::vector<double> symmetric(elements);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      symmetric[i * n + j] = 0.5 * (metric[i * n + j] + metric[j * n + i]);

  const auto eigen = symmetric_eigen(std::move(symmetric), n);
  const auto& q = eigen.vectors;
  const double largest = eigen.values.back();
  if (!(largest > 0.0)) throw std::runtime_error("DF metric has no positive response subspace");
  const double cutoff = relative_threshold * largest;
  const double resolution = 128 * std::numeric_limits<double>::epsilon() * largest;
  std::vector<std::uint8_t> retained(n);
  for (std::size_t i = 0; i < n; ++i) {
    double projected_value = 0.0;
    for (std::size_t row = 0; row < n; ++row)
      for (std::size_t column = 0; column < n; ++column)
        projected_value +=
            q[row * n + i] * function_value_matrix[row * n + column] * q[column * n + i];
    if (function == tensor::SymmetricMatrixFunction::pseudoinverse) {
      retained[i] = eigen.values[i] * projected_value > 0.5;
    } else {
      retained[i] = eigen.values[i] > 0.0 && std::sqrt(eigen.values[i]) * projected_value > 0.5;
    }
    if (relative_threshold > 0.0) {
      if (std::abs(eigen.values[i] - cutoff) <= resolution)
        throw std::runtime_error("DF metric rank crossing: eigenvalue is unresolved at the cutoff");
      if (static_cast<bool>(retained[i]) != (eigen.values[i] > cutoff))
        throw std::invalid_argument(
            "DF metric function active subspace differs from its threshold");
    }
  }
  if (std::none_of(retained.begin(), retained.end(), [](std::uint8_t keep) { return keep != 0; }))
    throw std::invalid_argument("DF metric function retains no positive subspace");

  return tensor::symmetric_matrix_function_vjp(eigen.values, q, retained, response, function,
                                               resolution);
}

DensityFittingMetricFactor factor_density_fitting_metric(const std::vector<double>& metric,
                                                         std::size_t dimension,
                                                         double relative_threshold) {
  std::size_t metric_elements = 0;
  if (dimension == 0 || !checked_multiply(dimension, dimension, metric_elements) ||
      metric.size() != metric_elements) {
    throw std::invalid_argument("metric dimensions are inconsistent");
  }
  if (!(relative_threshold > 0.0) || !(relative_threshold < 1.0)) {
    throw std::invalid_argument("metric relative threshold must lie strictly between zero and one");
  }
  std::vector<double> symmetric(metric.size());
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t column = 0; column < dimension; ++column) {
      const double value =
          0.5 * (metric[index(row, column, dimension)] + metric[index(column, row, dimension)]);
      if (!std::isfinite(value)) {
        throw std::invalid_argument("metric entries must be finite");
      }
      symmetric[index(row, column, dimension)] = value;
    }
  }
  const EigenResult eigen = symmetric_eigen(std::move(symmetric), dimension);
  const double largest = eigen.values.back();
  if (!(largest > 0.0) || !std::isfinite(largest)) {
    throw std::runtime_error("Coulomb metric has no positive eigenspace");
  }
  DensityFittingMetricFactor result;
  result.dimension = dimension;
  result.absolute_threshold = relative_threshold * largest;
  result.inverse_square_root.assign(metric_elements, 0.0);
  double smallest_retained = largest;
  for (std::size_t item = 0; item < dimension; ++item) {
    const double value = eigen.values[item];
    if (value <= result.absolute_threshold) continue;
    ++result.effective_rank;
    smallest_retained = std::min(smallest_retained, value);
    const double scale = 1.0 / std::sqrt(value);
    for (std::size_t row = 0; row < dimension; ++row) {
      for (std::size_t column = 0; column < dimension; ++column) {
        result.inverse_square_root[index(row, column, dimension)] +=
            eigen.vectors[index(row, item, dimension)] * scale *
            eigen.vectors[index(column, item, dimension)];
      }
    }
  }
  if (result.effective_rank == 0) {
    throw std::runtime_error("Coulomb metric threshold removed every auxiliary direction");
  }
  result.condition_number = largest / smallest_retained;
  return result;
}

}  // namespace vibeqc::integrals

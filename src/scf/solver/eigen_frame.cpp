#include "scf/solver/eigen_frame.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "scf/reference/linalg.hpp"

namespace vibeqc::scf::solver {
bool validate_eigen_frame(const std::vector<double>& matrix, const std::vector<double>* overlap,
                          const std::vector<double>& values,
                          const std::vector<double>& coefficients, std::size_t n,
                          EigenFrameDiagnostic& diagnostic, std::string& detail) {
  const auto finite = [](const auto& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (diagnostic.solver_info != 0 || n == 0 || n > std::numeric_limits<std::size_t>::max() / n ||
      matrix.size() != n * n || coefficients.size() != n * n || values.size() != n ||
      (overlap && overlap->size() != n * n) || !finite(matrix) || !finite(coefficients) ||
      !finite(values) || (overlap && !finite(*overlap)) ||
      !std::is_sorted(values.begin(), values.end())) {
    detail = "DF eigensolver returned invalid info, shape, finite values or eigenvalue order";
    return false;
  }
  // These host matrix checks share elementary products with the oracle,
  // never its eigensolver. They compare against the original F/S inputs.
  const auto fc = reference::multiply(matrix, coefficients, n);
  const auto sc = overlap ? reference::multiply(*overlap, coefficients, n) : coefficients;
  const auto gram = reference::multiply(reference::transpose(coefficients, n), sc, n);
  // Finite inputs may still overflow a product. Do not let NaNs disappear
  // through std::max or an infinite scale hide an invalid residual.
  if (!finite(fc) || !finite(sc) || !finite(gram)) {
    detail = "DF eigenframe validation products are nonfinite";
    return false;
  }
  double residual_norm = 0, matrix_norm = 0, coefficient_norm = 0, rhs_norm = 0;
  diagnostic.maximum_eigen_residual = diagnostic.maximum_metric_error = 0;
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      const auto k = row * n + column;
      const double rhs = sc[k] * values[column];
      const double residual = fc[k] - rhs;
      residual_norm = std::hypot(residual_norm, residual);
      matrix_norm = std::hypot(matrix_norm, matrix[k]);
      coefficient_norm = std::hypot(coefficient_norm, coefficients[k]);
      rhs_norm = std::hypot(rhs_norm, rhs);
      diagnostic.maximum_eigen_residual =
          std::max(diagnostic.maximum_eigen_residual, std::abs(residual));
      diagnostic.maximum_metric_error = std::max(diagnostic.maximum_metric_error,
                                                 std::abs(gram[k] - (row == column ? 1.0 : 0.0)));
    }
  }
  const double scale = matrix_norm * coefficient_norm + rhs_norm;
  diagnostic.scaled_eigen_residual = scale == 0 ? residual_norm : residual_norm / scale;
  if (!std::isfinite(scale) || !std::isfinite(residual_norm)) {
    detail = "DF eigenframe validation norm overflow";
    return false;
  }
  return accept_eigen_frame(diagnostic, detail);
}

bool accept_eigen_frame(const EigenFrameDiagnostic& diagnostic, std::string& detail) {
  if (diagnostic.solver_info != 0 || !std::isfinite(diagnostic.maximum_eigen_residual) ||
      !std::isfinite(diagnostic.maximum_metric_error) ||
      !std::isfinite(diagnostic.scaled_eigen_residual) || diagnostic.maximum_eigen_residual < 0 ||
      diagnostic.maximum_metric_error < 0 || diagnostic.scaled_eigen_residual < 0 ||
      diagnostic.maximum_eigen_residual > 1e-8 || diagnostic.maximum_metric_error > 1e-8 ||
      diagnostic.scaled_eigen_residual > 1e-12) {
    detail = "DF eigensystem failed physical eigen residual or metric orthogonality checks";
    return false;
  }
  return true;
}
}  // namespace vibeqc::scf::solver

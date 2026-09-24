#include "scf/reference/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "scf/reference/observation.hpp"

namespace vibeqc::scf::reference {

Matrix identity(std::size_t n) {
  Matrix result(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) result[index(i, i, n)] = 1.0;
  return result;
}

Matrix multiply(const Matrix& a, const Matrix& b, std::size_t n) {
  Matrix out(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      const double aik = a[index(i, k, n)];
      for (std::size_t j = 0; j < n; ++j) {
        out[index(i, j, n)] += aik * b[index(k, j, n)];
      }
    }
  }
  return out;
}

Matrix transpose(const Matrix& a, std::size_t n) {
  Matrix out(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      out[index(j, i, n)] = a[index(i, j, n)];
    }
  }
  return out;
}

// Keep the pivot-search loop's layout stable across unrelated link changes.
// On GCC/Zen 2 a 16-byte shift of this unchanged function slowed complete
// 192-AO DF endpoints by about 33%; 32-byte function alignment preserves the
// previous layout without changing pivot selection or floating-point order.
#if __has_cpp_attribute(gnu::aligned)
[[gnu::aligned(32)]]
#endif
#if __has_cpp_attribute(gnu::noinline)
// Keep the aligned numerical loop separate from the optional trace wrapper.
[[gnu::noinline]]
#endif
static EigenResult symmetric_eigen_impl(Matrix matrix, std::size_t n) {
  Matrix vectors = identity(n);
  const std::size_t max_sweeps = std::max<std::size_t>(50, 20 * n * n);
  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
    std::size_t p = 0;
    std::size_t q = 0;
    double largest = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const double candidate = std::abs(matrix[index(i, j, n)]);
        if (candidate > largest) {
          largest = candidate;
          p = i;
          q = j;
        }
      }
    }
    if (largest < 1.0e-14) break;

    const double app = matrix[index(p, p, n)];
    const double aqq = matrix[index(q, q, n)];
    const double apq = matrix[index(p, q, n)];
    const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(angle);
    const double s = std::sin(angle);

    for (std::size_t k = 0; k < n; ++k) {
      if (k == p || k == q) continue;
      const double mkp = matrix[index(k, p, n)];
      const double mkq = matrix[index(k, q, n)];
      matrix[index(k, p, n)] = matrix[index(p, k, n)] = c * mkp - s * mkq;
      matrix[index(k, q, n)] = matrix[index(q, k, n)] = s * mkp + c * mkq;
    }
    matrix[index(p, p, n)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    matrix[index(q, q, n)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    matrix[index(p, q, n)] = matrix[index(q, p, n)] = 0.0;

    for (std::size_t k = 0; k < n; ++k) {
      const double vkp = vectors[index(k, p, n)];
      const double vkq = vectors[index(k, q, n)];
      vectors[index(k, p, n)] = c * vkp - s * vkq;
      vectors[index(k, q, n)] = s * vkp + c * vkq;
    }
  }

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return matrix[index(a, a, n)] < matrix[index(b, b, n)];
  });
  EigenResult result;
  result.values.resize(n);
  result.vectors.resize(n * n);
  for (std::size_t column = 0; column < n; ++column) {
    const std::size_t source = order[column];
    result.values[column] = matrix[index(source, source, n)];
    for (std::size_t row = 0; row < n; ++row) {
      result.vectors[index(row, column, n)] = vectors[index(row, source, n)];
    }
  }
  return result;
}

EigenResult symmetric_eigen(Matrix matrix, std::size_t n) {
  // Instrument the call boundary so the independent aligned Jacobi loop is
  // unchanged. Only this leaf represents an actual reference diagonalization.
  scf::reference::observation::Scope trace("reference_eigensolve", n);
  return symmetric_eigen_impl(std::move(matrix), n);
}

Matrix symmetric_orthogonalizer(const Matrix& overlap, std::size_t n) {
  using namespace observation;
  Reason reason(active_reason == EigenReason::reference_export ||
                        active_reason == EigenReason::fallback
                    ? active_reason
                    : EigenReason::overlap);
  Scope trace("overlap_orthogonalization", n);
  const EigenResult eigen = symmetric_eigen(overlap, n);
  Matrix scaled = eigen.vectors;
  for (std::size_t column = 0; column < n; ++column) {
    if (eigen.values[column] < 1.0e-10) {
      throw std::runtime_error("overlap matrix is singular or severely linearly dependent");
    }
    const double factor = 1.0 / std::sqrt(eigen.values[column]);
    for (std::size_t row = 0; row < n; ++row) {
      scaled[index(row, column, n)] *= factor;
    }
  }
  return multiply(scaled, transpose(eigen.vectors, n), n);
}

EigenResult generalized_eigen(const Matrix& fock, const Matrix& orthogonalizer, std::size_t n) {
  scf::reference::observation::Scope trace("generalized_eigensolve", n);
  const Matrix transformed =
      multiply(transpose(orthogonalizer, n), multiply(fock, orthogonalizer, n), n);
  EigenResult result = symmetric_eigen(transformed, n);
  result.vectors = multiply(orthogonalizer, result.vectors, n);
  return result;
}

double dot(const Matrix& a, const Matrix& b) {
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
  return result;
}


}  // namespace vibeqc::scf::reference

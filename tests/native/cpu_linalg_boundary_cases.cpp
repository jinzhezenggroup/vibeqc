#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "tensor/cpu_linalg.hpp"

int main(int argc, char** argv) {
  using namespace vibeqc::tensor;
  if (argc != 2) return 2;
  const std::string mode(argv[1]);
  const CpuLinalgPlan plan{CpuLinalgProvider::scalar};
  const double nan = std::numeric_limits<double>::quiet_NaN();
  double a = 2.0, b = 3.0, c = nan;
  if (mode == "beta_zero") {
    cpu_gemm('N', 'N', 1, 1, 1, &a, &b, &c, 1.0, 0.0, plan);
    return c == 6.0 ? 0 : 1;
  }
  if (mode == "empty_inner") {
    cpu_gemm('N', 'N', 1, 1, 0, nullptr, nullptr, &c, 1.0, 0.0, plan);
    return c == 0.0 ? 0 : 1;
  }
  if (mode == "alpha_zero") {
    c = 4.0;
    cpu_gemm('N', 'N', 1, 1, 1, &nan, &nan, &c, 0.0, 2.0, plan);
    return c == 8.0 ? 0 : 1;
  }
  if (mode == "gemm_extent") {
    try {
      cpu_gemm('N', 'N', std::numeric_limits<std::size_t>::max() / 2 + 1, 2, 0, nullptr, nullptr,
               &c, 1.0, 0.0, plan);
    } catch (const std::length_error&) {
      return 0;
    }
    return 1;
  }
  if (mode == "syrk_alpha_zero") {
    double matrix[4]{nan, 9.0, nan, nan};
    cpu_syrk('L', 'N', 2, 1, &nan, matrix, 0.0, 0.0, plan);
    return matrix[0] == 0.0 && matrix[1] == 9.0 && matrix[2] == 0.0 && matrix[3] == 0.0 ? 0 : 1;
  }
  if (mode == "syrk_extent") {
    try {
      cpu_syrk('L', 'N', std::numeric_limits<std::size_t>::max() / 2 + 1, 2, nullptr, &c, 1.0, 0.0,
               plan);
    } catch (const std::length_error&) {
      return 0;
    }
    return 1;
  }
  if (mode == "cholesky_extent") {
    try {
      cpu_cholesky_lower(&c, std::size_t(1) << (sizeof(std::size_t) * 4), plan);
    } catch (const std::length_error&) {
      return 0;
    }
    return 1;
  }
  if (mode == "large_spectrum" || mode == "tiny_spectrum") {
    const double scale = mode == "large_spectrum" ? 1.0e308 : 1.0e-300;
    const auto result = cpu_symmetric_eigen({scale, 0.1 * scale, 0.1 * scale, -scale}, 2, plan);
    const double expected = std::hypot(1.0, 0.1);
    if (std::abs(result.values[0] / scale + expected) > 2e-13 ||
        std::abs(result.values[1] / scale - expected) > 2e-13)
      return 1;
    for (int col = 0; col < 2; ++col) {
      const double x = result.vectors[col], y = result.vectors[2 + col];
      const double value = result.values[col] / scale;
      if (std::abs(x + 0.1 * y - value * x) > 2e-13 || std::abs(0.1 * x - y - value * y) > 2e-13)
        return 1;
    }
    return 0;
  }
  return 2;
}

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#if defined(__CUDACC__)
#define VIBEQC_GRID_HD __host__ __device__
#else
#define VIBEQC_GRID_HD
#endif

namespace vibeqc_grid_adjoint {
VIBEQC_GRID_HD inline double portable_abs(double value) { return value < 0.0 ? -value : value; }

VIBEQC_GRID_HD inline double portable_exp(double value) {
#if defined(__CUDA_ARCH__)
  return ::exp(value);
#else
  return std::exp(value);
#endif
}

// Shared two-pass Becke reverse composition. Runtime owners supply O(natom)
// scratch per worker and transactional reduction storage. Local mathematical
// partials come exclusively from the compiler's grid_response Graphs.
template <class Norm>
VIBEQC_GRID_HD std::array<double, 4> distance(const double* a, const double* b, Norm norm,
                                              bool& valid) {
  double delta[3], scale = 0;
  for (size_t k = 0; k < 3; ++k) {
    delta[k] = a[k] - b[k];
    scale = std::max(scale, portable_abs(delta[k]));
  }
  if (!(scale > 0) || !std::isfinite(scale)) {
    valid = false;
    return {};
  }
  auto result = norm(delta[0] / scale, delta[1] / scale, delta[2] / scale);
  result[0] *= scale;
  for (double v : result)
    if (!std::isfinite(v)) {
      valid = false;
      return {};
    }
  return result;
}

template <class Norm, class Ratio, class Log, class Pair>
VIBEQC_GRID_HD bool contract_point(const double* point, const double* centers, size_t na,
                                   size_t owner, double seed, double* gradient, double* logs,
                                   double* products, double* bar_product, double* bar_distance,
                                   size_t* zeros, std::array<double, 4>* distances, Norm norm,
                                   Ratio ratio, Log logarithm, Pair pair) {
  bool valid = true;
  for (size_t a = 0; a < na; ++a) distances[a] = distance(point, centers + 3 * a, norm, valid);
  if (!valid) return false;
  if (na == 1) return true;
  for (size_t a = 0; a < na; ++a) logs[a] = 0;
  for (size_t a = 0; a < na; ++a) zeros[a] = 0;
  for (size_t a = 0; a < na; ++a) bar_distance[a] = 0;
  auto factor = [&](size_t a, size_t b, double separation) {
    auto r = ratio(distances[a][0] - distances[b][0], separation);
    const bool clipped = portable_abs(r[0]) >= 1;
    auto f = pair(std::clamp(r[0], -1.0, 1.0));
    if (clipped || f[0] < 0 || f[0] > 1) f[1] = 0;
    f[0] = std::clamp(f[0], 0.0, 1.0);
    return f;
  };
  for (size_t a = 0; a < na; ++a)
    for (size_t b = 0; b < a; ++b) {
      const double separation = distance(centers + 3 * a, centers + 3 * b, norm, valid)[0];
      const auto f = factor(a, b, separation);
      for (size_t side = 0; side < 2; ++side) {
        const size_t atom = side ? b : a;
        const double v = side ? 1 - f[0] : f[0];
        if (v > 0)
          logs[atom] += logarithm(v)[0];
        else
          ++zeros[atom];
      }
    }
  double maximum = -std::numeric_limits<double>::infinity();
  for (size_t a = 0; a < na; ++a)
    if (!zeros[a]) maximum = std::max(maximum, logs[a]);
  if (!std::isfinite(maximum)) return false;
  double total = 0;
  for (size_t a = 0; a < na; ++a) {
    products[a] = zeros[a] ? 0 : portable_exp(logs[a] - maximum);
    total += products[a];
  }
  // The selected normalized-product objective uses the SAME ratio
  // graph. A frozen log scale cancels between numerator/denominator.
  auto objective = ratio(products[owner], total);
  for (size_t a = 0; a < na; ++a)
    bar_product[a] = seed * (objective[2] + (a == size_t(owner) ? objective[1] : 0));
  for (size_t a = 0; a < na; ++a)
    for (size_t b = 0; b < a; ++b) {
      const auto separation = distance(centers + 3 * a, centers + 3 * b, norm, valid);
      const auto r = ratio(distances[a][0] - distances[b][0], separation[0]);
      const auto f = factor(a, b, separation[0]);
      // Saturated branches have exactly zero pullback. Skip before
      // exponentiation to avoid the undefined numerical form inf*0.
      if (f[1] == 0) continue;
      double bar_mu = 0;
      for (size_t side = 0; side < 2; ++side) {
        const size_t atom = side ? b : a;
        const double v = side ? 1 - f[0] : f[0];
        double derivative = 0;
        if (!zeros[atom]) derivative = products[atom] * logarithm(v)[1];
        // ONE exact zero leaves the product of all other factors;
        // two zeros kill the first derivative. Never divide by zero.
        else if (zeros[atom] == 1 && v == 0)
          derivative = portable_exp(logs[atom] - maximum);
        bar_mu += (side ? -1 : 1) * bar_product[atom] * derivative * f[1];
      }
      bar_distance[a] += bar_mu * r[1];
      bar_distance[b] -= bar_mu * r[1];
      for (size_t k = 0; k < 3; ++k) {
        const double value = bar_mu * r[2] * separation[k + 1];
        gradient[3 * a + k] += value;
        gradient[3 * b + k] -= value;
      }
    }
  for (size_t a = 0; a < na; ++a)
    for (size_t k = 0; k < 3; ++k) {
      const double value = bar_distance[a] * distances[a][k + 1];
      gradient[3 * a + k] -= value;
      gradient[3 * owner + k] += value;
    }
  return valid;
}
}  // namespace vibeqc_grid_adjoint
#undef VIBEQC_GRID_HD

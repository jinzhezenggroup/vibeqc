#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace vibeqc_grid_cpu {
// Local primal/partials are generated from the existing Graph programs. This
// template owns only traversal, branch policy and reverse composition. A frozen
// scale cancels in the norm pullback by homogeneity, including translations.
template <class Norm>
std::array<double, 4> distance(const double* a, const double* b, Norm norm) {
  double delta[3], scale = 0;
  for (size_t k = 0; k < 3; ++k) {
    delta[k] = a[k] - b[k];
    scale = std::max(scale, std::abs(delta[k]));
  }
  if (!(scale > 0) || !std::isfinite(scale)) throw 1;
  auto result = norm(delta[0] / scale, delta[1] / scale, delta[2] / scale);
  result[0] *= scale;
  for (double v : result)
    if (!std::isfinite(v)) throw 1;
  return result;
}

template <class Norm, class Ratio, class Log, class Pair>
int contract(const double* points, size_t np, const double* centers, size_t na,
             const int64_t* owners, const double* seeds, double* output, size_t output_count,
             size_t budget, size_t max_pairs, double tolerance, Norm norm, Ratio ratio,
             Log logarithm, Pair pair) noexcept {
  // Admission precedes pointer dereference and all shape-dependent storage.
  constexpr size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
  if (!na || na > limit / 30 || np > (limit - 30 * na) / 10 || output_count != 3 * na ||
      budget / 8 < 10 * np + 30 * na || !centers || !output ||
      (np && (!points || !owners || !seeds)) || !std::isfinite(tolerance) || tolerance < 0)
    return -1;
  // Count both pair passes and center validation without overflowing size_t.
  if (na - 1 > std::numeric_limits<size_t>::max() / na) return -1;
  const size_t pairs = na * (na - 1) / 2;
  if (pairs && (pairs > max_pairs || np > (max_pairs / pairs - 1) / 2)) return -1;
  try {
    for (size_t i = 0; i < 3 * na; ++i)
      if (!std::isfinite(centers[i])) return -2;
    for (size_t i = 0; i < np; ++i) {
      if (owners[i] < 0 || size_t(owners[i]) >= na || !std::isfinite(seeds[i])) return -2;
      for (size_t k = 0; k < 3; ++k)
        if (!std::isfinite(points[3 * i + k])) return -2;
    }
    // Reject nonsmooth geometry even for an empty point tile or zero seed.
    for (size_t a = 0; a < na; ++a)
      for (size_t b = 0; b < a; ++b)
        if (distance(centers + 3 * a, centers + 3 * b, norm)[0] <= tolerance) return -2;
    std::vector<double> gradient(3 * na, 0), logs(na), products(na), bar_product(na),
        bar_distance(na);
    std::vector<size_t> zeros(na);
    std::vector<std::array<double, 4>> distances(na);
    for (size_t p = 0; p < np; ++p) {
      for (size_t a = 0; a < na; ++a)
        distances[a] = distance(points + 3 * p, centers + 3 * a, norm);
      if (na == 1) continue;
      std::fill(logs.begin(), logs.end(), 0);
      std::fill(zeros.begin(), zeros.end(), 0);
      std::fill(bar_distance.begin(), bar_distance.end(), 0);
      auto factor = [&](size_t a, size_t b, double separation) {
        auto r = ratio(distances[a][0] - distances[b][0], separation);
        const bool clipped = std::abs(r[0]) >= 1;
        auto f = pair(std::clamp(r[0], -1.0, 1.0));
        if (clipped || f[0] < 0 || f[0] > 1) f[1] = 0;
        f[0] = std::clamp(f[0], 0.0, 1.0);
        return f;
      };
      for (size_t a = 0; a < na; ++a)
        for (size_t b = 0; b < a; ++b) {
          const double separation = distance(centers + 3 * a, centers + 3 * b, norm)[0];
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
      if (!std::isfinite(maximum)) return -2;
      double total = 0;
      for (size_t a = 0; a < na; ++a) {
        products[a] = zeros[a] ? 0 : std::exp(logs[a] - maximum);
        total += products[a];
      }
      // The selected normalized-product objective uses the SAME ratio
      // graph. A frozen log scale cancels between numerator/denominator.
      auto objective = ratio(products[owners[p]], total);
      for (size_t a = 0; a < na; ++a)
        bar_product[a] = seeds[p] * (objective[2] + (a == size_t(owners[p]) ? objective[1] : 0));
      for (size_t a = 0; a < na; ++a)
        for (size_t b = 0; b < a; ++b) {
          const auto separation = distance(centers + 3 * a, centers + 3 * b, norm);
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
              derivative = std::exp(logs[atom] - maximum);
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
          gradient[3 * owners[p] + k] += value;
        }
    }
    for (double v : gradient)
      if (!std::isfinite(v)) return -2;
    std::copy(gradient.begin(), gradient.end(), output);
    return 0;
  } catch (...) {
    return -3;
  }
}
}  // namespace vibeqc_grid_cpu

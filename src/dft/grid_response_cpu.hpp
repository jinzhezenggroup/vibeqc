#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "grid_response_adjoint.hpp"

namespace vibeqc_grid_cpu {
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
    bool valid = true;
    // Reject nonsmooth geometry even for an empty point tile or zero seed.
    for (size_t a = 0; a < na; ++a)
      for (size_t b = 0; b < a; ++b)
        if (vibeqc_grid_adjoint::distance(centers + 3 * a, centers + 3 * b, norm, valid)[0] <=
                tolerance ||
            !valid)
          return -2;
    std::vector<double> gradient(3 * na, 0), logs(na), products(na), bar_product(na),
        bar_distance(na);
    std::vector<size_t> zeros(na);
    std::vector<std::array<double, 4>> distances(na);
    for (size_t p = 0; p < np; ++p)
      if (!vibeqc_grid_adjoint::contract_point(
              points + 3 * p, centers, na, owners[p], seeds[p], gradient.data(), logs.data(),
              products.data(), bar_product.data(), bar_distance.data(), zeros.data(),
              distances.data(), norm, ratio, logarithm, pair))
        return -2;
    for (double v : gradient)
      if (!std::isfinite(v)) return -2;
    std::copy(gradient.begin(), gradient.end(), output);
    return 0;
  } catch (...) {
    return -3;
  }
}
}  // namespace vibeqc_grid_cpu

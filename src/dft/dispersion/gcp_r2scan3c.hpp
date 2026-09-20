// SPDX-License-Identifier: LGPL-3.0-or-later
// r2SCAN-3c gCP qualification/runtime kernel.
// Equations and parameters audited against simple-dftd3 commit
// 41d5a07b98ce15e97bec7a1815869725f6c7b0c2 (LGPL-3.0-or-later).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dft/dispersion/gcp_r2scan3c_data.hpp"
#include "generated_method_parameters.hpp"

namespace vibeqc::dft::dispersion {

enum class GCPStatus : std::int32_t {
  success = 0,
  invalid_argument = 1,
  unsupported_element = 2,
  coincident_atoms = 3,
  numerical_failure = 4,
};

struct GCPParameters {
  double sigma = ::vibeqc::generated::method_parameters::r2scan3cGcp().sigma;
  double alpha = ::vibeqc::generated::method_parameters::r2scan3cGcp().alpha;
  double beta = ::vibeqc::generated::method_parameters::r2scan3cGcp().beta;
  double damping_scale = ::vibeqc::generated::method_parameters::r2scan3cGcp().damping_scale;
  double damping_exponent = ::vibeqc::generated::method_parameters::r2scan3cGcp().damping_exponent;
};

inline constexpr GCPParameters r2scan3c_gcp_parameters() { return {}; }

namespace gcp_detail {
inline constexpr double kAngstromToBohr = 1.8897261254578281;

inline std::array<double, 9> aaux(double x) {
  std::array<double, 9> a{};
  const double ex = std::exp(-x), rx = 1.0 / x;
  a[0] = ex * rx;
  for (int k = 1; k < 9; ++k) {
    a[k] = (static_cast<double>(k) * a[k - 1] + ex) * rx;
  }
  return a;
}

inline std::array<double, 9> bint(double x) {
  std::array<double, 9> b{};
  if (std::abs(x) < 1.0e-6) {
    for (int k = 0; k < 9; k += 2) b[k] = 2.0 / static_cast<double>(k + 1);
    return b;
  }
  std::array<double, 13> pw{};
  pw[0] = 1.0;
  for (int i = 1; i < 13; ++i) {
    pw[i] = pw[i - 1] * (-x) / static_cast<double>(i);
  }
  for (int k = 0; k < 9; ++k) {
    double acc = 0.0;
    for (int i = k % 2; i < 13; i += 2) {
      acc += pw[i] / static_cast<double>(k + i + 1);
    }
    b[k] = 2.0 * acc;
  }
  return b;
}

inline std::array<double, 9> baux(double x) {
  std::array<double, 9> b{};
  const double ep = std::exp(x), em = std::exp(-x), rx = 1.0 / x;
  for (int k = 0; k < 9; ++k) {
    double term = rx;
    double sgn = (k % 2 == 0) ? 1.0 : -1.0;
    double sp = sgn * term, sm = term;
    for (int j = 1; j <= k; ++j) {
      term *= static_cast<double>(k - j + 1) * rx;
      sgn = -sgn;
      sp += sgn * term;
      sm += term;
    }
    b[k] = ep * sp - em * sm;
  }
  return b;
}

inline bool overlap(double r, int shell_a, int shell_b, double za, double zb, double* s,
                    double* ds) {
  const bool same = std::abs(za - zb) < 0.1;
  const int key = shell_a * shell_b;
  int m = 0, terms = 0;
  std::array<double, 6> wt{};
  std::array<int, 6> pa{}, qb{};
  double cnorm = 0.0;
  if (key == 1) {
    m = 3;
    terms = 2;
    wt = {1, -1, 0, 0, 0, 0};
    pa = {2, 0, 0, 0, 0, 0};
    qb = {0, 2, 0, 0, 0, 0};
    cnorm = 0.25 * std::sqrt(std::pow(za * zb, 3));
  } else if (key == 2) {
    if (shell_a >= shell_b) std::swap(za, zb);
    m = 4;
    terms = 4;
    wt = {1, -1, 1, -1, 0, 0};
    pa = {3, 0, 2, 1, 0, 0};
    qb = {0, 3, 1, 2, 0, 0};
    cnorm = std::sqrt(1.0 / 3.0) * std::sqrt(std::pow(za, 3) * std::pow(zb, 5)) * 0.125;
  } else if (key == 3) {
    if (shell_a >= shell_b) std::swap(za, zb);
    m = 5;
    terms = 4;
    wt = {1, -1, 2, -2, 0, 0};
    pa = {4, 0, 3, 1, 0, 0};
    qb = {0, 4, 1, 3, 0, 0};
    cnorm = std::sqrt(std::pow(za, 3) * std::pow(zb, 7) / 7.5) * 0.0625 / std::sqrt(3.0);
  } else if (key == 4) {
    m = 5;
    terms = 3;
    wt = {1, 1, -2, 0, 0, 0};
    pa = {4, 0, 2, 0, 0, 0};
    qb = {0, 4, 2, 0, 0, 0};
    cnorm = std::sqrt(std::pow(za * zb, 5)) * 0.0625 / 3.0;
  } else if (key == 6) {
    if (shell_a >= shell_b) std::swap(za, zb);
    m = 6;
    terms = 6;
    wt = {1, 1, -2, -2, 1, 1};
    pa = {5, 4, 3, 2, 1, 0};
    qb = {0, 1, 2, 3, 4, 5};
    cnorm = std::sqrt(std::pow(za, 5) * std::pow(zb, 7) / 7.5) * 0.03125 / 3.0;
  } else if (key == 9) {
    m = 7;
    terms = 4;
    wt = {1, -3, 3, -1, 0, 0};
    pa = {6, 4, 2, 0, 0, 0};
    qb = {0, 2, 4, 6, 0, 0};
    cnorm = std::sqrt(std::pow(za * zb, 7)) / 1440.0;
  } else {
    return false;
  }
  const double ha = 0.5 * (za + zb), hb = 0.5 * (zb - za);
  const auto av = aaux(ha * r);
  const auto bv = same ? bint(hb * r) : baux(hb * r);
  double f0 = 0.0, f1 = 0.0;
  for (int t = 0; t < terms; ++t) {
    f0 += wt[t] * av[pa[t]] * bv[qb[t]];
    f1 -= wt[t] * (ha * av[pa[t] + 1] * bv[qb[t]] + hb * av[pa[t]] * bv[qb[t] + 1]);
  }
  *s = cnorm * std::pow(r, m) * f0;
  *ds = cnorm * (static_cast<double>(m) * std::pow(r, m - 1) * f0 + std::pow(r, m) * f1);
  return std::isfinite(*s) && std::isfinite(*ds) && *s > 0.0;
}
}  // namespace gcp_detail

inline GCPStatus evaluate_r2scan3c_gcp(int n, const std::int32_t* z, const double* xyz,
                                       const GCPParameters& p, double* energy, double* gradient) {
  if (n < 0 || (n > 0 && (!z || !xyz || !gradient)) || !energy || gradient == xyz) {
    return GCPStatus::invalid_argument;
  }
  if (!(std::isfinite(p.sigma) && p.sigma > 0.0 && std::isfinite(p.alpha) && p.alpha > 0.0 &&
        std::isfinite(p.beta) && p.beta > 0.0 && std::isfinite(p.damping_scale) &&
        p.damping_scale > 0.0 && std::isfinite(p.damping_exponent) && p.damping_exponent > 0.0)) {
    return GCPStatus::invalid_argument;
  }
  *energy = 0.0;
  for (int k = 0; k < 3 * n; ++k) gradient[k] = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!::vibeqc::generated::method_parameters::r2scan3cGcpSupportsAtomicNumber(z[i]))
      return GCPStatus::unsupported_element;
    for (int k = 0; k < 3; ++k) {
      if (!std::isfinite(xyz[3 * i + k])) return GCPStatus::invalid_argument;
    }
  }
  for (int i = 0; i < n; ++i) {
    const auto& ei = gcp_data::kElements[static_cast<std::size_t>(z[i] - 1)];
    for (int j = 0; j < i; ++j) {
      const auto& ej = gcp_data::kElements[static_cast<std::size_t>(z[j] - 1)];
      std::array<double, 3> v{};
      double r2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        v[k] = xyz[3 * i + k] - xyz[3 * j + k];
        r2 += v[k] * v[k];
      }
      if (!(r2 > 0.0) || !std::isfinite(r2)) return GCPStatus::coincident_atoms;
      const double r = std::sqrt(r2);
      double s = 0.0, ds = 0.0;
      if (!gcp_detail::overlap(r, ei.shell, ej.shell, ei.slater, ej.slater, &s, &ds)) {
        return GCPStatus::numerical_failure;
      }
      const double bsse = std::exp(-p.alpha * std::pow(r, p.beta)) / std::sqrt(s);
      const int hi = std::max(z[i], z[j]), lo = std::min(z[i], z[j]);
      const std::size_t ridx = static_cast<std::size_t>(hi * (hi - 1) / 2 + lo - 1);
      const double r0 = gcp_data::kVdwAngstrom[ridx] * gcp_detail::kAngstromToBohr;
      const double x = r / r0;
      const double damp_power = p.damping_scale * std::pow(x, p.damping_exponent);
      const double damp = 1.0 - 1.0 / (1.0 + damp_power);
      const double ddamp = p.damping_scale * p.damping_exponent *
                           std::pow(x, p.damping_exponent - 1.0) / r0 /
                           std::pow(1.0 + damp_power, 2);
      const double xi = ei.xv >= 0.5 ? 1.0 / std::sqrt(ei.xv) : 0.0;
      const double xj = ej.xv >= 0.5 ? 1.0 / std::sqrt(ej.xv) : 0.0;
      const double pref = p.sigma * (ei.emiss * xj + ej.emiss * xi);
      const double pair = pref * bsse * damp;
      const double dbsse = bsse * (-p.alpha * p.beta * std::pow(r, p.beta - 1.0) - 0.5 * ds / s);
      const double dedr = pref * (dbsse * damp + bsse * ddamp);
      if (!(std::isfinite(pair) && std::isfinite(dedr))) {
        return GCPStatus::numerical_failure;
      }
      *energy += pair;
      for (int k = 0; k < 3; ++k) {
        const double g = dedr * v[k] / r;
        gradient[3 * i + k] += g;
        gradient[3 * j + k] -= g;
      }
    }
  }
  return std::isfinite(*energy) ? GCPStatus::success : GCPStatus::numerical_failure;
}

}  // namespace vibeqc::dft::dispersion

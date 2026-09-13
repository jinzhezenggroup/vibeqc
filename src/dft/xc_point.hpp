// Copyright (C) 2017 M.A.L. Marques
// Copyright (C) 2026 VibeQC contributors
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <cfloat>
#include <cmath>

// The CPU and ordinary-stream CUDA consumers use this same point contract.
// Parameters/composition follow xc/expressions.py and the vendored Libxc 7
// sources. See docs/xc_scf_domain.md for the algebra and boundary policy.
#if defined(__CUDACC__)
#define VIBEQC_XC_HD __host__ __device__
#else
#define VIBEQC_XC_HD
#endif

namespace vibeqc::dft::point {

inline constexpr const char* kDomain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";

/** First derivatives in (rho_a,rho_b,grad_a[3],grad_b[3]). Keeping Cartesian
 * gradient coefficients avoids an unrepresentable v_sigma times a zero
 * gradient in the vacuum tail. No tau or higher AO jets are requested. */
struct Value {
  double energy{};
  double rho[2]{};
  double gradient[2][3]{};
  bool valid{true};
};

namespace detail {
VIBEQC_XC_HD inline bool finite(double x) { return x >= -DBL_MAX && x <= DBL_MAX; }
/** Forward differentiation in locally scaled physical coordinates. The scale
 * is fixed during differentiation, so these are physical partial derivatives,
 * including at equal spin densities; it is not a density regularization. */
struct Jet {
  double v{};
  double d[8]{};
  VIBEQC_XC_HD Jet() {}
  VIBEQC_XC_HD Jet(double value) : v(value) {}
  VIBEQC_XC_HD static Jet variable(double value, unsigned index) {
    Jet out(value);
    out.d[index] = 1.0;
    return out;
  }
};
VIBEQC_XC_HD inline Jet operator+(const Jet& a, const Jet& b) {
  Jet out(a.v + b.v);
  for (unsigned i = 0; i < 8; ++i) out.d[i] = a.d[i] + b.d[i];
  return out;
}
VIBEQC_XC_HD inline Jet operator-(const Jet& a, const Jet& b) {
  Jet out(a.v - b.v);
  for (unsigned i = 0; i < 8; ++i) out.d[i] = a.d[i] - b.d[i];
  return out;
}
VIBEQC_XC_HD inline Jet operator-(const Jet& a) { return Jet(0.0) - a; }
VIBEQC_XC_HD inline Jet operator*(const Jet& a, const Jet& b) {
  Jet out(a.v * b.v);
  for (unsigned i = 0; i < 8; ++i) out.d[i] = a.d[i] * b.v + a.v * b.d[i];
  return out;
}
VIBEQC_XC_HD inline Jet operator/(const Jet& a, const Jet& b) {
  Jet out(a.v / b.v);
  // Do not square a small denominator: that spuriously underflows in tails.
  for (unsigned i = 0; i < 8; ++i) out.d[i] = (a.d[i] - out.v * b.d[i]) / b.v;
  return out;
}
VIBEQC_XC_HD inline Jet power(const Jet& a, double p) {
  Jet out(::pow(a.v, p));
  // All powers evaluated at zero here have p>1 and a zero first derivative.
  const double slope = a.v == 0.0 ? 0.0 : p * ::pow(a.v, p - 1.0);
  for (unsigned i = 0; i < 8; ++i) out.d[i] = slope * a.d[i];
  return out;
}
VIBEQC_XC_HD inline Jet log1p(const Jet& a) {
  Jet out(::log1p(a.v));
  for (unsigned i = 0; i < 8; ++i) out.d[i] = a.d[i] / (1.0 + a.v);
  return out;
}
VIBEQC_XC_HD inline Jet expm1(const Jet& a) {
  Jet out(::expm1(a.v));
  for (unsigned i = 0; i < 8; ++i) out.d[i] = ::exp(a.v) * a.d[i];
  return out;
}
VIBEQC_XC_HD inline Jet log1p_over_x(const Jet& a) {
  // Analytic continuation, including its derivative, when direct subtraction
  // in the quotient derivative would lose all significant digits.
  if (::fabs(a.v) < 1.0e-4)
    return 1.0 + a * (-0.5 + a * (1.0 / 3.0 + a * (-0.25 + a * (0.2 - a / 6.0))));
  return log1p(a) / a;
}

/** C2 extension only of the PBE spin interpolation u^(2/3). Energy and two
 * derivatives match the mathematical expression at u=1e-18; u=0 has its
 * exact value and a finite specified derivative. No rho/sigma clipping. */
VIBEQC_XC_HD inline Jet spin_two_thirds(const Jet& u) {
  constexpr double cutoff = 1.0e-18;
  if (u.v >= cutoff) return power(u, 2.0 / 3.0);
  const Jet t = u / cutoff;
  return 1.0e-12 * t * (14.0 / 9.0 + t * (-7.0 / 9.0 + t * (2.0 / 9.0)));
}

VIBEQC_XC_HD inline Jet pw_channel(const Jet& x, double a, double alpha, double b1, double b2,
                                   double b3, double b4) {
  constexpr double c = 0.6203504908994001;  // (3/(4*pi))^(1/3)
  const Jet x2 = x * x;
  const Jet q = b1 * ::sqrt(c) * x2 * x + b2 * c * x2 + b3 * ::pow(c, 1.5) * x + b4 * c * c;
  const Jet u = x2 * x2 / (2.0 * a * q);
  // This equals -2a(1+alpha*rs)log1p(1/(2a*aux)), rs=c/x^2.
  // Factoring log1p(u)=u*log1p_over_x(u) eliminates inverse-density powers.
  return -(x2 + alpha * c) * x2 / q * log1p_over_x(u);
}

VIBEQC_XC_HD inline Jet energy_per_scale(bool pbe, const Jet& a, const Jet& b, const Jet g[2][3],
                                         double scale) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double beta = 0.06672455060314922;
  constexpr double kappa = 0.804;
  const double gamma = (1.0 - ::log(2.0)) / (pi * pi);
  const double cx = 0.375 * ::pow(3.0 / pi, 1.0 / 3.0) * ::pow(4.0, 2.0 / 3.0);
  const double scale13 = ::cbrt(scale);
  const Jet n = a + b;
  const Jet x = ::pow(scale, 1.0 / 6.0) * power(n, 1.0 / 6.0);
  const Jet up = 2.0 * a / n, down = 2.0 * b / n, z = (a - b) / n;
  const Jet fz =
      (power(up, 4.0 / 3.0) + power(down, 4.0 / 3.0) - 2.0) / (::pow(2.0, 4.0 / 3.0) - 2.0);
  const Jet e0 =
      pw_channel(x, pbe ? 0.0310907 : 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294);
  const Jet e1 =
      pw_channel(x, pbe ? 0.01554535 : 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517);
  const Jet em =
      pw_channel(x, pbe ? 0.0168869 : 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671);
  const double fzz = pbe ? 1.709920934161365617563962776245 : 1.709921;
  Jet eps = e0 + power(z, 4.0) * fz * (e1 - e0 + em / fzz) - fz * em / fzz;
  Jet exchange;
  const Jet spin[2]{a, b};
  for (unsigned s = 0; s < 2; ++s) {
    // Empty spin exchange has zero energy and zero first derivative. Its
    // physical gradient is checked to be zero by the point entry below.
    if (spin[s].v == 0.0) continue;
    Jet enhancement(1.0);
    if (pbe) {
      Jet g2;
      for (unsigned k = 0; k < 3; ++k) g2 = g2 + g[s][k] * g[s][k];
      const Jet denominator = scale13 * scale13 * power(spin[s], 8.0 / 3.0);
      const double mu_x2s2 = beta * pi * pi / (12.0 * ::pow(6.0 * pi * pi, 2.0 / 3.0));
      enhancement =
          1.0 + kappa - kappa * kappa * denominator / (kappa * denominator + mu_x2s2 * g2);
    }
    exchange = exchange - cx * scale13 * power(spin[s], 4.0 / 3.0) * enhancement;
  }
  if (pbe) {
    const Jet phi = (spin_two_thirds(up) + spin_two_thirds(down)) / 2.0;
    const Jet phi3 = phi * phi * phi;
    Jet g2;
    for (unsigned k = 0; k < 3; ++k) {
      const Jet total = g[0][k] + g[1][k];
      g2 = g2 + total * total;
    }
    const Jet t2 = g2 / (16.0 * ::pow(2.0, 2.0 / 3.0) * 0.6203504908994001 * scale13 *
                         power(n, 7.0 / 3.0) * phi * phi);
    const Jet aa = beta / (gamma * expm1(-eps / (gamma * phi3)));
    const Jet v = 1.0 / (1.0 + aa * t2);
    // Combine eps_PW+H analytically, before floating-point evaluation:
    // G log1p(-(1-exp(eps_PW/G))/(1+u+u^2)), G=gamma*phi^3,
    // u=A*t2. v=1/(1+u) avoids u^2 overflow. This is essential for the
    // potential: differentiating cancellation between eps_PW and H creates
    // spurious tail gradient coefficients many orders above the true result.
    const Jet q = -expm1(eps / (gamma * phi3));
    eps = gamma * phi3 * log1p(-q * v * v / (1.0 - v + v * v));
  }
  return exchange + n * eps;
}
}  // namespace detail

/** Evaluate full-spin LDA_XC_PW or PBE energy and AO-potential coefficients.
 * Invalid inputs return valid=false on both CPU and CUDA (no device throw).
 * At the all-spin vacuum only the zero gradient is admissible. */
VIBEQC_XC_HD inline Value evaluate(bool pbe, const double rho[2], const double gradient[2][3]) {
  Value out;
  for (unsigned s = 0; s < 2; ++s) {
    if (!detail::finite(rho[s]) || rho[s] < 0.0) out.valid = false;
    for (unsigned k = 0; k < 3; ++k)
      if (!detail::finite(gradient[s][k]) || (rho[s] == 0.0 && gradient[s][k] != 0.0))
        out.valid = false;
  }
  const double scale = rho[0] + rho[1];
  if (!detail::finite(scale)) out.valid = false;
  if (!out.valid || scale == 0.0) return out;
  using detail::Jet;
  const Jet a = Jet::variable(rho[0] / scale, 0), b = Jet::variable(rho[1] / scale, 1);
  Jet g[2][3];
  for (unsigned s = 0; s < 2; ++s)
    for (unsigned k = 0; k < 3; ++k) g[s][k] = Jet::variable(gradient[s][k] / scale, 2 + 3 * s + k);
  const Jet energy = detail::energy_per_scale(pbe, a, b, g, scale);
  out.energy = scale * energy.v;
  out.valid = detail::finite(out.energy);
  for (unsigned s = 0; s < 2; ++s) {
    out.rho[s] = energy.d[s];
    out.valid = out.valid && detail::finite(out.rho[s]);
    for (unsigned k = 0; k < 3; ++k) {
      out.gradient[s][k] = energy.d[2 + 3 * s + k];
      out.valid = out.valid && detail::finite(out.gradient[s][k]);
    }
  }
  return out;
}
}  // namespace vibeqc::dft::point

#undef VIBEQC_XC_HD

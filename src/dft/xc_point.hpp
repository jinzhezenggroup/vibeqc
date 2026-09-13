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

/** Spin exchange and physical first derivatives. Choose the reduced gradient
 * or its reciprocal before squaring, so a tiny minority spin never forms
 * 0/0 from rho^(8/3) and |grad rho|^2. The two branches are algebraically
 * identical; no density floor or model extension is introduced here. */
struct Exchange {
  double energy{}, rho{}, gradient[3]{};
};
VIBEQC_XC_HD inline Exchange exchange(bool pbe, double rho, const double gradient[3]) {
  Exchange out;
  if (rho == 0.0) return out;
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double beta = 0.06672455060314922, kappa = 0.804;
  const double cx = 0.375 * ::pow(3.0 / pi, 1.0 / 3.0) * ::pow(4.0, 2.0 / 3.0);
  const double mu = beta * pi * pi / (12.0 * ::pow(6.0 * pi * pi, 2.0 / 3.0));
  const double rho13 = ::cbrt(rho), rho43 = rho * rho13;
  const double largest =
      ::fmax(::fabs(gradient[0]), ::fmax(::fabs(gradient[1]), ::fabs(gradient[2])));
  double enhancement = 1.0, radial_response = 0.0;
  if (pbe && largest != 0.0) {
    double direction[3], norm2 = 0.0;
    for (unsigned k = 0; k < 3; ++k) {
      direction[k] = gradient[k] / largest;
      norm2 += direction[k] * direction[k];
    }
    const double norm = ::sqrt(norm2);
    if (largest <= rho43 / norm) {
      double u[3], u2 = 0.0;
      for (unsigned k = 0; k < 3; ++k) {
        u[k] = gradient[k] / rho43;
        u2 += u[k] * u[k];
      }
      const double denominator = kappa + mu * u2;
      const double response = mu * kappa * kappa / (denominator * denominator);
      enhancement += kappa * mu * u2 / denominator;
      radial_response = response * u2;
      for (unsigned k = 0; k < 3; ++k) out.gradient[k] = -2.0 * cx * response * u[k];
    } else {
      // rho43 may itself underflow. Factoring rho/range first retains the
      // reciprocal reduced gradient and representable potential coefficients.
      const double t = (rho / largest) * (rho13 / norm), t2 = t * t;
      const double denominator = kappa * t2 + mu;
      const double response = mu * kappa * kappa / (denominator * denominator);
      enhancement += kappa - kappa * kappa * t2 / denominator;
      radial_response = response * t2;
      for (unsigned k = 0; k < 3; ++k)
        out.gradient[k] = -2.0 * cx * response * t2 * t * (direction[k] / norm);
    }
  }
  out.energy = -cx * rho43 * enhancement;
  // dF/drho contributes -(8/3) times the reduced-gradient response. Evaluate
  // the density derivative independently of energy underflow in rho43.
  out.rho = -cx * (4.0 / 3.0) * rho13 * (enhancement - 2.0 * radial_response);
  return out;
}

VIBEQC_XC_HD inline Jet correlation_per_scale(bool pbe, const Jet& a, const Jet& b, const Jet g[3],
                                              double scale, double gradient_ratio) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double beta = 0.06672455060314922;
  const double gamma = (1.0 - ::log(2.0)) / (pi * pi);
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
  if (pbe) {
    const Jet phi = (spin_two_thirds(up) + spin_two_thirds(down)) / 2.0;
    const Jet phi3 = phi * phi * phi;
    Jet g2;
    bool zero_gradient = true;
    for (unsigned k = 0; k < 3; ++k) {
      zero_gradient = zero_gradient && g[k].v == 0.0;
      g2 = g2 + g[k] * g[k];
    }
    // At exact cancellation correlation is PW. Check the components, not
    // g2.v: its square may underflow while a gradient derivative is nonzero.
    if (!zero_gradient) {
      // t2=g2/d. Form v=d/(d+A*g2) directly so neither a huge gradient/rho
      // ratio nor t2 or A*t2 is ever materialized. Multiplication order also
      // preserves d when gradient_ratio^2 alone would underflow.
      const Jet d = 16.0 * ::pow(2.0, 2.0 / 3.0) * 0.6203504908994001 * (gradient_ratio * scale13) *
                    gradient_ratio * power(n, 7.0 / 3.0) * phi * phi;
      const Jet aa = beta / (gamma * expm1(-eps / (gamma * phi3)));
      const Jet denominator = d + aa * g2;
      const Jet v = d / denominator;
      const Jet shape = 1.0 - v + v * v;
      if (v.v >= 0.5) {
        // For A*t2<=1, PW+H has no severe cancellation. This form also
        // retains PW at high density when 1-exp(eps/G) rounds to one.
        eps = eps + gamma * phi3 * log1p((beta / gamma) * g2 / (denominator * shape));
      } else {
        // Combine PW+H before evaluation in the large-gradient tail. The
        // equivalent bounded logarithm preserves tiny first derivatives.
        const Jet q = -expm1(eps / (gamma * phi3));
        eps = gamma * phi3 * log1p(-q * v * v / shape);
      }
    }
  }
  return n * eps;
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
  // Correlation depends only on the total gradient. Sum before normalizing
  // to retain cancellation between large opposite spin gradients. If a sum
  // exceeds FP64, normalize its finite summands instead. Both numerical
  // scales are held fixed during differentiation; they do not clip inputs.
  double total[3], gradient_scale = scale;
  for (unsigned k = 0; k < 3; ++k) {
    total[k] = gradient[0][k] + gradient[1][k];
    gradient_scale = detail::finite(total[k]) ? ::fmax(gradient_scale, ::fabs(total[k])) : DBL_MAX;
  }
  const double gradient_ratio = scale / gradient_scale;
  Jet g[3];
  for (unsigned k = 0; k < 3; ++k) {
    const double value = detail::finite(total[k])
                             ? total[k] / gradient_scale
                             : gradient[0][k] / gradient_scale + gradient[1][k] / gradient_scale;
    g[k] = Jet::variable(value, 2 + k);
    g[k].d[5 + k] = 1.0;  // Each independent spin contributes to the sum.
  }
  const Jet energy = detail::correlation_per_scale(pbe, a, b, g, scale, gradient_ratio);
  out.energy = scale * energy.v;
  out.valid = detail::finite(out.energy);
  for (unsigned s = 0; s < 2; ++s) {
    const auto x = detail::exchange(pbe, rho[s], gradient[s]);
    out.energy += x.energy;
    out.rho[s] = energy.d[s] + x.rho;
    out.valid = out.valid && detail::finite(out.rho[s]);
    for (unsigned k = 0; k < 3; ++k) {
      out.gradient[s][k] = gradient_ratio * energy.d[2 + 3 * s + k] + x.gradient[k];
      out.valid = out.valid && detail::finite(out.gradient[s][k]);
    }
  }
  out.valid = out.valid && detail::finite(out.energy);
  return out;
}
}  // namespace vibeqc::dft::point

#undef VIBEQC_XC_HD

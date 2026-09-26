#pragma once

#include "dft/xc_point.hpp"

#if defined(__CUDACC__)
#define VIBEQC_XC_RESPONSE_HD __host__ __device__
#else
#define VIBEQC_XC_RESPONSE_HD
#endif

namespace vibeqc::dft::point::detail {

/** Directional derivative of the existing point jet. This differentiates the
 * same scaled SCF expression; it does not store a point Hessian or introduce
 * another XC expression. The fixed scales are held constant in both passes. */
struct ResponseJet : Jet {
  static constexpr bool second_order = true;
  Jet tangent;
  VIBEQC_XC_RESPONSE_HD ResponseJet(double x = 0.0) : Jet(x) {}
  VIBEQC_XC_RESPONSE_HD ResponseJet(const Jet& x, const Jet& dx) : Jet(x), tangent(dx) {}
  VIBEQC_XC_RESPONSE_HD const Jet& base() const { return *this; }
  VIBEQC_XC_RESPONSE_HD static ResponseJet variable(double x, unsigned index, double direction) {
    return {Jet::variable(x, index), Jet(direction)};
  }
  VIBEQC_XC_RESPONSE_HD friend ResponseJet operator+(const ResponseJet& a, const ResponseJet& b) {
    return {a.base() + b.base(), a.tangent + b.tangent};
  }
  VIBEQC_XC_RESPONSE_HD friend ResponseJet operator-(const ResponseJet& a, const ResponseJet& b) {
    return {a.base() - b.base(), a.tangent - b.tangent};
  }
  VIBEQC_XC_RESPONSE_HD friend ResponseJet operator-(const ResponseJet& a) {
    return ResponseJet(0.0) - a;
  }
  VIBEQC_XC_RESPONSE_HD friend ResponseJet operator*(const ResponseJet& a, const ResponseJet& b) {
    return {a.base() * b.base(), a.tangent * b.base() + a.base() * b.tangent};
  }
  VIBEQC_XC_RESPONSE_HD friend ResponseJet operator/(const ResponseJet& a, const ResponseJet& b) {
    const Jet value = a.base() / b.base();
    return {value, (a.tangent - value * b.tangent) / b.base()};
  }
};
VIBEQC_XC_RESPONSE_HD inline ResponseJet power(const ResponseJet& x, double p) {
  return {power(x.base(), p), p * power(x.base(), p - 1.0) * x.tangent};
}
VIBEQC_XC_RESPONSE_HD inline ResponseJet log1p(const ResponseJet& x) {
  return {log1p(x.base()), x.tangent / (1.0 + x.base())};
}
VIBEQC_XC_RESPONSE_HD inline ResponseJet expm1(const ResponseJet& x) {
  Jet slope(::exp(x.v));
  for (unsigned i = 0; i < 8; ++i) slope.d[i] = slope.v * x.d[i];
  return {expm1(x.base()), slope * x.tangent};
}
}  // namespace vibeqc::dft::point::detail

namespace vibeqc::dft::point {
/** Spin-resolved directional derivative of the physical SCF potential.
 * The shared correlation expression and exchange potentials own the science.
 * An empty spin admits only zero density/gradient direction: orbital rotations
 * preserve that empty block, whereas its normal exchange Hessian is singular.
 * Numerical density/gradient scales remain fixed in both differentiation passes. */
VIBEQC_XC_RESPONSE_HD inline Value unrestricted_response(bool pbe, const double rho[2],
                                                         const double gradient[2][3],
                                                         const double delta_rho[2],
                                                         const double delta_gradient[2][3]) {
  Value out;
  bool zero_direction = true;
  for (unsigned s = 0; s < 2; ++s) {
    if (!detail::finite(rho[s]) || rho[s] < 0.0 || !detail::finite(delta_rho[s]) ||
        (rho[s] == 0.0 && delta_rho[s] != 0.0))
      out.valid = false;
    zero_direction = zero_direction && delta_rho[s] == 0.0;
    for (unsigned k = 0; k < 3; ++k) {
      if (!detail::valid_gradient_component(rho[s], gradient[s][k]) ||
          !detail::finite(delta_gradient[s][k]) || (rho[s] == 0.0 && delta_gradient[s][k] != 0.0))
        out.valid = false;
      zero_direction = zero_direction && delta_gradient[s][k] == 0.0;
    }
  }
  const double scale = rho[0] + rho[1];
  out.valid = out.valid && detail::finite(scale);
  if (!out.valid || zero_direction) return out;
  for (unsigned s = 0; s < 2; ++s)
    if (rho[s] > 0.0 && rho[s] / scale == 0.0) out.valid = false;
  if (!out.valid || scale == 0.0) {
    out.valid = false;
    return out;
  }
  using detail::ResponseJet;
  const auto a = ResponseJet::variable(rho[0] / scale, 0, delta_rho[0] / scale);
  const auto b = ResponseJet::variable(rho[1] / scale, 1, delta_rho[1] / scale);
  double total[3], gradient_scale = scale;
  for (unsigned k = 0; k < 3; ++k) {
    total[k] = gradient[0][k] + gradient[1][k];
    gradient_scale = detail::finite(total[k]) ? ::fmax(gradient_scale, ::fabs(total[k])) : DBL_MAX;
  }
  // Sum before normalization to retain opposing-spin cancellation, but scale
  // the summands separately when their otherwise valid sum would overflow.
  const auto normalized_sum = [gradient_scale](double x, double y) {
    const double sum = x + y;
    return detail::finite(sum) ? sum / gradient_scale : x / gradient_scale + y / gradient_scale;
  };
  const double ratio = scale / gradient_scale;
  ResponseJet g[3];
  for (unsigned k = 0; k < 3; ++k) {
    g[k] = ResponseJet::variable(normalized_sum(gradient[0][k], gradient[1][k]), 2 + k,
                                 normalized_sum(delta_gradient[0][k], delta_gradient[1][k]));
    g[k].d[5 + k] = 1.0;
  }
  const auto correlation = detail::correlation_per_scale(pbe, a, b, g, scale, ratio);
  for (unsigned s = 0; s < 2; ++s) {
    detail::Jet spin_rho(rho[s]), spin_gradient[3];
    spin_rho.d[0] = delta_rho[s];
    for (unsigned k = 0; k < 3; ++k) {
      spin_gradient[k] = detail::Jet(gradient[s][k]);
      spin_gradient[k].d[0] = delta_gradient[s][k];
    }
    const auto exchange = detail::exchange_value(pbe, spin_rho, spin_gradient);
    out.rho[s] = correlation.tangent.d[s] + exchange.rho.d[0];
    out.valid = out.valid && detail::finite(out.rho[s]);
    for (unsigned k = 0; k < 3; ++k) {
      out.gradient[s][k] = ratio * correlation.tangent.d[2 + 3 * s + k] + exchange.gradient[k].d[0];
      out.valid = out.valid && detail::finite(out.gradient[s][k]);
    }
  }
  return out;
}

/** Restricted total-density directional derivative of physical XC potential.
 * Only equal-spin RKS is qualified. Vacuum is allowed only with zero direction;
 * an undefined or nonrepresentable coefficient rejects the entire action. */
VIBEQC_XC_RESPONSE_HD inline Value restricted_response(bool pbe, double rho,
                                                       const double gradient[3], double delta_rho,
                                                       const double delta_gradient[3]) {
  Value out;
  if (!detail::finite(rho) || rho < 0.0 || !detail::finite(delta_rho)) out.valid = false;
  // RKS gradients are totals; SCF's vacuum admission sees the rounded equal-spin
  // gradient. A doubled cutoff would incorrectly accept halfway rounding ties.
  for (unsigned k = 0; k < 3; ++k)
    if (!detail::valid_gradient_component(rho, gradient[k] / 2.0) ||
        !detail::finite(delta_gradient[k]))
      out.valid = false;
  if (rho == 0.0) {
    if (delta_rho != 0.0) out.valid = false;
    for (unsigned k = 0; k < 3; ++k)
      if (delta_gradient[k] != 0.0) out.valid = false;
    return out;
  }
  // Zero is an exact direction even where an unused point Hessian would
  // overflow (for example rho^(4/3) underflow at a zero-gradient tail point).
  if (out.valid && delta_rho == 0.0 && delta_gradient[0] == 0.0 && delta_gradient[1] == 0.0 &&
      delta_gradient[2] == 0.0)
    return out;
  if (!out.valid || rho / 2.0 == 0.0) {
    out.valid = false;
    return out;
  }
  using detail::ResponseJet;
  const auto a = ResponseJet::variable(0.5, 0, (delta_rho / rho) / 2.0);
  const auto b = ResponseJet::variable(0.5, 1, (delta_rho / rho) / 2.0);
  double gradient_scale = rho;
  for (unsigned k = 0; k < 3; ++k) gradient_scale = ::fmax(gradient_scale, ::fabs(gradient[k]));
  const double ratio = rho / gradient_scale;
  ResponseJet g[3];
  for (unsigned k = 0; k < 3; ++k) {
    g[k] = ResponseJet::variable(gradient[k] / gradient_scale, 2 + k,
                                 delta_gradient[k] / gradient_scale);
    g[k].d[5 + k] = 1.0;
  }
  const auto correlation = detail::correlation_per_scale(pbe, a, b, g, rho, ratio);
  detail::Jet spin_rho(rho / 2.0), spin_gradient[3];
  spin_rho.d[0] = delta_rho / 2.0;
  for (unsigned k = 0; k < 3; ++k) {
    spin_gradient[k] = detail::Jet(gradient[k] / 2.0);
    spin_gradient[k].d[0] = delta_gradient[k] / 2.0;
  }
  const auto exchange = detail::exchange_value(pbe, spin_rho, spin_gradient);
  for (unsigned spin = 0; spin < 2; ++spin) {
    out.rho[spin] = correlation.tangent.d[spin] + exchange.rho.d[0];
    out.valid = out.valid && detail::finite(out.rho[spin]);
    for (unsigned k = 0; k < 3; ++k) {
      out.gradient[spin][k] =
          ratio * correlation.tangent.d[2 + 3 * spin + k] + exchange.gradient[k].d[0];
      out.valid = out.valid && detail::finite(out.gradient[spin][k]);
    }
  }
  return out;
}
}  // namespace vibeqc::dft::point

#undef VIBEQC_XC_RESPONSE_HD

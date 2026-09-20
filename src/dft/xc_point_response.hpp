#pragma once

#include "dft/xc_point.hpp"

namespace vibeqc::dft::point::detail {

/** Directional derivative of the existing point jet. This differentiates the
 * same scaled SCF expression; it does not store a point Hessian or introduce
 * another XC expression. The fixed scales are held constant in both passes. */
struct ResponseJet : Jet {
  static constexpr bool second_order = true;
  Jet tangent;
  ResponseJet(double x = 0.0) : Jet(x) {}
  ResponseJet(const Jet& x, const Jet& dx) : Jet(x), tangent(dx) {}
  const Jet& base() const { return *this; }
  static ResponseJet variable(double x, unsigned index, double direction) {
    return {Jet::variable(x, index), Jet(direction)};
  }
  friend ResponseJet operator+(const ResponseJet& a, const ResponseJet& b) {
    return {a.base() + b.base(), a.tangent + b.tangent};
  }
  friend ResponseJet operator-(const ResponseJet& a, const ResponseJet& b) {
    return {a.base() - b.base(), a.tangent - b.tangent};
  }
  friend ResponseJet operator-(const ResponseJet& a) { return ResponseJet(0.0) - a; }
  friend ResponseJet operator*(const ResponseJet& a, const ResponseJet& b) {
    return {a.base() * b.base(), a.tangent * b.base() + a.base() * b.tangent};
  }
  friend ResponseJet operator/(const ResponseJet& a, const ResponseJet& b) {
    const Jet value = a.base() / b.base();
    return {value, (a.tangent - value * b.tangent) / b.base()};
  }
};
inline ResponseJet power(const ResponseJet& x, double p) {
  return {power(x.base(), p), p * power(x.base(), p - 1.0) * x.tangent};
}
inline ResponseJet log1p(const ResponseJet& x) {
  return {log1p(x.base()), x.tangent / (1.0 + x.base())};
}
inline ResponseJet expm1(const ResponseJet& x) {
  Jet slope(::exp(x.v));
  for (unsigned i = 0; i < 8; ++i) slope.d[i] = slope.v * x.d[i];
  return {expm1(x.base()), slope * x.tangent};
}
}  // namespace vibeqc::dft::point::detail

namespace vibeqc::dft::point {
/** Restricted total-density directional derivative of physical XC potential.
 * Only equal-spin RKS is qualified. Vacuum is allowed only with zero direction;
 * an undefined or nonrepresentable coefficient rejects the entire action. */
inline Value restricted_response(bool pbe, double rho, const double gradient[3], double delta_rho,
                                 const double delta_gradient[3]) {
  Value out;
  if (!detail::finite(rho) || rho < 0.0 || !detail::finite(delta_rho)) out.valid = false;
  for (unsigned k = 0; k < 3; ++k)
    if (!detail::finite(gradient[k]) || !detail::finite(delta_gradient[k])) out.valid = false;
  if (rho == 0.0) {
    if (delta_rho != 0.0) out.valid = false;
    for (unsigned k = 0; k < 3; ++k)
      if (gradient[k] != 0.0 || delta_gradient[k] != 0.0) out.valid = false;
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

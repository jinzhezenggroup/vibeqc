#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dft/dispersion/d3_bj.hpp"

#if defined(__CUDACC__)
#define VIBEQC_D3_ZERO_HD __host__ __device__
#else
#define VIBEQC_D3_ZERO_HD
#endif

namespace vibeqc::dft::dispersion {

struct D3ZeroParameters {
  double s6{}, s8{}, rs6{}, rs8{}, alpha6{14.0};
  double cn_cutoff{}, pair_cutoff{}, pair_switch_width{};
};

VIBEQC_D3_ZERO_HD inline std::size_t d3_zero_workspace_elements(std::size_t atoms) {
  return d3_workspace_elements(atoms);
}

namespace d3_zero_detail {

VIBEQC_D3_ZERO_HD inline bool valid_parameters(const D3ZeroParameters& p) {
  using d3_detail::finite;
  const bool cn = p.cn_cutoff == 0.0 || (finite(p.cn_cutoff) && p.cn_cutoff > 0.0);
  const bool pair = p.pair_cutoff == 0.0 || (finite(p.pair_cutoff) && p.pair_cutoff > 0.0);
  return finite(p.s6) && finite(p.s8) && finite(p.rs6) && finite(p.rs8) && finite(p.alpha6) &&
         finite(p.pair_switch_width) && p.rs6 > 0.0 && p.rs8 > 0.0 && p.alpha6 > 0.0 && cn &&
         pair && p.pair_switch_width >= 0.0 &&
         (p.pair_switch_width == 0.0 ||
          (p.pair_cutoff > 0.0 && p.pair_switch_width < p.pair_cutoff));
}

struct DampedInversePower {
  double value{};
  double derivative_over_distance{};
};

VIBEQC_D3_ZERO_HD inline DampedInversePower damped_inverse_power(double r, double r2,
                                                                 double scaled_r0, double exponent,
                                                                 int power) {
  using d3_detail::finite;
  if (!(r > 0.0) || !(scaled_r0 > 0.0) || !(exponent > 0.0)) return {};
  const double ratio = scaled_r0 / r;
  const double x = 6.0 * pow(ratio, exponent);
  if (!finite(x)) {
    // Here 1+x rounds to x. Scale the complete weighted result instead of
    // rounding its tiny damping factor to zero before multiplying by r^-power.
    // The derivative is scaled independently since it can outlive value underflow.
    const double log_value =
        -static_cast<double>(power) * log2(r) - log2(6.0) - exponent * (log2(scaled_r0) - log2(r));
    const double factor = exponent - static_cast<double>(power);
    const double radial =
        factor == 0.0 ? 0.0 : copysign(exp2(log_value - log2(r2) + log2(fabs(factor))), factor);
    return {exp2(log_value), radial};
  }
  const double damping = 1.0 / (1.0 + x);
  double inverse = 1.0;
  for (int i = 0; i < power; ++i) inverse /= r;
  const double value = damping * inverse;
  const double derivative_over_distance =
      value / r2 * (exponent * (1.0 - damping) - static_cast<double>(power));
  return {value, derivative_over_distance};
}

}  // namespace d3_zero_detail

// Standalone non-periodic two-body D3(0) contribution with complete dE/dR,
// including coordination-number response of interpolated C6/C8 coefficients.
// Qualification only: this does not widen the public D3(BJ) runtime capability.
VIBEQC_D3_ZERO_HD inline D3Status evaluate_d3_zero(std::size_t n, const std::int32_t* z,
                                                   const double* xyz,
                                                   const D3ZeroParameters& parameters,
                                                   D3Tables tables, double* workspace,
                                                   std::size_t workspace_elements, double* energy,
                                                   double* gradient) {
  using namespace d3_detail;
  using namespace d3_zero_detail;
  if (!z || !xyz || !workspace || !energy || n == 0 || n > kD3MaximumAtomsPerSystem ||
      workspace_elements < d3_zero_workspace_elements(n) || !valid_parameters(parameters))
    return D3Status::invalid_argument;
  if (!tables.elements || !tables.pairs || !tables.reference_cn || !tables.reference_c6)
    return D3Status::invalid_argument;

  for (std::size_t atom = 0; atom < n; ++atom) {
    if (z[atom] < 1 || z[atom] > 86) return D3Status::unsupported;
    for (int axis = 0; axis < 3; ++axis)
      if (!finite(xyz[3 * atom + axis])) return D3Status::invalid_argument;
  }

  double* weights = workspace;
  double* derivatives = weights + 7 * n;
  double* adjoints = derivatives + 7 * n;
  double* cn = adjoints + n;
  for (std::size_t atom = 0; atom < n; ++atom) cn[atom] = adjoints[atom] = 0.0;
  *energy = 0.0;
  if (gradient)
    for (std::size_t i = 0; i < 3 * n; ++i) gradient[i] = 0.0;

  for (std::size_t second = 1; second < n; ++second) {
    for (std::size_t first = 0; first < second; ++first) {
      const double dx = xyz[3 * first] - xyz[3 * second];
      const double dy = xyz[3 * first + 1] - xyz[3 * second + 1];
      const double dz = xyz[3 * first + 2] - xyz[3 * second + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!finite(r2) || r2 < 1.0e-12) return D3Status::numerical_failure;
      if (parameters.cn_cutoff > 0.0 && r2 > parameters.cn_cutoff * parameters.cn_cutoff) continue;
      const double radius = tables.elements[z[first] - 1].covalent_radius +
                            tables.elements[z[second] - 1].covalent_radius;
      const double argument = 16.0 * (radius / sqrt(r2) - 1.0);
      const double value = logistic(argument);
      cn[first] += value;
      cn[second] += value;
    }
  }

  if (!prepare_weights(n, z, cn, tables, weights, derivatives)) return D3Status::numerical_failure;

  const double alpha8 = parameters.alpha6 + 2.0;
  if (!finite(alpha8) || !(alpha8 > 0.0)) return D3Status::invalid_argument;

  for (std::size_t second = 1; second < n; ++second) {
    for (std::size_t first = 0; first < second; ++first) {
      const double dx = xyz[3 * first] - xyz[3 * second];
      const double dy = xyz[3 * first + 1] - xyz[3 * second + 1];
      const double dz = xyz[3 * first + 2] - xyz[3 * second + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (parameters.pair_cutoff > 0.0 && r2 > parameters.pair_cutoff * parameters.pair_cutoff)
        continue;
      const double r = sqrt(r2);
      const auto c = coefficient(first, second, z, tables, weights, derivatives);
      if (!(c.c6 > 0.0) || !finite(c.c6) || !finite(c.first_cn) || !finite(c.second_cn))
        return D3Status::numerical_failure;
      const double rr =
          3.0 * tables.elements[z[first] - 1].r4r2 * tables.elements[z[second] - 1].r4r2;
      if (!(rr > 0.0) || !finite(rr)) return D3Status::numerical_failure;
      // Original D3(0) uses the tabulated pair R0AB, unlike BJ damping,
      // which constructs its damping radius from sqrt(C8/C6).
      const double r0 = tables.pairs[pair_index(z[first], z[second])].vdw_radius;
      if (!(r0 > 0.0) || !finite(r0)) return D3Status::numerical_failure;
      const auto term6 = damped_inverse_power(r, r2, parameters.rs6 * r0, parameters.alpha6, 6);
      const auto term8 = damped_inverse_power(r, r2, parameters.rs8 * r0, alpha8, 8);
      if (!finite(term6.value) || !finite(term8.value) || !finite(term6.derivative_over_distance) ||
          !finite(term8.derivative_over_distance))
        return D3Status::numerical_failure;

      const double phi = parameters.s6 * term6.value + parameters.s8 * rr * term8.value;
      const double phi_derivative_over_distance =
          parameters.s6 * term6.derivative_over_distance +
          parameters.s8 * rr * term8.derivative_over_distance;
      double cutoff_derivative = 0.0;
      const double cutoff =
          smooth_cutoff(r, parameters.pair_cutoff, parameters.pair_switch_width, cutoff_derivative);
      const double damping = cutoff * phi;
      const double pair_energy = -c.c6 * damping;
      if (!finite(pair_energy)) return D3Status::numerical_failure;
      *energy += pair_energy;
      adjoints[first] += -c.first_cn * damping;
      adjoints[second] += -c.second_cn * damping;
      if (gradient) {
        const double scale =
            -c.c6 * (cutoff * phi_derivative_over_distance + cutoff_derivative * phi / r);
        add_pair_gradient(first, second, dx, dy, dz, scale, gradient);
      }
    }
  }

  if (gradient) {
    for (std::size_t second = 1; second < n; ++second) {
      for (std::size_t first = 0; first < second; ++first) {
        const double dx = xyz[3 * first] - xyz[3 * second];
        const double dy = xyz[3 * first + 1] - xyz[3 * second + 1];
        const double dz = xyz[3 * first + 2] - xyz[3 * second + 2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (parameters.cn_cutoff > 0.0 && r2 > parameters.cn_cutoff * parameters.cn_cutoff)
          continue;
        const double r = sqrt(r2);
        const double radius = tables.elements[z[first] - 1].covalent_radius +
                              tables.elements[z[second] - 1].covalent_radius;
        const double argument = 16.0 * (radius / r - 1.0);
        const double e = exp(-fabs(argument));
        const double logistic_derivative = e / ((1.0 + e) * (1.0 + e));
        const double derivative = -16.0 * radius / r2 * logistic_derivative;
        const double scale = (adjoints[first] + adjoints[second]) * derivative / r;
        add_pair_gradient(first, second, dx, dy, dz, scale, gradient);
      }
    }
    for (std::size_t q = 0; q < 3 * n; ++q)
      if (!finite(gradient[q])) return D3Status::numerical_failure;
  }
  if (!finite(*energy)) return D3Status::numerical_failure;
  return D3Status::success;
}

}  // namespace vibeqc::dft::dispersion

#undef VIBEQC_D3_ZERO_HD

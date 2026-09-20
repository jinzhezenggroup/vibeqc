#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dft/dispersion/d3_bj.hpp"

#if defined(__CUDACC__)
#define VIBEQC_D3_ATM_HD __host__ __device__
#else
#define VIBEQC_D3_ATM_HD
#endif

namespace vibeqc::dft::dispersion {

// Standalone molecular D3(BJ)-ATM reference parameters. The BJ three-body term
// uses the upstream zero-damping ATM contract with rs9=4/3 and effective
// exponent alp+2=16 (the tabulated BJ alpha is 14).
struct D3ATMParameters {
  double s9{1.0};
  // Zero means an unscreened reference path. Positive values are bohr.
  double cn_cutoff{}, atm_cutoff{}, atm_switch_width{};
};

inline constexpr double kD3BjAtmRs9 = 4.0 / 3.0;
inline constexpr double kD3BjAtmAlpha = 16.0;

VIBEQC_D3_ATM_HD inline std::size_t d3_atm_workspace_elements(std::size_t atoms) {
  return d3_workspace_elements(atoms);
}

namespace d3_atm_detail {

VIBEQC_D3_ATM_HD inline bool valid_parameters(const D3ATMParameters& p) {
  using d3_detail::finite;
  const bool cn = p.cn_cutoff == 0.0 || (finite(p.cn_cutoff) && p.cn_cutoff > 0.0);
  const bool atm = p.atm_cutoff == 0.0 || (finite(p.atm_cutoff) && p.atm_cutoff > 0.0);
  return finite(p.s9) && finite(p.atm_switch_width) && p.s9 >= 0.0 && cn && atm &&
         p.atm_switch_width >= 0.0 &&
         (p.atm_switch_width == 0.0 || (p.atm_cutoff > 0.0 && p.atm_switch_width < p.atm_cutoff));
}

VIBEQC_D3_ATM_HD inline double atm_radial(double x, double y, double z, double r5, double damping,
                                          double angle, double damping_derivative, double c9) {
  const double angle_derivative =
      -0.375 *
      (x * x * x + x * x * (y + z) + x * (3.0 * y * y + 2.0 * y * z + 3.0 * z * z) -
       5.0 * (y - z) * (y - z) * (y + z)) /
      r5;
  return c9 * (-angle_derivative * damping + angle * damping_derivative) / x;
}

}  // namespace d3_atm_detail

// Standalone non-periodic D3(BJ)-ATM contribution with complete dE/dR,
// including coordination-number response of all three C6 coefficients. This
// intentionally does not alter the production D3 runtime or accept nonzero s9
// there; it is the independently gated scientific primitive for that follow-up.
// Workspace is exactly 16*n doubles, matching evaluate_d3_bj.
VIBEQC_D3_ATM_HD inline D3Status evaluate_d3_bj_atm(std::size_t n, const std::int32_t* z,
                                                    const double* xyz,
                                                    const D3ATMParameters& parameters,
                                                    D3Tables tables, double* workspace,
                                                    std::size_t workspace_elements, double* energy,
                                                    double* gradient) {
  using namespace d3_detail;
  using namespace d3_atm_detail;
  if (!z || !xyz || !workspace || !energy || n == 0 || n > kD3MaximumAtomsPerSystem ||
      workspace_elements < d3_atm_workspace_elements(n) || !valid_parameters(parameters))
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
  if (parameters.s9 == 0.0 || n < 3) return D3Status::success;

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

  for (std::size_t i = 2; i < n; ++i) {
    for (std::size_t j = 1; j < i; ++j) {
      const double xij = xyz[3 * i] - xyz[3 * j];
      const double yij = xyz[3 * i + 1] - xyz[3 * j + 1];
      const double zij = xyz[3 * i + 2] - xyz[3 * j + 2];
      const double a = xij * xij + yij * yij + zij * zij;
      if (!finite(a) || a < 1.0e-12) return D3Status::numerical_failure;
      if (parameters.atm_cutoff > 0.0 && a > parameters.atm_cutoff * parameters.atm_cutoff)
        continue;
      const double rij = sqrt(a);
      double dsij = 0.0;
      const double sij =
          smooth_cutoff(rij, parameters.atm_cutoff, parameters.atm_switch_width, dsij);
      const auto cij = coefficient(i, j, z, tables, weights, derivatives);

      for (std::size_t k = 0; k < j; ++k) {
        const double xik = xyz[3 * i] - xyz[3 * k];
        const double yik = xyz[3 * i + 1] - xyz[3 * k + 1];
        const double zik = xyz[3 * i + 2] - xyz[3 * k + 2];
        const double xjk = xyz[3 * j] - xyz[3 * k];
        const double yjk = xyz[3 * j + 1] - xyz[3 * k + 1];
        const double zjk = xyz[3 * j + 2] - xyz[3 * k + 2];
        const double b = xik * xik + yik * yik + zik * zik;
        const double c = xjk * xjk + yjk * yjk + zjk * zjk;
        if (!finite(b) || !finite(c) || b < 1.0e-12 || c < 1.0e-12)
          return D3Status::numerical_failure;
        if (parameters.atm_cutoff > 0.0 && (b > parameters.atm_cutoff * parameters.atm_cutoff ||
                                            c > parameters.atm_cutoff * parameters.atm_cutoff))
          continue;

        const auto cik = coefficient(i, k, z, tables, weights, derivatives);
        const auto cjk = coefficient(j, k, z, tables, weights, derivatives);
        if (!(cij.c6 > 0.0 && cik.c6 > 0.0 && cjk.c6 > 0.0) || !finite(cij.c6) || !finite(cik.c6) ||
            !finite(cjk.c6))
          return D3Status::numerical_failure;

        const double rik = sqrt(b), rjk = sqrt(c);
        double dsik = 0.0, dsjk = 0.0;
        const double sik =
            smooth_cutoff(rik, parameters.atm_cutoff, parameters.atm_switch_width, dsik);
        const double sjk =
            smooth_cutoff(rjk, parameters.atm_cutoff, parameters.atm_switch_width, dsjk);
        const double sw = sij * sik * sjk;

        const double r2p = a * b * c;
        const double r1p = sqrt(r2p);
        const double r3p = r2p * r1p;
        const double r5p = r3p * r2p;
        const double r0ij = kD3BjAtmRs9 * tables.pairs[pair_index(z[i], z[j])].vdw_radius;
        const double r0ik = kD3BjAtmRs9 * tables.pairs[pair_index(z[i], z[k])].vdw_radius;
        const double r0jk = kD3BjAtmRs9 * tables.pairs[pair_index(z[j], z[k])].vdw_radius;
        const double r0 = r0ij * r0ik * r0jk;
        if (!(r0 > 0.0) || !finite(r0)) return D3Status::numerical_failure;
        const double rp = pow(r0 / r1p, kD3BjAtmAlpha / 3.0);
        const double damping = 1.0 / (1.0 + 6.0 * rp);
        const double angle = 0.375 * (a + c - b) * (a - c + b) * (-a + c + b) / r5p + 1.0 / r3p;
        const double c9 = -parameters.s9 * sqrt(cij.c6 * cik.c6 * cjk.c6);
        const double de0 = angle * damping * c9;
        const double de = de0 * sw;
        if (!finite(de)) return D3Status::numerical_failure;
        *energy -= de;

        adjoints[i] -= 0.5 * de * (cij.first_cn / cij.c6 + cik.first_cn / cik.c6);
        adjoints[j] -= 0.5 * de * (cij.second_cn / cij.c6 + cjk.first_cn / cjk.c6);
        adjoints[k] -= 0.5 * de * (cik.second_cn / cik.c6 + cjk.second_cn / cjk.c6);

        if (gradient) {
          const double ddamping = -2.0 * kD3BjAtmAlpha * rp * damping * damping;
          const double scale_ij = sw * atm_radial(a, c, b, r5p, damping, angle, ddamping, c9) -
                                  de0 * dsij / rij * sik * sjk;
          const double scale_ik = sw * atm_radial(b, c, a, r5p, damping, angle, ddamping, c9) -
                                  de0 * dsik / rik * sij * sjk;
          const double scale_jk = sw * atm_radial(c, b, a, r5p, damping, angle, ddamping, c9) -
                                  de0 * dsjk / rjk * sij * sik;
          add_pair_gradient(i, j, xij, yij, zij, scale_ij, gradient);
          add_pair_gradient(i, k, xik, yik, zik, scale_ik, gradient);
          add_pair_gradient(j, k, xjk, yjk, zjk, scale_jk, gradient);
        }
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

#undef VIBEQC_D3_ATM_HD

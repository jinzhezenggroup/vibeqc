#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "d3_data.hpp"

#if defined(__CUDACC__)
#define VIBEQC_D3_HD __host__ __device__
#else
#define VIBEQC_D3_HD
#endif

namespace vibeqc::dft::dispersion {

inline constexpr std::size_t kD3MaximumAtomsPerSystem = 4096;

enum class D3Status : std::int32_t {
  success = 0,
  invalid_argument = 1,
  unsupported = 2,
  numerical_failure = 3,
};

struct D3Parameters {
  double s6{}, s8{}, a1{}, a2{}, s9{};
  // Zero means an unscreened reference path. Positive values are bohr.
  double cn_cutoff{}, pair_cutoff{}, pair_switch_width{};
};

struct D3Tables {
  const d3_data::ElementData* elements{};
  const d3_data::PairData* pairs{};
  const double* reference_cn{};
  const double* reference_c6{};
};

inline D3Tables d3_host_tables() {
  return {d3_data::kElements.data(), d3_data::kPairs.data(), d3_data::kReferenceCn.data(),
          d3_data::kReferenceC6.data()};
}

VIBEQC_D3_HD inline std::size_t d3_workspace_elements(std::size_t atoms) {
  return atoms <= kD3MaximumAtomsPerSystem ? 16 * atoms : 0;
}

namespace d3_detail {

VIBEQC_D3_HD inline bool finite(double x) { return x == x && x <= DBL_MAX && x >= -DBL_MAX; }

VIBEQC_D3_HD inline bool valid_parameters(const D3Parameters& p) {
  const bool cn = p.cn_cutoff == 0.0 || (finite(p.cn_cutoff) && p.cn_cutoff > 0.0);
  const bool pair = p.pair_cutoff == 0.0 || (finite(p.pair_cutoff) && p.pair_cutoff > 0.0);
  return finite(p.s6) && finite(p.s8) && finite(p.a1) && finite(p.a2) && finite(p.s9) &&
         finite(p.pair_switch_width) && p.s6 >= 0.0 && p.s8 >= 0.0 && p.a1 >= 0.0 && p.a2 >= 0.0 &&
         (p.a1 > 0.0 || p.a2 > 0.0) && p.s9 == 0.0 && cn && pair && p.pair_switch_width >= 0.0 &&
         (p.pair_switch_width == 0.0 ||
          (p.pair_cutoff > 0.0 && p.pair_switch_width < p.pair_cutoff));
}

VIBEQC_D3_HD inline double logistic(double argument) {
  const double e = exp(-fabs(argument));
  return argument >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
}

VIBEQC_D3_HD inline double smooth_cutoff(double r, double cutoff, double width, double& dr) {
  dr = 0.0;
  if (cutoff == 0.0 || width == 0.0 || r <= cutoff - width) return 1.0;
  if (r >= cutoff) return 0.0;
  const double x = (cutoff - r) / width;
  dr = -30.0 * x * x * (1.0 - x) * (1.0 - x) / width;
  return x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
}

VIBEQC_D3_HD inline std::size_t pair_index(int z_first, int z_second) {
  const int lo = z_first < z_second ? z_first : z_second;
  const int hi = z_first < z_second ? z_second : z_first;
  return static_cast<std::size_t>(lo - 1 + hi * (hi - 1) / 2);
}

VIBEQC_D3_HD inline double reference_c6(D3Tables tables, int zi, int zj, int ri, int rj) {
  const auto pair = tables.pairs[pair_index(zi, zj)];
  if (zi <= zj) {
    return tables.reference_c6[pair.c6_offset +
                               static_cast<std::size_t>(rj) * pair.first_reference_count + ri];
  }
  return tables.reference_c6[pair.c6_offset +
                             static_cast<std::size_t>(ri) * pair.first_reference_count + rj];
}

struct Coefficient {
  double c6{}, first_cn{}, second_cn{};
};

VIBEQC_D3_HD inline bool prepare_weights(std::size_t n, const std::int32_t* z, const double* cn,
                                         D3Tables tables, double* weights, double* derivatives) {
  for (std::size_t atom = 0; atom < n; ++atom) {
    const auto element = tables.elements[z[atom] - 1];
    if (element.reference_count == 0 || element.reference_count > 7 || !finite(cn[atom]))
      return false;
    double norm = 0.0;
    double derivative_norm = 0.0;
    double maximum_cn = -DBL_MAX;
    for (int ref = 0; ref < 7; ++ref) {
      const std::size_t index = 7 * atom + ref;
      weights[index] = derivatives[index] = 0.0;
      if (ref >= element.reference_count) continue;
      const double ref_cn = tables.reference_cn[element.reference_offset + ref];
      maximum_cn = fmax(maximum_cn, ref_cn);
      const double delta = ref_cn - cn[atom];
      const double u = exp(-4.0 * delta * delta);
      weights[index] = u;
      norm += u;
      derivative_norm += 8.0 * delta * u;
    }
    const double inverse_norm = 1.0 / norm;
    for (int ref = 0; ref < element.reference_count; ++ref) {
      const std::size_t index = 7 * atom + ref;
      const double ref_cn = tables.reference_cn[element.reference_offset + ref];
      const double delta = ref_cn - cn[atom];
      const double u = weights[index];
      double weight = u * inverse_norm;
      if (!finite(weight)) weight = ref_cn == maximum_cn ? 1.0 : 0.0;
      double derivative =
          8.0 * delta * u * inverse_norm - u * derivative_norm * inverse_norm * inverse_norm;
      if (!finite(derivative)) derivative = 0.0;
      weights[index] = weight;
      derivatives[index] = derivative;
    }
  }
  return true;
}

VIBEQC_D3_HD inline Coefficient coefficient(std::size_t first, std::size_t second,
                                            const std::int32_t* z, D3Tables tables,
                                            const double* weights, const double* derivatives) {
  const auto first_element = tables.elements[z[first] - 1];
  const auto second_element = tables.elements[z[second] - 1];
  Coefficient out{};
  for (int i = 0; i < first_element.reference_count; ++i) {
    for (int j = 0; j < second_element.reference_count; ++j) {
      const double ref = reference_c6(tables, z[first], z[second], i, j);
      const double wi = weights[7 * first + i], wj = weights[7 * second + j];
      out.c6 += wi * wj * ref;
      out.first_cn += derivatives[7 * first + i] * wj * ref;
      out.second_cn += wi * derivatives[7 * second + j] * ref;
    }
  }
  return out;
}

VIBEQC_D3_HD inline void add_pair_gradient(std::size_t first, std::size_t second, double dx,
                                           double dy, double dz, double scale, double* gradient) {
  gradient[3 * first] += scale * dx;
  gradient[3 * first + 1] += scale * dy;
  gradient[3 * first + 2] += scale * dz;
  gradient[3 * second] -= scale * dx;
  gradient[3 * second + 1] -= scale * dy;
  gradient[3 * second + 2] -= scale * dz;
}

}  // namespace d3_detail

// Complete two-body D3(BJ) dE/dR including coordination-number response.
// Workspace is exactly 16*n doubles: weights[7n], dweight/dCN[7n], adjoints[n], CN[n].
VIBEQC_D3_HD inline D3Status evaluate_d3_bj(std::size_t n, const std::int32_t* z, const double* xyz,
                                            const D3Parameters& parameters, D3Tables tables,
                                            double* workspace, std::size_t workspace_elements,
                                            double* energy, double* gradient) {
  using namespace d3_detail;
  if (!z || !xyz || !workspace || !energy || n == 0 || n > kD3MaximumAtomsPerSystem ||
      workspace_elements < d3_workspace_elements(n) || !valid_parameters(parameters))
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
      if (!finite(c.c6) || !finite(c.first_cn) || !finite(c.second_cn))
        return D3Status::numerical_failure;
      const double rr =
          3.0 * tables.elements[z[first] - 1].r4r2 * tables.elements[z[second] - 1].r4r2;
      const double rd = parameters.a1 * sqrt(rr) + parameters.a2;
      if (!(rr > 0.0) || !(rd > 0.0) || !finite(rr) || !finite(rd))
        return D3Status::numerical_failure;
      const double r4 = r2 * r2, r6 = r4 * r2, r8 = r4 * r4;
      const double rd2 = rd * rd, rd4 = rd2 * rd2, rd6 = rd4 * rd2, rd8 = rd4 * rd4;
      const double t6 = 1.0 / (r6 + rd6), t8 = 1.0 / (r8 + rd8);
      const double phi = parameters.s6 * t6 + parameters.s8 * rr * t8;
      const double derivative_over_distance =
          parameters.s6 * (-6.0 * r4 * t6 * t6) + parameters.s8 * rr * (-8.0 * r6 * t8 * t8);
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
            -c.c6 * (cutoff * derivative_over_distance + cutoff_derivative * phi / r);
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
  }
  if (!finite(*energy)) return D3Status::numerical_failure;
  return D3Status::success;
}

}  // namespace vibeqc::dft::dispersion

#undef VIBEQC_D3_HD

// SPDX-License-Identifier: GPL-3.0-or-later
// Adapted from xTBloom 2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3.
// See external/xtbloom-d3/manifest.json and CUDA_MKL_LINKING_EXCEPTION there.
// Repository-only qualification baseline: NOT a production DFT provider.
#pragma once
#include <cmath>
#include <cstdint>
#ifdef __CUDACC__
#define VIBEQC_D3_HD __host__ __device__
#else
#define VIBEQC_D3_HD
#endif
namespace vibeqc_d3_baseline {
using std::exp;
using std::fmax;
using std::isfinite;
using std::sqrt;
struct Plan {
  const std::int64_t* atom_offsets;
  const std::int64_t* pair_offsets;
  const double* covalent_radii;
  const double* reference_counts;
  const double* reference_cn;
  const double* reference_c6;
  const double* pair_rrij;
  const double* pair_damping_radii;
  double s6, s8, cn_cutoff, pair_cutoff, switch_width;
};
struct Workspace {
  double* weights;
  double* weight_cn_derivatives;
  double* coordination_adjoints;
};
VIBEQC_D3_HD inline double smooth_cutoff(double r, double cutoff, double width, double* dr) {
  *dr = 0.0;
  if (r > cutoff) return 0.0;
  if (width == 0.0 || r <= cutoff - width) return 1.0;
  if (r >= cutoff) return 0.0;
  const double x = (cutoff - r) / width;
  *dr = -30.0 * x * x * (1.0 - x) * (1.0 - x) / width;
  return x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
}
VIBEQC_D3_HD inline bool prepare_d3_weights(const Plan& plan, std::int64_t begin, std::int64_t end,
                                            const double* coordination,
                                            const Workspace& workspace) {
  for (std::int64_t atom = begin; atom < end; ++atom) {
    const std::uint8_t count = plan.reference_counts[atom];
    if (count == 0u || count > 7 || !isfinite(coordination[atom])) {
      return false;
    }
    const std::int64_t offset = atom * 7;
    double norm = 0.0;
    double derivative_norm = 0.0;
    double maximum_cn = -0x1.fffffffffffffp+1023;
    for (std::int64_t ref = 0; ref < 7; ++ref) {
      workspace.weights[offset + ref] = 0.0;
      workspace.weight_cn_derivatives[offset + ref] = 0.0;
      if (ref >= count) continue;
      const double ref_cn = plan.reference_cn[offset + ref];
      if (!isfinite(ref_cn)) return false;
      maximum_cn = fmax(maximum_cn, ref_cn);
      const double delta = ref_cn - coordination[atom];
      const double u = exp(-4.0 * delta * delta);
      workspace.weights[offset + ref] = u;
      norm += u;
      derivative_norm += 2.0 * 4.0 * delta * u;
    }
    const double inverse_norm = 1.0 / norm;
    for (std::int64_t ref = 0; ref < count; ++ref) {
      const double ref_cn = plan.reference_cn[offset + ref];
      const double delta = ref_cn - coordination[atom];
      const double u = workspace.weights[offset + ref];
      double weight = u * inverse_norm;
      if (!isfinite(weight)) weight = ref_cn == maximum_cn ? 1.0 : 0.0;
      double derivative =
          2.0 * 4.0 * delta * u * inverse_norm - u * derivative_norm * inverse_norm * inverse_norm;
      if (!isfinite(derivative)) derivative = 0.0;
      workspace.weights[offset + ref] = weight;
      workspace.weight_cn_derivatives[offset + ref] = derivative;
    }
  }
  return true;
}

VIBEQC_D3_HD inline bool pair_coefficient(const Plan& plan, std::int64_t packed_pair,
                                          std::int64_t first, std::int64_t second,
                                          const Workspace& workspace, double* c6, double* first_cn,
                                          double* second_cn) {
  *c6 = 0.0;
  *first_cn = 0.0;
  *second_cn = 0.0;
  const std::int64_t first_offset = first * 7;
  const std::int64_t second_offset = second * 7;
  const std::int64_t c6_offset = packed_pair * 49;
  const std::int64_t first_count = plan.reference_counts[first];
  const std::int64_t second_count = plan.reference_counts[second];
  for (std::int64_t i = 0; i < first_count; ++i) {
    for (std::int64_t j = 0; j < second_count; ++j) {
      const double ref = plan.reference_c6[c6_offset + i * 7 + j];
      const double wi = workspace.weights[first_offset + i];
      const double wj = workspace.weights[second_offset + j];
      const double di = workspace.weight_cn_derivatives[first_offset + i];
      const double dj = workspace.weight_cn_derivatives[second_offset + j];
      if (!isfinite(ref) || !isfinite(wi) || !isfinite(wj) || !isfinite(di) || !isfinite(dj)) {
        return false;
      }
      *c6 += wi * wj * ref;
      *first_cn += di * wj * ref;
      *second_cn += wi * dj * ref;
      if (!isfinite(*c6) || !isfinite(*first_cn) || !isfinite(*second_cn)) return false;
    }
  }
  return true;
}

VIBEQC_D3_HD inline bool add_d3(const Plan& plan, std::int64_t system, const double* positions,
                                const double* coordination, double* energy, double* gradient,
                                const Workspace& workspace) {
  const std::int64_t begin = plan.atom_offsets[system];
  const std::int64_t end = plan.atom_offsets[system + 1];
  if (!prepare_d3_weights(plan, begin, end, coordination, workspace)) return false;
  for (std::int64_t atom = begin; atom < end; ++atom) workspace.coordination_adjoints[atom] = 0.0;

  for (std::int64_t second = begin + 1; second < end; ++second) {
    const std::int64_t local_second = second - begin;
    for (std::int64_t first = begin; first < second; ++first) {
      const std::int64_t local_first = first - begin;
      const std::int64_t packed =
          plan.pair_offsets[system] + local_second * (local_second - 1) / 2 + local_first;
      double dx = positions[3 * first] - positions[3 * second];
      double dy = positions[3 * first + 1] - positions[3 * second + 1];
      double dz = positions[3 * first + 2] - positions[3 * second + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!isfinite(r2)) return false;
      if (r2 < 2.2204460492503131e-16 || r2 > plan.pair_cutoff * plan.pair_cutoff) continue;
      double c6 = 0.0, dfirst = 0.0, dsecond = 0.0;
      if (!pair_coefficient(plan, packed, first, second, workspace, &c6, &dfirst, &dsecond)) {
        return false;
      }
      const double r = sqrt(r2);
      const double rd = plan.pair_damping_radii[packed];
      const double rr = plan.pair_rrij[packed];
      if (!(rd > 0.0) || !(rr > 0.0) || !isfinite(rd) || !isfinite(rr)) return false;
      const double rd2 = rd * rd;
      const double rd4 = rd2 * rd2;
      const double rd6 = rd4 * rd2;
      const double rd8 = rd4 * rd4;
      const double r4 = r2 * r2;
      const double t6 = 1.0 / (r4 * r2 + rd6);
      const double t8 = 1.0 / (r4 * r4 + rd8);
      const double phi = plan.s6 * t6 + plan.s8 * rr * t8;
      const double derivative_over_distance =
          plan.s6 * (-6.0 * r4 * t6 * t6) + plan.s8 * rr * (-8.0 * r4 * r2 * t8 * t8);
      double cutoff_derivative = 0.0;
      const double cutoff =
          smooth_cutoff(r, plan.pair_cutoff, plan.switch_width, &cutoff_derivative);
      const double damping = cutoff * phi;
      const double e = -c6 * damping;
      if (!isfinite(e)) return false;
      if (energy != nullptr) *energy += e;
      workspace.coordination_adjoints[first] += -dfirst * damping;
      workspace.coordination_adjoints[second] += -dsecond * damping;
      if (gradient != nullptr) {
        const double scale =
            -c6 * (cutoff * derivative_over_distance + cutoff_derivative * phi / r);
        gradient[3 * first] += scale * dx;
        gradient[3 * first + 1] += scale * dy;
        gradient[3 * first + 2] += scale * dz;
        gradient[3 * second] -= scale * dx;
        gradient[3 * second + 1] -= scale * dy;
        gradient[3 * second + 2] -= scale * dz;
      }
    }
  }

  if (gradient != nullptr) {
    for (std::int64_t second = begin + 1; second < end; ++second) {
      for (std::int64_t first = begin; first < second; ++first) {
        const double dx = positions[3 * first] - positions[3 * second];
        const double dy = positions[3 * first + 1] - positions[3 * second + 1];
        const double dz = positions[3 * first + 2] - positions[3 * second + 2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < 1.0e-12 || r2 > plan.cn_cutoff * plan.cn_cutoff) continue;
        const double r = sqrt(r2);
        const double radius = plan.covalent_radii[first] + plan.covalent_radii[second];
        if (!(radius > 0.0) || !isfinite(radius)) return false;
        const double argument = 16.0 * (radius / r - 1.0);
        double logistic_derivative = 0.0;
        if (argument >= 0.0) {
          const double ex = exp(-argument);
          const double den = 1.0 + ex;
          logistic_derivative = ex / (den * den);
        } else {
          const double ex = exp(argument);
          const double den = 1.0 + ex;
          logistic_derivative = ex / (den * den);
        }
        const double derivative = -16.0 * radius / (r * r) * logistic_derivative;
        const double adj =
            workspace.coordination_adjoints[first] + workspace.coordination_adjoints[second];
        const double scale = adj * derivative / r;
        gradient[3 * first] += scale * dx;
        gradient[3 * first + 1] += scale * dy;
        gradient[3 * first + 2] += scale * dz;
        gradient[3 * second] -= scale * dx;
        gradient[3 * second + 1] -= scale * dy;
        gradient[3 * second + 2] -= scale * dz;
      }
    }
  }
  return energy == nullptr || isfinite(*energy);
}

// Packed input: positions[3*n], radii[n], counts[n], reference_cn[7*n],
// reference_c6[49*npair], rrij[npair], damping_radii[npair].
// Parameters: s6,s8,CN cutoff,pair cutoff,switch width. Scratch: 16*n doubles.
// Output: energy followed by 3*n derivatives dE/dR (not forces).
VIBEQC_D3_HD inline bool evaluate(std::int64_t n, const double* packed, const double* parameters,
                                  double* output, double* scratch) {
  const std::int64_t pairs = n * (n - 1) / 2;
  const std::int64_t atoms_off[2] = {0, n}, pairs_off[2] = {0, pairs};
  const double* radii = packed + 3 * n;
  const double* counts = radii + n;
  const double* refs = counts + n;
  const double* c6 = refs + 7 * n;
  Plan plan{
      atoms_off,       pairs_off,       radii,         counts,        refs,          c6,
      c6 + 49 * pairs, c6 + 50 * pairs, parameters[0], parameters[1], parameters[2], parameters[3],
      parameters[4]};
  Workspace workspace{scratch, scratch + 7 * n, scratch + 14 * n};
  double* cn = scratch + 15 * n;
  for (std::int64_t i = 0; i < n; ++i) cn[i] = 0.0;
  for (std::int64_t j = 1; j < n; ++j)
    for (std::int64_t i = 0; i < j; ++i) {
      double r2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        const double d = packed[3 * i + k] - packed[3 * j + k];
        r2 += d * d;
      }
      if (!isfinite(r2) || r2 < 1.0e-12) return false;
      if (r2 > plan.cn_cutoff * plan.cn_cutoff) continue;
      const double a = 16.0 * ((radii[i] + radii[j]) / sqrt(r2) - 1.0);
      const double e = exp(-fabs(a));
      const double value = a >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
      cn[i] += value;
      cn[j] += value;
    }
  for (std::int64_t i = 0; i < 1 + 3 * n; ++i) output[i] = 0.0;
  if (!add_d3(plan, 0, packed, cn, output, output + 1, workspace)) return false;
  for (std::int64_t i = 0; i < 1 + 3 * n; ++i)
    if (!isfinite(output[i])) return false;
  return true;
}
}  // namespace vibeqc_d3_baseline
#undef VIBEQC_D3_HD

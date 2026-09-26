// SPDX-License-Identifier: GPL-3.0-or-later
// Molecular D4 qualification baseline adapted from xTBloom
// 2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3, src/model/gfn2/d4.cpp and
// src/backends/cuda/gfn2_d4.cu. See d4_manifest.json and THIRD_PARTY_NOTICES.md.
// This is NOT a public DFT-D4 method: charges are independent inputs, and the
// bundled reference polarizabilities are GFN2-specific (not EEQ reference data).
#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dft/dispersion/d4_data.hpp"
#include "dft/dispersion/d4_math.hpp"
#include "generated_method_parameters.hpp"

#if defined(__CUDACC__)
#define VIBEQC_D4_HD __host__ __device__
#else
#define VIBEQC_D4_HD
#endif

namespace vibeqc::dft::dispersion {

// A deliberately bounded scalar baseline, not a promoted GPU schedule.
inline constexpr int kD4MaximumAtoms = 256;
inline constexpr int kD4MaximumReferences = 7;
enum class D4ReferenceModel : int { gfn2 = 1, eeq = 2 };
enum class D4Status : int { success, invalid_argument, unsupported, numerical_failure };

struct D4Parameters {
  D4ReferenceModel reference_model;
  double s6, s8, s9, a1, a2;
  double cn_cutoff, pair_cutoff, atm_cutoff;
  // Charge-scaling zeta parameters. Standard D4 uses 3/2; r2SCAN-3c uses 2/1.
  double ga = 3.0;
  double gc = 2.0;
};

// No generic/default DFT parameter alias is exposed.
VIBEQC_D4_HD inline D4Parameters gfn2_d4_parameters() {
  const auto p = ::vibeqc::generated::method_parameters::gfn2D4();
  return {D4ReferenceModel::gfn2, p.s6,         p.s8, p.s9, p.a1, p.a2, p.cn_cutoff,
          p.pair_cutoff,          p.atm_cutoff, p.ga, p.gc};
}

struct D4Tables {
  D4ReferenceModel reference_model;
  const data::D4ElementData* elements;
  const data::D4ReferenceData* references;
  const double* reference_c6;
  std::size_t element_count;
  std::size_t reference_count;
  std::size_t reference_c6_count;
  double ga;
  double gc;
};

// Host view; a device consumer must explicitly upload each array once at setup.
inline D4Tables gfn2_d4_host_tables() {
  return {D4ReferenceModel::gfn2,
          data::kElements.data(),
          data::kReferences.data(),
          data::kReferenceC6.data(),
          data::kElementCount,
          data::kReferenceCount,
          data::kReferenceC6.size(),
          3.0,
          2.0};
}

VIBEQC_D4_HD inline std::size_t d4_unbounded_workspace_elements(int atoms) {
  if (atoms < 0) return 0u;
  constexpr std::size_t kMaximumSize = static_cast<std::size_t>(-1);
  const std::size_t count = static_cast<std::size_t>(atoms);
  return count <= kMaximumSize / (27u * sizeof(double)) ? 27u * count : 0u;
}

VIBEQC_D4_HD inline std::size_t d4_workspace_elements(int atoms) {
  return atoms >= 0 && atoms <= kD4MaximumAtoms ? d4_unbounded_workspace_elements(atoms) : 0u;
}

namespace d4_detail {
VIBEQC_D4_HD inline bool finite(double x) { return x == x && x <= DBL_MAX && x >= -DBL_MAX; }
struct Range {
  std::uintptr_t begin, end;
};
VIBEQC_D4_HD inline bool range(const void* p, std::size_t bytes, Range& r) {
  const auto a = reinterpret_cast<std::uintptr_t>(p);
  if ((bytes && !p) || a > UINTPTR_MAX - bytes) return false;
  r = {a, a + bytes};
  return true;
}
VIBEQC_D4_HD inline bool overlaps(Range a, Range b) {
  return a.begin < a.end && b.begin < b.end && a.begin < b.end && b.begin < a.end;
}
VIBEQC_D4_HD inline bool valid_parameters(const D4Parameters& p) {
  return finite(p.s6) && finite(p.s8) && finite(p.s9) && finite(p.a1) && finite(p.a2) &&
         finite(p.cn_cutoff) && finite(p.pair_cutoff) && finite(p.atm_cutoff) && p.s6 >= 0.0 &&
         p.s9 >= 0.0 && p.a1 >= 0.0 && p.a2 > 0.0 && p.cn_cutoff > 0.0 && p.pair_cutoff > 0.0 &&
         p.atm_cutoff > 0.0 && finite(p.ga) && finite(p.gc) && p.ga > 0.0 && p.gc > 0.0;
}

// The qmod=0 branch is the continuous saturated limit. The source derivative
// formed 0/0 here; evaluating the limit also avoids a 0*inf near the boundary.
VIBEQC_D4_HD inline void charge_scale(double a, double c, double qref, double qmod, double& value,
                                      double& derivative) {
  const auto result = math::charge_scale(a, c, qref, qmod);
  value = result.value;
  derivative = result.derivative;
}

VIBEQC_D4_HD inline void weights(int n, const std::int32_t* z, const double* cn, const double* q,
                                 const D4Parameters& p, D4Tables t, double* w, double* wc,
                                 double* wq) {
  for (int i = 0; i < n; ++i) {
    math::atom_weights(t.elements[z[i] - 1], t.references, cn[i], q ? q[i] : 0.0, q == nullptr,
                       p.ga, p.gc, w + 7 * i, wc + 7 * i, wq + 7 * i);
  }
}

struct Coefficient {
  double c6, ci, cj, qi, qj;
};
VIBEQC_D4_HD inline Coefficient coefficient(int i, int j, const std::int32_t* z, D4Tables t,
                                            const double* w, const double* wc, const double* wq) {
  const auto shared = math::coefficient(t.elements[z[i] - 1], t.elements[z[j] - 1],
                                        math::PackedReferenceC6{t.reference_c6}, w + 7 * i,
                                        wc + 7 * i, wq + 7 * i, w + 7 * j, wc + 7 * j, wq + 7 * j);
  return {shared.c6, shared.first_cn, shared.second_cn, shared.first_charge, shared.second_charge};
}

VIBEQC_D4_HD inline double distance2(const double* xyz, int i, int j, double* v) {
  double rr = 0.0;
  for (int k = 0; k < 3; ++k) {
    v[k] = xyz[3 * i + k] - xyz[3 * j + k];
    rr += v[k] * v[k];
  }
  return rr;
}
VIBEQC_D4_HD inline double radius(int i, int j, const std::int32_t* z, D4Tables t,
                                  const D4Parameters& p) {
  return math::damping_radius(t.elements[z[i] - 1], t.elements[z[j] - 1], p.a1, p.a2);
}
VIBEQC_D4_HD inline void cn_pair(int i, int j, const std::int32_t* z, D4Tables t, double r,
                                 double& cn, double& dcdr) {
  const auto pair = math::coordination_pair(t.elements[z[i] - 1], t.elements[z[j] - 1], r);
  cn = pair.value;
  dcdr = pair.derivative;
}
VIBEQC_D4_HD inline void add_pair_gradient(int i, int j, const double* v, double scale,
                                           double* grad) {
  for (int a = 0; a < 3; ++a) {
    grad[3 * i + a] += scale * v[a];
    grad[3 * j + a] -= scale * v[a];
  }
}
VIBEQC_D4_HD inline double atm_radial(double x, double y, double z, double r5, double damp,
                                      double angle, double ddamp, double c9) {
  return math::atm_radial(x, y, z, r5, damp, angle, ddamp, c9);
}
}  // namespace d4_detail

// Shared cached-geometry D4 primitives for SCC/runtime consumers that already
// own and validate geometry/CN state. These helpers deliberately do not build
// geometry, pair lists, or CNs: they own only the charge/CN interpolation and
// pair-coefficient science also used by the complete fixed-charge evaluator.
struct D4CachedPairCoefficient {
  double c6 = 0.0;
  double first_cn = 0.0;
  double second_cn = 0.0;
  double first_charge = 0.0;
  double second_charge = 0.0;
};

VIBEQC_D4_HD inline void prepare_d4_cached_weights(int n, const std::int32_t* z, const double* cn,
                                                   const double* q, const D4Parameters& p,
                                                   D4Tables tables, double* weights,
                                                   double* cn_derivatives,
                                                   double* charge_derivatives) {
  d4_detail::weights(n, z, cn, q, p, tables, weights, cn_derivatives, charge_derivatives);
}

VIBEQC_D4_HD inline D4CachedPairCoefficient d4_cached_pair_coefficient(
    int first, int second, const std::int32_t* z, D4Tables tables, const double* weights,
    const double* cn_derivatives, const double* charge_derivatives) {
  const auto coefficient =
      d4_detail::coefficient(first, second, z, tables, weights, cn_derivatives, charge_derivatives);
  return {coefficient.c6, coefficient.ci, coefficient.cj, coefficient.qi, coefficient.qj};
}

// Complete derivative at independent/supplied charges, NOT complete DFT-D4
// forces. grad=dE/dR (Eh/bohr); dq=dE/dq (Eh/e); energy={two-body, zero-q ATM}.
// Add (dq/dR)^T*(dE/dq) through the selected charge provider for total forces.
// Molecular sharp-cutoff compatibility semantics; no PBC/Hessian capabilities.
// Every numeric buffer must be disjoint. Tables must be the immutable pinned
// arrays (or byte-identical device copies). Workspace is disposable; outputs
// are committed only on success. One CPU worker or CUDA lane owns one call.
VIBEQC_D4_HD inline D4Status evaluate_d4_fixed_charge_impl(
    int n, const std::int32_t* z, const double* xyz, const double* q, const D4Parameters& p,
    D4Tables t, double* workspace, std::size_t workspace_size, double* energy, double* grad,
    double* dq, bool enforce_atom_bound) {
  using namespace d4_detail;
  const std::size_t required_workspace = d4_unbounded_workspace_elements(n);
  if ((enforce_atom_bound && n > kD4MaximumAtoms) || p.reference_model != t.reference_model ||
      fabs(p.ga - t.ga) > 1.0e-15 || fabs(p.gc - t.gc) > 1.0e-15)
    return D4Status::unsupported;
  if (n < 0 || (n > 0 && required_workspace == 0u) || !valid_parameters(p) ||
      workspace_size < required_workspace)
    return D4Status::invalid_argument;
  if (t.element_count != data::kElementCount || t.reference_count != data::kReferenceCount ||
      t.reference_c6_count != data::kReferenceCount * (data::kReferenceCount + 1) / 2)
    return D4Status::unsupported;
  const std::size_t count = static_cast<std::size_t>(n);
  const void* ptrs[] = {z,    xyz, q,          workspace,    energy,
                        grad, dq,  t.elements, t.references, t.reference_c6};
  const std::size_t bytes[] = {count * sizeof(*z),
                               3 * count * sizeof(double),
                               count * sizeof(double),
                               required_workspace * sizeof(double),
                               2 * sizeof(double),
                               3 * count * sizeof(double),
                               count * sizeof(double),
                               t.element_count * sizeof(data::D4ElementData),
                               t.reference_count * sizeof(data::D4ReferenceData),
                               t.reference_c6_count * sizeof(double)};
  Range ranges[10];
  for (int a = 0; a < 10; ++a) {
    if (!range(ptrs[a], bytes[a], ranges[a]) ||
        (bytes[a] && reinterpret_cast<std::uintptr_t>(ptrs[a]) %
                         (a == 0 ? alignof(std::int32_t) : alignof(double))))
      return D4Status::invalid_argument;
    for (int b = 0; b < a; ++b)
      if (overlaps(ranges[a], ranges[b])) return D4Status::invalid_argument;
  }
  // Parameters live outside scratch as well: a malicious view cannot overwrite
  // a1/cutoffs in the middle of the calculation.
  Range pr;
  if (!range(&p, sizeof(p), pr)) return D4Status::invalid_argument;
  for (int a = 3; a <= 6; ++a)
    if (overlaps(pr, ranges[a])) return D4Status::invalid_argument;
  for (int i = 0; i < n; ++i) {
    if (z[i] < 1 || z[i] > static_cast<int>(t.element_count)) return D4Status::unsupported;
    if (!finite(q[i])) return D4Status::invalid_argument;
    for (int a = 0; a < 3; ++a)
      if (!finite(xyz[3 * i + a])) return D4Status::invalid_argument;
  }
  if (n == 0) {
    energy[0] = energy[1] = 0.0;
    return D4Status::success;
  }
  double* cn = workspace;
  double* w = cn + n;
  double* wc = w + 7 * n;
  double* wq = wc + 7 * n;
  double* adj = wq + 7 * n;
  double* g = adj + n;
  double* d = g + 3 * n;
  for (std::size_t a = 0; a < required_workspace; ++a) workspace[a] = 0.0;
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      double v[3];
      const double r2 = distance2(xyz, i, j, v);
      if (!finite(r2) || r2 < 1e-12) return D4Status::invalid_argument;
      if (sqrt(r2) <= p.cn_cutoff) {
        double cv, dc;
        cn_pair(i, j, z, t, sqrt(r2), cv, dc);
        cn[i] += cv;
        cn[j] += cv;
      }
    }
  weights(n, z, cn, q, p, t, w, wc, wq);
  double e2 = 0.0, e3 = 0.0;
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      double v[3];
      const double r2 = distance2(xyz, i, j, v);
      if (sqrt(r2) > p.pair_cutoff) continue;
      const auto c = coefficient(i, j, z, t, w, wc, wq);
      const double rr = 3.0 * t.elements[z[i] - 1].r4r2 * t.elements[z[j] - 1].r4r2;
      const double r0 = radius(i, j, z, t, p), u = r0 * r0;
      const double t6 = 1.0 / (r2 * r2 * r2 + u * u * u);
      const double t8 = 1.0 / (r2 * r2 * r2 * r2 + u * u * u * u);
      const double damp = p.s6 * t6 + p.s8 * rr * t8;
      const double ddamp = -6 * p.s6 * r2 * r2 * t6 * t6 - 8 * p.s8 * rr * r2 * r2 * r2 * t8 * t8;
      e2 -= c.c6 * damp;
      d[i] -= c.qi * damp;
      d[j] -= c.qj * damp;
      adj[i] -= c.ci * damp;
      adj[j] -= c.cj * damp;
      add_pair_gradient(i, j, v, -c.c6 * ddamp, g);
    }
  // ATM uses zero-charge reference weights, independently of input charges.
  if (p.s9 != 0.0) {
    weights(n, z, cn, nullptr, p, t, w, wc, wq);
    for (int i = 2; i < n; ++i)
      for (int j = 1; j < i; ++j) {
        double vij[3];
        const double a = distance2(xyz, i, j, vij);
        if (sqrt(a) > p.atm_cutoff) continue;
        const auto cij = coefficient(i, j, z, t, w, wc, wq);
        for (int k = 0; k < j; ++k) {
          double vik[3], vjk[3];
          const double b = distance2(xyz, i, k, vik), c = distance2(xyz, j, k, vjk);
          if (sqrt(b) > p.atm_cutoff || sqrt(c) > p.atm_cutoff) continue;
          const auto cik = coefficient(i, k, z, t, w, wc, wq);
          const auto cjk = coefficient(j, k, z, t, w, wc, wq);
          if (!(cij.c6 > 0.0 && cik.c6 > 0.0 && cjk.c6 > 0.0)) return D4Status::numerical_failure;
          const double r2p = a * b * c, r1p = sqrt(r2p), r3p = r2p * r1p, r5p = r3p * r2p;
          const double ratio =
              radius(i, j, z, t, p) * radius(i, k, z, t, p) * radius(j, k, z, t, p) / r1p;
          const double rp = pow(ratio, 16.0 / 3.0);
          const double damp = 1.0 / (1.0 + 6.0 * rp);
          const double angle = 0.375 * (a + c - b) * (a - c + b) * (-a + c + b) / r5p + 1.0 / r3p;
          const double c9 = -p.s9 * sqrt(cij.c6 * cik.c6 * cjk.c6);
          const double de = angle * damp * c9, ddamp = -32.0 * rp * damp * damp;
          e3 -= de;
          add_pair_gradient(i, j, vij, atm_radial(a, c, b, r5p, damp, angle, ddamp, c9), g);
          add_pair_gradient(i, k, vik, atm_radial(b, c, a, r5p, damp, angle, ddamp, c9), g);
          add_pair_gradient(j, k, vjk, atm_radial(c, b, a, r5p, damp, angle, ddamp, c9), g);
          adj[i] -= 0.5 * de * (cij.ci / cij.c6 + cik.ci / cik.c6);
          adj[j] -= 0.5 * de * (cij.cj / cij.c6 + cjk.ci / cjk.c6);
          adj[k] -= 0.5 * de * (cik.cj / cik.c6 + cjk.cj / cjk.c6);
        }
      }
  }
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      double v[3];
      const double r = sqrt(distance2(xyz, i, j, v));
      if (r > p.cn_cutoff) continue;
      double cv, dc;
      cn_pair(i, j, z, t, r, cv, dc);
      add_pair_gradient(i, j, v, dc * (adj[i] + adj[j]) / r, g);
    }
  if (!finite(e2) || !finite(e3)) return D4Status::numerical_failure;
  for (int i = 0; i < 3 * n; ++i)
    if (!finite(g[i])) return D4Status::numerical_failure;
  for (int i = 0; i < n; ++i)
    if (!finite(d[i])) return D4Status::numerical_failure;
  energy[0] = e2;
  energy[1] = e3;
  for (int i = 0; i < 3 * n; ++i) grad[i] = g[i];
  for (int i = 0; i < n; ++i) dq[i] = d[i];
  return D4Status::success;
}

VIBEQC_D4_HD inline D4Status evaluate_d4_fixed_charge(int n, const std::int32_t* z,
                                                      const double* xyz, const double* q,
                                                      const D4Parameters& p, D4Tables t,
                                                      double* workspace, std::size_t workspace_size,
                                                      double* energy, double* grad, double* dq) {
  return evaluate_d4_fixed_charge_impl(n, z, xyz, q, p, t, workspace, workspace_size, energy, grad,
                                       dq, true);
}

// Internal CPU owner used by GFN2 after its larger-system runtime has already
// supplied and validated storage. Public D4 and every CUDA schedule remain
// bounded by kD4MaximumAtoms through evaluate_d4_fixed_charge above.
inline D4Status evaluate_d4_fixed_charge_unbounded_cpu(int n, const std::int32_t* z,
                                                       const double* xyz, const double* q,
                                                       const D4Parameters& p, D4Tables t,
                                                       double* workspace,
                                                       std::size_t workspace_size, double* energy,
                                                       double* grad, double* dq) {
  return evaluate_d4_fixed_charge_impl(n, z, xyz, q, p, t, workspace, workspace_size, energy, grad,
                                       dq, false);
}

}  // namespace vibeqc::dft::dispersion
#undef VIBEQC_D4_HD

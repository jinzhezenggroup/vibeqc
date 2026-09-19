#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dft/dispersion/d4_eeq_data.hpp"
#include "dft/dispersion/d4_eeq_r2scan3c_c6.hpp"
#include "dft/dispersion/d4_reference.hpp"

#if defined(__CUDACC__)
#define VIBEQC_D4_EEQ_HD __host__ __device__
#else
#define VIBEQC_D4_EEQ_HD
#endif

namespace vibeqc::dft::dispersion {

enum class D4EEQProfile : int { standard = 1, r2scan3c = 2 };

struct EEQTables {
  const data::D4ElementData* elements;
  const eeq_data::EEQChargeElementData* charge_elements;
  std::size_t element_count;
};

inline EEQTables eeq2019_host_tables() {
  return {eeq_data::kElements.data(), eeq_data::kChargeElements.data(), eeq_data::kElementCount};
}

inline D4Tables eeq_d4_host_tables(D4EEQProfile profile) {
  const double* c6 = profile == D4EEQProfile::r2scan3c ? eeq_data::kReferenceC6R2SCAN3C.data()
                                                       : eeq_data::kReferenceC6Standard.data();
  const bool r2scan = profile == D4EEQProfile::r2scan3c;
  return {D4ReferenceModel::eeq,
          eeq_data::kElements.data(),
          eeq_data::kReferences.data(),
          c6,
          eeq_data::kElementCount,
          eeq_data::kReferenceCount,
          eeq_data::kReferenceC6Standard.size(),
          r2scan ? 2.0 : 3.0,
          r2scan ? 1.0 : 2.0};
}

inline D4Parameters r2scan3c_d4_parameters() {
  // Pinned DFT-D4 4.2.0: s6=1, s8=0, s9=2, a1=.42, a2=5.65,
  // plus the method-specific zeta profile ga=2/gc=1.
  return {D4ReferenceModel::eeq, 1.0, 0.0, 2.0, 0.42, 5.65, 30.0, 60.0, 40.0, 2.0, 1.0};
}

inline constexpr int kEEQMaximumAtoms = kD4MaximumAtoms;
inline constexpr double kEEQCutoff = 25.0;
inline constexpr double kEEQKcn = 7.5;
inline constexpr double kEEQMaximumCN = 8.0;
inline constexpr double kEEQRegularizer = 1.0e-14;
inline constexpr double kInvSqrtPi = 0.5641895835477562869480794515607726;
inline constexpr double kSqrtTwoOverPi = 0.7978845608028653558798921198687637;

VIBEQC_D4_EEQ_HD inline std::size_t eeq2019_workspace_elements(int atoms) {
  if (atoms < 0 || atoms > kEEQMaximumAtoms) return 0;
  const std::size_t n = static_cast<std::size_t>(atoms);
  const std::size_t m = n + 1;
  // raw CN + d(log-cut CN)/d(raw CN) + LU(A) + solution + all response RHS columns.
  return 2 * n + m * m + m + 3 * n * m;
}

namespace eeq_detail {

VIBEQC_D4_EEQ_HD inline double log_cn_cut(double cn) {
  return log1p(exp(kEEQMaximumCN)) - log1p(exp(kEEQMaximumCN - cn));
}

VIBEQC_D4_EEQ_HD inline double dlog_cn_cut(double cn) {
  return exp(kEEQMaximumCN) / (exp(kEEQMaximumCN) + exp(cn));
}

VIBEQC_D4_EEQ_HD inline bool finite(double value) {
  return (value == value && value <= DBL_MAX && value >= -DBL_MAX);
}

VIBEQC_D4_EEQ_HD inline bool lu_factor(double* a, int n, int* pivots) {
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    double maximum = fabs(a[static_cast<std::size_t>(k) * n + k]);
    for (int i = k + 1; i < n; ++i) {
      const double value = fabs(a[static_cast<std::size_t>(i) * n + k]);
      if (value > maximum) {
        maximum = value;
        pivot = i;
      }
    }
    if (!(maximum > 1.0e-15) || !finite(maximum)) return false;
    pivots[k] = pivot;
    if (pivot != k)
      for (int j = 0; j < n; ++j) {
        const auto p = static_cast<std::size_t>(pivot) * n + j;
        const auto q = static_cast<std::size_t>(k) * n + j;
        const double tmp = a[p];
        a[p] = a[q];
        a[q] = tmp;
      }
    const double diagonal = a[static_cast<std::size_t>(k) * n + k];
    for (int i = k + 1; i < n; ++i) {
      auto ik = static_cast<std::size_t>(i) * n + k;
      a[ik] /= diagonal;
      const double factor = a[ik];
      for (int j = k + 1; j < n; ++j)
        a[static_cast<std::size_t>(i) * n + j] -= factor * a[static_cast<std::size_t>(k) * n + j];
    }
  }
  return true;
}

VIBEQC_D4_EEQ_HD inline bool lu_solve(const double* lu, int n, const int* pivots, double* rhs) {
  for (int k = 0; k < n; ++k)
    if (pivots[k] != k) {
      const double tmp = rhs[k];
      rhs[k] = rhs[pivots[k]];
      rhs[pivots[k]] = tmp;
    }
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j) rhs[i] -= lu[static_cast<std::size_t>(i) * n + j] * rhs[j];
  for (int i = n - 1; i >= 0; --i) {
    for (int j = i + 1; j < n; ++j) rhs[i] -= lu[static_cast<std::size_t>(i) * n + j] * rhs[j];
    const double diagonal = lu[static_cast<std::size_t>(i) * n + i];
    if (!(fabs(diagonal) > 1.0e-15) || !finite(diagonal)) return false;
    rhs[i] /= diagonal;
    if (!finite(rhs[i])) return false;
  }
  return true;
}

VIBEQC_D4_EEQ_HD inline void eeq_cn_pair(const data::D4ElementData& a, const data::D4ElementData& b,
                                         double r, double& count, double& dcountdr) {
  const double rc = a.covalent_radius + b.covalent_radius;
  const double x = kEEQKcn * (r - rc) / rc;
  count = 0.5 * (1.0 + erf(-x));
  dcountdr = -kEEQKcn * exp(-x * x) * kInvSqrtPi / rc;
}

}  // namespace eeq_detail

// Molecular, nonperiodic EEQ2019 charge provider from pinned multicharge.
// dqdr layout is [coordinate=(3*atom+axis)][charge_atom], i.e.
// dqdr[(3*k+axis)*n+i] = dq_i / dR_{k,axis}.
VIBEQC_D4_EEQ_HD inline D4Status evaluate_eeq2019_with_tables(
    int n, const std::int32_t* z, const double* xyz, double total_charge, EEQTables t,
    double* workspace, std::size_t workspace_size, double* charges, double* dqdr) {
  using namespace eeq_detail;
  if (n < 0 || n > kEEQMaximumAtoms) return D4Status::unsupported;
  if (!finite(total_charge) || workspace_size < eeq2019_workspace_elements(n))
    return D4Status::invalid_argument;
  if (n == 0) return total_charge == 0.0 ? D4Status::success : D4Status::invalid_argument;
  if (!z || !xyz || !workspace || !charges || !dqdr) return D4Status::invalid_argument;
  if (!t.elements || !t.charge_elements || t.element_count != eeq_data::kElementCount)
    return D4Status::unsupported;
  const std::size_t count = static_cast<std::size_t>(n);
  const std::size_t used_workspace = eeq2019_workspace_elements(n);
  const void* ptrs[] = {z, xyz, workspace, charges, dqdr, t.elements, t.charge_elements};
  const std::size_t bytes[] = {
      count * sizeof(*z),
      3 * count * sizeof(double),
      used_workspace * sizeof(double),
      count * sizeof(double),
      3 * count * count * sizeof(double),
      t.element_count * sizeof(data::D4ElementData),
      t.element_count * sizeof(eeq_data::EEQChargeElementData),
  };
  d4_detail::Range ranges[7];
  for (int a = 0; a < 7; ++a) {
    if (!d4_detail::range(ptrs[a], bytes[a], ranges[a]) ||
        (bytes[a] && reinterpret_cast<std::uintptr_t>(ptrs[a]) %
                         (a == 0 ? alignof(std::int32_t) : alignof(double))))
      return D4Status::invalid_argument;
    for (int b = 0; b < a; ++b)
      if (d4_detail::overlaps(ranges[a], ranges[b])) return D4Status::invalid_argument;
  }
  for (int i = 0; i < n; ++i) {
    if (z[i] < 1 || z[i] > static_cast<int>(t.element_count)) return D4Status::unsupported;
    for (int a = 0; a < 3; ++a)
      if (!finite(xyz[3 * i + a])) return D4Status::invalid_argument;
  }

  const int m = n + 1;
  double* raw_cn = workspace;
  double* cut_factor = raw_cn + n;
  double* lu = cut_factor + n;
  double* solution = lu + static_cast<std::size_t>(m) * m;
  double* responses = solution + m;  // 3*n columns, each length m.
  for (std::size_t i = 0; i < eeq2019_workspace_elements(n); ++i) workspace[i] = 0.0;

  // First pass: raw EEQ coordination numbers.
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      double v[3];
      const double r2 = d4_detail::distance2(xyz, i, j, v);
      if (!finite(r2) || r2 < 1.0e-12) return D4Status::invalid_argument;
      const double r = sqrt(r2);
      if (r > kEEQCutoff) continue;
      double count, dcountdr;
      eeq_cn_pair(t.elements[z[i] - 1], t.elements[z[j] - 1], r, count, dcountdr);
      raw_cn[i] += count;
      raw_cn[j] += count;
    }

  // Build A and x after the same CN saturation used by multicharge.
  for (int i = 0; i < n; ++i) {
    const auto qpar = t.charge_elements[z[i] - 1];
    cut_factor[i] = dlog_cn_cut(raw_cn[i]);
    const double cn = log_cn_cut(raw_cn[i]);
    solution[i] = -qpar.chi + qpar.kcnchi * cn / sqrt(cn + kEEQRegularizer);
    lu[static_cast<std::size_t>(i) * m + i] = qpar.eta + kSqrtTwoOverPi / qpar.radius;
    lu[static_cast<std::size_t>(i) * m + n] = 1.0;
    lu[static_cast<std::size_t>(n) * m + i] = 1.0;
  }
  solution[n] = total_charge;

  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      double v[3];
      const double r2 = d4_detail::distance2(xyz, i, j, v);
      const double r = sqrt(r2);
      const auto qi = t.charge_elements[z[i] - 1];
      const auto qj = t.charge_elements[z[j] - 1];
      const double gamma = 1.0 / sqrt(qi.radius * qi.radius + qj.radius * qj.radius);
      const double value = erf(gamma * r) / r;
      lu[static_cast<std::size_t>(i) * m + j] = value;
      lu[static_cast<std::size_t>(j) * m + i] = value;
    }

  int pivots[kEEQMaximumAtoms + 1]{};
  if (!lu_factor(lu, m, pivots) || !lu_solve(lu, m, pivots, solution))
    return D4Status::numerical_failure;

  // Build all right-hand sides dx/dR - (dA/dR)y before solving them.
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      double v[3];
      const double r2 = d4_detail::distance2(xyz, i, j, v);
      const double r = sqrt(r2);
      const auto ei = t.elements[z[i] - 1];
      const auto ej = t.elements[z[j] - 1];
      const auto qi = t.charge_elements[z[i] - 1];
      const auto qj = t.charge_elements[z[j] - 1];

      // CN contribution to dx/dR.
      if (r <= kEEQCutoff) {
        double count, dcountdr;
        eeq_cn_pair(ei, ej, r, count, dcountdr);
        const double cni = log_cn_cut(raw_cn[i]);
        const double cnj = log_cn_cut(raw_cn[j]);
        const double xi = 0.5 * qi.kcnchi / sqrt(cni + kEEQRegularizer) * cut_factor[i];
        const double xj = 0.5 * qj.kcnchi / sqrt(cnj + kEEQRegularizer) * cut_factor[j];
        for (int axis = 0; axis < 3; ++axis) {
          const double g = dcountdr * v[axis] / r;
          const int ci = 3 * i + axis, cj = 3 * j + axis;
          responses[static_cast<std::size_t>(ci) * m + i] += xi * g;
          responses[static_cast<std::size_t>(ci) * m + j] += xj * g;
          responses[static_cast<std::size_t>(cj) * m + i] -= xi * g;
          responses[static_cast<std::size_t>(cj) * m + j] -= xj * g;
        }
      }

      // Coulomb-matrix contribution -(dA/dR)y.
      const double gamma = 1.0 / sqrt(qi.radius * qi.radius + qj.radius * qj.radius);
      const double arg = gamma * gamma * r2;
      const double dadr = 2.0 * gamma * exp(-arg) * kInvSqrtPi / r - erf(sqrt(arg)) / r2;
      for (int axis = 0; axis < 3; ++axis) {
        const double g = dadr * v[axis] / r;
        const int ci = 3 * i + axis, cj = 3 * j + axis;
        responses[static_cast<std::size_t>(ci) * m + i] -= g * solution[j];
        responses[static_cast<std::size_t>(ci) * m + j] -= g * solution[i];
        responses[static_cast<std::size_t>(cj) * m + i] += g * solution[j];
        responses[static_cast<std::size_t>(cj) * m + j] += g * solution[i];
      }
    }

  for (int coordinate = 0; coordinate < 3 * n; ++coordinate) {
    double* rhs = responses + static_cast<std::size_t>(coordinate) * m;
    if (!lu_solve(lu, m, pivots, rhs)) return D4Status::numerical_failure;
    for (int i = 0; i < n; ++i)
      if (!finite(rhs[i])) return D4Status::numerical_failure;
  }
  // Transactional publication: outputs are untouched until every response solve succeeds.
  for (int i = 0; i < n; ++i) charges[i] = solution[i];
  for (int coordinate = 0; coordinate < 3 * n; ++coordinate)
    for (int i = 0; i < n; ++i)
      dqdr[static_cast<std::size_t>(coordinate) * n + i] =
          responses[static_cast<std::size_t>(coordinate) * m + i];
  return D4Status::success;
}

inline D4Status evaluate_eeq2019(int n, const std::int32_t* z, const double* xyz,
                                 double total_charge, double* workspace, std::size_t workspace_size,
                                 double* charges, double* dqdr) {
  return evaluate_eeq2019_with_tables(n, z, xyz, total_charge, eeq2019_host_tables(), workspace,
                                      workspace_size, charges, dqdr);
}

VIBEQC_D4_EEQ_HD inline std::size_t complete_d4_eeq_workspace_elements(int atoms) {
  if (atoms < 0 || atoms > kD4MaximumAtoms) return 0;
  const std::size_t n = static_cast<std::size_t>(atoms);
  return eeq2019_workspace_elements(atoms) + d4_workspace_elements(atoms) + n + 3 * n * n + n +
         3 * n;
}

// Complete molecular D4 EEQ energy/gradient. The caller selects a table profile
// matching p.ga/p.gc; profile mismatches are rejected rather than silently mixed.
VIBEQC_D4_EEQ_HD inline D4Status evaluate_complete_d4_eeq_with_tables(
    int n, const std::int32_t* z, const double* xyz, double total_charge, const D4Parameters& p,
    D4EEQProfile profile, D4Tables d4_tables, EEQTables eeq_tables, double* workspace,
    std::size_t workspace_size, double* energy, double* gradient, double* charges) {
  if (n < 0 || n > kD4MaximumAtoms) return D4Status::unsupported;
  if (workspace_size < complete_d4_eeq_workspace_elements(n) || !workspace || !energy ||
      !gradient || (n && (!charges || !z || !xyz)))
    return D4Status::invalid_argument;
  if (profile != D4EEQProfile::standard && profile != D4EEQProfile::r2scan3c)
    return D4Status::unsupported;
  const bool r2scan = profile == D4EEQProfile::r2scan3c;
  if (p.reference_model != D4ReferenceModel::eeq || fabs(p.ga - (r2scan ? 2.0 : 3.0)) > 1.0e-15 ||
      fabs(p.gc - (r2scan ? 1.0 : 2.0)) > 1.0e-15)
    return D4Status::unsupported;

  const std::size_t count = static_cast<std::size_t>(n);
  const std::size_t used_workspace = complete_d4_eeq_workspace_elements(n);
  const void* mutable_ptrs[] = {z, xyz, workspace, energy, gradient, charges};
  const std::size_t mutable_bytes[] = {
      count * sizeof(*z), 3 * count * sizeof(double), used_workspace * sizeof(double),
      2 * sizeof(double), 3 * count * sizeof(double), count * sizeof(double)};
  d4_detail::Range mutable_ranges[6];
  for (int a = 0; a < 6; ++a) {
    if (!d4_detail::range(mutable_ptrs[a], mutable_bytes[a], mutable_ranges[a]) ||
        (mutable_bytes[a] && reinterpret_cast<std::uintptr_t>(mutable_ptrs[a]) %
                                 (a == 0 ? alignof(std::int32_t) : alignof(double))))
      return D4Status::invalid_argument;
    for (int b = 0; b < a; ++b)
      if (d4_detail::overlaps(mutable_ranges[a], mutable_ranges[b]))
        return D4Status::invalid_argument;
  }
  const void* table_ptrs[] = {d4_tables.elements, d4_tables.references, d4_tables.reference_c6,
                              eeq_tables.elements, eeq_tables.charge_elements};
  const std::size_t table_bytes[] = {
      d4_tables.element_count * sizeof(data::D4ElementData),
      d4_tables.reference_count * sizeof(data::D4ReferenceData),
      d4_tables.reference_c6_count * sizeof(double),
      eeq_tables.element_count * sizeof(data::D4ElementData),
      eeq_tables.element_count * sizeof(eeq_data::EEQChargeElementData)};
  for (int a = 0; a < 5; ++a) {
    d4_detail::Range table_range{};
    if (!d4_detail::range(table_ptrs[a], table_bytes[a], table_range) ||
        (table_bytes[a] && reinterpret_cast<std::uintptr_t>(table_ptrs[a]) % alignof(double)))
      return D4Status::invalid_argument;
    for (const auto& mutable_range : mutable_ranges)
      if (d4_detail::overlaps(table_range, mutable_range)) return D4Status::invalid_argument;
  }
  d4_detail::Range parameter_range{};
  if (!d4_detail::range(&p, sizeof(p), parameter_range)) return D4Status::invalid_argument;
  for (const auto& mutable_range : mutable_ranges)
    if (d4_detail::overlaps(parameter_range, mutable_range)) return D4Status::invalid_argument;

  double* eeq_workspace = workspace;
  double* d4_workspace = eeq_workspace + eeq2019_workspace_elements(n);
  double* q = d4_workspace + d4_workspace_elements(n);
  double* dqdr = q + n;
  double* dedq = dqdr + static_cast<std::size_t>(3) * n * n;
  double* g = dedq + n;

  auto status = evaluate_eeq2019_with_tables(n, z, xyz, total_charge, eeq_tables, eeq_workspace,
                                             eeq2019_workspace_elements(n), q, dqdr);
  if (status != D4Status::success) return status;
  double e[2] = {0.0, 0.0};
  status = evaluate_d4_fixed_charge(n, z, xyz, q, p, d4_tables, d4_workspace,
                                    d4_workspace_elements(n), e, g, dedq);
  if (status != D4Status::success) return status;
  for (int coordinate = 0; coordinate < 3 * n; ++coordinate)
    for (int i = 0; i < n; ++i)
      g[coordinate] += dqdr[static_cast<std::size_t>(coordinate) * n + i] * dedq[i];
  for (int coordinate = 0; coordinate < 3 * n; ++coordinate)
    if (!eeq_detail::finite(g[coordinate])) return D4Status::numerical_failure;
  energy[0] = e[0];
  energy[1] = e[1];
  for (int coordinate = 0; coordinate < 3 * n; ++coordinate) gradient[coordinate] = g[coordinate];
  for (int i = 0; i < n; ++i) charges[i] = q[i];
  return D4Status::success;
}

inline D4Status evaluate_complete_d4_eeq(int n, const std::int32_t* z, const double* xyz,
                                         double total_charge, const D4Parameters& p,
                                         D4EEQProfile profile, double* workspace,
                                         std::size_t workspace_size, double* energy,
                                         double* gradient, double* charges) {
  return evaluate_complete_d4_eeq_with_tables(n, z, xyz, total_charge, p, profile,
                                              eeq_d4_host_tables(profile), eeq2019_host_tables(),
                                              workspace, workspace_size, energy, gradient, charges);
}

}  // namespace vibeqc::dft::dispersion
#undef VIBEQC_D4_EEQ_HD

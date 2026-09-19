#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "d4_eeq_oracle_fixtures.hpp"

using namespace d4_eeq_tests;

namespace {
bool near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

int oracle_cases() {
  for (const auto& f : kEEQOracleFixtures) {
    std::vector<double> workspace(complete_d4_eeq_workspace_elements(f.atoms));
    std::vector<double> gradient(3 * f.atoms), charges(f.atoms);
    double energy[2]{};
    const auto status = evaluate_complete_d4_eeq(
        f.atoms, f.z.data(), f.xyz.data(), f.total_charge, f.parameters, f.profile,
        workspace.data(), workspace.size(), energy, gradient.data(), charges.data());
    if (status != D4Status::success) {
      std::fprintf(stderr, "%s: status=%d\n", f.name, static_cast<int>(status));
      return 1;
    }
    if (!near(energy[0] + energy[1], f.energy, 2e-13)) {
      std::fprintf(stderr, "%s: energy %.17g oracle %.17g\n", f.name, energy[0] + energy[1],
                   f.energy);
      return 2;
    }
    double charge_sum = 0.0;
    for (int i = 0; i < f.atoms; ++i) {
      charge_sum += charges[i];
      if (!near(charges[i], f.charges[i], 1e-10)) {
        std::fprintf(stderr, "%s: charge[%d] %.17g oracle %.17g\n", f.name, i, charges[i],
                     f.charges[i]);
        return 3;
      }
    }
    if (!near(charge_sum, f.total_charge, 2e-13)) return 4;
    for (int i = 0; i < 3 * f.atoms; ++i)
      if (!near(gradient[i], f.gradient[i], 2e-12)) {
        std::fprintf(stderr, "%s: gradient[%d] %.17g oracle %.17g\n", f.name, i, gradient[i],
                     f.gradient[i]);
        return 5;
      }
  }
  return 0;
}

int response_finite_differences() {
  const auto& f = kEEQOracleFixtures.front();
  const int n = f.atoms;
  std::vector<double> workspace(eeq2019_workspace_elements(n));
  std::vector<double> charges(n), dqdr(3 * n * n);
  if (evaluate_eeq2019(n, f.z.data(), f.xyz.data(), f.total_charge, workspace.data(),
                       workspace.size(), charges.data(), dqdr.data()) != D4Status::success)
    return 10;
  for (int c = 0; c < 3 * n; ++c) {
    double response_sum = 0.0;
    for (int i = 0; i < n; ++i) response_sum += dqdr[static_cast<std::size_t>(c) * n + i];
    if (std::fabs(response_sum) > 2e-13) return 11;
  }

  for (double h : {1e-4, 2e-5}) {
    double maximum = 0.0;
    for (int c = 0; c < 3 * n; ++c) {
      auto plus = f.xyz;
      auto minus = f.xyz;
      plus[c] += h;
      minus[c] -= h;
      std::vector<double> wp(workspace.size()), wm(workspace.size());
      std::vector<double> qp(n), qm(n), dp(dqdr.size()), dm(dqdr.size());
      if (evaluate_eeq2019(n, f.z.data(), plus.data(), f.total_charge, wp.data(), wp.size(),
                           qp.data(), dp.data()) != D4Status::success ||
          evaluate_eeq2019(n, f.z.data(), minus.data(), f.total_charge, wm.data(), wm.size(),
                           qm.data(), dm.data()) != D4Status::success)
        return 12;
      for (int i = 0; i < n; ++i) {
        const double numerical = (qp[i] - qm[i]) / (2 * h);
        maximum =
            std::max(maximum, std::fabs(numerical - dqdr[static_cast<std::size_t>(c) * n + i]));
      }
    }
    if (maximum > 2e-8) {
      std::fprintf(stderr, "dq/dR finite-difference error at h=%g: %.3e\n", h, maximum);
      return 13;
    }
  }
  return 0;
}

int complete_gradient_finite_difference() {
  const auto& f = kEEQOracleFixtures.front();
  const int n = f.atoms;
  std::vector<double> workspace(complete_d4_eeq_workspace_elements(n));
  std::vector<double> gradient(3 * n), charges(n);
  double energy[2]{};
  if (evaluate_complete_d4_eeq(n, f.z.data(), f.xyz.data(), f.total_charge, f.parameters, f.profile,
                               workspace.data(), workspace.size(), energy, gradient.data(),
                               charges.data()) != D4Status::success)
    return 20;
  for (double h : {1e-4, 2e-5}) {
    double maximum = 0.0;
    for (int c = 0; c < 3 * n; ++c) {
      auto plus = f.xyz;
      auto minus = f.xyz;
      plus[c] += h;
      minus[c] -= h;
      std::vector<double> wp(workspace.size()), wm(workspace.size());
      std::vector<double> gp(3 * n), gm(3 * n), qp(n), qm(n);
      double ep[2]{}, em[2]{};
      if (evaluate_complete_d4_eeq(n, f.z.data(), plus.data(), f.total_charge, f.parameters,
                                   f.profile, wp.data(), wp.size(), ep, gp.data(),
                                   qp.data()) != D4Status::success ||
          evaluate_complete_d4_eeq(n, f.z.data(), minus.data(), f.total_charge, f.parameters,
                                   f.profile, wm.data(), wm.size(), em, gm.data(),
                                   qm.data()) != D4Status::success)
        return 21;
      const double numerical = ((ep[0] + ep[1]) - (em[0] + em[1])) / (2 * h);
      maximum = std::max(maximum, std::fabs(numerical - gradient[c]));
    }
    if (maximum > 2e-9) {
      std::fprintf(stderr, "complete D4 gradient finite-difference error at h=%g: %.3e\n", h,
                   maximum);
      return 22;
    }
  }
  return 0;
}

int negative_cases() {
  const auto& f = kEEQOracleFixtures.front();
  std::vector<double> workspace(complete_d4_eeq_workspace_elements(f.atoms));
  std::vector<double> gradient(3 * f.atoms, 17.0), charges(f.atoms, 19.0);
  double energy[2]{23.0, 29.0};
  auto wrong = f.parameters;
  wrong.ga = 3.0;
  wrong.gc = 2.0;
  const auto status = evaluate_complete_d4_eeq(
      f.atoms, f.z.data(), f.xyz.data(), f.total_charge, wrong, D4EEQProfile::r2scan3c,
      workspace.data(), workspace.size(), energy, gradient.data(), charges.data());
  if (status != D4Status::unsupported) return 30;
  if (energy[0] != 23.0 || energy[1] != 29.0 ||
      !std::all_of(gradient.begin(), gradient.end(), [](double x) { return x == 17.0; }) ||
      !std::all_of(charges.begin(), charges.end(), [](double x) { return x == 19.0; }))
    return 31;

  const auto table_status = evaluate_complete_d4_eeq_with_tables(
      f.atoms, f.z.data(), f.xyz.data(), f.total_charge, f.parameters, D4EEQProfile::r2scan3c,
      eeq_d4_host_tables(D4EEQProfile::standard), eeq2019_host_tables(), workspace.data(),
      workspace.size(), energy, gradient.data(), charges.data());
  if (table_status != D4Status::unsupported) {
    std::fprintf(stderr, "table-profile mismatch status=%d\n", static_cast<int>(table_status));
    return 32;
  }
  if (energy[0] != 23.0 || energy[1] != 29.0 ||
      !std::all_of(gradient.begin(), gradient.end(), [](double x) { return x == 17.0; }) ||
      !std::all_of(charges.begin(), charges.end(), [](double x) { return x == 19.0; }))
    return 33;
  return 0;
}
}  // namespace

int main() {
  if (const int rc = oracle_cases()) return rc;
  if (const int rc = response_finite_differences()) return rc;
  if (const int rc = complete_gradient_finite_difference()) return rc;
  if (const int rc = negative_cases()) return rc;
  std::puts("D4 EEQ oracle/response/complete-gradient tests passed");
  return 0;
}

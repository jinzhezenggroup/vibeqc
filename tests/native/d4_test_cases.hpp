#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "d4_oracle_fixtures.hpp"
#include "dft/dispersion/d4_reference.hpp"

namespace d4_tests {
using namespace vibeqc::dft::dispersion;
struct Molecule {
  std::vector<std::int32_t> z;
  std::vector<double> xyz, q;
};
struct Result {
  D4Status status = D4Status::invalid_argument;
  std::array<double, 2> energy{17.0, 19.0};
  std::vector<double> gradient, dq;
  explicit Result(int n) : gradient(3 * n, 23.0), dq(n, 29.0) {}
  double total() const { return energy[0] + energy[1]; }
};
inline Molecule fixture() {
  return {{8, 1, 1, 6},
          {0, 0, 0, 1.43, 0, 1.1, -1.43, 0, 1.1, 2.7, 1.2, -0.4},
          {-0.6, 0.25, 0.25, 0.1}};
}
inline bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }
#define D4_CHECK(x)                                                   \
  do {                                                                \
    if (!(x)) {                                                       \
      std::fprintf(stderr, "D4 failure line %d: %s\n", __LINE__, #x); \
      return 1;                                                       \
    }                                                                 \
  } while (false)

template <class Eval>
int run_cases(Eval& evaluate) {
  double max_energy = 0.0, max_gradient = 0.0, max_charge = 0.0;
  for (const auto& f : kOracleFixtures) {
    Molecule input{{f.z.begin(), f.z.begin() + f.n},
                   {f.xyz.begin(), f.xyz.begin() + 3 * f.n},
                   {f.q.begin(), f.q.begin() + f.n}};
    const auto r = evaluate(input, f.parameters);
    D4_CHECK(r.status == D4Status::success);
    for (int i = 0; i < 2; ++i) {
      max_energy = std::max(max_energy, std::abs(r.energy[i] - f.energy[i]));
      D4_CHECK(near(r.energy[i], f.energy[i], 1e-14));
    }
    for (int i = 0; i < 3 * f.n; ++i) {
      max_gradient = std::max(max_gradient, std::abs(r.gradient[i] - f.gradient[i]));
      D4_CHECK(near(r.gradient[i], f.gradient[i], 2e-12));
    }
    for (int i = 0; i < f.n; ++i) {
      max_charge = std::max(max_charge, std::abs(r.dq[i] - f.dq[i]));
      D4_CHECK(near(r.dq[i], f.dq[i], 2e-12));
    }
  }
  std::printf("Independent dftd4 max errors: energy=%.3e gradient=%.3e charge=%.3e\n", max_energy,
              max_gradient, max_charge);
  auto m = fixture();
  auto p = gfn2_d4_parameters();
  const auto a = evaluate(m, p);
  D4_CHECK(a.status == D4Status::success);
  // Historical migration fixture; independent Fortran oracle lives separately.
  D4_CHECK(near(a.energy[0], -0.0005923540861122829, 2e-15));
  D4_CHECK(std::isfinite(a.energy[1]));
  for (double h : {1e-4, 1e-5, 3e-6}) {
    for (std::size_t k = 0; k < m.xyz.size(); ++k) {
      auto plus = m, minus = m;
      plus.xyz[k] += h;
      minus.xyz[k] -= h;
      const auto ep = evaluate(plus, p), em = evaluate(minus, p);
      D4_CHECK(ep.status == D4Status::success && em.status == D4Status::success);
      D4_CHECK(near((ep.total() - em.total()) / (2 * h), a.gradient[k], 3e-9));
    }
    for (std::size_t k = 0; k < m.q.size(); ++k) {
      auto plus = m, minus = m;
      plus.q[k] += h;
      minus.q[k] -= h;
      const auto ep = evaluate(plus, p), em = evaluate(minus, p);
      D4_CHECK(ep.status == D4Status::success && em.status == D4Status::success);
      D4_CHECK(near((ep.total() - em.total()) / (2 * h), a.dq[k], 2e-10));
      D4_CHECK(ep.energy[1] == em.energy[1]);
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    double sum = 0;
    for (std::size_t k = axis; k < a.gradient.size(); k += 3) sum += a.gradient[k];
    D4_CHECK(near(sum, 0, 2e-15));
  }
  auto translated = m;
  for (std::size_t k = 0; k < m.xyz.size(); ++k) translated.xyz[k] += 11.0 * (k % 3 + 1);
  const auto tr = evaluate(translated, p);
  D4_CHECK(tr.status == D4Status::success && near(tr.total(), a.total(), 2e-15));
  for (std::size_t k = 0; k < a.gradient.size(); ++k)
    D4_CHECK(near(a.gradient[k], tr.gradient[k], 2e-14));
  auto perm = m;
  for (int k = 0; k < 4; ++k) {
    perm.z[k] = m.z[3 - k];
    perm.q[k] = m.q[3 - k];
    for (int x = 0; x < 3; ++x) perm.xyz[3 * k + x] = m.xyz[3 * (3 - k) + x];
  }
  const auto pe = evaluate(perm, p);
  D4_CHECK(pe.status == D4Status::success && near(pe.total(), a.total(), 2e-15));
  for (int k = 0; k < 4; ++k) {
    D4_CHECK(near(pe.dq[k], a.dq[3 - k], 2e-14));
    for (int x = 0; x < 3; ++x)
      D4_CHECK(near(pe.gradient[3 * k + x], a.gradient[3 * (3 - k) + x], 2e-14));
  }
  // Independently vary pair/ATM scales and permit signed s8, unlike the
  // positive-damping-only GFN2 CUDA helper that must not become a DFT default.
  for (auto params :
       {D4Parameters{D4ReferenceModel::gfn2, 0.0, 0.0, 5.0, 0.52, 5.0, 30, 50, 25},
        D4Parameters{D4ReferenceModel::gfn2, 1.0, -2.7, 0.0, 0.52, 5.0, 30, 50, 25}}) {
    const auto b = evaluate(m, params);
    D4_CHECK(b.status == D4Status::success);
    for (std::size_t k = 0; k < m.xyz.size(); ++k) {
      auto plus = m, minus = m;
      plus.xyz[k] += 1e-5;
      minus.xyz[k] -= 1e-5;
      auto ep = evaluate(plus, params), em = evaluate(minus, params);
      D4_CHECK(ep.status == D4Status::success && em.status == D4Status::success);
      D4_CHECK(near((ep.total() - em.total()) / 2e-5, b.gradient[k], 3e-10));
    }
  }
  // Empty and single-atom cases need no fictitious pair storage.
  const auto empty = evaluate(Molecule{}, p);
  D4_CHECK(empty.status == D4Status::success && empty.total() == 0.0);
  const auto single = evaluate(Molecule{{8}, {0, 0, 0}, {-1}}, p);
  D4_CHECK(single.status == D4Status::success && single.total() == 0.0);
  // Every table element, including heavy elements, stays in bounds.
  for (int z = 1; z <= 86; ++z) {
    const auto el = evaluate(Molecule{{z, 1}, {0, 0, 0, 6, 1, -1}, {0.1, -0.1}}, p);
    D4_CHECK(el.status == D4Status::success && std::isfinite(el.total()));
  }
  auto boundary = m;
  boundary.q[0] = -data::kElements[7].effective_charge;
  const auto bd = evaluate(boundary, p);
  D4_CHECK(bd.status == D4Status::success && std::isfinite(bd.total()) && bd.dq[0] == 0.0);
  boundary.q[0] -= 0.01;
  const auto below = evaluate(boundary, p);
  D4_CHECK(below.status == D4Status::success && below.dq[0] == 0.0);
  auto unchanged = [&](const Result& b) {
    return b.energy == std::array<double, 2>{17, 19} &&
           std::all_of(b.gradient.begin(), b.gradient.end(), [](double x) { return x == 23; }) &&
           std::all_of(b.dq.begin(), b.dq.end(), [](double x) { return x == 29; });
  };
  auto wrong = p;
  wrong.reference_model = D4ReferenceModel::eeq;
  const auto eeq = evaluate(m, wrong);
  D4_CHECK(eeq.status == D4Status::unsupported && unchanged(eeq));
  for (int kind = 0; kind < 5; ++kind) {
    auto bad = m;
    if (kind == 0) bad.q[0] = std::numeric_limits<double>::quiet_NaN();
    if (kind == 1) bad.xyz[0] = std::numeric_limits<double>::infinity();
    if (kind == 2)
      for (int k = 0; k < 3; ++k) bad.xyz[3 + k] = bad.xyz[k];
    if (kind == 3) bad.z[0] = 0;
    if (kind == 4) bad.z[0] = 87;
    const auto b = evaluate(bad, p);
    D4_CHECK(b.status != D4Status::success && unchanged(b));
  }
  for (double bad : {-1.0, 0.0, std::numeric_limits<double>::quiet_NaN()}) {
    auto params = p;
    params.cn_cutoff = bad;
    const auto b = evaluate(m, params);
    D4_CHECK(b.status == D4Status::invalid_argument && unchanged(b));
  }
  // Repeated geometries must not reuse stale coefficients/coordination.
  auto moved = m;
  moved.xyz[0] += 0.12;
  const auto changed = evaluate(moved, p), restored = evaluate(m, p);
  D4_CHECK(changed.status == D4Status::success && changed.total() != a.total());
  D4_CHECK(restored.energy == a.energy && restored.gradient == a.gradient);
  std::puts("D4 energies, ATM, coordinate/charge derivatives and negative cases passed");
  return 0;
}
#undef D4_CHECK
}  // namespace d4_tests

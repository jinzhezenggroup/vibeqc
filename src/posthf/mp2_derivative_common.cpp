#include "posthf/mp2_derivative_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "hf/reference.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/df_progress_trace.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::mp2::detail {
namespace {
bool finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

std::size_t square(std::size_t value) { return posthf::checked_mul(value, value); }
std::size_t fourth(std::size_t value) { return square(square(value)); }

std::vector<std::size_t> shell_offsets(const core::System& system) {
  std::vector<std::size_t> offsets(system.shells.size() + 1, 0);
  for (std::size_t shell = 0; shell < system.shells.size(); ++shell) {
    const auto count = system.basis_representation == GENERATIVEQC_BASIS_SPHERICAL
                           ? 2 * system.shells[shell].angular_momentum + 1
                           : molecule::cartesian_count(system.shells[shell].angular_momentum);
    offsets[shell + 1] = posthf::checked_add(offsets[shell], count);
  }
  return offsets;
}

std::vector<double> pullback_matrix(std::span<const double> coefficients,
                                    std::span<const double> mo, std::size_t n) {
  const auto n2 = square(n);
  if (coefficients.size() != n2 || mo.size() != n2)
    throw std::invalid_argument("rank-2 AO pullback shape mismatch");
  std::vector<double> ao(n2), workspace(n2);
  tensor::cpu_congruence('N', n, coefficients.data(), mo.data(), ao.data(), workspace.data());
  return ao;
}

double symmetric_eri_weight(std::span<const double> weights, std::size_t n, std::size_t p,
                            std::size_t q, std::size_t r, std::size_t s) {
  const auto at = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return weights[((a * n + b) * n + c) * n + d];
  };
  // Project arbitrary adjoint weights onto the eight exact ERI permutations.
  // The same coefficient matrix acts on all four slots, so this projection
  // commutes with the MO-to-AO pullback. No additional rank-four buffer is live.
  return 0.125 * at(p, q, r, s) + 0.125 * at(q, p, r, s) + 0.125 * at(p, q, s, r) +
         0.125 * at(q, p, s, r) + 0.125 * at(r, s, p, q) + 0.125 * at(s, r, p, q) +
         0.125 * at(r, s, q, p) + 0.125 * at(s, r, q, p);
}

void transform_remaining_shells(const core::System& system, const hf::PhysicalReference& reference,
                                const std::vector<std::size_t>& offsets, std::size_t si,
                                std::span<const double> first,
                                const EriShellDerivativeAccumulate& eri_shell,
                                std::vector<double>& derivative, bool canonical_shells) {
  const auto n = reference.nbf;
  const auto di = offsets[si + 1] - offsets[si];
  for (std::size_t sj = 0; sj < system.shells.size(); ++sj) {
    if (canonical_shells && sj > si) continue;
    const auto dj = offsets[sj + 1] - offsets[sj];
    std::vector<double> second(posthf::checked_mul(posthf::checked_mul(di, dj), square(n)), 0.0);
    for (std::size_t iu = 0; iu < di; ++iu)
      for (std::size_t jv = 0; jv < dj; ++jv)
        for (std::size_t r = 0; r < n; ++r)
          for (std::size_t s = 0; s < n; ++s)
            for (std::size_t q = 0; q < n; ++q)
              second[((iu * dj + jv) * n + r) * n + s] +=
                  reference.coefficients[(offsets[sj] + jv) * n + q] *
                  first[((iu * n + q) * n + r) * n + s];
    for (std::size_t sk = 0; sk < system.shells.size(); ++sk) {
      if (canonical_shells && sk > si) continue;
      const auto dk = offsets[sk + 1] - offsets[sk];
      std::vector<double> third(
          posthf::checked_mul(posthf::checked_mul(posthf::checked_mul(di, dj), dk), n), 0.0);
      for (std::size_t iu = 0; iu < di; ++iu)
        for (std::size_t jv = 0; jv < dj; ++jv)
          for (std::size_t kw = 0; kw < dk; ++kw)
            for (std::size_t s = 0; s < n; ++s)
              for (std::size_t r = 0; r < n; ++r)
                third[((iu * dj + jv) * dk + kw) * n + s] +=
                    reference.coefficients[(offsets[sk] + kw) * n + r] *
                    second[((iu * dj + jv) * n + r) * n + s];
      for (std::size_t sl = 0; sl < system.shells.size(); ++sl) {
        if (canonical_shells && (sl > sk || (si == sk && sl > sj))) continue;
        const auto dl = offsets[sl + 1] - offsets[sl];
        std::vector<double> local(
            posthf::checked_mul(posthf::checked_mul(posthf::checked_mul(di, dj), dk), dl), 0.0);
        for (std::size_t iu = 0; iu < di; ++iu)
          for (std::size_t jv = 0; jv < dj; ++jv)
            for (std::size_t kw = 0; kw < dk; ++kw)
              for (std::size_t lx = 0; lx < dl; ++lx)
                for (std::size_t s = 0; s < n; ++s)
                  local[((iu * dj + jv) * dk + kw) * dl + lx] +=
                      reference.coefficients[(offsets[sl] + lx) * n + s] *
                      third[((iu * dj + jv) * dk + kw) * n + s];
        if (canonical_shells) {
          // All components within repeated shells remain present. Multiply by
          // the distinct shell-quartet orbit, not a fixed factor of eight;
          // center derivatives are still differentiated separately and then
          // scattered to atoms, including when several slots share an atom.
          const auto orbit =
              (si == sj ? 1 : 2) * (sk == sl ? 1 : 2) * (si == sk && sj == sl ? 1 : 2);
          for (double& value : local) value *= orbit;
        }
        const std::array<std::size_t, 4> shells{si, sj, sk, sl};
        eri_shell(shells, local, derivative);
      }
    }
  }
}
}  // namespace

std::vector<double> conventional_derivative_accumulate(
    const core::System& system, const hf::PhysicalReference& reference,
    const LagrangianWeights& weights, const OneElectronDerivativeContract& one_electron,
    const EriShellDerivativeAccumulate& eri_shell,
    const std::function<void(std::span<double>)>& finalize) {
  const auto n = reference.nbf;
  bool dense_two = false;
  if (!weights.two_electron.empty())
    dense_two = weights.two_electron.size() == fourth(n) && finite(weights.two_electron);
  const bool factorized_two = weights.two_electron.empty() &&
                              weights.two_electron_factors.orbitals == n &&
                              weights.two_electron_factors.occupied == reference.nocc &&
                              valid_factorized_two_electron_weights(weights.two_electron_factors);
  if (!n || molecule::ao_count(system) != n || reference.coefficients.size() != square(n) ||
      weights.orbitals != n || weights.occupied != reference.nocc ||
      weights.one_electron.size() != square(n) || weights.overlap.size() != square(n) ||
      (!dense_two && !factorized_two) || !finite(reference.coefficients) ||
      !finite(weights.one_electron) || !finite(weights.overlap) || !one_electron || !eri_shell)
    throw std::invalid_argument("conventional derivative reference/weight mismatch");

  runtime::df_progress::Scope trace("conventional_derivative");
  using Clock = std::chrono::steady_clock;
  const auto now = [&] { return trace.enabled() ? Clock::now() : Clock::time_point{}; };
  const auto started = now();
  const auto one_ao = pullback_matrix(reference.coefficients, weights.one_electron, n);
  const auto overlap_ao = pullback_matrix(reference.coefficients, weights.overlap, n);
  const auto one_started = now();
  auto derivative = one_electron(overlap_ao, one_ao);
  const auto two_started = now();
  if (derivative.size() != posthf::checked_mul(system.atoms.size(), std::size_t{3}) ||
      !finite(derivative))
    throw std::runtime_error("conventional one-electron derivative has the wrong shape");
  const auto offsets = shell_offsets(system);
  if (offsets.back() != n)
    throw std::runtime_error("conventional derivative shell offsets disagree with the reference");

  std::uint64_t shell_ns = 0, shell_calls = 0;
  EriShellDerivativeAccumulate measured_shell;
  if (trace.enabled()) {
    measured_shell = [&](const auto& shells, auto local, auto gradient) {
      const auto shell_started = Clock::now();
      eri_shell(shells, local, gradient);
      shell_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - shell_started)
                      .count();
      ++shell_calls;
    };
  }
  const auto& shell_contract = trace.enabled() ? measured_shell : eri_shell;
  for (std::size_t si = 0; si < system.shells.size(); ++si) {
    const auto di = offsets[si + 1] - offsets[si];
    std::vector<double> first(posthf::checked_mul(di, posthf::checked_mul(n, square(n))), 0.0);
    if (dense_two) {
      for (std::size_t iu = 0; iu < di; ++iu)
        for (std::size_t q = 0; q < n; ++q)
          for (std::size_t r = 0; r < n; ++r)
            for (std::size_t s = 0; s < n; ++s)
              for (std::size_t p = 0; p < n; ++p)
                first[((iu * n + q) * n + r) * n + s] +=
                    reference.coefficients[(offsets[si] + iu) * n + p] *
                    symmetric_eri_weight(weights.two_electron, n, p, q, r, s);
    } else {
      const auto& factors = weights.two_electron_factors;
      const auto occupied = factors.occupied, virtuals = n - occupied;
      auto g_index = [occupied, virtuals](std::size_t i, std::size_t j, std::size_t a,
                                          std::size_t b) {
        return ((i * occupied + j) * virtuals + a) * virtuals + b;
      };
      for (std::size_t iu = 0; iu < di; ++iu) {
        const auto mu = offsets[si] + iu;
        for (std::size_t p = 0; p < n; ++p) {
          const double coefficient = reference.coefficients[mu * n + p];
          if (coefficient == 0.0) continue;
          for (std::size_t q = 0; q < n; ++q) {
            const double weight = factors.fock[p * n + q];
            if (weight == 0.0) continue;
            for (std::size_t i = 0; i < occupied; ++i) {
              first[((iu * n + q) * n + i) * n + i] += 2.0 * coefficient * weight;
              first[((iu * n + i) * n + i) * n + q] -= coefficient * weight;
            }
          }
        }
        for (std::size_t i = 0; i < occupied; ++i) {
          const double coefficient = reference.coefficients[mu * n + i];
          if (coefficient == 0.0) continue;
          for (std::size_t j = 0; j < occupied; ++j)
            for (std::size_t a = 0; a < virtuals; ++a)
              for (std::size_t b = 0; b < virtuals; ++b)
                first[((iu * n + occupied + a) * n + j) * n + occupied + b] +=
                    coefficient * factors.correlation_iajb[g_index(i, j, a, b)];
        }
      }
    }
    transform_remaining_shells(system, reference, offsets, si, first, shell_contract, derivative,
                               dense_two);
  }
  const auto finalize_started = now();
  if (finalize) finalize(derivative);
  const auto finalize_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now() - finalize_started).count();
  if (trace.enabled()) shell_ns += finalize_ns;
  if (!finite(derivative)) throw std::runtime_error("conventional derivative is nonfinite");
  if (trace.enabled()) {
    using runtime::df_progress::Scope;
    const auto ns = [](auto interval) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(interval).count();
    };
    const auto two_ns = ns(now() - two_started);
    // Disjoint intervals. The shell callback may contain nested GPU consumer
    // work; the remainder measures host pullback/scatter and bookkeeping.
    Scope::number("ao_functions", n);
    Scope::number("canonical_shells", dense_two);
    Scope::number("rank_two_pullback_ns", ns(one_started - started));
    Scope::number("one_electron_ns", ns(two_started - one_started));
    Scope::number("two_electron_shell_calls", shell_calls);
    Scope::number("two_electron_shell_ns", shell_ns);
    Scope::number("two_electron_finalize_ns", finalize_ns);
    Scope::number("two_electron_pullback_scatter_ns", two_ns - shell_ns);
  }
  return derivative;
}

std::vector<double> conventional_derivative(const core::System& system,
                                            const hf::PhysicalReference& reference,
                                            const LagrangianWeights& weights,
                                            const OneElectronDerivativeContract& one_electron,
                                            const EriShellDerivativeContract& eri_shell) {
  if (!eri_shell) throw std::invalid_argument("missing conventional ERI derivative consumer");
  return conventional_derivative_accumulate(
      system, reference, weights, one_electron, [&](const auto& shells, auto local, auto gradient) {
        const auto center = eri_shell(shells, local);
        for (std::size_t slot = 0; slot < 4; ++slot)
          for (std::size_t axis = 0; axis < 3; ++axis)
            gradient[3 * system.shells[shells[slot]].atom_index + axis] += center[3 * slot + axis];
      });
}

}  // namespace generativeqc::mp2::detail

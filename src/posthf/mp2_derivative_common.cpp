#include "posthf/mp2_derivative_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "scf/types.hpp"

namespace vibeqc::mp2::detail {
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
    const auto count = system.basis_representation == VIBEQC_BASIS_SPHERICAL
                           ? 2 * system.shells[shell].angular_momentum + 1
                           : molecule::cartesian_count(system.shells[shell].angular_momentum);
    offsets[shell + 1] = posthf::checked_add(offsets[shell], count);
  }
  return offsets;
}

std::vector<double> pullback_matrix(std::span<const double> coefficients,
                                    std::span<const double> mo, std::size_t n) {
  std::vector<double> ao(square(n), 0.0);
  for (std::size_t u = 0; u < n; ++u)
    for (std::size_t v = 0; v < n; ++v)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q)
          ao[u * n + v] += coefficients[u * n + p] * mo[p * n + q] * coefficients[v * n + q];
  return ao;
}

void transform_remaining_shells(const core::System& system, const scf::PhysicalReference& reference,
                                const std::vector<std::size_t>& offsets, std::size_t si,
                                std::span<const double> first,
                                const EriShellDerivativeContract& eri_shell,
                                std::vector<double>& derivative) {
  const auto n = reference.nbf;
  const auto di = offsets[si + 1] - offsets[si];
  for (std::size_t sj = 0; sj < system.shells.size(); ++sj) {
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
        const std::array<std::size_t, 4> shells{si, sj, sk, sl};
        const auto center = eri_shell(shells, local);
        for (std::size_t slot = 0; slot < 4; ++slot) {
          const auto atom = system.shells[shells[slot]].atom_index;
          for (std::size_t axis = 0; axis < 3; ++axis)
            derivative[3 * atom + axis] += center[3 * slot + axis];
        }
      }
    }
  }
}
}  // namespace

std::vector<double> conventional_derivative(const core::System& system,
                                            const scf::PhysicalReference& reference,
                                            const LagrangianWeights& weights,
                                            const OneElectronDerivativeContract& one_electron,
                                            const EriShellDerivativeContract& eri_shell) {
  const auto n = reference.nbf;
  if (!n || molecule::ao_count(system) != n || reference.coefficients.size() != square(n) ||
      weights.orbitals != n || weights.occupied != reference.nocc ||
      weights.one_electron.size() != square(n) || weights.overlap.size() != square(n) ||
      weights.two_electron.size() != fourth(n) || !finite(reference.coefficients) ||
      !finite(weights.one_electron) || !finite(weights.overlap) || !finite(weights.two_electron) ||
      !one_electron || !eri_shell)
    throw std::invalid_argument("conventional derivative reference/weight mismatch");

  const auto one_ao = pullback_matrix(reference.coefficients, weights.one_electron, n);
  const auto overlap_ao = pullback_matrix(reference.coefficients, weights.overlap, n);
  auto derivative = one_electron(overlap_ao, one_ao);
  if (derivative.size() != posthf::checked_mul(system.atoms.size(), std::size_t{3}) ||
      !finite(derivative))
    throw std::runtime_error("conventional one-electron derivative has the wrong shape");
  const auto offsets = shell_offsets(system);
  if (offsets.back() != n)
    throw std::runtime_error("conventional derivative shell offsets disagree with the reference");

  for (std::size_t si = 0; si < system.shells.size(); ++si) {
    const auto di = offsets[si + 1] - offsets[si];
    std::vector<double> first(posthf::checked_mul(di, posthf::checked_mul(n, square(n))), 0.0);
    for (std::size_t iu = 0; iu < di; ++iu)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t r = 0; r < n; ++r)
          for (std::size_t s = 0; s < n; ++s)
            for (std::size_t p = 0; p < n; ++p)
              first[((iu * n + q) * n + r) * n + s] +=
                  reference.coefficients[(offsets[si] + iu) * n + p] *
                  weights.two_electron[((p * n + q) * n + r) * n + s];
    transform_remaining_shells(system, reference, offsets, si, first, eri_shell, derivative);
  }
  if (!finite(derivative)) throw std::runtime_error("conventional derivative is nonfinite");
  return derivative;
}

}  // namespace vibeqc::mp2::detail

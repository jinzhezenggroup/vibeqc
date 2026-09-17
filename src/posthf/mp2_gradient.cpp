#include "posthf/mp2_gradient.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "posthf/capacity.hpp"
#include "posthf/native_provider.hpp"

namespace vibeqc::mp2 {
namespace {
constexpr const char* energy_adjoint_hash = "mp2-canonical-energy-adjoint-v1";

std::size_t square(std::size_t value) { return posthf::checked_mul(value, value); }

std::size_t fourth_power(std::size_t value) { return square(square(value)); }

std::size_t eri_index(std::size_t n, std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
  return ((p * n + q) * n + r) * n + s;
}

std::size_t g_index(std::size_t no, std::size_t nv, std::size_t i, std::size_t j, std::size_t a,
                    std::size_t b) {
  return ((i * no + j) * nv + a) * nv + b;
}

bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

void validate_adjoint(const EnergyAdjoint& adjoint) {
  if (!adjoint.orbitals || !adjoint.occupied || adjoint.occupied >= adjoint.orbitals)
    throw std::invalid_argument("invalid canonical MP2 adjoint dimensions");
  const auto virtuals = adjoint.orbitals - adjoint.occupied;
  const auto expected = posthf::checked_mul(square(adjoint.occupied), square(virtuals));
  if (adjoint.integrals_iajb.size() != expected ||
      adjoint.orbital_energies.size() != adjoint.orbitals || !finite(adjoint.integrals_iajb) ||
      !finite(adjoint.orbital_energies))
    throw std::invalid_argument("inconsistent or nonfinite canonical MP2 adjoint");
}

std::vector<double> rotation_gradient(std::span<const double> one, std::span<const double> two,
                                      std::span<const double> h, std::span<const double> eri,
                                      std::size_t n) {
  std::vector<double> result(square(n), 0.0);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t t = 0; t < n; ++t) {
        result[t * n + p] += one[p * n + q] * h[t * n + q];
        result[t * n + q] += one[p * n + q] * h[p * n + t];
      }
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const double weight = two[eri_index(n, p, q, r, s)];
          if (weight == 0.0) continue;
          for (std::size_t t = 0; t < n; ++t) {
            result[t * n + p] += weight * eri[eri_index(n, t, q, r, s)];
            result[t * n + q] += weight * eri[eri_index(n, p, t, r, s)];
            result[t * n + r] += weight * eri[eri_index(n, p, q, t, s)];
            result[t * n + s] += weight * eri[eri_index(n, p, q, r, t)];
          }
        }
  if (!finite(result)) throw std::runtime_error("nonfinite MP2 rotation gradient");
  return result;
}

std::vector<double> canonical_fock(std::span<const double> h, std::span<const double> eri,
                                   std::size_t n, std::size_t occupied) {
  std::vector<double> fock(h.begin(), h.end());
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t i = 0; i < occupied; ++i)
        fock[p * n + q] += 2.0 * eri[eri_index(n, p, q, i, i)] - eri[eri_index(n, p, i, i, q)];
  double off_diagonal = 0.0;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      if (p != q) off_diagonal = std::max(off_diagonal, std::abs(fock[p * n + q]));
  if (off_diagonal > 1e-8)
    throw std::invalid_argument("MP2 orbital RHS requires a canonical RHF Fock matrix");
  return fock;
}

std::vector<std::size_t> all_orbitals(std::size_t n) {
  std::vector<std::size_t> result(n);
  for (std::size_t p = 0; p < n; ++p) result[p] = p;
  return result;
}

std::vector<double> streamed_fock(std::span<const double> h,
                                  const posthf::NativeBlockProvider& provider, std::size_t n,
                                  std::size_t occupied, bool cuda, int device_id) {
  auto fock = std::vector<double>(h.begin(), h.end());
  const auto all = all_orbitals(n);
  for (std::size_t i = 0; i < occupied; ++i) {
    const std::vector<std::size_t> orbital{i};
    const auto coulomb = provider.get({all, all, orbital, orbital}, cuda, device_id);
    const auto exchange = provider.get({all, orbital, orbital, all}, cuda, device_id);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        fock[p * n + q] += 2.0 * coulomb[p * n + q] - exchange[p * n + q];
  }
  double off_diagonal = 0.0;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      if (p != q) off_diagonal = std::max(off_diagonal, std::abs(fock[p * n + q]));
  if (off_diagonal > 1e-8)
    throw std::invalid_argument("streamed MP2 orbital RHS requires a canonical RHF Fock matrix");
  return fock;
}

std::vector<double> rotation_gradient_streamed(std::span<const double> one,
                                               std::span<const double> two,
                                               std::span<const double> h,
                                               const posthf::NativeBlockProvider& provider,
                                               std::size_t n, bool cuda, int device_id) {
  std::vector<double> result(square(n), 0.0);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t t = 0; t < n; ++t) {
        result[t * n + p] += one[p * n + q] * h[t * n + q];
        result[t * n + q] += one[p * n + q] * h[p * n + t];
      }
  const auto all = all_orbitals(n);
  for (std::size_t t = 0; t < n; ++t) {
    const std::vector<std::size_t> orbital{t};
    const auto first = provider.get({orbital, all, all, all}, cuda, device_id);
    const auto second = provider.get({all, orbital, all, all}, cuda, device_id);
    const auto third = provider.get({all, all, orbital, all}, cuda, device_id);
    const auto fourth = provider.get({all, all, all, orbital}, cuda, device_id);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t r = 0; r < n; ++r)
          for (std::size_t s = 0; s < n; ++s) {
            const double weight = two[eri_index(n, p, q, r, s)];
            if (weight == 0.0) continue;
            result[t * n + p] += weight * first[(q * n + r) * n + s];
            result[t * n + q] += weight * second[(p * n + r) * n + s];
            result[t * n + r] += weight * third[(p * n + q) * n + s];
            result[t * n + s] += weight * fourth[(p * n + q) * n + r];
          }
  }
  if (!finite(result)) throw std::runtime_error("nonfinite streamed MP2 rotation gradient");
  return result;
}

OrbitalRhs initial_orbital_weights(const EnergyAdjoint& adjoint) {
  const auto n = adjoint.orbitals, occupied = adjoint.occupied;
  const auto virtuals = n - occupied;
  OrbitalRhs result;
  result.orbitals = n;
  result.occupied = occupied;
  result.one_electron.assign(square(n), 0.0);
  result.two_electron.assign(fourth_power(n), 0.0);
  for (std::size_t p = 0; p < n; ++p) {
    const double weight = adjoint.orbital_energies[p];
    result.one_electron[p * n + p] = weight;
    for (std::size_t i = 0; i < occupied; ++i) {
      result.two_electron[eri_index(n, p, p, i, i)] += 2.0 * weight;
      result.two_electron[eri_index(n, p, i, i, p)] -= weight;
    }
  }
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b)
          result.two_electron[eri_index(n, i, occupied + a, j, occupied + b)] +=
              adjoint.integrals_iajb[g_index(occupied, virtuals, i, j, a, b)];
  return result;
}

void add_negative_fock_multiplier(std::vector<double>& one, std::vector<double>& two, std::size_t n,
                                  std::size_t occupied, std::size_t row, std::size_t column,
                                  double value) {
  one[row * n + column] -= value;
  for (std::size_t j = 0; j < occupied; ++j) {
    two[eri_index(n, row, column, j, j)] -= 2.0 * value;
    two[eri_index(n, row, j, j, column)] += value;
  }
}
}  // namespace

EnergyAdjoint canonical_energy_adjoint(std::span<const double> integrals_iajb,
                                       std::span<const double> orbital_energies,
                                       std::size_t occupied, double denominator_threshold) {
  if (!occupied || occupied >= orbital_energies.size() || !std::isfinite(denominator_threshold) ||
      denominator_threshold <= 0.0 || !finite(integrals_iajb) || !finite(orbital_energies))
    throw std::invalid_argument("invalid canonical MP2 adjoint inputs");
  const auto n = orbital_energies.size(), virtuals = n - occupied;
  const auto expected = posthf::checked_mul(square(occupied), square(virtuals));
  if (integrals_iajb.size() != expected)
    throw std::invalid_argument("canonical MP2 adjoint shape mismatch");
  EnergyAdjoint result;
  result.orbitals = n;
  result.occupied = occupied;
  result.integrals_iajb.assign(expected, 0.0);
  result.orbital_energies.assign(n, 0.0);
  result.equation_hash = energy_adjoint_hash;
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b) {
          const auto index = g_index(occupied, virtuals, i, j, a, b);
          const auto exchange_index = g_index(occupied, virtuals, i, j, b, a);
          const double direct = integrals_iajb[index];
          const double exchange = integrals_iajb[exchange_index];
          const double denominator = orbital_energies[i] + orbital_energies[j] -
                                     orbital_energies[occupied + a] -
                                     orbital_energies[occupied + b];
          if (!std::isfinite(denominator) || std::abs(denominator) <= denominator_threshold)
            throw std::invalid_argument("near-zero MP2 denominator; no regularization applied");
          result.integrals_iajb[index] += (4.0 * direct - exchange) / denominator;
          result.integrals_iajb[exchange_index] -= direct / denominator;
          const double numerator = 2.0 * direct * direct - direct * exchange;
          const double denominator_weight = -numerator / (denominator * denominator);
          result.orbital_energies[i] += denominator_weight;
          result.orbital_energies[j] += denominator_weight;
          result.orbital_energies[occupied + a] -= denominator_weight;
          result.orbital_energies[occupied + b] -= denominator_weight;
        }
  if (!finite(result.integrals_iajb) || !finite(result.orbital_energies))
    throw std::runtime_error("nonfinite canonical MP2 adjoint accumulation");
  return result;
}

OrbitalRhs canonical_orbital_rhs(std::span<const double> hcore_mo, std::span<const double> eri_mo,
                                 const EnergyAdjoint& adjoint, double same_space_threshold) {
  validate_adjoint(adjoint);
  const auto n = adjoint.orbitals, occupied = adjoint.occupied;
  if (hcore_mo.size() != square(n) || eri_mo.size() != fourth_power(n) || !finite(hcore_mo) ||
      !finite(eri_mo) || !std::isfinite(same_space_threshold) || same_space_threshold <= 0.0)
    throw std::invalid_argument("canonical orbital RHS inputs must be finite and consistent");
  auto fock = canonical_fock(hcore_mo, eri_mo, n, occupied);
  auto result = initial_orbital_weights(adjoint);
  const auto virtuals = n - occupied;
  auto gradient = rotation_gradient(result.one_electron, result.two_electron, hcore_mo, eri_mo, n);
  result.energy_gradient.resize(occupied * virtuals);
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      result.energy_gradient[i * virtuals + a] =
          gradient[i * n + occupied + a] - gradient[(occupied + a) * n + i];
  auto correct_block = [&](std::size_t begin, std::size_t end) {
    for (std::size_t p = begin; p < end; ++p)
      for (std::size_t q = p + 1; q < end; ++q) {
        const double denominator = fock[p * n + p] - fock[q * n + q];
        const double derivative = gradient[p * n + q] - gradient[q * n + p];
        if (std::abs(denominator) <= same_space_threshold) {
          if (std::abs(derivative) > same_space_threshold)
            throw std::invalid_argument(
                "same-space canonical MP2 response has a nonstationary degenerate subspace");
          continue;
        }
        add_negative_fock_multiplier(result.one_electron, result.two_electron, n, occupied, q, p,
                                     derivative / denominator);
      }
  };
  correct_block(0, occupied);
  correct_block(occupied, n);
  gradient = rotation_gradient(result.one_electron, result.two_electron, hcore_mo, eri_mo, n);
  result.response_rhs.resize(occupied * virtuals);
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      result.response_rhs[i * virtuals + a] =
          gradient[(occupied + a) * n + i] - gradient[i * n + occupied + a];
  if (!finite(result.energy_gradient) || !finite(result.response_rhs))
    throw std::runtime_error("nonfinite MP2 orbital RHS");
  return result;
}

OrbitalRhs canonical_orbital_rhs_streamed(const scf::PhysicalReference& reference,
                                          std::span<const double> hcore_mo,
                                          const posthf::NativeBlockProvider& provider,
                                          const EnergyAdjoint& adjoint, double same_space_threshold,
                                          bool cuda, int device_id) {
  validate_adjoint(adjoint);
  const auto n = adjoint.orbitals, occupied = adjoint.occupied;
  if (&provider.reference() != &reference || reference.nbf != n || reference.nocc != occupied ||
      reference.orbital_energies.size() != n || hcore_mo.size() != square(n) || !finite(hcore_mo) ||
      !std::isfinite(same_space_threshold) || same_space_threshold <= 0.0)
    throw std::invalid_argument("streamed MP2 orbital RHS reference/provider mismatch");
  auto fock = streamed_fock(hcore_mo, provider, n, occupied, cuda, device_id);
  for (std::size_t p = 0; p < n; ++p)
    if (std::abs(fock[p * n + p] - reference.orbital_energies[p]) > 1e-8)
      throw std::invalid_argument("streamed MP2 Fock spectrum differs from the reference");
  auto result = initial_orbital_weights(adjoint);
  const auto virtuals = n - occupied;
  auto gradient = rotation_gradient_streamed(result.one_electron, result.two_electron, hcore_mo,
                                             provider, n, cuda, device_id);
  result.energy_gradient.resize(occupied * virtuals);
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      result.energy_gradient[i * virtuals + a] =
          gradient[i * n + occupied + a] - gradient[(occupied + a) * n + i];
  auto correct_block = [&](std::size_t begin, std::size_t end) {
    for (std::size_t p = begin; p < end; ++p)
      for (std::size_t q = p + 1; q < end; ++q) {
        const double denominator = reference.orbital_energies[p] - reference.orbital_energies[q];
        const double derivative = gradient[p * n + q] - gradient[q * n + p];
        if (std::abs(denominator) <= same_space_threshold) {
          if (std::abs(derivative) > same_space_threshold)
            throw std::invalid_argument(
                "same-space streamed MP2 response has a nonstationary degenerate subspace");
          continue;
        }
        add_negative_fock_multiplier(result.one_electron, result.two_electron, n, occupied, q, p,
                                     derivative / denominator);
      }
  };
  correct_block(0, occupied);
  correct_block(occupied, n);
  gradient = rotation_gradient_streamed(result.one_electron, result.two_electron, hcore_mo,
                                        provider, n, cuda, device_id);
  result.response_rhs.resize(occupied * virtuals);
  for (std::size_t i = 0; i < occupied; ++i)
    for (std::size_t a = 0; a < virtuals; ++a)
      result.response_rhs[i * virtuals + a] =
          gradient[(occupied + a) * n + i] - gradient[i * n + occupied + a];
  if (!finite(result.energy_gradient) || !finite(result.response_rhs))
    throw std::runtime_error("nonfinite streamed MP2 orbital RHS");
  return result;
}

LagrangianWeights canonical_lagrangian_weights(std::span<const double> hcore_mo,
                                               std::span<const double> eri_mo,
                                               const EnergyAdjoint& adjoint,
                                               std::span<const double> response,
                                               double same_space_threshold) {
  auto orbital = canonical_orbital_rhs(hcore_mo, eri_mo, adjoint, same_space_threshold);
  const auto n = adjoint.orbitals, occupied = adjoint.occupied;
  const auto virtuals = n - occupied;
  if (response.size() != occupied * virtuals || !finite(response))
    throw std::invalid_argument("MP2 Z-vector does not match occupied-virtual layout");
  LagrangianWeights result;
  result.orbitals = n;
  result.occupied = occupied;
  result.one_electron = std::move(orbital.one_electron);
  result.two_electron = std::move(orbital.two_electron);
  for (std::size_t i = 0; i < occupied; ++i) {
    result.one_electron[i * n + i] += 2.0;
    for (std::size_t j = 0; j < occupied; ++j) {
      result.two_electron[eri_index(n, i, i, j, j)] += 2.0;
      result.two_electron[eri_index(n, i, j, j, i)] -= 1.0;
    }
    for (std::size_t a = 0; a < virtuals; ++a)
      add_negative_fock_multiplier(result.one_electron, result.two_electron, n, occupied,
                                   occupied + a, i, response[i * virtuals + a]);
  }
  const auto gradient =
      rotation_gradient(result.one_electron, result.two_electron, hcore_mo, eri_mo, n);
  result.overlap.resize(square(n));
  std::vector<double> stationarity(square(n));
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      result.overlap[p * n + q] = -0.25 * (gradient[p * n + q] + gradient[q * n + p]);
      stationarity[p * n + q] = gradient[p * n + q] - gradient[q * n + p];
    }
  result.stationarity_residual = response::stable_norm(stationarity);
  return result;
}

LagrangianWeights canonical_lagrangian_weights_streamed(
    const scf::PhysicalReference& reference, std::span<const double> hcore_mo,
    const posthf::NativeBlockProvider& provider, const EnergyAdjoint& adjoint,
    std::span<const double> response, double same_space_threshold, bool cuda, int device_id) {
  auto orbital = canonical_orbital_rhs_streamed(reference, hcore_mo, provider, adjoint,
                                                same_space_threshold, cuda, device_id);
  const auto n = adjoint.orbitals, occupied = adjoint.occupied;
  const auto virtuals = n - occupied;
  if (response.size() != posthf::checked_mul(occupied, virtuals) || !finite(response))
    throw std::invalid_argument("streamed MP2 Z-vector has the wrong layout");
  LagrangianWeights result;
  result.orbitals = n;
  result.occupied = occupied;
  result.one_electron = std::move(orbital.one_electron);
  result.two_electron = std::move(orbital.two_electron);
  for (std::size_t i = 0; i < occupied; ++i) {
    result.one_electron[i * n + i] += 2.0;
    for (std::size_t j = 0; j < occupied; ++j) {
      result.two_electron[eri_index(n, i, i, j, j)] += 2.0;
      result.two_electron[eri_index(n, i, j, j, i)] -= 1.0;
    }
    for (std::size_t a = 0; a < virtuals; ++a)
      add_negative_fock_multiplier(result.one_electron, result.two_electron, n, occupied,
                                   occupied + a, i, response[i * virtuals + a]);
  }
  const auto gradient = rotation_gradient_streamed(result.one_electron, result.two_electron,
                                                   hcore_mo, provider, n, cuda, device_id);
  result.overlap.resize(square(n));
  std::vector<double> stationarity(square(n));
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      result.overlap[p * n + q] = -0.25 * (gradient[p * n + q] + gradient[q * n + p]);
      stationarity[p * n + q] = gradient[p * n + q] - gradient[q * n + p];
    }
  result.stationarity_residual = response::stable_norm(stationarity);
  return result;
}

GradientResourcePlan conventional_gradient_plan(
    std::size_t orbitals, std::size_t occupied, std::size_t provider_bytes,
    const response::GmresPlan& response_plan, std::size_t maximum_shell_ao_count,
    std::size_t coordinate_count, std::size_t candidate_output_bytes, std::size_t budget_bytes,
    std::size_t derivative_backend_staging_bytes) {
  if (!orbitals || !occupied || occupied >= orbitals || !maximum_shell_ao_count ||
      !coordinate_count ||
      response_plan.dimension != posthf::checked_mul(occupied, orbitals - occupied))
    throw std::invalid_argument("invalid conventional MP2 gradient resource dimensions");
  const auto virtuals = orbitals - occupied;
  const auto n2 = square(orbitals), n4 = fourth_power(orbitals);
  const auto rotations = posthf::checked_mul(occupied, virtuals);
  const auto amplitudes = posthf::checked_mul(square(occupied), square(virtuals));
  GradientResourcePlan plan;
  plan.provider_bytes = provider_bytes;
  plan.adjoint_bytes =
      posthf::checked_mul(sizeof(double), posthf::checked_add(amplitudes, orbitals));
  plan.response_bytes = posthf::checked_add(
      response_plan.workspace_bytes,
      posthf::checked_mul(sizeof(double),
                          posthf::checked_add(posthf::checked_mul(6, rotations), n2)));
  auto relaxed_elements =
      posthf::checked_add(posthf::checked_mul(2, n4), posthf::checked_mul(3, n2));
  relaxed_elements = posthf::checked_add(relaxed_elements, posthf::checked_mul(2, rotations));
  plan.relaxed_weight_bytes = posthf::checked_mul(sizeof(double), relaxed_elements);
  plan.shell_cotangent_bytes =
      posthf::checked_mul(sizeof(double), fourth_power(maximum_shell_ao_count));
  const auto shell2 = square(maximum_shell_ao_count);
  const auto shell3 = posthf::checked_mul(shell2, maximum_shell_ao_count);
  const auto n3 = posthf::checked_mul(n2, orbitals);
  auto derivative_elements = posthf::checked_mul(2, n2);
  derivative_elements =
      posthf::checked_add(derivative_elements, posthf::checked_mul(maximum_shell_ao_count, n3));
  derivative_elements = posthf::checked_add(derivative_elements, posthf::checked_mul(shell2, n2));
  derivative_elements =
      posthf::checked_add(derivative_elements, posthf::checked_mul(shell3, orbitals));
  derivative_elements = posthf::checked_add(derivative_elements, coordinate_count);
  plan.derivative_staging_bytes = posthf::checked_mul(sizeof(double), derivative_elements);
  plan.derivative_backend_staging_bytes = derivative_backend_staging_bytes;
  plan.candidate_output_bytes = candidate_output_bytes;
  plan.peak_bytes = provider_bytes;
  for (auto bytes : {plan.adjoint_bytes, plan.response_bytes, plan.relaxed_weight_bytes,
                     plan.shell_cotangent_bytes, plan.derivative_staging_bytes,
                     plan.derivative_backend_staging_bytes, plan.candidate_output_bytes})
    plan.peak_bytes = posthf::checked_add(plan.peak_bytes, bytes);
  if (plan.peak_bytes > budget_bytes)
    throw std::length_error("conventional MP2 gradient exceeds numeric memory budget");
  return plan;
}

}  // namespace vibeqc::mp2

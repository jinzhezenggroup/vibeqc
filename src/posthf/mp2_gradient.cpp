#include "posthf/mp2_gradient.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "integrals/density_fitting_metric.hpp"
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
                                  const posthf::MOBlockProvider& provider, std::size_t n,
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
                                               const posthf::MOBlockProvider& provider,
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
                                          const posthf::MOBlockProvider& provider,
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
    const posthf::MOBlockProvider& provider, const EnergyAdjoint& adjoint,
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

DensityFittedLagrangianWeights density_fitted_lagrangian_weights(
    const scf::PhysicalReference& reference, const posthf::DensityFittedBlockProvider& provider,
    const LagrangianWeights& weights, std::size_t maximum_bytes) {
  const auto n = reference.nbf, occupied = reference.nocc, na = provider.auxiliary_count();
  if (&provider.reference() != &reference || !n || !occupied || occupied >= n || !na ||
      weights.orbitals != n || weights.occupied != occupied || !maximum_bytes)
    throw std::invalid_argument("RI-MP2 Lagrangian/reference/provider mismatch");
  const auto n2 = square(n), n4 = fourth_power(n), a2 = square(na);
  const auto three = posthf::checked_mul(n2, na);
  if (reference.coefficients.size() != n2 || weights.one_electron.size() != n2 ||
      weights.overlap.size() != n2 || weights.two_electron.size() != n4 ||
      provider.metric().size() != a2 || provider.inverse_square_root().size() != a2 ||
      provider.transformed_three_center().size() != three ||
      provider.whitened_three_center().size() != three || !finite(reference.coefficients) ||
      !finite(weights.one_electron) || !finite(weights.overlap) || !finite(weights.two_electron))
    throw std::invalid_argument("RI-MP2 Lagrangian weights have inconsistent dimensions");

  // Result: S/H/A/M. Scratch: bar_B, bar_A_MO and bar_X plus conservative
  // eigensystem workspace for the shared inverse-square-root pullback. Include
  // the provider and incoming relaxed weights because they coexist at this
  // stage of the endpoint.
  auto result_elements = posthf::checked_add(posthf::checked_mul(2, n2), three);
  result_elements = posthf::checked_add(result_elements, a2);
  auto scratch_elements = posthf::checked_mul(2, three);
  // bar_X and eigenvectors coexist with the VJP's symmetric/temp/transformed
  // matrices (its returned metric is charged above). Reserve one additional
  // matrix for eigensolver staging, plus eigenvalues and retained-rank flags.
  scratch_elements = posthf::checked_add(scratch_elements, posthf::checked_mul(6, a2));
  scratch_elements = posthf::checked_add(scratch_elements, posthf::checked_mul(2, na));
  scratch_elements = posthf::checked_add(scratch_elements, n2);
  auto weight_elements = posthf::checked_add(posthf::checked_mul(2, n2), n4);
  auto required = provider.provider_bytes();
  required = posthf::checked_add(required, posthf::checked_mul(weight_elements, sizeof(double)));
  required = posthf::checked_add(required, posthf::checked_mul(result_elements, sizeof(double)));
  required = posthf::checked_add(required, posthf::checked_mul(scratch_elements, sizeof(double)));
  if (required > maximum_bytes)
    throw std::length_error("RI-MP2 Lagrangian reverse exceeds memory budget");

  const auto& c = reference.coefficients;
  const auto& transformed = provider.transformed_three_center();
  const auto& whitened = provider.whitened_three_center();
  const auto& inverse_root = provider.inverse_square_root();
  auto three_index = [n, na](std::size_t p, std::size_t q, std::size_t aux) {
    return (p * n + q) * na + aux;
  };

  std::vector<double> bar_whitened(three, 0.0);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t aux = 0; aux < na; ++aux)
        for (std::size_t r = 0; r < n; ++r)
          for (std::size_t t = 0; t < n; ++t)
            bar_whitened[three_index(p, q, aux)] +=
                (weights.two_electron[eri_index(n, p, q, r, t)] +
                 weights.two_electron[eri_index(n, r, t, p, q)]) *
                whitened[three_index(r, t, aux)];

  std::vector<double> bar_transformed(three, 0.0);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t P = 0; P < na; ++P)
        for (std::size_t Q = 0; Q < na; ++Q)
          bar_transformed[three_index(p, q, P)] +=
              inverse_root[P * na + Q] * bar_whitened[three_index(p, q, Q)];

  DensityFittedLagrangianWeights result;
  result.orbitals = n;
  result.auxiliary = na;
  result.overlap.assign(n2, 0.0);
  result.one_electron.assign(n2, 0.0);
  result.three_center.assign(three, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = 0; nu < n; ++nu)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q) {
          const double coefficient = c[mu * n + p] * c[nu * n + q];
          result.overlap[mu * n + nu] += coefficient * weights.overlap[p * n + q];
          result.one_electron[mu * n + nu] += coefficient * weights.one_electron[p * n + q];
          if (coefficient == 0.0) continue;
          for (std::size_t P = 0; P < na; ++P)
            result.three_center[(mu * n + nu) * na + P] +=
                coefficient * bar_transformed[three_index(p, q, P)];
        }

  std::vector<double> bar_inverse_root(a2, 0.0);
  for (std::size_t P = 0; P < na; ++P)
    for (std::size_t Q = 0; Q < na; ++Q)
      for (std::size_t p = 0; p < n; ++p)
        for (std::size_t q = 0; q < n; ++q)
          bar_inverse_root[P * na + Q] +=
              transformed[three_index(p, q, P)] * bar_whitened[three_index(p, q, Q)];
  result.metric = integrals::density_fitting_metric_response(
      provider.metric(), inverse_root, bar_inverse_root, na, provider.relative_threshold(),
      tensor::SymmetricMatrixFunction::inverse_sqrt);
  result.workspace_bytes = posthf::checked_mul(scratch_elements, sizeof(double));
  result.planned_peak_bytes = required;
  if (!finite(result.overlap) || !finite(result.one_electron) || !finite(result.three_center) ||
      !finite(result.metric))
    throw std::runtime_error("RI-MP2 Lagrangian reverse produced nonfinite weights");
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

DensityFittedGradientResourcePlan density_fitted_gradient_plan(
    std::size_t orbitals, std::size_t occupied, std::size_t auxiliaries, std::size_t provider_bytes,
    const response::GmresPlan& response_plan, std::size_t cartesian_orbitals,
    std::size_t cartesian_auxiliaries, std::size_t coordinate_count,
    std::size_t candidate_output_bytes, std::size_t budget_bytes) {
  if (!orbitals || !occupied || occupied >= orbitals || !auxiliaries || !cartesian_orbitals ||
      !cartesian_auxiliaries || !coordinate_count ||
      response_plan.dimension != posthf::checked_mul(occupied, orbitals - occupied))
    throw std::invalid_argument("invalid RI-MP2 gradient resource dimensions");
  const auto virtuals = orbitals - occupied;
  const auto n2 = square(orbitals), n4 = fourth_power(orbitals);
  const auto a2 = square(auxiliaries);
  const auto three = posthf::checked_mul(n2, auxiliaries);
  const auto rotations = posthf::checked_mul(occupied, virtuals);
  const auto amplitudes = posthf::checked_mul(square(occupied), square(virtuals));

  DensityFittedGradientResourcePlan plan;
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

  auto reverse_result_elements =
      posthf::checked_add(posthf::checked_mul(2, n2), posthf::checked_add(three, a2));
  plan.reverse_result_bytes = posthf::checked_mul(sizeof(double), reverse_result_elements);
  auto reverse_workspace_elements = posthf::checked_mul(2, three);
  reverse_workspace_elements =
      posthf::checked_add(reverse_workspace_elements, posthf::checked_mul(6, a2));
  reverse_workspace_elements =
      posthf::checked_add(reverse_workspace_elements, posthf::checked_mul(2, auxiliaries));
  reverse_workspace_elements = posthf::checked_add(reverse_workspace_elements, n2);
  plan.reverse_workspace_bytes = posthf::checked_mul(sizeof(double), reverse_workspace_elements);

  const auto cartesian_matrix = square(cartesian_orbitals);
  const auto cartesian_three = posthf::checked_mul(cartesian_matrix, cartesian_auxiliaries);
  const auto cartesian_metric = square(cartesian_auxiliaries);
  auto derivative_elements =
      posthf::checked_add(cartesian_three, posthf::checked_add(cartesian_metric, coordinate_count));
  plan.derivative_staging_bytes = posthf::checked_mul(sizeof(double), derivative_elements);
  plan.candidate_output_bytes = candidate_output_bytes;

  plan.peak_bytes = provider_bytes;
  for (auto bytes : {plan.adjoint_bytes, plan.response_bytes, plan.relaxed_weight_bytes,
                     plan.reverse_result_bytes, plan.reverse_workspace_bytes,
                     plan.derivative_staging_bytes, plan.candidate_output_bytes})
    plan.peak_bytes = posthf::checked_add(plan.peak_bytes, bytes);
  if (plan.peak_bytes > budget_bytes)
    throw std::length_error("RI-MP2 gradient exceeds numeric memory budget");
  return plan;
}

}  // namespace vibeqc::mp2

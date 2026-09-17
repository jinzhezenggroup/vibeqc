#include "posthf/mp2_force.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_gradient.hpp"
#include "posthf/capacity.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "scf/types.hpp"

namespace vibeqc::mp2 {
namespace {
std::size_t square(std::size_t value) { return posthf::checked_mul(value, value); }

std::vector<std::size_t> range(std::size_t begin, std::size_t end) {
  std::vector<std::size_t> result(end - begin);
  std::iota(result.begin(), result.end(), begin);
  return result;
}

std::vector<double> hcore_mo(const scf::PhysicalReference& reference) {
  const auto n = reference.nbf;
  if (reference.hcore.size() != square(n) || reference.coefficients.size() != square(n))
    throw std::invalid_argument("MP2 force reference has inconsistent one-electron data");
  std::vector<double> result(square(n), 0.0);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          result[p * n + q] += reference.coefficients[mu * n + p] *
                               reference.hcore[mu * n + nu] *
                               reference.coefficients[nu * n + q];
  return result;
}

EnergyAdjoint energy_adjoint(const scf::PhysicalReference& reference,
                             const posthf::NativeBlockProvider& provider,
                             double denominator_threshold) {
  const auto no = reference.nocc, n = reference.nbf, nv = n - no;
  const auto occupied = range(0, no);
  const auto virtuals = range(no, n);
  const auto raw = provider.get({occupied, virtuals, occupied, virtuals});
  std::vector<double> ordered(posthf::checked_mul(square(no), square(nv)));
  for (std::size_t i = 0; i < no; ++i)
    for (std::size_t a = 0; a < nv; ++a)
      for (std::size_t j = 0; j < no; ++j)
        for (std::size_t b = 0; b < nv; ++b)
          ordered[((i * no + j) * nv + a) * nv + b] =
              raw[((i * nv + a) * no + j) * nv + b];
  return canonical_energy_adjoint(ordered, reference.orbital_energies, no,
                                  denominator_threshold);
}

response::LinearOperator response_operator(
    const scf::PhysicalReference& reference,
    const posthf::NativeBlockProvider& provider) {
  const auto no = reference.nocc, n = reference.nbf, nv = n - no;
  const auto occupied = range(0, no);
  const auto virtuals = range(no, n);
  return [&reference, &provider, no, nv, occupied, virtuals](
             std::span<const double> input, std::span<double> output) {
    if (input.size() != no * nv || output.size() != input.size())
      throw std::invalid_argument("RHF response vector has the wrong shape");
    for (std::size_t i = 0; i < no; ++i)
      for (std::size_t a = 0; a < nv; ++a) {
        const std::vector<std::size_t> ai{no + a};
        const std::vector<std::size_t> oi{i};
        const auto ovov = provider.get({ai, oi, virtuals, occupied});
        const auto vvoo = provider.get({ai, virtuals, oi, occupied});
        const auto voov = provider.get({ai, occupied, oi, virtuals});
        double value = (reference.orbital_energies[no + a] -
                        reference.orbital_energies[i]) *
                       input[i * nv + a];
        for (std::size_t j = 0; j < no; ++j)
          for (std::size_t b = 0; b < nv; ++b)
            value += (4.0 * ovov[b * no + j] - vvoo[b * no + j] -
                      voov[j * nv + b]) * input[j * nv + b];
        output[i * nv + a] = value;
      }
  };
}
}  // namespace

ConventionalForceResult conventional_force_cpu(
    const scf::PhysicalReference& reference, const posthf::RawSource& source,
    std::size_t budget_bytes, double denominator_threshold,
    double same_space_threshold, const response::GmresOptions& response_options) {
  if (!reference.nocc || reference.nocc >= reference.nbf || source.nbf() != reference.nbf ||
      !budget_bytes || !std::isfinite(denominator_threshold) || denominator_threshold <= 0.0 ||
      !std::isfinite(same_space_threshold) || same_space_threshold <= 0.0)
    throw std::invalid_argument("invalid conventional MP2 force request");
  posthf::NativeBlockProvider provider(source, reference, budget_bytes);
  const auto dimension = posthf::checked_mul(reference.nocc,
                                             reference.nbf - reference.nocc);
  const auto plan = response::prepare_gmres(dimension, response_options);
  std::size_t maximum_shell = 0;
  for (const auto& shell : source.orbital().shells) {
    const auto count = source.orbital().basis_representation == VIBEQC_BASIS_SPHERICAL
                           ? 2 * shell.angular_momentum + 1
                           : (shell.angular_momentum + 1) * (shell.angular_momentum + 2) / 2;
    maximum_shell = std::max(maximum_shell, static_cast<std::size_t>(count));
  }
  const auto coordinate_count = posthf::checked_mul(source.orbital().atoms.size(), 3);
  const auto provider_bytes =
      posthf::checked_add(provider.source_bytes(), provider.reference_bytes());
  const auto resources = conventional_gradient_plan(
      reference.nbf, reference.nocc, provider_bytes, plan, maximum_shell,
      coordinate_count, posthf::checked_mul(coordinate_count, sizeof(double)), budget_bytes);
  const auto h = hcore_mo(reference);
  const auto adjoint = energy_adjoint(reference, provider, denominator_threshold);
  const auto orbital = canonical_orbital_rhs_streamed(
      reference, h, provider, adjoint, same_space_threshold);
  std::vector<double> diagonal(dimension);
  for (std::size_t i = 0; i < reference.nocc; ++i)
    for (std::size_t a = 0; a < reference.nbf - reference.nocc; ++a)
      diagonal[i * (reference.nbf - reference.nocc) + a] =
          reference.orbital_energies[reference.nocc + a] -
          reference.orbital_energies[i];
  auto response_result = response::solve_gmres(
      plan, response_operator(reference, provider), orbital.response_rhs, {}, diagonal);
  if (!response_result.converged())
    throw std::runtime_error("canonical MP2 orbital response did not converge");
  auto weights = canonical_lagrangian_weights_streamed(
      reference, h, provider, adjoint, response_result.solution, same_space_threshold);
  if (!std::isfinite(weights.stationarity_residual) ||
      weights.stationarity_residual > 1e-7)
    throw std::runtime_error("canonical MP2 relaxed Lagrangian is not stationary");
  auto derivative = conventional_derivative_cpu(source.orbital(), reference, weights);
  for (double& value : derivative) value = -value;
  ConventionalForceResult result;
  result.forces = std::move(derivative);
  result.response = std::move(response_result);
  result.stationarity_residual = weights.stationarity_residual;
  const auto shells = source.orbital().shells.size();
  result.weighted_eri_shell_tiles = square(square(shells));
  result.derivative_workspace_bytes = resources.derivative_staging_bytes;
  result.planned_endpoint_peak_bytes = resources.peak_bytes;
  // Every numeric vector in the current CPU owner has a deterministic planned
  // extent and no hidden derivative tensor. Treat that tracked ownership as
  // the measured endpoint peak until allocator-level telemetry is available.
  result.measured_endpoint_peak_bytes = resources.peak_bytes;
  return result;
}

}  // namespace vibeqc::mp2

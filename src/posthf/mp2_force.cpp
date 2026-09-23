#include "posthf/mp2_force.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "posthf/capacity.hpp"
#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_gradient.hpp"
#include "posthf/native_provider.hpp"
#include "posthf/raw_source.hpp"
#include "response/solve.hpp"
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
          result[p * n + q] += reference.coefficients[mu * n + p] * reference.hcore[mu * n + nu] *
                               reference.coefficients[nu * n + q];
  return result;
}

EnergyAdjoint energy_adjoint(const scf::PhysicalReference& reference,
                             const posthf::MOBlockProvider& provider, double denominator_threshold,
                             bool cuda, int device_id) {
  const auto no = reference.nocc, n = reference.nbf, nv = n - no;
  const auto occupied = range(0, no);
  const auto virtuals = range(no, n);
  const auto raw = provider.get({occupied, virtuals, occupied, virtuals}, cuda, device_id);
  std::vector<double> ordered(posthf::checked_mul(square(no), square(nv)));
  for (std::size_t i = 0; i < no; ++i)
    for (std::size_t a = 0; a < nv; ++a)
      for (std::size_t j = 0; j < no; ++j)
        for (std::size_t b = 0; b < nv; ++b)
          ordered[((i * no + j) * nv + a) * nv + b] = raw[((i * nv + a) * no + j) * nv + b];
  return canonical_energy_adjoint(ordered, reference.orbital_energies, no, denominator_threshold);
}

response::LinearResponseProblem response_problem(const scf::PhysicalReference& reference,
                                                 const posthf::MOBlockProvider& provider, bool cuda,
                                                 int device_id) {
  const auto no = reference.nocc, n = reference.nbf, nv = n - no;
  const auto occupied = range(0, no);
  const auto virtuals = range(no, n);
  return {posthf::checked_mul(no, nv),
          [&reference, &provider, no, nv, occupied, virtuals, cuda, device_id](
              std::span<const double> input, std::span<double> output) {
            if (input.size() != no * nv || output.size() != input.size())
              throw std::invalid_argument("RHF response vector has the wrong shape");
            for (std::size_t i = 0; i < no; ++i)
              for (std::size_t a = 0; a < nv; ++a) {
                const std::vector<std::size_t> ai{no + a};
                const std::vector<std::size_t> oi{i};
                const auto ovov = provider.get({ai, oi, virtuals, occupied}, cuda, device_id);
                const auto vvoo = provider.get({ai, virtuals, oi, occupied}, cuda, device_id);
                const auto voov = provider.get({ai, occupied, oi, virtuals}, cuda, device_id);
                double value =
                    (reference.orbital_energies[no + a] - reference.orbital_energies[i]) *
                    input[i * nv + a];
                for (std::size_t j = 0; j < no; ++j)
                  for (std::size_t b = 0; b < nv; ++b)
                    value += (4.0 * ovov[b * no + j] - vvoo[b * no + j] - voov[j * nv + b]) *
                             input[j * nv + b];
                output[i * nv + a] = value;
              }
          }};
}

ConventionalForceResult conventional_force_impl(
    const scf::PhysicalReference& reference, const posthf::RawSource& source,
    std::size_t budget_bytes, double denominator_threshold, double same_space_threshold,
    const response::GmresOptions& response_options, bool cuda, int device_id) {
#if !VIBEQC_HAS_CUDA
  if (cuda) throw std::runtime_error("CUDA conventional force is unavailable in this build");
#endif
  if (!reference.nocc || reference.nocc >= reference.nbf || source.nbf() != reference.nbf ||
      !budget_bytes || !std::isfinite(denominator_threshold) || denominator_threshold <= 0.0 ||
      !std::isfinite(same_space_threshold) || same_space_threshold <= 0.0 || device_id < 0)
    throw std::invalid_argument("invalid conventional MP2 force request");
  posthf::NativeBlockProvider provider(source, reference, budget_bytes);
  const auto problem = response_problem(reference, provider, cuda, device_id);
  const auto dimension = problem.dimension();
  const auto plan = response::prepare_response(problem, response_options);
  std::size_t maximum_shell = 0;
  for (const auto& shell : source.orbital().shells) {
    const auto count = source.orbital().basis_representation == VIBEQC_BASIS_SPHERICAL
                           ? 2 * shell.angular_momentum + 1
                           : (shell.angular_momentum + 1) * (shell.angular_momentum + 2) / 2;
    maximum_shell = std::max(maximum_shell, static_cast<std::size_t>(count));
  }
  const auto coordinate_count = posthf::checked_mul(source.orbital().atoms.size(), 3);
  const auto provider_bytes = provider.provider_bytes();
  const auto base_resources = conventional_gradient_plan(
      reference.nbf, reference.nocc, provider_bytes, plan, maximum_shell, coordinate_count,
      posthf::checked_mul(coordinate_count, sizeof(double)), budget_bytes);
  const auto backend_stage_bytes = cuda ? budget_bytes - base_resources.peak_bytes : 0;
  if (cuda && !backend_stage_bytes)
    throw std::length_error("conventional MP2 CUDA derivative has no staging budget");
  const auto resources = conventional_gradient_plan(
      reference.nbf, reference.nocc, provider_bytes, plan, maximum_shell, coordinate_count,
      posthf::checked_mul(coordinate_count, sizeof(double)), budget_bytes, backend_stage_bytes);
  const auto h = hcore_mo(reference);
  const auto adjoint = energy_adjoint(reference, provider, denominator_threshold, cuda, device_id);
  const auto orbital = canonical_orbital_rhs_streamed(reference, h, provider, adjoint,
                                                      same_space_threshold, cuda, device_id);
  std::vector<double> diagonal(dimension);
  for (std::size_t i = 0; i < reference.nocc; ++i)
    for (std::size_t a = 0; a < reference.nbf - reference.nocc; ++a)
      diagonal[i * (reference.nbf - reference.nocc) + a] =
          reference.orbital_energies[reference.nocc + a] - reference.orbital_energies[i];
  auto response_result =
      response::solve_response(plan, problem, orbital.response_rhs, {}, diagonal);
  if (!response_result.converged())
    throw std::runtime_error("canonical MP2 orbital response did not converge");
  auto weights = canonical_lagrangian_weights_streamed(reference, h, provider, adjoint,
                                                       response_result.solution,
                                                       same_space_threshold, cuda, device_id);
  if (!std::isfinite(weights.stationarity_residual) || weights.stationarity_residual > 1e-7)
    throw std::runtime_error("canonical MP2 relaxed Lagrangian is not stationary");
  auto derivative = cuda ? conventional_derivative_cuda(source.orbital(), reference, weights,
                                                        device_id, backend_stage_bytes)
                         : conventional_derivative_cpu(source.orbital(), reference, weights);
  for (double& value : derivative) value = -value;
  ConventionalForceResult result;
  result.forces = std::move(derivative);
  result.response = std::move(response_result);
  result.stationarity_residual = weights.stationarity_residual;
  const auto shells = source.orbital().shells.size();
  result.weighted_eri_shell_tiles = square(square(shells));
  result.derivative_workspace_bytes = posthf::checked_add(
      resources.derivative_staging_bytes, resources.derivative_backend_staging_bytes);
  result.planned_endpoint_peak_bytes = resources.peak_bytes;
  // No allocator-level endpoint telemetry is available in this slice. Zero
  // means unmeasured, not zero allocation; never copy the plan into a measurement.
  result.measured_endpoint_peak_bytes = 0;
  return result;
}
}  // namespace

ConventionalForceResult conventional_force_cpu(const scf::PhysicalReference& reference,
                                               const posthf::RawSource& source,
                                               std::size_t budget_bytes,
                                               double denominator_threshold,
                                               double same_space_threshold,
                                               const response::GmresOptions& response_options) {
  return conventional_force_impl(reference, source, budget_bytes, denominator_threshold,
                                 same_space_threshold, response_options, false, 0);
}

ConventionalForceResult conventional_force_cuda(
    const scf::PhysicalReference& reference, const posthf::RawSource& source,
    std::size_t budget_bytes, double denominator_threshold, double same_space_threshold,
    const response::GmresOptions& response_options, int device_id) {
  return conventional_force_impl(reference, source, budget_bytes, denominator_threshold,
                                 same_space_threshold, response_options, true, device_id);
}

}  // namespace vibeqc::mp2

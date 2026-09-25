#include "dft/nonlocal_correlation/vv10_integration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "runtime/resource_usage.hpp"
#include "xc_cpu_generated.hpp"

namespace vibeqc::dft::nlc {
namespace {

std::size_t matrix_size(std::size_t n) {
  if (!n || n > std::numeric_limits<std::size_t>::max() / n)
    throw std::invalid_argument("invalid VV10 AO dimension");
  return n * n;
}

void validate_density(const AoBasis& basis, const MolecularGrid& grid,
                      const std::vector<double>& density, std::size_t tile_points) {
  const auto n = basis.nao;
  if (!tile_points) throw std::invalid_argument("VV10 AO tile size must be positive");
  if (grid.point_count() == 0 || grid.system().atoms.size() != basis.natom)
    throw std::invalid_argument("VV10 grid and AO basis are incompatible");
  if (density.size() != matrix_size(n))
    throw std::invalid_argument("VV10 density dimensions do not match the AO basis");
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = 0; nu < n; ++nu) {
      const auto value = density[mu * n + nu];
      const auto transpose = density[nu * n + mu];
      if (!std::isfinite(value)) throw std::invalid_argument("VV10 AO density must be finite");
      if (std::abs(value - transpose) >
          1.0e-12 + 1.0e-10 * std::max(std::abs(value), std::abs(transpose)))
        throw std::invalid_argument("VV10 AO density must be symmetric");
    }
}

void accumulate_features(const double* phi, const double* dx, const double* dy, const double* dz,
                         std::size_t n, const std::vector<double>& density, double& rho,
                         double* gradient) {
  rho = 0.0;
  gradient[0] = gradient[1] = gradient[2] = 0.0;
  for (std::size_t mu = 0; mu < n; ++mu) {
    double weighted = 0.0;
    for (std::size_t nu = 0; nu < n; ++nu) weighted += density[mu * n + nu] * phi[nu];
    rho += phi[mu] * weighted;
    gradient[0] += 2.0 * dx[mu] * weighted;
    gradient[1] += 2.0 * dy[mu] * weighted;
    gradient[2] += 2.0 * dz[mu] * weighted;
  }
}

void add_potential_point(std::vector<double>& potential, const double* phi, const double* dx,
                         const double* dy, const double* dz, std::size_t n, double weight,
                         double vrho, double vsigma, const double* gradient) {
  const double spatial[3]{2.0 * vsigma * gradient[0], 2.0 * vsigma * gradient[1],
                          2.0 * vsigma * gradient[2]};
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = 0; nu < n; ++nu) {
      const double weak = spatial[0] * (dx[mu] * phi[nu] + phi[mu] * dx[nu]) +
                          spatial[1] * (dy[mu] * phi[nu] + phi[mu] * dy[nu]) +
                          spatial[2] * (dz[mu] * phi[nu] + phi[mu] * dz[nu]);
      potential[mu * n + nu] += weight * (vrho * phi[mu] * phi[nu] + weak);
    }
}

}  // namespace

Vv10Integral integrate_vv10_rks(const AoBasis& basis, const MolecularGrid& grid,
                                const std::vector<double>& density, Vv10Plan& plan,
                                std::size_t tile_points, XcDensitySource source,
                                Vv10DensityDomain domain) {
  validate_density(basis, grid, density, tile_points);
  if (domain != Vv10DensityDomain::StrictPositive && domain != Vv10DensityDomain::MolecularV1)
    throw std::invalid_argument("unknown VV10 integration density domain");
  // The host AO bridge applies the same molecular padding before either
  // resident CUDA or CPU VV10 pair execution. Keep rVV10 outside this domain.
  if (domain == Vv10DensityDomain::MolecularV1 && plan.parameters().variant != Vv10Variant::vv10)
    throw std::invalid_argument("molecular VV10 density screening requires VV10");
  if (source.route != XcDensityRoute::DensityMatrix)
    throw std::invalid_argument(
        "self-consistent VV10 currently requires the density-matrix AO route");
  const auto n = basis.nao;
  const auto points = grid.point_count();
  if (plan.resources().point_count != points)
    throw std::invalid_argument("VV10 pair plan point count does not match the KS grid");

  std::vector<double> rho(points), gradient(3 * points), vrho(points), vsigma(points);
  std::vector<double> ao;
  const auto& xyz = grid.points();
  for (std::size_t begin = 0; begin < points; begin += tile_points) {
    const auto count = std::min(tile_points, points - begin);
    ao.resize(4 * count * n);
    basis.evaluate(xyz.data() + 3 * begin, count, 1, 0, n, ao.data(), ao.size());
    for (std::size_t p = 0; p < count; ++p) {
      const auto* phi = ao.data() + p * n;
      const auto* dx = ao.data() + (count + p) * n;
      const auto* dy = ao.data() + (2 * count + p) * n;
      const auto* dz = ao.data() + (3 * count + p) * n;
      accumulate_features(phi, dx, dy, dz, n, density, rho[begin + p],
                          gradient.data() + 3 * (begin + p));
    }
  }

  // Keep the prepared pair extent fixed. Inactive points carry zero quadrature
  // weight and benign dummy features; they do not contribute to either pair
  // sum, the local beta term, or the AO potential. Physical active features are
  // untouched. Validate BEFORE padding so screening cannot conceal bad input.
  std::vector<double> screened_weights;
  if (domain == Vv10DensityDomain::MolecularV1) {
    screened_weights = grid.weights();
    for (std::size_t p = 0; p < points; ++p) {
      if (!std::isfinite(rho[p]) || rho[p] < 0.0 || !std::isfinite(gradient[3 * p]) ||
          !std::isfinite(gradient[3 * p + 1]) || !std::isfinite(gradient[3 * p + 2]) ||
          !std::isfinite(screened_weights[p]))
        throw std::domain_error("invalid molecular VV10 density/gradient/weight");
      if (rho[p] < generated::kMolecularVv10DensityThreshold) {
        screened_weights[p] = 0.0;
        rho[p] = 1.0;
        gradient[3 * p] = gradient[3 * p + 1] = gradient[3 * p + 2] = 0.0;
      }
    }
  }
  const auto& effective_weights = screened_weights.empty() ? grid.weights() : screened_weights;
  double energy = 0.0;
  std::string detail;
  const auto status = plan.execute(xyz, effective_weights, rho, gradient, energy, vrho, vsigma,
                                   std::span<double>{}, std::span<double>{}, detail);
  if (status != VIBEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail.empty() ? "VV10 pair execution failed" : detail);

  Vv10Integral result;
  result.energy = energy;
  result.points = points;
  result.potential.assign(matrix_size(n), 0.0);
  const auto& weights = effective_weights;
  for (std::size_t begin = 0; begin < points; begin += tile_points) {
    const auto count = std::min(tile_points, points - begin);
    ao.resize(4 * count * n);
    basis.evaluate(xyz.data() + 3 * begin, count, 1, 0, n, ao.data(), ao.size());
    for (std::size_t p = 0; p < count; ++p) {
      const auto global = begin + p;
      const auto* phi = ao.data() + p * n;
      const auto* dx = ao.data() + (count + p) * n;
      const auto* dy = ao.data() + (2 * count + p) * n;
      const auto* dz = ao.data() + (3 * count + p) * n;
      add_potential_point(result.potential, phi, dx, dy, dz, n, weights[global], vrho[global],
                          vsigma[global], gradient.data() + 3 * global);
    }
  }
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = mu + 1; nu < n; ++nu) {
      const auto average = 0.5 * (result.potential[mu * n + nu] + result.potential[nu * n + mu]);
      result.potential[mu * n + nu] = result.potential[nu * n + mu] = average;
    }
  if (!std::isfinite(result.energy) ||
      !std::all_of(result.potential.begin(), result.potential.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("nonfinite self-consistent VV10 AO contribution");

  result.owned_numeric_bytes = runtime::vector_capacities(rho, gradient, vrho, vsigma, ao,
                                                          result.potential, screened_weights);
  return result;
}

SpinVv10Integral integrate_vv10_uks(const AoBasis& basis, const MolecularGrid& grid,
                                    const std::vector<double>& alpha_density,
                                    const std::vector<double>& beta_density, Vv10Plan& plan,
                                    std::size_t tile_points, Vv10DensityDomain domain) {
  validate_density(basis, grid, alpha_density, tile_points);
  validate_density(basis, grid, beta_density, tile_points);
  if (alpha_density.size() != beta_density.size())
    throw std::invalid_argument("VV10 UKS spin densities must have matching dimensions");
  std::vector<double> total(alpha_density.size());
  for (std::size_t i = 0; i < total.size(); ++i) total[i] = alpha_density[i] + beta_density[i];
  auto common = integrate_vv10_rks(basis, grid, total, plan, tile_points, {}, domain);

  SpinVv10Integral result;
  result.energy = common.energy;
  result.points = common.points;
  result.potential[0] = common.potential;
  result.potential[1] = std::move(common.potential);
  result.owned_numeric_bytes = runtime::add_capacity(
      common.owned_numeric_bytes, runtime::vector_capacities(total, result.potential[0]));
  return result;
}

}  // namespace vibeqc::dft::nlc

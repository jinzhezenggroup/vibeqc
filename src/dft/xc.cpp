#include "dft/xc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "dft/xc_point.hpp"
#include "runtime/resource_usage.hpp"
#include "xc_cpu_generated.hpp"

namespace vibeqc::dft {
namespace {

std::size_t matrix_size(std::size_t n) {
  if (!n || n > std::numeric_limits<std::size_t>::max() / n)
    throw std::invalid_argument("invalid XC AO dimension");
  return n * n;
}

void validate_density_matrix(const AoBasis& basis, const MolecularGrid& grid,
                             const std::vector<double>& density, std::size_t tile_points) {
  const std::size_t n = basis.nao;
  if (!tile_points) throw std::invalid_argument("XC tile size must be positive");
  if (grid.system().atoms.size() != basis.natom)
    throw std::invalid_argument("XC grid and AO basis atom counts differ");
  if (density.size() != matrix_size(n))
    throw std::invalid_argument("XC density dimensions do not match the AO basis");
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double value = density[i * n + j];
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite XC density matrix");
      if (std::abs(value - density[j * n + i]) >
          1.0e-12 + 1.0e-10 * std::max(std::abs(value), std::abs(density[j * n + i])))
        throw std::invalid_argument("XC density matrix must be symmetric");
    }
  }
}

/** Resolve once per XC call: an O(nAO^2) exact witness check, never an
 * O(nAO^2*nocc) reconstruction or density eigendecomposition. */
const scf::OccupiedDensityFactor* resolve_density_source(std::size_t n,
                                                         const std::vector<double>& density,
                                                         XcDensitySource source,
                                                         XcDensityDiagnostic& diagnostic) {
  diagnostic.requested = source.route;
  if (source.role != XcDensityRole::State && source.role != XcDensityRole::Response)
    throw std::invalid_argument("unsupported XC density role");
  diagnostic.active_ao = n;
  diagnostic.borrowed_density_bytes = runtime::vector_bytes(density);
  if (source.factor) {
    diagnostic.nocc = source.factor->rank();
    diagnostic.borrowed_factor_bytes = source.factor->numeric_capacity_bytes();
  }
  if (source.route == XcDensityRoute::DensityMatrix) return nullptr;
  if (source.route != XcDensityRoute::OccupiedOrbitals)
    throw std::invalid_argument("unsupported XC density route");
  if (source.role == XcDensityRole::Response)
    diagnostic.fallback = XcDensityFallback::Response;
  else if (!source.factor)
    diagnostic.fallback = XcDensityFallback::MissingFactor;
  else if (source.factor->nbf() != n)
    diagnostic.fallback = XcDensityFallback::Basis;
  else if (source.factor->identity() != source.identity)
    diagnostic.fallback = XcDensityFallback::Identity;
  else if (source.factor->spin() != scf::DensityFactorSpin::Restricted)
    diagnostic.fallback = XcDensityFallback::Spin;
  else if (!source.factor->matches(source.identity, scf::DensityFactorSpin::Restricted, density))
    diagnostic.fallback = XcDensityFallback::Density;
  else {
    diagnostic.executed = XcDensityRoute::OccupiedOrbitals;
    return source.factor;
  }
  return nullptr;
}

/** Same total-density features for both algorithms. C uses the generated
 * bilinears with B=C*sqrt(f), so closed-shell occupation is already included.
 * Its four scalar orbital jets are reduced immediately: no grid-by-orbital
 * array survives a point. The established D contraction retains all symmetric
 * cross terms, including for matrices accepted within symmetry tolerance.
 */
std::array<double, 5> rks_features(const double* phi,
                                   const std::array<const double*, 3>& derivatives, std::size_t n,
                                   const std::vector<double>& density,
                                   const scf::OccupiedDensityFactor* factor, bool need_gradient) {
  std::array<double, 5> features{};
  if (factor) {
    for (std::size_t o = 0; o < factor->rank(); ++o) {
      double work[4]{};
      for (std::size_t mu = 0; mu < n; ++mu) {
        const double b = factor->values()[mu * factor->rank() + o];
        work[0] += phi[mu] * b;
        if (need_gradient)
          for (unsigned axis = 0; axis < 3; ++axis) work[axis + 1] += derivatives[axis][mu] * b;
      }
      generated::add_features(work[0], work + 1, work, features.data(), need_gradient ? 3 : 1);
    }
  } else {
    for (std::size_t mu = 0; mu < n; ++mu) {
      for (std::size_t nu = 0; nu < n; ++nu) {
        const double d = density[mu * n + nu];
        features[0] += phi[mu] * d * phi[nu];
        if (need_gradient)
          for (unsigned axis = 0; axis < 3; ++axis)
            features[axis + 1] +=
                (derivatives[axis][mu] * phi[nu] + phi[mu] * derivatives[axis][nu]) * d;
      }
    }
  }
  return features;
}

void sample_xc_capacity(XcIntegral& result, const std::vector<double>& ao, std::size_t count) {
  auto& record = result.density_diagnostic;
  record.max_tile_points = std::max(record.max_tile_points, count);
  record.owned_numeric_bytes =
      std::max(record.owned_numeric_bytes, runtime::vector_capacities(ao, result.potential));
}

}  // namespace

XcIntegral integrate_lda_xc_pw_rks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& density, std::size_t tile_points,
                                   XcDensitySource source) {
  const std::size_t n = basis.nao;
  validate_density_matrix(basis, grid, density, tile_points);

  XcIntegral result;
  result.potential.assign(n * n, 0.0);
  result.points = grid.point_count();
  auto& record = result.density_diagnostic;
  record.npoint = result.points;
  record.ingredient_mask = 1;
  const auto* factor = resolve_density_source(n, density, source, record);
  std::vector<double> ao;
  const auto& points = grid.points();
  const auto& weights = grid.weights();
  for (std::size_t begin = 0; begin < result.points; begin += tile_points) {
    const std::size_t count = std::min(tile_points, result.points - begin);
    ao.resize(count * n);
    sample_xc_capacity(result, ao, count);
    basis.evaluate(points.data() + 3 * begin, count, 0, 0, n, ao.data(), ao.size());
    for (std::size_t point = 0; point < count; ++point) {
      const double* phi = ao.data() + point * n;
      const double rho = rks_features(phi, {}, n, density, factor, false)[0];
      if (!std::isfinite(rho) || rho < 0.0)
        throw std::domain_error("LDA tail-v1 requires finite nonnegative density");
      if (rho == 0.0) continue;
      const auto xc = generated::lda_xc_pw_unpolarized(rho);
      if (!std::isfinite(xc.energy_density) || !std::isfinite(xc.density_derivative))
        throw std::runtime_error("nonfinite generated LDA value");
      const double weight = weights[begin + point];
      result.energy += weight * xc.energy_density;
      result.electrons += weight * rho;
      const double coefficient = weight * xc.density_derivative;
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          result.potential[mu * n + nu] += coefficient * phi[mu] * phi[nu];
    }
  }
  return result;
}

namespace {
/** The same tile traversal, density convention and point evaluator serve both
 * spin functionals. Each quadrature weight is applied once to E and V. */
SpinXcIntegral integrate_spin_xc(const AoBasis& basis, const MolecularGrid& grid,
                                 const std::vector<double>& alpha_density,
                                 const std::vector<double>& beta_density, std::size_t tile_points,
                                 bool pbe, double exchange_scale = 1.0,
                                 double correlation_scale = 1.0) {
  validate_density_matrix(basis, grid, alpha_density, tile_points);
  validate_density_matrix(basis, grid, beta_density, tile_points);
  const std::size_t n = basis.nao;
  SpinXcIntegral result;
  for (auto& potential : result.potential) potential.assign(n * n, 0.0);
  result.points = grid.point_count();
  std::vector<double> ao;
  const std::vector<double>* densities[2]{&alpha_density, &beta_density};
  for (std::size_t begin = 0; begin < result.points; begin += tile_points) {
    const std::size_t count = std::min(tile_points, result.points - begin);
    ao.resize((pbe ? 4 : 1) * count * n);
    basis.evaluate(grid.points().data() + 3 * begin, count, pbe ? 1 : 0, 0, n, ao.data(),
                   ao.size());
    for (std::size_t p = 0; p < count; ++p) {
      const double* phi = ao.data() + p * n;
      std::array<const double*, 3> jets{};
      if (pbe)
        for (unsigned k = 0; k < 3; ++k) jets[k] = ao.data() + ((k + 1) * count + p) * n;
      double rho[2]{}, gradient[2][3]{};
      for (unsigned spin = 0; spin < 2; ++spin) {
        const auto features = rks_features(phi, jets, n, *densities[spin], nullptr, pbe);
        rho[spin] = features[0];
        for (unsigned k = 0; k < 3; ++k) gradient[spin][k] = features[k + 1];
      }
      const auto xc = point::evaluate(pbe, rho, gradient, exchange_scale, correlation_scale);
      if (!xc.valid) throw std::domain_error("invalid or unrepresentable semilocal spin features");
      const double weight = grid.weights()[begin + p];
      result.energy += weight * xc.energy;
      for (unsigned spin = 0; spin < 2; ++spin) {
        result.electrons[spin] += weight * rho[spin];
        for (std::size_t mu = 0; mu < n; ++mu) {
          for (std::size_t nu = 0; nu < n; ++nu) {
            double value = xc.rho[spin] * phi[mu] * phi[nu];
            if (pbe)
              for (unsigned k = 0; k < 3; ++k)
                value += xc.gradient[spin][k] * (jets[k][mu] * phi[nu] + phi[mu] * jets[k][nu]);
            result.potential[spin][mu * n + nu] += weight * value;
          }
        }
      }
    }
  }
  if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite spin XC integral");
  return result;
}
}  // namespace

SpinXcIntegral integrate_lda_xc_pw_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       std::size_t tile_points) {
  return integrate_spin_xc(basis, grid, alpha_density, beta_density, tile_points, false);
}

SpinXcIntegral integrate_pbe_uks_scaled(const AoBasis& basis, const MolecularGrid& grid,
                                        const std::vector<double>& alpha_density,
                                        const std::vector<double>& beta_density,
                                        std::size_t tile_points, double exchange_scale,
                                        double correlation_scale) {
  return integrate_spin_xc(basis, grid, alpha_density, beta_density, tile_points, true,
                           exchange_scale, correlation_scale);
}

SpinXcIntegral integrate_pbe_uks(const AoBasis& basis, const MolecularGrid& grid,
                                 const std::vector<double>& alpha_density,
                                 const std::vector<double>& beta_density, std::size_t tile_points) {
  return integrate_pbe_uks_scaled(basis, grid, alpha_density, beta_density, tile_points, 1.0, 1.0);
}

XcIntegral integrate_pbe_rks_impl(const AoBasis& basis, const MolecularGrid& grid,
                                  const std::vector<double>& density, std::size_t tile_points,
                                  bool allow_tail, XcDensitySource source,
                                  double exchange_scale = 1.0, double correlation_scale = 1.0) {
  const std::size_t n = basis.nao;
  validate_density_matrix(basis, grid, density, tile_points);

  XcIntegral result;
  result.potential.assign(n * n, 0.0);
  result.points = grid.point_count();
  auto& record = result.density_diagnostic;
  record.npoint = result.points;
  record.ingredient_mask = 3;
  const auto* factor = resolve_density_source(n, density, source, record);
  std::vector<double> ao;
  const auto& points = grid.points();
  const auto& weights = grid.weights();
  for (std::size_t begin = 0; begin < result.points; begin += tile_points) {
    const std::size_t count = std::min(tile_points, result.points - begin);
    ao.resize(4 * count * n);
    sample_xc_capacity(result, ao, count);
    basis.evaluate(points.data() + 3 * begin, count, 1, 0, n, ao.data(), ao.size());
    for (std::size_t point = 0; point < count; ++point) {
      const double* phi = ao.data() + point * n;
      const double* grad_x = ao.data() + (count + point) * n;
      const double* grad_y = ao.data() + (2 * count + point) * n;
      const double* grad_z = ao.data() + (3 * count + point) * n;
      const auto features = rks_features(phi, {grad_x, grad_y, grad_z}, n, density, factor, true);
      const double rho = features[0];
      const std::array<double, 3> gradient{features[1], features[2], features[3]};
      const double sigma =
          gradient[0] * gradient[0] + gradient[1] * gradient[1] + gradient[2] * gradient[2];
      if (!allow_tail && rho > 0.0 &&
          (rho < 1.0e-12 || rho > 1.0e12 || std::sqrt(sigma) / std::pow(rho, 4.0 / 3.0) > 1.0e6))
        throw std::domain_error("PBE features outside interior-v1");
      const double spin_rho[2]{rho / 2.0, rho / 2.0};
      const double spin_gradient[2][3]{{gradient[0] / 2.0, gradient[1] / 2.0, gradient[2] / 2.0},
                                       {gradient[0] / 2.0, gradient[1] / 2.0, gradient[2] / 2.0}};
      const auto xc =
          point::evaluate(true, spin_rho, spin_gradient, exchange_scale, correlation_scale);
      if (!xc.valid) throw std::domain_error("invalid or unrepresentable PBE features");
      const double weight = weights[begin + point];
      result.energy += weight * xc.energy;
      result.electrons += weight * rho;
      for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
          double value = xc.rho[0] * phi[mu] * phi[nu];
          value += xc.gradient[0][0] * (grad_x[mu] * phi[nu] + phi[mu] * grad_x[nu]) +
                   xc.gradient[0][1] * (grad_y[mu] * phi[nu] + phi[mu] * grad_y[nu]) +
                   xc.gradient[0][2] * (grad_z[mu] * phi[nu] + phi[mu] * grad_z[nu]);
          result.potential[mu * n + nu] += weight * value;
        }
      }
    }
  }
  if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite PBE energy");
  return result;
}

XcIntegral integrate_pbe_rks(const AoBasis& basis, const MolecularGrid& grid,
                             const std::vector<double>& density, std::size_t tile_points,
                             XcDensitySource source) {
  return integrate_pbe_rks_impl(basis, grid, density, tile_points, false, source);
}

XcIntegral integrate_pbe_rks_with_tail_scaled(const AoBasis& basis, const MolecularGrid& grid,
                                              const std::vector<double>& density,
                                              std::size_t tile_points, XcDensitySource source,
                                              double exchange_scale, double correlation_scale) {
  return integrate_pbe_rks_impl(basis, grid, density, tile_points, true, source, exchange_scale,
                                correlation_scale);
}

XcIntegral integrate_pbe_rks_with_tail(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& density, std::size_t tile_points,
                                       XcDensitySource source) {
  return integrate_pbe_rks_with_tail_scaled(basis, grid, density, tile_points, source, 1.0, 1.0);
}

// Keep the CPU slice's entry name bound to the same numerical contract.
SpinXcIntegral integrate_pbe_uks_with_tail(const AoBasis& basis, const MolecularGrid& grid,
                                           const std::vector<double>& alpha,
                                           const std::vector<double>& beta,
                                           std::size_t tile_points) {
  return integrate_pbe_uks(basis, grid, alpha, beta, tile_points);
}

}  // namespace vibeqc::dft

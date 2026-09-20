#include "dft/xc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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
                                   const scf::OccupiedDensityFactor* factor,
                                   unsigned ingredient_mask) {
  std::array<double, 5> features{};
  const bool need_first = (ingredient_mask & 14U) != 0;
  const bool need_tau = (ingredient_mask & 8U) != 0;
  if (factor) {
    for (std::size_t o = 0; o < factor->rank(); ++o) {
      double work[4]{};
      for (std::size_t mu = 0; mu < n; ++mu) {
        const double b = factor->values()[mu * factor->rank() + o];
        work[0] += phi[mu] * b;
        if (need_first)
          for (unsigned axis = 0; axis < 3; ++axis) work[axis + 1] += derivatives[axis][mu] * b;
      }
      generated::add_features(work[0], work + 1, work, features.data(), ingredient_mask);
    }
  } else {
    for (std::size_t mu = 0; mu < n; ++mu) {
      for (std::size_t nu = 0; nu < n; ++nu) {
        const double d = density[mu * n + nu];
        features[0] += phi[mu] * d * phi[nu];
        if (need_first)
          for (unsigned axis = 0; axis < 3; ++axis)
            features[axis + 1] +=
                (derivatives[axis][mu] * phi[nu] + phi[mu] * derivatives[axis][nu]) * d;
        if (need_tau)
          for (unsigned axis = 0; axis < 3; ++axis)
            features[4] += 0.5 * derivatives[axis][mu] * d * derivatives[axis][nu];
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
      const double rho = rks_features(phi, {}, n, density, factor, 1U)[0];
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
                                 bool pbe) {
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
        const auto features = rks_features(phi, jets, n, *densities[spin], nullptr, pbe ? 7U : 1U);
        rho[spin] = features[0];
        for (unsigned k = 0; k < 3; ++k) gradient[spin][k] = features[k + 1];
      }
      const auto xc = point::evaluate(pbe, rho, gradient);
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

SpinXcIntegral integrate_pbe_uks(const AoBasis& basis, const MolecularGrid& grid,
                                 const std::vector<double>& alpha_density,
                                 const std::vector<double>& beta_density, std::size_t tile_points) {
  return integrate_spin_xc(basis, grid, alpha_density, beta_density, tile_points, true);
}

R2scanPointValue evaluate_r2scan_point(const double rho[2], const double (&gradient)[2][3],
                                       const double tau[2]) {
  for (unsigned spin = 0; spin < 2; ++spin) {
    if (!std::isfinite(rho[spin]) || !std::isfinite(tau[spin]) || rho[spin] < 0.0 ||
        tau[spin] < 0.0)
      throw std::domain_error("r2SCAN requires finite nonnegative rho/tau");
    for (double component : gradient[spin])
      if (!std::isfinite(component))
        throw std::domain_error("r2SCAN requires finite density gradients");
  }
  const double total_density = rho[0] + rho[1];
  constexpr double tail_low = 1.0e-56;
  constexpr double tail_high = 1.0e-52;
  if (total_density <= tail_low) return {};

  double sigma[3]{};
  generated::sigma(gradient, sigma);
  auto raw =
      generated::r2scan_polarized(rho[0], rho[1], sigma[0], sigma[1], sigma[2], tau[0], tau[1]);
  if (!std::isfinite(raw.energy_density)) {
    char detail[512];
    std::snprintf(detail, sizeof(detail),
                  "r2SCAN nonfinite energy: rho=%.17e,%.17e tau=%.17e,%.17e "
                  "sigma=%.17e,%.17e,%.17e",
                  rho[0], rho[1], tau[0], tau[1], sigma[0], sigma[1], sigma[2]);
    throw std::domain_error(detail);
  }
  for (double derivative : raw.feature_derivative)
    if (!std::isfinite(derivative)) {
      char detail[512];
      std::snprintf(detail, sizeof(detail),
                    "r2SCAN nonfinite derivative: rho=%.17e,%.17e tau=%.17e,%.17e "
                    "sigma=%.17e,%.17e,%.17e",
                    rho[0], rho[1], tau[0], tau[1], sigma[0], sigma[1], sigma[2]);
      throw std::domain_error(detail);
    }
  if (total_density < tail_high) {
    const double width = tail_high - tail_low;
    const double x = (total_density - tail_low) / width;
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double scale = x3 * (10.0 + x * (-15.0 + 6.0 * x));
    const double dscale = 30.0 * x2 * (1.0 - x) * (1.0 - x) / width;
    const double unscaled_energy = raw.energy_density;
    raw.energy_density *= scale;
    raw.feature_derivative[0] = scale * raw.feature_derivative[0] + dscale * unscaled_energy;
    raw.feature_derivative[1] = scale * raw.feature_derivative[1] + dscale * unscaled_energy;
    for (unsigned i = 2; i < 7; ++i) raw.feature_derivative[i] *= scale;
  }

  R2scanPointValue out;
  out.energy = raw.energy_density;
  out.rho[0] = raw.feature_derivative[0];
  out.rho[1] = raw.feature_derivative[1];
  for (unsigned k = 0; k < 3; ++k) {
    out.gradient[0][k] = 2.0 * raw.feature_derivative[2] * gradient[0][k] +
                         raw.feature_derivative[3] * gradient[1][k];
    out.gradient[1][k] = raw.feature_derivative[3] * gradient[0][k] +
                         2.0 * raw.feature_derivative[4] * gradient[1][k];
  }
  // tau_s = 1/2 sum_mn D_s,mn grad(phi_m).grad(phi_n).
  out.kinetic[0] = 0.5 * raw.feature_derivative[5];
  out.kinetic[1] = 0.5 * raw.feature_derivative[6];
  return out;
}

XcIntegral integrate_r2scan_rks(const AoBasis& basis, const MolecularGrid& grid,
                                const std::vector<double>& density, std::size_t tile_points,
                                XcDensitySource source) {
  const std::size_t n = basis.nao;
  validate_density_matrix(basis, grid, density, tile_points);
  XcIntegral result;
  result.potential.assign(n * n, 0.0);
  result.points = grid.point_count();
  auto& record = result.density_diagnostic;
  record.npoint = result.points;
  record.ingredient_mask = 15U;
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
      std::array<const double*, 3> jets{};
      for (unsigned k = 0; k < 3; ++k) jets[k] = ao.data() + ((k + 1) * count + point) * n;
      const auto total = rks_features(phi, jets, n, density, factor, 15U);
      const double rho[2]{0.5 * total[0], 0.5 * total[0]};
      const double gradient[2][3]{{0.5 * total[1], 0.5 * total[2], 0.5 * total[3]},
                                  {0.5 * total[1], 0.5 * total[2], 0.5 * total[3]}};
      const double tau[2]{0.5 * total[4], 0.5 * total[4]};
      const auto xc = evaluate_r2scan_point(rho, gradient, tau);
      const double weight = weights[begin + point];
      result.energy += weight * xc.energy;
      result.electrons += weight * total[0];
      const double rho_coefficient = 0.5 * (xc.rho[0] + xc.rho[1]);
      const double kinetic_coefficient = 0.5 * (xc.kinetic[0] + xc.kinetic[1]);
      double gradient_coefficient[3]{};
      for (unsigned k = 0; k < 3; ++k)
        gradient_coefficient[k] = 0.5 * (xc.gradient[0][k] + xc.gradient[1][k]);
      for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
          double value = rho_coefficient * phi[mu] * phi[nu];
          for (unsigned k = 0; k < 3; ++k) {
            value += gradient_coefficient[k] * (jets[k][mu] * phi[nu] + phi[mu] * jets[k][nu]);
            value += kinetic_coefficient * jets[k][mu] * jets[k][nu];
          }
          result.potential[mu * n + nu] += weight * value;
        }
      }
    }
  }
  if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite r2SCAN RKS energy");
  return result;
}

SpinXcIntegral integrate_r2scan_uks(const AoBasis& basis, const MolecularGrid& grid,
                                    const std::vector<double>& alpha_density,
                                    const std::vector<double>& beta_density,
                                    std::size_t tile_points) {
  validate_density_matrix(basis, grid, alpha_density, tile_points);
  validate_density_matrix(basis, grid, beta_density, tile_points);
  const std::size_t n = basis.nao;
  SpinXcIntegral result;
  for (auto& potential : result.potential) potential.assign(n * n, 0.0);
  result.points = grid.point_count();
  const std::vector<double>* densities[2]{&alpha_density, &beta_density};
  std::vector<double> ao;
  for (std::size_t begin = 0; begin < result.points; begin += tile_points) {
    const std::size_t count = std::min(tile_points, result.points - begin);
    ao.resize(4 * count * n);
    basis.evaluate(grid.points().data() + 3 * begin, count, 1, 0, n, ao.data(), ao.size());
    for (std::size_t point = 0; point < count; ++point) {
      const double* phi = ao.data() + point * n;
      std::array<const double*, 3> jets{};
      for (unsigned k = 0; k < 3; ++k) jets[k] = ao.data() + ((k + 1) * count + point) * n;
      double rho[2]{}, gradient[2][3]{}, tau[2]{};
      for (unsigned spin = 0; spin < 2; ++spin) {
        const auto features = rks_features(phi, jets, n, *densities[spin], nullptr, 15U);
        rho[spin] = features[0];
        for (unsigned k = 0; k < 3; ++k) gradient[spin][k] = features[k + 1];
        tau[spin] = features[4];
      }
      const auto xc = evaluate_r2scan_point(rho, gradient, tau);
      const double weight = grid.weights()[begin + point];
      result.energy += weight * xc.energy;
      for (unsigned spin = 0; spin < 2; ++spin) {
        result.electrons[spin] += weight * rho[spin];
        for (std::size_t mu = 0; mu < n; ++mu) {
          for (std::size_t nu = 0; nu < n; ++nu) {
            double value = xc.rho[spin] * phi[mu] * phi[nu];
            for (unsigned k = 0; k < 3; ++k) {
              value += xc.gradient[spin][k] * (jets[k][mu] * phi[nu] + phi[mu] * jets[k][nu]);
              value += xc.kinetic[spin] * jets[k][mu] * jets[k][nu];
            }
            result.potential[spin][mu * n + nu] += weight * value;
          }
        }
      }
    }
  }
  if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite r2SCAN UKS energy");
  return result;
}

XcIntegral integrate_pbe_rks_impl(const AoBasis& basis, const MolecularGrid& grid,
                                  const std::vector<double>& density, std::size_t tile_points,
                                  bool allow_tail, XcDensitySource source) {
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
      const auto features = rks_features(phi, {grad_x, grad_y, grad_z}, n, density, factor, 7U);
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
      const auto xc = point::evaluate(true, spin_rho, spin_gradient);
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

XcIntegral integrate_pbe_rks_with_tail(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& density, std::size_t tile_points,
                                       XcDensitySource source) {
  return integrate_pbe_rks_impl(basis, grid, density, tile_points, true, source);
}

// Keep the CPU slice's entry name bound to the same numerical contract.
SpinXcIntegral integrate_pbe_uks_with_tail(const AoBasis& basis, const MolecularGrid& grid,
                                           const std::vector<double>& alpha,
                                           const std::vector<double>& beta,
                                           std::size_t tile_points) {
  return integrate_pbe_uks(basis, grid, alpha, beta, tile_points);
}

}  // namespace vibeqc::dft

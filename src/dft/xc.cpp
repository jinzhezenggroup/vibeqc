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

point::Value evaluate_generated_lda_point(const double rho[2]) {
  point::Value out;
  for (unsigned spin = 0; spin < 2; ++spin)
    if (!std::isfinite(rho[spin]) || rho[spin] < 0.0) {
      out.valid = false;
      return out;
    }
  const double total = rho[0] + rho[1];
  if (!std::isfinite(total)) {
    out.valid = false;
    return out;
  }
  if (total == 0.0) return out;
  const auto raw = generated::lda_xc_pw_polarized_production(rho[0], rho[1]);
  out.energy = raw.energy_density;
  out.rho[0] = raw.feature_derivative[0];
  out.rho[1] = raw.feature_derivative[1];
  out.valid = std::isfinite(out.energy) && std::isfinite(out.rho[0]) && std::isfinite(out.rho[1]);
  return out;
}

point::Value evaluate_generated_pbe_point(const double rho[2], const double gradient[2][3],
                                          double exchange_scale = 1.0,
                                          double correlation_scale = 1.0) {
  point::Value out;
  if (!std::isfinite(exchange_scale) || !std::isfinite(correlation_scale) || exchange_scale < 0.0 ||
      correlation_scale < 0.0) {
    out.valid = false;
    return out;
  }
  for (unsigned spin = 0; spin < 2; ++spin) {
    if (!std::isfinite(rho[spin]) || rho[spin] < 0.0) {
      out.valid = false;
      return out;
    }
    for (unsigned axis = 0; axis < 3; ++axis)
      if (!point::detail::valid_gradient_component(rho[spin], gradient[spin][axis])) {
        out.valid = false;
        return out;
      }
  }
  const double total = rho[0] + rho[1];
  if (!std::isfinite(total)) {
    out.valid = false;
    return out;
  }
  if (total == 0.0) return out;
  const auto raw = generated::pbe_polarized_production(rho[0], rho[1], gradient, exchange_scale,
                                                       correlation_scale);
  out.energy = raw.energy_density;
  out.rho[0] = raw.rho[0];
  out.rho[1] = raw.rho[1];
  for (unsigned spin = 0; spin < 2; ++spin)
    for (unsigned axis = 0; axis < 3; ++axis) out.gradient[spin][axis] = raw.gradient[spin][axis];
  out.valid = std::isfinite(out.energy) && std::isfinite(out.rho[0]) && std::isfinite(out.rho[1]);
  for (unsigned spin = 0; spin < 2; ++spin)
    for (unsigned axis = 0; axis < 3; ++axis)
      out.valid = out.valid && std::isfinite(out.gradient[spin][axis]);
  return out;
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
        const auto features = rks_features(phi, jets, n, *densities[spin], nullptr, pbe ? 7U : 1U);
        rho[spin] = features[0];
        for (unsigned k = 0; k < 3; ++k) gradient[spin][k] = features[k + 1];
      }
      const auto xc =
          pbe ? evaluate_generated_pbe_point(rho, gradient, exchange_scale, correlation_scale)
              : evaluate_generated_lda_point(rho);
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

namespace {

std::array<double, 3> validate_gga_interior_point(const double rho[2],
                                                  const double (&gradient)[2][3],
                                                  const char* method) {
  std::array<double, 3> sigma{};
  generated::sigma(gradient, sigma.data());
  const double total = rho[0] + rho[1];
  if (!std::isfinite(total) || total < 1.0e-12 || total > 1.0e12)
    throw std::domain_error(std::string(method) + " requires interior-v1 total density");
  for (unsigned spin = 0; spin < 2; ++spin) {
    const double same_sigma = sigma[spin == 0 ? 0 : 2];
    if (!std::isfinite(rho[spin]) || rho[spin] <= 0.0 || rho[spin] / total < 1.0e-10 ||
        !std::isfinite(same_sigma) || same_sigma <= 0.0)
      throw std::domain_error(std::string(method) + " requires interior-v1 spin density/gradient");
    const double reduced = std::sqrt(same_sigma) / std::pow(rho[spin], 4.0 / 3.0);
    if (!std::isfinite(reduced) || reduced > 1.0e6)
      throw std::domain_error(std::string(method) + " reduced gradient exceeds interior-v1");
    for (double component : gradient[spin])
      if (!std::isfinite(component))
        throw std::domain_error(std::string(method) + " requires finite density gradients");
  }
  const double bound = std::sqrt(sigma[0]) * std::sqrt(sigma[2]);
  if (!std::isfinite(sigma[1]) ||
      std::abs(sigma[1]) > bound * (1.0 + 16.0 * std::numeric_limits<double>::epsilon()))
    throw std::domain_error(std::string(method) + " spin-gradient Gram matrix is invalid");
  return sigma;
}

std::array<double, 3> validate_b3lyp_production_point(const double rho[2],
                                                      const double (&gradient)[2][3]) {
  std::array<double, 3> sigma{};
  generated::sigma(gradient, sigma.data());
  const double total = rho[0] + rho[1];
  if (!std::isfinite(total) || total < 0.0 || total > 1.0e12)
    throw std::domain_error("B3LYP requires finite nonnegative total density");
  for (unsigned spin = 0; spin < 2; ++spin) {
    const double same_sigma = sigma[spin == 0 ? 0 : 2];
    if (!std::isfinite(rho[spin]) || rho[spin] < 0.0 || !std::isfinite(same_sigma) ||
        same_sigma < 0.0)
      throw std::domain_error("B3LYP requires finite nonnegative spin density/gradient");
    for (double component : gradient[spin])
      if (!std::isfinite(component))
        throw std::domain_error("B3LYP requires finite density gradients");
    if (rho[spin] == 0.0 && same_sigma != 0.0)
      throw std::domain_error("B3LYP zero spin density requires zero same-spin gradient");
  }
  const double bound = std::sqrt(sigma[0]) * std::sqrt(sigma[2]);
  if (!std::isfinite(sigma[1]) ||
      std::abs(sigma[1]) > bound * (1.0 + 16.0 * std::numeric_limits<double>::epsilon()))
    throw std::domain_error("B3LYP spin-gradient Gram matrix is invalid");
  return sigma;
}

template <class Raw>
B3GgaPointValue map_gga_point(const Raw& raw, const double (&gradient)[2][3], const char* method) {
  if (!std::isfinite(raw.energy_density))
    throw std::domain_error(std::string("nonfinite generated ") + method + " semilocal energy");
  for (double derivative : raw.feature_derivative)
    if (!std::isfinite(derivative))
      throw std::domain_error(std::string("nonfinite generated ") + method +
                              " semilocal derivative");
  B3GgaPointValue out;
  out.energy = raw.energy_density;
  out.rho[0] = raw.feature_derivative[0];
  out.rho[1] = raw.feature_derivative[1];
  for (unsigned k = 0; k < 3; ++k) {
    out.gradient[0][k] = 2.0 * raw.feature_derivative[2] * gradient[0][k] +
                         raw.feature_derivative[3] * gradient[1][k];
    out.gradient[1][k] = raw.feature_derivative[3] * gradient[0][k] +
                         2.0 * raw.feature_derivative[4] * gradient[1][k];
  }
  return out;
}

using SemilocalEvaluator = SemilocalPointEvaluator;

XcIntegral integrate_semilocal_rks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& density, std::size_t tile_points,
                                   XcDensitySource source, unsigned ingredient_mask,
                                   SemilocalEvaluator evaluate, const char* method) {
  const std::size_t n = basis.nao;
  validate_density_matrix(basis, grid, density, tile_points);
  const bool need_first = (ingredient_mask & 14U) != 0;
  const bool need_tau = (ingredient_mask & 8U) != 0;

  XcIntegral result;
  result.potential.assign(n * n, 0.0);
  result.points = grid.point_count();
  auto& record = result.density_diagnostic;
  record.npoint = result.points;
  record.ingredient_mask = ingredient_mask;
  const auto* factor = resolve_density_source(n, density, source, record);
  std::vector<double> ao;
  for (std::size_t begin = 0; begin < result.points; begin += tile_points) {
    const std::size_t count = std::min(tile_points, result.points - begin);
    ao.resize((need_first ? 4 : 1) * count * n);
    sample_xc_capacity(result, ao, count);
    basis.evaluate(grid.points().data() + 3 * begin, count, need_first ? 1 : 0, 0, n, ao.data(),
                   ao.size());
    for (std::size_t point = 0; point < count; ++point) {
      const double* phi = ao.data() + point * n;
      std::array<const double*, 3> jets{};
      if (need_first)
        for (unsigned k = 0; k < 3; ++k) jets[k] = ao.data() + ((k + 1) * count + point) * n;
      const auto total = rks_features(phi, jets, n, density, factor, ingredient_mask);
      const double rho[2]{0.5 * total[0], 0.5 * total[0]};
      double gradient[2][3]{};
      if (need_first)
        for (unsigned k = 0; k < 3; ++k) gradient[0][k] = gradient[1][k] = 0.5 * total[k + 1];
      const double tau[2]{need_tau ? 0.5 * total[4] : 0.0, need_tau ? 0.5 * total[4] : 0.0};
      const auto xc = evaluate(rho, gradient, tau);
      const double weight = grid.weights()[begin + point];
      result.energy += weight * xc.energy;
      result.electrons += weight * total[0];

      const double rho_coefficient = 0.5 * (xc.rho[0] + xc.rho[1]);
      const double kinetic_coefficient = need_tau ? 0.5 * (xc.kinetic[0] + xc.kinetic[1]) : 0.0;
      double gradient_coefficient[3]{};
      if (need_first)
        for (unsigned k = 0; k < 3; ++k)
          gradient_coefficient[k] = 0.5 * (xc.gradient[0][k] + xc.gradient[1][k]);

      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu) {
          double value = rho_coefficient * phi[mu] * phi[nu];
          if (need_first)
            for (unsigned k = 0; k < 3; ++k) {
              value += gradient_coefficient[k] * (jets[k][mu] * phi[nu] + phi[mu] * jets[k][nu]);
              if (need_tau) value += kinetic_coefficient * jets[k][mu] * jets[k][nu];
            }
          result.potential[mu * n + nu] += weight * value;
        }
    }
  }
  if (!std::isfinite(result.energy))
    throw std::runtime_error(std::string("nonfinite ") + method + " RKS semilocal energy");
  return result;
}

SpinXcIntegral integrate_semilocal_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       std::size_t tile_points, unsigned ingredient_mask,
                                       SemilocalEvaluator evaluate, const char* method) {
  validate_density_matrix(basis, grid, alpha_density, tile_points);
  validate_density_matrix(basis, grid, beta_density, tile_points);
  const bool need_first = (ingredient_mask & 14U) != 0;
  const bool need_tau = (ingredient_mask & 8U) != 0;
  const std::size_t n = basis.nao;
  SpinXcIntegral result;
  for (auto& potential : result.potential) potential.assign(n * n, 0.0);
  result.points = grid.point_count();
  std::vector<double> ao;
  const std::vector<double>* densities[2]{&alpha_density, &beta_density};
  for (std::size_t begin = 0; begin < result.points; begin += tile_points) {
    const std::size_t count = std::min(tile_points, result.points - begin);
    ao.resize((need_first ? 4 : 1) * count * n);
    basis.evaluate(grid.points().data() + 3 * begin, count, need_first ? 1 : 0, 0, n, ao.data(),
                   ao.size());
    for (std::size_t point = 0; point < count; ++point) {
      const double* phi = ao.data() + point * n;
      std::array<const double*, 3> jets{};
      if (need_first)
        for (unsigned k = 0; k < 3; ++k) jets[k] = ao.data() + ((k + 1) * count + point) * n;
      double rho[2]{}, gradient[2][3]{}, tau[2]{};
      for (unsigned spin = 0; spin < 2; ++spin) {
        const auto features =
            rks_features(phi, jets, n, *densities[spin], nullptr, ingredient_mask);
        rho[spin] = features[0];
        if (need_first)
          for (unsigned k = 0; k < 3; ++k) gradient[spin][k] = features[k + 1];
        if (need_tau) tau[spin] = features[4];
      }
      const auto xc = evaluate(rho, gradient, tau);
      const double weight = grid.weights()[begin + point];
      result.energy += weight * xc.energy;
      for (unsigned spin = 0; spin < 2; ++spin) {
        result.electrons[spin] += weight * rho[spin];
        for (std::size_t mu = 0; mu < n; ++mu)
          for (std::size_t nu = 0; nu < n; ++nu) {
            double value = xc.rho[spin] * phi[mu] * phi[nu];
            if (need_first)
              for (unsigned k = 0; k < 3; ++k) {
                value += xc.gradient[spin][k] * (jets[k][mu] * phi[nu] + phi[mu] * jets[k][nu]);
                if (need_tau) value += xc.kinetic[spin] * jets[k][mu] * jets[k][nu];
              }
            result.potential[spin][mu * n + nu] += weight * value;
          }
      }
    }
  }
  if (!std::isfinite(result.energy))
    throw std::runtime_error(std::string("nonfinite ") + method + " UKS semilocal energy");
  return result;
}

}  // namespace

void validate_semilocal_point_program(const SemilocalPointProgram& program) {
  if (!program.identifier || !*program.identifier || !program.expression_identity ||
      !*program.expression_identity || !program.evaluate)
    throw std::invalid_argument("semilocal point program requires complete identity and evaluator");
  if (program.ingredient_mask != 1U && program.ingredient_mask != 7U &&
      program.ingredient_mask != 15U)
    throw std::invalid_argument("semilocal point program has unsupported ingredient mask");
  if (program.domain_version == 0U)
    throw std::invalid_argument("semilocal point program requires a domain version");
}

XcIntegral integrate_semilocal_rks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& density,
                                   const SemilocalPointProgram& program,
                                   std::size_t tile_points, XcDensitySource source) {
  validate_semilocal_point_program(program);
  return integrate_semilocal_rks(basis, grid, density, tile_points, source,
                                 program.ingredient_mask, program.evaluate, program.identifier);
}

SpinXcIntegral integrate_semilocal_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       const SemilocalPointProgram& program,
                                       std::size_t tile_points) {
  validate_semilocal_point_program(program);
  return integrate_semilocal_uks(basis, grid, alpha_density, beta_density, tile_points,
                                 program.ingredient_mask, program.evaluate, program.identifier);
}

B3lypPointValue evaluate_b3lyp_point(const double rho[2], const double (&gradient)[2][3]) {
  const auto sigma = validate_b3lyp_production_point(rho, gradient);
  return map_gga_point(generated::b3lyp_polarized(rho[0], rho[1], sigma[0], sigma[1], sigma[2]),
                       gradient, "B3LYP");
}

CamB3lypPointValue evaluate_cam_b3lyp_point(const double rho[2], const double (&gradient)[2][3]) {
  const auto sigma = validate_gga_interior_point(rho, gradient, "CAM-B3LYP");
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double beta_b88 = 0.0042;
  constexpr double gamma_b88 = 6.0;
  const double cx = 0.375 * std::pow(3.0 / pi, 1.0 / 3.0) * std::pow(4.0, 2.0 / 3.0);
  for (unsigned spin = 0; spin < 2; ++spin) {
    const double same_sigma = sigma[spin == 0 ? 0 : 2];
    const double x2 = same_sigma * std::pow(rho[spin], -8.0 / 3.0);
    const double x = std::sqrt(x2);
    const double enhancement =
        1.0 + beta_b88 / cx * x2 / (1.0 + gamma_b88 * beta_b88 * x * std::asinh(x));
    const double k_gga = std::sqrt(9.0 * pi / (2.0 * cx * enhancement)) * std::cbrt(rho[spin]);
    if (!std::isfinite(k_gga) || generated::kCamB3lypOmega / (2.0 * k_gga) >= 1.35)
      throw std::domain_error("CAM-B3LYP ITYH attenuation exceeds rsh-interior-v1");
  }
  return map_gga_point(generated::cam_b3lyp_polarized(rho[0], rho[1], sigma[0], sigma[1], sigma[2]),
                       gradient, "CAM-B3LYP");
}

Pw91PointValue evaluate_pw91_point(const double rho[2], const double (&gradient)[2][3]) {
  const auto sigma = validate_gga_interior_point(rho, gradient, "PW91");
  return map_gga_point(generated::pw91_polarized(rho[0], rho[1], sigma[0], sigma[1], sigma[2]),
                       gradient, "PW91");
}

namespace {
SemilocalPointValue evaluate_b3lyp_semilocal(const double rho[2], const double (&gradient)[2][3],
                                             const double[2]) {
  return evaluate_b3lyp_point(rho, gradient);
}
SemilocalPointValue evaluate_cam_b3lyp_semilocal(const double rho[2],
                                                 const double (&gradient)[2][3], const double[2]) {
  return evaluate_cam_b3lyp_point(rho, gradient);
}
SemilocalPointValue evaluate_pw91_semilocal(const double rho[2], const double (&gradient)[2][3],
                                            const double[2]) {
  return evaluate_pw91_point(rho, gradient);
}
}  // namespace

XcIntegral integrate_b3lyp_rks(const AoBasis& basis, const MolecularGrid& grid,
                               const std::vector<double>& density, std::size_t tile_points,
                               XcDensitySource source) {
  return integrate_semilocal_rks(basis, grid, density, tile_points, source, 7U,
                                 evaluate_b3lyp_semilocal, "B3LYP");
}

SpinXcIntegral integrate_b3lyp_uks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& alpha_density,
                                   const std::vector<double>& beta_density,
                                   std::size_t tile_points) {
  return integrate_semilocal_uks(basis, grid, alpha_density, beta_density, tile_points, 7U,
                                 evaluate_b3lyp_semilocal, "B3LYP");
}

XcIntegral integrate_cam_b3lyp_rks(const AoBasis& basis, const MolecularGrid& grid,
                                   const std::vector<double>& density, std::size_t tile_points,
                                   XcDensitySource source) {
  return integrate_semilocal_rks(basis, grid, density, tile_points, source, 7U,
                                 evaluate_cam_b3lyp_semilocal, "CAM-B3LYP");
}

SpinXcIntegral integrate_cam_b3lyp_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       std::size_t tile_points) {
  return integrate_semilocal_uks(basis, grid, alpha_density, beta_density, tile_points, 7U,
                                 evaluate_cam_b3lyp_semilocal, "CAM-B3LYP");
}

XcIntegral integrate_pw91_rks(const AoBasis& basis, const MolecularGrid& grid,
                              const std::vector<double>& density, std::size_t tile_points,
                              XcDensitySource source) {
  return integrate_semilocal_rks(basis, grid, density, tile_points, source, 7U,
                                 evaluate_pw91_semilocal, "PW91");
}

SpinXcIntegral integrate_pw91_uks(const AoBasis& basis, const MolecularGrid& grid,
                                  const std::vector<double>& alpha_density,
                                  const std::vector<double>& beta_density,
                                  std::size_t tile_points) {
  return integrate_semilocal_uks(basis, grid, alpha_density, beta_density, tile_points, 7U,
                                 evaluate_pw91_semilocal, "PW91");
}

Wb97mvPointValue evaluate_wb97mv_point(const double rho[2], const double (&gradient)[2][3],
                                       const double tau[2]) {
  for (unsigned spin = 0; spin < 2; ++spin) {
    if (!std::isfinite(rho[spin]) || !std::isfinite(tau[spin]) || rho[spin] < 0.0 ||
        tau[spin] < 0.0)
      throw std::domain_error("omegaB97M-V requires finite nonnegative rho/tau");
    for (double component : gradient[spin])
      if (!std::isfinite(component))
        throw std::domain_error("omegaB97M-V requires finite density gradients");
  }

  // Reproduce the pinned Libxc 7.0.0 work_mgga input policy before entering
  // the Maple-generated functional. This is an audited definition-domain
  // continuation, not a VibeQC density clip: total density below the functional
  // threshold is zeroed, while surviving spin features use Libxc's floors.
  const double total_density = rho[0] + rho[1];
  if (total_density < generated::kWb97mvDensityThreshold) return {};

  double sigma[3]{};
  generated::sigma(gradient, sigma);
  double work_rho[2]{
      std::max(generated::kWb97mvDensityThreshold, rho[0]),
      std::max(generated::kWb97mvDensityThreshold, rho[1]),
  };
  const double sigma_floor = generated::kWb97mvSigmaThreshold * generated::kWb97mvSigmaThreshold;
  double work_sigma[3]{
      std::max(sigma_floor, sigma[0]),
      sigma[1],
      std::max(sigma_floor, sigma[2]),
  };
  const double sigma_average = 0.5 * (work_sigma[0] + work_sigma[2]);
  work_sigma[1] = std::clamp(work_sigma[1], -sigma_average, sigma_average);
  double work_tau[2]{
      std::max(generated::kWb97mvTauThreshold, tau[0]),
      std::max(generated::kWb97mvTauThreshold, tau[1]),
  };
  const auto raw =
      generated::wb97mv_polarized(work_rho[0], work_rho[1], work_sigma[0], work_sigma[1],
                                  work_sigma[2], work_tau[0], work_tau[1]);
  if (!std::isfinite(raw.energy_density))
    throw std::domain_error("omegaB97M-V production semilocal energy is nonfinite");
  for (double derivative : raw.feature_derivative)
    if (!std::isfinite(derivative))
      throw std::domain_error("omegaB97M-V production semilocal derivative is nonfinite");

  Wb97mvPointValue out;
  out.energy = raw.energy_density;
  out.rho[0] = raw.feature_derivative[0];
  out.rho[1] = raw.feature_derivative[1];
  for (unsigned k = 0; k < 3; ++k) {
    out.gradient[0][k] = 2.0 * raw.feature_derivative[2] * gradient[0][k] +
                         raw.feature_derivative[3] * gradient[1][k];
    out.gradient[1][k] = raw.feature_derivative[3] * gradient[0][k] +
                         2.0 * raw.feature_derivative[4] * gradient[1][k];
  }
  out.kinetic[0] = 0.5 * raw.feature_derivative[5];
  out.kinetic[1] = 0.5 * raw.feature_derivative[6];
  return out;
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

XcIntegral integrate_wb97mv_rks(const AoBasis& basis, const MolecularGrid& grid,
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
      const auto xc = evaluate_wb97mv_point(rho, gradient, tau);
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
  if (!std::isfinite(result.energy))
    throw std::runtime_error("nonfinite omegaB97M-V RKS semilocal energy");
  return result;
}

SpinXcIntegral integrate_wb97mv_uks(const AoBasis& basis, const MolecularGrid& grid,
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
      const auto xc = evaluate_wb97mv_point(rho, gradient, tau);
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
  if (!std::isfinite(result.energy))
    throw std::runtime_error("nonfinite omegaB97M-V UKS semilocal energy");
  return result;
}

XcIntegral integrate_r2scan_rks(const AoBasis& basis, const MolecularGrid& grid,
                                const std::vector<double>& density, std::size_t tile_points,
                                XcDensitySource source) {
  return integrate_semilocal_rks(basis, grid, density, tile_points, source, 15U,
                                 evaluate_r2scan_point, "r2SCAN");
}

SpinXcIntegral integrate_r2scan_uks(const AoBasis& basis, const MolecularGrid& grid,
                                    const std::vector<double>& alpha_density,
                                    const std::vector<double>& beta_density,
                                    std::size_t tile_points) {
  return integrate_semilocal_uks(basis, grid, alpha_density, beta_density, tile_points, 15U,
                                 evaluate_r2scan_point, "r2SCAN");
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
      const auto xc =
          evaluate_generated_pbe_point(spin_rho, spin_gradient, exchange_scale, correlation_scale);
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

ExactIncrementalXcIntegral integrate_pbe_rks_incremental_exact(
    const AoBasis& basis, const MolecularGrid& grid, const std::vector<double>& anchor_density,
    const std::vector<double>& delta_density, std::size_t tile_points, double exchange_scale,
    double correlation_scale) {
  const std::size_t n = basis.nao;
  validate_density_matrix(basis, grid, anchor_density, tile_points);
  // delta-D is intentionally allowed to be indefinite; only shape, symmetry
  // and finiteness are required here. Physical-domain validation is applied to
  // the reconstructed total features before nonlinear XC evaluation.
  validate_density_matrix(basis, grid, delta_density, tile_points);
  if (!std::isfinite(exchange_scale) || !std::isfinite(correlation_scale) || exchange_scale < 0.0 ||
      correlation_scale < 0.0)
    throw std::invalid_argument("incremental PBE scales must be finite and nonnegative");

  ExactIncrementalXcIntegral result;
  result.total.potential.assign(n * n, 0.0);
  result.total.points = grid.point_count();
  result.total.density_diagnostic.npoint = result.total.points;
  result.total.density_diagnostic.ingredient_mask = 3U;
  result.total.density_diagnostic.active_ao = n;
  result.total.density_diagnostic.borrowed_density_bytes = runtime::add_capacity(
      runtime::vector_bytes(anchor_density), runtime::vector_bytes(delta_density));
  result.potential_difference.assign(n * n, 0.0);

  std::vector<double> ao;
  const auto& points = grid.points();
  const auto& weights = grid.weights();
  for (std::size_t begin = 0; begin < result.total.points; begin += tile_points) {
    const std::size_t count = std::min(tile_points, result.total.points - begin);
    ao.resize(4 * count * n);
    sample_xc_capacity(result.total, ao, count);
    basis.evaluate(points.data() + 3 * begin, count, 1, 0, n, ao.data(), ao.size());
    for (std::size_t point = 0; point < count; ++point) {
      const double* phi = ao.data() + point * n;
      const double* grad_x = ao.data() + (count + point) * n;
      const double* grad_y = ao.data() + (2 * count + point) * n;
      const double* grad_z = ao.data() + (3 * count + point) * n;
      const std::array<const double*, 3> jets{grad_x, grad_y, grad_z};

      const auto anchor = rks_features(phi, jets, n, anchor_density, nullptr, 7U);
      const auto delta = rks_features(phi, jets, n, delta_density, nullptr, 7U);
      std::array<double, 4> total{};
      for (unsigned i = 0; i < 4; ++i) total[i] = anchor[i] + delta[i];

      const double anchor_rho[2]{0.5 * anchor[0], 0.5 * anchor[0]};
      const double total_rho[2]{0.5 * total[0], 0.5 * total[0]};
      double anchor_gradient[2][3]{};
      double total_gradient[2][3]{};
      for (unsigned axis = 0; axis < 3; ++axis) {
        anchor_gradient[0][axis] = anchor_gradient[1][axis] = 0.5 * anchor[axis + 1];
        total_gradient[0][axis] = total_gradient[1][axis] = 0.5 * total[axis + 1];
      }
      const auto anchor_xc = evaluate_generated_pbe_point(anchor_rho, anchor_gradient,
                                                          exchange_scale, correlation_scale);
      const auto total_xc = evaluate_generated_pbe_point(total_rho, total_gradient, exchange_scale,
                                                         correlation_scale);
      if (!anchor_xc.valid || !total_xc.valid)
        throw std::domain_error("invalid or unrepresentable incremental PBE features");

      const double weight = weights[begin + point];
      result.anchor_energy += weight * anchor_xc.energy;
      result.total.energy += weight * total_xc.energy;
      result.total.electrons += weight * total[0];

      const double anchor_rho_coefficient = anchor_xc.rho[0];
      const double total_rho_coefficient = total_xc.rho[0];
      for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
          const double phi_pair = phi[mu] * phi[nu];
          double anchor_value = anchor_rho_coefficient * phi_pair;
          double total_value = total_rho_coefficient * phi_pair;
          for (unsigned axis = 0; axis < 3; ++axis) {
            const double derivative_pair = jets[axis][mu] * phi[nu] + phi[mu] * jets[axis][nu];
            anchor_value += anchor_xc.gradient[0][axis] * derivative_pair;
            total_value += total_xc.gradient[0][axis] * derivative_pair;
          }
          const std::size_t index = mu * n + nu;
          result.total.potential[index] += weight * total_value;
          result.potential_difference[index] += weight * (total_value - anchor_value);
        }
      }
    }
  }
  result.energy_difference = result.total.energy - result.anchor_energy;
  const auto finite = [](double value) { return std::isfinite(value); };
  if (!std::isfinite(result.total.energy) || !std::isfinite(result.anchor_energy) ||
      !std::isfinite(result.energy_difference) || !std::isfinite(result.total.electrons) ||
      !std::all_of(result.total.potential.begin(), result.total.potential.end(), finite) ||
      !std::all_of(result.potential_difference.begin(), result.potential_difference.end(), finite))
    throw std::runtime_error("nonfinite incremental PBE result");
  return result;
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

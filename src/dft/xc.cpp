#include "dft/xc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "dft/xc_point.hpp"
#include "runtime/resource_usage.hpp"
#include "xc_cpu_generated.hpp"

namespace vibeqc::dft {
namespace {

struct Dual {
  double value{};
  double rho{};
  double sigma{};

  Dual operator-() const { return {-value, -rho, -sigma}; }
};

struct SpinDual {
  double value{};
  double alpha{};
  double beta{};

  SpinDual operator-() const { return {-value, -alpha, -beta}; }
};

SpinDual operator+(SpinDual left, SpinDual right) {
  return {left.value + right.value, left.alpha + right.alpha, left.beta + right.beta};
}
SpinDual operator-(SpinDual left, SpinDual right) { return left + (-right); }
SpinDual operator*(SpinDual left, SpinDual right) {
  return {left.value * right.value, left.alpha * right.value + left.value * right.alpha,
          left.beta * right.value + left.value * right.beta};
}
SpinDual operator/(SpinDual left, SpinDual right) {
  const double inverse = 1.0 / right.value;
  const double scale = inverse * inverse;
  return {left.value * inverse, (left.alpha * right.value - left.value * right.alpha) * scale,
          (left.beta * right.value - left.value * right.beta) * scale};
}
SpinDual operator+(SpinDual left, double right) { return left + SpinDual{right, 0.0, 0.0}; }
SpinDual operator+(double left, SpinDual right) { return right + left; }
SpinDual operator-(SpinDual left, double right) { return left + (-right); }
SpinDual operator-(double left, SpinDual right) { return SpinDual{left, 0.0, 0.0} - right; }
SpinDual operator*(SpinDual left, double right) {
  return {left.value * right, left.alpha * right, left.beta * right};
}
SpinDual operator*(double left, SpinDual right) { return right * left; }
SpinDual operator/(SpinDual left, double right) { return left * (1.0 / right); }
SpinDual spin_pow(const SpinDual& input, double exponent) {
  if (input.value < 0.0) throw std::domain_error("fractional power of negative spin density");
  const double value = std::pow(input.value, exponent);
  const double scale = input.value == 0.0 ? 0.0 : exponent * std::pow(input.value, exponent - 1.0);
  return {value, scale * input.alpha, scale * input.beta};
}

SpinDual log1p_over_x(const SpinDual& input) {
  const double x = input.value;
  double value = 0.0;
  double derivative = 0.0;
  if (std::abs(x) < 1.0e-5) {
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double x4 = x3 * x;
    value = 1.0 - 0.5 * x + x2 / 3.0 - 0.25 * x3 + 0.2 * x4;
    derivative = -0.5 + (2.0 / 3.0) * x - 0.75 * x2 + 0.8 * x3;
  } else {
    const double logarithm = std::log1p(x);
    value = logarithm / x;
    derivative = (x / (1.0 + x) - logarithm) / (x * x);
  }
  return {value, derivative * input.alpha, derivative * input.beta};
}

struct PolarizedLdaValue {
  double energy_density{};
  std::array<double, 2> density_derivative{};
};

PolarizedLdaValue lda_xc_pw_polarized_with_tail(double rho_a, double rho_b) {
  if (!std::isfinite(rho_a) || !std::isfinite(rho_b) || rho_a < 0.0 || rho_b < 0.0)
    throw std::domain_error("LDA spin-tail-v2 requires finite nonnegative spin densities");
  if (rho_a == 0.0 && rho_b == 0.0) return {};

  const double total_density = rho_a + rho_b;
  const double spin_polarization = (rho_a - rho_b) / total_density;
  const SpinDual x{std::pow(total_density, 1.0 / 6.0), 1.0, 0.0};
  const SpinDual z{spin_polarization, 0.0, 1.0};
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double c = std::pow(3.0 / (4.0 * pi), 1.0 / 3.0);
  const double sqrt_c = std::sqrt(c);
  const std::array<double, 3> a{0.031091, 0.015545, 0.016887};
  const std::array<double, 3> alpha{0.21370, 0.20548, 0.11125};
  const std::array<double, 3> b1{7.5957, 14.1189, 10.357};
  const std::array<double, 3> b2{3.5876, 6.1977, 3.6231};
  const std::array<double, 3> b3{1.6382, 3.3662, 0.88026};
  const std::array<double, 3> b4{0.49294, 0.62517, 0.49671};
  const SpinDual x2 = x * x;
  const SpinDual x3 = x2 * x;
  const SpinDual x4 = x2 * x2;
  std::array<SpinDual, 3> correlation_scale;
  for (std::size_t i = 0; i < correlation_scale.size(); ++i) {
    const SpinDual q =
        b1[i] * sqrt_c * x3 + b2[i] * c * x2 + b3[i] * std::pow(c, 1.5) * x + b4[i] * c * c;
    const SpinDual u = x4 / (2.0 * a[i] * q);
    correlation_scale[i] = -(x2 + alpha[i] * c) / q * log1p_over_x(u);
  }

  const SpinDual up = 1.0 + z;
  const SpinDual down = 1.0 - z;
  const double spin_denominator = std::pow(2.0, 4.0 / 3.0) - 2.0;
  const SpinDual fz =
      (spin_pow(up, 4.0 / 3.0) + spin_pow(down, 4.0 / 3.0) - 2.0) / spin_denominator;
  constexpr double fz20 = 1.709921;
  const SpinDual z2 = z * z;
  const SpinDual z4 = z2 * z2;
  const SpinDual correlation_energy_scale =
      correlation_scale[0] +
      z4 * fz * (correlation_scale[1] - correlation_scale[0] + correlation_scale[2] / fz20) -
      fz * correlation_scale[2] / fz20;
  const double cx = (3.0 / 8.0) * std::pow(3.0 / pi, 1.0 / 3.0) * std::pow(4.0, 2.0 / 3.0);
  const SpinDual exchange_energy_scale =
      -cx * (spin_pow(up / 2.0, 4.0 / 3.0) + spin_pow(down / 2.0, 4.0 / 3.0));
  const SpinDual energy_scale = exchange_energy_scale + correlation_energy_scale;
  const double x2_value = x.value * x.value;
  const double radial_derivative =
      x2_value * (8.0 * energy_scale.value + x.value * energy_scale.alpha) / 6.0;
  const double spin_derivative = x2_value * energy_scale.beta;
  const double alpha_derivative = radial_derivative + (1.0 - spin_polarization) * spin_derivative;
  const double beta_derivative = radial_derivative - (1.0 + spin_polarization) * spin_derivative;
  const double energy_density = std::pow(x.value, 8.0) * energy_scale.value;
  if (!std::isfinite(energy_density) || !std::isfinite(alpha_derivative) ||
      !std::isfinite(beta_derivative)) {
    std::ostringstream detail;
    detail << "nonfinite LDA spin-tail-v2 value at rho_a=" << rho_a << ", rho_b=" << rho_b
           << ", rho=" << rho_a + rho_b;
    throw std::runtime_error(detail.str());
  }
  return {energy_density, {alpha_derivative, beta_derivative}};
}

Dual operator+(Dual left, Dual right) {
  return {left.value + right.value, left.rho + right.rho, left.sigma + right.sigma};
}
Dual operator-(Dual left, Dual right) { return left + (-right); }
Dual operator*(Dual left, Dual right) {
  return {left.value * right.value, left.rho * right.value + left.value * right.rho,
          left.sigma * right.value + left.value * right.sigma};
}
Dual operator/(Dual left, Dual right) {
  const double inverse = 1.0 / right.value;
  const double scale = inverse * inverse;
  return {left.value * inverse, (left.rho * right.value - left.value * right.rho) * scale,
          (left.sigma * right.value - left.value * right.sigma) * scale};
}

Dual operator+(double left, Dual right) { return Dual{left, 0.0, 0.0} + right; }
Dual operator-(double left, Dual right) { return Dual{left, 0.0, 0.0} - right; }
Dual operator*(Dual left, double right) {
  return {left.value * right, left.rho * right, left.sigma * right};
}
Dual operator*(double left, Dual right) { return right * left; }
Dual operator/(Dual left, double right) { return left * (1.0 / right); }
Dual operator/(double left, Dual right) { return Dual{left, 0.0, 0.0} / right; }

Dual expm1(const Dual& input) {
  const double value = std::expm1(input.value);
  const double derivative = std::exp(input.value);
  return {value, derivative * input.rho, derivative * input.sigma};
}

Dual log1p(const Dual& input) {
  const double scale = 1.0 / (1.0 + input.value);
  return {std::log1p(input.value), scale * input.rho, scale * input.sigma};
}

Dual pow(const Dual& input, double exponent) {
  const double value = std::pow(input.value, exponent);
  const double scale = exponent * std::pow(input.value, exponent - 1.0);
  return {value, scale * input.rho, scale * input.sigma};
}

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

struct PbeValue {
  double energy_density{};
  double density_derivative{};
  double sigma_derivative{};
};

PbeValue pbe_unpolarized(double rho, double sigma, bool allow_tail) {
  if (!std::isfinite(rho) || !std::isfinite(sigma) || rho < 0.0 || sigma < 0.0)
    throw std::domain_error("PBE tail-v1 requires finite nonnegative rho and sigma");
  if (rho == 0.0) {
    if (sigma != 0.0) throw std::domain_error("PBE vacuum requires zero gradient");
    return {};
  }
  const bool outside_density = rho < 1.0e-12 || rho > 1.0e12;
  const double reduced_gradient = outside_density ? std::numeric_limits<double>::infinity()
                                                  : std::sqrt(sigma) / std::pow(rho, 4.0 / 3.0);
  const bool outside_domain = outside_density || reduced_gradient > 1.0e6;
  if (outside_domain) {
    if (!allow_tail) {
      if (outside_density) throw std::domain_error("PBE total density outside interior-v1");
      throw std::domain_error("PBE reduced gradient exceeds interior-v1 1e6");
    }
    const auto lda = generated::lda_xc_pw_unpolarized(rho);
    if (!std::isfinite(lda.energy_density) || !std::isfinite(lda.density_derivative))
      throw std::runtime_error("nonfinite PBE LDA fallback tail value");
    return {lda.energy_density, lda.density_derivative, 0.0};
  }
  const Dual r{rho, 1.0, 0.0};
  const Dual s{sigma, 0.0, 1.0};
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double c = std::pow(3.0 / (4.0 * pi), 1.0 / 3.0);
  const double cx = (3.0 / 8.0) * std::pow(3.0 / pi, 1.0 / 3.0) * std::pow(4.0, 2.0 / 3.0);
  const double kappa = 0.8040;
  const double mu = 0.06672455060314922 * pi * pi / 3.0;
  const double x2s2 = 1.0 / (4.0 * std::pow(6.0 * pi * pi, 2.0 / 3.0));

  const Dual spin_density = r / 2.0;
  const Dual s2 = x2s2 * (s / 4.0) * pow(spin_density, -8.0 / 3.0);
  const Dual enhancement = 1.0 + kappa * (1.0 - kappa / (kappa + mu * s2));
  const Dual exchange = -2.0 * cx * pow(spin_density, 4.0 / 3.0) * enhancement;

  const Dual rs = c * pow(r, -1.0 / 3.0);
  const double a = 0.0310907;
  const double alpha = 0.21370;
  const double b1 = 7.5957;
  const double b2 = 3.5876;
  const double b3 = 1.6382;
  const double b4 = 0.49294;
  const Dual aux = b1 * pow(rs, 0.5) + b2 * rs + b3 * pow(rs, 1.5) + b4 * pow(rs, 2.0);
  const Dual eps_pw = -2.0 * a * (1.0 + alpha * rs) * log1p(1.0 / (2.0 * a * aux));

  const double beta = 0.06672455060314922;
  const double gamma = (1.0 - std::log(2.0)) / (pi * pi);
  const Dual t2 = (s * pow(r, -8.0 / 3.0)) / (16.0 * std::pow(2.0, 2.0 / 3.0) * rs);
  const Dual a_pbe = beta / (gamma * expm1(-eps_pw / gamma));
  const Dual f1 = t2 + a_pbe * t2 * t2;
  const Dual f2 = beta * f1 / (gamma * (1.0 + a_pbe * f1));
  const Dual correlation = r * (eps_pw + gamma * log1p(f2));
  const Dual total = exchange + correlation;
  if (!std::isfinite(total.value) || !std::isfinite(total.rho) || !std::isfinite(total.sigma))
    throw std::runtime_error("nonfinite PBE tail-v1 value");
  return {total.value, total.rho, total.sigma};
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

SpinXcIntegral integrate_lda_xc_pw_uks(const AoBasis& basis, const MolecularGrid& grid,
                                       const std::vector<double>& alpha_density,
                                       const std::vector<double>& beta_density,
                                       std::size_t tile_points) {
  validate_density_matrix(basis, grid, alpha_density, tile_points);
  validate_density_matrix(basis, grid, beta_density, tile_points);
  const std::size_t n = basis.nao;
  SpinXcIntegral result;
  result.potential[0].assign(n * n, 0.0);
  result.potential[1].assign(n * n, 0.0);
  result.points = grid.point_count();
  std::vector<double> ao;
  const auto& points = grid.points();
  const auto& weights = grid.weights();
  for (std::size_t begin = 0; begin < result.points; begin += tile_points) {
    const std::size_t count = std::min(tile_points, result.points - begin);
    ao.resize(count * n);
    basis.evaluate(points.data() + 3 * begin, count, 0, 0, n, ao.data(), ao.size());
    for (std::size_t point = 0; point < count; ++point) {
      const double* phi = ao.data() + point * n;
      double rho_a = 0.0;
      double rho_b = 0.0;
      for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
          const double product = phi[mu] * phi[nu];
          rho_a += product * alpha_density[mu * n + nu];
          rho_b += product * beta_density[mu * n + nu];
        }
      }
      if (!std::isfinite(rho_a) || !std::isfinite(rho_b) || rho_a < 0.0 || rho_b < 0.0)
        throw std::domain_error("LDA UKS requires finite nonnegative spin densities");
      if (rho_a == 0.0 && rho_b == 0.0) continue;
      const auto xc = lda_xc_pw_polarized_with_tail(rho_a, rho_b);
      if (!std::isfinite(xc.energy_density))
        throw std::runtime_error("nonfinite polarized LDA tail value");
      const double weight = weights[begin + point];
      result.energy += weight * xc.energy_density;
      result.electrons[0] += weight * rho_a;
      result.electrons[1] += weight * rho_b;
      for (std::size_t spin = 0; spin < 2; ++spin) {
        if (!std::isfinite(xc.density_derivative[spin]))
          throw std::runtime_error("nonfinite polarized LDA tail derivative");
        const double coefficient = weight * xc.density_derivative[spin];
        for (std::size_t mu = 0; mu < n; ++mu)
          for (std::size_t nu = 0; nu < n; ++nu)
            result.potential[spin][mu * n + nu] += coefficient * phi[mu] * phi[nu];
      }
    }
  }
  if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite polarized LDA energy");
  return result;
}

SpinXcIntegral integrate_pbe_uks_with_tail(const AoBasis& basis, const MolecularGrid& grid,
                                           const std::vector<double>& alpha_density,
                                           const std::vector<double>& beta_density,
                                           std::size_t tile_points) {
  validate_density_matrix(basis, grid, alpha_density, tile_points);
  validate_density_matrix(basis, grid, beta_density, tile_points);
  const std::size_t n = basis.nao;
  SpinXcIntegral result;
  result.potential[0].assign(n * n, 0.0);
  result.potential[1].assign(n * n, 0.0);
  result.points = grid.point_count();
  std::vector<double> ao;
  const auto& points = grid.points();
  const auto& weights = grid.weights();
  for (std::size_t begin = 0; begin < result.points; begin += tile_points) {
    const std::size_t count = std::min(tile_points, result.points - begin);
    ao.resize(4 * count * n);
    basis.evaluate(points.data() + 3 * begin, count, 1, 0, n, ao.data(), ao.size());
    for (std::size_t point = 0; point < count; ++point) {
      const double* phi = ao.data() + point * n;
      const std::array<const double*, 3> derivative{
          ao.data() + (count + point) * n,
          ao.data() + (2 * count + point) * n,
          ao.data() + (3 * count + point) * n,
      };
      std::array<double, 2> rho{};
      std::array<std::array<double, 3>, 2> gradient{};
      for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
          const std::size_t index = mu * n + nu;
          const double product = phi[mu] * phi[nu];
          for (std::size_t spin = 0; spin < 2; ++spin) {
            const double density = spin == 0 ? alpha_density[index] : beta_density[index];
            rho[spin] += product * density;
            for (std::size_t axis = 0; axis < 3; ++axis)
              gradient[spin][axis] +=
                  (derivative[axis][mu] * phi[nu] + phi[mu] * derivative[axis][nu]) * density;
          }
        }
      }
      // Differentiate the same stable PBE expression in Cartesian gradients.
      // This keeps energy and potential continuous at an empty spin and avoids
      // forming a divergent v_sigma followed by multiplication by zero.
      const double spin_gradient[2][3]{{gradient[0][0], gradient[0][1], gradient[0][2]},
                                       {gradient[1][0], gradient[1][1], gradient[1][2]}};
      const auto xc = point::evaluate(true, rho.data(), spin_gradient);
      if (!xc.valid) throw std::domain_error("invalid or unrepresentable polarized PBE point");
      const double weight = weights[begin + point];
      result.energy += weight * xc.energy;
      result.electrons[0] += weight * rho[0];
      result.electrons[1] += weight * rho[1];
      for (std::size_t spin = 0; spin < 2; ++spin) {
        for (std::size_t mu = 0; mu < n; ++mu) {
          for (std::size_t nu = 0; nu < n; ++nu) {
            double value = xc.rho[spin] * phi[mu] * phi[nu];
            for (std::size_t axis = 0; axis < 3; ++axis)
              value += xc.gradient[spin][axis] *
                       (derivative[axis][mu] * phi[nu] + phi[mu] * derivative[axis][nu]);
            result.potential[spin][mu * n + nu] += weight * value;
          }
        }
      }
    }
  }
  if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite polarized PBE energy");
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
      const auto features = rks_features(phi, {grad_x, grad_y, grad_z}, n, density, factor, true);
      const double rho = features[0];
      const std::array<double, 3> gradient{features[1], features[2], features[3]};
      const double sigma =
          gradient[0] * gradient[0] + gradient[1] * gradient[1] + gradient[2] * gradient[2];
      const auto xc = pbe_unpolarized(rho, sigma, allow_tail);
      const double weight = weights[begin + point];
      result.energy += weight * xc.energy_density;
      result.electrons += weight * rho;
      for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
          double value = xc.density_derivative * phi[mu] * phi[nu];
          value += 2.0 * xc.sigma_derivative *
                   (gradient[0] * (grad_x[mu] * phi[nu] + phi[mu] * grad_x[nu]) +
                    gradient[1] * (grad_y[mu] * phi[nu] + phi[mu] * grad_y[nu]) +
                    gradient[2] * (grad_z[mu] * phi[nu] + phi[mu] * grad_z[nu]));
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

}  // namespace vibeqc::dft

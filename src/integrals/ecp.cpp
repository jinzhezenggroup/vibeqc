#include "integrals/ecp.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "molecule/basis.hpp"

namespace vibeqc::integrals {
namespace {
constexpr double pi = std::numbers::pi;

std::vector<std::array<double, 2>> legendre(unsigned n) {
  std::vector<std::array<double, 2>> result(n);
  for (unsigned i = 0; i < (n + 1) / 2; ++i) {
    double z = std::cos(pi * (i + .75) / (n + .5)), derivative = 0;
    for (unsigned iteration = 0; iteration < 64; ++iteration) {
      double p = 1, previous = 0;
      for (unsigned k = 1; k <= n; ++k) {
        const double next = ((2 * k - 1) * z * p - (k - 1) * previous) / k;
        previous = p;
        p = next;
      }
      derivative = n * (z * p - previous) / (z * z - 1);
      const double delta = p / derivative;
      z -= delta;
      if (std::abs(delta) < 2e-15) break;
      if (iteration == 63) throw std::runtime_error("ECP quadrature root did not converge");
    }
    const double weight = 2 / ((1 - z * z) * derivative * derivative);
    result[i] = {-z, weight};
    result[n - i - 1] = {z, weight};
  }
  return result;
}

double ipow(double x, unsigned n) {
  double v = 1;
  for (unsigned i = 0; i < n; ++i) v *= x;
  return v;
}

struct AO {
  const core::Shell* shell;
  molecule::AoExpansion expansion;
};

// Value and analytic derivative with respect to the AO's basis center.
std::array<double, 4> evaluate(const AO& ao, const core::System& system,
                               const std::array<double, 3>& point) {
  std::array<double, 3> x{};
  double rr = 0;
  for (unsigned d = 0; d < 3; ++d) {
    x[d] = point[d] - system.atoms[ao.shell->atom_index].position[d];
    rr += x[d] * x[d];
  }
  std::array<double, 4> v{};
  for (const auto& term : ao.expansion) {
    double polynomial = 1;
    for (unsigned d = 0; d < 3; ++d) polynomial *= ipow(x[d], term.component[d]);
    const double factor =
        term.coefficient * molecule::cartesian_component_normalization(term.component);
    for (const auto& primitive : ao.shell->primitives) {
      const double radial = factor * primitive.coefficient * std::exp(-primitive.exponent * rr);
      v[0] += radial * polynomial;
      for (unsigned d = 0; d < 3; ++d) {
        double lowered = 0;
        if (term.component[d]) {
          lowered = term.component[d];
          for (unsigned e = 0; e < 3; ++e) lowered *= ipow(x[e], term.component[e] - (d == e));
        }
        v[d + 1] += radial * (2 * primitive.exponent * x[d] * polynomial - lowered);
      }
    }
  }
  return v;
}

// Any orthonormal real basis within l gives the same summed projector.
std::array<double, 9> harmonics(double x, double y, double z) {
  return {1 / std::sqrt(4 * pi),
          std::sqrt(3 / (4 * pi)) * x,
          std::sqrt(3 / (4 * pi)) * y,
          std::sqrt(3 / (4 * pi)) * z,
          std::sqrt(15 / (4 * pi)) * x * y,
          std::sqrt(15 / (4 * pi)) * y * z,
          std::sqrt(5 / (16 * pi)) * (3 * z * z - 1),
          std::sqrt(15 / (4 * pi)) * x * z,
          std::sqrt(15 / (16 * pi)) * (x * x - y * y)};
}
}  // namespace

void ecp_quadrature(unsigned radial, unsigned angular, std::vector<EcpRadialPoint>& radii,
                    std::vector<EcpSpherePoint>& sphere) {
  if (radial < 16 || radial > 512 || angular < 8 || angular > 96)
    throw std::invalid_argument("invalid ECP quadrature grid");
  for (const auto& tw : legendre(radial)) {
    const double t = (tw[0] + 1) / 2;
    radii.push_back({t / (1 - t), tw[1] / (2 * (1 - t) * (1 - t))});
  }
  const unsigned nphi = 2 * angular;
  for (const auto& zw : legendre(angular))
    for (unsigned k = 0; k < nphi; ++k) {
      const double phi = 2 * pi * k / nphi, s = std::sqrt(1 - zw[0] * zw[0]);
      EcpSpherePoint p{s * std::cos(phi), s * std::sin(phi), zw[0], zw[1] * 2 * pi / nphi, {}};
      const auto y = harmonics(p.x, p.y, p.z);
      std::copy(y.begin(), y.end(), p.harmonics);
      sphere.push_back(p);
    }
}

EcpData ecp_integrals(const core::System& system, unsigned radial, unsigned angular,
                      bool derivatives) {
  if (radial < 16 || radial > 512 || angular < 8 || angular > 96)
    throw std::invalid_argument("ECP quadrature requires 16..512 radial and 8..96 polar points");
  std::vector<AO> aos;
  for (const auto& shell : system.shells)
    for (const auto& expansion :
         molecule::ao_expansions(shell.angular_momentum, system.basis_representation))
      aos.push_back({&shell, expansion});
  const std::size_t n = aos.size(), size = n * n;
  if (n > 256 || system.atoms.size() > 128)
    throw std::invalid_argument("ECP baseline supports at most 256 AOs and 128 atoms");
  EcpData out{n,
              derivatives ? 3 * system.atoms.size() : 0,
              std::vector<double>(size),
              std::vector<double>(size),
              {},
              {}};
  out.local_derivative.resize(out.ncoord * size);
  out.nonlocal_derivative.resize(out.ncoord * size);
  if (system.ecp_terms.empty()) return out;
  const auto radial_grid = legendre(radial), polar = legendre(angular);
  const unsigned nphi = 2 * angular;
  struct Point {
    std::array<double, 3> xyz;
    double weight;
    std::array<double, 9> y;
  };
  std::vector<Point> sphere;
  for (const auto& zw : polar)
    for (unsigned k = 0; k < nphi; ++k) {
      const double phi = 2 * pi * k / nphi, s = std::sqrt(1 - zw[0] * zw[0]);
      const double x = s * std::cos(phi), y = s * std::sin(phi);
      sphere.push_back({{x, y, zw[0]}, zw[1] * 2 * pi / nphi, harmonics(x, y, zw[0])});
    }
  // A radial shell is the lifetime boundary: no full molecular quadrature/AO tensor.
  std::vector<std::array<double, 4>> values(n * sphere.size());
  std::vector<std::array<double, 4>> projections(n * 9);
  for (std::size_t center = 0; center < system.atoms.size(); ++center) {
    if (system.atoms[center].ecp_core == 0) continue;
    for (const auto& tw : radial_grid) {
      const double t = (tw[0] + 1) / 2, r = t / (1 - t);
      const double rw = tw[1] / (2 * (1 - t) * (1 - t));
      std::array<double, 4> potentials{};
      for (const auto& term : system.ecp_terms)
        if (term.atom_index == center)
          potentials[term.channel + 1] +=
              rw * term.coefficient * ipow(r, term.power) * std::exp(-term.exponent * r * r);
      if (std::all_of(potentials.begin(), potentials.end(), [](double v) { return v == 0; }))
        continue;
      std::fill(projections.begin(), projections.end(), std::array<double, 4>{});
      for (std::size_t q = 0; q < sphere.size(); ++q) {
        std::array<double, 3> point{};
        for (unsigned d = 0; d < 3; ++d)
          point[d] = system.atoms[center].position[d] + r * sphere[q].xyz[d];
        for (std::size_t a = 0; a < n; ++a) {
          const auto v = evaluate(aos[a], system, point);
          values[a * sphere.size() + q] = v;
          for (unsigned m = 0; m < 9; ++m)
            for (unsigned d = 0; d < (derivatives ? 4U : 1U); ++d)
              projections[a * 9 + m][d] += sphere[q].weight * sphere[q].y[m] * v[d];
        }
      }
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b <= a; ++b) {
          // 0: value; 1..3: A derivative; 4..6: B derivative.
          std::array<double, 7> local{}, nonlocal{};
          for (std::size_t q = 0; q < sphere.size(); ++q) {
            const auto& va = values[a * sphere.size() + q];
            const auto& vb = values[b * sphere.size() + q];
            const double w = potentials[0] * sphere[q].weight;
            local[0] += w * va[0] * vb[0];
            if (derivatives)
              for (unsigned d = 0; d < 3; ++d) {
                local[d + 1] += w * va[d + 1] * vb[0];
                local[d + 4] += w * va[0] * vb[d + 1];
              }
          }
          for (unsigned l = 0; l <= 2; ++l)
            for (unsigned m = l * l; m < (l + 1) * (l + 1); ++m) {
              const auto& va = projections[a * 9 + m];
              const auto& vb = projections[b * 9 + m];
              const double w = potentials[l + 1];
              nonlocal[0] += w * va[0] * vb[0];
              if (derivatives)
                for (unsigned d = 0; d < 3; ++d) {
                  nonlocal[d + 1] += w * va[d + 1] * vb[0];
                  nonlocal[d + 4] += w * va[0] * vb[d + 1];
                }
            }
          for (unsigned part = 0; part < 2; ++part) {
            const auto& v = part == 0 ? local : nonlocal;
            auto& matrix = part == 0 ? out.local : out.nonlocal;
            auto& gradient = part == 0 ? out.local_derivative : out.nonlocal_derivative;
            for (unsigned transpose = 0; transpose < (a == b ? 1U : 2U); ++transpose) {
              const auto item = transpose ? b * n + a : a * n + b;
              matrix[item] += v[0];
              if (derivatives)
                for (unsigned d = 0; d < 3; ++d) {
                  gradient[(aos[a].shell->atom_index * 3 + d) * size + item] += v[d + 1];
                  gradient[(aos[b].shell->atom_index * 3 + d) * size + item] += v[d + 4];
                  // Complete A/B/C translation: C is a real ECP center, not a shell.
                  gradient[(center * 3 + d) * size + item] -= v[d + 1] + v[d + 4];
                }
            }
          }
        }
    }
  }
  return out;
}

EcpData checked_ecp_integrals(const core::System& system, bool derivatives) {
  const auto coarse = ecp_integrals(system, 160, 32, derivatives);
  auto fine = ecp_integrals(system, 224, 44, derivatives);
  const auto check = [](const auto& a, const auto& b, double tolerance) {
    for (std::size_t i = 0; i < a.size(); ++i)
      if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i] - b[i]) > tolerance)
        throw std::runtime_error(
            "ECP quadrature convergence gate failed; this input is outside the validated execution "
            "domain");
  };
  check(coarse.local, fine.local, 2e-9);
  check(coarse.nonlocal, fine.nonlocal, 2e-9);
  check(coarse.local_derivative, fine.local_derivative, 2e-8);
  check(coarse.nonlocal_derivative, fine.nonlocal_derivative, 2e-8);
  return fine;
}

void add_ecp(const EcpData& ecp, std::vector<double>& hcore, std::vector<double>& derivative) {
  if (hcore.size() != ecp.local.size() || derivative.size() != ecp.local_derivative.size())
    throw std::invalid_argument("ECP and one-electron matrix layouts differ");
  for (std::size_t i = 0; i < hcore.size(); ++i) hcore[i] += ecp.local[i] + ecp.nonlocal[i];
  for (std::size_t i = 0; i < derivative.size(); ++i)
    derivative[i] += ecp.local_derivative[i] + ecp.nonlocal_derivative[i];
}
}  // namespace vibeqc::integrals

#include "integrals/s_integrals.hpp"

#include "molecule/basis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vibeqc::integrals {
namespace {

// Dynamic forward derivatives make the CPU implementation a compact and
// independent oracle for both integral values and every nuclear coordinate.
// The optimized CUDA backend uses a one-coordinate dual scalar instead.
struct Jet {
  double value{};
  std::vector<double> derivative;

  Jet() = default;
  Jet(double v, std::size_t ncoord) : value(v), derivative(ncoord, 0.0) {}

  static Jet variable(double v, std::size_t ncoord, std::size_t coordinate) {
    Jet result(v, ncoord);
    result.derivative[coordinate] = 1.0;
    return result;
  }
};

Jet operator+(const Jet& a, const Jet& b) {
  Jet out(a.value + b.value, a.derivative.size());
  for (std::size_t i = 0; i < out.derivative.size(); ++i) {
    out.derivative[i] = a.derivative[i] + b.derivative[i];
  }
  return out;
}

Jet operator-(const Jet& a, const Jet& b) {
  Jet out(a.value - b.value, a.derivative.size());
  for (std::size_t i = 0; i < out.derivative.size(); ++i) {
    out.derivative[i] = a.derivative[i] - b.derivative[i];
  }
  return out;
}

Jet operator*(const Jet& a, const Jet& b) {
  Jet out(a.value * b.value, a.derivative.size());
  for (std::size_t i = 0; i < out.derivative.size(); ++i) {
    out.derivative[i] =
        a.derivative[i] * b.value + a.value * b.derivative[i];
  }
  return out;
}

Jet operator/(const Jet& a, const Jet& b) {
  Jet out(a.value / b.value, a.derivative.size());
  const double inverse_square = 1.0 / (b.value * b.value);
  for (std::size_t i = 0; i < out.derivative.size(); ++i) {
    out.derivative[i] =
        (a.derivative[i] * b.value - a.value * b.derivative[i]) *
        inverse_square;
  }
  return out;
}

Jet operator*(double a, const Jet& b) { return Jet(a, b.derivative.size()) * b; }
Jet operator/(const Jet& a, double b) { return a / Jet(b, a.derivative.size()); }
Jet operator/(double a, const Jet& b) { return Jet(a, b.derivative.size()) / b; }

Jet exp(const Jet& x) {
  const double value = std::exp(x.value);
  Jet out(value, x.derivative.size());
  for (std::size_t i = 0; i < out.derivative.size(); ++i) {
    out.derivative[i] = value * x.derivative[i];
  }
  return out;
}

Jet sqrt(const Jet& x) {
  const double value = std::sqrt(x.value);
  Jet out(value, x.derivative.size());
  const double factor = 0.5 / value;
  for (std::size_t i = 0; i < out.derivative.size(); ++i) {
    out.derivative[i] = factor * x.derivative[i];
  }
  return out;
}

Jet erf(const Jet& x) {
  Jet out(std::erf(x.value), x.derivative.size());
  const double factor = 2.0 / std::sqrt(std::numbers::pi) *
                        std::exp(-x.value * x.value);
  for (std::size_t i = 0; i < out.derivative.size(); ++i) {
    out.derivative[i] = factor * x.derivative[i];
  }
  return out;
}

using Vec3 = std::array<Jet, 3>;

Jet distance_squared(const Vec3& a, const Vec3& b) {
  Jet result(0.0, a[0].derivative.size());
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const Jet delta = a[axis] - b[axis];
    result = result + delta * delta;
  }
  return result;
}

Vec3 product_center(double alpha, const Vec3& a, double beta, const Vec3& b) {
  const double p = alpha + beta;
  return {(alpha * a[0] + beta * b[0]) / p,
          (alpha * a[1] + beta * b[1]) / p,
          (alpha * a[2] + beta * b[2]) / p};
}

std::vector<Jet> boys_values(unsigned maximum_order, const Jet& argument) {
  std::vector<Jet> values(maximum_order + 1,
                          Jet(0.0, argument.derivative.size()));
  if (argument.value < 6.0) {
    // Differentiate the entire Taylor series as algebra, avoiding an explicit
    // derivative branch and the F0 erf singularity at T=0.
    for (unsigned order = 0; order <= maximum_order; ++order) {
      Jet term(1.0, argument.derivative.size());
      Jet sum(0.0, argument.derivative.size());
      for (unsigned k = 0; k < 80; ++k) {
        sum = sum + term /
                        static_cast<double>(2 * order + 2 * k + 1);
        term = term * (-1.0 * argument) / static_cast<double>(k + 1);
        if (std::abs(term.value) < 1.0e-18) break;
      }
      values[order] = std::move(sum);
    }
    return values;
  }

  values[0] = 0.5 * sqrt(Jet(std::numbers::pi, argument.derivative.size()) /
                           argument) *
              erf(sqrt(argument));
  const Jet exponential = exp(-1.0 * argument);
  for (unsigned order = 1; order <= maximum_order; ++order) {
    values[order] =
        ((2.0 * static_cast<double>(order) - 1.0) * values[order - 1] -
         exponential) /
        (2.0 * argument);
  }
  return values;
}

struct HermiteCoefficients {
  unsigned jdim{};
  unsigned tdim{};
  std::vector<Jet> data;

  Jet& at(unsigned i, unsigned j, unsigned t) {
    return data[(static_cast<std::size_t>(i) * jdim + j) * tdim + t];
  }
  const Jet& at(unsigned i, unsigned j, unsigned t) const {
    return data[(static_cast<std::size_t>(i) * jdim + j) * tdim + t];
  }
};

HermiteCoefficients fill_hermite(unsigned maximum_i,
                                 unsigned maximum_j,
                                 const Jet& product,
                                 const Jet& center_a,
                                 const Jet& center_b,
                                 double alpha,
                                 double beta) {
  HermiteCoefficients coefficients;
  const unsigned idim = maximum_i + 1;
  coefficients.jdim = maximum_j + 1;
  coefficients.tdim = maximum_i + maximum_j + 2;
  coefficients.data.assign(
      static_cast<std::size_t>(idim) * coefficients.jdim * coefficients.tdim,
      Jet(0.0, center_a.derivative.size()));

  const double p = alpha + beta;
  const double mu = alpha * beta / p;
  const Jet ab = center_a - center_b;
  coefficients.at(0, 0, 0) = exp(-mu * ab * ab);
  const Jet pa = product - center_a;
  const Jet pb = product - center_b;
  const double inverse_two_p = 0.5 / p;

  for (unsigned i = 0; i <= maximum_i; ++i) {
    for (unsigned j = 0; j <= maximum_j; ++j) {
      if (i == 0 && j == 0) continue;
      if (i > 0) {
        coefficients.at(i, j, 0) =
            pa * coefficients.at(i - 1, j, 0) +
            coefficients.at(i - 1, j, 1);
      } else {
        coefficients.at(i, j, 0) =
            pb * coefficients.at(i, j - 1, 0) +
            coefficients.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coefficients.at(i, j, t) =
              pa * coefficients.at(i - 1, j, t) +
              inverse_two_p * coefficients.at(i - 1, j, t - 1) +
              static_cast<double>(t + 1) *
                  coefficients.at(i - 1, j, t + 1);
        } else {
          coefficients.at(i, j, t) =
              pb * coefficients.at(i, j - 1, t) +
              inverse_two_p * coefficients.at(i, j - 1, t - 1) +
              static_cast<double>(t + 1) *
                  coefficients.at(i, j - 1, t + 1);
        }
      }
    }
  }
  return coefficients;
}

struct CoulombAuxiliary {
  unsigned dim{};
  std::vector<Jet> data;

  Jet& at(unsigned n, unsigned t, unsigned u, unsigned v) {
    return data[(((static_cast<std::size_t>(n) * dim + t) * dim + u) * dim) +
                v];
  }
  const Jet& at(unsigned n, unsigned t, unsigned u, unsigned v) const {
    return data[(((static_cast<std::size_t>(n) * dim + t) * dim + u) * dim) +
                v];
  }
};

CoulombAuxiliary fill_coulomb(unsigned maximum_angular,
                              double exponent,
                              const Vec3& product,
                              const Vec3& center) {
  CoulombAuxiliary auxiliary;
  auxiliary.dim = maximum_angular + 1;
  const std::size_t size = static_cast<std::size_t>(auxiliary.dim) *
                           auxiliary.dim * auxiliary.dim * auxiliary.dim;
  auxiliary.data.assign(size, Jet(0.0, product[0].derivative.size()));

  const Vec3 pc{product[0] - center[0], product[1] - center[1],
                product[2] - center[2]};
  const std::vector<Jet> boys =
      boys_values(maximum_angular,
                  exponent * distance_squared(product, center));
  double factor = 1.0;
  for (unsigned n = 0; n <= maximum_angular; ++n) {
    auxiliary.at(n, 0, 0, 0) = factor * boys[n];
    factor *= -2.0 * exponent;
  }

  for (unsigned v = 1; v <= maximum_angular; ++v) {
    for (unsigned n = 0; n + v <= maximum_angular; ++n) {
      Jet value = pc[2] * auxiliary.at(n + 1, 0, 0, v - 1);
      if (v > 1) {
        value = value + static_cast<double>(v - 1) *
                            auxiliary.at(n + 1, 0, 0, v - 2);
      }
      auxiliary.at(n, 0, 0, v) = std::move(value);
    }
  }
  for (unsigned v = 0; v <= maximum_angular; ++v) {
    for (unsigned u = 1; u + v <= maximum_angular; ++u) {
      for (unsigned n = 0; n + u + v <= maximum_angular; ++n) {
        Jet value = pc[1] * auxiliary.at(n + 1, 0, u - 1, v);
        if (u > 1) {
          value = value + static_cast<double>(u - 1) *
                              auxiliary.at(n + 1, 0, u - 2, v);
        }
        auxiliary.at(n, 0, u, v) = std::move(value);
      }
    }
  }
  for (unsigned v = 0; v <= maximum_angular; ++v) {
    for (unsigned u = 0; u + v <= maximum_angular; ++u) {
      for (unsigned t = 1; t + u + v <= maximum_angular; ++t) {
        for (unsigned n = 0; n + t + u + v <= maximum_angular; ++n) {
          Jet value = pc[0] * auxiliary.at(n + 1, t - 1, u, v);
          if (t > 1) {
            value = value + static_cast<double>(t - 1) *
                                auxiliary.at(n + 1, t - 2, u, v);
          }
          auxiliary.at(n, t, u, v) = std::move(value);
        }
      }
    }
  }
  return auxiliary;
}

Jet primitive_overlap_cartesian(
    double alpha,
    const Vec3& a,
    const molecule::CartesianComponent& angular_a,
    double beta,
    const Vec3& b,
    const molecule::CartesianComponent& angular_b) {
  const double p = alpha + beta;
  const Vec3 product = product_center(alpha, a, beta, b);
  Jet result(std::pow(std::numbers::pi / p, 1.5),
             a[0].derivative.size());
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const HermiteCoefficients coefficients =
        fill_hermite(angular_a[axis], angular_b[axis], product[axis],
                     a[axis], b[axis], alpha, beta);
    result = result * coefficients.at(angular_a[axis], angular_b[axis], 0);
  }
  return result;
}

Jet primitive_kinetic_cartesian(
    double alpha,
    const Vec3& a,
    const molecule::CartesianComponent& angular_a,
    double beta,
    const Vec3& b,
    const molecule::CartesianComponent& angular_b) {
  const unsigned total_b = angular_b[0] + angular_b[1] + angular_b[2];
  Jet result = beta * (2.0 * static_cast<double>(total_b) + 3.0) *
               primitive_overlap_cartesian(alpha, a, angular_a, beta, b,
                                           angular_b);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    molecule::CartesianComponent raised = angular_b;
    raised[axis] += 2;
    result = result - 2.0 * beta * beta *
                          primitive_overlap_cartesian(
                              alpha, a, angular_a, beta, b, raised);
    if (angular_b[axis] >= 2) {
      molecule::CartesianComponent lowered = angular_b;
      lowered[axis] -= 2;
      result = result -
               0.5 * static_cast<double>(angular_b[axis] *
                                         (angular_b[axis] - 1)) *
                   primitive_overlap_cartesian(alpha, a, angular_a, beta, b,
                                               lowered);
    }
  }
  return result;
}

Jet primitive_nuclear_attraction_cartesian(
    double alpha,
    const Vec3& a,
    const molecule::CartesianComponent& angular_a,
    double beta,
    const Vec3& b,
    const molecule::CartesianComponent& angular_b,
    const std::vector<Vec3>& atoms,
    const core::System& system) {
  const double p = alpha + beta;
  const Vec3 product = product_center(alpha, a, beta, b);
  std::array<HermiteCoefficients, 3> coefficients{
      fill_hermite(angular_a[0], angular_b[0], product[0], a[0], b[0], alpha,
                   beta),
      fill_hermite(angular_a[1], angular_b[1], product[1], a[1], b[1], alpha,
                   beta),
      fill_hermite(angular_a[2], angular_b[2], product[2], a[2], b[2], alpha,
                   beta)};
  const unsigned maximum = angular_a[0] + angular_a[1] + angular_a[2] +
                           angular_b[0] + angular_b[1] + angular_b[2];
  Jet result(0.0, a[0].derivative.size());
  for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
    const CoulombAuxiliary auxiliary =
        fill_coulomb(maximum, p, product, atoms[atom]);
    Jet value(0.0, a[0].derivative.size());
    for (unsigned t = 0; t <= angular_a[0] + angular_b[0]; ++t) {
      for (unsigned u = 0; u <= angular_a[1] + angular_b[1]; ++u) {
        for (unsigned v = 0; v <= angular_a[2] + angular_b[2]; ++v) {
          value = value +
                  coefficients[0].at(angular_a[0], angular_b[0], t) *
                      coefficients[1].at(angular_a[1], angular_b[1], u) *
                      coefficients[2].at(angular_a[2], angular_b[2], v) *
                      auxiliary.at(0, t, u, v);
        }
      }
    }
    result = result -
             static_cast<double>(system.atoms[atom].atomic_number) *
                 (2.0 * std::numbers::pi / p) * value;
  }
  return result;
}

Jet primitive_eri_cartesian(
    double alpha,
    const Vec3& a,
    const molecule::CartesianComponent& angular_a,
    double beta,
    const Vec3& b,
    const molecule::CartesianComponent& angular_b,
    double gamma,
    const Vec3& c,
    const molecule::CartesianComponent& angular_c,
    double delta,
    const Vec3& d,
    const molecule::CartesianComponent& angular_d) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double rho = p * q / (p + q);
  const Vec3 product_p = product_center(alpha, a, beta, b);
  const Vec3 product_q = product_center(gamma, c, delta, d);
  std::array<HermiteCoefficients, 3> first_coefficients{
      fill_hermite(angular_a[0], angular_b[0], product_p[0], a[0], b[0],
                   alpha, beta),
      fill_hermite(angular_a[1], angular_b[1], product_p[1], a[1], b[1],
                   alpha, beta),
      fill_hermite(angular_a[2], angular_b[2], product_p[2], a[2], b[2],
                   alpha, beta)};
  std::array<HermiteCoefficients, 3> second_coefficients{
      fill_hermite(angular_c[0], angular_d[0], product_q[0], c[0], d[0],
                   gamma, delta),
      fill_hermite(angular_c[1], angular_d[1], product_q[1], c[1], d[1],
                   gamma, delta),
      fill_hermite(angular_c[2], angular_d[2], product_q[2], c[2], d[2],
                   gamma, delta)};
  const unsigned maximum =
      angular_a[0] + angular_a[1] + angular_a[2] +
      angular_b[0] + angular_b[1] + angular_b[2] +
      angular_c[0] + angular_c[1] + angular_c[2] +
      angular_d[0] + angular_d[1] + angular_d[2];
  const CoulombAuxiliary auxiliary =
      fill_coulomb(maximum, rho, product_p, product_q);

  Jet value(0.0, a[0].derivative.size());
  for (unsigned t = 0; t <= angular_a[0] + angular_b[0]; ++t) {
    for (unsigned u = 0; u <= angular_a[1] + angular_b[1]; ++u) {
      for (unsigned v = 0; v <= angular_a[2] + angular_b[2]; ++v) {
        const Jet first =
            first_coefficients[0].at(angular_a[0], angular_b[0], t) *
            first_coefficients[1].at(angular_a[1], angular_b[1], u) *
            first_coefficients[2].at(angular_a[2], angular_b[2], v);
        for (unsigned tau = 0; tau <= angular_c[0] + angular_d[0]; ++tau) {
          for (unsigned nu = 0; nu <= angular_c[1] + angular_d[1]; ++nu) {
            for (unsigned phi = 0; phi <= angular_c[2] + angular_d[2]; ++phi) {
              const double sign = ((tau + nu + phi) & 1U) == 0 ? 1.0 : -1.0;
              value = value + sign * first *
                  second_coefficients[0].at(angular_c[0], angular_d[0], tau) *
                  second_coefficients[1].at(angular_c[1], angular_d[1], nu) *
                  second_coefficients[2].at(angular_c[2], angular_d[2], phi) *
                  auxiliary.at(0, t + tau, u + nu, v + phi);
            }
          }
        }
      }
    }
  }
  const double prefactor =
      2.0 * std::pow(std::numbers::pi, 2.5) /
      (p * q * std::sqrt(p + q));
  return prefactor * value;
}

struct AoView {
  const core::Shell* shell{};
  molecule::CartesianComponent angular{};
  double component_normalization{};
};

std::vector<AoView> expand_cartesian_aos(const core::System& system) {
  std::vector<AoView> aos;
  aos.reserve(molecule::cartesian_ao_count(system));
  for (const core::Shell& shell : system.shells) {
    for (const molecule::CartesianComponent& component :
         molecule::cartesian_components(shell.angular_momentum)) {
      aos.push_back(
          {&shell, component,
           molecule::cartesian_component_normalization(component)});
    }
  }
  return aos;
}

struct GlobalExpansionTerm {
  std::size_t cartesian_ao{};
  double coefficient{};
};

using GlobalAoExpansion = std::vector<GlobalExpansionTerm>;

std::vector<GlobalAoExpansion> spherical_expansions(
    const core::System& system) {
  std::vector<GlobalAoExpansion> expansions;
  expansions.reserve(molecule::ao_count(system));
  std::size_t cartesian_offset = 0;
  for (const core::Shell& shell : system.shells) {
    const std::vector<molecule::CartesianComponent> components =
        molecule::cartesian_components(shell.angular_momentum);
    for (const molecule::AoExpansion& shell_expansion :
         molecule::ao_expansions(
             shell.angular_momentum, VIBEQC_BASIS_SPHERICAL)) {
      GlobalAoExpansion expansion;
      expansion.reserve(shell_expansion.size());
      for (const molecule::CartesianExpansionTerm& term : shell_expansion) {
        const auto component = std::find(
            components.begin(), components.end(), term.component);
        if (component == components.end()) {
          throw std::logic_error(
              "spherical expansion references an unknown Cartesian AO");
        }
        expansion.push_back(
            {cartesian_offset +
                 static_cast<std::size_t>(component - components.begin()),
             term.coefficient});
      }
      expansions.push_back(std::move(expansion));
    }
    cartesian_offset += components.size();
  }
  return expansions;
}

std::vector<GlobalAoExpansion> public_ao_expansions(
    const core::System& system) {
  if (system.basis_representation == VIBEQC_BASIS_SPHERICAL) {
    return spherical_expansions(system);
  }
  std::vector<GlobalAoExpansion> expansions;
  const std::size_t count = molecule::cartesian_ao_count(system);
  expansions.reserve(count);
  for (std::size_t ao = 0; ao < count; ++ao) {
    expansions.push_back({{ao, 1.0}});
  }
  return expansions;
}

std::size_t matrix_index(std::size_t i, std::size_t j, std::size_t n) {
  return i * n + j;
}

std::size_t three_center_index(std::size_t i,
                               std::size_t j,
                               std::size_t auxiliary,
                               std::size_t n,
                               std::size_t naux) {
  return (i * n + j) * naux + auxiliary;
}

std::size_t eri_index(std::size_t i,
                      std::size_t j,
                      std::size_t k,
                      std::size_t l,
                      std::size_t n) {
  return ((i * n + j) * n + k) * n + l;
}

void unpack_jets(const std::vector<Jet>& source,
                 std::vector<double>& values,
                 std::vector<double>& derivatives,
                 std::size_t ncoord) {
  values.resize(source.size());
  derivatives.resize(source.size() * ncoord);
  for (std::size_t item = 0; item < source.size(); ++item) {
    values[item] = source[item].value;
    for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate) {
      derivatives[coordinate * source.size() + item] =
          source[item].derivative[coordinate];
    }
  }
}

std::vector<double> transform_matrix(
    const double* source,
    std::size_t cartesian_count,
    const std::vector<GlobalAoExpansion>& target_aos) {
  const std::size_t target_count = target_aos.size();
  std::vector<double> transformed(target_count * target_count, 0.0);
  for (std::size_t p = 0; p < target_count; ++p) {
    for (std::size_t q = 0; q < target_count; ++q) {
      double value = 0.0;
      for (const GlobalExpansionTerm& i : target_aos[p]) {
        for (const GlobalExpansionTerm& j : target_aos[q]) {
          value += i.coefficient * j.coefficient *
                   source[matrix_index(
                       i.cartesian_ao, j.cartesian_ao, cartesian_count)];
        }
      }
      transformed[matrix_index(p, q, target_count)] = value;
    }
  }
  return transformed;
}

std::vector<double> transform_eri(
    const double* source,
    std::size_t cartesian_count,
    const std::vector<GlobalAoExpansion>& target_aos) {
  const std::size_t target_count = target_aos.size();
  std::vector<double> transformed(
      target_count * target_count * target_count * target_count, 0.0);
  for (std::size_t p = 0; p < target_count; ++p) {
    for (std::size_t q = 0; q < target_count; ++q) {
      for (std::size_t r = 0; r < target_count; ++r) {
        for (std::size_t s = 0; s < target_count; ++s) {
          double value = 0.0;
          for (const GlobalExpansionTerm& i : target_aos[p]) {
            for (const GlobalExpansionTerm& j : target_aos[q]) {
              for (const GlobalExpansionTerm& k : target_aos[r]) {
                for (const GlobalExpansionTerm& l : target_aos[s]) {
                  value += i.coefficient * j.coefficient * k.coefficient *
                           l.coefficient *
                           source[eri_index(
                               i.cartesian_ao, j.cartesian_ao,
                               k.cartesian_ao, l.cartesian_ao,
                               cartesian_count)];
                }
              }
            }
          }
          transformed[eri_index(p, q, r, s, target_count)] = value;
        }
      }
    }
  }
  return transformed;
}

std::vector<double> transform_three_center(
    const double* source,
    std::size_t cartesian_count,
    std::size_t cartesian_auxiliary_count,
    const std::vector<GlobalAoExpansion>& target_aos,
    const std::vector<GlobalAoExpansion>& target_auxiliary_aos) {
  const std::size_t target_count = target_aos.size();
  const std::size_t target_auxiliary_count = target_auxiliary_aos.size();
  std::vector<double> transformed(
      target_count * target_count * target_auxiliary_count, 0.0);
  for (std::size_t p = 0; p < target_count; ++p) {
    for (std::size_t q = 0; q < target_count; ++q) {
      for (std::size_t auxiliary = 0;
           auxiliary < target_auxiliary_count; ++auxiliary) {
        double value = 0.0;
        for (const GlobalExpansionTerm& i : target_aos[p]) {
          for (const GlobalExpansionTerm& j : target_aos[q]) {
            for (const GlobalExpansionTerm& item :
                 target_auxiliary_aos[auxiliary]) {
              value += i.coefficient * j.coefficient * item.coefficient *
                  source[three_center_index(
                      i.cartesian_ao, j.cartesian_ao, item.cartesian_ao,
                      cartesian_count, cartesian_auxiliary_count)];
            }
          }
        }
        transformed[three_center_index(
            p, q, auxiliary, target_count, target_auxiliary_count)] = value;
      }
    }
  }
  return transformed;
}

void require_matching_density_fitting_geometry(
    const core::System& orbital_system,
    const core::System& auxiliary_system) {
  if (orbital_system.atoms.size() != auxiliary_system.atoms.size()) {
    throw std::invalid_argument(
        "orbital and auxiliary systems must contain the same atoms");
  }
  for (std::size_t atom = 0; atom < orbital_system.atoms.size(); ++atom) {
    const core::Atom& orbital = orbital_system.atoms[atom];
    const core::Atom& auxiliary = auxiliary_system.atoms[atom];
    if (orbital.atomic_number != auxiliary.atomic_number ||
        orbital.position != auxiliary.position) {
      throw std::invalid_argument(
          "orbital and auxiliary systems must share exact geometry");
    }
  }
}

}  // namespace

DensityFittingIntegralData build_density_fitting_integrals(
    const core::System& orbital_system,
    const core::System& auxiliary_system) {
  require_matching_density_fitting_geometry(
      orbital_system, auxiliary_system);

  DensityFittingIntegralData cartesian;
  cartesian.nbf = molecule::cartesian_ao_count(orbital_system);
  cartesian.naux = molecule::cartesian_ao_count(auxiliary_system);
  cartesian.ncoord = orbital_system.atoms.size() * 3;
  const std::vector<AoView> orbital_aos =
      expand_cartesian_aos(orbital_system);
  const std::vector<AoView> auxiliary_aos =
      expand_cartesian_aos(auxiliary_system);

  std::vector<Vec3> atom_coordinates;
  atom_coordinates.reserve(orbital_system.atoms.size());
  for (std::size_t atom = 0; atom < orbital_system.atoms.size(); ++atom) {
    Vec3 position;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      position[axis] = Jet::variable(
          orbital_system.atoms[atom].position[axis], cartesian.ncoord,
          atom * 3 + axis);
    }
    atom_coordinates.push_back(std::move(position));
  }

  const molecule::CartesianComponent zero_angular{0, 0, 0};
  std::vector<Jet> metric(
      cartesian.naux * cartesian.naux, Jet(0.0, cartesian.ncoord));
  for (std::size_t p = 0; p < cartesian.naux; ++p) {
    const AoView& first_auxiliary = auxiliary_aos[p];
    const Vec3& first_center =
        atom_coordinates[first_auxiliary.shell->atom_index];
    for (std::size_t q = 0; q < cartesian.naux; ++q) {
      const AoView& second_auxiliary = auxiliary_aos[q];
      const Vec3& second_center =
          atom_coordinates[second_auxiliary.shell->atom_index];
      Jet value(0.0, cartesian.ncoord);
      const double component_factor =
          first_auxiliary.component_normalization *
          second_auxiliary.component_normalization;
      for (const core::Primitive& first_primitive :
           first_auxiliary.shell->primitives) {
        for (const core::Primitive& second_primitive :
             second_auxiliary.shell->primitives) {
          const double weight = component_factor *
              first_primitive.coefficient * second_primitive.coefficient;
          value = value + weight * primitive_eri_cartesian(
              first_primitive.exponent, first_center,
              first_auxiliary.angular, 0.0, first_center, zero_angular,
              second_primitive.exponent, second_center,
              second_auxiliary.angular, 0.0, second_center, zero_angular);
        }
      }
      metric[matrix_index(p, q, cartesian.naux)] = std::move(value);
    }
  }

  std::vector<Jet> three_center(
      cartesian.nbf * cartesian.nbf * cartesian.naux,
      Jet(0.0, cartesian.ncoord));
  for (std::size_t i = 0; i < cartesian.nbf; ++i) {
    const AoView& first_ao = orbital_aos[i];
    const Vec3& first_center = atom_coordinates[first_ao.shell->atom_index];
    for (std::size_t j = 0; j < cartesian.nbf; ++j) {
      const AoView& second_ao = orbital_aos[j];
      const Vec3& second_center = atom_coordinates[second_ao.shell->atom_index];
      for (std::size_t p = 0; p < cartesian.naux; ++p) {
        const AoView& auxiliary_ao = auxiliary_aos[p];
        const Vec3& auxiliary_center =
            atom_coordinates[auxiliary_ao.shell->atom_index];
        Jet value(0.0, cartesian.ncoord);
        const double component_factor = first_ao.component_normalization *
            second_ao.component_normalization *
            auxiliary_ao.component_normalization;
        for (const core::Primitive& first_primitive :
             first_ao.shell->primitives) {
          for (const core::Primitive& second_primitive :
               second_ao.shell->primitives) {
            for (const core::Primitive& auxiliary_primitive :
                 auxiliary_ao.shell->primitives) {
              const double weight = component_factor *
                  first_primitive.coefficient * second_primitive.coefficient *
                  auxiliary_primitive.coefficient;
              value = value + weight * primitive_eri_cartesian(
                  first_primitive.exponent, first_center, first_ao.angular,
                  second_primitive.exponent, second_center, second_ao.angular,
                  auxiliary_primitive.exponent, auxiliary_center,
                  auxiliary_ao.angular, 0.0, auxiliary_center, zero_angular);
            }
          }
        }
        three_center[three_center_index(
            i, j, p, cartesian.nbf, cartesian.naux)] = std::move(value);
      }
    }
  }

  unpack_jets(
      metric, cartesian.metric, cartesian.metric_derivative,
      cartesian.ncoord);
  unpack_jets(
      three_center, cartesian.three_center,
      cartesian.three_center_derivative, cartesian.ncoord);

  const std::vector<GlobalAoExpansion> target_orbital_aos =
      public_ao_expansions(orbital_system);
  const std::vector<GlobalAoExpansion> target_auxiliary_aos =
      public_ao_expansions(auxiliary_system);
  if (target_orbital_aos.size() == cartesian.nbf &&
      target_auxiliary_aos.size() == cartesian.naux) {
    return cartesian;
  }

  DensityFittingIntegralData transformed;
  transformed.nbf = target_orbital_aos.size();
  transformed.naux = target_auxiliary_aos.size();
  transformed.ncoord = cartesian.ncoord;
  transformed.metric = transform_matrix(
      cartesian.metric.data(), cartesian.naux, target_auxiliary_aos);
  transformed.three_center = transform_three_center(
      cartesian.three_center.data(), cartesian.nbf, cartesian.naux,
      target_orbital_aos, target_auxiliary_aos);
  const std::size_t cartesian_metric_size = cartesian.naux * cartesian.naux;
  const std::size_t cartesian_three_center_size =
      cartesian.nbf * cartesian.nbf * cartesian.naux;
  transformed.metric_derivative.reserve(
      transformed.ncoord * transformed.naux * transformed.naux);
  transformed.three_center_derivative.reserve(
      transformed.ncoord * transformed.nbf * transformed.nbf *
      transformed.naux);
  for (std::size_t coordinate = 0; coordinate < transformed.ncoord;
       ++coordinate) {
    std::vector<double> metric_derivative = transform_matrix(
        cartesian.metric_derivative.data() +
            coordinate * cartesian_metric_size,
        cartesian.naux, target_auxiliary_aos);
    std::vector<double> three_center_derivative = transform_three_center(
        cartesian.three_center_derivative.data() +
            coordinate * cartesian_three_center_size,
        cartesian.nbf, cartesian.naux, target_orbital_aos,
        target_auxiliary_aos);
    transformed.metric_derivative.insert(
        transformed.metric_derivative.end(), metric_derivative.begin(),
        metric_derivative.end());
    transformed.three_center_derivative.insert(
        transformed.three_center_derivative.end(),
        three_center_derivative.begin(), three_center_derivative.end());
  }
  return transformed;
}

DensityFittingIntegralData transform_density_fitting_integrals(
    const DensityFittingIntegralData& cartesian,
    const core::System& orbital_system,
    const core::System& auxiliary_system) {
  require_matching_density_fitting_geometry(orbital_system, auxiliary_system);
  const std::size_t cartesian_nbf =
      molecule::cartesian_ao_count(orbital_system);
  const std::size_t cartesian_naux =
      molecule::cartesian_ao_count(auxiliary_system);
  const std::size_t ncoord = orbital_system.atoms.size() * 3;
  const std::size_t metric_size = cartesian_naux * cartesian_naux;
  const std::size_t tensor_size = cartesian_nbf * cartesian_nbf * cartesian_naux;
  if (cartesian.nbf != cartesian_nbf || cartesian.naux != cartesian_naux ||
      cartesian.ncoord != ncoord || cartesian.metric.size() != metric_size ||
      cartesian.three_center.size() != tensor_size ||
      cartesian.metric_derivative.size() != ncoord * metric_size ||
      cartesian.three_center_derivative.size() != ncoord * tensor_size) {
    throw std::invalid_argument(
        "Cartesian density-fitting tensor dimensions are inconsistent");
  }

  const std::vector<GlobalAoExpansion> target_orbital_aos =
      public_ao_expansions(orbital_system);
  const std::vector<GlobalAoExpansion> target_auxiliary_aos =
      public_ao_expansions(auxiliary_system);
  if (target_orbital_aos.size() == cartesian_nbf &&
      target_auxiliary_aos.size() == cartesian_naux) {
    return cartesian;
  }

  DensityFittingIntegralData transformed;
  transformed.nbf = target_orbital_aos.size();
  transformed.naux = target_auxiliary_aos.size();
  transformed.ncoord = ncoord;
  transformed.metric = transform_matrix(
      cartesian.metric.data(), cartesian_naux, target_auxiliary_aos);
  transformed.three_center = transform_three_center(
      cartesian.three_center.data(), cartesian_nbf, cartesian_naux,
      target_orbital_aos, target_auxiliary_aos);
  transformed.metric_derivative.reserve(ncoord * transformed.naux * transformed.naux);
  transformed.three_center_derivative.reserve(
      ncoord * transformed.nbf * transformed.nbf * transformed.naux);
  for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate) {
    const std::vector<double> metric_derivative = transform_matrix(
        cartesian.metric_derivative.data() + coordinate * metric_size,
        cartesian_naux, target_auxiliary_aos);
    const std::vector<double> three_center_derivative = transform_three_center(
        cartesian.three_center_derivative.data() + coordinate * tensor_size,
        cartesian_nbf, cartesian_naux, target_orbital_aos,
        target_auxiliary_aos);
    transformed.metric_derivative.insert(
        transformed.metric_derivative.end(), metric_derivative.begin(),
        metric_derivative.end());
    transformed.three_center_derivative.insert(
        transformed.three_center_derivative.end(),
        three_center_derivative.begin(), three_center_derivative.end());
  }
  return transformed;
}

IntegralData transform_integrals(const IntegralData& cartesian,
                                 const core::System& system) {
  const std::size_t cartesian_nbf = molecule::cartesian_ao_count(system);
  const std::size_t ncoord = system.atoms.size() * 3;
  const std::size_t matrix_size = cartesian_nbf * cartesian_nbf;
  const std::size_t eri_size = matrix_size * matrix_size;
  if (cartesian.nbf != cartesian_nbf || cartesian.ncoord != ncoord ||
      cartesian.overlap.size() != matrix_size ||
      cartesian.hcore.size() != matrix_size ||
      cartesian.overlap_derivative.size() != ncoord * matrix_size ||
      cartesian.hcore_derivative.size() != ncoord * matrix_size ||
      (!cartesian.eri.empty() && cartesian.eri.size() != eri_size) ||
      (!cartesian.eri_derivative.empty() &&
       cartesian.eri_derivative.size() != ncoord * eri_size)) {
    throw std::invalid_argument(
        "Cartesian one-electron tensor dimensions are inconsistent");
  }
  const std::vector<GlobalAoExpansion> target_aos = public_ao_expansions(system);
  if (target_aos.size() == cartesian_nbf) return cartesian;

  IntegralData transformed;
  transformed.nbf = target_aos.size();
  transformed.ncoord = ncoord;
  transformed.overlap = transform_matrix(
      cartesian.overlap.data(), cartesian_nbf, target_aos);
  transformed.hcore = transform_matrix(
      cartesian.hcore.data(), cartesian_nbf, target_aos);
  if (!cartesian.eri.empty()) {
    transformed.eri = transform_eri(
        cartesian.eri.data(), cartesian_nbf, target_aos);
  }
  const std::size_t transformed_matrix_size = transformed.nbf * transformed.nbf;
  const std::size_t transformed_eri_size =
      transformed_matrix_size * transformed_matrix_size;
  transformed.overlap_derivative.reserve(ncoord * transformed_matrix_size);
  transformed.hcore_derivative.reserve(ncoord * transformed_matrix_size);
  if (!cartesian.eri_derivative.empty()) {
    transformed.eri_derivative.reserve(ncoord * transformed_eri_size);
  }
  for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate) {
    const std::vector<double> overlap_derivative = transform_matrix(
        cartesian.overlap_derivative.data() + coordinate * matrix_size,
        cartesian_nbf, target_aos);
    const std::vector<double> hcore_derivative = transform_matrix(
        cartesian.hcore_derivative.data() + coordinate * matrix_size,
        cartesian_nbf, target_aos);
    transformed.overlap_derivative.insert(
        transformed.overlap_derivative.end(), overlap_derivative.begin(),
        overlap_derivative.end());
    transformed.hcore_derivative.insert(
        transformed.hcore_derivative.end(), hcore_derivative.begin(),
        hcore_derivative.end());
    if (!cartesian.eri_derivative.empty()) {
      const std::vector<double> eri_derivative = transform_eri(
          cartesian.eri_derivative.data() + coordinate * eri_size,
          cartesian_nbf, target_aos);
      transformed.eri_derivative.insert(
          transformed.eri_derivative.end(), eri_derivative.begin(),
          eri_derivative.end());
    }
  }
  transformed.nuclear_repulsion = cartesian.nuclear_repulsion;
  transformed.nuclear_repulsion_derivative =
      cartesian.nuclear_repulsion_derivative;
  return transformed;
}

IntegralData build_integrals(const core::System& system) {
  IntegralData out;
  out.nbf = molecule::cartesian_ao_count(system);
  out.ncoord = system.atoms.size() * 3;
  const std::size_t n = out.nbf;
  const std::vector<AoView> aos = expand_cartesian_aos(system);

  std::vector<Vec3> atom_coordinates;
  atom_coordinates.reserve(system.atoms.size());
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom) {
    Vec3 position;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      position[axis] = Jet::variable(system.atoms[atom].position[axis],
                                     out.ncoord, atom * 3 + axis);
    }
    atom_coordinates.push_back(std::move(position));
  }

  std::vector<Jet> overlap(n * n, Jet(0.0, out.ncoord));
  std::vector<Jet> hcore(n * n, Jet(0.0, out.ncoord));
  std::vector<Jet> eri(n * n * n * n, Jet(0.0, out.ncoord));

  for (std::size_t i = 0; i < n; ++i) {
    const AoView& ao_i = aos[i];
    const Vec3& a = atom_coordinates[ao_i.shell->atom_index];
    for (std::size_t j = 0; j < n; ++j) {
      const AoView& ao_j = aos[j];
      const Vec3& b = atom_coordinates[ao_j.shell->atom_index];
      Jet sij(0.0, out.ncoord);
      Jet hij(0.0, out.ncoord);
      const double component_factor =
          ao_i.component_normalization * ao_j.component_normalization;
      for (const core::Primitive& pi : ao_i.shell->primitives) {
        for (const core::Primitive& pj : ao_j.shell->primitives) {
          const double weight =
              component_factor * pi.coefficient * pj.coefficient;
          sij = sij + weight * primitive_overlap_cartesian(
                                   pi.exponent, a, ao_i.angular,
                                   pj.exponent, b, ao_j.angular);
          hij = hij + weight *
              (primitive_kinetic_cartesian(
                   pi.exponent, a, ao_i.angular,
                   pj.exponent, b, ao_j.angular) +
               primitive_nuclear_attraction_cartesian(
                   pi.exponent, a, ao_i.angular,
                   pj.exponent, b, ao_j.angular,
                   atom_coordinates, system));
        }
      }
      overlap[matrix_index(i, j, n)] = std::move(sij);
      hcore[matrix_index(i, j, n)] = std::move(hij);
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    const AoView& ao_i = aos[i];
    const Vec3& a = atom_coordinates[ao_i.shell->atom_index];
    for (std::size_t j = 0; j < n; ++j) {
      const AoView& ao_j = aos[j];
      const Vec3& b = atom_coordinates[ao_j.shell->atom_index];
      for (std::size_t k = 0; k < n; ++k) {
        const AoView& ao_k = aos[k];
        const Vec3& c = atom_coordinates[ao_k.shell->atom_index];
        for (std::size_t l = 0; l < n; ++l) {
          const AoView& ao_l = aos[l];
          const Vec3& d = atom_coordinates[ao_l.shell->atom_index];
          Jet value(0.0, out.ncoord);
          const double component_factor =
              ao_i.component_normalization * ao_j.component_normalization *
              ao_k.component_normalization * ao_l.component_normalization;
          for (const core::Primitive& pi : ao_i.shell->primitives) {
            for (const core::Primitive& pj : ao_j.shell->primitives) {
              for (const core::Primitive& pk : ao_k.shell->primitives) {
                for (const core::Primitive& pl : ao_l.shell->primitives) {
                  const double weight = component_factor * pi.coefficient *
                                        pj.coefficient * pk.coefficient *
                                        pl.coefficient;
                  value = value + weight * primitive_eri_cartesian(
                      pi.exponent, a, ao_i.angular,
                      pj.exponent, b, ao_j.angular,
                      pk.exponent, c, ao_k.angular,
                      pl.exponent, d, ao_l.angular);
                }
              }
            }
          }
          eri[eri_index(i, j, k, l, n)] = std::move(value);
        }
      }
    }
  }

  Jet nuclear_repulsion(0.0, out.ncoord);
  for (std::size_t a = 0; a < system.atoms.size(); ++a) {
    for (std::size_t b = 0; b < a; ++b) {
      nuclear_repulsion = nuclear_repulsion +
          static_cast<double>(system.atoms[a].atomic_number *
                              system.atoms[b].atomic_number) /
              sqrt(distance_squared(atom_coordinates[a], atom_coordinates[b]));
    }
  }

  unpack_jets(overlap, out.overlap, out.overlap_derivative, out.ncoord);
  unpack_jets(hcore, out.hcore, out.hcore_derivative, out.ncoord);
  unpack_jets(eri, out.eri, out.eri_derivative, out.ncoord);
  out.nuclear_repulsion = nuclear_repulsion.value;
  out.nuclear_repulsion_derivative = std::move(nuclear_repulsion.derivative);
  if (system.basis_representation == VIBEQC_BASIS_SPHERICAL) {
    const std::vector<GlobalAoExpansion> target_aos =
        spherical_expansions(system);
    IntegralData spherical;
    spherical.nbf = target_aos.size();
    spherical.ncoord = out.ncoord;
    spherical.overlap = transform_matrix(
        out.overlap.data(), out.nbf, target_aos);
    spherical.hcore = transform_matrix(
        out.hcore.data(), out.nbf, target_aos);
    spherical.eri = transform_eri(out.eri.data(), out.nbf, target_aos);
    const std::size_t cartesian_matrix_size = out.nbf * out.nbf;
    const std::size_t cartesian_eri_size =
        cartesian_matrix_size * cartesian_matrix_size;
    const std::size_t spherical_matrix_size = spherical.nbf * spherical.nbf;
    const std::size_t spherical_eri_size =
        spherical_matrix_size * spherical_matrix_size;
    spherical.overlap_derivative.reserve(
        spherical.ncoord * spherical_matrix_size);
    spherical.hcore_derivative.reserve(
        spherical.ncoord * spherical_matrix_size);
    spherical.eri_derivative.reserve(
        spherical.ncoord * spherical_eri_size);
    for (std::size_t coordinate = 0; coordinate < spherical.ncoord;
         ++coordinate) {
      std::vector<double> overlap_derivative = transform_matrix(
          out.overlap_derivative.data() + coordinate * cartesian_matrix_size,
          out.nbf, target_aos);
      std::vector<double> hcore_derivative = transform_matrix(
          out.hcore_derivative.data() + coordinate * cartesian_matrix_size,
          out.nbf, target_aos);
      std::vector<double> eri_derivative = transform_eri(
          out.eri_derivative.data() + coordinate * cartesian_eri_size,
          out.nbf, target_aos);
      spherical.overlap_derivative.insert(
          spherical.overlap_derivative.end(), overlap_derivative.begin(),
          overlap_derivative.end());
      spherical.hcore_derivative.insert(
          spherical.hcore_derivative.end(), hcore_derivative.begin(),
          hcore_derivative.end());
      spherical.eri_derivative.insert(
          spherical.eri_derivative.end(), eri_derivative.begin(),
          eri_derivative.end());
    }
    spherical.nuclear_repulsion = out.nuclear_repulsion;
    spherical.nuclear_repulsion_derivative =
        std::move(out.nuclear_repulsion_derivative);
    return spherical;
  }
  return out;
}

}  // namespace vibeqc::integrals

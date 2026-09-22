#include "integrals/s_integrals.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include "generated_one_electron_st_cpu.hpp"
#include "integrals/ecp.hpp"
#include "integrals/generated_df_cpu.hpp"
#include "integrals/range_moments.hpp"
#include "molecule/basis.hpp"
#include "posthf/raw_source.hpp"

namespace vibeqc::integrals {
namespace {

std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("CPU integral tensor extent overflows size_t");
  return a * b;
}

std::size_t checked_sum(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::overflow_error("CPU integral tensor extent overflows size_t");
  return a + b;
}

// Dynamic forward derivatives remain the structurally independent host reference
// for raw integral validation and for families not yet promoted to generated CPU
// production code.  Production s/p/d/f overlap/kinetic/nuclear-attraction instead
// consumes the same compiler-owned mathematical DAG used by the CUDA one-electron lowering.
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
    out.derivative[i] = a.derivative[i] * b.value + a.value * b.derivative[i];
  }
  return out;
}

Jet operator/(const Jet& a, const Jet& b) {
  Jet out(a.value / b.value, a.derivative.size());
  const double inverse_square = 1.0 / (b.value * b.value);
  for (std::size_t i = 0; i < out.derivative.size(); ++i) {
    out.derivative[i] = (a.derivative[i] * b.value - a.value * b.derivative[i]) * inverse_square;
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
  return {(alpha * a[0] + beta * b[0]) / p, (alpha * a[1] + beta * b[1]) / p,
          (alpha * a[2] + beta * b[2]) / p};
}

std::vector<Jet> boys_values(unsigned maximum_order, const Jet& argument) {
  // A positive series avoids cancellation near T=0 and unstable upward
  // recurrence when the requested order exceeds T. Include F_(n+1) so
  // nuclear derivatives use the exact identity even at coincident centers.
  const double t = argument.value;
  std::vector<double> scalar(maximum_order + 2);
  const double exponential = std::exp(-t);
  if (t < static_cast<double>(maximum_order) + 16.0) {
    for (unsigned n = 0; n < scalar.size(); ++n) {
      double term = 1.0 / (2.0 * n + 1.0);
      double sum = term;
      for (unsigned k = 1; k < 512; ++k) {
        term *= 2.0 * t / (2.0 * n + 2.0 * k + 1.0);
        sum += term;
        if (term <= sum * 2.0e-16) break;
      }
      scalar[n] = exponential * sum;
    }
  } else {
    scalar[0] = 0.5 * std::sqrt(std::numbers::pi / t) * std::erf(std::sqrt(t));
    for (unsigned n = 1; n < scalar.size(); ++n)
      scalar[n] = ((2.0 * n - 1.0) * scalar[n - 1] - exponential) / (2.0 * t);
  }
  std::vector<Jet> values(maximum_order + 1, Jet(0.0, argument.derivative.size()));
  for (unsigned n = 0; n <= maximum_order; ++n) {
    values[n].value = scalar[n];
    for (std::size_t coordinate = 0; coordinate < argument.derivative.size(); ++coordinate)
      values[n].derivative[coordinate] = -scalar[n + 1] * argument.derivative[coordinate];
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

HermiteCoefficients fill_hermite(unsigned maximum_i, unsigned maximum_j, const Jet& product,
                                 const Jet& center_a, const Jet& center_b, double alpha,
                                 double beta) {
  HermiteCoefficients coefficients;
  const unsigned idim = maximum_i + 1;
  coefficients.jdim = maximum_j + 1;
  coefficients.tdim = maximum_i + maximum_j + 2;
  coefficients.data.assign(static_cast<std::size_t>(idim) * coefficients.jdim * coefficients.tdim,
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
        coefficients.at(i, j, 0) = pa * coefficients.at(i - 1, j, 0) + coefficients.at(i - 1, j, 1);
      } else {
        coefficients.at(i, j, 0) = pb * coefficients.at(i, j - 1, 0) + coefficients.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coefficients.at(i, j, t) = pa * coefficients.at(i - 1, j, t) +
                                     inverse_two_p * coefficients.at(i - 1, j, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i - 1, j, t + 1);
        } else {
          coefficients.at(i, j, t) = pb * coefficients.at(i, j - 1, t) +
                                     inverse_two_p * coefficients.at(i, j - 1, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i, j - 1, t + 1);
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
    return data[(((static_cast<std::size_t>(n) * dim + t) * dim + u) * dim) + v];
  }
  const Jet& at(unsigned n, unsigned t, unsigned u, unsigned v) const {
    return data[(((static_cast<std::size_t>(n) * dim + t) * dim + u) * dim) + v];
  }
};

CoulombAuxiliary fill_coulomb_recurrence(unsigned maximum_angular, double exponent,
                                         const Vec3& product, const Vec3& center,
                                         const std::vector<Jet>& radial) {
  if (radial.size() != static_cast<std::size_t>(maximum_angular) + 1)
    throw std::invalid_argument("Coulomb radial moment count does not match angular order");
  CoulombAuxiliary auxiliary;
  auxiliary.dim = maximum_angular + 1;
  const std::size_t size =
      static_cast<std::size_t>(auxiliary.dim) * auxiliary.dim * auxiliary.dim * auxiliary.dim;
  auxiliary.data.assign(size, Jet(0.0, radial.front().derivative.size()));

  const Vec3 pc{product[0] - center[0], product[1] - center[1], product[2] - center[2]};
  double factor = 1.0;
  for (unsigned n = 0; n <= maximum_angular; ++n) {
    auxiliary.at(n, 0, 0, 0) = factor * radial[n];
    factor *= -2.0 * exponent;
  }

  for (unsigned v = 1; v <= maximum_angular; ++v) {
    for (unsigned n = 0; n + v <= maximum_angular; ++n) {
      Jet value = pc[2] * auxiliary.at(n + 1, 0, 0, v - 1);
      if (v > 1) value = value + static_cast<double>(v - 1) * auxiliary.at(n + 1, 0, 0, v - 2);
      auxiliary.at(n, 0, 0, v) = std::move(value);
    }
  }
  for (unsigned v = 0; v <= maximum_angular; ++v) {
    for (unsigned u = 1; u + v <= maximum_angular; ++u) {
      for (unsigned n = 0; n + u + v <= maximum_angular; ++n) {
        Jet value = pc[1] * auxiliary.at(n + 1, 0, u - 1, v);
        if (u > 1) value = value + static_cast<double>(u - 1) * auxiliary.at(n + 1, 0, u - 2, v);
        auxiliary.at(n, 0, u, v) = std::move(value);
      }
    }
  }
  for (unsigned v = 0; v <= maximum_angular; ++v) {
    for (unsigned u = 0; u + v <= maximum_angular; ++u) {
      for (unsigned t = 1; t + u + v <= maximum_angular; ++t) {
        for (unsigned n = 0; n + t + u + v <= maximum_angular; ++n) {
          Jet value = pc[0] * auxiliary.at(n + 1, t - 1, u, v);
          if (t > 1) value = value + static_cast<double>(t - 1) * auxiliary.at(n + 1, t - 2, u, v);
          auxiliary.at(n, t, u, v) = std::move(value);
        }
      }
    }
  }
  return auxiliary;
}

CoulombAuxiliary fill_coulomb(unsigned maximum_angular, double exponent, const Vec3& product,
                              const Vec3& center) {
  return fill_coulomb_recurrence(
      maximum_angular, exponent, product, center,
      boys_values(maximum_angular, exponent * distance_squared(product, center)));
}

CoulombAuxiliary fill_range_coulomb(unsigned maximum_angular, double exponent, const Vec3& product,
                                    const Vec3& center, CoulombRange range, double omega) {
  if (range == CoulombRange::Full)
    throw std::invalid_argument("range ERI helper requires short- or long-range operator");
  if (product[0].derivative.size() != 0 || center[0].derivative.size() != 0)
    throw std::logic_error("value-only range ERI helper cannot publish nuclear derivatives");
  if (maximum_angular > 13)
    throw std::invalid_argument("range ERI angular order exceeds validated radial moments");

  std::array<double, 14> moments{};
  if (!range_moments(maximum_angular, exponent * distance_squared(product, center).value, exponent,
                     range, omega, moments.data()))
    throw std::invalid_argument("invalid range-separated ERI radial inputs");
  std::vector<Jet> radial;
  radial.reserve(static_cast<std::size_t>(maximum_angular) + 1);
  for (unsigned n = 0; n <= maximum_angular; ++n) radial.emplace_back(moments[n], 0);
  return fill_coulomb_recurrence(maximum_angular, exponent, product, center, radial);
}

// Retained as the structurally independent S/T oracle and the explicit g-shell
// CPU fallback. Production s/p/d/f S/T uses the compiler-owned generated DAG.
Jet reference_overlap_cartesian(double alpha, const Vec3& a,
                                const molecule::CartesianComponent& angular_a, double beta,
                                const Vec3& b, const molecule::CartesianComponent& angular_b) {
  const double p = alpha + beta;
  const Vec3 product = product_center(alpha, a, beta, b);
  Jet result(std::pow(std::numbers::pi / p, 1.5), a[0].derivative.size());
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const HermiteCoefficients coefficients = fill_hermite(
        angular_a[axis], angular_b[axis], product[axis], a[axis], b[axis], alpha, beta);
    result = result * coefficients.at(angular_a[axis], angular_b[axis], 0);
  }
  return result;
}

Jet reference_kinetic_cartesian(double alpha, const Vec3& a,
                                const molecule::CartesianComponent& angular_a, double beta,
                                const Vec3& b, const molecule::CartesianComponent& angular_b) {
  const unsigned total_b = angular_b[0] + angular_b[1] + angular_b[2];
  Jet result = beta * (2.0 * static_cast<double>(total_b) + 3.0) *
               reference_overlap_cartesian(alpha, a, angular_a, beta, b, angular_b);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    molecule::CartesianComponent raised = angular_b;
    raised[axis] += 2;
    result = result -
             2.0 * beta * beta * reference_overlap_cartesian(alpha, a, angular_a, beta, b, raised);
    if (angular_b[axis] >= 2) {
      molecule::CartesianComponent lowered = angular_b;
      lowered[axis] -= 2;
      result = result - 0.5 * static_cast<double>(angular_b[axis] * (angular_b[axis] - 1)) *
                            reference_overlap_cartesian(alpha, a, angular_a, beta, b, lowered);
    }
  }
  return result;
}

struct ProductionST {
  Jet overlap;
  Jet kinetic;
};

unsigned generated_component(const molecule::CartesianComponent& angular) {
  if (angular[0] + angular[1] + angular[2] > 3U) return 20U;
  return generated_one_electron_cpu::component_index(angular[0], angular[1], angular[2]);
}

ProductionST production_overlap_kinetic_cartesian(double alpha, const Vec3& a,
                                                  const molecule::CartesianComponent& angular_a,
                                                  std::size_t atom_a, double beta, const Vec3& b,
                                                  const molecule::CartesianComponent& angular_b,
                                                  std::size_t atom_b) {
  const unsigned first = generated_component(angular_a);
  const unsigned second = generated_component(angular_b);
  const std::size_t ncoord = a[0].derivative.size();
  if (first >= 20 || second >= 20) {
    return {reference_overlap_cartesian(alpha, a, angular_a, beta, b, angular_b),
            reference_kinetic_cartesian(alpha, a, angular_a, beta, b, angular_b)};
  }
  const auto pair = generated_one_electron_cpu::make_pair(
      alpha, beta, a[0].value, a[1].value, a[2].value, b[0].value, b[1].value, b[2].value);
  const auto values = generated_one_electron_cpu::overlap_kinetic(pair, first, second);
  ProductionST result{Jet(values.overlap, ncoord), Jet(values.kinetic, ncoord)};
  if (ncoord == 0) return result;

  const auto gradient = generated_one_electron_cpu::overlap_kinetic_gradient(pair, first, second);
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const std::size_t ca = 3 * atom_a + axis, cb = 3 * atom_b + axis;
    result.overlap.derivative[ca] += gradient.first[axis];
    result.overlap.derivative[cb] -= gradient.first[axis];
    result.kinetic.derivative[ca] += gradient.second[axis];
    result.kinetic.derivative[cb] -= gradient.second[axis];
  }
  return result;
}

double production_overlap_value_cartesian(double alpha, const Vec3& a,
                                          const molecule::CartesianComponent& angular_a,
                                          double beta, const Vec3& b,
                                          const molecule::CartesianComponent& angular_b) {
  const unsigned first = generated_component(angular_a);
  const unsigned second = generated_component(angular_b);
  if (first >= 20 || second >= 20)
    return reference_overlap_cartesian(alpha, a, angular_a, beta, b, angular_b).value;
  const auto pair = generated_one_electron_cpu::make_pair(
      alpha, beta, a[0].value, a[1].value, a[2].value, b[0].value, b[1].value, b[2].value);
  return generated_one_electron_cpu::overlap_kinetic(pair, first, second).overlap;
}

Jet primitive_coulomb_potential_cartesian(double alpha, const Vec3& a,
                                          const molecule::CartesianComponent& angular_a,
                                          double beta, const Vec3& b,
                                          const molecule::CartesianComponent& angular_b,
                                          const Vec3& center) {
  const double p = alpha + beta;
  const Vec3 product = product_center(alpha, a, beta, b);
  std::array<HermiteCoefficients, 3> coefficients{
      fill_hermite(angular_a[0], angular_b[0], product[0], a[0], b[0], alpha, beta),
      fill_hermite(angular_a[1], angular_b[1], product[1], a[1], b[1], alpha, beta),
      fill_hermite(angular_a[2], angular_b[2], product[2], a[2], b[2], alpha, beta)};
  const unsigned maximum =
      angular_a[0] + angular_a[1] + angular_a[2] + angular_b[0] + angular_b[1] + angular_b[2];
  const CoulombAuxiliary auxiliary = fill_coulomb(maximum, p, product, center);
  Jet value(0.0, a[0].derivative.size());
  for (unsigned t = 0; t <= angular_a[0] + angular_b[0]; ++t) {
    for (unsigned u = 0; u <= angular_a[1] + angular_b[1]; ++u) {
      for (unsigned v = 0; v <= angular_a[2] + angular_b[2]; ++v) {
        value = value + coefficients[0].at(angular_a[0], angular_b[0], t) *
                            coefficients[1].at(angular_a[1], angular_b[1], u) *
                            coefficients[2].at(angular_a[2], angular_b[2], v) *
                            auxiliary.at(0, t, u, v);
      }
    }
  }
  return (2.0 * std::numbers::pi / p) * value;
}

Jet primitive_nuclear_attraction_cartesian(double alpha, const Vec3& a,
                                           const molecule::CartesianComponent& angular_a,
                                           double beta, const Vec3& b,
                                           const molecule::CartesianComponent& angular_b,
                                           const std::vector<Vec3>& atoms,
                                           const core::System& system) {
  Jet result(0.0, a[0].derivative.size());
  for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
    result = result - static_cast<double>(system.atoms[atom].ionic_charge()) *
                          primitive_coulomb_potential_cartesian(alpha, a, angular_a, beta, b,
                                                                angular_b, atoms[atom]);
  }
  return result;
}

// Production s/p/d/f V reuses the compiler-owned one-electron DAG.  The
// dynamic-Jet implementation above remains the independent oracle and the
// explicit g-shell fallback.
Jet production_nuclear_attraction_cartesian(double alpha, const Vec3& a,
                                            const molecule::CartesianComponent& angular_a,
                                            std::size_t atom_a, double beta, const Vec3& b,
                                            const molecule::CartesianComponent& angular_b,
                                            std::size_t atom_b, const std::vector<Vec3>& atoms,
                                            const core::System& system) {
  const unsigned first = generated_component(angular_a);
  const unsigned second = generated_component(angular_b);
  const std::size_t ncoord = a[0].derivative.size();
  if (first >= 20 || second >= 20)
    return primitive_nuclear_attraction_cartesian(alpha, a, angular_a, beta, b, angular_b, atoms,
                                                  system);

  const auto pair = generated_one_electron_cpu::make_pair(
      alpha, beta, a[0].value, a[1].value, a[2].value, b[0].value, b[1].value, b[2].value);
  Jet result(0.0, ncoord);
  for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
    const Vec3& center = atoms[atom];
    const double charge = static_cast<double>(system.atoms[atom].ionic_charge());
    const double value = generated_one_electron_cpu::attraction(
        pair, first, second, center[0].value, center[1].value, center[2].value);
    // The generated unit-charge V and its derivatives already include -1/r.
    result.value += charge * value;
    if (ncoord == 0) continue;

    const auto gradient = generated_one_electron_cpu::attraction_gradient(
        pair, first, second, center[0].value, center[1].value, center[2].value);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double da = charge * gradient.first[axis];
      const double db = charge * gradient.second[axis];
      result.derivative[3 * atom_a + axis] += da;
      result.derivative[3 * atom_b + axis] += db;
      result.derivative[3 * atom + axis] -= da + db;
    }
  }
  return result;
}

Jet primitive_eri_cartesian(double alpha, const Vec3& a,
                            const molecule::CartesianComponent& angular_a, double beta,
                            const Vec3& b, const molecule::CartesianComponent& angular_b,
                            double gamma, const Vec3& c,
                            const molecule::CartesianComponent& angular_c, double delta,
                            const Vec3& d, const molecule::CartesianComponent& angular_d) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double rho = p * q / (p + q);
  const Vec3 product_p = product_center(alpha, a, beta, b);
  const Vec3 product_q = product_center(gamma, c, delta, d);
  std::array<HermiteCoefficients, 3> first_coefficients{
      fill_hermite(angular_a[0], angular_b[0], product_p[0], a[0], b[0], alpha, beta),
      fill_hermite(angular_a[1], angular_b[1], product_p[1], a[1], b[1], alpha, beta),
      fill_hermite(angular_a[2], angular_b[2], product_p[2], a[2], b[2], alpha, beta)};
  std::array<HermiteCoefficients, 3> second_coefficients{
      fill_hermite(angular_c[0], angular_d[0], product_q[0], c[0], d[0], gamma, delta),
      fill_hermite(angular_c[1], angular_d[1], product_q[1], c[1], d[1], gamma, delta),
      fill_hermite(angular_c[2], angular_d[2], product_q[2], c[2], d[2], gamma, delta)};
  const unsigned maximum = angular_a[0] + angular_a[1] + angular_a[2] + angular_b[0] +
                           angular_b[1] + angular_b[2] + angular_c[0] + angular_c[1] +
                           angular_c[2] + angular_d[0] + angular_d[1] + angular_d[2];
  const CoulombAuxiliary auxiliary = fill_coulomb(maximum, rho, product_p, product_q);

  Jet value(0.0, a[0].derivative.size());
  for (unsigned t = 0; t <= angular_a[0] + angular_b[0]; ++t) {
    for (unsigned u = 0; u <= angular_a[1] + angular_b[1]; ++u) {
      for (unsigned v = 0; v <= angular_a[2] + angular_b[2]; ++v) {
        const Jet first = first_coefficients[0].at(angular_a[0], angular_b[0], t) *
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
  const double prefactor = 2.0 * std::pow(std::numbers::pi, 2.5) / (p * q * std::sqrt(p + q));
  return prefactor * value;
}

double primitive_range_eri_cartesian(double alpha, const Vec3& a,
                                     const molecule::CartesianComponent& angular_a, double beta,
                                     const Vec3& b, const molecule::CartesianComponent& angular_b,
                                     double gamma, const Vec3& c,
                                     const molecule::CartesianComponent& angular_c, double delta,
                                     const Vec3& d, const molecule::CartesianComponent& angular_d,
                                     CoulombRange range, double omega) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double rho = p * q / (p + q);
  const Vec3 product_p = product_center(alpha, a, beta, b);
  const Vec3 product_q = product_center(gamma, c, delta, d);
  std::array<HermiteCoefficients, 3> first_coefficients{
      fill_hermite(angular_a[0], angular_b[0], product_p[0], a[0], b[0], alpha, beta),
      fill_hermite(angular_a[1], angular_b[1], product_p[1], a[1], b[1], alpha, beta),
      fill_hermite(angular_a[2], angular_b[2], product_p[2], a[2], b[2], alpha, beta)};
  std::array<HermiteCoefficients, 3> second_coefficients{
      fill_hermite(angular_c[0], angular_d[0], product_q[0], c[0], d[0], gamma, delta),
      fill_hermite(angular_c[1], angular_d[1], product_q[1], c[1], d[1], gamma, delta),
      fill_hermite(angular_c[2], angular_d[2], product_q[2], c[2], d[2], gamma, delta)};
  const unsigned maximum = angular_a[0] + angular_a[1] + angular_a[2] + angular_b[0] +
                           angular_b[1] + angular_b[2] + angular_c[0] + angular_c[1] +
                           angular_c[2] + angular_d[0] + angular_d[1] + angular_d[2];
  const CoulombAuxiliary auxiliary =
      fill_range_coulomb(maximum, rho, product_p, product_q, range, omega);

  Jet value(0.0, 0);
  for (unsigned t = 0; t <= angular_a[0] + angular_b[0]; ++t) {
    for (unsigned u = 0; u <= angular_a[1] + angular_b[1]; ++u) {
      for (unsigned v = 0; v <= angular_a[2] + angular_b[2]; ++v) {
        const Jet first = first_coefficients[0].at(angular_a[0], angular_b[0], t) *
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
  const double prefactor = 2.0 * std::pow(std::numbers::pi, 2.5) / (p * q * std::sqrt(p + q));
  return prefactor * value.value;
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
      aos.push_back({&shell, component, molecule::cartesian_component_normalization(component)});
    }
  }
  return aos;
}

struct GlobalExpansionTerm {
  std::size_t cartesian_ao{};
  double coefficient{};
};

using GlobalAoExpansion = std::vector<GlobalExpansionTerm>;

std::vector<GlobalAoExpansion> spherical_expansions(const core::System& system) {
  std::vector<GlobalAoExpansion> expansions;
  expansions.reserve(molecule::ao_count(system));
  std::size_t cartesian_offset = 0;
  for (const core::Shell& shell : system.shells) {
    const std::vector<molecule::CartesianComponent> components =
        molecule::cartesian_components(shell.angular_momentum);
    for (const molecule::AoExpansion& shell_expansion :
         molecule::ao_expansions(shell.angular_momentum, VIBEQC_BASIS_SPHERICAL)) {
      GlobalAoExpansion expansion;
      expansion.reserve(shell_expansion.size());
      for (const molecule::CartesianExpansionTerm& term : shell_expansion) {
        const auto component = std::find(components.begin(), components.end(), term.component);
        if (component == components.end()) {
          throw std::logic_error("spherical expansion references an unknown Cartesian AO");
        }
        expansion.push_back(
            {cartesian_offset + static_cast<std::size_t>(component - components.begin()),
             term.coefficient});
      }
      expansions.push_back(std::move(expansion));
    }
    cartesian_offset += components.size();
  }
  return expansions;
}

std::vector<GlobalAoExpansion> public_ao_expansions(const core::System& system) {
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

std::size_t matrix_index(std::size_t i, std::size_t j, std::size_t n) { return i * n + j; }

std::size_t three_center_index(std::size_t i, std::size_t j, std::size_t auxiliary, std::size_t n,
                               std::size_t naux) {
  return (i * n + j) * naux + auxiliary;
}

std::size_t eri_index(std::size_t i, std::size_t j, std::size_t k, std::size_t l, std::size_t n) {
  return ((i * n + j) * n + k) * n + l;
}

void unpack_jets(const std::vector<Jet>& source, std::vector<double>& values,
                 std::vector<double>& derivatives, std::size_t ncoord) {
  values.resize(source.size());
  derivatives.resize(source.size() * ncoord);
  for (std::size_t item = 0; item < source.size(); ++item) {
    values[item] = source[item].value;
    for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate) {
      derivatives[coordinate * source.size() + item] = source[item].derivative[coordinate];
    }
  }
}

std::vector<double> transform_matrix(const double* source, std::size_t cartesian_count,
                                     const std::vector<GlobalAoExpansion>& target_aos) {
  const std::size_t target_count = target_aos.size();
  std::vector<double> transformed(target_count * target_count, 0.0);
  for (std::size_t p = 0; p < target_count; ++p) {
    for (std::size_t q = 0; q < target_count; ++q) {
      double value = 0.0;
      for (const GlobalExpansionTerm& i : target_aos[p]) {
        for (const GlobalExpansionTerm& j : target_aos[q]) {
          value += i.coefficient * j.coefficient *
                   source[matrix_index(i.cartesian_ao, j.cartesian_ao, cartesian_count)];
        }
      }
      transformed[matrix_index(p, q, target_count)] = value;
    }
  }
  return transformed;
}

std::vector<double> transform_eri(const double* source, std::size_t cartesian_count,
                                  const std::vector<GlobalAoExpansion>& target_aos) {
  const std::size_t target_count = target_aos.size();
  std::vector<double> transformed(target_count * target_count * target_count * target_count, 0.0);
  for (std::size_t p = 0; p < target_count; ++p) {
    for (std::size_t q = 0; q < target_count; ++q) {
      for (std::size_t r = 0; r < target_count; ++r) {
        for (std::size_t s = 0; s < target_count; ++s) {
          double value = 0.0;
          for (const GlobalExpansionTerm& i : target_aos[p]) {
            for (const GlobalExpansionTerm& j : target_aos[q]) {
              for (const GlobalExpansionTerm& k : target_aos[r]) {
                for (const GlobalExpansionTerm& l : target_aos[s]) {
                  value += i.coefficient * j.coefficient * k.coefficient * l.coefficient *
                           source[eri_index(i.cartesian_ao, j.cartesian_ao, k.cartesian_ao,
                                            l.cartesian_ao, cartesian_count)];
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
    const double* source, std::size_t cartesian_count, std::size_t cartesian_auxiliary_count,
    const std::vector<GlobalAoExpansion>& target_aos,
    const std::vector<GlobalAoExpansion>& target_auxiliary_aos) {
  const std::size_t target_count = target_aos.size();
  const std::size_t target_auxiliary_count = target_auxiliary_aos.size();
  std::vector<double> transformed(target_count * target_count * target_auxiliary_count, 0.0);
  for (std::size_t p = 0; p < target_count; ++p) {
    for (std::size_t q = 0; q < target_count; ++q) {
      for (std::size_t auxiliary = 0; auxiliary < target_auxiliary_count; ++auxiliary) {
        double value = 0.0;
        for (const GlobalExpansionTerm& i : target_aos[p]) {
          for (const GlobalExpansionTerm& j : target_aos[q]) {
            for (const GlobalExpansionTerm& item : target_auxiliary_aos[auxiliary]) {
              value += i.coefficient * j.coefficient * item.coefficient *
                       source[three_center_index(i.cartesian_ao, j.cartesian_ao, item.cartesian_ao,
                                                 cartesian_count, cartesian_auxiliary_count)];
            }
          }
        }
        transformed[three_center_index(p, q, auxiliary, target_count, target_auxiliary_count)] =
            value;
      }
    }
  }
  return transformed;
}

std::vector<double> pullback_matrix_weights(std::span<const double> source,
                                            std::size_t cartesian_count,
                                            const std::vector<GlobalAoExpansion>& public_aos) {
  std::vector<double> result(checked_product(cartesian_count, cartesian_count), 0.0);
  const std::size_t public_count = public_aos.size();
  for (std::size_t p = 0; p < public_count; ++p) {
    for (std::size_t q = 0; q < public_count; ++q) {
      const double weight = source[matrix_index(p, q, public_count)];
      if (weight == 0.0) continue;
      for (const GlobalExpansionTerm& i : public_aos[p]) {
        for (const GlobalExpansionTerm& j : public_aos[q]) {
          result[matrix_index(i.cartesian_ao, j.cartesian_ao, cartesian_count)] +=
              weight * i.coefficient * j.coefficient;
        }
      }
    }
  }
  return result;
}

std::vector<double> pullback_three_center_weights(
    std::span<const double> source, std::size_t cartesian_count,
    std::size_t cartesian_auxiliary_count, const std::vector<GlobalAoExpansion>& public_aos,
    const std::vector<GlobalAoExpansion>& public_auxiliary_aos) {
  const std::size_t cartesian_matrix = checked_product(cartesian_count, cartesian_count);
  std::vector<double> result(checked_product(cartesian_matrix, cartesian_auxiliary_count), 0.0);
  const std::size_t public_count = public_aos.size();
  const std::size_t public_auxiliary_count = public_auxiliary_aos.size();
  for (std::size_t p = 0; p < public_count; ++p) {
    for (std::size_t q = 0; q < public_count; ++q) {
      for (std::size_t auxiliary = 0; auxiliary < public_auxiliary_count; ++auxiliary) {
        const double weight =
            source[three_center_index(p, q, auxiliary, public_count, public_auxiliary_count)];
        if (weight == 0.0) continue;
        for (const GlobalExpansionTerm& i : public_aos[p]) {
          for (const GlobalExpansionTerm& j : public_aos[q]) {
            for (const GlobalExpansionTerm& item : public_auxiliary_aos[auxiliary]) {
              result[three_center_index(i.cartesian_ao, j.cartesian_ao, item.cartesian_ao,
                                        cartesian_count, cartesian_auxiliary_count)] +=
                  weight * i.coefficient * j.coefficient * item.coefficient;
            }
          }
        }
      }
    }
  }
  return result;
}
void require_matching_density_fitting_geometry(const core::System& orbital_system,
                                               const core::System& auxiliary_system) {
  if (orbital_system.atoms.size() != auxiliary_system.atoms.size()) {
    throw std::invalid_argument("orbital and auxiliary systems must contain the same atoms");
  }
  for (std::size_t atom = 0; atom < orbital_system.atoms.size(); ++atom) {
    const core::Atom& orbital = orbital_system.atoms[atom];
    const core::Atom& auxiliary = auxiliary_system.atoms[atom];
    if (orbital.atomic_number != auxiliary.atomic_number ||
        orbital.position != auxiliary.position) {
      throw std::invalid_argument("orbital and auxiliary systems must share exact geometry");
    }
  }
}

}  // namespace

std::vector<double> contract_weighted_density_fitting_derivative(
    const core::System& orbital_system, const core::System& auxiliary_system,
    std::span<const double> metric_weights, std::span<const double> three_center_weights) {
  require_matching_density_fitting_geometry(orbital_system, auxiliary_system);
  const std::size_t public_nbf = molecule::ao_count(orbital_system);
  const std::size_t public_naux = molecule::ao_count(auxiliary_system);
  const std::size_t public_matrix = checked_product(public_nbf, public_nbf);
  const std::size_t metric_size = checked_product(public_naux, public_naux);
  const std::size_t three_center_size = checked_product(public_matrix, public_naux);
  if (metric_weights.size() != metric_size || three_center_weights.size() != three_center_size) {
    throw std::invalid_argument("weighted DF derivative dimensions are inconsistent");
  }
  const auto finite = [](double value) { return std::isfinite(value); };
  if (!std::all_of(metric_weights.begin(), metric_weights.end(), finite) ||
      !std::all_of(three_center_weights.begin(), three_center_weights.end(), finite)) {
    throw std::invalid_argument("weighted DF derivative requires finite weights");
  }

  const bool generated_supported =
      std::all_of(orbital_system.shells.begin(), orbital_system.shells.end(),
                  [](const core::Shell& shell) { return shell.angular_momentum <= 3; }) &&
      std::all_of(auxiliary_system.shells.begin(), auxiliary_system.shells.end(),
                  [](const core::Shell& shell) { return shell.angular_momentum <= 3; });
  const std::size_t ncoord = checked_product(orbital_system.atoms.size(), std::size_t{3});
  if (!generated_supported) {
    const DensityFittingIntegralData full =
        build_density_fitting_integrals(orbital_system, auxiliary_system, true);
    std::vector<double> result(ncoord, 0.0);
    for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate) {
      const double* dm = full.metric_derivative.data() + coordinate * metric_size;
      const double* db = full.three_center_derivative.data() + coordinate * three_center_size;
      double value = 0.0;
      for (std::size_t item = 0; item < metric_size; ++item)
        value += metric_weights[item] * dm[item];
      for (std::size_t item = 0; item < three_center_size; ++item)
        value += three_center_weights[item] * db[item];
      result[coordinate] = value;
    }
    return result;
  }

  const std::vector<GlobalAoExpansion> public_aos = public_ao_expansions(orbital_system);
  const std::vector<GlobalAoExpansion> public_auxiliary_aos =
      public_ao_expansions(auxiliary_system);
  const std::vector<AoView> orbital_aos = expand_cartesian_aos(orbital_system);
  const std::vector<AoView> auxiliary_aos = expand_cartesian_aos(auxiliary_system);
  const std::size_t cartesian_nbf = orbital_aos.size();
  const std::size_t cartesian_naux = auxiliary_aos.size();
  const std::vector<double> cartesian_metric_weights =
      pullback_matrix_weights(metric_weights, cartesian_naux, public_auxiliary_aos);
  const std::vector<double> cartesian_three_center_weights = pullback_three_center_weights(
      three_center_weights, cartesian_nbf, cartesian_naux, public_aos, public_auxiliary_aos);

  using GeneratedAngular = generated_df_cpu::Angular;
  using GeneratedVec3 = generated_df_cpu::Vec3;
  const auto center = [&](std::size_t atom) {
    const auto& r = orbital_system.atoms[atom].position;
    return GeneratedVec3{r[0], r[1], r[2]};
  };
  const auto angular = [](const molecule::CartesianComponent& a) {
    return GeneratedAngular{a[0], a[1], a[2]};
  };
  std::vector<double> result(ncoord, 0.0);
  const auto scatter = [&](std::size_t atom, double weight, GeneratedVec3 value) {
    result[3 * atom] += weight * value.x;
    result[3 * atom + 1] += weight * value.y;
    result[3 * atom + 2] += weight * value.z;
  };

  for (std::size_t p = 0; p < cartesian_naux; ++p) {
    const AoView& first = auxiliary_aos[p];
    const auto first_center = center(first.shell->atom_index);
    for (std::size_t q = 0; q < cartesian_naux; ++q) {
      const double external = cartesian_metric_weights[matrix_index(p, q, cartesian_naux)];
      if (external == 0.0) continue;
      const AoView& second = auxiliary_aos[q];
      const auto second_center = center(second.shell->atom_index);
      const double component_factor =
          first.component_normalization * second.component_normalization;
      for (const auto& first_primitive : first.shell->primitives) {
        for (const auto& second_primitive : second.shell->primitives) {
          const double weight = external * component_factor * first_primitive.coefficient *
                                second_primitive.coefficient;
          const auto response = generated_df_cpu::metric_derivative(
              first_primitive.exponent, first_center, angular(first.angular),
              second_primitive.exponent, second_center, angular(second.angular));
          scatter(first.shell->atom_index, weight, response.first);
          scatter(second.shell->atom_index, weight, response.third);
        }
      }
    }
  }

  for (std::size_t i = 0; i < cartesian_nbf; ++i) {
    const AoView& first = orbital_aos[i];
    const auto first_center = center(first.shell->atom_index);
    for (std::size_t j = 0; j < cartesian_nbf; ++j) {
      const AoView& second = orbital_aos[j];
      const auto second_center = center(second.shell->atom_index);
      for (std::size_t p = 0; p < cartesian_naux; ++p) {
        const double external = cartesian_three_center_weights[three_center_index(
            i, j, p, cartesian_nbf, cartesian_naux)];
        if (external == 0.0) continue;
        const AoView& auxiliary = auxiliary_aos[p];
        const auto auxiliary_center = center(auxiliary.shell->atom_index);
        const double component_factor = first.component_normalization *
                                        second.component_normalization *
                                        auxiliary.component_normalization;
        for (const auto& first_primitive : first.shell->primitives) {
          for (const auto& second_primitive : second.shell->primitives) {
            for (const auto& auxiliary_primitive : auxiliary.shell->primitives) {
              const double weight = external * component_factor * first_primitive.coefficient *
                                    second_primitive.coefficient * auxiliary_primitive.coefficient;
              const auto response = generated_df_cpu::three_center_derivative(
                  first_primitive.exponent, first_center, angular(first.angular),
                  second_primitive.exponent, second_center, angular(second.angular),
                  auxiliary_primitive.exponent, auxiliary_center, angular(auxiliary.angular));
              scatter(first.shell->atom_index, weight, response.first);
              scatter(second.shell->atom_index, weight, response.second);
              scatter(auxiliary.shell->atom_index, weight, response.third);
            }
          }
        }
      }
    }
  }
  return result;
}
DensityFittingIntegralData build_density_fitting_integrals(const core::System& orbital_system,
                                                           const core::System& auxiliary_system,
                                                           bool include_derivatives) {
  require_matching_density_fitting_geometry(orbital_system, auxiliary_system);

  DensityFittingIntegralData cartesian;
  cartesian.nbf = molecule::cartesian_ao_count(orbital_system);
  cartesian.naux = molecule::cartesian_ao_count(auxiliary_system);
  cartesian.ncoord = include_derivatives ? orbital_system.atoms.size() * 3 : 0;
  const std::vector<AoView> orbital_aos = expand_cartesian_aos(orbital_system);
  const std::vector<AoView> auxiliary_aos = expand_cartesian_aos(auxiliary_system);

  const bool generated_supported =
      std::all_of(orbital_aos.begin(), orbital_aos.end(),
                  [](const AoView& ao) { return ao.shell->angular_momentum <= 3; }) &&
      std::all_of(auxiliary_aos.begin(), auxiliary_aos.end(),
                  [](const AoView& ao) { return ao.shell->angular_momentum <= 3; });
  const bool generated_derivatives = include_derivatives && generated_supported;
  const bool generated_values = !include_derivatives && generated_supported;
  if (generated_derivatives) {
    using GeneratedAngular = generated_df_cpu::Angular;
    using GeneratedVec3 = generated_df_cpu::Vec3;
    auto center = [&](std::size_t atom) {
      const auto& r = orbital_system.atoms[atom].position;
      return GeneratedVec3{r[0], r[1], r[2]};
    };
    auto angular = [](const molecule::CartesianComponent& a) {
      return GeneratedAngular{a[0], a[1], a[2]};
    };
    auto scatter = [](std::vector<double>& derivative, std::size_t stride, std::size_t item,
                      std::size_t atom, double weight, GeneratedVec3 value) {
      derivative[(3 * atom) * stride + item] += weight * value.x;
      derivative[(3 * atom + 1) * stride + item] += weight * value.y;
      derivative[(3 * atom + 2) * stride + item] += weight * value.z;
    };

    const std::size_t metric_size = cartesian.naux * cartesian.naux;
    cartesian.metric.assign(metric_size, 0.0);
    cartesian.metric_derivative.assign(cartesian.ncoord * metric_size, 0.0);
    for (std::size_t p = 0; p < cartesian.naux; ++p) {
      const AoView& first = auxiliary_aos[p];
      const auto first_center = center(first.shell->atom_index);
      for (std::size_t q = 0; q < cartesian.naux; ++q) {
        const AoView& second = auxiliary_aos[q];
        const auto second_center = center(second.shell->atom_index);
        const std::size_t item = matrix_index(p, q, cartesian.naux);
        const double component_factor =
            first.component_normalization * second.component_normalization;
        for (const auto& first_primitive : first.shell->primitives) {
          for (const auto& second_primitive : second.shell->primitives) {
            const double weight =
                component_factor * first_primitive.coefficient * second_primitive.coefficient;
            const auto response = generated_df_cpu::metric_derivative(
                first_primitive.exponent, first_center, angular(first.angular),
                second_primitive.exponent, second_center, angular(second.angular));
            cartesian.metric[item] += weight * response.value;
            scatter(cartesian.metric_derivative, metric_size, item, first.shell->atom_index, weight,
                    response.first);
            scatter(cartesian.metric_derivative, metric_size, item, second.shell->atom_index,
                    weight, response.third);
          }
        }
      }
    }

    const std::size_t tensor_size = cartesian.nbf * cartesian.nbf * cartesian.naux;
    cartesian.three_center.assign(tensor_size, 0.0);
    cartesian.three_center_derivative.assign(cartesian.ncoord * tensor_size, 0.0);
    for (std::size_t i = 0; i < cartesian.nbf; ++i) {
      const AoView& first = orbital_aos[i];
      const auto first_center = center(first.shell->atom_index);
      for (std::size_t j = 0; j < cartesian.nbf; ++j) {
        const AoView& second = orbital_aos[j];
        const auto second_center = center(second.shell->atom_index);
        for (std::size_t p = 0; p < cartesian.naux; ++p) {
          const AoView& auxiliary = auxiliary_aos[p];
          const auto auxiliary_center = center(auxiliary.shell->atom_index);
          const std::size_t item = three_center_index(i, j, p, cartesian.nbf, cartesian.naux);
          const double component_factor = first.component_normalization *
                                          second.component_normalization *
                                          auxiliary.component_normalization;
          for (const auto& first_primitive : first.shell->primitives) {
            for (const auto& second_primitive : second.shell->primitives) {
              for (const auto& auxiliary_primitive : auxiliary.shell->primitives) {
                const double weight = component_factor * first_primitive.coefficient *
                                      second_primitive.coefficient *
                                      auxiliary_primitive.coefficient;
                const auto response = generated_df_cpu::three_center_derivative(
                    first_primitive.exponent, first_center, angular(first.angular),
                    second_primitive.exponent, second_center, angular(second.angular),
                    auxiliary_primitive.exponent, auxiliary_center, angular(auxiliary.angular));
                cartesian.three_center[item] += weight * response.value;
                scatter(cartesian.three_center_derivative, tensor_size, item,
                        first.shell->atom_index, weight, response.first);
                scatter(cartesian.three_center_derivative, tensor_size, item,
                        second.shell->atom_index, weight, response.second);
                scatter(cartesian.three_center_derivative, tensor_size, item,
                        auxiliary.shell->atom_index, weight, response.third);
              }
            }
          }
        }
      }
    }
  } else if (generated_values) {
    using GeneratedAngular = generated_df_cpu::Angular;
    using GeneratedVec3 = generated_df_cpu::Vec3;
    auto center = [&](std::size_t atom) {
      const auto& r = orbital_system.atoms[atom].position;
      return GeneratedVec3{r[0], r[1], r[2]};
    };
    auto angular = [](const molecule::CartesianComponent& a) {
      return GeneratedAngular{a[0], a[1], a[2]};
    };

    const std::size_t metric_size = cartesian.naux * cartesian.naux;
    cartesian.metric.assign(metric_size, 0.0);
    for (std::size_t p = 0; p < cartesian.naux; ++p) {
      const AoView& first = auxiliary_aos[p];
      const auto first_center = center(first.shell->atom_index);
      for (std::size_t q = 0; q < cartesian.naux; ++q) {
        const AoView& second = auxiliary_aos[q];
        const auto second_center = center(second.shell->atom_index);
        const std::size_t item = matrix_index(p, q, cartesian.naux);
        const double component_factor =
            first.component_normalization * second.component_normalization;
        for (const auto& first_primitive : first.shell->primitives) {
          for (const auto& second_primitive : second.shell->primitives) {
            const double weight =
                component_factor * first_primitive.coefficient * second_primitive.coefficient;
            cartesian.metric[item] +=
                weight * generated_df_cpu::metric_value(
                             first_primitive.exponent, first_center, angular(first.angular),
                             second_primitive.exponent, second_center, angular(second.angular));
          }
        }
      }
    }

    const std::size_t tensor_size = cartesian.nbf * cartesian.nbf * cartesian.naux;
    cartesian.three_center.assign(tensor_size, 0.0);
    for (std::size_t i = 0; i < cartesian.nbf; ++i) {
      const AoView& first = orbital_aos[i];
      const auto first_center = center(first.shell->atom_index);
      for (std::size_t j = 0; j < cartesian.nbf; ++j) {
        const AoView& second = orbital_aos[j];
        const auto second_center = center(second.shell->atom_index);
        for (std::size_t p = 0; p < cartesian.naux; ++p) {
          const AoView& auxiliary = auxiliary_aos[p];
          const auto auxiliary_center = center(auxiliary.shell->atom_index);
          const std::size_t item = three_center_index(i, j, p, cartesian.nbf, cartesian.naux);
          const double component_factor = first.component_normalization *
                                          second.component_normalization *
                                          auxiliary.component_normalization;
          for (const auto& first_primitive : first.shell->primitives) {
            for (const auto& second_primitive : second.shell->primitives) {
              for (const auto& auxiliary_primitive : auxiliary.shell->primitives) {
                const double weight = component_factor * first_primitive.coefficient *
                                      second_primitive.coefficient *
                                      auxiliary_primitive.coefficient;
                cartesian.three_center[item] +=
                    weight * generated_df_cpu::three_center_value(
                                 first_primitive.exponent, first_center, angular(first.angular),
                                 second_primitive.exponent, second_center, angular(second.angular),
                                 auxiliary_primitive.exponent, auxiliary_center,
                                 angular(auxiliary.angular));
              }
            }
          }
        }
      }
    }
  } else {
    std::vector<Vec3> atom_coordinates;
    atom_coordinates.reserve(orbital_system.atoms.size());
    for (std::size_t atom = 0; atom < orbital_system.atoms.size(); ++atom) {
      Vec3 position;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const double coordinate = orbital_system.atoms[atom].position[axis];
        position[axis] = include_derivatives
                             ? Jet::variable(coordinate, cartesian.ncoord, atom * 3 + axis)
                             : Jet(coordinate, 0);
      }
      atom_coordinates.push_back(std::move(position));
    }

    const molecule::CartesianComponent zero_angular{0, 0, 0};
    std::vector<Jet> metric(cartesian.naux * cartesian.naux, Jet(0.0, cartesian.ncoord));
    for (std::size_t p = 0; p < cartesian.naux; ++p) {
      const AoView& first_auxiliary = auxiliary_aos[p];
      const Vec3& first_center = atom_coordinates[first_auxiliary.shell->atom_index];
      for (std::size_t q = 0; q < cartesian.naux; ++q) {
        const AoView& second_auxiliary = auxiliary_aos[q];
        const Vec3& second_center = atom_coordinates[second_auxiliary.shell->atom_index];
        Jet value(0.0, cartesian.ncoord);
        const double component_factor =
            first_auxiliary.component_normalization * second_auxiliary.component_normalization;
        for (const core::Primitive& first_primitive : first_auxiliary.shell->primitives) {
          for (const core::Primitive& second_primitive : second_auxiliary.shell->primitives) {
            const double weight =
                component_factor * first_primitive.coefficient * second_primitive.coefficient;
            value =
                value + weight * primitive_eri_cartesian(first_primitive.exponent, first_center,
                                                         first_auxiliary.angular, 0.0, first_center,
                                                         zero_angular, second_primitive.exponent,
                                                         second_center, second_auxiliary.angular,
                                                         0.0, second_center, zero_angular);
          }
        }
        metric[matrix_index(p, q, cartesian.naux)] = std::move(value);
      }
    }

    std::vector<Jet> three_center(cartesian.nbf * cartesian.nbf * cartesian.naux,
                                  Jet(0.0, cartesian.ncoord));
    for (std::size_t i = 0; i < cartesian.nbf; ++i) {
      const AoView& first_ao = orbital_aos[i];
      const Vec3& first_center = atom_coordinates[first_ao.shell->atom_index];
      for (std::size_t j = 0; j < cartesian.nbf; ++j) {
        const AoView& second_ao = orbital_aos[j];
        const Vec3& second_center = atom_coordinates[second_ao.shell->atom_index];
        for (std::size_t p = 0; p < cartesian.naux; ++p) {
          const AoView& auxiliary_ao = auxiliary_aos[p];
          const Vec3& auxiliary_center = atom_coordinates[auxiliary_ao.shell->atom_index];
          Jet value(0.0, cartesian.ncoord);
          const double component_factor = first_ao.component_normalization *
                                          second_ao.component_normalization *
                                          auxiliary_ao.component_normalization;
          for (const core::Primitive& first_primitive : first_ao.shell->primitives) {
            for (const core::Primitive& second_primitive : second_ao.shell->primitives) {
              for (const core::Primitive& auxiliary_primitive : auxiliary_ao.shell->primitives) {
                const double weight = component_factor * first_primitive.coefficient *
                                      second_primitive.coefficient *
                                      auxiliary_primitive.coefficient;
                value = value +
                        weight * primitive_eri_cartesian(
                                     first_primitive.exponent, first_center, first_ao.angular,
                                     second_primitive.exponent, second_center, second_ao.angular,
                                     auxiliary_primitive.exponent, auxiliary_center,
                                     auxiliary_ao.angular, 0.0, auxiliary_center, zero_angular);
              }
            }
          }
          three_center[three_center_index(i, j, p, cartesian.nbf, cartesian.naux)] =
              std::move(value);
        }
      }
    }

    unpack_jets(metric, cartesian.metric, cartesian.metric_derivative, cartesian.ncoord);
    unpack_jets(three_center, cartesian.three_center, cartesian.three_center_derivative,
                cartesian.ncoord);
  }

  const std::vector<GlobalAoExpansion> target_orbital_aos = public_ao_expansions(orbital_system);
  const std::vector<GlobalAoExpansion> target_auxiliary_aos =
      public_ao_expansions(auxiliary_system);
  if (target_orbital_aos.size() == cartesian.nbf && target_auxiliary_aos.size() == cartesian.naux) {
    return cartesian;
  }

  DensityFittingIntegralData transformed;
  transformed.nbf = target_orbital_aos.size();
  transformed.naux = target_auxiliary_aos.size();
  transformed.ncoord = cartesian.ncoord;
  transformed.metric =
      transform_matrix(cartesian.metric.data(), cartesian.naux, target_auxiliary_aos);
  transformed.three_center =
      transform_three_center(cartesian.three_center.data(), cartesian.nbf, cartesian.naux,
                             target_orbital_aos, target_auxiliary_aos);
  const std::size_t cartesian_metric_size = cartesian.naux * cartesian.naux;
  const std::size_t cartesian_three_center_size = cartesian.nbf * cartesian.nbf * cartesian.naux;
  transformed.metric_derivative.reserve(transformed.ncoord * transformed.naux * transformed.naux);
  transformed.three_center_derivative.reserve(transformed.ncoord * transformed.nbf *
                                              transformed.nbf * transformed.naux);
  for (std::size_t coordinate = 0; coordinate < transformed.ncoord; ++coordinate) {
    std::vector<double> metric_derivative =
        transform_matrix(cartesian.metric_derivative.data() + coordinate * cartesian_metric_size,
                         cartesian.naux, target_auxiliary_aos);
    std::vector<double> three_center_derivative = transform_three_center(
        cartesian.three_center_derivative.data() + coordinate * cartesian_three_center_size,
        cartesian.nbf, cartesian.naux, target_orbital_aos, target_auxiliary_aos);
    transformed.metric_derivative.insert(transformed.metric_derivative.end(),
                                         metric_derivative.begin(), metric_derivative.end());
    transformed.three_center_derivative.insert(transformed.three_center_derivative.end(),
                                               three_center_derivative.begin(),
                                               three_center_derivative.end());
  }
  return transformed;
}

DensityFittingIntegralData transform_density_fitting_integrals(
    const DensityFittingIntegralData& cartesian, const core::System& orbital_system,
    const core::System& auxiliary_system) {
  require_matching_density_fitting_geometry(orbital_system, auxiliary_system);
  const std::size_t cartesian_nbf = molecule::cartesian_ao_count(orbital_system);
  const std::size_t cartesian_naux = molecule::cartesian_ao_count(auxiliary_system);
  const std::size_t ncoord = orbital_system.atoms.size() * 3;
  const std::size_t metric_size = cartesian_naux * cartesian_naux;
  const std::size_t tensor_size = cartesian_nbf * cartesian_nbf * cartesian_naux;
  // A fused consumer may omit both derivative tensors. Partial omission is
  // invalid because the two arrays describe the same physical coordinate set.
  const bool derivatives =
      !cartesian.metric_derivative.empty() || !cartesian.three_center_derivative.empty();
  if (cartesian.nbf != cartesian_nbf || cartesian.naux != cartesian_naux ||
      cartesian.ncoord != ncoord || cartesian.metric.size() != metric_size ||
      cartesian.three_center.size() != tensor_size ||
      (derivatives && (cartesian.metric_derivative.size() != ncoord * metric_size ||
                       cartesian.three_center_derivative.size() != ncoord * tensor_size))) {
    throw std::invalid_argument("Cartesian density-fitting tensor dimensions are inconsistent");
  }

  const std::vector<GlobalAoExpansion> target_orbital_aos = public_ao_expansions(orbital_system);
  const std::vector<GlobalAoExpansion> target_auxiliary_aos =
      public_ao_expansions(auxiliary_system);
  if (target_orbital_aos.size() == cartesian_nbf && target_auxiliary_aos.size() == cartesian_naux) {
    return cartesian;
  }

  DensityFittingIntegralData transformed;
  transformed.nbf = target_orbital_aos.size();
  transformed.naux = target_auxiliary_aos.size();
  transformed.ncoord = ncoord;
  transformed.metric =
      transform_matrix(cartesian.metric.data(), cartesian_naux, target_auxiliary_aos);
  transformed.three_center =
      transform_three_center(cartesian.three_center.data(), cartesian_nbf, cartesian_naux,
                             target_orbital_aos, target_auxiliary_aos);
  if (!derivatives) return transformed;
  transformed.metric_derivative.reserve(ncoord * transformed.naux * transformed.naux);
  transformed.three_center_derivative.reserve(ncoord * transformed.nbf * transformed.nbf *
                                              transformed.naux);
  for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate) {
    const std::vector<double> metric_derivative =
        transform_matrix(cartesian.metric_derivative.data() + coordinate * metric_size,
                         cartesian_naux, target_auxiliary_aos);
    const std::vector<double> three_center_derivative = transform_three_center(
        cartesian.three_center_derivative.data() + coordinate * tensor_size, cartesian_nbf,
        cartesian_naux, target_orbital_aos, target_auxiliary_aos);
    transformed.metric_derivative.insert(transformed.metric_derivative.end(),
                                         metric_derivative.begin(), metric_derivative.end());
    transformed.three_center_derivative.insert(transformed.three_center_derivative.end(),
                                               three_center_derivative.begin(),
                                               three_center_derivative.end());
  }
  return transformed;
}

void cross_overlap(const core::System& target, const core::System& source,
                   std::span<double> output) {
  const auto nt = molecule::ao_count(target), ns = molecule::ao_count(source);
  if (nt == 0 || ns == 0 || nt > std::numeric_limits<std::size_t>::max() / ns ||
      output.size() != nt * ns) {
    throw std::invalid_argument("cross-overlap output must match target/source AO dimensions");
  }
  const auto target_cartesian = expand_cartesian_aos(target);
  const auto source_cartesian = expand_cartesian_aos(source);
  const auto target_public = public_ao_expansions(target);
  const auto source_public = public_ao_expansions(source);
  auto centers = [](const core::System& system) {
    std::vector<Vec3> result;
    result.reserve(system.atoms.size());
    for (const auto& atom : system.atoms) {
      result.push_back(
          {Jet(atom.position[0], 0), Jet(atom.position[1], 0), Jet(atom.position[2], 0)});
    }
    return result;
  };
  const auto target_centers = centers(target), source_centers = centers(source);
  // Expand only the AO pair being written. The bounded sparse spherical terms
  // avoid a second Cartesian rectangular matrix and preserve both AO orders.
  for (std::size_t i = 0; i < nt; ++i) {
    for (std::size_t j = 0; j < ns; ++j) {
      double value = 0;
      for (const auto& t : target_public[i]) {
        const auto& a = target_cartesian[t.cartesian_ao];
        for (const auto& s : source_public[j]) {
          const auto& b = source_cartesian[s.cartesian_ao];
          const double factor =
              t.coefficient * s.coefficient * a.component_normalization * b.component_normalization;
          for (const auto& p : a.shell->primitives) {
            for (const auto& q : b.shell->primitives) {
              value += factor * p.coefficient * q.coefficient *
                       production_overlap_value_cartesian(
                           p.exponent, target_centers[a.shell->atom_index], a.angular, q.exponent,
                           source_centers[b.shell->atom_index], b.angular);
            }
          }
        }
      }
      output[i * ns + j] = value;
    }
  }
}

namespace {

EspProbeDerivativeData build_esp_integrals_impl(const core::System& system,
                                                std::span<const double> points_xyz,
                                                bool include_probe_derivatives) {
  if (points_xyz.size() % 3 != 0) {
    throw std::invalid_argument("ESP probe coordinates must be xyz triples");
  }
  for (double coordinate : points_xyz) {
    if (!std::isfinite(coordinate)) throw std::invalid_argument("nonfinite ESP probe coordinate");
  }

  const auto cartesian_aos = expand_cartesian_aos(system);
  const auto public_aos = public_ao_expansions(system);
  if (cartesian_aos.empty() || public_aos.empty()) {
    throw std::invalid_argument("ESP integrals require a nonempty AO basis");
  }
  const std::size_t cartesian_nbf = cartesian_aos.size();
  const std::size_t nbf = public_aos.size();
  const std::size_t npoint = points_xyz.size() / 3;
  const std::size_t cartesian_matrix_size = checked_product(cartesian_nbf, cartesian_nbf);
  const std::size_t matrix_size = checked_product(nbf, nbf);
  const std::size_t derivative_count = include_probe_derivatives ? 3 : 0;

  std::vector<Vec3> centers;
  centers.reserve(system.atoms.size());
  for (const auto& atom : system.atoms) {
    centers.push_back({Jet(atom.position[0], derivative_count),
                       Jet(atom.position[1], derivative_count),
                       Jet(atom.position[2], derivative_count)});
  }

  EspProbeDerivativeData result;
  result.nbf = nbf;
  result.npoint = npoint;
  result.values.resize(checked_product(npoint, matrix_size));
  if (include_probe_derivatives)
    result.probe_derivative.resize(checked_product(checked_product(3, npoint), matrix_size));

  std::vector<double> cartesian(cartesian_matrix_size);
  std::vector<double> cartesian_derivative;
  if (include_probe_derivatives) cartesian_derivative.resize(3 * cartesian_matrix_size);

  for (std::size_t point = 0; point < npoint; ++point) {
    Vec3 probe;
    for (unsigned axis = 0; axis < 3; ++axis) {
      const double coordinate = points_xyz[3 * point + axis];
      probe[axis] =
          include_probe_derivatives ? Jet::variable(coordinate, 3, axis) : Jet(coordinate, 0);
    }

    for (std::size_t i = 0; i < cartesian_nbf; ++i) {
      const auto& first = cartesian_aos[i];
      const auto& first_center = centers[first.shell->atom_index];
      for (std::size_t j = 0; j <= i; ++j) {
        const auto& second = cartesian_aos[j];
        const auto& second_center = centers[second.shell->atom_index];
        Jet value(0.0, derivative_count);
        const double angular_normalization =
            first.component_normalization * second.component_normalization;
        for (const auto& p : first.shell->primitives) {
          for (const auto& q : second.shell->primitives) {
            value = value + angular_normalization * p.coefficient * q.coefficient *
                                primitive_coulomb_potential_cartesian(
                                    p.exponent, first_center, first.angular, q.exponent,
                                    second_center, second.angular, probe);
          }
        }
        const auto ij = matrix_index(i, j, cartesian_nbf);
        const auto ji = matrix_index(j, i, cartesian_nbf);
        cartesian[ij] = cartesian[ji] = value.value;
        if (include_probe_derivatives) {
          for (unsigned axis = 0; axis < 3; ++axis) {
            const auto offset = axis * cartesian_matrix_size;
            cartesian_derivative[offset + ij] = value.derivative[axis];
            cartesian_derivative[offset + ji] = value.derivative[axis];
          }
        }
      }
    }

    const double* source = cartesian.data();
    std::vector<double> transformed;
    if (nbf != cartesian_nbf) {
      transformed = transform_matrix(cartesian.data(), cartesian_nbf, public_aos);
      source = transformed.data();
    }
    std::copy(source, source + matrix_size, result.values.begin() + point * matrix_size);

    if (include_probe_derivatives) {
      for (unsigned axis = 0; axis < 3; ++axis) {
        const double* derivative_source =
            cartesian_derivative.data() + axis * cartesian_matrix_size;
        std::vector<double> transformed_derivative;
        if (nbf != cartesian_nbf) {
          transformed_derivative = transform_matrix(derivative_source, cartesian_nbf, public_aos);
          derivative_source = transformed_derivative.data();
        }
        std::copy(derivative_source, derivative_source + matrix_size,
                  result.probe_derivative.begin() + (3 * point + axis) * matrix_size);
      }
    }
  }
  return result;
}

}  // namespace

EspIntegralData build_esp_integrals(const core::System& system,
                                    std::span<const double> points_xyz) {
  auto result = build_esp_integrals_impl(system, points_xyz, false);
  return {result.nbf, result.npoint, std::move(result.values)};
}

EspProbeDerivativeData build_esp_integrals_with_probe_derivatives(
    const core::System& system, std::span<const double> points_xyz) {
  return build_esp_integrals_impl(system, points_xyz, true);
}

IntegralData transform_integrals(const IntegralData& cartesian, const core::System& system) {
  const std::size_t cartesian_nbf = molecule::cartesian_ao_count(system);
  const std::size_t ncoord = system.atoms.size() * 3;
  const std::size_t matrix_size = cartesian_nbf * cartesian_nbf;
  const std::size_t eri_size = matrix_size * matrix_size;
  // A fused derivative consumer needs only transformed values. Both AO
  // derivative arrays may be absent; a partial or malformed pair is invalid.
  const bool one_electron_derivatives =
      !cartesian.overlap_derivative.empty() || !cartesian.hcore_derivative.empty();
  if (cartesian.nbf != cartesian_nbf || cartesian.ncoord != ncoord ||
      cartesian.overlap.size() != matrix_size || cartesian.hcore.size() != matrix_size ||
      (one_electron_derivatives && (cartesian.overlap_derivative.size() != ncoord * matrix_size ||
                                    cartesian.hcore_derivative.size() != ncoord * matrix_size)) ||
      (!cartesian.eri.empty() && cartesian.eri.size() != eri_size) ||
      (!cartesian.eri_derivative.empty() && cartesian.eri_derivative.size() != ncoord * eri_size)) {
    throw std::invalid_argument("Cartesian one-electron tensor dimensions are inconsistent");
  }
  const std::vector<GlobalAoExpansion> target_aos = public_ao_expansions(system);
  if (target_aos.size() == cartesian_nbf) return cartesian;

  IntegralData transformed;
  transformed.nbf = target_aos.size();
  transformed.ncoord = ncoord;
  transformed.overlap = transform_matrix(cartesian.overlap.data(), cartesian_nbf, target_aos);
  transformed.hcore = transform_matrix(cartesian.hcore.data(), cartesian_nbf, target_aos);
  if (!cartesian.eri.empty()) {
    transformed.eri = transform_eri(cartesian.eri.data(), cartesian_nbf, target_aos);
  }
  const std::size_t transformed_matrix_size = transformed.nbf * transformed.nbf;
  const std::size_t transformed_eri_size = transformed_matrix_size * transformed_matrix_size;
  if (one_electron_derivatives) {
    transformed.overlap_derivative.reserve(ncoord * transformed_matrix_size);
    transformed.hcore_derivative.reserve(ncoord * transformed_matrix_size);
  }
  if (!cartesian.eri_derivative.empty()) {
    transformed.eri_derivative.reserve(ncoord * transformed_eri_size);
  }
  for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate) {
    if (one_electron_derivatives) {
      const std::vector<double> overlap_derivative =
          transform_matrix(cartesian.overlap_derivative.data() + coordinate * matrix_size,
                           cartesian_nbf, target_aos);
      const std::vector<double> hcore_derivative = transform_matrix(
          cartesian.hcore_derivative.data() + coordinate * matrix_size, cartesian_nbf, target_aos);
      transformed.overlap_derivative.insert(transformed.overlap_derivative.end(),
                                            overlap_derivative.begin(), overlap_derivative.end());
      transformed.hcore_derivative.insert(transformed.hcore_derivative.end(),
                                          hcore_derivative.begin(), hcore_derivative.end());
    }
    if (!cartesian.eri_derivative.empty()) {
      const std::vector<double> eri_derivative = transform_eri(
          cartesian.eri_derivative.data() + coordinate * eri_size, cartesian_nbf, target_aos);
      transformed.eri_derivative.insert(transformed.eri_derivative.end(), eri_derivative.begin(),
                                        eri_derivative.end());
    }
  }
  transformed.nuclear_repulsion = cartesian.nuclear_repulsion;
  transformed.nuclear_repulsion_derivative = cartesian.nuclear_repulsion_derivative;
  return transformed;
}

IntegralData build_integrals(const core::System& system, bool include_derivatives,
                             bool include_eri) {
  IntegralData out;
  out.nbf = molecule::cartesian_ao_count(system);
  out.ncoord = include_derivatives ? system.atoms.size() * 3 : 0;
  const std::size_t n = out.nbf;
  const std::size_t n2 = checked_product(n, n);
  const std::size_t n4 = include_eri ? checked_product(n2, n2) : 0;
  if (include_eri) {
    checked_product(n4, sizeof(Jet));
    checked_product(checked_product(n4, out.ncoord), sizeof(double));
  }
  const std::vector<AoView> aos = expand_cartesian_aos(system);

  std::vector<Vec3> atom_coordinates;
  atom_coordinates.reserve(system.atoms.size());
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom) {
    Vec3 position;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double coordinate = system.atoms[atom].position[axis];
      position[axis] = include_derivatives ? Jet::variable(coordinate, out.ncoord, atom * 3 + axis)
                                           : Jet(coordinate, 0);
    }
    atom_coordinates.push_back(std::move(position));
  }

  std::vector<Jet> overlap(n * n, Jet(0.0, out.ncoord));
  std::vector<Jet> hcore(n * n, Jet(0.0, out.ncoord));
  std::vector<Jet> eri;
  if (include_eri) eri.assign(n4, Jet(0.0, out.ncoord));

  for (std::size_t i = 0; i < n; ++i) {
    const AoView& ao_i = aos[i];
    const Vec3& a = atom_coordinates[ao_i.shell->atom_index];
    for (std::size_t j = 0; j < n; ++j) {
      const AoView& ao_j = aos[j];
      const Vec3& b = atom_coordinates[ao_j.shell->atom_index];
      Jet sij(0.0, out.ncoord);
      Jet hij(0.0, out.ncoord);
      const double component_factor = ao_i.component_normalization * ao_j.component_normalization;
      for (const core::Primitive& pi : ao_i.shell->primitives) {
        for (const core::Primitive& pj : ao_j.shell->primitives) {
          const double weight = component_factor * pi.coefficient * pj.coefficient;
          const ProductionST st = production_overlap_kinetic_cartesian(
              pi.exponent, a, ao_i.angular, ao_i.shell->atom_index, pj.exponent, b, ao_j.angular,
              ao_j.shell->atom_index);
          sij = sij + weight * st.overlap;
          hij =
              hij + weight * (st.kinetic + production_nuclear_attraction_cartesian(
                                               pi.exponent, a, ao_i.angular, ao_i.shell->atom_index,
                                               pj.exponent, b, ao_j.angular, ao_j.shell->atom_index,
                                               atom_coordinates, system));
        }
      }
      overlap[matrix_index(i, j, n)] = std::move(sij);
      hcore[matrix_index(i, j, n)] = std::move(hij);
    }
  }

  if (include_eri) {
    for (std::size_t i = 0; i < n; ++i) {
      const AoView& ao_i = aos[i];
      const Vec3& a = atom_coordinates[ao_i.shell->atom_index];
      for (std::size_t j = 0; j <= i; ++j) {
        const AoView& ao_j = aos[j];
        const Vec3& b = atom_coordinates[ao_j.shell->atom_index];
        for (std::size_t k = 0; k < n; ++k) {
          const AoView& ao_k = aos[k];
          const Vec3& c = atom_coordinates[ao_k.shell->atom_index];
          for (std::size_t l = 0; l <= k; ++l) {
            if (i * (i + 1) / 2 + j < k * (k + 1) / 2 + l) continue;
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
                    const double weight = component_factor * pi.coefficient * pj.coefficient *
                                          pk.coefficient * pl.coefficient;
                    value = value + weight * primitive_eri_cartesian(pi.exponent, a, ao_i.angular,
                                                                     pj.exponent, b, ao_j.angular,
                                                                     pk.exponent, c, ao_k.angular,
                                                                     pl.exponent, d, ao_l.angular);
                  }
                }
              }
            }
            // Eightfold ERI symmetry also holds for derivatives with respect
            // to physical atoms. Compute each expensive high-l recurrence once.
            for (const auto& indices : std::array<std::array<std::size_t, 4>, 8>{{{i, j, k, l},
                                                                                  {j, i, k, l},
                                                                                  {i, j, l, k},
                                                                                  {j, i, l, k},
                                                                                  {k, l, i, j},
                                                                                  {l, k, i, j},
                                                                                  {k, l, j, i},
                                                                                  {l, k, j, i}}})
              eri[eri_index(indices[0], indices[1], indices[2], indices[3], n)] = value;
          }
        }
      }
    }
  }

  Jet nuclear_repulsion(0.0, out.ncoord);
  for (std::size_t a = 0; a < system.atoms.size(); ++a) {
    for (std::size_t b = 0; b < a; ++b) {
      nuclear_repulsion =
          nuclear_repulsion +
          static_cast<double>(system.atoms[a].ionic_charge() * system.atoms[b].ionic_charge()) /
              sqrt(distance_squared(atom_coordinates[a], atom_coordinates[b]));
    }
  }

  unpack_jets(overlap, out.overlap, out.overlap_derivative, out.ncoord);
  unpack_jets(hcore, out.hcore, out.hcore_derivative, out.ncoord);
  if (include_eri) unpack_jets(eri, out.eri, out.eri_derivative, out.ncoord);
  out.nuclear_repulsion = nuclear_repulsion.value;
  out.nuclear_repulsion_derivative = std::move(nuclear_repulsion.derivative);
  if (!system.ecp_terms.empty()) {
    auto cartesian = system;
    cartesian.basis_representation = VIBEQC_BASIS_CARTESIAN;
    add_ecp(checked_ecp_integrals(cartesian, include_derivatives), out.hcore, out.hcore_derivative);
  }
  if (system.basis_representation == VIBEQC_BASIS_SPHERICAL) {
    const std::vector<GlobalAoExpansion> target_aos = spherical_expansions(system);
    IntegralData spherical;
    spherical.nbf = target_aos.size();
    spherical.ncoord = out.ncoord;
    spherical.overlap = transform_matrix(out.overlap.data(), out.nbf, target_aos);
    spherical.hcore = transform_matrix(out.hcore.data(), out.nbf, target_aos);
    if (include_eri) spherical.eri = transform_eri(out.eri.data(), out.nbf, target_aos);
    const std::size_t cartesian_matrix_size = out.nbf * out.nbf;
    const std::size_t cartesian_eri_size =
        include_eri ? cartesian_matrix_size * cartesian_matrix_size : 0;
    const std::size_t spherical_matrix_size = spherical.nbf * spherical.nbf;
    const std::size_t spherical_eri_size =
        include_eri ? spherical_matrix_size * spherical_matrix_size : 0;
    spherical.overlap_derivative.reserve(spherical.ncoord * spherical_matrix_size);
    spherical.hcore_derivative.reserve(spherical.ncoord * spherical_matrix_size);
    if (include_eri) spherical.eri_derivative.reserve(spherical.ncoord * spherical_eri_size);
    for (std::size_t coordinate = 0; coordinate < spherical.ncoord; ++coordinate) {
      std::vector<double> overlap_derivative = transform_matrix(
          out.overlap_derivative.data() + coordinate * cartesian_matrix_size, out.nbf, target_aos);
      std::vector<double> hcore_derivative = transform_matrix(
          out.hcore_derivative.data() + coordinate * cartesian_matrix_size, out.nbf, target_aos);
      std::vector<double> eri_derivative;
      if (include_eri)
        eri_derivative = transform_eri(out.eri_derivative.data() + coordinate * cartesian_eri_size,
                                       out.nbf, target_aos);
      spherical.overlap_derivative.insert(spherical.overlap_derivative.end(),
                                          overlap_derivative.begin(), overlap_derivative.end());
      spherical.hcore_derivative.insert(spherical.hcore_derivative.end(), hcore_derivative.begin(),
                                        hcore_derivative.end());
      if (include_eri)
        spherical.eri_derivative.insert(spherical.eri_derivative.end(), eri_derivative.begin(),
                                        eri_derivative.end());
    }
    spherical.nuclear_repulsion = out.nuclear_repulsion;
    spherical.nuclear_repulsion_derivative = std::move(out.nuclear_repulsion_derivative);
    return spherical;
  }
  return out;
}

std::vector<double> build_range_eri(const core::System& system, CoulombRange range, double omega) {
  if (range == CoulombRange::Full || !std::isfinite(omega) || omega < 0.0)
    throw std::invalid_argument("range ERI requires a finite nonnegative short/long omega");

  const std::size_t cartesian_nbf = molecule::cartesian_ao_count(system);
  const std::size_t n2 = checked_product(cartesian_nbf, cartesian_nbf);
  std::vector<double> eri(checked_product(n2, n2), 0.0);
  const std::vector<AoView> aos = expand_cartesian_aos(system);
  std::vector<Vec3> atom_coordinates;
  atom_coordinates.reserve(system.atoms.size());
  for (const auto& atom : system.atoms)
    atom_coordinates.push_back(
        {Jet(atom.position[0], 0), Jet(atom.position[1], 0), Jet(atom.position[2], 0)});

  for (std::size_t i = 0; i < cartesian_nbf; ++i) {
    const AoView& ao_i = aos[i];
    const Vec3& a = atom_coordinates[ao_i.shell->atom_index];
    for (std::size_t j = 0; j <= i; ++j) {
      const AoView& ao_j = aos[j];
      const Vec3& b = atom_coordinates[ao_j.shell->atom_index];
      for (std::size_t k = 0; k < cartesian_nbf; ++k) {
        const AoView& ao_k = aos[k];
        const Vec3& c = atom_coordinates[ao_k.shell->atom_index];
        for (std::size_t l = 0; l <= k; ++l) {
          if (i * (i + 1) / 2 + j < k * (k + 1) / 2 + l) continue;
          const AoView& ao_l = aos[l];
          const Vec3& d = atom_coordinates[ao_l.shell->atom_index];
          double value = 0.0;
          const double component_factor =
              ao_i.component_normalization * ao_j.component_normalization *
              ao_k.component_normalization * ao_l.component_normalization;
          for (const core::Primitive& pi : ao_i.shell->primitives)
            for (const core::Primitive& pj : ao_j.shell->primitives)
              for (const core::Primitive& pk : ao_k.shell->primitives)
                for (const core::Primitive& pl : ao_l.shell->primitives) {
                  const double weight = component_factor * pi.coefficient * pj.coefficient *
                                        pk.coefficient * pl.coefficient;
                  value += weight * primitive_range_eri_cartesian(
                                        pi.exponent, a, ao_i.angular, pj.exponent, b, ao_j.angular,
                                        pk.exponent, c, ao_k.angular, pl.exponent, d, ao_l.angular,
                                        range, omega);
                }
          if (!std::isfinite(value))
            throw std::runtime_error("nonfinite range-separated ERI value");
          for (const auto& indices : std::array<std::array<std::size_t, 4>, 8>{{{i, j, k, l},
                                                                                {j, i, k, l},
                                                                                {i, j, l, k},
                                                                                {j, i, l, k},
                                                                                {k, l, i, j},
                                                                                {l, k, i, j},
                                                                                {k, l, j, i},
                                                                                {l, k, j, i}}})
            eri[eri_index(indices[0], indices[1], indices[2], indices[3], cartesian_nbf)] = value;
        }
      }
    }
  }
  if (system.basis_representation != VIBEQC_BASIS_SPHERICAL) return eri;
  return transform_eri(eri.data(), cartesian_nbf, spherical_expansions(system));
}

std::array<double, 12> contract_weighted_eri_shell_derivative(
    const core::System& system, const std::array<std::size_t, 4>& shell_indices,
    std::span<const double> weights) {
  std::array<const core::Shell*, 4> shells{};
  std::array<std::vector<molecule::AoExpansion>, 4> expansions{};
  std::size_t expected = 1;
  for (std::size_t slot = 0; slot < shells.size(); ++slot) {
    if (shell_indices[slot] >= system.shells.size())
      throw std::invalid_argument("weighted ERI shell index is out of range");
    shells[slot] = &system.shells[shell_indices[slot]];
    expansions[slot] =
        molecule::ao_expansions(shells[slot]->angular_momentum, system.basis_representation);
    expected = checked_product(expected, expansions[slot].size());
  }
  if (weights.size() != expected || !std::all_of(weights.begin(), weights.end(),
                                                 [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("weighted ERI shell weights are inconsistent or nonfinite");

  constexpr std::size_t coordinates = 12;
  std::array<Vec3, 4> centers{};
  for (std::size_t slot = 0; slot < centers.size(); ++slot) {
    const auto atom = shells[slot]->atom_index;
    if (atom >= system.atoms.size())
      throw std::invalid_argument("weighted ERI shell atom is out of range");
    for (std::size_t axis = 0; axis < 3; ++axis)
      centers[slot][axis] =
          Jet::variable(system.atoms[atom].position[axis], coordinates, 3 * slot + axis);
  }

  Jet contracted(0.0, coordinates);
  std::size_t cursor = 0;
  for (const auto& ao_i : expansions[0])
    for (const auto& ao_j : expansions[1])
      for (const auto& ao_k : expansions[2])
        for (const auto& ao_l : expansions[3]) {
          const double public_weight = weights[cursor++];
          if (public_weight == 0.0) continue;
          for (const auto& ei : ao_i)
            for (const auto& ej : ao_j)
              for (const auto& ek : ao_k)
                for (const auto& el : ao_l) {
                  const std::array<const molecule::CartesianExpansionTerm*, 4> terms{&ei, &ej, &ek,
                                                                                     &el};
                  double component_weight = public_weight;
                  for (const auto* term : terms)
                    component_weight *=
                        term->coefficient *
                        molecule::cartesian_component_normalization(term->component);
                  for (const auto& pi : shells[0]->primitives)
                    for (const auto& pj : shells[1]->primitives)
                      for (const auto& pk : shells[2]->primitives)
                        for (const auto& pl : shells[3]->primitives) {
                          const double primitive_weight = component_weight * pi.coefficient *
                                                          pj.coefficient * pk.coefficient *
                                                          pl.coefficient;
                          contracted = contracted + primitive_weight *
                                                        primitive_eri_cartesian(
                                                            pi.exponent, centers[0], ei.component,
                                                            pj.exponent, centers[1], ej.component,
                                                            pk.exponent, centers[2], ek.component,
                                                            pl.exponent, centers[3], el.component);
                        }
                }
        }
  if (!std::isfinite(contracted.value) ||
      !std::all_of(contracted.derivative.begin(), contracted.derivative.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("weighted ERI shell derivative is nonfinite");
  std::array<double, coordinates> result{};
  std::copy(contracted.derivative.begin(), contracted.derivative.end(), result.begin());
  return result;
}

std::vector<double> contract_weighted_one_electron_derivative(
    const core::System& system, std::span<const double> overlap_weights,
    std::span<const double> hcore_weights, bool include_nuclear_repulsion) {
  if (!system.ecp_terms.empty())
    throw std::invalid_argument("streamed one-electron ECP derivatives are not implemented");
  const auto n = molecule::ao_count(system);
  const auto matrix_size = checked_product(n, n);
  if (overlap_weights.size() != matrix_size || hcore_weights.size() != matrix_size ||
      !std::all_of(overlap_weights.begin(), overlap_weights.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(hcore_weights.begin(), hcore_weights.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("streamed one-electron weights are inconsistent or nonfinite");

  const auto ncoord = checked_product(system.atoms.size(), std::size_t{3});
  std::vector<Vec3> atoms(system.atoms.size());
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    for (std::size_t axis = 0; axis < 3; ++axis)
      atoms[atom][axis] = Jet::variable(system.atoms[atom].position[axis], ncoord, 3 * atom + axis);

  std::vector<std::size_t> offsets(system.shells.size() + 1, 0);
  std::vector<std::vector<molecule::AoExpansion>> expansions(system.shells.size());
  for (std::size_t shell = 0; shell < system.shells.size(); ++shell) {
    if (system.shells[shell].atom_index >= system.atoms.size())
      throw std::invalid_argument("streamed one-electron shell atom is out of range");
    expansions[shell] =
        molecule::ao_expansions(system.shells[shell].angular_momentum, system.basis_representation);
    offsets[shell + 1] = checked_sum(offsets[shell], expansions[shell].size());
  }
  if (offsets.back() != n)
    throw std::runtime_error("streamed one-electron shell offsets disagree with the system");

  Jet contracted(0.0, ncoord);
  for (std::size_t si = 0; si < system.shells.size(); ++si) {
    const auto& shell_i = system.shells[si];
    const auto& center_i = atoms[shell_i.atom_index];
    for (std::size_t sj = 0; sj < system.shells.size(); ++sj) {
      const auto& shell_j = system.shells[sj];
      const auto& center_j = atoms[shell_j.atom_index];
      for (std::size_t i = 0; i < expansions[si].size(); ++i)
        for (std::size_t j = 0; j < expansions[sj].size(); ++j) {
          const auto index = (offsets[si] + i) * n + offsets[sj] + j;
          const double overlap_weight = overlap_weights[index];
          const double hcore_weight = hcore_weights[index];
          if (overlap_weight == 0.0 && hcore_weight == 0.0) continue;
          for (const auto& ei : expansions[si][i])
            for (const auto& ej : expansions[sj][j]) {
              const double component_weight =
                  ei.coefficient * molecule::cartesian_component_normalization(ei.component) *
                  ej.coefficient * molecule::cartesian_component_normalization(ej.component);
              for (const auto& pi : shell_i.primitives)
                for (const auto& pj : shell_j.primitives) {
                  const double primitive_weight =
                      component_weight * pi.coefficient * pj.coefficient;
                  const ProductionST st = production_overlap_kinetic_cartesian(
                      pi.exponent, center_i, ei.component, shell_i.atom_index, pj.exponent,
                      center_j, ej.component, shell_j.atom_index);
                  if (overlap_weight != 0.0)
                    contracted = contracted + overlap_weight * primitive_weight * st.overlap;
                  if (hcore_weight != 0.0)
                    contracted =
                        contracted +
                        hcore_weight * primitive_weight *
                            (st.kinetic + production_nuclear_attraction_cartesian(
                                              pi.exponent, center_i, ei.component,
                                              shell_i.atom_index, pj.exponent, center_j,
                                              ej.component, shell_j.atom_index, atoms, system));
                }
            }
        }
    }
  }
  if (include_nuclear_repulsion)
    for (std::size_t a = 0; a < system.atoms.size(); ++a)
      for (std::size_t b = 0; b < a; ++b)
        contracted = contracted + static_cast<double>(system.atoms[a].ionic_charge() *
                                                      system.atoms[b].ionic_charge()) /
                                      sqrt(distance_squared(atoms[a], atoms[b]));
  if (!std::isfinite(contracted.value) ||
      !std::all_of(contracted.derivative.begin(), contracted.derivative.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("streamed one-electron derivative is nonfinite");
  return contracted.derivative;
}

}  // namespace vibeqc::integrals

// Keep this adapter in the evaluator translation unit so it reuses the exact
// contracted primitive mathematics without exporting recurrence internals.
namespace vibeqc::posthf {
using namespace vibeqc::integrals;

struct RawSource::Impl {
  core::System orbital, auxiliary;
  std::vector<double> ecp_matrix;
  bool has_auxiliary = false;
  std::vector<AoView> aos, aux;
  std::vector<GlobalAoExpansion> public_aos, public_aux;
  std::vector<Vec3> centers;

  double cartesian(Operator op, const std::array<const AoView*, 4>& slots) const {
    const bool one = op == Operator::overlap || op == Operator::hcore;
    const unsigned rank = op == Operator::eri ? 4 : op == Operator::three_center ? 3 : 2;
    std::array<core::Primitive, 4> primitive{};
    double result = 0;
    auto evaluate = [&](auto&& self, unsigned slot, double weight) -> void {
      if (slot < rank) {
        for (const auto& p : slots[slot]->shell->primitives) {
          primitive[slot] = p;
          self(self, slot + 1, weight * p.coefficient * slots[slot]->component_normalization);
        }
        return;
      }
      const auto& a = centers[slots[0]->shell->atom_index];
      const auto& b = centers[slots[1]->shell->atom_index];
      if (one) {
        if (op == Operator::overlap) {
          result +=
              weight * reference_overlap_cartesian(primitive[0].exponent, a, slots[0]->angular,
                                                   primitive[1].exponent, b, slots[1]->angular)
                           .value;
        } else {
          result +=
              weight * (reference_kinetic_cartesian(primitive[0].exponent, a, slots[0]->angular,
                                                    primitive[1].exponent, b, slots[1]->angular) +
                        primitive_nuclear_attraction_cartesian(
                            primitive[0].exponent, a, slots[0]->angular, primitive[1].exponent, b,
                            slots[1]->angular, centers, orbital))
                           .value;
        }
        return;
      }
      const molecule::CartesianComponent zero{0, 0, 0};
      if (op == Operator::metric) {
        result += weight * primitive_eri_cartesian(primitive[0].exponent, a, slots[0]->angular, 0,
                                                   a, zero, primitive[1].exponent, b,
                                                   slots[1]->angular, 0, b, zero)
                               .value;
      } else {
        const auto& c = centers[slots[2]->shell->atom_index];
        const auto& d = rank == 4 ? centers[slots[3]->shell->atom_index] : c;
        result += weight * primitive_eri_cartesian(primitive[0].exponent, a, slots[0]->angular,
                                                   primitive[1].exponent, b, slots[1]->angular,
                                                   primitive[2].exponent, c, slots[2]->angular,
                                                   rank == 4 ? primitive[3].exponent : 0, d,
                                                   rank == 4 ? slots[3]->angular : zero)
                               .value;
      }
    };
    evaluate(evaluate, 0, 1);
    return result;
  }

  double value(Operator op, const std::array<std::size_t, 4>& indices) const {
    const unsigned rank = op == Operator::eri ? 4 : op == Operator::three_center ? 3 : 2;
    std::array<const AoView*, 4> slots{};
    double result = 0;
    auto expand = [&](auto&& self, unsigned slot, double weight) -> void {
      if (slot == rank) {
        result += weight * cartesian(op, slots);
        return;
      }
      const bool auxiliary_slot =
          op == Operator::metric || (op == Operator::three_center && slot == 2);
      const auto& expansion = (auxiliary_slot ? public_aux : public_aos)[indices[slot]];
      const auto& source = auxiliary_slot ? aux : aos;
      for (const auto& term : expansion) {
        slots[slot] = &source[term.cartesian_ao];
        self(self, slot + 1, weight * term.coefficient);
      }
    };
    expand(expand, 0, 1);
    if (op == Operator::hcore && !ecp_matrix.empty())
      result += ecp_matrix[indices[0] * public_aos.size() + indices[1]];
    return result;
  }
};

RawSource::RawSource(core::System orbital, const core::System* auxiliary)
    : impl_(std::make_unique<Impl>()) {
  impl_->orbital = std::move(orbital);
  for (const auto& shell : impl_->orbital.shells)
    if (shell.angular_momentum > 4)
      throw std::invalid_argument("raw post-HF source supports through g");
  impl_->aos = expand_cartesian_aos(impl_->orbital);
  impl_->public_aos = public_ao_expansions(impl_->orbital);
  if (!impl_->orbital.ecp_terms.empty()) {
    const auto ecp = checked_ecp_integrals(impl_->orbital, false);
    impl_->ecp_matrix = ecp.local;
    for (std::size_t i = 0; i < impl_->ecp_matrix.size(); ++i)
      impl_->ecp_matrix[i] += ecp.nonlocal[i];
  }
  for (const auto& atom : impl_->orbital.atoms) {
    Vec3 position;
    for (unsigned axis = 0; axis < 3; ++axis) position[axis] = Jet(atom.position[axis], 0);
    impl_->centers.push_back(std::move(position));
  }
  if (auxiliary) {
    require_matching_density_fitting_geometry(impl_->orbital, *auxiliary);
    for (const auto& shell : auxiliary->shells)
      if (shell.angular_momentum > 4)
        throw std::invalid_argument("raw auxiliary source supports through g");
    impl_->auxiliary = *auxiliary;
    impl_->has_auxiliary = true;
    impl_->aux = expand_cartesian_aos(impl_->auxiliary);
    impl_->public_aux = public_ao_expansions(impl_->auxiliary);
  }
}
RawSource::~RawSource() = default;
const core::System& RawSource::orbital() const { return impl_->orbital; }
const core::System& RawSource::auxiliary() const {
  if (!impl_->has_auxiliary) throw std::invalid_argument("auxiliary basis required");
  return impl_->auxiliary;
}
std::size_t RawSource::retained_numeric_bytes() const {
  auto bytes = source_capacity(impl_->orbital);
  if (impl_->has_auxiliary) bytes = checked_add(bytes, source_capacity(impl_->auxiliary));
  return checked_add(bytes, checked_mul(sizeof(double), impl_->ecp_matrix.capacity()));
}
std::size_t RawSource::nbf() const { return impl_->public_aos.size(); }
std::size_t RawSource::naux() const { return impl_->public_aux.size(); }
void RawSource::read(Operator op, const std::array<std::size_t, 4>& begin,
                     const std::array<std::size_t, 4>& count, double* out,
                     std::size_t elements) const {
  if (op < Operator::overlap || op > Operator::three_center)
    throw std::invalid_argument("unknown raw operator");
  const unsigned rank = op == Operator::eri ? 4 : op == Operator::three_center ? 3 : 2;
  if ((op == Operator::metric || op == Operator::three_center) && !impl_->has_auxiliary)
    throw std::invalid_argument("auxiliary basis required");
  std::size_t size = 1;
  for (unsigned slot = 0; slot < 4; ++slot) {
    const std::size_t dimension =
        slot >= rank
            ? 1
            : (op == Operator::metric || (op == Operator::three_center && slot == 2) ? naux()
                                                                                     : nbf());
    if (begin[slot] > dimension || count[slot] > dimension - begin[slot])
      throw std::invalid_argument("raw tile exceeds source bounds");
    if (count[slot] && size > SIZE_MAX / count[slot])
      throw std::overflow_error("raw tile size overflow");
    size *= count[slot];
  }
  if (size != elements || (size && !out)) throw std::invalid_argument("raw output size mismatch");
  std::size_t cursor = 0;
  std::array<std::size_t, 4> indices{};
  auto traverse = [&](auto&& self, unsigned slot) -> void {
    if (slot == 4) {
      out[cursor++] = impl_->value(op, indices);
      return;
    }
    for (std::size_t i = 0; i < count[slot]; ++i) {
      indices[slot] = begin[slot] + i;
      self(self, slot + 1);
    }
  };
  traverse(traverse, 0);
}
}  // namespace vibeqc::posthf

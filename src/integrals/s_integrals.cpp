#include "integrals/s_integrals.hpp"

#include "molecule/basis.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace qce::integrals {
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

std::vector<AoView> expand_aos(const core::System& system) {
  std::vector<AoView> aos;
  aos.reserve(molecule::ao_count(system));
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

std::size_t matrix_index(std::size_t i, std::size_t j, std::size_t n) {
  return i * n + j;
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

}  // namespace

IntegralData build_cartesian_integrals(const core::System& system) {
  IntegralData out;
  out.nbf = molecule::ao_count(system);
  out.ncoord = system.atoms.size() * 3;
  const std::size_t n = out.nbf;
  const std::vector<AoView> aos = expand_aos(system);

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
          const double weight = component_factor * pi.coefficient * pj.coefficient;
          sij = sij + weight * primitive_overlap_cartesian(
                                   pi.exponent, a, ao_i.angular,
                                   pj.exponent, b, ao_j.angular);
          hij = hij + weight *
              (primitive_kinetic_cartesian(pi.exponent, a, ao_i.angular,
                                           pj.exponent, b, ao_j.angular) +
               primitive_nuclear_attraction_cartesian(
                   pi.exponent, a, ao_i.angular, pj.exponent, b, ao_j.angular,
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
  return out;
}

}  // namespace qce::integrals

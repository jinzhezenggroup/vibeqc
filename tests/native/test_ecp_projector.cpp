#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "generated_ecp_ao.cuh"

namespace {
constexpr double pi = std::numbers::pi;
struct Jet {
  double v[4];
};
struct Point {
  double x, y, z, weight;
  double harmonics[9];
};
struct Radial {
  double r, weight;
};
struct Term {
  unsigned atom_index;
  int channel;
  unsigned power;
  double exponent, coefficient;
};
void near(double actual, double expected) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 3e-12 * (1 + std::abs(expected)))
    throw std::runtime_error("generated ECP projector disagrees with addition-theorem oracle");
}

struct Primitive {
  double exponent, coefficient;
};
struct Component {
  unsigned x, y, z;
  double coefficient;
};
struct AO {
  int primitive_offset, primitive_count, term_count;
  double x, y, z;
  Component components[3];
};

// Independent long-double polynomial oracle; no generated AO or derivatives.
long double ao_value(const AO& ao, const Primitive* primitives, const std::array<double, 3>& node) {
  const long double x = node[0] - ao.x, y = node[1] - ao.y, z = node[2] - ao.z;
  long double polynomial = 0, radial = 0;
  for (int t = 0; t < ao.term_count; ++t) {
    const auto c = ao.components[t];
    polynomial += c.coefficient * std::pow(x, c.x) * std::pow(y, c.y) * std::pow(z, c.z);
  }
  for (int k = 0; k < ao.primitive_count; ++k) {
    const auto p = primitives[ao.primitive_offset + k];
    radial += p.coefficient * std::exp(-p.exponent * (x * x + y * y + z * z));
  }
  return polynomial * radial;
}

void check_ao_consumer() {
  const Primitive primitives[] = {{9, 1234}, {0.13, 0.73}, {1.1, -0.61}, {3.7, 0.29}};
  const Point point{0.36, -0.48, 0.8, 0, {}};
  const std::array<double, 3> center{0.31, -0.22, 0.17};
  for (unsigned lx = 0; lx <= 2; ++lx)
    for (unsigned ly = 0; lx + ly <= 2; ++ly)
      for (unsigned lz = 0; lx + ly + lz <= 2; ++lz)
        for (int count : {1, 3})
          for (double radius : {0.0, 1e-7, 0.73, 4.1}) {
            // Three terms exercise spherical-d-like signed mixtures; offset 1
            // ensures the sentinel primitive must never enter the contraction.
            AO ao{1,
                  count,
                  count,
                  -0.4,
                  0.19,
                  -0.11,
                  {{lx, ly, lz, 0.71}, {0, 2, 0, -0.37}, {0, 0, 2, 0.53}}};
            const std::array<double, 3> node{center[0] + radius * point.x,
                                             center[1] + radius * point.y,
                                             center[2] + radius * point.z};
            for (bool coincident : {false, true}) {
              if (coincident) {
                ao.x = node[0];
                ao.y = node[1];
                ao.z = node[2];
              }
              for (bool derivatives : {false, true}) {
                const auto result = vibeqc::generated::ecp_evaluate_ao<Jet>(
                    ao, primitives, point, radius, center[0], center[1], center[2], derivatives);
                near(result.v[0], static_cast<double>(ao_value(ao, primitives, node)));
                for (int axis = 0; axis < 3; ++axis) {
                  if (!derivatives) {
                    near(result.v[axis + 1], 0);
                    continue;
                  }
                  for (double step : {2e-5, 1e-5}) {
                    auto plus = node, minus = node;
                    plus[axis] -= step;
                    minus[axis] += step;
                    const double expected = static_cast<double>(
                        (ao_value(ao, primitives, plus) - ao_value(ao, primitives, minus)) /
                        (2 * step));
                    if (!std::isfinite(result.v[axis + 1]) ||
                        std::abs(result.v[axis + 1] - expected) > 3e-9)
                      throw std::runtime_error("contracted AO center finite difference failed");
                  }
                }
              }
            }
          }
}

void check_weighted_consumer() {
  // Deliberately nonsymmetric fixed weights: no triangular shortcut or factor 2.
  constexpr int n = 3, size = n * n;
  std::array<double, size> weights{}, local{}, nonlocal{}, dl{}, dn{};
  for (int i = 0; i < size; ++i) {
    weights[i] = std::sin(0.3 + 1.7 * i);
    local[i] = std::cos(0.7 * i);
    nonlocal[i] = std::sin(1.3 * i);
    dl[i] = std::sin(0.6 + i);
    dn[i] = std::cos(0.1 + 0.9 * i);
    near(vibeqc::generated::ecp_add_operator(1.7, local[i], nonlocal[i]),
         static_cast<double>(1.7L + local[i] + nonlocal[i]));
  }
  for (int component = 0; component < 3; ++component) {
    auto a = dl, b = dn;
    if (component == 0) a.fill(0);
    if (component == 1) b.fill(0);
    const double force =
        vibeqc::generated::ecp_force_component(a.data(), b.data(), weights.data(), size);
    for (long double step : {2e-5L, 1e-5L}) {
      long double plus = 0, minus = 0;
      for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
          const int i = row * n + col;
          const long double value = static_cast<long double>(local[i]) + nonlocal[i];
          const long double derivative = static_cast<long double>(a[i]) + b[i];
          plus += weights[i] * (value + step * derivative);
          minus += weights[i] * (value - step * derivative);
        }
      near(force, static_cast<double>((minus - plus) / (2 * step)));
    }
  }
  near(vibeqc::generated::ecp_force_component(nullptr, nullptr, nullptr, 0), 0);
}

// This oracle never forms AO projections. It contracts pairs of angular
// nodes using the spherical-harmonic addition theorem and Legendre P_l.
double kernel(int l, const Point& a, const Point& b) {
  const double x = a.x * b.x + a.y * b.y + a.z * b.z;
  return (2 * l + 1) / (4 * pi) * (l == 0 ? 1 : l == 1 ? x : (3 * x * x - 1) / 2);
}

void check() {
  constexpr int nq = 7;
  std::array<Point, nq> sphere{};
  std::array<Jet, nq> a{}, b{};
  for (int q = 0; q < nq; ++q) {
    const double z = -0.9 + 0.27 * q, phi = 0.31 + q * 1.47;
    const double x = std::sqrt(1 - z * z) * std::cos(phi);
    const double y = std::sqrt(1 - z * z) * std::sin(phi);
    sphere[q] = {x,
                 y,
                 z,
                 0.2 + 0.03 * q,
                 {1 / std::sqrt(4 * pi), std::sqrt(3 / (4 * pi)) * x, std::sqrt(3 / (4 * pi)) * y,
                  std::sqrt(3 / (4 * pi)) * z, std::sqrt(15 / (4 * pi)) * x * y,
                  std::sqrt(15 / (4 * pi)) * y * z, std::sqrt(5 / (16 * pi)) * (3 * z * z - 1),
                  std::sqrt(15 / (4 * pi)) * x * z, std::sqrt(15 / (16 * pi)) * (x * x - y * y)}};
    for (int d = 0; d < 4; ++d) {
      a[q].v[d] = std::sin(0.7 + 1.3 * q + d * 0.4);
      b[q].v[d] = std::cos(-0.3 + 0.7 * q - d * 0.9);
    }
  }
  for (const bool derivatives : {false, true}) {
    auto av = a, bv = b;
    if (!derivatives)
      for (int q = 0; q < nq; ++q)
        for (int d = 1; d < 4; ++d)
          av[q].v[d] = bv[q].v[d] = std::numeric_limits<double>::quiet_NaN();
    std::array<Jet, 9> pa{}, pb{};
    for (int m = 0; m < 9; ++m) {
      pa[m] = vibeqc::generated::ecp_project(av.data(), sphere.data(), nq, m, derivatives);
      pb[m] = vibeqc::generated::ecp_project(bv.data(), sphere.data(), nq, m, derivatives);
    }
    for (int channel = -1; channel <= 2; ++channel)
      for (unsigned power = 0; power <= 4; ++power)
        for (double r : {0.0, 1e-7, 0.37, 1.9, 12.0}) {
          const Radial radial{r, 0.73};
          const Term terms[] = {{1, channel, power, 0.67, -1.2},
                                {1, channel, power, 1.23, 0.81},
                                {0, channel, power, 0.11, 123.0}};
          double parts[2][10];
          vibeqc::generated::ecp_contract(terms, 3, radial, sphere.data(), nq, 1, av.data(),
                                          bv.data(), pa.data(), pb.data(), derivatives, parts);
          const double potential =
              radial.weight * std::pow(r, power) *
              (-1.2 * std::exp(-0.67 * r * r) + 0.81 * std::exp(-1.23 * r * r));
          const int part = channel == -1 ? 0 : 1;
          double expected[10]{};
          for (int q = 0; q < nq; ++q)
            for (int k = 0; k < nq; ++k) {
              const double w =
                  potential * sphere[q].weight *
                  (channel == -1 ? (q == k ? 1.0 : 0.0)
                                 : sphere[k].weight * kernel(channel, sphere[q], sphere[k]));
              expected[0] += w * a[q].v[0] * b[k].v[0];
              if (derivatives)
                for (int d = 1; d <= 3; ++d) {
                  expected[d] += w * a[q].v[d] * b[k].v[0];
                  expected[d + 3] += w * a[q].v[0] * b[k].v[d];
                }
            }
          for (int d = 0; d < 3; ++d) expected[d + 7] = -expected[d + 1] - expected[d + 4];
          for (int d = 0; d < 10; ++d) {
            near(parts[part][d], expected[d]);
            near(parts[1 - part][d], 0);
          }
        }
  }
  if (!std::isnan(vibeqc::generated::ecp_radial(5, 1, 1, 1, 1)))
    throw std::runtime_error("unsupported radial power silently accepted");
}
}  // namespace

int main() {
  try {
    check();
    check_ao_consumer();
    check_weighted_consumer();
    std::cout << "Generated ECP: all powers/channels, A/B/C jets and value-only gates passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

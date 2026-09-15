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
    std::cout << "Generated ECP: all powers/channels, A/B/C jets and value-only gates passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

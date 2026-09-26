// Whole CPU provider qualification against the retained independent algorithm.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "ecp_reference.hpp"
#include "molecule/basis.hpp"

namespace {
using vibeqc::core::System;
using vibeqc::integrals::EcpData;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance = 2e-11) {
  require(std::isfinite(actual) && std::isfinite(expected) &&
              std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)),
          "generated CPU ECP differs from independent oracle");
}
void compare(const EcpData& a, const EcpData& b) {
  require(a.nbf == b.nbf && a.ncoord == b.ncoord, "provider output dimensions");
  const std::vector<double>* actual[] = {&a.local, &a.nonlocal, &a.local_derivative,
                                         &a.nonlocal_derivative};
  const std::vector<double>* expected[] = {&b.local, &b.nonlocal, &b.local_derivative,
                                           &b.nonlocal_derivative};
  for (unsigned part = 0; part < 4; ++part) {
    require(actual[part]->size() == expected[part]->size(), "provider vector shape");
    for (std::size_t i = 0; i < actual[part]->size(); ++i)
      near((*actual[part])[i], (*expected[part])[i]);
  }
}
System fixture(vibeqc_basis_representation representation, bool high_angular = true) {
  System system;
  // Center 2 has ECP terms but no basis; center 1 is all-electron. Center 0
  // exercises coincident basis/ECP derivative scatter and signed contractions.
  system.atoms = {
      {11, {0.13, -0.21, 0.17}, 10}, {1, {0.43, 0.19, 3.2}, 0}, {11, {-0.63, 0.24, -1.6}, 10}};
  system.shells = {{0, 0, {{0.73, 0.8}, {0.21, -0.12}}}, {1, 1, {{0.45, 0.9}, {1.15, -0.12}}}};
  if (high_angular) {
    system.shells.push_back({0, 2, {{0.35, 1.0}}});
    system.shells.push_back({1, 3, {{0.55, 0.81}, {0.17, 0.23}}});
  }
  system.basis_representation = representation;
  system.multiplicity = 2;
  for (unsigned center : {0U, 2U})
    for (int channel = -1; channel <= 3; ++channel)
      for (unsigned power = 0; power <= 4; ++power) {
        system.ecp_terms.push_back({center, channel, power, 0.67, -0.12});
        system.ecp_terms.push_back({center, channel, power, 1.23, 0.081});
      }
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          detail.c_str());
  return system;
}

double contract(const EcpData& data) {
  double result = 0;
  for (std::size_t i = 0; i < data.nbf * data.nbf; ++i)
    result += std::sin(0.7 + i * 1.3) * (data.local[i] + data.nonlocal[i]);
  return result;
}
void check_raw(vibeqc_basis_representation representation) {
  auto system = fixture(representation);
  const auto actual = vibeqc::integrals::ecp_integrals(system, 32, 12, true);
  compare(actual, vibeqc::testing::ecp_integrals(system, 32, 12, true));
  const auto values = vibeqc::integrals::ecp_integrals(system, 32, 12, false);
  compare(values, vibeqc::testing::ecp_integrals(system, 32, 12, false));
  for (std::size_t i = 0; i < actual.local.size(); ++i) {
    near(actual.local[i], values.local[i]);
    near(actual.nonlocal[i], values.nonlocal[i]);
    for (unsigned d = 0; d < 3; ++d) {
      double sum = 0;
      for (unsigned atom = 0; atom < 3; ++atom)
        sum += actual.local_derivative[(atom * 3 + d) * actual.local.size() + i] +
               actual.nonlocal_derivative[(atom * 3 + d) * actual.local.size() + i];
      near(sum, 0);
    }
  }
  // Independent energy finite differences, two steps, arbitrary nonsymmetric
  // AO weights, all physical centers (including the basis-free ECP center).
  for (unsigned coordinate = 0; coordinate < 9; ++coordinate) {
    double analytic = 0;
    for (std::size_t i = 0; i < actual.local.size(); ++i) {
      const auto offset = coordinate * actual.local.size() + i;
      analytic += std::sin(0.7 + i * 1.3) *
                  (actual.local_derivative[offset] + actual.nonlocal_derivative[offset]);
    }
    for (double h : {2e-4, 1e-4}) {
      auto plus = system, minus = system;
      plus.atoms[coordinate / 3].position[coordinate % 3] += h;
      minus.atoms[coordinate / 3].position[coordinate % 3] -= h;
      const auto fd = (contract(vibeqc::testing::ecp_integrals(plus, 32, 12, false)) -
                       contract(vibeqc::testing::ecp_integrals(minus, 32, 12, false))) /
                      (2 * h);
      near(analytic, fd, 2e-7);
    }
  }
  // Changed geometry and replay do not retain the previous center/grid/AO state.
  system.atoms[1].position[2] += 0.17;
  compare(vibeqc::integrals::ecp_integrals(system, 32, 12, true),
          vibeqc::testing::ecp_integrals(system, 32, 12, true));
  compare(actual, vibeqc::integrals::ecp_integrals(fixture(representation), 32, 12, true));
}

template <class F>
void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "invalid ECP input accepted");
}
void check_checked_and_failures() {
  auto system = fixture(VIBEQC_BASIS_SPHERICAL, false);
  for (bool derivatives : {false, true})
    compare(vibeqc::integrals::checked_ecp_integrals(system, derivatives),
            vibeqc::testing::checked_ecp_integrals(system, derivatives));
  for (auto grid : {std::array<unsigned, 2>{15, 8}, {513, 8}, {16, 7}, {16, 97}})
    rejects([&] { vibeqc::integrals::ecp_integrals(system, grid[0], grid[1]); });
  auto empty = system;
  empty.ecp_terms.clear();
  compare(vibeqc::integrals::ecp_integrals(empty), vibeqc::testing::ecp_integrals(empty));
  auto too_large = system;
  too_large.atoms.resize(129);
  rejects([&] { vibeqc::integrals::ecp_integrals(too_large); });
  too_large = system;
  too_large.shells.resize(257, system.shells[0]);
  rejects([&] { vibeqc::integrals::ecp_integrals(too_large); });
  auto unsupported = system;
  unsupported.shells[0].angular_momentum = 4;
  for (auto representation : {VIBEQC_BASIS_CARTESIAN, VIBEQC_BASIS_SPHERICAL}) {
    unsupported.basis_representation = representation;
    rejects([&] { vibeqc::integrals::ecp_integrals(unsupported); });
  }
  for (double poison :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    auto invalid = system;
    invalid.ecp_terms[0].coefficient = poison;
    rejects([&] { vibeqc::integrals::checked_ecp_integrals(invalid, false); });
    rejects([&] { vibeqc::testing::checked_ecp_integrals(invalid, false); });
  }
  compare(vibeqc::integrals::checked_ecp_integrals(system, true),
          vibeqc::testing::checked_ecp_integrals(system, true));
  auto data = vibeqc::integrals::ecp_integrals(system, 16, 8, true);
  std::vector<double> hcore(data.local.size(), 0.7), derivative(data.local_derivative.size(), -0.2);
  vibeqc::integrals::add_ecp(data, hcore, derivative);
  for (std::size_t i = 0; i < hcore.size(); ++i)
    near(hcore[i], 0.7 + data.local[i] + data.nonlocal[i]);
  for (std::size_t i = 0; i < derivative.size(); ++i)
    near(derivative[i], -0.2 + data.local_derivative[i] + data.nonlocal_derivative[i]);
  hcore.pop_back();
  rejects([&] { vibeqc::integrals::add_ecp(data, hcore, derivative); });
}

void check_larger() {
  auto system = fixture(VIBEQC_BASIS_CARTESIAN);
  // 80 AOs, beyond the public stationary s/p slice: same one-layer provider.
  const auto shells = system.shells;
  for (unsigned repeat = 1; repeat < 4; ++repeat)
    system.shells.insert(system.shells.end(), shells.begin(), shells.end());
  const auto start = std::chrono::steady_clock::now();
  const auto actual = vibeqc::integrals::ecp_integrals(system, 16, 8, true);
  const auto middle = std::chrono::steady_clock::now();
  const auto expected = vibeqc::testing::ecp_integrals(system, 16, 8, true);
  const auto end = std::chrono::steady_clock::now();
  compare(actual, expected);
  require(actual.nbf == 80, "larger fixture must exercise 80 AOs");
  std::cout << "80-AO full raw provider seconds: generated="
            << std::chrono::duration<double>(middle - start).count()
            << " oracle=" << std::chrono::duration<double>(end - middle).count() << '\n';
}
}  // namespace

int main() {
  try {
    check_raw(VIBEQC_BASIS_CARTESIAN);
    check_raw(VIBEQC_BASIS_SPHERICAL);
    check_checked_and_failures();
    check_larger();
    std::cout << "CPU ECP generated/oracle matrix, derivative, replay and rejection gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

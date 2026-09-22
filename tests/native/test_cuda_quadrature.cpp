// Independent scalar triangular Becke traversal and analytic coincident gates.
// Run only inside a finite scheduler allocation. No timing is an SCF endpoint.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "dft/grid.hpp"
#include "runtime/resource_ledger.hpp"

namespace {
using vibeqc::core::System;
using vibeqc::dft::GridSpec;
using vibeqc::dft::MolecularGrid;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void close(const std::vector<double>& actual, const std::vector<double>& expected, double atol,
           double rtol, const char* message) {
  require(actual.size() == expected.size(), "export size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (!std::isfinite(actual[i]) ||
        std::abs(actual[i] - expected[i]) > atol + rtol * std::abs(expected[i])) {
      std::cerr << message << " at " << i << ": " << actual[i] << " != " << expected[i] << '\n';
      throw std::runtime_error(message);
    }
}

System centers(std::size_t count) {
  System system;
  for (std::size_t a = 0; a < count; ++a)
    system.atoms.push_back(
        {a % 3 == 0 ? 8 : 1,
         {2.8 * double(a % 4), 3.1 * double((a / 4) % 4), 3.4 * double(a / 16)}});
  return system;
}

void compare(const System& system, GridSpec spec, bool derivative) {
  const MolecularGrid reference(system, spec);
  const auto actual = MolecularGrid::from_cuda(system, spec, 0);
  require(actual.spec() == reference.spec(), "grid identity changed");
  require(actual.owners() == reference.owners(), "grid owner/order changed");
  close(actual.points(), reference.points(), 2e-13, 3e-15, "grid coordinate mismatch");
  // Relative tails are tested as well as absolute weights; this catches a
  // mu reversal that replaces the reference's oriented log1p complement.
  close(actual.weights(), reference.weights(), 2e-12, 3e-11, "grid weight mismatch");
  close(actual.atomic_weights(), reference.atomic_weights(), 0, 0, "atomic export mismatch");
  if (derivative) {
    std::vector<double> sensitivity(actual.point_count());
    for (std::size_t p = 0; p < sensitivity.size(); ++p) {
      const auto& xyz = reference.points();
      sensitivity[p] = std::exp(-0.2 * (xyz[3 * p] * xyz[3 * p] + xyz[3 * p + 1] * xyz[3 * p + 1] +
                                        xyz[3 * p + 2] * xyz[3 * p + 2]));
    }
    close(actual.contract_weight_derivative(sensitivity),
          reference.contract_weight_derivative(sensitivity), 2e-10, 2e-10,
          "contracted derivative export mismatch");
  }
}

void accounting(const System& system, GridSpec spec) {
  using namespace vibeqc::runtime;
  const auto points =
      system.atoms.size() * spec.radial_points * spec.angular_polar * spec.angular_azimuth;
  const auto expected = vibeqc::dft::cuda_quadrature_bytes(system.atoms.size(), points);
  auto ledger = std::make_shared<DeviceResourceLedger>(DeviceResourceLedger{expected, 0});
  active_device_resource_ledger = ledger;
  (void)MolecularGrid::from_cuda(system, spec, 0);
  require(ledger->peak == expected && ledger->live == 0 && ledger->allocations == 2,
          "quadrature shape inventory differs from actual allocations");
  ledger->limit = expected - 1;
  bool rejected = false;
  try {
    (void)MolecularGrid::from_cuda(system, spec, 0);
  } catch (const std::bad_alloc&) {
    rejected = true;
  }
  require(rejected && ledger->live == 0 && ledger->rejected == 1,
          "quadrature OOM did not fail closed/clean up");
  active_device_resource_ledger.reset();
}
}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try {
    if (argc == 2) {
      const auto system = centers(std::stoul(argv[1]));
      GridSpec spec;
      spec.radial_points = 54;
      spec.angular_polar = 16;
      spec.angular_azimuth = 32;
      const auto start = std::chrono::steady_clock::now();
      const auto grid = MolecularGrid::from_cuda(system, spec, 0);
      std::cout.precision(12);
      std::cout << "{\"atoms\":" << system.atoms.size() << ",\"points\":" << grid.point_count()
                << ",\"cuda_grid_prepare_seconds\":"
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                << ",\"device_bytes\":"
                << vibeqc::dft::cuda_quadrature_bytes(system.atoms.size(), grid.point_count())
                << "}\n";
      return 0;
    }
    GridSpec spec;
    spec.radial_points = 5;
    spec.angular_polar = 7;
    spec.angular_azimuth = 23;
    auto system = centers(6);  // 4830 points crosses a tile and ends partway through an atom.
    for (unsigned version : {1U, 2U}) {
      spec.version = version;
      spec.element_radii[1] = 0.8;
      spec.element_radii[8] = 1.4;
      for (unsigned iterations = 1; iterations <= 5; ++iterations) {
        spec.partition_iterations = iterations;
        compare(system, spec, iterations == 3);
      }
    }
    spec.version = 1;
    spec.element_radii = {};
    compare(centers(1), spec, true);  // v1 fallback and analytic partition=1
    system.atoms[1].position = system.atoms[0].position;
    system.atoms[2].position = system.atoms[0].position;
    system.atoms[2].position[0] += 5e-13;
    compare(system, spec, true);
    for (auto& atom : system.atoms)
      for (double& x : atom.position) x += 7.125;
    compare(system, spec, true);
    std::reverse(system.atoms.begin(), system.atoms.end());
    compare(system, spec, true);
    accounting(system, spec);
    // Dynamic atom scratch, grid-stride coverage and tiny/partial tiles.
    spec.radial_points = 2;
    spec.angular_polar = 3;
    spec.angular_azimuth = 5;
    compare(centers(96), spec, false);
    System coincident;
    coincident.atoms = {{1, {0, 0, 0}}, {1, {0, 0, 0}}};
    const auto same = MolecularGrid::from_cuda(coincident, spec, 0);
    auto half = same.atomic_weights();
    for (double& w : half) w *= 0.5;
    close(same.weights(), half, 1e-13, 1e-14, "coincident partition differs from one half");
    spec.version = 2;
    bool rejected = false;
    try {
      (void)MolecularGrid::from_cuda(coincident, spec, 0);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "missing v2 radius accepted");
    rejected = false;
    try {
      (void)vibeqc::dft::cuda_quadrature_bytes(96, std::numeric_limits<std::size_t>::max());
    } catch (const std::overflow_error&) {
      rejected = true;
    }
    require(rejected, "export overflow accepted");
    std::cout << "CUDA quadrature scalar/analytic/resource/derivative gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

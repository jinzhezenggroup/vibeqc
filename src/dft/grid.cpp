#include "dft/grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace vibeqc::dft {
namespace {

std::size_t multiply(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("DFT grid size overflow");
  return a * b;
}

std::pair<std::vector<double>, std::vector<double>> gauss_legendre(std::size_t count) {
  if (!count || count > 512) throw std::invalid_argument("unsupported Gauss-Legendre order");
  std::vector<double> nodes(count), weights(count);
  const std::size_t half = (count + 1) / 2;
  for (std::size_t i = 0; i < half; ++i) {
    double root = std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) /
                           (static_cast<double>(count) + 0.5));
    double derivative = 0.0;
    for (unsigned iteration = 0; iteration < 100; ++iteration) {
      double previous = 1.0, current = root;
      for (std::size_t degree = 2; degree <= count; ++degree) {
        const double next =
            ((2.0 * degree - 1.0) * root * current - (degree - 1.0) * previous) / degree;
        previous = current;
        current = next;
      }
      derivative = count * (root * current - previous) / (root * root - 1.0);
      const double next = root - current / derivative;
      if (std::abs(next - root) <= 4.0 * std::numeric_limits<double>::epsilon()) {
        root = next;
        break;
      }
      root = next;
      if (iteration == 99) throw std::runtime_error("Gauss-Legendre root did not converge");
    }
    double previous = 1.0, current = root;
    for (std::size_t degree = 2; degree <= count; ++degree) {
      const double next =
          ((2.0 * degree - 1.0) * root * current - (degree - 1.0) * previous) / degree;
      previous = current;
      current = next;
    }
    derivative = count * (root * current - previous) / (root * root - 1.0);
    const double weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
    nodes[i] = -root;
    nodes[count - 1 - i] = root;
    weights[i] = weight;
    weights[count - 1 - i] = weight;
  }
  return {std::move(nodes), std::move(weights)};
}

double distance(const double* a, const double* b) {
  return std::hypot(std::hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]);
}

double owner_partition(const double* point, const core::System& system, std::size_t owner,
                       const GridSpec& spec) {
  const std::size_t atoms = system.atoms.size();
  std::vector<double> logs(atoms, 0.0);
  for (std::size_t a = 0; a < atoms; ++a) {
    for (std::size_t b = 0; b < a; ++b) {
      const double separation =
          distance(system.atoms[a].position.data(), system.atoms[b].position.data());
      double mu = 0.0;
      if (separation > spec.coincident_tolerance) {
        mu = std::clamp((distance(point, system.atoms[a].position.data()) -
                         distance(point, system.atoms[b].position.data())) /
                            separation,
                        -1.0, 1.0);
      }
      for (unsigned iteration = 0; iteration < spec.partition_iterations; ++iteration)
        mu = 0.5 * mu * (3.0 - mu * mu);
      const double pair = std::clamp(0.5 * (1.0 - mu), 0.0, 1.0);
      logs[a] += std::log(pair);
      logs[b] += std::log1p(-pair);
    }
  }
  const double maximum = *std::max_element(logs.begin(), logs.end());
  double sum = 0.0;
  for (double& value : logs) {
    value = std::exp(value - maximum);
    sum += value;
  }
  if (!(sum > 0.0) || !std::isfinite(sum))
    throw std::runtime_error("invalid Becke partition normalization");
  return logs[owner] / sum;
}

}  // namespace

MolecularGrid::MolecularGrid(const core::System& system, GridSpec spec)
    : system_(system), spec_(spec) {
  if (system_.atoms.empty()) throw std::invalid_argument("a DFT grid requires atoms");
  if (spec_.version != 1 || !spec_.radial_points || spec_.radial_points > 512 ||
      !spec_.angular_polar || spec_.angular_polar > 256 || spec_.angular_azimuth < 3 ||
      spec_.angular_azimuth > 1024 || !spec_.partition_iterations ||
      spec_.partition_iterations > 5 || !std::isfinite(spec_.coincident_tolerance) ||
      spec_.coincident_tolerance < 0.0)
    throw std::invalid_argument("unsupported DFT grid prescription/version");

  auto [polar, polar_weights] = gauss_legendre(spec_.angular_polar);
  auto [radial_nodes, radial_weights] = gauss_legendre(spec_.radial_points);
  const std::size_t angular = multiply(spec_.angular_polar, spec_.angular_azimuth);
  const std::size_t per_atom = multiply(spec_.radial_points, angular);
  const std::size_t total = multiply(system_.atoms.size(), per_atom);
  points_.reserve(multiply(total, 3));
  weights_.reserve(total);
  owners_.reserve(total);
  const double azimuth_weight = 2.0 * std::numbers::pi / spec_.angular_azimuth;
  for (std::size_t owner = 0; owner < system_.atoms.size(); ++owner) {
    const auto& center = system_.atoms[owner].position;
    for (std::size_t radial = 0; radial < spec_.radial_points; ++radial) {
      const double t = 0.5 * (radial_nodes[radial] + 1.0);
      const double wt = 0.5 * radial_weights[radial];
      const double r = t / (1.0 - t);
      const double wr = wt * r * r / ((1.0 - t) * (1.0 - t));
      for (std::size_t z = 0; z < spec_.angular_polar; ++z) {
        const double ring = std::sqrt(std::max(0.0, 1.0 - polar[z] * polar[z]));
        for (std::size_t azimuth = 0; azimuth < spec_.angular_azimuth; ++azimuth) {
          const double phi = 2.0 * std::numbers::pi * azimuth / spec_.angular_azimuth;
          const double point[3]{center[0] + r * ring * std::cos(phi),
                                center[1] + r * ring * std::sin(phi), center[2] + r * polar[z]};
          points_.insert(points_.end(), point, point + 3);
          weights_.push_back(wr * polar_weights[z] * azimuth_weight *
                             owner_partition(point, system_, owner, spec_));
          owners_.push_back(static_cast<std::uint32_t>(owner));
        }
      }
    }
  }
}

}  // namespace vibeqc::dft

#include "dft/grid.hpp"

#include <algorithm>
#include <array>
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

double radial_measure(double radius, double node, double weight) {
  const double t = 0.5 * (node + 1.0);
  const double r = radius * t / (1.0 - t);
  return 0.5 * weight * radius * r * r / ((1.0 - t) * (1.0 - t));
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

void validate_grid_spec(const GridSpec& spec) {
  if ((spec.version != 1 && spec.version != 2) || !spec.radial_points || spec.radial_points > 512 ||
      !spec.angular_polar || spec.angular_polar > 256 || spec.angular_azimuth < 3 ||
      spec.angular_azimuth > 1024 || !spec.partition_iterations || spec.partition_iterations > 5 ||
      !std::isfinite(spec.coincident_tolerance) || spec.coincident_tolerance < 0.0)
    throw std::invalid_argument("unsupported DFT grid prescription/version");
  for (double radius : spec.element_radii)
    if (!std::isfinite(radius) || radius < 0.0)
      throw std::invalid_argument("invalid DFT element radius");
}

MolecularGrid::MolecularGrid(const core::System& system, GridSpec spec)
    : system_(system), spec_(spec) {
  if (system_.atoms.empty()) throw std::invalid_argument("a DFT grid requires atoms");
  validate_grid_spec(spec_);

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
    const auto z = system_.atoms[owner].atomic_number;
    if (z < 1 || z > 118) throw std::invalid_argument("invalid grid atomic number");
    const double stored = spec_.element_radii[z];
    if (spec_.version >= 2 && !(stored > 0.0))
      throw std::invalid_argument("production DFT grid has no sourced element radius");
    const double radius = stored > 0.0 ? stored : 1.0;
    for (std::size_t radial = 0; radial < spec_.radial_points; ++radial) {
      const double t = 0.5 * (radial_nodes[radial] + 1.0);
      const double r = radius * t / (1.0 - t);
      const double wr = radial_measure(radius, radial_nodes[radial], radial_weights[radial]);
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

std::vector<double> MolecularGrid::atomic_weights() const {
  const auto [polar, polar_weights] = gauss_legendre(spec_.angular_polar);
  const auto [nodes, weights] = gauss_legendre(spec_.radial_points);
  const double azimuth_weight = 2.0 * std::numbers::pi / spec_.angular_azimuth;
  std::vector<double> result;
  result.reserve(point_count());
  for (const auto& atom : system_.atoms) {
    const double stored = spec_.element_radii[atom.atomic_number];
    if (spec_.version >= 2 && !(stored > 0.0))
      throw std::invalid_argument("production DFT grid has no sourced element radius");
    const double radius = stored > 0.0 ? stored : 1.0;
    for (std::size_t radial = 0; radial < nodes.size(); ++radial)
      for (double polar_weight : polar_weights)
        for (std::size_t azimuth = 0; azimuth < spec_.angular_azimuth; ++azimuth)
          result.push_back(radial_measure(radius, nodes[radial], weights[radial]) * polar_weight *
                           azimuth_weight);
  }
  return result;
}

std::vector<double> MolecularGrid::contract_weight_derivative(
    std::span<const double> weight_sensitivity) const {
  if (weight_sensitivity.size() != point_count())
    throw std::invalid_argument("DFT grid weight sensitivity count mismatch");
  for (double value : weight_sensitivity)
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite DFT grid weight sensitivity");

  const std::size_t atoms = system_.atoms.size();
  const std::size_t ncoord = multiply(atoms, std::size_t{3});
  std::vector<double> result(ncoord, 0.0);
  if (atoms == 1) return result;

  std::vector<double> logs(atoms), partition(atoms), log_derivative(multiply(atoms, ncoord));
  std::vector<double> average(ncoord);
  for (std::size_t point_index = 0; point_index < point_count(); ++point_index) {
    std::fill(logs.begin(), logs.end(), 0.0);
    std::fill(log_derivative.begin(), log_derivative.end(), 0.0);
    const double* point = points_.data() + 3 * point_index;
    const std::size_t owner = owners_[point_index];
    if (owner >= atoms) throw std::logic_error("DFT grid owner is out of range");

    for (std::size_t a = 0; a < atoms; ++a) {
      for (std::size_t b = 0; b < a; ++b) {
        const auto& ra_center = system_.atoms[a].position;
        const auto& rb_center = system_.atoms[b].position;
        double separation_vector[3]{}, point_a[3]{}, point_b[3]{};
        double separation2 = 0.0, distance_a2 = 0.0, distance_b2 = 0.0;
        for (unsigned axis = 0; axis < 3; ++axis) {
          separation_vector[axis] = ra_center[axis] - rb_center[axis];
          point_a[axis] = point[axis] - ra_center[axis];
          point_b[axis] = point[axis] - rb_center[axis];
          separation2 += separation_vector[axis] * separation_vector[axis];
          distance_a2 += point_a[axis] * point_a[axis];
          distance_b2 += point_b[axis] * point_b[axis];
        }
        const double separation = std::sqrt(separation2);
        const double distance_a = std::sqrt(distance_a2);
        const double distance_b = std::sqrt(distance_b2);
        double mu = 0.0;
        std::array<std::size_t, 3> pair_atoms{owner, 0, 0};
        std::array<std::array<double, 3>, 3> pair_derivative{};
        std::size_t pair_atom_count = 1;
        const auto pair_block = [&](std::size_t atom) {
          for (std::size_t block = 0; block < pair_atom_count; ++block)
            if (pair_atoms[block] == atom) return block;
          pair_atoms[pair_atom_count] = atom;
          return pair_atom_count++;
        };
        const std::size_t a_block = pair_block(a);
        const std::size_t b_block = pair_block(b);
        if (separation > spec_.coincident_tolerance) {
          const double numerator = distance_a - distance_b;
          const double raw_mu = numerator / separation;
          mu = std::clamp(raw_mu, -1.0, 1.0);
          if (raw_mu > -1.0 && raw_mu < 1.0) {
            for (unsigned axis = 0; axis < 3; ++axis) {
              const double unit_a = distance_a > 0.0 ? point_a[axis] / distance_a : 0.0;
              const double unit_b = distance_b > 0.0 ? point_b[axis] / distance_b : 0.0;
              const double unit_ab = separation_vector[axis] / separation;
              const double numerator_owner = unit_a - unit_b;
              pair_derivative[0][axis] += numerator_owner / separation;
              pair_derivative[a_block][axis] +=
                  -unit_a / separation - numerator * unit_ab / separation2;
              pair_derivative[b_block][axis] +=
                  unit_b / separation + numerator * unit_ab / separation2;
            }
          }
        }
        for (unsigned iteration = 0; iteration < spec_.partition_iterations; ++iteration) {
          const double slope = 1.5 * (1.0 - mu * mu);
          for (std::size_t block = 0; block < pair_atom_count; ++block)
            for (double& derivative : pair_derivative[block]) derivative *= slope;
          mu = 0.5 * mu * (3.0 - mu * mu);
        }
        const double pair = std::clamp(0.5 * (1.0 - mu), 0.0, 1.0);
        logs[a] += std::log(pair);
        logs[b] += std::log1p(-pair);
        if (pair > 0.0 && pair < 1.0) {
          for (std::size_t block = 0; block < pair_atom_count; ++block)
            for (unsigned axis = 0; axis < 3; ++axis) {
              const std::size_t coordinate = 3 * pair_atoms[block] + axis;
              const double pair_response = -0.5 * pair_derivative[block][axis];
              log_derivative[a * ncoord + coordinate] += pair_response / pair;
              log_derivative[b * ncoord + coordinate] -= pair_response / (1.0 - pair);
            }
        }
      }
    }

    const double maximum = *std::max_element(logs.begin(), logs.end());
    double normalization = 0.0;
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      partition[atom] = std::exp(logs[atom] - maximum);
      normalization += partition[atom];
    }
    if (!(normalization > 0.0) || !std::isfinite(normalization))
      throw std::runtime_error("invalid differentiated Becke partition normalization");
    for (double& value : partition) value /= normalization;
    std::fill(average.begin(), average.end(), 0.0);
    for (std::size_t atom = 0; atom < atoms; ++atom)
      for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate)
        average[coordinate] += partition[atom] * log_derivative[atom * ncoord + coordinate];

    const double scale = weight_sensitivity[point_index] * weights_[point_index];
    for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate) {
      result[coordinate] +=
          scale * (log_derivative[owner * ncoord + coordinate] - average[coordinate]);
    }
  }
  if (!std::all_of(result.begin(), result.end(), [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("nonfinite contracted DFT grid weight derivative");
  return result;
}

}  // namespace vibeqc::dft

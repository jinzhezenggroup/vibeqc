#include "molecule/basis.hpp"

#include <cmath>
#include <limits>
#include <numbers>

namespace qce::molecule {
namespace {

constexpr unsigned kMaximumPublicAngularMomentum = 3;

double radial_primitive_normalization(double exponent, unsigned angular_momentum) {
  return std::pow(2.0 * exponent / std::numbers::pi, 0.75) *
         std::pow(4.0 * exponent, 0.5 * static_cast<double>(angular_momentum));
}

double normalized_same_center_overlap(double alpha,
                                      double beta,
                                      unsigned angular_momentum) {
  const double ratio = 2.0 * std::sqrt(alpha * beta) / (alpha + beta);
  return std::pow(ratio, static_cast<double>(angular_momentum) + 1.5);
}

double odd_double_factorial(unsigned angular_power) noexcept {
  double value = 1.0;
  for (unsigned factor = 1; factor < 2 * angular_power; factor += 2) {
    value *= static_cast<double>(factor);
  }
  return value;
}

}  // namespace

std::vector<CartesianComponent> cartesian_components(unsigned l) {
  std::vector<CartesianComponent> components;
  components.reserve(cartesian_count(l));
  // CCA/libcint order: x power decreases first; for a fixed x power, y
  // increases and z is the remaining power.  Examples are p=(x,y,z) and
  // d=(xx,xy,xz,yy,yz,zz).
  for (int lx = static_cast<int>(l); lx >= 0; --lx) {
    for (unsigned lz = 0; lz <= l - static_cast<unsigned>(lx); ++lz) {
      const unsigned ly = l - static_cast<unsigned>(lx) - lz;
      components.push_back(
          {static_cast<unsigned>(lx), ly, lz});
    }
  }
  return components;
}

std::size_t ao_count(const core::System& system) noexcept {
  std::size_t count = 0;
  for (const core::Shell& shell : system.shells) {
    const std::size_t functions = cartesian_count(shell.angular_momentum);
    if (functions > std::numeric_limits<std::size_t>::max() - count) return 0;
    count += functions;
  }
  return count;
}

double cartesian_component_normalization(
    const CartesianComponent& component) noexcept {
  const double denominator =
      odd_double_factorial(component[0]) *
      odd_double_factorial(component[1]) *
      odd_double_factorial(component[2]);
  return 1.0 / std::sqrt(denominator);
}

qce_status validate_and_normalize(core::System& system, std::string& detail) {
  if (system.atoms.empty() || system.shells.empty()) {
    detail = "a system requires at least one atom and one basis shell";
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  int nuclear_charge = 0;
  for (const auto& atom : system.atoms) {
    if (atom.atomic_number <= 0) {
      detail = "atomic numbers must be positive";
      return QCE_STATUS_INVALID_ARGUMENT;
    }
    nuclear_charge += atom.atomic_number;
  }
  system.electron_count = nuclear_charge - system.charge;
  if (system.electron_count <= 0 || system.multiplicity == 0) {
    detail = "electron count must be positive and multiplicity nonzero";
    return QCE_STATUS_INVALID_ARGUMENT;
  }

  for (auto& shell : system.shells) {
    if (shell.atom_index >= system.atoms.size()) {
      detail = "shell atom index is out of range";
      return QCE_STATUS_INVALID_ARGUMENT;
    }
    if (shell.angular_momentum > kMaximumPublicAngularMomentum) {
      detail = "the executable Cartesian path currently supports s through f shells";
      return QCE_STATUS_NOT_IMPLEMENTED;
    }
    if (shell.primitives.empty()) {
      detail = "each shell requires at least one primitive";
      return QCE_STATUS_INVALID_ARGUMENT;
    }
    for (auto& primitive : shell.primitives) {
      if (!(primitive.exponent > 0.0) || !std::isfinite(primitive.coefficient)) {
        detail = "primitive exponents must be positive and coefficients finite";
        return QCE_STATUS_INVALID_ARGUMENT;
      }
    }

    // The overlap of two individually normalized primitives on the same
    // center depends only on total angular momentum, not on the Cartesian
    // distribution.  Therefore one contraction scale is valid for every AO
    // in the shell.  We retain a radial primitive factor in the coefficient;
    // the small component-dependent double-factorial factor is applied when
    // the shell is expanded into Cartesian AOs.
    double norm2 = 0.0;
    for (const auto& a : shell.primitives) {
      for (const auto& b : shell.primitives) {
        norm2 += a.coefficient * b.coefficient *
                 normalized_same_center_overlap(
                     a.exponent, b.exponent, shell.angular_momentum);
      }
    }
    if (!(norm2 > 0.0) || !std::isfinite(norm2)) {
      detail = "contracted shell has an invalid normalization";
      return QCE_STATUS_NUMERICAL_FAILURE;
    }
    const double scale = 1.0 / std::sqrt(norm2);
    for (auto& primitive : shell.primitives) {
      primitive.coefficient *=
          scale * radial_primitive_normalization(
                      primitive.exponent, shell.angular_momentum);
    }
  }
  return QCE_STATUS_SUCCESS;
}

}  // namespace qce::molecule

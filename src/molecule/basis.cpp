#include "molecule/basis.hpp"

#include <cmath>
#include <limits>
#include <numbers>

namespace vibeqc::molecule {
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

std::vector<AoExpansion> ao_expansions(
    unsigned l, vibeqc_basis_representation representation) {
  const std::vector<CartesianComponent> cartesian = cartesian_components(l);
  if (representation == VIBEQC_BASIS_CARTESIAN || l < 2) {
    std::vector<AoExpansion> expansions;
    expansions.reserve(cartesian.size());
    for (const CartesianComponent& component : cartesian) {
      expansions.push_back({{component, 1.0}});
    }
    return expansions;
  }

  if (l == 2) {
    // Cartesian order: xx, xy, xz, yy, yz, zz. Spherical columns follow
    // libcint/PySCF m=-2,-1,0,1,2 real-harmonic order.
    const double root_three_over_two = std::sqrt(3.0) / 2.0;
    return {
        {{{1, 1, 0}, 1.0}},
        {{{0, 1, 1}, 1.0}},
        {{{2, 0, 0}, -0.5}, {{0, 2, 0}, -0.5}, {{0, 0, 2}, 1.0}},
        {{{1, 0, 1}, 1.0}},
        {{{2, 0, 0}, root_three_over_two},
         {{0, 2, 0}, -root_three_over_two}},
    };
  }
  if (l == 3) {
    // Cartesian order: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz,
    // yzz, zzz. Coefficients act on individually normalized Cartesian AOs.
    const double three_over_root_eight = 3.0 / std::sqrt(8.0);
    const double root_five_over_eight = std::sqrt(5.0 / 8.0);
    const double root_three_over_forty = std::sqrt(3.0 / 40.0);
    const double root_three_over_eight = std::sqrt(3.0 / 8.0);
    const double root_six_over_five = std::sqrt(6.0 / 5.0);
    const double three_over_root_twenty = 3.0 / std::sqrt(20.0);
    const double root_three_over_two = std::sqrt(3.0) / 2.0;
    return {
        {{{2, 1, 0}, three_over_root_eight},
         {{0, 3, 0}, -root_five_over_eight}},
        {{{1, 1, 1}, 1.0}},
        {{{2, 1, 0}, -root_three_over_forty},
         {{0, 3, 0}, -root_three_over_eight},
         {{0, 1, 2}, root_six_over_five}},
        {{{2, 0, 1}, -three_over_root_twenty},
         {{0, 2, 1}, -three_over_root_twenty},
         {{0, 0, 3}, 1.0}},
        {{{3, 0, 0}, -root_three_over_eight},
         {{1, 2, 0}, -root_three_over_forty},
         {{1, 0, 2}, root_six_over_five}},
        {{{2, 0, 1}, root_three_over_two},
         {{0, 2, 1}, -root_three_over_two}},
        {{{3, 0, 0}, root_five_over_eight},
         {{1, 2, 0}, -three_over_root_eight}},
    };
  }
  return {};
}

std::size_t ao_count(const core::System& system) noexcept {
  std::size_t count = 0;
  for (const core::Shell& shell : system.shells) {
    const std::size_t functions =
        system.basis_representation == VIBEQC_BASIS_SPHERICAL
        ? 2 * static_cast<std::size_t>(shell.angular_momentum) + 1
        : cartesian_count(shell.angular_momentum);
    if (functions > std::numeric_limits<std::size_t>::max() - count) return 0;
    count += functions;
  }
  return count;
}

std::size_t cartesian_ao_count(const core::System& system) noexcept {
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

vibeqc_status validate_and_normalize(core::System& system, std::string& detail) {
  if (system.atoms.empty() || system.shells.empty()) {
    detail = "a system requires at least one atom and one basis shell";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  int nuclear_charge = 0;
  for (const auto& atom : system.atoms) {
    if (atom.atomic_number <= 0) {
      detail = "atomic numbers must be positive";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    nuclear_charge += atom.atomic_number;
  }
  system.electron_count = nuclear_charge - system.charge;
  if (system.electron_count <= 0 || system.multiplicity == 0) {
    detail = "electron count must be positive and multiplicity nonzero";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (system.basis_representation != VIBEQC_BASIS_CARTESIAN &&
      system.basis_representation != VIBEQC_BASIS_SPHERICAL) {
    detail = "basis representation must be Cartesian or spherical";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  for (auto& shell : system.shells) {
    if (shell.atom_index >= system.atoms.size()) {
      detail = "shell atom index is out of range";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    if (shell.angular_momentum > kMaximumPublicAngularMomentum) {
      detail = "the executable Cartesian path currently supports s through f shells";
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    }
    if (shell.primitives.empty()) {
      detail = "each shell requires at least one primitive";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    for (auto& primitive : shell.primitives) {
      if (!(primitive.exponent > 0.0) || !std::isfinite(primitive.coefficient)) {
        detail = "primitive exponents must be positive and coefficients finite";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
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
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    const double scale = 1.0 / std::sqrt(norm2);
    for (auto& primitive : shell.primitives) {
      primitive.coefficient *=
          scale * radial_primitive_normalization(
                      primitive.exponent, shell.angular_momentum);
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::molecule

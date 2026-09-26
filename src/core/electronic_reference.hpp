#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace vibeqc::core {

/** Borrowed canonical-orbital data for one spin channel.
 * Storage remains owned by the producing method state. */
struct OrbitalChannelView {
  std::size_t occupied{};
  std::span<const double> coefficients;
  std::span<const double> orbital_energies;
  std::span<const double> density;
  std::span<const double> fock;
  std::span<const double> weighted_density;
};

/** Method-neutral, read-only electronic reference contract.
 * Restricted references use one channel; unrestricted references use two.
 * All matrices are row-major nbf-by-nbf and orbital coefficients are columns. */
struct ElectronicReferenceView {
  std::size_t basis_functions{};
  std::size_t spin_channels{};
  std::span<const double> overlap;
  std::span<const double> hcore;
  std::array<OrbitalChannelView, 2> channels{};
  double energy{};

  [[nodiscard]] bool restricted() const noexcept { return spin_channels == 1; }
  [[nodiscard]] std::size_t virtual_orbitals(std::size_t spin) const {
    if (spin >= spin_channels || spin >= channels.size())
      throw std::out_of_range("electronic reference spin index");
    if (spin_channels > channels.size() || !basis_functions ||
        channels[spin].occupied > basis_functions)
      throw std::invalid_argument("invalid electronic reference orbital dimensions");
    return basis_functions - channels[spin].occupied;
  }
};

[[nodiscard]] inline bool electronic_reference_shape_valid(
    const ElectronicReferenceView& reference) noexcept {
  const auto n = reference.basis_functions;
  if (!n || !reference.spin_channels || reference.spin_channels > reference.channels.size() ||
      n > std::numeric_limits<std::size_t>::max() / n)
    return false;
  const auto matrix_size = n * n;
  if (reference.overlap.size() != matrix_size || reference.hcore.size() != matrix_size)
    return false;

  for (std::size_t spin = 0; spin < reference.spin_channels; ++spin) {
    const auto& channel = reference.channels[spin];
    if (channel.occupied > n || channel.coefficients.size() != matrix_size ||
        channel.orbital_energies.size() != n || channel.density.size() != matrix_size ||
        channel.fock.size() != matrix_size ||
        (!channel.weighted_density.empty() && channel.weighted_density.size() != matrix_size))
      return false;
  }
  return true;
}

inline void validate_electronic_reference_shape(const ElectronicReferenceView& reference) {
  if (!electronic_reference_shape_valid(reference))
    throw std::invalid_argument("invalid electronic reference shape");
}

}  // namespace vibeqc::core

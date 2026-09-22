#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/types.hpp"

namespace vibeqc::molecule {

/** Exact immutable integral-source identity, not a hash or mutable pointer token.
 * Coulomb integral values depend on geometry, basis and representation, not occupations.
 * This is not an electronic-method or ECP-operator identity.
 * Construction uses two bounded passes to avoid temporary vector reallocations;
 * matches performs no allocation, GPU access or approximate numeric comparison. */
class BasisGeometryIdentity {
 public:
  explicit BasisGeometryIdentity(const core::System& system) {
    std::size_t count = 0;
    visit(system, [&](std::uint64_t) {
      if (count == words_.max_size()) throw std::length_error("DF source identity is too large");
      ++count;
      return true;
    });
    words_.reserve(count);
    visit(system, [&](std::uint64_t word) {
      words_.push_back(word);
      return true;
    });
  }

  [[nodiscard]] bool matches(const core::System& system) const noexcept {
    std::size_t index = 0;
    return visit(system,
                 [&](std::uint64_t word) {
                   return index < words_.size() && words_[index++] == word;
                 }) &&
           index == words_.size();
  }

  [[nodiscard]] std::size_t storage_bytes() const noexcept {
    return words_.capacity() * sizeof(std::uint64_t);
  }

 private:
  template <class Sink>
  static bool visit(const core::System& system, Sink&& sink) {
    if (!sink(1) || !sink(static_cast<std::uint64_t>(system.basis_representation)) ||
        !sink(system.atoms.size()))
      return false;
    for (const auto& atom : system.atoms) {
      if (!sink(static_cast<std::uint64_t>(atom.atomic_number)) ||
          !sink(static_cast<std::uint64_t>(atom.ecp_core)))
        return false;
      for (double coordinate : atom.position)
        if (!sink(std::bit_cast<std::uint64_t>(coordinate))) return false;
    }
    if (!sink(system.shells.size())) return false;
    for (const auto& shell : system.shells) {
      if (!sink(shell.atom_index) || !sink(shell.angular_momentum) ||
          !sink(shell.primitives.size()))
        return false;
      for (const auto& primitive : shell.primitives)
        if (!sink(std::bit_cast<std::uint64_t>(primitive.exponent)) ||
            !sink(std::bit_cast<std::uint64_t>(primitive.coefficient)))
          return false;
    }
    return true;
  }

  std::vector<std::uint64_t> words_;
};

}  // namespace vibeqc::molecule

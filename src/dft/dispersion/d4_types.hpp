#pragma once

#include <cstdint>

namespace vibeqc::dft::dispersion::data {

struct D4ElementData {
  std::uint16_t reference_offset;
  std::uint8_t reference_count;
  double covalent_radius;
  double electronegativity;
  double effective_charge;
  double hardness;
  double r4r2;
};

struct D4ReferenceData {
  double coordination_number;
  double charge;
  std::uint8_t gaussian_count;
};

}  // namespace vibeqc::dft::dispersion::data

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "core/types.hpp"
#include "molecule/basis.hpp"

namespace {
vibeqc::core::System system() {
  vibeqc::core::System result;
  result.atoms = {{26, {0, 0, 0}}};
  result.shells = {{0, 0, {{80, 1}, {5, 0}}}};
  result.charge = 24;
  return result;
}
void require(bool condition, const char* text) {
  if (!condition) throw std::runtime_error(text);
}
}  // namespace
int main() {
  try {
    std::string detail;
    auto valid = system();
    require(vibeqc::molecule::validate_and_normalize(valid, detail) == VIBEQC_STATUS_SUCCESS,
            "all-electron Fe ion with explicit small basis must be accepted");
    require(valid.electron_count == 2, "ionic charge must remain distinct from nuclear Z");
    require(
        valid.shells[0].primitives.size() == 2 && valid.shells[0].primitives[1].coefficient == 0,
        "zero primitives must be retained");
    for (double exponent : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
      auto input = system();
      input.shells[0].primitives[0].exponent = exponent;
      require(
          vibeqc::molecule::validate_and_normalize(input, detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "nonpositive or nonfinite exponent must fail before normalization");
    }
    auto overflow = system();
    overflow.charge = std::numeric_limits<int>::min();
    require(vibeqc::molecule::validate_and_normalize(overflow, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "electron count must not overflow signed ABI storage");
    auto underflow = system();
    underflow.shells[0].primitives[1] = {0.01, std::numeric_limits<double>::denorm_min()};
    require(vibeqc::molecule::validate_and_normalize(underflow, detail) ==
                VIBEQC_STATUS_NUMERICAL_FAILURE,
            "native normalization must not silently erase a nonzero primitive");
    auto spin_overflow = system();
    spin_overflow.multiplicity = std::numeric_limits<unsigned>::max();
    require(vibeqc::molecule::validate_and_normalize(spin_overflow, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "multiplicity must not wrap signed native occupation storage");
    spin_overflow.charge = 26 - std::numeric_limits<int>::max();
    spin_overflow.multiplicity = 2;
    require(vibeqc::molecule::validate_and_normalize(spin_overflow, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "electron plus spin occupation intermediate must not overflow");
    auto unsupported = system();
    unsupported.shells[0].angular_momentum = 4;
    require(vibeqc::molecule::validate_and_normalize(unsupported, detail) ==
                    VIBEQC_STATUS_NOT_IMPLEMENTED &&
                detail.find("l=4") != std::string::npos,
            "high angular momentum must identify the missing shell");
    auto invalid_atom = system();
    invalid_atom.atoms[0].atomic_number = 119;
    require(vibeqc::molecule::validate_and_normalize(invalid_atom, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "invalid atomic number must not become a nuclear charge");
    auto invalid_position = system();
    invalid_position.atoms[0].position[1] = std::numeric_limits<double>::infinity();
    require(vibeqc::molecule::validate_and_normalize(invalid_position, detail) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "nonfinite coordinates must fail before integral execution");
    std::cout << "basis contract gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

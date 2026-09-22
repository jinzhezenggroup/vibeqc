#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "scf/solver/warm_subspace.hpp"

namespace {
using Matrix = std::vector<double>;

void require(bool value, const char* detail) {
  if (!value) throw std::runtime_error(detail);
}

void invariant_subspace_survives_occupied_rotation() {
  using namespace vibeqc::scf::solver;
  const double a = std::sqrt(0.5);
  // The first two columns are an arbitrary rotation inside the occupied
  // subspace. They are not eigenvectors of the 1/2 block, but their projector
  // is invariant because F has no occupied/virtual coupling.
  const Matrix fock{1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0};
  const Matrix orbitals{a, -a, 0.0, a, a, 0.0, 0.0, 0.0, 1.0};
  const auto diagnostic = inspect_warm_occupied_subspace(fock, orbitals, 3, 2);
  require(diagnostic.finite, "finite invariant subspace reported non-finite");
  require(diagnostic.maximum_residual < 1.0e-14,
          "occupied rotation manufactured an external residual");
  std::string detail;
  require(accept_warm_occupied_subspace(diagnostic, 1.0e-12, 1.0e-12, detail),
          "invariant occupied subspace rejected");
}

void occupied_virtual_coupling_triggers_fallback() {
  using namespace vibeqc::scf::solver;
  const Matrix fock{1.0, 0.0, 0.125, 0.0, 2.0, 0.0, 0.125, 0.0, 5.0};
  const Matrix identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const auto diagnostic = inspect_warm_occupied_subspace(fock, identity, 3, 2);
  require(diagnostic.finite, "finite coupled subspace reported non-finite");
  require(std::abs(diagnostic.maximum_residual - 0.125) < 1.0e-14,
          "occupied/virtual coupling magnitude was not recovered");
  require(std::abs(diagnostic.frobenius_residual - 0.125) < 1.0e-14,
          "occupied residual norm was not recovered");
  std::string detail;
  require(!accept_warm_occupied_subspace(diagnostic, 0.1, 1.0, detail),
          "large occupied/virtual coupling did not request dense fallback");
  require(accept_warm_occupied_subspace(diagnostic, 0.13, 1.0, detail),
          "explicit relaxed probe gate rejected expected evidence");
}

void invalid_evidence_is_fail_closed() {
  using namespace vibeqc::scf::solver;
  Matrix fock{1.0, 0.0, 0.0, 2.0};
  Matrix orbitals{1.0, 0.0, 0.0, 1.0};
  orbitals[0] = std::numeric_limits<double>::quiet_NaN();
  const auto diagnostic = inspect_warm_occupied_subspace(fock, orbitals, 2, 1);
  require(!diagnostic.finite, "non-finite orbital frame was accepted as finite");
  std::string detail;
  require(!accept_warm_occupied_subspace(diagnostic, 1.0, 1.0, detail),
          "non-finite warm evidence did not fail closed");

  bool threw = false;
  try {
    (void)inspect_warm_occupied_subspace(Matrix{1.0}, Matrix{1.0}, 2, 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "shape mismatch did not reject warm-subspace input");
}

void finite_input_scale_overflow_requests_fallback() {
  using namespace vibeqc::scf::solver;
  // Every entry and the coupling residual is finite, but ||F||*||C||+||FC||
  // overflows binary64. Dividing by infinity must not manufacture a zero gate.
  const Matrix fock{8.0e307, 0.0, 0.125, 0.0, 8.0e307, 0.0, 0.125, 0.0, 8.0e307};
  const Matrix identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const auto diagnostic = inspect_warm_occupied_subspace(fock, identity, 3, 2);
  std::string detail;
  require(!diagnostic.finite, "overflowed residual normalization reported valid evidence");
  require(!accept_warm_occupied_subspace(diagnostic, 1.0, 0.0, detail),
          "overflowed scale manufactured an accepted zero residual");
}

void full_space_is_trivially_invariant() {
  using namespace vibeqc::scf::solver;
  const Matrix fock{2.0, 0.25, 0.25, 3.0};
  const Matrix identity{1.0, 0.0, 0.0, 1.0};
  const auto diagnostic = inspect_warm_occupied_subspace(fock, identity, 2, 2);
  require(diagnostic.maximum_residual < 1.0e-14,
          "full orbital space should have no external residual");
}

}  // namespace

int main() {
  try {
    invariant_subspace_survives_occupied_rotation();
    occupied_virtual_coupling_triggers_fallback();
    invalid_evidence_is_fail_closed();
    full_space_is_trivially_invariant();
    finite_input_scale_overflow_requests_fallback();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}

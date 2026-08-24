#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/rhf.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

qce::core::System helium_hydrogen_sd() {
  qce::core::System system;
  system.atoms = {{2, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.5, 1.0}}},
      {0, 2, {{0.8, 1.0}}},
      {1, 0, {{1.2, 1.0}}},
  };
  system.charge = 1;
  system.multiplicity = 1;
  system.basis_representation = QCE_BASIS_SPHERICAL;
  std::string detail;
  require(qce::molecule::validate_and_normalize(system, detail) ==
              QCE_STATUS_SUCCESS,
          "spherical s/d system normalization failed");
  return system;
}

qce::core::System single_f_shell() {
  qce::core::System system;
  system.atoms = {{2, {0.0, 0.0, 0.0}}};
  system.shells = {{0, 3, {{0.6, 1.0}}}};
  system.basis_representation = QCE_BASIS_SPHERICAL;
  std::string detail;
  require(qce::molecule::validate_and_normalize(system, detail) ==
              QCE_STATUS_SUCCESS,
          "spherical f system normalization failed");
  return system;
}

}  // namespace

int main() {
  try {
    const qce::core::System sd = helium_hydrogen_sd();
    require(qce::molecule::cartesian_ao_count(sd) == 8,
            "Cartesian source count for the s/d system is incorrect");
    require(qce::molecule::ao_count(sd) == 7,
            "spherical s/d AO count is incorrect");

    qce::core::ScfOptions options;
    options.energy_tolerance = 1.0e-12;
    options.density_tolerance = 1.0e-10;
    const qce::core::ScfResult result = qce::scf::run_rhf(sd, options);
    require(result.converged, "spherical s/d RHF did not converge");
    // Independent PySCF/libcint reference with cart=False and the exact same
    // one-primitive basis definitions.
    require(std::abs(result.energy - (-2.3341870407859284)) < 3.0e-12,
            "spherical s/d energy differs from PySCF");
    require(std::abs(result.forces[2] - 0.3502792384052442) < 8.0e-12 &&
                std::abs(result.forces[5] + 0.3502792384052438) < 8.0e-12,
            "spherical s/d analytic force differs from PySCF");

    const qce::integrals::IntegralData f_integrals =
        qce::integrals::build_integrals(single_f_shell());
    require(f_integrals.nbf == 7,
            "spherical f transform produced the wrong AO count");
    for (std::size_t row = 0; row < f_integrals.nbf; ++row) {
      for (std::size_t column = 0; column < f_integrals.nbf; ++column) {
        const double expected = row == column ? 1.0 : 0.0;
        require(std::abs(
                    f_integrals.overlap[row * f_integrals.nbf + column] -
                    expected) < 3.0e-13,
                "real-spherical f functions are not orthonormal");
      }
    }

    std::cout << "validated real-spherical d/f transforms and RHF gradient\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}


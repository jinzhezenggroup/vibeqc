#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/rhf.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(double actual,
                   double expected,
                   double tolerance,
                   const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(message) + ": actual=" +
                             std::to_string(actual) +
                             " expected=" + std::to_string(expected));
  }
}

std::size_t matrix_index(std::size_t row, std::size_t column, std::size_t n) {
  return row * n + column;
}

std::size_t eri_index(std::size_t i,
                      std::size_t j,
                      std::size_t k,
                      std::size_t l,
                      std::size_t n) {
  return ((i * n + j) * n + k) * n + l;
}

qce::core::System hydrogen_sp_dimer() {
  qce::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.2, 1.0}}}, {0, 1, {{0.7, 1.0}}},
      {1, 0, {{1.2, 1.0}}}, {1, 1, {{0.7, 1.0}}},
  };
  system.multiplicity = 1;
  std::string detail;
  require(qce::molecule::validate_and_normalize(system, detail) ==
              QCE_STATUS_SUCCESS,
          "s/p system normalization failed");
  return system;
}

qce::core::System helium_hydrogen_sdf() {
  qce::core::System system;
  system.atoms = {{2, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.5, 1.0}}},
      {0, 2, {{0.8, 1.0}}},
      {0, 3, {{0.6, 1.0}}},
      {1, 0, {{1.2, 1.0}}},
  };
  system.charge = 1;
  system.multiplicity = 1;
  std::string detail;
  require(qce::molecule::validate_and_normalize(system, detail) ==
              QCE_STATUS_SUCCESS,
          "s/d/f system normalization failed");
  return system;
}

}  // namespace

int main() {
  try {
    const qce::core::System system = hydrogen_sp_dimer();
    require(qce::molecule::ao_count(system) == 8,
            "s/p shell expansion produced the wrong AO count");
    const qce::scf::CudaRhfBasisLayoutStats layout =
        qce::scf::inspect_rhf_cuda_basis_layout({system});
    require(layout.shell_count == 4 && layout.shell_pair_count == 10 &&
                layout.shell_quartet_count == 55 &&
                layout.ao_count == 8,
            "CUDA basis layout lost the shell-to-AO topology");
    require(layout.unique_primitive_count == 4,
            "CUDA basis layout duplicated shell primitives");
    require(layout.expanded_primitive_references == 8,
            "expanded primitive diagnostic has the wrong Cartesian count");
    require(layout.device_basis_bytes == 452,
            "CUDA basis topology payload changed unexpectedly");
    const qce::scf::CudaRhfBasisLayoutStats sdf_layout =
        qce::scf::inspect_rhf_cuda_basis_layout({helium_hydrogen_sdf()});
    require(sdf_layout.shell_count == 4 &&
                sdf_layout.shell_pair_count == 10 &&
                sdf_layout.shell_quartet_count == 55 &&
                sdf_layout.ao_count == 18,
            "s/d/f CUDA shell-to-AO topology is inconsistent");
    require(sdf_layout.unique_primitive_count == 4 &&
                sdf_layout.expanded_primitive_references == 18,
            "s/d/f primitive storage was expanded per Cartesian component");
    require(sdf_layout.device_basis_bytes == 602,
            "s/d/f CUDA basis topology payload changed unexpectedly");
    const qce::integrals::IntegralData integrals =
        qce::integrals::build_cartesian_integrals(system);
    require(integrals.nbf == 8, "integral engine reported the wrong AO count");

    // Values were generated independently with PySCF 2.11/libcint using
    // cart=True and the raw basis {s: (1.2,1), p: (0.7,1)} on each hydrogen.
    require_close(std::accumulate(integrals.overlap.begin(),
                                  integrals.overlap.end(), 0.0),
                  10.256697920226276, 2.0e-12,
                  "Cartesian overlap checksum differs from libcint");
    require_close(std::accumulate(integrals.hcore.begin(),
                                  integrals.hcore.end(), 0.0),
                  -2.580918402607146, 3.0e-12,
                  "Cartesian core-Hamiltonian checksum differs from libcint");
    require_close(std::accumulate(integrals.eri.begin(), integrals.eri.end(),
                                  0.0),
                  83.63765625818158, 2.0e-11,
                  "Cartesian ERI checksum differs from libcint");

    const std::size_t n = integrals.nbf;
    require_close(integrals.overlap[matrix_index(0, 7, n)],
                  -0.5894285335123269, 2.0e-13,
                  "s-p overlap ordering/sign differs from libcint");
    require_close(integrals.hcore[matrix_index(3, 7, n)],
                  -0.24217477539847332, 3.0e-13,
                  "p-p core Hamiltonian differs from libcint");
    require_close(integrals.eri[eri_index(0, 1, 0, 1, n)],
                  0.12127971271176609, 3.0e-13,
                  "mixed s/p ERI differs from libcint");
    require_close(integrals.eri[eri_index(0, 7, 4, 3, n)],
                  -0.3003230202786651, 3.0e-13,
                  "four-center s/p ERI ordering/sign differs from libcint");

    std::cout << "validated 8-AO Cartesian s/p integrals against PySCF/libcint\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

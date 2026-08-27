#include "vibeqc/vibeqc.h"
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

vibeqc::core::System hydrogen_molecular_ion(double distance) {
  vibeqc::core::System system;
  system.atoms = {
      {1, {0.0, 0.0, -0.5 * distance}},
      {1, {0.0, 0.0, 0.5 * distance}},
  };
  system.shells = {
      {0, 0, {{3.42525091, 0.15432897},
              {0.62391373, 0.53532814},
              {0.16885540, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897},
              {0.62391373, 0.53532814},
              {0.16885540, 0.44463454}}},
  };
  system.charge = 1;
  system.multiplicity = 2;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "H2+ UHF system normalization failed");
  return system;
}

vibeqc::scf::ScfResult evaluate(double distance,
                                const std::vector<double>* warm = nullptr) {
  vibeqc::scf::ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1.0e-12;
  options.density_tolerance = 1.0e-10;
  return vibeqc::scf::run_uhf(hydrogen_molecular_ion(distance), options, warm);
}

#if VIBEQC_HAS_CUDA
bool cuda_device_available() {
  // A CUDA-enabled build may still run on a login node without an allocated
  // device. Probe through the public runtime contract so the CPU UHF oracle
  // remains testable there without hiding CUDA failures on allocated workers.
  vibeqc_context_descriptor descriptor{
      sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, 0, VIBEQC_BACKEND_CUDA};
  vibeqc_context* context = nullptr;
  const vibeqc_status status = vibeqc_context_create(&descriptor, &context);
  if (context != nullptr) vibeqc_context_destroy(context);
  return status == VIBEQC_STATUS_SUCCESS;
}
#endif

}  // namespace

int main() {
  try {
    const vibeqc::scf::ScfResult center = evaluate(1.4);
    require(center.converged, "H2+ UHF did not converge");
    require(std::abs(center.energy - (-0.53851134755010321)) < 2.0e-10,
            "H2+ UHF energy differs from PySCF");
    require(center.density.size() == 8,
            "UHF warm state does not contain alpha and beta matrices");
    require(std::abs(center.forces[2] - (-0.19038408605055362)) < 3.0e-9 &&
                std::abs(center.forces[5] - 0.19038408605055368) < 3.0e-9,
            "H2+ analytic UHF force differs from PySCF");

    const double step = 1.0e-4;
    const vibeqc::scf::ScfResult plus = evaluate(1.4 + step);
    const vibeqc::scf::ScfResult minus = evaluate(1.4 - step);
    require(plus.converged && minus.converged,
            "finite-difference UHF points did not converge");
    const double derivative = (plus.energy - minus.energy) / (2.0 * step);
    require(std::abs(center.forces[5] + derivative) < 2.0e-6,
            "analytic UHF force disagrees with energy finite differences");

    const vibeqc::scf::ScfResult warm = evaluate(1.4, &center.density);
    require(warm.converged && warm.initial_density_used,
            "UHF packed spin density was not accepted as a warm start");
    require(std::abs(warm.energy - center.energy) < 2.0e-12,
            "warm UHF energy changed the converged state");

#if VIBEQC_HAS_CUDA
    if (cuda_device_available()) {
      vibeqc::scf::ScfOptions cuda_options;
      cuda_options.max_iterations = 100;
      cuda_options.energy_tolerance = 1.0e-12;
      cuda_options.density_tolerance = 1.0e-10;
      const vibeqc::scf::ScfResult cuda = vibeqc::scf::run_uhf_cuda(
          hydrogen_molecular_ion(1.4), cuda_options, 0);
      require(cuda.converged, "CUDA H2+ UHF did not converge");
      require(std::abs(cuda.energy - center.energy) < 2.0e-10,
              "CUDA H2+ UHF energy differs from the CPU oracle");
      require(cuda.density.size() == center.density.size(),
              "CUDA UHF did not retain both spin-density matrices");
      for (std::size_t coordinate = 0; coordinate < center.forces.size();
           ++coordinate) {
        require(std::abs(cuda.forces[coordinate] - center.forces[coordinate]) <
                    3.0e-9,
                "CUDA analytic UHF force differs from the CPU oracle");
      }
      const vibeqc::scf::ScfResult cuda_warm = vibeqc::scf::run_uhf_cuda(
          hydrogen_molecular_ion(1.4), cuda_options, 0, &cuda.density);
      require(cuda_warm.converged && cuda_warm.initial_density_used,
              "CUDA UHF packed spin density was not accepted as a warm start");
    } else {
      std::cout << "CUDA UHF checks skipped: no allocated CUDA device\n";
    }
#endif

    std::cout << "H2+ UHF energy: " << center.energy << '\n';
    std::cout << "H2+ UHF force z(atom 1): " << center.forces[5] << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

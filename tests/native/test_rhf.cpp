#include "qce/qce.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

struct Evaluation {
  double energy{};
  std::array<double, 6> forces{};
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Evaluation h2(double distance) {
  qce_context_descriptor context_descriptor{
      sizeof(qce_context_descriptor), QCE_ABI_VERSION, 0,
      QCE_BACKEND_CPU_REFERENCE};
  qce_context* context = nullptr;
  require(qce_context_create(&context_descriptor, &context) == QCE_STATUS_SUCCESS,
          "context creation failed");

  const std::array<qce_atom, 2> atoms{{
      {1, 0.0, 0.0, -0.5 * distance},
      {1, 0.0, 0.0, 0.5 * distance},
  }};
  const std::array<qce_primitive, 6> primitives{{
      {3.42525091, 0.15432897},
      {0.62391373, 0.53532814},
      {0.16885540, 0.44463454},
      {3.42525091, 0.15432897},
      {0.62391373, 0.53532814},
      {0.16885540, 0.44463454},
  }};
  const std::array<qce_shell, 2> shells{{
      {0, 0, 0, 3},
      {1, 0, 3, 3},
  }};
  qce_system_descriptor system_descriptor{
      sizeof(qce_system_descriptor), QCE_ABI_VERSION,
      atoms.data(), static_cast<uint32_t>(atoms.size()),
      shells.data(), static_cast<uint32_t>(shells.size()),
      primitives.data(), static_cast<uint32_t>(primitives.size()),
      0, 1};
  qce_system* system = nullptr;
  const qce_status system_status =
      qce_system_create(context, &system_descriptor, &system);
  require(system_status == QCE_STATUS_SUCCESS, "system creation failed");

  qce_method_descriptor method{
      sizeof(qce_method_descriptor), QCE_ABI_VERSION, QCE_METHOD_RHF,
      100, 8, 1.0e-12, 1.0e-10, 1.0e-14};
  qce_calculation* calculation = nullptr;
  require(qce_calculation_prepare(context, system, &method, &calculation) ==
              QCE_STATUS_SUCCESS,
          "calculation preparation failed");

  Evaluation evaluation;
  qce_result_descriptor result{
      sizeof(qce_result_descriptor), QCE_ABI_VERSION, 0.0,
      evaluation.forces.data(), static_cast<uint32_t>(evaluation.forces.size()),
      0, 0.0, 0.0, 0, QCE_BACKEND_CPU_REFERENCE};
  const qce_status status = qce_calculation_execute(calculation, &result);
  require(status == QCE_STATUS_SUCCESS, "RHF execution failed");
  require(result.converged == 1, "RHF did not report convergence");
  evaluation.energy = result.energy;

  qce_calculation_destroy(calculation);
  qce_system_destroy(system);
  qce_context_destroy(context);
  return evaluation;
}

}  // namespace

int main() {
  try {
    int available = -1;
    require(qce_method_available(QCE_METHOD_RHF, &available) == QCE_STATUS_SUCCESS &&
                available == 1,
            "RHF capability query failed");
    require(qce_method_available(QCE_METHOD_UHF, &available) ==
                QCE_STATUS_SUCCESS && available == 1,
            "UHF capability query failed");
    require(qce_method_available(QCE_METHOD_WB97M_V, &available) ==
                QCE_STATUS_SUCCESS && available == 0,
            "wB97M-V must remain explicitly unavailable");

    const Evaluation center = h2(1.4);
    require(std::abs(center.energy - (-1.11671432506255)) < 2.0e-9,
            "H2/STO-3G RHF energy differs from the reference");
    for (int axis = 0; axis < 3; ++axis) {
      require(std::abs(center.forces[axis] + center.forces[3 + axis]) < 2.0e-10,
              "forces violate translational invariance");
    }

    const double step = 1.0e-4;
    const Evaluation plus = h2(1.4 + step);
    const Evaluation minus = h2(1.4 - step);
    const double d_energy_d_distance = (plus.energy - minus.energy) / (2.0 * step);
    // With atoms at +/-R/2, force_z(atom 1) equals -dE/dR.
    require(std::abs(center.forces[5] + d_energy_d_distance) < 2.0e-6,
            "analytic RHF force disagrees with finite differences");

    std::cout << "H2 energy: " << center.energy << '\n';
    std::cout << "H2 force z(atom 1): " << center.forces[5] << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

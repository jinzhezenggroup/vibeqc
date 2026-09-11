#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "vibeqc/vibeqc.h"

namespace {

struct Evaluation {
  double energy{};
  std::array<double, 6> forces{};
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// Exercise output selection on one retained C ABI calculation, including
// energy-only as its first execution and forces after an energy-only replay.
Evaluation h2(double distance, bool verify_energy_only = false,
              vibeqc_backend backend = VIBEQC_BACKEND_CPU_REFERENCE,
              vibeqc_density_fitting_mode df_mode = VIBEQC_DENSITY_FITTING_NONE,
              bool unrestricted = false, std::uint64_t df_budget = 0) {
  vibeqc_context_descriptor context_descriptor{sizeof(vibeqc_context_descriptor),
                                               VIBEQC_ABI_VERSION, 0, backend};
  vibeqc_context* context = nullptr;
  require(vibeqc_context_create(&context_descriptor, &context) == VIBEQC_STATUS_SUCCESS,
          "context creation failed");

  const std::array<vibeqc_atom, 2> atoms{{
      {1, 0.0, 0.0, -0.5 * distance},
      {1, 0.0, 0.0, 0.5 * distance},
  }};
  const std::array<vibeqc_primitive, 6> primitives{{
      {3.42525091, 0.15432897},
      {0.62391373, 0.53532814},
      {0.16885540, 0.44463454},
      {3.42525091, 0.15432897},
      {0.62391373, 0.53532814},
      {0.16885540, 0.44463454},
  }};
  const std::array<vibeqc_shell, 2> shells{{
      {0, 0, 0, 3},
      {1, 0, 3, 3},
  }};
  vibeqc_system_descriptor system_descriptor{sizeof(vibeqc_system_descriptor),
                                             VIBEQC_ABI_VERSION,
                                             atoms.data(),
                                             static_cast<uint32_t>(atoms.size()),
                                             shells.data(),
                                             static_cast<uint32_t>(shells.size()),
                                             primitives.data(),
                                             static_cast<uint32_t>(primitives.size()),
                                             0,
                                             1};
  system_descriptor.charge = unrestricted ? 1 : 0;
  system_descriptor.multiplicity = unrestricted ? 2 : 1;
  vibeqc_system* system = nullptr;
  const vibeqc_status system_status = vibeqc_system_create(context, &system_descriptor, &system);
  require(system_status == VIBEQC_STATUS_SUCCESS, "system creation failed");

  vibeqc_method_descriptor method{sizeof(vibeqc_method_descriptor),
                                  VIBEQC_ABI_VERSION,
                                  VIBEQC_METHOD_RHF,
                                  100,
                                  8,
                                  1.0e-12,
                                  1.0e-10,
                                  1.0e-14};
  method.method = unrestricted ? VIBEQC_METHOD_UHF : VIBEQC_METHOD_RHF;
  method.density_fitting_mode = df_mode;
  method.density_fitting_memory_budget_bytes = df_budget;
  vibeqc_calculation* calculation = nullptr;
  require(
      vibeqc_calculation_prepare(context, system, &method, &calculation) == VIBEQC_STATUS_SUCCESS,
      "calculation preparation failed");

  Evaluation evaluation;
  vibeqc_result_descriptor result{sizeof(vibeqc_result_descriptor),
                                  VIBEQC_ABI_VERSION,
                                  0.0,
                                  evaluation.forces.data(),
                                  static_cast<uint32_t>(evaluation.forces.size()),
                                  0,
                                  0.0,
                                  0.0,
                                  0,
                                  VIBEQC_BACKEND_CPU_REFERENCE};
  vibeqc_result_descriptor first_energy_only{sizeof(vibeqc_result_descriptor),
                                             VIBEQC_ABI_VERSION,
                                             0.0,
                                             nullptr,
                                             0,
                                             0,
                                             0.0,
                                             0.0,
                                             0,
                                             backend};
  if (verify_energy_only) {
    require(vibeqc_calculation_execute(calculation, &first_energy_only) == VIBEQC_STATUS_SUCCESS,
            "first energy-only execution failed");
    require(first_energy_only.executed_backend == backend,
            "energy-only execution used an unexpected backend");
  }
  const vibeqc_status status = vibeqc_calculation_execute(calculation, &result);
  require(status == VIBEQC_STATUS_SUCCESS, "RHF execution failed");
  require(result.converged == 1, "RHF did not report convergence");
  evaluation.energy = result.energy;

  if (verify_energy_only) {
    vibeqc_result_descriptor energy_only{
        sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
        VIBEQC_BACKEND_CPU_REFERENCE};
    require(vibeqc_calculation_execute(calculation, &energy_only) == VIBEQC_STATUS_SUCCESS,
            "energy-only execution failed");
    const double tolerance = backend == VIBEQC_BACKEND_CUDA ? 2.0e-9 : 1.0e-14;
    require(std::abs(energy_only.energy - evaluation.energy) < tolerance &&
                std::abs(first_energy_only.energy - evaluation.energy) < tolerance,
            "omitting force storage changed the energy");
    const auto expected_forces = evaluation.forces;
    evaluation.forces.fill(NAN);
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS,
            "force execution after energy-only replay failed");
    for (std::size_t i = 0; i < expected_forces.size(); ++i) {
      require(std::isfinite(evaluation.forces[i]) &&
                  std::abs(evaluation.forces[i] - expected_forces[i]) < tolerance,
              "energy-only replay changed or suppressed later forces");
    }
  }

  vibeqc_calculation_destroy(calculation);
  vibeqc_system_destroy(system);
  vibeqc_context_destroy(context);
  return evaluation;
}

/** A single-atom RHF preparation with one s-shell of \p primitive_count primitives. */
struct PreparedSingleAtom {
  vibeqc_context* context;
  vibeqc_system* system;
  vibeqc_calculation* calculation;
};

PreparedSingleAtom prepare_single_atom_rhf(int atomic_number, std::size_t primitive_count) {
  vibeqc_context_descriptor context_descriptor{sizeof(vibeqc_context_descriptor),
                                               VIBEQC_ABI_VERSION, 0, VIBEQC_BACKEND_CPU_REFERENCE};
  vibeqc_context* context = nullptr;
  require(vibeqc_context_create(&context_descriptor, &context) == VIBEQC_STATUS_SUCCESS,
          "single-atom context creation failed");

  const std::array<vibeqc_atom, 1> atoms{{{atomic_number, 0.0, 0.0, 0.0}}};
  const vibeqc_primitive primitives[4] = {{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}};
  const std::array<vibeqc_shell, 1> shells{{{0, 0, 0, static_cast<uint32_t>(primitive_count)}}};
  vibeqc_system_descriptor system_descriptor{sizeof(vibeqc_system_descriptor),
                                             VIBEQC_ABI_VERSION,
                                             atoms.data(),
                                             static_cast<uint32_t>(atoms.size()),
                                             shells.data(),
                                             static_cast<uint32_t>(shells.size()),
                                             primitives,
                                             static_cast<uint32_t>(primitive_count),
                                             0,
                                             1};
  vibeqc_system* system = nullptr;
  require(vibeqc_system_create(context, &system_descriptor, &system) == VIBEQC_STATUS_SUCCESS,
          "single-atom system creation failed");

  vibeqc_method_descriptor method{sizeof(vibeqc_method_descriptor),
                                  VIBEQC_ABI_VERSION,
                                  VIBEQC_METHOD_RHF,
                                  100,
                                  8,
                                  1.0e-12,
                                  1.0e-10,
                                  1.0e-14};
  method.density_fitting_mode = VIBEQC_DENSITY_FITTING_NONE;
  method.density_fitting_memory_budget_bytes = 0;
  method.precision_mode = VIBEQC_PRECISION_FP64;
  vibeqc_calculation* calculation = nullptr;
  require(
      vibeqc_calculation_prepare(context, system, &method, &calculation) == VIBEQC_STATUS_SUCCESS,
      "single-atom calculation preparation failed");

  return {context, system, calculation};
}

/**
 * The provenance getter must report availability honestly. Both the availability
 * query (a NULL \p out) and the copy-out are gated on whether a completed
 * execution has populated the record: UNAVAILABLE before a run and after one that
 * threw, SUCCESS after a normal return. This pins the \p precision_available gate
 * so a stale record can never be serialized from a failed or not-yet-run run.
 */
void verify_precision_provenance_gate() {
  // He: Z=2, two s primitives -> two AOs, one occupied pair, converges.
  {
    const PreparedSingleAtom he = prepare_single_atom_rhf(2, 2);
    vibeqc_precision_provenance prov{sizeof(vibeqc_precision_provenance), VIBEQC_ABI_VERSION};
    require(vibeqc_calculation_get_precision_provenance(he.calculation, &prov) ==
                VIBEQC_STATUS_PRECISION_UNAVAILABLE,
            "provenance must be unavailable before any execution");
    require(vibeqc_calculation_get_precision_provenance(he.calculation, nullptr) ==
                VIBEQC_STATUS_PRECISION_UNAVAILABLE,
            "availability query must be unavailable before any execution");

    double forces[3] = {0.0, 0.0, 0.0};
    vibeqc_result_descriptor result{
        sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, forces, 3, 0, 0.0, 0.0, 0,
        VIBEQC_BACKEND_CPU_REFERENCE};
    const vibeqc_status executed = vibeqc_calculation_execute(he.calculation, &result);
    require(executed == VIBEQC_STATUS_SUCCESS && result.converged == 1,
            "He RHF reference run did not converge");
    require(
        vibeqc_calculation_get_precision_provenance(he.calculation, &prov) == VIBEQC_STATUS_SUCCESS,
        "provenance must be available after a completed run");
    require(prov.requested_mode == VIBEQC_PRECISION_FP64,
            "completed run reports the requested fp64 policy");
    require(vibeqc_calculation_get_precision_provenance(he.calculation, nullptr) ==
                VIBEQC_STATUS_SUCCESS,
            "availability query must be available after a completed run");
    // Both descriptor fields are part of the contract: an exactly sized struct
    // that advertises a foreign ABI must be rejected and left untouched rather
    // than filled with the current layout.
    vibeqc_precision_provenance foreign_abi{sizeof(vibeqc_precision_provenance),
                                            VIBEQC_ABI_VERSION + 1U};
    foreign_abi.policy_version = 4242U;
    foreign_abi.requested_mode = 4242;
    require(vibeqc_calculation_get_precision_provenance(he.calculation, &foreign_abi) ==
                VIBEQC_STATUS_ABI_MISMATCH,
            "a foreign abi_version must be rejected");
    require(foreign_abi.policy_version == 4242U && foreign_abi.requested_mode == 4242 &&
                foreign_abi.struct_size == sizeof(vibeqc_precision_provenance) &&
                foreign_abi.mixed_precision_reserved_error == 0.0 &&
                foreign_abi.refinement_iterations == 0,
            "a rejected descriptor must not be modified");
    vibeqc_precision_provenance short_size{sizeof(vibeqc_precision_provenance) - 1U,
                                           VIBEQC_ABI_VERSION};
    require(vibeqc_calculation_get_precision_provenance(he.calculation, &short_size) ==
                VIBEQC_STATUS_ABI_MISMATCH,
            "a short descriptor must be rejected");
    vibeqc_calculation_destroy(he.calculation);
    vibeqc_system_destroy(he.system);
    vibeqc_context_destroy(he.context);
  }
  // Be: Z=4, one s primitive -> one AO but two occupied pairs, so the cold
  // host plan throws after prepare succeeds ("basis has fewer orbitals than
  // occupied electron pairs"). A throw must reset the record back to
  // unavailable rather than leak the previous run's provenance.
  {
    const PreparedSingleAtom be = prepare_single_atom_rhf(4, 1);
    vibeqc_precision_provenance prov{sizeof(vibeqc_precision_provenance), VIBEQC_ABI_VERSION};
    require(vibeqc_calculation_get_precision_provenance(be.calculation, &prov) ==
                VIBEQC_STATUS_PRECISION_UNAVAILABLE,
            "provenance must be unavailable before a failing execution");

    double forces[3] = {0.0, 0.0, 0.0};
    vibeqc_result_descriptor result{
        sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, forces, 3, 0, 0.0, 0.0, 0,
        VIBEQC_BACKEND_CPU_REFERENCE};
    const vibeqc_status executed = vibeqc_calculation_execute(be.calculation, &result);
    require(executed != VIBEQC_STATUS_SUCCESS, "Be/1s RHF must fail to converge");
    require(vibeqc_calculation_get_precision_provenance(be.calculation, &prov) ==
                VIBEQC_STATUS_PRECISION_UNAVAILABLE,
            "provenance must fall back to unavailable after a failed execution");
    require(vibeqc_calculation_get_precision_provenance(be.calculation, nullptr) ==
                VIBEQC_STATUS_PRECISION_UNAVAILABLE,
            "availability query must fall back to unavailable after a failed execution");
    vibeqc_calculation_destroy(be.calculation);
    vibeqc_system_destroy(be.system);
    vibeqc_context_destroy(be.context);
  }
}

}  // namespace

int main() {
  try {
    int available = -1;
    require(vibeqc_method_available(VIBEQC_METHOD_RHF, &available) == VIBEQC_STATUS_SUCCESS &&
                available == 1,
            "RHF capability query failed");
    require(vibeqc_method_available(VIBEQC_METHOD_UHF, &available) == VIBEQC_STATUS_SUCCESS &&
                available == 1,
            "UHF capability query failed");
    require(vibeqc_method_available(VIBEQC_METHOD_WB97M_V, &available) == VIBEQC_STATUS_SUCCESS &&
                available == 0,
            "wB97M-V must remain explicitly unavailable");

    verify_precision_provenance_gate();

    vibeqc_method_capabilities_descriptor capabilities{
        sizeof(vibeqc_method_capabilities_descriptor), VIBEQC_ABI_VERSION, 0, 0, 0, 0, 0};
    require(
        vibeqc_method_get_capabilities(VIBEQC_METHOD_RHF, &capabilities) == VIBEQC_STATUS_SUCCESS,
        "RHF detailed capability query failed");
    require(
        capabilities.family == VIBEQC_METHOD_FAMILY_HARTREE_FOCK && capabilities.available == 1 &&
            capabilities.supports_batch == 1 &&
            capabilities.supported_properties == (VIBEQC_PROPERTY_ENERGY | VIBEQC_PROPERTY_FORCES),
        "RHF detailed capabilities are incorrect");
    require(vibeqc_method_get_capabilities(VIBEQC_METHOD_RCCSD_T, &capabilities) ==
                    VIBEQC_STATUS_SUCCESS &&
                capabilities.family == VIBEQC_METHOD_FAMILY_COUPLED_CLUSTER &&
                capabilities.available == 0,
            "RCCSD(T) reserved capabilities are incorrect");

    const Evaluation center = h2(1.4, true);
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

    for (bool unrestricted : {false, true}) {
      h2(1.4, true, VIBEQC_BACKEND_CPU_REFERENCE, VIBEQC_DENSITY_FITTING_CPU_REFERENCE,
         unrestricted);
    }
#if VIBEQC_HAS_CUDA
    vibeqc_context_descriptor probe{sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, 0,
                                    VIBEQC_BACKEND_CUDA};
    vibeqc_context* cuda_context = nullptr;
    if (vibeqc_context_create(&probe, &cuda_context) == VIBEQC_STATUS_SUCCESS) {
      vibeqc_context_destroy(cuda_context);
      for (bool unrestricted : {false, true}) {
        for (std::uint64_t budget : {0ULL, 8ULL * 1024ULL * 1024ULL}) {
          h2(1.4, true, VIBEQC_BACKEND_CUDA, VIBEQC_DENSITY_FITTING_CUDA, unrestricted, budget);
        }
      }
    }
#endif

    std::cout << "H2 energy: " << center.energy << '\n';
    std::cout << "H2 force z(atom 1): " << center.forces[5] << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

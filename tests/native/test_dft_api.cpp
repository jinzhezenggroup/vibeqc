#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "vibeqc/vibeqc.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Fixture {
  vibeqc_context* context{};
  vibeqc_system* system{};

  explicit Fixture(vibeqc_backend backend = VIBEQC_BACKEND_CPU_REFERENCE, int charge = 0,
                   std::uint32_t multiplicity = 1) {
    vibeqc_context_descriptor context_descriptor{sizeof(vibeqc_context_descriptor),
                                                 VIBEQC_ABI_VERSION, 0, backend};
    require(vibeqc_context_create(&context_descriptor, &context) == VIBEQC_STATUS_SUCCESS,
            "DFT context creation failed");
    system = create_system(context, charge, multiplicity);
  }

  static vibeqc_system* create_system(vibeqc_context* context, int charge = 0,
                                      std::uint32_t multiplicity = 1) {
    const std::array<vibeqc_atom, 2> atoms{{
        {1, 0.0, 0.0, -0.7},
        {1, 0.0, 0.0, 0.7},
    }};
    const std::array<vibeqc_primitive, 6> primitives{{
        {3.425250914, 0.1543289673},
        {0.6239137298, 0.5353281423},
        {0.168855404, 0.4446345422},
        {3.425250914, 0.1543289673},
        {0.6239137298, 0.5353281423},
        {0.168855404, 0.4446345422},
    }};
    const std::array<vibeqc_shell, 2> shells{{{0, 0, 0, 3}, {1, 0, 3, 3}}};
    vibeqc_system_descriptor descriptor{sizeof(vibeqc_system_descriptor),
                                        VIBEQC_ABI_VERSION,
                                        atoms.data(),
                                        static_cast<uint32_t>(atoms.size()),
                                        shells.data(),
                                        static_cast<uint32_t>(shells.size()),
                                        primitives.data(),
                                        static_cast<uint32_t>(primitives.size()),
                                        charge,
                                        multiplicity,
                                        VIBEQC_BASIS_CARTESIAN};
    vibeqc_system* created = nullptr;
    require(vibeqc_system_create(context, &descriptor, &created) == VIBEQC_STATUS_SUCCESS,
            "DFT system creation failed");
    return created;
  }

  ~Fixture() {
    vibeqc_system_destroy(system);
    vibeqc_context_destroy(context);
  }
};

vibeqc_method_descriptor lda_method() {
  return {sizeof(vibeqc_method_descriptor),
          VIBEQC_ABI_VERSION,
          VIBEQC_METHOD_LDA_RKS,
          200,
          8,
          1.0e-12,
          1.0e-10,
          1.0e-12,
          VIBEQC_DENSITY_FITTING_NONE,
          nullptr,
          1.0e-10,
          0};
}

}  // namespace

int main() {
  try {
    vibeqc_method_capabilities_descriptor capabilities{
        sizeof(vibeqc_method_capabilities_descriptor), VIBEQC_ABI_VERSION, 0, 0, 0, 0, 0};
    require(vibeqc_method_get_capabilities(VIBEQC_METHOD_LDA_RKS, &capabilities) ==
                VIBEQC_STATUS_SUCCESS,
            "LDA RKS capability query failed");
    require(capabilities.family == VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL &&
                capabilities.supported_properties == VIBEQC_PROPERTY_ENERGY &&
                capabilities.available == 1 && capabilities.supports_batch == 0,
            "LDA RKS capabilities are incorrect");
    require(vibeqc_method_get_capabilities(VIBEQC_METHOD_PBE_RKS, &capabilities) ==
                    VIBEQC_STATUS_SUCCESS &&
                capabilities.family == VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL &&
                capabilities.supported_properties == VIBEQC_PROPERTY_ENERGY &&
                capabilities.available == 1 && capabilities.supports_batch == 0,
            "PBE RKS capabilities are incorrect");
    for (vibeqc_method method : {VIBEQC_METHOD_LDA_UKS, VIBEQC_METHOD_PBE_UKS}) {
      require(vibeqc_method_get_capabilities(method, &capabilities) == VIBEQC_STATUS_SUCCESS &&
                  capabilities.family == VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL &&
                  capabilities.available == 0,
              "reserved DFT method capabilities are incorrect");
    }

    Fixture fixture;
    auto method = lda_method();
    vibeqc_calculation* calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_SUCCESS,
            "LDA RKS preparation failed");
    vibeqc_result_descriptor result{
        sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
        VIBEQC_BACKEND_CPU_REFERENCE};
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
                result.converged == 1 && std::isfinite(result.energy) &&
                result.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE,
            "LDA RKS energy-only execution failed");
    require(std::abs(result.energy - (-1.121017859421488)) < 2.0e-12,
            "LDA RKS H2 regression energy changed");

    std::array<double, 6> forces{};
    result.forces = forces.data();
    result.force_count = static_cast<uint32_t>(forces.size());
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_NOT_IMPLEMENTED,
            "LDA RKS force request was not rejected");
    const char* detail = vibeqc_context_get_last_detail(fixture.context);
    require(detail != nullptr && std::string(detail).find("issue #163") != std::string::npos,
            "LDA RKS force rejection omitted its capability boundary");
    vibeqc_calculation_destroy(calculation);

    method = lda_method();
    method.method = VIBEQC_METHOD_PBE_RKS;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_SUCCESS,
            "PBE RKS preparation failed");
    result = {sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
              VIBEQC_BACKEND_CPU_REFERENCE};
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
                result.converged == 1 && std::isfinite(result.energy) &&
                result.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE,
            "PBE RKS energy-only execution failed");
    require(std::abs(result.energy - (-1.1520643753396715)) < 2.0e-12,
            "PBE RKS H2 implementation regression energy changed");
    std::cout << std::setprecision(17) << "PBE RKS H2 energy: " << result.energy << "\n";
    vibeqc_calculation_destroy(calculation);

    method.density_fitting_mode = VIBEQC_DENSITY_FITTING_CPU_REFERENCE;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_NOT_IMPLEMENTED,
            "LDA RKS accepted density fitting");

    for (const auto [charge, multiplicity] :
         {std::pair{1, std::uint32_t{2}}, std::pair{0, std::uint32_t{3}}}) {
      Fixture invalid_spin(VIBEQC_BACKEND_CPU_REFERENCE, charge, multiplicity);
      method = lda_method();
      calculation = nullptr;
      require(vibeqc_calculation_prepare(invalid_spin.context, invalid_spin.system, &method,
                                         &calculation) == VIBEQC_STATUS_INVALID_ARGUMENT &&
                  calculation == nullptr,
              "LDA RKS accepted an odd-electron or non-singlet system");
      detail = vibeqc_context_get_last_detail(invalid_spin.context);
      require(detail != nullptr && std::string(detail).find("multiplicity 1") != std::string::npos,
              "LDA RKS spin rejection omitted its closed-shell boundary");
    }

    for (vibeqc_method reserved : {VIBEQC_METHOD_LDA_UKS, VIBEQC_METHOD_PBE_UKS}) {
      method = lda_method();
      method.method = reserved;
      calculation = nullptr;
      require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                      VIBEQC_STATUS_NOT_IMPLEMENTED &&
                  calculation == nullptr,
              "reserved DFT method reached prepared execution");
    }

    method = lda_method();
    method.max_iterations = 1;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_SUCCESS,
            "one-iteration LDA RKS preparation failed");
    vibeqc_result_descriptor unconverged{
        sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
        VIBEQC_BACKEND_CPU_REFERENCE};
    require(vibeqc_calculation_execute(calculation, &unconverged) == VIBEQC_STATUS_NOT_CONVERGED &&
                unconverged.converged == 0 && unconverged.iterations == 1 &&
                std::isfinite(unconverged.energy),
            "LDA RKS nonconvergence status or diagnostics are incorrect");
    vibeqc_calculation_destroy(calculation);

    vibeqc_system* systems[]{fixture.system};
    method = lda_method();
    vibeqc_batch* batch = nullptr;
    require(vibeqc_batch_prepare(fixture.context, systems, 1, &method, 0, &batch) ==
                VIBEQC_STATUS_NOT_IMPLEMENTED,
            "LDA RKS accepted prepared batch execution");

#if VIBEQC_HAS_CUDA
    vibeqc_context_descriptor cuda_descriptor{sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION,
                                              0, VIBEQC_BACKEND_CUDA};
    vibeqc_context* cuda_context = nullptr;
    if (vibeqc_context_create(&cuda_descriptor, &cuda_context) == VIBEQC_STATUS_SUCCESS) {
      vibeqc_system* cuda_system = Fixture::create_system(cuda_context);
      vibeqc_calculation* cuda_calculation = nullptr;
      require(vibeqc_calculation_prepare(cuda_context, cuda_system, &method, &cuda_calculation) ==
                  VIBEQC_STATUS_NOT_IMPLEMENTED,
              "LDA RKS accepted the CUDA backend");
      vibeqc_system_destroy(cuda_system);
      vibeqc_context_destroy(cuda_context);
    }
#endif
    std::cout << "LDA RKS public CPU energy-only contract passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "vibeqc/vibeqc.h"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if VIBEQC_HAS_CUDA
extern "C" void ks_cuda_fail_next_runtime_for_test_v1();
extern "C" void xc_cuda_fail_next_runtime_for_test_v1();
extern "C" void xc_cuda_fail_next_allocation_for_test_v1();
#endif

namespace {
// Fail exactly one grid-coordinate allocation after warm-state import. This
// executable-only interposition exercises real constructor unwinding without
// adding fault controls to the production API or exhausting machine memory.
thread_local std::size_t fail_allocation_bytes = 0;

}  // namespace

void* operator new(std::size_t bytes) {
  if (bytes && bytes == fail_allocation_bytes) {
    fail_allocation_bytes = 0;
    throw std::bad_alloc();
  }
  if (void* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

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

void ks_short_method_descriptor_rejected() {
#if defined(__unix__) || defined(__APPLE__)
  // Validation must reject a short allocation from its advertised size before
  // any current-layout suffix is read.
  const auto page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  require(page_size > sizeof(vibeqc_method_descriptor), "invalid system page size");
  struct Pages {
    void* address;
    std::size_t bytes;
    ~Pages() { munmap(address, bytes); }
  };
  void* address =
      mmap(nullptr, 2 * page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  require(address != MAP_FAILED, "cannot allocate guarded descriptor pages");
  Pages pages{address, 2 * page_size};
  auto* boundary = static_cast<unsigned char*>(address) + page_size;
  require(mprotect(boundary, page_size, PROT_NONE) == 0,
          "cannot protect the absent descriptor suffix");
  Fixture fixture;
  for (const auto size : {offsetof(vibeqc_method_descriptor, density_fitting_mode),
                          offsetof(vibeqc_method_descriptor, precision_mode)}) {
    auto method = lda_method();
    method.struct_size = static_cast<std::uint32_t>(size);
    auto* truncated = reinterpret_cast<vibeqc_method_descriptor*>(boundary - size);
    std::memcpy(truncated, &method, size);
    vibeqc_calculation* calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, truncated, &calculation) ==
                VIBEQC_STATUS_ABI_MISMATCH,
            "short method descriptor was accepted");
    vibeqc_system* systems[]{fixture.system};
    vibeqc_batch* batch = nullptr;
    require(vibeqc_batch_prepare(fixture.context, systems, 1, truncated, 0, &batch) ==
                VIBEQC_STATUS_ABI_MISMATCH,
            "short batch method descriptor was accepted");
  }
#endif
}

void ks_option_snapshot() {
  require(vibeqc_ks_options_version() == 1, "semantic KS option ABI unavailable");
  Fixture fixture;
  auto method = lda_method();
  std::array<double, 119> radii;
  radii.fill(1.0);
  radii[1] = 1.3;
  const std::array<vibeqc_ks_semilocal_component, 2> components{
      {{"LDA_X", 1.0}, {"LDA_C_PW", 1.0}}};
  vibeqc_ks_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = VIBEQC_ABI_VERSION;
  options.scf_domain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
  options.grid_version = 1;
  options.radial_points = 32;
  options.angular_polar = 10;
  options.angular_azimuth = 20;
  options.partition_iterations = 2;
  options.coincident_tolerance = 1e-12;
  options.tile_points = 31;
  options.element_radii = radii.data();
  options.element_radius_count = radii.size();
  options.xc_execution_schedule = VIBEQC_XC_EXECUTION_DEVICE_FUSED;
  options.spin_channels = 1;
  options.semilocal_components = components.data();
  options.semilocal_component_count = components.size();
  method.ks_options = &options;
  vibeqc_calculation* calculation = nullptr;
  require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
              VIBEQC_STATUS_SUCCESS,
          "custom semantic KS preparation failed");
  options.radial_points = 0;
  radii[1] = std::numeric_limits<double>::quiet_NaN();
  vibeqc_result_descriptor result{
      sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0, nullptr, 0, 0, 0, 0, 0,
      VIBEQC_BACKEND_CPU_REFERENCE};
  require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS,
          "KS consumed caller options after prepare");
  const double energy = result.energy;
  require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
              std::abs(result.energy - energy) < 1e-11,
          "KS snapshot replay changed");
  vibeqc_calculation_destroy(calculation);
  options.struct_size = 8;
  calculation = nullptr;
  require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
              VIBEQC_STATUS_ABI_MISMATCH,
          "truncated semantic KS snapshot accepted");
  options.struct_size = sizeof(options);
  options.scf_domain = "not-a-qualified-domain";
  require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
              VIBEQC_STATUS_NOT_IMPLEMENTED,
          "unknown KS domain policy accepted");
  method.method = VIBEQC_METHOD_RHF;
  options.scf_domain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
  require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
              VIBEQC_STATUS_INVALID_ARGUMENT,
          "HF ignored a KS model option");
}

void ks_option_semantic_plan() {
  Fixture fixture;
  auto method = lda_method();
  method.method = VIBEQC_METHOD_PBE_RKS;
  const std::array<vibeqc_ks_semilocal_component, 2> components{
      {{"GGA_X_PBE", 1.0}, {"GGA_C_PBE", 1.0}}};
  std::array<vibeqc_ks_exchange_term, 2> exchange{{
      {VIBEQC_KS_EXCHANGE_SHORT_RANGE, 0.2, 0.3, -0.1},
      {VIBEQC_KS_EXCHANGE_LONG_RANGE, 0.8, 0.3, -0.4},
  }};
  vibeqc_ks_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = VIBEQC_ABI_VERSION;
  options.scf_domain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
  options.grid_version = 1;
  options.radial_points = 32;
  options.angular_polar = 10;
  options.angular_azimuth = 20;
  options.partition_iterations = 2;
  options.coincident_tolerance = 1e-12;
  options.tile_points = 31;
  options.xc_execution_schedule = VIBEQC_XC_EXECUTION_DEVICE_FUSED;
  options.spin_channels = 1;
  options.semilocal_components = components.data();
  options.semilocal_component_count = components.size();
  options.exchange_terms = exchange.data();
  options.exchange_term_count = exchange.size();
  method.ks_options = &options;
  vibeqc_calculation* calculation = nullptr;
  require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
              VIBEQC_STATUS_SUCCESS,
          "semantic range exchange did not attach the native two-Fock consumer");
  vibeqc_result_descriptor result{
      sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0, nullptr, 0, 0, 0, 0, 0,
      VIBEQC_BACKEND_CPU_REFERENCE};
  require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
              std::isfinite(result.energy),
          "semantic range exchange failed through the native two-Fock endpoint");
  vibeqc_calculation_destroy(calculation);
  calculation = nullptr;
  exchange[0].fock_coefficient = -0.2;
  require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
              VIBEQC_STATUS_INVALID_ARGUMENT,
          "semantic range exchange accepted an inconsistent K coefficient");
}

void pbe0_composition_snapshot() {
  Fixture fixture;
  auto method = lda_method();
  method.method = VIBEQC_METHOD_PBE0_RKS;
  const std::array<vibeqc_ks_semilocal_component, 2> components{
      {{"GGA_C_PBE", 1.0}, {"GGA_X_PBE", 0.75}}};
  std::array<vibeqc_ks_exchange_term, 1> exchange{
      {{VIBEQC_KS_EXCHANGE_FULL_RANGE, 0.25, 0.0, -0.125}}};
  vibeqc_ks_options options{};
  options.struct_size = sizeof(options);
  options.abi_version = VIBEQC_ABI_VERSION;
  options.scf_domain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
  options.grid_version = 1;
  options.radial_points = 64;
  options.angular_polar = 12;
  options.angular_azimuth = 24;
  options.partition_iterations = 3;
  options.coincident_tolerance = 1e-12;
  options.tile_points = 256;
  options.xc_execution_schedule = VIBEQC_XC_EXECUTION_DEVICE_FUSED;
  options.spin_channels = 1;
  options.semilocal_components = components.data();
  options.semilocal_component_count = components.size();
  options.exchange_terms = exchange.data();
  options.exchange_term_count = exchange.size();
  method.ks_options = &options;

  vibeqc_calculation* calculation = nullptr;
  require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
              VIBEQC_STATUS_SUCCESS,
          "PBE0 RKS semantic composition preparation failed");
  vibeqc_result_descriptor result{
      sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
      VIBEQC_BACKEND_CPU_REFERENCE};
  require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
              result.converged && std::abs(result.energy - (-1.1543107969377155)) < 2e-12,
          "PBE0 RKS semantic composition energy changed");
  vibeqc_calculation_destroy(calculation);

  method.ks_options = nullptr;
  calculation = nullptr;
  require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
              VIBEQC_STATUS_NOT_IMPLEMENTED,
          "PBE0 RKS silently inferred composition without compiler plan");
  method.ks_options = &options;
  exchange[0].fock_coefficient = -0.25;
  require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
              VIBEQC_STATUS_INVALID_ARGUMENT,
          "PBE0 RKS accepted an unrestricted exchange coefficient");
}

void warm_preparation_failure(bool retained_plan) {
  Fixture fixture;
  auto method = lda_method();
  vibeqc_system* systems[]{fixture.system, fixture.system};
  vibeqc_batch *source = nullptr, *target = nullptr;
  require(vibeqc_batch_prepare(fixture.context, systems, 2, &method,
                               VIBEQC_BATCH_ENABLE_WARM_STARTS, &source) == VIBEQC_STATUS_SUCCESS,
          "warm failure source preparation failed");
  std::unique_ptr<vibeqc_batch, decltype(&vibeqc_batch_destroy)> source_owner(
      source, &vibeqc_batch_destroy);
  require(vibeqc_batch_prepare(fixture.context, systems, 2, &method,
                               VIBEQC_BATCH_ENABLE_WARM_STARTS, &target) == VIBEQC_STATUS_SUCCESS,
          "warm failure target preparation failed");
  std::unique_ptr<vibeqc_batch, decltype(&vibeqc_batch_destroy)> target_owner(
      target, &vibeqc_batch_destroy);
  const std::array<double, 6> changed{0.0, 0.0, -0.9, 0.0, 0.0, 0.9};
  std::array<vibeqc_batch_input_descriptor, 2> inputs;
  for (auto& input : inputs)
    input = {sizeof(input), VIBEQC_ABI_VERSION, changed.data(), changed.size()};
  std::array<vibeqc_batch_item_result_descriptor, 2> results{};
  const auto execute = [&](vibeqc_batch* batch, bool moved) {
    for (auto& result : results) {
      result = {};
      result.struct_size = sizeof(result);
      result.abi_version = VIBEQC_ABI_VERSION;
    }
    return vibeqc_batch_execute(batch, moved ? inputs.data() : nullptr, moved ? inputs.size() : 0,
                                results.data(), results.size());
  };
  require(execute(source, true) == VIBEQC_STATUS_SUCCESS && results[0].converged,
          "warm failure source did not converge");
  const double expected = results[0].energy;
  if (retained_plan)
    require(execute(target, false) == VIBEQC_STATUS_SUCCESS && results[0].converged,
            "warm failure target's original geometry did not converge");
  std::array<std::array<double, 4>, 2> densities{};
  std::array<std::array<double, 6>, 2> coordinates{};
  std::array<vibeqc_hf_warm_state, 2> seeds{};
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    seeds[i].struct_size = sizeof(seeds[i]);
    seeds[i].abi_version = VIBEQC_ABI_VERSION;
    seeds[i].density = densities[i].data();
    seeds[i].density_count = densities[i].size();
    seeds[i].coordinates = coordinates[i].data();
    seeds[i].coordinate_count = coordinates[i].size();
    require(vibeqc_batch_get_hf_warm_state(source, i, &seeds[i]) == VIBEQC_STATUS_SUCCESS &&
                seeds[i].present,
            "warm failure seed export failed");
  }
  require(vibeqc_batch_restore_hf_warm_states(target, seeds.data(), seeds.size()) ==
              VIBEQC_STATUS_SUCCESS,
          "warm failure seed import failed");
  // H2's default quadrature has two atoms times 48*16*32 points. Its xyz
  // vector is allocated inside preparation, after the imported seed is ready.
  fail_allocation_bytes = 3 * 2 * 48 * 16 * 32 * sizeof(double);
  const auto status = execute(target, true);
  const bool injected = fail_allocation_bytes == 0;
  fail_allocation_bytes = 0;
  require(injected, "warm preparation allocation fault was not reached");
  require(status == VIBEQC_STATUS_SUCCESS && results[0].status == VIBEQC_STATUS_OUT_OF_MEMORY &&
              !results[0].converged && !results[0].warm_start_fallback,
          "failed warm preparation retried a missing or stale calculation");
  require(results[1].status == VIBEQC_STATUS_SUCCESS && results[1].warm_start_used &&
              std::abs(results[1].energy - expected) < 2e-10,
          "warm preparation failure corrupted its neighbor");
  require(execute(target, true) == VIBEQC_STATUS_SUCCESS && results[0].converged &&
              results[0].warm_start_used && std::abs(results[0].energy - expected) < 2e-10,
          "warm preparation failure lost the imported seed or target geometry");
}

void warm_execution_allocation_failure() {
  Fixture fixture;
  auto method = lda_method();
  vibeqc_system* systems[]{fixture.system, fixture.system};
  vibeqc_batch* batch = nullptr;
  require(vibeqc_batch_prepare(fixture.context, systems, 2, &method,
                               VIBEQC_BATCH_ENABLE_WARM_STARTS, &batch) == VIBEQC_STATUS_SUCCESS,
          "warm execution failure preparation failed");
  std::unique_ptr<vibeqc_batch, decltype(&vibeqc_batch_destroy)> owner(batch,
                                                                       &vibeqc_batch_destroy);
  std::array<vibeqc_batch_item_result_descriptor, 2> results{};
  const auto execute = [&] {
    for (auto& item : results) {
      item = {};
      item.struct_size = sizeof(item);
      item.abi_version = VIBEQC_ABI_VERSION;
    }
    return vibeqc_batch_execute(batch, nullptr, 0, results.data(), results.size());
  };
  require(execute() == VIBEQC_STATUS_SUCCESS && results[0].converged && results[1].converged,
          "warm execution failure source did not converge");
  const auto energy = results[0].energy;
  // Geometry and owners are already prepared. Fail the first 2x2 SCF matrix
  // allocation, then prove that a cold retry cannot hide this resource error.
  fail_allocation_bytes = 4 * sizeof(double);
  const auto status = execute();
  const bool injected = fail_allocation_bytes == 0;
  fail_allocation_bytes = 0;
  require(injected && status == VIBEQC_STATUS_SUCCESS &&
              results[0].status == VIBEQC_STATUS_OUT_OF_MEMORY && results[0].warm_start_used &&
              !results[0].warm_start_fallback && results[1].converged,
          "warm SCF allocation failure was hidden by a cold retry or affected its neighbor");
  require(execute() == VIBEQC_STATUS_SUCCESS && results[0].converged &&
              results[0].warm_start_used && std::abs(results[0].energy - energy) < 2e-10,
          "warm SCF allocation failure lost its last-good seed");
}

}  // namespace

int main() {
  try {
    ks_short_method_descriptor_rejected();
    ks_option_snapshot();
    ks_option_semantic_plan();
    pbe0_composition_snapshot();
    warm_preparation_failure(false);
    warm_preparation_failure(true);
    warm_execution_allocation_failure();
    vibeqc_method_capabilities_descriptor capabilities{
        sizeof(vibeqc_method_capabilities_descriptor), VIBEQC_ABI_VERSION, 0, 0, 0, 0, 0};
    for (vibeqc_method registered :
         {VIBEQC_METHOD_LDA_RKS, VIBEQC_METHOD_PBE_RKS, VIBEQC_METHOD_R2SCAN_RKS,
          VIBEQC_METHOD_LDA_UKS, VIBEQC_METHOD_PBE_UKS, VIBEQC_METHOD_R2SCAN_UKS,
          VIBEQC_METHOD_PBE0_RKS, VIBEQC_METHOD_PBE0_UKS, VIBEQC_METHOD_PBE_D4_RKS}) {
      require(vibeqc_method_get_capabilities(registered, &capabilities) == VIBEQC_STATUS_SUCCESS &&
                  capabilities.family == VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL &&
                  capabilities.supported_properties == VIBEQC_PROPERTY_ENERGY &&
                  capabilities.available == 1 && capabilities.supports_batch == 1,
              "registered KS capabilities are incorrect");
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
    const double pbe_energy = result.energy;
    std::cout << std::setprecision(17) << "PBE RKS H2 energy: " << result.energy << "\n";
    vibeqc_calculation_destroy(calculation);

    const std::array<std::int32_t, 2> d4_z{1, 1};
    const std::array<double, 6> d4_xyz{0.0, 0.0, -0.7, 0.0, 0.0, 0.7};
    vibeqc_d4_system_descriptor d4_system{sizeof(vibeqc_d4_system_descriptor),
                                          VIBEQC_ABI_VERSION,
                                          d4_z.data(),
                                          d4_xyz.data(),
                                          static_cast<std::uint32_t>(d4_z.size()),
                                          0.0};
    vibeqc_d4_bj_eeq_descriptor d4_model{sizeof(vibeqc_d4_bj_eeq_descriptor),
                                         VIBEQC_ABI_VERSION,
                                         VIBEQC_D4_PROFILE_STANDARD_EEQ,
                                         1.0,
                                         0.95948085,
                                         1.0,
                                         0.38574991,
                                         4.80688534,
                                         3.0,
                                         2.0,
                                         30.0,
                                         60.0,
                                         40.0,
                                         64u << 20};
    {
      auto inaccessible = d4_system;
      inaccessible.atomic_numbers =
          reinterpret_cast<const std::int32_t*>(static_cast<std::uintptr_t>(1));
      inaccessible.coordinates = reinterpret_cast<const double*>(static_cast<std::uintptr_t>(1));
      vibeqc_d4_batch* rejected =
          reinterpret_cast<vibeqc_d4_batch*>(static_cast<std::uintptr_t>(1));

      auto invalid_budget = d4_model;
      invalid_budget.maximum_bytes = 0;
      require(vibeqc_d4_batch_prepare(fixture.context, &inaccessible, 1, &invalid_budget,
                                      &rejected) == VIBEQC_STATUS_INVALID_ARGUMENT &&
                  rejected == nullptr,
              "D4 zero-budget admission touched caller arrays or published a batch");

      rejected = reinterpret_cast<vibeqc_d4_batch*>(static_cast<std::uintptr_t>(1));
      invalid_budget.maximum_bytes = 1;
      require(vibeqc_d4_batch_prepare(fixture.context, &inaccessible, 1, &invalid_budget,
                                      &rejected) == VIBEQC_STATUS_OUT_OF_MEMORY &&
                  rejected == nullptr,
              "D4 impossible-budget admission touched caller arrays or published a batch");

      rejected = reinterpret_cast<vibeqc_d4_batch*>(static_cast<std::uintptr_t>(1));
      auto oversized = inaccessible;
      oversized.atom_count = 257;
      require(vibeqc_d4_batch_prepare(fixture.context, &oversized, 1, &d4_model, &rejected) ==
                      VIBEQC_STATUS_NOT_IMPLEMENTED &&
                  rejected == nullptr,
              "D4 oversized-system admission touched caller arrays or published a batch");
    }
    vibeqc_d4_batch* d4_batch = nullptr;
    require(vibeqc_d4_batch_prepare(fixture.context, &d4_system, 1, &d4_model, &d4_batch) ==
                VIBEQC_STATUS_SUCCESS,
            "standalone public D4 preparation failed");
    vibeqc_d4_batch_item_result_descriptor d4_result{};
    d4_result.struct_size = sizeof(d4_result);
    d4_result.abi_version = VIBEQC_ABI_VERSION;
    require(vibeqc_d4_batch_execute(d4_batch, nullptr, 0, &d4_result, 1) == VIBEQC_STATUS_SUCCESS &&
                d4_result.status == VIBEQC_STATUS_SUCCESS && std::isfinite(d4_result.energy),
            "standalone public D4 execution failed");
    const double d4_energy = d4_result.energy;
    vibeqc_d4_batch_destroy(d4_batch);

    method = lda_method();
    method.method = VIBEQC_METHOD_PBE_D4_RKS;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_SUCCESS,
            "PBE-D4 RKS preparation failed");
    result = {sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
              VIBEQC_BACKEND_CPU_REFERENCE};
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
                result.converged == 1 && std::isfinite(result.energy) &&
                std::abs(result.energy - (pbe_energy + d4_energy)) < 2e-12,
            "PBE-D4 public named method did not add the production D4 correction");
    result.forces = forces.data();
    result.force_count = static_cast<uint32_t>(forces.size());
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_NOT_IMPLEMENTED,
            "PBE-D4 silently widened the public force capability");
    detail = vibeqc_context_get_last_detail(fixture.context);
    require(detail != nullptr && std::string(detail).find("issue #163") != std::string::npos,
            "PBE-D4 force rejection omitted the PBE stationary-gradient boundary");
    vibeqc_calculation_destroy(calculation);

    method = lda_method();
    method.method = VIBEQC_METHOD_R2SCAN_RKS;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_SUCCESS,
            "r2SCAN RKS preparation failed");
    result = {sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
              VIBEQC_BACKEND_CPU_REFERENCE};
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
                result.converged == 1 && std::isfinite(result.energy) &&
                result.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE,
            "r2SCAN RKS energy-only execution failed");
    const double r2scan_rks_energy = result.energy;
    result.forces = forces.data();
    result.force_count = static_cast<uint32_t>(forces.size());
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_NOT_IMPLEMENTED,
            "r2SCAN RKS force request was not rejected");
    detail = vibeqc_context_get_last_detail(fixture.context);
    require(detail != nullptr && std::string(detail).find("issue #164") != std::string::npos,
            "r2SCAN force rejection omitted its #164 capability boundary");
    vibeqc_calculation_destroy(calculation);

    method = lda_method();
    method.density_fitting_mode = VIBEQC_DENSITY_FITTING_CPU_REFERENCE;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_SUCCESS,
            "LDA RKS density-fitting preparation failed");
    result.forces = nullptr;
    result.force_count = 0;
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
                result.converged == 1 && std::isfinite(result.energy),
            "LDA RKS density-fitted energy failed");
    result.forces = forces.data();
    result.force_count = static_cast<uint32_t>(forces.size());
    require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_NOT_IMPLEMENTED,
            "DFT density fitting exposed an incomplete force consumer");
    vibeqc_calculation_destroy(calculation);

    method = lda_method();
    method.density_fitting_mode = static_cast<vibeqc_density_fitting_mode>(99);
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "LDA RKS misclassified an unknown density-fitting mode");

    for (double vibeqc_method_descriptor::* field : {
             &vibeqc_method_descriptor::energy_tolerance,
             &vibeqc_method_descriptor::density_tolerance,
             &vibeqc_method_descriptor::screening_tolerance,
         }) {
      method = lda_method();
      method.*field = std::numeric_limits<double>::quiet_NaN();
      require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                  VIBEQC_STATUS_INVALID_ARGUMENT,
              "LDA RKS accepted a NaN tolerance");
    }

    method = lda_method();
    method.precision_mode = VIBEQC_PRECISION_AUTO;
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_NOT_IMPLEMENTED,
            "LDA RKS silently accepted automatic precision");
    method.precision_mode = static_cast<vibeqc_precision_mode>(99);
    require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                VIBEQC_STATUS_INVALID_ARGUMENT,
            "LDA RKS misclassified an unknown precision mode");

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

    for (vibeqc_method spin_method :
         {VIBEQC_METHOD_LDA_UKS, VIBEQC_METHOD_PBE_UKS, VIBEQC_METHOD_R2SCAN_UKS}) {
      method = lda_method();
      method.method = spin_method;
      calculation = nullptr;
      require(vibeqc_calculation_prepare(fixture.context, fixture.system, &method, &calculation) ==
                      VIBEQC_STATUS_SUCCESS &&
                  calculation != nullptr,
              "UKS singlet preparation failed");
      result = {
          sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
          VIBEQC_BACKEND_CPU_REFERENCE};
      require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_SUCCESS &&
                  result.converged == 1 && result.density_rms < 1.0e-9,
              "UKS singlet execution failed");
      const double expected =
          spin_method == VIBEQC_METHOD_LDA_UKS
              ? -1.121017859421488
              : (spin_method == VIBEQC_METHOD_PBE_UKS ? -1.1520643753396715 : r2scan_rks_energy);
      require(std::abs(result.energy - expected) < 2.0e-12,
              "equal-spin UKS and RKS energies disagree");
      result.forces = forces.data();
      result.force_count = static_cast<uint32_t>(forces.size());
      require(vibeqc_calculation_execute(calculation, &result) == VIBEQC_STATUS_NOT_IMPLEMENTED,
              "UKS force request exposed an unsupported buffer");
      vibeqc_calculation_destroy(calculation);
    }

    Fixture open_shell(VIBEQC_BACKEND_CPU_REFERENCE, -1, 2);
    method = lda_method();
    method.method = VIBEQC_METHOD_LDA_UKS;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(open_shell.context, open_shell.system, &method,
                                       &calculation) == VIBEQC_STATUS_SUCCESS &&
                calculation != nullptr,
            "LDA UKS preparation rejected a valid doublet");
    result = {sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
              VIBEQC_BACKEND_CPU_REFERENCE};
    const vibeqc_status uks_status = vibeqc_calculation_execute(calculation, &result);
    if (uks_status != VIBEQC_STATUS_SUCCESS) {
      const char* uks_detail = vibeqc_context_get_last_detail(open_shell.context);
      throw std::runtime_error(std::string("LDA UKS execution failed: ") +
                               (uks_detail == nullptr ? "no detail" : uks_detail));
    }
    require(result.converged == 1 && std::isfinite(result.energy) &&
                std::isfinite(result.density_rms) && result.density_rms < 1.0e-8 &&
                result.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE,
            "LDA UKS energy-only result is invalid");
    vibeqc_calculation_destroy(calculation);

    method.method = VIBEQC_METHOD_PBE_UKS;
    calculation = nullptr;
    require(vibeqc_calculation_prepare(open_shell.context, open_shell.system, &method,
                                       &calculation) == VIBEQC_STATUS_SUCCESS &&
                calculation != nullptr,
            "PBE UKS preparation rejected a valid doublet");
    result = {sizeof(vibeqc_result_descriptor), VIBEQC_ABI_VERSION, 0.0, nullptr, 0, 0, 0.0, 0.0, 0,
              VIBEQC_BACKEND_CPU_REFERENCE};
    const vibeqc_status pbe_uks_status = vibeqc_calculation_execute(calculation, &result);
    if (pbe_uks_status != VIBEQC_STATUS_SUCCESS) {
      const char* pbe_uks_detail = vibeqc_context_get_last_detail(open_shell.context);
      throw std::runtime_error(std::string("PBE UKS execution failed: ") +
                               (pbe_uks_detail == nullptr ? "no detail" : pbe_uks_detail));
    }
    require(result.converged == 1 && std::isfinite(result.energy) &&
                std::isfinite(result.density_rms) && result.density_rms < 1.0e-8,
            "PBE UKS energy-only result is invalid");
    vibeqc_calculation_destroy(calculation);

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
                VIBEQC_STATUS_SUCCESS,
            "LDA RKS batch preparation failed");
    vibeqc_batch_item_result_descriptor item{};
    item.struct_size = sizeof(item);
    item.abi_version = VIBEQC_ABI_VERSION;
    require(vibeqc_batch_get_scf_diagnostic(batch, 0, nullptr) == VIBEQC_STATUS_NOT_IMPLEMENTED,
            "unexecuted KS batch exposed stale diagnostics");
    require(vibeqc_batch_execute(batch, nullptr, 0, &item, 1) == VIBEQC_STATUS_SUCCESS &&
                item.status == VIBEQC_STATUS_SUCCESS && item.converged,
            "LDA RKS native batch energy failed");
    vibeqc_scf_diagnostic diagnostic{sizeof(vibeqc_scf_diagnostic), VIBEQC_ABI_VERSION};
    require(vibeqc_batch_get_scf_diagnostic(batch, 0, &diagnostic) == VIBEQC_STATUS_SUCCESS &&
                diagnostic.physical_residual_rms < 1e-9 &&
                diagnostic.density_rms == item.density_rms,
            "KS batch physical diagnostic is missing or inconsistent");
    item.forces = forces.data();
    item.force_count = forces.size();
    require(vibeqc_batch_execute(batch, nullptr, 0, &item, 1) == VIBEQC_STATUS_NOT_IMPLEMENTED &&
                vibeqc_batch_get_scf_diagnostic(batch, 0, nullptr) == VIBEQC_STATUS_NOT_IMPLEMENTED,
            "KS batch force rejection retained stale diagnostics");
    vibeqc_batch_destroy(batch);

#if VIBEQC_HAS_CUDA
    vibeqc_context_descriptor cuda_descriptor{sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION,
                                              0, VIBEQC_BACKEND_CUDA};
    vibeqc_context* cuda_context = nullptr;
    if (vibeqc_context_create(&cuda_descriptor, &cuda_context) == VIBEQC_STATUS_SUCCESS) {
      for (auto ks : {VIBEQC_METHOD_LDA_RKS, VIBEQC_METHOD_PBE_RKS, VIBEQC_METHOD_R2SCAN_RKS,
                      VIBEQC_METHOD_PBE0_RKS, VIBEQC_METHOD_LDA_UKS, VIBEQC_METHOD_PBE_UKS,
                      VIBEQC_METHOD_R2SCAN_UKS, VIBEQC_METHOD_PBE0_UKS}) {
        const bool uks = ks == VIBEQC_METHOD_LDA_UKS || ks == VIBEQC_METHOD_PBE_UKS ||
                         ks == VIBEQC_METHOD_R2SCAN_UKS || ks == VIBEQC_METHOD_PBE0_UKS;
        const bool pbe0 = ks == VIBEQC_METHOD_PBE0_RKS || ks == VIBEQC_METHOD_PBE0_UKS;
        Fixture cpu_fixture(VIBEQC_BACKEND_CPU_REFERENCE, uks ? 1 : 0, uks ? 2 : 1);
        vibeqc_system* cuda_system = Fixture::create_system(cuda_context, uks ? 1 : 0, uks ? 2 : 1);
        method = lda_method();
        method.method = ks;
        const std::array<vibeqc_ks_semilocal_component, 2> pbe0_components{
            {{{"GGA_C_PBE", 1.0}, {"GGA_X_PBE", 0.75}}}};
        std::array<vibeqc_ks_exchange_term, 1> pbe0_exchange{{{
            VIBEQC_KS_EXCHANGE_FULL_RANGE, 0.25, 0.0, uks ? -0.25 : -0.125}}};
        vibeqc_ks_options pbe0_options{};
        if (pbe0) {
          pbe0_options.struct_size = sizeof(pbe0_options);
          pbe0_options.abi_version = VIBEQC_ABI_VERSION;
          pbe0_options.scf_domain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
          pbe0_options.grid_version = 1;
          pbe0_options.radial_points = 64;
          pbe0_options.angular_polar = 12;
          pbe0_options.angular_azimuth = 24;
          pbe0_options.partition_iterations = 3;
          pbe0_options.coincident_tolerance = 1e-12;
          pbe0_options.tile_points = 256;
          pbe0_options.xc_execution_schedule = VIBEQC_XC_EXECUTION_DEVICE_FUSED;
          pbe0_options.spin_channels = uks ? 2 : 1;
          pbe0_options.semilocal_components = pbe0_components.data();
          pbe0_options.semilocal_component_count = pbe0_components.size();
          pbe0_options.exchange_terms = pbe0_exchange.data();
          pbe0_options.exchange_term_count = pbe0_exchange.size();
          method.ks_options = &pbe0_options;
        }
        vibeqc_calculation *cpu_calculation = nullptr, *cuda_calculation = nullptr;
        require(vibeqc_calculation_prepare(cpu_fixture.context, cpu_fixture.system, &method,
                                           &cpu_calculation) == VIBEQC_STATUS_SUCCESS &&
                    vibeqc_calculation_prepare(cuda_context, cuda_system, &method,
                                               &cuda_calculation) == VIBEQC_STATUS_SUCCESS,
                "CPU/CUDA KS preparation failed");
        auto cpu_result = unconverged, cuda_result = unconverged;
        const auto cpu_status = vibeqc_calculation_execute(cpu_calculation, &cpu_result);
        const auto cuda_status = vibeqc_calculation_execute(cuda_calculation, &cuda_result);
        const char* cuda_detail = vibeqc_context_get_last_detail(cuda_context);
        require(cpu_status == VIBEQC_STATUS_SUCCESS && cuda_status == VIBEQC_STATUS_SUCCESS &&
                    cuda_result.executed_backend == VIBEQC_BACKEND_CUDA &&
                    cuda_result.density_rms < 1e-9 &&
                    std::abs(cuda_result.energy - cpu_result.energy) < 1e-10,
                ("public native CUDA KS energy/residual/backend differs from CPU: method=" +
                 std::to_string(ks) + " CPU status=" + std::to_string(cpu_status) +
                 " CUDA status=" + std::to_string(cuda_status) +
                 " detail=" + (cuda_detail ? cuda_detail : ""))
                    .c_str());
        const auto cold = cuda_result;
        require(
            vibeqc_calculation_execute(cuda_calculation, &cuda_result) == VIBEQC_STATUS_SUCCESS &&
                cuda_result.iterations <= cold.iterations &&
                std::abs(cuda_result.energy - cold.energy) < 1e-11,
            "public CUDA KS compatible replay changed the endpoint");

        auto auto_method = method;
        auto_method.precision_mode = VIBEQC_PRECISION_AUTO;
        vibeqc_calculation* auto_calculation = nullptr;
        const auto auto_status =
            vibeqc_calculation_prepare(cuda_context, cuda_system, &auto_method, &auto_calculation);
        // LDA/PBE admit mixed-J acceleration. r2SCAN and exact-exchange PBE0
        // remain strict FP64 until their independent precision qualifications land.
        if (ks == VIBEQC_METHOD_R2SCAN_RKS || ks == VIBEQC_METHOD_R2SCAN_UKS || pbe0) {
          require(auto_status == VIBEQC_STATUS_NOT_IMPLEMENTED && auto_calculation == nullptr,
                  "strict-FP64 CUDA KS method lost its explicit precision rejection");
        } else {
          require(auto_status == VIBEQC_STATUS_SUCCESS && auto_calculation != nullptr,
                  "CUDA KS automatic-precision preparation failed");
          auto auto_result = unconverged;
          const char* saved_chunk = std::getenv("VIBEQC_CUDA_KS_CHUNK");
          const std::string saved_chunk_value = saved_chunk ? saved_chunk : "";
          require(setenv("VIBEQC_CUDA_KS_CHUNK", "2", 1) == 0,
                  "cannot enable the CUDA KS chunk integration regression");
          require(
              vibeqc_calculation_execute(auto_calculation, &auto_result) == VIBEQC_STATUS_SUCCESS &&
                  auto_result.converged == 1 && std::abs(auto_result.energy - cold.energy) < 2e-8,
              "CUDA KS mixed-J target refinement changed the FP64 endpoint");
          vibeqc_precision_provenance precision{sizeof(vibeqc_precision_provenance),
                                                VIBEQC_ABI_VERSION};
          require(vibeqc_calculation_get_precision_provenance(auto_calculation, &precision) ==
                          VIBEQC_STATUS_SUCCESS &&
                      precision.requested_mode == VIBEQC_PRECISION_AUTO &&
                      precision.effective_bits == 32U && precision.strict_refinement_applied == 1 &&
                      precision.refinement_iterations >= 1U,
                  "CUDA KS mixed-J provenance omitted actual FP32 work or FP64 refinement");
          require(saved_chunk ? setenv("VIBEQC_CUDA_KS_CHUNK", saved_chunk_value.c_str(), 1) == 0
                              : unsetenv("VIBEQC_CUDA_KS_CHUNK") == 0,
                  "cannot restore the CUDA KS chunk integration regression environment");
          vibeqc_calculation_destroy(auto_calculation);
        }

        if (ks == VIBEQC_METHOD_LDA_RKS) {
          // Cover both the owner and generated-XC error boundaries. Neither
          // runtime nor allocation failure may be hidden by a cold warm retry.
          using fail_function = void (*)();
          const std::array<std::pair<fail_function, vibeqc_status>, 3> failures{{
              {&ks_cuda_fail_next_runtime_for_test_v1, VIBEQC_STATUS_CUDA_ERROR},
              {&xc_cuda_fail_next_runtime_for_test_v1, VIBEQC_STATUS_CUDA_ERROR},
              {&xc_cuda_fail_next_allocation_for_test_v1, VIBEQC_STATUS_OUT_OF_MEMORY},
          }};
          for (const auto& [fail, expected_status] : failures) {
            fail();
            require(vibeqc_calculation_execute(cuda_calculation, &cuda_result) == expected_status,
                    "CUDA KS runtime failure lost its public status");
            require(
                vibeqc_calculation_execute(cuda_calculation, &cuda_result) == VIBEQC_STATUS_SUCCESS,
                "CUDA KS runtime failure prevented single-point recovery");
            vibeqc_system* cuda_systems[]{cuda_system};
            vibeqc_batch* cuda_batch = nullptr;
            require(vibeqc_batch_prepare(cuda_context, cuda_systems, 1, &method,
                                         VIBEQC_BATCH_ENABLE_WARM_STARTS,
                                         &cuda_batch) == VIBEQC_STATUS_SUCCESS &&
                        cuda_batch != nullptr,
                    "LDA RKS CUDA warm batch preparation failed");
            vibeqc_batch_item_result_descriptor cuda_item{};
            cuda_item.struct_size = sizeof(cuda_item);
            cuda_item.abi_version = VIBEQC_ABI_VERSION;
            require(vibeqc_batch_execute(cuda_batch, nullptr, 0, &cuda_item, 1) ==
                            VIBEQC_STATUS_SUCCESS &&
                        cuda_item.status == VIBEQC_STATUS_SUCCESS,
                    "LDA RKS CUDA warm batch did not establish a seed");
            cuda_item = {};
            cuda_item.struct_size = sizeof(cuda_item);
            cuda_item.abi_version = VIBEQC_ABI_VERSION;
            fail();
            require(vibeqc_batch_execute(cuda_batch, nullptr, 0, &cuda_item, 1) ==
                            VIBEQC_STATUS_SUCCESS &&
                        cuda_item.status == expected_status && cuda_item.warm_start_used &&
                        !cuda_item.warm_start_fallback,
                    "CUDA warm replay retried or misclassified a device failure");
            cuda_item = {};
            cuda_item.struct_size = sizeof(cuda_item);
            cuda_item.abi_version = VIBEQC_ABI_VERSION;
            require(vibeqc_batch_execute(cuda_batch, nullptr, 0, &cuda_item, 1) ==
                            VIBEQC_STATUS_SUCCESS &&
                        cuda_item.status == VIBEQC_STATUS_SUCCESS && cuda_item.warm_start_used,
                    "CUDA warm replay did not recover after a device failure");
            vibeqc_batch_destroy(cuda_batch);
          }
        }
        cuda_result.forces = forces.data();
        cuda_result.force_count = static_cast<uint32_t>(forces.size());
        require(vibeqc_calculation_execute(cuda_calculation, &cuda_result) ==
                    VIBEQC_STATUS_NOT_IMPLEMENTED,
                "public CUDA KS force request was not rejected");
        vibeqc_calculation_destroy(cpu_calculation);
        vibeqc_calculation_destroy(cuda_calculation);
        vibeqc_system_destroy(cuda_system);
      }
      vibeqc_context_destroy(cuda_context);
    }
#endif
    std::cout << "Registered KS energy-only backend and force-rejection contract passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

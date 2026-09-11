#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "vibeqc/fock.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void checked(vibeqc_fock_plan* plan, vibeqc_status status) {
  // Read the error string after execution; a prior c_str pointer can be
  // invalidated when execution replaces the detail buffer.
  require(status == VIBEQC_STATUS_SUCCESS, vibeqc_fock_plan_last_error(plan));
}

// Only public handles cross this test boundary. Release context/system and
// mutate caller inputs before evaluating the owned source under sanitizers.
void exercise(vibeqc_backend backend, bool unrestricted, bool fitted) {
  vibeqc_context_descriptor context_spec{sizeof(context_spec), VIBEQC_ABI_VERSION, 0, backend};
  vibeqc_context* context = nullptr;
  require(vibeqc_context_create(&context_spec, &context) == 0, "context create");
  std::array<vibeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
  std::array<vibeqc_primitive, 2> primitives{{{1, 1}, {1, 1}}};
  std::array<vibeqc_shell, 2> shells{{{0, 0, 0, 1}, {1, 0, 1, 1}}};
  vibeqc_system_descriptor system_spec{sizeof(system_spec),  VIBEQC_ABI_VERSION,
                                       atoms.data(),         2,
                                       shells.data(),        2,
                                       primitives.data(),    2,
                                       unrestricted ? 1 : 0, unrestricted ? 2U : 1U};
  vibeqc_system* system = nullptr;
  require(vibeqc_system_create(context, &system_spec, &system) == 0, "system create");
  vibeqc_fock_spec spec{
      sizeof(spec),
      VIBEQC_ABI_VERSION,
      1,
      unrestricted ? VIBEQC_FOCK_UNRESTRICTED : VIBEQC_FOCK_RESTRICTED,
      1,
      {1, 1, VIBEQC_FOCK_FULL_RANGE, 0, fitted ? VIBEQC_FOCK_DENSITY_FITTED : VIBEQC_FOCK_EXACT},
      {1, unrestricted ? -1.0 : -0.5, VIBEQC_FOCK_FULL_RANGE, 0, VIBEQC_FOCK_EXACT}};
  vibeqc_fock_plan* plan = nullptr;
  require(vibeqc_fock_plan_create(context, system, nullptr, &spec, nullptr, &plan) == 0,
          "plan create");
  auto invalid = spec;
  invalid.abi_version += 1;
  auto* failed = plan;
  require(vibeqc_fock_plan_create(context, system, nullptr, &invalid, nullptr, &failed) ==
                  VIBEQC_STATUS_ABI_MISMATCH &&
              failed == nullptr,
          "ABI failure handle");
  // Invalid controls fail before source preparation even when a DF cutoff
  // would be mathematically irrelevant to this exact-only request.
  for (double value :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0}) {
    for (bool metric : {false, true}) {
      vibeqc_fock_controls controls{sizeof(controls), VIBEQC_ABI_VERSION, 1e-12, 0, 0};
      (metric ? controls.metric_relative_threshold : controls.screening_tolerance) = value;
      failed = plan;
      require(vibeqc_fock_plan_create(context, system, nullptr, &spec, &controls, &failed) ==
                      VIBEQC_STATUS_INVALID_ARGUMENT &&
                  failed == nullptr,
              "invalid Fock controls must not publish a plan");
    }
  }
  vibeqc_fock_controls controls{sizeof(controls), VIBEQC_ABI_VERSION, 1e-12, 1.0, 0};
  require(vibeqc_fock_plan_create(context, system, nullptr, &spec, &controls, &failed) ==
                  VIBEQC_STATUS_INVALID_ARGUMENT &&
              failed == nullptr,
          "unit metric cutoff rejected");
  controls.metric_relative_threshold = 0;
  require(vibeqc_fock_plan_create(context, system, nullptr, &spec, &controls, &failed) == 0,
          "zero metric cutoff retains default behavior");
  vibeqc_fock_diagnostic defaulted{sizeof(defaulted), VIBEQC_ABI_VERSION};
  require(vibeqc_fock_plan_diagnostic(failed, &defaulted) == 0 &&
              defaulted.metric_relative_threshold == (fitted ? 1e-10 : 0.0),
          "default metric cutoff diagnostic");
  vibeqc_fock_plan_destroy(failed);
  vibeqc_system_destroy(system);
  vibeqc_context_destroy(context);
  atoms[1].z = 70;
  primitives[0].exponent = 19;
  vibeqc_fock_diagnostic diag{sizeof(diag), VIBEQC_ABI_VERSION};
  require(vibeqc_fock_plan_diagnostic(plan, &diag) == 0 && diag.nbf == 2 && diag.backend == backend,
          "diagnostics after source release");

  std::array<double, 8> density{};
  std::array<double, 6> forces{};
  vibeqc_fock_scf_result out{sizeof(out), VIBEQC_ABI_VERSION};
  out.density = density.data();
  out.density_count = unrestricted ? 8 : 4;
  checked(plan, vibeqc_fock_plan_solve(plan, nullptr, nullptr, 0, &out));
  const double energy = out.energy;
  out.forces = forces.data();
  out.force_count = forces.size();
  // Deliberate input/output alias is safe: a complete seed snapshot is owned.
  checked(plan, vibeqc_fock_plan_solve(plan, nullptr, density.data(), out.density_count, &out));
  require(out.initial_density_used && std::abs(out.energy - energy) < 1e-10, "warm result");
  require(std::abs(forces[2] + forces[5]) < 1e-10, "force invariance");
  const auto saved_density = density;
  const auto saved_forces = forces;
  const auto saved_out = out;
  vibeqc_fock_scf_controls limited{sizeof(limited), VIBEQC_ABI_VERSION, 1, 8, 1e-10, 1e-8};
  require(vibeqc_fock_plan_solve(plan, &limited, nullptr, 0, &out) == VIBEQC_STATUS_NOT_CONVERGED,
          "nonconverged status");
  require(std::memcmp(&out, &saved_out, sizeof(out)) == 0 && density == saved_density &&
              forces == saved_forces,
          "failure publication");
  out.forces = density.data();
  require(vibeqc_fock_plan_solve(plan, nullptr, nullptr, 0, &out) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "overlapping SCF outputs");
  out = saved_out;

  std::array<double, 4> j{}, ka{}, kb{}, fa{}, fb{};
  vibeqc_fock_result fixed{sizeof(fixed), VIBEQC_ABI_VERSION};
  fixed.matrix_count = 4;
  fixed.coulomb = j.data();
  fixed.exchange_alpha = ka.data();
  fixed.exchange_beta = unrestricted ? kb.data() : nullptr;
  fixed.fock_alpha = fa.data();
  fixed.fock_beta = unrestricted ? fb.data() : nullptr;
  fixed.gradient = forces.data();
  fixed.gradient_count = 6;
  checked(plan, vibeqc_fock_plan_evaluate(plan, density.data(), 4,
                                          unrestricted ? density.data() + 4 : nullptr,
                                          unrestricted ? 4 : 0, &fixed));
  require(std::abs(fixed.energy_one_electron + fixed.energy_two_electron + fixed.nuclear_repulsion -
                   energy) < 1e-10,
          "fixed-density energy");
  vibeqc_fock_plan_destroy(plan);
  vibeqc_fock_plan_destroy(nullptr);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto backend = argc == 2 && std::string(argv[1]) == "cuda" ? VIBEQC_BACKEND_CUDA
                                                                     : VIBEQC_BACKEND_CPU_REFERENCE;
    for (bool spin : {false, true})
      for (bool fitted : {false, true}) exercise(backend, spin, fitted);
    std::cout << "public Fock lifecycle and SCF passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

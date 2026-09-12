#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "posthf/mp2_cpu_generated.hpp"
#include "posthf/native_provider.hpp"
#include "scf/mean_field.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc::core::System h2() {
  vibeqc::core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<vibeqc::core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "H2 setup");
  return system;
}

void generated_equations() {
  for (unsigned tile : {1, 2, 4, 8}) {
    const auto plan = vibeqc::mp2::generated::cpu_plan(tile);
    std::vector<double> g(tile * tile), x(tile * tile), ea(tile), eb(tile);
    for (unsigned a = 0; a < tile; ++a) {
      ea[a] = 0.2 + 0.1 * a;
      eb[a] = 0.3 + 0.2 * a;
      for (unsigned b = 0; b < tile; ++b) {
        g[a * tile + b] = 0.02 * (1 + a + 2 * b);
        x[a * tile + b] = 0.01 * (2 + 3 * a + b);
      }
    }
    double expected[2]{}, actual[2]{};
    // Independent coordinate loops, not the generated reduction/CPU interpreter.
    for (unsigned a = 0; a < tile; ++a)
      for (unsigned b = 0; b < tile; ++b) {
        const auto q = a * tile + b;
        const double d = -0.8 - 0.5 - ea[a] - eb[b];
        expected[0] += g[q] * g[q] / d;
        expected[1] += g[q] * (g[q] - x[q]) / d;
      }
    plan.run(g.data(), x.data(), -0.8, -0.5, ea.data(), eb.data(), actual);
    for (unsigned k = 0; k < 2; ++k)
      require(std::abs(actual[k] - expected[k]) < 1e-12, "native TensorIR OS/SS factors");
  }
}

void provider_and_reference() {
  const auto system = h2();
  vibeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0;
  options.energy_tolerance = options.density_tolerance = 1e-11;
  options.reference_memory_budget_bytes = 256ULL << 20;
  auto hf = vibeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference && hf.forces.empty(), "bounded reference owns no HF forces");
  const auto& ref = *hf.reference;
  vibeqc::posthf::RawSource source(system);
  vibeqc::posthf::NativeBlockProvider provider(source, ref, 256ULL << 20, 1);
  const vibeqc::posthf::MOSlots slots{{{1, 0}, {0, 1}, {1, 0}, {0, 1}}};
  const auto values = provider.get(slots);
  // The full AO tensor exists only in this deliberately tiny independent
  // eight-loop transform oracle. The native consumer never allocates it.
  const auto oracle = vibeqc::integrals::build_integrals(system);
  auto idx = [](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return ((a * 2 + b) * 2 + c) * 2 + d;
  };
  for (std::size_t p = 0; p < 2; ++p)
    for (std::size_t q = 0; q < 2; ++q)
      for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t s = 0; s < 2; ++s) {
          double expected = 0;
          for (std::size_t u = 0; u < 2; ++u)
            for (std::size_t v = 0; v < 2; ++v)
              for (std::size_t w = 0; w < 2; ++w)
                for (std::size_t x = 0; x < 2; ++x)
                  expected += oracle.eri[idx(u, v, w, x)] * ref.coefficients[2 * u + slots[0][p]] *
                              ref.coefficients[2 * v + slots[1][q]] *
                              ref.coefficients[2 * w + slots[2][r]] *
                              ref.coefficients[2 * x + slots[3][s]];
          require(std::abs(values[idx(p, q, r, s)] - expected) < 1e-11,
                  "native MO index dictionary and permutation");
        }
  auto bad = ref;
  bad.coefficients[0] *= 1.2;
  bool rejected = false;
  try {
    vibeqc::scf::validate_physical_reference(bad);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "nonorthogonal reference was accepted");
  bad = ref;
  bad.density[0] += 0.1;
  rejected = false;
  try {
    vibeqc::scf::validate_physical_reference(bad);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "inconsistent physical density was accepted");
  bad = ref;
  // Finite entries can produce inf/NaN products. Comparison-only validators
  // otherwise accept NaN residuals because each greater-than test is false.
  bad.coefficients.assign(4, std::numeric_limits<double>::max());
  rejected = false;
  try {
    vibeqc::scf::validate_physical_reference(bad);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "overflowing reference validation was accepted");
  const auto plan = provider.plan({2, 2, 2, 2});
  require(plan.allocation_bytes == 0 && plan.device_bytes == 0,
          "CPU provider must not claim GPU allocations");
  bool overflow = false;
  try {
    vibeqc::posthf::numeric_block_plan(1, 0, 0, {SIZE_MAX, 2, 2, 2}, {1, 1, 1, 1}, false);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  require(overflow, "native capacity overflow was not rejected");

  auto spherical = system;
  spherical.basis_representation = VIBEQC_BASIS_SPHERICAL;
  spherical.shells[0].angular_momentum = 3;
  const auto cart = vibeqc::molecule::cartesian_ao_count(spherical);
  const auto n = vibeqc::molecule::ao_count(spherical);
  const auto without_eri = vibeqc::posthf::rhf_reference_capacity(spherical, 8, false);
  const auto with_eri = vibeqc::posthf::rhf_reference_capacity(spherical, 8, true);
  require(with_eri - without_eri >= 8 * (2 * cart * cart * cart * cart + n * n * n * n),
          "CPU reference budget omitted simultaneous Cartesian/spherical tensors");
}
}  // namespace
int main() {
  try {
    generated_equations();
    provider_and_reference();
    std::cout << "MP2 native contracts passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

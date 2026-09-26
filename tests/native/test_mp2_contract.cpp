#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "posthf/cuda_derivative.hpp"
#include "posthf/mp2_cpu_generated.hpp"
#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_energy.hpp"
#include "posthf/mp2_force.hpp"
#include "posthf/native_provider.hpp"
#include "scf/interaction_source_view.hpp"
#include "scf/mean_field.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class ForwardingInteractionSource final : public vibeqc::integrals::ElectronInteractionSource {
 public:
  explicit ForwardingInteractionSource(const vibeqc::posthf::RawSource& source) : source_(source) {}

  const vibeqc::core::System& orbital() const override { return source_.orbital(); }
  std::size_t nbf() const override { return source_.nbf(); }
  std::size_t naux() const override { return source_.naux(); }
  std::size_t retained_numeric_bytes() const override { return source_.retained_numeric_bytes(); }
  bool supports(Operator op) const noexcept override { return source_.supports(op); }
  void read(Operator op, const std::array<std::size_t, 4>& begin,
            const std::array<std::size_t, 4>& count, double* out,
            std::size_t elements) const override {
    source_.read(op, begin, count, out, elements);
  }

 private:
  const vibeqc::posthf::RawSource& source_;
};

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
    const auto tile_elements = static_cast<std::size_t>(tile) * tile;
    std::vector<double> g(tile_elements), x(tile_elements), ea(tile), eb(tile);
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
  const auto common = ref.electronic_reference();
  require(vibeqc::core::electronic_reference_shape_valid(common),
          "RHF common electronic reference shape");
  require(common.restricted() && common.basis_functions == ref.nbf &&
              common.channels[0].occupied == ref.nocc &&
              common.channels[0].coefficients.data() == ref.coefficients.data(),
          "RHF common electronic reference copied or changed orbital ownership");
  vibeqc::posthf::RawSource source(system);
  vibeqc::posthf::NativeBlockProvider provider(source, ref, 256ULL << 20, 1);
  const vibeqc::posthf::MOSlots slots{{{1, 0}, {0, 1}, {1, 0}, {0, 1}}};
  const auto values = provider.get(slots);
  vibeqc::posthf::RawSource source_with_auxiliary(system, &system);
  const auto source_and_auxiliary_bytes = vibeqc::posthf::checked_add(
      vibeqc::posthf::source_capacity(source_with_auxiliary.orbital()),
      vibeqc::posthf::source_capacity(source_with_auxiliary.auxiliary()));
  require(source_with_auxiliary.retained_numeric_bytes() >= source_and_auxiliary_bytes,
          "shared raw source omitted its retained auxiliary basis");
  vibeqc::posthf::NativeBlockProvider auxiliary_provider(source_with_auxiliary, ref, 256ULL << 20,
                                                         1);
  require(auxiliary_provider.source_bytes() >= source_and_auxiliary_bytes,
          "MO provider omitted the auxiliary source from endpoint memory admission");
  ForwardingInteractionSource generic_source(source);
  vibeqc::posthf::NativeBlockProvider generic_provider(generic_source, ref, 256ULL << 20, 1);
  require(generic_provider.get(slots) == values,
          "native MO provider still depends on the concrete RawSource type");

  auto exact_spec = vibeqc::scf::make_hf_fock_spec(vibeqc::scf::FockSpin::Restricted,
                                                   vibeqc::scf::FockApproximation::Exact);
  exact_spec.derivative_order = 0;
  vibeqc::scf::PreparedFockPlan exact_plan(
      system, nullptr,
      vibeqc::scf::resolve_fock_build(exact_spec, vibeqc::scf::FockBackend::Cpu, 0.0));
  vibeqc::scf::PreparedFockInteractionSourceView exact_source(exact_plan);
  require(exact_source.supports(vibeqc::integrals::ElectronInteractionOperator::eri) &&
              !exact_source.supports(vibeqc::integrals::ElectronInteractionOperator::metric),
          "prepared exact source advertised incorrect capabilities");
  require(exact_source.retained_numeric_bytes() == exact_plan.cpu_observation_capacity(),
          "prepared source residency diverged from its owner");
  vibeqc::posthf::NativeBlockProvider exact_provider(exact_source, ref, 256ULL << 20, 1);
  const auto exact_values = exact_provider.get(slots);
  require(exact_values.size() == values.size(), "prepared exact MO block shape changed");
  for (std::size_t q = 0; q < values.size(); ++q)
    require(std::abs(exact_values[q] - values[q]) < 1e-11,
            "prepared exact owner changed the MO block");

  auto df_spec = vibeqc::scf::make_hf_fock_spec(vibeqc::scf::FockSpin::Restricted,
                                                vibeqc::scf::FockApproximation::DensityFitted);
  df_spec.derivative_order = 0;
  vibeqc::scf::PreparedFockPlan df_plan(
      system, &system, vibeqc::scf::resolve_fock_build(df_spec, vibeqc::scf::FockBackend::Cpu));
  vibeqc::scf::PreparedFockInteractionSourceView df_source(df_plan);
  require(!df_source.supports(vibeqc::integrals::ElectronInteractionOperator::eri) &&
              df_source.supports(vibeqc::integrals::ElectronInteractionOperator::metric) &&
              df_source.supports(vibeqc::integrals::ElectronInteractionOperator::three_center),
          "prepared DF source advertised incorrect capabilities");
  const auto* fitted = df_plan.cpu_fitted_data();
  require(fitted != nullptr, "prepared DF source lost its CPU resident owner");
  std::vector<double> metric(fitted->raw.metric.size());
  df_source.read(vibeqc::integrals::ElectronInteractionOperator::metric, {0, 0, 0, 0},
                 {fitted->raw.naux, fitted->raw.naux, 1, 1}, metric.data(), metric.size());
  require(metric == fitted->raw.metric, "prepared DF metric view changed resident values");
  std::vector<double> three_center(fitted->raw.three_center.size());
  df_source.read(vibeqc::integrals::ElectronInteractionOperator::three_center, {0, 0, 0, 0},
                 {fitted->raw.nbf, fitted->raw.nbf, fitted->raw.naux, 1}, three_center.data(),
                 three_center.size());
  require(three_center == fitted->raw.three_center,
          "prepared DF three-center view changed resident values");
  std::array<double, 1> sentinel{123.0};
  bool unsupported_rejected = false;
  try {
    df_source.read(vibeqc::integrals::ElectronInteractionOperator::eri, {0, 0, 0, 0}, {1, 1, 1, 1},
                   sentinel.data(), 1);
  } catch (const std::invalid_argument&) {
    unsupported_rejected = true;
  }
  require(unsupported_rejected && sentinel[0] == 123.0,
          "unsupported prepared interaction read modified caller output");
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

  const vibeqc::posthf::MOSlots second_slots{{{0, 1}, {1, 0}, {0, 1}, {1, 0}}};
  vibeqc::posthf::ProviderWork sequential_work, batched_work;
  const auto sequential_first = provider.get(slots, false, 0, nullptr, &sequential_work);
  const auto sequential_second = provider.get(second_slots, false, 0, nullptr, &sequential_work);
  const auto batched = provider.get_many({slots, second_slots}, false, 0, nullptr, &batched_work);
  require(batched.size() == 2, "native MO batch output count");
  for (std::size_t q = 0; q < sequential_first.size(); ++q)
    require(std::abs(batched[0][q] - sequential_first[q]) < 1e-12,
            "native MO batch first block changed values");
  for (std::size_t q = 0; q < sequential_second.size(); ++q)
    require(std::abs(batched[1][q] - sequential_second[q]) < 1e-12,
            "native MO batch second block changed values");
  require(batched_work.mo_blocks == sequential_work.mo_blocks && batched_work.mo_blocks == 2,
          "native MO batch work did not count transformed blocks");
  require(2 * batched_work.source_reads == sequential_work.source_reads,
          "native MO batch did not reuse AO source reads");
  require(2 * batched_work.source_values == sequential_work.source_values,
          "native MO batch did not reuse AO source values");
  require(batched_work.transform_fmas == sequential_work.transform_fmas,
          "native MO batch changed AO-to-MO transform work");

  const std::array<std::size_t, 4> batch_shape{2, 2, 2, 2};
  const auto cuda_plan = provider.plan(batch_shape, true);
  const auto cpu_common = provider.batch_bytes(batch_shape, 0, false);
  const auto cuda_common = provider.batch_bytes(batch_shape, 0, true);
  const auto cuda_one = provider.batch_bytes(batch_shape, 1, true);
  const auto cuda_two = provider.batch_bytes(batch_shape, 2, true);
  require(cuda_plan.device_bytes >= cuda_plan.aligned_numeric,
          "CUDA batch plan fixed allowance underflow");
  require(cuda_common - cpu_common == cuda_plan.device_bytes - cuda_plan.aligned_numeric,
          "CUDA batch common capacity did not isolate the shared owner");
  require(cuda_one >= cuda_common && cuda_two - cuda_one == cuda_one - cuda_common,
          "CUDA batch request capacity is not additive after the shared owner");
  const auto shared_two_budget = cuda_two;
  vibeqc::posthf::NativeBlockProvider shared_cuda_provider(source, ref, shared_two_budget);
  require(shared_cuda_provider.batch_capacity(batch_shape, true) >= 2,
          "shared CUDA owner capacity was charged once per request");

  require(provider.batch_capacity(batch_shape) >= 2,
          "native MO batch capacity is unexpectedly one");
  const auto single_request_bytes = provider.batch_bytes(batch_shape, 1);
  vibeqc::posthf::NativeBlockProvider tight_provider(source, ref, single_request_bytes, 1);
  require(tight_provider.batch_capacity(batch_shape) == 1,
          "native MO batch capacity ignored the memory budget");
  bool batch_rejected = false;
  try {
    (void)tight_provider.get_many({slots, second_slots});
  } catch (const std::length_error&) {
    batch_rejected = true;
  }
  require(batch_rejected, "native MO batch exceeded the memory budget");
  // Heterogeneous shapes and zero-padded tail columns must keep request order.
  const vibeqc::posthf::MOSlots padded{{{0}, {1, vibeqc::posthf::padded_mo}, {1}, {0}}};
  const std::vector<vibeqc::posthf::MOSlots> mixed{slots, padded, second_slots};
  const auto mixed_values = provider.get_many(mixed);
  for (std::size_t request = 0; request < mixed.size(); ++request) {
    const auto single = provider.get(mixed[request]);
    require(single == mixed_values[request], "heterogeneous MO batch changed values or order");
  }
  auto invalid = slots;
  invalid[3][0] = ref.nbf;
  vibeqc::posthf::ProviderWork rejected_work;
  bool invalid_rejected = false;
  try {
    (void)provider.get_many({slots, invalid}, false, 0, nullptr, &rejected_work);
  } catch (const std::invalid_argument&) {
    invalid_rejected = true;
  }
  require(invalid_rejected && rejected_work.source_scans == 0 && rejected_work.source_reads == 0 &&
              rejected_work.mo_blocks == 0,
          "invalid later MO request reached AO traversal or published work");

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

void conventional_energy_batch_fallback_matches() {
  auto system = h2();
  system.shells.push_back({0, 1, {{0.7, 1.0}}});
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "p-shell MP2 fixture normalization");
  vibeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0;
  options.energy_tolerance = options.density_tolerance = 1e-11;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = vibeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference, "p-shell MP2 reference");
  const auto& ref = *hf.reference;
  vibeqc::posthf::RawSource source(system);
  constexpr unsigned tile = 2;
  const auto kernel = vibeqc::mp2::generated::cpu_plan(tile);
  const auto reserve = kernel.numeric_bytes + 32ULL * tile * tile + 16ULL * tile + 64;
  vibeqc::posthf::NativeBlockProvider widest_provider(source, ref, 256ULL << 20,
                                                      std::numeric_limits<unsigned>::max());
  std::size_t minimum_provider_bytes = std::numeric_limits<std::size_t>::max();
  for (std::size_t axis_tile = 1; axis_tile <= widest_provider.tile_shape()[0]; ++axis_tile) {
    vibeqc::posthf::NativeBlockProvider candidate(source, ref, 256ULL << 20,
                                                  static_cast<unsigned>(axis_tile));
    minimum_provider_bytes =
        std::min(minimum_provider_bytes, candidate.batch_bytes({1, tile, 1, tile}, 1));
  }
  const auto minimum_budget = minimum_provider_bytes + reserve;
  const auto roomy =
      vibeqc::mp2::conventional_energy(ref, source, 256ULL << 20, 1e-10, tile, false, 0);
  const auto tight =
      vibeqc::mp2::conventional_energy(ref, source, minimum_budget, 1e-10, tile, false, 0);
  require(roomy.tiles > 1 && roomy.tiles == tight.tiles, "complete MP2 tile sequence changed");
  require(std::abs(roomy.opposite_spin - tight.opposite_spin) < 1e-13 &&
              std::abs(roomy.same_spin - tight.same_spin) < 1e-13,
          "memory-bounded fallback changed MP2 energy");
  require(roomy.provider_work.source_reads < tight.provider_work.source_reads &&
              roomy.provider_work.transform_fmas <= tight.provider_work.transform_fmas,
          "source-tile scheduling did not reduce provider work");
  require(tight.numeric_capacity_bytes <= minimum_budget, "tight numeric budget exceeded");
  bool rejected = false;
  try {
    (void)vibeqc::mp2::conventional_energy(ref, source, minimum_budget - 1, 1e-10, tile, false, 0);
  } catch (const std::length_error&) {
    rejected = true;
  }
  require(rejected, "one-byte-below the minimum source-tile budget was accepted");
}

void conventional_energy_reuses_ao_scans() {
  const auto system = h2();
  vibeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0;
  options.energy_tolerance = options.density_tolerance = 1e-11;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = vibeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference, "batched MP2 energy reference");
  vibeqc::posthf::RawSource source(system);
  const auto energy =
      vibeqc::mp2::conventional_energy(*hf.reference, source, 256ULL << 20, 1e-10, 1, false, 0);
  require(energy.tiles == 1 && energy.provider_work.mo_blocks == 2 &&
              energy.provider_work.source_scans == 1,
          "batched MP2 energy request/source-scan count");
  std::size_t full_ao_values = 1;
  for (unsigned k = 0; k < 4; ++k) full_ao_values *= hf.reference->nbf;
  require(energy.provider_work.source_values == full_ao_values,
          "batched MP2 energy rescanned the AO tensor");
  require(energy.provider_work.source_reads == full_ao_values,
          "batched MP2 energy source-read count changed unexpectedly");
}

void shell_local_weighted_eri_derivative() {
  const auto system = h2();
  const auto oracle = vibeqc::integrals::build_integrals(system);
  const std::array<std::size_t, 4> shells{0, 1, 0, 1};
  const std::array<double, 1> weights{0.37};
  const auto center =
      vibeqc::integrals::contract_weighted_eri_shell_derivative(system, shells, weights);
  std::array<double, 6> scattered{};
  for (std::size_t slot = 0; slot < 4; ++slot)
    for (std::size_t axis = 0; axis < 3; ++axis)
      scattered[3 * system.shells[shells[slot]].atom_index + axis] += center[3 * slot + axis];
  const auto eri = ((0 * 2 + 1) * 2 + 0) * 2 + 1;
  for (std::size_t coordinate = 0; coordinate < scattered.size(); ++coordinate) {
    const double expected = weights[0] * oracle.eri_derivative[coordinate * 16 + eri];
    require(std::abs(scattered[coordinate] - expected) < 1e-11,
            "shell-local weighted ERI derivative differs from dense oracle");
  }
}

void cuda_shell_derivative_budget_failure_is_transactional() {
  const auto system = h2();
  const std::array<std::size_t, 4> shells{0, 1, 0, 1};
  const std::array<double, 1> weights{0.37};
  std::array<double, 12> center{};
  center.fill(123.0);
  std::string detail;
  // Reject before device execution: a nonzero one-byte budget cannot hold
  // even the fixed staging state. A valid request can succeed on GPU hosts,
  // so device availability must not determine this negative-path contract.
  const auto status = vibeqc::posthf::contract_weighted_eri_shell_derivative_cuda(
      0, system, shells, weights, 1, center, detail);
#if VIBEQC_HAS_CUDA
  require(status == VIBEQC_STATUS_OUT_OF_MEMORY,
          "CUDA derivative accepted an insufficient staging budget");
#else
  require(status == VIBEQC_STATUS_NOT_IMPLEMENTED, "CPU build lost CUDA stub status");
#endif
  require(std::all_of(center.begin(), center.end(), [](double value) { return value == 123.0; }),
          "failed CUDA shell derivative modified caller output");
}

void streamed_one_electron_derivative() {
  const auto system = h2();
  const auto oracle = vibeqc::integrals::build_integrals(system, true, false);
  const std::array<double, 4> overlap_weights{0.2, -0.1, 0.3, 0.4};
  const std::array<double, 4> hcore_weights{-0.5, 0.7, -0.2, 0.6};
  const auto derivative = vibeqc::integrals::contract_weighted_one_electron_derivative(
      system, overlap_weights, hcore_weights, true);
  require(derivative.size() == 6, "streamed one-electron derivative shape");
  for (std::size_t coordinate = 0; coordinate < derivative.size(); ++coordinate) {
    double expected = oracle.nuclear_repulsion_derivative[coordinate];
    for (std::size_t element = 0; element < 4; ++element)
      expected += overlap_weights[element] * oracle.overlap_derivative[coordinate * 4 + element] +
                  hcore_weights[element] * oracle.hcore_derivative[coordinate * 4 + element];
    require(std::abs(derivative[coordinate] - expected) < 1e-11,
            "streamed one-electron derivative differs from dense oracle");
  }
}

void conventional_derivative_from_mo_weights() {
  const auto system = h2();
  vibeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0;
  options.energy_tolerance = options.density_tolerance = 1e-11;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = vibeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference, "MO derivative reference");
  const auto& ref = *hf.reference;
  vibeqc::mp2::LagrangianWeights weights;
  weights.orbitals = 2;
  weights.occupied = 1;
  weights.one_electron = {0.2, -0.3, 0.4, 0.1};
  weights.overlap = {-0.1, 0.5, 0.6, -0.2};
  weights.two_electron.resize(16);
  for (std::size_t i = 0; i < weights.two_electron.size(); ++i)
    weights.two_electron[i] = 0.01 * static_cast<double>(i + 1);
  const auto derivative = vibeqc::mp2::conventional_derivative_cpu(system, ref, weights);

  const auto oracle = vibeqc::integrals::build_integrals(system);
  std::array<double, 4> one_ao{}, overlap_ao{};
  std::array<double, 16> two_ao{};
  auto eri = [](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * 2 + q) * 2 + r) * 2 + s;
  };
  for (std::size_t u = 0; u < 2; ++u)
    for (std::size_t v = 0; v < 2; ++v)
      for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t q = 0; q < 2; ++q) {
          const double pullback = ref.coefficients[2 * u + p] * ref.coefficients[2 * v + q];
          one_ao[2 * u + v] += pullback * weights.one_electron[2 * p + q];
          overlap_ao[2 * u + v] += pullback * weights.overlap[2 * p + q];
        }
  for (std::size_t u = 0; u < 2; ++u)
    for (std::size_t v = 0; v < 2; ++v)
      for (std::size_t w = 0; w < 2; ++w)
        for (std::size_t x = 0; x < 2; ++x)
          for (std::size_t p = 0; p < 2; ++p)
            for (std::size_t q = 0; q < 2; ++q)
              for (std::size_t r = 0; r < 2; ++r)
                for (std::size_t s = 0; s < 2; ++s)
                  two_ao[eri(u, v, w, x)] +=
                      ref.coefficients[2 * u + p] * ref.coefficients[2 * v + q] *
                      ref.coefficients[2 * w + r] * ref.coefficients[2 * x + s] *
                      weights.two_electron[eri(p, q, r, s)];
  require(derivative.size() == 6, "conventional derivative shape");
  for (std::size_t coordinate = 0; coordinate < derivative.size(); ++coordinate) {
    double expected = oracle.nuclear_repulsion_derivative[coordinate];
    for (std::size_t element = 0; element < 4; ++element)
      expected += one_ao[element] * oracle.hcore_derivative[coordinate * 4 + element] +
                  overlap_ao[element] * oracle.overlap_derivative[coordinate * 4 + element];
    for (std::size_t element = 0; element < 16; ++element)
      expected += two_ao[element] * oracle.eri_derivative[coordinate * 16 + element];
    require(std::abs(derivative[coordinate] - expected) < 1e-10,
            "conventional MO-weight derivative differs from dense oracle");
  }
#if !VIBEQC_HAS_CUDA
  bool cuda_rejected = false;
  try {
    (void)vibeqc::mp2::conventional_derivative_cuda(system, ref, weights, 0, 1ULL << 20);
  } catch (const std::runtime_error&) {
    cuda_rejected = true;
  }
  require(cuda_rejected, "CPU build accepted a CUDA conventional derivative");
#endif
}

double conventional_total_energy(vibeqc::core::System system) {
  vibeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0;
  options.energy_tolerance = options.density_tolerance = 1e-12;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = vibeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference, "finite-difference MP2 reference");
  vibeqc::posthf::RawSource source(std::move(system));
  const auto correlation =
      vibeqc::mp2::conventional_energy(*hf.reference, source, 256ULL << 20, 1e-10, 1, false, 0);
  return hf.reference->energy + correlation.opposite_spin + correlation.same_spin;
}

void complete_conventional_force_matches_resolved_energy() {
  const auto system = h2();
  vibeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0;
  options.energy_tolerance = options.density_tolerance = 1e-12;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = vibeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference, "analytic MP2 force reference");
  vibeqc::posthf::RawSource source(system);
  vibeqc::response::GmresOptions response;
  response.relative_tolerance = 1e-12;
  response.absolute_tolerance = 1e-13;
  response.restart = 8;
  response.max_iterations = 40;
  response.max_workspace_bytes = 64ULL << 20;
  const auto analytic = vibeqc::mp2::conventional_force_cpu(*hf.reference, source, 256ULL << 20,
                                                            1e-10, 1e-10, response);
  require(analytic.response.converged(), "MP2 Z-vector did not converge");
  require(analytic.forces.size() == 6, "MP2 force shape");
  const double step = 1e-4;
  auto plus = system;
  auto minus = system;
  plus.atoms[0].position[2] += step;
  minus.atoms[0].position[2] -= step;
  const double finite =
      (conventional_total_energy(std::move(plus)) - conventional_total_energy(std::move(minus))) /
      (2.0 * step);
  require(std::abs(analytic.forces[2] + finite) < 2e-6,
          "complete conventional MP2 force differs from resolved finite difference");
#if !VIBEQC_HAS_CUDA
  bool cuda_rejected = false;
  try {
    (void)vibeqc::mp2::conventional_force_cuda(*hf.reference, source, 256ULL << 20, 1e-10, 1e-10,
                                               response, 0);
  } catch (const std::runtime_error&) {
    cuda_rejected = true;
  }
  require(cuda_rejected, "CPU build accepted a CUDA conventional force owner");
#endif
}
}  // namespace
int main() {
  try {
    generated_equations();
    provider_and_reference();
    conventional_energy_reuses_ao_scans();
    conventional_energy_batch_fallback_matches();
    shell_local_weighted_eri_derivative();
    cuda_shell_derivative_budget_failure_is_transactional();
    streamed_one_electron_derivative();
    conventional_derivative_from_mo_weights();
    complete_conventional_force_matches_resolved_energy();
    std::cout << "MP2 native contracts passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

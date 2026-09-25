#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "api/handles.hpp"
#include "api/ks_snapshot.hpp"
#include "dft/cuda_ks.hpp"
#include "dft/xc.hpp"
#include "methods/dft_method.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda_fock_execution.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"

namespace {
using namespace vibeqc;
using scf::reference::Matrix;
void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
bool expect_iteration_chunking() {
  const char* selection = std::getenv("VIBEQC_CUDA_KS_CHUNK");
  return selection != nullptr && std::string(selection) == "2";
}
core::System hydrogens(unsigned count, bool restricted, double shift = 0.0) {
  core::System system;
  system.multiplicity = restricted ? 1 : 2;
  system.charge = !restricted && count == 2 ? 1 : 0;
  for (unsigned i = 0; i < count; ++i) {
    system.atoms.push_back({1, {0.15 * i * i, 0.13 * i, (1.5 + shift) * i}});
    system.shells.push_back(
        {i,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
  }
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS, detail);
  return system;
}
scf::ResolvedFockBuild strategy(bool restricted, scf::FockBackend backend) {
  scf::FockBuildSpec spec;
  spec.spin = restricted ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted;
  spec.exchange.present = false;
  spec.derivative_order = 0;
  return scf::resolve_fock_build(spec, backend, 1e-12);
}

void prepared_cuda_fock_seam() {
  const auto system = hydrogens(2, true);
  const scf::PreparedFockPlan cpu(system, nullptr, strategy(true, scf::FockBackend::Cpu));
  require(!scf::prepared_cuda_fock_binding(cpu),
          "CPU Fock owner unexpectedly exposed a CUDA execution binding");

  scf::FockBuildSpec spec;
  spec.spin = scf::FockSpin::Restricted;
  spec.derivative_order = 0;
  const auto resolved = scf::resolve_fock_build(spec, scf::FockBackend::Cuda, 1e-12);
  const scf::PreparedFockPlan hybrid(system, nullptr, resolved, 0);
  const auto binding = scf::prepared_cuda_fock_binding(hybrid);
  require(binding && binding.nbf == hybrid.one_electron().nbf && binding.stream != nullptr &&
              binding.source_identity != nullptr,
          "prepared full-range CUDA J/K owner lacks the method-neutral execution binding");

  auto range_spec = spec;
  range_spec.coulomb.present = false;
  range_spec.exchange = {true, -0.19, scf::FockOperator::LongRange, 0.33,
                         scf::FockApproximation::Exact};
  const auto range_resolved = scf::resolve_fock_build(range_spec, scf::FockBackend::Cuda, 0.0);
  const scf::PreparedFockPlan range(system, nullptr, range_resolved, 0);
  require(static_cast<bool>(scf::prepared_cuda_fock_binding(range)),
          "unscreened value-only CUDA range exchange lacks the prepared execution binding");

  const auto screened_resolved = scf::resolve_fock_build(range_spec, scf::FockBackend::Cuda, 1e-12);
  const scf::PreparedFockPlan screened_range(system, nullptr, screened_resolved, 0);
  require(!scf::prepared_cuda_fock_binding(screened_range),
          "screened CUDA range exchange bypassed the qualification gate");
}

/** Independently rebuild the retained density with CPU integrals/XC. This
 * catches a converged flag or energy belonging to the preceding generation. */
void physical_check(const scf::PreparedFockPlan& cpu, const dft::AoBasis& basis,
                    const dft::MolecularGrid& grid, std::uint32_t functional,
                    const scf::ScfResult& result) {
  using namespace scf::reference;
  const auto n = basis.nao, elements = n * n;
  const bool uks = cpu.strategy().spec.spin == scf::FockSpin::Unrestricted;
  Matrix a(result.density.begin(), result.density.begin() + elements), b;
  if (uks) b.assign(result.density.begin() + elements, result.density.end());
  const auto jk = cpu.build(a, b);
  auto fock = scf::assemble_fock(cpu.strategy(), cpu.one_electron().hcore, jk);
  double xc_energy;
  if (uks) {
    const auto xc = functional == 2U   ? dft::integrate_r2scan_uks(basis, grid, a, b)
                    : functional == 1U ? dft::integrate_pbe_uks(basis, grid, a, b)
                                       : dft::integrate_lda_xc_pw_uks(basis, grid, a, b);
    xc_energy = xc.energy;
    for (std::size_t i = 0; i < elements; ++i) {
      fock.alpha[i] += xc.potential[0][i];
      fock.beta[i] += xc.potential[1][i];
    }
  } else {
    const auto xc = functional == 2U   ? dft::integrate_r2scan_rks(basis, grid, a)
                    : functional == 1U ? dft::integrate_pbe_rks_with_tail(basis, grid, a)
                                       : dft::integrate_lda_xc_pw_rks(basis, grid, a);
    xc_energy = xc.energy;
    for (std::size_t i = 0; i < elements; ++i) fock.alpha[i] += xc.potential[i];
  }
  const auto& ints = cpu.one_electron();
  const double energy = ints.nuclear_repulsion + dot(a, ints.hcore) +
                        (uks ? dot(b, ints.hcore) : 0.0) +
                        scf::contract_fock_energy(cpu.strategy(), jk, a, b) + xc_energy;
  const auto residual_a = commutator_residual(fock.alpha, a, ints.overlap, n);
  const auto residual_b = uks ? commutator_residual(fock.beta, b, ints.overlap, n) : Matrix{};
  const double residual = std::max(residual_rms(residual_a), uks ? residual_rms(residual_b) : 0.0);
  const double public_residual =
      residual_rms(uks ? concatenate(residual_a, residual_b) : residual_a);
  require(std::abs(energy - result.energy) < 2e-11, "CUDA E and returned D are inconsistent");
  require(std::abs(residual - result.dft_diagnostic.physical_residual) < 2e-11,
          "CUDA physical residual belongs to another state");
  require(std::abs(public_residual - result.physical_residual_rms) < 2e-11,
          "CUDA public RMS no longer combines the physical spin-matrix entries");
  if (result.converged)
    require(residual < 1e-9, "CUDA reported convergence above the physical gate");
  require(std::abs(energy - (result.energy + xc_energy)) > 0.05,
          "CUDA endpoint gate does not detect XC double counting");
}

void compare_rks_chunk_history(bool pbe) {
  const auto system = hydrogens(2, true);
  const dft::AoBasis basis(system);
  const dft::GridSpec grid_spec{1, 24, 12, 24, 3, 1e-12};
  const dft::MolecularGrid grid(system, grid_spec);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.max_iterations = 150;
  const auto solve = [&](const char* width) {
    require(::setenv("VIBEQC_CUDA_KS_CHUNK", width, 1) == 0,
            "could not select CUDA RKS history route");
    const scf::PreparedFockPlan gpu(system, nullptr, strategy(true, scf::FockBackend::Cuda), 0);
    dft::CudaKsPlan plan(gpu, basis, grid, options,
                         pbe ? dft::SemilocalFamily::Pbe : dft::SemilocalFamily::Lda, 257);
    auto result = plan.run(nullptr, false, false);
    return std::pair{std::move(result), plan.transfers()};
  };
  const auto ordinary = solve("1");
  const auto chunked = solve("2");
  const auto& left = ordinary.first.dft_diagnostic.history;
  const auto& right = chunked.first.dft_diagnostic.history;
  require(ordinary.first.converged && chunked.first.converged &&
              ordinary.first.iterations == chunked.first.iterations &&
              left.size() == right.size() &&
              std::abs(ordinary.first.energy - chunked.first.energy) < 1e-13,
          "CUDA RKS chunk changed the ordinary physical trajectory");
  for (std::size_t i = 0; i < left.size(); ++i) {
    const bool energy_change_equal =
        (std::isinf(left[i].energy_change) && std::isinf(right[i].energy_change)) ||
        std::abs(left[i].energy_change - right[i].energy_change) < 1e-13;
    require(left[i].iteration == right[i].iteration && energy_change_equal &&
                std::abs(left[i].components.total() - right[i].components.total()) < 1e-13 &&
                std::abs(left[i].density_change - right[i].density_change) < 1e-13 &&
                std::abs(left[i].physical_residual - right[i].physical_residual) < 1e-13 &&
                std::abs(left[i].electrons[0] - right[i].electrons[0]) < 1e-13 &&
                std::abs(left[i].electrons[1] - right[i].electrons[1]) < 1e-13 &&
                left[i].occupation_stabilized == right[i].occupation_stabilized,
            "CUDA RKS chunk changed an ordinary iteration-history row");
  }
  require(ordinary.second.iteration_synchronizations == ordinary.second.iterations &&
              chunked.second.iteration_synchronizations < chunked.second.iterations,
          "CUDA RKS history comparison did not exercise both fence cadences");
  require(::setenv("VIBEQC_CUDA_KS_CHUNK", "2", 1) == 0,
          "could not restore CUDA RKS chunk qualification");
}

/** OH exercises the stationary integer-occupation cycle from #305 on CUDA.
 * Rebuild every returned physical quantity with the unshifted CPU operator. */
void run_hydroxyl(bool pbe) {
  core::System system;
  system.multiplicity = 2;
  system.atoms = {{8, {0, 0, 0}}, {1, {0, 0, 1.8}}};
  system.shells = {
      {0,
       0,
       {{130.7093214, 0.1543289673}, {23.80886605, 0.5353281423}, {6.443608313, 0.4446345422}}},
      {0,
       0,
       {{5.033151319, -0.09996722919}, {1.169596125, 0.3995128261}, {0.38038896, 0.7001154689}}},
      {0, 1, {{5.033151319, 0.155916275}, {1.169596125, 0.6076837186}, {0.38038896, 0.3919573931}}},
      {1,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}}};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS, detail);
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system);
  const scf::PreparedFockPlan cpu(system, nullptr, strategy(false, scf::FockBackend::Cpu));
  const scf::PreparedFockPlan gpu(system, nullptr, strategy(false, scf::FockBackend::Cuda), 0);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.max_iterations = 200;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  dft::CudaKsPlan plan(gpu, basis, grid, options,
                       pbe ? dft::SemilocalFamily::Pbe : dft::SemilocalFamily::Lda);
  const auto cold = plan.run(nullptr, false, false);
  require(cold.converged && !plan.failed(), "CUDA OH occupation cycle did not converge");
  const auto reference = scf::run_uks(cpu, basis, grid, options, pbe);
  require(reference.converged && std::abs(reference.energy - cold.energy) < 1e-10,
          "CUDA OH endpoint disagrees with independently solved CPU UKS");
  const auto cold_execution = plan.transfers();
  require(cold_execution.iterations == cold.iterations &&
              cold_execution.iteration_chunks == cold_execution.iteration_synchronizations &&
              cold_execution.iteration_synchronizations == cold_execution.iterations,
          "CUDA OH must retain the host-controlled one-fence-per-iteration path");
  require(plan.transfers().matrix_d2h_bytes == 0,
          "CUDA occupation stabilization exported iteration matrices");
  physical_check(cpu, basis, grid, pbe, plan.result());
  const auto& history = cold.dft_diagnostic.history;
  const auto cycle = std::find_if(history.begin(), history.end(), [&](const auto& item) {
    return item.iteration > 1 && item.energy_change < options.energy_tolerance &&
           item.physical_residual < options.density_tolerance &&
           item.density_change >= options.density_tolerance;
  });
  if (!pbe) require(cycle != history.end(), "OH regression did not exercise the occupation cycle");
  if (cycle != history.end()) {
    require(cycle->iteration < cold.iterations,
            "stationary energy bypassed the subsequent density-change gate");
    require(plan.transfers().occupation_stabilized_proposals > 0,
            "CUDA stationary cycle did not apply the CPU-compatible proposal policy");
    require(history.back().occupation_stabilized,
            "CUDA final closure discarded the qualified stationary occupation choice");
  }
  const auto stabilized_rows = std::count_if(
      history.begin(), history.end(), [](const auto& item) { return item.occupation_stabilized; });
  require(static_cast<std::uint64_t>(stabilized_rows) ==
              plan.transfers().occupation_stabilized_proposals,
          "CUDA history did not record every stabilized occupation proposal");
  for (const auto& item : history)
    require(std::abs(item.electrons[0] - 5) < 1e-10 && std::abs(item.electrons[1] - 4) < 1e-10,
            "occupation stabilization changed the requested spin populations");
  const auto warm = plan.run();
  require(
      warm.converged && warm.initial_density_used && std::abs(warm.energy - cold.energy) < 1e-10,
      "CUDA OH resident replay lost its physical endpoint");
  physical_check(cpu, basis, grid, pbe, warm);
  const auto restarted = plan.run(nullptr, false);
  require(restarted.converged && !restarted.initial_density_used &&
              std::abs(restarted.energy - cold.energy) < 1e-10,
          "CUDA OH cold restart retained stale proposal control");
  physical_check(cpu, basis, grid, pbe, restarted);
  std::cout << "KS OH pbe=" << pbe << " iterations=" << cold.iterations << '\n';
}

void run_case(unsigned atoms, bool restricted, std::uint32_t functional) {
  const auto system = hydrogens(atoms, restricted);
  const dft::AoBasis basis(system);
  const dft::GridSpec grid_spec{1, 24, 12, 24, 3, 1e-12};
  const dft::MolecularGrid grid(system, grid_spec);
  const scf::PreparedFockPlan cpu(system, nullptr, strategy(restricted, scf::FockBackend::Cpu));
  const scf::PreparedFockPlan gpu(system, nullptr, strategy(restricted, scf::FockBackend::Cuda), 0);
  scf::ScfOptions options;
  options.compute_forces = false;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.max_iterations = 150;
  dft::CudaKsPlan plan(gpu, basis, grid, options, dft::semilocal_family_from_code(functional), 257);
  dft::CudaKsFinalStateToken unavailable;
  std::string snapshot_detail;
  require(plan.final_state_token(unavailable, snapshot_detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "fresh CUDA KS owner published a final-state token");
  plan.begin(nullptr, false);
  while (plan.active()) {
    plan.enqueue_iteration();
    require(plan.pending(), "CUDA iteration did not retain pending state");
    plan.finish_iteration();
  }
  require(plan.transfers().matrix_d2h_bytes == 0, "CUDA SCF staged an iteration matrix");
  const auto result = plan.result();
  const auto cold_execution = plan.transfers();
  require(cold_execution.iterations == result.iterations &&
              cold_execution.iteration_chunks == cold_execution.iteration_synchronizations,
          "CUDA KS chunk accounting does not match the physical trajectory");
  if (restricted && expect_iteration_chunking() && result.iterations > 1) {
    require(cold_execution.iteration_synchronizations < cold_execution.iterations,
            "qualified CUDA RKS retained a mandatory host fence after every iteration");
    require(cold_execution.execution_region_bindings == 1 &&
                cold_execution.execution_region_executions == cold_execution.iteration_chunks &&
                cold_execution.execution_region_failures == 0,
            "CUDA KS device chunks bypassed the shared compiled-execution lifecycle");
  } else {
    require(cold_execution.iteration_synchronizations == cold_execution.iterations,
            "ordinary CUDA KS baseline changed its host-fence cadence");
    require(cold_execution.execution_region_bindings == 0 &&
                cold_execution.execution_region_executions == 0,
            "ordinary CUDA KS unexpectedly bound a compiled execution region");
  }
  require(cold_execution.submitted_iterations >= cold_execution.iterations &&
              cold_execution.submitted_iterations <=
                  cold_execution.iterations + cold_execution.iteration_chunks,
          "CUDA KS speculative work escaped the bounded chunk contract");
  if (!result.converged || plan.failed()) {
    std::cerr << "failed atoms=" << atoms << " restricted=" << restricted
              << " functional=" << functional << " iter=" << result.iterations
              << " residual=" << result.dft_diagnostic.physical_residual
              << " density=" << result.density_rms << '\n';
    throw std::runtime_error("native CUDA KS did not converge");
  }
  const auto reference = [&] {
    if (!restricted)
      return functional == 2U ? scf::run_r2scan_uks(cpu, basis, grid, options)
                              : scf::run_uks(cpu, basis, grid, options, functional == 1U);
    if (functional == 2U) return scf::run_r2scan_rks(cpu, basis, grid, options);
    return functional == 1U ? scf::run_pbe_rks(cpu, basis, grid, options)
                            : scf::run_lda_rks(cpu, basis, grid, options);
  }();
  require(reference.converged && std::abs(reference.energy - result.energy) < 1e-10,
          "CPU/CUDA SCF endpoints disagree");
  physical_check(cpu, basis, grid, functional, result);
  const auto before_snapshot = plan.transfers();
  dft::CudaKsFinalStateToken token;
  require(plan.final_state_token(token, snapshot_detail) == VIBEQC_STATUS_SUCCESS, snapshot_detail);
  require(plan.transfers().final_state_d2h_bytes == before_snapshot.final_state_d2h_bytes,
          "CUDA KS token query transferred device state");
  dft::VerifiedKsFinalState snapshot;
  require(plan.read_final_state(token, false, snapshot, snapshot_detail) == VIBEQC_STATUS_SUCCESS,
          snapshot_detail);
  const auto snapshot_transfer = plan.transfers();
  const auto spins = restricted ? 1U : 2U;
  const auto expected_snapshot_bytes =
      spins * (3 * basis.nao * basis.nao + basis.nao) * sizeof(double) + spins * sizeof(int);
  require(snapshot.weighted_density.empty() && snapshot.density.size() == spins &&
              snapshot.fock.size() == spins && snapshot.orbitals.size() == spins &&
              snapshot.identity.model.grid == grid_spec &&
              snapshot.identity.model.functional == functional &&
              snapshot.identity.model.spins == spins &&
              snapshot.identity.determinant.model == gpu.strategy() &&
              snapshot.identity.determinant.factor.orbital_generation ==
                  snapshot.identity.determinant.factor.density_generation &&
              std::abs(snapshot.components.total() - result.energy) < 1e-12,
          "CUDA KS final snapshot lost model, generation or physical state");
  require(snapshot_transfer.final_state_d2h_bytes - before_snapshot.final_state_d2h_bytes ==
                  expected_snapshot_bytes &&
              snapshot_transfer.final_state_reads == before_snapshot.final_state_reads + 1 &&
              snapshot_transfer.synchronizations == before_snapshot.synchronizations + 1,
          "CUDA KS final snapshot transfer accounting is incomplete");
  require(plan.read_final_state(token, true, snapshot, snapshot_detail) == VIBEQC_STATUS_SUCCESS &&
              snapshot.weighted_density.size() == spins,
          snapshot_detail);
  if (!restricted && atoms == 1)
    require(std::all_of(snapshot.weighted_density[1].begin(), snapshot.weighted_density[1].end(),
                        [](double value) { return value == 0.0; }),
            "empty beta occupation produced nonzero W");

  const std::vector<std::function<void(dft::CudaKsFinalStateToken&)>> stale_tokens{
      [](auto& value) { ++value.version; },
      [](auto& value) { ++value.identity.determinant.factor.basis; },
      [](auto& value) { ++value.identity.determinant.factor.reference; },
      [](auto& value) { ++value.identity.determinant.factor.orbital_generation; },
      [](auto& value) { ++value.identity.determinant.factor.density_generation; },
      [](auto& value) { ++value.identity.determinant.solve_epoch; },
      [](auto& value) { value.identity.determinant.model.screening_tolerance *= 2; },
      [](auto& value) { value.identity.determinant.occupied[0] = 0; },
      [](auto& value) { ++value.identity.model.owner; },
      [](auto& value) { ++value.identity.model.grid.radial_points; },
      [](auto& value) {
        value.identity.model.functional = (value.identity.model.functional + 1U) % 3U;
      },
      [](auto& value) { ++value.identity.model.tile_points; },
      [](auto& value) { ++value.identity.model.device; },
      [](auto& value) { ++value.identity.model.scf_domain_version; }};
  for (const auto& mutate : stale_tokens) {
    auto stale = token;
    mutate(stale);
    dft::VerifiedKsFinalState rejected;
    const auto before_rejection = plan.transfers();
    require(plan.read_final_state(stale, false, rejected, snapshot_detail) ==
                    VIBEQC_STATUS_INVALID_ARGUMENT &&
                rejected.density.empty() &&
                plan.transfers().final_state_d2h_bytes == before_rejection.final_state_d2h_bytes,
            "stale CUDA KS token transferred or published state");
  }
  plan.invalidate_final_state();
  require(plan.final_state_token(unavailable, snapshot_detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "explicit result invalidation preserved CUDA KS eligibility");
  require(plan.run(nullptr, true, false).converged &&
              plan.final_state_token(token, snapshot_detail) == VIBEQC_STATUS_SUCCESS,
          "CUDA KS owner did not recover eligibility after explicit invalidation");
  require(result.iterations == result.dft_diagnostic.history.size(),
          "missing CUDA iteration history");
  const auto before = plan.transfers();
  const auto warm = plan.run();
  const auto after = plan.transfers();
  require(warm.converged && warm.initial_density_used && warm.iterations <= result.iterations &&
              std::abs(warm.energy - result.energy) < 1e-11 &&
              before.density_h2d_bytes == after.density_h2d_bytes,
          "unchanged-geometry replay did not reuse resident warm density");
  dft::VerifiedKsFinalState stale_snapshot;
  require(plan.read_final_state(token, false, stale_snapshot, snapshot_detail) ==
                  VIBEQC_STATUS_INVALID_ARGUMENT &&
              stale_snapshot.density.empty(),
          "warm replay accepted the preceding CUDA KS solve epoch");
  const auto energy_only = plan.run(nullptr, true, false);
  require(energy_only.converged && energy_only.density.empty() &&
              plan.transfers().matrix_d2h_bytes == after.matrix_d2h_bytes,
          "energy-only CUDA KS exported a final density matrix");
  const auto frozen_density = plan.warm_density();
  plan.set_warm_start_updates(false);
  const auto frozen_before = plan.transfers();
  const auto frozen_result = plan.run(nullptr, false, false);
  require(frozen_result.converged && !frozen_result.initial_density_used &&
              plan.transfers().matrix_d2h_bytes == frozen_before.matrix_d2h_bytes,
          "frozen cold solve performed an implicit density export");
  require(plan.warm_density() == frozen_density,
          "successful frozen solve replaced the resident last-good density");
  plan.clear_warm_start();
  require(plan.warm_density().empty(), "cleared CUDA seed remains visible");
  require(plan.run(nullptr, true, false).converged && plan.warm_density().empty(),
          "frozen CUDA owner established a new seed");
  plan.set_warm_start_updates(true);
  require(plan.run(nullptr, false, false).converged && !plan.warm_density().empty(),
          "unfrozen CUDA owner failed to establish a seed");

  if (atoms > 1) {
    auto invalid = result.density;
    const auto matrix_size = static_cast<std::size_t>(atoms) * atoms;
    std::fill(invalid.begin(), invalid.begin() + matrix_size, 0.0);
    invalid[0] = -1.0;
    invalid[matrix_size - 1] = 2.0;
    const auto failed = plan.run(&invalid);
    require(plan.failed() && !failed.converged, "invalid grid density did not fail the CUDA item");
    require(plan.final_state_token(unavailable, snapshot_detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
            "failed CUDA KS solve preserved final-state eligibility");
    const auto recovered = plan.run();
    require(recovered.converged && std::abs(recovered.energy - result.energy) < 1e-11,
            "failed CUDA item replaced its last-good warm state");
    if (restricted && expect_iteration_chunking()) {
      const auto recovered_execution = plan.transfers();
      require(recovered_execution.execution_region_failures >= 1 &&
                  recovered_execution.execution_region_recoveries >= 1,
              "CUDA KS device region did not record failure/recovery ownership");
    }
    const auto moved = hydrogens(atoms, restricted, 0.2);
    const scf::PreparedFockPlan new_gpu(moved, nullptr,
                                        strategy(restricted, scf::FockBackend::Cuda), 0);
    const dft::AoBasis new_basis(moved);
    const dft::MolecularGrid new_grid(moved, grid_spec);
    bool stale = false;
    try {
      dft::CudaKsPlan wrong(new_gpu, new_basis, grid, options,
                            dft::semilocal_family_from_code(functional));
    } catch (const std::invalid_argument&) {
      stale = true;
    }
    require(stale, "CUDA SCF accepted a same-shape old grid");
    dft::CudaKsPlan changed(new_gpu, new_basis, new_grid, options,
                            dft::semilocal_family_from_code(functional));
    auto seed = plan.warm_density();
    for (auto& value : seed) value *= 1.3;
    const auto moved_warm = changed.run(&seed), moved_cold = changed.run(nullptr, false);
    require(moved_warm.converged && moved_cold.converged &&
                std::abs(moved_warm.energy - moved_cold.energy) < 1e-10,
            "changed-geometry warm normalization changed the endpoint");
  }
  options.max_iterations = 1;
  dft::CudaKsPlan unfinished(gpu, basis, grid, options,
                             dft::semilocal_family_from_code(functional));
  const auto limited = unfinished.run();
  require(!limited.converged && !unfinished.failed(), "iteration limit misreported its status");
  require(unfinished.warm_density().empty(), "unfinished solve published a good warm state");
  require(
      unfinished.final_state_token(unavailable, snapshot_detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
      "unfinished CUDA KS solve published a final-state token");
  physical_check(cpu, basis, grid, functional, limited);

  // Exact arena request is charged through the existing #203 device ledger.
  auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
  ledger->limit = plan.resources().state_device_bytes + plan.resources().xc_device_bytes;
  ledger->device = 0;
  const auto previous = runtime::active_device_resource_ledger;
  runtime::active_device_resource_ledger = ledger;
  try {
    {
      dft::CudaKsPlan measured(gpu, basis, grid, options,
                               dft::semilocal_family_from_code(functional), 257);
      require(ledger->live ==
                  measured.resources().state_device_bytes + measured.resources().xc_device_bytes,
              "CUDA KS resource request omits explicit allocations");
    }
    require(ledger->live == 0, "CUDA KS retained charged memory after destruction");
  } catch (...) {
    runtime::active_device_resource_ledger = previous;
    throw;
  }
  runtime::active_device_resource_ledger = previous;
  std::cout << "KS atoms=" << atoms << " restricted=" << restricted << " functional=" << functional
            << " iterations=" << result.iterations
            << " residual=" << result.dft_diagnostic.physical_residual << '\n';
}
/** Exercise the C validation layer, which can reject a request before the
 * prepared method's execute() invalidation is reached. */
void rejected_api_requests_revoke_tokens() {
  vibeqc_context_descriptor context_spec{sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, 0,
                                         VIBEQC_BACKEND_CUDA};
  vibeqc_context* raw_context{};
  require(vibeqc_context_create(&context_spec, &raw_context) == VIBEQC_STATUS_SUCCESS,
          "KS token API test context failed");
  std::unique_ptr<vibeqc_context, decltype(&vibeqc_context_destroy)> context(
      raw_context, vibeqc_context_destroy);
  vibeqc_system system{hydrogens(2, true)};
  vibeqc_method_descriptor method{sizeof(vibeqc_method_descriptor),
                                  VIBEQC_ABI_VERSION,
                                  VIBEQC_METHOD_LDA_RKS,
                                  150,
                                  8,
                                  1e-12,
                                  1e-10,
                                  1e-12,
                                  VIBEQC_DENSITY_FITTING_NONE,
                                  nullptr,
                                  1e-10,
                                  0};
  vibeqc_calculation* raw_calculation{};
  require(vibeqc_calculation_prepare(context.get(), &system, &method, &raw_calculation) ==
              VIBEQC_STATUS_SUCCESS,
          "KS token API test preparation failed");
  std::unique_ptr<vibeqc_calculation, decltype(&vibeqc_calculation_destroy)> calculation(
      raw_calculation, vibeqc_calculation_destroy);
  std::string detail;
  dft::CudaKsFinalStateToken token;
  dft::VerifiedKsFinalState snapshot;
  for (int rejected = 0; rejected < 3; ++rejected) {
    vibeqc_result_descriptor output{};
    output.struct_size = sizeof(output);
    output.abi_version = VIBEQC_ABI_VERSION;
    require(vibeqc_calculation_execute(calculation.get(), &output) == VIBEQC_STATUS_SUCCESS &&
                methods::detail::dft_final_state_token(*calculation->plan, token, detail) ==
                    VIBEQC_STATUS_SUCCESS,
            "KS calculation failed to publish current token");
    if (rejected == 1) ++output.abi_version;
    if (rejected == 2) output.force_count = 1;
    require(vibeqc_calculation_execute(calculation.get(), rejected == 0 ? nullptr : &output) !=
                VIBEQC_STATUS_SUCCESS,
            "malformed KS calculation unexpectedly executed");
    require(methods::detail::read_dft_final_state(*calculation->plan, token, false, snapshot,
                                                  detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
            "rejected C calculation request retained a previous token");
  }

  const vibeqc_system* systems[]{&system, &system};
  vibeqc_batch* raw_batch{};
  require(vibeqc_batch_prepare(context.get(), systems, 2, &method, VIBEQC_BATCH_ENABLE_WARM_STARTS,
                               &raw_batch) == VIBEQC_STATUS_SUCCESS,
          "KS token batch preparation failed");
  std::unique_ptr<vibeqc_batch, decltype(&vibeqc_batch_destroy)> batch(raw_batch,
                                                                       vibeqc_batch_destroy);
  for (int rejected = 0; rejected < 3; ++rejected) {
    vibeqc_batch_item_result_descriptor outputs[2]{};
    for (auto& output : outputs) {
      output.struct_size = sizeof(output);
      output.abi_version = VIBEQC_ABI_VERSION;
    }
    require(vibeqc_batch_execute(batch.get(), nullptr, 0, outputs, 2) == VIBEQC_STATUS_SUCCESS,
            "KS token batch execution failed");
    dft::CudaKsFinalStateToken tokens[2];
    for (std::size_t i = 0; i < 2; ++i)
      require(methods::detail::dft_final_state_token(*batch->plan, i, tokens[i], detail) ==
                  VIBEQC_STATUS_SUCCESS,
              "KS batch failed to publish current token");
    std::uint64_t metadata[16]{};
    vibeqc_ks_snapshot* raw_snapshot{};
    require(vibeqc_ks_snapshot_create_v1(batch.get(), 0, &raw_snapshot, metadata, 16) ==
                VIBEQC_STATUS_SUCCESS,
            "stationary bridge did not consume the native #162 handoff");
    std::unique_ptr<vibeqc_ks_snapshot, decltype(&vibeqc_ks_snapshot_destroy_v1)> proof(
        raw_snapshot, vibeqc_ks_snapshot_destroy_v1);
    std::vector<double> values(metadata[15], 79.0);
    require(metadata[0] == 3 && metadata[1] == 2 && metadata[2] == 1 &&
                vibeqc_ks_snapshot_copy_v1(batch.get(), proof.get(), values.data(),
                                           values.size()) == VIBEQC_STATUS_SUCCESS,
            "stationary bridge failed to export the native sources");
    require(values.size() >= 3 && values[values.size() - 3] > 0 && values[values.size() - 2] == 1 &&
                values[values.size() - 1] == 1,
            "CUDA KS snapshot v3 omitted measured export work");
    require(vibeqc_ks_snapshot_check_v1(batch.get(), proof.get()) == VIBEQC_STATUS_SUCCESS,
            "current stationary proof failed validation");
    if (rejected == 2) ++outputs[1].abi_version;
    require(vibeqc_batch_execute(batch.get(), nullptr, 0, rejected == 0 ? nullptr : outputs,
                                 rejected == 1 ? 1 : 2) != VIBEQC_STATUS_SUCCESS,
            "malformed KS batch unexpectedly executed");
    std::fill(values.begin(), values.end(), 79.0);
    require(
        vibeqc_ks_snapshot_check_v1(batch.get(), proof.get()) == VIBEQC_STATUS_INVALID_ARGUMENT &&
            vibeqc_ks_snapshot_copy_v1(batch.get(), proof.get(), values.data(), values.size()) ==
                VIBEQC_STATUS_INVALID_ARGUMENT &&
            std::all_of(values.begin(), values.end(), [](double v) { return v == 79.0; }),
        "revoked stationary proof validated or copied stale arrays");
    for (std::size_t i = 0; i < 2; ++i)
      require(methods::detail::read_dft_final_state(*batch->plan, i, tokens[i], false, snapshot,
                                                    detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
              "rejected C batch request retained a previous token");
  }
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    prepared_cuda_fock_seam();
    if (std::getenv("VIBEQC_CUDA_KS_CHUNK") == nullptr) {
      require(::setenv("VIBEQC_CUDA_KS_CHUNK", "2", 1) == 0,
              "could not enable CUDA RKS chunk qualification");
      for (bool pbe : {false, true}) {
        compare_rks_chunk_history(pbe);
        run_case(2, true, pbe);
      }
      require(::unsetenv("VIBEQC_CUDA_KS_CHUNK") == 0,
              "could not restore CUDA KS synchronization baseline");
    }
    rejected_api_requests_revoke_tokens();
    for (bool pbe : {false, true}) {
      run_case(2, true, pbe);
      run_hydroxyl(pbe);
      for (unsigned atoms : {1U, 2U, 3U}) run_case(atoms, false, pbe);
    }
    run_case(2, true, 2U);
    run_case(2, false, 2U);
    std::cout << "Native CUDA LDA/PBE/r2SCAN KS SCF, physical-state, warm/failure/resource gates "
                 "passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

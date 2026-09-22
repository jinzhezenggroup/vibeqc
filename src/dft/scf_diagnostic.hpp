#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace vibeqc::dft {

/** Physical energy terms, never the half-trace of an XC-containing Fock. */
struct EnergyComponents {
  double nuclear{}, one_electron{}, hartree{}, xc{}, exact_exchange{};
  double total() const noexcept { return nuclear + one_electron + hartree + xc + exact_exchange; }
};

/** A physical evaluation and its proposed density change. Each spin residual
 * and density update must pass separately; an empty spin cannot dilute RMS. */
struct ScfIteration {
  unsigned iteration{};
  EnergyComponents components;
  double energy_change{}, density_change{}, physical_residual{};
  std::array<double, 2> electrons{};
  /** Only the orbital proposal is shifted; the recorded physical terms are not. */
  bool occupation_stabilized{};
};

/** Run-local incremental-XC accounting. The experimental path never survives
 * a geometry/basis/grid replay, so every solve starts with a fresh anchor. */
struct IncrementalXcDiagnostic {
  std::uint32_t policy_version{1};
  /** Fresh per-solve identity minted only after geometry/basis/grid validation. */
  std::uint64_t model_identity{};
  bool enabled{};
  std::size_t full_builds{};
  std::size_t incremental_updates{};
  std::size_t periodic_rebuilds{};
  std::size_t drift_rebuilds{};
  std::size_t noise_rebuilds{};
  std::size_t stagnation_rebuilds{};
  std::size_t fallback_rebuilds{};
  std::size_t strict_final_builds{};
  std::size_t strict_refinement_iterations{};
  std::size_t final_audits{};
  std::size_t audit_failures{};
  std::uint64_t anchor_generation{};
  double max_anchor_delta_rms{};
  std::size_t retained_anchor_bytes{};
  std::size_t peak_update_buffer_bytes{};
  std::size_t peak_replacement_overlap_bytes{};
};

/** Method-owned diagnostics for one solve; history is reset for every replay.
 * Electron counts use Tr(D_s S), independently of quadrature electron counts.
 */
struct ScfDiagnostic {
  std::array<std::size_t, 2> occupations{};
  std::array<double, 2> electrons{};
  std::size_t grid_points{}, tile_points{}, ao_order{}, fock_builds{};
  std::uint32_t scf_domain_version{1};
  bool initial_density_used{};
  double density_change{};
  double physical_residual{std::numeric_limits<double>::infinity()};
  EnergyComponents components;
  IncrementalXcDiagnostic incremental_xc;
  std::vector<ScfIteration> history;
};
}  // namespace vibeqc::dft

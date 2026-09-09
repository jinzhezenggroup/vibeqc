#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vibeqc::scf {

/** Owned physical state, before DIIS. A callback receives a const, synchronous
 * view; copy the value to retain it. Each solve has a fresh generation, including
 * repeated solves of one geometry. UHF arrays contain alpha then beta blocks.
 * Occupations are electron counts per block, with maximum occupancy 2 for RHF
 * and 1 for UHF. The baseline is the traditional DIIS/Aufbau next density.
 */
struct ScfSnapshot {
  std::uint64_t generation{};
  unsigned iteration{};
  std::size_t nbf{};
  std::vector<unsigned> electrons;
  double occupation_weight{};
  std::vector<double> density, fock, residual, overlap, baseline;
  double energy{}, residual_rms{};
  std::size_t fock_builds{};
};

enum class ProposalRepresentation { none, ensemble_density, determinant_density, reset };

/** Densities must already satisfy symmetry, spin traces and metric occupation
 * bounds. Determinants additionally satisfy P S P = weight P. No rescaling or
 * orthogonalization is hidden in acceptance. Orbital/rotation adapters must
 * construct a valid density and report their own transformation cost.
 */
struct ScfProposal {
  ProposalRepresentation representation{ProposalRepresentation::none};
  std::uint64_t generation{};
  unsigned iteration{};
  std::vector<double> density;
};

enum class ProposalAction { baseline, accepted, damped, rejected, reset };
struct ProposalDecision {
  ProposalAction action{ProposalAction::baseline};
  std::string reason;
  unsigned trials{};
  double fraction{}, energy{}, residual_rms{};
  double inference_seconds{}, validation_seconds{}, operator_seconds{};
};

/** Optional, per-call CPU RHF/UHF instrumentation. The caller owns this object
 * until the synchronous solve returns. It is never retained by a plan or shared
 * between batch items. Missing proposals use the traditional path. Observation
 * exceptions abort the instrumented solve; proposal exceptions safely reset
 * DIIS. GPU/DFT integrations are separate capabilities, never implicit fallbacks.
 */
struct ScfHooks {
  std::function<ScfProposal(const ScfSnapshot&)> propose;
  std::function<void(const ScfSnapshot&, const ProposalDecision&)> observe;
};

}  // namespace vibeqc::scf

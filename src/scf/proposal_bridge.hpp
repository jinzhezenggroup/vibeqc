#pragma once

#include <cstddef>
#include <cstdint>

/** Private SOL01 host-only bridge, separate from the stable calculator C ABI.
 * All pointers are borrowed for the callback duration; adapters must copy them.
 * A snapshot contains spins*nbf*nbf matrix entries except overlap (nbf*nbf).
 */
struct ScfSnapshotViewV1 {
  std::uint64_t generation;
  unsigned iteration;
  std::size_t nbf, spins, fock_builds;
  const unsigned* electrons;
  double weight, energy, residual_rms;
  const double *density, *fock, *residual, *overlap, *baseline;
};

struct ScfDecisionViewV1 {
  int action;
  const char* reason;
  unsigned trials;
  double fraction, energy, residual_rms;
  double inference_seconds, validation_seconds, operator_seconds;
};

// Return representation 0:none, 1:ensemble, 2:determinant, 3:reset. The output
// has spins*nbf*nbf entries and must name the exact parent generation/iteration.
using ScfProposeV1 = int (*)(const ScfSnapshotViewV1*, double*, std::uint64_t*, unsigned*);
using ScfObserveV1 = void (*)(const ScfSnapshotViewV1*, const ScfDecisionViewV1*);

#pragma once

#include <vector>

namespace vibeqc::scf {

/** Owned scientific seed and its source geometry, never runtime/device state.
 * RHF stores the spin-summed density; UHF stores alpha then beta, row-major.
 * Diagnostics describe the source solve only and cannot establish convergence
 * of a new target. Coordinates follow the prepared plan's ordered nuclei.
 */
struct HfWarmState {
  std::vector<double> density;
  std::vector<double> coordinates;
  double energy{};
  double energy_change{};
  double density_rms{};
  int iterations{};
};

}  // namespace vibeqc::scf

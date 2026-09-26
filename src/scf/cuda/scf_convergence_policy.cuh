#pragma once

#include <math_constants.h>

#include <cmath>
#include <cstddef>

#include "scf/cuda/scf_constants.hpp"

namespace vibeqc::scf::cuda_execution {

/** Common FP64 HF energy comparison budget. Both quartet scatter and fitted
 * contractions/reductions lose a few representable values at stationarity.
 * This guard scales with energy, never with the requested density tolerance;
 * density and current physical-Fock stationarity must still pass separately.
 * An absent or nonfinite baseline cannot manufacture first-step convergence. */
__device__ inline double hf_energy_roundoff_guard(bool enabled, double energy,
                                                  double previous_energy) {
  if (!enabled || !isfinite(energy) || !isfinite(previous_energy)) return 0.0;
  const double scale = fmax(1.0, fmax(fabs(energy), fabs(previous_energy)));
  return kHfEnergyRoundoffFactor * kDoubleMachineEpsilon * scale;
}

/** Shared direct/DF stopping rule, independent of how a solver obtained its
 * baseline. The reported energy change remains the unmodified absolute delta.
 * A trusted warm baseline permits one step; a missing baseline requires a
 * second physical energy evaluation. NaNs/infinities fail every route closed. */
__device__ inline bool hf_iteration_converged(double energy, double previous_energy,
                                              double energy_tolerance, double density_tolerance,
                                              double density_rms, double physical_maximum,
                                              bool guard_roundoff = true) {
  return isfinite(energy) && isfinite(previous_energy) && isfinite(density_rms) &&
         isfinite(physical_maximum) &&
         fabs(energy - previous_energy) <
             energy_tolerance + hf_energy_roundoff_guard(guard_roundoff, energy, previous_energy) &&
         density_rms < density_tolerance && physical_maximum <= fmin(1e-8, density_tolerance);
}

/** One-warp maximum of the physical pre-DIIS FDS-SDF residual. Null is reserved
 * for compatibility callers without residual storage and coarse mixed stages;
 * production FP64 HF supplies the current residual. UHF passes both spins. */
__device__ inline double maximum_physical_residual(const double* values, std::size_t size) {
  if (values == nullptr) return 0.0;
  double maximum = 0.0;
  for (std::size_t element = threadIdx.x; element < size; element += blockDim.x)
    maximum = isfinite(values[element]) ? fmax(maximum, fabs(values[element])) : CUDART_INF;
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1)
    maximum = fmax(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
  return __shfl_sync(0xffffffffU, maximum, 0);
}

}  // namespace vibeqc::scf::cuda_execution

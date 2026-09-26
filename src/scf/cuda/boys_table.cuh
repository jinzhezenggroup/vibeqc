#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained Boys table recurrence, shared by direct and one-electron
// evaluators with the existing precision and small-argument branches.
namespace vibeqc::scf::cuda_execution {

template <unsigned MaximumOrder, typename Scalar>
__device__ inline void boys_values(Scalar argument, Scalar* values) {
  static_assert(MaximumOrder <= kMaximumCoulombOrder);
  if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    // The alternating high-order series loses too many digits in FP32 near
    // its branch boundary.  Evaluate the small special-function table in FP64
    // and round once before the substantially larger Cartesian recurrence.
    double accurate_values[MaximumOrder + 1];
    boys_values<MaximumOrder, double>(static_cast<double>(argument.value), accurate_values);
    for (unsigned order = 0; order <= MaximumOrder; ++order) {
      values[order] = MixedPrecisionFloat{accurate_values[order]};
    }
  } else {
    for (unsigned order = 0; order <= MaximumOrder; ++order) {
      values[order] = scalar<Scalar>(0.0);
    }
    // Low Boys orders tolerate upward recurrence much earlier than the generic
    // high-order path. Avoid a long alternating series for the dominant direct
    // s/p/d quartets once cancellation is bounded; the thresholds keep the
    // worst relative error below 4e-15 against a high-precision oracle.
    constexpr double series_threshold = MaximumOrder == 0   ? 1.0e-8
                                        : MaximumOrder == 1 ? 0.25
                                        : MaximumOrder == 2 ? 0.75
                                        : MaximumOrder == 3 ? 1.25
                                        : MaximumOrder == 4 ? 2.0
                                                            : 6.0;
    if (scalar_value(argument) < series_threshold) {
      // Evaluate only the highest requested Boys order by its convergent power
      // series. Lower orders follow from the stable downward recurrence, which
      // removes MaximumOrder duplicate series from every primitive quartet.
      Scalar term = scalar<Scalar>(1.0);
      Scalar sum = scalar<Scalar>(0.0);
      for (unsigned k = 0; k < 80; ++k) {
        sum = sum + term / static_cast<double>(2 * MaximumOrder + 2 * k + 1);
        term = term * (-1.0 * argument) / static_cast<double>(k + 1);
        if (fabs(scalar_value(term)) < 1.0e-18) break;
      }
      values[MaximumOrder] = sum;
      const Scalar exponential = qexp(-1.0 * argument);
      for (unsigned order = MaximumOrder; order > 0; --order) {
        values[order - 1] =
            (2.0 * argument * values[order] + exponential) / static_cast<double>(2 * order - 1);
      }
      return;
    }

    values[0] = 0.5 * qsqrt(scalar<Scalar>(kPi) / argument) * qerf(qsqrt(argument));
    const Scalar exponential = qexp(-1.0 * argument);
    for (unsigned order = 1; order <= MaximumOrder; ++order) {
      values[order] = ((2.0 * static_cast<double>(order) - 1.0) * values[order - 1] - exponential) /
                      (2.0 * argument);
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution

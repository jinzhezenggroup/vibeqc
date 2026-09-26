#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for shell pair hermite.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** Hermite workspace bounded by one exact shell-pair class. */
template <typename Scalar, unsigned FirstAngular, unsigned SecondAngular>
struct ShellPairHermiteCoefficients {
  static constexpr unsigned kIDimension = FirstAngular + 1;
  static constexpr unsigned kJDimension = SecondAngular + 1;
  // One zero boundary element is required because the recurrence reads t+1.
  static constexpr unsigned kTDimension = FirstAngular + SecondAngular + 2;
  Scalar data[kIDimension * kJDimension * kTDimension];

  __device__ inline Scalar& at(unsigned i, unsigned j, unsigned t) {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
  __device__ inline const Scalar& at(unsigned i, unsigned j, unsigned t) const {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
};

template <unsigned FirstAngular, unsigned SecondAngular, typename Scalar>
__device__ inline void fill_shell_pair_hermite(
    unsigned maximum_i, unsigned maximum_j, Scalar product, Scalar center_a, Scalar center_b,
    double alpha, double beta,
    ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>& coefficients) {
  static_assert(FirstAngular <= kMaximumAngularMomentum);
  static_assert(SecondAngular <= kMaximumAngularMomentum);
  for (unsigned item = 0;
       item < ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>::kIDimension *
                  ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>::kJDimension *
                  ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>::kTDimension;
       ++item) {
    coefficients.data[item] = scalar<Scalar>(0.0);
  }
  const double p = alpha + beta;
  const double mu = alpha * beta / p;
  const Scalar ab = center_a - center_b;
  coefficients.at(0, 0, 0) = qexp(-mu * ab * ab);
  const Scalar pa = product - center_a;
  const Scalar pb = product - center_b;
  const double inverse_two_p = 0.5 / p;

  for (unsigned i = 0; i <= maximum_i; ++i) {
    for (unsigned j = 0; j <= maximum_j; ++j) {
      if (i == 0 && j == 0) continue;
      if (i > 0) {
        coefficients.at(i, j, 0) = pa * coefficients.at(i - 1, j, 0) + coefficients.at(i - 1, j, 1);
      } else {
        coefficients.at(i, j, 0) = pb * coefficients.at(i, j - 1, 0) + coefficients.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coefficients.at(i, j, t) = pa * coefficients.at(i - 1, j, t) +
                                     inverse_two_p * coefficients.at(i - 1, j, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i - 1, j, t + 1);
        } else {
          coefficients.at(i, j, t) = pb * coefficients.at(i, j - 1, t) +
                                     inverse_two_p * coefficients.at(i, j - 1, t - 1) +
                                     static_cast<double>(t + 1) * coefficients.at(i, j - 1, t + 1);
        }
      }
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution

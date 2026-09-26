#pragma once

#include <cuda_runtime.h>

#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained Cartesian Hermite workspace and recurrence. The asymmetric
// dimensions include kinetic-operator raising and the zero recurrence boundary.
namespace vibeqc::scf::cuda_execution {

template <typename Scalar>
struct HermiteCoefficients {
  Scalar data[kHermiteIDimension * kHermiteJDimension * kHermiteTDimension];

  __device__ inline Scalar& at(unsigned i, unsigned j, unsigned t) {
    return data[(i * kHermiteJDimension + j) * kHermiteTDimension + t];
  }
  __device__ inline const Scalar& at(unsigned i, unsigned j, unsigned t) const {
    return data[(i * kHermiteJDimension + j) * kHermiteTDimension + t];
  }
};

template <typename Scalar>
__device__ inline void fill_hermite(unsigned maximum_i, unsigned maximum_j, Scalar product,
                                    Scalar center_a, Scalar center_b, double alpha, double beta,
                                    HermiteCoefficients<Scalar>& coefficients) {
  for (int item = 0; item < kHermiteIDimension * kHermiteJDimension * kHermiteTDimension; ++item) {
    coefficients.data[item] = scalar<Scalar>(0.0);
  }
  using Real = EvaluationReal<Scalar>;
  const Real alpha_value{alpha};
  const Real beta_value{beta};
  const Real p = alpha_value + beta_value;
  const Real mu = alpha_value * beta_value / p;
  const Scalar ab = center_a - center_b;
  coefficients.at(0, 0, 0) = qexp((-1.0 * mu) * ab * ab);
  const Scalar pa = product - center_a;
  const Scalar pb = product - center_b;
  const Real inverse_two_p = Real{0.5} / p;

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

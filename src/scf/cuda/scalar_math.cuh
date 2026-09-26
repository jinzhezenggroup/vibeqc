#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

// Shared device scalar/AD and special-function arithmetic. Ordinary inline
// linkage permits reuse across CUDA owners without forcing compiler inlining.
namespace vibeqc::scf::cuda_execution {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Dual {
  double value;
  double derivative;
};

/**
 * Forward-mode scalar carrying all Cartesian derivatives of one atom.
 *
 * Two-electron force kernels differentiate the same shell quartet along x,
 * y, and z. Propagating those components together avoids recomputing the
 * geometry-independent value recurrence three times for every center.
 */
struct Dual3 {
  double value;
  double derivative_x;
  double derivative_y;
  double derivative_z;
};

/**
 * FP32 evaluation scalar that prevents double literals from promoting the
 * mixed ERI recurrence back to FP64.
 *
 * Density reads, screening, shell contraction, Fock accumulation, and every
 * derivative path remain double precision.  This wrapper is deliberately
 * device-only so mixed precision cannot leak into the public or host ABI.
 */
struct MixedPrecisionFloat {
  float value;

  __device__ inline MixedPrecisionFloat() : value(0.0F) {}
  __device__ inline MixedPrecisionFloat(double input) : value(static_cast<float>(input)) {}
};

/** Use FP32 recurrence coefficients only for the mixed value evaluator. */
template <typename Scalar>
using EvaluationReal =
    std::conditional_t<std::is_same_v<Scalar, MixedPrecisionFloat>, MixedPrecisionFloat, double>;

__device__ __forceinline__ MixedPrecisionFloat operator+(MixedPrecisionFloat a,
                                                         MixedPrecisionFloat b) {
  return static_cast<double>(a.value + b.value);
}
__device__ __forceinline__ MixedPrecisionFloat operator-(MixedPrecisionFloat a,
                                                         MixedPrecisionFloat b) {
  return static_cast<double>(a.value - b.value);
}
__device__ __forceinline__ MixedPrecisionFloat operator*(MixedPrecisionFloat a,
                                                         MixedPrecisionFloat b) {
  return static_cast<double>(a.value * b.value);
}
__device__ __forceinline__ MixedPrecisionFloat operator/(MixedPrecisionFloat a,
                                                         MixedPrecisionFloat b) {
  return static_cast<double>(a.value / b.value);
}
__device__ __forceinline__ MixedPrecisionFloat operator*(MixedPrecisionFloat a, double b) {
  return a * MixedPrecisionFloat{b};
}
__device__ __forceinline__ MixedPrecisionFloat operator*(double a, MixedPrecisionFloat b) {
  return MixedPrecisionFloat{a} * b;
}

__device__ inline Dual operator+(Dual a, Dual b) {
  return {a.value + b.value, a.derivative + b.derivative};
}
__device__ inline Dual operator-(Dual a, Dual b) {
  return {a.value - b.value, a.derivative - b.derivative};
}
__device__ inline Dual operator*(Dual a, Dual b) {
  return {a.value * b.value, a.derivative * b.value + a.value * b.derivative};
}
__device__ inline Dual operator/(Dual a, Dual b) {
  const double inverse_square = 1.0 / (b.value * b.value);
  return {a.value / b.value, (a.derivative * b.value - a.value * b.derivative) * inverse_square};
}
__device__ inline Dual operator*(double a, Dual b) { return Dual{a, 0.0} * b; }
__device__ inline Dual operator/(Dual a, double b) { return a / Dual{b, 0.0}; }
__device__ inline Dual operator/(double a, Dual b) { return Dual{a, 0.0} / b; }

__device__ inline Dual3 operator+(Dual3 a, Dual3 b) {
  return {a.value + b.value, a.derivative_x + b.derivative_x, a.derivative_y + b.derivative_y,
          a.derivative_z + b.derivative_z};
}
__device__ inline Dual3 operator-(Dual3 a, Dual3 b) {
  return {a.value - b.value, a.derivative_x - b.derivative_x, a.derivative_y - b.derivative_y,
          a.derivative_z - b.derivative_z};
}
__device__ inline Dual3 operator*(Dual3 a, Dual3 b) {
  return {
      a.value * b.value,
      a.derivative_x * b.value + a.value * b.derivative_x,
      a.derivative_y * b.value + a.value * b.derivative_y,
      a.derivative_z * b.value + a.value * b.derivative_z,
  };
}
__device__ inline Dual3 operator/(Dual3 a, Dual3 b) {
  const double inverse_square = 1.0 / (b.value * b.value);
  return {
      a.value / b.value,
      (a.derivative_x * b.value - a.value * b.derivative_x) * inverse_square,
      (a.derivative_y * b.value - a.value * b.derivative_y) * inverse_square,
      (a.derivative_z * b.value - a.value * b.derivative_z) * inverse_square,
  };
}
__device__ inline Dual3 operator*(double a, Dual3 b) { return Dual3{a, 0.0, 0.0, 0.0} * b; }
__device__ inline Dual3 operator/(Dual3 a, double b) { return a / Dual3{b, 0.0, 0.0, 0.0}; }

template <typename Scalar>
__device__ inline Scalar scalar(double value, double derivative = 0.0) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    return {value, derivative};
  } else if constexpr (std::is_same_v<Scalar, Dual3>) {
    (void)derivative;
    return {value, 0.0, 0.0, 0.0};
  } else {
    (void)derivative;
    return value;
  }
}

template <typename Scalar>
__device__ inline double scalar_value(Scalar value) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    return value.value;
  } else if constexpr (std::is_same_v<Scalar, Dual3>) {
    return value.value;
  } else if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    return static_cast<double>(value.value);
  } else {
    return value;
  }
}

template <typename Scalar>
__device__ inline Scalar qexp(Scalar value) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    const double result = exp(value.value);
    return {result, result * value.derivative};
  } else if constexpr (std::is_same_v<Scalar, Dual3>) {
    const double result = exp(value.value);
    return {result, result * value.derivative_x, result * value.derivative_y,
            result * value.derivative_z};
  } else if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    return static_cast<double>(expf(value.value));
  } else {
    return exp(value);
  }
}

template <typename Scalar>
__device__ inline Scalar qsqrt(Scalar value) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    const double result = sqrt(value.value);
    return {result, 0.5 * value.derivative / result};
  } else if constexpr (std::is_same_v<Scalar, Dual3>) {
    const double result = sqrt(value.value);
    const double scale = 0.5 / result;
    return {result, scale * value.derivative_x, scale * value.derivative_y,
            scale * value.derivative_z};
  } else if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    return static_cast<double>(sqrtf(value.value));
  } else {
    return sqrt(value);
  }
}

template <typename Scalar>
__device__ inline Scalar qerf(Scalar value) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    const double result = erf(value.value);
    const double factor = 2.0 / sqrt(kPi) * exp(-value.value * value.value);
    return {result, factor * value.derivative};
  } else if constexpr (std::is_same_v<Scalar, Dual3>) {
    const double result = erf(value.value);
    const double factor = 2.0 / sqrt(kPi) * exp(-value.value * value.value);
    return {result, factor * value.derivative_x, factor * value.derivative_y,
            factor * value.derivative_z};
  } else if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    return static_cast<double>(erff(value.value));
  } else {
    return erf(value);
  }
}

template <typename Scalar>
__device__ inline Scalar boys0(Scalar x) {
  if (scalar_value(x) < 1.0e-8) {
    const Scalar x2 = x * x;
    const Scalar x3 = x2 * x;
    const Scalar x4 = x3 * x;
    return scalar<Scalar>(1.0) - x / 3.0 + x2 / 10.0 - x3 / 42.0 + x4 / 216.0;
  }
  return 0.5 * qsqrt(scalar<Scalar>(kPi) / x) * qerf(qsqrt(x));
}

}  // namespace vibeqc::scf::cuda_execution

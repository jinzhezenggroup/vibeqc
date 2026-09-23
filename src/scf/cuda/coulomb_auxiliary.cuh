#pragma once

#include <cuda_runtime.h>

#include "integrals/range_moments.hpp"
#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/gaussian_geometry.cuh"

// Retained Coulomb auxiliary recurrence, bounded by the instantiated
// angular order. Layout assertions guard the existing workspace contract.
namespace vibeqc::scf::cuda_execution {

template <typename Scalar, unsigned MaximumAngular>
struct CoulombAuxiliary {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);

  __host__ __device__ static constexpr unsigned choose3(unsigned value) {
    return value < 3 ? 0 : value * (value - 1) * (value - 2) / 6;
  }

  __host__ __device__ static constexpr unsigned choose4(unsigned value) {
    return value < 4 ? 0 : value * (value - 1) * (value - 2) * (value - 3) / 24;
  }

  static constexpr unsigned kStateCount = choose4(MaximumAngular + 4);
  Scalar data[kStateCount];

  __device__ inline unsigned index(unsigned n, unsigned t, unsigned u, unsigned v) const {
    const unsigned total = choose4(MaximumAngular + 4);
    const unsigned n_offset = total - choose4(MaximumAngular - n + 4);
    const unsigned remaining_after_n = MaximumAngular - n;
    const unsigned t_offset = choose3(remaining_after_n + 3) - choose3(remaining_after_n - t + 3);
    const unsigned remaining_after_t = remaining_after_n - t;
    const unsigned u_offset = u * (remaining_after_t + 1) - u * (u - 1) / 2;
    return n_offset + t_offset + u_offset + v;
  }

  __device__ inline Scalar& at(unsigned n, unsigned t, unsigned u, unsigned v) {
    return data[index(n, t, u, v)];
  }
  __device__ inline const Scalar& at(unsigned n, unsigned t, unsigned u, unsigned v) const {
    return data[index(n, t, u, v)];
  }
};

// Keep each angular specialization at the exact simplex size so lower-order
// work initializes and indexes only the states its recurrence can reach.
static_assert(CoulombAuxiliary<double, 0>::kStateCount == 1);
static_assert(CoulombAuxiliary<double, 1>::kStateCount == 5);
static_assert(CoulombAuxiliary<double, 2>::kStateCount == 15);
static_assert(CoulombAuxiliary<double, 6>::kStateCount == 210);
static_assert(CoulombAuxiliary<double, 12>::kStateCount == 1820);

template <unsigned MaximumAngular, typename Scalar>
__device__ inline void fill_coulomb_from_moments(
    EvaluationReal<Scalar> exponent, const Vec3<Scalar>& pc, const Scalar* boys,
    CoulombAuxiliary<Scalar, MaximumAngular>& auxiliary) {
  for (unsigned item = 0; item < CoulombAuxiliary<Scalar, MaximumAngular>::kStateCount; ++item) {
    auxiliary.data[item] = scalar<Scalar>(0.0);
  }
  EvaluationReal<Scalar> factor{1.0};
  for (unsigned n = 0; n <= MaximumAngular; ++n) {
    auxiliary.at(n, 0, 0, 0) = factor * boys[n];
    factor = factor * (-2.0 * exponent);
  }

  for (unsigned v = 1; v <= MaximumAngular; ++v) {
    for (unsigned n = 0; n + v <= MaximumAngular; ++n) {
      Scalar value = pc.z * auxiliary.at(n + 1, 0, 0, v - 1);
      if (v > 1) {
        value = value + static_cast<double>(v - 1) * auxiliary.at(n + 1, 0, 0, v - 2);
      }
      auxiliary.at(n, 0, 0, v) = value;
    }
  }
  for (unsigned v = 0; v <= MaximumAngular; ++v) {
    for (unsigned u = 1; u + v <= MaximumAngular; ++u) {
      for (unsigned n = 0; n + u + v <= MaximumAngular; ++n) {
        Scalar value = pc.y * auxiliary.at(n + 1, 0, u - 1, v);
        if (u > 1) {
          value = value + static_cast<double>(u - 1) * auxiliary.at(n + 1, 0, u - 2, v);
        }
        auxiliary.at(n, 0, u, v) = value;
      }
    }
  }
  for (unsigned v = 0; v <= MaximumAngular; ++v) {
    for (unsigned u = 0; u + v <= MaximumAngular; ++u) {
      for (unsigned t = 1; t + u + v <= MaximumAngular; ++t) {
        for (unsigned n = 0; n + t + u + v <= MaximumAngular; ++n) {
          Scalar value = pc.x * auxiliary.at(n + 1, t - 1, u, v);
          if (t > 1) {
            value = value + static_cast<double>(t - 1) * auxiliary.at(n + 1, t - 2, u, v);
          }
          auxiliary.at(n, t, u, v) = value;
        }
      }
    }
  }
}

template <unsigned MaximumAngular, typename Scalar>
__device__ inline void fill_coulomb(EvaluationReal<Scalar> exponent, const Vec3<Scalar>& product,
                                    const Vec3<Scalar>& center,
                                    CoulombAuxiliary<Scalar, MaximumAngular>& auxiliary) {
  const Vec3<Scalar> pc{product.x - center.x, product.y - center.y, product.z - center.z};
  Scalar boys[MaximumAngular + 1];
  boys_values<MaximumAngular>(exponent * distance_squared(product, center), boys);
  fill_coulomb_from_moments<MaximumAngular>(exponent, pc, boys, auxiliary);
}

/** Range-separated direct exchange reuses the same Hermite recurrence as the
 * full-Coulomb path. Only the radial moment source changes, through the common
 * CPU/CUDA range_moments primitive. */
template <unsigned MaximumAngular>
__device__ inline bool fill_coulomb_range(
    double exponent, const Vec3<double>& product, const Vec3<double>& center,
    vibeqc::integrals::CoulombRange range, double omega,
    CoulombAuxiliary<double, MaximumAngular>& auxiliary) {
  double boys[MaximumAngular + 1];
  const double argument = exponent * distance_squared(product, center);
  if (!vibeqc::integrals::bounded_range_moments<MaximumAngular>(
          MaximumAngular, argument, exponent, range, omega, boys))
    return false;
  const Vec3<double> pc{product.x - center.x, product.y - center.y, product.z - center.z};
  fill_coulomb_from_moments<MaximumAngular>(exponent, pc, boys, auxiliary);
  return true;
}

}  // namespace vibeqc::scf::cuda_execution

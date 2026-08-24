#include "scf/rhf.hpp"

#include "molecule/basis.hpp"
#include "scf/direct_task_layout.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace qce::scf {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kMaximumAngularMomentum = 3;
constexpr std::size_t kMaximumAoExpansionTerms =
    molecule::kMaximumAoExpansionTerms;
constexpr int kHermiteIDimension = kMaximumAngularMomentum + 1;
constexpr int kHermiteJDimension = kMaximumAngularMomentum + 3;
constexpr int kHermiteTDimension = 2 * kMaximumAngularMomentum + 4;
constexpr int kMaximumCoulombOrder = 4 * kMaximumAngularMomentum;
// Small fixed-topology fleet buckets benefit from evaluating ERIs once and
// replaying them from the persistent arena. Larger AO spaces switch to fused
// direct J/K so device memory remains O(N^2), not O(N^4).
constexpr std::size_t kPersistentEriAoLimit = 16;
// Below the persistent-ERI boundary, one lightweight kernel avoids cuBLAS
// launch overhead. Production direct-J/K workloads use batched GEMM.
constexpr std::size_t kCublasMatrixProductAoThreshold = 17;
// Capture-safe scalar kernels are small or register-heavy and use one warp per
// block. Direct quartets keep their separately documented virtual tiling.
constexpr unsigned kCaptureSafeKernelThreads = 32;

/** Geometry-dependent direct-J/K work emitted by shell-bound compaction. */
struct ActiveShellQuartetTile {
  std::uint32_t first_pair;
  std::uint32_t second_pair;
  std::uint32_t tile;
};

static_assert(sizeof(ActiveShellQuartetTile) == 3 * sizeof(std::uint32_t));

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

__device__ Dual operator+(Dual a, Dual b) {
  return {a.value + b.value, a.derivative + b.derivative};
}
__device__ Dual operator-(Dual a, Dual b) {
  return {a.value - b.value, a.derivative - b.derivative};
}
__device__ Dual operator*(Dual a, Dual b) {
  return {a.value * b.value,
          a.derivative * b.value + a.value * b.derivative};
}
__device__ Dual operator/(Dual a, Dual b) {
  const double inverse_square = 1.0 / (b.value * b.value);
  return {a.value / b.value,
          (a.derivative * b.value - a.value * b.derivative) * inverse_square};
}
__device__ Dual operator-(double a, Dual b) { return Dual{a, 0.0} - b; }
__device__ Dual operator*(double a, Dual b) { return Dual{a, 0.0} * b; }
__device__ Dual operator/(Dual a, double b) { return a / Dual{b, 0.0}; }
__device__ Dual operator/(double a, Dual b) { return Dual{a, 0.0} / b; }

__device__ Dual3 operator+(Dual3 a, Dual3 b) {
  return {a.value + b.value, a.derivative_x + b.derivative_x,
          a.derivative_y + b.derivative_y,
          a.derivative_z + b.derivative_z};
}
__device__ Dual3 operator-(Dual3 a, Dual3 b) {
  return {a.value - b.value, a.derivative_x - b.derivative_x,
          a.derivative_y - b.derivative_y,
          a.derivative_z - b.derivative_z};
}
__device__ Dual3 operator*(Dual3 a, Dual3 b) {
  return {
      a.value * b.value,
      a.derivative_x * b.value + a.value * b.derivative_x,
      a.derivative_y * b.value + a.value * b.derivative_y,
      a.derivative_z * b.value + a.value * b.derivative_z,
  };
}
__device__ Dual3 operator/(Dual3 a, Dual3 b) {
  const double inverse_square = 1.0 / (b.value * b.value);
  return {
      a.value / b.value,
      (a.derivative_x * b.value - a.value * b.derivative_x) * inverse_square,
      (a.derivative_y * b.value - a.value * b.derivative_y) * inverse_square,
      (a.derivative_z * b.value - a.value * b.derivative_z) * inverse_square,
  };
}
__device__ Dual3 operator*(double a, Dual3 b) {
  return Dual3{a, 0.0, 0.0, 0.0} * b;
}
__device__ Dual3 operator/(Dual3 a, double b) {
  return a / Dual3{b, 0.0, 0.0, 0.0};
}

template <typename Scalar>
__device__ Scalar scalar(double value, double derivative = 0.0) {
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
__device__ double scalar_value(Scalar value) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    return value.value;
  } else if constexpr (std::is_same_v<Scalar, Dual3>) {
    return value.value;
  } else {
    return value;
  }
}

template <typename Scalar>
__device__ Scalar qexp(Scalar value) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    const double result = exp(value.value);
    return {result, result * value.derivative};
  } else if constexpr (std::is_same_v<Scalar, Dual3>) {
    const double result = exp(value.value);
    return {result, result * value.derivative_x,
            result * value.derivative_y, result * value.derivative_z};
  } else {
    return exp(value);
  }
}

template <typename Scalar>
__device__ Scalar qsqrt(Scalar value) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    const double result = sqrt(value.value);
    return {result, 0.5 * value.derivative / result};
  } else if constexpr (std::is_same_v<Scalar, Dual3>) {
    const double result = sqrt(value.value);
    const double scale = 0.5 / result;
    return {result, scale * value.derivative_x,
            scale * value.derivative_y, scale * value.derivative_z};
  } else {
    return sqrt(value);
  }
}

template <typename Scalar>
__device__ Scalar qerf(Scalar value) {
  if constexpr (std::is_same_v<Scalar, Dual>) {
    const double result = erf(value.value);
    const double factor = 2.0 / sqrt(kPi) * exp(-value.value * value.value);
    return {result, factor * value.derivative};
  } else if constexpr (std::is_same_v<Scalar, Dual3>) {
    const double result = erf(value.value);
    const double factor = 2.0 / sqrt(kPi) * exp(-value.value * value.value);
    return {result, factor * value.derivative_x,
            factor * value.derivative_y, factor * value.derivative_z};
  } else {
    return erf(value);
  }
}

template <typename Scalar>
__device__ Scalar boys0(Scalar x) {
  if (scalar_value(x) < 1.0e-8) {
    const Scalar x2 = x * x;
    const Scalar x3 = x2 * x;
    const Scalar x4 = x3 * x;
    return scalar<Scalar>(1.0) - x / 3.0 + x2 / 10.0 - x3 / 42.0 + x4 / 216.0;
  }
  return 0.5 * qsqrt(scalar<Scalar>(kPi) / x) * qerf(qsqrt(x));
}

template <typename Scalar>
struct Vec3 {
  Scalar x;
  Scalar y;
  Scalar z;
};

struct DeviceBatch {
  std::int32_t batch_size;
  std::int32_t nbf;
  // Direct shell quartets always use normalized Cartesian source AOs. This
  // equals nbf for Cartesian public bases and is larger for spherical d/f.
  std::int32_t direct_nbf;
  std::int64_t total_atoms;
  std::int64_t total_shells;
  std::int64_t total_shell_pairs;
  std::int64_t total_shell_quartets;
  const std::int64_t* atom_offsets;
  const std::int32_t* atom_systems;
  const std::int32_t* atomic_numbers;
  const double* positions;
  const std::int64_t* system_shell_offsets;
  const std::int32_t* shell_atoms;
  const std::uint8_t* shell_angular;
  const std::int64_t* shell_ao_offsets;
  const std::int64_t* shell_direct_ao_offsets;
  const std::int64_t* shell_primitive_offsets;
  const std::int64_t* system_shell_pair_offsets;
  const std::int64_t* system_shell_quartet_offsets;
  const std::int32_t* shell_pair_systems;
  const std::int32_t* shell_pair_first;
  const std::int32_t* shell_pair_second;
  // Every target AO refers back to one physical shell and carries up to three
  // normalized Cartesian expansion terms. Cartesian AOs use one term; real
  // spherical d/f AOs use the sparse solid-harmonic combinations.
  const std::int32_t* ao_shells;
  const std::uint8_t* ao_term_counts;
  const std::uint8_t* ao_term_angular;
  const double* ao_term_coefficients;
  const std::int32_t* direct_ao_shells;
  const std::uint8_t* direct_ao_angular;
  const double* direct_ao_coefficients;
  // Column-major C with public AO rows and Cartesian source AO columns:
  // phi_public = C * phi_cartesian.
  const double* ao_to_direct_transform;
  const double* primitive_exponents;
  const double* primitive_coefficients;
  const std::int32_t* occupied;
};

__device__ std::size_t matrix_index(std::size_t row,
                                    std::size_t column,
                                    std::size_t n) {
  // CUDA dense matrices are column-major so they can be submitted directly to
  // cuSOLVER without iteration-level transposes.
  return row + column * n;
}

__device__ std::size_t eri_index(std::size_t i,
                                 std::size_t j,
                                 std::size_t k,
                                 std::size_t l,
                                 std::size_t n) {
  return ((i * n + j) * n + k) * n + l;
}

__device__ void decode_lower_triangle(std::size_t packed,
                                      std::size_t& first,
                                      std::size_t& second) {
  first = static_cast<std::size_t>(
      0.5 * (sqrt(8.0 * static_cast<double>(packed) + 1.0) - 1.0));
  while ((first + 1) * (first + 2) / 2 <= packed) ++first;
  while (first * (first + 1) / 2 > packed) --first;
  second = packed - first * (first + 1) / 2;
}

__device__ std::int32_t shell_quartet_system(const DeviceBatch& batch,
                                             std::size_t quartet) {
  std::int32_t lower = 0;
  std::int32_t upper = batch.batch_size;
  while (lower + 1 < upper) {
    const std::int32_t middle = lower + (upper - lower) / 2;
    if (static_cast<std::size_t>(batch.system_shell_quartet_offsets[middle]) <=
        quartet) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

__device__ std::size_t shell_ao_pair_count(const DeviceBatch& batch,
                                           std::size_t shell_pair) {
  const std::int32_t first_shell = batch.shell_pair_first[shell_pair];
  const std::int32_t second_shell = batch.shell_pair_second[shell_pair];
  const std::size_t first_count =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[first_shell + 1] -
                               batch.shell_direct_ao_offsets[first_shell]);
  const std::size_t second_count =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[second_shell + 1] -
                               batch.shell_direct_ao_offsets[second_shell]);
  return first_shell == second_shell
      ? first_count * (first_count + 1) / 2
      : first_count * second_count;
}

__device__ void decode_shell_ao_pair(const DeviceBatch& batch,
                                     std::size_t shell_pair,
                                     std::size_t ordinal,
                                     std::size_t system_ao_begin,
                                     std::size_t& first,
                                     std::size_t& second) {
  const std::int32_t first_shell = batch.shell_pair_first[shell_pair];
  const std::int32_t second_shell = batch.shell_pair_second[shell_pair];
  const std::size_t first_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[first_shell]);
  const std::size_t second_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[second_shell]);
  const std::size_t second_count =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[second_shell + 1]) -
      second_begin;
  std::size_t first_component = 0;
  std::size_t second_component = 0;
  if (first_shell == second_shell) {
    decode_lower_triangle(ordinal, first_component, second_component);
  } else {
    first_component = ordinal / second_count;
    second_component = ordinal % second_count;
  }
  first = first_begin + first_component - system_ao_begin;
  second = second_begin + second_component - system_ao_begin;
}

template <typename Scalar>
__device__ Vec3<Scalar> atom_position(const DeviceBatch& batch,
                                      std::int64_t atom,
                                      std::int64_t derivative_coordinate) {
  const std::int64_t base = atom * 3;
  if constexpr (std::is_same_v<Scalar, Dual3>) {
    // Any coordinate belonging to the requested atom denotes the combined
    // x/y/z seed. Negative coordinates continue to mean value-only mode.
    const bool differentiated = derivative_coordinate >= 0 &&
        derivative_coordinate / 3 == atom;
    return {
        {batch.positions[base], differentiated ? 1.0 : 0.0, 0.0, 0.0},
        {batch.positions[base + 1], 0.0, differentiated ? 1.0 : 0.0, 0.0},
        {batch.positions[base + 2], 0.0, 0.0,
         differentiated ? 1.0 : 0.0},
    };
  } else {
    return {
        scalar<Scalar>(batch.positions[base],
                       derivative_coordinate == base ? 1.0 : 0.0),
        scalar<Scalar>(batch.positions[base + 1],
                       derivative_coordinate == base + 1 ? 1.0 : 0.0),
        scalar<Scalar>(batch.positions[base + 2],
                       derivative_coordinate == base + 2 ? 1.0 : 0.0),
    };
  }
}

template <typename Scalar>
__device__ Scalar distance_squared(const Vec3<Scalar>& first,
                                   const Vec3<Scalar>& second) {
  const Scalar dx = first.x - second.x;
  const Scalar dy = first.y - second.y;
  const Scalar dz = first.z - second.z;
  return dx * dx + dy * dy + dz * dz;
}

template <typename Scalar>
__device__ Vec3<Scalar> product_center(double alpha,
                                      const Vec3<Scalar>& first,
                                      double beta,
                                      const Vec3<Scalar>& second) {
  const double exponent = alpha + beta;
  return {(alpha * first.x + beta * second.x) / exponent,
          (alpha * first.y + beta * second.y) / exponent,
          (alpha * first.z + beta * second.z) / exponent};
}

template <typename Scalar>
__device__ Scalar primitive_overlap(double alpha,
                                    const Vec3<Scalar>& first,
                                    double beta,
                                    const Vec3<Scalar>& second) {
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  return pow(kPi / exponent, 1.5) *
         qexp(-reduced * distance_squared(first, second));
}

template <typename Scalar>
__device__ Scalar primitive_kinetic(double alpha,
                                    const Vec3<Scalar>& first,
                                    double beta,
                                    const Vec3<Scalar>& second) {
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  const Scalar squared_distance = distance_squared(first, second);
  return reduced * (3.0 - 2.0 * reduced * squared_distance) *
         primitive_overlap(alpha, first, beta, second);
}

template <typename Scalar>
__device__ Scalar primitive_nuclear_attraction(
    const DeviceBatch& batch,
    std::int32_t system,
    double alpha,
    const Vec3<Scalar>& first,
    double beta,
    const Vec3<Scalar>& second,
    std::int64_t derivative_coordinate) {
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  const Vec3<Scalar> center = product_center(alpha, first, beta, second);
  const Scalar prefactor =
      (2.0 * kPi / exponent) *
      qexp(-reduced * distance_squared(first, second));
  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t atom = batch.atom_offsets[system];
       atom < batch.atom_offsets[system + 1]; ++atom) {
    const Scalar argument =
        exponent * distance_squared(center, atom_position<Scalar>(
                                                batch, atom, derivative_coordinate));
    result = result - static_cast<double>(batch.atomic_numbers[atom]) *
                          prefactor * boys0(argument);
  }
  return result;
}

template <typename Scalar>
__device__ Scalar primitive_eri(double alpha,
                                const Vec3<Scalar>& first,
                                double beta,
                                const Vec3<Scalar>& second,
                                double gamma,
                                const Vec3<Scalar>& third,
                                double delta,
                                const Vec3<Scalar>& fourth) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const Vec3<Scalar> center_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> center_q = product_center(gamma, third, delta, fourth);
  const double rho = p * q / (p + q);
  const double prefactor =
      2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));
  return prefactor *
         qexp(-mu * distance_squared(first, second) -
              nu * distance_squared(third, fourth)) *
         boys0(rho * distance_squared(center_p, center_q));
}

struct Angular {
  unsigned x;
  unsigned y;
  unsigned z;
};

__device__ unsigned angular_axis(const Angular& angular, int axis) {
  return axis == 0 ? angular.x : (axis == 1 ? angular.y : angular.z);
}

__device__ void add_angular_axis(Angular& angular, int axis, int delta) {
  unsigned* value = axis == 0 ? &angular.x : (axis == 1 ? &angular.y : &angular.z);
  *value = static_cast<unsigned>(static_cast<int>(*value) + delta);
}

__device__ unsigned angular_total(const Angular& angular) {
  return angular.x + angular.y + angular.z;
}

/** Match the host planner's symmetry-reduced s/p/d/f shell-class encoding. */
__host__ __device__ constexpr unsigned direct_triangular_class_high(
    unsigned index) {
  unsigned high = 0;
  while ((high + 1) * (high + 2) / 2 <= index) ++high;
  return high;
}

/** Resolve one class template to its exact Coulomb recurrence order. */
__host__ __device__ constexpr unsigned direct_shell_class_angular_order(
    unsigned shell_class) {
  const unsigned first_pair = direct_triangular_class_high(shell_class);
  const unsigned second_pair =
      shell_class - first_pair * (first_pair + 1) / 2;
  const unsigned first_high = direct_triangular_class_high(first_pair);
  const unsigned first_low =
      first_pair - first_high * (first_high + 1) / 2;
  const unsigned second_high = direct_triangular_class_high(second_pair);
  const unsigned second_low =
      second_pair - second_high * (second_high + 1) / 2;
  return first_high + first_low + second_high + second_low;
}

__host__ __device__ constexpr unsigned direct_shell_pair_class_cuda(
    unsigned first,
    unsigned second) {
  const unsigned high = first > second ? first : second;
  const unsigned low = first > second ? second : first;
  return high * (high + 1) / 2 + low;
}

__device__ unsigned direct_quartet_shell_class_device(
    unsigned first,
    unsigned second,
    unsigned third,
    unsigned fourth) {
  const unsigned first_pair = direct_shell_pair_class_cuda(first, second);
  const unsigned second_pair = direct_shell_pair_class_cuda(third, fourth);
  const unsigned high_pair = max(first_pair, second_pair);
  const unsigned low_pair = min(first_pair, second_pair);
  return high_pair * (high_pair + 1) / 2 + low_pair;
}

__device__ bool is_s_function(const Angular& angular) {
  return angular_total(angular) == 0;
}

__device__ Angular ao_angular(const DeviceBatch& batch,
                              std::int64_t ao,
                              unsigned term = 0) {
  const std::size_t offset =
      (static_cast<std::size_t>(ao) * kMaximumAoExpansionTerms + term) * 3;
  return {batch.ao_term_angular[offset], batch.ao_term_angular[offset + 1],
          batch.ao_term_angular[offset + 2]};
}

__device__ double ao_term_coefficient(const DeviceBatch& batch,
                                      std::int64_t ao,
                                      unsigned term) {
  return batch.ao_term_coefficients[
      static_cast<std::size_t>(ao) * kMaximumAoExpansionTerms + term];
}

__device__ Angular direct_ao_angular(const DeviceBatch& batch,
                                     std::int64_t ao) {
  const std::size_t offset = static_cast<std::size_t>(ao) * 3;
  return {batch.direct_ao_angular[offset], batch.direct_ao_angular[offset + 1],
          batch.direct_ao_angular[offset + 2]};
}

template <typename Scalar>
__device__ Scalar vec_axis(const Vec3<Scalar>& vector, int axis) {
  return axis == 0 ? vector.x : (axis == 1 ? vector.y : vector.z);
}

template <unsigned MaximumOrder, typename Scalar>
__device__ void boys_values(Scalar argument, Scalar* values) {
  static_assert(MaximumOrder <= kMaximumCoulombOrder);
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = scalar<Scalar>(0.0);
  }
  if (scalar_value(argument) < 6.0) {
    for (unsigned order = 0; order <= MaximumOrder; ++order) {
      Scalar term = scalar<Scalar>(1.0);
      Scalar sum = scalar<Scalar>(0.0);
      for (unsigned k = 0; k < 80; ++k) {
        sum = sum + term / static_cast<double>(2 * order + 2 * k + 1);
        term = term * (-1.0 * argument) / static_cast<double>(k + 1);
        if (fabs(scalar_value(term)) < 1.0e-18) break;
      }
      values[order] = sum;
    }
    return;
  }

  values[0] = 0.5 * qsqrt(scalar<Scalar>(kPi) / argument) *
              qerf(qsqrt(argument));
  const Scalar exponential = qexp(-1.0 * argument);
  for (unsigned order = 1; order <= MaximumOrder; ++order) {
    values[order] =
        ((2.0 * static_cast<double>(order) - 1.0) * values[order - 1] -
         exponential) /
        (2.0 * argument);
  }
}

template <typename Scalar>
struct HermiteCoefficients {
  Scalar data[kHermiteIDimension * kHermiteJDimension * kHermiteTDimension];

  __device__ Scalar& at(unsigned i, unsigned j, unsigned t) {
    return data[(i * kHermiteJDimension + j) * kHermiteTDimension + t];
  }
  __device__ const Scalar& at(unsigned i, unsigned j, unsigned t) const {
    return data[(i * kHermiteJDimension + j) * kHermiteTDimension + t];
  }
};

template <typename Scalar>
__device__ void fill_hermite(unsigned maximum_i,
                             unsigned maximum_j,
                             Scalar product,
                             Scalar center_a,
                             Scalar center_b,
                             double alpha,
                             double beta,
                             HermiteCoefficients<Scalar>& coefficients) {
  for (int item = 0;
       item < kHermiteIDimension * kHermiteJDimension * kHermiteTDimension;
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
        coefficients.at(i, j, 0) =
            pa * coefficients.at(i - 1, j, 0) +
            coefficients.at(i - 1, j, 1);
      } else {
        coefficients.at(i, j, 0) =
            pb * coefficients.at(i, j - 1, 0) +
            coefficients.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coefficients.at(i, j, t) =
              pa * coefficients.at(i - 1, j, t) +
              inverse_two_p * coefficients.at(i - 1, j, t - 1) +
              static_cast<double>(t + 1) *
                  coefficients.at(i - 1, j, t + 1);
        } else {
          coefficients.at(i, j, t) =
              pb * coefficients.at(i, j - 1, t) +
              inverse_two_p * coefficients.at(i, j - 1, t - 1) +
              static_cast<double>(t + 1) *
                  coefficients.at(i, j - 1, t + 1);
        }
      }
    }
  }
}

/** Hermite workspace bounded by one exact shell-pair class. */
template <typename Scalar, unsigned FirstAngular, unsigned SecondAngular>
struct ShellPairHermiteCoefficients {
  static constexpr unsigned kIDimension = FirstAngular + 1;
  static constexpr unsigned kJDimension = SecondAngular + 1;
  // One zero boundary element is required because the recurrence reads t+1.
  static constexpr unsigned kTDimension =
      FirstAngular + SecondAngular + 2;
  Scalar data[kIDimension * kJDimension * kTDimension];

  __device__ Scalar& at(unsigned i, unsigned j, unsigned t) {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
  __device__ const Scalar& at(unsigned i, unsigned j, unsigned t) const {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
};

template <unsigned FirstAngular, unsigned SecondAngular, typename Scalar>
__device__ void fill_shell_pair_hermite(
    unsigned maximum_i,
    unsigned maximum_j,
    Scalar product,
    Scalar center_a,
    Scalar center_b,
    double alpha,
    double beta,
    ShellPairHermiteCoefficients<Scalar, FirstAngular, SecondAngular>&
        coefficients) {
  static_assert(FirstAngular <= kMaximumAngularMomentum);
  static_assert(SecondAngular <= kMaximumAngularMomentum);
  for (unsigned item = 0;
       item < ShellPairHermiteCoefficients<
                  Scalar, FirstAngular, SecondAngular>::kIDimension *
              ShellPairHermiteCoefficients<
                  Scalar, FirstAngular, SecondAngular>::kJDimension *
              ShellPairHermiteCoefficients<
                  Scalar, FirstAngular, SecondAngular>::kTDimension;
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
        coefficients.at(i, j, 0) =
            pa * coefficients.at(i - 1, j, 0) +
            coefficients.at(i - 1, j, 1);
      } else {
        coefficients.at(i, j, 0) =
            pb * coefficients.at(i, j - 1, 0) +
            coefficients.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coefficients.at(i, j, t) =
              pa * coefficients.at(i - 1, j, t) +
              inverse_two_p * coefficients.at(i - 1, j, t - 1) +
              static_cast<double>(t + 1) *
                  coefficients.at(i - 1, j, t + 1);
        } else {
          coefficients.at(i, j, t) =
              pb * coefficients.at(i, j - 1, t) +
              inverse_two_p * coefficients.at(i, j - 1, t - 1) +
              static_cast<double>(t + 1) *
                  coefficients.at(i, j - 1, t + 1);
        }
      }
    }
  }
}

template <typename Scalar, unsigned MaximumAngular>
struct CoulombAuxiliary {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);

  __host__ __device__ static constexpr unsigned choose3(unsigned value) {
    return value < 3 ? 0 : value * (value - 1) * (value - 2) / 6;
  }

  __host__ __device__ static constexpr unsigned choose4(unsigned value) {
    return value < 4 ? 0
                     : value * (value - 1) * (value - 2) * (value - 3) / 24;
  }

  static constexpr unsigned kStateCount = choose4(MaximumAngular + 4);
  Scalar data[kStateCount];

  __device__ unsigned index(unsigned n,
                            unsigned t,
                            unsigned u,
                            unsigned v) const {
    const unsigned total = choose4(MaximumAngular + 4);
    const unsigned n_offset =
        total - choose4(MaximumAngular - n + 4);
    const unsigned remaining_after_n = MaximumAngular - n;
    const unsigned t_offset =
        choose3(remaining_after_n + 3) -
        choose3(remaining_after_n - t + 3);
    const unsigned remaining_after_t = remaining_after_n - t;
    const unsigned u_offset =
        u * (remaining_after_t + 1) - u * (u - 1) / 2;
    return n_offset + t_offset + u_offset + v;
  }

  __device__ Scalar& at(unsigned n, unsigned t, unsigned u, unsigned v) {
    return data[index(n, t, u, v)];
  }
  __device__ const Scalar& at(unsigned n,
                              unsigned t,
                              unsigned u,
                              unsigned v) const {
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
__device__ void fill_coulomb(double exponent,
                             const Vec3<Scalar>& product,
                             const Vec3<Scalar>& center,
                             CoulombAuxiliary<Scalar, MaximumAngular>&
                                 auxiliary) {
  for (unsigned item = 0;
       item < CoulombAuxiliary<Scalar, MaximumAngular>::kStateCount; ++item) {
    auxiliary.data[item] = scalar<Scalar>(0.0);
  }
  const Vec3<Scalar> pc{product.x - center.x, product.y - center.y,
                        product.z - center.z};
  Scalar boys[MaximumAngular + 1];
  boys_values<MaximumAngular>(
      exponent * distance_squared(product, center), boys);
  double factor = 1.0;
  for (unsigned n = 0; n <= MaximumAngular; ++n) {
    auxiliary.at(n, 0, 0, 0) = factor * boys[n];
    factor *= -2.0 * exponent;
  }

  for (unsigned v = 1; v <= MaximumAngular; ++v) {
    for (unsigned n = 0; n + v <= MaximumAngular; ++n) {
      Scalar value = pc.z * auxiliary.at(n + 1, 0, 0, v - 1);
      if (v > 1) {
        value = value + static_cast<double>(v - 1) *
                            auxiliary.at(n + 1, 0, 0, v - 2);
      }
      auxiliary.at(n, 0, 0, v) = value;
    }
  }
  for (unsigned v = 0; v <= MaximumAngular; ++v) {
    for (unsigned u = 1; u + v <= MaximumAngular; ++u) {
      for (unsigned n = 0; n + u + v <= MaximumAngular; ++n) {
        Scalar value = pc.y * auxiliary.at(n + 1, 0, u - 1, v);
        if (u > 1) {
          value = value + static_cast<double>(u - 1) *
                              auxiliary.at(n + 1, 0, u - 2, v);
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
            value = value + static_cast<double>(t - 1) *
                                auxiliary.at(n + 1, t - 2, u, v);
          }
          auxiliary.at(n, t, u, v) = value;
        }
      }
    }
  }
}

template <typename Scalar>
__device__ Scalar primitive_overlap_cartesian(
    double alpha,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<Scalar>& second,
    const Angular& angular_second) {
  const double p = alpha + beta;
  const Vec3<Scalar> product = product_center(alpha, first, beta, second);
  Scalar result = scalar<Scalar>(pow(kPi / p, 1.5));
  for (int axis = 0; axis < 3; ++axis) {
    HermiteCoefficients<Scalar> coefficients;
    const unsigned first_power = angular_axis(angular_first, axis);
    const unsigned second_power = angular_axis(angular_second, axis);
    fill_hermite(first_power, second_power, vec_axis(product, axis),
                 vec_axis(first, axis), vec_axis(second, axis), alpha, beta,
                 coefficients);
    result = result * coefficients.at(first_power, second_power, 0);
  }
  return result;
}

template <typename Scalar>
__device__ Scalar primitive_kinetic_cartesian(
    double alpha,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<Scalar>& second,
    const Angular& angular_second) {
  Scalar result = beta * (2.0 * static_cast<double>(angular_total(angular_second)) +
                          3.0) *
                  primitive_overlap_cartesian(alpha, first, angular_first,
                                              beta, second, angular_second);
  for (int axis = 0; axis < 3; ++axis) {
    Angular raised = angular_second;
    add_angular_axis(raised, axis, 2);
    result = result - 2.0 * beta * beta *
                          primitive_overlap_cartesian(
                              alpha, first, angular_first, beta, second, raised);
    const unsigned power = angular_axis(angular_second, axis);
    if (power >= 2) {
      Angular lowered = angular_second;
      add_angular_axis(lowered, axis, -2);
      result = result - 0.5 * static_cast<double>(power * (power - 1)) *
                            primitive_overlap_cartesian(
                                alpha, first, angular_first, beta, second,
                                lowered);
    }
  }
  return result;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ __noinline__ Scalar nuclear_attraction_cartesian_value(
    const DeviceBatch& batch,
    std::int32_t system,
    double exponent,
    const Vec3<Scalar>& product,
    const Angular& angular_first,
    const Angular& angular_second,
    const HermiteCoefficients<Scalar>* coefficients,
    std::int64_t derivative_coordinate) {
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum);
  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t atom = batch.atom_offsets[system];
       atom < batch.atom_offsets[system + 1]; ++atom) {
    CoulombAuxiliary<Scalar, MaximumAngular> auxiliary;
    fill_coulomb<MaximumAngular>(
        exponent, product,
        atom_position<Scalar>(batch, atom, derivative_coordinate), auxiliary);
    Scalar value = scalar<Scalar>(0.0);
    for (unsigned t = 0; t <= angular_first.x + angular_second.x; ++t) {
      for (unsigned u = 0; u <= angular_first.y + angular_second.y; ++u) {
        for (unsigned v = 0; v <= angular_first.z + angular_second.z; ++v) {
          value = value +
              coefficients[0].at(angular_first.x, angular_second.x, t) *
              coefficients[1].at(angular_first.y, angular_second.y, u) *
              coefficients[2].at(angular_first.z, angular_second.z, v) *
              auxiliary.at(0, t, u, v);
        }
      }
    }
    result = result - static_cast<double>(batch.atomic_numbers[atom]) *
                          (2.0 * kPi / exponent) * value;
  }
  return result;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ Scalar primitive_nuclear_attraction_cartesian(
    const DeviceBatch& batch,
    std::int32_t system,
    double alpha,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<Scalar>& second,
    const Angular& angular_second,
    std::int64_t derivative_coordinate) {
  const double exponent = alpha + beta;
  const Vec3<Scalar> product = product_center(alpha, first, beta, second);
  HermiteCoefficients<Scalar> coefficients[3];
  for (int axis = 0; axis < 3; ++axis) {
    fill_hermite(angular_axis(angular_first, axis),
                 angular_axis(angular_second, axis), vec_axis(product, axis),
                 vec_axis(first, axis), vec_axis(second, axis), alpha, beta,
                 coefficients[axis]);
  }

  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum);
  return nuclear_attraction_cartesian_value<MaximumAngular>(
      batch, system, exponent, product, angular_first, angular_second,
      coefficients, derivative_coordinate);
}

template <unsigned MaximumAngular,
          typename Scalar,
          typename FirstCoefficients,
          typename SecondCoefficients>
__device__ __noinline__ Scalar eri_cartesian_value(
    double p,
    double q,
    double rho,
    const Vec3<Scalar>& product_p,
    const Vec3<Scalar>& product_q,
    const Angular& angular_first,
    const Angular& angular_second,
    const Angular& angular_third,
    const Angular& angular_fourth,
    const FirstCoefficients* first_coefficients,
    const SecondCoefficients* second_coefficients) {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  CoulombAuxiliary<Scalar, MaximumAngular> auxiliary;
  fill_coulomb<MaximumAngular>(rho, product_p, product_q, auxiliary);

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned t = 0; t <= angular_first.x + angular_second.x; ++t) {
    for (unsigned u = 0; u <= angular_first.y + angular_second.y; ++u) {
      for (unsigned v = 0; v <= angular_first.z + angular_second.z; ++v) {
        const Scalar first_value =
            first_coefficients[0].at(angular_first.x, angular_second.x, t) *
            first_coefficients[1].at(angular_first.y, angular_second.y, u) *
            first_coefficients[2].at(angular_first.z, angular_second.z, v);
        for (unsigned tau = 0; tau <= angular_third.x + angular_fourth.x;
             ++tau) {
          for (unsigned nu = 0; nu <= angular_third.y + angular_fourth.y;
               ++nu) {
            for (unsigned phi = 0;
                 phi <= angular_third.z + angular_fourth.z; ++phi) {
              const double sign = ((tau + nu + phi) & 1U) == 0 ? 1.0 : -1.0;
              value = value + sign * first_value *
                  second_coefficients[0].at(angular_third.x,
                                            angular_fourth.x, tau) *
                  second_coefficients[1].at(angular_third.y,
                                            angular_fourth.y, nu) *
                  second_coefficients[2].at(angular_third.z,
                                            angular_fourth.z, phi) *
                  auxiliary.at(0, t + tau, u + nu, v + phi);
            }
          }
        }
      }
    }
  }
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * value;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ Scalar primitive_eri_cartesian(
    double alpha,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<Scalar>& second,
    const Angular& angular_second,
    double gamma,
    const Vec3<Scalar>& third,
    const Angular& angular_third,
    double delta,
    const Vec3<Scalar>& fourth,
    const Angular& angular_fourth) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  HermiteCoefficients<Scalar> first_coefficients[3];
  HermiteCoefficients<Scalar> second_coefficients[3];
  for (int axis = 0; axis < 3; ++axis) {
    fill_hermite(angular_axis(angular_first, axis),
                 angular_axis(angular_second, axis), vec_axis(product_p, axis),
                 vec_axis(first, axis), vec_axis(second, axis), alpha, beta,
                 first_coefficients[axis]);
    fill_hermite(angular_axis(angular_third, axis),
                 angular_axis(angular_fourth, axis), vec_axis(product_q, axis),
                 vec_axis(third, axis), vec_axis(fourth, axis), gamma, delta,
                 second_coefficients[axis]);
  }
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  return eri_cartesian_value<MaximumAngular>(
      p, q, rho, product_p, product_q, angular_first, angular_second,
      angular_third, angular_fourth, first_coefficients, second_coefficients);
}

/**
 * Closed first-order Hermite contraction for canonical (p s | s s).
 *
 * The exact order-1 shell class has only one Cartesian component on the first
 * center. Generating its two reachable Hermite terms directly avoids all six
 * coefficient workspaces and the generic six-deep component contraction.
 */
template <typename Scalar>
__device__ Scalar primitive_eri_psss(
    int axis,
    double alpha,
    const Vec3<Scalar>& first,
    double beta,
    const Vec3<Scalar>& second,
    double gamma,
    const Vec3<Scalar>& third,
    double delta,
    const Vec3<Scalar>& fourth) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  Scalar boys[2];
  boys_values<1>(rho * distance_squared(product_p, product_q), boys);
  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) -
           nu * distance_squared(third, fourth));
  const Scalar pa = vec_axis(product_p, axis) - vec_axis(first, axis);
  const Scalar pq = vec_axis(product_p, axis) - vec_axis(product_q, axis);
  const Scalar value = pa * boys[0] - (rho / p) * pq * boys[1];
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

/** One nonzero three-dimensional Hermite coefficient through order two. */
template <typename Scalar>
struct LowOrderHermiteTerm {
  // Each Cartesian derivative occupies two bits. Adding two states therefore
  // combines pair derivatives without carrying between x, y, and z.
  unsigned derivative_state;
  Scalar coefficient;
};

/** Compact shell-pair expansion; total order two reaches at most four terms. */
template <typename Scalar>
struct LowOrderPairExpansion {
  LowOrderHermiteTerm<Scalar> terms[4];
};

__device__ unsigned low_order_derivative_state(int axis) {
  return 1U << (2 * axis);
}

__device__ unsigned low_order_derivative_total(unsigned state) {
  return (state & 3U) + ((state >> 2U) & 3U) + ((state >> 4U) & 3U);
}

/**
 * Generate only the nonzero Hermite terms of one order-0/1/2 shell pair.
 *
 * The Gaussian pair decay is deliberately excluded and applied once by the
 * primitive quartet. At order two, retaining duplicate first-derivative terms
 * for a repeated axis keeps one runtime component path for d and p-p AOs while
 * still bounding the expansion at four entries.
 */
template <unsigned FirstShellAngular,
          unsigned SecondShellAngular,
          typename Scalar>
__device__ LowOrderPairExpansion<Scalar> make_low_order_pair_expansion(
    double exponent,
    const Vec3<Scalar>& product,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    const Vec3<Scalar>& second,
    const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 2);
  LowOrderPairExpansion<Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[2];
    Scalar shifts[2];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = low_order_derivative_state(axis);
      for (unsigned quantum = 0;
           quantum < angular_axis(angular_first, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(first, axis);
        ++quantum_count;
      }
      for (unsigned quantum = 0;
           quantum < angular_axis(angular_second, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(second, axis);
        ++quantum_count;
      }
    }

    expansion.terms[0] = {0U, shifts[0]};
    expansion.terms[1] = {
        derivative_states[0], scalar<Scalar>(inverse_two_exponent)};
    if constexpr (PairOrder == 2) {
      // A repeated Cartesian axis contributes the recurrence's +1/(2p)
      // correction. The two first-derivative entries then share a state and
      // sum to the exact E1 coefficient during contraction.
      const double repeated_axis_correction =
          derivative_states[0] == derivative_states[1]
          ? inverse_two_exponent
          : 0.0;
      expansion.terms[0].coefficient =
          shifts[0] * shifts[1] +
          scalar<Scalar>(repeated_axis_correction);
      expansion.terms[1].coefficient =
          inverse_two_exponent * shifts[1];
      expansion.terms[2] = {
          derivative_states[1], inverse_two_exponent * shifts[0]};
      expansion.terms[3] = {
          derivative_states[0] + derivative_states[1],
          scalar<Scalar>(inverse_two_exponent * inverse_two_exponent)};
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most two. */
template <typename Scalar>
__device__ Scalar low_order_coulomb(
    unsigned derivative_state,
    double rho,
    const Vec3<Scalar>& product_difference,
    const Scalar* boys) {
  const unsigned x_order = derivative_state & 3U;
  const unsigned y_order = (derivative_state >> 2U) & 3U;
  const unsigned z_order = (derivative_state >> 4U) & 3U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order == 0) return boys[0];

  if (total_order == 1) {
    const Scalar coordinate =
        x_order != 0 ? product_difference.x
                     : (y_order != 0 ? product_difference.y
                                     : product_difference.z);
    return (-2.0 * rho) * coordinate * boys[1];
  }

  const double second_order_factor = 4.0 * rho * rho;
  if (x_order == 2 || y_order == 2 || z_order == 2) {
    const Scalar coordinate =
        x_order == 2 ? product_difference.x
                     : (y_order == 2 ? product_difference.y
                                     : product_difference.z);
    return second_order_factor * coordinate * coordinate * boys[2] -
           (2.0 * rho) * boys[1];
  }

  Scalar coordinate_product = scalar<Scalar>(1.0);
  if (x_order != 0) coordinate_product = coordinate_product * product_difference.x;
  if (y_order != 0) coordinate_product = coordinate_product * product_difference.y;
  if (z_order != 0) coordinate_product = coordinate_product * product_difference.z;
  return second_order_factor * coordinate_product * boys[2];
}

/**
 * Closed order-2 contraction for canonical (d s|s s), (p p|s s), and
 * (p s|p s) primitive quartets.
 */
template <unsigned FirstShellAngular,
          unsigned SecondShellAngular,
          unsigned ThirdShellAngular,
          unsigned FourthShellAngular,
          typename Scalar>
__device__ Scalar primitive_eri_order2(
    double alpha,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<Scalar>& second,
    const Angular& angular_second,
    double gamma,
    const Vec3<Scalar>& third,
    const Angular& angular_third,
    double delta,
    const Vec3<Scalar>& fourth,
    const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder =
      FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder =
      ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 2);
  static_assert(FirstPairOrder <= 2 && SecondPairOrder <= 2);
  constexpr unsigned FirstTermCount =
      FirstPairOrder == 0 ? 1 : (FirstPairOrder == 1 ? 2 : 4);
  constexpr unsigned SecondTermCount =
      SecondPairOrder == 0 ? 1 : (SecondPairOrder == 1 ? 2 : 4);

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const LowOrderPairExpansion<Scalar> first_expansion =
      make_low_order_pair_expansion<FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const LowOrderPairExpansion<Scalar> second_expansion =
      make_low_order_pair_expansion<ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[3];
  boys_values<2>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount;
         ++second_term) {
      const LowOrderHermiteTerm<Scalar>& first_item =
          first_expansion.terms[first_term];
      const LowOrderHermiteTerm<Scalar>& second_item =
          second_expansion.terms[second_term];
      const double sign =
          (low_order_derivative_total(second_item.derivative_state) & 1U) == 0
          ? 1.0
          : -1.0;
      value = value +
          sign * first_item.coefficient * second_item.coefficient *
          low_order_coulomb(
              first_item.derivative_state + second_item.derivative_state,
              rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) -
           nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

/** Exact-sized sparse pair expansion used only by total-order-3 quartets. */
template <unsigned PairOrder, typename Scalar>
struct ThirdOrderPairExpansion {
  static_assert(PairOrder <= 3);
  LowOrderHermiteTerm<Scalar> terms[1U << PairOrder];
};

/**
 * Generate a shell pair through order three from its angular quanta.
 *
 * The base expansion is the product of one first-order factor per quantum.
 * Two quanta on the same Cartesian axis additionally have one Gaussian Wick
 * contraction, 1/(2p). Through order three, adding that contraction to the
 * surviving base terms produces the complete Hermite expansion while keeping
 * the exact 1/2/4/8-term bound.
 */
template <unsigned FirstShellAngular,
          unsigned SecondShellAngular,
          typename Scalar>
__device__ ThirdOrderPairExpansion<
    FirstShellAngular + SecondShellAngular, Scalar>
make_third_order_pair_expansion(
    double exponent,
    const Vec3<Scalar>& product,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    const Vec3<Scalar>& second,
    const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 3);
  constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  ThirdOrderPairExpansion<PairOrder, Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[QuantumStorage];
    Scalar shifts[QuantumStorage];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = low_order_derivative_state(axis);
      for (unsigned quantum = 0;
           quantum < angular_axis(angular_first, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(first, axis);
        ++quantum_count;
      }
      for (unsigned quantum = 0;
           quantum < angular_axis(angular_second, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(second, axis);
        ++quantum_count;
      }
    }

    for (unsigned subset = 0; subset < (1U << PairOrder); ++subset) {
      unsigned derivative_state = 0;
      Scalar coefficient = scalar<Scalar>(1.0);
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if ((subset & (1U << quantum)) != 0) {
          derivative_state += derivative_states[quantum];
          coefficient = inverse_two_exponent * coefficient;
        } else {
          coefficient = coefficient * shifts[quantum];
        }
      }
      expansion.terms[subset] = {derivative_state, coefficient};
    }

    if constexpr (PairOrder == 2) {
      if (derivative_states[0] == derivative_states[1]) {
        expansion.terms[0].coefficient =
            expansion.terms[0].coefficient +
            scalar<Scalar>(inverse_two_exponent);
      }
    } else if constexpr (PairOrder == 3) {
      for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1;
             second_quantum < 3; ++second_quantum) {
          if (derivative_states[first_quantum] !=
              derivative_states[second_quantum]) {
            continue;
          }
          const unsigned remaining_quantum =
              3U - first_quantum - second_quantum;
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              inverse_two_exponent * shifts[remaining_quantum];
          const unsigned surviving_derivative = 1U << remaining_quantum;
          expansion.terms[surviving_derivative].coefficient =
              expansion.terms[surviving_derivative].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most three. */
template <typename Scalar>
__device__ Scalar third_order_coulomb(
    unsigned derivative_state,
    double rho,
    const Vec3<Scalar>& product_difference,
    const Scalar* boys) {
  const unsigned x_order = derivative_state & 3U;
  const unsigned y_order = (derivative_state >> 2U) & 3U;
  const unsigned z_order = (derivative_state >> 4U) & 3U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 3) {
    return low_order_coulomb(
        derivative_state, rho, product_difference, boys);
  }

  const double third_order_factor = -8.0 * rho * rho * rho;
  if (x_order == 3 || y_order == 3 || z_order == 3) {
    const Scalar coordinate =
        x_order == 3 ? product_difference.x
                     : (y_order == 3 ? product_difference.y
                                     : product_difference.z);
    return third_order_factor * coordinate * coordinate * coordinate *
               boys[3] +
           (12.0 * rho * rho) * coordinate * boys[2];
  }

  if (x_order == 2 || y_order == 2 || z_order == 2) {
    const Scalar repeated_coordinate =
        x_order == 2 ? product_difference.x
                     : (y_order == 2 ? product_difference.y
                                     : product_difference.z);
    const Scalar single_coordinate =
        x_order == 1 ? product_difference.x
                     : (y_order == 1 ? product_difference.y
                                     : product_difference.z);
    return third_order_factor * repeated_coordinate * repeated_coordinate *
               single_coordinate * boys[3] +
           (4.0 * rho * rho) * single_coordinate * boys[2];
  }

  return third_order_factor * product_difference.x * product_difference.y *
         product_difference.z * boys[3];
}

/**
 * Closed order-3 contraction for canonical (f s|s s), (d p|s s),
 * (d s|p s), and (p p|p s) primitive quartets.
 */
template <unsigned FirstShellAngular,
          unsigned SecondShellAngular,
          unsigned ThirdShellAngular,
          unsigned FourthShellAngular,
          typename Scalar>
__device__ Scalar primitive_eri_order3(
    double alpha,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<Scalar>& second,
    const Angular& angular_second,
    double gamma,
    const Vec3<Scalar>& third,
    const Angular& angular_third,
    double delta,
    const Vec3<Scalar>& fourth,
    const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder =
      FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder =
      ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 3);
  static_assert(FirstPairOrder <= 3 && SecondPairOrder <= 3);
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const ThirdOrderPairExpansion<FirstPairOrder, Scalar> first_expansion =
      make_third_order_pair_expansion<
          FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const ThirdOrderPairExpansion<SecondPairOrder, Scalar> second_expansion =
      make_third_order_pair_expansion<
          ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[4];
  boys_values<3>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount;
         ++second_term) {
      const LowOrderHermiteTerm<Scalar>& first_item =
          first_expansion.terms[first_term];
      const LowOrderHermiteTerm<Scalar>& second_item =
          second_expansion.terms[second_term];
      const double sign =
          (low_order_derivative_total(second_item.derivative_state) & 1U) == 0
          ? 1.0
          : -1.0;
      value = value +
          sign * first_item.coefficient * second_item.coefficient *
          third_order_coulomb(
              first_item.derivative_state + second_item.derivative_state,
              rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) -
           nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

/** One sparse Hermite coefficient with three bits per Cartesian derivative. */
template <typename Scalar>
struct FourthOrderHermiteTerm {
  // Order four needs values 0--4 on one axis, so the two-bit encoding used by
  // lower orders is deliberately widened only for this specialization.
  unsigned derivative_state;
  Scalar coefficient;
};

/** Exact-sized sparse shell-pair expansion through total order four. */
template <unsigned PairOrder, typename Scalar>
struct FourthOrderPairExpansion {
  static_assert(PairOrder <= 4);
  FourthOrderHermiteTerm<Scalar> terms[1U << PairOrder];
};

__device__ unsigned fourth_order_derivative_state(int axis) {
  return 1U << (3 * axis);
}

__device__ unsigned fourth_order_derivative_total(unsigned state) {
  return (state & 7U) + ((state >> 3U) & 7U) + ((state >> 6U) & 7U);
}

/**
 * Generate the exact Wick expansion of one shell pair through order four.
 *
 * Base subset terms represent uncontracted angular quanta. Every same-axis
 * pair adds one 1/(2p) contraction times the uncontracted remaining factors;
 * order four additionally admits the three possible disjoint pairings. All
 * contributions merge into the existing 2^N subset slots, so no generic
 * recurrence workspace is required.
 */
template <unsigned FirstShellAngular,
          unsigned SecondShellAngular,
          typename Scalar>
__device__ FourthOrderPairExpansion<
    FirstShellAngular + SecondShellAngular, Scalar>
make_fourth_order_pair_expansion(
    double exponent,
    const Vec3<Scalar>& product,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    const Vec3<Scalar>& second,
    const Angular& angular_second) {
  constexpr unsigned PairOrder = FirstShellAngular + SecondShellAngular;
  static_assert(PairOrder <= 4);
  constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  FourthOrderPairExpansion<PairOrder, Scalar> expansion;
  const double inverse_two_exponent = 0.5 / exponent;

  if constexpr (PairOrder == 0) {
    expansion.terms[0] = {0U, scalar<Scalar>(1.0)};
  } else {
    unsigned derivative_states[QuantumStorage];
    Scalar shifts[QuantumStorage];
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const unsigned state = fourth_order_derivative_state(axis);
      for (unsigned quantum = 0;
           quantum < angular_axis(angular_first, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(first, axis);
        ++quantum_count;
      }
      for (unsigned quantum = 0;
           quantum < angular_axis(angular_second, axis); ++quantum) {
        derivative_states[quantum_count] = state;
        shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(second, axis);
        ++quantum_count;
      }
    }

    for (unsigned subset = 0; subset < (1U << PairOrder); ++subset) {
      unsigned derivative_state = 0;
      Scalar coefficient = scalar<Scalar>(1.0);
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if ((subset & (1U << quantum)) != 0) {
          derivative_state += derivative_states[quantum];
          coefficient = inverse_two_exponent * coefficient;
        } else {
          coefficient = coefficient * shifts[quantum];
        }
      }
      expansion.terms[subset] = {derivative_state, coefficient};
    }

    if constexpr (PairOrder == 2) {
      if (derivative_states[0] == derivative_states[1]) {
        expansion.terms[0].coefficient =
            expansion.terms[0].coefficient +
            scalar<Scalar>(inverse_two_exponent);
      }
    } else if constexpr (PairOrder == 3) {
      for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1;
             second_quantum < 3; ++second_quantum) {
          if (derivative_states[first_quantum] !=
              derivative_states[second_quantum]) {
            continue;
          }
          const unsigned remaining_quantum =
              3U - first_quantum - second_quantum;
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              inverse_two_exponent * shifts[remaining_quantum];
          const unsigned surviving_derivative = 1U << remaining_quantum;
          expansion.terms[surviving_derivative].coefficient =
              expansion.terms[surviving_derivative].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    } else if constexpr (PairOrder == 4) {
      for (unsigned first_quantum = 0; first_quantum < 4; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1;
             second_quantum < 4; ++second_quantum) {
          if (derivative_states[first_quantum] !=
              derivative_states[second_quantum]) {
            continue;
          }
          unsigned remaining[2];
          unsigned remaining_count = 0;
          for (unsigned quantum = 0; quantum < 4; ++quantum) {
            if (quantum != first_quantum && quantum != second_quantum) {
              remaining[remaining_count++] = quantum;
            }
          }
          const unsigned first_remaining = remaining[0];
          const unsigned second_remaining = remaining[1];
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              inverse_two_exponent * shifts[first_remaining] *
                  shifts[second_remaining];
          expansion.terms[1U << first_remaining].coefficient =
              expansion.terms[1U << first_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent) *
                  shifts[second_remaining];
          expansion.terms[1U << second_remaining].coefficient =
              expansion.terms[1U << second_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent) *
                  shifts[first_remaining];
          const unsigned both_remaining =
              (1U << first_remaining) | (1U << second_remaining);
          expansion.terms[both_remaining].coefficient =
              expansion.terms[both_remaining].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent *
                             inverse_two_exponent);
        }
      }

      constexpr unsigned Pairings[3][4] = {
          {0, 1, 2, 3},
          {0, 2, 1, 3},
          {0, 3, 1, 2},
      };
      for (unsigned pairing = 0; pairing < 3; ++pairing) {
        if (derivative_states[Pairings[pairing][0]] ==
                derivative_states[Pairings[pairing][1]] &&
            derivative_states[Pairings[pairing][2]] ==
                derivative_states[Pairings[pairing][3]]) {
          expansion.terms[0].coefficient =
              expansion.terms[0].coefficient +
              scalar<Scalar>(inverse_two_exponent * inverse_two_exponent);
        }
      }
    }
  }
  return expansion;
}

/** Evaluate a Cartesian Coulomb derivative of total order at most four. */
template <typename Scalar>
__device__ Scalar fourth_order_coulomb(
    unsigned derivative_state,
    double rho,
    const Vec3<Scalar>& product_difference,
    const Scalar* boys) {
  const unsigned x_order = derivative_state & 7U;
  const unsigned y_order = (derivative_state >> 3U) & 7U;
  const unsigned z_order = (derivative_state >> 6U) & 7U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 4) {
    const unsigned lower_order_state =
        x_order | (y_order << 2U) | (z_order << 4U);
    return third_order_coulomb(
        lower_order_state, rho, product_difference, boys);
  }

  const double fourth_order_factor = 16.0 * rho * rho * rho * rho;
  const double third_order_factor = -8.0 * rho * rho * rho;
  const double second_order_factor = 4.0 * rho * rho;
  if (x_order == 4 || y_order == 4 || z_order == 4) {
    const Scalar coordinate =
        x_order == 4 ? product_difference.x
                     : (y_order == 4 ? product_difference.y
                                     : product_difference.z);
    const Scalar coordinate_squared = coordinate * coordinate;
    return fourth_order_factor * coordinate_squared * coordinate_squared *
               boys[4] +
           (6.0 * third_order_factor) * coordinate_squared * boys[3] +
           (3.0 * second_order_factor) * boys[2];
  }

  if (x_order == 3 || y_order == 3 || z_order == 3) {
    const Scalar repeated_coordinate =
        x_order == 3 ? product_difference.x
                     : (y_order == 3 ? product_difference.y
                                     : product_difference.z);
    const Scalar single_coordinate =
        x_order == 1 ? product_difference.x
                     : (y_order == 1 ? product_difference.y
                                     : product_difference.z);
    return fourth_order_factor * repeated_coordinate * repeated_coordinate *
               repeated_coordinate * single_coordinate * boys[4] +
           (3.0 * third_order_factor) * repeated_coordinate *
               single_coordinate * boys[3];
  }

  if ((x_order == 2 && y_order == 2) ||
      (x_order == 2 && z_order == 2) ||
      (y_order == 2 && z_order == 2)) {
    Scalar first_coordinate = product_difference.x;
    Scalar second_coordinate = product_difference.y;
    if (x_order == 0) {
      first_coordinate = product_difference.y;
      second_coordinate = product_difference.z;
    } else if (y_order == 0) {
      second_coordinate = product_difference.z;
    }
    const Scalar first_squared = first_coordinate * first_coordinate;
    const Scalar second_squared = second_coordinate * second_coordinate;
    return fourth_order_factor * first_squared * second_squared * boys[4] +
           third_order_factor * (first_squared + second_squared) * boys[3] +
           second_order_factor * boys[2];
  }

  const Scalar repeated_coordinate =
      x_order == 2 ? product_difference.x
                   : (y_order == 2 ? product_difference.y
                                   : product_difference.z);
  Scalar single_product = scalar<Scalar>(1.0);
  if (x_order == 1) single_product = single_product * product_difference.x;
  if (y_order == 1) single_product = single_product * product_difference.y;
  if (z_order == 1) single_product = single_product * product_difference.z;
  return fourth_order_factor * repeated_coordinate * repeated_coordinate *
             single_product * boys[4] +
         third_order_factor * single_product * boys[3];
}

/**
 * Closed order-4 contraction for canonical (f p|s s), (d d|s s),
 * (f s|p s), (d p|p s), (d s|d s), (d s|p p), and (p p|p p) quartets.
 */
template <unsigned FirstShellAngular,
          unsigned SecondShellAngular,
          unsigned ThirdShellAngular,
          unsigned FourthShellAngular,
          typename Scalar>
__device__ Scalar primitive_eri_order4(
    double alpha,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<Scalar>& second,
    const Angular& angular_second,
    double gamma,
    const Vec3<Scalar>& third,
    const Angular& angular_third,
    double delta,
    const Vec3<Scalar>& fourth,
    const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder =
      FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder =
      ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 4);
  static_assert(FirstPairOrder <= 4 && SecondPairOrder <= 4);
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const FourthOrderPairExpansion<FirstPairOrder, Scalar> first_expansion =
      make_fourth_order_pair_expansion<
          FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const FourthOrderPairExpansion<SecondPairOrder, Scalar> second_expansion =
      make_fourth_order_pair_expansion<
          ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[5];
  boys_values<4>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount;
         ++second_term) {
      const FourthOrderHermiteTerm<Scalar>& first_item =
          first_expansion.terms[first_term];
      const FourthOrderHermiteTerm<Scalar>& second_item =
          second_expansion.terms[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U) ==
              0
          ? 1.0
          : -1.0;
      value = value +
          sign * first_item.coefficient * second_item.coefficient *
          fourth_order_coulomb(
              first_item.derivative_state + second_item.derivative_state,
              rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) -
           nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

/**
 * Evaluate one Cartesian primitive quartet with exact shell-pair workspaces.
 *
 * Axis powers remain AO-component data, while the enclosing shell angular
 * momenta bound every Hermite dimension at compile time. This avoids charging
 * an s/p/d task for the generic f/f pair workspace.
 */
template <unsigned FirstShellAngular,
          unsigned SecondShellAngular,
          unsigned ThirdShellAngular,
          unsigned FourthShellAngular,
          typename Scalar>
__device__ Scalar primitive_eri_cartesian_shell_class(
    double alpha,
    const Vec3<Scalar>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<Scalar>& second,
    const Angular& angular_second,
    double gamma,
    const Vec3<Scalar>& third,
    const Angular& angular_third,
    double delta,
    const Vec3<Scalar>& fourth,
    const Angular& angular_fourth) {
  constexpr unsigned MaximumAngular =
      FirstShellAngular + SecondShellAngular + ThirdShellAngular +
      FourthShellAngular;
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  if constexpr (FirstShellAngular == 1 && SecondShellAngular == 0 &&
                ThirdShellAngular == 0 && FourthShellAngular == 0) {
    const int axis = angular_first.x == 1 ? 0 : (angular_first.y == 1 ? 1 : 2);
    return primitive_eri_psss(
        axis, alpha, first, beta, second, gamma, third, delta, fourth);
  } else if constexpr (MaximumAngular == 2) {
    return primitive_eri_order2<
        FirstShellAngular, SecondShellAngular,
        ThirdShellAngular, FourthShellAngular>(
        alpha, first, angular_first, beta, second, angular_second,
        gamma, third, angular_third, delta, fourth, angular_fourth);
  } else if constexpr (MaximumAngular == 3) {
    return primitive_eri_order3<
        FirstShellAngular, SecondShellAngular,
        ThirdShellAngular, FourthShellAngular>(
        alpha, first, angular_first, beta, second, angular_second,
        gamma, third, angular_third, delta, fourth, angular_fourth);
  } else if constexpr (MaximumAngular == 4) {
    return primitive_eri_order4<
        FirstShellAngular, SecondShellAngular,
        ThirdShellAngular, FourthShellAngular>(
        alpha, first, angular_first, beta, second, angular_second,
        gamma, third, angular_third, delta, fourth, angular_fourth);
  } else {
    const double p = alpha + beta;
    const double q = gamma + delta;
    const double rho = p * q / (p + q);
    const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
    const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
    ShellPairHermiteCoefficients<
        Scalar, FirstShellAngular, SecondShellAngular>
        first_coefficients[3];
    ShellPairHermiteCoefficients<
        Scalar, ThirdShellAngular, FourthShellAngular>
        second_coefficients[3];
    for (int axis = 0; axis < 3; ++axis) {
      fill_shell_pair_hermite<FirstShellAngular, SecondShellAngular>(
          angular_axis(angular_first, axis),
          angular_axis(angular_second, axis), vec_axis(product_p, axis),
          vec_axis(first, axis), vec_axis(second, axis), alpha, beta,
          first_coefficients[axis]);
      fill_shell_pair_hermite<ThirdShellAngular, FourthShellAngular>(
          angular_axis(angular_third, axis),
          angular_axis(angular_fourth, axis), vec_axis(product_q, axis),
          vec_axis(third, axis), vec_axis(fourth, axis), gamma, delta,
          second_coefficients[axis]);
    }
    return eri_cartesian_value<MaximumAngular>(
        p, q, rho, product_p, product_q, angular_first, angular_second,
        angular_third, angular_fourth, first_coefficients,
        second_coefficients);
  }
}

template <typename Scalar>
__device__ Scalar contracted_overlap(const DeviceBatch& batch,
                                     std::int32_t system,
                                     std::int32_t i,
                                     std::int32_t j,
                                     std::int64_t derivative_coordinate) {
  const std::int64_t ao_i = static_cast<std::int64_t>(system) * batch.nbf + i;
  const std::int64_t ao_j = static_cast<std::int64_t>(system) * batch.nbf + j;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const Vec3<Scalar> first = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_i], derivative_coordinate);
  const Vec3<Scalar> second = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_j], derivative_coordinate);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const Angular angular_first = ao_angular(batch, ao_i, 0);
  const Angular angular_second = ao_angular(batch, ao_j, 0);
  const bool all_s = first_terms == 1 && second_terms == 1 &&
                     is_s_function(angular_first) &&
                     is_s_function(angular_second);
  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      const double primitive_weight = batch.primitive_coefficients[a] *
                                      batch.primitive_coefficients[b];
      if (all_s) {
        result = result + primitive_weight *
            ao_term_coefficient(batch, ao_i, 0) *
            ao_term_coefficient(batch, ao_j, 0) *
            primitive_overlap(batch.primitive_exponents[a], first,
                              batch.primitive_exponents[b], second);
      } else {
        for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
          const Angular first_angular =
              ao_angular(batch, ao_i, first_term);
          const double first_coefficient =
              ao_term_coefficient(batch, ao_i, first_term);
          for (unsigned second_term = 0; second_term < second_terms;
               ++second_term) {
            result = result + primitive_weight * first_coefficient *
                ao_term_coefficient(batch, ao_j, second_term) *
                primitive_overlap_cartesian(
                    batch.primitive_exponents[a], first, first_angular,
                    batch.primitive_exponents[b], second,
                    ao_angular(batch, ao_j, second_term));
          }
        }
      }
    }
  }
  return result;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ __noinline__ Scalar contracted_hcore_cartesian(
    const DeviceBatch& batch,
    std::int32_t system,
    std::int64_t ao_i,
    std::int64_t ao_j,
    std::int32_t shell_i,
    std::int32_t shell_j,
    const Vec3<Scalar>& first,
    const Vec3<Scalar>& second,
    unsigned first_terms,
    unsigned second_terms,
    std::int64_t derivative_coordinate) {
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum);
  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      const double weight = batch.primitive_coefficients[a] *
                            batch.primitive_coefficients[b];
      for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
        const Angular first_angular = ao_angular(batch, ao_i, first_term);
        const double first_coefficient =
            ao_term_coefficient(batch, ao_i, first_term);
        for (unsigned second_term = 0; second_term < second_terms;
             ++second_term) {
          const Angular second_angular =
              ao_angular(batch, ao_j, second_term);
          result = result + weight * first_coefficient *
              ao_term_coefficient(batch, ao_j, second_term) *
              (primitive_kinetic_cartesian(
                   batch.primitive_exponents[a], first, first_angular,
                   batch.primitive_exponents[b], second, second_angular) +
               primitive_nuclear_attraction_cartesian<MaximumAngular>(
                   batch, system, batch.primitive_exponents[a], first,
                   first_angular, batch.primitive_exponents[b], second,
                   second_angular, derivative_coordinate));
        }
      }
    }
  }
  return result;
}

template <typename Scalar>
__device__ Scalar contracted_hcore(const DeviceBatch& batch,
                                   std::int32_t system,
                                   std::int32_t i,
                                   std::int32_t j,
                                   std::int64_t derivative_coordinate) {
  const std::int64_t ao_i = static_cast<std::int64_t>(system) * batch.nbf + i;
  const std::int64_t ao_j = static_cast<std::int64_t>(system) * batch.nbf + j;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const Vec3<Scalar> first = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_i], derivative_coordinate);
  const Vec3<Scalar> second = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_j], derivative_coordinate);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const Angular angular_first = ao_angular(batch, ao_i, 0);
  const Angular angular_second = ao_angular(batch, ao_j, 0);
  const bool all_s = first_terms == 1 && second_terms == 1 &&
                     is_s_function(angular_first) &&
                     is_s_function(angular_second);
  if (all_s) {
    Scalar result = scalar<Scalar>(0.0);
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        const double weight = batch.primitive_coefficients[a] *
                              batch.primitive_coefficients[b];
        result = result + weight * ao_term_coefficient(batch, ao_i, 0) *
            ao_term_coefficient(batch, ao_j, 0) *
            (primitive_kinetic(batch.primitive_exponents[a], first,
                               batch.primitive_exponents[b], second) +
             primitive_nuclear_attraction(
                 batch, system, batch.primitive_exponents[a], first,
                 batch.primitive_exponents[b], second, derivative_coordinate));
      }
    }
    return result;
  }

  const unsigned maximum = batch.shell_angular[shell_i] +
                           batch.shell_angular[shell_j];
  switch (maximum) {
    case 1:
      return contracted_hcore_cartesian<1>(
          batch, system, ao_i, ao_j, shell_i, shell_j, first, second,
          first_terms, second_terms, derivative_coordinate);
    case 2:
      return contracted_hcore_cartesian<2>(
          batch, system, ao_i, ao_j, shell_i, shell_j, first, second,
          first_terms, second_terms, derivative_coordinate);
    case 3:
      return contracted_hcore_cartesian<3>(
          batch, system, ao_i, ao_j, shell_i, shell_j, first, second,
          first_terms, second_terms, derivative_coordinate);
    case 4:
      return contracted_hcore_cartesian<4>(
          batch, system, ao_i, ao_j, shell_i, shell_j, first, second,
          first_terms, second_terms, derivative_coordinate);
    case 5:
      return contracted_hcore_cartesian<5>(
          batch, system, ao_i, ao_j, shell_i, shell_j, first, second,
          first_terms, second_terms, derivative_coordinate);
    case 6:
      return contracted_hcore_cartesian<6>(
          batch, system, ao_i, ao_j, shell_i, shell_j, first, second,
          first_terms, second_terms, derivative_coordinate);
  }
  return scalar<Scalar>(0.0);
}

template <unsigned MaximumAngular, typename Scalar>
__device__ __noinline__ Scalar contracted_eri_cartesian(
    const DeviceBatch& batch,
    std::int64_t ao_i,
    std::int64_t ao_j,
    std::int64_t ao_k,
    std::int64_t ao_l,
    std::int32_t shell_i,
    std::int32_t shell_j,
    std::int32_t shell_k,
    std::int32_t shell_l,
    std::int64_t derivative_coordinate) {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  const Vec3<Scalar> first = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_i], derivative_coordinate);
  const Vec3<Scalar> second = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_j], derivative_coordinate);
  const Vec3<Scalar> third = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_k], derivative_coordinate);
  const Vec3<Scalar> fourth = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_l], derivative_coordinate);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const unsigned third_terms = batch.ao_term_counts[ao_k];
  const unsigned fourth_terms = batch.ao_term_counts[ao_l];

  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[shell_k];
           c < batch.shell_primitive_offsets[shell_k + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[shell_l];
             d < batch.shell_primitive_offsets[shell_l + 1]; ++d) {
          const double weight = batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] *
                                batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          for (unsigned first_term = 0; first_term < first_terms;
               ++first_term) {
            const Angular first_angular =
                ao_angular(batch, ao_i, first_term);
            const double first_coefficient =
                ao_term_coefficient(batch, ao_i, first_term);
            for (unsigned second_term = 0; second_term < second_terms;
                 ++second_term) {
              const Angular second_angular =
                  ao_angular(batch, ao_j, second_term);
              const double second_coefficient =
                  ao_term_coefficient(batch, ao_j, second_term);
              for (unsigned third_term = 0; third_term < third_terms;
                   ++third_term) {
                const Angular third_angular =
                    ao_angular(batch, ao_k, third_term);
                const double third_coefficient =
                    ao_term_coefficient(batch, ao_k, third_term);
                for (unsigned fourth_term = 0; fourth_term < fourth_terms;
                     ++fourth_term) {
                  result = result + weight * first_coefficient *
                      second_coefficient * third_coefficient *
                      ao_term_coefficient(batch, ao_l, fourth_term) *
                      primitive_eri_cartesian<MaximumAngular>(
                          batch.primitive_exponents[a], first, first_angular,
                          batch.primitive_exponents[b], second,
                          second_angular, batch.primitive_exponents[c], third,
                          third_angular, batch.primitive_exponents[d], fourth,
                          ao_angular(batch, ao_l, fourth_term));
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

template <unsigned MaximumAngular, typename Scalar>
__device__ Scalar contracted_eri_order(const DeviceBatch& batch,
                                       std::int32_t system,
                                       std::int32_t i,
                                       std::int32_t j,
                                       std::int32_t k,
                                       std::int32_t l,
                                       std::int64_t derivative_coordinate) {
  static_assert(MaximumAngular <= kMaximumCoulombOrder);
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.nbf;
  const std::int64_t ao_i = base + i;
  const std::int64_t ao_j = base + j;
  const std::int64_t ao_k = base + k;
  const std::int64_t ao_l = base + l;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const std::int32_t shell_k = batch.ao_shells[ao_k];
  const std::int32_t shell_l = batch.ao_shells[ao_l];
  if constexpr (MaximumAngular == 0) {
    const Vec3<Scalar> first = atom_position<Scalar>(
        batch, batch.shell_atoms[shell_i], derivative_coordinate);
    const Vec3<Scalar> second = atom_position<Scalar>(
        batch, batch.shell_atoms[shell_j], derivative_coordinate);
    const Vec3<Scalar> third = atom_position<Scalar>(
        batch, batch.shell_atoms[shell_k], derivative_coordinate);
    const Vec3<Scalar> fourth = atom_position<Scalar>(
        batch, batch.shell_atoms[shell_l], derivative_coordinate);
    Scalar result = scalar<Scalar>(0.0);
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        for (std::int64_t c = batch.shell_primitive_offsets[shell_k];
             c < batch.shell_primitive_offsets[shell_k + 1]; ++c) {
          for (std::int64_t d = batch.shell_primitive_offsets[shell_l];
               d < batch.shell_primitive_offsets[shell_l + 1]; ++d) {
            const double weight = batch.primitive_coefficients[a] *
                                  batch.primitive_coefficients[b] *
                                  batch.primitive_coefficients[c] *
                                  batch.primitive_coefficients[d];
            result = result + weight *
                ao_term_coefficient(batch, ao_i, 0) *
                ao_term_coefficient(batch, ao_j, 0) *
                ao_term_coefficient(batch, ao_k, 0) *
                ao_term_coefficient(batch, ao_l, 0) *
                primitive_eri(batch.primitive_exponents[a], first,
                              batch.primitive_exponents[b], second,
                              batch.primitive_exponents[c], third,
                              batch.primitive_exponents[d], fourth);
          }
        }
      }
    }
    return result;
  } else {
    return contracted_eri_cartesian<MaximumAngular, Scalar>(
        batch, ao_i, ao_j, ao_k, ao_l, shell_i, shell_j, shell_k, shell_l,
        derivative_coordinate);
  }
}

template <typename Scalar>
__device__ Scalar contracted_eri(const DeviceBatch& batch,
                                 std::int32_t system,
                                 std::int32_t i,
                                 std::int32_t j,
                                 std::int32_t k,
                                 std::int32_t l,
                                 std::int64_t derivative_coordinate) {
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.nbf;
  const std::int32_t shell_i = batch.ao_shells[base + i];
  const std::int32_t shell_j = batch.ao_shells[base + j];
  const std::int32_t shell_k = batch.ao_shells[base + k];
  const std::int32_t shell_l = batch.ao_shells[base + l];
  // Shell angular momentum is invariant across Cartesian expansion terms, so
  // one contracted-quartet dispatch covers every primitive and sparse
  // spherical term below it.
  const unsigned maximum = batch.shell_angular[shell_i] +
                           batch.shell_angular[shell_j] +
                           batch.shell_angular[shell_k] +
                           batch.shell_angular[shell_l];
  switch (maximum) {
    case 0:
      return contracted_eri_order<0, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 1:
      return contracted_eri_order<1, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 2:
      return contracted_eri_order<2, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 3:
      return contracted_eri_order<3, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 4:
      return contracted_eri_order<4, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 5:
      return contracted_eri_order<5, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 6:
      return contracted_eri_order<6, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 7:
      return contracted_eri_order<7, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 8:
      return contracted_eri_order<8, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 9:
      return contracted_eri_order<9, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 10:
      return contracted_eri_order<10, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 11:
      return contracted_eri_order<11, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
    case 12:
      return contracted_eri_order<12, Scalar>(
          batch, system, i, j, k, l, derivative_coordinate);
  }
  return scalar<Scalar>(0.0);
}

/**
 * Contract one quartet of normalized Cartesian source AOs.
 *
 * Public spherical AOs are handled by transforming their density before this
 * evaluator and their Fock matrix afterwards. Each source AO therefore has
 * exactly one angular component, eliminating the sparse term-product loops
 * from the dominant direct Fock and force recurrences.
 */
template <unsigned FirstShellAngular,
          unsigned SecondShellAngular,
          unsigned ThirdShellAngular,
          unsigned FourthShellAngular,
          typename Scalar>
__device__ __noinline__ Scalar contracted_eri_cartesian_source_shell_class(
    const DeviceBatch& batch,
    std::int64_t ao_i,
    std::int64_t ao_j,
    std::int64_t ao_k,
    std::int64_t ao_l,
    std::int32_t shell_i,
    std::int32_t shell_j,
    std::int32_t shell_k,
    std::int32_t shell_l,
    std::int64_t derivative_coordinate) {
  constexpr unsigned MaximumAngular =
      FirstShellAngular + SecondShellAngular + ThirdShellAngular +
      FourthShellAngular;
  const Vec3<Scalar> first = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_i], derivative_coordinate);
  const Vec3<Scalar> second = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_j], derivative_coordinate);
  const Vec3<Scalar> third = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_k], derivative_coordinate);
  const Vec3<Scalar> fourth = atom_position<Scalar>(
      batch, batch.shell_atoms[shell_l], derivative_coordinate);
  const Angular angular_first = direct_ao_angular(batch, ao_i);
  const Angular angular_second = direct_ao_angular(batch, ao_j);
  const Angular angular_third = direct_ao_angular(batch, ao_k);
  const Angular angular_fourth = direct_ao_angular(batch, ao_l);
  const double angular_coefficient =
      batch.direct_ao_coefficients[ao_i] *
      batch.direct_ao_coefficients[ao_j] *
      batch.direct_ao_coefficients[ao_k] *
      batch.direct_ao_coefficients[ao_l];

  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[shell_k];
           c < batch.shell_primitive_offsets[shell_k + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[shell_l];
             d < batch.shell_primitive_offsets[shell_l + 1]; ++d) {
          const double weight = angular_coefficient *
              batch.primitive_coefficients[a] *
              batch.primitive_coefficients[b] *
              batch.primitive_coefficients[c] *
              batch.primitive_coefficients[d];
          if constexpr (MaximumAngular == 0) {
            result = result + weight * primitive_eri(
                batch.primitive_exponents[a], first,
                batch.primitive_exponents[b], second,
                batch.primitive_exponents[c], third,
                batch.primitive_exponents[d], fourth);
          } else {
            result = result + weight *
                primitive_eri_cartesian_shell_class<
                    FirstShellAngular, SecondShellAngular,
                    ThirdShellAngular, FourthShellAngular>(
                    batch.primitive_exponents[a], first, angular_first,
                    batch.primitive_exponents[b], second, angular_second,
                    batch.primitive_exponents[c], third, angular_third,
                    batch.primitive_exponents[d], fourth, angular_fourth);
          }
        }
      }
    }
  }
  return result;
}

/** Cartesian derivatives of one contracted quartet, indexed by input slot. */
struct CartesianQuartetGradient {
  double center[4][3];
};

/**
 * Evaluate all four center derivatives of an ssss or canonical psss primitive.
 *
 * Coordinate differentiation changes only the Gaussian pair decay, product
 * centers, and Boys argument. Computing those shared values once is much
 * cheaper than replaying the complete primitive with one Dual3 seed per
 * independent atom. `p_axis` is ignored for the order-zero specialization.
 */
template <unsigned AngularOrder>
__device__ void primitive_eri_order01_gradient(
    int p_axis,
    double alpha,
    const Vec3<double>& first,
    double beta,
    const Vec3<double>& second,
    double gamma,
    const Vec3<double>& third,
    double delta,
    const Vec3<double>& fourth,
    double (&gradient)[4][3]) {
  static_assert(AngularOrder <= 1);
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<double> product_p =
      product_center(alpha, first, beta, second);
  const Vec3<double> product_q =
      product_center(gamma, third, delta, fourth);
  const Vec3<double> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };
  double boys[AngularOrder + 2];
  boys_values<AngularOrder + 1>(
      rho * distance_squared(product_p, product_q), boys);
  const double pair_decay = exp(
      -mu * distance_squared(first, second) -
      nu * distance_squared(third, fourth));
  const double prefactor =
      2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;

  // d(P-Q)/d(A,B,C,D); the same scalar applies independently to x/y/z.
  const double product_scales[4] = {
      alpha / p, beta / p, -gamma / q, -delta / q};
  double value = boys[0];
  double pa = 0.0;
  double pq_axis = 0.0;
  const double coulomb_scale = rho / p;
  if constexpr (AngularOrder == 1) {
    pa = vec_axis(product_p, p_axis) - vec_axis(first, p_axis);
    pq_axis = vec_axis(product_difference, p_axis);
    value = pa * boys[0] - coulomb_scale * pq_axis * boys[1];
  }

  for (unsigned center = 0; center < 4; ++center) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double decay_derivative = 0.0;
      if (center < 2) {
        const double difference =
            vec_axis(first, coordinate) - vec_axis(second, coordinate);
        decay_derivative =
            (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
      } else {
        const double difference =
            vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
        decay_derivative =
            (center == 2 ? -2.0 * nu : 2.0 * nu) * difference;
      }
      const double argument_derivative =
          2.0 * rho * product_scales[center] *
          vec_axis(product_difference, coordinate);
      double value_derivative = -boys[1] * argument_derivative;
      if constexpr (AngularOrder == 1) {
        double pa_derivative = 0.0;
        if (coordinate == p_axis) {
          if (center == 0) {
            pa_derivative = alpha / p - 1.0;
          } else if (center == 1) {
            pa_derivative = beta / p;
          }
        }
        const double pq_derivative =
            coordinate == p_axis ? product_scales[center] : 0.0;
        value_derivative =
            pa_derivative * boys[0] - pa * boys[1] * argument_derivative -
            coulomb_scale * pq_derivative * boys[1] +
            coulomb_scale * pq_axis * boys[2] * argument_derivative;
      }
      gradient[center][coordinate] =
          prefactor * (value_derivative + value * decay_derivative);
    }
  }
}

/**
 * Contract explicit order-zero/one primitive gradients into input AO slots.
 *
 * The order-one integral is canonicalized to (p s|s s), while `original`
 * preserves the caller's slot-to-atom mapping for force accumulation.
 */
template <unsigned AngularOrder>
__device__ CartesianQuartetGradient
contracted_eri_cartesian_source_order01_gradient(
    const DeviceBatch& batch,
    std::int32_t system,
    std::int32_t i,
    std::int32_t j,
    std::int32_t k,
    std::int32_t l) {
  static_assert(AngularOrder <= 1);
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base =
      static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if constexpr (AngularOrder == 1) {
    unsigned p_slot = 0;
    for (unsigned slot = 0; slot < 4; ++slot) {
      if (batch.shell_angular[slots[slot].shell] == 1) p_slot = slot;
    }
    if (p_slot == 1) {
      const SourceSlot swap = slots[0];
      slots[0] = slots[1];
      slots[1] = swap;
    } else if (p_slot >= 2) {
      if (p_slot == 3) {
        const SourceSlot swap = slots[2];
        slots[2] = slots[3];
        slots[3] = swap;
      }
      const SourceSlot first_swap = slots[0];
      slots[0] = slots[2];
      slots[2] = first_swap;
      const SourceSlot second_swap = slots[1];
      slots[1] = slots[3];
      slots[3] = second_swap;
    }
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  int p_axis = 0;
  if constexpr (AngularOrder == 1) {
    const Angular angular = direct_ao_angular(batch, slots[0].ao);
    p_axis = angular.x == 1 ? 0 : (angular.y == 1 ? 1 : 2);
  }
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] *
      batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] *
      batch.direct_ao_coefficients[slots[3].ao];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient *
              batch.primitive_coefficients[a] *
              batch.primitive_coefficients[b] *
              batch.primitive_coefficients[c] *
              batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          primitive_eri_order01_gradient<AngularOrder>(
              p_axis, batch.primitive_exponents[a], positions[0],
              batch.primitive_exponents[b], positions[1],
              batch.primitive_exponents[c], positions[2],
              batch.primitive_exponents[d], positions[3],
              primitive_gradient);
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

/** One order-two shell-pair term and its two-center coefficient gradient. */
struct LowOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
  double center[2][3];
};

struct LowOrderPairGradientExpansion {
  LowOrderPairGradientTerm terms[4];
  unsigned count;
};

/**
 * Build the exact order-0/1/2 Hermite pair expansion and center gradients.
 *
 * Pair decay is handled once by the primitive quartet. The only
 * coordinate-dependent pair coefficients are products of P-A/P-B shifts;
 * 1/(2p) contraction corrections are coordinate independent.
 */
__device__ LowOrderPairGradientExpansion
make_low_order_pair_gradient_expansion(
    double alpha,
    const Vec3<double>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<double>& second,
    const Angular& angular_second) {
  const double exponent = alpha + beta;
  const Vec3<double> product =
      product_center(alpha, first, beta, second);
  const double inverse_two_exponent = 0.5 / exponent;
  unsigned derivative_states[2]{};
  double shifts[2]{};
  double shift_gradients[2][2][3]{};
  unsigned quantum_count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    for (unsigned quantum = 0;
         quantum < angular_axis(angular_first, axis); ++quantum) {
      derivative_states[quantum_count] = low_order_derivative_state(axis);
      shifts[quantum_count] =
          vec_axis(product, axis) - vec_axis(first, axis);
      shift_gradients[quantum_count][0][axis] = alpha / exponent - 1.0;
      shift_gradients[quantum_count][1][axis] = beta / exponent;
      ++quantum_count;
    }
    for (unsigned quantum = 0;
         quantum < angular_axis(angular_second, axis); ++quantum) {
      derivative_states[quantum_count] = low_order_derivative_state(axis);
      shifts[quantum_count] =
          vec_axis(product, axis) - vec_axis(second, axis);
      shift_gradients[quantum_count][0][axis] = alpha / exponent;
      shift_gradients[quantum_count][1][axis] = beta / exponent - 1.0;
      ++quantum_count;
    }
  }

  LowOrderPairGradientExpansion expansion{};
  if (quantum_count == 0) {
    expansion.count = 1;
    expansion.terms[0].coefficient = 1.0;
    return expansion;
  }
  if (quantum_count == 1) {
    expansion.count = 2;
    expansion.terms[0].coefficient = shifts[0];
    expansion.terms[1].derivative_state = derivative_states[0];
    expansion.terms[1].coefficient = inverse_two_exponent;
    for (unsigned center = 0; center < 2; ++center) {
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        expansion.terms[0].center[center][coordinate] =
            shift_gradients[0][center][coordinate];
      }
    }
    return expansion;
  }

  expansion.count = 4;
  expansion.terms[0].coefficient = shifts[0] * shifts[1] +
      (derivative_states[0] == derivative_states[1]
           ? inverse_two_exponent
           : 0.0);
  expansion.terms[1].derivative_state = derivative_states[0];
  expansion.terms[1].coefficient = inverse_two_exponent * shifts[1];
  expansion.terms[2].derivative_state = derivative_states[1];
  expansion.terms[2].coefficient = inverse_two_exponent * shifts[0];
  expansion.terms[3].derivative_state =
      derivative_states[0] + derivative_states[1];
  expansion.terms[3].coefficient =
      inverse_two_exponent * inverse_two_exponent;
  for (unsigned center = 0; center < 2; ++center) {
    for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
      expansion.terms[0].center[center][coordinate] =
          shift_gradients[0][center][coordinate] * shifts[1] +
          shifts[0] * shift_gradients[1][center][coordinate];
      expansion.terms[1].center[center][coordinate] =
          inverse_two_exponent *
          shift_gradients[1][center][coordinate];
      expansion.terms[2].center[center][coordinate] =
          inverse_two_exponent *
          shift_gradients[0][center][coordinate];
    }
  }
  return expansion;
}

/** Evaluate all center derivatives of one total-order-two primitive quartet. */
__device__ void primitive_eri_order2_gradient(
    double alpha,
    const Vec3<double>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<double>& second,
    const Angular& angular_second,
    double gamma,
    const Vec3<double>& third,
    const Angular& angular_third,
    double delta,
    const Vec3<double>& fourth,
    const Angular& angular_fourth,
    double (&gradient)[4][3]) {
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<double> product_p =
      product_center(alpha, first, beta, second);
  const Vec3<double> product_q =
      product_center(gamma, third, delta, fourth);
  const Vec3<double> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };
  const LowOrderPairGradientExpansion first_expansion =
      make_low_order_pair_gradient_expansion(
          alpha, first, angular_first, beta, second, angular_second);
  const LowOrderPairGradientExpansion second_expansion =
      make_low_order_pair_gradient_expansion(
          gamma, third, angular_third, delta, fourth, angular_fourth);
  double boys[4];
  boys_values<3>(rho * distance_squared(product_p, product_q), boys);
  const double product_scales[4] = {
      alpha / p, beta / p, -gamma / q, -delta / q};
  double value = 0.0;
  double value_gradient[4][3]{};
  for (unsigned first_term = 0; first_term < first_expansion.count;
       ++first_term) {
    for (unsigned second_term = 0; second_term < second_expansion.count;
         ++second_term) {
      const LowOrderPairGradientTerm& first_item =
          first_expansion.terms[first_term];
      const LowOrderPairGradientTerm& second_item =
          second_expansion.terms[second_term];
      const double sign =
          (low_order_derivative_total(second_item.derivative_state) & 1U) == 0
          ? 1.0
          : -1.0;
      const unsigned derivative_state =
          first_item.derivative_state + second_item.derivative_state;
      const double coulomb = low_order_coulomb(
          derivative_state, rho, product_difference, boys);
      const double coefficient =
          sign * first_item.coefficient * second_item.coefficient;
      value += coefficient * coulomb;
      for (unsigned center = 0; center < 4; ++center) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          double coefficient_derivative = 0.0;
          if (center < 2) {
            coefficient_derivative = sign *
                first_item.center[center][coordinate] *
                second_item.coefficient;
          } else {
            coefficient_derivative = sign * first_item.coefficient *
                second_item.center[center - 2][coordinate];
          }
          const double coulomb_derivative = product_scales[center] *
              third_order_coulomb(
                  derivative_state +
                      low_order_derivative_state(coordinate),
                  rho, product_difference, boys);
          value_gradient[center][coordinate] +=
              coefficient_derivative * coulomb +
              coefficient * coulomb_derivative;
        }
      }
    }
  }

  const double pair_decay = exp(
      -mu * distance_squared(first, second) -
      nu * distance_squared(third, fourth));
  const double prefactor =
      2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;
  for (unsigned center = 0; center < 4; ++center) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double decay_derivative = 0.0;
      if (center < 2) {
        const double difference =
            vec_axis(first, coordinate) - vec_axis(second, coordinate);
        decay_derivative =
            (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
      } else {
        const double difference =
            vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
        decay_derivative =
            (center == 2 ? -2.0 * nu : 2.0 * nu) * difference;
      }
      gradient[center][coordinate] = prefactor *
          (value_gradient[center][coordinate] + value * decay_derivative);
    }
  }
}

/** Canonicalize and contract all-center gradients for total angular order 2. */
__device__ CartesianQuartetGradient
contracted_eri_cartesian_source_order2_gradient(
    const DeviceBatch& batch,
    std::int32_t system,
    std::int32_t i,
    std::int32_t j,
    std::int32_t k,
    std::int32_t l) {
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base =
      static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if (batch.shell_angular[slots[0].shell] <
      batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] <
      batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell],
      batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell],
      batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] *
      batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] *
      batch.direct_ao_coefficients[slots[3].ao];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient *
              batch.primitive_coefficients[a] *
              batch.primitive_coefficients[b] *
              batch.primitive_coefficients[c] *
              batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          primitive_eri_order2_gradient(
              batch.primitive_exponents[a], positions[0], angular[0],
              batch.primitive_exponents[b], positions[1], angular[1],
              batch.primitive_exponents[c], positions[2], angular[2],
              batch.primitive_exponents[d], positions[3], angular[3],
              primitive_gradient);
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

/** Canonicalize one Cartesian source quartet to its exact shell class. */
template <unsigned ShellClass, typename Scalar>
__device__ Scalar contracted_eri_cartesian_source_shell_class(
    const DeviceBatch& batch,
    std::int32_t system,
    std::int32_t i,
    std::int32_t j,
    std::int32_t k,
    std::int32_t l,
    std::int64_t derivative_coordinate) {
  static_assert(ShellClass < detail::kDirectQuartetShellClassCount);
  constexpr unsigned FirstPairClass =
      direct_triangular_class_high(ShellClass);
  constexpr unsigned SecondPairClass =
      ShellClass - FirstPairClass * (FirstPairClass + 1) / 2;
  constexpr unsigned FirstShellAngular =
      direct_triangular_class_high(FirstPairClass);
  constexpr unsigned SecondShellAngular =
      FirstPairClass -
      FirstShellAngular * (FirstShellAngular + 1) / 2;
  constexpr unsigned ThirdShellAngular =
      direct_triangular_class_high(SecondPairClass);
  constexpr unsigned FourthShellAngular =
      SecondPairClass -
      ThirdShellAngular * (ThirdShellAngular + 1) / 2;

  const std::int64_t base =
      static_cast<std::int64_t>(system) * batch.direct_nbf;
  std::int32_t shell_i = batch.direct_ao_shells[base + i];
  std::int32_t shell_j = batch.direct_ao_shells[base + j];
  std::int32_t shell_k = batch.direct_ao_shells[base + k];
  std::int32_t shell_l = batch.direct_ao_shells[base + l];
  unsigned angular_i = batch.shell_angular[shell_i];
  unsigned angular_j = batch.shell_angular[shell_j];
  unsigned angular_k = batch.shell_angular[shell_k];
  unsigned angular_l = batch.shell_angular[shell_l];

  if (angular_i < angular_j) {
    const std::int32_t ao = i;
    i = j;
    j = ao;
    const std::int32_t shell = shell_i;
    shell_i = shell_j;
    shell_j = shell;
    const unsigned angular = angular_i;
    angular_i = angular_j;
    angular_j = angular;
  }
  if (angular_k < angular_l) {
    const std::int32_t ao = k;
    k = l;
    l = ao;
    const std::int32_t shell = shell_k;
    shell_k = shell_l;
    shell_l = shell;
    const unsigned angular = angular_k;
    angular_k = angular_l;
    angular_l = angular;
  }
  const unsigned first_pair_class =
      direct_shell_pair_class_cuda(angular_i, angular_j);
  const unsigned second_pair_class =
      direct_shell_pair_class_cuda(angular_k, angular_l);
  if (first_pair_class < second_pair_class) {
    const std::int32_t first_ao = i;
    const std::int32_t second_ao = j;
    i = k;
    j = l;
    k = first_ao;
    l = second_ao;
    const std::int32_t first_shell = shell_i;
    const std::int32_t second_shell = shell_j;
    shell_i = shell_k;
    shell_j = shell_l;
    shell_k = first_shell;
    shell_l = second_shell;
  }

  return contracted_eri_cartesian_source_shell_class<
      FirstShellAngular, SecondShellAngular, ThirdShellAngular,
      FourthShellAngular, Scalar>(
      batch, base + i, base + j, base + k, base + l,
      shell_i, shell_j, shell_k, shell_l, derivative_coordinate);
}

/** Dispatch one angular-order task to its Cartesian source evaluator. */
template <unsigned AngularOrder, typename Scalar>
__device__ Scalar dispatch_contracted_eri_cartesian_source_shell_class(
    unsigned runtime_shell_class,
    const DeviceBatch& batch,
    std::int32_t system,
    std::int32_t i,
    std::int32_t j,
    std::int32_t k,
    std::int32_t l,
    std::int64_t derivative_coordinate) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
#define QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(ShellClass)                  \
  case ShellClass:                                                     \
    if constexpr (direct_shell_class_angular_order(ShellClass) ==      \
                  AngularOrder) {                                     \
      return contracted_eri_cartesian_source_shell_class<              \
          ShellClass, Scalar>(batch, system, i, j, k, l,               \
                              derivative_coordinate);                  \
    }                                                                 \
    break
  switch (runtime_shell_class) {
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(0);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(1);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(2);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(3);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(4);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(5);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(6);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(7);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(8);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(9);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(10);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(11);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(12);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(13);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(14);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(15);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(16);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(17);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(18);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(19);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(20);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(21);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(22);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(23);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(24);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(25);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(26);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(27);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(28);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(29);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(30);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(31);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(32);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(33);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(34);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(35);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(36);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(37);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(38);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(39);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(40);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(41);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(42);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(43);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(44);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(45);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(46);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(47);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(48);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(49);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(50);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(51);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(52);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(53);
    QCE_DIRECT_SOURCE_SHELL_CLASS_CASE(54);
  }
#undef QCE_DIRECT_SOURCE_SHELL_CLASS_CASE
  return scalar<Scalar>(0.0);
}

template <typename Scalar>
__device__ Scalar contracted_eri_cartesian_source(
    const DeviceBatch& batch,
    std::int32_t system,
    std::int32_t i,
    std::int32_t j,
    std::int32_t k,
    std::int32_t l,
    std::int64_t derivative_coordinate) {
  const std::int64_t base =
      static_cast<std::int64_t>(system) * batch.direct_nbf;
  const std::int32_t shell_i = batch.direct_ao_shells[base + i];
  const std::int32_t shell_j = batch.direct_ao_shells[base + j];
  const std::int32_t shell_k = batch.direct_ao_shells[base + k];
  const std::int32_t shell_l = batch.direct_ao_shells[base + l];
  const unsigned angular_order = batch.shell_angular[shell_i] +
                                 batch.shell_angular[shell_j] +
                                 batch.shell_angular[shell_k] +
                                 batch.shell_angular[shell_l];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[shell_i], batch.shell_angular[shell_j],
      batch.shell_angular[shell_k], batch.shell_angular[shell_l]);
#define QCE_DIRECT_SOURCE_ANGULAR_CASE(Order)                           \
  case Order:                                                          \
    return dispatch_contracted_eri_cartesian_source_shell_class<       \
        Order, Scalar>(shell_class, batch, system, i, j, k, l,         \
                       derivative_coordinate)
  switch (angular_order) {
    QCE_DIRECT_SOURCE_ANGULAR_CASE(0);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(1);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(2);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(3);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(4);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(5);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(6);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(7);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(8);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(9);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(10);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(11);
    QCE_DIRECT_SOURCE_ANGULAR_CASE(12);
  }
#undef QCE_DIRECT_SOURCE_ANGULAR_CASE
  return scalar<Scalar>(0.0);
}

__global__ void build_one_electron_integrals_kernel(DeviceBatch batch,
                                                     const std::int32_t* pair_first,
                                                     const std::int32_t* pair_second,
                                                     std::size_t pair_count,
                                                     double* overlap,
                                                     double* hcore) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) return;
  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t pair = element % pair_count;
  const std::size_t row = static_cast<std::size_t>(pair_first[pair]);
  const std::size_t column = static_cast<std::size_t>(pair_second[pair]);
  const double overlap_value = contracted_overlap<double>(
      batch, system, static_cast<std::int32_t>(row),
      static_cast<std::int32_t>(column), -1);
  const double hcore_value = contracted_hcore<double>(
      batch, system, static_cast<std::int32_t>(row),
      static_cast<std::int32_t>(column), -1);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  overlap[matrix_offset + matrix_index(row, column, n)] = overlap_value;
  hcore[matrix_offset + matrix_index(row, column, n)] = hcore_value;
  if (row != column) {
    overlap[matrix_offset + matrix_index(column, row, n)] = overlap_value;
    hcore[matrix_offset + matrix_index(column, row, n)] = hcore_value;
  }
}

__global__ void build_eri_kernel(DeviceBatch batch, double* eri) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t eri_size = n * n * n * n;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * eri_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / eri_size);
  std::size_t local = element % eri_size;
  const std::int32_t l = static_cast<std::int32_t>(local % n);
  local /= n;
  const std::int32_t k = static_cast<std::int32_t>(local % n);
  local /= n;
  const std::int32_t j = static_cast<std::int32_t>(local % n);
  const std::int32_t i = static_cast<std::int32_t>(local / n);
  eri[element] = contracted_eri<double>(batch, system, i, j, k, l, -1);
}

__global__ void build_nuclear_repulsion_kernel(DeviceBatch batch,
                                                double* nuclear_repulsion) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch.batch_size) return;
  double result = 0.0;
  for (std::int64_t first = batch.atom_offsets[system];
       first < batch.atom_offsets[system + 1]; ++first) {
    const Vec3<double> a = atom_position<double>(batch, first, -1);
    for (std::int64_t second = batch.atom_offsets[system]; second < first; ++second) {
      const Vec3<double> b = atom_position<double>(batch, second, -1);
      result += static_cast<double>(batch.atomic_numbers[first] *
                                    batch.atomic_numbers[second]) /
                sqrt(distance_squared(a, b));
    }
  }
  nuclear_repulsion[system] = result;
}

__global__ void initialize_state_kernel(std::int32_t batch_size,
                                        std::uint8_t* active,
                                        std::uint8_t* converged,
                                        std::uint8_t* failed,
                                        std::uint32_t* iterations,
                                        double* previous_energy,
                                        double* energy_change,
                                        double* density_rms,
                                        std::uint32_t* diis_count,
                                        std::uint32_t* diis_head) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size) return;
  active[system] = 1;
  converged[system] = 0;
  failed[system] = 0;
  iterations[system] = 0;
  previous_energy[system] = CUDART_INF;
  energy_change[system] = CUDART_INF;
  density_rms[system] = CUDART_INF;
  diis_count[system] = 0;
  diis_head[system] = 0;
}

__global__ void copy_matrix_kernel(std::size_t elements,
                                   const double* source,
                                   double* destination) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element < elements) destination[element] = source[element];
}

__global__ void inspect_solver_kernel(std::int32_t batch_size,
                                      const int* info,
                                      std::uint8_t* active,
                                      std::uint8_t* failed,
                                      std::uint8_t* converged) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size || info[system] == 0) return;
  active[system] = 0;
  failed[system] = 1;
  converged[system] = 0;
}

__global__ void expand_spin_active_kernel(std::int32_t batch_size,
                                          std::int32_t spin_count,
                                          const std::uint8_t* active,
                                          std::uint8_t* spin_active) {
  const std::int32_t state = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  const std::int32_t state_count = batch_size * spin_count;
  if (state < state_count) spin_active[state] = active[state / spin_count];
}

__global__ void inspect_spin_solver_kernel(std::int32_t batch_size,
                                           std::int32_t spin_count,
                                           const int* info,
                                           std::uint8_t* active,
                                           std::uint8_t* failed,
                                           std::uint8_t* converged) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size) return;
  for (std::int32_t spin = 0; spin < spin_count; ++spin) {
    if (info[system * spin_count + spin] != 0) {
      active[system] = 0;
      failed[system] = 1;
      converged[system] = 0;
      return;
    }
  }
}

constexpr std::int32_t kSmallEigensolverLimit = 16;
constexpr std::int32_t kBatchedEigensolverLimit = 32;
constexpr unsigned kGraphEigensolverThreads = 64;
constexpr unsigned kCyclicGraphEigensolverThreads = 256;
constexpr std::int32_t kCyclicGraphEigensolverLimit = 256;

__global__ void symmetric_eigen_small_kernel(std::int32_t batch_size,
                                             std::int32_t nbf,
                                             double* matrices,
                                             double* eigenvalues,
                                             int* info,
                                             const std::uint8_t* active) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || threadIdx.x != 0) return;
  info[system] = 0;
  if (active != nullptr && active[system] == 0) return;
  if (nbf <= 0 || nbf > kSmallEigensolverLimit) {
    info[system] = -1;
    return;
  }
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double matrix[kSmallEigensolverLimit * kSmallEigensolverLimit];
  double vectors[kSmallEigensolverLimit * kSmallEigensolverLimit];
  for (std::size_t column = 0; column < n; ++column) {
    for (std::size_t row = 0; row < n; ++row) {
      matrix[matrix_index(row, column, n)] =
          matrices[offset + matrix_index(row, column, n)];
      vectors[matrix_index(row, column, n)] = row == column ? 1.0 : 0.0;
    }
  }

  const std::size_t maximum_sweeps =
      20 * matrix_size > 50 ? 20 * matrix_size : 50;
  for (std::size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
    std::size_t p = 0;
    std::size_t q = 0;
    double largest = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t column = row + 1; column < n; ++column) {
        const double candidate =
            fabs(matrix[matrix_index(row, column, n)]);
        if (candidate > largest) {
          largest = candidate;
          p = row;
          q = column;
        }
      }
    }
    if (largest < 1.0e-14) break;
    if (sweep + 1 == maximum_sweeps) info[system] = 1;

    const double app = matrix[matrix_index(p, p, n)];
    const double aqq = matrix[matrix_index(q, q, n)];
    const double apq = matrix[matrix_index(p, q, n)];
    const double angle = 0.5 * atan2(2.0 * apq, aqq - app);
    const double cosine = cos(angle);
    const double sine = sin(angle);
    for (std::size_t k = 0; k < n; ++k) {
      if (k == p || k == q) continue;
      const double mkp = matrix[matrix_index(k, p, n)];
      const double mkq = matrix[matrix_index(k, q, n)];
      matrix[matrix_index(k, p, n)] =
          matrix[matrix_index(p, k, n)] = cosine * mkp - sine * mkq;
      matrix[matrix_index(k, q, n)] =
          matrix[matrix_index(q, k, n)] = sine * mkp + cosine * mkq;
    }
    matrix[matrix_index(p, p, n)] =
        cosine * cosine * app - 2.0 * sine * cosine * apq +
        sine * sine * aqq;
    matrix[matrix_index(q, q, n)] =
        sine * sine * app + 2.0 * sine * cosine * apq +
        cosine * cosine * aqq;
    matrix[matrix_index(p, q, n)] = 0.0;
    matrix[matrix_index(q, p, n)] = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
      const double vkp = vectors[matrix_index(row, p, n)];
      const double vkq = vectors[matrix_index(row, q, n)];
      vectors[matrix_index(row, p, n)] = cosine * vkp - sine * vkq;
      vectors[matrix_index(row, q, n)] = sine * vkp + cosine * vkq;
    }
  }

  // Stable selection sort keeps the same ascending eigenpair convention used
  // by the CPU oracle and cuSOLVER path.
  for (std::size_t column = 0; column < n; ++column) {
    std::size_t selected = column;
    for (std::size_t candidate = column + 1; candidate < n; ++candidate) {
      if (matrix[matrix_index(candidate, candidate, n)] <
          matrix[matrix_index(selected, selected, n)]) {
        selected = candidate;
      }
    }
    if (selected != column) {
      const double diagonal = matrix[matrix_index(column, column, n)];
      matrix[matrix_index(column, column, n)] =
          matrix[matrix_index(selected, selected, n)];
      matrix[matrix_index(selected, selected, n)] = diagonal;
      for (std::size_t row = 0; row < n; ++row) {
        const double swap = vectors[matrix_index(row, column, n)];
        vectors[matrix_index(row, column, n)] =
            vectors[matrix_index(row, selected, n)];
        vectors[matrix_index(row, selected, n)] = swap;
      }
    }
    eigenvalues[static_cast<std::size_t>(system) * n + column] =
        matrix[matrix_index(column, column, n)];
  }
  for (std::size_t element = 0; element < matrix_size; ++element) {
    matrices[offset + element] = vectors[element];
  }
}

/**
 * Graph-capture-safe Jacobi eigensolver for AO matrices above the provider's
 * small batched range.
 *
 * One block owns one physical or spin state. Threads cooperatively select the
 * largest off-diagonal element and apply its row/column rotation, while a
 * separate arena matrix retains eigenvectors. This keeps the device-tail SCF
 * loop intact for realistic named bases without cuSOLVER's capture-time host
 * synchronization or a fixed compile-time AO limit.
 */
__global__ void symmetric_eigen_graph_maximum_pivot_kernel(
    std::int32_t batch_size,
    std::int32_t nbf,
    double* matrices,
    double* eigenvectors,
    double* eigenvalues,
    int* info,
    const std::uint8_t* active) {
  static_assert(kGraphEigensolverThreads > 0 &&
                (kGraphEigensolverThreads &
                 (kGraphEigensolverThreads - 1)) == 0);
  const std::int32_t state = static_cast<std::int32_t>(blockIdx.x);
  if (state >= batch_size) return;
  if (active != nullptr && active[state] == 0) {
    if (threadIdx.x == 0) info[state] = 0;
    return;
  }
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t matrix_offset =
      static_cast<std::size_t>(state) * matrix_size;
  const std::size_t eigenvalue_offset = static_cast<std::size_t>(state) * n;

  __shared__ double block_maximum[kGraphEigensolverThreads];
  __shared__ std::size_t block_index[kGraphEigensolverThreads];
  __shared__ std::size_t pivot_p;
  __shared__ std::size_t pivot_q;
  __shared__ double pivot_cosine;
  __shared__ double pivot_sine;
  __shared__ int converged;

  for (std::size_t element = threadIdx.x; element < matrix_size;
       element += blockDim.x) {
    const std::size_t row = element / n;
    const std::size_t column = element % n;
    eigenvectors[matrix_offset + element] = row == column ? 1.0 : 0.0;
  }
  if (threadIdx.x == 0) {
    info[state] = 1;
    converged = 0;
  }
  __syncthreads();

  const std::size_t maximum_rotations =
      20 * matrix_size > 50 ? 20 * matrix_size : 50;
  for (std::size_t rotation = 0; rotation < maximum_rotations; ++rotation) {
    double local_maximum = 0.0;
    std::size_t local_index = 0;
    for (std::size_t element = threadIdx.x; element < matrix_size;
         element += blockDim.x) {
      const std::size_t row = element / n;
      const std::size_t column = element % n;
      if (row >= column) continue;
      const double candidate = fabs(matrices[matrix_offset + element]);
      if (candidate > local_maximum) {
        local_maximum = candidate;
        local_index = element;
      }
    }
    block_maximum[threadIdx.x] = local_maximum;
    block_index[threadIdx.x] = local_index;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride &&
          block_maximum[threadIdx.x + stride] >
              block_maximum[threadIdx.x]) {
        block_maximum[threadIdx.x] =
            block_maximum[threadIdx.x + stride];
        block_index[threadIdx.x] = block_index[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      if (block_maximum[0] < 1.0e-14) {
        converged = 1;
        info[state] = 0;
      } else {
        pivot_p = block_index[0] / n;
        pivot_q = block_index[0] % n;
        const double app =
            matrices[matrix_offset + matrix_index(pivot_p, pivot_p, n)];
        const double aqq =
            matrices[matrix_offset + matrix_index(pivot_q, pivot_q, n)];
        const double apq =
            matrices[matrix_offset + matrix_index(pivot_p, pivot_q, n)];
        const double angle = 0.5 * atan2(2.0 * apq, aqq - app);
        pivot_cosine = cos(angle);
        pivot_sine = sin(angle);
      }
    }
    __syncthreads();
    if (converged != 0) break;

    for (std::size_t k = threadIdx.x; k < n; k += blockDim.x) {
      if (k != pivot_p && k != pivot_q) {
        const double mkp =
            matrices[matrix_offset + matrix_index(k, pivot_p, n)];
        const double mkq =
            matrices[matrix_offset + matrix_index(k, pivot_q, n)];
        const double next_p = pivot_cosine * mkp - pivot_sine * mkq;
        const double next_q = pivot_sine * mkp + pivot_cosine * mkq;
        matrices[matrix_offset + matrix_index(k, pivot_p, n)] = next_p;
        matrices[matrix_offset + matrix_index(pivot_p, k, n)] = next_p;
        matrices[matrix_offset + matrix_index(k, pivot_q, n)] = next_q;
        matrices[matrix_offset + matrix_index(pivot_q, k, n)] = next_q;
      }
      const double vkp =
          eigenvectors[matrix_offset + matrix_index(k, pivot_p, n)];
      const double vkq =
          eigenvectors[matrix_offset + matrix_index(k, pivot_q, n)];
      eigenvectors[matrix_offset + matrix_index(k, pivot_p, n)] =
          pivot_cosine * vkp - pivot_sine * vkq;
      eigenvectors[matrix_offset + matrix_index(k, pivot_q, n)] =
          pivot_sine * vkp + pivot_cosine * vkq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const double app =
          matrices[matrix_offset + matrix_index(pivot_p, pivot_p, n)];
      const double aqq =
          matrices[matrix_offset + matrix_index(pivot_q, pivot_q, n)];
      const double apq =
          matrices[matrix_offset + matrix_index(pivot_p, pivot_q, n)];
      matrices[matrix_offset + matrix_index(pivot_p, pivot_p, n)] =
          pivot_cosine * pivot_cosine * app -
          2.0 * pivot_sine * pivot_cosine * apq +
          pivot_sine * pivot_sine * aqq;
      matrices[matrix_offset + matrix_index(pivot_q, pivot_q, n)] =
          pivot_sine * pivot_sine * app +
          2.0 * pivot_sine * pivot_cosine * apq +
          pivot_cosine * pivot_cosine * aqq;
      matrices[matrix_offset + matrix_index(pivot_p, pivot_q, n)] = 0.0;
      matrices[matrix_offset + matrix_index(pivot_q, pivot_p, n)] = 0.0;
    }
    __syncthreads();
  }

  // Stable selection sort preserves the ascending eigenpair convention of
  // both the CPU oracle and the two smaller CUDA solver paths.
  for (std::size_t column = 0; column < n; ++column) {
    if (threadIdx.x == 0) {
      std::size_t selected = column;
      for (std::size_t candidate = column + 1; candidate < n; ++candidate) {
        if (matrices[matrix_offset + matrix_index(candidate, candidate, n)] <
            matrices[matrix_offset + matrix_index(selected, selected, n)]) {
          selected = candidate;
        }
      }
      block_index[0] = selected;
      if (selected != column) {
        const double diagonal =
            matrices[matrix_offset + matrix_index(column, column, n)];
        matrices[matrix_offset + matrix_index(column, column, n)] =
            matrices[matrix_offset + matrix_index(selected, selected, n)];
        matrices[matrix_offset + matrix_index(selected, selected, n)] =
            diagonal;
      }
    }
    __syncthreads();
    const std::size_t selected = block_index[0];
    if (selected != column) {
      for (std::size_t row = threadIdx.x; row < n; row += blockDim.x) {
        const double swap =
            eigenvectors[matrix_offset + matrix_index(row, column, n)];
        eigenvectors[matrix_offset + matrix_index(row, column, n)] =
            eigenvectors[matrix_offset + matrix_index(row, selected, n)];
        eigenvectors[matrix_offset + matrix_index(row, selected, n)] = swap;
      }
    }
    __syncthreads();
  }
  for (std::size_t column = threadIdx.x; column < n; column += blockDim.x) {
    eigenvalues[eigenvalue_offset + column] =
        matrices[matrix_offset + matrix_index(column, column, n)];
  }
  __syncthreads();
  for (std::size_t element = threadIdx.x; element < matrix_size;
       element += blockDim.x) {
    matrices[matrix_offset + element] =
        eigenvectors[matrix_offset + element];
  }
}

/**
 * Return one disjoint round-robin pair for a cyclic Jacobi sweep.
 *
 * Even dimensions keep the final orbital fixed while the other n-1 orbitals
 * rotate around it. Odd dimensions use one implicit dummy orbital and omit
 * its pair. Every off-diagonal pair appears exactly once per full sweep.
 */
__device__ void cyclic_jacobi_pair(std::size_t n,
                                   std::size_t round,
                                   std::size_t compact_pair,
                                   std::size_t& first,
                                   std::size_t& second) {
  const bool odd = (n & 1U) != 0;
  const std::size_t schedule_size = odd ? n + 1 : n;
  const std::size_t rotating = schedule_size - 1;
  const std::size_t pair = odd ? compact_pair + 1 : compact_pair;
  if (pair == 0) {
    first = schedule_size - 1;
    second = round;
    return;
  }
  first = (round + pair) % rotating;
  second = (round + rotating - pair) % rotating;
}

/**
 * Graph-capture-safe cyclic Jacobi eigensolver for realistic AO matrices.
 *
 * A sweep consists of n-1 round-robin rounds. Each round diagonalizes n/2
 * disjoint 2x2 pivots simultaneously, then applies their block-diagonal
 * rotation to matrix columns, matrix rows, and eigenvectors. This reduces a
 * sweep to O(n^3) work, whereas selecting one global maximum before every
 * rotation rescans O(n^2) entries and becomes O(n^4). One block still owns a
 * complete state, preserving the device-tail Graph and per-system active mask.
 */
__global__ void symmetric_eigen_graph_cyclic_kernel(
    std::int32_t batch_size,
    std::int32_t nbf,
    double* matrices,
    double* eigenvectors,
    double* eigenvalues,
    int* info,
    const std::uint8_t* active) {
  static_assert(kCyclicGraphEigensolverThreads > 0 &&
                (kCyclicGraphEigensolverThreads &
                 (kCyclicGraphEigensolverThreads - 1)) == 0);
  const std::int32_t state = static_cast<std::int32_t>(blockIdx.x);
  if (state >= batch_size) return;
  if (active != nullptr && active[state] == 0) {
    if (threadIdx.x == 0) info[state] = 0;
    return;
  }

  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t matrix_offset =
      static_cast<std::size_t>(state) * matrix_size;
  const std::size_t eigenvalue_offset = static_cast<std::size_t>(state) * n;
  const std::size_t pair_count = n / 2;
  const std::size_t round_count = (n & 1U) != 0 ? n : n - 1;

  extern __shared__ double rotation_parameters[];
  double* rotation_cosines = rotation_parameters;
  double* rotation_sines = rotation_parameters + pair_count;
  __shared__ double block_maximum[kCyclicGraphEigensolverThreads];
  __shared__ std::size_t selected_column;
  __shared__ int converged;

  for (std::size_t element = threadIdx.x; element < matrix_size;
       element += blockDim.x) {
    const std::size_t row = element / n;
    const std::size_t column = element % n;
    eigenvectors[matrix_offset + element] = row == column ? 1.0 : 0.0;
  }
  if (threadIdx.x == 0) {
    info[state] = 1;
    converged = 0;
  }
  __syncthreads();

  constexpr std::size_t maximum_sweeps = 50;
  for (std::size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
    for (std::size_t round = 0; round < round_count; ++round) {
      for (std::size_t pair = threadIdx.x; pair < pair_count;
           pair += blockDim.x) {
        std::size_t first = 0;
        std::size_t second = 0;
        cyclic_jacobi_pair(n, round, pair, first, second);
        const double first_diagonal =
            matrices[matrix_offset + matrix_index(first, first, n)];
        const double second_diagonal =
            matrices[matrix_offset + matrix_index(second, second, n)];
        const double off_diagonal =
            matrices[matrix_offset + matrix_index(first, second, n)];
        double angle = 0.5 * atan2(
            2.0 * off_diagonal, second_diagonal - first_diagonal);
        // The principal atan2 branch can choose an almost-pi/2 rotation that
        // merely swaps diagonal entries. Maximum-pivot Jacobi tolerates that,
        // but a parallel cyclic ordering can repeat the swaps indefinitely.
        // The equivalent rotation in [-pi/4, pi/4] is the standard cyclic
        // Jacobi choice and guarantees progress without changing eigenpairs.
        if (angle > 0.25 * kPi) {
          angle -= 0.5 * kPi;
        } else if (angle < -0.25 * kPi) {
          angle += 0.5 * kPi;
        }
        rotation_cosines[pair] = cos(angle);
        rotation_sines[pair] = sin(angle);
      }
      __syncthreads();

      // Right multiplication A <- A Q. Disjoint column pairs make every
      // output element unique within this stage.
      const std::size_t pair_elements = n * pair_count;
      for (std::size_t task = threadIdx.x; task < pair_elements;
           task += blockDim.x) {
        const std::size_t row = task / pair_count;
        const std::size_t pair = task % pair_count;
        std::size_t first = 0;
        std::size_t second = 0;
        cyclic_jacobi_pair(n, round, pair, first, second);
        const double first_value =
            matrices[matrix_offset + matrix_index(row, first, n)];
        const double second_value =
            matrices[matrix_offset + matrix_index(row, second, n)];
        const double cosine = rotation_cosines[pair];
        const double sine = rotation_sines[pair];
        matrices[matrix_offset + matrix_index(row, first, n)] =
            cosine * first_value - sine * second_value;
        matrices[matrix_offset + matrix_index(row, second, n)] =
            sine * first_value + cosine * second_value;
      }
      __syncthreads();

      // Left multiplication A <- Q^T A uses the same disjoint row pairs.
      for (std::size_t task = threadIdx.x; task < pair_elements;
           task += blockDim.x) {
        const std::size_t column = task / pair_count;
        const std::size_t pair = task % pair_count;
        std::size_t first = 0;
        std::size_t second = 0;
        cyclic_jacobi_pair(n, round, pair, first, second);
        const double first_value =
            matrices[matrix_offset + matrix_index(first, column, n)];
        const double second_value =
            matrices[matrix_offset + matrix_index(second, column, n)];
        const double cosine = rotation_cosines[pair];
        const double sine = rotation_sines[pair];
        matrices[matrix_offset + matrix_index(first, column, n)] =
            cosine * first_value - sine * second_value;
        matrices[matrix_offset + matrix_index(second, column, n)] =
            sine * first_value + cosine * second_value;
      }
      __syncthreads();

      for (std::size_t pair = threadIdx.x; pair < pair_count;
           pair += blockDim.x) {
        std::size_t first = 0;
        std::size_t second = 0;
        cyclic_jacobi_pair(n, round, pair, first, second);
        matrices[matrix_offset + matrix_index(first, second, n)] = 0.0;
        matrices[matrix_offset + matrix_index(second, first, n)] = 0.0;
      }

      // Accumulate V <- V Q after the matrix similarity transform.
      for (std::size_t task = threadIdx.x; task < pair_elements;
           task += blockDim.x) {
        const std::size_t row = task / pair_count;
        const std::size_t pair = task % pair_count;
        std::size_t first = 0;
        std::size_t second = 0;
        cyclic_jacobi_pair(n, round, pair, first, second);
        const double first_value =
            eigenvectors[matrix_offset + matrix_index(row, first, n)];
        const double second_value =
            eigenvectors[matrix_offset + matrix_index(row, second, n)];
        const double cosine = rotation_cosines[pair];
        const double sine = rotation_sines[pair];
        eigenvectors[matrix_offset + matrix_index(row, first, n)] =
            cosine * first_value - sine * second_value;
        eigenvectors[matrix_offset + matrix_index(row, second, n)] =
            sine * first_value + cosine * second_value;
      }
      __syncthreads();
    }

    double local_maximum = 0.0;
    for (std::size_t element = threadIdx.x; element < matrix_size;
         element += blockDim.x) {
      const std::size_t row = element / n;
      const std::size_t column = element % n;
      if (row < column) {
        local_maximum = fmax(
            local_maximum,
            fabs(matrices[matrix_offset + element]));
      }
    }
    block_maximum[threadIdx.x] = local_maximum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
        block_maximum[threadIdx.x] = fmax(
            block_maximum[threadIdx.x],
            block_maximum[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0 && block_maximum[0] < 1.0e-13) {
      converged = 1;
      info[state] = 0;
    }
    __syncthreads();
    if (converged != 0) break;
  }
  // Stable selection sort preserves the ascending eigenpair convention of
  // the CPU oracle and cuSOLVER path.
  for (std::size_t column = 0; column < n; ++column) {
    if (threadIdx.x == 0) {
      std::size_t selected = column;
      for (std::size_t candidate = column + 1; candidate < n; ++candidate) {
        if (matrices[matrix_offset + matrix_index(candidate, candidate, n)] <
            matrices[matrix_offset + matrix_index(selected, selected, n)]) {
          selected = candidate;
        }
      }
      selected_column = selected;
      if (selected != column) {
        const double diagonal =
            matrices[matrix_offset + matrix_index(column, column, n)];
        matrices[matrix_offset + matrix_index(column, column, n)] =
            matrices[matrix_offset + matrix_index(selected, selected, n)];
        matrices[matrix_offset + matrix_index(selected, selected, n)] =
            diagonal;
      }
    }
    __syncthreads();
    if (selected_column != column) {
      for (std::size_t row = threadIdx.x; row < n; row += blockDim.x) {
        const double swap =
            eigenvectors[matrix_offset + matrix_index(row, column, n)];
        eigenvectors[matrix_offset + matrix_index(row, column, n)] =
            eigenvectors[matrix_offset +
                         matrix_index(row, selected_column, n)];
        eigenvectors[matrix_offset +
                     matrix_index(row, selected_column, n)] = swap;
      }
    }
    __syncthreads();
  }
  for (std::size_t column = threadIdx.x; column < n; column += blockDim.x) {
    eigenvalues[eigenvalue_offset + column] =
        matrices[matrix_offset + matrix_index(column, column, n)];
  }
  __syncthreads();
  for (std::size_t element = threadIdx.x; element < matrix_size;
       element += blockDim.x) {
    matrices[matrix_offset + element] =
        eigenvectors[matrix_offset + element];
  }
}

__global__ void build_orthogonalizer_kernel(std::int32_t batch_size,
                                            std::int32_t nbf,
                                            const double* eigenvectors,
                                            const double* eigenvalues,
                                            const std::uint8_t* active,
                                            double* orthogonalizer,
                                            std::uint8_t* failed) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const double* vectors = eigenvectors + static_cast<std::size_t>(system) * matrix_size;
  const double* values = eigenvalues + static_cast<std::size_t>(system) * n;
  double result = 0.0;
  for (std::size_t orbital = 0; orbital < n; ++orbital) {
    if (!(values[orbital] > 1.0e-10)) {
      failed[system] = 1;
      return;
    }
    result += vectors[matrix_index(row, orbital, n)] *
              vectors[matrix_index(column, orbital, n)] /
              sqrt(values[orbital]);
  }
  orthogonalizer[element] = result;
}

__global__ void matrix_product_kernel(std::int32_t batch_size,
                                      std::int32_t nbf,
                                      const double* left,
                                      bool transpose_left,
                                      const double* right,
                                      const std::uint8_t* active,
                                      double* output) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double value = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t left_index = transpose_left
        ? matrix_index(k, row, n)
        : matrix_index(row, k, n);
    value += left[offset + left_index] *
             right[offset + matrix_index(k, column, n)];
  }
  output[element] = value;
}

__global__ void broadcast_spin_matrix_kernel(std::int32_t batch_size,
                                             std::int32_t spin_count,
                                             std::int32_t nbf,
                                             const double* physical_matrices,
                                             const std::uint8_t* active,
                                             double* spin_matrices) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t spin_elements =
      static_cast<std::size_t>(batch_size) * spin_count * matrix_size;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= spin_elements) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active != nullptr && active[system] == 0) return;
  spin_matrices[element] =
      physical_matrices[system * matrix_size + element % matrix_size];
}

__global__ void spin_matrix_product_kernel(std::int32_t batch_size,
                                           std::int32_t spin_count,
                                           std::int32_t nbf,
                                           const double* left,
                                           bool left_is_spin,
                                           bool transpose_left,
                                           const double* right,
                                           bool right_is_spin,
                                           const std::uint8_t* active,
                                           double* output) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t left_offset =
      (left_is_spin ? state : system) * matrix_size;
  const std::size_t right_offset =
      (right_is_spin ? state : system) * matrix_size;
  double value = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t left_index = transpose_left
        ? matrix_index(k, row, n)
        : matrix_index(row, k, n);
    value += left[left_offset + left_index] *
             right[right_offset + matrix_index(k, column, n)];
  }
  output[element] = value;
}

__global__ void build_density_kernel(std::int32_t batch_size,
                                     std::int32_t nbf,
                                     const std::int32_t* occupied,
                                     const double* coefficients,
                                     const std::uint8_t* active,
                                     double* density) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[system]; ++orbital) {
    value += 2.0 * coefficients[offset + matrix_index(row, orbital, n)] *
             coefficients[offset + matrix_index(column, orbital, n)];
  }
  density[element] = value;
}

__global__ void build_spin_density_kernel(std::int32_t batch_size,
                                          std::int32_t spin_count,
                                          std::int32_t nbf,
                                          const std::int32_t* occupied,
                                          const double* coefficients,
                                          const std::uint8_t* active,
                                          double* density) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = state * matrix_size;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[state]; ++orbital) {
    value += coefficients[offset + matrix_index(row, orbital, n)] *
             coefficients[offset + matrix_index(column, orbital, n)];
  }
  density[element] = value;
}

__global__ void mix_open_shell_guess_kernel(std::int32_t batch_size,
                                            std::int32_t nbf,
                                            const std::int32_t* occupied,
                                            const std::uint8_t* active,
                                            double* coefficients) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * n) return;
  const std::size_t system = element / n;
  if (active != nullptr && active[system] == 0) return;
  const std::size_t row = element % n;
  const std::int32_t alpha_occupied = occupied[system * 2];
  const std::int32_t beta_occupied = occupied[system * 2 + 1];
  if (alpha_occupied == beta_occupied || beta_occupied <= 0 ||
      beta_occupied >= nbf) {
    return;
  }

  // Match the CPU open-shell cold guess: preserve the beta orbital metric
  // while breaking exact spatial symmetry between its frontier orbitals.
  constexpr double cosine = 0.7071067811865476;
  constexpr double sine = 0.7071067811865476;
  const std::size_t matrix_size = n * n;
  const std::size_t offset = (system * 2 + 1) * matrix_size;
  const std::size_t occupied_orbital =
      static_cast<std::size_t>(beta_occupied - 1);
  const std::size_t virtual_orbital =
      static_cast<std::size_t>(beta_occupied);
  const double occupied_value =
      coefficients[offset + matrix_index(row, occupied_orbital, n)];
  const double virtual_value =
      coefficients[offset + matrix_index(row, virtual_orbital, n)];
  coefficients[offset + matrix_index(row, occupied_orbital, n)] =
      cosine * occupied_value + sine * virtual_value;
  coefficients[offset + matrix_index(row, virtual_orbital, n)] =
      -sine * occupied_value + cosine * virtual_value;
}

__global__ void apply_warm_density_kernel(std::int32_t batch_size,
                                          std::int32_t nbf,
                                          const std::int32_t* occupied,
                                          const std::uint8_t* warm_mask,
                                          const double* warm_density,
                                          const double* overlap,
                                          double* density) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || warm_mask[system] == 0 || threadIdx.x != 0) return;
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double trace = 0.0;
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      const double symmetric = 0.5 *
          (warm_density[offset + matrix_index(row, column, n)] +
           warm_density[offset + matrix_index(column, row, n)]);
      density[offset + matrix_index(row, column, n)] = symmetric;
      trace += symmetric * overlap[offset + matrix_index(column, row, n)];
    }
  }
  const double target = 2.0 * occupied[system];
  if (isfinite(trace) && fabs(trace) > 1.0e-14) {
    const double scale = target / trace;
    for (std::size_t element = 0; element < matrix_size; ++element) {
      density[offset + element] *= scale;
    }
  }
}

__global__ void apply_uhf_warm_density_kernel(
    std::int32_t batch_size,
    std::int32_t nbf,
    const std::int32_t* occupied,
    const std::uint8_t* warm_mask,
    const double* warm_density,
    const double* overlap,
    double* density) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || warm_mask[system] == 0 || threadIdx.x != 0) return;
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t overlap_offset =
      static_cast<std::size_t>(system) * matrix_size;
  for (std::int32_t spin = 0; spin < 2; ++spin) {
    const std::size_t state = static_cast<std::size_t>(system) * 2 + spin;
    const std::size_t offset = state * matrix_size;
    double trace = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t column = 0; column < n; ++column) {
        const double symmetric = 0.5 *
            (warm_density[offset + matrix_index(row, column, n)] +
             warm_density[offset + matrix_index(column, row, n)]);
        density[offset + matrix_index(row, column, n)] = symmetric;
        trace += symmetric *
            overlap[overlap_offset + matrix_index(column, row, n)];
      }
    }
    const double target = static_cast<double>(occupied[state]);
    if (target == 0.0) {
      for (std::size_t element = 0; element < matrix_size; ++element) {
        density[offset + element] = 0.0;
      }
    } else if (isfinite(trace) && fabs(trace) > 1.0e-14) {
      const double scale = target / trace;
      for (std::size_t element = 0; element < matrix_size; ++element) {
        density[offset + element] *= scale;
      }
    }
  }
}

__global__ void build_fock_kernel(std::int32_t batch_size,
                                  std::int32_t nbf,
                                  const double* hcore,
                                  const double* eri,
                                  const double* density,
                                  const std::uint8_t* active,
                                  double* fock) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local % n;
  const std::size_t j = local / n;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t eri_offset = static_cast<std::size_t>(system) * eri_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const double pkl = density[matrix_offset + matrix_index(k, l, n)];
      coulomb += pkl * eri[eri_offset + eri_index(i, j, k, l, n)];
      exchange += pkl * eri[eri_offset + eri_index(i, k, j, l, n)];
    }
  }
  fock[element] = hcore[element] + coulomb - 0.5 * exchange;
}

__global__ void build_uhf_fock_kernel(std::int32_t batch_size,
                                      std::int32_t nbf,
                                      const double* hcore,
                                      const double* eri,
                                      const double* density,
                                      const std::uint8_t* active,
                                      double* fock) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * 2 * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / 2;
  if (active != nullptr && active[system] == 0) return;
  const std::size_t spin = state % 2;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local % n;
  const std::size_t j = local / n;
  const std::size_t physical_matrix_offset = system * matrix_size;
  const std::size_t eri_offset = system * eri_size;
  const std::size_t alpha_offset = system * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t spin_offset = alpha_offset + spin * matrix_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const std::size_t kl = matrix_index(k, l, n);
      const double total = density[alpha_offset + kl] +
                           density[beta_offset + kl];
      coulomb += total * eri[eri_offset + eri_index(i, j, k, l, n)];
      exchange += density[spin_offset + kl] *
                  eri[eri_offset + eri_index(i, k, j, l, n)];
    }
  }
  fock[element] = hcore[physical_matrix_offset + local] + coulomb - exchange;
}

__global__ void build_schwarz_bounds_packed_kernel(
    DeviceBatch batch,
    std::size_t pair_count,
    double* schwarz_bounds) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system =
      static_cast<std::int32_t>(element / pair_count);
  const std::size_t pair = element % pair_count;
  std::size_t i = 0;
  std::size_t j = 0;
  decode_lower_triangle(pair, i, j);
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_offset =
      static_cast<std::size_t>(system) * n * n;
  const double diagonal = contracted_eri_cartesian_source<double>(
      batch, system, static_cast<std::int32_t>(i),
      static_cast<std::int32_t>(j), static_cast<std::int32_t>(i),
      static_cast<std::int32_t>(j), -1);
  // fabs is conservative when roundoff makes a non-negative diagonal
  // slightly negative; it never converts that noise into a false zero.
  const double bound = sqrt(fabs(diagonal));
  schwarz_bounds[matrix_offset + matrix_index(i, j, n)] = bound;
  if (i != j) {
    schwarz_bounds[matrix_offset + matrix_index(j, i, n)] = bound;
  }
}

__global__ void reduce_shell_pair_bounds_kernel(
    DeviceBatch batch,
    const double* schwarz_bounds,
    double* shell_pair_bounds) {
  extern __shared__ double block_maxima[];
  const std::size_t shell_pair = static_cast<std::size_t>(blockIdx.x);
  if (shell_pair >= static_cast<std::size_t>(batch.total_shell_pairs)) return;
  const std::int32_t system = batch.shell_pair_systems[shell_pair];
  const std::int32_t first_shell = batch.shell_pair_first[shell_pair];
  const std::int32_t second_shell = batch.shell_pair_second[shell_pair];
  const std::size_t first_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[first_shell]);
  const std::size_t first_count =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[first_shell + 1]) -
      first_begin;
  const std::size_t second_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[second_shell]);
  const std::size_t second_count =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[second_shell + 1]) -
      second_begin;
  const bool same_shell = first_shell == second_shell;
  const std::size_t ao_pair_count = same_shell
      ? first_count * (first_count + 1) / 2
      : first_count * second_count;
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * n * n;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;

  double local_maximum = 0.0;
  for (std::size_t ordinal = threadIdx.x; ordinal < ao_pair_count;
       ordinal += blockDim.x) {
    std::size_t first_component = 0;
    std::size_t second_component = 0;
    if (same_shell) {
      decode_lower_triangle(ordinal, first_component, second_component);
    } else {
      first_component = ordinal / second_count;
      second_component = ordinal % second_count;
    }
    const std::size_t i = first_begin + first_component - system_ao_begin;
    const std::size_t j = second_begin + second_component - system_ao_begin;
    local_maximum = fmax(
        local_maximum,
        schwarz_bounds[matrix_offset + matrix_index(i, j, n)]);
  }

  block_maxima[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_maxima[threadIdx.x] =
          fmax(block_maxima[threadIdx.x], block_maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) shell_pair_bounds[shell_pair] = block_maxima[0];
}

__global__ void compact_active_shell_quartet_tiles_kernel(
    DeviceBatch batch,
    double screening_tolerance,
    const double* shell_pair_bounds,
    const std::uint32_t* active_shell_quartet_tile_offsets,
    std::uint32_t* active_shell_quartet_tile_counts,
    ActiveShellQuartetTile* active_shell_quartet_tiles) {
  const std::size_t shell_quartet =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (shell_quartet >=
      static_cast<std::size_t>(batch.total_shell_quartets)) {
    return;
  }

  const std::int32_t system = shell_quartet_system(batch, shell_quartet);
  const std::size_t local_quartet = shell_quartet -
      static_cast<std::size_t>(batch.system_shell_quartet_offsets[system]);
  std::size_t first_pair_local = 0;
  std::size_t second_pair_local = 0;
  decode_lower_triangle(local_quartet, first_pair_local, second_pair_local);
  const std::size_t pair_begin =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
  const std::size_t first_pair = pair_begin + first_pair_local;
  const std::size_t second_pair = pair_begin + second_pair_local;
  if (shell_pair_bounds[first_pair] * shell_pair_bounds[second_pair] <
      screening_tolerance) {
    return;
  }

  const std::size_t first_ao_pair_count =
      shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count =
      shell_ao_pair_count(batch, second_pair);
  const std::size_t ao_quartet_count = first_pair == second_pair
      ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
      : first_ao_pair_count * second_ao_pair_count;
  const std::uint32_t tile_count = static_cast<std::uint32_t>(
      (ao_quartet_count + detail::kDirectQuartetTileSize - 1) /
      detail::kDirectQuartetTileSize);
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const unsigned angular_order = batch.shell_angular[first_shell] +
                                 batch.shell_angular[second_shell] +
                                 batch.shell_angular[third_shell] +
                                 batch.shell_angular[fourth_shell];
  if (angular_order >= detail::kDirectQuartetAngularOrderCount) return;

  // Compaction expands each active shell quartet into only its populated AO
  // tiles inside a fixed angular-order partition. Exact shell-class dispatch
  // happens inside the consumer so Graph replay retains only 13 launch nodes.
  // Order within one partition need not be stable because consumers use
  // double atomics and promise numerical, rather than bitwise, replay.
  const std::uint32_t slot = active_shell_quartet_tile_offsets[angular_order] +
      atomicAdd(active_shell_quartet_tile_counts + angular_order, tile_count);
  for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
    active_shell_quartet_tiles[slot + tile] = {
        static_cast<std::uint32_t>(first_pair),
        static_cast<std::uint32_t>(second_pair), tile};
  }
}

__global__ void build_fock_direct_packed_kernel(
    DeviceBatch batch,
    double screening_tolerance,
    const double* hcore,
    const std::int32_t* ao_pair_first,
    const std::int32_t* ao_pair_second,
    std::size_t pair_count,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock) {
  extern __shared__ double pair_sums[];
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t matrix_element = static_cast<std::size_t>(blockIdx.x);
  if (matrix_element >=
      static_cast<std::size_t>(batch.batch_size) * matrix_size) {
    return;
  }
  const std::int32_t system =
      static_cast<std::int32_t>(matrix_element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local_matrix = matrix_element % matrix_size;
  const std::size_t i = local_matrix % n;
  const std::size_t j = local_matrix / n;
  const std::size_t matrix_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const double bound_ij =
      schwarz_bounds[matrix_offset + matrix_index(i, j, n)];

  double contribution = 0.0;
  for (std::size_t pair = threadIdx.x; pair < pair_count;
       pair += blockDim.x) {
    const std::size_t k = static_cast<std::size_t>(ao_pair_first[pair]);
    const std::size_t l = static_cast<std::size_t>(ao_pair_second[pair]);
    const double pkl = density[matrix_offset + matrix_index(k, l, n)];
    if (pkl == 0.0) continue;

    if (bound_ij *
            schwarz_bounds[matrix_offset + matrix_index(k, l, n)] >=
        screening_tolerance) {
      const double pair_weight = k == l ? pkl : 2.0 * pkl;
      contribution += pair_weight * contracted_eri<double>(
          batch, system, static_cast<std::int32_t>(i),
          static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
          static_cast<std::int32_t>(l), -1);
    }
    if (schwarz_bounds[matrix_offset + matrix_index(i, k, n)] *
            schwarz_bounds[matrix_offset + matrix_index(j, l, n)] >=
        screening_tolerance) {
      contribution -= 0.5 * pkl * contracted_eri<double>(
          batch, system, static_cast<std::int32_t>(i),
          static_cast<std::int32_t>(k), static_cast<std::int32_t>(j),
          static_cast<std::int32_t>(l), -1);
    }
    if (k != l &&
        schwarz_bounds[matrix_offset + matrix_index(i, l, n)] *
                schwarz_bounds[matrix_offset + matrix_index(j, k, n)] >=
            screening_tolerance) {
      contribution -= 0.5 * pkl * contracted_eri<double>(
          batch, system, static_cast<std::int32_t>(i),
          static_cast<std::int32_t>(l), static_cast<std::int32_t>(j),
          static_cast<std::int32_t>(k), -1);
    }
  }

  pair_sums[threadIdx.x] = contribution;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      pair_sums[threadIdx.x] += pair_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    fock[matrix_element] = hcore[matrix_element] + pair_sums[0];
  }
}

__global__ void build_uhf_fock_direct_packed_kernel(
    DeviceBatch batch,
    double screening_tolerance,
    const double* hcore,
    const std::int32_t* ao_pair_first,
    const std::int32_t* ao_pair_second,
    std::size_t pair_count,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock) {
  extern __shared__ double pair_sums[];
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t matrix_element = static_cast<std::size_t>(blockIdx.x);
  if (matrix_element >=
      static_cast<std::size_t>(batch.batch_size) * 2 * matrix_size) {
    return;
  }
  const std::size_t state = matrix_element / matrix_size;
  const std::size_t system = state / 2;
  if (active != nullptr && active[system] == 0) return;
  const std::size_t spin = state % 2;
  const std::size_t local_matrix = matrix_element % matrix_size;
  const std::size_t i = local_matrix % n;
  const std::size_t j = local_matrix / n;
  const std::size_t physical_offset = system * matrix_size;
  const std::size_t alpha_offset = system * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t spin_offset = alpha_offset + spin * matrix_size;
  const double bound_ij =
      schwarz_bounds[physical_offset + matrix_index(i, j, n)];

  double contribution = 0.0;
  for (std::size_t pair = threadIdx.x; pair < pair_count;
       pair += blockDim.x) {
    const std::size_t k = static_cast<std::size_t>(ao_pair_first[pair]);
    const std::size_t l = static_cast<std::size_t>(ao_pair_second[pair]);
    const std::size_t kl = matrix_index(k, l, n);
    const double alpha = density[alpha_offset + kl];
    const double beta = density[beta_offset + kl];
    const double same_spin = density[spin_offset + kl];
    const double total = alpha + beta;
    if (total == 0.0 && same_spin == 0.0) continue;

    if (total != 0.0 &&
        bound_ij * schwarz_bounds[physical_offset + kl] >=
            screening_tolerance) {
      const double pair_weight = k == l ? total : 2.0 * total;
      contribution += pair_weight * contracted_eri<double>(
          batch, static_cast<std::int32_t>(system),
          static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
          static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), -1);
    }
    if (same_spin != 0.0 &&
        schwarz_bounds[physical_offset + matrix_index(i, k, n)] *
                schwarz_bounds[physical_offset + matrix_index(j, l, n)] >=
            screening_tolerance) {
      contribution -= same_spin * contracted_eri<double>(
          batch, static_cast<std::int32_t>(system),
          static_cast<std::int32_t>(i), static_cast<std::int32_t>(k),
          static_cast<std::int32_t>(j), static_cast<std::int32_t>(l), -1);
    }
    if (k != l && same_spin != 0.0 &&
        schwarz_bounds[physical_offset + matrix_index(i, l, n)] *
                schwarz_bounds[physical_offset + matrix_index(j, k, n)] >=
            screening_tolerance) {
      contribution -= same_spin * contracted_eri<double>(
          batch, static_cast<std::int32_t>(system),
          static_cast<std::int32_t>(i), static_cast<std::int32_t>(l),
          static_cast<std::int32_t>(j), static_cast<std::int32_t>(k), -1);
    }
  }

  pair_sums[threadIdx.x] = contribution;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      pair_sums[threadIdx.x] += pair_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    fock[matrix_element] = hcore[physical_offset + local_matrix] + pair_sums[0];
  }
}

__device__ void eri_symmetry_permutation(unsigned permutation,
                                         std::size_t i,
                                         std::size_t j,
                                         std::size_t k,
                                         std::size_t l,
                                         std::size_t& a,
                                         std::size_t& b,
                                         std::size_t& c,
                                         std::size_t& d) {
  switch (permutation) {
    case 0: a = i; b = j; c = k; d = l; break;
    case 1: a = j; b = i; c = k; d = l; break;
    case 2: a = i; b = j; c = l; d = k; break;
    case 3: a = j; b = i; c = l; d = k; break;
    case 4: a = k; b = l; c = i; d = j; break;
    case 5: a = l; b = k; c = i; d = j; break;
    case 6: a = k; b = l; c = j; d = i; break;
    default: a = l; b = k; c = j; d = i; break;
  }
}

__device__ bool unique_eri_symmetry_permutation(unsigned permutation,
                                                std::size_t i,
                                                std::size_t j,
                                                std::size_t k,
                                                std::size_t l,
                                                std::size_t a,
                                                std::size_t b,
                                                std::size_t c,
                                                std::size_t d) {
  for (unsigned previous = 0; previous < permutation; ++previous) {
    std::size_t pa = 0;
    std::size_t pb = 0;
    std::size_t pc = 0;
    std::size_t pd = 0;
    eri_symmetry_permutation(previous, i, j, k, l, pa, pb, pc, pd);
    if (a == pa && b == pb && c == pc && d == pd) return false;
  }
  return true;
}

__global__ void initialize_direct_fock_kernel(
    std::int32_t batch_size,
    std::int32_t matrices_per_system,
    std::int32_t nbf,
    const double* hcore,
    const std::uint8_t* active,
    double* fock) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t matrix_count =
      static_cast<std::size_t>(batch_size) * matrices_per_system;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system =
      state / static_cast<std::size_t>(matrices_per_system);
  if (active != nullptr && active[system] == 0) return;
  fock[element] = hcore[system * matrix_size + element % matrix_size];
}

__global__ void clear_active_matrices_kernel(
    std::int32_t batch_size,
    std::int32_t matrices_per_system,
    std::int32_t nbf,
    const std::uint8_t* active,
    double* matrices) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t matrix_count =
      static_cast<std::size_t>(batch_size) * matrices_per_system;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system =
      state / static_cast<std::size_t>(matrices_per_system);
  if (active[system] != 0) matrices[element] = 0.0;
}

/** First stage of D_cart = C^T D_public C. */
__global__ void transform_density_to_direct_right_kernel(
    std::int32_t batch_size,
    std::int32_t spin_count,
    std::int32_t nbf,
    std::int32_t direct_nbf,
    const double* transform,
    const double* density,
    const std::uint8_t* active,
    double* temporary) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t direct_n = static_cast<std::size_t>(direct_nbf);
  const std::size_t rectangular_size = n * direct_n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * rectangular_size) return;
  const std::size_t state = element / rectangular_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;
  const std::size_t local = element % rectangular_size;
  const std::size_t row = local % n;
  const std::size_t direct_column = local / n;
  const std::size_t density_offset = state * n * n;
  const std::size_t transform_offset = system * rectangular_size;
  double value = 0.0;
  for (std::size_t column = 0; column < n; ++column) {
    value += density[density_offset + matrix_index(row, column, n)] *
        transform[transform_offset + column + direct_column * n];
  }
  temporary[element] = value;
}

/** Second stage of D_cart = C^T (D_public C). */
__global__ void transform_density_to_direct_left_kernel(
    std::int32_t batch_size,
    std::int32_t spin_count,
    std::int32_t nbf,
    std::int32_t direct_nbf,
    const double* transform,
    const double* temporary,
    const std::uint8_t* active,
    double* direct_density) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t direct_n = static_cast<std::size_t>(direct_nbf);
  const std::size_t matrix_size = direct_n * direct_n;
  const std::size_t rectangular_size = n * direct_n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t direct_row = local % direct_n;
  const std::size_t direct_column = local / direct_n;
  const std::size_t transform_offset = system * rectangular_size;
  const std::size_t temporary_offset = state * rectangular_size;
  double value = 0.0;
  for (std::size_t row = 0; row < n; ++row) {
    value += transform[transform_offset + row + direct_row * n] *
        temporary[temporary_offset + row + direct_column * n];
  }
  direct_density[element] = value;
}

/** First stage of F_public = C F_cart C^T. */
__global__ void transform_direct_fock_left_kernel(
    std::int32_t batch_size,
    std::int32_t spin_count,
    std::int32_t nbf,
    std::int32_t direct_nbf,
    const double* transform,
    const double* direct_fock,
    const std::uint8_t* active,
    double* temporary) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t direct_n = static_cast<std::size_t>(direct_nbf);
  const std::size_t direct_matrix_size = direct_n * direct_n;
  const std::size_t rectangular_size = n * direct_n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * rectangular_size) return;
  const std::size_t state = element / rectangular_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;
  const std::size_t local = element % rectangular_size;
  const std::size_t public_row = local % n;
  const std::size_t direct_column = local / n;
  const std::size_t transform_offset = system * rectangular_size;
  const std::size_t direct_offset = state * direct_matrix_size;
  double value = 0.0;
  for (std::size_t direct_row = 0; direct_row < direct_n; ++direct_row) {
    value += transform[transform_offset + public_row + direct_row * n] *
        direct_fock[direct_offset +
                    matrix_index(direct_row, direct_column, direct_n)];
  }
  temporary[element] = value;
}

/** Finish F_public = (C F_cart) C^T and restore the one-electron matrix. */
__global__ void transform_direct_fock_right_kernel(
    std::int32_t batch_size,
    std::int32_t spin_count,
    std::int32_t nbf,
    std::int32_t direct_nbf,
    const double* transform,
    const double* temporary,
    const double* hcore,
    const std::uint8_t* active,
    double* fock) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t direct_n = static_cast<std::size_t>(direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t rectangular_size = n * direct_n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t public_row = local % n;
  const std::size_t public_column = local / n;
  const std::size_t transform_offset = system * rectangular_size;
  const std::size_t temporary_offset = state * rectangular_size;
  double value = hcore[system * matrix_size + local];
  for (std::size_t direct_column = 0; direct_column < direct_n;
       ++direct_column) {
    value += temporary[temporary_offset + public_row + direct_column * n] *
        transform[transform_offset + public_column + direct_column * n];
  }
  fock[element] = value;
}

template <bool Unrestricted, unsigned AngularOrder>
__global__ void build_fock_direct_quartet_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  const std::size_t active_subtile = static_cast<std::size_t>(blockIdx.x);
  const std::size_t active_tile =
      active_subtile / detail::kDirectQuartetSubtilesPerTile;
  if (active_tile >=
      static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const std::size_t subtile =
      active_subtile % detail::kDirectQuartetSubtilesPerTile;
  const ActiveShellQuartetTile task =
      active_shell_quartet_tiles[active_tile];
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active != nullptr && active[system] == 0) return;
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_ao_pair_count =
      shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count =
      shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
      ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
      : first_ao_pair_count * second_ao_pair_count;
  const std::size_t ordinal =
      static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize +
      subtile * blockDim.x + threadIdx.x;
  if (ordinal < ao_quartet_count) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    std::size_t l = 0;
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin,
                         i, j);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin,
                         k, l);
    if (schwarz_bounds[physical_offset + matrix_index(i, j, n)] *
            schwarz_bounds[physical_offset + matrix_index(k, l, n)] <
        screening_tolerance) {
      return;
    }
    const double integral =
        dispatch_contracted_eri_cartesian_source_shell_class<
            AngularOrder, double>(
            shell_class, batch, system, static_cast<std::int32_t>(i),
            static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
            static_cast<std::int32_t>(l), -1);
    if (integral == 0.0) return;

    for (unsigned permutation = 0; permutation < 8; ++permutation) {
      std::size_t a = 0;
      std::size_t b = 0;
      std::size_t c = 0;
      std::size_t d = 0;
      eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);
      if (!unique_eri_symmetry_permutation(
              permutation, i, j, k, l, a, b, c, d)) {
        continue;
      }
      const std::size_t ab = matrix_index(a, b, n);
      const std::size_t ac = matrix_index(a, c, n);
      const std::size_t cd = matrix_index(c, d, n);
      const std::size_t bd = matrix_index(b, d, n);
      if constexpr (Unrestricted) {
        const double alpha_cd = density[spin_offset + cd];
        const double beta_cd = density[spin_offset + matrix_size + cd];
        const double total_cd = alpha_cd + beta_cd;
        if (total_cd != 0.0) {
          atomicAdd(fock + spin_offset + ab, total_cd * integral);
          atomicAdd(fock + spin_offset + matrix_size + ab,
                    total_cd * integral);
        }
        const double alpha_bd = density[spin_offset + bd];
        const double beta_bd = density[spin_offset + matrix_size + bd];
        if (alpha_bd != 0.0) {
          atomicAdd(fock + spin_offset + ac, -alpha_bd * integral);
        }
        if (beta_bd != 0.0) {
          atomicAdd(fock + spin_offset + matrix_size + ac,
                    -beta_bd * integral);
        }
      } else {
        const double density_cd = density[physical_offset + cd];
        const double density_bd = density[physical_offset + bd];
        if (density_cd != 0.0) {
          atomicAdd(fock + physical_offset + ab, density_cd * integral);
        }
        if (density_bd != 0.0) {
          atomicAdd(fock + physical_offset + ac,
                    -0.5 * density_bd * integral);
        }
      }
    }
  }
}

__global__ void build_commutator_residual_kernel(
    std::int32_t batch_size,
    std::int32_t nbf,
    const double* fock,
    const double* density,
    const double* overlap,
    const std::uint8_t* active,
    double* residual) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double fps = 0.0;
  double spf = 0.0;
  for (std::size_t first = 0; first < n; ++first) {
    for (std::size_t second = 0; second < n; ++second) {
      fps += fock[offset + matrix_index(row, first, n)] *
             density[offset + matrix_index(first, second, n)] *
             overlap[offset + matrix_index(second, column, n)];
      spf += overlap[offset + matrix_index(row, first, n)] *
             density[offset + matrix_index(first, second, n)] *
             fock[offset + matrix_index(second, column, n)];
    }
  }
  residual[element] = fps - spf;
}

__global__ void build_spin_commutator_residual_kernel(
    std::int32_t batch_size,
    std::int32_t spin_count,
    std::int32_t nbf,
    const double* fock,
    const double* density,
    const double* overlap,
    const std::uint8_t* active,
    double* residual) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count =
      static_cast<std::size_t>(batch_size) * spin_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / static_cast<std::size_t>(spin_count);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t spin_offset = state * matrix_size;
  const std::size_t overlap_offset = system * matrix_size;
  double fps = 0.0;
  double spf = 0.0;
  for (std::size_t first = 0; first < n; ++first) {
    for (std::size_t second = 0; second < n; ++second) {
      fps += fock[spin_offset + matrix_index(row, first, n)] *
             density[spin_offset + matrix_index(first, second, n)] *
             overlap[overlap_offset + matrix_index(second, column, n)];
      spf += overlap[overlap_offset + matrix_index(row, first, n)] *
             density[spin_offset + matrix_index(first, second, n)] *
             fock[spin_offset + matrix_index(second, column, n)];
    }
  }
  residual[element] = fps - spf;
}

__global__ void update_diis_kernel(std::int32_t batch_size,
                                   std::int32_t nbf,
                                   std::int32_t matrices_per_system,
                                   std::uint32_t history_capacity,
                                   const double* fock,
                                   const double* residual,
                                   const std::uint8_t* active,
                                   double* fock_history,
                                   double* residual_history,
                                   double* linear_system,
                                   double* coefficients,
                                   std::uint32_t* history_count,
                                   std::uint32_t* history_head,
                                   double* effective_fock) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || active[system] == 0 || threadIdx.x != 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t vector_size =
      matrix_size * static_cast<std::size_t>(matrices_per_system);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * vector_size;
  if (history_capacity < 2) {
    for (std::size_t element = 0; element < vector_size; ++element) {
      effective_fock[matrix_offset + element] = fock[matrix_offset + element];
    }
    return;
  }

  const std::size_t history_stride =
      static_cast<std::size_t>(history_capacity) * vector_size;
  const std::uint32_t slot = history_head[system];
  const std::size_t slot_offset =
      static_cast<std::size_t>(system) * history_stride +
      static_cast<std::size_t>(slot) * vector_size;
  for (std::size_t element = 0; element < vector_size; ++element) {
    fock_history[slot_offset + element] = fock[matrix_offset + element];
    residual_history[slot_offset + element] = residual[matrix_offset + element];
  }
  const std::uint32_t count =
      history_count[system] < history_capacity
          ? history_count[system] + 1
          : history_capacity;
  history_count[system] = count;
  history_head[system] = (slot + 1) % history_capacity;
  if (count < 2) {
    for (std::size_t element = 0; element < vector_size; ++element) {
      effective_fock[matrix_offset + element] = fock[matrix_offset + element];
    }
    return;
  }

  const std::uint32_t dimension = count + 1;
  const std::size_t system_stride =
      static_cast<std::size_t>(history_capacity + 1) * (history_capacity + 1);
  double* matrix = linear_system + static_cast<std::size_t>(system) * system_stride;
  double* rhs = coefficients +
      static_cast<std::size_t>(system) * (history_capacity + 1);
  for (std::uint32_t row = 0; row < dimension; ++row) {
    rhs[row] = row == count ? -1.0 : 0.0;
    for (std::uint32_t column = 0; column < dimension; ++column) {
      matrix[static_cast<std::size_t>(row) * dimension + column] = 0.0;
    }
  }
  for (std::uint32_t row = 0; row < count; ++row) {
    const std::size_t row_offset =
        static_cast<std::size_t>(system) * history_stride +
        static_cast<std::size_t>(row) * vector_size;
    for (std::uint32_t column = 0; column < count; ++column) {
      const std::size_t column_offset =
          static_cast<std::size_t>(system) * history_stride +
          static_cast<std::size_t>(column) * vector_size;
      double dot = 0.0;
      for (std::size_t element = 0; element < vector_size; ++element) {
        dot += residual_history[row_offset + element] *
               residual_history[column_offset + element];
      }
      matrix[static_cast<std::size_t>(row) * dimension + column] = dot;
    }
    matrix[static_cast<std::size_t>(row) * dimension + count] = -1.0;
    matrix[static_cast<std::size_t>(count) * dimension + row] = -1.0;
  }

  bool nonsingular = true;
  for (std::uint32_t column = 0; column < dimension; ++column) {
    std::uint32_t pivot = column;
    for (std::uint32_t row = column + 1; row < dimension; ++row) {
      if (fabs(matrix[static_cast<std::size_t>(row) * dimension + column]) >
          fabs(matrix[static_cast<std::size_t>(pivot) * dimension + column])) {
        pivot = row;
      }
    }
    const double diagonal =
        matrix[static_cast<std::size_t>(pivot) * dimension + column];
    if (fabs(diagonal) < 1.0e-14) {
      nonsingular = false;
      break;
    }
    if (pivot != column) {
      for (std::uint32_t item = 0; item < dimension; ++item) {
        const std::size_t first =
            static_cast<std::size_t>(column) * dimension + item;
        const std::size_t second =
            static_cast<std::size_t>(pivot) * dimension + item;
        const double swap = matrix[first];
        matrix[first] = matrix[second];
        matrix[second] = swap;
      }
      const double swap = rhs[column];
      rhs[column] = rhs[pivot];
      rhs[pivot] = swap;
    }
    const double scale =
        matrix[static_cast<std::size_t>(column) * dimension + column];
    for (std::uint32_t item = column; item < dimension; ++item) {
      matrix[static_cast<std::size_t>(column) * dimension + item] /= scale;
    }
    rhs[column] /= scale;
    for (std::uint32_t row = 0; row < dimension; ++row) {
      if (row == column) continue;
      const double factor =
          matrix[static_cast<std::size_t>(row) * dimension + column];
      for (std::uint32_t item = column; item < dimension; ++item) {
        matrix[static_cast<std::size_t>(row) * dimension + item] -=
            factor * matrix[static_cast<std::size_t>(column) * dimension + item];
      }
      rhs[row] -= factor * rhs[column];
    }
  }

  for (std::size_t element = 0; element < vector_size; ++element) {
    double value = fock[matrix_offset + element];
    if (nonsingular) {
      value = 0.0;
      for (std::uint32_t item = 0; item < count; ++item) {
        const std::size_t item_offset =
            static_cast<std::size_t>(system) * history_stride +
            static_cast<std::size_t>(item) * vector_size;
        value += rhs[item] * fock_history[item_offset + element];
      }
    }
    effective_fock[matrix_offset + element] = value;
  }
}

__global__ void compute_energy_kernel(std::int32_t batch_size,
                                      std::int32_t nbf,
                                      const double* density,
                                      const double* hcore,
                                      const double* fock,
                                      const double* nuclear_repulsion,
                                      const std::uint8_t* active,
                                      double* energy) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size || (active != nullptr && active[system] == 0)) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double value = nuclear_repulsion[system];
  for (std::size_t element = 0; element < matrix_size; ++element) {
    value += 0.5 * density[offset + element] *
             (hcore[offset + element] + fock[offset + element]);
  }
  energy[system] = value;
}

__global__ void compute_uhf_energy_kernel(std::int32_t batch_size,
                                          std::int32_t nbf,
                                          const double* density,
                                          const double* hcore,
                                          const double* fock,
                                          const double* nuclear_repulsion,
                                          const std::uint8_t* active,
                                          double* energy) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size || (active != nullptr && active[system] == 0)) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t alpha_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  double value = nuclear_repulsion[system];
  for (std::size_t element = 0; element < matrix_size; ++element) {
    value += 0.5 * density[alpha_offset + element] *
             (hcore[physical_offset + element] + fock[alpha_offset + element]);
    value += 0.5 * density[beta_offset + element] *
             (hcore[physical_offset + element] + fock[beta_offset + element]);
  }
  energy[system] = value;
}

__global__ void update_convergence_kernel(std::int32_t batch_size,
                                          std::int32_t nbf,
                                          double energy_tolerance,
                                          double density_tolerance,
                                          const double* energy,
                                          double* previous_energy,
                                          const double* next_density,
                                          double* density,
                                          std::uint8_t* active,
                                          std::uint8_t* converged,
                                          std::uint32_t* iterations,
                                          double* energy_change,
                                          double* density_rms) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  const std::uint32_t iteration = iterations[system] + 1;
  double square = 0.0;
  for (std::size_t element = 0; element < matrix_size; ++element) {
    const double delta = next_density[offset + element] - density[offset + element];
    square += delta * delta;
    density[offset + element] = next_density[offset + element];
  }
  const double change = isfinite(previous_energy[system])
      ? fabs(energy[system] - previous_energy[system])
      : CUDART_INF;
  const double rms = sqrt(square / static_cast<double>(matrix_size));
  iterations[system] = iteration;
  energy_change[system] = change;
  density_rms[system] = rms;
  if (iteration > 1 && change < energy_tolerance && rms < density_tolerance) {
    converged[system] = 1;
    active[system] = 0;
  } else {
    previous_energy[system] = energy[system];
  }
}

__global__ void update_uhf_convergence_kernel(
    std::int32_t batch_size,
    std::int32_t nbf,
    double energy_tolerance,
    double density_tolerance,
    const double* energy,
    double* previous_energy,
    const double* next_density,
    double* density,
    std::uint8_t* active,
    std::uint8_t* converged,
    std::uint32_t* iterations,
    double* energy_change,
    double* density_rms) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t vector_size = 2 * matrix_size;
  const std::size_t offset = static_cast<std::size_t>(system) * vector_size;
  double square = 0.0;
  for (std::size_t element = 0; element < vector_size; ++element) {
    const double delta = next_density[offset + element] - density[offset + element];
    square += delta * delta;
    density[offset + element] = next_density[offset + element];
  }
  const double change = isfinite(previous_energy[system])
      ? fabs(energy[system] - previous_energy[system])
      : CUDART_INF;
  const double rms = sqrt(square / static_cast<double>(vector_size));
  previous_energy[system] = energy[system];
  energy_change[system] = change;
  density_rms[system] = rms;
  ++iterations[system];
  if (iterations[system] > 1 && change < energy_tolerance &&
      rms < density_tolerance) {
    converged[system] = 1;
    active[system] = 0;
  }
}

__global__ void tail_rhf_loop_kernel(std::int32_t batch_size,
                                     std::uint32_t maximum_iterations,
                                     const std::uint8_t* active,
                                     const std::uint32_t* iterations) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  bool continue_loop = false;
  for (std::int32_t system = 0; system < batch_size; ++system) {
    continue_loop = continue_loop ||
        (active[system] == 1 && iterations[system] < maximum_iterations);
  }
  if (!continue_loop) return;

  // Re-launch the currently executing one-iteration Graph on its tail stream.
  // This is the same device-resident early-stop pattern used by xTBloom: the
  // host submits one Graph and never polls convergence between iterations.
  const cudaGraphExec_t current = cudaGetCurrentGraphExec();
  if (current != nullptr) {
    (void)cudaGraphLaunch(current, cudaStreamGraphTailLaunch);
  }
}

__global__ void select_converged_kernel(std::int32_t batch_size,
                                        const std::uint8_t* converged,
                                        const std::uint8_t* failed,
                                        std::uint8_t* active) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system < batch_size) {
    active[system] = converged[system] == 1 && failed[system] == 0 ? 1 : 0;
  }
}

__global__ void build_weighted_density_kernel(std::int32_t batch_size,
                                              std::int32_t nbf,
                                              const std::int32_t* occupied,
                                              const double* coefficients,
                                              const double* orbital_energies,
                                              const std::uint8_t* active,
                                              double* weighted_density) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t eigen_offset = static_cast<std::size_t>(system) * n;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[system]; ++orbital) {
    value += 2.0 * orbital_energies[eigen_offset + orbital] *
             coefficients[offset + matrix_index(row, orbital, n)] *
             coefficients[offset + matrix_index(column, orbital, n)];
  }
  weighted_density[element] = value;
}

__global__ void build_spin_weighted_density_kernel(
    std::int32_t batch_size,
    std::int32_t nbf,
    const std::int32_t* occupied,
    const double* coefficients,
    const double* orbital_energies,
    const std::uint8_t* active,
    double* weighted_density) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t state_count = static_cast<std::size_t>(batch_size) * 2;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= state_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / 2;
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t row = local % n;
  const std::size_t column = local / n;
  const std::size_t offset = state * matrix_size;
  const std::size_t eigen_offset = state * n;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[state]; ++orbital) {
    value += orbital_energies[eigen_offset + orbital] *
             coefficients[offset + matrix_index(row, orbital, n)] *
             coefficients[offset + matrix_index(column, orbital, n)];
  }
  weighted_density[element] = value;
}

__global__ void sum_uhf_spin_matrices_kernel(std::int32_t batch_size,
                                             std::int32_t nbf,
                                             const double* spin_matrices,
                                             const std::uint8_t* active,
                                             double* total_matrices) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::size_t system = element / matrix_size;
  if (active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t alpha_offset = system * 2 * matrix_size;
  total_matrices[element] = spin_matrices[alpha_offset + local] +
                            spin_matrices[alpha_offset + matrix_size + local];
}

__global__ void nuclear_force_kernel(DeviceBatch batch,
                                     const std::uint8_t* active,
                                     double* forces) {
  const std::int64_t coordinate =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (coordinate >= batch.total_atoms * 3) return;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  Dual derivative{0.0, 0.0};
  for (std::int64_t first = batch.atom_offsets[system];
       first < batch.atom_offsets[system + 1]; ++first) {
    const Vec3<Dual> a = atom_position<Dual>(batch, first, coordinate);
    for (std::int64_t second = batch.atom_offsets[system]; second < first; ++second) {
      const Vec3<Dual> b = atom_position<Dual>(batch, second, coordinate);
      derivative = derivative +
          static_cast<double>(batch.atomic_numbers[first] *
                              batch.atomic_numbers[second]) /
          qsqrt(distance_squared(a, b));
    }
  }
  forces[coordinate] = -derivative.derivative;
}

__global__ void one_electron_force_kernel(DeviceBatch batch,
                                          const std::int32_t* pair_first,
                                          const std::int32_t* pair_second,
                                          std::size_t pair_count,
                                          const double* density,
                                          const double* weighted_density,
                                          const std::uint8_t* active,
                                          double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t work_per_coordinate = pair_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count =
      static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * work_per_coordinate) return;
  const std::int64_t coordinate = static_cast<std::int64_t>(
      element / work_per_coordinate);
  const std::size_t local = element % work_per_coordinate;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t i = static_cast<std::size_t>(pair_first[local]);
  const std::size_t j = static_cast<std::size_t>(pair_second[local]);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double pij = density[matrix_offset + matrix_index(i, j, n)];
  const double wij = weighted_density[matrix_offset + matrix_index(i, j, n)];
  const Dual ds = contracted_overlap<Dual>(
      batch, system, static_cast<std::int32_t>(i),
      static_cast<std::int32_t>(j), coordinate);
  const Dual dh = contracted_hcore<Dual>(
      batch, system, static_cast<std::int32_t>(i),
      static_cast<std::int32_t>(j), coordinate);
  const double pair_weight = i == j ? 1.0 : 2.0;
  atomicAdd(forces + coordinate,
            -pair_weight * (pij * dh.derivative - wij * ds.derivative));
}

__global__ void two_electron_force_kernel(DeviceBatch batch,
                                          const double* density,
                                          const std::uint8_t* active,
                                          double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t quartet_count = matrix_size * matrix_size;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count =
      static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * quartet_count) return;
  const std::int64_t coordinate =
      static_cast<std::int64_t>(element / quartet_count);
  std::size_t local = element % quartet_count;
  const std::size_t l = local % n;
  local /= n;
  const std::size_t k = local % n;
  local /= n;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double coefficient =
      0.5 * density[matrix_offset + matrix_index(i, j, n)] *
                density[matrix_offset + matrix_index(k, l, n)] -
      0.25 * density[matrix_offset + matrix_index(i, k, n)] *
                 density[matrix_offset + matrix_index(j, l, n)];
  if (coefficient == 0.0) return;
  const Dual derivative = contracted_eri<Dual>(
      batch, system, static_cast<std::int32_t>(i),
      static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
      static_cast<std::int32_t>(l), coordinate);
  atomicAdd(forces + coordinate, -coefficient * derivative.derivative);
}

__global__ void two_electron_uhf_force_kernel(
    DeviceBatch batch,
    const double* spin_density,
    const std::uint8_t* active,
    double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t quartet_count = matrix_size * matrix_size;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count =
      static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * quartet_count) return;
  const std::int64_t coordinate =
      static_cast<std::int64_t>(element / quartet_count);
  std::size_t local = element % quartet_count;
  const std::size_t l = local % n;
  local /= n;
  const std::size_t k = local % n;
  local /= n;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t alpha_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t ij = matrix_index(i, j, n);
  const std::size_t kl = matrix_index(k, l, n);
  const double alpha_ij = spin_density[alpha_offset + ij];
  const double beta_ij = spin_density[beta_offset + ij];
  const double alpha_kl = spin_density[alpha_offset + kl];
  const double beta_kl = spin_density[beta_offset + kl];
  const double total_ij = alpha_ij + beta_ij;
  const double total_kl = alpha_kl + beta_kl;
  const double coefficient =
      0.5 * total_ij * total_kl -
      0.5 * spin_density[alpha_offset + matrix_index(i, k, n)] *
            spin_density[alpha_offset + matrix_index(j, l, n)] -
      0.5 * spin_density[beta_offset + matrix_index(i, k, n)] *
            spin_density[beta_offset + matrix_index(j, l, n)];
  if (coefficient == 0.0) return;
  const Dual derivative = contracted_eri<Dual>(
      batch, system, static_cast<std::int32_t>(i),
      static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
      static_cast<std::int32_t>(l), coordinate);
  atomicAdd(forces + coordinate, -coefficient * derivative.derivative);
}

__global__ void two_electron_force_direct_kernel(
    DeviceBatch batch,
    double screening_tolerance,
    const std::int32_t* pair_first,
    const std::int32_t* pair_second,
    std::size_t pair_count,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t work_per_coordinate = matrix_size * pair_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count =
      static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * work_per_coordinate) return;
  const std::int64_t coordinate =
      static_cast<std::int64_t>(element / work_per_coordinate);
  std::size_t local = element % work_per_coordinate;
  const std::size_t packed_kl = local % pair_count;
  local /= pair_count;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::size_t k =
      static_cast<std::size_t>(pair_first[packed_kl]);
  const std::size_t l =
      static_cast<std::size_t>(pair_second[packed_kl]);
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double pij = density[matrix_offset + matrix_index(i, j, n)];
  const double pkl = density[matrix_offset + matrix_index(k, l, n)];
  if (pij == 0.0 || pkl == 0.0) return;

  double energy_derivative = 0.0;
  const double coulomb_bound =
      schwarz_bounds[matrix_offset + matrix_index(i, j, n)] *
      schwarz_bounds[matrix_offset + matrix_index(k, l, n)];
  if (coulomb_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i),
        static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
        static_cast<std::int32_t>(l), coordinate);
    // The packed density pair represents both (k,l) and (l,k) when k != l.
    const double coulomb_coefficient =
        k == l ? 0.5 * pij * pkl : pij * pkl;
    energy_derivative += coulomb_coefficient * derivative.derivative;
  }

  const double exchange_bound =
      schwarz_bounds[matrix_offset + matrix_index(i, k, n)] *
      schwarz_bounds[matrix_offset + matrix_index(j, l, n)];
  if (exchange_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i),
        static_cast<std::int32_t>(k), static_cast<std::int32_t>(j),
        static_cast<std::int32_t>(l), coordinate);
    energy_derivative -= 0.25 * pij * pkl * derivative.derivative;
  }
  if (k != l) {
    // Packing (k,l) also represents the swapped density pair. Unlike the
    // Coulomb term, exchange maps it to a distinct integral permutation.
    const double transposed_exchange_bound =
        schwarz_bounds[matrix_offset + matrix_index(i, l, n)] *
        schwarz_bounds[matrix_offset + matrix_index(j, k, n)];
    if (transposed_exchange_bound >= screening_tolerance) {
      const Dual derivative = contracted_eri<Dual>(
          batch, system, static_cast<std::int32_t>(i),
          static_cast<std::int32_t>(l), static_cast<std::int32_t>(j),
          static_cast<std::int32_t>(k), coordinate);
      energy_derivative -= 0.25 * pij * pkl * derivative.derivative;
    }
  }
  if (energy_derivative != 0.0) {
    atomicAdd(forces + coordinate, -energy_derivative);
  }
}

__global__ void two_electron_uhf_force_direct_kernel(
    DeviceBatch batch,
    double screening_tolerance,
    const std::int32_t* pair_first,
    const std::int32_t* pair_second,
    std::size_t pair_count,
    const double* schwarz_bounds,
    const double* spin_density,
    const std::uint8_t* active,
    double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t work_per_coordinate = matrix_size * pair_count;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count =
      static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * work_per_coordinate) return;
  const std::int64_t coordinate =
      static_cast<std::int64_t>(element / work_per_coordinate);
  std::size_t local = element % work_per_coordinate;
  const std::size_t packed_kl = local % pair_count;
  local /= pair_count;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::size_t k = static_cast<std::size_t>(pair_first[packed_kl]);
  const std::size_t l = static_cast<std::size_t>(pair_second[packed_kl]);
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t alpha_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t ij = matrix_index(i, j, n);
  const std::size_t kl = matrix_index(k, l, n);
  const double alpha_ij = spin_density[alpha_offset + ij];
  const double beta_ij = spin_density[beta_offset + ij];
  const double alpha_kl = spin_density[alpha_offset + kl];
  const double beta_kl = spin_density[beta_offset + kl];
  const double total_ij = alpha_ij + beta_ij;
  const double total_kl = alpha_kl + beta_kl;

  double energy_derivative = 0.0;
  const double coulomb_bound =
      schwarz_bounds[physical_offset + ij] *
      schwarz_bounds[physical_offset + kl];
  if (total_ij != 0.0 && total_kl != 0.0 &&
      coulomb_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i),
        static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
        static_cast<std::int32_t>(l), coordinate);
    const double coefficient =
        k == l ? 0.5 * total_ij * total_kl : total_ij * total_kl;
    energy_derivative += coefficient * derivative.derivative;
  }

  const double exchange_pair =
      alpha_ij * alpha_kl + beta_ij * beta_kl;
  const double exchange_bound =
      schwarz_bounds[physical_offset + matrix_index(i, k, n)] *
      schwarz_bounds[physical_offset + matrix_index(j, l, n)];
  if (exchange_pair != 0.0 && exchange_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i),
        static_cast<std::int32_t>(k), static_cast<std::int32_t>(j),
        static_cast<std::int32_t>(l), coordinate);
    energy_derivative -= 0.5 * exchange_pair * derivative.derivative;
  }
  if (k != l && exchange_pair != 0.0) {
    const double transposed_exchange_bound =
        schwarz_bounds[physical_offset + matrix_index(i, l, n)] *
        schwarz_bounds[physical_offset + matrix_index(j, k, n)];
    if (transposed_exchange_bound >= screening_tolerance) {
      const Dual derivative = contracted_eri<Dual>(
          batch, system, static_cast<std::int32_t>(i),
          static_cast<std::int32_t>(l), static_cast<std::int32_t>(j),
          static_cast<std::int32_t>(k), coordinate);
      energy_derivative -= 0.5 * exchange_pair * derivative.derivative;
    }
  }
  if (energy_derivative != 0.0) {
    atomicAdd(forces + coordinate, -energy_derivative);
  }
}

template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  const std::size_t active_subtile = static_cast<std::size_t>(blockIdx.x);
  const std::size_t active_tile =
      active_subtile / detail::kDirectQuartetSubtilesPerTile;
  // Consume the identical compact tile list as direct Fock so energy and
  // derivative screening cover precisely the same AO-quartet domain.
  if (active_tile >=
      static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const std::size_t subtile =
      active_subtile % detail::kDirectQuartetSubtilesPerTile;
  const ActiveShellQuartetTile task =
      active_shell_quartet_tiles[active_tile];
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_ao_pair_count =
      shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count =
      shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
      ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
      : first_ao_pair_count * second_ao_pair_count;
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[first_shell], batch.shell_atoms[second_shell],
      batch.shell_atoms[third_shell], batch.shell_atoms[fourth_shell]};
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);

  const std::size_t ordinal =
      static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize +
      subtile * blockDim.x + threadIdx.x;
  if (ordinal < ao_quartet_count) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    std::size_t l = 0;
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin,
                         i, j);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin,
                         k, l);
    if (schwarz_bounds[physical_offset + matrix_index(i, j, n)] *
            schwarz_bounds[physical_offset + matrix_index(k, l, n)] <
        screening_tolerance) {
      return;
    }

    double coefficient = 0.0;
    for (unsigned permutation = 0; permutation < 8; ++permutation) {
      std::size_t a = 0;
      std::size_t b = 0;
      std::size_t c = 0;
      std::size_t d = 0;
      eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);
      if (!unique_eri_symmetry_permutation(
              permutation, i, j, k, l, a, b, c, d)) {
        continue;
      }
      const std::size_t ab = matrix_index(a, b, n);
      const std::size_t ac = matrix_index(a, c, n);
      const std::size_t cd = matrix_index(c, d, n);
      const std::size_t bd = matrix_index(b, d, n);
      if constexpr (Unrestricted) {
        const double total_ab = density[spin_offset + ab] +
            density[spin_offset + matrix_size + ab];
        const double total_cd = density[spin_offset + cd] +
            density[spin_offset + matrix_size + cd];
        coefficient += 0.5 * total_ab * total_cd;
        coefficient -= 0.5 *
            (density[spin_offset + ac] * density[spin_offset + bd] +
             density[spin_offset + matrix_size + ac] *
                 density[spin_offset + matrix_size + bd]);
      } else {
        coefficient +=
            0.5 * density[physical_offset + ab] *
                density[physical_offset + cd] -
            0.25 * density[physical_offset + ac] *
                density[physical_offset + bd];
      }
    }
    if (coefficient == 0.0) return;

    // An ERI is invariant when all four basis centers translate together, so
    // its derivatives over the unique participating atoms sum to zero. Build
    // that unique list, evaluate only N-1 centers, and recover the final one
    // from the negative sum. This halves two-center work and removes one third
    // of three-center work without changing the analytic-gradient contract.
    std::int32_t unique_center_atoms[4];
    unsigned unique_center_count = 0;
    for (unsigned center = 0; center < 4; ++center) {
      bool duplicate_center = false;
      for (unsigned previous = 0; previous < unique_center_count; ++previous) {
        duplicate_center = duplicate_center ||
            center_atoms[center] == unique_center_atoms[previous];
      }
      if (!duplicate_center) {
        unique_center_atoms[unique_center_count++] = center_atoms[center];
      }
    }
    double explicit_unique_gradient[4][3]{};
    if constexpr (AngularOrder <= 2) {
      CartesianQuartetGradient explicit_gradient{};
      if constexpr (AngularOrder <= 1) {
        explicit_gradient =
            contracted_eri_cartesian_source_order01_gradient<AngularOrder>(
                batch, system, static_cast<std::int32_t>(i),
                static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
                static_cast<std::int32_t>(l));
      } else {
        explicit_gradient = contracted_eri_cartesian_source_order2_gradient(
            batch, system, static_cast<std::int32_t>(i),
            static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
            static_cast<std::int32_t>(l));
      }
      for (unsigned center = 0; center < 4; ++center) {
        unsigned unique_center = 0;
        while (unique_center_atoms[unique_center] != center_atoms[center]) {
          ++unique_center;
        }
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          explicit_unique_gradient[unique_center][coordinate] +=
              explicit_gradient.center[center][coordinate];
        }
      }
    }
    double derivative_sum_x = 0.0;
    double derivative_sum_y = 0.0;
    double derivative_sum_z = 0.0;
    for (unsigned center = 0; center + 1 < unique_center_count; ++center) {
      const std::int64_t coordinate =
          static_cast<std::int64_t>(unique_center_atoms[center]) * 3;
      double derivative_x = 0.0;
      double derivative_y = 0.0;
      double derivative_z = 0.0;
      if constexpr (AngularOrder <= 2) {
        derivative_x = explicit_unique_gradient[center][0];
        derivative_y = explicit_unique_gradient[center][1];
        derivative_z = explicit_unique_gradient[center][2];
      } else {
        const Dual3 derivative =
            dispatch_contracted_eri_cartesian_source_shell_class<
                AngularOrder, Dual3>(
                shell_class, batch, system, static_cast<std::int32_t>(i),
                static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
                static_cast<std::int32_t>(l), coordinate);
        derivative_x = derivative.derivative_x;
        derivative_y = derivative.derivative_y;
        derivative_z = derivative.derivative_z;
      }
      derivative_sum_x += derivative_x;
      derivative_sum_y += derivative_y;
      derivative_sum_z += derivative_z;
      if (derivative_x != 0.0) {
        atomicAdd(forces + coordinate,
                  -coefficient * derivative_x);
      }
      if (derivative_y != 0.0) {
        atomicAdd(forces + coordinate + 1,
                  -coefficient * derivative_y);
      }
      if (derivative_z != 0.0) {
        atomicAdd(forces + coordinate + 2,
                  -coefficient * derivative_z);
      }
    }
    if (unique_center_count > 1) {
      const std::int64_t final_coordinate =
          static_cast<std::int64_t>(
              unique_center_atoms[unique_center_count - 1]) * 3;
      if (derivative_sum_x != 0.0) {
        atomicAdd(forces + final_coordinate,
                  coefficient * derivative_sum_x);
      }
      if (derivative_sum_y != 0.0) {
        atomicAdd(forces + final_coordinate + 1,
                  coefficient * derivative_sum_y);
      }
      if (derivative_sum_z != 0.0) {
        atomicAdd(forces + final_coordinate + 2,
                  coefficient * derivative_sum_z);
      }
    }
  }
}

template <bool Unrestricted, unsigned AngularOrder = 0>
void launch_angular_fock_quartets(
    cudaStream_t stream,
    const std::array<std::size_t,
                     detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t,
                     detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch,
    const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock) {
  if constexpr (AngularOrder < detail::kDirectQuartetAngularOrderCount) {
    if (capacities[AngularOrder] != 0) {
      build_fock_direct_quartet_kernel<Unrestricted, AngularOrder><<<
          static_cast<unsigned>(
              capacities[AngularOrder] *
              detail::kDirectQuartetSubtilesPerTile),
          detail::kDirectQuartetThreads, 0, stream>>>(
          batch, active_tile_counts + AngularOrder,
          active_tiles + offsets[AngularOrder], screening_tolerance,
          schwarz_bounds, density, active, fock);
    }
    launch_angular_fock_quartets<Unrestricted, AngularOrder + 1>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        screening_tolerance, schwarz_bounds, density, active, fock);
  }
}

template <bool Unrestricted, unsigned AngularOrder = 0>
void launch_angular_force_quartets(
    cudaStream_t stream,
    const std::array<std::size_t,
                     detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t,
                     detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch,
    const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces) {
  if constexpr (AngularOrder < detail::kDirectQuartetAngularOrderCount) {
    if (capacities[AngularOrder] != 0) {
      two_electron_force_quartet_kernel<Unrestricted, AngularOrder><<<
          static_cast<unsigned>(
              capacities[AngularOrder] *
              detail::kDirectQuartetSubtilesPerTile),
          detail::kDirectQuartetThreads, 0, stream>>>(
          batch, active_tile_counts + AngularOrder,
          active_tiles + offsets[AngularOrder], screening_tolerance,
          schwarz_bounds, density, active, forces);
    }
    launch_angular_force_quartets<Unrestricted, AngularOrder + 1>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        screening_tolerance, schwarz_bounds, density, active, forces);
  }
}

bool checked_multiply(std::size_t first,
                      std::size_t second,
                      std::size_t& result) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

bool checked_add(std::size_t first, std::size_t second, std::size_t& result) {
  if (second > std::numeric_limits<std::size_t>::max() - first) return false;
  result = first + second;
  return true;
}

struct ArenaLayout {
  std::size_t bytes{};
  std::size_t atom_offsets{};
  std::size_t atom_systems{};
  std::size_t atomic_numbers{};
  std::size_t positions{};
  std::size_t system_shell_offsets{};
  std::size_t shell_atoms{};
  std::size_t shell_angular{};
  std::size_t shell_ao_offsets{};
  std::size_t shell_direct_ao_offsets{};
  std::size_t shell_primitive_offsets{};
  std::size_t system_shell_pair_offsets{};
  std::size_t system_shell_quartet_offsets{};
  std::size_t shell_pair_systems{};
  std::size_t shell_pair_first{};
  std::size_t shell_pair_second{};
  std::size_t ao_shells{};
  std::size_t ao_term_counts{};
  std::size_t ao_term_angular{};
  std::size_t ao_term_coefficients{};
  std::size_t direct_ao_shells{};
  std::size_t direct_ao_angular{};
  std::size_t direct_ao_coefficients{};
  std::size_t ao_to_direct_transform{};
  std::size_t primitive_exponents{};
  std::size_t primitive_coefficients{};
  std::size_t occupied{};
  std::size_t warm_mask{};
  std::size_t warm_density{};
  std::size_t overlap{};
  std::size_t hcore{};
  std::size_t eri{};
  std::size_t schwarz_bounds{};
  std::size_t direct_density{};
  std::size_t direct_fock{};
  std::size_t direct_transform_temporary{};
  std::size_t shell_pair_bounds{};
  std::size_t active_shell_quartet_tile_offsets{};
  std::size_t active_shell_quartet_tile_counts{};
  std::size_t active_shell_quartet_tiles{};
  std::size_t ao_pair_first{};
  std::size_t ao_pair_second{};
  std::size_t nuclear_repulsion{};
  std::size_t orthogonalizer{};
  std::size_t temporary{};
  std::size_t eigensystem{};
  std::size_t coefficients{};
  std::size_t eigenvalues{};
  std::size_t density{};
  std::size_t next_density{};
  std::size_t fock{};
  std::size_t residual{};
  std::size_t weighted_density{};
  std::size_t total_density{};
  std::size_t total_weighted_density{};
  std::size_t fock_history{};
  std::size_t residual_history{};
  std::size_t diis_linear_system{};
  std::size_t diis_coefficients{};
  std::size_t diis_count{};
  std::size_t diis_head{};
  std::size_t energy{};
  std::size_t previous_energy{};
  std::size_t energy_change{};
  std::size_t density_rms{};
  std::size_t forces{};
  std::size_t active{};
  std::size_t converged{};
  std::size_t failed{};
  std::size_t spin_active{};
  std::size_t iterations{};
  std::size_t solver_info{};
};

template <typename T>
bool append_array(std::size_t count, std::size_t& cursor, std::size_t& offset) {
  const std::size_t remainder = cursor % alignof(T);
  if (remainder != 0 &&
      !checked_add(cursor, alignof(T) - remainder, cursor)) return false;
  offset = cursor;
  std::size_t bytes = 0;
  return checked_multiply(count, sizeof(T), bytes) &&
         checked_add(cursor, bytes, cursor);
}

bool make_layout(std::size_t batch_size,
                 std::size_t nbf,
                 std::size_t direct_nbf,
                 std::size_t atoms,
                 std::size_t shell_count,
                 std::size_t shell_pair_count,
                 std::size_t shell_quartet_tile_count,
                 std::size_t primitives,
                 std::size_t diis_history,
                 std::size_t spin_count,
                 bool persistent_eri,
                 bool transformed_direct,
                 ArenaLayout& layout) {
  std::size_t matrix_size = 0;
  std::size_t eri_size = 0;
  std::size_t matrices = 0;
  std::size_t spin_matrices = 0;
  std::size_t eris = 0;
  std::size_t aos = 0;
  std::size_t direct_aos = 0;
  std::size_t direct_matrix_size = 0;
  std::size_t direct_matrices = 0;
  std::size_t direct_spin_matrices = 0;
  std::size_t transform_elements = 0;
  std::size_t transform_temporaries = 0;
  std::size_t nbf_plus_one = 0;
  std::size_t pair_product = 0;
  if (!checked_multiply(nbf, nbf, matrix_size) ||
      !checked_multiply(matrix_size, matrix_size, eri_size) ||
      !checked_multiply(batch_size, matrix_size, matrices) ||
      !checked_multiply(matrices, spin_count, spin_matrices) ||
      !checked_multiply(batch_size, nbf, aos) ||
      !checked_multiply(batch_size, direct_nbf, direct_aos) ||
      !checked_multiply(direct_nbf, direct_nbf, direct_matrix_size) ||
      !checked_multiply(batch_size, direct_matrix_size, direct_matrices) ||
      !checked_multiply(direct_matrices, spin_count, direct_spin_matrices) ||
      !checked_multiply(aos, direct_nbf, transform_elements) ||
      !checked_multiply(transform_elements, spin_count,
                        transform_temporaries) ||
      !checked_add(nbf, 1, nbf_plus_one) ||
      !checked_multiply(nbf, nbf_plus_one, pair_product)) return false;
  const std::size_t pair_count = pair_product / 2;
  if (persistent_eri && !checked_multiply(batch_size, eri_size, eris)) {
    return false;
  }
  std::size_t history_matrices = 0;
  std::size_t diis_dimension = 0;
  std::size_t diis_linear_elements = 0;
  if (!checked_multiply(spin_matrices, diis_history, history_matrices) ||
      !checked_add(diis_history, 1, diis_dimension) ||
      !checked_multiply(diis_dimension, diis_dimension, diis_linear_elements) ||
      !checked_multiply(diis_linear_elements, batch_size,
                        diis_linear_elements)) return false;
  std::size_t cursor = 0;
  ArenaLayout made{};
  if (!append_array<std::int64_t>(batch_size + 1, cursor, made.atom_offsets) ||
      !append_array<std::int32_t>(atoms, cursor, made.atom_systems) ||
      !append_array<std::int32_t>(atoms, cursor, made.atomic_numbers) ||
      !append_array<double>(atoms * 3, cursor, made.positions) ||
      !append_array<std::int64_t>(batch_size + 1, cursor,
                                  made.system_shell_offsets) ||
      !append_array<std::int32_t>(shell_count, cursor, made.shell_atoms) ||
      !append_array<std::uint8_t>(shell_count, cursor, made.shell_angular) ||
      !append_array<std::int64_t>(shell_count + 1, cursor,
                                  made.shell_ao_offsets) ||
      !append_array<std::int64_t>(shell_count + 1, cursor,
                                  made.shell_direct_ao_offsets) ||
      !append_array<std::int64_t>(shell_count + 1, cursor,
                                  made.shell_primitive_offsets) ||
      !append_array<std::int64_t>(batch_size + 1, cursor,
                                  made.system_shell_pair_offsets) ||
      !append_array<std::int64_t>(batch_size + 1, cursor,
                                  made.system_shell_quartet_offsets) ||
      !append_array<std::int32_t>(shell_pair_count, cursor,
                                  made.shell_pair_systems) ||
      !append_array<std::int32_t>(shell_pair_count, cursor,
                                  made.shell_pair_first) ||
      !append_array<std::int32_t>(shell_pair_count, cursor,
                                  made.shell_pair_second) ||
      !append_array<std::int32_t>(aos, cursor, made.ao_shells) ||
      !append_array<std::uint8_t>(aos, cursor, made.ao_term_counts) ||
      !append_array<std::uint8_t>(
          aos * kMaximumAoExpansionTerms * 3, cursor,
          made.ao_term_angular) ||
      !append_array<double>(aos * kMaximumAoExpansionTerms, cursor,
                            made.ao_term_coefficients) ||
      !append_array<std::int32_t>(direct_aos, cursor,
                                  made.direct_ao_shells) ||
      !append_array<std::uint8_t>(direct_aos * 3, cursor,
                                  made.direct_ao_angular) ||
      !append_array<double>(direct_aos, cursor,
                            made.direct_ao_coefficients) ||
      !append_array<double>(transformed_direct ? transform_elements : 0,
                            cursor, made.ao_to_direct_transform) ||
      !append_array<double>(primitives, cursor, made.primitive_exponents) ||
      !append_array<double>(primitives, cursor, made.primitive_coefficients) ||
      !append_array<std::int32_t>(batch_size * spin_count, cursor,
                                  made.occupied) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.warm_mask) ||
      !append_array<double>(spin_matrices, cursor, made.warm_density) ||
      !append_array<double>(matrices, cursor, made.overlap) ||
      !append_array<double>(matrices, cursor, made.hcore) ||
      !append_array<double>(eris, cursor, made.eri) ||
      !append_array<double>(
          persistent_eri ? 0
                         : (transformed_direct ? direct_matrices : matrices),
          cursor,
                            made.schwarz_bounds) ||
      !append_array<double>(transformed_direct ? direct_spin_matrices : 0,
                            cursor, made.direct_density) ||
      !append_array<double>(transformed_direct ? direct_spin_matrices : 0,
                            cursor, made.direct_fock) ||
      !append_array<double>(transformed_direct ? transform_temporaries : 0,
                            cursor, made.direct_transform_temporary) ||
      !append_array<double>(persistent_eri ? 0 : shell_pair_count, cursor,
                            made.shell_pair_bounds) ||
      !append_array<std::uint32_t>(
          persistent_eri ? 0 : detail::kDirectQuartetAngularOrderCount + 1,
          cursor, made.active_shell_quartet_tile_offsets) ||
      !append_array<std::uint32_t>(
          persistent_eri ? 0 : detail::kDirectQuartetAngularOrderCount,
          cursor, made.active_shell_quartet_tile_counts) ||
      !append_array<ActiveShellQuartetTile>(
          persistent_eri ? 0 : shell_quartet_tile_count, cursor,
          made.active_shell_quartet_tiles) ||
      !append_array<std::int32_t>(pair_count, cursor,
                                  made.ao_pair_first) ||
      !append_array<std::int32_t>(pair_count, cursor,
                                  made.ao_pair_second) ||
      !append_array<double>(batch_size, cursor, made.nuclear_repulsion) ||
      !append_array<double>(matrices, cursor, made.orthogonalizer) ||
      !append_array<double>(spin_matrices, cursor, made.temporary) ||
      !append_array<double>(spin_matrices, cursor, made.eigensystem) ||
      !append_array<double>(spin_matrices, cursor, made.coefficients) ||
      !append_array<double>(batch_size * spin_count * nbf, cursor,
                            made.eigenvalues) ||
      !append_array<double>(spin_matrices, cursor, made.density) ||
      !append_array<double>(spin_matrices, cursor, made.next_density) ||
      !append_array<double>(spin_matrices, cursor, made.fock) ||
      !append_array<double>(spin_matrices, cursor, made.residual) ||
      !append_array<double>(spin_matrices, cursor, made.weighted_density) ||
      !append_array<double>(spin_count == 2 ? matrices : 0, cursor,
                            made.total_density) ||
      !append_array<double>(spin_count == 2 ? matrices : 0, cursor,
                            made.total_weighted_density) ||
      !append_array<double>(history_matrices, cursor, made.fock_history) ||
      !append_array<double>(history_matrices, cursor, made.residual_history) ||
      !append_array<double>(diis_linear_elements, cursor,
                            made.diis_linear_system) ||
      !append_array<double>(batch_size * diis_dimension, cursor,
                            made.diis_coefficients) ||
      !append_array<std::uint32_t>(batch_size, cursor, made.diis_count) ||
      !append_array<std::uint32_t>(batch_size, cursor, made.diis_head) ||
      !append_array<double>(batch_size, cursor, made.energy) ||
      !append_array<double>(batch_size, cursor, made.previous_energy) ||
      !append_array<double>(batch_size, cursor, made.energy_change) ||
      !append_array<double>(batch_size, cursor, made.density_rms) ||
      !append_array<double>(atoms * 3, cursor, made.forces) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.active) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.converged) ||
      !append_array<std::uint8_t>(batch_size, cursor, made.failed) ||
      !append_array<std::uint8_t>(batch_size * spin_count, cursor,
                                  made.spin_active) ||
      !append_array<std::uint32_t>(batch_size, cursor, made.iterations) ||
      !append_array<int>(batch_size * spin_count, cursor,
                         made.solver_info)) return false;
  made.bytes = cursor;
  layout = made;
  return true;
}

template <typename T>
T* arena_pointer(void* arena, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(arena) + offset);
}

struct HostBatch {
  std::size_t nbf{};
  std::size_t direct_nbf{};
  std::size_t spin_count{1};
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atom_systems;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> positions;
  std::vector<std::int64_t> system_shell_offsets;
  std::vector<std::int32_t> shell_atoms;
  std::vector<std::uint8_t> shell_angular;
  std::vector<std::int64_t> shell_ao_offsets;
  std::vector<std::int64_t> shell_direct_ao_offsets;
  std::vector<std::int64_t> shell_primitive_offsets;
  std::vector<std::int64_t> system_shell_pair_offsets;
  std::vector<std::int64_t> system_shell_quartet_offsets;
  std::vector<std::int32_t> shell_pair_systems;
  std::vector<std::int32_t> shell_pair_first;
  std::vector<std::int32_t> shell_pair_second;
  std::vector<std::int32_t> ao_shells;
  std::vector<std::uint8_t> ao_term_counts;
  std::vector<std::uint8_t> ao_term_angular;
  std::vector<double> ao_term_coefficients;
  std::vector<std::int32_t> direct_ao_shells;
  std::vector<std::uint8_t> direct_ao_angular;
  std::vector<double> direct_ao_coefficients;
  std::vector<double> ao_to_direct_transform;
  std::vector<double> primitive_exponents;
  std::vector<double> primitive_coefficients;
  std::vector<std::int32_t> occupied;
  std::vector<std::uint8_t> warm_mask;
  std::vector<double> warm_density;
};

bool pack_host_batch(const std::vector<core::System>& systems,
                     const std::vector<const std::vector<double>*>& initial_densities,
                     HostBatch& host,
                     bool unrestricted = false) {
  if (systems.empty() || systems.size() != initial_densities.size()) return false;
  host.nbf = molecule::ao_count(systems.front());
  host.direct_nbf = molecule::cartesian_ao_count(systems.front());
  host.spin_count = unrestricted ? 2 : 1;
  if (host.nbf == 0 || host.direct_nbf == 0 ||
      host.nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      host.direct_nbf >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      systems.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return false;
  const std::size_t matrix_size = host.nbf * host.nbf;
  host.atom_offsets.push_back(0);
  host.system_shell_offsets.push_back(0);
  host.shell_ao_offsets.push_back(0);
  host.shell_direct_ao_offsets.push_back(0);
  host.shell_primitive_offsets.push_back(0);
  host.system_shell_pair_offsets.push_back(0);
  host.system_shell_quartet_offsets.push_back(0);
  host.warm_density.resize(
      systems.size() * host.spin_count * matrix_size, 0.0);
  if (host.direct_nbf != host.nbf && host.nbf > kPersistentEriAoLimit) {
    host.ao_to_direct_transform.resize(
        systems.size() * host.nbf * host.direct_nbf, 0.0);
  }
  for (std::size_t system_index = 0; system_index < systems.size(); ++system_index) {
    const core::System& system = systems[system_index];
    if (molecule::ao_count(system) != host.nbf ||
        molecule::cartesian_ao_count(system) != host.direct_nbf ||
        system.electron_count <= 0) {
      return false;
    }
    const int spin_excess = static_cast<int>(system.multiplicity) - 1;
    if ((!unrestricted &&
         (system.electron_count % 2 != 0 || system.multiplicity != 1)) ||
        (unrestricted &&
         (spin_excess < 0 || spin_excess > system.electron_count ||
          ((system.electron_count + spin_excess) & 1) != 0))) {
      return false;
    }
    const std::int64_t atom_base = static_cast<std::int64_t>(host.atomic_numbers.size());
    for (const core::Atom& atom : system.atoms) {
      host.atom_systems.push_back(static_cast<std::int32_t>(system_index));
      host.atomic_numbers.push_back(atom.atomic_number);
      host.positions.insert(host.positions.end(), atom.position.begin(), atom.position.end());
    }
    host.atom_offsets.push_back(static_cast<std::int64_t>(host.atomic_numbers.size()));
    const std::size_t system_ao_begin = host.ao_shells.size();
    const std::size_t system_direct_ao_begin = host.direct_ao_shells.size();
    const std::size_t system_shell_begin = host.shell_atoms.size();
    for (const core::Shell& shell : system.shells) {
      if (shell.angular_momentum > kMaximumAngularMomentum ||
          shell.atom_index >= system.atoms.size()) return false;
      if (host.shell_atoms.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return false;
      }
      const std::int32_t shell_index =
          static_cast<std::int32_t>(host.shell_atoms.size());
      host.shell_atoms.push_back(
          atom_base + static_cast<std::int64_t>(shell.atom_index));
      host.shell_angular.push_back(
          static_cast<std::uint8_t>(shell.angular_momentum));
      for (const core::Primitive& primitive : shell.primitives) {
        host.primitive_exponents.push_back(primitive.exponent);
        host.primitive_coefficients.push_back(primitive.coefficient);
      }
      host.shell_primitive_offsets.push_back(
          static_cast<std::int64_t>(host.primitive_exponents.size()));
      const std::vector<molecule::CartesianComponent> cartesian_components =
          molecule::cartesian_components(shell.angular_momentum);
      const std::size_t direct_shell_begin =
          host.direct_ao_shells.size() - system_direct_ao_begin;
      for (const molecule::CartesianComponent& component :
           cartesian_components) {
        host.direct_ao_shells.push_back(shell_index);
        host.direct_ao_angular.push_back(
            static_cast<std::uint8_t>(component[0]));
        host.direct_ao_angular.push_back(
            static_cast<std::uint8_t>(component[1]));
        host.direct_ao_angular.push_back(
            static_cast<std::uint8_t>(component[2]));
        host.direct_ao_coefficients.push_back(
            molecule::cartesian_component_normalization(component));
      }
      host.shell_direct_ao_offsets.push_back(
          static_cast<std::int64_t>(host.direct_ao_shells.size()));
      for (const molecule::AoExpansion& expansion : molecule::ao_expansions(
               shell.angular_momentum, system.basis_representation)) {
        if (expansion.empty() ||
            expansion.size() > kMaximumAoExpansionTerms) {
          return false;
        }
        const std::size_t target_ao =
            host.ao_shells.size() - system_ao_begin;
        host.ao_shells.push_back(shell_index);
        host.ao_term_counts.push_back(
            static_cast<std::uint8_t>(expansion.size()));
        for (std::size_t term_index = 0;
             term_index < kMaximumAoExpansionTerms; ++term_index) {
          if (term_index < expansion.size()) {
            const molecule::CartesianExpansionTerm& term =
                expansion[term_index];
            host.ao_term_angular.push_back(
                static_cast<std::uint8_t>(term.component[0]));
            host.ao_term_angular.push_back(
                static_cast<std::uint8_t>(term.component[1]));
            host.ao_term_angular.push_back(
                static_cast<std::uint8_t>(term.component[2]));
            host.ao_term_coefficients.push_back(
                term.coefficient *
                molecule::cartesian_component_normalization(term.component));
            if (!host.ao_to_direct_transform.empty()) {
              const auto component = std::find(
                  cartesian_components.begin(), cartesian_components.end(),
                  term.component);
              if (component == cartesian_components.end()) return false;
              const std::size_t direct_ao = direct_shell_begin +
                  static_cast<std::size_t>(
                      component - cartesian_components.begin());
              const std::size_t transform_offset =
                  system_index * host.nbf * host.direct_nbf;
              host.ao_to_direct_transform[
                  transform_offset + target_ao + direct_ao * host.nbf] =
                  term.coefficient;
            }
          } else {
            host.ao_term_angular.insert(
                host.ao_term_angular.end(), {0, 0, 0});
            host.ao_term_coefficients.push_back(0.0);
          }
        }
      }
      host.shell_ao_offsets.push_back(
          static_cast<std::int64_t>(host.ao_shells.size()));
    }
    if (host.ao_shells.size() - system_ao_begin != host.nbf ||
        host.direct_ao_shells.size() - system_direct_ao_begin !=
            host.direct_nbf) {
      return false;
    }
    host.system_shell_offsets.push_back(
        static_cast<std::int64_t>(host.shell_atoms.size()));
    for (std::size_t first = system_shell_begin;
         first < host.shell_atoms.size(); ++first) {
      for (std::size_t second = system_shell_begin; second <= first; ++second) {
        host.shell_pair_systems.push_back(
            static_cast<std::int32_t>(system_index));
        host.shell_pair_first.push_back(static_cast<std::int32_t>(first));
        host.shell_pair_second.push_back(static_cast<std::int32_t>(second));
      }
    }
    const std::size_t system_shell_pair_end = host.shell_pair_first.size();
    const std::size_t system_shell_pair_count =
        system_shell_pair_end -
        static_cast<std::size_t>(host.system_shell_pair_offsets.back());
    host.system_shell_pair_offsets.push_back(
        static_cast<std::int64_t>(system_shell_pair_end));
    std::size_t system_shell_pair_plus_one = 0;
    std::size_t system_shell_quartet_count = 0;
    if (!checked_add(system_shell_pair_count, 1,
                     system_shell_pair_plus_one) ||
        !checked_multiply(system_shell_pair_count,
                          system_shell_pair_plus_one,
                          system_shell_quartet_count)) {
      return false;
    }
    system_shell_quartet_count /= 2;
    const std::int64_t previous_quartet_offset =
        host.system_shell_quartet_offsets.back();
    if (system_shell_quartet_count > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max() -
            previous_quartet_offset)) {
      return false;
    }
    host.system_shell_quartet_offsets.push_back(
        previous_quartet_offset +
        static_cast<std::int64_t>(system_shell_quartet_count));
    if (unrestricted) {
      const int alpha = (system.electron_count + spin_excess) / 2;
      host.occupied.push_back(alpha);
      host.occupied.push_back(system.electron_count - alpha);
    } else {
      host.occupied.push_back(system.electron_count / 2);
    }
    const std::vector<double>* warm = initial_densities[system_index];
    const std::size_t warm_size = host.spin_count * matrix_size;
    const bool valid_warm = warm != nullptr && warm->size() == warm_size &&
        std::all_of(warm->begin(), warm->end(),
                    [](double value) { return std::isfinite(value); });
    host.warm_mask.push_back(valid_warm ? 1 : 0);
    if (valid_warm) {
      std::copy(warm->begin(), warm->end(),
                host.warm_density.begin() + system_index * warm_size);
    }
  }
  return true;
}

qce_status cuda_status(cudaError_t status) {
  if (status == cudaSuccess) return QCE_STATUS_SUCCESS;
  return status == cudaErrorMemoryAllocation ? QCE_STATUS_OUT_OF_MEMORY
                                              : QCE_STATUS_CUDA_ERROR;
}

qce_status solver_status(cusolverStatus_t status) {
  if (status == CUSOLVER_STATUS_SUCCESS) return QCE_STATUS_SUCCESS;
  return status == CUSOLVER_STATUS_ALLOC_FAILED ? QCE_STATUS_OUT_OF_MEMORY
                                                : QCE_STATUS_CUDA_ERROR;
}

qce_status blas_status(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) return QCE_STATUS_SUCCESS;
  return status == CUBLAS_STATUS_ALLOC_FAILED ? QCE_STATUS_OUT_OF_MEMORY
                                              : QCE_STATUS_CUDA_ERROR;
}

void fill_global_failure(std::vector<RhfBucketItem>& outputs, qce_status status) {
  for (RhfBucketItem& output : outputs) output.status = status;
}

class CudaResources {
 public:
  ~CudaResources() {
    if (device_id_ >= 0) (void)cudaSetDevice(device_id_);
    if (iteration_graph_exec_ != nullptr) {
      (void)cudaGraphExecDestroy(iteration_graph_exec_);
    }
    if (iteration_graph_ != nullptr) (void)cudaGraphDestroy(iteration_graph_);
    if (jacobi_ != nullptr) (void)cusolverDnDestroySyevjInfo(jacobi_);
    if (solver_ != nullptr) (void)cusolverDnDestroy(solver_);
    if (blas_ != nullptr) (void)cublasDestroy(blas_);
    if (stream_ != nullptr) {
      // Both allocations come from CUDA's stream-ordered device pool. Queue
      // their release on the owning bucket stream so destroying one plan does
      // not impose a device-wide synchronization on unrelated workloads.
      if (solver_workspace_ != nullptr) {
        (void)cudaFreeAsync(solver_workspace_, stream_);
      }
      if (arena_ != nullptr) (void)cudaFreeAsync(arena_, stream_);
      (void)cudaStreamSynchronize(stream_);
      (void)cudaStreamDestroy(stream_);
    }
  }

  int device_id_{-1};
  cudaStream_t stream_{};
  cublasHandle_t blas_{};
  cusolverDnHandle_t solver_{};
  syevjInfo_t jacobi_{};
  cudaGraph_t iteration_graph_{};
  cudaGraphExec_t iteration_graph_exec_{};
  void* arena_{};
  double* solver_workspace_{};
};

qce_status copy_to_device(void* destination,
                          const void* source,
                          std::size_t bytes,
                          cudaStream_t stream) {
  if (bytes == 0) return QCE_STATUS_SUCCESS;
  return cuda_status(cudaMemcpyAsync(destination, source, bytes,
                                     cudaMemcpyHostToDevice, stream));
}

qce_status launch_matrix_product(CudaResources& resources,
                                 int batch_size,
                                 int nbf,
                                 const double* left,
                                 bool transpose_left,
                                 const double* right,
                                 const std::uint8_t* active,
                                 double* output,
                                 bool use_cublas) {
  const std::size_t matrix_size =
      static_cast<std::size_t>(nbf) * static_cast<std::size_t>(nbf);
  if (!use_cublas) {
    const std::size_t elements =
        static_cast<std::size_t>(batch_size) * matrix_size;
    const unsigned blocks = static_cast<unsigned>(
        (elements + kCaptureSafeKernelThreads - 1) /
        kCaptureSafeKernelThreads);
    matrix_product_kernel<<<blocks, kCaptureSafeKernelThreads, 0,
                            resources.stream_>>>(
        batch_size, nbf, left, transpose_left, right, active, output);
    return cuda_status(cudaPeekAtLastError());
  }

  const double alpha = 1.0;
  const double beta = 0.0;
  const cublasOperation_t operation =
      transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
  return blas_status(cublasDgemmStridedBatched(
      resources.blas_, operation, CUBLAS_OP_N, nbf, nbf, nbf, &alpha,
      left, nbf, static_cast<long long>(matrix_size), right, nbf,
      static_cast<long long>(matrix_size), &beta, output, nbf,
      static_cast<long long>(matrix_size), batch_size));
}

/**
 * Multiply system-major spin matrices while broadcasting physical operands.
 *
 * A physical matrix repeats for alpha and beta, which is not one constant
 * stride over the interleaved state array. One strided-batched GEMM per spin
 * preserves the existing [system][spin][matrix] storage without pointer lists.
 */
qce_status launch_spin_matrix_product(CudaResources& resources,
                                      int batch_size,
                                      int spin_count,
                                      int nbf,
                                      const double* left,
                                      bool left_is_spin,
                                      bool transpose_left,
                                      const double* right,
                                      bool right_is_spin,
                                      const std::uint8_t* active,
                                      double* output,
                                      bool use_cublas) {
  const std::size_t matrix_size =
      static_cast<std::size_t>(nbf) * static_cast<std::size_t>(nbf);
  if (!use_cublas) {
    const std::size_t elements = static_cast<std::size_t>(batch_size) *
                                 static_cast<std::size_t>(spin_count) *
                                 matrix_size;
    const unsigned blocks = static_cast<unsigned>(
        (elements + kCaptureSafeKernelThreads - 1) /
        kCaptureSafeKernelThreads);
    spin_matrix_product_kernel<<<blocks, kCaptureSafeKernelThreads, 0,
                                 resources.stream_>>>(
        batch_size, spin_count, nbf, left, left_is_spin, transpose_left,
        right, right_is_spin, active, output);
    return cuda_status(cudaPeekAtLastError());
  }

  const double alpha = 1.0;
  const double beta = 0.0;
  const cublasOperation_t operation =
      transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
  const long long physical_stride = static_cast<long long>(matrix_size);
  const long long spin_stride =
      static_cast<long long>(matrix_size * static_cast<std::size_t>(spin_count));
  for (int spin = 0; spin < spin_count; ++spin) {
    const std::size_t spin_offset =
        static_cast<std::size_t>(spin) * matrix_size;
    const double* spin_left = left + (left_is_spin ? spin_offset : 0);
    const double* spin_right = right + (right_is_spin ? spin_offset : 0);
    const cublasStatus_t status = cublasDgemmStridedBatched(
        resources.blas_, operation, CUBLAS_OP_N, nbf, nbf, nbf, &alpha,
        spin_left, nbf, left_is_spin ? spin_stride : physical_stride,
        spin_right, nbf, right_is_spin ? spin_stride : physical_stride,
        &beta, output + spin_offset, nbf, spin_stride, batch_size);
    if (status != CUBLAS_STATUS_SUCCESS) return blas_status(status);
  }
  return QCE_STATUS_SUCCESS;
}

qce_status launch_solver(CudaResources& resources,
                         int nbf,
                         int batch_size,
                         double* matrices,
                         double* eigenvector_workspace,
                         double* eigenvalues,
                         int lwork,
                         int* info,
                         const std::uint8_t* active) {
  if (nbf <= kSmallEigensolverLimit) {
    symmetric_eigen_small_kernel<<<static_cast<unsigned>(batch_size), 1, 0,
                                   resources.stream_>>>(
        batch_size, nbf, matrices, eigenvalues, info, active);
    return cuda_status(cudaPeekAtLastError());
  }
  if (nbf <= kBatchedEigensolverLimit) {
    const cusolverStatus_t status = cusolverDnDsyevjBatched(
        resources.solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        nbf, matrices, nbf, eigenvalues, resources.solver_workspace_, lwork,
        info, resources.jacobi_, batch_size);
    return status == CUSOLVER_STATUS_SUCCESS ? QCE_STATUS_SUCCESS
                                             : solver_status(status);
  }
  if (nbf <= kCyclicGraphEigensolverLimit) {
    symmetric_eigen_graph_cyclic_kernel<<<
        static_cast<unsigned>(batch_size), kCyclicGraphEigensolverThreads,
        static_cast<std::size_t>(nbf) * sizeof(double), resources.stream_>>>(
        batch_size, nbf, matrices, eigenvector_workspace, eigenvalues, info,
        active);
  } else {
    // Retain the unbounded maximum-pivot implementation above the configured
    // cyclic range instead of introducing a new AO-count limit.
    symmetric_eigen_graph_maximum_pivot_kernel<<<
        static_cast<unsigned>(batch_size), kGraphEigensolverThreads, 0,
        resources.stream_>>>(
        batch_size, nbf, matrices, eigenvector_workspace, eigenvalues, info,
        active);
  }
  return cuda_status(cudaPeekAtLastError());
}

}  // namespace

struct CudaRhfBucketPlan {
  CudaResources resources;
  ArenaLayout layout;
  HostBatch topology;
  core::ScfOptions options;
  std::size_t batch_size{};
  std::size_t nbf{};
  std::size_t direct_nbf{};
  std::size_t total_atoms{};
  std::size_t total_shells{};
  std::size_t total_shell_pairs{};
  std::size_t total_shell_quartets{};
  std::size_t total_shell_quartet_tiles{};
  std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>
      shell_quartet_tile_capacities{};
  std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>
      shell_quartet_tile_offsets{};
  std::size_t primitive_count{};
  std::size_t diis_history{};
  int lwork{};
  bool persistent_eri{};
  bool quartet_direct{};
  bool transformed_direct{};
  bool unrestricted{};
  bool cublas_enabled{true};
  bool retry_without_cublas{};
  bool initialized{};
};

namespace {

bool same_topology(const HostBatch& first, const HostBatch& second) {
  return first.nbf == second.nbf &&
         first.direct_nbf == second.direct_nbf &&
         first.spin_count == second.spin_count &&
         first.atom_offsets == second.atom_offsets &&
         first.atom_systems == second.atom_systems &&
         first.atomic_numbers == second.atomic_numbers &&
         first.system_shell_offsets == second.system_shell_offsets &&
         first.shell_atoms == second.shell_atoms &&
         first.shell_angular == second.shell_angular &&
         first.shell_ao_offsets == second.shell_ao_offsets &&
         first.shell_direct_ao_offsets == second.shell_direct_ao_offsets &&
         first.shell_primitive_offsets == second.shell_primitive_offsets &&
         first.system_shell_pair_offsets == second.system_shell_pair_offsets &&
         first.system_shell_quartet_offsets ==
             second.system_shell_quartet_offsets &&
         first.shell_pair_systems == second.shell_pair_systems &&
         first.shell_pair_first == second.shell_pair_first &&
         first.shell_pair_second == second.shell_pair_second &&
         first.ao_shells == second.ao_shells &&
         first.ao_term_counts == second.ao_term_counts &&
         first.ao_term_angular == second.ao_term_angular &&
         first.ao_term_coefficients == second.ao_term_coefficients &&
         first.direct_ao_shells == second.direct_ao_shells &&
         first.direct_ao_angular == second.direct_ao_angular &&
         first.direct_ao_coefficients == second.direct_ao_coefficients &&
         first.ao_to_direct_transform == second.ao_to_direct_transform &&
         first.primitive_exponents == second.primitive_exponents &&
         first.primitive_coefficients == second.primitive_coefficients &&
         first.occupied == second.occupied;
}

bool same_options(const core::ScfOptions& first,
                  const core::ScfOptions& second) {
  return first.max_iterations == second.max_iterations &&
         first.diis_history == second.diis_history &&
         first.energy_tolerance == second.energy_tolerance &&
         first.density_tolerance == second.density_tolerance &&
         first.screening_tolerance == second.screening_tolerance;
}

std::vector<RhfBucketItem> execute_hf_cuda_bucket(
    CudaRhfBucketPlan& plan,
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool unrestricted) {
  std::vector<RhfBucketItem> outputs(systems.size());
  HostBatch host;
  if (!pack_host_batch(systems, initial_densities, host, unrestricted)) {
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }

  const std::size_t batch_size = systems.size();
  const std::size_t nbf = host.nbf;
  const std::size_t direct_nbf = host.direct_nbf;
  const std::size_t spin_count = host.spin_count;
  std::size_t spin_batch_size = 0;
  std::size_t matrix_size = 0;
  std::size_t eri_size = 0;
  std::size_t matrix_elements = 0;
  std::size_t spin_matrix_elements = 0;
  std::size_t eri_elements = 0;
  std::size_t nbf_plus_one = 0;
  std::size_t pair_product = 0;
  std::size_t direct_matrix_size = 0;
  std::size_t direct_matrix_elements = 0;
  std::size_t direct_spin_matrix_elements = 0;
  std::size_t direct_nbf_plus_one = 0;
  std::size_t direct_pair_product = 0;
  std::size_t public_ao_elements = 0;
  std::size_t rectangular_matrix_elements = 0;
  std::size_t spin_rectangular_matrix_elements = 0;
  if (!checked_multiply(nbf, nbf, matrix_size) ||
      !checked_multiply(matrix_size, matrix_size, eri_size) ||
      !checked_multiply(batch_size, matrix_size, matrix_elements) ||
      !checked_multiply(batch_size, spin_count, spin_batch_size) ||
      !checked_multiply(matrix_elements, spin_count, spin_matrix_elements) ||
      !checked_multiply(batch_size, eri_size, eri_elements) ||
      !checked_add(nbf, 1, nbf_plus_one) ||
      !checked_multiply(nbf, nbf_plus_one, pair_product) ||
      !checked_multiply(direct_nbf, direct_nbf, direct_matrix_size) ||
      !checked_multiply(batch_size, direct_matrix_size,
                        direct_matrix_elements) ||
      !checked_multiply(direct_matrix_elements, spin_count,
                        direct_spin_matrix_elements) ||
      !checked_add(direct_nbf, 1, direct_nbf_plus_one) ||
      !checked_multiply(direct_nbf, direct_nbf_plus_one,
                        direct_pair_product) ||
      !checked_multiply(batch_size, nbf, public_ao_elements) ||
      !checked_multiply(public_ao_elements, direct_nbf,
                        rectangular_matrix_elements) ||
      !checked_multiply(rectangular_matrix_elements, spin_count,
                        spin_rectangular_matrix_elements)) {
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t pair_count = pair_product / 2;
  const std::size_t direct_pair_count = direct_pair_product / 2;
  std::size_t pair_elements = 0;
  std::size_t direct_pair_elements = 0;
  if (!checked_multiply(batch_size, pair_count, pair_elements) ||
      !checked_multiply(batch_size, direct_pair_count,
                        direct_pair_elements)) {
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (matrix_elements > std::numeric_limits<unsigned>::max() ||
      spin_matrix_elements > std::numeric_limits<unsigned>::max() ||
      direct_matrix_elements > std::numeric_limits<unsigned>::max() ||
      direct_spin_matrix_elements > std::numeric_limits<unsigned>::max() ||
      rectangular_matrix_elements > std::numeric_limits<unsigned>::max() ||
      spin_rectangular_matrix_elements >
          std::numeric_limits<unsigned>::max() ||
      direct_pair_elements > std::numeric_limits<unsigned>::max() ||
      spin_batch_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_atoms = host.atomic_numbers.size();
  const std::size_t total_shells = host.shell_atoms.size();
  const std::size_t total_shell_pairs = host.shell_pair_first.size();
  if (host.system_shell_quartet_offsets.size() != batch_size + 1 ||
      host.system_shell_quartet_offsets.back() < 0) {
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_quartets =
      static_cast<std::size_t>(host.system_shell_quartet_offsets.back());
  const bool requested_persistent_eri = nbf <= kPersistentEriAoLimit;
  const bool requested_quartet_direct =
      !requested_persistent_eri &&
      std::all_of(host.shell_angular.begin(), host.shell_angular.end(),
                  [](std::uint8_t angular) { return angular <= 3; });
  const bool requested_transformed_direct =
      requested_quartet_direct && direct_nbf != nbf;
  detail::DirectQuartetTaskLayout direct_task_layout{};
  std::size_t total_shell_quartet_tiles = 0;
  if (requested_quartet_direct) {
    if (!detail::make_direct_quartet_task_layout(
            host.shell_direct_ao_offsets, host.shell_angular,
            host.system_shell_pair_offsets, host.shell_pair_first,
            host.shell_pair_second,
            direct_task_layout) ||
        direct_task_layout.shell_quartet_count != total_shell_quartets) {
      fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
    total_shell_quartet_tiles = direct_task_layout.exact_tile_count;
  }
  // Direct consumers expand each compact logical tile into one-warp blocks;
  // validate the resulting fixed Graph grid before narrowing it to unsigned.
  if (total_shell_pairs > std::numeric_limits<unsigned>::max() ||
      total_shell_quartets > std::numeric_limits<unsigned>::max() ||
      total_shell_quartet_tiles >
          std::numeric_limits<unsigned>::max() /
              detail::kDirectQuartetSubtilesPerTile) {
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  std::size_t force_coordinate_count = 0;
  std::size_t one_electron_force_elements = 0;
  std::size_t force_matrix_elements = 0;
  std::size_t persistent_force_elements = 0;
  std::size_t direct_force_elements = 0;
  if (!checked_multiply(total_atoms, 3, force_coordinate_count) ||
      !checked_multiply(force_coordinate_count, pair_count,
                        one_electron_force_elements) ||
      !checked_multiply(force_coordinate_count, matrix_size,
                        force_matrix_elements) ||
      !checked_multiply(force_coordinate_count, eri_size,
                        persistent_force_elements) ||
      !checked_multiply(force_matrix_elements, pair_count,
                        direct_force_elements)) {
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t diis_history = std::max<std::size_t>(1, options.diis_history);
  if (diis_history > 64) {
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const bool first_setup = !plan.initialized;
  if (!first_setup &&
      (plan.resources.device_id_ != device_id ||
       !same_topology(plan.topology, host) ||
       !same_options(plan.options, options) ||
       plan.unrestricted != unrestricted)) {
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (first_setup) {
    if (!make_layout(batch_size, nbf, direct_nbf, total_atoms, total_shells,
                     total_shell_pairs, total_shell_quartet_tiles,
                     host.primitive_exponents.size(), diis_history,
                     host.spin_count,
                     requested_persistent_eri,
                     requested_transformed_direct,
                     plan.layout)) {
      fill_global_failure(outputs, QCE_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    plan.batch_size = batch_size;
    plan.nbf = nbf;
    plan.direct_nbf = direct_nbf;
    plan.total_atoms = total_atoms;
    plan.total_shells = total_shells;
    plan.total_shell_pairs = total_shell_pairs;
    plan.total_shell_quartets = total_shell_quartets;
    plan.total_shell_quartet_tiles = total_shell_quartet_tiles;
    plan.shell_quartet_tile_capacities =
        direct_task_layout.angular_order_tile_counts;
    for (std::size_t order = 0;
         order < direct_task_layout.angular_order_tile_offsets.size();
         ++order) {
      plan.shell_quartet_tile_offsets[order] = static_cast<std::uint32_t>(
          direct_task_layout.angular_order_tile_offsets[order]);
    }
    plan.primitive_count = host.primitive_exponents.size();
    plan.diis_history = diis_history;
    plan.persistent_eri = requested_persistent_eri;
    plan.quartet_direct = requested_quartet_direct;
    plan.transformed_direct = requested_transformed_direct;
    plan.unrestricted = unrestricted;
    plan.options = options;
    plan.topology = host;
    // Positions and warm guesses are dynamic execution inputs, not part of
    // the immutable fixed-topology cache identity.
    plan.topology.positions.clear();
    plan.topology.warm_mask.clear();
    plan.topology.warm_density.clear();
    plan.resources.device_id_ = device_id;
  }
  ArenaLayout& layout = plan.layout;
  CudaResources& resources = plan.resources;
  const bool persistent_eri = plan.persistent_eri;
  const bool quartet_direct = plan.quartet_direct;
  const bool transformed_direct = plan.transformed_direct;
  const bool use_cusolver =
      nbf > static_cast<std::size_t>(kSmallEigensolverLimit) &&
      nbf <= static_cast<std::size_t>(kBatchedEigensolverLimit);
  const bool use_cublas = plan.cublas_enabled &&
      nbf >= kCublasMatrixProductAoThreshold;
  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  cublasStatus_t blas_error = CUBLAS_STATUS_SUCCESS;
  cusolverStatus_t solver_error = CUSOLVER_STATUS_SUCCESS;
  if (first_setup) {
    if ((cuda_error = cudaStreamCreateWithFlags(&resources.stream_,
                                                 cudaStreamNonBlocking)) != cudaSuccess ||
        (cuda_error = cudaMallocAsync(
             &resources.arena_, layout.bytes, resources.stream_)) != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (use_cublas) {
      blas_error = cublasCreate(&resources.blas_);
      if (blas_error == CUBLAS_STATUS_SUCCESS) {
        blas_error = cublasSetStream(resources.blas_, resources.stream_);
      }
      if (blas_error == CUBLAS_STATUS_SUCCESS) {
        blas_error = cublasSetPointerMode(
            resources.blas_, CUBLAS_POINTER_MODE_HOST);
      }
      if (blas_error != CUBLAS_STATUS_SUCCESS) {
        plan.retry_without_cublas = true;
        fill_global_failure(outputs, blas_status(blas_error));
        return outputs;
      }
    }
    if (use_cusolver) {
      solver_error = cusolverDnCreate(&resources.solver_);
      if (solver_error != CUSOLVER_STATUS_SUCCESS ||
          (solver_error = cusolverDnSetStream(resources.solver_, resources.stream_)) !=
              CUSOLVER_STATUS_SUCCESS ||
          (solver_error = cusolverDnCreateSyevjInfo(&resources.jacobi_)) !=
              CUSOLVER_STATUS_SUCCESS ||
          (solver_error = cusolverDnXsyevjSetTolerance(resources.jacobi_, 1.0e-13)) !=
              CUSOLVER_STATUS_SUCCESS ||
          (solver_error = cusolverDnXsyevjSetMaxSweeps(resources.jacobi_, 100)) !=
              CUSOLVER_STATUS_SUCCESS ||
          (solver_error = cusolverDnXsyevjSetSortEig(resources.jacobi_, 1)) !=
              CUSOLVER_STATUS_SUCCESS) {
        fill_global_failure(outputs, solver_status(solver_error));
        return outputs;
      }
    }
  }

  auto atom_offsets = arena_pointer<std::int64_t>(resources.arena_, layout.atom_offsets);
  auto atom_systems = arena_pointer<std::int32_t>(resources.arena_, layout.atom_systems);
  auto atomic_numbers = arena_pointer<std::int32_t>(resources.arena_, layout.atomic_numbers);
  auto positions = arena_pointer<double>(resources.arena_, layout.positions);
  auto system_shell_offsets = arena_pointer<std::int64_t>(
      resources.arena_, layout.system_shell_offsets);
  auto shell_atoms = arena_pointer<std::int32_t>(resources.arena_, layout.shell_atoms);
  auto shell_angular =
      arena_pointer<std::uint8_t>(resources.arena_, layout.shell_angular);
  auto shell_ao_offsets = arena_pointer<std::int64_t>(
      resources.arena_, layout.shell_ao_offsets);
  auto shell_direct_ao_offsets = arena_pointer<std::int64_t>(
      resources.arena_, layout.shell_direct_ao_offsets);
  auto shell_primitive_offsets = arena_pointer<std::int64_t>(
      resources.arena_, layout.shell_primitive_offsets);
  auto system_shell_pair_offsets = arena_pointer<std::int64_t>(
      resources.arena_, layout.system_shell_pair_offsets);
  auto system_shell_quartet_offsets = arena_pointer<std::int64_t>(
      resources.arena_, layout.system_shell_quartet_offsets);
  auto shell_pair_systems = arena_pointer<std::int32_t>(
      resources.arena_, layout.shell_pair_systems);
  auto shell_pair_first = arena_pointer<std::int32_t>(
      resources.arena_, layout.shell_pair_first);
  auto shell_pair_second = arena_pointer<std::int32_t>(
      resources.arena_, layout.shell_pair_second);
  auto ao_shells =
      arena_pointer<std::int32_t>(resources.arena_, layout.ao_shells);
  auto ao_term_counts =
      arena_pointer<std::uint8_t>(resources.arena_, layout.ao_term_counts);
  auto ao_term_angular =
      arena_pointer<std::uint8_t>(resources.arena_, layout.ao_term_angular);
  auto ao_term_coefficients =
      arena_pointer<double>(resources.arena_, layout.ao_term_coefficients);
  auto direct_ao_shells = arena_pointer<std::int32_t>(
      resources.arena_, layout.direct_ao_shells);
  auto direct_ao_angular = arena_pointer<std::uint8_t>(
      resources.arena_, layout.direct_ao_angular);
  auto direct_ao_coefficients = arena_pointer<double>(
      resources.arena_, layout.direct_ao_coefficients);
  auto ao_to_direct_transform = arena_pointer<double>(
      resources.arena_, layout.ao_to_direct_transform);
  auto primitive_exponents =
      arena_pointer<double>(resources.arena_, layout.primitive_exponents);
  auto primitive_coefficients =
      arena_pointer<double>(resources.arena_, layout.primitive_coefficients);
  auto occupied = arena_pointer<std::int32_t>(resources.arena_, layout.occupied);
  auto warm_mask = arena_pointer<std::uint8_t>(resources.arena_, layout.warm_mask);
  auto warm_density = arena_pointer<double>(resources.arena_, layout.warm_density);
  auto overlap = arena_pointer<double>(resources.arena_, layout.overlap);
  auto hcore = arena_pointer<double>(resources.arena_, layout.hcore);
  auto eri = arena_pointer<double>(resources.arena_, layout.eri);
  auto schwarz_bounds =
      arena_pointer<double>(resources.arena_, layout.schwarz_bounds);
  auto direct_density =
      arena_pointer<double>(resources.arena_, layout.direct_density);
  auto direct_fock =
      arena_pointer<double>(resources.arena_, layout.direct_fock);
  auto direct_transform_temporary = arena_pointer<double>(
      resources.arena_, layout.direct_transform_temporary);
  auto shell_pair_bounds =
      arena_pointer<double>(resources.arena_, layout.shell_pair_bounds);
  auto active_shell_quartet_tile_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.active_shell_quartet_tile_offsets);
  auto active_shell_quartet_tile_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.active_shell_quartet_tile_counts);
  auto active_shell_quartet_tiles = arena_pointer<ActiveShellQuartetTile>(
      resources.arena_, layout.active_shell_quartet_tiles);
  auto ao_pair_first =
      arena_pointer<std::int32_t>(resources.arena_, layout.ao_pair_first);
  auto ao_pair_second =
      arena_pointer<std::int32_t>(resources.arena_, layout.ao_pair_second);
  auto nuclear_repulsion =
      arena_pointer<double>(resources.arena_, layout.nuclear_repulsion);
  auto orthogonalizer =
      arena_pointer<double>(resources.arena_, layout.orthogonalizer);
  auto temporary = arena_pointer<double>(resources.arena_, layout.temporary);
  auto eigensystem = arena_pointer<double>(resources.arena_, layout.eigensystem);
  auto coefficients = arena_pointer<double>(resources.arena_, layout.coefficients);
  auto eigenvalues = arena_pointer<double>(resources.arena_, layout.eigenvalues);
  auto density = arena_pointer<double>(resources.arena_, layout.density);
  auto next_density = arena_pointer<double>(resources.arena_, layout.next_density);
  auto fock = arena_pointer<double>(resources.arena_, layout.fock);
  auto residual = arena_pointer<double>(resources.arena_, layout.residual);
  auto weighted_density =
      arena_pointer<double>(resources.arena_, layout.weighted_density);
  auto total_density =
      arena_pointer<double>(resources.arena_, layout.total_density);
  auto total_weighted_density =
      arena_pointer<double>(resources.arena_, layout.total_weighted_density);
  auto fock_history =
      arena_pointer<double>(resources.arena_, layout.fock_history);
  auto residual_history =
      arena_pointer<double>(resources.arena_, layout.residual_history);
  auto diis_linear_system =
      arena_pointer<double>(resources.arena_, layout.diis_linear_system);
  auto diis_coefficients =
      arena_pointer<double>(resources.arena_, layout.diis_coefficients);
  auto diis_count =
      arena_pointer<std::uint32_t>(resources.arena_, layout.diis_count);
  auto diis_head =
      arena_pointer<std::uint32_t>(resources.arena_, layout.diis_head);
  auto energy = arena_pointer<double>(resources.arena_, layout.energy);
  auto previous_energy =
      arena_pointer<double>(resources.arena_, layout.previous_energy);
  auto energy_change = arena_pointer<double>(resources.arena_, layout.energy_change);
  auto density_rms = arena_pointer<double>(resources.arena_, layout.density_rms);
  auto forces = arena_pointer<double>(resources.arena_, layout.forces);
  auto active = arena_pointer<std::uint8_t>(resources.arena_, layout.active);
  auto converged = arena_pointer<std::uint8_t>(resources.arena_, layout.converged);
  auto failed = arena_pointer<std::uint8_t>(resources.arena_, layout.failed);
  auto spin_active =
      arena_pointer<std::uint8_t>(resources.arena_, layout.spin_active);
  auto iterations = arena_pointer<std::uint32_t>(resources.arena_, layout.iterations);
  auto solver_info = arena_pointer<int>(resources.arena_, layout.solver_info);

  std::vector<std::int32_t> host_pair_first;
  std::vector<std::int32_t> host_pair_second;
  if (first_setup) {
    host_pair_first.reserve(pair_count);
    host_pair_second.reserve(pair_count);
    // Canonical lower-triangle order is stable for the lifetime of a topology
    // plan. Upload it once so one-electron and direct-J/K consumers reuse the
    // same device metadata without rebuilding or decoding pair indices.
    for (std::size_t first = 0; first < nbf; ++first) {
      for (std::size_t second = 0; second <= first; ++second) {
        host_pair_first.push_back(static_cast<std::int32_t>(first));
        host_pair_second.push_back(static_cast<std::int32_t>(second));
      }
    }
  }

  const std::size_t shell_quartet_offset_bytes = quartet_direct
      ? plan.shell_quartet_tile_offsets.size() * sizeof(std::uint32_t)
      : 0;
  const std::pair<const void*, std::pair<void*, std::size_t>> static_uploads[] = {
      {host.atom_offsets.data(), {atom_offsets, host.atom_offsets.size() * sizeof(std::int64_t)}},
      {host.atom_systems.data(), {atom_systems, host.atom_systems.size() * sizeof(std::int32_t)}},
      {host.atomic_numbers.data(), {atomic_numbers, host.atomic_numbers.size() * sizeof(std::int32_t)}},
      {host.system_shell_offsets.data(),
       {system_shell_offsets,
        host.system_shell_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_atoms.data(), {shell_atoms, host.shell_atoms.size() * sizeof(std::int32_t)}},
      {host.shell_angular.data(),
       {shell_angular, host.shell_angular.size() * sizeof(std::uint8_t)}},
      {host.shell_ao_offsets.data(),
       {shell_ao_offsets, host.shell_ao_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_direct_ao_offsets.data(),
       {shell_direct_ao_offsets,
        host.shell_direct_ao_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_primitive_offsets.data(),
       {shell_primitive_offsets,
        host.shell_primitive_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_pair_offsets.data(),
       {system_shell_pair_offsets,
        host.system_shell_pair_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_quartet_offsets.data(),
       {system_shell_quartet_offsets,
        host.system_shell_quartet_offsets.size() * sizeof(std::int64_t)}},
      {host.shell_pair_systems.data(),
       {shell_pair_systems,
        host.shell_pair_systems.size() * sizeof(std::int32_t)}},
      {host.shell_pair_first.data(),
       {shell_pair_first, host.shell_pair_first.size() * sizeof(std::int32_t)}},
      {host.shell_pair_second.data(),
       {shell_pair_second,
        host.shell_pair_second.size() * sizeof(std::int32_t)}},
      {host.ao_shells.data(),
       {ao_shells, host.ao_shells.size() * sizeof(std::int32_t)}},
      {host.ao_term_counts.data(),
       {ao_term_counts, host.ao_term_counts.size() * sizeof(std::uint8_t)}},
      {host.ao_term_angular.data(),
       {ao_term_angular,
        host.ao_term_angular.size() * sizeof(std::uint8_t)}},
      {host.ao_term_coefficients.data(),
       {ao_term_coefficients,
        host.ao_term_coefficients.size() * sizeof(double)}},
      {host.direct_ao_shells.data(),
       {direct_ao_shells,
        host.direct_ao_shells.size() * sizeof(std::int32_t)}},
      {host.direct_ao_angular.data(),
       {direct_ao_angular,
        host.direct_ao_angular.size() * sizeof(std::uint8_t)}},
      {host.direct_ao_coefficients.data(),
       {direct_ao_coefficients,
        host.direct_ao_coefficients.size() * sizeof(double)}},
      {host.ao_to_direct_transform.data(),
       {ao_to_direct_transform,
        host.ao_to_direct_transform.size() * sizeof(double)}},
      {host.primitive_exponents.data(), {primitive_exponents, host.primitive_exponents.size() * sizeof(double)}},
      {host.primitive_coefficients.data(), {primitive_coefficients, host.primitive_coefficients.size() * sizeof(double)}},
      {host.occupied.data(), {occupied, host.occupied.size() * sizeof(std::int32_t)}},
      {plan.shell_quartet_tile_offsets.data(),
       {active_shell_quartet_tile_offsets, shell_quartet_offset_bytes}},
      {host_pair_first.data(),
       {ao_pair_first, host_pair_first.size() * sizeof(std::int32_t)}},
      {host_pair_second.data(),
       {ao_pair_second, host_pair_second.size() * sizeof(std::int32_t)}},
  };
  const std::pair<const void*, std::pair<void*, std::size_t>> dynamic_uploads[] = {
      {host.positions.data(), {positions, host.positions.size() * sizeof(double)}},
      {host.warm_mask.data(), {warm_mask, host.warm_mask.size() * sizeof(std::uint8_t)}},
      {host.warm_density.data(), {warm_density, host.warm_density.size() * sizeof(double)}},
  };
  if (first_setup) {
    for (const auto& upload : static_uploads) {
      const qce_status status = copy_to_device(
          upload.second.first, upload.first, upload.second.second,
          resources.stream_);
      if (status != QCE_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
    }
  }
  for (const auto& upload : dynamic_uploads) {
    const qce_status status = copy_to_device(
        upload.second.first, upload.first, upload.second.second,
        resources.stream_);
    if (status != QCE_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  }

  DeviceBatch device_batch{
      static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
      static_cast<std::int32_t>(direct_nbf),
      static_cast<std::int64_t>(total_atoms),
      static_cast<std::int64_t>(total_shells),
      static_cast<std::int64_t>(total_shell_pairs),
      static_cast<std::int64_t>(total_shell_quartets), atom_offsets,
      atom_systems, atomic_numbers, positions, system_shell_offsets,
      shell_atoms, shell_angular, shell_ao_offsets, shell_direct_ao_offsets,
      shell_primitive_offsets, system_shell_pair_offsets,
      system_shell_quartet_offsets,
      shell_pair_systems, shell_pair_first, shell_pair_second, ao_shells,
      ao_term_counts, ao_term_angular, ao_term_coefficients,
      direct_ao_shells, direct_ao_angular, direct_ao_coefficients,
      ao_to_direct_transform,
      primitive_exponents,
      primitive_coefficients, occupied};

  if (first_setup && use_cusolver) {
    solver_error = cusolverDnDsyevjBatched_bufferSize(
        resources.solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(nbf), eigensystem, static_cast<int>(nbf), eigenvalues,
        &plan.lwork, resources.jacobi_, static_cast<int>(spin_batch_size));
    if (solver_error != CUSOLVER_STATUS_SUCCESS || plan.lwork <= 0) {
      fill_global_failure(outputs, solver_status(solver_error));
      return outputs;
    }
    if ((cuda_error = cudaMallocAsync(
             reinterpret_cast<void**>(&resources.solver_workspace_),
             static_cast<std::size_t>(plan.lwork) * sizeof(double),
             resources.stream_)) != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  } else if (first_setup) {
    plan.lwork = 0;
  }
  const int lwork = plan.lwork;

  constexpr unsigned threads =
      kCaptureSafeKernelThreads;
  const auto blocks_for = [](std::size_t elements) {
    return static_cast<unsigned>((elements + threads - 1) / threads);
  };
  const auto multiply_matrices = [&](const double* left,
                                     bool transpose_left,
                                     const double* right,
                                     double* output) {
    const qce_status product_status = launch_matrix_product(
        resources, static_cast<int>(batch_size), static_cast<int>(nbf),
        left, transpose_left, right, active, output, use_cublas);
    if (use_cublas && product_status != QCE_STATUS_SUCCESS) {
      plan.retry_without_cublas = true;
    }
    return product_status;
  };
  const auto multiply_spin_matrices = [&](const double* left,
                                          bool left_is_spin,
                                          bool transpose_left,
                                          const double* right,
                                          bool right_is_spin,
                                          double* output) {
    const qce_status product_status = launch_spin_matrix_product(
        resources, static_cast<int>(batch_size), 2, static_cast<int>(nbf),
        left, left_is_spin, transpose_left, right, right_is_spin, active,
        output, use_cublas);
    if (use_cublas && product_status != QCE_STATUS_SUCCESS) {
      plan.retry_without_cublas = true;
    }
    return product_status;
  };
  const auto launch_fock_builder = [&](const double* density_input) {
    const double* quartet_density = density_input;
    double* quartet_fock = fock;
    if (quartet_direct && transformed_direct) {
      transform_density_to_direct_right_kernel<<<
          blocks_for(spin_rectangular_matrix_elements), threads, 0,
          resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf),
          static_cast<std::int32_t>(direct_nbf), ao_to_direct_transform,
          density_input, active, direct_transform_temporary);
      transform_density_to_direct_left_kernel<<<
          blocks_for(direct_spin_matrix_elements), threads, 0,
          resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf),
          static_cast<std::int32_t>(direct_nbf), ao_to_direct_transform,
          direct_transform_temporary, active, direct_density);
      clear_active_matrices_kernel<<<
          blocks_for(direct_spin_matrix_elements), threads, 0,
          resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(direct_nbf), active, direct_fock);
      quartet_density = direct_density;
      quartet_fock = direct_fock;
    }
    if (unrestricted && persistent_eri) {
      build_uhf_fock_kernel<<<blocks_for(spin_matrix_elements), threads, 0,
                              resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
          hcore, eri, density_input, active, fock);
    } else if (unrestricted && quartet_direct) {
      if (!transformed_direct) {
        initialize_direct_fock_kernel<<<blocks_for(spin_matrix_elements),
                                        threads, 0, resources.stream_>>>(
            static_cast<std::int32_t>(batch_size), 2,
            static_cast<std::int32_t>(nbf), hcore, active, fock);
      }
      launch_angular_fock_quartets<true>(
          resources.stream_, plan.shell_quartet_tile_capacities,
          plan.shell_quartet_tile_offsets, device_batch,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          options.screening_tolerance, schwarz_bounds, quartet_density, active,
          quartet_fock);
    } else if (unrestricted) {
      build_uhf_fock_direct_packed_kernel<<<
          static_cast<unsigned>(spin_matrix_elements), threads,
          threads * sizeof(double), resources.stream_>>>(
          device_batch, options.screening_tolerance, hcore, ao_pair_first,
          ao_pair_second, pair_count, schwarz_bounds, density_input, active,
          fock);
    } else if (persistent_eri) {
      build_fock_kernel<<<blocks_for(matrix_elements), threads, 0,
                          resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
          hcore, eri, density_input, active, fock);
    } else if (quartet_direct) {
      if (!transformed_direct) {
        initialize_direct_fock_kernel<<<blocks_for(matrix_elements), threads,
                                        0, resources.stream_>>>(
            static_cast<std::int32_t>(batch_size), 1,
            static_cast<std::int32_t>(nbf), hcore, active, fock);
      }
      launch_angular_fock_quartets<false>(
          resources.stream_, plan.shell_quartet_tile_capacities,
          plan.shell_quartet_tile_offsets, device_batch,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          options.screening_tolerance, schwarz_bounds, quartet_density, active,
          quartet_fock);
    } else {
      build_fock_direct_packed_kernel<<<
          static_cast<unsigned>(matrix_elements), threads,
          threads * sizeof(double), resources.stream_>>>(
          device_batch, options.screening_tolerance, hcore, ao_pair_first,
          ao_pair_second, pair_count, schwarz_bounds, density_input, active,
          fock);
    }
    if (quartet_direct && transformed_direct) {
      transform_direct_fock_left_kernel<<<
          blocks_for(spin_rectangular_matrix_elements), threads, 0,
          resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf),
          static_cast<std::int32_t>(direct_nbf), ao_to_direct_transform,
          direct_fock, active, direct_transform_temporary);
      transform_direct_fock_right_kernel<<<
          blocks_for(spin_matrix_elements), threads, 0, resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf),
          static_cast<std::int32_t>(direct_nbf), ao_to_direct_transform,
          direct_transform_temporary, hcore, active, fock);
    }
  };
  initialize_state_kernel<<<blocks_for(batch_size), threads, 0, resources.stream_>>>(
      static_cast<std::int32_t>(batch_size), active, converged, failed,
      iterations, previous_energy, energy_change, density_rms,
      diis_count, diis_head);
  build_one_electron_integrals_kernel<<<blocks_for(pair_elements), threads, 0,
                                        resources.stream_>>>(
      device_batch, ao_pair_first, ao_pair_second, pair_count, overlap, hcore);
  if (persistent_eri) {
    build_eri_kernel<<<blocks_for(eri_elements), threads, 0, resources.stream_>>>(
        device_batch, eri);
  } else {
    build_schwarz_bounds_packed_kernel<<<blocks_for(direct_pair_elements),
                                         threads,
                                         0, resources.stream_>>>(
        device_batch, direct_pair_count, schwarz_bounds);
    if (quartet_direct) {
      reduce_shell_pair_bounds_kernel<<<
          static_cast<unsigned>(total_shell_pairs), threads,
          threads * sizeof(double), resources.stream_>>>(
          device_batch, schwarz_bounds, shell_pair_bounds);
      cuda_error = cudaMemsetAsync(
          active_shell_quartet_tile_counts, 0,
          detail::kDirectQuartetAngularOrderCount * sizeof(std::uint32_t),
          resources.stream_);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
      compact_active_shell_quartet_tiles_kernel<<<
          blocks_for(total_shell_quartets), threads, 0, resources.stream_>>>(
          device_batch, options.screening_tolerance, shell_pair_bounds,
          active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles);
    }
  }
  build_nuclear_repulsion_kernel<<<blocks_for(batch_size), threads, 0,
                                    resources.stream_>>>(
      device_batch, nuclear_repulsion);

  copy_matrix_kernel<<<blocks_for(matrix_elements), threads, 0, resources.stream_>>>(
      matrix_elements, overlap, eigensystem);
  qce_status status = launch_solver(resources, static_cast<int>(nbf),
                                    static_cast<int>(batch_size), eigensystem,
                                    temporary, eigenvalues, lwork, solver_info,
                                    active);
  if (status != QCE_STATUS_SUCCESS) {
    fill_global_failure(outputs, status);
    return outputs;
  }
  inspect_solver_kernel<<<blocks_for(batch_size), threads, 0, resources.stream_>>>(
      static_cast<std::int32_t>(batch_size), solver_info, active, failed, converged);
  build_orthogonalizer_kernel<<<blocks_for(matrix_elements), threads, 0,
                                resources.stream_>>>(
      static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
      eigensystem, eigenvalues, active, orthogonalizer, failed);

  status = multiply_matrices(hcore, false, orthogonalizer, temporary);
  if (status == QCE_STATUS_SUCCESS) {
    status = multiply_matrices(
        orthogonalizer, true, temporary, eigensystem);
  }
  if (status != QCE_STATUS_SUCCESS) {
    fill_global_failure(outputs, status);
    return outputs;
  }
  status = launch_solver(resources, static_cast<int>(nbf),
                         static_cast<int>(batch_size), eigensystem,
                         temporary, eigenvalues, lwork, solver_info, active);
  if (status != QCE_STATUS_SUCCESS) {
    fill_global_failure(outputs, status);
    return outputs;
  }
  inspect_solver_kernel<<<blocks_for(batch_size), threads, 0, resources.stream_>>>(
      static_cast<std::int32_t>(batch_size), solver_info, active, failed, converged);
  if (unrestricted) {
    status = multiply_matrices(
        orthogonalizer, false, eigensystem, temporary);
    if (status != QCE_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    broadcast_spin_matrix_kernel<<<blocks_for(spin_matrix_elements), threads, 0,
                                   resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), 2,
        static_cast<std::int32_t>(nbf), temporary, active, coefficients);
    mix_open_shell_guess_kernel<<<blocks_for(batch_size * nbf), threads, 0,
                                  resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        occupied, active, coefficients);
    build_spin_density_kernel<<<blocks_for(spin_matrix_elements), threads, 0,
                                resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), 2,
        static_cast<std::int32_t>(nbf), occupied, coefficients, active, density);
    apply_uhf_warm_density_kernel<<<static_cast<unsigned>(batch_size), 1, 0,
                                    resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        occupied, warm_mask, warm_density, overlap, density);
  } else {
    status = multiply_matrices(
        orthogonalizer, false, eigensystem, coefficients);
    if (status != QCE_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    build_density_kernel<<<blocks_for(matrix_elements), threads, 0,
                           resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        occupied, coefficients, active, density);
    apply_warm_density_kernel<<<static_cast<unsigned>(batch_size), 1, 0,
                                resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        occupied, warm_mask, warm_density, overlap, density);
  }

  if (first_setup) {
    // Graph construction is allocation-permitted setup work. Synchronize once
    // so capture cannot race the initial guess; fixed-topology replays reuse
    // this executable and do not repeat the fence or provider setup.
    cuda_error = cudaStreamSynchronize(resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamBeginCapture(resources.stream_,
                                          cudaStreamCaptureModeThreadLocal);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    launch_fock_builder(density);
    if (unrestricted) {
      compute_uhf_energy_kernel<<<blocks_for(batch_size), threads, 0,
                                  resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
          density, hcore, fock, nuclear_repulsion, active, energy);
      build_spin_commutator_residual_kernel<<<
          blocks_for(spin_matrix_elements), threads, 0, resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), 2,
          static_cast<std::int32_t>(nbf), fock, density, overlap, active,
          residual);
    } else {
      compute_energy_kernel<<<blocks_for(batch_size), threads, 0,
                              resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
          density, hcore, fock, nuclear_repulsion, active, energy);
      build_commutator_residual_kernel<<<blocks_for(matrix_elements), threads, 0,
                                         resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
          fock, density, overlap, active, residual);
    }
    update_diis_kernel<<<static_cast<unsigned>(batch_size), 1, 0,
                         resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        unrestricted ? 2 : 1,
        static_cast<std::uint32_t>(diis_history), fock, residual, active,
        fock_history, residual_history, diis_linear_system, diis_coefficients,
        diis_count, diis_head, eigensystem);
    if (unrestricted) {
      status = multiply_spin_matrices(
          eigensystem, true, false, orthogonalizer, false, temporary);
      if (status == QCE_STATUS_SUCCESS) {
        status = multiply_spin_matrices(
            orthogonalizer, false, true, temporary, true, eigensystem);
      }
      if (status == QCE_STATUS_SUCCESS) {
        expand_spin_active_kernel<<<blocks_for(spin_batch_size), threads, 0,
                                    resources.stream_>>>(
            static_cast<std::int32_t>(batch_size), 2, active, spin_active);
        status = launch_solver(resources, static_cast<int>(nbf),
                               static_cast<int>(spin_batch_size), eigensystem,
                               temporary, eigenvalues, lwork, solver_info,
                               spin_active);
      }
    } else {
      status = multiply_matrices(
          eigensystem, false, orthogonalizer, temporary);
      if (status == QCE_STATUS_SUCCESS) {
        status = multiply_matrices(
            orthogonalizer, true, temporary, eigensystem);
      }
      if (status == QCE_STATUS_SUCCESS) {
        status = launch_solver(resources, static_cast<int>(nbf),
                               static_cast<int>(batch_size), eigensystem,
                               temporary, eigenvalues, lwork, solver_info,
                               active);
      }
    }
    if (status == QCE_STATUS_SUCCESS) {
      if (unrestricted) {
        inspect_spin_solver_kernel<<<blocks_for(batch_size), threads, 0,
                                     resources.stream_>>>(
            static_cast<std::int32_t>(batch_size), 2, solver_info, active,
            failed, converged);
        status = multiply_spin_matrices(
            orthogonalizer, false, false, eigensystem, true, coefficients);
        if (status == QCE_STATUS_SUCCESS) {
          build_spin_density_kernel<<<blocks_for(spin_matrix_elements), threads,
                                      0, resources.stream_>>>(
              static_cast<std::int32_t>(batch_size), 2,
              static_cast<std::int32_t>(nbf), occupied, coefficients, active,
              next_density);
          update_uhf_convergence_kernel<<<blocks_for(batch_size), threads, 0,
                                          resources.stream_>>>(
              static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance,
              options.density_tolerance, energy, previous_energy,
              next_density, density, active, converged, iterations,
              energy_change, density_rms);
        }
      } else {
        inspect_solver_kernel<<<blocks_for(batch_size), threads, 0,
                                resources.stream_>>>(
            static_cast<std::int32_t>(batch_size), solver_info, active, failed,
            converged);
        status = multiply_matrices(
            orthogonalizer, false, eigensystem, coefficients);
        if (status == QCE_STATUS_SUCCESS) {
          build_density_kernel<<<blocks_for(matrix_elements), threads, 0,
                                 resources.stream_>>>(
              static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), occupied, coefficients, active,
              next_density);
          update_convergence_kernel<<<blocks_for(batch_size), threads, 0,
                                      resources.stream_>>>(
              static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance,
              options.density_tolerance, energy, previous_energy,
              next_density, density, active, converged, iterations,
              energy_change, density_rms);
        }
      }
      if (status == QCE_STATUS_SUCCESS) {
        tail_rhf_loop_kernel<<<1, 1, 0, resources.stream_>>>(
            static_cast<std::int32_t>(batch_size), options.max_iterations,
            active, iterations);
      }
    }
    cuda_error = cudaStreamEndCapture(resources.stream_, &resources.iteration_graph_);
    if (status != QCE_STATUS_SUCCESS || cuda_error != cudaSuccess ||
        resources.iteration_graph_ == nullptr) {
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs, status != QCE_STATUS_SUCCESS
                                       ? status
                                       : cuda_status(cuda_error));
      return outputs;
    }
    cuda_error = cudaGraphInstantiate(&resources.iteration_graph_exec_,
                                      resources.iteration_graph_,
                                      cudaGraphInstantiateFlagDeviceLaunch);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaGraphUpload(resources.iteration_graph_exec_, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    plan.initialized = true;
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaGraphLaunch(resources.iteration_graph_exec_, resources.stream_);
  }
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }

  // Rebuild from the converged density and diagonalize the un-extrapolated
  // Fock matrix. These orbitals define the energy-weighted density in the
  // Pulay term, matching the CPU analytic-gradient oracle.
  select_converged_kernel<<<blocks_for(batch_size), threads, 0, resources.stream_>>>(
      static_cast<std::int32_t>(batch_size), converged, failed, active);
  launch_fock_builder(density);
  if (unrestricted) {
    status = multiply_spin_matrices(
        fock, true, false, orthogonalizer, false, temporary);
    if (status == QCE_STATUS_SUCCESS) {
      status = multiply_spin_matrices(
          orthogonalizer, false, true, temporary, true, eigensystem);
    }
    if (status == QCE_STATUS_SUCCESS) {
      expand_spin_active_kernel<<<blocks_for(spin_batch_size), threads, 0,
                                  resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), 2, active, spin_active);
      status = launch_solver(resources, static_cast<int>(nbf),
                             static_cast<int>(spin_batch_size), eigensystem,
                             temporary, eigenvalues, lwork, solver_info,
                             spin_active);
    }
  } else {
    status = multiply_matrices(fock, false, orthogonalizer, temporary);
    if (status == QCE_STATUS_SUCCESS) {
      status = multiply_matrices(
          orthogonalizer, true, temporary, eigensystem);
    }
    if (status == QCE_STATUS_SUCCESS) {
      status = launch_solver(resources, static_cast<int>(nbf),
                             static_cast<int>(batch_size), eigensystem,
                             temporary, eigenvalues, lwork, solver_info,
                             active);
    }
  }
  if (status != QCE_STATUS_SUCCESS) {
    fill_global_failure(outputs, status);
    return outputs;
  }
  if (unrestricted) {
    inspect_spin_solver_kernel<<<blocks_for(batch_size), threads, 0,
                                 resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), 2, solver_info, active, failed,
        converged);
    status = multiply_spin_matrices(
        orthogonalizer, false, false, eigensystem, true, coefficients);
    if (status != QCE_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    build_spin_density_kernel<<<blocks_for(spin_matrix_elements), threads, 0,
                                resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), 2,
        static_cast<std::int32_t>(nbf), occupied, coefficients, active, density);
  } else {
    inspect_solver_kernel<<<blocks_for(batch_size), threads, 0,
                            resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), solver_info, active, failed,
        converged);
    status = multiply_matrices(
        orthogonalizer, false, eigensystem, coefficients);
    if (status != QCE_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    build_density_kernel<<<blocks_for(matrix_elements), threads, 0,
                           resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        occupied, coefficients, active, density);
  }
  launch_fock_builder(density);
  if (unrestricted) {
    compute_uhf_energy_kernel<<<blocks_for(batch_size), threads, 0,
                                resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        density, hcore, fock, nuclear_repulsion, active, energy);
    build_spin_weighted_density_kernel<<<blocks_for(spin_matrix_elements),
                                         threads, 0, resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        occupied, coefficients, eigenvalues, active, weighted_density);
    sum_uhf_spin_matrices_kernel<<<blocks_for(matrix_elements), threads, 0,
                                   resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        density, active, total_density);
    sum_uhf_spin_matrices_kernel<<<blocks_for(matrix_elements), threads, 0,
                                   resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        weighted_density, active, total_weighted_density);
  } else {
    compute_energy_kernel<<<blocks_for(batch_size), threads, 0,
                            resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        density, hcore, fock, nuclear_repulsion, active, energy);
    build_weighted_density_kernel<<<blocks_for(matrix_elements), threads, 0,
                                    resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        occupied, coefficients, eigenvalues, active, weighted_density);
  }
  cuda_error = cudaMemsetAsync(forces, 0, total_atoms * 3 * sizeof(double),
                               resources.stream_);
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  nuclear_force_kernel<<<blocks_for(force_coordinate_count), threads, 0,
                          resources.stream_>>>(
      device_batch, active, forces);
  one_electron_force_kernel<<<blocks_for(one_electron_force_elements), threads,
                               0, resources.stream_>>>(
      device_batch, ao_pair_first, ao_pair_second, pair_count,
      unrestricted ? total_density : density,
      unrestricted ? total_weighted_density : weighted_density, active, forces);
  if (unrestricted && persistent_eri) {
    two_electron_uhf_force_kernel<<<blocks_for(persistent_force_elements),
                                    threads, 0, resources.stream_>>>(
        device_batch, density, active, forces);
  } else if (unrestricted && quartet_direct) {
    launch_angular_force_quartets<true>(
        resources.stream_, plan.shell_quartet_tile_capacities,
        plan.shell_quartet_tile_offsets, device_batch,
        active_shell_quartet_tile_counts, active_shell_quartet_tiles,
        options.screening_tolerance, schwarz_bounds,
        transformed_direct ? direct_density : density, active, forces);
  } else if (unrestricted) {
    two_electron_uhf_force_direct_kernel<<<
        blocks_for(direct_force_elements), threads, 0, resources.stream_>>>(
        device_batch, options.screening_tolerance, ao_pair_first,
        ao_pair_second, pair_count, schwarz_bounds, density, active, forces);
  } else if (persistent_eri) {
    two_electron_force_kernel<<<blocks_for(persistent_force_elements), threads,
                                 0, resources.stream_>>>(
        device_batch, density, active, forces);
  } else if (quartet_direct) {
    launch_angular_force_quartets<false>(
        resources.stream_, plan.shell_quartet_tile_capacities,
        plan.shell_quartet_tile_offsets, device_batch,
        active_shell_quartet_tile_counts, active_shell_quartet_tiles,
        options.screening_tolerance, schwarz_bounds,
        transformed_direct ? direct_density : density, active, forces);
  } else {
    two_electron_force_direct_kernel<<<
        blocks_for(direct_force_elements), threads, 0, resources.stream_>>>(
        device_batch, options.screening_tolerance, ao_pair_first,
        ao_pair_second, pair_count, schwarz_bounds, density, active,
        forces);
  }

  cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }

  std::vector<double> host_energy(batch_size);
  std::vector<double> host_energy_change(batch_size);
  std::vector<double> host_density_rms(batch_size);
  std::vector<double> host_density(spin_matrix_elements);
  std::vector<double> host_forces(total_atoms * 3);
  std::vector<std::uint8_t> host_converged(batch_size);
  std::vector<std::uint8_t> host_failed(batch_size);
  std::vector<std::uint32_t> host_iterations(batch_size);
  const struct Download {
    void* host;
    const void* device;
    std::size_t bytes;
  } downloads[] = {
      {host_energy.data(), energy, batch_size * sizeof(double)},
      {host_energy_change.data(), energy_change, batch_size * sizeof(double)},
      {host_density_rms.data(), density_rms, batch_size * sizeof(double)},
      {host_density.data(), density, spin_matrix_elements * sizeof(double)},
      {host_forces.data(), forces, total_atoms * 3 * sizeof(double)},
      {host_converged.data(), converged, batch_size * sizeof(std::uint8_t)},
      {host_failed.data(), failed, batch_size * sizeof(std::uint8_t)},
      {host_iterations.data(), iterations, batch_size * sizeof(std::uint32_t)},
  };
  for (const Download& download : downloads) {
    cuda_error = cudaMemcpyAsync(download.host, download.device, download.bytes,
                                 cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  cuda_error = cudaStreamSynchronize(resources.stream_);
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }

  for (std::size_t system = 0; system < batch_size; ++system) {
    RhfBucketItem& output = outputs[system];
    core::ScfResult& result = output.scf;
    result.energy = host_energy[system];
    result.iterations = host_iterations[system];
    result.energy_change = host_energy_change[system];
    result.density_rms = host_density_rms[system];
    result.converged = host_converged[system] != 0 && host_failed[system] == 0;
    result.initial_density_used = host.warm_mask[system] != 0;
    const std::size_t density_stride = spin_count * matrix_size;
    result.density.assign(host_density.begin() + system * density_stride,
                          host_density.begin() + (system + 1) * density_stride);
    const std::size_t atom_begin = static_cast<std::size_t>(host.atom_offsets[system]);
    const std::size_t atom_end = static_cast<std::size_t>(host.atom_offsets[system + 1]);
    result.forces.assign(host_forces.begin() + atom_begin * 3,
                         host_forces.begin() + atom_end * 3);
    output.status = host_failed[system] != 0
        ? QCE_STATUS_NUMERICAL_FAILURE
        : (result.converged ? QCE_STATUS_SUCCESS : QCE_STATUS_SCF_NOT_CONVERGED);
  }
  return outputs;
}

}  // namespace

CudaRhfBasisLayoutStats inspect_rhf_cuda_basis_layout(
    const std::vector<core::System>& systems) {
  std::vector<const std::vector<double>*> initial_densities(systems.size(),
                                                            nullptr);
  HostBatch host;
  if (!pack_host_batch(systems, initial_densities, host)) {
    throw std::invalid_argument("systems cannot be represented by one CUDA RHF bucket");
  }

  std::size_t expanded_primitive_references = 0;
  for (const core::System& system : systems) {
    for (const core::Shell& shell : system.shells) {
      std::size_t shell_references = 0;
      if (!checked_multiply(molecule::cartesian_count(shell.angular_momentum),
                            shell.primitives.size(), shell_references) ||
          !checked_add(expanded_primitive_references, shell_references,
                       expanded_primitive_references)) {
        throw std::overflow_error("expanded CUDA primitive reference count overflowed");
      }
    }
  }

  const std::size_t device_basis_bytes =
      host.system_shell_offsets.size() * sizeof(std::int64_t) +
      host.shell_atoms.size() * sizeof(std::int32_t) +
      host.shell_angular.size() * sizeof(std::uint8_t) +
      host.shell_ao_offsets.size() * sizeof(std::int64_t) +
      host.shell_direct_ao_offsets.size() * sizeof(std::int64_t) +
      host.shell_primitive_offsets.size() * sizeof(std::int64_t) +
      host.system_shell_pair_offsets.size() * sizeof(std::int64_t) +
      host.system_shell_quartet_offsets.size() * sizeof(std::int64_t) +
      host.shell_pair_systems.size() * sizeof(std::int32_t) +
      host.shell_pair_first.size() * sizeof(std::int32_t) +
      host.shell_pair_second.size() * sizeof(std::int32_t) +
      host.ao_shells.size() * sizeof(std::int32_t) +
      host.ao_term_counts.size() * sizeof(std::uint8_t) +
      host.ao_term_angular.size() * sizeof(std::uint8_t) +
      host.ao_term_coefficients.size() * sizeof(double) +
      host.direct_ao_shells.size() * sizeof(std::int32_t) +
      host.direct_ao_angular.size() * sizeof(std::uint8_t) +
      host.direct_ao_coefficients.size() * sizeof(double) +
      host.ao_to_direct_transform.size() * sizeof(double) +
      host.primitive_exponents.size() * sizeof(double) +
      host.primitive_coefficients.size() * sizeof(double);
  return {
      systems.size(),
      host.shell_atoms.size(),
      host.shell_pair_first.size(),
      static_cast<std::size_t>(host.system_shell_quartet_offsets.back()),
      host.ao_shells.size(),
      host.primitive_exponents.size(),
      expanded_primitive_references,
      device_basis_bytes,
  };
}

namespace {

std::vector<RhfBucketItem> run_hf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool unrestricted) {
  if (plan == nullptr) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  HostBatch candidate;
  if (!pack_host_batch(systems, initial_densities, candidate, unrestricted)) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, QCE_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (*plan != nullptr && (*plan)->initialized &&
      ((*plan)->resources.device_id_ != device_id ||
       !same_topology((*plan)->topology, candidate) ||
       !same_options((*plan)->options, options) ||
       (*plan)->unrestricted != unrestricted)) {
    delete *plan;
    *plan = nullptr;
  }
  if (*plan == nullptr) {
    *plan = new (std::nothrow) CudaRhfBucketPlan{};
    if (*plan == nullptr) {
      std::vector<RhfBucketItem> outputs(systems.size());
      fill_global_failure(outputs, QCE_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
  }
  std::vector<RhfBucketItem> outputs = execute_hf_cuda_bucket(
      **plan, systems, options, initial_densities, device_id, unrestricted);
  const bool retry_without_cublas =
      !(*plan)->initialized && (*plan)->retry_without_cublas;
  if (!(*plan)->initialized) {
    delete *plan;
    *plan = nullptr;
  }
  if (retry_without_cublas) {
    // Provider setup or graph capture can reject a cuBLAS implementation on a
    // particular CUDA release. Rebuild once with the numerically identical
    // native kernel so public CUDA execution remains available.
    *plan = new (std::nothrow) CudaRhfBucketPlan{};
    if (*plan == nullptr) {
      fill_global_failure(outputs, QCE_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    (*plan)->cublas_enabled = false;
    outputs = execute_hf_cuda_bucket(
        **plan, systems, options, initial_densities, device_id, unrestricted);
    if (!(*plan)->initialized) {
      delete *plan;
      *plan = nullptr;
    }
  }
  return outputs;
}

}  // namespace

std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id) {
  return run_hf_cuda_bucket_cached(
      plan, systems, options, initial_densities, device_id, false);
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id) {
  return run_hf_cuda_bucket_cached(
      plan, systems, options, initial_densities, device_id, true);
}

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept {
  delete plan;
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id) {
  CudaRhfBucketPlan* plan = nullptr;
  std::vector<RhfBucketItem> outputs = run_rhf_cuda_bucket_cached(
      &plan, systems, options, initial_densities, device_id);
  destroy_rhf_cuda_bucket_plan(plan);
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const core::ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id) {
  CudaRhfBucketPlan* plan = nullptr;
  std::vector<RhfBucketItem> outputs = run_uhf_cuda_bucket_cached(
      &plan, systems, options, initial_densities, device_id);
  destroy_rhf_cuda_bucket_plan(plan);
  return outputs;
}

core::ScfResult run_rhf_cuda(const core::System& system,
                             const core::ScfOptions& options,
                             int device_id,
                             const std::vector<double>* initial_density) {
  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  std::vector<RhfBucketItem> result =
      run_rhf_cuda_bucket(systems, options, initial_densities, device_id);
  if (result.empty()) throw std::runtime_error("CUDA RHF returned no result");
  if (result.front().status == QCE_STATUS_CUDA_ERROR ||
      result.front().status == QCE_STATUS_OUT_OF_MEMORY) {
    throw std::runtime_error("CUDA RHF execution failed");
  }
  return std::move(result.front().scf);
}

core::ScfResult run_uhf_cuda(const core::System& system,
                             const core::ScfOptions& options,
                             int device_id,
                             const std::vector<double>* initial_density) {
  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  std::vector<RhfBucketItem> result =
      run_uhf_cuda_bucket(systems, options, initial_densities, device_id);
  if (result.empty()) throw std::runtime_error("CUDA UHF returned no result");
  if (result.front().status == QCE_STATUS_CUDA_ERROR ||
      result.front().status == QCE_STATUS_OUT_OF_MEMORY) {
    throw std::runtime_error("CUDA UHF execution failed");
  }
  return std::move(result.front().scf);
}

}  // namespace qce::scf

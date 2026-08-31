#include "scf/rhf.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"

#include "molecule/basis.hpp"
#include "scf/cuda_eigensolver_policy.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/direct_task_layout.hpp"
#include "scf/generated_shell_task.hpp"

#include "scf/aot_shell_registry.hpp"

#include <cublas_v2.h>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace vibeqc::scf {
namespace {

// Runtime policy parsing lives in a host-only C++ TU. Keep the numerical CUDA
// source's call sites unchanged while making policy-only edits incremental.
// The legacy spellings below remain documented here for source-level tooling;
// their actual ``std::getenv`` calls and thresholds are implemented by
// ``scf/cuda/rhf_policy.cpp`` so editing policy does not rebuild this TU:
//   std::getenv("VIBEQC_FINAL_FOCK_REBUILD")
//   std::getenv("VIBEQC_PPPS_SIGNATURE_BUCKETING")
//   std::getenv("VIBEQC_PPPS_BLOCK_THREADS")
//   std::getenv("VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING")
//   std::getenv("VIBEQC_ONE_ELECTRON_FORCE_SCALAR")
//   std::getenv("VIBEQC_PSSS_RESIDENT_BRA")
//   scalar_one_electron_force_environment == nullptr
//   kTightConvergedFockReuseDensityRms = 1.0e-12
//   kExpandedConvergedFockReuseDensityTolerance = 1.0e-9
//   kExpandedConvergedFockReuseDensityRms = 2.0e-9
using cuda_policy::bounded_direct_aot_only_diagnostic_requested;
using cuda_policy::bounded_direct_count_diagnostic_requested;
using cuda_policy::bounded_direct_fock_only_diagnostic_requested;
using cuda_policy::bounded_direct_streaming_override_requested;
using cuda_policy::bounded_fock_class_timing_requested;
using cuda_policy::configured_mixed_precision_fock_threshold;
using cuda_policy::converged_fock_reuse_density_rms;
using cuda_policy::direct_tile_validation_requested;
using cuda_policy::force_density_product_screening_requested;
using cuda_policy::graph_native_eigensolver_override_requested;
using cuda_policy::ppps_resident_block_threads_requested;
using cuda_policy::ppps_signature_bucketing_requested;
using cuda_policy::ppss_signature_bucketing_requested;
using cuda_policy::psps_signature_bucketing_requested;
using cuda_policy::resident_ppps_bra_requested;
using cuda_policy::resident_psss_bra_requested;
using cuda_policy::one_electron_force_scalar_requested;
using cuda_policy::reuse_converged_fock_requested;
using cuda_policy::xsyev_probe_skip_diagnostic_requested;

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
// Generated descriptors are a small staging cache, not topology.
// Dominant Fock classes stream directly from O(N_shell^2) shell-pair metadata;
// force classes that outgrow this cache are replayed losslessly through
// exact-class pages and the same generated consumers. Keeping the cache modest
// also leaves room for the large AOT module and CUDA Graph on a 32 GiB device.
constexpr std::size_t kBoundedGeneratedTasksPerShellPair = 1024;
constexpr std::size_t kBoundedGeneratedMaximumTaskCapacity =
    8U * 1024U * 1024U;
// Matrix reductions use one complete warp per system. Keep this independent
// from the generic capture-safe launch width so tuning other kernels cannot
// silently drop reductions from additional warps.
constexpr unsigned kMatrixReductionThreads = 32;
// External warm densities require an O(N^2) symmetry and metric-trace pass
// before they can enter a captured SCF replay. One block owns each system so
// batch-size-one production runs can spread that setup scan across the GPU.
constexpr unsigned kWarmDensityThreads = 256;
static_assert(kWarmDensityThreads % 32 == 0);
// Persistent direct-force workers retain one AO-quartet warp per block. Eight
// resident workers per SM balance the high-register force kernels while
// replacing topology-capacity grids with device-side work stealing.
// Fock, force, and generated-shell queues all launch one-warp workers and use
// the same occupancy target. Keep one shared limit so their scheduling policy
// cannot drift when the device SM count changes.
constexpr unsigned kPersistentQuartetWarpsPerMultiprocessor = 8;
// Resident psss force blocks keep one p-s primitive-pair list in shared
// memory while their threads traverse the system's s-s ket pairs. Large
// contracted bases fall back to the established compact-tile worker.
constexpr unsigned kResidentPsssThreads = 128;
constexpr std::size_t kResidentPsssMaximumBraPrimitivePairs = 64;
// Orders zero through six have dedicated analytic derivatives and enough work
// to amortize the device queue. Higher generic Dual3 orders retain fixed grids
// because queue state raises their already-maximal register footprint without
// improving the 96-AO profile.
constexpr unsigned kPersistentForceAngularOrderCount = 7;
// Fock orders zero through five stay below the worst high-order register
// footprint and contain the largest topology-capacity tails at 192 AOs.
constexpr unsigned kPersistentFockAngularOrderCount = 6;
static_assert(
    kPersistentFockAngularOrderCount <=
    detail::kDirectQuartetAngularOrderCount);
// Orders zero through two retain their specialized FP64 workers.  Keeping the
// mixed queue limited to the remaining partitions avoids duplicating the
// dominant low-order topology capacity solely for records that can never be
// routed to an FP32 recurrence.
constexpr unsigned kMixedFockMinimumAngularOrder = 3;
static_assert(
    kMixedFockMinimumAngularOrder <
    detail::kDirectQuartetAngularOrderCount);
// An ssss shell quartet is exactly one Cartesian AO quartet. Assign one whole
// shell task to each lane instead of leaving 31 lanes idle in the generic
// one-tile-per-warp mapping. Higher classes require a genuinely shell-fused
// contraction so their common primitive/root setup is not repeated per AO.
constexpr unsigned kPackedSsssAngularOrderCount = 1;
// Total angular order one contains only psss. Its three Cartesian outputs
// share every primitive-pair, product-center, Boys, and decay calculation, so
// one lane should own the complete shell task instead of one AO component.
constexpr unsigned kFusedPsssAngularOrder = 1;
// Order two is fully covered by psps, ppss, and dsss. A single shell-task
// worker can dispatch those three exact recurrences without another queue.
constexpr unsigned kFusedOrderTwoAngularOrder = 2;
constexpr unsigned kSsssShellClass = 0;
constexpr unsigned kPsssShellClass = 1;
// Triangular shell-class numbering maps (p s | p s) to class two. Keep the
// exact value next to the order-two dispatch because the fused force worker
// must also mask this class out of the generic AO-component fallback.
constexpr unsigned kPspsShellClass = 2;
constexpr unsigned kPpssShellClass = 3;
// Canonical (p p|p s) is the fourth triangular pair-of-pairs class:
// pair(pp)=2 and pair(ps)=1, hence 2*(2+1)/2 + 1 == 4.
constexpr unsigned kPppsShellClass = 4;
constexpr unsigned kPppsAngularOrder = 3;
constexpr unsigned kPpppShellClass = 5;
constexpr unsigned kDsssShellClass = 6;
constexpr unsigned kDspsShellClass = 7;
constexpr unsigned kDsppShellClass = 8;
constexpr unsigned kDpssShellClass = 10;
constexpr unsigned kDppsShellClass = 11;
constexpr unsigned kDpppShellClass = 12;
constexpr unsigned kDpdsShellClass = 13;
constexpr unsigned kDpdpShellClass = 14;
constexpr unsigned kDdpsShellClass = 16;
constexpr unsigned kDdppShellClass = 17;
constexpr unsigned kDddpShellClass = 19;
constexpr unsigned kDdddShellClass = 20;
constexpr unsigned kDdddAngularOrder = 8;
constexpr std::uint64_t kDdddShellClassMask =
    std::uint64_t{1} << kDdddShellClass;
// The production profile covers the contiguous canonical class range from
// ssss through dddd. Generated resident-bra Fock kernels own classes 0..19;
// dddd Fock uses the native exact recurrence below because the generated
// value consumer is not numerically reliable for production tasks. A
// separately qualified generated force consumer may still own dddd gradients.
// Both routes enumerate pair-class segments directly and therefore avoid a
// whole-topology generic fallback scan.
constexpr std::uint64_t kStreamingFockShellClassMask =
    (std::uint64_t{1} << 21U) - 1U;
constexpr std::uint64_t kGeneratedStreamingFockShellClassMask =
    kStreamingFockShellClassMask & ~kDdddShellClassMask;
constexpr std::uint64_t kNativeStreamingFockShellClassMask =
    kDdddShellClassMask;
// Fixed-topology ssss/psss already have handwritten Fock consumers, while the
// generated dddd consumer is rejected above. Keep those bits out of the fixed
// mask so the established exact routes remain single-counted and correct.
constexpr std::uint64_t kFixedTopologyGeneratedFockExclusionMask =
    (std::uint64_t{1} << 0U) |
    (std::uint64_t{1} << 1U) |
    kDdddShellClassMask;
// The generated resident ppps consumer stages one pp primitive-pair list in
// shared memory.  Larger lists stay on the established ordinary task path.
constexpr unsigned kGeneratedPppsResidentMaximumBraPrimitivePairs = 64;
// Signature bucketing groups both ordered PPPS orientations by the ket
// primitive-pair count. Counts 0..63 are exact and 64 is an overflow bucket;
// bundled production bases currently use no more than 15 ket pairs here.
constexpr unsigned kPppsSignaturePrimitivePairBuckets = 65;
constexpr unsigned kPppsSignatureBucketCount =
    2 * kPppsSignaturePrimitivePairBuckets;
// Whole-task and subgroup-task workers advance independent quartets in
// lockstep. Group both primitive-pair loop lengths and both pair orientations
// so each hardware warp executes a uniform recurrence slice. The same compact
// page-local histogram serves all selected scalar classes and PPPS without a
// topology-sized sort or a class-specific scientific fallback.
constexpr unsigned kBoundedForceSignatureOrientationCount = 4;
constexpr unsigned kBoundedForceSignatureBucketCount =
    kBoundedForceSignatureOrientationCount *
    kPppsSignaturePrimitivePairBuckets *
    kPppsSignaturePrimitivePairBuckets;
constexpr unsigned kBoundedForceSignatureScanThreads = 256;
constexpr unsigned kBoundedForceSignatureScanBlockCount =
    (kBoundedForceSignatureBucketCount +
     kBoundedForceSignatureScanThreads - 1U) /
    kBoundedForceSignatureScanThreads;
constexpr std::uint64_t kBoundedForceSignatureShellClassMask =
    (std::uint64_t{1} << kSsssShellClass) |
    (std::uint64_t{1} << kPsssShellClass) |
    (std::uint64_t{1} << kPspsShellClass) |
    (std::uint64_t{1} << kPpssShellClass) |
    (std::uint64_t{1} << kPppsShellClass) |
    (std::uint64_t{1} << kPpppShellClass) |
    (std::uint64_t{1} << kDsssShellClass) |
    (std::uint64_t{1} << kDspsShellClass) |
    (std::uint64_t{1} << kDsppShellClass) |
    (std::uint64_t{1} << kDpssShellClass) |
    (std::uint64_t{1} << kDppsShellClass) |
    (std::uint64_t{1} << kDpppShellClass) |
    (std::uint64_t{1} << kDpdsShellClass) |
    (std::uint64_t{1} << kDpdpShellClass) |
    (std::uint64_t{1} << kDdpsShellClass) |
    (std::uint64_t{1} << kDdppShellClass) |
    (std::uint64_t{1} << kDddpShellClass) |
    (std::uint64_t{1} << kDdddShellClass);
// The scalar PSPS and PPSS force workers assign one complete task to each
// lane. Group both canonical pair loop lengths so a warp advances through
// equal primitive work instead of serializing on the longest lane. Counts
// 0..63 are exact and 64 is the overflow bucket, matching the PPPS convention.
constexpr unsigned kLowOrderSignaturePrimitivePairBuckets = 65;
constexpr unsigned kLowOrderSignatureBucketsPerClass =
    kLowOrderSignaturePrimitivePairBuckets *
    kLowOrderSignaturePrimitivePairBuckets;
constexpr unsigned kLowOrderSignatureClassCount = 2;
constexpr unsigned kLowOrderSignatureElementCount =
    kLowOrderSignatureClassCount * kLowOrderSignatureBucketsPerClass;
static_assert(detail::kDirectQuartetThreads == 32);
// Generated order-five classes are removed from a compact generic fallback
// queue. Keeping the order explicit avoids coupling runtime selection to one
// generated shell class such as dppp.
constexpr unsigned kGenericOrderFiveAngularOrder = 5;
// Direct J/K scatters millions of independently evaluated AO quartets through
// FP64 atomics. Their nondeterministic accumulation order changes the total
// energy by a small number of representable values even after the density is
// stationary. Add only a machine-precision-scaled comparison guard; the
// requested absolute tolerance remains the dominant term for ordinary cases.
constexpr double kDirectFockEnergyRoundoffFactor = 16.0;
constexpr double kDoubleMachineEpsilon = 2.2204460492503131e-16;
// Force-product screening is an additional approximation on top of the Fock
// quartet gate. Do not inherit deliberately loose SCF screening thresholds:
// doing so removes derivative terms that remain present in the screened
// energy and breaks finite-difference consistency. Production 1e-14 runs keep
// the intended gate strength, while looser diagnostic runs use this cap.
constexpr double kForceDensityProductScreeningTolerance = 1.0e-14;
constexpr unsigned kBoundedDirectThreads =
    static_cast<unsigned>(detail::kBoundedDirectQueueCapacity);
static_assert(kBoundedDirectThreads % detail::kDirectQuartetThreads == 0);
static_assert(kBoundedDirectThreads <= 1024);

using GeneratedShellTask = detail::GeneratedShellTask;
using GeneratedPppsResidentTask = detail::GeneratedPppsResidentTask;
using GeneratedShellPairStream = detail::GeneratedShellPairStream;

/** Geometry-dependent direct-J/K work emitted by shell-bound compaction. */
struct ActiveShellQuartetTile {
  std::uint32_t first_pair;
  std::uint32_t second_pair;
  std::uint32_t tile;
};

static_assert(sizeof(ActiveShellQuartetTile) == 3 * sizeof(std::uint32_t));

/**
 * First invalid descriptor found by the optional post-compaction validator.
 *
 * The validator writes one record with an atomic first-writer-wins protocol,
 * so it can run on the same stream as compaction without device printf or a
 * host synchronization in the captured graph.  ``error`` is initialized to
 * ``kDirectTileValidationNoError`` before each replay.
 */
struct DirectTileValidationRecord {
  std::uint32_t error;
  std::uint32_t angular_order;
  std::uint32_t slot;
  std::uint32_t tile;
  std::uint32_t first_pair;
  std::uint32_t second_pair;
  std::int32_t shell[4];
  std::uint32_t direct_nbf;
  std::uint32_t first_pair_count;
  std::uint32_t second_pair_count;
  std::uint32_t i;
  std::uint32_t j;
  std::uint32_t k;
  std::uint32_t l;
  std::uint32_t active_tile_count;
  std::uint32_t partition_capacity;
  std::uint32_t partition_begin;
};

static_assert(sizeof(DirectTileValidationRecord) == 20 * sizeof(std::uint32_t));
constexpr std::uint32_t kDirectTileValidationNoError =
    std::numeric_limits<std::uint32_t>::max();
enum class DirectTileValidationError : std::uint32_t {
  count_exceeds_capacity = 1,
  pair_out_of_bounds = 2,
  shell_out_of_bounds = 3,
  tile_out_of_bounds = 4,
  ao_range_invalid = 5,
};

/** One static resident-bra task over a compact contiguous ket-pair chunk. */
struct PsssResidentTask {
  std::uint32_t bra_pair;
  std::uint32_t ket_begin;
  std::uint32_t ket_count;
};

static_assert(sizeof(PsssResidentTask) == 3 * sizeof(std::uint32_t));

/** Optional final-density profiling counters; never touched in normal runs. */
struct DeviceShellClassProfileEntry {
  unsigned long long shell_quartets;
  unsigned long long tiles;
  unsigned long long ao_quartets;
  unsigned long long primitive_quartets;
};

static_assert(sizeof(DeviceShellClassProfileEntry) ==
              sizeof(CudaRhfShellClassProfileEntry));

/** Raw spin-resolved density magnitudes for one direct-AO shell block. */
struct ShellPairDensityBounds {
  double coulomb;
  double exchange_alpha;
  double exchange_beta;
};

static_assert(sizeof(ShellPairDensityBounds) == 3 * sizeof(double));
static_assert(sizeof(ShellPairDensityBounds) ==
              sizeof(detail::GeneratedShellPairDensityBounds));
static_assert(alignof(ShellPairDensityBounds) ==
              alignof(detail::GeneratedShellPairDensityBounds));

/** Select the density gate applied while compacting direct shell quartets. */
enum class DirectScreeningPurpose : std::uint8_t {
  Fock,
  Force,
};

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

  __device__ MixedPrecisionFloat() : value(0.0F) {}
  __device__ MixedPrecisionFloat(double input)
      : value(static_cast<float>(input)) {}
};

/** Use FP32 recurrence coefficients only for the mixed value evaluator. */
template <typename Scalar>
using EvaluationReal = std::conditional_t<
    std::is_same_v<Scalar, MixedPrecisionFloat>, MixedPrecisionFloat, double>;

__device__ __forceinline__ MixedPrecisionFloat operator+(
    MixedPrecisionFloat a, MixedPrecisionFloat b) {
  return static_cast<double>(a.value + b.value);
}
__device__ __forceinline__ MixedPrecisionFloat operator-(
    MixedPrecisionFloat a, MixedPrecisionFloat b) {
  return static_cast<double>(a.value - b.value);
}
__device__ __forceinline__ MixedPrecisionFloat operator*(
    MixedPrecisionFloat a, MixedPrecisionFloat b) {
  return static_cast<double>(a.value * b.value);
}
__device__ __forceinline__ MixedPrecisionFloat operator/(
    MixedPrecisionFloat a, MixedPrecisionFloat b) {
  return static_cast<double>(a.value / b.value);
}
__device__ __forceinline__ MixedPrecisionFloat operator*(
    MixedPrecisionFloat a, double b) {
  return a * MixedPrecisionFloat{b};
}
__device__ __forceinline__ MixedPrecisionFloat operator*(
    double a, MixedPrecisionFloat b) {
  return MixedPrecisionFloat{a} * b;
}

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
  } else if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    return static_cast<double>(value.value);
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
  } else if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    return static_cast<double>(expf(value.value));
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
  } else if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    return static_cast<double>(sqrtf(value.value));
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
  } else if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    return static_cast<double>(erff(value.value));
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

/** Geometry and contraction data shared by every quartet using a shell pair. */
struct PrimitivePairData {
  double exponent_sum;
  double reduced_exponent;
  Vec3<double> product_center;
  double weighted_coefficient;
  double first_product_scale;
  double second_product_scale;
};

static_assert(sizeof(PrimitivePairData) == 8 * sizeof(double));
static_assert(sizeof(PrimitivePairData) ==
              sizeof(detail::GeneratedPrimitivePairData));
static_assert(alignof(PrimitivePairData) ==
              alignof(detail::GeneratedPrimitivePairData));

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
  std::int64_t total_shell_pair_blocks;
  std::int64_t total_shell_pair_block_quartets;
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
  const std::int64_t* system_shell_pair_block_offsets;
  const std::int64_t* system_shell_pair_block_quartet_offsets;
  const std::int32_t* shell_pair_systems;
  const std::int32_t* shell_pair_first;
  const std::int32_t* shell_pair_second;
  const std::int64_t* shell_pair_primitive_offsets;
  const PrimitivePairData* shell_primitive_pairs;
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

/** Resolve one packed lower-triangular shell-pair-block task to its system. */
__device__ std::int32_t shell_pair_block_quartet_system(
    const DeviceBatch& batch,
    std::size_t block_quartet) {
  std::int32_t lower = 0;
  std::int32_t upper = batch.batch_size;
  while (lower + 1 < upper) {
    const std::int32_t middle = lower + (upper - lower) / 2;
    if (static_cast<std::size_t>(
            batch.system_shell_pair_block_quartet_offsets[middle]) <=
        block_quartet) {
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

__device__ bool direct_shell_ao_range_valid(
    const DeviceBatch& batch,
    std::int32_t shell,
    std::size_t system_ao_begin,
    std::size_t direct_nbf,
    std::size_t& count) {
  const std::int64_t begin_value = batch.shell_direct_ao_offsets[shell];
  const std::int64_t end_value = batch.shell_direct_ao_offsets[shell + 1];
  if (begin_value < 0 || end_value < begin_value) return false;
  const std::size_t begin = static_cast<std::size_t>(begin_value);
  const std::size_t end = static_cast<std::size_t>(end_value);
  if (begin < system_ao_begin || end < begin ||
      end - system_ao_begin > direct_nbf) {
    return false;
  }
  count = end - begin;
  return true;
}

/** Record one validation failure without perturbing the production path. */
__device__ void record_direct_tile_validation_failure(
    DirectTileValidationRecord* record,
    DirectTileValidationError error,
    unsigned angular_order,
    std::size_t slot,
    const ActiveShellQuartetTile& tile,
    const std::int32_t* shells,
    std::size_t direct_nbf,
    std::size_t first_pair_count,
    std::size_t second_pair_count,
    std::size_t i,
    std::size_t j,
    std::size_t k,
    std::size_t l,
    std::size_t active_tile_count,
    std::size_t partition_capacity,
    std::size_t partition_begin) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(&record->error, kDirectTileValidationNoError, code) !=
      kDirectTileValidationNoError) {
    return;
  }
  record->angular_order = angular_order;
  record->slot = static_cast<std::uint32_t>(slot);
  record->tile = tile.tile;
  record->first_pair = tile.first_pair;
  record->second_pair = tile.second_pair;
#pragma unroll
  for (unsigned center = 0; center < 4U; ++center) {
    record->shell[center] = shells == nullptr ? -1 : shells[center];
  }
  record->direct_nbf = static_cast<std::uint32_t>(direct_nbf);
  record->first_pair_count = static_cast<std::uint32_t>(first_pair_count);
  record->second_pair_count = static_cast<std::uint32_t>(second_pair_count);
  record->i = static_cast<std::uint32_t>(i);
  record->j = static_cast<std::uint32_t>(j);
  record->k = static_cast<std::uint32_t>(k);
  record->l = static_cast<std::uint32_t>(l);
  record->active_tile_count = static_cast<std::uint32_t>(active_tile_count);
  record->partition_capacity = static_cast<std::uint32_t>(partition_capacity);
  record->partition_begin = static_cast<std::uint32_t>(partition_begin);
}

__device__ bool decode_direct_tile_ao_ordinal(
    const DeviceBatch& batch,
    const ActiveShellQuartetTile& tile,
    std::size_t ordinal,
    std::size_t first_pair_ao_count,
    std::size_t second_pair_ao_count,
    std::size_t system_ao_begin,
    std::size_t direct_nbf,
    std::size_t& i,
    std::size_t& j,
    std::size_t& k,
    std::size_t& l) {
  std::size_t first_ao_pair = ordinal / second_pair_ao_count;
  std::size_t second_ao_pair = ordinal % second_pair_ao_count;
  if (tile.first_pair == tile.second_pair) {
    decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
  }
  decode_shell_ao_pair(batch, tile.first_pair, first_ao_pair,
                       system_ao_begin, i, j);
  decode_shell_ao_pair(batch, tile.second_pair, second_ao_pair,
                       system_ao_begin, k, l);
  return i < direct_nbf && j < direct_nbf && k < direct_nbf &&
      l < direct_nbf;
}

/** Validate compact direct tiles before any generated or handwritten consumer. */
__global__ void validate_direct_tile_descriptors_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_tile_offsets,
    const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles,
    std::size_t total_tile_capacity,
    DirectTileValidationRecord* record) {
  const std::size_t slot =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= total_tile_capacity) return;

  unsigned angular_order = 0;
  while (angular_order + 1U < detail::kDirectQuartetAngularOrderCount &&
         slot >= active_tile_offsets[angular_order + 1U]) {
    ++angular_order;
  }
  const std::size_t partition_begin = active_tile_offsets[angular_order];
  const std::size_t partition_end = active_tile_offsets[angular_order + 1U];
  const std::size_t partition_capacity = partition_end >= partition_begin
      ? partition_end - partition_begin
      : 0U;
  const std::size_t active_tile_count = active_tile_counts[angular_order];
  ActiveShellQuartetTile empty_tile{};
  if (partition_end < partition_begin || active_tile_count > partition_capacity) {
    record_direct_tile_validation_failure(
        record, DirectTileValidationError::count_exceeds_capacity,
        angular_order, slot, empty_tile, nullptr,
        static_cast<std::size_t>(batch.direct_nbf), 0, 0, 0, 0, 0, 0,
        active_tile_count, partition_capacity, partition_begin);
    return;
  }
  if (slot - partition_begin >= active_tile_count) return;

  const ActiveShellQuartetTile tile = active_tiles[slot];
  if (tile.first_pair >= static_cast<std::uint32_t>(batch.total_shell_pairs) ||
      tile.second_pair >= static_cast<std::uint32_t>(batch.total_shell_pairs)) {
    record_direct_tile_validation_failure(
        record, DirectTileValidationError::pair_out_of_bounds,
        angular_order, slot, tile, nullptr,
        static_cast<std::size_t>(batch.direct_nbf), 0, 0, 0, 0, 0, 0,
        active_tile_count, partition_capacity, partition_begin);
    return;
  }

  const std::int32_t shells[4] = {
      batch.shell_pair_first[tile.first_pair],
      batch.shell_pair_second[tile.first_pair],
      batch.shell_pair_first[tile.second_pair],
      batch.shell_pair_second[tile.second_pair],
  };
  for (unsigned center = 0; center < 4U; ++center) {
    if (shells[center] < 0 ||
        shells[center] >= static_cast<std::int32_t>(batch.total_shells)) {
      record_direct_tile_validation_failure(
          record, DirectTileValidationError::shell_out_of_bounds,
          angular_order, slot, tile, shells,
          static_cast<std::size_t>(batch.direct_nbf), 0, 0, 0, 0, 0, 0,
          active_tile_count, partition_capacity, partition_begin);
      return;
    }
  }

  const std::int32_t system = batch.shell_pair_systems[tile.first_pair];
  const std::int32_t second_system = batch.shell_pair_systems[tile.second_pair];
  if (system < 0 || system >= batch.batch_size || second_system != system) {
    record_direct_tile_validation_failure(
        record, DirectTileValidationError::shell_out_of_bounds,
        angular_order, slot, tile, shells,
        static_cast<std::size_t>(batch.direct_nbf), 0, 0, 0, 0, 0, 0,
        active_tile_count, partition_capacity, partition_begin);
    return;
  }
  const std::size_t direct_nbf = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t system_ao_begin =
      static_cast<std::size_t>(system) * direct_nbf;
  std::size_t shell_counts[4]{};
  for (unsigned center = 0; center < 4U; ++center) {
    if (!direct_shell_ao_range_valid(batch, shells[center], system_ao_begin,
                                     direct_nbf, shell_counts[center])) {
      record_direct_tile_validation_failure(
          record, DirectTileValidationError::ao_range_invalid,
          angular_order, slot, tile, shells, direct_nbf, 0, 0, 0, 0, 0, 0,
          active_tile_count, partition_capacity, partition_begin);
      return;
    }
  }
  const std::size_t first_pair_ao_count = shells[0] == shells[1]
      ? shell_counts[0] * (shell_counts[0] + 1U) / 2U
      : shell_counts[0] * shell_counts[1];
  const std::size_t second_pair_ao_count = shells[2] == shells[3]
      ? shell_counts[2] * (shell_counts[2] + 1U) / 2U
      : shell_counts[2] * shell_counts[3];
  const std::size_t ao_quartet_count = tile.first_pair == tile.second_pair
      ? first_pair_ao_count * (first_pair_ao_count + 1U) / 2U
      : first_pair_ao_count * second_pair_ao_count;
  const std::size_t expected_tiles =
      (ao_quartet_count + detail::kDirectQuartetTileSize - 1U) /
      detail::kDirectQuartetTileSize;
  if (tile.tile >= expected_tiles || first_pair_ao_count == 0U ||
      second_pair_ao_count == 0U) {
    record_direct_tile_validation_failure(
        record, DirectTileValidationError::tile_out_of_bounds,
        angular_order, slot, tile, shells, direct_nbf, first_pair_ao_count,
        second_pair_ao_count, 0, 0, 0, 0, active_tile_count,
        partition_capacity, partition_begin);
    return;
  }

  const std::size_t ordinal = static_cast<std::size_t>(tile.tile) *
      detail::kDirectQuartetTileSize;
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t k = 0;
  std::size_t l = 0;
  if (!decode_direct_tile_ao_ordinal(
          batch, tile, ordinal, first_pair_ao_count, second_pair_ao_count,
          system_ao_begin, direct_nbf, i, j, k, l)) {
    record_direct_tile_validation_failure(
        record, DirectTileValidationError::ao_range_invalid,
        angular_order, slot, tile, shells, direct_nbf, first_pair_ao_count,
        second_pair_ao_count, i, j, k, l, active_tile_count,
        partition_capacity, partition_begin);
    return;
  }
  const std::size_t last_ordinal = min(
      ao_quartet_count - 1U,
      ordinal + detail::kDirectQuartetTileSize - 1U);
  if (!decode_direct_tile_ao_ordinal(
          batch, tile, last_ordinal, first_pair_ao_count,
          second_pair_ao_count, system_ao_begin, direct_nbf, i, j, k, l)) {
    record_direct_tile_validation_failure(
        record, DirectTileValidationError::ao_range_invalid,
        angular_order, slot, tile, shells, direct_nbf, first_pair_ao_count,
        second_pair_ao_count, i, j, k, l, active_tile_count,
        partition_capacity, partition_begin);
  }
}

/** Return the packed lower-triangle index for two shells in one system. */
__device__ std::size_t system_shell_pair_index(
    const DeviceBatch& batch,
    std::int32_t system,
    std::int32_t first_shell,
    std::int32_t second_shell) {
  const std::size_t shell_begin =
      static_cast<std::size_t>(batch.system_shell_offsets[system]);
  const std::size_t first =
      static_cast<std::size_t>(first_shell) - shell_begin;
  const std::size_t second =
      static_cast<std::size_t>(second_shell) - shell_begin;
  const std::size_t high = first > second ? first : second;
  const std::size_t low = first > second ? second : first;
  return static_cast<std::size_t>(
      batch.system_shell_pair_offsets[system]) + high * (high + 1) / 2 + low;
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
  using Real = EvaluationReal<Scalar>;
  const Real alpha_value{alpha};
  const Real beta_value{beta};
  const Real exponent = alpha_value + beta_value;
  return {(alpha_value * first.x + beta_value * second.x) / exponent,
          (alpha_value * first.y + beta_value * second.y) / exponent,
          (alpha_value * first.z + beta_value * second.z) / exponent};
}

/** Precompute geometry shared by every primitive quartet using a shell pair. */
__global__ void build_shell_primitive_pair_cache_kernel(
    DeviceBatch batch,
    PrimitivePairData* shell_primitive_pairs) {
  const std::size_t shell_pair = static_cast<std::size_t>(blockIdx.x);
  if (shell_pair >= static_cast<std::size_t>(batch.total_shell_pairs)) return;
  const std::int32_t first_shell = batch.shell_pair_first[shell_pair];
  const std::int32_t second_shell = batch.shell_pair_second[shell_pair];
  const std::int64_t first_begin =
      batch.shell_primitive_offsets[first_shell];
  const std::int64_t second_begin =
      batch.shell_primitive_offsets[second_shell];
  const std::size_t first_count = static_cast<std::size_t>(
      batch.shell_primitive_offsets[first_shell + 1] - first_begin);
  const std::size_t second_count = static_cast<std::size_t>(
      batch.shell_primitive_offsets[second_shell + 1] - second_begin);
  const std::size_t pair_count = first_count * second_count;
  const Vec3<double> first = atom_position<double>(
      batch, batch.shell_atoms[first_shell], -1);
  const Vec3<double> second = atom_position<double>(
      batch, batch.shell_atoms[second_shell], -1);
  const double squared_distance = distance_squared(first, second);
  const std::size_t output_begin = static_cast<std::size_t>(
      batch.shell_pair_primitive_offsets[shell_pair]);
  for (std::size_t ordinal = threadIdx.x; ordinal < pair_count;
       ordinal += blockDim.x) {
    const std::int64_t first_primitive = first_begin +
        static_cast<std::int64_t>(ordinal / second_count);
    const std::int64_t second_primitive = second_begin +
        static_cast<std::int64_t>(ordinal % second_count);
    const double alpha = batch.primitive_exponents[first_primitive];
    const double beta = batch.primitive_exponents[second_primitive];
    const double exponent_sum = alpha + beta;
    const double reduced_exponent = alpha * beta / exponent_sum;
    shell_primitive_pairs[output_begin + ordinal] = {
        exponent_sum,
        reduced_exponent,
        product_center(alpha, first, beta, second),
        batch.primitive_coefficients[first_primitive] *
            batch.primitive_coefficients[second_primitive] *
            exp(-reduced_exponent * squared_distance),
        alpha / exponent_sum,
        beta / exponent_sum,
    };
  }
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
  return reduced * (scalar<Scalar>(3.0) - 2.0 * reduced * squared_distance) *
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
  if constexpr (std::is_same_v<Scalar, MixedPrecisionFloat>) {
    // The alternating high-order series loses too many digits in FP32 near
    // its branch boundary.  Evaluate the small special-function table in FP64
    // and round once before the substantially larger Cartesian recurrence.
    double accurate_values[MaximumOrder + 1];
    boys_values<MaximumOrder, double>(
        static_cast<double>(argument.value), accurate_values);
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
  constexpr double series_threshold = MaximumOrder == 0 ? 1.0e-8
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
      sum = sum + term /
          static_cast<double>(2 * MaximumOrder + 2 * k + 1);
      term = term * (-1.0 * argument) / static_cast<double>(k + 1);
      if (fabs(scalar_value(term)) < 1.0e-18) break;
    }
    values[MaximumOrder] = sum;
    const Scalar exponential = qexp(-1.0 * argument);
    for (unsigned order = MaximumOrder; order > 0; --order) {
      values[order - 1] =
          (2.0 * argument * values[order] + exponential) /
          static_cast<double>(2 * order - 1);
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
__device__ void fill_coulomb(EvaluationReal<Scalar> exponent,
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
  EvaluationReal<Scalar> factor{1.0};
  for (unsigned n = 0; n <= MaximumAngular; ++n) {
    auxiliary.at(n, 0, 0, 0) = factor * boys[n];
    factor = factor * (-2.0 * exponent);
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

/**
 * Compact value-only overlap used by analytic one-electron derivatives.
 *
 * Center differentiation raises one basis function by one quantum, while the
 * kinetic operator may raise that result by two more. The generic Hermite
 * workspace is intentionally not enlarged for this force-only requirement;
 * the separable Obara-Saika overlap recurrence needs only the final t=0
 * integral for first-center order <=3 and second-center order <=6.
 */
__device__ double primitive_overlap_cartesian_compact(
    double alpha,
    const Vec3<double>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<double>& second,
    const Angular& angular_second) {
  constexpr unsigned FirstDimension = kMaximumAngularMomentum + 1;
  constexpr unsigned SecondDimension = kMaximumAngularMomentum + 4;
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  const double inverse_two_exponent = 0.5 / exponent;
  const Vec3<double> product = product_center(alpha, first, beta, second);
  double result = 1.0;
  for (int axis = 0; axis < 3; ++axis) {
    const unsigned first_power = angular_axis(angular_first, axis);
    const unsigned second_power = angular_axis(angular_second, axis);
    double values[FirstDimension][SecondDimension]{};
    const double first_coordinate = vec_axis(first, axis);
    const double second_coordinate = vec_axis(second, axis);
    const double difference = first_coordinate - second_coordinate;
    values[0][0] = sqrt(kPi / exponent) *
        exp(-reduced * difference * difference);
    const double product_first =
        vec_axis(product, axis) - first_coordinate;
    const double product_second =
        vec_axis(product, axis) - second_coordinate;
    for (unsigned i = 1; i <= first_power; ++i) {
      values[i][0] = product_first * values[i - 1][0];
      if (i > 1) {
        values[i][0] += static_cast<double>(i - 1) *
            inverse_two_exponent * values[i - 2][0];
      }
    }
    for (unsigned j = 1; j <= second_power; ++j) {
      values[0][j] = product_second * values[0][j - 1];
      if (j > 1) {
        values[0][j] += static_cast<double>(j - 1) *
            inverse_two_exponent * values[0][j - 2];
      }
      for (unsigned i = 1; i <= first_power; ++i) {
        values[i][j] = product_first * values[i - 1][j] +
            static_cast<double>(j) * inverse_two_exponent *
                values[i - 1][j - 1];
        if (i > 1) {
          values[i][j] += static_cast<double>(i - 1) *
              inverse_two_exponent * values[i - 2][j];
        }
      }
    }
    result *= values[first_power][second_power];
  }
  return result;
}

/** Value-only kinetic integral backed by the compact overlap recurrence. */
__device__ double primitive_kinetic_cartesian_compact(
    double alpha,
    const Vec3<double>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<double>& second,
    const Angular& angular_second) {
  double result = beta *
      (2.0 * static_cast<double>(angular_total(angular_second)) + 3.0) *
      primitive_overlap_cartesian_compact(
          alpha, first, angular_first, beta, second, angular_second);
  for (int axis = 0; axis < 3; ++axis) {
    Angular raised = angular_second;
    add_angular_axis(raised, axis, 2);
    result -= 2.0 * beta * beta * primitive_overlap_cartesian_compact(
        alpha, first, angular_first, beta, second, raised);
    const unsigned power = angular_axis(angular_second, axis);
    if (power >= 2) {
      Angular lowered = angular_second;
      add_angular_axis(lowered, axis, -2);
      result -= 0.5 * static_cast<double>(power * (power - 1)) *
          primitive_overlap_cartesian_compact(
              alpha, first, angular_first, beta, second, lowered);
    }
  }
  return result;
}

/** Differentiate an overlap integral at its second Gaussian center. */
__device__ void primitive_overlap_second_center_gradient(
    double alpha,
    const Vec3<double>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<double>& second,
    const Angular& angular_second,
    double (&gradient)[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    Angular raised = angular_second;
    add_angular_axis(raised, axis, 1);
    double value = 2.0 * beta * primitive_overlap_cartesian_compact(
        alpha, first, angular_first, beta, second, raised);
    const unsigned power = angular_axis(angular_second, axis);
    if (power > 0) {
      Angular lowered = angular_second;
      add_angular_axis(lowered, axis, -1);
      value -= static_cast<double>(power) *
          primitive_overlap_cartesian_compact(
              alpha, first, angular_first, beta, second, lowered);
    }
    gradient[axis] = value;
  }
}

/** Differentiate a kinetic integral at its second Gaussian center. */
__device__ void primitive_kinetic_second_center_gradient(
    double alpha,
    const Vec3<double>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<double>& second,
    const Angular& angular_second,
    double (&gradient)[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    Angular raised = angular_second;
    add_angular_axis(raised, axis, 1);
    double value = 2.0 * beta * primitive_kinetic_cartesian_compact(
        alpha, first, angular_first, beta, second, raised);
    const unsigned power = angular_axis(angular_second, axis);
    if (power > 0) {
      Angular lowered = angular_second;
      add_angular_axis(lowered, axis, -1);
      value -= static_cast<double>(power) *
          primitive_kinetic_cartesian_compact(
              alpha, first, angular_first, beta, second, lowered);
    }
    gradient[axis] = value;
  }
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

/** Compact Hermite workspace including one raised quantum on either center. */
struct OneElectronDerivativeHermiteCoefficients {
  static constexpr unsigned kIDimension = kMaximumAngularMomentum + 2;
  static constexpr unsigned kJDimension = kMaximumAngularMomentum + 2;
  static constexpr unsigned kTDimension = 2 * kMaximumAngularMomentum + 4;
  double data[kIDimension * kJDimension * kTDimension];

  __device__ double& at(unsigned i, unsigned j, unsigned t) {
    return data[(i * kJDimension + j) * kTDimension + t];
  }

  __device__ double at(unsigned i, unsigned j, unsigned t) const {
    return data[(i * kJDimension + j) * kTDimension + t];
  }
};

/** Fill every coefficient required by first derivatives of one shell pair. */
__device__ void fill_one_electron_derivative_hermite(
    unsigned maximum_i,
    unsigned maximum_j,
    double product,
    double center_a,
    double center_b,
    double alpha,
    double beta,
    OneElectronDerivativeHermiteCoefficients& coefficients) {
  for (unsigned item = 0;
       item < OneElectronDerivativeHermiteCoefficients::kIDimension *
                  OneElectronDerivativeHermiteCoefficients::kJDimension *
                  OneElectronDerivativeHermiteCoefficients::kTDimension;
       ++item) {
    coefficients.data[item] = 0.0;
  }
  const double exponent = alpha + beta;
  const double reduced = alpha * beta / exponent;
  const double difference = center_a - center_b;
  coefficients.at(0, 0, 0) = exp(-reduced * difference * difference);
  const double product_first = product - center_a;
  const double product_second = product - center_b;
  const double inverse_two_exponent = 0.5 / exponent;
  for (unsigned i = 0; i <= maximum_i; ++i) {
    for (unsigned j = 0; j <= maximum_j; ++j) {
      if (i == 0 && j == 0) continue;
      if (i > 0) {
        coefficients.at(i, j, 0) =
            product_first * coefficients.at(i - 1, j, 0) +
            coefficients.at(i - 1, j, 1);
      } else {
        coefficients.at(i, j, 0) =
            product_second * coefficients.at(i, j - 1, 0) +
            coefficients.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coefficients.at(i, j, t) =
              product_first * coefficients.at(i - 1, j, t) +
              inverse_two_exponent * coefficients.at(i - 1, j, t - 1) +
              static_cast<double>(t + 1) *
                  coefficients.at(i - 1, j, t + 1);
        } else {
          coefficients.at(i, j, t) =
              product_second * coefficients.at(i, j - 1, t) +
              inverse_two_exponent * coefficients.at(i, j - 1, t - 1) +
              static_cast<double>(t + 1) *
                  coefficients.at(i, j - 1, t + 1);
        }
      }
    }
  }
}

template <unsigned MaximumAngular>
__device__ __noinline__ void
primitive_nuclear_attraction_cartesian_atom_gradient_from_hermite(
    const DeviceBatch& batch,
    double alpha,
    const Angular& angular_first,
    double beta,
    const Angular& angular_second,
    std::int64_t atom,
    double exponent,
    const Vec3<double>& product,
    const OneElectronDerivativeHermiteCoefficients* coefficients,
    double (&first_gradient)[3],
    double (&second_gradient)[3]) {
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum + 1);
  CoulombAuxiliary<double, MaximumAngular> auxiliary;
  fill_coulomb<MaximumAngular>(
      exponent, product, atom_position<double>(batch, atom, -1), auxiliary);
  for (unsigned axis = 0; axis < 3; ++axis) {
    first_gradient[axis] = 0.0;
    second_gradient[axis] = 0.0;
  }
  const unsigned x_limit = angular_first.x + angular_second.x + 1;
  const unsigned y_limit = angular_first.y + angular_second.y + 1;
  const unsigned z_limit = angular_first.z + angular_second.z + 1;
  const double attraction_scale =
      -static_cast<double>(batch.atomic_numbers[atom]) *
      (2.0 * kPi / exponent);
  for (unsigned t = 0; t <= x_limit; ++t) {
    for (unsigned u = 0; u <= y_limit; ++u) {
      for (unsigned v = 0; v <= z_limit; ++v) {
        if (t + u + v > MaximumAngular) continue;
        const unsigned orders[3] = {t, u, v};
        double base[3];
        double first_derivative[3];
        double second_derivative[3];
        for (int axis = 0; axis < 3; ++axis) {
          const unsigned first_power = angular_axis(angular_first, axis);
          const unsigned second_power = angular_axis(angular_second, axis);
          const unsigned order = orders[axis];
          base[axis] = coefficients[axis].at(
              first_power, second_power, order);
          first_derivative[axis] = 2.0 * alpha * coefficients[axis].at(
              first_power + 1, second_power, order);
          if (first_power > 0) {
            first_derivative[axis] -= static_cast<double>(first_power) *
                coefficients[axis].at(
                    first_power - 1, second_power, order);
          }
          second_derivative[axis] = 2.0 * beta * coefficients[axis].at(
              first_power, second_power + 1, order);
          if (second_power > 0) {
            second_derivative[axis] -= static_cast<double>(second_power) *
                coefficients[axis].at(
                    first_power, second_power - 1, order);
          }
        }
        const double coulomb = attraction_scale * auxiliary.at(0, t, u, v);
        first_gradient[0] +=
            first_derivative[0] * base[1] * base[2] * coulomb;
        first_gradient[1] +=
            base[0] * first_derivative[1] * base[2] * coulomb;
        first_gradient[2] +=
            base[0] * base[1] * first_derivative[2] * coulomb;
        second_gradient[0] +=
            second_derivative[0] * base[1] * base[2] * coulomb;
        second_gradient[1] +=
            base[0] * second_derivative[1] * base[2] * coulomb;
        second_gradient[2] +=
            base[0] * base[1] * second_derivative[2] * coulomb;
      }
    }
  }
}

/**
 * Evaluate both basis-center gradients of one nucleus's attraction integral.
 *
 * All raised/lowered Cartesian components share one Hermite workspace, one
 * Boys sequence, and one Coulomb auxiliary recurrence. The separate
 * precomputed entry point lets a cooperative AO-pair worker build the Hermite
 * coefficients once and reuse them across all nuclear centers in its warp.
 */
template <unsigned MaximumAngular>
__device__ __noinline__ void
primitive_nuclear_attraction_cartesian_atom_gradient(
    const DeviceBatch& batch,
    double alpha,
    const Vec3<double>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<double>& second,
    const Angular& angular_second,
    std::int64_t atom,
    double (&first_gradient)[3],
    double (&second_gradient)[3]) {
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum + 1);
  const double exponent = alpha + beta;
  const Vec3<double> product = product_center(alpha, first, beta, second);
  OneElectronDerivativeHermiteCoefficients coefficients[3];
  for (int axis = 0; axis < 3; ++axis) {
    fill_one_electron_derivative_hermite(
        angular_axis(angular_first, axis) + 1,
        angular_axis(angular_second, axis) + 1, vec_axis(product, axis),
        vec_axis(first, axis), vec_axis(second, axis), alpha, beta,
        coefficients[axis]);
  }
  primitive_nuclear_attraction_cartesian_atom_gradient_from_hermite<
      MaximumAngular>(
      batch, alpha, angular_first, beta, angular_second, atom, exponent,
      product, coefficients, first_gradient, second_gradient);
}

template <unsigned MaximumAngular,
          typename Scalar,
          typename FirstCoefficients,
          typename SecondCoefficients>
__device__ __noinline__ Scalar eri_cartesian_value(
    EvaluationReal<Scalar> p,
    EvaluationReal<Scalar> q,
    EvaluationReal<Scalar> rho,
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
  const EvaluationReal<Scalar> prefactor =
      EvaluationReal<Scalar>{2.0 * pow(kPi, 2.5)} /
      (p * q * qsqrt(p + q));
  return prefactor * value;
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

/** Ten unique Cartesian Coulomb states through total order two. */
struct Order2CoulombValues {
  double c0;
  double cx;
  double cy;
  double cz;
  double cxx;
  double cxy;
  double cxz;
  double cyy;
  double cyz;
  double czz;
};

/** Build the complete order-two Coulomb tensor once per primitive quartet. */
__device__ __forceinline__ Order2CoulombValues order2_coulomb_values(
    double rho,
    double x,
    double y,
    double z,
    const double (&boys)[3]) {
  const double twice_rho = 2.0 * rho;
  const double twice_rho_squared = twice_rho * twice_rho;
  return {
      boys[0],
      -twice_rho * x * boys[1],
      -twice_rho * y * boys[1],
      -twice_rho * z * boys[1],
      twice_rho_squared * x * x * boys[2] - twice_rho * boys[1],
      twice_rho_squared * x * y * boys[2],
      twice_rho_squared * x * z * boys[2],
      twice_rho_squared * y * y * boys[2] - twice_rho * boys[1],
      twice_rho_squared * y * z * boys[2],
      twice_rho_squared * z * z * boys[2] - twice_rho * boys[1],
  };
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

/** Cartesian source component count for one s, p, or d shell. */
template <unsigned ShellAngular>
__host__ __device__ constexpr unsigned order2_shell_component_count() {
  static_assert(ShellAngular <= 2);
  return (ShellAngular + 1) * (ShellAngular + 2) / 2;
}

/** Return one CCA-ordered Cartesian component through d angular momentum. */
template <unsigned ShellAngular>
__device__ __forceinline__ Angular order2_shell_component(unsigned component) {
  static_assert(ShellAngular <= 2);
  if constexpr (ShellAngular == 0) {
    (void)component;
    return {0, 0, 0};
  } else if constexpr (ShellAngular == 1) {
    return component == 0 ? Angular{1, 0, 0}
         : component == 1 ? Angular{0, 1, 0}
                          : Angular{0, 0, 1};
  } else {
    switch (component) {
      case 0: return {2, 0, 0};
      case 1: return {1, 1, 0};
      case 2: return {1, 0, 1};
      case 3: return {0, 2, 0};
      case 4: return {0, 1, 1};
      default: return {0, 0, 2};
    }
  }
}

/** Maximum full Cartesian output count among the three order-two classes. */
struct Order2IntegralVector {
  double component[9];
};

/**
 * Contract one canonical order-two shell quartet into all requested outputs.
 *
 * `active_component_mask` uses the canonical full Cartesian product order.
 * Symmetry-diagonal tasks therefore avoid computing product entries that are
 * absent from their lower-triangular AO-quartet domain.
 */
template <unsigned FirstShellAngular,
          unsigned SecondShellAngular,
          unsigned ThirdShellAngular,
          unsigned FourthShellAngular>
__device__ __noinline__ Order2IntegralVector
contracted_eri_cartesian_source_order2_shell(
    const DeviceBatch& batch,
    std::int32_t first_shell,
    std::int32_t second_shell,
    std::int32_t third_shell,
    std::int32_t fourth_shell,
    unsigned active_component_mask) {
  static_assert(FirstShellAngular + SecondShellAngular +
                ThirdShellAngular + FourthShellAngular == 2);
  constexpr unsigned FirstCount =
      order2_shell_component_count<FirstShellAngular>();
  constexpr unsigned SecondCount =
      order2_shell_component_count<SecondShellAngular>();
  constexpr unsigned ThirdCount =
      order2_shell_component_count<ThirdShellAngular>();
  constexpr unsigned FourthCount =
      order2_shell_component_count<FourthShellAngular>();
  constexpr unsigned OutputCount =
      FirstCount * SecondCount * ThirdCount * FourthCount;
  static_assert(OutputCount <= 9);

  const std::int32_t shells[4] = {
      first_shell, second_shell, third_shell, fourth_shell};
  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[first_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[second_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[third_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1),
  };
  const std::int64_t ao_begin[4] = {
      batch.shell_direct_ao_offsets[first_shell],
      batch.shell_direct_ao_offsets[second_shell],
      batch.shell_direct_ao_offsets[third_shell],
      batch.shell_direct_ao_offsets[fourth_shell],
  };
  const double first_pair_distance =
      distance_squared(positions[0], positions[1]);
  const double second_pair_distance =
      distance_squared(positions[2], positions[3]);

  Order2IntegralVector result{};
  for (std::int64_t a = batch.shell_primitive_offsets[shells[0]];
       a < batch.shell_primitive_offsets[shells[0] + 1]; ++a) {
    const double alpha = batch.primitive_exponents[a];
    const double coefficient_a = batch.primitive_coefficients[a];
    for (std::int64_t b = batch.shell_primitive_offsets[shells[1]];
         b < batch.shell_primitive_offsets[shells[1] + 1]; ++b) {
      const double beta = batch.primitive_exponents[b];
      const double p = alpha + beta;
      const double mu = alpha * beta / p;
      const Vec3<double> product_p =
          product_center(alpha, positions[0], beta, positions[1]);
      const double first_pair_coefficient =
          coefficient_a * batch.primitive_coefficients[b];
      for (std::int64_t c = batch.shell_primitive_offsets[shells[2]];
           c < batch.shell_primitive_offsets[shells[2] + 1]; ++c) {
        const double gamma = batch.primitive_exponents[c];
        const double first_three_coefficient =
            first_pair_coefficient * batch.primitive_coefficients[c];
        for (std::int64_t d = batch.shell_primitive_offsets[shells[3]];
             d < batch.shell_primitive_offsets[shells[3] + 1]; ++d) {
          const double delta = batch.primitive_exponents[d];
          const double q = gamma + delta;
          const double nu = gamma * delta / q;
          const double rho = p * q / (p + q);
          const Vec3<double> product_q =
              product_center(gamma, positions[2], delta, positions[3]);
          const double x = product_p.x - product_q.x;
          const double y = product_p.y - product_q.y;
          const double z = product_p.z - product_q.z;
          double boys[3];
          boys_values<2>(rho * (x * x + y * y + z * z), boys);
          const Order2CoulombValues coulomb =
              order2_coulomb_values(rho, x, y, z, boys);
          const double pair_decay = exp(
              -mu * first_pair_distance - nu * second_pair_distance);
          const double primitive_coefficient =
              first_three_coefficient * batch.primitive_coefficients[d];
          const double common = primitive_coefficient *
              2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;

          if constexpr (FirstShellAngular == 1 &&
                        SecondShellAngular == 0 &&
                        ThirdShellAngular == 1 &&
                        FourthShellAngular == 0) {
            const double hp = 0.5 / p;
            const double hq = 0.5 / q;
            const double hpq = hp * hq;
            const Vec3<double> pa{
                product_p.x - positions[0].x,
                product_p.y - positions[0].y,
                product_p.z - positions[0].z};
            const Vec3<double> qc{
                product_q.x - positions[2].x,
                product_q.y - positions[2].y,
                product_q.z - positions[2].z};
            if ((active_component_mask & (1U << 0)) != 0U) {
              result.component[0] += common *
                  (pa.x * qc.x * coulomb.c0 + hp * qc.x * coulomb.cx -
                   hq * pa.x * coulomb.cx - hpq * coulomb.cxx);
            }
            if ((active_component_mask & (1U << 1)) != 0U) {
              result.component[1] += common *
                  (pa.x * qc.y * coulomb.c0 + hp * qc.y * coulomb.cx -
                   hq * pa.x * coulomb.cy - hpq * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 2)) != 0U) {
              result.component[2] += common *
                  (pa.x * qc.z * coulomb.c0 + hp * qc.z * coulomb.cx -
                   hq * pa.x * coulomb.cz - hpq * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 3)) != 0U) {
              result.component[3] += common *
                  (pa.y * qc.x * coulomb.c0 + hp * qc.x * coulomb.cy -
                   hq * pa.y * coulomb.cx - hpq * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 4)) != 0U) {
              result.component[4] += common *
                  (pa.y * qc.y * coulomb.c0 + hp * qc.y * coulomb.cy -
                   hq * pa.y * coulomb.cy - hpq * coulomb.cyy);
            }
            if ((active_component_mask & (1U << 5)) != 0U) {
              result.component[5] += common *
                  (pa.y * qc.z * coulomb.c0 + hp * qc.z * coulomb.cy -
                   hq * pa.y * coulomb.cz - hpq * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 6)) != 0U) {
              result.component[6] += common *
                  (pa.z * qc.x * coulomb.c0 + hp * qc.x * coulomb.cz -
                   hq * pa.z * coulomb.cx - hpq * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 7)) != 0U) {
              result.component[7] += common *
                  (pa.z * qc.y * coulomb.c0 + hp * qc.y * coulomb.cz -
                   hq * pa.z * coulomb.cy - hpq * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 8)) != 0U) {
              result.component[8] += common *
                  (pa.z * qc.z * coulomb.c0 + hp * qc.z * coulomb.cz -
                   hq * pa.z * coulomb.cz - hpq * coulomb.czz);
            }
          } else if constexpr (FirstShellAngular == 1 &&
                               SecondShellAngular == 1) {
            const double h = 0.5 / p;
            const double h2 = h * h;
            const Vec3<double> pa{
                product_p.x - positions[0].x,
                product_p.y - positions[0].y,
                product_p.z - positions[0].z};
            const Vec3<double> pb{
                product_p.x - positions[1].x,
                product_p.y - positions[1].y,
                product_p.z - positions[1].z};
            if ((active_component_mask & (1U << 0)) != 0U) {
              result.component[0] += common *
                  ((pa.x * pb.x + h) * coulomb.c0 +
                   h * (pb.x + pa.x) * coulomb.cx + h2 * coulomb.cxx);
            }
            if ((active_component_mask & (1U << 1)) != 0U) {
              result.component[1] += common *
                  (pa.x * pb.y * coulomb.c0 + h * pb.y * coulomb.cx +
                   h * pa.x * coulomb.cy + h2 * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 2)) != 0U) {
              result.component[2] += common *
                  (pa.x * pb.z * coulomb.c0 + h * pb.z * coulomb.cx +
                   h * pa.x * coulomb.cz + h2 * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 3)) != 0U) {
              result.component[3] += common *
                  (pa.y * pb.x * coulomb.c0 + h * pb.x * coulomb.cy +
                   h * pa.y * coulomb.cx + h2 * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 4)) != 0U) {
              result.component[4] += common *
                  ((pa.y * pb.y + h) * coulomb.c0 +
                   h * (pb.y + pa.y) * coulomb.cy + h2 * coulomb.cyy);
            }
            if ((active_component_mask & (1U << 5)) != 0U) {
              result.component[5] += common *
                  (pa.y * pb.z * coulomb.c0 + h * pb.z * coulomb.cy +
                   h * pa.y * coulomb.cz + h2 * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 6)) != 0U) {
              result.component[6] += common *
                  (pa.z * pb.x * coulomb.c0 + h * pb.x * coulomb.cz +
                   h * pa.z * coulomb.cx + h2 * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 7)) != 0U) {
              result.component[7] += common *
                  (pa.z * pb.y * coulomb.c0 + h * pb.y * coulomb.cz +
                   h * pa.z * coulomb.cy + h2 * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 8)) != 0U) {
              result.component[8] += common *
                  ((pa.z * pb.z + h) * coulomb.c0 +
                   h * (pb.z + pa.z) * coulomb.cz + h2 * coulomb.czz);
            }
          } else {
            static_assert(
                FirstShellAngular == 2 && SecondShellAngular == 0 &&
                ThirdShellAngular == 0 && FourthShellAngular == 0);
            const double h = 0.5 / p;
            const double h2 = h * h;
            const Vec3<double> pa{
                product_p.x - positions[0].x,
                product_p.y - positions[0].y,
                product_p.z - positions[0].z};
            if ((active_component_mask & (1U << 0)) != 0U) {
              result.component[0] += common *
                  ((pa.x * pa.x + h) * coulomb.c0 +
                   2.0 * h * pa.x * coulomb.cx + h2 * coulomb.cxx);
            }
            if ((active_component_mask & (1U << 1)) != 0U) {
              result.component[1] += common *
                  (pa.x * pa.y * coulomb.c0 + h * pa.y * coulomb.cx +
                   h * pa.x * coulomb.cy + h2 * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 2)) != 0U) {
              result.component[2] += common *
                  (pa.x * pa.z * coulomb.c0 + h * pa.z * coulomb.cx +
                   h * pa.x * coulomb.cz + h2 * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 3)) != 0U) {
              result.component[3] += common *
                  ((pa.y * pa.y + h) * coulomb.c0 +
                   2.0 * h * pa.y * coulomb.cy + h2 * coulomb.cyy);
            }
            if ((active_component_mask & (1U << 4)) != 0U) {
              result.component[4] += common *
                  (pa.y * pa.z * coulomb.c0 + h * pa.z * coulomb.cy +
                   h * pa.y * coulomb.cz + h2 * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 5)) != 0U) {
              result.component[5] += common *
                  ((pa.z * pa.z + h) * coulomb.c0 +
                   2.0 * h * pa.z * coulomb.cz + h2 * coulomb.czz);
            }
          }
        }
      }
    }
  }

  // Cartesian normalization is primitive-independent. Applying it once after
  // contraction avoids four coefficient loads for every component of every
  // primitive quartet.
  unsigned output = 0;
#pragma unroll
  for (unsigned first_component = 0;
       first_component < FirstCount; ++first_component) {
#pragma unroll
    for (unsigned second_component = 0;
         second_component < SecondCount; ++second_component) {
#pragma unroll
      for (unsigned third_component = 0;
           third_component < ThirdCount; ++third_component) {
#pragma unroll
        for (unsigned fourth_component = 0;
             fourth_component < FourthCount;
             ++fourth_component, ++output) {
          if ((active_component_mask & (1U << output)) == 0U) continue;
          result.component[output] *=
              batch.direct_ao_coefficients[
                  ao_begin[0] + first_component] *
              batch.direct_ao_coefficients[
                  ao_begin[1] + second_component] *
              batch.direct_ao_coefficients[
                  ao_begin[2] + third_component] *
              batch.direct_ao_coefficients[
                  ao_begin[3] + fourth_component];
        }
      }
    }
  }
  return result;
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

/** Value and P-Q chain derivatives of a weighted order-two Hermite DAG. */
struct WeightedOrder2Coulomb {
  double c0;
  double cx;
  double cy;
  double cz;
  double value;
  double chain[3];
};

/**
 * Contract the ten order-zero-through-two Hermite coefficients with exactly
 * twenty unique Cartesian Coulomb states. The returned chain is only the
 * derivative through P-Q; shell-class helpers add their explicit PA/PB/QC
 * coefficient derivatives and Gaussian-pair decay separately.
 */
__device__ __forceinline__ WeightedOrder2Coulomb
contract_weighted_order2_coulomb(
    double rho,
    double x,
    double y,
    double z,
    const double (&boys)[4],
    double h0,
    double hx,
    double hy,
    double hz,
    double hxx,
    double hxy,
    double hxz,
    double hyy,
    double hyz,
    double hzz) {
  const double twice_rho = 2.0 * rho;
  const double twice_rho_squared = twice_rho * twice_rho;
  const double twice_rho_cubed = twice_rho_squared * twice_rho;
  const double c0 = boys[0];
  const double cx = -twice_rho * x * boys[1];
  const double cy = -twice_rho * y * boys[1];
  const double cz = -twice_rho * z * boys[1];
  const double cxx =
      twice_rho_squared * x * x * boys[2] - twice_rho * boys[1];
  const double cxy = twice_rho_squared * x * y * boys[2];
  const double cxz = twice_rho_squared * x * z * boys[2];
  const double cyy =
      twice_rho_squared * y * y * boys[2] - twice_rho * boys[1];
  const double cyz = twice_rho_squared * y * z * boys[2];
  const double czz =
      twice_rho_squared * z * z * boys[2] - twice_rho * boys[1];
  const double cxxx =
      -twice_rho_cubed * x * x * x * boys[3] +
      3.0 * twice_rho_squared * x * boys[2];
  const double cxxy =
      -twice_rho_cubed * x * x * y * boys[3] +
      twice_rho_squared * y * boys[2];
  const double cxxz =
      -twice_rho_cubed * x * x * z * boys[3] +
      twice_rho_squared * z * boys[2];
  const double cxyy =
      -twice_rho_cubed * x * y * y * boys[3] +
      twice_rho_squared * x * boys[2];
  const double cxyz = -twice_rho_cubed * x * y * z * boys[3];
  const double cxzz =
      -twice_rho_cubed * x * z * z * boys[3] +
      twice_rho_squared * x * boys[2];
  const double cyyy =
      -twice_rho_cubed * y * y * y * boys[3] +
      3.0 * twice_rho_squared * y * boys[2];
  const double cyyz =
      -twice_rho_cubed * y * y * z * boys[3] +
      twice_rho_squared * z * boys[2];
  const double cyzz =
      -twice_rho_cubed * y * z * z * boys[3] +
      twice_rho_squared * y * boys[2];
  const double czzz =
      -twice_rho_cubed * z * z * z * boys[3] +
      3.0 * twice_rho_squared * z * boys[2];

  WeightedOrder2Coulomb result{};
  result.c0 = c0;
  result.cx = cx;
  result.cy = cy;
  result.cz = cz;
  result.value =
      h0 * c0 + hx * cx + hy * cy + hz * cz +
      hxx * cxx + hxy * cxy + hxz * cxz +
      hyy * cyy + hyz * cyz + hzz * czz;
  result.chain[0] =
      h0 * cx + hx * cxx + hy * cxy + hz * cxz +
      hxx * cxxx + hxy * cxxy + hxz * cxxz +
      hyy * cxyy + hyz * cxyz + hzz * cxzz;
  result.chain[1] =
      h0 * cy + hx * cxy + hy * cyy + hz * cyz +
      hxx * cxxy + hxy * cxyy + hxz * cxyz +
      hyy * cyyy + hyz * cyyz + hzz * cyzz;
  result.chain[2] =
      h0 * cz + hx * cxz + hy * cyz + hz * czz +
      hxx * cxxz + hxy * cxyz + hxz * cxzz +
      hyy * cyyz + hyz * cyzz + hzz * czzz;
  return result;
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
    using Real = EvaluationReal<Scalar>;
    const Real alpha_value{alpha};
    const Real beta_value{beta};
    const Real gamma_value{gamma};
    const Real delta_value{delta};
    const Real p = alpha_value + beta_value;
    const Real q = gamma_value + delta_value;
    const Real rho = p * q / (p + q);
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

/**
 * Contract exact all-center one-electron derivatives for one public AO pair.
 *
 * The Gaussian center identity
 * `d_B g_b = 2 beta g_(b+1) - l_b g_(b-1)` converts derivatives into
 * value-only integrals. Overlap and kinetic derivatives at the first center
 * follow from translation. For each nuclear-attraction term, one shared
 * Hermite/Coulomb recurrence forms both basis-center derivatives, and the
 * nuclear-center derivative is their negative sum.
 */
template <unsigned MaximumAngular>
__device__ void contracted_one_electron_force_pair(
    const DeviceBatch& batch,
    std::int32_t system,
    std::int32_t i,
    std::int32_t j,
    double density,
    double weighted_density,
    double* forces) {
  static_assert(MaximumAngular >= 1);
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum + 1);
  const std::int64_t ao_i = static_cast<std::int64_t>(system) * batch.nbf + i;
  const std::int64_t ao_j = static_cast<std::int64_t>(system) * batch.nbf + j;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const std::int64_t first_atom = batch.shell_atoms[shell_i];
  const std::int64_t second_atom = batch.shell_atoms[shell_j];
  const Vec3<double> first = atom_position<double>(batch, first_atom, -1);
  const Vec3<double> second = atom_position<double>(batch, second_atom, -1);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const double pair_weight = i == j ? 1.0 : 2.0;
  const double density_scale = -pair_weight * density;
  const double overlap_scale = pair_weight * weighted_density;
  double first_force[3]{};
  double second_force[3]{};

  // Overlap and kinetic operators are translation invariant, so the first
  // basis-center derivatives are the negatives of the second-center values.
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      const double primitive_weight = batch.primitive_coefficients[a] *
          batch.primitive_coefficients[b];
      for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
        const Angular angular_first = ao_angular(batch, ao_i, first_term);
        const double first_coefficient =
            ao_term_coefficient(batch, ao_i, first_term);
        for (unsigned second_term = 0; second_term < second_terms;
             ++second_term) {
          const Angular angular_second =
              ao_angular(batch, ao_j, second_term);
          const double weight = primitive_weight * first_coefficient *
              ao_term_coefficient(batch, ao_j, second_term);
          double overlap_gradient[3];
          double kinetic_gradient[3];
          primitive_overlap_second_center_gradient(
              batch.primitive_exponents[a], first, angular_first,
              batch.primitive_exponents[b], second, angular_second,
              overlap_gradient);
          primitive_kinetic_second_center_gradient(
              batch.primitive_exponents[a], first, angular_first,
              batch.primitive_exponents[b], second, angular_second,
              kinetic_gradient);
          for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
            const double contribution = weight *
                (density_scale * kinetic_gradient[coordinate] +
                 overlap_scale * overlap_gradient[coordinate]);
            first_force[coordinate] -= contribution;
            second_force[coordinate] += contribution;
          }
        }
      }
    }
  }

  // Keep one local accumulator per target nucleus. This preserves one atomic
  // add per Cartesian component and AO pair instead of one per primitive.
  for (std::int64_t atom = batch.atom_offsets[system];
       atom < batch.atom_offsets[system + 1]; ++atom) {
    double nuclear_force[3]{};
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        const double primitive_weight = batch.primitive_coefficients[a] *
            batch.primitive_coefficients[b];
        for (unsigned first_term = 0; first_term < first_terms;
             ++first_term) {
          const Angular angular_first = ao_angular(batch, ao_i, first_term);
          const double first_coefficient =
              ao_term_coefficient(batch, ao_i, first_term);
          for (unsigned second_term = 0; second_term < second_terms;
               ++second_term) {
            const Angular angular_second =
                ao_angular(batch, ao_j, second_term);
            const double weight = density_scale * primitive_weight *
                first_coefficient *
                ao_term_coefficient(batch, ao_j, second_term);
            double first_gradient[3];
            double second_gradient[3];
            primitive_nuclear_attraction_cartesian_atom_gradient<
                MaximumAngular>(
                batch, batch.primitive_exponents[a], first, angular_first,
                batch.primitive_exponents[b], second, angular_second, atom,
                first_gradient, second_gradient);
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              first_force[coordinate] += weight * first_gradient[coordinate];
              second_force[coordinate] +=
                  weight * second_gradient[coordinate];
              nuclear_force[coordinate] -= weight *
                  (first_gradient[coordinate] +
                   second_gradient[coordinate]);
            }
          }
        }
      }
    }
    const std::int64_t coordinate = atom * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (nuclear_force[axis] != 0.0) {
        atomicAdd(forces + coordinate + axis, nuclear_force[axis]);
      }
    }
  }

  const std::int64_t first_coordinate = first_atom * 3;
  const std::int64_t second_coordinate = second_atom * 3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (first_force[axis] != 0.0) {
      atomicAdd(forces + first_coordinate + axis, first_force[axis]);
    }
    if (second_force[axis] != 0.0) {
      atomicAdd(forces + second_coordinate + axis, second_force[axis]);
    }
  }
}

/**
 * Contract one AO pair with nuclear centers distributed across one warp.
 *
 * A block owns one public AO pair. Lane zero evaluates the translation-
 * invariant overlap/kinetic terms once, while every lane owns one point-charge
 * auxiliary center at a time. The large Hermite table depends only on the AO
 * pair and primitive pair, so lane zero materializes it in shared memory and
 * all nuclear-center lanes reuse it for their Coulomb recurrences. Basis-center
 * attraction derivatives are reduced in registers; this preserves the scalar
 * path's one force update per AO pair and center.
 */
template <unsigned MaximumAngular>
__device__ void contracted_one_electron_force_pair_cooperative(
    const DeviceBatch& batch,
    std::int32_t system,
    std::int32_t i,
    std::int32_t j,
    double density,
    double weighted_density,
    double* forces,
    OneElectronDerivativeHermiteCoefficients* shared_coefficients) {
  static_assert(MaximumAngular >= 1);
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum + 1);
  const unsigned lane = threadIdx.x;
  const std::int64_t ao_i = static_cast<std::int64_t>(system) * batch.nbf + i;
  const std::int64_t ao_j = static_cast<std::int64_t>(system) * batch.nbf + j;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const std::int64_t first_atom = batch.shell_atoms[shell_i];
  const std::int64_t second_atom = batch.shell_atoms[shell_j];
  const Vec3<double> first = atom_position<double>(batch, first_atom, -1);
  const Vec3<double> second = atom_position<double>(batch, second_atom, -1);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const double pair_weight = i == j ? 1.0 : 2.0;
  const double density_scale = -pair_weight * density;
  const double overlap_scale = pair_weight * weighted_density;
  double first_force[3]{};
  double second_force[3]{};

  // These terms do not depend on a point-charge center. Keeping them on lane
  // zero avoids repeating the compact overlap recurrence across the warp.
  if (lane == 0U) {
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        const double primitive_weight = batch.primitive_coefficients[a] *
            batch.primitive_coefficients[b];
        for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
          const Angular angular_first = ao_angular(batch, ao_i, first_term);
          const double first_coefficient =
              ao_term_coefficient(batch, ao_i, first_term);
          for (unsigned second_term = 0; second_term < second_terms;
               ++second_term) {
            const Angular angular_second =
                ao_angular(batch, ao_j, second_term);
            const double weight = primitive_weight * first_coefficient *
                ao_term_coefficient(batch, ao_j, second_term);
            double overlap_gradient[3];
            double kinetic_gradient[3];
            primitive_overlap_second_center_gradient(
                batch.primitive_exponents[a], first, angular_first,
                batch.primitive_exponents[b], second, angular_second,
                overlap_gradient);
            primitive_kinetic_second_center_gradient(
                batch.primitive_exponents[a], first, angular_first,
                batch.primitive_exponents[b], second, angular_second,
                kinetic_gradient);
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              const double contribution = weight *
                  (density_scale * kinetic_gradient[coordinate] +
                   overlap_scale * overlap_gradient[coordinate]);
              first_force[coordinate] -= contribution;
              second_force[coordinate] += contribution;
            }
          }
        }
      }
    }
  }

  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom_base = atom_begin; atom_base < atom_end;
       atom_base += warpSize) {
    const std::int64_t atom = atom_base + lane;
    double nuclear_force[3]{};
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      const double alpha = batch.primitive_exponents[a];
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        const double beta = batch.primitive_exponents[b];
        const double exponent = alpha + beta;
        const Vec3<double> product = product_center(alpha, first, beta, second);
        const double primitive_weight = batch.primitive_coefficients[a] *
            batch.primitive_coefficients[b];
        for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
          const Angular angular_first = ao_angular(batch, ao_i, first_term);
          const double first_coefficient =
              ao_term_coefficient(batch, ao_i, first_term);
          for (unsigned second_term = 0; second_term < second_terms;
               ++second_term) {
            const Angular angular_second =
                ao_angular(batch, ao_j, second_term);
            const double weight = density_scale * primitive_weight *
                first_coefficient *
                ao_term_coefficient(batch, ao_j, second_term);
            if (lane == 0U) {
              for (int axis = 0; axis < 3; ++axis) {
                fill_one_electron_derivative_hermite(
                    angular_axis(angular_first, axis) + 1,
                    angular_axis(angular_second, axis) + 1,
                    vec_axis(product, axis), vec_axis(first, axis),
                    vec_axis(second, axis), alpha, beta,
                    shared_coefficients[axis]);
              }
            }
            __syncthreads();
            if (atom < atom_end) {
              double first_gradient[3];
              double second_gradient[3];
              primitive_nuclear_attraction_cartesian_atom_gradient_from_hermite<
                  MaximumAngular>(
                  batch, alpha, angular_first, beta, angular_second, atom,
                  exponent, product, shared_coefficients, first_gradient,
                  second_gradient);
              for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
                first_force[coordinate] +=
                    weight * first_gradient[coordinate];
                second_force[coordinate] +=
                    weight * second_gradient[coordinate];
                nuclear_force[coordinate] -= weight *
                    (first_gradient[coordinate] +
                     second_gradient[coordinate]);
              }
            }
            // No lane may overwrite the shared Hermite table while another
            // lane is still evaluating its point-charge recurrence.
            __syncthreads();
          }
        }
      }
    }
    if (atom < atom_end) {
      const std::int64_t coordinate = atom * 3;
      for (unsigned axis = 0; axis < 3; ++axis) {
        if (nuclear_force[axis] != 0.0) {
          atomicAdd(forces + coordinate + axis, nuclear_force[axis]);
        }
      }
    }
  }

  for (unsigned offset = warpSize / 2; offset != 0; offset /= 2) {
    for (unsigned axis = 0; axis < 3; ++axis) {
      first_force[axis] +=
          __shfl_down_sync(0xffffffffU, first_force[axis], offset);
      second_force[axis] +=
          __shfl_down_sync(0xffffffffU, second_force[axis], offset);
    }
  }
  if (lane == 0U) {
    const std::int64_t first_coordinate = first_atom * 3;
    const std::int64_t second_coordinate = second_atom * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (first_force[axis] != 0.0) {
        atomicAdd(forces + first_coordinate + axis, first_force[axis]);
      }
      if (second_force[axis] != 0.0) {
        atomicAdd(forces + second_coordinate + axis, second_force[axis]);
      }
    }
  }
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
  using CoefficientScalar = std::conditional_t<
      std::is_same_v<Scalar, MixedPrecisionFloat>, Scalar, double>;
  const CoefficientScalar angular_coefficient = CoefficientScalar{
      batch.direct_ao_coefficients[ao_i] *
      batch.direct_ao_coefficients[ao_j] *
      batch.direct_ao_coefficients[ao_k] *
      batch.direct_ao_coefficients[ao_l]};

  Scalar result = scalar<Scalar>(0.0);
  for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
       a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
         b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[shell_k];
           c < batch.shell_primitive_offsets[shell_k + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[shell_l];
             d < batch.shell_primitive_offsets[shell_l + 1]; ++d) {
          const CoefficientScalar weight = angular_coefficient *
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

/** Three contracted Cartesian components of canonical (p s | s s). */
struct PsssIntegralVector {
  double axis[3];
};

/** Density-weighted psss derivatives for the first three canonical centers. */
struct PsssWeightedGradient {
  double center[3][3];
};

/**
 * Contract all psss Cartesian outputs with one shared primitive traversal.
 *
 * The p_x, p_y, and p_z values differ only in the final PA/PQ component of
 * the closed order-one expression. Keeping the complete shell task in one
 * lane removes threefold repetition of product centers, pair decay, and Boys
 * values while retaining the established CCA x/y/z component order.
 */
__device__ __noinline__ PsssIntegralVector
contracted_eri_cartesian_source_psss(
    const DeviceBatch& batch,
    std::size_t first_shell_pair,
    std::size_t second_shell_pair,
    std::int32_t p_shell,
    std::int32_t paired_s_shell,
    std::int32_t third_shell,
    std::int32_t fourth_shell) {
  const Vec3<double> first = atom_position<double>(
      batch, batch.shell_atoms[p_shell], -1);

  const std::int64_t p_ao_begin = batch.shell_direct_ao_offsets[p_shell];
  const double s_angular_coefficient =
      batch.direct_ao_coefficients[
          batch.shell_direct_ao_offsets[paired_s_shell]] *
      batch.direct_ao_coefficients[
          batch.shell_direct_ao_offsets[third_shell]] *
      batch.direct_ao_coefficients[
          batch.shell_direct_ao_offsets[fourth_shell]];
  const double angular_coefficient[3] = {
      s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin],
      s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin + 1],
      s_angular_coefficient * batch.direct_ao_coefficients[p_ao_begin + 2],
  };

  PsssIntegralVector result{};
  const std::int64_t first_pair_begin =
      batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end =
      batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin =
      batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end =
      batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end; ++first_primitive) {
    const PrimitivePairData first_pair =
        batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const Vec3<double> product_p = first_pair.product_center;
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end; ++second_primitive) {
      const PrimitivePairData second_pair =
          batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      double boys[2];
      boys_values<1>(
          rho * distance_squared(product_p, product_q), boys);
      const double prefactor =
          first_pair.weighted_coefficient *
          second_pair.weighted_coefficient *
          2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));
      const double coulomb_scale = rho / p;
      result.axis[0] += angular_coefficient[0] * prefactor *
          ((product_p.x - first.x) * boys[0] -
           coulomb_scale * (product_p.x - product_q.x) * boys[1]);
      result.axis[1] += angular_coefficient[1] * prefactor *
          ((product_p.y - first.y) * boys[0] -
           coulomb_scale * (product_p.y - product_q.y) * boys[1]);
      result.axis[2] += angular_coefficient[2] * prefactor *
          ((product_p.z - first.z) * boys[0] -
           coulomb_scale * (product_p.z - product_q.z) * boys[1]);
    }
  }
  return result;
}

/**
 * Contract all three psss component gradients in one primitive traversal.
 *
 * `density_coefficient` already contains the exact eightfold RHF/UHF density
 * contraction for each p axis. Linearity lets the three component gradients
 * be combined before the primitive loops: PA, P-Q, and their coordinate
 * derivatives become short weighted dot products, while product centers,
 * decay, and Boys values are evaluated only once. The fourth-center gradient
 * is intentionally omitted and restored from translation by the force task.
 */
template <bool ResidentBra>
__device__ __noinline__ PsssWeightedGradient
contracted_eri_cartesian_source_psss_weighted_gradient(
    const DeviceBatch& batch,
    std::size_t first_shell_pair,
    std::size_t second_shell_pair,
    std::int32_t p_shell,
    std::int32_t paired_s_shell,
    std::int32_t third_shell,
    std::int32_t fourth_shell,
    const double (&density_coefficient)[3],
    const PrimitivePairData* resident_first_pairs,
    std::int64_t resident_first_pair_count) {
  const Vec3<double> first = atom_position<double>(
      batch, batch.shell_atoms[p_shell], -1);
  const Vec3<double> second = atom_position<double>(
      batch, batch.shell_atoms[paired_s_shell], -1);
  const Vec3<double> third = atom_position<double>(
      batch, batch.shell_atoms[third_shell], -1);
  const Vec3<double> fourth = atom_position<double>(
      batch, batch.shell_atoms[fourth_shell], -1);

  const std::int64_t p_ao_begin = batch.shell_direct_ao_offsets[p_shell];
  const double s_angular_coefficient =
      batch.direct_ao_coefficients[
          batch.shell_direct_ao_offsets[paired_s_shell]] *
      batch.direct_ao_coefficients[
          batch.shell_direct_ao_offsets[third_shell]] *
      batch.direct_ao_coefficients[
          batch.shell_direct_ao_offsets[fourth_shell]];
  const double axis_weight[3] = {
      density_coefficient[0] * s_angular_coefficient *
          batch.direct_ao_coefficients[p_ao_begin],
      density_coefficient[1] * s_angular_coefficient *
          batch.direct_ao_coefficients[p_ao_begin + 1],
      density_coefficient[2] * s_angular_coefficient *
          batch.direct_ao_coefficients[p_ao_begin + 2],
  };

  PsssWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == p_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_shell;
  const std::int64_t first_pair_begin = ResidentBra
      ? 0
      : batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = ResidentBra
      ? resident_first_pair_count
      : batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin =
      batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end =
      batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end; ++first_primitive) {
    const PrimitivePairData first_pair = ResidentBra
        ? resident_first_pairs[first_primitive]
        : batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const Vec3<double> product_p = first_pair.product_center;
    const double first_product_scale = first_pair_matches_canonical_order
        ? first_pair.first_product_scale
        : first_pair.second_product_scale;
    const double second_product_scale = first_pair_matches_canonical_order
        ? first_pair.second_product_scale
        : first_pair.first_product_scale;
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end; ++second_primitive) {
      const PrimitivePairData second_pair =
          batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const Vec3<double> product_difference{
          product_p.x - product_q.x,
          product_p.y - product_q.y,
          product_p.z - product_q.z,
      };
      const Vec3<double> pa{
          product_p.x - first.x,
          product_p.y - first.y,
          product_p.z - first.z,
      };
      double boys[3];
      boys_values<2>(
          rho * distance_squared(product_p, product_q), boys);
      const double prefactor =
          first_pair.weighted_coefficient *
          second_pair.weighted_coefficient *
          2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q));
      const double coulomb_scale = rho / p;
      const double weighted_pa =
          axis_weight[0] * pa.x + axis_weight[1] * pa.y +
          axis_weight[2] * pa.z;
      const double weighted_pq =
          axis_weight[0] * product_difference.x +
          axis_weight[1] * product_difference.y +
          axis_weight[2] * product_difference.z;
      const double weighted_value =
          weighted_pa * boys[0] -
          coulomb_scale * weighted_pq * boys[1];
      const double third_product_scale =
          second_pair_matches_canonical_order
              ? second_pair.first_product_scale
              : second_pair.second_product_scale;
      const double product_scales[3] = {
          first_product_scale, second_product_scale,
          -third_product_scale};

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          double decay_derivative = 0.0;
          if (center < 2) {
            const double difference =
                vec_axis(first, coordinate) -
                vec_axis(second, coordinate);
            decay_derivative =
                (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
          } else {
            const double difference =
                vec_axis(third, coordinate) -
                vec_axis(fourth, coordinate);
            decay_derivative = -2.0 * nu * difference;
          }
          const double argument_derivative =
              2.0 * rho * product_scales[center] *
              vec_axis(product_difference, coordinate);
          const double pa_derivative = center == 0
              ? first_product_scale - 1.0
              : (center == 1 ? second_product_scale : 0.0);
          const double weighted_value_derivative =
              axis_weight[coordinate] * pa_derivative * boys[0] -
              weighted_pa * boys[1] * argument_derivative -
              coulomb_scale * axis_weight[coordinate] *
                  product_scales[center] * boys[1] +
              coulomb_scale * weighted_pq * boys[2] *
                  argument_derivative;
          result.center[center][coordinate] += prefactor *
              (weighted_value_derivative +
               weighted_value * decay_derivative);
        }
      }
    }
  }
  return result;
}

/** Density-weighted psps derivatives for the first three canonical centers. */
struct PspsWeightedGradient {
  double center[3][3];
};

/**
 * Contract all nine canonical (p s | p s) components through one closed DAG.
 *
 * `component_weight` contains the screened density contraction and all four
 * Cartesian AO normalizations for the exact decoded AO-quartet domain. In
 * particular, missing entries from an identical-shell-pair triangular domain
 * remain zero; mirroring them would double-count the established ERI
 * multiplicity. The recurrence below forms the ten unique Coulomb states
 * through order two and their ten order-three raises exactly once per
 * primitive quartet, then contracts value and three coordinate derivatives.
 */
__device__ __noinline__ PspsWeightedGradient
contracted_eri_cartesian_source_psps_weighted_gradient(
    const DeviceBatch& batch,
    std::size_t first_shell_pair,
    std::size_t second_shell_pair,
    std::int32_t first_p_shell,
    std::int32_t first_s_shell,
    std::int32_t second_p_shell,
    std::int32_t second_s_shell,
    const double (&component_weight)[9]) {
  const Vec3<double> first =
      atom_position<double>(batch, batch.shell_atoms[first_p_shell], -1);
  const Vec3<double> second =
      atom_position<double>(batch, batch.shell_atoms[first_s_shell], -1);
  const Vec3<double> third =
      atom_position<double>(batch, batch.shell_atoms[second_p_shell], -1);
  const Vec3<double> fourth =
      atom_position<double>(batch, batch.shell_atoms[second_s_shell], -1);

  PspsWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == first_p_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == second_p_shell;
  const std::int64_t first_pair_begin =
      batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end =
      batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin =
      batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end =
      batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair =
        batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const double inverse_two_p = 0.5 / p;
    const Vec3<double> product_p = first_pair.product_center;
    const Vec3<double> pa{
        product_p.x - first.x,
        product_p.y - first.y,
        product_p.z - first.z,
    };
    const double first_product_scale = first_pair_matches_canonical_order
                                           ? first_pair.first_product_scale
                                           : first_pair.second_product_scale;
    const double second_product_scale = first_pair_matches_canonical_order
                                            ? first_pair.second_product_scale
                                            : first_pair.first_product_scale;
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair =
          batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const double inverse_two_q = 0.5 / q;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const Vec3<double> qc{
          product_q.x - third.x,
          product_q.y - third.y,
          product_q.z - third.z,
      };
      const double x = product_p.x - product_q.x;
      const double y = product_p.y - product_q.y;
      const double z = product_p.z - product_q.z;
      double boys[4];
      boys_values<3>(rho * (x * x + y * y + z * z), boys);

      const double row_x = component_weight[0] * qc.x +
                           component_weight[1] * qc.y +
                           component_weight[2] * qc.z;
      const double row_y = component_weight[3] * qc.x +
                           component_weight[4] * qc.y +
                           component_weight[5] * qc.z;
      const double row_z = component_weight[6] * qc.x +
                           component_weight[7] * qc.y +
                           component_weight[8] * qc.z;
      const double column_x = component_weight[0] * pa.x +
                              component_weight[3] * pa.y +
                              component_weight[6] * pa.z;
      const double column_y = component_weight[1] * pa.x +
                              component_weight[4] * pa.y +
                              component_weight[7] * pa.z;
      const double column_z = component_weight[2] * pa.x +
                              component_weight[5] * pa.y +
                              component_weight[8] * pa.z;
      const double h0 = pa.x * row_x + pa.y * row_y + pa.z * row_z;
      const double hx = inverse_two_p * row_x - inverse_two_q * column_x;
      const double hy = inverse_two_p * row_y - inverse_two_q * column_y;
      const double hz = inverse_two_p * row_z - inverse_two_q * column_z;
      const double second_scale = -inverse_two_p * inverse_two_q;
      const double hxx = second_scale * component_weight[0];
      const double hxy =
          second_scale * (component_weight[1] + component_weight[3]);
      const double hxz =
          second_scale * (component_weight[2] + component_weight[6]);
      const double hyy = second_scale * component_weight[4];
      const double hyz =
          second_scale * (component_weight[5] + component_weight[7]);
      const double hzz = second_scale * component_weight[8];

      const WeightedOrder2Coulomb coulomb = contract_weighted_order2_coulomb(
          rho, x, y, z, boys, h0, hx, hy, hz, hxx, hxy, hxz, hyy, hyz, hzz);

      const double first_explicit[3] = {
          row_x * coulomb.c0 -
              inverse_two_q * (component_weight[0] * coulomb.cx +
                               component_weight[1] * coulomb.cy +
                               component_weight[2] * coulomb.cz),
          row_y * coulomb.c0 -
              inverse_two_q * (component_weight[3] * coulomb.cx +
                               component_weight[4] * coulomb.cy +
                               component_weight[5] * coulomb.cz),
          row_z * coulomb.c0 -
              inverse_two_q * (component_weight[6] * coulomb.cx +
                               component_weight[7] * coulomb.cy +
                               component_weight[8] * coulomb.cz),
      };
      const double second_explicit[3] = {
          column_x * coulomb.c0 +
              inverse_two_p * (component_weight[0] * coulomb.cx +
                               component_weight[3] * coulomb.cy +
                               component_weight[6] * coulomb.cz),
          column_y * coulomb.c0 +
              inverse_two_p * (component_weight[1] * coulomb.cx +
                               component_weight[4] * coulomb.cy +
                               component_weight[7] * coulomb.cz),
          column_z * coulomb.c0 +
              inverse_two_p * (component_weight[2] * coulomb.cx +
                               component_weight[5] * coulomb.cy +
                               component_weight[8] * coulomb.cz),
      };
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;
      const double product_scale[3] = {
          first_product_scale, second_product_scale, -third_product_scale};
      const double shift_scale[3] = {first_product_scale - 1.0,
                                     second_product_scale,
                                     third_product_scale - 1.0};
      const double prefactor = first_pair.weighted_coefficient *
                               second_pair.weighted_coefficient * 2.0 *
                               pow(kPi, 2.5) / (p * q * sqrt(p + q));

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          const double pair_coefficient_derivative =
              shift_scale[center] * (center < 2 ? first_explicit[coordinate]
                                                : second_explicit[coordinate]);
          double decay_derivative = 0.0;
          if (center < 2) {
            const double difference =
                vec_axis(first, coordinate) - vec_axis(second, coordinate);
            decay_derivative =
                (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
          } else {
            const double difference =
                vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
            decay_derivative = -2.0 * nu * difference;
          }
          result.center[center][coordinate] +=
              prefactor * (pair_coefficient_derivative +
                           product_scale[center] * coulomb.chain[coordinate] +
                           coulomb.value * decay_derivative);
        }
      }
    }
  }
  return result;
}

/**
 * Contract canonical (p p | s s) through the shared order-two Coulomb DAG.
 */
__device__ __noinline__ PspsWeightedGradient
contracted_eri_cartesian_source_ppss_weighted_gradient(
    const DeviceBatch& batch,
    std::size_t first_shell_pair,
    std::size_t second_shell_pair,
    std::int32_t first_p_shell,
    std::int32_t second_p_shell,
    std::int32_t third_s_shell,
    std::int32_t fourth_s_shell,
    const double (&component_weight)[9]) {
  const Vec3<double> first =
      atom_position<double>(batch, batch.shell_atoms[first_p_shell], -1);
  const Vec3<double> second =
      atom_position<double>(batch, batch.shell_atoms[second_p_shell], -1);
  const Vec3<double> third =
      atom_position<double>(batch, batch.shell_atoms[third_s_shell], -1);
  const Vec3<double> fourth =
      atom_position<double>(batch, batch.shell_atoms[fourth_s_shell], -1);

  PspsWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == first_p_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_s_shell;
  const std::int64_t first_pair_begin =
      batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end =
      batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin =
      batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end =
      batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair =
        batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const double inverse_two_p = 0.5 / p;
    const Vec3<double> product_p = first_pair.product_center;
    const Vec3<double> pa{
        product_p.x - first.x,
        product_p.y - first.y,
        product_p.z - first.z,
    };
    const Vec3<double> pb{
        product_p.x - second.x,
        product_p.y - second.y,
        product_p.z - second.z,
    };
    const double first_product_scale = first_pair_matches_canonical_order
                                           ? first_pair.first_product_scale
                                           : first_pair.second_product_scale;
    const double second_product_scale = first_pair_matches_canonical_order
                                            ? first_pair.second_product_scale
                                            : first_pair.first_product_scale;
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair =
          batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const double x = product_p.x - product_q.x;
      const double y = product_p.y - product_q.y;
      const double z = product_p.z - product_q.z;
      double boys[4];
      boys_values<3>(rho * (x * x + y * y + z * z), boys);

      const double row_x = component_weight[0] * pb.x +
                           component_weight[1] * pb.y +
                           component_weight[2] * pb.z;
      const double row_y = component_weight[3] * pb.x +
                           component_weight[4] * pb.y +
                           component_weight[5] * pb.z;
      const double row_z = component_weight[6] * pb.x +
                           component_weight[7] * pb.y +
                           component_weight[8] * pb.z;
      const double column_x = component_weight[0] * pa.x +
                              component_weight[3] * pa.y +
                              component_weight[6] * pa.z;
      const double column_y = component_weight[1] * pa.x +
                              component_weight[4] * pa.y +
                              component_weight[7] * pa.z;
      const double column_z = component_weight[2] * pa.x +
                              component_weight[5] * pa.y +
                              component_weight[8] * pa.z;
      const double h0 =
          pa.x * row_x + pa.y * row_y + pa.z * row_z +
          inverse_two_p *
              (component_weight[0] + component_weight[4] + component_weight[8]);
      const double hx = inverse_two_p * (row_x + column_x);
      const double hy = inverse_two_p * (row_y + column_y);
      const double hz = inverse_two_p * (row_z + column_z);
      const double second_scale = inverse_two_p * inverse_two_p;
      const double hxx = second_scale * component_weight[0];
      const double hxy =
          second_scale * (component_weight[1] + component_weight[3]);
      const double hxz =
          second_scale * (component_weight[2] + component_weight[6]);
      const double hyy = second_scale * component_weight[4];
      const double hyz =
          second_scale * (component_weight[5] + component_weight[7]);
      const double hzz = second_scale * component_weight[8];
      const WeightedOrder2Coulomb coulomb = contract_weighted_order2_coulomb(
          rho, x, y, z, boys, h0, hx, hy, hz, hxx, hxy, hxz, hyy, hyz, hzz);

      const double first_shift_scale = first_product_scale - 1.0;
      const double second_shift_scale = first_product_scale;
      const double explicit_first[3] = {
          (first_shift_scale * row_x + second_shift_scale * column_x) *
                  coulomb.c0 +
              inverse_two_p *
                  (first_shift_scale * (component_weight[0] * coulomb.cx +
                                        component_weight[1] * coulomb.cy +
                                        component_weight[2] * coulomb.cz) +
                   second_shift_scale * (component_weight[0] * coulomb.cx +
                                         component_weight[3] * coulomb.cy +
                                         component_weight[6] * coulomb.cz)),
          (first_shift_scale * row_y + second_shift_scale * column_y) *
                  coulomb.c0 +
              inverse_two_p *
                  (first_shift_scale * (component_weight[3] * coulomb.cx +
                                        component_weight[4] * coulomb.cy +
                                        component_weight[5] * coulomb.cz) +
                   second_shift_scale * (component_weight[1] * coulomb.cx +
                                         component_weight[4] * coulomb.cy +
                                         component_weight[7] * coulomb.cz)),
          (first_shift_scale * row_z + second_shift_scale * column_z) *
                  coulomb.c0 +
              inverse_two_p *
                  (first_shift_scale * (component_weight[6] * coulomb.cx +
                                        component_weight[7] * coulomb.cy +
                                        component_weight[8] * coulomb.cz) +
                   second_shift_scale * (component_weight[2] * coulomb.cx +
                                         component_weight[5] * coulomb.cy +
                                         component_weight[8] * coulomb.cz)),
      };
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;
      const double product_scale[3] = {
          first_product_scale, second_product_scale, -third_product_scale};
      const double prefactor = first_pair.weighted_coefficient *
                               second_pair.weighted_coefficient * 2.0 *
                               pow(kPi, 2.5) / (p * q * sqrt(p + q));

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          const double pair_coefficient_derivative =
              center == 0 ? explicit_first[coordinate]
                          : (center == 1 ? -explicit_first[coordinate] : 0.0);
          double decay_derivative = 0.0;
          if (center < 2) {
            const double difference =
                vec_axis(first, coordinate) - vec_axis(second, coordinate);
            decay_derivative =
                (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
          } else {
            const double difference =
                vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
            decay_derivative = -2.0 * nu * difference;
          }
          result.center[center][coordinate] +=
              prefactor * (pair_coefficient_derivative +
                           product_scale[center] * coulomb.chain[coordinate] +
                           coulomb.value * decay_derivative);
        }
      }
    }
  }
  return result;
}

/**
 * Contract canonical (d s | s s) in CCA xx,xy,xz,yy,yz,zz order.
 */
__device__ __noinline__ PspsWeightedGradient
contracted_eri_cartesian_source_dsss_weighted_gradient(
    const DeviceBatch& batch,
    std::size_t first_shell_pair,
    std::size_t second_shell_pair,
    std::int32_t d_shell,
    std::int32_t paired_s_shell,
    std::int32_t third_s_shell,
    std::int32_t fourth_s_shell,
    const double* component_weight) {
  const Vec3<double> first =
      atom_position<double>(batch, batch.shell_atoms[d_shell], -1);
  const Vec3<double> second =
      atom_position<double>(batch, batch.shell_atoms[paired_s_shell], -1);
  const Vec3<double> third =
      atom_position<double>(batch, batch.shell_atoms[third_s_shell], -1);
  const Vec3<double> fourth =
      atom_position<double>(batch, batch.shell_atoms[fourth_s_shell], -1);

  PspsWeightedGradient result{};
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == d_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_s_shell;
  const std::int64_t first_pair_begin =
      batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end =
      batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin =
      batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end =
      batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair =
        batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const double inverse_two_p = 0.5 / p;
    const Vec3<double> product_p = first_pair.product_center;
    const Vec3<double> pa{
        product_p.x - first.x,
        product_p.y - first.y,
        product_p.z - first.z,
    };
    const double first_product_scale = first_pair_matches_canonical_order
                                           ? first_pair.first_product_scale
                                           : first_pair.second_product_scale;
    const double second_product_scale = first_pair_matches_canonical_order
                                            ? first_pair.second_product_scale
                                            : first_pair.first_product_scale;
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair =
          batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const double rho = p * q / (p + q);
      const Vec3<double> product_q = second_pair.product_center;
      const double x = product_p.x - product_q.x;
      const double y = product_p.y - product_q.y;
      const double z = product_p.z - product_q.z;
      double boys[4];
      boys_values<3>(rho * (x * x + y * y + z * z), boys);

      const double gx = 2.0 * component_weight[0] * pa.x +
                        component_weight[1] * pa.y + component_weight[2] * pa.z;
      const double gy = component_weight[1] * pa.x +
                        2.0 * component_weight[3] * pa.y +
                        component_weight[4] * pa.z;
      const double gz = component_weight[2] * pa.x +
                        component_weight[4] * pa.y +
                        2.0 * component_weight[5] * pa.z;
      const double h0 = component_weight[0] * (pa.x * pa.x + inverse_two_p) +
                        component_weight[1] * pa.x * pa.y +
                        component_weight[2] * pa.x * pa.z +
                        component_weight[3] * (pa.y * pa.y + inverse_two_p) +
                        component_weight[4] * pa.y * pa.z +
                        component_weight[5] * (pa.z * pa.z + inverse_two_p);
      const double hx = inverse_two_p * gx;
      const double hy = inverse_two_p * gy;
      const double hz = inverse_two_p * gz;
      const double second_scale = inverse_two_p * inverse_two_p;
      const WeightedOrder2Coulomb coulomb =
          contract_weighted_order2_coulomb(rho,
                                           x,
                                           y,
                                           z,
                                           boys,
                                           h0,
                                           hx,
                                           hy,
                                           hz,
                                           second_scale * component_weight[0],
                                           second_scale * component_weight[1],
                                           second_scale * component_weight[2],
                                           second_scale * component_weight[3],
                                           second_scale * component_weight[4],
                                           second_scale * component_weight[5]);
      const double shift_scale = first_product_scale - 1.0;
      const double explicit_first[3] = {
          shift_scale *
              (gx * coulomb.c0 +
               inverse_two_p * (2.0 * component_weight[0] * coulomb.cx +
                                component_weight[1] * coulomb.cy +
                                component_weight[2] * coulomb.cz)),
          shift_scale *
              (gy * coulomb.c0 +
               inverse_two_p * (component_weight[1] * coulomb.cx +
                                2.0 * component_weight[3] * coulomb.cy +
                                component_weight[4] * coulomb.cz)),
          shift_scale *
              (gz * coulomb.c0 +
               inverse_two_p * (component_weight[2] * coulomb.cx +
                                component_weight[4] * coulomb.cy +
                                2.0 * component_weight[5] * coulomb.cz)),
      };
      const double third_product_scale = second_pair_matches_canonical_order
                                             ? second_pair.first_product_scale
                                             : second_pair.second_product_scale;
      const double product_scale[3] = {
          first_product_scale, second_product_scale, -third_product_scale};
      const double prefactor = first_pair.weighted_coefficient *
                               second_pair.weighted_coefficient * 2.0 *
                               pow(kPi, 2.5) / (p * q * sqrt(p + q));

#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
          const double pair_coefficient_derivative =
              center == 0 ? explicit_first[coordinate]
                          : (center == 1 ? -explicit_first[coordinate] : 0.0);
          double decay_derivative = 0.0;
          if (center < 2) {
            const double difference =
                vec_axis(first, coordinate) - vec_axis(second, coordinate);
            decay_derivative =
                (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
          } else {
            const double difference =
                vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
            decay_derivative = -2.0 * nu * difference;
          }
          result.center[center][coordinate] +=
              prefactor * (pair_coefficient_derivative +
                           product_scale[center] * coulomb.chain[coordinate] +
                           coulomb.value * decay_derivative);
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

/** Density-weighted ssss derivatives for the first three input centers. */
struct SsssWeightedGradient {
  double center[3][3];
};

/**
 * Contract an ssss shell quartet from the reusable primitive-pair cache.
 *
 * The generic order-zero path rebuilds both primitive pairs for every
 * primitive quartet, including product centers, Gaussian pair decay, and
 * coefficient products. Those quantities already live in PrimitivePairData
 * and are shared with direct Fock. Only the inter-pair Boys argument and the
 * derivative chain therefore remain here. The fourth center is omitted by
 * translational invariance and reconstructed by the force-task consumer.
 */
__device__ __forceinline__ SsssWeightedGradient
contracted_eri_cartesian_source_ssss_weighted_gradient(
    const DeviceBatch& batch,
    std::size_t first_shell_pair,
    std::size_t second_shell_pair,
    std::int32_t first_shell,
    std::int32_t second_shell,
    std::int32_t third_shell,
    std::int32_t fourth_shell,
    double component_weight) {
  const Vec3<double> first = atom_position<double>(
      batch, batch.shell_atoms[first_shell], -1);
  const Vec3<double> second = atom_position<double>(
      batch, batch.shell_atoms[second_shell], -1);
  const Vec3<double> third = atom_position<double>(
      batch, batch.shell_atoms[third_shell], -1);
  const Vec3<double> fourth = atom_position<double>(
      batch, batch.shell_atoms[fourth_shell], -1);
  const Vec3<double> first_difference{
      first.x - second.x,
      first.y - second.y,
      first.z - second.z,
  };
  const Vec3<double> second_difference{
      third.x - fourth.x,
      third.y - fourth.y,
      third.z - fourth.z,
  };

  SsssWeightedGradient result{};
  const std::int64_t first_pair_begin =
      batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end =
      batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin =
      batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end =
      batch.shell_pair_primitive_offsets[second_shell_pair + 1];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end; ++first_primitive) {
    const PrimitivePairData first_pair =
        batch.shell_primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end; ++second_primitive) {
      const PrimitivePairData second_pair =
          batch.shell_primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double rho = p * q / (p + q);
      const Vec3<double> product_difference{
          first_pair.product_center.x - second_pair.product_center.x,
          first_pair.product_center.y - second_pair.product_center.y,
          first_pair.product_center.z - second_pair.product_center.z,
      };
      double boys[2];
      boys_values<1>(
          rho * distance_squared(first_pair.product_center,
                                 second_pair.product_center),
          boys);
      const double prefactor =
          component_weight * first_pair.weighted_coefficient *
          second_pair.weighted_coefficient * 2.0 * pow(kPi, 2.5) /
          (p * q * sqrt(p + q));
      const double product_chain[3] = {
          -2.0 * rho * product_difference.x * boys[1],
          -2.0 * rho * product_difference.y * boys[1],
          -2.0 * rho * product_difference.z * boys[1],
      };
      const double first_decay[3] = {
          -2.0 * first_pair.reduced_exponent * first_difference.x * boys[0],
          -2.0 * first_pair.reduced_exponent * first_difference.y * boys[0],
          -2.0 * first_pair.reduced_exponent * first_difference.z * boys[0],
      };
      const double third_decay[3] = {
          -2.0 * second_pair.reduced_exponent * second_difference.x * boys[0],
          -2.0 * second_pair.reduced_exponent * second_difference.y * boys[0],
          -2.0 * second_pair.reduced_exponent * second_difference.z * boys[0],
      };

#pragma unroll
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        result.center[0][coordinate] += prefactor *
            (first_pair.first_product_scale * product_chain[coordinate] +
             first_decay[coordinate]);
        result.center[1][coordinate] += prefactor *
            (first_pair.second_product_scale * product_chain[coordinate] -
             first_decay[coordinate]);
        result.center[2][coordinate] += prefactor *
            (-second_pair.first_product_scale * product_chain[coordinate] +
             third_decay[coordinate]);
      }
    }
  }
  return result;
}

/**
 * Evaluate the independent center derivatives of an ssss or canonical psss
 * primitive.
 *
 * Coordinate differentiation changes only the Gaussian pair decay, product
 * centers, and Boys argument. Computing those shared values once is much
 * cheaper than replaying the complete primitive with one Dual3 seed per
 * independent atom. `p_axis` is ignored for the order-zero specialization.
 */
template <unsigned AngularOrder>
__device__ void primitive_eri_order01_gradient(int p_axis,
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
  const Vec3<double> product_p = product_center(alpha, first, beta, second);
  const Vec3<double> product_q = product_center(gamma, third, delta, fourth);
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

  // A simultaneous translation of all four Gaussian centers leaves the ERI
  // unchanged. Evaluate only three centers and recover the fourth exactly;
  // the force consumer already relies on this invariant across unique atoms.
  for (unsigned center = 0; center < 3; ++center) {
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
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] -
        gradient[2][coordinate];
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

/** One order-two shell-pair term and its first-center coefficient gradient. */
struct LowOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
  double first_center[3];
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
  double first_center_shift_gradients[2][3]{};
  unsigned quantum_count = 0;
  for (int axis = 0; axis < 3; ++axis) {
    for (unsigned quantum = 0;
         quantum < angular_axis(angular_first, axis); ++quantum) {
      derivative_states[quantum_count] = low_order_derivative_state(axis);
      shifts[quantum_count] =
          vec_axis(product, axis) - vec_axis(first, axis);
      first_center_shift_gradients[quantum_count][axis] =
          alpha / exponent - 1.0;
      ++quantum_count;
    }
    for (unsigned quantum = 0;
         quantum < angular_axis(angular_second, axis); ++quantum) {
      derivative_states[quantum_count] = low_order_derivative_state(axis);
      shifts[quantum_count] =
          vec_axis(product, axis) - vec_axis(second, axis);
      first_center_shift_gradients[quantum_count][axis] = alpha / exponent;
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
    for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
      expansion.terms[0].first_center[coordinate] =
          first_center_shift_gradients[0][coordinate];
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
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    expansion.terms[0].first_center[coordinate] =
        first_center_shift_gradients[0][coordinate] * shifts[1] +
        shifts[0] * first_center_shift_gradients[1][coordinate];
    expansion.terms[1].first_center[coordinate] =
        inverse_two_exponent *
        first_center_shift_gradients[1][coordinate];
    expansion.terms[2].first_center[coordinate] =
        inverse_two_exponent *
        first_center_shift_gradients[0][coordinate];
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
  const double first_product_scale = alpha / p;
  const double second_product_scale = beta / p;
  const double third_product_scale = -gamma / q;
  double value = 0.0;
  // Pair coefficients are translation invariant within each pair, so the
  // second-center coefficient derivative is the negative of the first. The
  // fourth full-center derivative is restored below from total translation.
  // Keeping only three accumulators also avoids spilling another vector in
  // this order-two kernel, whose remaining gap is dominated by local state.
  double value_gradient[3][3]{};
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
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        const double first_pair_gradient = sign *
            first_item.first_center[coordinate] * second_item.coefficient;
        const double second_pair_gradient = sign * first_item.coefficient *
            second_item.first_center[coordinate];
        // The raised Coulomb state is shared by every center; only the
        // product-center chain-rule scale differs.
        const double scaled_coulomb_derivative = coefficient *
            third_order_coulomb(
                derivative_state + low_order_derivative_state(coordinate),
                rho, product_difference, boys);
        value_gradient[0][coordinate] +=
            first_pair_gradient * coulomb +
            first_product_scale * scaled_coulomb_derivative;
        value_gradient[1][coordinate] +=
            -first_pair_gradient * coulomb +
            second_product_scale * scaled_coulomb_derivative;
        value_gradient[2][coordinate] +=
            second_pair_gradient * coulomb +
            third_product_scale * scaled_coulomb_derivative;
      }
    }
  }

  const double pair_decay = exp(
      -mu * distance_squared(first, second) -
      nu * distance_squared(third, fourth));
  const double prefactor =
      2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;
  for (unsigned center = 0; center < 3; ++center) {
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
        decay_derivative = -2.0 * nu * difference;
      }
      gradient[center][coordinate] = prefactor *
          (value_gradient[center][coordinate] + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] -
        gradient[2][coordinate];
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

/** One sparse coefficient term in an order-three differentiated pair. */
struct ThirdOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
};

template <unsigned PairOrder>
struct ThirdOrderPairGradientExpansion {
  static_assert(PairOrder <= 3);
  static constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  ThirdOrderPairGradientTerm terms[1U << PairOrder];
  unsigned axes[QuantumStorage];
  double shifts[QuantumStorage];
  double first_center_shift_gradients[QuantumStorage];
  double inverse_two_exponent;
};

/**
 * Differentiate the exact subset/Wick pair expansion through order three.
 *
 * Three-bit Cartesian derivative fields are used because differentiating an
 * order-three Coulomb state can raise one axis to order four. Exponents are
 * fixed nuclear-coordinate parameters, so only P-A/P-B shift products carry
 * coefficient derivatives; Wick factors 1/(2p) remain constant.
 */
template <unsigned PairOrder>
__device__ ThirdOrderPairGradientExpansion<PairOrder>
make_third_order_pair_gradient_expansion(
    double alpha,
    const Vec3<double>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<double>& second,
    const Angular& angular_second) {
  static_assert(PairOrder <= 3);
  ThirdOrderPairGradientExpansion<PairOrder> expansion{};
  if constexpr (PairOrder == 0) {
    expansion.terms[0].coefficient = 1.0;
  } else {
    const double exponent = alpha + beta;
    const Vec3<double> product = product_center(alpha, first, beta, second);
    expansion.inverse_two_exponent = 0.5 / exponent;
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis);
           ++quantum) {
        expansion.axes[quantum_count] = static_cast<unsigned>(axis);
        expansion.shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(first, axis);
        expansion.first_center_shift_gradients[quantum_count] =
            alpha / exponent - 1.0;
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis);
           ++quantum) {
        expansion.axes[quantum_count] = static_cast<unsigned>(axis);
        expansion.shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(second, axis);
        expansion.first_center_shift_gradients[quantum_count] =
            alpha / exponent;
        ++quantum_count;
      }
    }

    for (unsigned subset = 0; subset < (1U << PairOrder); ++subset) {
      ThirdOrderPairGradientTerm& term = expansion.terms[subset];
      term.coefficient = 1.0;
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if ((subset & (1U << quantum)) != 0) {
          term.derivative_state +=
              fourth_order_derivative_state(expansion.axes[quantum]);
          term.coefficient *= expansion.inverse_two_exponent;
        } else {
          term.coefficient *= expansion.shifts[quantum];
        }
      }
    }

    if constexpr (PairOrder == 2) {
      if (expansion.axes[0] == expansion.axes[1]) {
        expansion.terms[0].coefficient += expansion.inverse_two_exponent;
      }
    } else if constexpr (PairOrder == 3) {
      for (unsigned first_quantum = 0; first_quantum < 3; ++first_quantum) {
        for (unsigned second_quantum = first_quantum + 1; second_quantum < 3;
             ++second_quantum) {
          if (expansion.axes[first_quantum] !=
              expansion.axes[second_quantum]) {
            continue;
          }
          const unsigned remaining_quantum =
              3U - first_quantum - second_quantum;
          ThirdOrderPairGradientTerm& value_term = expansion.terms[0];
          value_term.coefficient +=
              expansion.inverse_two_exponent *
              expansion.shifts[remaining_quantum];
          expansion.terms[1U << remaining_quantum].coefficient +=
              expansion.inverse_two_exponent *
              expansion.inverse_two_exponent;
        }
      }
    }
  }
  return expansion;
}

/** Differentiate one pair coefficient with respect to its first center. */
template <unsigned PairOrder>
__device__ double third_order_pair_first_center_gradient(
    const ThirdOrderPairGradientExpansion<PairOrder>& expansion,
    unsigned subset,
    unsigned coordinate) {
  if constexpr (PairOrder == 0) {
    return 0.0;
  } else {
    double gradient = 0.0;
    for (unsigned differentiated = 0;
         differentiated < PairOrder; ++differentiated) {
      if ((subset & (1U << differentiated)) != 0 ||
          expansion.axes[differentiated] != coordinate) {
        continue;
      }
      double derivative =
          expansion.first_center_shift_gradients[differentiated];
      for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
        if (quantum == differentiated) continue;
        derivative *= (subset & (1U << quantum)) != 0
            ? expansion.inverse_two_exponent
            : expansion.shifts[quantum];
      }
      gradient += derivative;
    }
    if constexpr (PairOrder == 3) {
      if (subset == 0) {
        for (unsigned first_quantum = 0; first_quantum < 3;
             ++first_quantum) {
          for (unsigned second_quantum = first_quantum + 1;
               second_quantum < 3; ++second_quantum) {
            if (expansion.axes[first_quantum] !=
                expansion.axes[second_quantum]) {
              continue;
            }
            const unsigned remaining_quantum =
                3U - first_quantum - second_quantum;
            if (expansion.axes[remaining_quantum] == coordinate) {
              gradient += expansion.inverse_two_exponent *
                  expansion.first_center_shift_gradients[remaining_quantum];
            }
          }
        }
      }
    }
    return gradient;
  }
}

/** Evaluate all-center derivatives of one canonical order-three primitive. */
template <unsigned FirstPairOrder, unsigned SecondPairOrder>
__device__ void primitive_eri_order3_gradient(
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
  static_assert(FirstPairOrder + SecondPairOrder == 3);
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
  const ThirdOrderPairGradientExpansion<FirstPairOrder> first_expansion =
      make_third_order_pair_gradient_expansion<FirstPairOrder>(
          alpha, first, angular_first, beta, second, angular_second);
  const ThirdOrderPairGradientExpansion<SecondPairOrder> second_expansion =
      make_third_order_pair_gradient_expansion<SecondPairOrder>(
          gamma, third, angular_third, delta, fourth, angular_fourth);
  double boys[5];
  boys_values<4>(rho * distance_squared(product_p, product_q), boys);
  const double first_product_scale = alpha / p;
  const double second_product_scale = beta / p;
  const double third_product_scale = -gamma / q;
  double value = 0.0;
  // The fourth center is restored from translational invariance. Besides
  // removing one quarter of the center updates, this keeps the hot primitive
  // path from spilling another three-component accumulator to local memory.
  double value_gradient[3][3]{};
  for (unsigned first_term = 0; first_term < (1U << FirstPairOrder);
       ++first_term) {
    for (unsigned second_term = 0; second_term < (1U << SecondPairOrder);
         ++second_term) {
      const ThirdOrderPairGradientTerm& first_item =
          first_expansion.terms[first_term];
      const ThirdOrderPairGradientTerm& second_item =
          second_expansion.terms[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U)
              == 0
          ? 1.0
          : -1.0;
      const unsigned derivative_state =
          first_item.derivative_state + second_item.derivative_state;
      const double coulomb = fourth_order_coulomb(
          derivative_state, rho, product_difference, boys);
      const double coefficient =
          sign * first_item.coefficient * second_item.coefficient;
      value += coefficient * coulomb;
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        const double first_pair_gradient =
            third_order_pair_first_center_gradient(
                first_expansion, first_term, coordinate);
        const double second_pair_gradient =
            third_order_pair_first_center_gradient(
                second_expansion, second_term, coordinate);
        const double first_coefficient_gradient =
            sign * first_pair_gradient * second_item.coefficient;
        const double second_coefficient_gradient =
            sign * first_item.coefficient * second_pair_gradient;
        // The Cartesian Coulomb derivative is center independent; only the
        // product-center chain-rule scale changes. Compute it once instead of
        // repeating the fourth-order closed form for all four centers.
        const double scaled_coulomb_derivative = coefficient *
            fourth_order_coulomb(
                derivative_state +
                    fourth_order_derivative_state(coordinate),
                rho, product_difference, boys);
        value_gradient[0][coordinate] +=
            first_coefficient_gradient * coulomb +
            first_product_scale * scaled_coulomb_derivative;
        value_gradient[1][coordinate] +=
            -first_coefficient_gradient * coulomb +
            second_product_scale * scaled_coulomb_derivative;
        value_gradient[2][coordinate] +=
            second_coefficient_gradient * coulomb +
            third_product_scale * scaled_coulomb_derivative;
      }
    }
  }

  const double pair_decay = exp(
      -mu * distance_squared(first, second) -
      nu * distance_squared(third, fourth));
  const double prefactor =
      2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;
  for (unsigned center = 0; center < 3; ++center) {
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
        decay_derivative = -2.0 * nu * difference;
      }
      gradient[center][coordinate] = prefactor *
          (value_gradient[center][coordinate] + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] -
        gradient[2][coordinate];
  }
}

/** Canonicalize and contract all-center gradients for total angular order 3. */
__device__ CartesianQuartetGradient
contracted_eri_cartesian_source_order3_gradient(
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
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] +
      batch.shell_angular[slots[1].shell];
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
          if (first_pair_order == 3) {
            primitive_eri_order3_gradient<3, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          } else {
            primitive_eri_order3_gradient<2, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          }
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

/** One sparse coefficient term and its first-center gradient. */
struct HighOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
  double first_center[3];
};

template <unsigned PairOrder>
struct HighOrderPairGradientGeometry {
  static_assert(PairOrder <= 6);
  static constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  unsigned axes[QuantumStorage];
  double shifts[QuantumStorage];
  double first_center_shift_gradients[QuantumStorage];
  double inverse_two_exponent;
};

/** Add one Wick matching to one derivative subset and its gradient. */
template <unsigned PairOrder>
__device__ void add_high_order_wick_matching_term(
    HighOrderPairGradientTerm& term,
    const HighOrderPairGradientGeometry<PairOrder>& geometry,
    unsigned subset,
    unsigned removed,
    unsigned contraction_count) {
  static_assert(PairOrder <= 6);
  if ((subset & removed) != 0) return;
  double inverse_factor = 1.0;
  const unsigned inverse_count =
      contraction_count + static_cast<unsigned>(__popc(subset));
  for (unsigned factor = 0; factor < inverse_count; ++factor) {
    inverse_factor *= geometry.inverse_two_exponent;
  }

  double coefficient = inverse_factor;
  for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
    const unsigned bit = 1U << quantum;
    if (((subset | removed) & bit) == 0) {
      coefficient *= geometry.shifts[quantum];
    }
  }
  term.coefficient += coefficient;

  // Differentiate the same surviving product while its factors are hot.
  // The old path revisited every matching once for coefficients and three
  // more times for Cartesian gradients. Accumulating by the differentiated
  // quantum avoids that repeated combinatorial walk and keeps the gradient
  // sparse in its quantum's Cartesian axis.
  for (unsigned differentiated = 0; differentiated < PairOrder;
       ++differentiated) {
    const unsigned differentiated_bit = 1U << differentiated;
    if (((subset | removed) & differentiated_bit) != 0) continue;
    double derivative = inverse_factor *
        geometry.first_center_shift_gradients[differentiated];
    for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
      const unsigned bit = 1U << quantum;
      if (quantum == differentiated ||
          ((subset | removed) & bit) != 0) {
        continue;
      }
      derivative *= geometry.shifts[quantum];
    }
    term.first_center[geometry.axes[differentiated]] += derivative;
  }
}

/**
 * Generate the exact subset/Wick pair expansion through angular order six.
 *
 * One contraction removes a same-axis quantum pair. Three disjoint
 * contractions are the highest possible matching at order six. Expanding
 * every surviving quantum into either its center shift or Hermite derivative
 * covers the complete Gaussian product recurrence without a dense
 * coefficient workspace. Pair masks are ordered to visit every disjoint Wick
 * matching exactly once.
 */
template <unsigned PairOrder>
__device__ HighOrderPairGradientGeometry<PairOrder>
make_high_order_pair_gradient_geometry(
    double alpha,
    const Vec3<double>& first,
    const Angular& angular_first,
    double beta,
    const Vec3<double>& second,
    const Angular& angular_second) {
  static_assert(PairOrder <= 6);
  HighOrderPairGradientGeometry<PairOrder> geometry{};
  if constexpr (PairOrder != 0) {
    const double exponent = alpha + beta;
    const Vec3<double> product = product_center(alpha, first, beta, second);
    geometry.inverse_two_exponent = 0.5 / exponent;
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis);
           ++quantum) {
        geometry.axes[quantum_count] = static_cast<unsigned>(axis);
        geometry.shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(first, axis);
        geometry.first_center_shift_gradients[quantum_count] =
            alpha / exponent - 1.0;
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis);
           ++quantum) {
        geometry.axes[quantum_count] = static_cast<unsigned>(axis);
        geometry.shifts[quantum_count] =
            vec_axis(product, axis) - vec_axis(second, axis);
        geometry.first_center_shift_gradients[quantum_count] =
            alpha / exponent;
        ++quantum_count;
      }
    }
  }
  return geometry;
}

/** Generate one exact subset/Wick coefficient and its center gradient. */
template <unsigned PairOrder>
__device__ HighOrderPairGradientTerm make_high_order_pair_gradient_term(
    const HighOrderPairGradientGeometry<PairOrder>& geometry,
    unsigned subset) {
  HighOrderPairGradientTerm term{};
  if constexpr (PairOrder == 0) {
    // The scalar pair has one unit term and no center derivative. Keeping it
    // out of the combinatorial loops also avoids zero-trip unsigned-loop
    // diagnostics in CUDA's template instantiation.
    term.coefficient = 1.0;
  } else {
    for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
      if ((subset & (1U << quantum)) != 0) {
        term.derivative_state +=
            fourth_order_derivative_state(geometry.axes[quantum]);
      }
    }
    add_high_order_wick_matching_term(
        term, geometry, subset, 0U, 0U);
    for (unsigned first_quantum = 0; first_quantum < PairOrder;
         ++first_quantum) {
      for (unsigned second_quantum = first_quantum + 1;
           second_quantum < PairOrder; ++second_quantum) {
        if (geometry.axes[first_quantum] != geometry.axes[second_quantum]) {
          continue;
        }
        const unsigned first_pair =
            (1U << first_quantum) | (1U << second_quantum);
        add_high_order_wick_matching_term(
            term, geometry, subset, first_pair, 1U);
        for (unsigned third_quantum = 0; third_quantum < PairOrder;
             ++third_quantum) {
          for (unsigned fourth_quantum = third_quantum + 1;
               fourth_quantum < PairOrder; ++fourth_quantum) {
            const unsigned second_pair =
                (1U << third_quantum) | (1U << fourth_quantum);
            if ((first_pair & second_pair) != 0 ||
                first_pair >= second_pair ||
                geometry.axes[third_quantum] !=
                    geometry.axes[fourth_quantum]) {
              continue;
            }
            add_high_order_wick_matching_term(
                term, geometry, subset, first_pair | second_pair, 2U);
            if constexpr (PairOrder == 6) {
              for (unsigned fifth_quantum = 0;
                   fifth_quantum < PairOrder; ++fifth_quantum) {
                for (unsigned sixth_quantum = fifth_quantum + 1;
                     sixth_quantum < PairOrder; ++sixth_quantum) {
                  const unsigned third_pair =
                      (1U << fifth_quantum) | (1U << sixth_quantum);
                  if (((first_pair | second_pair) & third_pair) != 0 ||
                      second_pair >= third_pair ||
                      geometry.axes[fifth_quantum] !=
                          geometry.axes[sixth_quantum]) {
                    continue;
                  }
                  add_high_order_wick_matching_term(
                      term, geometry, subset,
                      first_pair | second_pair | third_pair, 3U);
                }
              }
            }
          }
        }
      }
    }
  }
  return term;
}

/** Primitive-local powers reused by one bounded high-order Coulomb recurrence. */
template <unsigned MaximumOrder>
struct HighOrderCoulombWorkspace {
  static_assert(MaximumOrder >= 5 && MaximumOrder <= 7);
  Vec3<double> difference;
  double coordinate_powers[3][MaximumOrder + 1];
  double negative_two_rho_powers[MaximumOrder + 1];
};

template <unsigned MaximumOrder>
__device__ HighOrderCoulombWorkspace<MaximumOrder>
make_high_order_coulomb_workspace(
    double rho,
    const Vec3<double>& difference) {
  HighOrderCoulombWorkspace<MaximumOrder> workspace{};
  workspace.difference = difference;
  for (unsigned axis = 0; axis < 3; ++axis) {
    workspace.coordinate_powers[axis][0] = 1.0;
    for (unsigned power = 1; power <= MaximumOrder; ++power) {
      workspace.coordinate_powers[axis][power] =
          workspace.coordinate_powers[axis][power - 1] *
          vec_axis(difference, static_cast<int>(axis));
    }
  }
  workspace.negative_two_rho_powers[0] = 1.0;
  for (unsigned power = 1; power <= MaximumOrder; ++power) {
    workspace.negative_two_rho_powers[power] =
        workspace.negative_two_rho_powers[power - 1] * (-2.0 * rho);
  }
  return workspace;
}

/** Number of ways to form `pairs` disjoint contractions from one axis. */
__device__ unsigned axis_wick_multiplicity(unsigned order, unsigned pairs) {
  if (pairs == 0) return 1U;
  if (pairs == 1) return order * (order - 1U) / 2U;
  if (pairs == 2) {
    return order * (order - 1U) * (order - 2U) * (order - 3U) / 8U;
  }
  return order * (order - 1U) * (order - 2U) * (order - 3U) *
      (order - 4U) * (order - 5U) / 48U;
}

/** Evaluate one Cartesian Coulomb derivative through `MaximumOrder`. */
template <unsigned MaximumOrder>
__device__ double high_order_coulomb(
    unsigned derivative_state,
    double rho,
    const HighOrderCoulombWorkspace<MaximumOrder>& workspace,
    const double* boys) {
  const unsigned x_order = derivative_state & 7U;
  const unsigned y_order = (derivative_state >> 3U) & 7U;
  const unsigned z_order = (derivative_state >> 6U) & 7U;
  const unsigned total_order = x_order + y_order + z_order;
  if (total_order < 5) {
    return fourth_order_coulomb(
        derivative_state, rho, workspace.difference, boys);
  }

  double value = 0.0;
  for (unsigned x_pairs = 0; x_pairs <= x_order / 2U; ++x_pairs) {
    for (unsigned y_pairs = 0; y_pairs <= y_order / 2U; ++y_pairs) {
      for (unsigned z_pairs = 0; z_pairs <= z_order / 2U; ++z_pairs) {
        const unsigned contraction_count = x_pairs + y_pairs + z_pairs;
        const unsigned boys_order = total_order - contraction_count;
        const unsigned multiplicity =
            axis_wick_multiplicity(x_order, x_pairs) *
            axis_wick_multiplicity(y_order, y_pairs) *
            axis_wick_multiplicity(z_order, z_pairs);
        value += static_cast<double>(multiplicity) *
            workspace.negative_two_rho_powers[boys_order] *
            workspace.coordinate_powers[0][x_order - 2U * x_pairs] *
            workspace.coordinate_powers[1][y_order - 2U * y_pairs] *
            workspace.coordinate_powers[2][z_order - 2U * z_pairs] *
            boys[boys_order];
      }
    }
  }
  return value;
}

/** Evaluate all-center derivatives of one canonical order-four to-six primitive. */
template <unsigned FirstPairOrder, unsigned SecondPairOrder>
__device__ void primitive_eri_order456_gradient(
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
  constexpr unsigned AngularOrder = FirstPairOrder + SecondPairOrder;
  constexpr unsigned CoulombOrder = AngularOrder + 1;
  static_assert(
      AngularOrder == 4 || AngularOrder == 5 || AngularOrder == 6);
  static_assert(FirstPairOrder >= SecondPairOrder);
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
  const HighOrderPairGradientGeometry<FirstPairOrder> first_geometry =
      make_high_order_pair_gradient_geometry<FirstPairOrder>(
          alpha, first, angular_first, beta, second, angular_second);
  const HighOrderPairGradientGeometry<SecondPairOrder> second_geometry =
      make_high_order_pair_gradient_geometry<SecondPairOrder>(
          gamma, third, angular_third, delta, fourth, angular_fourth);
  double boys[AngularOrder + 2];
  boys_values<AngularOrder + 1>(
      rho * distance_squared(product_p, product_q), boys);
  const HighOrderCoulombWorkspace<CoulombOrder> coulomb_workspace =
      make_high_order_coulomb_workspace<CoulombOrder>(
          rho, product_difference);
  const double first_product_scale = alpha / p;
  const double second_product_scale = beta / p;
  const double third_product_scale = -gamma / q;
  double value = 0.0;
  double value_gradient[3][3]{};
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;
  // Canonical pair ordering keeps the second expansion small (at most eight
  // terms through total order six). Materialize those terms once; generate
  // each larger first-pair term immediately before consuming it so its
  // coefficient and gradient do not create another full local array.
  HighOrderPairGradientTerm second_items[SecondTermCount];
  for (unsigned second_term = 0; second_term < SecondTermCount;
       ++second_term) {
    second_items[second_term] = make_high_order_pair_gradient_term(
        second_geometry, second_term);
  }
  for (unsigned first_term = 0; first_term < FirstTermCount;
       ++first_term) {
    const HighOrderPairGradientTerm first_item =
        make_high_order_pair_gradient_term(first_geometry, first_term);
    for (unsigned second_term = 0; second_term < SecondTermCount;
         ++second_term) {
      const HighOrderPairGradientTerm& second_item =
          second_items[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U)
              == 0
          ? 1.0
          : -1.0;
      const unsigned derivative_state =
          first_item.derivative_state + second_item.derivative_state;
      const double coulomb = high_order_coulomb<CoulombOrder>(
          derivative_state, rho, coulomb_workspace, boys);
      const double coefficient =
          sign * first_item.coefficient * second_item.coefficient;
      value += coefficient * coulomb;
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        const double first_coefficient_gradient =
            sign * first_item.first_center[coordinate] *
            second_item.coefficient;
        const double second_coefficient_gradient =
            sign * first_item.coefficient *
            second_item.first_center[coordinate];
        const double scaled_coulomb_derivative = coefficient *
            high_order_coulomb<CoulombOrder>(
                derivative_state +
                    fourth_order_derivative_state(coordinate),
                rho, coulomb_workspace, boys);
        value_gradient[0][coordinate] +=
            first_coefficient_gradient * coulomb +
            first_product_scale * scaled_coulomb_derivative;
        value_gradient[1][coordinate] +=
            -first_coefficient_gradient * coulomb +
            second_product_scale * scaled_coulomb_derivative;
        value_gradient[2][coordinate] +=
            second_coefficient_gradient * coulomb +
            third_product_scale * scaled_coulomb_derivative;
      }
    }
  }

  const double pair_decay = exp(
      -mu * distance_squared(first, second) -
      nu * distance_squared(third, fourth));
  const double prefactor =
      2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;
  for (unsigned center = 0; center < 3; ++center) {
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
        decay_derivative = -2.0 * nu * difference;
      }
      gradient[center][coordinate] = prefactor *
          (value_gradient[center][coordinate] + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] -
        gradient[2][coordinate];
  }
}

/** Canonicalize and contract all-center gradients for total angular order 4. */
__device__ CartesianQuartetGradient
contracted_eri_cartesian_source_order4_gradient(
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
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] +
      batch.shell_angular[slots[1].shell];
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
          if (first_pair_order == 4) {
            primitive_eri_order456_gradient<4, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          } else if (first_pair_order == 3) {
            primitive_eri_order456_gradient<3, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          } else {
            primitive_eri_order456_gradient<2, 2>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          }
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

/** Canonicalize and contract all-center gradients for total angular order 5. */
__device__ CartesianQuartetGradient
contracted_eri_cartesian_source_order5_gradient(
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
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] +
      batch.shell_angular[slots[1].shell];
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
          if (first_pair_order == 5) {
            primitive_eri_order456_gradient<5, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          } else if (first_pair_order == 4) {
            primitive_eri_order456_gradient<4, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          } else {
            primitive_eri_order456_gradient<3, 2>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          }
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

/** Canonicalize and contract all-center gradients for total angular order 6. */
__device__ CartesianQuartetGradient
contracted_eri_cartesian_source_order6_gradient(
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
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] +
      batch.shell_angular[slots[1].shell];
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
          if (first_pair_order == 6) {
            primitive_eri_order456_gradient<6, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          } else if (first_pair_order == 5) {
            primitive_eri_order456_gradient<5, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          } else if (first_pair_order == 4) {
            primitive_eri_order456_gradient<4, 2>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          } else {
            primitive_eri_order456_gradient<3, 3>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3],
                primitive_gradient);
          }
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
#define VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(ShellClass)                  \
  case ShellClass:                                                     \
    if constexpr (direct_shell_class_angular_order(ShellClass) ==      \
                  AngularOrder) {                                     \
      return contracted_eri_cartesian_source_shell_class<              \
          ShellClass, Scalar>(batch, system, i, j, k, l,               \
                              derivative_coordinate);                  \
    }                                                                 \
    break
  switch (runtime_shell_class) {
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(0);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(1);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(2);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(3);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(4);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(5);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(6);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(7);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(8);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(9);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(10);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(11);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(12);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(13);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(14);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(15);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(16);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(17);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(18);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(19);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(20);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(21);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(22);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(23);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(24);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(25);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(26);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(27);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(28);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(29);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(30);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(31);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(32);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(33);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(34);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(35);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(36);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(37);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(38);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(39);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(40);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(41);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(42);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(43);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(44);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(45);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(46);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(47);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(48);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(49);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(50);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(51);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(52);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(53);
    VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE(54);
  }
#undef VIBEQC_DIRECT_SOURCE_SHELL_CLASS_CASE
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
#define VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(Order)                           \
  case Order:                                                          \
    return dispatch_contracted_eri_cartesian_source_shell_class<       \
        Order, Scalar>(shell_class, batch, system, i, j, k, l,         \
                       derivative_coordinate)
  switch (angular_order) {
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(0);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(1);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(2);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(3);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(4);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(5);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(6);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(7);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(8);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(9);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(10);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(11);
    VIBEQC_DIRECT_SOURCE_ANGULAR_CASE(12);
  }
#undef VIBEQC_DIRECT_SOURCE_ANGULAR_CASE
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

/**
 * Evaluate the two DF targets directly from the shared Cartesian quartet
 * recurrence.  A zero-exponent s function is used as the fourth center: with
 * delta=0, `(mu nu|P)` and `(P|Q)` are exactly the three-/two-center Coulomb
 * integrals used by the host oracle, while no O(N^4) ERI tensor is retained.
 * The combined batch contains orbital shells, auxiliary shells, and one such
 * dummy shell, so all geometry and normalization data follow the production
 * DeviceBatch layout without a second recurrence implementation.
 */
template <bool Derivative>
__global__ void build_cuda_df_integrals_kernel(
    DeviceBatch batch, std::size_t orbital_count, std::size_t auxiliary_count,
    std::size_t dummy_index, std::size_t metric_elements,
    std::size_t three_center_elements, std::size_t system_base,
    std::size_t launch_batch_size, std::int64_t derivative_coordinate,
    double* metric, double* three_center) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) *
      blockDim.x + threadIdx.x;
  const std::size_t per_system = metric_elements + three_center_elements;
  const std::size_t total = launch_batch_size * per_system;
  if (element >= total) return;
  const std::size_t local_system = element / per_system;
  const std::size_t system = system_base + local_system;
  const std::size_t system_local = element % per_system;
  const std::int64_t system_derivative_coordinate =
      derivative_coordinate < 0
          ? derivative_coordinate
          : derivative_coordinate + batch.atom_offsets[system] * 3;

  if (system_local < metric_elements) {
    const std::size_t first_aux = system_local / auxiliary_count;
    const std::size_t second_aux = system_local % auxiliary_count;
    const auto value = contracted_eri<
        std::conditional_t<Derivative, Dual, double>>(
        batch, static_cast<std::int32_t>(system),
        static_cast<std::int32_t>(orbital_count + first_aux),
        static_cast<std::int32_t>(dummy_index),
        static_cast<std::int32_t>(orbital_count + second_aux),
        static_cast<std::int32_t>(dummy_index),
        system_derivative_coordinate);
    if constexpr (Derivative) {
      metric[local_system * metric_elements + system_local] =
          value.derivative;
    } else {
      metric[local_system * metric_elements + system_local] = value;
    }
    return;
  }

  const std::size_t local = system_local - metric_elements;
  const std::size_t orbital_pair = local / auxiliary_count;
  const std::size_t auxiliary = local % auxiliary_count;
  const std::size_t first_orbital = orbital_pair / orbital_count;
  const std::size_t second_orbital = orbital_pair % orbital_count;
  const auto value = contracted_eri<
      std::conditional_t<Derivative, Dual, double>>(
      batch, static_cast<std::int32_t>(system),
      static_cast<std::int32_t>(first_orbital),
      static_cast<std::int32_t>(second_orbital),
      static_cast<std::int32_t>(orbital_count + auxiliary),
      static_cast<std::int32_t>(dummy_index),
      system_derivative_coordinate);
  if constexpr (Derivative) {
    three_center[local_system * three_center_elements + local] =
        value.derivative;
  } else {
    three_center[local_system * three_center_elements + local] = value;
  }
}

/** Evaluate one-electron matrices and their first-coordinate response. */
template <bool Derivative>
__global__ void build_cuda_one_electron_integrals_kernel(
    DeviceBatch batch, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count,
    std::int64_t derivative_coordinate, double* overlap, double* hcore) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) *
      blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t pair = element % pair_count;
  const std::size_t row = static_cast<std::size_t>(pair_first[pair]);
  const std::size_t column = static_cast<std::size_t>(pair_second[pair]);
  const std::int64_t system_derivative_coordinate =
      derivative_coordinate < 0
          ? derivative_coordinate
          : derivative_coordinate + batch.atom_offsets[system] * 3;
  if constexpr (Derivative) {
    const Dual overlap_value = contracted_overlap<Dual>(
        batch, system, static_cast<std::int32_t>(row),
        static_cast<std::int32_t>(column), system_derivative_coordinate);
    const Dual hcore_value = contracted_hcore<Dual>(
        batch, system, static_cast<std::int32_t>(row),
        static_cast<std::int32_t>(column), system_derivative_coordinate);
    const std::size_t matrix_offset = static_cast<std::size_t>(system) *
        n * n;
    overlap[matrix_offset + row * n + column] = overlap_value.derivative;
    hcore[matrix_offset + row * n + column] = hcore_value.derivative;
    if (row != column) {
      overlap[matrix_offset + column * n + row] = overlap_value.derivative;
      hcore[matrix_offset + column * n + row] = hcore_value.derivative;
    }
  } else {
    const double overlap_value = contracted_overlap<double>(
        batch, system, static_cast<std::int32_t>(row),
        static_cast<std::int32_t>(column), -1);
    const double hcore_value = contracted_hcore<double>(
        batch, system, static_cast<std::int32_t>(row),
        static_cast<std::int32_t>(column), -1);
    const std::size_t matrix_offset = static_cast<std::size_t>(system) *
        n * n;
    overlap[matrix_offset + row * n + column] = overlap_value;
    hcore[matrix_offset + row * n + column] = hcore_value;
    if (row != column) {
      overlap[matrix_offset + column * n + row] = overlap_value;
      hcore[matrix_offset + column * n + row] = hcore_value;
    }
  }
}

template <bool Derivative>
__global__ void build_cuda_nuclear_repulsion_kernel(
    DeviceBatch batch, std::int64_t derivative_coordinate,
    double* nuclear_repulsion) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch.batch_size) return;
  const std::int64_t system_derivative_coordinate =
      derivative_coordinate < 0
          ? derivative_coordinate
          : derivative_coordinate + batch.atom_offsets[system] * 3;
  if constexpr (Derivative) {
    Dual result{0.0, 0.0};
    for (std::int64_t first = batch.atom_offsets[system];
         first < batch.atom_offsets[system + 1]; ++first) {
      const Vec3<Dual> a = atom_position<Dual>(
          batch, first, system_derivative_coordinate);
      for (std::int64_t second = batch.atom_offsets[system];
           second < first; ++second) {
        const Vec3<Dual> b = atom_position<Dual>(
            batch, second, system_derivative_coordinate);
        result = result + static_cast<double>(
            batch.atomic_numbers[first] * batch.atomic_numbers[second]) /
            qsqrt(distance_squared(a, b));
      }
    }
    nuclear_repulsion[system] = result.derivative;
  } else {
    double result = 0.0;
    for (std::int64_t first = batch.atom_offsets[system];
         first < batch.atom_offsets[system + 1]; ++first) {
      const Vec3<double> a = atom_position<double>(batch, first, -1);
      for (std::int64_t second = batch.atom_offsets[system];
           second < first; ++second) {
        const Vec3<double> b = atom_position<double>(batch, second, -1);
        result += static_cast<double>(
            batch.atomic_numbers[first] * batch.atomic_numbers[second]) /
            sqrt(distance_squared(a, b));
      }
    }
    nuclear_repulsion[system] = result;
  }
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
                                        bool reuse_previous_energy,
                                        const double* energy,
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
  previous_energy[system] =
      reuse_previous_energy ? energy[system] : CUDART_INF;
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

/** Copy complete per-system matrices selected by a device-resident mask. */
__global__ void copy_selected_matrices_kernel(
    std::int32_t batch_size,
    std::int32_t matrices_per_system,
    std::int32_t nbf,
    const std::uint8_t* selected,
    const double* source,
    double* destination) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t matrix_count =
      static_cast<std::size_t>(batch_size) * matrices_per_system;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_count * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system =
      state / static_cast<std::size_t>(matrices_per_system);
  if (selected[system] != 0) destination[element] = source[element];
}

/** Compact device record retained only by the opt-in profiling Graph. */
struct DeviceInactiveEigensolverProfileEntry {
  std::uint64_t solver_start_nanoseconds;
  std::uint64_t solver_elapsed_nanoseconds;
  std::uint32_t iteration;
  std::uint32_t family;
  std::uint32_t physical_system_count;
  std::uint32_t solver_batch_count;
  std::uint32_t active_physical_count;
  std::uint32_t active_solver_count;
  std::uint32_t inactive_input_nonfinite_count;
  std::uint32_t inactive_submission_nonfinite_count;
  std::uint32_t inactive_info_nonzero_count;
  std::uint32_t inactive_touch_flags;
  std::uint32_t provider_invoked;
};

__device__ __forceinline__ std::uint64_t globaltimer_nanoseconds() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

/** Record a stream-ordered timestamp immediately before one Fock class. */
__global__ void start_bounded_fock_class_timer_kernel(
    unsigned shell_class,
    std::uint64_t* starts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  starts[shell_class] = globaltimer_nanoseconds();
}

/** Accumulate exact stream time consumed by one bounded Fock class launch. */
__global__ void finish_bounded_fock_class_timer_kernel(
    unsigned shell_class,
    const std::uint64_t* starts,
    std::uint64_t* elapsed,
    std::uint32_t* launches) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const std::uint64_t stop = globaltimer_nanoseconds();
  elapsed[shell_class] += stop - starts[shell_class];
  ++launches[shell_class];
}

/** Allocate and initialize the record owned by this sequential Graph replay. */
__global__ void begin_inactive_eigensolver_profile_kernel(
    std::int32_t physical_batch_size,
    std::int32_t solver_batch_size,
    std::uint32_t family,
    bool provider_invoked,
    bool cublas_transformed_inactive,
    const std::uint8_t* physical_active,
    const std::uint8_t* solver_active,
    std::uint32_t capacity,
    std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const std::uint32_t index = atomicAdd(count, 1U);
  if (index >= capacity) return;
  std::uint32_t active_physical_count = 0;
  std::uint32_t active_solver_count = 0;
  for (std::int32_t system = 0; system < physical_batch_size; ++system) {
    active_physical_count += physical_active[system] != 0 ? 1U : 0U;
  }
  for (std::int32_t state = 0; state < solver_batch_size; ++state) {
    active_solver_count += solver_active[state] != 0 ? 1U : 0U;
  }
  const bool has_inactive =
      active_solver_count < static_cast<std::uint32_t>(solver_batch_size);
  DeviceInactiveEigensolverProfileEntry& entry = entries[index];
  entry.solver_start_nanoseconds = 0U;
  entry.solver_elapsed_nanoseconds = 0U;
  entry.iteration = index + 1U;
  entry.family = family;
  entry.physical_system_count =
      static_cast<std::uint32_t>(physical_batch_size);
  entry.solver_batch_count = static_cast<std::uint32_t>(solver_batch_size);
  entry.active_physical_count = active_physical_count;
  entry.active_solver_count = active_solver_count;
  entry.inactive_input_nonfinite_count = 0U;
  entry.inactive_submission_nonfinite_count = 0U;
  entry.inactive_info_nonzero_count = 0U;
  entry.inactive_touch_flags =
      has_inactive && cublas_transformed_inactive
      ? VIBEQC_EIGENSOLVER_INACTIVE_TOUCH_CUBLAS_TRANSFORM
      : 0U;
  entry.provider_invoked = provider_invoked ? 1U : 0U;
}

/**
 * Replace every inactive provider input with an identity matrix.
 *
 * cuSOLVER providers cannot consume the active mask. Identity substitution
 * guarantees finite, well-conditioned input without changing the fixed batch
 * size. The optional diagnostic counts non-finite values before replacement;
 * it is not part of the production fast path when profiling is disabled.
 */
__global__ void sanitize_inactive_solver_input_kernel(
    std::int32_t solver_batch_size,
    std::int32_t nbf,
    const std::uint8_t* solver_active,
    double* matrices,
    int* info,
    std::uint32_t profile_capacity,
    const std::uint32_t* profile_count,
    DeviceInactiveEigensolverProfileEntry* profile_entries) {
  const std::int32_t state = static_cast<std::int32_t>(blockIdx.x);
  if (state >= solver_batch_size) return;
  if (threadIdx.x == 0) info[state] = 0;
  if (solver_active[state] != 0) return;
  __shared__ unsigned matrix_nonfinite;
  __shared__ unsigned submission_nonfinite;
  if (threadIdx.x == 0) {
    matrix_nonfinite = 0U;
    submission_nonfinite = 0U;
  }
  __syncthreads();
  DeviceInactiveEigensolverProfileEntry* profile = nullptr;
  if (profile_entries != nullptr && profile_count != nullptr &&
      *profile_count != 0U && *profile_count <= profile_capacity) {
    profile = profile_entries + (*profile_count - 1U);
    if (threadIdx.x == 0) {
      atomicOr(&profile->inactive_touch_flags,
               VIBEQC_EIGENSOLVER_INACTIVE_TOUCH_IDENTITY_SANITIZE);
    }
  }
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t offset = static_cast<std::size_t>(state) * matrix_size;
  for (std::size_t element = threadIdx.x; element < matrix_size;
       element += blockDim.x) {
    const double input = matrices[offset + element];
    if (profile != nullptr && !isfinite(input)) {
      atomicExch(&matrix_nonfinite, 1U);
    }
    const std::size_t row = element % n;
    const std::size_t column = element / n;
    matrices[offset + element] = row == column ? 1.0 : 0.0;
  }
  __syncthreads();
  if (profile != nullptr && threadIdx.x == 0 && matrix_nonfinite != 0U) {
    atomicAdd(&profile->inactive_input_nonfinite_count, 1U);
  }
  if (profile != nullptr) {
    for (std::size_t element = threadIdx.x; element < matrix_size;
         element += blockDim.x) {
      if (!isfinite(matrices[offset + element])) {
        atomicExch(&submission_nonfinite, 1U);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && submission_nonfinite != 0U) {
      atomicAdd(&profile->inactive_submission_nonfinite_count, 1U);
    }
  }
}

__global__ void start_inactive_eigensolver_timer_kernel(
    std::uint32_t capacity,
    const std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || *count == 0U ||
      *count > capacity) {
    return;
  }
  entries[*count - 1U].solver_start_nanoseconds = globaltimer_nanoseconds();
}

__global__ void finish_inactive_eigensolver_profile_kernel(
    std::int32_t solver_batch_size,
    const std::uint8_t* solver_active,
    const int* info,
    std::uint32_t capacity,
    const std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || *count == 0U ||
      *count > capacity) {
    return;
  }
  DeviceInactiveEigensolverProfileEntry& entry = entries[*count - 1U];
  const std::uint64_t stop = globaltimer_nanoseconds();
  entry.solver_elapsed_nanoseconds = stop - entry.solver_start_nanoseconds;
  std::uint32_t inactive_info_nonzero_count = 0U;
  for (std::int32_t state = 0; state < solver_batch_size; ++state) {
    if (solver_active[state] == 0 && info[state] != 0) {
      ++inactive_info_nonzero_count;
    }
  }
  entry.inactive_info_nonzero_count = inactive_info_nonzero_count;
}

__global__ void inspect_solver_kernel(std::int32_t batch_size,
                                      const int* info,
                                      std::uint8_t* active,
                                      std::uint8_t* failed,
                                      std::uint8_t* converged) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  // An inactive state has already converged or failed. Provider writes to its
  // fixed-batch info slot must never overwrite that terminal status.
  if (system >= batch_size || active[system] == 0 || info[system] == 0) return;
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
  if (system >= batch_size || active[system] == 0) return;
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

template <unsigned BlockThreads>
__device__ double warm_density_block_sum(double value, double* warp_sums) {
  static_assert(BlockThreads % 32 == 0);
  constexpr unsigned kWarpWidth = 32;
  const unsigned lane = threadIdx.x % kWarpWidth;
  const unsigned warp = threadIdx.x / kWarpWidth;
  for (unsigned delta = kWarpWidth / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();

  // The first warp reduces the block's partial sums. All lanes participate in
  // the shuffle so the full mask remains valid; unused lanes contribute 0.
  value = warp == 0 && lane < BlockThreads / kWarpWidth
      ? warp_sums[lane]
      : 0.0;
  if (warp == 0) {
    for (unsigned delta = kWarpWidth / 2; delta != 0; delta >>= 1) {
      value += __shfl_down_sync(0xffffffffU, value, delta);
    }
  }
  if (threadIdx.x == 0) warp_sums[0] = value;
  __syncthreads();
  return warp_sums[0];
}

__global__ void apply_warm_density_kernel(std::int32_t batch_size,
                                          std::int32_t nbf,
                                          const std::int32_t* occupied,
                                          const std::uint8_t* warm_mask,
                                          const double* warm_density,
                                          const double* overlap,
                                          double* density,
                                          std::uint8_t* warm_invalid) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || warm_mask[system] == 0) return;
  __shared__ double warp_sums[kWarmDensityThreads / 32];
  __shared__ double scale;
  __shared__ int valid_trace;
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double trace = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_size;
       element += blockDim.x) {
    const std::size_t row = element % n;
    const std::size_t column = element / n;
    const std::size_t transpose = matrix_index(column, row, n);
    const double symmetric = 0.5 *
        (warm_density[offset + element] + warm_density[offset + transpose]);
    density[offset + element] = symmetric;
    trace += symmetric * overlap[offset + transpose];
  }
  trace = warm_density_block_sum<kWarmDensityThreads>(trace, warp_sums);
  if (threadIdx.x == 0) {
    valid_trace = isfinite(trace) && trace > 0.0;
    warm_invalid[system] = valid_trace ? 0 : 1;
    scale = valid_trace ? 2.0 * occupied[system] / trace : 0.0;
  }
  __syncthreads();
  if (valid_trace != 0) {
    for (std::size_t element = threadIdx.x; element < matrix_size;
         element += blockDim.x) {
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
    double* density,
    std::uint8_t* warm_invalid) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || warm_mask[system] == 0) return;
  __shared__ double warp_sums[kWarmDensityThreads / 32];
  __shared__ double scale;
  __shared__ int valid_trace;
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t overlap_offset =
      static_cast<std::size_t>(system) * matrix_size;
  if (threadIdx.x == 0) warm_invalid[system] = 0;
  __syncthreads();
  for (std::int32_t spin = 0; spin < 2; ++spin) {
    const std::size_t state = static_cast<std::size_t>(system) * 2 + spin;
    const std::size_t offset = state * matrix_size;
    const double target = static_cast<double>(occupied[state]);
    if (target == 0.0) {
      for (std::size_t element = threadIdx.x; element < matrix_size;
           element += blockDim.x) {
        density[offset + element] = 0.0;
      }
      __syncthreads();
      continue;
    }
    double trace = 0.0;
    for (std::size_t element = threadIdx.x; element < matrix_size;
         element += blockDim.x) {
      const std::size_t row = element % n;
      const std::size_t column = element / n;
      const std::size_t transpose = matrix_index(column, row, n);
      const double symmetric = 0.5 *
          (warm_density[offset + element] + warm_density[offset + transpose]);
      density[offset + element] = symmetric;
      trace += symmetric * overlap[overlap_offset + transpose];
    }
    trace = warm_density_block_sum<kWarmDensityThreads>(trace, warp_sums);
    if (threadIdx.x == 0) {
      valid_trace = isfinite(trace) && trace > 0.0;
      scale = valid_trace ? target / trace : 0.0;
      if (valid_trace == 0) warm_invalid[system] = 1;
    }
    __syncthreads();
    if (valid_trace != 0) {
      for (std::size_t element = threadIdx.x; element < matrix_size;
           element += blockDim.x) {
        density[offset + element] *= scale;
      }
    }
    // Both spin passes reuse the same reduction and scalar slots.
    __syncthreads();
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

/**
 * Reduce the current AO density to the shell-block magnitudes used by J/K.
 *
 * The direct quartet kernels scatter every ERI symmetry permutation, so a
 * shell quartet can contribute through its two Coulomb density blocks or any
 * of its four crossed exchange blocks. UHF alpha and beta exchange bounds
 * remain separate so the force gate never invents an opposite-spin product.
 * RHF stores its one physical density in the alpha field; the Fock gate
 * applies the existing one-half exchange factor when it consumes that field.
 */
template <bool Unrestricted>
__global__ void reduce_shell_pair_density_bounds_kernel(
    DeviceBatch batch,
    const double* density,
    const std::uint8_t* active,
    ShellPairDensityBounds* shell_pair_density_bounds) {
  extern __shared__ double block_maxima[];
  double* coulomb_maxima = block_maxima;
  double* exchange_alpha_maxima = block_maxima + blockDim.x;
  double* exchange_beta_maxima = block_maxima + 2 * blockDim.x;
  const std::size_t shell_pair = static_cast<std::size_t>(blockIdx.x);
  if (shell_pair >= static_cast<std::size_t>(batch.total_shell_pairs)) return;
  const std::int32_t system = batch.shell_pair_systems[shell_pair];
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t ao_pair_count = shell_ao_pair_count(batch, shell_pair);

  double local_coulomb = 0.0;
  double local_exchange_alpha = 0.0;
  double local_exchange_beta = 0.0;
  if (active == nullptr || active[system] != 0) {
    for (std::size_t ordinal = threadIdx.x; ordinal < ao_pair_count;
         ordinal += blockDim.x) {
      std::size_t first = 0;
      std::size_t second = 0;
      decode_shell_ao_pair(batch, shell_pair, ordinal, system_ao_begin,
                           first, second);
      const std::size_t forward = matrix_index(first, second, n);
      const std::size_t reverse = matrix_index(second, first, n);
      if constexpr (Unrestricted) {
        const double alpha_forward = density[spin_offset + forward];
        const double beta_forward =
            density[spin_offset + matrix_size + forward];
        const double alpha_reverse = density[spin_offset + reverse];
        const double beta_reverse =
            density[spin_offset + matrix_size + reverse];
        local_coulomb = fmax(
            local_coulomb,
            fmax(fabs(alpha_forward + beta_forward),
                 fabs(alpha_reverse + beta_reverse)));
        local_exchange_alpha = fmax(
            local_exchange_alpha,
            fmax(fabs(alpha_forward), fabs(alpha_reverse)));
        local_exchange_beta = fmax(
            local_exchange_beta,
            fmax(fabs(beta_forward), fabs(beta_reverse)));
      } else {
        const double magnitude = fmax(
            fabs(density[physical_offset + forward]),
            fabs(density[physical_offset + reverse]));
        local_coulomb = fmax(local_coulomb, magnitude);
        local_exchange_alpha = fmax(local_exchange_alpha, magnitude);
      }
    }
  }

  coulomb_maxima[threadIdx.x] = local_coulomb;
  exchange_alpha_maxima[threadIdx.x] = local_exchange_alpha;
  exchange_beta_maxima[threadIdx.x] = local_exchange_beta;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      coulomb_maxima[threadIdx.x] = fmax(
          coulomb_maxima[threadIdx.x],
          coulomb_maxima[threadIdx.x + stride]);
      exchange_alpha_maxima[threadIdx.x] = fmax(
          exchange_alpha_maxima[threadIdx.x],
          exchange_alpha_maxima[threadIdx.x + stride]);
      exchange_beta_maxima[threadIdx.x] = fmax(
          exchange_beta_maxima[threadIdx.x],
          exchange_beta_maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    shell_pair_density_bounds[shell_pair] = {
        coulomb_maxima[0], exchange_alpha_maxima[0],
        exchange_beta_maxima[0]};
  }
}

/**
 * Reduce one permutation-contiguous shell-pair block to its Schwarz maximum.
 *
 * The permutation is refreshed whenever geometry changes so similarly sized
 * pairs share a coarse gate and class streams retain a monotonic Schwarz tail.
 */
__global__ void reduce_bounded_shell_pair_block_bounds_kernel(
    DeviceBatch batch,
    const std::uint32_t* shell_pair_order,
    const double* shell_pair_bounds,
    double* shell_pair_block_bounds) {
  extern __shared__ double block_maxima[];
  const std::size_t block = static_cast<std::size_t>(blockIdx.x);
  if (block >= static_cast<std::size_t>(batch.total_shell_pair_blocks)) return;

  std::int32_t system = 0;
  while (system + 1 < batch.batch_size &&
         static_cast<std::size_t>(
             batch.system_shell_pair_block_offsets[system + 1]) <= block) {
    ++system;
  }
  const std::size_t local_block = block - static_cast<std::size_t>(
      batch.system_shell_pair_block_offsets[system]);
  const std::size_t pair_begin = static_cast<std::size_t>(
      batch.system_shell_pair_offsets[system]);
  const std::size_t pair_end = static_cast<std::size_t>(
      batch.system_shell_pair_offsets[system + 1]);
  const std::size_t ordered_begin = pair_begin +
      local_block * detail::kBoundedDirectShellPairBlockSize;
  const std::size_t ordered_end = min(
      pair_end,
      ordered_begin + detail::kBoundedDirectShellPairBlockSize);

  double local_maximum = 0.0;
  for (std::size_t ordered = ordered_begin + threadIdx.x;
       ordered < ordered_end; ordered += blockDim.x) {
    local_maximum = fmax(
        local_maximum, shell_pair_bounds[shell_pair_order[ordered]]);
  }
  block_maxima[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_maxima[threadIdx.x] = fmax(
          block_maxima[threadIdx.x], block_maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    shell_pair_block_bounds[block] = block_maxima[0];
  }
}

/** Conservative density maximum used only to reject complete pair blocks. */
__global__ void reduce_bounded_system_density_bounds_kernel(
    DeviceBatch batch,
    const ShellPairDensityBounds* shell_pair_density_bounds,
    double* system_density_bounds) {
  extern __shared__ double block_maxima[];
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch.batch_size) return;
  const std::size_t pair_begin = static_cast<std::size_t>(
      batch.system_shell_pair_offsets[system]);
  const std::size_t pair_end = static_cast<std::size_t>(
      batch.system_shell_pair_offsets[system + 1]);
  double local_maximum = 0.0;
  for (std::size_t pair = pair_begin + threadIdx.x;
       pair < pair_end; pair += blockDim.x) {
    const ShellPairDensityBounds bound = shell_pair_density_bounds[pair];
    local_maximum = fmax(
        local_maximum,
        fmax(bound.coulomb,
             fmax(bound.exchange_alpha, bound.exchange_beta)));
  }
  block_maxima[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_maxima[threadIdx.x] = fmax(
          block_maxima[threadIdx.x], block_maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) system_density_bounds[system] = block_maxima[0];
}

__global__ void clear_active_shell_quartet_tile_counts_kernel(
    std::uint32_t* active_shell_quartet_tile_counts,
    std::uint32_t* persistent_fock_task_heads,
    std::uint32_t* fp32_shell_quartet_tile_counts,
    std::uint32_t* fp32_persistent_fock_task_heads) {
  const std::size_t order =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (order < detail::kDirectQuartetAngularOrderCount) {
    active_shell_quartet_tile_counts[order] = 0;
    if (fp32_shell_quartet_tile_counts != nullptr) {
      fp32_shell_quartet_tile_counts[order] = 0;
    }
  }
  // Reset queue state in the same captured Graph node as the active counts.
  // A separate tiny kernel here would be replayed for every SCF iteration.
  if (order < kPersistentFockAngularOrderCount) {
    persistent_fock_task_heads[order] = 0;
    if (fp32_persistent_fock_task_heads != nullptr) {
      fp32_persistent_fock_task_heads[order] = 0;
    }
  }
}

/** Apply the shell-level Schwarz and density gate for one direct consumer. */
template <bool Unrestricted, DirectScreeningPurpose Purpose>
__device__ __forceinline__ bool direct_shell_quartet_survives_screening(
    const DeviceBatch& batch,
    std::size_t first_pair,
    std::size_t second_pair,
    double screening_tolerance,
    const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds,
    double* fock_contribution_bound = nullptr) {
  const double quartet_bound =
      shell_pair_bounds[first_pair] * shell_pair_bounds[second_pair];
  if (quartet_bound < screening_tolerance) return false;

  const std::int32_t system = batch.shell_pair_systems[first_pair];
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const std::size_t ac_pair = system_shell_pair_index(
      batch, system, first_shell, third_shell);
  const std::size_t ad_pair = system_shell_pair_index(
      batch, system, first_shell, fourth_shell);
  const std::size_t bc_pair = system_shell_pair_index(
      batch, system, second_shell, third_shell);
  const std::size_t bd_pair = system_shell_pair_index(
      batch, system, second_shell, fourth_shell);
  const ShellPairDensityBounds ab = shell_pair_density_bounds[first_pair];
  const ShellPairDensityBounds cd = shell_pair_density_bounds[second_pair];
  const ShellPairDensityBounds ac = shell_pair_density_bounds[ac_pair];
  const ShellPairDensityBounds ad = shell_pair_density_bounds[ad_pair];
  const ShellPairDensityBounds bc = shell_pair_density_bounds[bc_pair];
  const ShellPairDensityBounds bd = shell_pair_density_bounds[bd_pair];

  double fock_density_bound = fmax(
      ab.coulomb, cd.coulomb);
  if constexpr (Unrestricted) {
    fock_density_bound = fmax(
        fock_density_bound,
        fmax(fmax(ac.exchange_alpha, ac.exchange_beta),
             fmax(ad.exchange_alpha, ad.exchange_beta)));
    fock_density_bound = fmax(
        fock_density_bound,
        fmax(fmax(bc.exchange_alpha, bc.exchange_beta),
             fmax(bd.exchange_alpha, bd.exchange_beta)));
  } else {
    // Preserve the established RHF Fock gate exactly: F = J - K/2.
    const double exchange_bound = fmax(
        fmax(ac.exchange_alpha, ad.exchange_alpha),
        fmax(bc.exchange_alpha, bd.exchange_alpha));
    fock_density_bound = fmax(fock_density_bound, 0.5 * exchange_bound);
  }
  const double contribution_bound = quartet_bound * fock_density_bound;
  if (fock_contribution_bound != nullptr) {
    *fock_contribution_bound = contribution_bound;
  }
  if (contribution_bound < screening_tolerance) return false;
  if constexpr (Purpose == DirectScreeningPurpose::Fock) return true;

  // Screen J and each same-spin K contraction independently. Combining the
  // exact symmetry-reduced coefficient here would exploit cancellation and
  // can make loose-screening analytic forces disagree with finite differences.
  const double force_screening_tolerance =
      fmin(screening_tolerance, kForceDensityProductScreeningTolerance);
  if (quartet_bound * ab.coulomb * cd.coulomb >=
      force_screening_tolerance) {
    return true;
  }
  if constexpr (Unrestricted) {
    return quartet_bound * ac.exchange_alpha * bd.exchange_alpha >=
               force_screening_tolerance ||
        quartet_bound * ac.exchange_beta * bd.exchange_beta >=
               force_screening_tolerance ||
        quartet_bound * ad.exchange_alpha * bc.exchange_alpha >=
               force_screening_tolerance ||
        quartet_bound * ad.exchange_beta * bc.exchange_beta >=
               force_screening_tolerance;
  } else {
    return quartet_bound * ac.exchange_alpha * bd.exchange_alpha >=
               force_screening_tolerance ||
        quartet_bound * ad.exchange_alpha * bc.exchange_alpha >=
               force_screening_tolerance;
  }
}

template <bool Unrestricted, DirectScreeningPurpose Purpose>
__global__ void compact_active_shell_quartet_tiles_kernel(
    DeviceBatch batch,
    double screening_tolerance,
    const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint8_t* active,
    const std::uint32_t* active_shell_quartet_tile_offsets,
    std::uint32_t* active_shell_quartet_tile_counts,
    ActiveShellQuartetTile* active_shell_quartet_tiles,
    bool mixed_precision_enabled,
    double fp64_threshold,
    const std::uint32_t* fp32_shell_quartet_tile_offsets,
    std::uint32_t* fp32_shell_quartet_tile_counts,
    ActiveShellQuartetTile* fp32_shell_quartet_tiles) {
  const std::size_t shell_quartet =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (shell_quartet >=
      static_cast<std::size_t>(batch.total_shell_quartets)) {
    return;
  }

  const std::int32_t system = shell_quartet_system(batch, shell_quartet);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local_quartet = shell_quartet -
      static_cast<std::size_t>(batch.system_shell_quartet_offsets[system]);
  std::size_t first_pair_local = 0;
  std::size_t second_pair_local = 0;
  decode_lower_triangle(local_quartet, first_pair_local, second_pair_local);
  const std::size_t pair_begin =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
  const std::size_t first_pair = pair_begin + first_pair_local;
  const std::size_t second_pair = pair_begin + second_pair_local;
  double contribution_bound = 0.0;
  if (!direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
          batch, first_pair, second_pair, screening_tolerance,
          shell_pair_bounds, shell_pair_density_bounds,
          Purpose == DirectScreeningPurpose::Fock
              ? &contribution_bound
              : nullptr)) return;

  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];

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
  bool use_fp32 = false;
  if constexpr (Purpose == DirectScreeningPurpose::Fock) {
    // Low-order shell-fused workers remain FP64; routing them through the
    // generic evaluator would conflate precision with a scheduling regression.
    use_fp32 = mixed_precision_enabled &&
        angular_order >= kMixedFockMinimumAngularOrder &&
        fp32_shell_quartet_tile_counts != nullptr &&
        contribution_bound < fp64_threshold;
  }
  std::uint32_t* selected_counts = use_fp32
      ? fp32_shell_quartet_tile_counts
      : active_shell_quartet_tile_counts;
  ActiveShellQuartetTile* selected_tiles = use_fp32
      ? fp32_shell_quartet_tiles
      : active_shell_quartet_tiles;
  const std::uint32_t* selected_offsets = use_fp32
      ? fp32_shell_quartet_tile_offsets
      : active_shell_quartet_tile_offsets;
  const std::uint32_t slot = selected_offsets[angular_order] +
      atomicAdd(selected_counts + angular_order, tile_count);
  for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
    selected_tiles[slot + tile] = {
        static_cast<std::uint32_t>(first_pair),
        static_cast<std::uint32_t>(second_pair), tile};
  }
}

constexpr std::uint8_t kNoGeneratedShellClass =
    std::numeric_limits<std::uint8_t>::max();

/**
 * Return the canonical ``pp`` pair for an active ppps tile.
 *
 * Direct compaction stores an unordered pair-of-pairs, while generated
 * kernels consume the pair with the larger triangular class in slot zero.
 * Keeping this test in one device helper makes resident grouping use exactly
 * the same symmetry convention as generated task materialization.  The
 * primitive-pair limit is part of the predicate: a bra that cannot fit in
 * shared memory must remain visible to the ordinary generated ppps queue.
 */
__device__ __forceinline__ bool resident_ppps_bra_pair(
    const DeviceBatch& batch,
    const ActiveShellQuartetTile& tile,
    std::uint32_t& bra_pair) {
  if (tile.tile != 0U) return false;
  const std::int32_t first_shell = batch.shell_pair_first[tile.first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[tile.first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[tile.second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[tile.second_pair];
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (first_pair_class == 2U && second_pair_class == 1U) {
    bra_pair = tile.first_pair;
  } else if (first_pair_class == 1U && second_pair_class == 2U) {
    bra_pair = tile.second_pair;
  } else {
    return false;
  }
  const std::int64_t begin = batch.shell_pair_primitive_offsets[bra_pair];
  const std::int64_t end =
      batch.shell_pair_primitive_offsets[static_cast<std::size_t>(bra_pair) + 1U];
  const std::int64_t count = end - begin;
  return count > 0 && count <= static_cast<std::int64_t>(
      kGeneratedPppsResidentMaximumBraPrimitivePairs);
}

/** Return the exact orientation/ket-primitive bucket for one resident tile. */
__device__ __forceinline__ unsigned resident_ppps_signature_bucket(
    const DeviceBatch& batch,
    const ActiveShellQuartetTile& tile,
    std::uint32_t bra_pair) {
  const bool pair_exchanged = bra_pair == tile.second_pair;
  const std::uint32_t ket_pair =
      pair_exchanged ? tile.first_pair : tile.second_pair;
  const std::int64_t ket_begin =
      batch.shell_pair_primitive_offsets[ket_pair];
  const std::int64_t ket_end =
      batch.shell_pair_primitive_offsets[ket_pair + 1U];
  const std::uint64_t ket_count = ket_end > ket_begin
      ? static_cast<std::uint64_t>(ket_end - ket_begin)
      : 0U;
  const unsigned primitive_bucket = static_cast<unsigned>(min(
      ket_count,
      static_cast<std::uint64_t>(kPppsSignaturePrimitivePairBuckets - 1U)));
  return (pair_exchanged ? kPppsSignaturePrimitivePairBuckets : 0U) +
      primitive_bucket;
}

/** Return the page-local loop/orientation signature for one bounded task. */
__device__ __forceinline__ unsigned bounded_force_signature_bucket(
    const DeviceBatch& batch,
    std::uint32_t first_pair,
    std::uint32_t second_pair) {
  const std::int64_t first_begin =
      batch.shell_pair_primitive_offsets[first_pair];
  const std::int64_t first_end =
      batch.shell_pair_primitive_offsets[first_pair + 1U];
  const std::int64_t second_begin =
      batch.shell_pair_primitive_offsets[second_pair];
  const std::int64_t second_end =
      batch.shell_pair_primitive_offsets[second_pair + 1U];
  const std::uint64_t first_count = first_end > first_begin
      ? static_cast<std::uint64_t>(first_end - first_begin)
      : 0U;
  const std::uint64_t second_count = second_end > second_begin
      ? static_cast<std::uint64_t>(second_end - second_begin)
      : 0U;
  const unsigned first_bucket = static_cast<unsigned>(min(
      first_count,
      static_cast<std::uint64_t>(
          kPppsSignaturePrimitivePairBuckets - 1U)));
  const unsigned second_bucket = static_cast<unsigned>(min(
      second_count,
      static_cast<std::uint64_t>(
          kPppsSignaturePrimitivePairBuckets - 1U)));
  const std::int32_t first_shell = batch.shell_pair_first[first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
  const unsigned orientation =
      (batch.shell_angular[first_shell] <
           batch.shell_angular[second_shell]
       ? 2U
       : 0U) |
      (batch.shell_angular[third_shell] <
           batch.shell_angular[fourth_shell]
       ? 1U
       : 0U);
  return (orientation * kPppsSignaturePrimitivePairBuckets + first_bucket) *
      kPppsSignaturePrimitivePairBuckets + second_bucket;
}

/** Return the ordered primitive-pair loop signature for one low-order tile. */
__device__ __forceinline__ unsigned generated_low_order_signature_bucket(
    const DeviceBatch& batch,
    const ActiveShellQuartetTile& tile) {
  const std::int64_t first_begin =
      batch.shell_pair_primitive_offsets[tile.first_pair];
  const std::int64_t first_end =
      batch.shell_pair_primitive_offsets[tile.first_pair + 1U];
  const std::int64_t second_begin =
      batch.shell_pair_primitive_offsets[tile.second_pair];
  const std::int64_t second_end =
      batch.shell_pair_primitive_offsets[tile.second_pair + 1U];
  const std::uint64_t first_count = first_end > first_begin
      ? static_cast<std::uint64_t>(first_end - first_begin)
      : 0U;
  const std::uint64_t second_count = second_end > second_begin
      ? static_cast<std::uint64_t>(second_end - second_begin)
      : 0U;
  const unsigned first_bucket = static_cast<unsigned>(min(
      first_count,
      static_cast<std::uint64_t>(
          kLowOrderSignaturePrimitivePairBuckets - 1U)));
  const unsigned second_bucket = static_cast<unsigned>(min(
      second_count,
      static_cast<std::uint64_t>(
          kLowOrderSignaturePrimitivePairBuckets - 1U)));
  return first_bucket * kLowOrderSignaturePrimitivePairBuckets + second_bucket;
}

/** Map each supported scalar class to its private signature histogram. */
__device__ __forceinline__ unsigned generated_low_order_signature_index(
    unsigned shell_class,
    unsigned signature) {
  const unsigned class_slot = shell_class == kPspsShellClass ? 0U : 1U;
  return class_slot * kLowOrderSignatureBucketsPerClass + signature;
}

/**
 * Fill the stable generated task ABI from one canonicalized shell quartet.
 *
 * Both the ordinary class queue and the resident ppps queue use this helper.
 * In particular, the two one-bit pair-orientation mask records the swaps
 * applied before pair-exchange canonicalization; generated force code uses
 * it to map primitive-pair product scales back to physical centers.
 */
__device__ __forceinline__ void populate_generated_shell_task(
    const DeviceBatch& batch,
    const ActiveShellQuartetTile& tile,
    GeneratedShellTask& task) {
  std::int32_t shells[4] = {
      batch.shell_pair_first[tile.first_pair],
      batch.shell_pair_second[tile.first_pair],
      batch.shell_pair_first[tile.second_pair],
      batch.shell_pair_second[tile.second_pair],
  };
  std::uint32_t shell_pairs[2] = {tile.first_pair, tile.second_pair};
  std::uint32_t reversed_shell_pair_mask = 0U;
  if (batch.shell_angular[shells[0]] < batch.shell_angular[shells[1]]) {
    const std::int32_t swap = shells[0];
    shells[0] = shells[1];
    shells[1] = swap;
    reversed_shell_pair_mask |= 1U;
  }
  if (batch.shell_angular[shells[2]] < batch.shell_angular[shells[3]]) {
    const std::int32_t swap = shells[2];
    shells[2] = shells[3];
    shells[3] = swap;
    reversed_shell_pair_mask |= 2U;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[shells[0]], batch.shell_angular[shells[1]]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[shells[2]], batch.shell_angular[shells[3]]);
  if (first_pair_class < second_pair_class) {
    const std::int32_t first = shells[0];
    const std::int32_t second = shells[1];
    shells[0] = shells[2];
    shells[1] = shells[3];
    shells[2] = first;
    shells[3] = second;
    const std::uint32_t pair_swap = shell_pairs[0];
    shell_pairs[0] = shell_pairs[1];
    shell_pairs[1] = pair_swap;
    reversed_shell_pair_mask =
        ((reversed_shell_pair_mask & 1U) << 1U) |
        ((reversed_shell_pair_mask & 2U) >> 1U);
  }

  const std::int32_t system = batch.shell_pair_systems[tile.first_pair];
  const std::size_t matrix_order =
      static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = matrix_order * matrix_order;
  const std::size_t system_ao_begin =
      static_cast<std::size_t>(system) * matrix_order;
#pragma unroll
  for (unsigned center = 0; center < 4U; ++center) {
    const std::int32_t shell = shells[center];
    task.primitive_begin[center] = static_cast<std::uint64_t>(
        batch.shell_primitive_offsets[shell]);
    task.primitive_end[center] = static_cast<std::uint64_t>(
        batch.shell_primitive_offsets[shell + 1]);
    const std::size_t ao_begin = static_cast<std::size_t>(
        batch.shell_direct_ao_offsets[shell]);
    task.ao_begin[center] = static_cast<std::uint64_t>(
        ao_begin - system_ao_begin);
    task.ao_coefficient_begin[center] = static_cast<std::uint64_t>(ao_begin);
    task.shell[center] = static_cast<std::uint32_t>(shell);
    task.atom[center] = static_cast<std::uint32_t>(
        batch.shell_atoms[shell]);
  }
  task.density_offset = static_cast<std::uint64_t>(
      static_cast<std::size_t>(system) * matrix_size);
  task.spin_offset = static_cast<std::uint64_t>(
      static_cast<std::size_t>(system) * 2U * matrix_size);
  task.matrix_order = static_cast<std::uint32_t>(matrix_order);
  task.shell_pair[0] = shell_pairs[0];
  task.shell_pair[1] = shell_pairs[1];
  task.reversed_shell_pair_mask = reversed_shell_pair_mask;
}

/**
 * Classify every active logical quartet once for all generated consumers.
 *
 * The byte tag is retained across the device-side prefix sum so task
 * materialization does not repeat exact-class decoding. Slots for AO tiles
 * beyond tile zero and classes disabled by the runtime mask remain tagged as
 * unclassified and fall through to the handwritten consumers.
 */
__global__ void classify_generated_shell_tasks_kernel(
    DeviceBatch batch,
    std::size_t total_tile_capacity,
    const std::uint32_t* active_shell_quartet_tile_offsets,
    const std::uint32_t* active_shell_quartet_tile_counts,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t enabled_shell_class_mask,
    const std::uint64_t* enabled_shell_class_mask_pointer,
    bool exclude_resident_ppps,
    std::uint32_t* generated_task_counts,
    std::uint8_t* generated_shell_classes,
    std::uint64_t low_order_signature_mask,
    std::uint32_t* low_order_signature_counts) {
  const std::size_t slot =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= total_tile_capacity) return;
  generated_shell_classes[slot] = kNoGeneratedShellClass;

  unsigned angular_order = 0;
  while (angular_order + 1 < detail::kDirectQuartetAngularOrderCount &&
         slot >= active_shell_quartet_tile_offsets[angular_order + 1]) {
    ++angular_order;
  }
  const std::size_t partition_begin =
      active_shell_quartet_tile_offsets[angular_order];
  if (slot - partition_begin >= active_shell_quartet_tile_counts[angular_order]) {
    return;
  }

  const ActiveShellQuartetTile tile = active_shell_quartet_tiles[slot];
  if (tile.tile != 0U) return;

  const std::int32_t first_shell = batch.shell_pair_first[tile.first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[tile.first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[tile.second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[tile.second_pair];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (enabled_shell_class_mask_pointer != nullptr) {
    enabled_shell_class_mask = *enabled_shell_class_mask_pointer;
  }
  if (shell_class >= detail::kDirectQuartetShellClassCount ||
      (enabled_shell_class_mask &
       (std::uint64_t{1} << shell_class)) == 0U) {
    return;
  }
  // Force preparation can route eligible canonical ppps quartets through the
  // resident-bra consumer.  Leave oversized bra primitive lists in this
  // ordinary class queue; otherwise their force contribution would vanish.
  if (exclude_resident_ppps && shell_class == kPppsShellClass) {
    std::uint32_t bra_pair = 0;
    if (resident_ppps_bra_pair(batch, tile, bra_pair)) return;
  }
  generated_shell_classes[slot] = static_cast<std::uint8_t>(shell_class);
  atomicAdd(generated_task_counts + shell_class, 1U);
  if (low_order_signature_counts != nullptr &&
      (low_order_signature_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    const unsigned signature =
        generated_low_order_signature_bucket(batch, tile);
    atomicAdd(
        low_order_signature_counts + generated_low_order_signature_index(
            shell_class, signature),
        1U);
  }
}

/** Build compact class slices and reset their materialization/worker cursors. */
__global__ void prefix_generated_shell_task_counts_kernel(
    const std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_write_counts,
    std::uint32_t* generated_task_heads) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  std::uint32_t offset = 0;
  generated_task_offsets[0] = 0;
  for (unsigned shell_class = 0;
       shell_class < detail::kDirectQuartetShellClassCount; ++shell_class) {
    generated_task_write_counts[shell_class] = 0;
    generated_task_heads[shell_class] = 0;
    offset += generated_task_counts[shell_class];
    generated_task_offsets[shell_class + 1] = offset;
  }
}

/** Prefix selected scalar signature slices inside their exact-class ranges. */
__global__ void prefix_low_order_signature_counts_kernel(
    const std::uint32_t* generated_task_offsets,
    std::uint64_t low_order_signature_mask,
    std::uint32_t* low_order_signature_counts,
    std::uint32_t* low_order_signature_offsets) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  for (unsigned class_slot = 0; class_slot < kLowOrderSignatureClassCount;
       ++class_slot) {
    const unsigned shell_class =
        class_slot == 0U ? kPspsShellClass : kPpssShellClass;
    if ((low_order_signature_mask &
         (std::uint64_t{1} << shell_class)) == 0U) {
      continue;
    }
    std::uint32_t offset = generated_task_offsets[shell_class];
    const unsigned signature_begin =
        class_slot * kLowOrderSignatureBucketsPerClass;
    for (unsigned signature = 0;
         signature < kLowOrderSignatureBucketsPerClass; ++signature) {
      const unsigned index = signature_begin + signature;
      const std::uint32_t count = low_order_signature_counts[index];
      low_order_signature_offsets[index] = offset;
      // Reuse the count array as the scatter cursor after preserving the class
      // total in generated_task_counts for the persistent worker.
      low_order_signature_counts[index] = 0U;
      offset += count;
    }
  }
}

/** Canonicalize classified quartets into contiguous exact-class slices. */
__global__ void materialize_generated_shell_tasks_kernel(
    DeviceBatch batch,
    std::size_t total_tile_capacity,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    const std::uint8_t* generated_shell_classes,
    const std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_write_counts,
    GeneratedShellTask* generated_tasks,
    std::uint64_t low_order_signature_mask,
    const std::uint32_t* low_order_signature_offsets,
    std::uint32_t* low_order_signature_write_counts) {
  const std::size_t active_tile =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (active_tile >= total_tile_capacity) return;
  const unsigned shell_class = generated_shell_classes[active_tile];
  if (shell_class == kNoGeneratedShellClass) return;

  const ActiveShellQuartetTile tile = active_shell_quartet_tiles[active_tile];
  std::uint32_t task_index = 0U;
  if (low_order_signature_offsets != nullptr &&
      low_order_signature_write_counts != nullptr &&
      (low_order_signature_mask &
       (std::uint64_t{1} << shell_class)) != 0U) {
    const unsigned signature =
        generated_low_order_signature_bucket(batch, tile);
    const unsigned index = generated_low_order_signature_index(
        shell_class, signature);
    task_index = low_order_signature_offsets[index] +
        atomicAdd(low_order_signature_write_counts + index, 1U);
  } else {
    task_index = generated_task_offsets[shell_class] +
        atomicAdd(generated_task_write_counts + shell_class, 1U);
  }
  populate_generated_shell_task(batch, tile, generated_tasks[task_index]);
}

/**
 * Count force-eligible canonical ppps tiles by their ``pp`` bra pair.
 *
 * This is intentionally indexed by the global shell-pair ordinal rather than
 * by a host-built map.  A direct batch can contain roughly 18k shell pairs;
 * three compact uint32 arrays make that histogram inexpensive and, more
 * importantly, keep active-system and force-screening decisions on device.
 */
__global__ void count_ppps_resident_bra_tasks_kernel(
    DeviceBatch batch,
    std::size_t active_tile_capacity,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t enabled_shell_class_mask,
    std::uint32_t* resident_bra_counts,
    std::uint32_t* resident_signature_counts) {
  const std::size_t slot =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= active_tile_capacity ||
      slot >= *active_shell_quartet_tile_count) return;
  const ActiveShellQuartetTile tile = active_shell_quartet_tiles[slot];
  if ((enabled_shell_class_mask & (std::uint64_t{1} << kPppsShellClass)) ==
      0U) {
    return;
  }
  std::uint32_t bra_pair = 0;
  if (resident_ppps_bra_pair(batch, tile, bra_pair)) {
    atomicAdd(resident_bra_counts + bra_pair, 1U);
    if (resident_signature_counts != nullptr) {
      const unsigned signature =
          resident_ppps_signature_bucket(batch, tile, bra_pair);
      atomicAdd(
          resident_signature_counts +
              static_cast<std::size_t>(bra_pair) *
                  kPppsSignatureBucketCount +
              signature,
          1U);
    }
  }
}

/**
 * Prefix the ppps bra histogram and initialize one descriptor per bra.
 *
 * Descriptors are stored at their shell-pair ordinal. Inactive ordinals have
 * a zero ``ket_count`` and are harmless when the resident launch uses the
 * fixed shell-pair capacity; this avoids a device-to-host count readback and
 * keeps the force path graph/replay safe. ``resident_bra_offsets`` indexes
 * the transient ppps-sized tail of the generated-task arena. The final-force
 * stream launches the resident consumer before ordinary preparation is
 * allowed to overwrite that tail.
 */
__global__ void prefix_ppps_resident_bra_tasks_kernel(
    std::size_t total_shell_pairs,
    const std::uint32_t* resident_bra_counts,
    std::uint32_t* resident_bra_offsets,
    std::uint32_t* resident_bra_write_counts,
    GeneratedPppsResidentTask* resident_tasks) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  std::uint32_t offset = 0U;
  resident_bra_offsets[0] = 0U;
  for (std::size_t bra_pair = 0; bra_pair < total_shell_pairs; ++bra_pair) {
    resident_bra_write_counts[bra_pair] = 0U;
    const std::uint32_t count = resident_bra_counts[bra_pair];
    resident_tasks[bra_pair] = {
        static_cast<std::uint32_t>(bra_pair), offset, count};
    // Host topology validation bounds the resident allocation below
    // UINT32_MAX: every resident ket is one active ppps tile and the tile
    // capacity is checked before this kernel is launched.
    offset += count;
    resident_bra_offsets[bra_pair + 1U] = offset;
    resident_tasks[bra_pair].ket_begin = resident_bra_offsets[bra_pair];
  }
}

/** Build per-bra orientation/primitive bucket offsets for stable scattering. */
__global__ void prefix_ppps_resident_signature_buckets_kernel(
    std::size_t total_shell_pairs,
    const std::uint32_t* resident_bra_offsets,
    std::uint32_t* resident_signature_counts,
    std::uint32_t* resident_signature_offsets) {
  const std::size_t bra_pair = blockIdx.x;
  if (bra_pair >= total_shell_pairs || threadIdx.x != 0U) return;
  std::uint32_t offset = resident_bra_offsets[bra_pair];
  const std::size_t bucket_begin =
      bra_pair * kPppsSignatureBucketCount;
  for (unsigned bucket = 0U; bucket < kPppsSignatureBucketCount; ++bucket) {
    const std::size_t index = bucket_begin + bucket;
    const std::uint32_t count = resident_signature_counts[index];
    resident_signature_offsets[index] = offset;
    resident_signature_counts[index] = 0U;
    offset += count;
  }
}

/** Materialize eligible ppps tasks into the bra-grouped resident array. */
__global__ void materialize_ppps_resident_bra_tasks_kernel(
    DeviceBatch batch,
    std::size_t active_tile_capacity,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    const std::uint32_t* resident_bra_offsets,
    std::uint32_t* resident_bra_write_counts,
    const std::uint32_t* resident_signature_offsets,
    std::uint32_t* resident_signature_write_counts,
    GeneratedShellTask* resident_ket_tasks,
    std::uint32_t* resident_ket_signatures) {
  const std::size_t slot =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= active_tile_capacity ||
      slot >= *active_shell_quartet_tile_count) return;
  const ActiveShellQuartetTile tile = active_shell_quartet_tiles[slot];
  std::uint32_t bra_pair = 0;
  if (!resident_ppps_bra_pair(batch, tile, bra_pair)) return;
  const bool pair_exchanged = bra_pair == tile.second_pair;
  std::uint32_t ket_index = 0U;
  if (resident_signature_offsets != nullptr &&
      resident_signature_write_counts != nullptr) {
    const unsigned signature =
        resident_ppps_signature_bucket(batch, tile, bra_pair);
    const std::size_t bucket_index =
        static_cast<std::size_t>(bra_pair) * kPppsSignatureBucketCount +
        signature;
    ket_index = resident_signature_offsets[bucket_index] +
        atomicAdd(resident_signature_write_counts + bucket_index, 1U);
  } else {
    ket_index = resident_bra_offsets[bra_pair] +
        atomicAdd(resident_bra_write_counts + bra_pair, 1U);
  }
  populate_generated_shell_task(batch, tile, resident_ket_tasks[ket_index]);
  if (resident_ket_signatures != nullptr) {
    const GeneratedShellTask& task = resident_ket_tasks[ket_index];
    const std::int64_t ket_begin =
        batch.shell_pair_primitive_offsets[task.shell_pair[1]];
    const std::int64_t ket_end =
        batch.shell_pair_primitive_offsets[task.shell_pair[1] + 1U];
    const std::uint64_t ket_count = ket_end > ket_begin
        ? static_cast<std::uint64_t>(ket_end - ket_begin)
        : 0U;
    constexpr std::uint32_t kCountMask = 0x7fffffffU;
    const std::uint32_t encoded_count = ket_count > kCountMask
        ? kCountMask
        : static_cast<std::uint32_t>(ket_count);
    const std::uint32_t orientation =
        pair_exchanged ? 0x80000000U : 0U;
    resident_ket_signatures[ket_index] = orientation | encoded_count;
  }
}

/** Prepare the resident ppps histogram, prefix, descriptors, and ket records. */
cudaError_t prepare_ppps_resident_tasks(
    cudaStream_t stream,
    std::size_t active_tile_capacity,
    std::size_t total_shell_pairs,
    DeviceBatch batch,
    const std::uint32_t* active_tile_count,
    const ActiveShellQuartetTile* active_tiles,
    GeneratedPppsResidentTask* resident_tasks,
    GeneratedShellTask* resident_ket_tasks,
    std::uint32_t* resident_bra_counts,
    std::uint32_t* resident_bra_offsets,
    std::uint32_t* resident_bra_write_counts,
    std::uint32_t* resident_signature_counts,
    std::uint32_t* resident_signature_offsets,
    std::uint32_t* resident_ket_signatures,
    std::uint64_t enabled_mask) {
  if (active_tile_capacity == 0 || total_shell_pairs == 0 ||
      resident_tasks == nullptr || resident_ket_tasks == nullptr ||
      resident_bra_counts == nullptr || resident_bra_offsets == nullptr ||
      resident_bra_write_counts == nullptr ||
      (enabled_mask & (std::uint64_t{1} << kPppsShellClass)) == 0U) {
    return cudaSuccess;
  }
  cudaError_t error = cudaMemsetAsync(
      resident_bra_counts, 0, total_shell_pairs * sizeof(std::uint32_t),
      stream);
  if (error != cudaSuccess) return error;
  if (resident_signature_counts != nullptr) {
    error = cudaMemsetAsync(
        resident_signature_counts, 0,
        total_shell_pairs * kPppsSignatureBucketCount *
            sizeof(std::uint32_t),
        stream);
    if (error != cudaSuccess) return error;
  }
  constexpr unsigned preparation_threads = kCaptureSafeKernelThreads;
  const unsigned preparation_blocks = static_cast<unsigned>(
      (active_tile_capacity + preparation_threads - 1) /
      preparation_threads);
  count_ppps_resident_bra_tasks_kernel<<<
      preparation_blocks, preparation_threads, 0, stream>>>(
      batch, active_tile_capacity, active_tile_count, active_tiles,
      enabled_mask, resident_bra_counts, resident_signature_counts);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  prefix_ppps_resident_bra_tasks_kernel<<<1, 1, 0, stream>>>(
      total_shell_pairs, resident_bra_counts, resident_bra_offsets,
      resident_bra_write_counts, resident_tasks);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  if (resident_signature_counts != nullptr &&
      resident_signature_offsets != nullptr) {
    prefix_ppps_resident_signature_buckets_kernel<<<
        static_cast<unsigned>(total_shell_pairs), 1, 0, stream>>>(
        total_shell_pairs, resident_bra_offsets, resident_signature_counts,
        resident_signature_offsets);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
  }
  materialize_ppps_resident_bra_tasks_kernel<<<
      preparation_blocks, preparation_threads, 0, stream>>>(
      batch, active_tile_capacity, active_tile_count, active_tiles,
      resident_bra_offsets, resident_bra_write_counts,
      resident_signature_offsets, resident_signature_counts,
      resident_ket_tasks, resident_ket_signatures);
  return cudaPeekAtLastError();
}

/**
 * Compact the order-five fallback after excluding currently enabled AOT
 * classes. This stays separate from exact-class compaction so runtime masks
 * such as ``none``, ``dppp``, and ``all`` retain a correct generic fallback.
 */
__global__ void compact_generic_order5_tiles_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint64_t generated_shell_class_mask,
    const std::uint64_t* generated_shell_class_mask_pointer,
    std::uint32_t* generic_tile_count,
    ActiveShellQuartetTile* generic_tiles) {
  // Fock graph replay uploads its runtime selection to device memory, while
  // the final force path supplies a host-resolved value outside the graph.
  if (generated_shell_class_mask_pointer != nullptr) {
    generated_shell_class_mask = *generated_shell_class_mask_pointer;
  }
  const std::size_t active_tile =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (active_tile >=
      static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const ActiveShellQuartetTile tile =
      active_shell_quartet_tiles[active_tile];
  const std::int32_t first_shell = batch.shell_pair_first[tile.first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[tile.first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[tile.second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[tile.second_pair];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (shell_class < 64U &&
      (generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    return;
  }
  const std::uint32_t slot = atomicAdd(generic_tile_count, 1U);
  generic_tiles[slot] = tile;
}

/**
 * Summarize the exact tile list consumed by the final Fock and force kernels.
 *
 * The fixed grid walks topology capacity, but only slots below each compacted
 * angular partition's active count contribute. Profiling is opt-in, so these
 * atomics and the partition lookup never enter production timing runs.
 */
__global__ void profile_active_shell_quartet_tiles_kernel(
    DeviceBatch batch,
    std::size_t total_tile_capacity,
    const std::uint32_t* active_shell_quartet_tile_offsets,
    const std::uint32_t* active_shell_quartet_tile_counts,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    DeviceShellClassProfileEntry* profile) {
  const std::size_t slot =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= total_tile_capacity) return;

  unsigned angular_order = 0;
  while (angular_order + 1 < detail::kDirectQuartetAngularOrderCount &&
         slot >= active_shell_quartet_tile_offsets[angular_order + 1]) {
    ++angular_order;
  }
  const std::size_t partition_begin =
      active_shell_quartet_tile_offsets[angular_order];
  if (slot - partition_begin >=
      active_shell_quartet_tile_counts[angular_order]) {
    return;
  }

  const ActiveShellQuartetTile task = active_shell_quartet_tiles[slot];
  const std::int32_t first_shell = batch.shell_pair_first[task.first_pair];
  const std::int32_t second_shell = batch.shell_pair_second[task.first_pair];
  const std::int32_t third_shell = batch.shell_pair_first[task.second_pair];
  const std::int32_t fourth_shell = batch.shell_pair_second[task.second_pair];
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[first_shell], batch.shell_angular[second_shell],
      batch.shell_angular[third_shell], batch.shell_angular[fourth_shell]);
  if (shell_class >= detail::kDirectQuartetShellClassCount) return;

  const std::size_t first_ao_pair_count =
      shell_ao_pair_count(batch, task.first_pair);
  const std::size_t second_ao_pair_count =
      shell_ao_pair_count(batch, task.second_pair);
  const std::size_t ao_quartet_count = task.first_pair == task.second_pair
      ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
      : first_ao_pair_count * second_ao_pair_count;
  const std::size_t tile_begin =
      static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize;
  if (tile_begin >= ao_quartet_count) return;
  const std::size_t tile_ao_quartets =
      min(detail::kDirectQuartetTileSize, ao_quartet_count - tile_begin);

  unsigned long long primitive_quartets =
      static_cast<unsigned long long>(tile_ao_quartets);
  const std::int32_t shells[4] = {
      first_shell, second_shell, third_shell, fourth_shell};
  for (const std::int32_t shell : shells) {
    primitive_quartets *= static_cast<unsigned long long>(
        batch.shell_primitive_offsets[shell + 1] -
        batch.shell_primitive_offsets[shell]);
  }

  DeviceShellClassProfileEntry& entry = profile[shell_class];
  if (task.tile == 0) atomicAdd(&entry.shell_quartets, 1ULL);
  atomicAdd(&entry.tiles, 1ULL);
  atomicAdd(&entry.ao_quartets,
            static_cast<unsigned long long>(tile_ao_quartets));
  atomicAdd(&entry.primitive_quartets, primitive_quartets);
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

/** Test uniqueness directly from the canonical pair symmetries. */
__device__ bool unique_eri_symmetry_permutation(unsigned permutation,
                                                std::size_t i,
                                                std::size_t j,
                                                std::size_t k,
                                                std::size_t l) {
  const bool pair_swapped = permutation >= 4;
  const bool first_pair_diagonal = pair_swapped ? k == l : i == j;
  const bool second_pair_diagonal = pair_swapped ? i == j : k == l;
  if ((permutation & 1U) != 0 && first_pair_diagonal) return false;
  if ((permutation & 2U) != 0 && second_pair_diagonal) return false;
  return !pair_swapped || i != k || j != l;
}

/** Scatter one symmetry-canonical ERI into the direct RHF/UHF Fock matrix. */
template <bool Unrestricted>
__device__ __forceinline__ void accumulate_direct_fock_integral(
    std::size_t n,
    std::size_t physical_offset,
    std::size_t spin_offset,
    const double* density,
    double* fock,
    std::size_t i,
    std::size_t j,
    std::size_t k,
    std::size_t l,
    double integral) {
  const std::size_t matrix_size = n * n;
  for (unsigned permutation = 0; permutation < 8; ++permutation) {
    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) {
      continue;
    }
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::size_t d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);
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

template <bool Unrestricted,
          unsigned AngularOrder,
          typename EvalScalar = double>
__device__ __forceinline__ void contract_fock_direct_quartet_subtile(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock,
    const std::uint64_t* generated_fock_shell_class_mask,
    std::size_t active_subtile,
    unsigned ao_quartet_lane) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  constexpr std::size_t subtiles_per_tile =
      detail::direct_quartet_subtiles_per_tile(AngularOrder);
  const std::size_t active_tile =
      active_subtile / subtiles_per_tile;
  if (active_tile >=
      static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const std::size_t subtile =
      active_subtile % subtiles_per_tile;
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
  if (generated_fock_shell_class_mask != nullptr &&
      ((*generated_fock_shell_class_mask &
        (std::uint64_t{1} << shell_class)) != 0U)) {
    return;
  }

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
      subtile * detail::kDirectQuartetThreads + ao_quartet_lane;
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
    const EvalScalar evaluated_integral =
        dispatch_contracted_eri_cartesian_source_shell_class<
            AngularOrder, EvalScalar>(
            shell_class, batch, system, static_cast<std::int32_t>(i),
            static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
            static_cast<std::int32_t>(l), -1);
    const double integral = scalar_value(evaluated_integral);
    if (integral == 0.0) return;
    accumulate_direct_fock_integral<Unrestricted>(
        n, physical_offset, spin_offset, density, fock,
        i, j, k, l, integral);
  }
}

/**
 * Evaluate and scatter one complete order-one shell task.
 *
 * Shell-pair topology is index-canonical rather than angular-canonical, so
 * the p shell can occupy any input slot. Integral evaluation is reordered to
 * canonical (p s|s s), while screening and Fock scatter keep the original AO
 * slots to preserve the existing eightfold symmetry semantics.
 */
template <bool Unrestricted>
__device__ __noinline__ void contract_fock_direct_psss_task(
    const DeviceBatch& batch,
    ActiveShellQuartetTile task,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock) {
  // A psss shell quartet has three AO outputs and therefore exactly one tile.
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active != nullptr && active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  unsigned p_slot = 4;
  unsigned p_count = 0;
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[raw_shell[slot]];
    if (angular == 1) {
      p_slot = slot;
      ++p_count;
    } else if (angular != 0) {
      return;
    }
  }
  if (p_count != 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  std::size_t raw_ao[4] = {
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[raw_shell[0]]) - system_ao_begin,
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[raw_shell[1]]) - system_ao_begin,
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[raw_shell[2]]) - system_ao_begin,
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[raw_shell[3]]) - system_ao_begin,
  };
  const std::size_t p_ao_begin = raw_ao[p_slot];
  unsigned active_axis_mask = 0;
  for (unsigned axis = 0; axis < 3; ++axis) {
    raw_ao[p_slot] = p_ao_begin + axis;
    const double first_bound = schwarz_bounds[
        physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)];
    const double second_bound = schwarz_bounds[
        physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)];
    if (first_bound * second_bound >= screening_tolerance) {
      active_axis_mask |= 1U << axis;
    }
  }
  if (active_axis_mask == 0) return;

  std::int32_t canonical_shell[4] = {
      raw_shell[0], raw_shell[1], raw_shell[2], raw_shell[3]};
  std::size_t canonical_pair[2] = {first_pair, second_pair};
  if (p_slot == 1) {
    const std::int32_t swap = canonical_shell[0];
    canonical_shell[0] = canonical_shell[1];
    canonical_shell[1] = swap;
  } else if (p_slot >= 2) {
    if (p_slot == 3) {
      const std::int32_t swap = canonical_shell[2];
      canonical_shell[2] = canonical_shell[3];
      canonical_shell[3] = swap;
    }
    const std::int32_t first_swap = canonical_shell[0];
    canonical_shell[0] = canonical_shell[2];
    canonical_shell[2] = first_swap;
    const std::int32_t second_swap = canonical_shell[1];
    canonical_shell[1] = canonical_shell[3];
    canonical_shell[3] = second_swap;
    const std::size_t pair_swap = canonical_pair[0];
    canonical_pair[0] = canonical_pair[1];
    canonical_pair[1] = pair_swap;
  }

  const PsssIntegralVector integral = contracted_eri_cartesian_source_psss(
      batch, canonical_pair[0], canonical_pair[1],
      canonical_shell[0], canonical_shell[1],
      canonical_shell[2], canonical_shell[3]);
  for (unsigned axis = 0; axis < 3; ++axis) {
    if ((active_axis_mask & (1U << axis)) == 0 ||
        integral.axis[axis] == 0.0) {
      continue;
    }
    raw_ao[p_slot] = p_ao_begin + axis;
    accumulate_direct_fock_integral<Unrestricted>(
        n, physical_offset, spin_offset, density, fock,
        raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3], integral.axis[axis]);
  }
}

/** One canonical shell slot and its position in the original quartet. */
struct Order2SourceSlot {
  std::int32_t shell;
  unsigned original;
};

/** Map raw AO slots to the fused vector's canonical Cartesian product. */
__device__ __forceinline__ unsigned order2_component_index(
    const DeviceBatch& batch,
    const Order2SourceSlot (&slots)[4],
    const std::size_t (&raw_ao)[4],
    std::size_t system_ao_begin) {
  unsigned output = 0;
#pragma unroll
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[slots[slot].shell];
    const unsigned component_count = (angular + 1) * (angular + 2) / 2;
    const std::size_t local_begin = static_cast<std::size_t>(
        batch.shell_direct_ao_offsets[slots[slot].shell]) - system_ao_begin;
    const unsigned component = static_cast<unsigned>(
        raw_ao[slots[slot].original] - local_begin);
    output = output * component_count + component;
  }
  return output;
}

/** Evaluate and scatter one complete psps, ppss, or dsss shell task. */
template <bool Unrestricted>
__device__ __noinline__ void contract_fock_direct_order2_task(
    const DeviceBatch& batch,
    ActiveShellQuartetTile task,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active != nullptr && active[system] == 0) return;

  Order2SourceSlot slots[4] = {
      {batch.shell_pair_first[first_pair], 0},
      {batch.shell_pair_second[first_pair], 1},
      {batch.shell_pair_first[second_pair], 2},
      {batch.shell_pair_second[second_pair], 3},
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[slots[0].shell],
      batch.shell_angular[slots[1].shell],
      batch.shell_angular[slots[2].shell],
      batch.shell_angular[slots[3].shell]);
  if (shell_class != 2U && shell_class != 3U && shell_class != 6U) return;
  // Generated order-two workers execute before this handwritten fallback.
  // Honor the exact-class mask here as the generic subtile path does, or the
  // same shell quartet is scattered into the Fock matrix twice.
  if (generated_fock_shell_class_mask != nullptr &&
      ((*generated_fock_shell_class_mask &
        (std::uint64_t{1} << shell_class)) != 0U)) {
    return;
  }

  if (batch.shell_angular[slots[0].shell] <
      batch.shell_angular[slots[1].shell]) {
    const Order2SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] <
      batch.shell_angular[slots[3].shell]) {
    const Order2SourceSlot swap = slots[2];
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
    const Order2SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const Order2SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

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

  unsigned active_component_mask = 0;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(
        batch, first_pair, first_ao_pair, system_ao_begin,
        raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(
        batch, second_pair, second_ao_pair, system_ao_begin,
        raw_ao[2], raw_ao[3]);
    if (schwarz_bounds[
            physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[
                physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    active_component_mask |=
        1U << order2_component_index(
            batch, slots, raw_ao, system_ao_begin);
  }
  if (active_component_mask == 0) return;

  Order2IntegralVector integral{};
  switch (shell_class) {
    case 2:
      integral = contracted_eri_cartesian_source_order2_shell<1, 0, 1, 0>(
          batch, slots[0].shell, slots[1].shell,
          slots[2].shell, slots[3].shell, active_component_mask);
      break;
    case 3:
      integral = contracted_eri_cartesian_source_order2_shell<1, 1, 0, 0>(
          batch, slots[0].shell, slots[1].shell,
          slots[2].shell, slots[3].shell, active_component_mask);
      break;
    default:
      integral = contracted_eri_cartesian_source_order2_shell<2, 0, 0, 0>(
          batch, slots[0].shell, slots[1].shell,
          slots[2].shell, slots[3].shell, active_component_mask);
      break;
  }

  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(
        batch, first_pair, first_ao_pair, system_ao_begin,
        raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(
        batch, second_pair, second_ao_pair, system_ao_begin,
        raw_ao[2], raw_ao[3]);
    const unsigned component = order2_component_index(
        batch, slots, raw_ao, system_ao_begin);
    if ((active_component_mask & (1U << component)) == 0 ||
        integral.component[component] == 0.0) {
      continue;
    }
    accumulate_direct_fock_integral<Unrestricted>(
        n, physical_offset, spin_offset, density, fock,
        raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3],
        integral.component[component]);
  }
}

/** Fixed-capacity wrapper retained for high-register angular orders. */
template <bool Unrestricted,
          unsigned AngularOrder,
          typename EvalScalar = double>
__global__ void build_fock_direct_quartet_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  contract_fock_direct_quartet_subtile<
      Unrestricted, AngularOrder, EvalScalar>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles,
      screening_tolerance, schwarz_bounds, density, active, fock,
      generated_fock_shell_class_mask,
      static_cast<std::size_t>(blockIdx.x), threadIdx.x);
}

/** Pack exact ssss shell tasks across all lanes of one worker warp. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void build_fock_direct_quartet_packed_persistent_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint32_t* task_head,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
  double* fock) {
  static_assert(AngularOrder < kPackedSsssAngularOrderCount);
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(
          task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      // Order zero has one subtile and one exact logical tile per shell
      // quartet, so packed_item is also its compact subtile index.
      contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder>(
          batch, active_shell_quartet_tile_count,
          active_shell_quartet_tiles, screening_tolerance, schwarz_bounds,
          density, active, fock, nullptr, packed_item, 0U);
    }
  }
}

/** Consume complete psss shell tasks, one independent task per lane. */
template <bool Unrestricted>
__global__ void build_fock_direct_psss_persistent_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint32_t* task_head,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock) {
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(
          task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_fock_direct_psss_task<Unrestricted>(
          batch, active_shell_quartet_tiles[packed_item],
          screening_tolerance, schwarz_bounds, density, active, fock);
    }
    // Tail lanes must remain live until the next warp-uniform queue exit so
    // the full-mask shuffle above is valid on every persistent iteration.
  }
}

/** Consume complete order-two shell tasks, one independent task per lane. */
template <bool Unrestricted>
__global__ void build_fock_direct_order2_persistent_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint32_t* task_head,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(
          task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_fock_direct_order2_task<Unrestricted>(
          batch, active_shell_quartet_tiles[packed_item],
          screening_tolerance, schwarz_bounds, density, active, fock,
          generated_fock_shell_class_mask);
    }
  }
}

/** Consume only the active compacted Fock domain from a device queue. */
template <bool Unrestricted,
          unsigned AngularOrder,
          typename EvalScalar = double>
__global__ void build_fock_direct_quartet_persistent_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint32_t* task_head,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  const unsigned lane = threadIdx.x % warpSize;
  constexpr std::uint32_t subtiles_per_tile = static_cast<std::uint32_t>(
      detail::direct_quartet_subtiles_per_tile(AngularOrder));
  const std::uint32_t work_count =
      *active_shell_quartet_tile_count * subtiles_per_tile;
  while (true) {
    std::uint32_t active_subtile = 0;
    if (lane == 0) active_subtile = atomicAdd(task_head, 1U);
    active_subtile = __shfl_sync(0xffffffffU, active_subtile, 0);
    if (active_subtile >= work_count) return;
    contract_fock_direct_quartet_subtile<
        Unrestricted, AngularOrder, EvalScalar>(
        batch, active_shell_quartet_tile_count, active_shell_quartet_tiles,
        screening_tolerance, schwarz_bounds, density, active, fock,
        generated_fock_shell_class_mask,
        active_subtile, threadIdx.x);
  }
}

/** Subtract the second GEMM product from the first in a batched matrix set. */
__global__ void subtract_matrix_batches_kernel(
    std::int32_t batch_size,
    std::int32_t matrices_per_system,
    std::int32_t nbf,
    const double* subtract,
    const std::uint8_t* active,
    double* minuend) {
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(batch_size) *
                            static_cast<std::size_t>(matrices_per_system) *
                            matrix_size;
  if (element >= total) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system =
      state / static_cast<std::size_t>(matrices_per_system);
  if (active != nullptr && active[system] == 0) return;
  minuend[element] -= subtract[element];
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
  // One warp owns one system.  History vectors and the O(N^2) residual-dot
  // products are distributed across lanes, while the small dense DIIS solve
  // remains in lane zero.  This preserves the original dot-product order for
  // each B-matrix entry and avoids the old single-thread N^2 bottleneck.
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t vector_size =
      matrix_size * static_cast<std::size_t>(matrices_per_system);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * vector_size;
  if (history_capacity < 2) {
    for (std::size_t element = threadIdx.x; element < vector_size;
         element += blockDim.x) {
      effective_fock[matrix_offset + element] = fock[matrix_offset + element];
    }
    return;
  }

  const std::size_t history_stride =
      static_cast<std::size_t>(history_capacity) * vector_size;
  std::uint32_t slot = 0;
  if (threadIdx.x == 0) slot = history_head[system];
  slot = __shfl_sync(0xffffffffU, slot, 0);
  const std::size_t slot_offset =
      static_cast<std::size_t>(system) * history_stride +
      static_cast<std::size_t>(slot) * vector_size;
  for (std::size_t element = threadIdx.x; element < vector_size;
       element += blockDim.x) {
    fock_history[slot_offset + element] = fock[matrix_offset + element];
    residual_history[slot_offset + element] = residual[matrix_offset + element];
  }
  __syncwarp();
  std::uint32_t count = 0;
  if (threadIdx.x == 0) {
    count = history_count[system] < history_capacity
        ? history_count[system] + 1
        : history_capacity;
    history_count[system] = count;
    history_head[system] = (slot + 1) % history_capacity;
  }
  count = __shfl_sync(0xffffffffU, count, 0);
  if (count < 2) {
    for (std::size_t element = threadIdx.x; element < vector_size;
         element += blockDim.x) {
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
  const std::size_t linear_elements =
      static_cast<std::size_t>(dimension) * dimension;
  for (std::size_t element = threadIdx.x; element < linear_elements;
       element += blockDim.x) {
    matrix[element] = 0.0;
  }
  for (std::uint32_t row = threadIdx.x; row < dimension;
       row += blockDim.x) {
    rhs[row] = row == count ? -1.0 : 0.0;
  }
  __syncwarp();
  const std::size_t dot_count = static_cast<std::size_t>(count) * count;
  for (std::size_t pair = threadIdx.x; pair < dot_count;
       pair += blockDim.x) {
    const std::uint32_t row = static_cast<std::uint32_t>(pair / count);
    const std::uint32_t column = static_cast<std::uint32_t>(pair % count);
    const std::size_t row_offset =
        static_cast<std::size_t>(system) * history_stride +
        static_cast<std::size_t>(row) * vector_size;
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
  __syncwarp();
  if (threadIdx.x == 0) {
    for (std::uint32_t row = 0; row < count; ++row) {
      matrix[static_cast<std::size_t>(row) * dimension + count] = -1.0;
      matrix[static_cast<std::size_t>(count) * dimension + row] = -1.0;
    }
  }
  __syncwarp();

  int nonsingular = 1;
  if (threadIdx.x == 0) {
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
        nonsingular = 0;
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
  }
  __syncwarp();
  // The solve is lane-zero-only; broadcast its success flag before any lane
  // decides whether it should form the extrapolated Fock matrix.
  nonsingular = __shfl_sync(0xffffffffU, nonsingular, 0);

  for (std::size_t element = threadIdx.x; element < vector_size;
       element += blockDim.x) {
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
  // One warp owns one system.  The previous one-thread-per-system mapping
  // made the N^2 contraction and its global-memory latency completely serial
  // at large AO counts; all callers launch exactly one 32-thread block per
  // system, which also keeps this graph-capture-safe.
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || (active != nullptr && active[system] == 0)) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double value = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_size;
       element += blockDim.x) {
    value += 0.5 * density[offset + element] *
             (hcore[offset + element] + fock[offset + element]);
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (threadIdx.x == 0) energy[system] = nuclear_repulsion[system] + value;
}

__global__ void compute_uhf_energy_kernel(std::int32_t batch_size,
                                          std::int32_t nbf,
                                          const double* density,
                                          const double* hcore,
                                          const double* fock,
                                          const double* nuclear_repulsion,
                                          const std::uint8_t* active,
                                          double* energy) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || (active != nullptr && active[system] == 0)) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t alpha_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  double value = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_size;
       element += blockDim.x) {
    value += 0.5 * density[alpha_offset + element] *
             (hcore[physical_offset + element] + fock[alpha_offset + element]);
    value += 0.5 * density[beta_offset + element] *
             (hcore[physical_offset + element] + fock[beta_offset + element]);
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (threadIdx.x == 0) energy[system] = nuclear_repulsion[system] + value;
}

/** Comparison guard for the nondeterministic FP64 direct-Fock reduction. */
__device__ __forceinline__ double direct_fock_energy_roundoff_guard(
    bool enabled,
    double energy,
    double previous_energy) {
  if (!enabled || !isfinite(previous_energy)) return 0.0;
  const double energy_scale =
      fmax(1.0, fmax(fabs(energy), fabs(previous_energy)));
  return kDirectFockEnergyRoundoffFactor * kDoubleMachineEpsilon *
      energy_scale;
}

template <bool RetainConvergedDensity>
__global__ void update_convergence_kernel(std::int32_t batch_size,
                                          std::int32_t nbf,
                                          double energy_tolerance,
                                          double density_tolerance,
                                          bool guard_direct_fock_roundoff,
                                          const double* energy,
                                          double* previous_energy,
                                          const double* next_density,
                                          double* density,
                                          std::uint8_t* active,
                                          std::uint8_t* converged,
                                          std::uint32_t* iterations,
                                          double* energy_change,
                                          double* density_rms) {
  // A warp owns one system.  This is intentionally a one-warp block because
  // all scalar state transitions are performed by lane zero after the warp
  // reduction; the matrix walk itself is spread over the 32 lanes.
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double square = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_size;
       element += blockDim.x) {
    const double delta = next_density[offset + element] - density[offset + element];
    square += delta * delta;
    if constexpr (!RetainConvergedDensity) {
      density[offset + element] = next_density[offset + element];
    }
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    square += __shfl_down_sync(0xffffffffU, square, delta);
  }
  int copy_next_density = 0;
  if (threadIdx.x == 0) {
    const std::uint32_t iteration = iterations[system] + 1;
    const bool has_energy_baseline = isfinite(previous_energy[system]);
    const double change = has_energy_baseline
        ? fabs(energy[system] - previous_energy[system])
        : CUDART_INF;
    const double roundoff_guard = direct_fock_energy_roundoff_guard(
        guard_direct_fock_roundoff, energy[system], previous_energy[system]);
    const double rms = sqrt(square / static_cast<double>(matrix_size));
    iterations[system] = iteration;
    energy_change[system] = change;
    density_rms[system] = rms;
    const bool did_converge =
        (iteration > 1 || has_energy_baseline) &&
        change < energy_tolerance + roundoff_guard &&
        rms < density_tolerance;
    if (did_converge) {
      converged[system] = 1;
      active[system] = 0;
    } else {
      previous_energy[system] = energy[system];
      copy_next_density = 1;
    }
  }
  copy_next_density = __shfl_sync(0xffffffffU, copy_next_density, 0);
  if constexpr (RetainConvergedDensity) {
    // The raw Fock matrix still corresponds to P_n. Advance to P_{n+1} only
    // when another SCF iteration is required, so finalization can reuse the
    // already computed F(P_n) after convergence instead of rebuilding it.
    if (copy_next_density != 0) {
      for (std::size_t element = threadIdx.x; element < matrix_size;
           element += blockDim.x) {
        density[offset + element] = next_density[offset + element];
      }
    }
  }
}

template <bool RetainConvergedDensity>
__global__ void update_uhf_convergence_kernel(
    std::int32_t batch_size,
    std::int32_t nbf,
    double energy_tolerance,
    double density_tolerance,
    bool guard_direct_fock_roundoff,
    const double* energy,
    double* previous_energy,
    const double* next_density,
    double* density,
    std::uint8_t* active,
    std::uint8_t* converged,
                                          std::uint32_t* iterations,
                                          double* energy_change,
                                          double* density_rms) {
  // Keep UHF's two spin matrices under one warp so the convergence reduction
  // and scalar state transition have the same ordering as RHF.
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t vector_size = 2 * matrix_size;
  const std::size_t offset = static_cast<std::size_t>(system) * vector_size;
  double square = 0.0;
  for (std::size_t element = threadIdx.x; element < vector_size;
       element += blockDim.x) {
    const double delta = next_density[offset + element] - density[offset + element];
    square += delta * delta;
    if constexpr (!RetainConvergedDensity) {
      density[offset + element] = next_density[offset + element];
    }
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    square += __shfl_down_sync(0xffffffffU, square, delta);
  }
  int copy_next_density = 0;
  if (threadIdx.x == 0) {
    const bool has_energy_baseline = isfinite(previous_energy[system]);
    const double change = has_energy_baseline
        ? fabs(energy[system] - previous_energy[system])
        : CUDART_INF;
    const double roundoff_guard = direct_fock_energy_roundoff_guard(
        guard_direct_fock_roundoff, energy[system], previous_energy[system]);
    const double rms = sqrt(square / static_cast<double>(vector_size));
    // Preserve the existing UHF baseline update semantics, including the
    // converged iteration, because it is observable by the next warm replay.
    previous_energy[system] = energy[system];
    energy_change[system] = change;
    density_rms[system] = rms;
    const std::uint32_t iteration = ++iterations[system];
    const bool did_converge =
        (iteration > 1 || has_energy_baseline) &&
        change < energy_tolerance + roundoff_guard &&
        rms < density_tolerance;
    if (did_converge) {
      converged[system] = 1;
      active[system] = 0;
    } else {
      copy_next_density = 1;
    }
  }
  copy_next_density = __shfl_sync(0xffffffffU, copy_next_density, 0);
  if constexpr (RetainConvergedDensity) {
    // Preserve each system's spin densities paired with its raw alpha/beta
    // Fock matrices until per-system finalization selects reuse or rebuild.
    if (copy_next_density != 0) {
      for (std::size_t element = threadIdx.x; element < vector_size;
           element += blockDim.x) {
        density[offset + element] = next_density[offset + element];
      }
    }
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

/**
 * Partition converged systems between retained-Fock reuse and exact rebuild.
 *
 * A converged system still owns P_n/F(P_n) because the templated convergence
 * kernel did not advance its density. Only a looser final step restores
 * P_{n+1} and becomes active for the legacy Fock builder. The reuse mask is
 * retained until forces finish so the accepted P_{n+1} warm state can then be
 * restored independently for every system in the bucket.
 */
__global__ void select_final_fock_rebuild_kernel(
    std::int32_t batch_size,
    double reuse_density_rms,
    const double* density_rms,
    const std::uint8_t* converged,
    const std::uint8_t* failed,
    std::uint8_t* reuse_mask,
    std::uint8_t* active,
    std::uint32_t* rebuild_count) {
  const std::int32_t system = static_cast<std::int32_t>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch_size) return;
  const bool valid = converged[system] == 1 && failed[system] == 0;
  const bool reuse = valid && density_rms[system] <= reuse_density_rms;
  reuse_mask[system] = reuse ? 1 : 0;
  active[system] = valid && !reuse ? 1 : 0;
  if (active[system] != 0) atomicAdd(rebuild_count, 1U);
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

__global__ void one_electron_force_scalar_kernel(
    DeviceBatch batch,
    const std::int32_t* pair_first,
    const std::int32_t* pair_second,
    std::size_t pair_count,
    const double* density,
    const double* weighted_density,
    const std::uint8_t* active,
    double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system =
      static_cast<std::int32_t>(element / pair_count);
  const std::size_t local = element % pair_count;
  if (active[system] == 0) return;
  const std::size_t i = static_cast<std::size_t>(pair_first[local]);
  const std::size_t j = static_cast<std::size_t>(pair_second[local]);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double pij = density[matrix_offset + matrix_index(i, j, n)];
  const double wij = weighted_density[matrix_offset + matrix_index(i, j, n)];
  if (pij == 0.0 && wij == 0.0) return;
  const std::int64_t ao_i = static_cast<std::int64_t>(system) * batch.nbf +
      static_cast<std::int32_t>(i);
  const std::int64_t ao_j = static_cast<std::int64_t>(system) * batch.nbf +
      static_cast<std::int32_t>(j);
  const unsigned maximum =
      batch.shell_angular[batch.ao_shells[ao_i]] +
      batch.shell_angular[batch.ao_shells[ao_j]] + 1U;
#define VIBEQC_ONE_ELECTRON_FORCE_CASE(Order)                              \
  case Order:                                                          \
    contracted_one_electron_force_pair<Order>(                         \
        batch, system, static_cast<std::int32_t>(i),                    \
        static_cast<std::int32_t>(j), pij, wij, forces);                \
    break
  switch (maximum) {
    VIBEQC_ONE_ELECTRON_FORCE_CASE(1);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(2);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(3);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(4);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(5);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(6);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(7);
  }
#undef VIBEQC_ONE_ELECTRON_FORCE_CASE
}

/**
 * Treat nuclei as a prepared point-charge auxiliary dimension.
 *
 * One warp owns one AO pair and reuses a shared primitive-pair Hermite table
 * while its lanes evaluate independent nuclear centers. This avoids a host or
 * device launch per nucleus and preserves the public-basis density contraction
 * used by the scalar accuracy oracle.
 */
__global__ void one_electron_force_cooperative_kernel(
    DeviceBatch batch,
    const std::int32_t* pair_first,
    const std::int32_t* pair_second,
    std::size_t pair_count,
    const double* density,
    const double* weighted_density,
    const std::uint8_t* active,
    double* forces) {
  extern __shared__ double one_electron_shared[];
  if (blockDim.x != warpSize) return;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x);
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system =
      static_cast<std::int32_t>(element / pair_count);
  const std::size_t local = element % pair_count;
  if (active[system] == 0) return;
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t i = static_cast<std::size_t>(pair_first[local]);
  const std::size_t j = static_cast<std::size_t>(pair_second[local]);
  const std::size_t matrix_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const double pij = density[matrix_offset + matrix_index(i, j, n)];
  const double wij =
      weighted_density[matrix_offset + matrix_index(i, j, n)];
  if (pij == 0.0 && wij == 0.0) return;
  const std::int64_t ao_i = static_cast<std::int64_t>(system) * batch.nbf +
      static_cast<std::int32_t>(i);
  const std::int64_t ao_j = static_cast<std::int64_t>(system) * batch.nbf +
      static_cast<std::int32_t>(j);
  const unsigned maximum =
      batch.shell_angular[batch.ao_shells[ao_i]] +
      batch.shell_angular[batch.ao_shells[ao_j]] + 1U;
  auto* shared_coefficients = reinterpret_cast<
      OneElectronDerivativeHermiteCoefficients*>(one_electron_shared);
#define VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(Order)                  \
  case Order:                                                              \
    contracted_one_electron_force_pair_cooperative<Order>(                 \
        batch, system, static_cast<std::int32_t>(i),                        \
        static_cast<std::int32_t>(j), pij, wij, forces,                     \
        shared_coefficients);                                               \
    break
  switch (maximum) {
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(1);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(2);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(3);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(4);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(5);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(6);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(7);
  }
#undef VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE
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

/** Exact symmetry-reduced density coefficient for one force AO quartet. */
template <bool Unrestricted>
__device__ __forceinline__ double direct_force_density_coefficient(
    std::size_t n,
    std::size_t physical_offset,
    std::size_t spin_offset,
    const double* density,
    std::size_t i,
    std::size_t j,
    std::size_t k,
    std::size_t l) {
  const std::size_t matrix_size = n * n;
  double coefficient = 0.0;
  for (unsigned permutation = 0; permutation < 8; ++permutation) {
    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) {
      continue;
    }
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::size_t d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);
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
  return coefficient;
}

/** Evaluate and write one complete density-weighted ssss force shell task. */
template <bool Unrestricted>
__device__ __forceinline__ void contract_two_electron_force_ssss_task(
    const DeviceBatch& batch,
    ActiveShellQuartetTile task,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  // Every s shell contains one Cartesian AO, so a valid ssss shell quartet
  // occupies exactly the first compact tile and needs no AO-pair decoding.
  if (task.tile != 0U) return;
  if ((generated_shell_class_mask & std::uint64_t{1}) != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t shells[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  for (unsigned slot = 0; slot < 4; ++slot) {
    if (batch.shell_angular[shells[slot]] != 0U) return;
  }

  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[shells[0]], batch.shell_atoms[shells[1]],
      batch.shell_atoms[shells[2]], batch.shell_atoms[shells[3]],
  };
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
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t ao[4] = {
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[0]]) -
          system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[1]]) -
          system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[2]]) -
          system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[shells[3]]) -
          system_ao_begin,
  };
  if (schwarz_bounds[physical_offset + matrix_index(ao[0], ao[1], n)] *
          schwarz_bounds[physical_offset + matrix_index(ao[2], ao[3], n)] <
      screening_tolerance) {
    return;
  }
  const double density_coefficient =
      direct_force_density_coefficient<Unrestricted>(
          n, physical_offset, spin_offset, density,
          ao[0], ao[1], ao[2], ao[3]);
  if (density_coefficient == 0.0) return;
  const double component_weight = density_coefficient *
      batch.direct_ao_coefficients[system_ao_begin + ao[0]] *
      batch.direct_ao_coefficients[system_ao_begin + ao[1]] *
      batch.direct_ao_coefficients[system_ao_begin + ao[2]] *
      batch.direct_ao_coefficients[system_ao_begin + ao[3]];
  const SsssWeightedGradient gradient =
      contracted_eri_cartesian_source_ssss_weighted_gradient(
          batch, first_pair, second_pair, shells[0], shells[1], shells[2],
          shells[3], component_weight);

  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate =
        static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned center = 0; center < 3; ++center) {
        const double value = gradient.center[center][axis];
        fourth_derivative -= value;
        if (center_atoms[center] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (center_atoms[3] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) *
      3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

/** Evaluate and write one complete density-weighted psss force shell task. */
template <bool Unrestricted, bool ResidentBra = false>
__device__ __noinline__ void contract_two_electron_force_psss_task(
    const DeviceBatch& batch,
    ActiveShellQuartetTile task,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask,
    const PrimitivePairData* resident_first_pairs = nullptr,
    std::int64_t resident_first_pair_count = 0) {
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class < 64U &&
      (generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    return;
  }
  unsigned p_slot = 4;
  unsigned p_count = 0;
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[raw_shell[slot]];
    if (angular == 1) {
      p_slot = slot;
      ++p_count;
    } else if (angular != 0) {
      return;
    }
  }
  if (p_count != 1) return;

  const std::int32_t center_atoms[4] = {
      batch.shell_atoms[raw_shell[0]], batch.shell_atoms[raw_shell[1]],
      batch.shell_atoms[raw_shell[2]], batch.shell_atoms[raw_shell[3]],
  };
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
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  std::size_t raw_ao[4] = {
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[raw_shell[0]]) - system_ao_begin,
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[raw_shell[1]]) - system_ao_begin,
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[raw_shell[2]]) - system_ao_begin,
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[raw_shell[3]]) - system_ao_begin,
  };
  const std::size_t p_ao_begin = raw_ao[p_slot];
  double density_coefficient[3]{};
  for (unsigned axis = 0; axis < 3; ++axis) {
    raw_ao[p_slot] = p_ao_begin + axis;
    if (schwarz_bounds[
            physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[
                physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    density_coefficient[axis] =
        direct_force_density_coefficient<Unrestricted>(
            n, physical_offset, spin_offset, density,
            raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3]);
  }
  if (density_coefficient[0] == 0.0 &&
      density_coefficient[1] == 0.0 &&
      density_coefficient[2] == 0.0) {
    return;
  }

  std::int32_t slots[4] = {
      raw_shell[0], raw_shell[1], raw_shell[2], raw_shell[3]};
  std::size_t canonical_pair[2] = {first_pair, second_pair};
  if (p_slot == 1) {
    const std::int32_t swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  } else if (p_slot >= 2) {
    if (p_slot == 3) {
      const std::int32_t swap = slots[2];
      slots[2] = slots[3];
      slots[3] = swap;
    }
    const std::int32_t first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const std::int32_t second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
    const std::size_t pair_swap = canonical_pair[0];
    canonical_pair[0] = canonical_pair[1];
    canonical_pair[1] = pair_swap;
  }

  const PsssWeightedGradient gradient =
      contracted_eri_cartesian_source_psss_weighted_gradient<ResidentBra>(
          batch, canonical_pair[0], canonical_pair[1],
          slots[0], slots[1], slots[2], slots[3],
          density_coefficient, resident_first_pairs,
          resident_first_pair_count);
  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate =
        static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned canonical = 0; canonical < 3; ++canonical) {
        const double value = gradient.center[canonical][axis];
        fourth_derivative -= value;
        if (batch.shell_atoms[slots[canonical]] ==
            unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (batch.shell_atoms[slots[3]] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(
          unique_center_atoms[unique_center_count - 1]) * 3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

/** Evaluate and write one complete density-weighted psps force shell task. */
template <bool Unrestricted>
__device__ __noinline__ void contract_two_electron_force_psps_task(
    const DeviceBatch& batch,
    ActiveShellQuartetTile task,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  // A psps shell quartet has at most nine Cartesian AO quartets and therefore
  // always fits in the first compact tile.
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class != kPspsShellClass) return;
  if ((generated_shell_class_mask &
       (std::uint64_t{1} << kPspsShellClass)) != 0U) {
    return;
  }

  unsigned first_p_slot = 4;
  unsigned second_p_slot = 4;
  for (unsigned slot = 0; slot < 2; ++slot) {
    if (batch.shell_angular[raw_shell[slot]] == 1U) first_p_slot = slot;
  }
  for (unsigned slot = 2; slot < 4; ++slot) {
    if (batch.shell_angular[raw_shell[slot]] == 1U) second_p_slot = slot;
  }
  if (first_p_slot >= 2 || second_p_slot < 2 || second_p_slot >= 4) return;
  const unsigned canonical_raw_slot[4] = {
      first_p_slot, 1U - first_p_slot,
      second_p_slot, 5U - second_p_slot,
  };
  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]], raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]], raw_shell[canonical_raw_slot[3]],
  };

  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    const std::int32_t atom = batch.shell_atoms[canonical_shell[center]];
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center ||
          atom == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = atom;
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_p_ao_begin =
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[canonical_shell[0]]) -
      system_ao_begin;
  const std::size_t second_p_ao_begin =
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[canonical_shell[2]]) -
      system_ao_begin;
  const std::size_t first_ao_pair_count =
      shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count =
      shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count = same_shell_pair
      ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
      : first_ao_pair_count * second_ao_pair_count;

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch, first_pair, first_ao_pair, system_ao_begin,
                         raw_ao[0], raw_ao[1]);
    decode_shell_ao_pair(batch, second_pair, second_ao_pair, system_ao_begin,
                         raw_ao[2], raw_ao[3]);
    if (schwarz_bounds[
            physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[
                physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    const double density_coefficient =
        direct_force_density_coefficient<Unrestricted>(
            n, physical_offset, spin_offset, density,
            raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3]);
    if (density_coefficient == 0.0) continue;
    const unsigned first_axis =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[0]] - first_p_ao_begin);
    const unsigned second_axis = static_cast<unsigned>(
        raw_ao[canonical_raw_slot[2]] - second_p_ao_begin);
    if (first_axis >= 3 || second_axis >= 3)
      return;
    const double angular_coefficient =
        batch.direct_ao_coefficients[system_ao_begin + raw_ao[0]] *
        batch.direct_ao_coefficients[system_ao_begin + raw_ao[1]] *
        batch.direct_ao_coefficients[system_ao_begin + raw_ao[2]] *
        batch.direct_ao_coefficients[system_ao_begin + raw_ao[3]];
    component_weight[first_axis * 3 + second_axis] +=
        density_coefficient * angular_coefficient;
    any_component = true;
  }
  if (!any_component)
    return;

  const PspsWeightedGradient gradient =
      contracted_eri_cartesian_source_psps_weighted_gradient(
          batch, first_pair, second_pair, canonical_shell[0],
          canonical_shell[1], canonical_shell[2], canonical_shell[3],
          component_weight);
  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate =
        static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned canonical = 0; canonical < 3; ++canonical) {
        const double value = gradient.center[canonical][axis];
        fourth_derivative -= value;
        if (batch.shell_atoms[canonical_shell[canonical]] ==
            unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (batch.shell_atoms[canonical_shell[3]] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) *
      3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

/** Evaluate one closed ppss or dsss shell task over its exact AO domain. */
template <bool Unrestricted, unsigned TargetShellClass>
__device__ __noinline__ void contract_two_electron_force_pair_order2_task(
    const DeviceBatch& batch,
    ActiveShellQuartetTile task,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  static_assert(TargetShellClass == kPpssShellClass ||
                TargetShellClass == kDsssShellClass);
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class =
      direct_quartet_shell_class_device(batch.shell_angular[raw_shell[0]],
                                        batch.shell_angular[raw_shell[1]],
                                        batch.shell_angular[raw_shell[2]],
                                        batch.shell_angular[raw_shell[3]]);
  if (shell_class != TargetShellClass) return;
  if ((generated_shell_class_mask & (std::uint64_t{1} << TargetShellClass)) !=
      0U) {
    return;
  }

  unsigned canonical_raw_slot[4];
  if constexpr (TargetShellClass == kPpssShellClass) {
    const bool first_pair_is_pp = batch.shell_angular[raw_shell[0]] == 1U &&
                                  batch.shell_angular[raw_shell[1]] == 1U;
    const unsigned pair_begin = first_pair_is_pp ? 0U : 2U;
    const unsigned other_pair_begin = first_pair_is_pp ? 2U : 0U;
    canonical_raw_slot[0] = pair_begin;
    canonical_raw_slot[1] = pair_begin + 1U;
    canonical_raw_slot[2] = other_pair_begin;
    canonical_raw_slot[3] = other_pair_begin + 1U;
  } else {
    unsigned d_slot = 4U;
    for (unsigned slot = 0; slot < 4; ++slot) {
      if (batch.shell_angular[raw_shell[slot]] == 2U) d_slot = slot;
    }
    if (d_slot >= 4U) return;
    const unsigned pair_begin = d_slot < 2U ? 0U : 2U;
    const unsigned other_pair_begin = pair_begin == 0U ? 2U : 0U;
    canonical_raw_slot[0] = d_slot;
    canonical_raw_slot[1] = pair_begin + (d_slot == pair_begin ? 1U : 0U);
    canonical_raw_slot[2] = other_pair_begin;
    canonical_raw_slot[3] = other_pair_begin + 1U;
  }
  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]],
      raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]],
      raw_shell[canonical_raw_slot[3]],
  };
  const std::size_t canonical_pair[2] = {
      canonical_raw_slot[0] < 2U ? first_pair : second_pair,
      canonical_raw_slot[2] < 2U ? first_pair : second_pair,
  };

  std::int32_t unique_center_atoms[4];
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    const std::int32_t atom = batch.shell_atoms[canonical_shell[center]];
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center =
          duplicate_center || atom == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = atom;
    }
  }
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset =
      static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset =
      static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_component_begin =
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[canonical_shell[0]]) -
      system_ao_begin;
  const std::size_t second_component_begin =
      static_cast<std::size_t>(
          batch.shell_direct_ao_offsets[canonical_shell[1]]) -
      system_ao_begin;
  const std::size_t first_ao_pair_count =
      shell_ao_pair_count(batch, first_pair);
  const std::size_t second_ao_pair_count =
      shell_ao_pair_count(batch, second_pair);
  const bool same_shell_pair = first_pair == second_pair;
  const std::size_t ao_quartet_count =
      same_shell_pair ? first_ao_pair_count * (first_ao_pair_count + 1) / 2
                      : first_ao_pair_count * second_ao_pair_count;

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_count; ++ordinal) {
    std::size_t first_ao_pair = 0;
    std::size_t second_ao_pair = 0;
    if (same_shell_pair) {
      decode_lower_triangle(ordinal, first_ao_pair, second_ao_pair);
    } else {
      first_ao_pair = ordinal / second_ao_pair_count;
      second_ao_pair = ordinal % second_ao_pair_count;
    }
    std::size_t raw_ao[4];
    decode_shell_ao_pair(batch,
                         first_pair,
                         first_ao_pair,
                         system_ao_begin,
                         raw_ao[0],
                         raw_ao[1]);
    decode_shell_ao_pair(batch,
                         second_pair,
                         second_ao_pair,
                         system_ao_begin,
                         raw_ao[2],
                         raw_ao[3]);
    if (schwarz_bounds[physical_offset +
                       matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset +
                           matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    const double density_coefficient =
        direct_force_density_coefficient<Unrestricted>(n,
                                                       physical_offset,
                                                       spin_offset,
                                                       density,
                                                       raw_ao[0],
                                                       raw_ao[1],
                                                       raw_ao[2],
                                                       raw_ao[3]);
    if (density_coefficient == 0.0) continue;
    const unsigned first_component = static_cast<unsigned>(
        raw_ao[canonical_raw_slot[0]] - first_component_begin);
    unsigned output = first_component;
    if constexpr (TargetShellClass == kPpssShellClass) {
      const unsigned second_component = static_cast<unsigned>(
          raw_ao[canonical_raw_slot[1]] - second_component_begin);
      if (first_component >= 3U || second_component >= 3U) return;
      output = first_component * 3U + second_component;
    } else if (first_component >= 6U) {
      return;
    }
    const double angular_coefficient =
        batch.direct_ao_coefficients[system_ao_begin + raw_ao[0]] *
        batch.direct_ao_coefficients[system_ao_begin + raw_ao[1]] *
        batch.direct_ao_coefficients[system_ao_begin + raw_ao[2]] *
        batch.direct_ao_coefficients[system_ao_begin + raw_ao[3]];
    component_weight[output] += density_coefficient * angular_coefficient;
    any_component = true;
  }
  if (!any_component) return;

  PspsWeightedGradient gradient{};
  if constexpr (TargetShellClass == kPpssShellClass) {
    gradient = contracted_eri_cartesian_source_ppss_weighted_gradient(
        batch,
        canonical_pair[0],
        canonical_pair[1],
        canonical_shell[0],
        canonical_shell[1],
        canonical_shell[2],
        canonical_shell[3],
        component_weight);
  } else {
    gradient = contracted_eri_cartesian_source_dsss_weighted_gradient(
        batch,
        canonical_pair[0],
        canonical_pair[1],
        canonical_shell[0],
        canonical_shell[1],
        canonical_shell[2],
        canonical_shell[3],
        component_weight);
  }

  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate =
        static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned canonical = 0; canonical < 3; ++canonical) {
        const double value = gradient.center[canonical][axis];
        fourth_derivative -= value;
        if (batch.shell_atoms[canonical_shell[canonical]] ==
            unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (batch.shell_atoms[canonical_shell[3]] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }
  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) *
      3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

template <bool Unrestricted, unsigned AngularOrder>
__device__ __forceinline__ void contract_two_electron_force_quartet_subtile(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask,
    std::size_t active_subtile,
    unsigned ao_quartet_lane) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  constexpr std::size_t subtiles_per_tile =
      detail::direct_quartet_subtiles_per_tile(AngularOrder);
  const std::size_t active_tile =
      active_subtile / subtiles_per_tile;
  // Consume the identical compact tile list as direct Fock so energy and
  // derivative screening cover precisely the same AO-quartet domain.
  if (active_tile >=
      static_cast<std::size_t>(*active_shell_quartet_tile_count)) {
    return;
  }
  const std::size_t subtile =
      active_subtile % subtiles_per_tile;
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
  // Generated consumers contract the complete exact class independently.
  // The host-selected bit mask keeps the generic fallback active for classes
  // disabled during production bisection.
  if (shell_class < 64U &&
      (generated_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U) {
    return;
  }

  const std::size_t ordinal =
      static_cast<std::size_t>(task.tile) * detail::kDirectQuartetTileSize +
      subtile * detail::kDirectQuartetThreads + ao_quartet_lane;
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

    const double coefficient = direct_force_density_coefficient<Unrestricted>(
        n, physical_offset, spin_offset, density, i, j, k, l);
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
    if constexpr (AngularOrder <= 6) {
      CartesianQuartetGradient explicit_gradient{};
      if constexpr (AngularOrder <= 1) {
        explicit_gradient =
            contracted_eri_cartesian_source_order01_gradient<AngularOrder>(
                batch, system, static_cast<std::int32_t>(i),
                static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
                static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 2) {
        explicit_gradient = contracted_eri_cartesian_source_order2_gradient(
            batch, system, static_cast<std::int32_t>(i),
            static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
            static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 3) {
        explicit_gradient = contracted_eri_cartesian_source_order3_gradient(
            batch, system, static_cast<std::int32_t>(i),
            static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
            static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 4) {
        explicit_gradient = contracted_eri_cartesian_source_order4_gradient(
            batch, system, static_cast<std::int32_t>(i),
            static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
            static_cast<std::int32_t>(l));
      } else if constexpr (AngularOrder == 5) {
        explicit_gradient = contracted_eri_cartesian_source_order5_gradient(
            batch, system, static_cast<std::int32_t>(i),
            static_cast<std::int32_t>(j), static_cast<std::int32_t>(k),
            static_cast<std::int32_t>(l));
      } else {
        explicit_gradient = contracted_eri_cartesian_source_order6_gradient(
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
      if constexpr (AngularOrder <= 6) {
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

/** Runtime angular dispatch used only by the generic large-topology path. */
template <bool Unrestricted>
__device__ __noinline__ void contract_bounded_direct_fock_subtile(
    DeviceBatch batch,
    unsigned angular_order,
    const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock,
    std::size_t subtile,
    unsigned lane) {
#define VIBEQC_BOUNDED_FOCK_CASE(order)                                      \
  case order:                                                               \
    contract_fock_direct_quartet_subtile<Unrestricted, order>(               \
        batch, queue_count, task, screening_tolerance, schwarz_bounds,       \
        density, active, fock, nullptr, subtile, lane);                      \
    break
  switch (angular_order) {
    VIBEQC_BOUNDED_FOCK_CASE(0);
    VIBEQC_BOUNDED_FOCK_CASE(1);
    VIBEQC_BOUNDED_FOCK_CASE(2);
    VIBEQC_BOUNDED_FOCK_CASE(3);
    VIBEQC_BOUNDED_FOCK_CASE(4);
    VIBEQC_BOUNDED_FOCK_CASE(5);
    VIBEQC_BOUNDED_FOCK_CASE(6);
    VIBEQC_BOUNDED_FOCK_CASE(7);
    VIBEQC_BOUNDED_FOCK_CASE(8);
    VIBEQC_BOUNDED_FOCK_CASE(9);
    VIBEQC_BOUNDED_FOCK_CASE(10);
    VIBEQC_BOUNDED_FOCK_CASE(11);
    VIBEQC_BOUNDED_FOCK_CASE(12);
    default: break;
  }
#undef VIBEQC_BOUNDED_FOCK_CASE
}

template <bool Unrestricted>
__device__ __noinline__ void contract_bounded_direct_force_subtile(
    DeviceBatch batch,
    unsigned angular_order,
    const std::uint32_t* queue_count,
    const ActiveShellQuartetTile* task,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::size_t subtile,
    unsigned lane) {
#define VIBEQC_BOUNDED_FORCE_CASE(order)                                     \
  case order:                                                               \
    contract_two_electron_force_quartet_subtile<Unrestricted, order>(        \
        batch, queue_count, task, screening_tolerance, schwarz_bounds,       \
        density, active, forces, 0U, subtile, lane);                         \
    break
  switch (angular_order) {
    VIBEQC_BOUNDED_FORCE_CASE(0);
    VIBEQC_BOUNDED_FORCE_CASE(1);
    VIBEQC_BOUNDED_FORCE_CASE(2);
    VIBEQC_BOUNDED_FORCE_CASE(3);
    VIBEQC_BOUNDED_FORCE_CASE(4);
    VIBEQC_BOUNDED_FORCE_CASE(5);
    VIBEQC_BOUNDED_FORCE_CASE(6);
    VIBEQC_BOUNDED_FORCE_CASE(7);
    VIBEQC_BOUNDED_FORCE_CASE(8);
    VIBEQC_BOUNDED_FORCE_CASE(9);
    VIBEQC_BOUNDED_FORCE_CASE(10);
    VIBEQC_BOUNDED_FORCE_CASE(11);
    VIBEQC_BOUNDED_FORCE_CASE(12);
    default: break;
  }
#undef VIBEQC_BOUNDED_FORCE_CASE
}

/** Accumulate the same final-density shell-class ledger as exact compaction. */
__device__ void profile_bounded_direct_shell_quartet(
    DeviceBatch batch,
    const ActiveShellQuartetTile& task,
    DeviceShellClassProfileEntry* profile) {
  if (profile == nullptr) return;
  const std::int32_t shells[4] = {
      batch.shell_pair_first[task.first_pair],
      batch.shell_pair_second[task.first_pair],
      batch.shell_pair_first[task.second_pair],
      batch.shell_pair_second[task.second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[shells[0]], batch.shell_angular[shells[1]],
      batch.shell_angular[shells[2]], batch.shell_angular[shells[3]]);
  if (shell_class >= detail::kDirectQuartetShellClassCount) return;
  const std::size_t first_count =
      shell_ao_pair_count(batch, task.first_pair);
  const std::size_t second_count =
      shell_ao_pair_count(batch, task.second_pair);
  const std::size_t ao_quartets = task.first_pair == task.second_pair
      ? first_count * (first_count + 1) / 2
      : first_count * second_count;
  const std::size_t tiles =
      (ao_quartets + detail::kDirectQuartetTileSize - 1) /
      detail::kDirectQuartetTileSize;
  unsigned long long primitive_quartets =
      static_cast<unsigned long long>(ao_quartets);
  for (const std::int32_t shell : shells) {
    primitive_quartets *= static_cast<unsigned long long>(
        batch.shell_primitive_offsets[shell + 1] -
        batch.shell_primitive_offsets[shell]);
  }
  DeviceShellClassProfileEntry& entry = profile[shell_class];
  atomicAdd(&entry.shell_quartets, 1ULL);
  atomicAdd(&entry.tiles, static_cast<unsigned long long>(tiles));
  atomicAdd(&entry.ao_quartets,
            static_cast<unsigned long long>(ao_quartets));
  atomicAdd(&entry.primitive_quartets, primitive_quartets);
}

/**
 * Stream only canonical dddd work from class-major shell-pair segments.
 *
 * This is an exact class-specific route, not the bounded generic fallback:
 * its outer domain is the dd-pair triangle and every accepted quartet is
 * consumed immediately.  It replaces the currently unreliable generated
 * dddd value/gradient consumer while retaining O(N_shell^2) topology storage
 * and zero whole-topology scan when all production classes are covered.
 */
template <bool Unrestricted,
          DirectScreeningPurpose Purpose,
          bool Force>
__global__ __launch_bounds__(detail::kDirectQuartetThreads)
void bounded_direct_dddd_streaming_kernel(
    DeviceBatch batch,
    const GeneratedShellPairStream* topology_pointer,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* output,
    std::uint32_t* bra_head,
    DeviceShellClassProfileEntry* profile) {
  static_assert(detail::kDirectQuartetThreads == 32);
  constexpr std::uint32_t kSkip = 0U;
  constexpr std::uint32_t kConsume = 1U;
  constexpr std::uint32_t kFinished = 2U;
  constexpr std::size_t kSubtilesPerTile =
      detail::direct_quartet_subtiles_per_tile(kDdddAngularOrder);

  __shared__ ActiveShellQuartetTile task;
  __shared__ std::uint32_t queue_count;
  __shared__ std::uint32_t candidate_ordinal;
  __shared__ std::uint32_t stream_state;
  __shared__ std::uint32_t tile_count;

  const unsigned lane = threadIdx.x;
  const GeneratedShellPairStream& topology = *topology_pointer;
  const std::size_t stride =
      static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::size_t dd_pair_class = 5U;
  const auto* density_bounds =
      reinterpret_cast<const ShellPairDensityBounds*>(
          topology.shell_pair_density_bounds);

  if (lane == 0U) queue_count = 1U;
  __syncwarp();
  while (true) {
    if (lane == 0U) {
      candidate_ordinal = atomicAdd(bra_head, 1U);
      std::uint64_t remaining = candidate_ordinal;
      stream_state = kFinished;
      for (std::int32_t system = 0; system < topology.batch_size; ++system) {
        const std::uint32_t pair_begin = topology.pair_class_offsets[
            dd_pair_class * stride + static_cast<std::size_t>(system)];
        const std::uint32_t pair_end = topology.pair_class_offsets[
            dd_pair_class * stride + static_cast<std::size_t>(system) + 1U];
        const std::uint64_t pair_count = pair_end - pair_begin;
        const std::uint64_t system_candidates =
            pair_count * (pair_count + 1U) / 2U;
        if (remaining >= system_candidates) {
          remaining -= system_candidates;
          continue;
        }

        std::size_t bra_local = 0U;
        std::size_t ket_local = 0U;
        decode_lower_triangle(
            static_cast<std::size_t>(remaining), bra_local, ket_local);
        const std::uint32_t bra_pair =
            topology.pair_order[pair_begin + bra_local];
        const std::uint32_t ket_pair =
            topology.pair_order[pair_begin + ket_local];
        bool keep = (active == nullptr || active[system] != 0U) &&
            topology.shell_pair_bounds[bra_pair] *
                topology.shell_pair_bounds[ket_pair] >= screening_tolerance;
        if (keep) {
          keep = direct_shell_quartet_survives_screening<
              Unrestricted, Purpose>(
              batch, bra_pair, ket_pair, screening_tolerance,
              topology.shell_pair_bounds, density_bounds);
        }
        stream_state = keep ? kConsume : kSkip;
        if (keep) {
          task = {bra_pair, ket_pair, 0U};
          const std::size_t first_count =
              shell_ao_pair_count(batch, bra_pair);
          const std::size_t second_count =
              shell_ao_pair_count(batch, ket_pair);
          const std::size_t ao_quartets = bra_pair == ket_pair
              ? first_count * (first_count + 1U) / 2U
              : first_count * second_count;
          tile_count = static_cast<std::uint32_t>(
              (ao_quartets + detail::kDirectQuartetTileSize - 1U) /
              detail::kDirectQuartetTileSize);
        }
        break;
      }
    }
    __syncwarp();
    if (stream_state == kFinished) return;
    if (stream_state != kConsume) continue;

    if constexpr (Force) {
      if (lane == 0U) {
        profile_bounded_direct_shell_quartet(batch, task, profile);
      }
      __syncwarp();
    }
    for (std::uint32_t tile = 0U; tile < tile_count; ++tile) {
      if (lane == 0U) task.tile = tile;
      __syncwarp();
      for (std::size_t subtile = 0U;
           subtile < kSubtilesPerTile; ++subtile) {
        if constexpr (Force) {
          contract_two_electron_force_quartet_subtile<
              Unrestricted, kDdddAngularOrder>(
              batch, &queue_count, &task, screening_tolerance,
              schwarz_bounds, density, active, output, 0U, subtile, lane);
        } else {
          contract_fock_direct_quartet_subtile<
              Unrestricted, kDdddAngularOrder>(
              batch, &queue_count, &task, screening_tolerance,
              schwarz_bounds, density, active, output, nullptr,
              subtile, lane);
        }
      }
      __syncwarp();
    }
  }
}

/**
 * Materialize one page of an overflowed exact class for its generated kernel.
 *
 * Page membership is determined by the unscreened candidate ordinal so every
 * launch covers a disjoint, deterministic slice without a device-to-host
 * synchronization. Surviving tasks are compacted within that page and then
 * consumed by the exact generated shell-class kernel; this is scheduling,
 * not a generic integral-evaluation fallback.
 */
template <bool Unrestricted, DirectScreeningPurpose Purpose>
__global__ void compact_bounded_exact_class_force_wave_kernel(
    DeviceBatch batch,
    const GeneratedShellPairStream* topology_pointer,
    unsigned shell_class,
    unsigned high_pair_class,
    unsigned low_pair_class,
    double screening_tolerance,
    std::uint64_t page_begin,
    std::uint32_t page_capacity,
    GeneratedShellTask* tasks,
    std::uint32_t* task_count,
    std::uint32_t* bra_head,
    const std::uint32_t* overflow,
    bool force_execution,
    std::uint32_t* signature_counts,
    const std::uint32_t* signature_offsets) {
  __shared__ std::uint32_t bra_ordinal;
  if (shell_class >= detail::kDirectQuartetShellClassCount ||
      (!force_execution && overflow[shell_class] == 0U)) {
    return;
  }
  const GeneratedShellPairStream& topology = *topology_pointer;
  const std::size_t stride =
      static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::uint32_t bra_begin =
      topology.pair_class_offsets[high_pair_class * stride];
  const std::uint32_t bra_end = topology.pair_class_offsets[
      high_pair_class * stride + topology.batch_size];
  const auto* density_bounds =
      reinterpret_cast<const ShellPairDensityBounds*>(
          topology.shell_pair_density_bounds);

  while (true) {
    if (threadIdx.x == 0U) bra_ordinal = atomicAdd(bra_head, 1U);
    __syncthreads();
    if (bra_ordinal >= bra_end - bra_begin) return;
    const std::uint32_t bra_pair =
        topology.pair_order[bra_begin + bra_ordinal];
    const std::int32_t system = topology.shell_pair_systems[bra_pair];
    if (topology.active != nullptr && topology.active[system] == 0U) {
      continue;
    }
    const std::uint32_t ket_begin = topology.pair_class_offsets[
        low_pair_class * stride + static_cast<std::size_t>(system)];
    const std::uint32_t ket_end = topology.pair_class_offsets[
        low_pair_class * stride + static_cast<std::size_t>(system) + 1U];
    const std::uint32_t system_bra_begin = topology.pair_class_offsets[
        high_pair_class * stride + static_cast<std::size_t>(system)];
    const std::uint64_t bra_local =
        bra_begin + bra_ordinal - system_bra_begin;
    const std::uint64_t ket_count = ket_end - ket_begin;
    std::uint64_t system_candidate_begin = 0U;
    for (std::int32_t previous = 0; previous < system; ++previous) {
      const std::uint32_t previous_bra_begin =
          topology.pair_class_offsets[
              high_pair_class * stride + static_cast<std::size_t>(previous)];
      const std::uint32_t previous_bra_end =
          topology.pair_class_offsets[
              high_pair_class * stride +
              static_cast<std::size_t>(previous) + 1U];
      const std::uint32_t previous_ket_begin =
          topology.pair_class_offsets[
              low_pair_class * stride + static_cast<std::size_t>(previous)];
      const std::uint32_t previous_ket_end =
          topology.pair_class_offsets[
              low_pair_class * stride +
              static_cast<std::size_t>(previous) + 1U];
      system_candidate_begin +=
          static_cast<std::uint64_t>(previous_bra_end - previous_bra_begin) *
          static_cast<std::uint64_t>(previous_ket_end - previous_ket_begin);
    }
    for (std::uint32_t ket_ordinal = ket_begin + threadIdx.x;
         ket_ordinal < ket_end; ket_ordinal += blockDim.x) {
      const std::uint64_t candidate_ordinal =
          system_candidate_begin + bra_local * ket_count +
          (ket_ordinal - ket_begin);
      if (candidate_ordinal < page_begin) continue;
      if (candidate_ordinal - page_begin >= page_capacity) break;
      const std::uint32_t ket_pair = topology.pair_order[ket_ordinal];
      if (topology.shell_pair_bounds[bra_pair] *
              topology.shell_pair_bounds[ket_pair] < screening_tolerance) {
        break;
      }
      if (high_pair_class == low_pair_class && bra_pair < ket_pair) continue;
      if (!direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
              batch, bra_pair, ket_pair, screening_tolerance,
              topology.shell_pair_bounds, density_bounds)) {
        continue;
      }
      std::uint32_t ordinal = 0U;
      if (signature_counts != nullptr) {
        const unsigned signature =
            bounded_force_signature_bucket(batch, bra_pair, ket_pair);
        const std::uint32_t signature_ordinal =
            atomicAdd(signature_counts + signature, 1U);
        if (signature_offsets == nullptr) continue;
        ordinal = signature_offsets[signature] + signature_ordinal;
      } else {
        ordinal = atomicAdd(task_count, 1U);
      }
      populate_generated_shell_task(
          batch, {bra_pair, ket_pair, 0U}, tasks[ordinal]);
    }
    __syncthreads();
  }
}

/** Scan bounded signature chunks in parallel and reset them for scatter. */
__global__ void scan_bounded_force_signature_counts_kernel(
    std::uint32_t* signature_counts,
    std::uint32_t* signature_offsets,
    std::uint32_t* block_offsets) {
  using BlockScan = cub::BlockScan<
      std::uint32_t, kBoundedForceSignatureScanThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  const unsigned signature =
      blockIdx.x * kBoundedForceSignatureScanThreads + threadIdx.x;
  const std::uint32_t count =
      signature < kBoundedForceSignatureBucketCount
      ? signature_counts[signature]
      : 0U;
  std::uint32_t local_offset = 0U;
  std::uint32_t block_total = 0U;
  BlockScan(scan_storage).ExclusiveSum(count, local_offset, block_total);
  if (signature < kBoundedForceSignatureBucketCount) {
    signature_offsets[signature] = local_offset;
    signature_counts[signature] = 0U;
  }
  if (threadIdx.x == 0U) block_offsets[blockIdx.x] = block_total;
}

/** Complete the chunk prefix and publish the bounded page task count. */
__global__ void prefix_bounded_force_signature_blocks_kernel(
    std::uint32_t* signature_offsets,
    std::uint32_t* block_offsets,
    std::uint32_t* task_count) {
  using BlockScan = cub::BlockScan<
      std::uint32_t, kBoundedForceSignatureScanThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  const std::uint32_t count =
      threadIdx.x < kBoundedForceSignatureScanBlockCount
      ? block_offsets[threadIdx.x]
      : 0U;
  std::uint32_t block_offset = 0U;
  std::uint32_t page_total = 0U;
  BlockScan(scan_storage).ExclusiveSum(count, block_offset, page_total);
  if (threadIdx.x < kBoundedForceSignatureScanBlockCount) {
    block_offsets[threadIdx.x] = block_offset;
  }
  __syncthreads();
  for (unsigned signature = threadIdx.x;
       signature < kBoundedForceSignatureBucketCount;
       signature += blockDim.x) {
    signature_offsets[signature] += block_offsets[
        signature / kBoundedForceSignatureScanThreads];
  }
  if (threadIdx.x == 0U) *task_count = page_total;
}

template <DirectScreeningPurpose Purpose>
__device__ __forceinline__ bool
bounded_direct_block_pair_survives_screening(
    std::size_t first_block,
    std::size_t second_block,
    std::int32_t system,
    double screening_tolerance,
    const double* shell_pair_block_bounds,
    const double* system_density_bounds);

/** Read the runtime exact-class mask used by the bounded generated routes. */
__device__ __forceinline__ bool bounded_generated_class_enabled(
    unsigned shell_class,
    const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask) {
  if (enabled_mask_pointer != nullptr) enabled_mask = *enabled_mask_pointer;
  return shell_class < detail::kDirectQuartetShellClassCount &&
      (enabled_mask & (std::uint64_t{1} << shell_class)) != 0U;
}

/**
 * Materialize every enabled exact class in one hierarchical scan.
 *
 * Each class owns a fixed slice whose setup-time weight comes from shell-pair
 * angular histograms. Overflow is recorded per class so a later exact-class
 * page stream can recover only that class without discarding unrelated
 * generated routes or repeating a whole-topology integral evaluation.
 */
template <bool Unrestricted,
          DirectScreeningPurpose Purpose,
          bool Materialize>
__global__ void compact_bounded_generated_tasks_kernel(
    DeviceBatch batch,
    double screening_tolerance,
    const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds,
    const double* system_density_bounds,
    const std::uint8_t* active,
    const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask,
    std::uint64_t excluded_mask,
    const std::uint32_t* selected_classes,
    const std::uint32_t* selected_any,
    unsigned long long* global_cursor,
    GeneratedShellTask* tasks,
    std::uint32_t* task_counts,
    const std::uint32_t* task_offsets,
    std::uint32_t* overflow) {
  __shared__ unsigned long long block_quartet;
  if (selected_any != nullptr && *selected_any == 0U) return;
  const std::size_t total = static_cast<std::size_t>(
      batch.total_shell_pair_block_quartets);
  while (true) {
    if (threadIdx.x == 0) block_quartet = atomicAdd(global_cursor, 1ULL);
    __syncthreads();
    if (block_quartet >= total) return;

    const std::size_t packed_block_quartet =
        static_cast<std::size_t>(block_quartet);
    const std::int32_t system = shell_pair_block_quartet_system(
        batch, packed_block_quartet);
    if (active != nullptr && active[system] == 0) continue;
    const std::size_t local_block_quartet = packed_block_quartet -
        static_cast<std::size_t>(
            batch.system_shell_pair_block_quartet_offsets[system]);
    std::size_t first_block_local = 0;
    std::size_t second_block_local = 0;
    decode_lower_triangle(
        local_block_quartet, first_block_local, second_block_local);
    const std::size_t system_block_begin = static_cast<std::size_t>(
        batch.system_shell_pair_block_offsets[system]);
    const std::size_t first_block = system_block_begin + first_block_local;
    const std::size_t second_block = system_block_begin + second_block_local;
    if (!bounded_direct_block_pair_survives_screening<Purpose>(
            first_block, second_block, system, screening_tolerance,
            shell_pair_block_bounds, system_density_bounds)) {
      continue;
    }

    const std::size_t system_pair_begin = static_cast<std::size_t>(
        batch.system_shell_pair_offsets[system]);
    const std::size_t system_pair_end = static_cast<std::size_t>(
        batch.system_shell_pair_offsets[system + 1]);
    const std::size_t first_ordered_begin = system_pair_begin +
        first_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t second_ordered_begin = system_pair_begin +
        second_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t first_count = min(
        detail::kBoundedDirectShellPairBlockSize,
        system_pair_end - first_ordered_begin);
    const std::size_t second_count = min(
        detail::kBoundedDirectShellPairBlockSize,
        system_pair_end - second_ordered_begin);
    const bool same_block = first_block == second_block;
    const std::size_t candidate_count = same_block
        ? first_count * (first_count + 1) / 2
        : first_count * second_count;
    for (std::size_t candidate = threadIdx.x; candidate < candidate_count;
         candidate += blockDim.x) {
      std::size_t first_local = 0;
      std::size_t second_local = 0;
      if (same_block) {
        decode_lower_triangle(candidate, first_local, second_local);
      } else {
        first_local = candidate / second_count;
        second_local = candidate % second_count;
      }
      const std::size_t first_pair = shell_pair_order[
          first_ordered_begin + first_local];
      const std::size_t second_pair = shell_pair_order[
          second_ordered_begin + second_local];
      if (!direct_shell_quartet_survives_screening<Unrestricted, Purpose>(
              batch, first_pair, second_pair, screening_tolerance,
              shell_pair_bounds, shell_pair_density_bounds)) {
        continue;
      }
      const std::int32_t first_shell = batch.shell_pair_first[first_pair];
      const std::int32_t second_shell = batch.shell_pair_second[first_pair];
      const std::int32_t third_shell = batch.shell_pair_first[second_pair];
      const std::int32_t fourth_shell = batch.shell_pair_second[second_pair];
      const unsigned shell_class = direct_quartet_shell_class_device(
          batch.shell_angular[first_shell],
          batch.shell_angular[second_shell],
          batch.shell_angular[third_shell],
          batch.shell_angular[fourth_shell]);
      if (!bounded_generated_class_enabled(
              shell_class, enabled_mask_pointer, enabled_mask)) {
        continue;
      }
      if ((excluded_mask & (std::uint64_t{1} << shell_class)) != 0U) {
        continue;
      }
      if (selected_classes != nullptr &&
          selected_classes[shell_class] == 0U) {
        continue;
      }
      if constexpr (Materialize) {
        const std::uint32_t class_slot =
            atomicAdd(task_counts + shell_class, 1U);
        const std::uint32_t class_capacity =
            task_offsets[shell_class + 1U] - task_offsets[shell_class];
        if (class_slot >= class_capacity) {
          atomicExch(overflow + shell_class, 1U);
          continue;
        }
        const std::uint32_t slot =
            task_offsets[shell_class] + class_slot;
        const ActiveShellQuartetTile tile{
            static_cast<std::uint32_t>(first_pair),
            static_cast<std::uint32_t>(second_pair), 0U};
        populate_generated_shell_task(batch, tile, tasks[slot]);
      } else {
        atomicAdd(task_counts + shell_class, 1U);
      }
    }
    __syncthreads();
  }
}

/**
 * Normalize the first generated wave and plan an exact overflow-only retry.
 *
 * The retry reuses the complete task arena after successful first-wave
 * consumers drain it. Exact observed counts define the second-wave slices;
 * if their sum still exceeds the arena, proportional slices preserve useful
 * generated work while the remaining classes are completed by exact-class
 * paged compaction.
 */
__global__ void prepare_bounded_generated_retry_kernel(
    std::uint32_t task_capacity,
    const std::uint32_t* task_offsets,
    std::uint32_t* task_counts,
    std::uint32_t* task_heads,
    std::uint32_t* overflow,
    std::uint32_t* retry_mask,
    std::uint32_t* retry_offsets,
    std::uint32_t* retry_any,
    bool preserve_overflow_counts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  std::uint64_t retry_total = 0;
  for (unsigned shell_class = 0;
       shell_class < detail::kDirectQuartetShellClassCount;
       ++shell_class) {
    const std::uint32_t capacity =
        task_offsets[shell_class + 1U] - task_offsets[shell_class];
    const bool retry = overflow[shell_class] != 0U ||
        task_counts[shell_class] > capacity;
    retry_mask[shell_class] = retry ? 1U : 0U;
    if (retry) retry_total += task_counts[shell_class];
  }

  std::uint64_t assigned = 0;
  if (retry_total > task_capacity) {
    for (unsigned shell_class = 0;
         shell_class < detail::kDirectQuartetShellClassCount;
         ++shell_class) {
      if (retry_mask[shell_class] == 0U) continue;
      assigned += static_cast<std::uint64_t>(task_capacity) *
          task_counts[shell_class] / retry_total;
    }
  }
  std::uint64_t extra = retry_total > task_capacity
      ? static_cast<std::uint64_t>(task_capacity) - assigned
      : 0U;
  std::uint64_t cursor = 0;
  for (unsigned shell_class = 0;
       shell_class < detail::kDirectQuartetShellClassCount;
       ++shell_class) {
    retry_offsets[shell_class] = static_cast<std::uint32_t>(cursor);
    if (retry_mask[shell_class] != 0U) {
      std::uint64_t capacity = retry_total <= task_capacity
          ? task_counts[shell_class]
          : static_cast<std::uint64_t>(task_capacity) *
              task_counts[shell_class] / retry_total;
      if (extra != 0U) {
        ++capacity;
        --extra;
      }
      cursor += capacity;
    }
    if (!preserve_overflow_counts && retry_mask[shell_class] != 0U) {
      task_counts[shell_class] = 0U;
    }
    task_heads[shell_class] = 0U;
    overflow[shell_class] = retry_mask[shell_class];
  }
  retry_offsets[detail::kDirectQuartetShellClassCount] =
      static_cast<std::uint32_t>(cursor);
  *retry_any = retry_total == 0U ? 0U : 1U;
}

/** Disable only generated classes that exceeded their fixed arena slice. */
__global__ void normalize_bounded_generated_task_counts_kernel(
    const std::uint32_t* task_offsets,
    std::uint32_t* task_counts,
    std::uint32_t* task_heads,
    std::uint32_t* overflow,
    bool preserve_overflow_counts) {
  const unsigned shell_class = blockIdx.x * blockDim.x + threadIdx.x;
  if (shell_class >= detail::kDirectQuartetShellClassCount) return;
  const std::uint32_t capacity =
      task_offsets[shell_class + 1U] - task_offsets[shell_class];
  if (overflow[shell_class] != 0U || task_counts[shell_class] > capacity) {
    if (!preserve_overflow_counts) task_counts[shell_class] = 0U;
    overflow[shell_class] = 1U;
  }
  task_heads[shell_class] = 0U;
}

/** Profile a successfully materialized bounded generated queue once. */
__global__ void profile_bounded_generated_tasks_kernel(
    DeviceBatch batch,
    const GeneratedShellTask* tasks,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    DeviceShellClassProfileEntry* profile) {
  const std::uint32_t count = *task_count;
  const std::uint32_t offset = *task_offset;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t task = blockIdx.x * blockDim.x + threadIdx.x;
       task < count; task += stride) {
    profile_bounded_direct_shell_quartet(
        batch,
        {tasks[offset + task].shell_pair[0],
         tasks[offset + task].shell_pair[1], 0U},
        profile);
  }
}

/** Safely reject a complete shell-pair-block product before exact screening. */
template <DirectScreeningPurpose Purpose>
__device__ __forceinline__ bool
bounded_direct_block_pair_survives_screening(
    std::size_t first_block,
    std::size_t second_block,
    std::int32_t system,
    double screening_tolerance,
    const double* shell_pair_block_bounds,
    const double* system_density_bounds) {
  const double quartet_bound =
      shell_pair_block_bounds[first_block] *
      shell_pair_block_bounds[second_block];
  if (quartet_bound < screening_tolerance) return false;
  const double density_bound = system_density_bounds[system];
  if (quartet_bound * density_bound < screening_tolerance) return false;
  if constexpr (Purpose == DirectScreeningPurpose::Force) {
    const double force_tolerance = fmin(
        screening_tolerance, kForceDensityProductScreeningTolerance);
    if (quartet_bound * density_bound * density_bound < force_tolerance) {
      return false;
    }
  }
  return true;
}

/**
 * Enumerate, screen, queue, and drain shell pair-of-pairs hierarchically.
 *
 * A persistent CTA first claims one pair-block product. Conservative Schwarz
 * and density maxima reject the complete block without visiting its members;
 * surviving blocks are expanded in fixed 256-candidate chunks and retain the
 * exact shell-quartet predicate. This keeps storage bounded while replacing
 * the former unconditional O(N_shell^4) scan with a small O(N_shell^4/B^2)
 * outer domain plus exact work only in surviving blocks.
 */
template <bool Unrestricted,
          DirectScreeningPurpose Purpose,
          bool Force>
__global__ __launch_bounds__(kBoundedDirectThreads, 1)
void bounded_direct_shell_quartet_kernel(
    DeviceBatch batch,
    double screening_tolerance,
    const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds,
    const std::uint32_t* shell_pair_order,
    const double* shell_pair_block_bounds,
    const double* system_density_bounds,
    const std::uint64_t* enabled_mask_pointer,
    std::uint64_t enabled_mask,
    const std::uint32_t* bounded_generated_overflow,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* output,
    unsigned long long* global_cursor,
    DeviceShellClassProfileEntry* profile) {
  __shared__ ActiveShellQuartetTile queue[
      detail::kBoundedDirectQueueCapacity];
  __shared__ std::uint32_t queue_count;
  __shared__ unsigned long long block_quartet;
  const unsigned lane = threadIdx.x % detail::kDirectQuartetThreads;
  const unsigned warp = threadIdx.x / detail::kDirectQuartetThreads;
  const std::size_t total =
      static_cast<std::size_t>(batch.total_shell_pair_block_quartets);

  while (true) {
    if (threadIdx.x == 0) {
      block_quartet = atomicAdd(global_cursor, 1ULL);
    }
    __syncthreads();
    if (block_quartet >= total) return;

    const std::size_t packed_block_quartet =
        static_cast<std::size_t>(block_quartet);
    const std::int32_t system = shell_pair_block_quartet_system(
        batch, packed_block_quartet);
    if (active != nullptr && active[system] == 0) continue;
    const std::size_t local_block_quartet = packed_block_quartet -
        static_cast<std::size_t>(
            batch.system_shell_pair_block_quartet_offsets[system]);
    std::size_t first_block_local = 0;
    std::size_t second_block_local = 0;
    decode_lower_triangle(
        local_block_quartet, first_block_local, second_block_local);
    const std::size_t system_block_begin = static_cast<std::size_t>(
        batch.system_shell_pair_block_offsets[system]);
    const std::size_t first_block = system_block_begin + first_block_local;
    const std::size_t second_block = system_block_begin + second_block_local;
    if (!bounded_direct_block_pair_survives_screening<Purpose>(
            first_block, second_block, system, screening_tolerance,
            shell_pair_block_bounds, system_density_bounds)) {
      continue;
    }

    const std::size_t system_pair_begin = static_cast<std::size_t>(
        batch.system_shell_pair_offsets[system]);
    const std::size_t system_pair_end = static_cast<std::size_t>(
        batch.system_shell_pair_offsets[system + 1]);
    const std::size_t first_ordered_begin = system_pair_begin +
        first_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t second_ordered_begin = system_pair_begin +
        second_block_local * detail::kBoundedDirectShellPairBlockSize;
    const std::size_t first_count = min(
        detail::kBoundedDirectShellPairBlockSize,
        system_pair_end - first_ordered_begin);
    const std::size_t second_count = min(
        detail::kBoundedDirectShellPairBlockSize,
        system_pair_end - second_ordered_begin);
    const bool same_block = first_block == second_block;
    const std::size_t candidate_count = same_block
        ? first_count * (first_count + 1) / 2
        : first_count * second_count;

    for (std::size_t candidate_begin = 0;
         candidate_begin < candidate_count;
         candidate_begin += detail::kBoundedDirectQueueCapacity) {
      if (threadIdx.x == 0) queue_count = 0;
      __syncthreads();
      const std::size_t candidate = candidate_begin + threadIdx.x;
      if (candidate < candidate_count) {
        std::size_t first_local = 0;
        std::size_t second_local = 0;
        if (same_block) {
          decode_lower_triangle(candidate, first_local, second_local);
        } else {
          first_local = candidate / second_count;
          second_local = candidate % second_count;
        }
        const std::size_t first_pair = shell_pair_order[
            first_ordered_begin + first_local];
        const std::size_t second_pair = shell_pair_order[
            second_ordered_begin + second_local];
        if (direct_shell_quartet_survives_screening<
                Unrestricted, Purpose>(
                batch, first_pair, second_pair, screening_tolerance,
                shell_pair_bounds, shell_pair_density_bounds)) {
          const std::int32_t first_shell =
              batch.shell_pair_first[first_pair];
          const std::int32_t second_shell =
              batch.shell_pair_second[first_pair];
          const std::int32_t third_shell =
              batch.shell_pair_first[second_pair];
          const std::int32_t fourth_shell =
              batch.shell_pair_second[second_pair];
          const unsigned shell_class = direct_quartet_shell_class_device(
              batch.shell_angular[first_shell],
              batch.shell_angular[second_shell],
              batch.shell_angular[third_shell],
              batch.shell_angular[fourth_shell]);
          const bool generated_class =
              bounded_generated_overflow[shell_class] == 0U &&
              bounded_generated_class_enabled(
                  shell_class, enabled_mask_pointer, enabled_mask);
          if (!generated_class) {
            const std::uint32_t slot = atomicAdd(&queue_count, 1U);
            queue[slot] = {
                static_cast<std::uint32_t>(first_pair),
                static_cast<std::uint32_t>(second_pair), 0U};
            if constexpr (Force) {
              profile_bounded_direct_shell_quartet(
                  batch, queue[slot], profile);
            }
          }
        }
      }
      __syncthreads();

      // Low-order shell tasks fit in one scalar lane. Drain up to 256 of
      // them concurrently before assigning the larger classes one warp each;
      // the former generic path spent 31 idle lanes on every ssss/psss/order2
      // task and dominates molecular systems built from s/p/d basis shells.
      for (std::uint32_t slot = threadIdx.x; slot < queue_count;
           slot += blockDim.x) {
        const ActiveShellQuartetTile task = queue[slot];
        const std::int32_t first_shell =
            batch.shell_pair_first[task.first_pair];
        const std::int32_t second_shell =
            batch.shell_pair_second[task.first_pair];
        const std::int32_t third_shell =
            batch.shell_pair_first[task.second_pair];
        const std::int32_t fourth_shell =
            batch.shell_pair_second[task.second_pair];
        const unsigned angular_order =
            batch.shell_angular[first_shell] +
            batch.shell_angular[second_shell] +
            batch.shell_angular[third_shell] +
            batch.shell_angular[fourth_shell];
        if constexpr (Force) {
          if (angular_order == 0U) {
            contract_two_electron_force_ssss_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density,
                active, output, 0U);
          } else if (angular_order == 1U) {
            contract_two_electron_force_psss_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density,
                active, output, 0U);
          } else if (angular_order == 2U) {
            contract_two_electron_force_psps_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density,
                active, output, 0U);
            contract_two_electron_force_pair_order2_task<
                Unrestricted, kPpssShellClass>(
                batch, task, screening_tolerance, schwarz_bounds, density,
                active, output, 0U);
            contract_two_electron_force_pair_order2_task<
                Unrestricted, kDsssShellClass>(
                batch, task, screening_tolerance, schwarz_bounds, density,
                active, output, 0U);
          }
        } else {
          if (angular_order == 0U) {
            contract_fock_direct_quartet_subtile<Unrestricted, 0U>(
                batch, &queue_count, queue + slot, screening_tolerance,
                schwarz_bounds, density, active, output, nullptr, 0U, 0U);
          } else if (angular_order == 1U) {
            contract_fock_direct_psss_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density,
                active, output);
          } else if (angular_order == 2U) {
            contract_fock_direct_order2_task<Unrestricted>(
                batch, task, screening_tolerance, schwarz_bounds, density,
                active, output, nullptr);
          }
        }
      }
      __syncthreads();

      for (std::uint32_t slot = warp; slot < queue_count;
           slot += kBoundedDirectThreads / detail::kDirectQuartetThreads) {
        const ActiveShellQuartetTile base = queue[slot];
        const std::int32_t first_shell =
            batch.shell_pair_first[base.first_pair];
        const std::int32_t second_shell =
            batch.shell_pair_second[base.first_pair];
        const std::int32_t third_shell =
            batch.shell_pair_first[base.second_pair];
        const std::int32_t fourth_shell =
            batch.shell_pair_second[base.second_pair];
        const unsigned angular_order =
            batch.shell_angular[first_shell] +
            batch.shell_angular[second_shell] +
            batch.shell_angular[third_shell] +
            batch.shell_angular[fourth_shell];
        if (angular_order <= 2U) continue;
        const std::size_t first_ao_count =
            shell_ao_pair_count(batch, base.first_pair);
        const std::size_t second_ao_count =
            shell_ao_pair_count(batch, base.second_pair);
        const std::size_t ao_quartets = base.first_pair == base.second_pair
            ? first_ao_count * (first_ao_count + 1) / 2
            : first_ao_count * second_ao_count;
        const std::uint32_t tile_count = static_cast<std::uint32_t>(
            (ao_quartets + detail::kDirectQuartetTileSize - 1) /
            detail::kDirectQuartetTileSize);
        const std::size_t subtile_count =
            detail::direct_quartet_subtiles_per_tile(angular_order);
        for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
          if (lane == 0) queue[slot].tile = tile;
          __syncwarp();
          for (std::size_t subtile = 0; subtile < subtile_count; ++subtile) {
            if constexpr (Force) {
              contract_bounded_direct_force_subtile<Unrestricted>(
                  batch, angular_order, &queue_count, queue + slot,
                  screening_tolerance, schwarz_bounds, density, active,
                  output, subtile, lane);
            } else {
              contract_bounded_direct_fock_subtile<Unrestricted>(
                  batch, angular_order, &queue_count, queue + slot,
                  screening_tolerance, schwarz_bounds, density, active,
                  output, subtile, lane);
            }
          }
          __syncwarp();
        }
      }
      __syncthreads();
    }
  }
}

/** Fixed-capacity wrapper for the small generic high-order force grids. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  contract_two_electron_force_quartet_subtile<Unrestricted, AngularOrder>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles,
      screening_tolerance, schwarz_bounds, density, active, forces,
      generated_shell_class_mask,
      static_cast<std::size_t>(blockIdx.x), threadIdx.x);
}

/** Pack independent ssss derivative shell tasks across one worker warp. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_packed_persistent_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint32_t* task_head,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  static_assert(AngularOrder < kPackedSsssAngularOrderCount);
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(
          task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_two_electron_force_ssss_task<Unrestricted>(
          batch, active_shell_quartet_tiles[packed_item], screening_tolerance,
          schwarz_bounds, density, active, forces,
          generated_shell_class_mask);
    }
  }
}

/**
 * Keep one p-s bra resident while block threads traverse all s-s ket pairs.
 *
 * Total angular order one contains only psss quartets, so enumerating each
 * p-s shell pair once and each s-s shell pair in its system once preserves
 * unique quartet ownership without consulting the unordered compact queue.
 * The shared primitive-pair records remove the remaining repeated bra loads
 * across the hundreds of ket tasks normally associated with one 192-AO bra.
 */
template <bool Unrestricted>
// Four resident 128-thread blocks cap this register-heavy contraction at
// 128 registers/thread on sm_120.  The extra occupancy hides the long
// primitive-pair dependency chain without changing the resident-bra schedule.
__global__ __launch_bounds__(kResidentPsssThreads, 4)
void two_electron_force_psss_resident_bra_kernel(
    DeviceBatch batch,
    const PsssResidentTask* resident_tasks,
    const std::uint32_t* resident_ket_pairs,
    std::size_t resident_task_count,
    double screening_tolerance,
    const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds,
    bool force_density_product_screening,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  extern __shared__ PrimitivePairData resident_first_pairs[];
  const std::size_t task_index = static_cast<std::size_t>(blockIdx.x);
  if (task_index >= resident_task_count) return;
  const PsssResidentTask task = resident_tasks[task_index];
  const std::size_t bra_pair = task.bra_pair;

  const std::int32_t system = batch.shell_pair_systems[bra_pair];
  if (active[system] == 0) return;
  const std::int32_t bra_first_shell = batch.shell_pair_first[bra_pair];
  const std::int32_t bra_second_shell = batch.shell_pair_second[bra_pair];
  const unsigned bra_first_angular = batch.shell_angular[bra_first_shell];
  const unsigned bra_second_angular = batch.shell_angular[bra_second_shell];
  if (bra_first_angular + bra_second_angular != 1U) return;

  const std::int64_t bra_primitive_begin =
      batch.shell_pair_primitive_offsets[bra_pair];
  const std::int64_t bra_primitive_count =
      batch.shell_pair_primitive_offsets[bra_pair + 1U] -
      bra_primitive_begin;
  if (bra_primitive_count <= 0 ||
      bra_primitive_count > static_cast<std::int64_t>(
          kResidentPsssMaximumBraPrimitivePairs)) return;
  for (std::int64_t primitive = threadIdx.x;
       primitive < bra_primitive_count; primitive += blockDim.x) {
    resident_first_pairs[primitive] =
        batch.shell_primitive_pairs[bra_primitive_begin + primitive];
  }
  __syncthreads();

  for (std::size_t local_ket = threadIdx.x;
       local_ket < task.ket_count; local_ket += blockDim.x) {
    const std::size_t ket_pair =
        resident_ket_pairs[task.ket_begin + local_ket];
    const std::size_t first_pair = bra_pair > ket_pair ? bra_pair : ket_pair;
    const std::size_t second_pair = bra_pair > ket_pair ? ket_pair : bra_pair;
    const bool survives_screening = force_density_product_screening
        ? direct_shell_quartet_survives_screening<
              Unrestricted, DirectScreeningPurpose::Force>(
              batch, first_pair, second_pair, screening_tolerance,
              shell_pair_bounds, shell_pair_density_bounds)
        : direct_shell_quartet_survives_screening<
              Unrestricted, DirectScreeningPurpose::Fock>(
              batch, first_pair, second_pair, screening_tolerance,
              shell_pair_bounds, shell_pair_density_bounds);
    if (!survives_screening) {
      continue;
    }
    contract_two_electron_force_psss_task<Unrestricted, true>(
        batch,
        {static_cast<std::uint32_t>(first_pair),
         static_cast<std::uint32_t>(second_pair), 0U},
        screening_tolerance, schwarz_bounds, density, active, forces,
        generated_shell_class_mask, resident_first_pairs,
        bra_primitive_count);
  }
}

/** Consume complete density-weighted psss force tasks, one task per lane. */
template <bool Unrestricted>
__global__ void two_electron_force_psss_persistent_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint32_t* task_head,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(
          task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_two_electron_force_psss_task<Unrestricted>(
          batch, active_shell_quartet_tiles[packed_item],
          screening_tolerance, schwarz_bounds, density, active, forces,
          generated_shell_class_mask);
    }
    // Keep tail lanes live through the next full-mask queue broadcast.
  }
}

/** Scan compact order-two tiles and consume complete psps shell tasks. */
template <bool Unrestricted>
__global__ void two_electron_force_psps_grid_stride_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t task_index = blockIdx.x * blockDim.x + threadIdx.x;
       task_index < work_count; task_index += stride) {
    contract_two_electron_force_psps_task<Unrestricted>(
        batch, active_shell_quartet_tiles[task_index], screening_tolerance,
        schwarz_bounds, density, active, forces,
        generated_shell_class_mask);
  }
}

/** Scan compact order-two tiles for one exact ppss or dsss class. */
template <bool Unrestricted, unsigned TargetShellClass>
__global__ void two_electron_force_pair_order2_grid_stride_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  static_assert(
      TargetShellClass == kPpssShellClass ||
      TargetShellClass == kDsssShellClass);
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t task_index = blockIdx.x * blockDim.x + threadIdx.x;
       task_index < work_count; task_index += stride) {
    contract_two_electron_force_pair_order2_task<
        Unrestricted, TargetShellClass>(
        batch, active_shell_quartet_tiles[task_index], screening_tolerance,
        schwarz_bounds, density, active, forces,
        generated_shell_class_mask);
  }
}

/**
 * Persistent one-warp workers dynamically consume only compacted force work.
 *
 * The topology-capacity launch remains useful for CUDA Graph Fock replay, but
 * the final force executes outside that iterative Graph. A device task head
 * therefore removes empty capacity blocks and balances irregular AO-quartet
 * derivative cost without introducing a host readback of compacted counts.
 */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void two_electron_force_quartet_persistent_kernel(
    DeviceBatch batch,
    const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles,
    std::uint32_t* task_head,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  const unsigned lane = threadIdx.x % warpSize;
  constexpr std::uint32_t subtiles_per_tile = static_cast<std::uint32_t>(
      detail::direct_quartet_subtiles_per_tile(AngularOrder));
  const std::uint32_t work_count =
      *active_shell_quartet_tile_count * subtiles_per_tile;
  while (true) {
    std::uint32_t active_subtile = 0;
    if (lane == 0) active_subtile = atomicAdd(task_head, 1U);
    active_subtile = __shfl_sync(0xffffffffU, active_subtile, 0);
    if (active_subtile >= work_count) return;
    contract_two_electron_force_quartet_subtile<Unrestricted, AngularOrder>(
        batch, active_shell_quartet_tile_count, active_shell_quartet_tiles,
        screening_tolerance, schwarz_bounds, density, active, forces,
        generated_shell_class_mask,
        active_subtile, threadIdx.x);
  }
}

/** Prepare compact exact-class slices shared by generated Fock and force code. */
cudaError_t prepare_generated_shell_tasks(
    cudaStream_t stream,
    std::size_t total_tile_capacity,
    std::size_t generated_task_capacity,
    const std::uint32_t* active_tile_offsets,
    DeviceBatch batch,
    const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks,
    std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts,
    std::uint32_t* generated_task_heads,
    std::uint64_t low_order_signature_mask,
    std::uint32_t* low_order_signature_counts,
    std::uint32_t* low_order_signature_offsets,
    std::uint64_t enabled_mask,
    const std::uint64_t* enabled_mask_pointer,
    bool exclude_resident_ppps) {
  if ((enabled_mask == 0U && enabled_mask_pointer == nullptr) ||
      generated_task_capacity == 0 ||
      total_tile_capacity == 0) {
    return cudaSuccess;
  }
  cudaError_t error = cudaMemsetAsync(
      generated_task_counts, 0,
      detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), stream);
  if (error != cudaSuccess) return error;
  if (low_order_signature_counts != nullptr) {
    error = cudaMemsetAsync(
        low_order_signature_counts, 0,
        kLowOrderSignatureElementCount * sizeof(std::uint32_t), stream);
    if (error != cudaSuccess) return error;
  }
  constexpr unsigned preparation_threads = kCaptureSafeKernelThreads;
  const unsigned preparation_blocks = static_cast<unsigned>(
      (total_tile_capacity + preparation_threads - 1) /
      preparation_threads);
  classify_generated_shell_tasks_kernel<<<
      preparation_blocks, preparation_threads, 0, stream>>>(
      batch, total_tile_capacity, active_tile_offsets, active_tile_counts,
      active_tiles, enabled_mask, enabled_mask_pointer, exclude_resident_ppps,
      generated_task_counts, generated_shell_classes,
      low_order_signature_mask, low_order_signature_counts);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  prefix_generated_shell_task_counts_kernel<<<1, 1, 0, stream>>>(
      generated_task_counts, generated_task_offsets,
      generated_task_write_counts, generated_task_heads);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  if (low_order_signature_counts != nullptr &&
      low_order_signature_offsets != nullptr) {
    prefix_low_order_signature_counts_kernel<<<1, 1, 0, stream>>>(
        generated_task_offsets, low_order_signature_mask,
        low_order_signature_counts, low_order_signature_offsets);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
  }
  materialize_generated_shell_tasks_kernel<<<
      preparation_blocks, preparation_threads, 0, stream>>>(
      batch, total_tile_capacity, active_tiles, generated_shell_classes,
      generated_task_offsets, generated_task_write_counts, generated_tasks,
      low_order_signature_mask, low_order_signature_offsets,
      low_order_signature_counts);
  return cudaPeekAtLastError();
}

/**
 * Bucket all enabled generated classes once, then launch their force slices.
 *
 * Counts, offsets, and worker heads remain device-resident, so adding an AOT
 * class does not add a host synchronization or another scan of every active
 * quartet. The generated persistent kernels apply their class offset when
 * loading tasks from the shared compact allocation.
 */
cudaError_t launch_generated_shell_class_forces(
    cudaStream_t stream,
    std::size_t total_tile_capacity,
    std::size_t generated_task_capacity,
    const std::array<std::size_t,
                     detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::uint32_t* active_tile_offsets,
    DeviceBatch batch,
    const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks,
    std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts,
    std::uint32_t* generated_task_heads,
    std::uint32_t* low_order_signature_counts,
    std::uint32_t* low_order_signature_offsets,
    GeneratedPppsResidentTask* resident_ppps_tasks,
    GeneratedShellTask* resident_ppps_ket_tasks,
    std::uint32_t* resident_ppps_bra_counts,
    std::uint32_t* resident_ppps_bra_offsets,
    std::uint32_t* resident_ppps_bra_write_counts,
    std::uint32_t* resident_ppps_signature_counts,
    std::uint32_t* resident_ppps_signature_offsets,
    std::uint32_t* resident_ppps_signatures,
    std::size_t total_shell_pairs,
    bool resident_ppps_enabled,
    bool resident_ppps_signature_bucketing,
    bool psps_signature_bucketing,
    bool ppss_signature_bucketing,
    unsigned resident_ppps_block_threads,
    unsigned persistent_worker_blocks,
    bool unrestricted,
    std::uint64_t enabled_mask,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces) {
  bool use_resident_ppps = resident_ppps_enabled &&
      (enabled_mask & (std::uint64_t{1} << kPppsShellClass)) != 0U;
  if (total_tile_capacity == 0 ||
      (enabled_mask == 0U && !use_resident_ppps) ||
      (generated_task_capacity == 0 && !use_resident_ppps)) {
    return cudaSuccess;
  }
  cudaError_t error = cudaSuccess;
  if (use_resident_ppps) {
    std::size_t ppps_tile_offset = 0;
    for (unsigned order = 0; order < kPppsAngularOrder; ++order) {
      ppps_tile_offset += capacities[order];
    }
    // ppps has total angular order three.  Restrict both grouping scans to
    // that fixed partition instead of rereading every active shell quartet.
    error = prepare_ppps_resident_tasks(
        stream, capacities[kPppsAngularOrder], total_shell_pairs, batch,
        active_tile_counts + kPppsAngularOrder,
        active_tiles + ppps_tile_offset, resident_ppps_tasks,
        resident_ppps_ket_tasks, resident_ppps_bra_counts,
        resident_ppps_bra_offsets, resident_ppps_bra_write_counts,
        resident_ppps_signature_bucketing
            ? resident_ppps_signature_counts
            : nullptr,
        resident_ppps_signature_bucketing
            ? resident_ppps_signature_offsets
            : nullptr,
        resident_ppps_signatures,
        enabled_mask);
    if (error != cudaSuccess) return error;
  }
  if (use_resident_ppps) {
    // Probe the selected AOT profile before excluding eligible ppps records
    // from the ordinary queue.  A portable profile may contain the ordinary
    // ppps class without its optional resident route; in that case fall back
    // losslessly instead of dropping the resident-eligible quartets.
    error = generated::launch_ppps_resident(
        stream, unrestricted, resident_ppps_tasks, resident_ppps_ket_tasks,
        batch.shell_pair_primitive_offsets, batch.shell_primitive_pairs,
        batch.direct_ao_coefficients, batch.positions, screening_tolerance,
        schwarz_bounds, density, forces, resident_ppps_block_threads,
        total_shell_pairs);
    if (error == cudaErrorNotSupported) {
      use_resident_ppps = false;
    } else if (error != cudaSuccess) {
      return error;
    }
  }
  const std::uint64_t low_order_signature_mask =
      (psps_signature_bucketing
           ? (std::uint64_t{1} << kPspsShellClass)
           : 0U) |
      (ppss_signature_bucketing
           ? (std::uint64_t{1} << kPpssShellClass)
           : 0U);
  error = prepare_generated_shell_tasks(
      stream, total_tile_capacity, generated_task_capacity,
      active_tile_offsets, batch, active_tile_counts, active_tiles,
      generated_tasks, generated_shell_classes, generated_task_offsets,
      generated_task_counts, generated_task_write_counts,
      generated_task_heads, low_order_signature_mask,
      low_order_signature_mask != 0U ? low_order_signature_counts : nullptr,
      low_order_signature_mask != 0U ? low_order_signature_offsets : nullptr,
      enabled_mask, nullptr, use_resident_ppps);
  if (error != cudaSuccess) return error;

  std::size_t kernel_count = 0;
  const generated::ShellKernelMetadata* kernels =
      generated::selected_shell_kernels(kernel_count);
  for (std::size_t kernel_index = 0; kernel_index < kernel_count;
       ++kernel_index) {
    const generated::ShellKernelMetadata& kernel = kernels[kernel_index];
    if ((enabled_mask & (std::uint64_t{1} << kernel.shell_class)) == 0U) {
      continue;
    }
    const unsigned worker_blocks = std::min(
        static_cast<unsigned>(capacities[kernel.angular_order]),
        persistent_worker_blocks);
    error = generated::launch_shell_class(
        kernel.shell_class, stream, unrestricted, worker_blocks,
        generated_tasks, generated_task_offsets + kernel.shell_class,
        batch.shell_pair_primitive_offsets, batch.shell_primitive_pairs,
        batch.direct_ao_coefficients, batch.positions, screening_tolerance,
        schwarz_bounds, density, forces,
        generated_task_counts + kernel.shell_class,
        generated_task_heads + kernel.shell_class);
    if (error != cudaSuccess) return error;
  }
  // The resident launch precedes ordinary preparation so an unsupported
  // optional route can select the complete fallback queue without dropping
  // any eligible ppps records.
  return cudaSuccess;
}

/**
 * Bucket all enabled generated classes once, then launch their Fock slices.
 *
 * The mask stays device-resident so graph replay can change the environment
 * selection without changing its fixed preparation and worker launch nodes.
 */
cudaError_t launch_generated_shell_class_focks(
    cudaStream_t stream,
    std::size_t total_tile_capacity,
    std::size_t generated_task_capacity,
    const std::array<std::size_t,
                     detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::uint32_t* active_tile_offsets,
    DeviceBatch batch,
    const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks,
    std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts,
    std::uint32_t* generated_task_heads,
    const std::uint64_t* enabled_mask,
    unsigned persistent_worker_blocks,
    bool unrestricted,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock) {
  cudaError_t error = prepare_generated_shell_tasks(
      stream, total_tile_capacity, generated_task_capacity,
      active_tile_offsets, batch, active_tile_counts, active_tiles,
      generated_tasks, generated_shell_classes, generated_task_offsets,
      generated_task_counts, generated_task_write_counts,
      generated_task_heads, 0U, nullptr, nullptr, 0U, enabled_mask, false);
  if (error != cudaSuccess) return error;

  std::size_t kernel_count = 0;
  const generated::ShellKernelMetadata* kernels =
      generated::selected_fock_shell_kernels(kernel_count);
  for (std::size_t kernel_index = 0; kernel_index < kernel_count;
       ++kernel_index) {
    const generated::ShellKernelMetadata& kernel = kernels[kernel_index];
    const unsigned worker_blocks = std::min(
        static_cast<unsigned>(capacities[kernel.angular_order]),
        persistent_worker_blocks);
    error = generated::launch_shell_class_fock(
        kernel.shell_class, stream, unrestricted, worker_blocks,
        generated_tasks, generated_task_offsets + kernel.shell_class,
        batch.shell_pair_primitive_offsets, batch.shell_primitive_pairs,
        batch.direct_ao_coefficients, batch.positions, screening_tolerance,
        schwarz_bounds, density, fock,
        generated_task_counts + kernel.shell_class,
        generated_task_heads + kernel.shell_class);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

/** Bucket the FP32 queue and launch only generated mixed-Fock capabilities. */
cudaError_t launch_generated_shell_class_mixed_focks(
    cudaStream_t stream,
    std::size_t total_tile_capacity,
    std::size_t generated_task_capacity,
    const std::array<std::size_t,
                     detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::uint32_t* active_tile_offsets,
    DeviceBatch batch,
    const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles,
    GeneratedShellTask* generated_tasks,
    std::uint8_t* generated_shell_classes,
    std::uint32_t* generated_task_offsets,
    std::uint32_t* generated_task_counts,
    std::uint32_t* generated_task_write_counts,
    std::uint32_t* generated_task_heads,
    const std::uint64_t* enabled_mask,
    unsigned persistent_worker_blocks,
    bool unrestricted,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock) {
  const std::uint64_t capability_mask =
      generated::enabled_mixed_fock_shell_class_mask();
  if (capability_mask == 0U) return cudaSuccess;
  cudaError_t error = prepare_generated_shell_tasks(
      stream, total_tile_capacity, generated_task_capacity,
      active_tile_offsets, batch, active_tile_counts, active_tiles,
      generated_tasks, generated_shell_classes, generated_task_offsets,
      generated_task_counts, generated_task_write_counts,
      generated_task_heads, 0U, nullptr, nullptr, 0U, enabled_mask, false);
  if (error != cudaSuccess) return error;

  std::size_t kernel_count = 0;
  const generated::ShellKernelMetadata* kernels =
      generated::selected_fock_shell_kernels(kernel_count);
  for (std::size_t kernel_index = 0; kernel_index < kernel_count;
       ++kernel_index) {
    const generated::ShellKernelMetadata& kernel = kernels[kernel_index];
    if ((capability_mask & (std::uint64_t{1} << kernel.shell_class)) == 0U) {
      continue;
    }
    const unsigned worker_blocks = std::min(
        static_cast<unsigned>(capacities[kernel.angular_order]),
        persistent_worker_blocks);
    error = generated::launch_shell_class_mixed_fock(
        kernel.shell_class, stream, unrestricted, worker_blocks,
        generated_tasks, generated_task_offsets + kernel.shell_class,
        batch.shell_pair_primitive_offsets, batch.shell_primitive_pairs,
        batch.direct_ao_coefficients, batch.positions, screening_tolerance,
        schwarz_bounds, density, fock,
        generated_task_counts + kernel.shell_class,
        generated_task_heads + kernel.shell_class);
    if (error != cudaSuccess) return error;
  }
  return cudaSuccess;
}

template <bool Unrestricted,
          typename EvalScalar = double,
          unsigned AngularOrder = 0>
void launch_angular_fock_quartets(
    cudaStream_t stream,
    const std::array<std::size_t,
                     detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t,
                     detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch,
    const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles,
    const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles,
    std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if constexpr (AngularOrder < detail::kDirectQuartetAngularOrderCount) {
    if (capacities[AngularOrder] != 0) {
      const std::uint32_t* order_tile_count =
          active_tile_counts + AngularOrder;
      const ActiveShellQuartetTile* order_tiles =
          active_tiles + offsets[AngularOrder];
      if constexpr (
          AngularOrder == kGenericOrderFiveAngularOrder &&
          std::is_same_v<EvalScalar, double>) {
        // Every order-five class currently enabled for generated Fock owns an
        // exact queue. Avoid making the generic persistent worker claim all
        // six subtiles only to decode the class and return.
        order_tile_count = generic_order5_tile_count;
        order_tiles = generic_order5_tiles;
      }
      if constexpr (std::is_same_v<EvalScalar, MixedPrecisionFloat>) {
        // Mixed work begins at order three.  It keeps an independent queue and
        // persistent head so the ERI recurrence contains no per-warp precision
        // branch and low-order shell-fused workers remain unchanged.
        if constexpr (AngularOrder >= kMixedFockMinimumAngularOrder &&
                      AngularOrder < kPersistentFockAngularOrderCount) {
          const unsigned capacity_blocks = static_cast<unsigned>(
              capacities[AngularOrder] *
              detail::direct_quartet_subtiles_per_tile(AngularOrder));
          build_fock_direct_quartet_persistent_kernel<
              Unrestricted, AngularOrder, EvalScalar><<<
                  std::min(capacity_blocks, persistent_worker_blocks),
                  detail::kDirectQuartetThreads, 0, stream>>>(
              batch, order_tile_count, order_tiles,
              persistent_task_heads + AngularOrder, screening_tolerance,
              schwarz_bounds, density, active, fock,
              generated_fock_shell_class_mask);
        } else if constexpr (AngularOrder >=
                             kPersistentFockAngularOrderCount) {
          build_fock_direct_quartet_kernel<
              Unrestricted, AngularOrder, EvalScalar><<<
                  static_cast<unsigned>(
                      capacities[AngularOrder] *
                      detail::direct_quartet_subtiles_per_tile(AngularOrder)),
                  detail::kDirectQuartetThreads, 0, stream>>>(
              batch, order_tile_count, order_tiles, screening_tolerance,
              schwarz_bounds, density, active, fock,
              generated_fock_shell_class_mask);
        }
      } else if constexpr (AngularOrder < kPackedSsssAngularOrderCount) {
        const unsigned capacity_workers = static_cast<unsigned>(
            (capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
            detail::kDirectQuartetThreads);
        build_fock_direct_quartet_packed_persistent_kernel<
            Unrestricted, AngularOrder><<<
                std::min(capacity_workers, persistent_worker_blocks),
                detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles,
            persistent_task_heads + AngularOrder, screening_tolerance,
            schwarz_bounds, density, active, fock);
      } else if constexpr (AngularOrder == kFusedPsssAngularOrder) {
        const unsigned capacity_workers = static_cast<unsigned>(
            (capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
            detail::kDirectQuartetThreads);
        build_fock_direct_psss_persistent_kernel<Unrestricted><<<
            std::min(capacity_workers, persistent_worker_blocks),
            detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles,
            persistent_task_heads + AngularOrder, screening_tolerance,
            schwarz_bounds, density, active, fock);
      } else if constexpr (AngularOrder == kFusedOrderTwoAngularOrder) {
        const unsigned capacity_workers = static_cast<unsigned>(
            (capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
            detail::kDirectQuartetThreads);
        build_fock_direct_order2_persistent_kernel<Unrestricted><<<
            std::min(capacity_workers, persistent_worker_blocks),
            detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles,
            persistent_task_heads + AngularOrder, screening_tolerance,
            schwarz_bounds, density, active, fock,
            generated_fock_shell_class_mask);
      } else if constexpr (AngularOrder <
                           kPersistentFockAngularOrderCount) {
        const unsigned capacity_blocks = static_cast<unsigned>(
            capacities[AngularOrder] *
            detail::direct_quartet_subtiles_per_tile(AngularOrder));
        build_fock_direct_quartet_persistent_kernel<
            Unrestricted, AngularOrder><<<
                std::min(capacity_blocks, persistent_worker_blocks),
                detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles,
            persistent_task_heads + AngularOrder, screening_tolerance,
            schwarz_bounds, density, active, fock,
            generated_fock_shell_class_mask);
      } else {
        build_fock_direct_quartet_kernel<Unrestricted, AngularOrder><<<
            static_cast<unsigned>(
                capacities[AngularOrder] *
                detail::direct_quartet_subtiles_per_tile(AngularOrder)),
            detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles, screening_tolerance,
            schwarz_bounds, density, active, fock,
            generated_fock_shell_class_mask);
      }
    }
    launch_angular_fock_quartets<
        Unrestricted, EvalScalar, AngularOrder + 1>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        generic_order5_tile_count, generic_order5_tiles,
        persistent_task_heads, persistent_worker_blocks,
        screening_tolerance, schwarz_bounds, density, active, fock,
        generated_fock_shell_class_mask);
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
    const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles,
    std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks,
    const PsssResidentTask* psss_resident_tasks,
    const std::uint32_t* psss_resident_ket_pairs,
    std::size_t psss_resident_task_count,
    std::size_t resident_psss_bra_primitive_pairs,
    double screening_tolerance,
    const double* shell_pair_bounds,
    const ShellPairDensityBounds* shell_pair_density_bounds,
    bool force_density_product_screening,
    const double* schwarz_bounds,
    const double* density,
    const std::uint8_t* active,
    double* forces,
    std::uint64_t generated_shell_class_mask) {
  if constexpr (AngularOrder < detail::kDirectQuartetAngularOrderCount) {
    if (capacities[AngularOrder] != 0) {
      const std::uint32_t* order_tile_count =
          active_tile_counts + AngularOrder;
      const ActiveShellQuartetTile* order_tiles =
          active_tiles + offsets[AngularOrder];
      if constexpr (AngularOrder == kGenericOrderFiveAngularOrder) {
        // Generated classes have exact queues. The generic order-five worker
        // consumes only classes not selected by the current runtime mask.
        order_tile_count = generic_order5_tile_count;
        order_tiles = generic_order5_tiles;
      }
      if constexpr (AngularOrder < kPackedSsssAngularOrderCount) {
        const unsigned capacity_workers = static_cast<unsigned>(
            (capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
            detail::kDirectQuartetThreads);
        two_electron_force_quartet_packed_persistent_kernel<
            Unrestricted, AngularOrder><<<
                std::min(capacity_workers, persistent_worker_blocks),
                detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles,
            persistent_task_heads + AngularOrder, screening_tolerance,
            schwarz_bounds, density, active, forces,
            generated_shell_class_mask);
      } else if constexpr (AngularOrder == kFusedPsssAngularOrder) {
        if (psss_resident_task_count != 0 &&
            resident_psss_bra_primitive_pairs != 0 &&
            resident_psss_bra_primitive_pairs <=
                kResidentPsssMaximumBraPrimitivePairs) {
          two_electron_force_psss_resident_bra_kernel<Unrestricted><<<
              static_cast<unsigned>(psss_resident_task_count),
              kResidentPsssThreads,
              resident_psss_bra_primitive_pairs * sizeof(PrimitivePairData),
              stream>>>(
              batch, psss_resident_tasks, psss_resident_ket_pairs,
              psss_resident_task_count, screening_tolerance, shell_pair_bounds,
              shell_pair_density_bounds, force_density_product_screening,
              schwarz_bounds, density, active, forces,
              generated_shell_class_mask);
        } else {
          const unsigned capacity_workers = static_cast<unsigned>(
              (capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
              detail::kDirectQuartetThreads);
          two_electron_force_psss_persistent_kernel<Unrestricted><<<
              std::min(capacity_workers, persistent_worker_blocks),
              detail::kDirectQuartetThreads, 0, stream>>>(
              batch, order_tile_count, order_tiles,
              persistent_task_heads + AngularOrder, screening_tolerance,
              schwarz_bounds, density, active, forces,
              generated_shell_class_mask);
        }
      } else if constexpr (AngularOrder == kFusedOrderTwoAngularOrder) {
        const unsigned capacity_workers = static_cast<unsigned>(
            (capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
            detail::kDirectQuartetThreads);
        two_electron_force_psps_grid_stride_kernel<Unrestricted><<<
            std::min(capacity_workers, persistent_worker_blocks),
            detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles, screening_tolerance,
            schwarz_bounds, density, active, forces,
            generated_shell_class_mask);
        two_electron_force_pair_order2_grid_stride_kernel<
            Unrestricted, kPpssShellClass><<<
                std::min(capacity_workers, persistent_worker_blocks),
                detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles, screening_tolerance,
            schwarz_bounds, density, active, forces,
            generated_shell_class_mask);
        two_electron_force_pair_order2_grid_stride_kernel<
            Unrestricted, kDsssShellClass><<<
                std::min(capacity_workers, persistent_worker_blocks),
                detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles, screening_tolerance,
            schwarz_bounds, density, active, forces,
            generated_shell_class_mask);

        // Exact shell workers own all three order-two classes unless an
        // enabled generated kernel already consumed one. The generic launch
        // is retained as a guarded safety net for future class additions.
        const std::uint64_t generic_shell_class_mask =
            generated_shell_class_mask |
            (std::uint64_t{1} << kPspsShellClass) |
            (std::uint64_t{1} << kPpssShellClass) |
            (std::uint64_t{1} << kDsssShellClass);
        const unsigned capacity_blocks = static_cast<unsigned>(
            capacities[AngularOrder] *
            detail::direct_quartet_subtiles_per_tile(AngularOrder));
        two_electron_force_quartet_persistent_kernel<
            Unrestricted, AngularOrder><<<
                std::min(capacity_blocks, persistent_worker_blocks),
                detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles,
            persistent_task_heads + AngularOrder, screening_tolerance,
            schwarz_bounds, density, active, forces,
            generic_shell_class_mask);
      } else if constexpr (AngularOrder <
                           kPersistentForceAngularOrderCount) {
        const unsigned capacity_blocks = static_cast<unsigned>(
            capacities[AngularOrder] *
            detail::direct_quartet_subtiles_per_tile(AngularOrder));
        two_electron_force_quartet_persistent_kernel<
            Unrestricted, AngularOrder><<<
                std::min(capacity_blocks, persistent_worker_blocks),
                detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles,
            persistent_task_heads + AngularOrder, screening_tolerance,
            schwarz_bounds, density, active, forces,
            generated_shell_class_mask);
      } else {
        two_electron_force_quartet_kernel<Unrestricted, AngularOrder><<<
            static_cast<unsigned>(
              capacities[AngularOrder] *
                detail::direct_quartet_subtiles_per_tile(AngularOrder)),
            detail::kDirectQuartetThreads, 0, stream>>>(
            batch, order_tile_count, order_tiles, screening_tolerance,
            schwarz_bounds, density, active, forces,
            generated_shell_class_mask);
      }
    }
    launch_angular_force_quartets<Unrestricted, AngularOrder + 1>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        generic_order5_tile_count, generic_order5_tiles,
        persistent_task_heads, persistent_worker_blocks,
        psss_resident_tasks, psss_resident_ket_pairs,
        psss_resident_task_count,
        resident_psss_bra_primitive_pairs, screening_tolerance,
        shell_pair_bounds, shell_pair_density_bounds,
        force_density_product_screening, schwarz_bounds, density, active, forces,
        generated_shell_class_mask);
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
  std::size_t system_shell_pair_block_offsets{};
  std::size_t system_shell_pair_block_quartet_offsets{};
  std::size_t shell_pair_systems{};
  std::size_t shell_pair_first{};
  std::size_t shell_pair_second{};
  std::size_t shell_pair_primitive_offsets{};
  std::size_t shell_primitive_pairs{};
  std::size_t psss_resident_tasks{};
  std::size_t psss_resident_ket_pairs{};
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
  // Setup-only flags for rejecting an external warm density before graph
  // capture.  They are deliberately separate from `failed`, whose lifetime
  // spans the SCF graph and denotes numerical solver failures.
  std::size_t warm_invalid{};
  std::size_t overlap{};
  std::size_t hcore{};
  std::size_t eri{};
  std::size_t schwarz_bounds{};
  std::size_t direct_density{};
  std::size_t direct_fock{};
  std::size_t direct_transform_temporary{};
  std::size_t shell_pair_bounds{};
  std::size_t shell_pair_density_bounds{};
  std::size_t bounded_direct_shell_pair_order{};
  std::size_t bounded_stream_shell_pair_order{};
  std::size_t bounded_stream_pair_class_offsets{};
  std::size_t bounded_stream_topology{};
  std::size_t bounded_direct_shell_pair_block_bounds{};
  std::size_t bounded_direct_system_density_bounds{};
  std::size_t bounded_direct_generated_tasks{};
  std::size_t bounded_direct_generated_task_counts{};
  std::size_t bounded_direct_generated_task_offsets{};
  std::size_t bounded_direct_generated_retry_task_offsets{};
  std::size_t bounded_direct_generated_task_heads{};
  std::size_t bounded_direct_generated_overflow{};
  std::size_t bounded_direct_generated_retry_mask{};
  std::size_t bounded_direct_generated_retry_any{};
  std::size_t bounded_force_signature_counts{};
  std::size_t bounded_force_signature_offsets{};
  std::size_t bounded_force_signature_block_offsets{};
  std::size_t bounded_fock_class_timer_starts{};
  std::size_t bounded_fock_class_timer_elapsed{};
  std::size_t bounded_fock_class_timer_launches{};
  std::size_t active_shell_quartet_tile_offsets{};
  std::size_t active_shell_quartet_tile_counts{};
  std::size_t active_shell_quartet_tiles{};
  std::size_t fp32_shell_quartet_tile_offsets{};
  std::size_t fp32_shell_quartet_tile_counts{};
  std::size_t fp32_shell_quartet_tiles{};
  std::size_t shell_class_profile{};
  std::size_t persistent_fock_task_heads{};
  std::size_t fp32_persistent_fock_task_heads{};
  std::size_t persistent_force_task_heads{};
  std::size_t generated_shell_tasks{};
  std::size_t generated_shell_classes{};
  std::size_t generated_shell_task_offsets{};
  std::size_t generated_shell_task_counts{};
  std::size_t generated_shell_task_write_counts{};
  std::size_t generated_shell_task_heads{};
  std::size_t generated_low_order_signature_counts{};
  std::size_t generated_low_order_signature_offsets{};
  std::size_t generated_ppps_resident_tasks{};
  std::size_t generated_ppps_resident_bra_counts{};
  std::size_t generated_ppps_resident_bra_offsets{};
  std::size_t generated_ppps_resident_bra_write_counts{};
  std::size_t generated_ppps_resident_signature_counts{};
  std::size_t generated_ppps_resident_signature_offsets{};
  std::size_t generated_ppps_resident_signatures{};
  std::size_t generated_fock_shell_class_mask{};
  std::size_t generated_mixed_fock_shell_class_mask{};
  std::size_t generic_order5_tiles{};
  std::size_t generic_order5_tile_count{};
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
  std::size_t final_fock_reuse_mask{};
  std::size_t final_fock_rebuild_count{};
  std::size_t spin_active{};
  std::size_t iterations{};
  std::size_t solver_info{};
  std::size_t inactive_eigensolver_profile_count{};
  std::size_t inactive_eigensolver_profile{};
  std::size_t bounded_direct_cursor{};
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
                 std::size_t shell_pair_block_count,
                 std::size_t bounded_generated_task_capacity,
                 std::size_t shell_pair_primitive_count,
                 std::size_t psss_resident_task_count,
                 std::size_t psss_resident_ket_pair_count,
                 std::size_t shell_quartet_tile_count,
                 std::size_t fp32_shell_quartet_tile_count,
                 std::size_t generated_shell_task_capacity,
                 std::size_t ppps_resident_ket_task_capacity,
                 std::size_t generic_order5_tile_capacity,
                 std::size_t primitives,
                 std::size_t diis_history,
                 std::size_t eigensolver_profile_capacity,
                 std::size_t spin_count,
                 bool persistent_eri,
                 bool transformed_direct,
                 bool shell_class_profiling,
                 bool inactive_eigensolver_profiling,
                 bool bounded_fock_class_timing,
                 bool bounded_direct_streaming,
                 bool mixed_precision_fock,
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
  std::size_t ppps_signature_elements = 0;
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
      !checked_multiply(
          ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
          kPppsSignatureBucketCount, ppps_signature_elements) ||
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
      !append_array<std::int64_t>(batch_size + 1, cursor,
                                  made.system_shell_pair_block_offsets) ||
      !append_array<std::int64_t>(
          batch_size + 1, cursor,
          made.system_shell_pair_block_quartet_offsets) ||
      !append_array<std::int32_t>(shell_pair_count, cursor,
                                  made.shell_pair_systems) ||
      !append_array<std::int32_t>(shell_pair_count, cursor,
                                  made.shell_pair_first) ||
      !append_array<std::int32_t>(shell_pair_count, cursor,
                                  made.shell_pair_second) ||
      !append_array<std::int64_t>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming
              ? 0
              : shell_pair_count + 1,
          cursor, made.shell_pair_primitive_offsets) ||
      !append_array<PrimitivePairData>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming
              ? 0
              : shell_pair_primitive_count,
          cursor, made.shell_primitive_pairs) ||
      !append_array<PsssResidentTask>(
          psss_resident_task_count, cursor, made.psss_resident_tasks) ||
      !append_array<std::uint32_t>(
          psss_resident_ket_pair_count, cursor,
          made.psss_resident_ket_pairs) ||
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
      !append_array<std::uint8_t>(batch_size, cursor, made.warm_invalid) ||
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
      !append_array<ShellPairDensityBounds>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming
              ? 0
              : shell_pair_count,
          cursor,
          made.shell_pair_density_bounds) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? shell_pair_count : 0, cursor,
          made.bounded_direct_shell_pair_order) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? shell_pair_count : 0, cursor,
          made.bounded_stream_shell_pair_order) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming
              ? detail::kDirectShellPairClassCount * (batch_size + 1)
              : 0,
          cursor, made.bounded_stream_pair_class_offsets) ||
      !append_array<GeneratedShellPairStream>(
          bounded_direct_streaming ? 1 : 0, cursor,
          made.bounded_stream_topology) ||
      !append_array<double>(
          bounded_direct_streaming ? shell_pair_block_count : 0, cursor,
          made.bounded_direct_shell_pair_block_bounds) ||
      !append_array<double>(
          bounded_direct_streaming ? batch_size : 0, cursor,
          made.bounded_direct_system_density_bounds) ||
      !append_array<GeneratedShellTask>(
          bounded_direct_streaming ? bounded_generated_task_capacity : 0,
          cursor, made.bounded_direct_generated_tasks) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming
              ? detail::kDirectQuartetShellClassCount
              : 0,
          cursor, made.bounded_direct_generated_task_counts) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming
              ? detail::kDirectQuartetShellClassCount + 1
              : 0,
          cursor, made.bounded_direct_generated_task_offsets) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming
              ? detail::kDirectQuartetShellClassCount + 1
              : 0,
          cursor, made.bounded_direct_generated_retry_task_offsets) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming
              ? detail::kDirectQuartetShellClassCount
              : 0,
          cursor, made.bounded_direct_generated_task_heads) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming
              ? detail::kDirectQuartetShellClassCount
              : 0,
          cursor,
          made.bounded_direct_generated_overflow) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming
              ? detail::kDirectQuartetShellClassCount
              : 0,
          cursor, made.bounded_direct_generated_retry_mask) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? 1 : 0, cursor,
          made.bounded_direct_generated_retry_any) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? kBoundedForceSignatureBucketCount : 0,
          cursor, made.bounded_force_signature_counts) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming ? kBoundedForceSignatureBucketCount : 0,
          cursor, made.bounded_force_signature_offsets) ||
      !append_array<std::uint32_t>(
          bounded_direct_streaming
              ? kBoundedForceSignatureScanBlockCount
              : 0,
          cursor, made.bounded_force_signature_block_offsets) ||
      !append_array<std::uint64_t>(
          bounded_fock_class_timing
              ? detail::kDirectQuartetShellClassCount
              : 0,
          cursor, made.bounded_fock_class_timer_starts) ||
      !append_array<std::uint64_t>(
          bounded_fock_class_timing
              ? detail::kDirectQuartetShellClassCount
              : 0,
          cursor, made.bounded_fock_class_timer_elapsed) ||
      !append_array<std::uint32_t>(
          bounded_fock_class_timing
              ? detail::kDirectQuartetShellClassCount
              : 0,
          cursor, made.bounded_fock_class_timer_launches) ||
      !append_array<std::uint32_t>(
          persistent_eri || bounded_direct_streaming
              ? 0
              : detail::kDirectQuartetAngularOrderCount + 1,
          cursor, made.active_shell_quartet_tile_offsets) ||
      !append_array<std::uint32_t>(
          persistent_eri || bounded_direct_streaming
              ? 0
              : detail::kDirectQuartetAngularOrderCount,
          cursor, made.active_shell_quartet_tile_counts) ||
      !append_array<ActiveShellQuartetTile>(
          persistent_eri || bounded_direct_streaming
              ? 0
              : shell_quartet_tile_count,
          cursor,
          made.active_shell_quartet_tiles) ||
      !append_array<std::uint32_t>(
          mixed_precision_fock
              ? detail::kDirectQuartetAngularOrderCount + 1
              : 0,
          cursor, made.fp32_shell_quartet_tile_offsets) ||
      !append_array<std::uint32_t>(
          mixed_precision_fock
              ? detail::kDirectQuartetAngularOrderCount
              : 0,
          cursor, made.fp32_shell_quartet_tile_counts) ||
      !append_array<ActiveShellQuartetTile>(
          mixed_precision_fock ? fp32_shell_quartet_tile_count : 0,
          cursor, made.fp32_shell_quartet_tiles) ||
      !append_array<DeviceShellClassProfileEntry>(
          shell_class_profiling ? detail::kDirectQuartetShellClassCount : 0,
          cursor, made.shell_class_profile) ||
      !append_array<std::uint32_t>(
          shell_quartet_tile_count == 0
              ? 0
              : kPersistentFockAngularOrderCount,
          cursor, made.persistent_fock_task_heads) ||
      !append_array<std::uint32_t>(
          mixed_precision_fock ? kPersistentFockAngularOrderCount : 0,
          cursor, made.fp32_persistent_fock_task_heads) ||
      !append_array<std::uint32_t>(
          shell_quartet_tile_count == 0
              ? 0
              : kPersistentForceAngularOrderCount,
          cursor, made.persistent_force_task_heads) ||
      !append_array<GeneratedShellTask>(
          generated_shell_task_capacity, cursor, made.generated_shell_tasks) ||
      !append_array<std::uint8_t>(
          generated_shell_task_capacity == 0 ? 0 : shell_quartet_tile_count,
          cursor, made.generated_shell_classes) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0
              ? 0
              : detail::kDirectQuartetShellClassCount + 1,
          cursor, made.generated_shell_task_offsets) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0
              ? 0
              : detail::kDirectQuartetShellClassCount,
          cursor, made.generated_shell_task_counts) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0
              ? 0
              : detail::kDirectQuartetShellClassCount,
          cursor, made.generated_shell_task_write_counts) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0
              ? 0
              : detail::kDirectQuartetShellClassCount,
          cursor, made.generated_shell_task_heads) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0
              ? 0
              : kLowOrderSignatureElementCount,
          cursor, made.generated_low_order_signature_counts) ||
      !append_array<std::uint32_t>(
          generated_shell_task_capacity == 0
              ? 0
              : kLowOrderSignatureElementCount,
          cursor, made.generated_low_order_signature_offsets) ||
      !append_array<GeneratedPppsResidentTask>(
          ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
          cursor, made.generated_ppps_resident_tasks) ||
      !append_array<std::uint32_t>(
          ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
          cursor, made.generated_ppps_resident_bra_counts) ||
      !append_array<std::uint32_t>(
          ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count + 1,
          cursor, made.generated_ppps_resident_bra_offsets) ||
      !append_array<std::uint32_t>(
          ppps_resident_ket_task_capacity == 0 ? 0 : shell_pair_count,
          cursor, made.generated_ppps_resident_bra_write_counts) ||
      !append_array<std::uint32_t>(
          ppps_signature_elements,
          cursor, made.generated_ppps_resident_signature_counts) ||
      !append_array<std::uint32_t>(
          ppps_signature_elements,
          cursor, made.generated_ppps_resident_signature_offsets) ||
      !append_array<std::uint32_t>(
          shell_class_profiling ? ppps_resident_ket_task_capacity : 0,
          cursor, made.generated_ppps_resident_signatures) ||
      !append_array<std::uint64_t>(
          shell_quartet_tile_count == 0 && !bounded_direct_streaming ? 0 : 1,
          cursor, made.generated_fock_shell_class_mask) ||
      !append_array<std::uint64_t>(
          mixed_precision_fock ? 1 : 0,
          cursor, made.generated_mixed_fock_shell_class_mask) ||
      !append_array<ActiveShellQuartetTile>(
          generic_order5_tile_capacity, cursor, made.generic_order5_tiles) ||
      !append_array<std::uint32_t>(
          generic_order5_tile_capacity == 0 ? 0 : 1,
          cursor, made.generic_order5_tile_count) ||
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
      !append_array<std::uint8_t>(batch_size, cursor,
                                  made.final_fock_reuse_mask) ||
      !append_array<std::uint32_t>(1, cursor,
                                   made.final_fock_rebuild_count) ||
      !append_array<std::uint8_t>(batch_size * spin_count, cursor,
                                  made.spin_active) ||
      !append_array<std::uint32_t>(batch_size, cursor, made.iterations) ||
      !append_array<int>(batch_size * spin_count, cursor,
                         made.solver_info) ||
      !append_array<std::uint32_t>(
          inactive_eigensolver_profiling ? 1 : 0, cursor,
          made.inactive_eigensolver_profile_count) ||
      !append_array<DeviceInactiveEigensolverProfileEntry>(
          inactive_eigensolver_profiling ? eigensolver_profile_capacity : 0,
          cursor,
          made.inactive_eigensolver_profile) ||
      !append_array<std::uint64_t>(
          bounded_direct_streaming ? 1 : 0, cursor,
          made.bounded_direct_cursor)) return false;
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
  std::vector<std::int64_t> system_shell_pair_block_offsets;
  std::vector<std::int64_t> system_shell_pair_block_quartet_offsets;
  std::vector<std::int32_t> shell_pair_systems;
  std::vector<std::int32_t> shell_pair_first;
  std::vector<std::int32_t> shell_pair_second;
  std::vector<std::int64_t> shell_pair_primitive_offsets;
  std::vector<PsssResidentTask> psss_resident_tasks;
  std::vector<std::uint32_t> psss_resident_ket_pairs;
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

/** Simulate hardware CTA assignment with one descriptor at a time per SM. */
double ppps_profile_schedule_makespan(const std::vector<double>& weights,
                                      unsigned multiprocessor_count) {
  if (weights.empty() || multiprocessor_count == 0U) return 0.0;
  std::vector<double> loads(multiprocessor_count, 0.0);
  for (const double weight : weights) {
    auto next = std::min_element(loads.begin(), loads.end());
    *next += weight;
  }
  return *std::max_element(loads.begin(), loads.end());
}

/**
 * Summarize the exact compacted PPPS queue copied from the device.
 *
 * Signatures retain device materialization order, so the warp-divergence
 * denominator measures the queue that the production kernel actually saw.
 * The scheduling model intentionally stays descriptor-only: it estimates the
 * fixed-bra tail across physical SMs without claiming to reproduce occupancy
 * or instruction-level latency.
 */
CudaPppsQueueProfile build_ppps_queue_profile(
    const HostBatch& host,
    const std::vector<std::uint32_t>& descriptor_counts,
    const std::vector<std::uint32_t>& ordered_signatures,
    unsigned multiprocessor_count) {
  CudaPppsQueueProfile profile;
  profile.descriptor_slots = descriptor_counts.size();
  std::array<std::vector<double>, kPppsProfileBlockThreads.size()>
      task_schedule_weights;
  std::array<std::vector<double>, kPppsProfileBlockThreads.size()>
      primitive_schedule_weights;
  std::size_t ket_begin = 0;
  constexpr std::uint32_t kCountMask = 0x7fffffffU;
  constexpr std::uint32_t kOrientationMask = 0x80000000U;

  for (std::size_t bra_pair = 0; bra_pair < descriptor_counts.size();
       ++bra_pair) {
    const std::size_t ket_count = descriptor_counts[bra_pair];
    if (ket_count == 0U) continue;
    if (ket_begin > ordered_signatures.size() ||
        ket_count > ordered_signatures.size() - ket_begin) {
      // A truncated diagnostic must never be mistaken for valid queue data.
      return {};
    }
    ++profile.non_empty_descriptors;
    profile.tasks += ket_count;
    if (profile.ket_count_histogram.size() <= ket_count) {
      profile.ket_count_histogram.resize(ket_count + 1U, 0U);
    }
    ++profile.ket_count_histogram[ket_count];

    const std::int64_t bra_begin =
        host.shell_pair_primitive_offsets[bra_pair];
    const std::int64_t bra_end =
        host.shell_pair_primitive_offsets[bra_pair + 1U];
    const std::uint64_t bra_primitives = bra_end > bra_begin
        ? static_cast<std::uint64_t>(bra_end - bra_begin)
        : 0U;
    const std::size_t bra_bucket = std::min<std::uint64_t>(
        bra_primitives,
        CudaPppsQueueProfile::kPrimitivePairBucketCount - 1U);
    std::vector<std::uint64_t> primitive_counts(ket_count, 0U);

    for (std::size_t local_ket = 0; local_ket < ket_count; ++local_ket) {
      const std::uint32_t signature =
          ordered_signatures[ket_begin + local_ket];
      const std::size_t orientation =
          (signature & kOrientationMask) == 0U ? 0U : 1U;
      const std::uint64_t ket_primitives = signature & kCountMask;
      const std::uint64_t primitive_work =
          bra_primitives * ket_primitives;
      primitive_counts[local_ket] = primitive_work;
      profile.primitive_work += primitive_work;
      ++profile.orientation_tasks[orientation];
      profile.orientation_primitive_work[orientation] += primitive_work;
      ++profile.bra_primitive_tasks[bra_bucket];
      profile.bra_primitive_work[bra_bucket] += primitive_work;
      const std::size_t ket_bucket = std::min<std::uint64_t>(
          ket_primitives,
          CudaPppsQueueProfile::kPrimitivePairBucketCount - 1U);
      ++profile.ket_primitive_tasks[ket_bucket];
      profile.ket_primitive_work[ket_bucket] += primitive_work;
    }

    for (std::size_t warp_begin = 0; warp_begin < ket_count;
         warp_begin += 32U) {
      const std::size_t warp_end = std::min(ket_count, warp_begin + 32U);
      const std::uint64_t maximum = *std::max_element(
          primitive_counts.begin() + static_cast<std::ptrdiff_t>(warp_begin),
          primitive_counts.begin() + static_cast<std::ptrdiff_t>(warp_end));
      profile.primitive_warp_slots += 32U * maximum;
    }

    for (std::size_t candidate = 0;
         candidate < kPppsProfileBlockThreads.size(); ++candidate) {
      const std::size_t block_threads =
          kPppsProfileBlockThreads[candidate];
      const std::size_t rounds =
          (ket_count + block_threads - 1U) / block_threads;
      profile.lane_slots[candidate] += block_threads * rounds;
      task_schedule_weights[candidate].push_back(
          static_cast<double>(rounds));
      std::uint64_t descriptor_primitive_time = 0U;
      for (std::size_t round_begin = 0; round_begin < ket_count;
           round_begin += block_threads) {
        const std::size_t round_end =
            std::min(ket_count, round_begin + block_threads);
        descriptor_primitive_time += *std::max_element(
            primitive_counts.begin() +
                static_cast<std::ptrdiff_t>(round_begin),
            primitive_counts.begin() +
                static_cast<std::ptrdiff_t>(round_end));
      }
      primitive_schedule_weights[candidate].push_back(
          static_cast<double>(descriptor_primitive_time));
    }
    ket_begin += ket_count;
  }

  for (std::size_t candidate = 0;
       candidate < kPppsProfileBlockThreads.size(); ++candidate) {
    const double task_total = std::accumulate(
        task_schedule_weights[candidate].begin(),
        task_schedule_weights[candidate].end(), 0.0);
    const double primitive_total = std::accumulate(
        primitive_schedule_weights[candidate].begin(),
        primitive_schedule_weights[candidate].end(), 0.0);
    profile.task_schedule_ideal[candidate] =
        task_total / static_cast<double>(multiprocessor_count);
    profile.task_schedule_makespan[candidate] =
        ppps_profile_schedule_makespan(
            task_schedule_weights[candidate], multiprocessor_count);
    profile.primitive_schedule_ideal[candidate] =
        primitive_total / static_cast<double>(multiprocessor_count);
    profile.primitive_schedule_makespan[candidate] =
        ppps_profile_schedule_makespan(
            primitive_schedule_weights[candidate], multiprocessor_count);
  }
  return profile;
}

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
  host.system_shell_pair_block_offsets.push_back(0);
  host.system_shell_pair_block_quartet_offsets.push_back(0);
  host.shell_pair_primitive_offsets.push_back(0);
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
        const std::int64_t first_primitive_count =
            host.shell_primitive_offsets[first + 1] -
            host.shell_primitive_offsets[first];
        const std::int64_t second_primitive_count =
            host.shell_primitive_offsets[second + 1] -
            host.shell_primitive_offsets[second];
        if (first_primitive_count <= 0 || second_primitive_count <= 0 ||
            first_primitive_count >
                std::numeric_limits<std::int64_t>::max() /
                    second_primitive_count) {
          return false;
        }
        const std::int64_t pair_primitive_count =
            first_primitive_count * second_primitive_count;
        if (host.shell_pair_primitive_offsets.back() >
            std::numeric_limits<std::int64_t>::max() -
                pair_primitive_count) {
          return false;
        }
        host.shell_pair_primitive_offsets.push_back(
            host.shell_pair_primitive_offsets.back() +
            pair_primitive_count);
      }
    }
    const std::size_t system_shell_pair_end = host.shell_pair_first.size();
    const std::size_t system_shell_pair_begin = static_cast<std::size_t>(
        host.system_shell_pair_offsets.back());
    const std::size_t system_shell_pair_count =
        system_shell_pair_end - system_shell_pair_begin;
    std::vector<std::uint32_t> psss_bra_pairs;
    const std::size_t resident_ket_begin =
        host.psss_resident_ket_pairs.size();
    for (std::size_t pair = system_shell_pair_begin;
         pair < system_shell_pair_end; ++pair) {
      if (pair > std::numeric_limits<std::uint32_t>::max()) return false;
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      const unsigned first_angular = host.shell_angular[first_shell];
      const unsigned second_angular = host.shell_angular[second_shell];
      if (first_angular + second_angular == 1U) {
        psss_bra_pairs.push_back(static_cast<std::uint32_t>(pair));
      } else if (first_angular == 0U && second_angular == 0U) {
        host.psss_resident_ket_pairs.push_back(
            static_cast<std::uint32_t>(pair));
      }
    }
    const std::size_t resident_ket_count =
        host.psss_resident_ket_pairs.size() - resident_ket_begin;
    if (resident_ket_begin > std::numeric_limits<std::uint32_t>::max() ||
        resident_ket_count > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    for (const std::uint32_t bra_pair : psss_bra_pairs) {
      for (std::size_t ket = 0; ket < resident_ket_count;
           ket += kResidentPsssThreads) {
        const std::size_t chunk_count = std::min<std::size_t>(
            kResidentPsssThreads, resident_ket_count - ket);
        const std::size_t chunk_begin = resident_ket_begin + ket;
        if (chunk_begin > std::numeric_limits<std::uint32_t>::max()) {
          return false;
        }
        host.psss_resident_tasks.push_back(
            {bra_pair, static_cast<std::uint32_t>(chunk_begin),
             static_cast<std::uint32_t>(chunk_count)});
      }
    }
    host.system_shell_pair_offsets.push_back(
        static_cast<std::int64_t>(system_shell_pair_end));
    const std::size_t system_shell_pair_block_count =
        detail::bounded_direct_queue_refill_count(
            system_shell_pair_count,
            detail::kBoundedDirectShellPairBlockSize);
    const std::int64_t previous_block_offset =
        host.system_shell_pair_block_offsets.back();
    if (system_shell_pair_block_count > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max() -
            previous_block_offset)) {
      return false;
    }
    host.system_shell_pair_block_offsets.push_back(
        previous_block_offset +
        static_cast<std::int64_t>(system_shell_pair_block_count));
    std::size_t system_shell_pair_block_plus_one = 0;
    std::size_t system_shell_pair_block_quartet_count = 0;
    if (!checked_add(system_shell_pair_block_count, 1,
                     system_shell_pair_block_plus_one) ||
        !checked_multiply(system_shell_pair_block_count,
                          system_shell_pair_block_plus_one,
                          system_shell_pair_block_quartet_count)) {
      return false;
    }
    system_shell_pair_block_quartet_count /= 2;
    const std::int64_t previous_block_quartet_offset =
        host.system_shell_pair_block_quartet_offsets.back();
    if (system_shell_pair_block_quartet_count > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max() -
            previous_block_quartet_offset)) {
      return false;
    }
    host.system_shell_pair_block_quartet_offsets.push_back(
        previous_block_quartet_offset +
        static_cast<std::int64_t>(
            system_shell_pair_block_quartet_count));
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
    // A supplied warm state is an explicit input, not an optional hint.  The
    // CPU path rejects malformed matrices; silently converting one to a cold
    // guess would make CUDA and CPU disagree and could hide a caller bug.
    if (warm != nullptr &&
        (warm->size() != warm_size ||
         !std::all_of(warm->begin(), warm->end(),
                      [](double value) { return std::isfinite(value); }))) {
      return false;
    }
    const bool valid_warm = warm != nullptr;
    host.warm_mask.push_back(valid_warm ? 1 : 0);
    if (valid_warm) {
      std::copy(warm->begin(), warm->end(),
                host.warm_density.begin() + system_index * warm_size);
    }
  }
  return true;
}

vibeqc_status cuda_status(cudaError_t status) {
  if (status == cudaSuccess) return VIBEQC_STATUS_SUCCESS;
  return status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                              : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status solver_status(cusolverStatus_t status) {
  if (status == CUSOLVER_STATUS_SUCCESS) return VIBEQC_STATUS_SUCCESS;
  return status == CUSOLVER_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status blas_status(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) return VIBEQC_STATUS_SUCCESS;
  return status == CUBLAS_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                              : VIBEQC_STATUS_CUDA_ERROR;
}

void fill_global_failure(std::vector<RhfBucketItem>& outputs, vibeqc_status status) {
  for (RhfBucketItem& output : outputs) output.status = status;
}

class CudaResources {
 public:
  ~CudaResources() {
    if (device_id_ >= 0) (void)cudaSetDevice(device_id_);
    if (post_eigensolver_graph_exec_ != nullptr) {
      (void)cudaGraphExecDestroy(post_eigensolver_graph_exec_);
    }
    if (post_eigensolver_graph_ != nullptr) {
      (void)cudaGraphDestroy(post_eigensolver_graph_);
    }
    if (iteration_graph_exec_ != nullptr) {
      (void)cudaGraphExecDestroy(iteration_graph_exec_);
    }
    if (iteration_graph_ != nullptr) (void)cudaGraphDestroy(iteration_graph_);
    if (jacobi_ != nullptr) (void)cusolverDnDestroySyevjInfo(jacobi_);
    if (solver_parameters_ != nullptr) {
      (void)cusolverDnDestroyParams(solver_parameters_);
    }
    if (solver_ != nullptr) (void)cusolverDnDestroy(solver_);
    if (blas_ != nullptr) (void)cublasDestroy(blas_);
    if (stream_ != nullptr) {
      // Both allocations come from CUDA's stream-ordered device pool. Queue
      // their release on the owning bucket stream so destroying one plan does
      // not impose a device-wide synchronization on unrelated workloads.
      if (solver_workspace_ != nullptr) {
        (void)cudaFreeAsync(solver_workspace_, stream_);
      }
      if (direct_tile_validation_ != nullptr) {
        (void)cudaFreeAsync(direct_tile_validation_, stream_);
      }
      if (arena_ != nullptr) (void)cudaFreeAsync(arena_, stream_);
      (void)cudaStreamSynchronize(stream_);
      (void)cudaStreamDestroy(stream_);
    }
    std::free(solver_host_workspace_);
  }

  int device_id_{-1};
  cudaStream_t stream_{};
  cublasHandle_t blas_{};
  cusolverDnHandle_t solver_{};
  cusolverDnParams_t solver_parameters_{};
  syevjInfo_t jacobi_{};
  cudaGraph_t iteration_graph_{};
  cudaGraphExec_t iteration_graph_exec_{};
  // cuSOLVER XsyevBatched above 512 AOs executes efficiently on an ordinary
  // stream but rejects CUDA Graph capture on CUDA 12.9.  Large-matrix SCF
  // therefore replays a pre-solver Graph, launches the provider normally,
  // then replays this post-solver Graph under host convergence control.
  cudaGraph_t post_eigensolver_graph_{};
  cudaGraphExec_t post_eigensolver_graph_exec_{};
  void* arena_{};
  DirectTileValidationRecord* direct_tile_validation_{};
  void* solver_workspace_{};
  std::size_t solver_workspace_bytes_{};
  void* solver_host_workspace_{};
  std::size_t solver_host_workspace_bytes_{};
};

vibeqc_status copy_to_device(void* destination,
                          const void* source,
                          std::size_t bytes,
                          cudaStream_t stream) {
  if (bytes == 0) return VIBEQC_STATUS_SUCCESS;
  return cuda_status(cudaMemcpyAsync(destination, source, bytes,
                                     cudaMemcpyHostToDevice, stream));
}

vibeqc_status launch_matrix_product(CudaResources& resources,
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
vibeqc_status launch_spin_matrix_product(CudaResources& resources,
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
  return VIBEQC_STATUS_SUCCESS;
}

struct EigensolverProfileLaunch {
  std::int32_t physical_batch_size{};
  const std::uint8_t* physical_active{};
  bool cublas_transformed_inactive{};
  std::uint32_t capacity{};
  std::uint32_t* count{};
  DeviceInactiveEigensolverProfileEntry* entries{};
};

bool provider_eigensolver(CudaEigensolverFamily family) {
  return family == CudaEigensolverFamily::jacobi_batched ||
      family == CudaEigensolverFamily::xsyev_batched;
}

vibeqc_status launch_solver(CudaResources& resources,
                            CudaEigensolverFamily family,
                            int nbf,
                            int batch_size,
                            double* matrices,
                            double* eigenvector_workspace,
                            double* eigenvalues,
                            int lwork,
                            int* info,
                            const std::uint8_t* active,
                            const EigensolverProfileLaunch* profile = nullptr) {
  const bool provider_invoked = provider_eigensolver(family);
  if (profile != nullptr) {
    begin_inactive_eigensolver_profile_kernel<<<1, 1, 0,
                                                resources.stream_>>>(
        profile->physical_batch_size, batch_size,
        static_cast<std::uint32_t>(family), provider_invoked,
        profile->cublas_transformed_inactive, profile->physical_active, active,
        profile->capacity, profile->count, profile->entries);
    const cudaError_t profile_error = cudaPeekAtLastError();
    if (profile_error != cudaSuccess) return cuda_status(profile_error);
  }
  if (provider_invoked) {
    // One block per solver state returns immediately for active matrices. The
    // homogeneous fast path therefore pays one tiny mask kernel while a
    // divergent provider batch receives finite identity placeholders.
    sanitize_inactive_solver_input_kernel<<<
        static_cast<unsigned>(batch_size), kCaptureSafeKernelThreads, 0,
        resources.stream_>>>(
        batch_size, nbf, active, matrices, info,
        profile == nullptr ? 0U : profile->capacity,
        profile == nullptr ? nullptr : profile->count,
        profile == nullptr ? nullptr : profile->entries);
    const cudaError_t sanitize_error = cudaPeekAtLastError();
    if (sanitize_error != cudaSuccess) return cuda_status(sanitize_error);
  }
  if (profile != nullptr) {
    start_inactive_eigensolver_timer_kernel<<<1, 1, 0, resources.stream_>>>(
        profile->capacity, profile->count, profile->entries);
    const cudaError_t profile_error = cudaPeekAtLastError();
    if (profile_error != cudaSuccess) return cuda_status(profile_error);
  }
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  if (family == CudaEigensolverFamily::small_native) {
    symmetric_eigen_small_kernel<<<static_cast<unsigned>(batch_size), 1, 0,
                                   resources.stream_>>>(
        batch_size, nbf, matrices, eigenvalues, info, active);
    status = cuda_status(cudaPeekAtLastError());
  } else if (family == CudaEigensolverFamily::jacobi_batched) {
    const cusolverStatus_t status = cusolverDnDsyevjBatched(
        resources.solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        nbf, matrices, nbf, eigenvalues,
        static_cast<double*>(resources.solver_workspace_), lwork, info,
        resources.jacobi_, batch_size);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      return solver_status(status);
    }
  } else if (family == CudaEigensolverFamily::xsyev_batched) {
    // The setup-time exact-stack probe has already captured, instantiated,
    // host-replayed, and device-tail-replayed this signature.
    const cusolverStatus_t status = cusolverDnXsyevBatched(
        resources.solver_, resources.solver_parameters_,
        CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, nbf, CUDA_R_64F,
        matrices, nbf, CUDA_R_64F, eigenvalues, CUDA_R_64F,
        resources.solver_workspace_, resources.solver_workspace_bytes_,
        resources.solver_host_workspace_,
        resources.solver_host_workspace_bytes_, info, batch_size);
    if (status != CUSOLVER_STATUS_SUCCESS) {
      return solver_status(status);
    }
  } else {
    // API-ineligible or Graph-rejected signatures retain the unbounded native
    // implementation without treating a provider limitation as a calculation
    // failure.
    symmetric_eigen_graph_maximum_pivot_kernel<<<
        static_cast<unsigned>(batch_size), kGraphEigensolverThreads, 0,
        resources.stream_>>>(
        batch_size, nbf, matrices, eigenvector_workspace, eigenvalues, info,
        active);
    status = cuda_status(cudaPeekAtLastError());
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (profile != nullptr) {
    finish_inactive_eigensolver_profile_kernel<<<1, 1, 0,
                                                 resources.stream_>>>(
        batch_size, active, info, profile->capacity, profile->count,
        profile->entries);
    status = cuda_status(cudaPeekAtLastError());
  }
  return status;
}

}  // namespace

struct CudaRhfBucketPlan {
  CudaResources resources;
  ArenaLayout layout;
  HostBatch topology;
  // Geometry-derived arena state is reusable until coordinates change.
  std::vector<double> cached_positions;
  // The current device density and its associated convergence seed are one
  // cache, while a fixed benchmark dm0 and seed are a separate cache. The
  // distinction matters because finalization advances the returned density
  // after evaluating the final energy, so repeated fixed-dm0 replays cease to
  // be resident hits even though they must retain the original energy seed.
  std::vector<double> resident_warm_positions;
  std::vector<double> resident_warm_density;
  std::vector<double> resident_previous_energy;
  std::vector<double> frozen_warm_positions;
  std::vector<double> frozen_warm_density;
  std::vector<double> frozen_previous_energy;
  std::optional<CudaRhfShellClassProfile> last_shell_class_profile;
  std::optional<CudaPppsQueueProfile> last_ppps_queue_profile;
  std::optional<CudaInactiveEigensolverProfile>
      last_inactive_eigensolver_profile;
  CudaEigensolverDiagnostic eigensolver_diagnostic;
  ScfOptions options;
  std::size_t batch_size{};
  std::size_t nbf{};
  std::size_t direct_nbf{};
  std::size_t total_atoms{};
  std::size_t total_shells{};
  std::size_t total_shell_pairs{};
  std::size_t total_shell_quartets{};
  std::size_t total_shell_pair_blocks{};
  std::size_t total_shell_pair_block_quartets{};
  std::size_t total_shell_quartet_tiles{};
  std::vector<std::uint32_t> bounded_direct_shell_pair_order;
  std::vector<std::uint32_t> bounded_stream_shell_pair_order;
  std::vector<std::uint32_t> bounded_stream_pair_class_offsets;
  std::size_t bounded_generated_task_capacity{};
  std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1>
      bounded_generated_task_offsets{};
  std::array<std::uint64_t, detail::kDirectQuartetShellClassCount>
      bounded_generated_task_upper_bounds{};
  std::size_t generated_shell_task_capacity{};
  std::size_t resident_ppps_ket_task_capacity{};
  std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>
      shell_quartet_tile_capacities{};
  std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>
      shell_quartet_tile_offsets{};
  std::size_t fp32_shell_quartet_tile_capacity{};
  std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>
      fp32_shell_quartet_tile_offsets{};
  unsigned persistent_quartet_worker_blocks{};
  std::size_t resident_psss_bra_primitive_pairs{};
  std::size_t resident_psss_task_count{};
  std::size_t primitive_count{};
  std::size_t diis_history{};
  int lwork{};
  bool persistent_eri{};
  bool quartet_direct{};
  bool transformed_direct{};
  bool bounded_direct_streaming{};
  bool unrestricted{};
  bool shell_class_profiling{};
  bool inactive_eigensolver_profiling{};
  bool bounded_fock_class_timing{};
  bool graph_native_eigensolver_override{};
  bool reuse_converged_fock{};
  bool mixed_precision_fock{};
  double mixed_precision_fock_threshold{};
  bool warm_start_updates_enabled{true};
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
         first.system_shell_pair_block_offsets ==
             second.system_shell_pair_block_offsets &&
         first.system_shell_pair_block_quartet_offsets ==
             second.system_shell_pair_block_quartet_offsets &&
         first.shell_pair_systems == second.shell_pair_systems &&
         first.shell_pair_first == second.shell_pair_first &&
         first.shell_pair_second == second.shell_pair_second &&
         first.shell_pair_primitive_offsets ==
             second.shell_pair_primitive_offsets &&
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

bool same_options(const ScfOptions& first,
                  const ScfOptions& second) {
  return first.max_iterations == second.max_iterations &&
         first.diis_history == second.diis_history &&
         first.energy_tolerance == second.energy_tolerance &&
         first.density_tolerance == second.density_tolerance &&
         first.screening_tolerance == second.screening_tolerance;
}

/**
 * Partition the bounded generated-task cache among non-streaming classes.
 *
 * Potential quartet counts are derived from the ten shell-pair angular
 * histograms, so setup stays O(N_shell_pairs) even when the exact quartet
 * topology has billions of entries. Dominant Fock classes need no slice
 * because they enumerate pair segments directly. A class that outgrows its
 * proportional slice is replayed independently through exact-class paged
 * compaction and its generated consumer; it cannot invalidate faster routes
 * for unrelated classes.
 */
std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1>
make_bounded_generated_task_offsets(
    const HostBatch& host,
    std::size_t task_capacity,
    std::array<std::uint64_t,
               detail::kDirectQuartetShellClassCount>* upper_bounds) {
  std::array<std::uint32_t, detail::kDirectQuartetShellClassCount + 1>
      offsets{};
  if (task_capacity == 0) return offsets;

  std::array<bool, detail::kDirectQuartetShellClassCount> force_compiled{};
  std::array<bool, detail::kDirectQuartetShellClassCount> fock_compiled{};
  const auto include_kernels = [](
      std::array<bool, detail::kDirectQuartetShellClassCount>& compiled,
      const generated::ShellKernelMetadata* kernels,
      std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
      if (kernels[index].shell_class < compiled.size()) {
        compiled[kernels[index].shell_class] = true;
      }
    }
  };
  std::size_t force_count = 0;
  const generated::ShellKernelMetadata* force_kernels =
      generated::selected_shell_kernels(force_count);
  include_kernels(force_compiled, force_kernels, force_count);
  std::size_t fock_count = 0;
  const generated::ShellKernelMetadata* fock_kernels =
      generated::selected_fock_shell_kernels(fock_count);
  include_kernels(fock_compiled, fock_kernels, fock_count);
  std::array<bool, detail::kDirectQuartetShellClassCount> compiled{};
  for (std::size_t shell_class = 0; shell_class < compiled.size();
       ++shell_class) {
    const bool queued_fock = fock_compiled[shell_class] &&
        (kStreamingFockShellClassMask &
         (std::uint64_t{1} << shell_class)) == 0U;
    // Fock-only spd rows still compile an exact dormant force symbol for the
    // bounded path.  Include their fixed-capacity slices without advertising
    // them to ordinary fixed-topology force planning.
    const bool queued_force =
        (force_compiled[shell_class] ||
         (fock_compiled[shell_class] &&
          (kStreamingFockShellClassMask &
           (std::uint64_t{1} << shell_class)) != 0U)) &&
        (kDdddShellClassMask &
         (std::uint64_t{1} << shell_class)) == 0U;
    // Direct pair-class streams consume no generated descriptor slice.
    compiled[shell_class] = queued_force || queued_fock;
  }

  std::array<std::uint64_t, detail::kDirectQuartetShellClassCount> weights{};
  for (std::size_t system = 0;
       system + 1 < host.system_shell_pair_offsets.size(); ++system) {
    std::array<std::uint64_t, detail::kDirectShellPairClassCount>
        pair_class_counts{};
    const std::size_t pair_begin = static_cast<std::size_t>(
        host.system_shell_pair_offsets[system]);
    const std::size_t pair_end = static_cast<std::size_t>(
        host.system_shell_pair_offsets[system + 1]);
    for (std::size_t pair = pair_begin; pair < pair_end; ++pair) {
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      const std::size_t pair_class = detail::direct_shell_pair_class(
          host.shell_angular[first_shell], host.shell_angular[second_shell]);
      ++pair_class_counts[pair_class];
    }
    for (std::size_t high = 0; high < pair_class_counts.size(); ++high) {
      for (std::size_t low = 0; low <= high; ++low) {
        const std::uint64_t high_count = pair_class_counts[high];
        const std::uint64_t low_count = pair_class_counts[low];
        const std::uint64_t quartets = high == low
            ? high_count * (high_count + 1U) / 2U
            : high_count * low_count;
        weights[high * (high + 1U) / 2U + low] += quartets;
      }
    }
  }
  if (upper_bounds != nullptr) *upper_bounds = weights;

  std::size_t compiled_count = 0;
  std::uint64_t total_weight = 0;
  for (std::size_t shell_class = 0; shell_class < compiled.size();
       ++shell_class) {
    if (!compiled[shell_class]) continue;
    ++compiled_count;
    total_weight += weights[shell_class];
  }
  if (compiled_count == 0) return offsets;

  std::array<std::uint64_t, detail::kDirectQuartetShellClassCount>
      capacities{};
  const std::uint64_t minimum_per_class = std::min<std::uint64_t>(
      4096U, task_capacity / compiled_count);
  const std::uint64_t reserved = minimum_per_class * compiled_count;
  const std::uint64_t proportional = task_capacity - reserved;
  std::uint64_t assigned = 0;
  for (std::size_t shell_class = 0; shell_class < compiled.size();
       ++shell_class) {
    if (!compiled[shell_class]) continue;
    const std::uint64_t share = total_weight == 0
        ? proportional / compiled_count
        : proportional * weights[shell_class] / total_weight;
    capacities[shell_class] = minimum_per_class + share;
    assigned += capacities[shell_class];
  }
  // Integer division leaves fewer than one task per compiled class. Assign
  // those slots round-robin; exact proportions do not depend on the tail.
  for (std::size_t shell_class = 0; assigned < task_capacity;
       shell_class = (shell_class + 1U) % compiled.size()) {
    if (!compiled[shell_class]) continue;
    ++capacities[shell_class];
    ++assigned;
  }

  std::uint64_t cursor = 0;
  for (std::size_t shell_class = 0; shell_class < compiled.size();
       ++shell_class) {
    offsets[shell_class] = static_cast<std::uint32_t>(cursor);
    cursor += capacities[shell_class];
  }
  offsets[detail::kDirectQuartetShellClassCount] =
      static_cast<std::uint32_t>(cursor);
  return offsets;
}

/** Return shell classes that can actually occur within this batch topology. */
std::uint64_t present_direct_shell_class_mask(const HostBatch& host) {
  static_assert(detail::kDirectQuartetShellClassCount <= 64U);
  std::uint64_t mask = 0U;
  for (std::size_t system = 0;
       system + 1U < host.system_shell_pair_offsets.size(); ++system) {
    std::array<bool, detail::kDirectShellPairClassCount> present_pairs{};
    const std::size_t pair_begin = static_cast<std::size_t>(
        host.system_shell_pair_offsets[system]);
    const std::size_t pair_end = static_cast<std::size_t>(
        host.system_shell_pair_offsets[system + 1U]);
    for (std::size_t pair = pair_begin; pair < pair_end; ++pair) {
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      present_pairs[detail::direct_shell_pair_class(
          host.shell_angular[first_shell],
          host.shell_angular[second_shell])] = true;
    }
    for (std::size_t high = 0; high < present_pairs.size(); ++high) {
      if (!present_pairs[high]) continue;
      for (std::size_t low = 0; low <= high; ++low) {
        if (!present_pairs[low]) continue;
        const std::size_t shell_class = high * (high + 1U) / 2U + low;
        mask |= std::uint64_t{1} << shell_class;
      }
    }
  }
  return mask;
}

/** Build class-major/system-major shell-pair segments for AOT streaming. */
bool make_bounded_stream_shell_pair_order(
    const HostBatch& host,
    std::vector<std::uint32_t>& pair_order,
    std::vector<std::uint32_t>& pair_class_offsets) {
  if (host.system_shell_pair_offsets.empty() ||
      host.shell_pair_first.size() != host.shell_pair_second.size() ||
      host.shell_pair_first.size() >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return false;
  }
  const std::size_t batch_size = host.system_shell_pair_offsets.size() - 1U;
  const std::size_t stride = batch_size + 1U;
  pair_order.clear();
  pair_order.reserve(host.shell_pair_first.size());
  pair_class_offsets.assign(
      detail::kDirectShellPairClassCount * stride, 0U);
  for (std::size_t pair_class = 0;
       pair_class < detail::kDirectShellPairClassCount; ++pair_class) {
    for (std::size_t system = 0; system < batch_size; ++system) {
      pair_class_offsets[pair_class * stride + system] =
          static_cast<std::uint32_t>(pair_order.size());
      const std::size_t pair_begin = static_cast<std::size_t>(
          host.system_shell_pair_offsets[system]);
      const std::size_t pair_end = static_cast<std::size_t>(
          host.system_shell_pair_offsets[system + 1U]);
      for (std::size_t pair = pair_begin; pair < pair_end; ++pair) {
        const std::int32_t first_shell = host.shell_pair_first[pair];
        const std::int32_t second_shell = host.shell_pair_second[pair];
        if (detail::direct_shell_pair_class(
                host.shell_angular[first_shell],
                host.shell_angular[second_shell]) == pair_class) {
          pair_order.push_back(static_cast<std::uint32_t>(pair));
        }
      }
    }
    pair_class_offsets[pair_class * stride + batch_size] =
        static_cast<std::uint32_t>(pair_order.size());
  }
  return pair_order.size() == host.shell_pair_first.size();
}

std::vector<RhfBucketItem> execute_hf_cuda_bucket(
    CudaRhfBucketPlan& plan,
    const HostBatch& host,
    const ScfOptions& options,
    int device_id,
    bool unrestricted,
    bool shell_class_profiling,
    bool inactive_eigensolver_profiling) {
  const std::size_t batch_size = host.warm_mask.size();
  std::vector<RhfBucketItem> outputs(batch_size);
  plan.last_shell_class_profile.reset();
  plan.last_ppps_queue_profile.reset();
  plan.last_inactive_eigensolver_profile.reset();

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
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t pair_count = pair_product / 2;
  const std::size_t direct_pair_count = direct_pair_product / 2;
  std::size_t pair_elements = 0;
  std::size_t direct_pair_elements = 0;
  if (!checked_multiply(batch_size, pair_count, pair_elements) ||
      !checked_multiply(batch_size, direct_pair_count,
                        direct_pair_elements)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
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
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_atoms = host.atomic_numbers.size();
  const std::size_t total_shells = host.shell_atoms.size();
  const std::size_t total_shell_pairs = host.shell_pair_first.size();
  if (host.shell_pair_primitive_offsets.size() != total_shell_pairs + 1 ||
      host.shell_pair_primitive_offsets.back() < 0) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_pair_primitives =
      static_cast<std::size_t>(host.shell_pair_primitive_offsets.back());
  if (host.system_shell_quartet_offsets.size() != batch_size + 1 ||
      host.system_shell_quartet_offsets.back() < 0) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_quartets =
      static_cast<std::size_t>(host.system_shell_quartet_offsets.back());
  if (host.system_shell_pair_block_offsets.size() != batch_size + 1 ||
      host.system_shell_pair_block_quartet_offsets.size() != batch_size + 1 ||
      host.system_shell_pair_block_offsets.back() < 0 ||
      host.system_shell_pair_block_quartet_offsets.back() < 0) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t total_shell_pair_blocks = static_cast<std::size_t>(
      host.system_shell_pair_block_offsets.back());
  const std::size_t total_shell_pair_block_quartets =
      static_cast<std::size_t>(
          host.system_shell_pair_block_quartet_offsets.back());
  const bool requested_persistent_eri = nbf <= kPersistentEriAoLimit;
  const bool requested_quartet_direct =
      !requested_persistent_eri &&
      std::all_of(host.shell_angular.begin(), host.shell_angular.end(),
                  [](std::uint8_t angular) { return angular <= 3; });
  const bool requested_transformed_direct =
      requested_quartet_direct && direct_nbf != nbf;
  bool requested_bounded_direct_streaming =
      requested_quartet_direct &&
      (detail::direct_topology_requires_bounded_streaming(
           total_shell_quartets) ||
       bounded_direct_streaming_override_requested());
  const bool cooperative_one_electron_force =
      one_electron_force_scalar_requested();
  const bool requested_graph_native_eigensolver_override =
      graph_native_eigensolver_override_requested();
  const bool xsyev_probe_skip_diagnostic =
      xsyev_probe_skip_diagnostic_requested();
  // Read this on every cached execution so one prepared batch can provide a
  // fixed-dm0 old/new A/B without rebuilding its immutable topology plan.
  const bool force_density_product_screening =
      force_density_product_screening_requested();
  const bool bounded_direct_count_diagnostic =
      bounded_direct_count_diagnostic_requested();
  const bool bounded_direct_aot_only_diagnostic =
      bounded_direct_aot_only_diagnostic_requested();
  const bool bounded_direct_fock_only_diagnostic =
      bounded_direct_fock_only_diagnostic_requested();
  const bool bounded_fock_class_timing =
      bounded_fock_class_timing_requested();
  const bool direct_tile_validation = direct_tile_validation_requested();
  // Read this per execution so one prepared topology can compare the new
  // route with the complete ordinary ppps queue in the same binary.
  const bool resident_ppps_bra = resident_ppps_bra_requested();
  const bool resident_ppps_signature_bucketing =
      ppps_signature_bucketing_requested();
  const bool psps_signature_bucketing =
      psps_signature_bucketing_requested();
  const bool ppss_signature_bucketing =
      ppss_signature_bucketing_requested();
  const unsigned resident_ppps_block_threads =
      ppps_resident_block_threads_requested();
  const bool first_setup = !plan.initialized;
  detail::DirectQuartetTaskLayout direct_task_layout{};
  std::size_t total_shell_quartet_tiles = 0;
  if (requested_quartet_direct && first_setup &&
      !requested_bounded_direct_streaming) {
    // Exact topology capacities are immutable for a prepared bucket. Their
    // pair-of-pairs enumeration is O(n_shell_pairs^2), so recomputing it on
    // every warm replay adds substantial host latency at large AO counts.
    if (!detail::make_direct_quartet_task_layout(
            host.shell_direct_ao_offsets, host.shell_angular,
            host.system_shell_pair_offsets, host.shell_pair_first,
            host.shell_pair_second,
            direct_task_layout) ||
        direct_task_layout.shell_quartet_count != total_shell_quartets) {
      fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
    total_shell_quartet_tiles = direct_task_layout.exact_tile_count;
    if (total_shell_quartet_tiles >
        detail::kDirectFixedTopologyTileLimit) {
      // A high-angular topology can exceed the fixed grid before its quartet
      // count alone proves that fact. Discard the setup-only exact counts and
      // use the same bounded device enumerator as obviously large systems.
      requested_bounded_direct_streaming = true;
      direct_task_layout = {};
      total_shell_quartet_tiles = 0;
    }
  } else if (requested_quartet_direct) {
    requested_bounded_direct_streaming =
        first_setup ? requested_bounded_direct_streaming
                    : plan.bounded_direct_streaming;
    total_shell_quartet_tiles = requested_bounded_direct_streaming
        ? 0
        : plan.total_shell_quartet_tiles;
  }
  const std::optional<double> requested_mixed_precision_fock_threshold =
      requested_quartet_direct
      ? configured_mixed_precision_fock_threshold(
            options.screening_tolerance)
      : std::nullopt;
  const bool requested_mixed_precision_fock =
      requested_mixed_precision_fock_threshold.has_value();
  // An iterative mixed Fock is not the exact final matrix associated with the
  // converged density. Force a complete FP64 rebuild before final energy,
  // orbitals, and analytic forces consume it.
  const bool requested_reuse_converged_fock =
      reuse_converged_fock_requested() && !requested_mixed_precision_fock;
  // Direct consumers expand each compact logical tile into one-warp blocks;
  // validate the resulting fixed Graph grid before narrowing it to unsigned.
  if (total_shell_pairs > std::numeric_limits<unsigned>::max() ||
      (!requested_bounded_direct_streaming &&
       total_shell_quartets > std::numeric_limits<unsigned>::max()) ||
      host.psss_resident_tasks.size() >
          std::numeric_limits<unsigned>::max() ||
      (!requested_bounded_direct_streaming &&
       total_shell_quartet_tiles >
           detail::kDirectFixedTopologyTileLimit)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  std::size_t force_coordinate_count = 0;
  std::size_t one_electron_force_elements = 0;
  std::size_t force_matrix_elements = 0;
  std::size_t persistent_force_elements = 0;
  std::size_t direct_force_elements = 0;
  if (!checked_multiply(total_atoms, 3, force_coordinate_count) ||
      !checked_multiply(batch_size, pair_count,
                        one_electron_force_elements) ||
      !checked_multiply(force_coordinate_count, matrix_size,
                        force_matrix_elements) ||
      !checked_multiply(force_coordinate_count, eri_size,
                        persistent_force_elements) ||
      !checked_multiply(force_matrix_elements, pair_count,
                        direct_force_elements)) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::size_t diis_history = std::max<std::size_t>(1, options.diis_history);
  if (diis_history > 64) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (!first_setup &&
      (plan.resources.device_id_ != device_id ||
       !same_topology(plan.topology, host) ||
       !same_options(plan.options, options) ||
       plan.unrestricted != unrestricted ||
       plan.bounded_direct_streaming !=
           requested_bounded_direct_streaming ||
       plan.shell_class_profiling != shell_class_profiling ||
       plan.inactive_eigensolver_profiling !=
           inactive_eigensolver_profiling ||
       plan.bounded_fock_class_timing != bounded_fock_class_timing ||
       plan.graph_native_eigensolver_override !=
           requested_graph_native_eigensolver_override ||
       plan.reuse_converged_fock != requested_reuse_converged_fock ||
       plan.mixed_precision_fock != requested_mixed_precision_fock ||
       plan.mixed_precision_fock_threshold !=
           requested_mixed_precision_fock_threshold.value_or(0.0))) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  std::size_t generated_shell_task_capacity =
      first_setup ? 0 : plan.generated_shell_task_capacity;
  std::size_t resident_ppps_ket_task_capacity =
      first_setup ? 0 : plan.resident_ppps_ket_task_capacity;
  std::size_t fp32_shell_quartet_tile_capacity =
      first_setup ? 0 : plan.fp32_shell_quartet_tile_capacity;
  if (first_setup && requested_mixed_precision_fock) {
    for (std::size_t order = kMixedFockMinimumAngularOrder;
         order < detail::kDirectQuartetAngularOrderCount; ++order) {
      if (!checked_add(
              fp32_shell_quartet_tile_capacity,
              direct_task_layout.angular_order_tile_counts[order],
              fp32_shell_quartet_tile_capacity)) {
        fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
        return outputs;
      }
    }
    if (fp32_shell_quartet_tile_capacity >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
      fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
  }
  const std::size_t generic_order5_tile_capacity =
      requested_quartet_direct && !requested_bounded_direct_streaming
      ? (first_setup
             ? direct_task_layout.angular_order_tile_counts[
                   kGenericOrderFiveAngularOrder]
             : plan.shell_quartet_tile_capacities[
                   kGenericOrderFiveAngularOrder])
      : 0;
  std::size_t bounded_generated_task_capacity =
      first_setup ? 0 : plan.bounded_generated_task_capacity;
  if (first_setup && requested_bounded_direct_streaming) {
    if (!checked_multiply(total_shell_pairs,
                          kBoundedGeneratedTasksPerShellPair,
                          bounded_generated_task_capacity)) {
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    bounded_generated_task_capacity = std::min(
        bounded_generated_task_capacity,
        kBoundedGeneratedMaximumTaskCapacity);
  }
  if (requested_quartet_direct && first_setup &&
      !requested_bounded_direct_streaming) {
    // The shared generated-task arena serves both exact Fock and force
    // consumers.  Their registries are intentionally not identical: for
    // example, `ssss` has a generated Fock consumer but remains on the
    // handwritten force path.  Build the capacity from their union so a
    // Fock-only class cannot leave its persistent kernel with a zero-sized
    // task arena.
    std::array<bool, detail::kDirectQuartetShellClassCount>
        generated_task_classes{};
    const auto include_generated_task_classes =
        [&](const generated::ShellKernelMetadata* kernels,
            std::size_t kernel_count) {
          for (std::size_t kernel_index = 0; kernel_index < kernel_count;
               ++kernel_index) {
            const unsigned shell_class = kernels[kernel_index].shell_class;
            if (shell_class < generated_task_classes.size()) {
              generated_task_classes[shell_class] = true;
            }
          }
        };
    std::size_t force_kernel_count = 0;
    const generated::ShellKernelMetadata* force_kernels =
        generated::selected_shell_kernels(force_kernel_count);
    include_generated_task_classes(force_kernels, force_kernel_count);
    std::size_t fock_kernel_count = 0;
    const generated::ShellKernelMetadata* fock_kernels =
        generated::selected_fock_shell_kernels(fock_kernel_count);
    include_generated_task_classes(fock_kernels, fock_kernel_count);
    for (std::size_t shell_class = 0;
         shell_class < generated_task_classes.size(); ++shell_class) {
      if (!generated_task_classes[shell_class]) continue;
      if (!checked_add(
              generated_shell_task_capacity,
              direct_task_layout.shell_class_tile_counts[shell_class],
              generated_shell_task_capacity)) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
    }
    // Every active canonical ppps shell quartet occupies one tile at angular
    // order three, so this exact class count is the maximum resident ket
    // record count.  Reserve a reusable tail of the existing generated-task
    // arena only when the selected AOT bundle contains a ppps force consumer;
    // portable/stub builds then retain their original arena footprint.
    bool ppps_force_available = false;
    for (std::size_t kernel_index = 0; kernel_index < force_kernel_count;
         ++kernel_index) {
      if (force_kernels[kernel_index].shell_class == kPppsShellClass) {
        ppps_force_available = true;
        break;
      }
    }
    resident_ppps_ket_task_capacity = ppps_force_available
        ? direct_task_layout.shell_class_tile_counts[kPppsShellClass]
        : 0;
  }
  if (resident_ppps_ket_task_capacity > generated_shell_task_capacity ||
      resident_ppps_ket_task_capacity > static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max())) {
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  if (first_setup) {
    if (!make_layout(batch_size, nbf, direct_nbf, total_atoms, total_shells,
                     total_shell_pairs, total_shell_pair_blocks,
                     bounded_generated_task_capacity,
                     total_shell_pair_primitives,
                     requested_quartet_direct
                         && !requested_bounded_direct_streaming
                         ? host.psss_resident_tasks.size()
                         : 0,
                     requested_quartet_direct
                         && !requested_bounded_direct_streaming
                         ? host.psss_resident_ket_pairs.size()
                         : 0,
                     total_shell_quartet_tiles,
                     fp32_shell_quartet_tile_capacity,
                     generated_shell_task_capacity,
                     resident_ppps_ket_task_capacity,
                     generic_order5_tile_capacity,
                     host.primitive_exponents.size(), diis_history,
                     options.max_iterations,
                     host.spin_count,
                     requested_persistent_eri,
                     requested_transformed_direct,
                     shell_class_profiling,
                     inactive_eigensolver_profiling,
                     bounded_fock_class_timing,
                     requested_bounded_direct_streaming,
                     requested_mixed_precision_fock,
                     plan.layout)) {
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    plan.batch_size = batch_size;
    plan.nbf = nbf;
    plan.direct_nbf = direct_nbf;
    plan.total_atoms = total_atoms;
    plan.total_shells = total_shells;
    plan.total_shell_pairs = total_shell_pairs;
    plan.total_shell_quartets = total_shell_quartets;
    plan.total_shell_pair_blocks = total_shell_pair_blocks;
    plan.total_shell_pair_block_quartets =
        total_shell_pair_block_quartets;
    plan.total_shell_quartet_tiles = total_shell_quartet_tiles;
    plan.bounded_generated_task_capacity = bounded_generated_task_capacity;
    plan.bounded_generated_task_offsets = requested_bounded_direct_streaming
        ? make_bounded_generated_task_offsets(
              host, bounded_generated_task_capacity,
              &plan.bounded_generated_task_upper_bounds)
        : std::array<std::uint32_t,
                     detail::kDirectQuartetShellClassCount + 1>{};
    if (!requested_bounded_direct_streaming) {
      plan.bounded_generated_task_upper_bounds.fill(0U);
    }
    if (requested_bounded_direct_streaming) {
      plan.bounded_direct_shell_pair_order.resize(total_shell_pairs);
      std::iota(plan.bounded_direct_shell_pair_order.begin(),
                plan.bounded_direct_shell_pair_order.end(), 0U);
      if (!make_bounded_stream_shell_pair_order(
              host, plan.bounded_stream_shell_pair_order,
              plan.bounded_stream_pair_class_offsets)) {
        fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
        return outputs;
      }
    }
    plan.resident_psss_bra_primitive_pairs = 0;
    const bool resident_psss_enabled = resident_psss_bra_requested();
    plan.resident_psss_task_count =
        requested_quartet_direct && !requested_bounded_direct_streaming &&
            resident_psss_enabled
        ? host.psss_resident_tasks.size()
        : 0;
    for (std::size_t pair = 0; pair < total_shell_pairs; ++pair) {
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      if (host.shell_angular[first_shell] +
              host.shell_angular[second_shell] != 1U) {
        continue;
      }
      const std::size_t primitive_pairs = static_cast<std::size_t>(
          host.shell_pair_primitive_offsets[pair + 1] -
          host.shell_pair_primitive_offsets[pair]);
      plan.resident_psss_bra_primitive_pairs = std::max(
          plan.resident_psss_bra_primitive_pairs, primitive_pairs);
    }
    plan.generated_shell_task_capacity = generated_shell_task_capacity;
    plan.resident_ppps_ket_task_capacity =
        resident_ppps_ket_task_capacity;
    plan.shell_quartet_tile_capacities =
        direct_task_layout.angular_order_tile_counts;
    for (std::size_t order = 0;
         order < direct_task_layout.angular_order_tile_offsets.size();
         ++order) {
      plan.shell_quartet_tile_offsets[order] = static_cast<std::uint32_t>(
          direct_task_layout.angular_order_tile_offsets[order]);
    }
    plan.fp32_shell_quartet_tile_capacity =
        fp32_shell_quartet_tile_capacity;
    std::size_t fp32_tile_offset = 0;
    for (std::size_t order = 0;
         order < detail::kDirectQuartetAngularOrderCount; ++order) {
      plan.fp32_shell_quartet_tile_offsets[order] =
          static_cast<std::uint32_t>(fp32_tile_offset);
      if (requested_mixed_precision_fock &&
          order >= kMixedFockMinimumAngularOrder) {
        fp32_tile_offset +=
            direct_task_layout.angular_order_tile_counts[order];
      }
    }
    plan.fp32_shell_quartet_tile_offsets[
        detail::kDirectQuartetAngularOrderCount] =
        static_cast<std::uint32_t>(fp32_tile_offset);
    plan.primitive_count = host.primitive_exponents.size();
    plan.diis_history = diis_history;
    plan.persistent_eri = requested_persistent_eri;
    plan.quartet_direct = requested_quartet_direct;
    plan.transformed_direct = requested_transformed_direct;
    plan.bounded_direct_streaming =
        requested_bounded_direct_streaming;
    plan.unrestricted = unrestricted;
    plan.shell_class_profiling = shell_class_profiling;
    plan.inactive_eigensolver_profiling =
        inactive_eigensolver_profiling;
    plan.bounded_fock_class_timing = bounded_fock_class_timing;
    plan.graph_native_eigensolver_override =
        requested_graph_native_eigensolver_override;
    plan.reuse_converged_fock = requested_reuse_converged_fock;
    plan.mixed_precision_fock = requested_mixed_precision_fock;
    plan.mixed_precision_fock_threshold =
        requested_mixed_precision_fock_threshold.value_or(0.0);
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
  const bool bounded_direct_streaming = plan.bounded_direct_streaming;
  const bool reuse_converged_fock = plan.reuse_converged_fock;
  const bool mixed_precision_fock = plan.mixed_precision_fock;
  const double mixed_precision_fock_threshold =
      plan.mixed_precision_fock_threshold;
  if (first_setup) {
    plan.eigensolver_diagnostic = {};
    plan.eigensolver_diagnostic.matrix_dimension = nbf;
    plan.eigensolver_diagnostic.physical_system_count = batch_size;
    plan.eigensolver_diagnostic.solver_batch_count = spin_batch_size;
    if (nbf <= static_cast<std::size_t>(kSmallEigensolverLimit)) {
      plan.eigensolver_diagnostic.ordinary_family =
          CudaEigensolverFamily::small_native;
      plan.eigensolver_diagnostic.family =
          CudaEigensolverFamily::small_native;
    } else if (nbf <= static_cast<std::size_t>(kBatchedEigensolverLimit)) {
      plan.eigensolver_diagnostic.ordinary_family =
          CudaEigensolverFamily::jacobi_batched;
      plan.eigensolver_diagnostic.family =
          CudaEigensolverFamily::jacobi_batched;
    } else {
      plan.eigensolver_diagnostic.xsyev_probe =
          probe_xsyev_batched_device_launch_graph(
              device_id, nbf, spin_batch_size);
      const XsyevBatchedDispatch dispatch = select_xsyev_batched_dispatch(
          plan.eigensolver_diagnostic.xsyev_probe);
      plan.eigensolver_diagnostic.family =
          dispatch.device_launch_graph_provider
          ? CudaEigensolverFamily::xsyev_batched
          : CudaEigensolverFamily::graph_native;
      plan.eigensolver_diagnostic.ordinary_family =
          dispatch.ordinary_stream_provider
          ? CudaEigensolverFamily::xsyev_batched
          : CudaEigensolverFamily::graph_native;
      plan.eigensolver_diagnostic.selection_source =
          xsyev_probe_skip_diagnostic
          ? CudaEigensolverSelectionSource::benchmark_override
          : plan.eigensolver_diagnostic.xsyev_probe.graph_eligible
          ? CudaEigensolverSelectionSource::exact_probe
          : CudaEigensolverSelectionSource::exact_probe_fallback;
    }
    if (requested_graph_native_eigensolver_override) {
      plan.eigensolver_diagnostic.family =
          CudaEigensolverFamily::graph_native;
      plan.eigensolver_diagnostic.selection_source =
          CudaEigensolverSelectionSource::benchmark_override;
    }
  }
  const CudaEigensolverFamily graph_eigensolver_family =
      plan.eigensolver_diagnostic.family;
  const CudaEigensolverFamily ordinary_eigensolver_family =
      plan.eigensolver_diagnostic.ordinary_family;
  const bool use_jacobi =
      graph_eigensolver_family == CudaEigensolverFamily::jacobi_batched ||
      ordinary_eigensolver_family == CudaEigensolverFamily::jacobi_batched;
  const bool use_cusolver = use_jacobi ||
      ordinary_eigensolver_family == CudaEigensolverFamily::xsyev_batched;
  const bool geometry_changed =
      first_setup || plan.cached_positions != host.positions;
  const bool all_systems_warm = std::all_of(
      host.warm_mask.begin(), host.warm_mask.end(),
      [](std::uint8_t value) { return value != 0; });
  const bool any_system_warm = std::any_of(
      host.warm_mask.begin(), host.warm_mask.end(),
      [](std::uint8_t value) { return value != 0; });
  // Density residency and the previous-energy baseline are deliberately
  // independent. A fixed warm start can reuse its frozen energy seed after a
  // prior replay advanced the device density. Geometry remains part of a
  // resident-density hit because applying an external dm0 also renormalizes
  // its electron trace against the geometry-dependent overlap matrix.
  const bool device_resident_density_hit =
      all_systems_warm &&
      plan.resident_warm_positions == host.positions &&
      plan.resident_warm_density == host.warm_density;
  const bool resident_energy_baseline_hit =
      device_resident_density_hit &&
      plan.resident_warm_positions == host.positions &&
      plan.resident_previous_energy.size() == batch_size;
  const bool frozen_energy_baseline_hit =
      all_systems_warm && !plan.warm_start_updates_enabled &&
      plan.frozen_warm_density == host.warm_density &&
      plan.frozen_warm_positions == host.positions &&
      plan.frozen_previous_energy.size() == batch_size;
  const bool cached_energy_baseline_hit =
      frozen_energy_baseline_hit || resident_energy_baseline_hit;
  // Copy the tiny seed vector locally before invalidating residency. Any
  // early CUDA or validation failure below may have partially changed the
  // device density; only a fully successful execution republishes it.
  std::vector<double> host_previous_energy_seed;
  if (frozen_energy_baseline_hit) {
    host_previous_energy_seed = plan.frozen_previous_energy;
  } else if (resident_energy_baseline_hit) {
    host_previous_energy_seed = plan.resident_previous_energy;
  }
  plan.resident_warm_positions.clear();
  plan.resident_warm_density.clear();
  plan.resident_previous_energy.clear();
  const bool use_cublas = plan.cublas_enabled &&
      nbf >= kCublasMatrixProductAoThreshold;
  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  if (first_setup && quartet_direct) {
    int multiprocessor_count = 0;
    cuda_error = cudaDeviceGetAttribute(
        &multiprocessor_count, cudaDevAttrMultiProcessorCount, device_id);
    if (cuda_error != cudaSuccess || multiprocessor_count <= 0 ||
        static_cast<unsigned>(multiprocessor_count) >
            std::numeric_limits<unsigned>::max() /
                kPersistentQuartetWarpsPerMultiprocessor) {
      fill_global_failure(
          outputs, cuda_error == cudaSuccess
              ? VIBEQC_STATUS_INVALID_ARGUMENT
              : cuda_status(cuda_error));
      return outputs;
    }
    plan.persistent_quartet_worker_blocks =
        static_cast<unsigned>(multiprocessor_count) *
        kPersistentQuartetWarpsPerMultiprocessor;
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
    if (quartet_direct &&
        (cuda_error = cudaMallocAsync(
             &resources.direct_tile_validation_,
             sizeof(DirectTileValidationRecord), resources.stream_)) !=
            cudaSuccess) {
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
              CUSOLVER_STATUS_SUCCESS) {
        fill_global_failure(outputs, solver_status(solver_error));
        return outputs;
      }
      if (use_jacobi) {
        if ((solver_error = cusolverDnCreateSyevjInfo(&resources.jacobi_)) !=
                CUSOLVER_STATUS_SUCCESS ||
            (solver_error = cusolverDnXsyevjSetTolerance(
                 resources.jacobi_, 1.0e-13)) != CUSOLVER_STATUS_SUCCESS ||
            (solver_error = cusolverDnXsyevjSetMaxSweeps(
                 resources.jacobi_, 100)) != CUSOLVER_STATUS_SUCCESS ||
            (solver_error = cusolverDnXsyevjSetSortEig(
                 resources.jacobi_, 1)) != CUSOLVER_STATUS_SUCCESS) {
          fill_global_failure(outputs, solver_status(solver_error));
          return outputs;
        }
      } else if ((solver_error = cusolverDnCreateParams(
                      &resources.solver_parameters_)) !=
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
  auto system_shell_pair_block_offsets = arena_pointer<std::int64_t>(
      resources.arena_, layout.system_shell_pair_block_offsets);
  auto system_shell_pair_block_quartet_offsets =
      arena_pointer<std::int64_t>(
          resources.arena_,
          layout.system_shell_pair_block_quartet_offsets);
  auto shell_pair_systems = arena_pointer<std::int32_t>(
      resources.arena_, layout.shell_pair_systems);
  auto shell_pair_first = arena_pointer<std::int32_t>(
      resources.arena_, layout.shell_pair_first);
  auto shell_pair_second = arena_pointer<std::int32_t>(
      resources.arena_, layout.shell_pair_second);
  auto shell_pair_primitive_offsets = arena_pointer<std::int64_t>(
      resources.arena_, layout.shell_pair_primitive_offsets);
  auto shell_primitive_pairs = arena_pointer<PrimitivePairData>(
      resources.arena_, layout.shell_primitive_pairs);
  auto psss_resident_tasks = arena_pointer<PsssResidentTask>(
      resources.arena_, layout.psss_resident_tasks);
  auto psss_resident_ket_pairs = arena_pointer<std::uint32_t>(
      resources.arena_, layout.psss_resident_ket_pairs);
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
  auto warm_invalid = arena_pointer<std::uint8_t>(resources.arena_, layout.warm_invalid);
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
  auto shell_pair_density_bounds = arena_pointer<ShellPairDensityBounds>(
      resources.arena_, layout.shell_pair_density_bounds);
  auto bounded_direct_shell_pair_order = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_direct_shell_pair_order);
  auto bounded_stream_shell_pair_order = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_stream_shell_pair_order);
  auto bounded_stream_pair_class_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_stream_pair_class_offsets);
  auto bounded_stream_topology = arena_pointer<GeneratedShellPairStream>(
      resources.arena_, layout.bounded_stream_topology);
  auto bounded_direct_shell_pair_block_bounds = arena_pointer<double>(
      resources.arena_, layout.bounded_direct_shell_pair_block_bounds);
  auto bounded_direct_system_density_bounds = arena_pointer<double>(
      resources.arena_, layout.bounded_direct_system_density_bounds);
  auto bounded_direct_generated_tasks = arena_pointer<GeneratedShellTask>(
      resources.arena_, layout.bounded_direct_generated_tasks);
  auto bounded_direct_generated_task_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_direct_generated_task_counts);
  auto bounded_direct_generated_task_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_direct_generated_task_offsets);
  auto bounded_direct_generated_retry_task_offsets =
      arena_pointer<std::uint32_t>(
          resources.arena_,
          layout.bounded_direct_generated_retry_task_offsets);
  auto bounded_direct_generated_task_heads = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_direct_generated_task_heads);
  auto bounded_direct_generated_overflow = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_direct_generated_overflow);
  auto bounded_direct_generated_retry_mask = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_direct_generated_retry_mask);
  auto bounded_direct_generated_retry_any = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_direct_generated_retry_any);
  auto bounded_force_signature_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_force_signature_counts);
  auto bounded_force_signature_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_force_signature_offsets);
  auto bounded_force_signature_block_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.bounded_force_signature_block_offsets);
  std::uint64_t* bounded_fock_class_timer_starts =
      bounded_fock_class_timing
      ? arena_pointer<std::uint64_t>(
            resources.arena_, layout.bounded_fock_class_timer_starts)
      : nullptr;
  std::uint64_t* bounded_fock_class_timer_elapsed =
      bounded_fock_class_timing
      ? arena_pointer<std::uint64_t>(
            resources.arena_, layout.bounded_fock_class_timer_elapsed)
      : nullptr;
  std::uint32_t* bounded_fock_class_timer_launches =
      bounded_fock_class_timing
      ? arena_pointer<std::uint32_t>(
            resources.arena_, layout.bounded_fock_class_timer_launches)
      : nullptr;
  auto active_shell_quartet_tile_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.active_shell_quartet_tile_offsets);
  auto active_shell_quartet_tile_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.active_shell_quartet_tile_counts);
  auto active_shell_quartet_tiles = arena_pointer<ActiveShellQuartetTile>(
      resources.arena_, layout.active_shell_quartet_tiles);
  std::uint32_t* fp32_shell_quartet_tile_offsets = mixed_precision_fock
      ? arena_pointer<std::uint32_t>(
            resources.arena_, layout.fp32_shell_quartet_tile_offsets)
      : nullptr;
  std::uint32_t* fp32_shell_quartet_tile_counts = mixed_precision_fock
      ? arena_pointer<std::uint32_t>(
            resources.arena_, layout.fp32_shell_quartet_tile_counts)
      : nullptr;
  ActiveShellQuartetTile* fp32_shell_quartet_tiles = mixed_precision_fock
      ? arena_pointer<ActiveShellQuartetTile>(
            resources.arena_, layout.fp32_shell_quartet_tiles)
      : nullptr;
  auto shell_class_profile = arena_pointer<DeviceShellClassProfileEntry>(
      resources.arena_, layout.shell_class_profile);
  auto persistent_fock_task_heads = arena_pointer<std::uint32_t>(
      resources.arena_, layout.persistent_fock_task_heads);
  std::uint32_t* fp32_persistent_fock_task_heads = mixed_precision_fock
      ? arena_pointer<std::uint32_t>(
            resources.arena_, layout.fp32_persistent_fock_task_heads)
      : nullptr;
  auto persistent_force_task_heads = arena_pointer<std::uint32_t>(
      resources.arena_, layout.persistent_force_task_heads);
  auto generated_shell_tasks = arena_pointer<GeneratedShellTask>(
      resources.arena_, layout.generated_shell_tasks);
  auto generated_shell_classes = arena_pointer<std::uint8_t>(
      resources.arena_, layout.generated_shell_classes);
  auto generated_shell_task_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_shell_task_offsets);
  auto generated_shell_task_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_shell_task_counts);
  auto generated_shell_task_write_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_shell_task_write_counts);
  auto generated_shell_task_heads = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_shell_task_heads);
  auto generated_low_order_signature_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_low_order_signature_counts);
  auto generated_low_order_signature_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_low_order_signature_offsets);
  auto generated_ppps_resident_tasks = arena_pointer<GeneratedPppsResidentTask>(
      resources.arena_, layout.generated_ppps_resident_tasks);
  // Final force preparation is ordered after the last Fock consumer on the
  // same stream.  Reuse the ppps-sized tail of the ordinary generated-task
  // arena for resident ket records, then let ordinary force preparation
  // overwrite it only after the resident launch completes.  This avoids a
  // multi-gigabyte duplicate queue at the 384-AO endpoint.
  GeneratedShellTask* generated_ppps_resident_ket_tasks = nullptr;
  if (resident_ppps_ket_task_capacity != 0) {
    generated_ppps_resident_ket_tasks = generated_shell_tasks +
        (generated_shell_task_capacity - resident_ppps_ket_task_capacity);
  }
  auto generated_ppps_resident_bra_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_ppps_resident_bra_counts);
  auto generated_ppps_resident_bra_offsets = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_ppps_resident_bra_offsets);
  auto generated_ppps_resident_bra_write_counts = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generated_ppps_resident_bra_write_counts);
  auto generated_ppps_resident_signature_counts =
      arena_pointer<std::uint32_t>(
          resources.arena_,
          layout.generated_ppps_resident_signature_counts);
  auto generated_ppps_resident_signature_offsets =
      arena_pointer<std::uint32_t>(
          resources.arena_,
          layout.generated_ppps_resident_signature_offsets);
  // A zero-sized arena slice still has an offset, so do not turn it into a
  // writable pointer when profiling did not allocate per-task signatures.
  std::uint32_t* generated_ppps_resident_signatures =
      shell_class_profiling
      ? arena_pointer<std::uint32_t>(
            resources.arena_, layout.generated_ppps_resident_signatures)
      : nullptr;
  auto generated_fock_shell_class_mask = arena_pointer<std::uint64_t>(
      resources.arena_, layout.generated_fock_shell_class_mask);
  std::uint64_t* generated_mixed_fock_shell_class_mask = mixed_precision_fock
      ? arena_pointer<std::uint64_t>(
            resources.arena_, layout.generated_mixed_fock_shell_class_mask)
      : nullptr;
  auto generic_order5_tiles = arena_pointer<ActiveShellQuartetTile>(
      resources.arena_, layout.generic_order5_tiles);
  auto generic_order5_tile_count = arena_pointer<std::uint32_t>(
      resources.arena_, layout.generic_order5_tile_count);
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
  auto final_fock_reuse_mask = arena_pointer<std::uint8_t>(
      resources.arena_, layout.final_fock_reuse_mask);
  auto final_fock_rebuild_count = arena_pointer<std::uint32_t>(
      resources.arena_, layout.final_fock_rebuild_count);
  auto spin_active =
      arena_pointer<std::uint8_t>(resources.arena_, layout.spin_active);
  auto iterations = arena_pointer<std::uint32_t>(resources.arena_, layout.iterations);
  auto solver_info = arena_pointer<int>(resources.arena_, layout.solver_info);
  std::uint32_t* inactive_eigensolver_profile_count =
      inactive_eigensolver_profiling
      ? arena_pointer<std::uint32_t>(
            resources.arena_, layout.inactive_eigensolver_profile_count)
      : nullptr;
  DeviceInactiveEigensolverProfileEntry* inactive_eigensolver_profile =
      inactive_eigensolver_profiling
      ? arena_pointer<DeviceInactiveEigensolverProfileEntry>(
            resources.arena_, layout.inactive_eigensolver_profile)
      : nullptr;
  auto bounded_direct_cursor = arena_pointer<unsigned long long>(
      resources.arena_, layout.bounded_direct_cursor);

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

  const std::size_t shell_quartet_offset_bytes =
      quartet_direct && !bounded_direct_streaming
      ? plan.shell_quartet_tile_offsets.size() * sizeof(std::uint32_t)
      : 0;
  const std::size_t fp32_shell_quartet_offset_bytes = mixed_precision_fock
      ? plan.fp32_shell_quartet_tile_offsets.size() * sizeof(std::uint32_t)
      : 0;
  const GeneratedShellPairStream host_bounded_stream_topology{
      static_cast<std::int32_t>(batch_size),
      static_cast<std::uint32_t>(direct_nbf),
      system_shell_offsets,
      system_shell_pair_offsets,
      shell_atoms,
      shell_angular,
      shell_direct_ao_offsets,
      shell_primitive_offsets,
      shell_pair_systems,
      shell_pair_first,
      shell_pair_second,
      bounded_stream_shell_pair_order,
      bounded_stream_pair_class_offsets,
      shell_pair_bounds,
      reinterpret_cast<const detail::GeneratedShellPairDensityBounds*>(
          shell_pair_density_bounds),
      active};
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
      {host.system_shell_pair_block_offsets.data(),
       {system_shell_pair_block_offsets,
        host.system_shell_pair_block_offsets.size() * sizeof(std::int64_t)}},
      {host.system_shell_pair_block_quartet_offsets.data(),
       {system_shell_pair_block_quartet_offsets,
        host.system_shell_pair_block_quartet_offsets.size() *
            sizeof(std::int64_t)}},
      {host.shell_pair_systems.data(),
       {shell_pair_systems,
        host.shell_pair_systems.size() * sizeof(std::int32_t)}},
      {host.shell_pair_first.data(),
       {shell_pair_first, host.shell_pair_first.size() * sizeof(std::int32_t)}},
      {host.shell_pair_second.data(),
       {shell_pair_second,
        host.shell_pair_second.size() * sizeof(std::int32_t)}},
      {host.shell_pair_primitive_offsets.data(),
       {shell_pair_primitive_offsets,
        quartet_direct
            ? host.shell_pair_primitive_offsets.size() *
                  sizeof(std::int64_t)
            : 0}},
      {host.psss_resident_tasks.data(),
       {psss_resident_tasks,
        quartet_direct && !bounded_direct_streaming
            ? host.psss_resident_tasks.size() * sizeof(PsssResidentTask)
            : 0}},
      {host.psss_resident_ket_pairs.data(),
       {psss_resident_ket_pairs,
        quartet_direct && !bounded_direct_streaming
            ? host.psss_resident_ket_pairs.size() * sizeof(std::uint32_t)
            : 0}},
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
      {plan.fp32_shell_quartet_tile_offsets.data(),
       {fp32_shell_quartet_tile_offsets,
        fp32_shell_quartet_offset_bytes}},
      {plan.bounded_generated_task_offsets.data(),
       {bounded_direct_generated_task_offsets,
        bounded_direct_streaming
            ? plan.bounded_generated_task_offsets.size() *
                  sizeof(std::uint32_t)
            : 0}},
      {plan.bounded_direct_shell_pair_order.data(),
       {bounded_direct_shell_pair_order,
        bounded_direct_streaming
            ? plan.bounded_direct_shell_pair_order.size() *
                  sizeof(std::uint32_t)
            : 0}},
      {plan.bounded_stream_shell_pair_order.data(),
       {bounded_stream_shell_pair_order,
        bounded_direct_streaming
            ? plan.bounded_stream_shell_pair_order.size() *
                  sizeof(std::uint32_t)
            : 0}},
      {plan.bounded_stream_pair_class_offsets.data(),
       {bounded_stream_pair_class_offsets,
        bounded_direct_streaming
            ? plan.bounded_stream_pair_class_offsets.size() *
                  sizeof(std::uint32_t)
            : 0}},
      {&host_bounded_stream_topology,
       {bounded_stream_topology,
        bounded_direct_streaming ? sizeof(GeneratedShellPairStream) : 0}},
      {host_pair_first.data(),
       {ao_pair_first, host_pair_first.size() * sizeof(std::int32_t)}},
      {host_pair_second.data(),
       {ao_pair_second, host_pair_second.size() * sizeof(std::int32_t)}},
  };
  // Registry selections may include f-shell kernels for a batch containing
  // only s/p/d shells.  Intersect with the topology before deciding whether
  // the streaming set is complete; otherwise an impossible class forces an
  // O(N_shell^4) generic bounded scan.
  const std::uint64_t host_present_shell_class_mask =
      quartet_direct ? present_direct_shell_class_mask(host) : 0U;
  const std::uint64_t host_generated_fock_shell_class_mask =
      (generated::enabled_fock_shell_class_mask() &
       host_present_shell_class_mask) &
      (bounded_direct_streaming
           ? std::numeric_limits<std::uint64_t>::max()
           : ~kFixedTopologyGeneratedFockExclusionMask);
  const std::uint64_t host_generated_mixed_fock_shell_class_mask =
      mixed_precision_fock
      ? generated::enabled_mixed_fock_shell_class_mask() &
          host_generated_fock_shell_class_mask & host_present_shell_class_mask
      : 0U;
  const std::uint64_t host_generated_streaming_fock_shell_class_mask =
      host_generated_fock_shell_class_mask &
      kGeneratedStreamingFockShellClassMask;
  const std::uint64_t host_native_streaming_fock_shell_class_mask =
      host_generated_fock_shell_class_mask &
      kNativeStreamingFockShellClassMask;
  const std::uint64_t host_streaming_fock_shell_class_mask =
      host_generated_streaming_fock_shell_class_mask |
      host_native_streaming_fock_shell_class_mask;
  const std::uint64_t host_uncovered_fock_shell_class_mask =
      host_present_shell_class_mask &
      ~host_generated_fock_shell_class_mask;
  const std::pair<const void*, std::pair<void*, std::size_t>>
      dynamic_uploads[] = {
      {host.warm_mask.data(),
       {warm_mask,
        device_resident_density_hit
            ? 0
            : host.warm_mask.size() * sizeof(std::uint8_t)}},
      {host.warm_density.data(),
       {warm_density,
        device_resident_density_hit
            ? 0
            : host.warm_density.size() * sizeof(double)}},
      {&host_generated_fock_shell_class_mask,
       {generated_fock_shell_class_mask,
        quartet_direct ? sizeof(std::uint64_t) : 0}},
      {&host_generated_mixed_fock_shell_class_mask,
       {generated_mixed_fock_shell_class_mask,
        mixed_precision_fock ? sizeof(std::uint64_t) : 0}},
  };
  if (first_setup) {
    for (const auto& upload : static_uploads) {
      const vibeqc_status status = copy_to_device(
          upload.second.first, upload.first, upload.second.second,
          resources.stream_);
      if (status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
    }
  }
  if (geometry_changed) {
    const vibeqc_status position_status = copy_to_device(
        positions, host.positions.data(),
        host.positions.size() * sizeof(double), resources.stream_);
    if (position_status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, position_status);
      return outputs;
    }
  }
  for (const auto& upload : dynamic_uploads) {
    const vibeqc_status status = copy_to_device(
        upload.second.first, upload.first, upload.second.second,
        resources.stream_);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  }
  if (cached_energy_baseline_hit) {
    // The SCF graph initializes previous_energy from this device buffer. A
    // frozen replay therefore restores its original seed explicitly instead
    // of relying on whatever energy the most recent resident density left.
    const vibeqc_status status = copy_to_device(
        energy, host_previous_energy_seed.data(),
        host_previous_energy_seed.size() * sizeof(double), resources.stream_);
    if (status != VIBEQC_STATUS_SUCCESS) {
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
      static_cast<std::int64_t>(total_shell_quartets),
      static_cast<std::int64_t>(total_shell_pair_blocks),
      static_cast<std::int64_t>(total_shell_pair_block_quartets), atom_offsets,
      atom_systems, atomic_numbers, positions, system_shell_offsets,
      shell_atoms, shell_angular, shell_ao_offsets, shell_direct_ao_offsets,
      shell_primitive_offsets, system_shell_pair_offsets,
      system_shell_quartet_offsets, system_shell_pair_block_offsets,
      system_shell_pair_block_quartet_offsets,
      shell_pair_systems, shell_pair_first, shell_pair_second,
      shell_pair_primitive_offsets, shell_primitive_pairs, ao_shells,
      ao_term_counts, ao_term_angular, ao_term_coefficients,
      direct_ao_shells, direct_ao_angular, direct_ao_coefficients,
      ao_to_direct_transform,
      primitive_exponents,
      primitive_coefficients, occupied};

  if (quartet_direct && geometry_changed) {
    build_shell_primitive_pair_cache_kernel<<<
        static_cast<unsigned>(total_shell_pairs),
        detail::kDirectQuartetThreads, 0, resources.stream_>>>(
        device_batch, shell_primitive_pairs);
    cuda_error = cudaPeekAtLastError();
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }

  if (first_setup && use_cusolver) {
    if (use_jacobi) {
      solver_error = cusolverDnDsyevjBatched_bufferSize(
          resources.solver_, CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_LOWER, static_cast<int>(nbf), eigensystem,
          static_cast<int>(nbf), eigenvalues, &plan.lwork,
          resources.jacobi_, static_cast<int>(spin_batch_size));
      resources.solver_workspace_bytes_ =
          static_cast<std::size_t>(plan.lwork) * sizeof(double);
    } else {
      // RHF submits batch_size matrices; UHF additionally submits the doubled
      // spin batch. Query both actual capacities because cuSOLVER does not
      // guarantee workspace sizes are monotonic in batch count.
      const std::array<int, 2> capacities{
          static_cast<int>(batch_size), static_cast<int>(spin_batch_size)};
      for (const int capacity : capacities) {
        std::size_t device_bytes = 0;
        std::size_t host_bytes = 0;
        solver_error = cusolverDnXsyevBatched_bufferSize(
            resources.solver_, resources.solver_parameters_,
            CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            static_cast<int>(nbf), CUDA_R_64F, eigensystem,
            static_cast<int>(nbf), CUDA_R_64F, eigenvalues, CUDA_R_64F,
            &device_bytes, &host_bytes, capacity);
        if (solver_error != CUSOLVER_STATUS_SUCCESS) break;
        resources.solver_workspace_bytes_ = std::max(
            resources.solver_workspace_bytes_, device_bytes);
        resources.solver_host_workspace_bytes_ = std::max(
            resources.solver_host_workspace_bytes_, host_bytes);
      }
      plan.lwork = 0;
    }
    if (solver_error != CUSOLVER_STATUS_SUCCESS) {
      fill_global_failure(outputs, solver_status(solver_error));
      return outputs;
    }
    if (plan.lwork < 0 || resources.solver_workspace_bytes_ == 0) {
      fill_global_failure(outputs, VIBEQC_STATUS_CUDA_ERROR);
      return outputs;
    }
    if ((cuda_error = cudaMallocAsync(
             &resources.solver_workspace_,
             resources.solver_workspace_bytes_,
             resources.stream_)) != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (resources.solver_host_workspace_bytes_ != 0) {
      resources.solver_host_workspace_ =
          std::malloc(resources.solver_host_workspace_bytes_);
      if (resources.solver_host_workspace_ == nullptr) {
        fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
        return outputs;
      }
    }
    if (use_cublas) {
      blas_error = cublasSetWorkspace(
          resources.blas_, resources.solver_workspace_,
          resources.solver_workspace_bytes_);
      if (blas_error != CUBLAS_STATUS_SUCCESS) {
        plan.retry_without_cublas = true;
        fill_global_failure(outputs, blas_status(blas_error));
        return outputs;
      }
    }
  } else if (first_setup) {
    plan.lwork = 0;
  }
  const int lwork = plan.lwork;

  constexpr unsigned threads =
      kCaptureSafeKernelThreads;
  constexpr unsigned matrix_reduction_threads = kMatrixReductionThreads;
  const auto blocks_for = [](std::size_t elements) {
    return static_cast<unsigned>((elements + threads - 1) / threads);
  };
  const auto multiply_matrices = [&](const double* left,
                                     bool transpose_left,
                                     const double* right,
                                     double* output) {
    const vibeqc_status product_status = launch_matrix_product(
        resources, static_cast<int>(batch_size), static_cast<int>(nbf),
        left, transpose_left, right, active, output, use_cublas);
    if (use_cublas && product_status != VIBEQC_STATUS_SUCCESS) {
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
    const vibeqc_status product_status = launch_spin_matrix_product(
        resources, static_cast<int>(batch_size), 2, static_cast<int>(nbf),
        left, left_is_spin, transpose_left, right, right_is_spin, active,
        output, use_cublas);
    if (use_cublas && product_status != VIBEQC_STATUS_SUCCESS) {
      plan.retry_without_cublas = true;
    }
    return product_status;
  };
  const auto build_commutator_residual = [&]() -> vibeqc_status {
    // [F, P]S is evaluated as four O(N^3) products.  `temporary` and
    // `eigensystem` are iteration scratch at this point: the former holds the
    // first product until it is folded into `residual`, while the latter is
    // overwritten before DIIS uses it as its effective-Fock output.  Keeping
    // the products in separate launches also lets the existing cuBLAS
    // strided-batched wrapper handle RHF and interleaved UHF layouts alike.
    vibeqc_status product_status = VIBEQC_STATUS_SUCCESS;
    if (unrestricted) {
      product_status = multiply_spin_matrices(
          fock, true, false, density, true, temporary);
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_spin_matrices(
            temporary, true, false, overlap, false, residual);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_spin_matrices(
            overlap, false, false, density, true, eigensystem);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_spin_matrices(
            eigensystem, true, false, fock, true, temporary);
      }
    } else {
      product_status = multiply_matrices(fock, false, density, temporary);
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_matrices(temporary, false, overlap, residual);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_matrices(overlap, false, density, eigensystem);
      }
      if (product_status == VIBEQC_STATUS_SUCCESS) {
        product_status = multiply_matrices(eigensystem, false, fock, temporary);
      }
    }
    if (product_status != VIBEQC_STATUS_SUCCESS) return product_status;
    subtract_matrix_batches_kernel<<<
        blocks_for(spin_matrix_elements), threads, 0, resources.stream_>>>(
        static_cast<std::int32_t>(batch_size),
        static_cast<std::int32_t>(spin_count), static_cast<std::int32_t>(nbf),
        temporary, active, residual);
    return cuda_status(cudaPeekAtLastError());
  };
  const auto launch_direct_quartet_metadata =
      [&](const double* density_input,
          bool allow_mixed_precision) -> cudaError_t {
    if (!quartet_direct) return cudaSuccess;
    if (direct_tile_validation &&
        resources.direct_tile_validation_ != nullptr) {
      cudaError_t validation_error = cudaMemsetAsync(
          resources.direct_tile_validation_, 0xff,
          sizeof(DirectTileValidationRecord), resources.stream_);
      if (validation_error != cudaSuccess) return validation_error;
    }
    const double* quartet_density = density_input;
    if (transformed_direct) {
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
      quartet_density = direct_density;
    }
    if (!bounded_direct_streaming) {
      clear_active_shell_quartet_tile_counts_kernel<<<
          blocks_for(detail::kDirectQuartetAngularOrderCount), threads, 0,
          resources.stream_>>>(active_shell_quartet_tile_counts,
                               persistent_fock_task_heads,
                               fp32_shell_quartet_tile_counts,
                               fp32_persistent_fock_task_heads);
    }
    if (unrestricted) {
      reduce_shell_pair_density_bounds_kernel<true><<<
          static_cast<unsigned>(total_shell_pairs), threads,
          3 * threads * sizeof(double), resources.stream_>>>(
          device_batch, quartet_density, active,
          shell_pair_density_bounds);
      if (bounded_direct_streaming) {
        reduce_bounded_system_density_bounds_kernel<<<
            static_cast<unsigned>(batch_size), threads,
            threads * sizeof(double), resources.stream_>>>(
            device_batch, shell_pair_density_bounds,
            bounded_direct_system_density_bounds);
        return cudaPeekAtLastError();
      }
      compact_active_shell_quartet_tiles_kernel<
          true, DirectScreeningPurpose::Fock><<<
              blocks_for(total_shell_quartets), threads, 0,
              resources.stream_>>>(
          device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active,
          active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          allow_mixed_precision && mixed_precision_fock,
          mixed_precision_fock_threshold,
          fp32_shell_quartet_tile_offsets,
          fp32_shell_quartet_tile_counts, fp32_shell_quartet_tiles);
    } else {
      reduce_shell_pair_density_bounds_kernel<false><<<
          static_cast<unsigned>(total_shell_pairs), threads,
          3 * threads * sizeof(double), resources.stream_>>>(
          device_batch, quartet_density, active,
          shell_pair_density_bounds);
      if (bounded_direct_streaming) {
        reduce_bounded_system_density_bounds_kernel<<<
            static_cast<unsigned>(batch_size), threads,
            threads * sizeof(double), resources.stream_>>>(
            device_batch, shell_pair_density_bounds,
            bounded_direct_system_density_bounds);
        return cudaPeekAtLastError();
      }
      compact_active_shell_quartet_tiles_kernel<
          false, DirectScreeningPurpose::Fock><<<
              blocks_for(total_shell_quartets), threads, 0,
              resources.stream_>>>(
          device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active,
          active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          allow_mixed_precision && mixed_precision_fock,
          mixed_precision_fock_threshold,
          fp32_shell_quartet_tile_offsets,
          fp32_shell_quartet_tile_counts, fp32_shell_quartet_tiles);
    }
    if (direct_tile_validation &&
        resources.direct_tile_validation_ != nullptr) {
      validate_direct_tile_descriptors_kernel<<<
          blocks_for(plan.total_shell_quartet_tiles), threads, 0,
          resources.stream_>>>(
          device_batch, active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          plan.total_shell_quartet_tiles, resources.direct_tile_validation_);
    }
    return cudaPeekAtLastError();
  };
  const auto launch_direct_force_compaction = [&]() -> cudaError_t {
    if (!quartet_direct || !force_density_product_screening) {
      return cudaSuccess;
    }
    if (bounded_direct_streaming) {
      // The force kernel reapplies the stronger density-product predicate as
      // it enumerates pair-of-pairs, so no intermediate force queue exists.
      return cudaSuccess;
    }
    // The final Fock/metadata path above has already reduced the selected
    // density in the direct Cartesian AO domain. Reuse those bounds and
    // overwrite the no-longer-needed Fock queue with its force-only subset.
    clear_active_shell_quartet_tile_counts_kernel<<<
        blocks_for(detail::kDirectQuartetAngularOrderCount), threads, 0,
        resources.stream_>>>(active_shell_quartet_tile_counts,
                             persistent_fock_task_heads,
                             fp32_shell_quartet_tile_counts,
                             fp32_persistent_fock_task_heads);
    if (unrestricted) {
      compact_active_shell_quartet_tiles_kernel<
          true, DirectScreeningPurpose::Force><<<
              blocks_for(total_shell_quartets), threads, 0,
              resources.stream_>>>(
          device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active,
          active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          false, 0.0, nullptr, nullptr, nullptr);
    } else {
      compact_active_shell_quartet_tiles_kernel<
          false, DirectScreeningPurpose::Force><<<
              blocks_for(total_shell_quartets), threads, 0,
              resources.stream_>>>(
          device_batch, options.screening_tolerance, shell_pair_bounds,
          shell_pair_density_bounds, active,
          active_shell_quartet_tile_offsets,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          false, 0.0, nullptr, nullptr, nullptr);
    }
    return cudaPeekAtLastError();
  };
  std::size_t bounded_fock_kernel_count = 0;
  const generated::ShellKernelMetadata* bounded_fock_kernels =
      generated::selected_fock_shell_kernels(bounded_fock_kernel_count);
  const auto launch_bounded_streaming_fock =
      [&](bool is_unrestricted,
          const double* quartet_density,
          double* quartet_fock,
          bool allow_mixed_precision) -> cudaError_t {
    // Every selected class owns one independent queue head.  Reset the
    // complete fixed-size head array in one asynchronous memset before the
    // class-major launches instead of issuing one host API call per class.
    // The heads are disjoint, so this preserves launch ordering and atomic
    // accumulation semantics while removing serial dispatch overhead from
    // the bounded warm path.
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_heads, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
        resources.stream_);
    if (error != cudaSuccess) return error;
    for (std::size_t kernel_index = 0;
         kernel_index < bounded_fock_kernel_count; ++kernel_index) {
      const unsigned shell_class =
          bounded_fock_kernels[kernel_index].shell_class;
      if ((host_generated_streaming_fock_shell_class_mask &
           (std::uint64_t{1} << shell_class)) == 0U) {
        continue;
      }
      if (bounded_fock_class_timing) {
        start_bounded_fock_class_timer_kernel<<<1, 1, 0,
                                                resources.stream_>>>(
            shell_class, bounded_fock_class_timer_starts);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
      }
      error = generated::launch_shell_class_streaming_fock(
          shell_class, resources.stream_, is_unrestricted,
          plan.persistent_quartet_worker_blocks, bounded_stream_topology,
          device_batch.shell_pair_primitive_offsets,
          device_batch.shell_primitive_pairs,
          device_batch.direct_ao_coefficients, device_batch.positions,
          options.screening_tolerance,
          allow_mixed_precision && mixed_precision_fock &&
              (host_generated_mixed_fock_shell_class_mask &
               (std::uint64_t{1} << shell_class)) != 0U,
          mixed_precision_fock_threshold, schwarz_bounds, quartet_density,
          quartet_fock,
          bounded_direct_generated_task_heads + shell_class);
      if (error != cudaSuccess) return error;
      if (bounded_fock_class_timing) {
        finish_bounded_fock_class_timer_kernel<<<1, 1, 0,
                                                 resources.stream_>>>(
            shell_class, bounded_fock_class_timer_starts,
            bounded_fock_class_timer_elapsed,
            bounded_fock_class_timer_launches);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return error;
      }
    }
    if ((host_native_streaming_fock_shell_class_mask &
         kDdddShellClassMask) == 0U) {
      return cudaSuccess;
    }
    // The complete head-array reset above also covers native DDDD.  Do not
    // issue a second class-specific memset here: the native fallback uses the
    // same disjoint head slot as generated classes.
    if (bounded_fock_class_timing) {
      start_bounded_fock_class_timer_kernel<<<1, 1, 0,
                                              resources.stream_>>>(
          kDdddShellClass, bounded_fock_class_timer_starts);
      error = cudaPeekAtLastError();
      if (error != cudaSuccess) return error;
    }
    if (is_unrestricted) {
      bounded_direct_dddd_streaming_kernel<
          true, DirectScreeningPurpose::Fock, false><<<
              plan.persistent_quartet_worker_blocks,
              detail::kDirectQuartetThreads, 0, resources.stream_>>>(
          device_batch, bounded_stream_topology,
          options.screening_tolerance, schwarz_bounds, quartet_density,
          active, quartet_fock,
          bounded_direct_generated_task_heads + kDdddShellClass, nullptr);
    } else {
      bounded_direct_dddd_streaming_kernel<
          false, DirectScreeningPurpose::Fock, false><<<
              plan.persistent_quartet_worker_blocks,
              detail::kDirectQuartetThreads, 0, resources.stream_>>>(
          device_batch, bounded_stream_topology,
          options.screening_tolerance, schwarz_bounds, quartet_density,
          active, quartet_fock,
          bounded_direct_generated_task_heads + kDdddShellClass, nullptr);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (bounded_fock_class_timing) {
      finish_bounded_fock_class_timer_kernel<<<1, 1, 0,
                                               resources.stream_>>>(
          kDdddShellClass, bounded_fock_class_timer_starts,
          bounded_fock_class_timer_elapsed,
          bounded_fock_class_timer_launches);
      error = cudaPeekAtLastError();
    }
    return error;
  };
  const auto launch_bounded_generated_fock =
      [&](bool is_unrestricted,
          const double* quartet_density,
          double* quartet_fock,
          bool allow_mixed_precision) -> cudaError_t {
    // The bounded Fock path follows the same hard routing invariant as force:
    // every present class must have a generated or native exact consumer.
    // Missing classes are unsupported instead of silently invoking the
    // whole-topology generic evaluator.
    if (host_uncovered_fock_shell_class_mask != 0U &&
        !bounded_direct_aot_only_diagnostic) {
      return cudaErrorNotSupported;
    }
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_counts, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
        resources.stream_);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(
        bounded_direct_generated_overflow,
        bounded_fock_kernel_count == 0 ? 1 : 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
        resources.stream_);
    if (error != cudaSuccess || bounded_fock_kernel_count == 0) return error;
    if (!bounded_direct_count_diagnostic &&
        (host_generated_fock_shell_class_mask &
         ~host_streaming_fock_shell_class_mask) == 0U) {
      return launch_bounded_streaming_fock(
          is_unrestricted, quartet_density, quartet_fock,
          allow_mixed_precision);
    }
    error = cudaMemsetAsync(
        bounded_direct_cursor, 0, sizeof(unsigned long long),
        resources.stream_);
    if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(                            \
    unrestricted_value, task_offsets, selected_classes, selected_any)   \
    compact_bounded_generated_tasks_kernel<                             \
        unrestricted_value, DirectScreeningPurpose::Fock, true><<<      \
            plan.persistent_quartet_worker_blocks,                      \
            kBoundedDirectThreads, 0, resources.stream_>>>(              \
        device_batch, options.screening_tolerance, shell_pair_bounds,    \
        shell_pair_density_bounds, bounded_direct_shell_pair_order,      \
        bounded_direct_shell_pair_block_bounds,                          \
        bounded_direct_system_density_bounds, active,                    \
        generated_fock_shell_class_mask, 0U,                             \
        host_streaming_fock_shell_class_mask, selected_classes,          \
        selected_any, bounded_direct_cursor,                              \
        bounded_direct_generated_tasks,                                  \
        bounded_direct_generated_task_counts,                            \
        task_offsets,                                                     \
        bounded_direct_generated_overflow)
    if (is_unrestricted) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(
          true, bounded_direct_generated_task_offsets, nullptr, nullptr);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(
          false, bounded_direct_generated_task_offsets, nullptr, nullptr);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    prepare_bounded_generated_retry_kernel<<<
        1, 1, 0,
        resources.stream_>>>(
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity),
        bounded_direct_generated_task_offsets,
        bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads,
        bounded_direct_generated_overflow,
        bounded_direct_generated_retry_mask,
        bounded_direct_generated_retry_task_offsets,
        bounded_direct_generated_retry_any,
        bounded_direct_count_diagnostic);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (bounded_direct_count_diagnostic) return cudaSuccess;
    const auto consume_generated_wave =
        [&](const std::uint32_t* task_offsets) -> cudaError_t {
      for (std::size_t kernel_index = 0;
           kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const unsigned shell_class =
            bounded_fock_kernels[kernel_index].shell_class;
        if ((host_generated_fock_shell_class_mask &
             (std::uint64_t{1} << shell_class)) == 0U) {
          continue;
        }
        cudaError_t launch_error = generated::launch_shell_class_fock(
            shell_class, resources.stream_, is_unrestricted,
            plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks, task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets,
            device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density,
            quartet_fock,
            bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (launch_error != cudaSuccess) return launch_error;
      }
      return cudaSuccess;
    };
    error = consume_generated_wave(bounded_direct_generated_task_offsets);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(
        bounded_direct_generated_task_counts, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
        resources.stream_);
    if (error == cudaSuccess) {
      error = cudaMemsetAsync(
          bounded_direct_generated_overflow, 0,
          detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
          resources.stream_);
    }
    if (error == cudaSuccess) {
      error = cudaMemsetAsync(
          bounded_direct_cursor, 0, sizeof(unsigned long long),
          resources.stream_);
    }
    if (error != cudaSuccess) return error;
    if (is_unrestricted) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(
          true, bounded_direct_generated_retry_task_offsets,
          bounded_direct_generated_retry_mask,
          bounded_direct_generated_retry_any);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK(
          false, bounded_direct_generated_retry_task_offsets,
          bounded_direct_generated_retry_mask,
          bounded_direct_generated_retry_any);
    }
#undef VIBEQC_LAUNCH_BOUNDED_GENERATED_FOCK
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    normalize_bounded_generated_task_counts_kernel<<<
        blocks_for(detail::kDirectQuartetShellClassCount), threads, 0,
        resources.stream_>>>(
        bounded_direct_generated_retry_task_offsets,
        bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads,
        bounded_direct_generated_overflow, false);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    error = consume_generated_wave(
        bounded_direct_generated_retry_task_offsets);
    if (error != cudaSuccess) return error;

    return launch_bounded_streaming_fock(
        is_unrestricted, quartet_density, quartet_fock,
        allow_mixed_precision);
  };
  const auto launch_fock_builder =
      [&](const double* density_input,
          bool allow_mixed_precision) -> cudaError_t {
    const double* quartet_density =
        transformed_direct ? direct_density : density_input;
    double* quartet_fock = transformed_direct ? direct_fock : fock;
    if (quartet_direct) {
      cudaError_t metadata_error =
          launch_direct_quartet_metadata(
              density_input, allow_mixed_precision);
      if (metadata_error != cudaSuccess) return metadata_error;
      // Validation mode intentionally stops after compaction.  Continuing
      // into a consumer would turn a descriptor report into a secondary
      // illegal access and would obscure whether the queue itself is valid.
      if (direct_tile_validation &&
          resources.direct_tile_validation_ != nullptr) {
        return cudaSuccess;
      }
    }
    if (quartet_direct && transformed_direct) {
      clear_active_matrices_kernel<<<
          blocks_for(direct_spin_matrix_elements), threads, 0,
          resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(direct_nbf), active, direct_fock);
    }
    if (quartet_direct) {
      if (plan.shell_quartet_tile_capacities[
              kGenericOrderFiveAngularOrder] != 0) {
        cudaError_t compact_error = cudaMemsetAsync(
            generic_order5_tile_count, 0, sizeof(std::uint32_t),
            resources.stream_);
        if (compact_error != cudaSuccess) return compact_error;
        compact_generic_order5_tiles_kernel<<<
            blocks_for(plan.shell_quartet_tile_capacities[
                kGenericOrderFiveAngularOrder]),
            threads, 0, resources.stream_>>>(
            device_batch,
            active_shell_quartet_tile_counts + kGenericOrderFiveAngularOrder,
            active_shell_quartet_tiles +
                plan.shell_quartet_tile_offsets[
                    kGenericOrderFiveAngularOrder],
            0U, generated_fock_shell_class_mask,
            generic_order5_tile_count, generic_order5_tiles);
        compact_error = cudaPeekAtLastError();
        if (compact_error != cudaSuccess) return compact_error;
      }
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
      if (bounded_direct_streaming) {
        cudaError_t streaming_error = launch_bounded_generated_fock(
            true, quartet_density, quartet_fock, allow_mixed_precision);
        if (streaming_error != cudaSuccess ||
            bounded_direct_count_diagnostic) {
          return streaming_error;
        }
        if (bounded_direct_aot_only_diagnostic) return cudaSuccess;
      } else {
        cudaError_t generated_error = launch_generated_shell_class_focks(
            resources.stream_, plan.total_shell_quartet_tiles,
            plan.generated_shell_task_capacity,
            plan.shell_quartet_tile_capacities,
            active_shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generated_shell_tasks, generated_shell_classes,
            generated_shell_task_offsets, generated_shell_task_counts,
            generated_shell_task_write_counts, generated_shell_task_heads,
            generated_fock_shell_class_mask,
            plan.persistent_quartet_worker_blocks, true,
            options.screening_tolerance, schwarz_bounds, quartet_density,
            quartet_fock);
        if (generated_error != cudaSuccess) return generated_error;
        launch_angular_fock_quartets<true>(
            resources.stream_, plan.shell_quartet_tile_capacities,
            plan.shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generic_order5_tile_count, generic_order5_tiles,
            persistent_fock_task_heads,
            plan.persistent_quartet_worker_blocks,
            options.screening_tolerance, schwarz_bounds, quartet_density,
            active, quartet_fock, generated_fock_shell_class_mask);
        if (allow_mixed_precision && mixed_precision_fock) {
          generated_error = launch_generated_shell_class_mixed_focks(
              resources.stream_, plan.fp32_shell_quartet_tile_capacity,
              plan.generated_shell_task_capacity,
              plan.shell_quartet_tile_capacities,
              fp32_shell_quartet_tile_offsets, device_batch,
              fp32_shell_quartet_tile_counts, fp32_shell_quartet_tiles,
              generated_shell_tasks, generated_shell_classes,
              generated_shell_task_offsets, generated_shell_task_counts,
              generated_shell_task_write_counts, generated_shell_task_heads,
              generated_mixed_fock_shell_class_mask,
              plan.persistent_quartet_worker_blocks, true,
              options.screening_tolerance, schwarz_bounds, quartet_density,
              quartet_fock);
          if (generated_error != cudaSuccess) return generated_error;
          launch_angular_fock_quartets<true, MixedPrecisionFloat>(
              resources.stream_, plan.shell_quartet_tile_capacities,
              plan.fp32_shell_quartet_tile_offsets, device_batch,
              fp32_shell_quartet_tile_counts, fp32_shell_quartet_tiles,
              nullptr, nullptr, fp32_persistent_fock_task_heads,
              plan.persistent_quartet_worker_blocks,
              options.screening_tolerance, schwarz_bounds, quartet_density,
              active, quartet_fock, generated_mixed_fock_shell_class_mask);
        }
      }
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
      if (bounded_direct_streaming) {
        cudaError_t streaming_error = launch_bounded_generated_fock(
            false, quartet_density, quartet_fock, allow_mixed_precision);
        if (streaming_error != cudaSuccess ||
            bounded_direct_count_diagnostic) {
          return streaming_error;
        }
        if (bounded_direct_aot_only_diagnostic) return cudaSuccess;
      } else {
        cudaError_t generated_error = launch_generated_shell_class_focks(
            resources.stream_, plan.total_shell_quartet_tiles,
            plan.generated_shell_task_capacity,
            plan.shell_quartet_tile_capacities,
            active_shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generated_shell_tasks, generated_shell_classes,
            generated_shell_task_offsets, generated_shell_task_counts,
            generated_shell_task_write_counts, generated_shell_task_heads,
            generated_fock_shell_class_mask,
            plan.persistent_quartet_worker_blocks, false,
            options.screening_tolerance, schwarz_bounds, quartet_density,
            quartet_fock);
        if (generated_error != cudaSuccess) return generated_error;
        launch_angular_fock_quartets<false>(
            resources.stream_, plan.shell_quartet_tile_capacities,
            plan.shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generic_order5_tile_count, generic_order5_tiles,
            persistent_fock_task_heads,
            plan.persistent_quartet_worker_blocks,
            options.screening_tolerance, schwarz_bounds, quartet_density,
            active, quartet_fock, generated_fock_shell_class_mask);
        if (allow_mixed_precision && mixed_precision_fock) {
          generated_error = launch_generated_shell_class_mixed_focks(
              resources.stream_, plan.fp32_shell_quartet_tile_capacity,
              plan.generated_shell_task_capacity,
              plan.shell_quartet_tile_capacities,
              fp32_shell_quartet_tile_offsets, device_batch,
              fp32_shell_quartet_tile_counts, fp32_shell_quartet_tiles,
              generated_shell_tasks, generated_shell_classes,
              generated_shell_task_offsets, generated_shell_task_counts,
              generated_shell_task_write_counts, generated_shell_task_heads,
              generated_mixed_fock_shell_class_mask,
              plan.persistent_quartet_worker_blocks, false,
              options.screening_tolerance, schwarz_bounds, quartet_density,
              quartet_fock);
          if (generated_error != cudaSuccess) return generated_error;
          launch_angular_fock_quartets<false, MixedPrecisionFloat>(
              resources.stream_, plan.shell_quartet_tile_capacities,
              plan.fp32_shell_quartet_tile_offsets, device_batch,
              fp32_shell_quartet_tile_counts, fp32_shell_quartet_tiles,
              nullptr, nullptr, fp32_persistent_fock_task_heads,
              plan.persistent_quartet_worker_blocks,
              options.screening_tolerance, schwarz_bounds, quartet_density,
              active, quartet_fock, generated_mixed_fock_shell_class_mask);
        }
      }
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
    return cudaPeekAtLastError();
  };
  initialize_state_kernel<<<blocks_for(batch_size), threads, 0,
                            resources.stream_>>>(
      static_cast<std::int32_t>(batch_size), cached_energy_baseline_hit, energy,
      active, converged, failed, iterations, previous_energy, energy_change,
      density_rms, diis_count, diis_head);
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  if (geometry_changed) {
    build_one_electron_integrals_kernel<<<blocks_for(pair_elements), threads, 0,
                                          resources.stream_>>>(
        device_batch, ao_pair_first, ao_pair_second, pair_count, overlap,
        hcore);
    if (persistent_eri) {
      build_eri_kernel<<<blocks_for(eri_elements), threads, 0,
                         resources.stream_>>>(device_batch, eri);
    } else {
      build_schwarz_bounds_packed_kernel<<<blocks_for(direct_pair_elements),
                                           threads, 0, resources.stream_>>>(
          device_batch, direct_pair_count, schwarz_bounds);
      if (quartet_direct) {
        reduce_shell_pair_bounds_kernel<<<
            static_cast<unsigned>(total_shell_pairs), threads,
            threads * sizeof(double), resources.stream_>>>(
            device_batch, schwarz_bounds, shell_pair_bounds);
        cuda_error = cudaPeekAtLastError();
        if (cuda_error != cudaSuccess) {
          fill_global_failure(outputs, cuda_status(cuda_error));
          return outputs;
        }
        if (bounded_direct_streaming) {
          // Keep every system and class segment Schwarz-descending for the
          // current geometry.  Generic blocks then group similar work, while
          // generated resident-bra streams may safely stop at the first full
          // ket chunk below the geometry-only gate.
          std::vector<double> host_shell_pair_bounds(total_shell_pairs);
          cuda_error = cudaMemcpyAsync(
              host_shell_pair_bounds.data(), shell_pair_bounds,
              total_shell_pairs * sizeof(double), cudaMemcpyDeviceToHost,
              resources.stream_);
          if (cuda_error == cudaSuccess) {
            cuda_error = cudaStreamSynchronize(resources.stream_);
          }
          if (cuda_error != cudaSuccess) {
            fill_global_failure(outputs, cuda_status(cuda_error));
            return outputs;
          }
          for (std::size_t system = 0; system < batch_size; ++system) {
            const std::size_t pair_begin = static_cast<std::size_t>(
                host.system_shell_pair_offsets[system]);
            const std::size_t pair_end = static_cast<std::size_t>(
                host.system_shell_pair_offsets[system + 1]);
            std::stable_sort(
                plan.bounded_direct_shell_pair_order.begin() + pair_begin,
                plan.bounded_direct_shell_pair_order.begin() + pair_end,
                [&](std::uint32_t first, std::uint32_t second) {
                  return host_shell_pair_bounds[first] >
                      host_shell_pair_bounds[second];
                });
          }
          const std::size_t class_stride = batch_size + 1U;
          for (std::size_t pair_class = 0;
               pair_class < detail::kDirectShellPairClassCount;
               ++pair_class) {
            for (std::size_t system = 0; system < batch_size; ++system) {
              const std::size_t segment_begin =
                  plan.bounded_stream_pair_class_offsets[
                      pair_class * class_stride + system];
              const std::size_t segment_end =
                  plan.bounded_stream_pair_class_offsets[
                      pair_class * class_stride + system + 1U];
              std::stable_sort(
                  plan.bounded_stream_shell_pair_order.begin() + segment_begin,
                  plan.bounded_stream_shell_pair_order.begin() + segment_end,
                  [&](std::uint32_t first, std::uint32_t second) {
                    return host_shell_pair_bounds[first] >
                        host_shell_pair_bounds[second];
                  });
            }
          }
          const vibeqc_status order_upload_status = copy_to_device(
              bounded_direct_shell_pair_order,
              plan.bounded_direct_shell_pair_order.data(),
              total_shell_pairs * sizeof(std::uint32_t), resources.stream_);
          if (order_upload_status != VIBEQC_STATUS_SUCCESS) {
            fill_global_failure(outputs, order_upload_status);
            return outputs;
          }
          const vibeqc_status stream_order_upload_status = copy_to_device(
              bounded_stream_shell_pair_order,
              plan.bounded_stream_shell_pair_order.data(),
              total_shell_pairs * sizeof(std::uint32_t), resources.stream_);
          if (stream_order_upload_status != VIBEQC_STATUS_SUCCESS) {
            fill_global_failure(outputs, stream_order_upload_status);
            return outputs;
          }
        }
        if (bounded_direct_streaming) {
          reduce_bounded_shell_pair_block_bounds_kernel<<<
              static_cast<unsigned>(total_shell_pair_blocks), threads,
              threads * sizeof(double), resources.stream_>>>(
              device_batch, bounded_direct_shell_pair_order,
              shell_pair_bounds, bounded_direct_shell_pair_block_bounds);
        }
      }
    }
    build_nuclear_repulsion_kernel<<<blocks_for(batch_size), threads, 0,
                                      resources.stream_>>>(
        device_batch, nuclear_repulsion);

    copy_matrix_kernel<<<blocks_for(matrix_elements), threads, 0,
                         resources.stream_>>>(
        matrix_elements, overlap, eigensystem);
    status = launch_solver(
        resources, ordinary_eigensolver_family, static_cast<int>(nbf),
        static_cast<int>(batch_size),
        eigensystem, temporary, eigenvalues, lwork, solver_info, active);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    inspect_solver_kernel<<<blocks_for(batch_size), threads, 0,
                            resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), solver_info, active, failed,
        converged);
    build_orthogonalizer_kernel<<<blocks_for(matrix_elements), threads, 0,
                                  resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        eigensystem, eigenvalues, active, orthogonalizer, failed);
  }

  // A valid warm density supersedes the core-Hamiltonian guess. Homogeneous
  // warm replay can therefore skip its transforms, eigensolve, and density
  // construction without changing mixed warm/cold bucket semantics.
  if (!all_systems_warm) {
    status = multiply_matrices(hcore, false, orthogonalizer, temporary);
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = multiply_matrices(
          orthogonalizer, true, temporary, eigensystem);
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    status = launch_solver(
        resources, ordinary_eigensolver_family, static_cast<int>(nbf),
        static_cast<int>(batch_size),
        eigensystem, temporary, eigenvalues, lwork, solver_info, active);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
    inspect_solver_kernel<<<blocks_for(batch_size), threads, 0,
                            resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), solver_info, active, failed,
        converged);
    if (unrestricted) {
      status = multiply_matrices(
          orthogonalizer, false, eigensystem, temporary);
      if (status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
      broadcast_spin_matrix_kernel<<<blocks_for(spin_matrix_elements), threads,
                                     0, resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), 2,
          static_cast<std::int32_t>(nbf), temporary, active, coefficients);
      mix_open_shell_guess_kernel<<<blocks_for(batch_size * nbf), threads, 0,
                                    resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(nbf), occupied, active, coefficients);
      build_spin_density_kernel<<<blocks_for(spin_matrix_elements), threads, 0,
                                  resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), 2,
          static_cast<std::int32_t>(nbf), occupied, coefficients, active,
          density);
    } else {
      status = multiply_matrices(
          orthogonalizer, false, eigensystem, coefficients);
      if (status != VIBEQC_STATUS_SUCCESS) {
        fill_global_failure(outputs, status);
        return outputs;
      }
      build_density_kernel<<<blocks_for(matrix_elements), threads, 0,
                             resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(nbf), occupied, coefficients, active,
          density);
    }
  }
  std::vector<std::uint8_t> host_warm_invalid;
  if (!device_resident_density_hit && any_system_warm) {
    // The normalization kernel also performs the CPU-equivalent metric trace
    // check.  Fence only this exceptional input-validation path; a resident
    // replay skips both the host upload and this O(N^2) setup scan.
    cuda_error = cudaMemsetAsync(warm_invalid, 0, batch_size * sizeof(std::uint8_t),
                                 resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (unrestricted) {
      apply_uhf_warm_density_kernel<<<static_cast<unsigned>(batch_size),
                                      kWarmDensityThreads, 0,
                                      resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(nbf), occupied, warm_mask, warm_density,
          overlap, density, warm_invalid);
    } else {
      apply_warm_density_kernel<<<static_cast<unsigned>(batch_size),
                                  kWarmDensityThreads, 0, resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(nbf), occupied, warm_mask, warm_density,
          overlap, density, warm_invalid);
    }
    host_warm_invalid.resize(batch_size, 0);
    cuda_error = cudaMemcpyAsync(
        host_warm_invalid.data(), warm_invalid,
        batch_size * sizeof(std::uint8_t), cudaMemcpyDeviceToHost,
        resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (std::any_of(host_warm_invalid.begin(), host_warm_invalid.end(),
                    [](std::uint8_t value) { return value != 0; })) {
      // The validation kernel has already symmetrized the candidate in the
      // resident density buffer. Residency was invalidated before execution,
      // and a rejected trace must not publish a replacement cache entry. A
      // separately frozen valid dm0/energy pair remains safe to replay.
      fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
      return outputs;
    }
  }

  const EigensolverProfileLaunch graph_eigensolver_profile{
      static_cast<std::int32_t>(batch_size),
      active,
      use_cublas,
      static_cast<std::uint32_t>(options.max_iterations),
      inactive_eigensolver_profile_count,
      inactive_eigensolver_profile,
  };
  const EigensolverProfileLaunch* graph_eigensolver_profile_pointer =
      inactive_eigensolver_profiling ? &graph_eigensolver_profile : nullptr;

  const bool fock_only_iteration =
      bounded_direct_fock_only_diagnostic && bounded_direct_streaming;
  // CUDA 12.9 XsyevBatched remains fast and correct on an ordinary stream for
  // matrices above 512 AOs, but the exact-signature probe shows that provider
  // rejecting Graph capture.  Keep the expensive Fock and matrix work in two
  // reusable Graphs while the host inserts the ordinary provider call between
  // them and checks the tiny physical active mask once per SCF iteration.
  const bool split_provider_iteration =
      !fock_only_iteration &&
      ordinary_eigensolver_family == CudaEigensolverFamily::xsyev_batched &&
      graph_eigensolver_family != CudaEigensolverFamily::xsyev_batched;

  const auto launch_iteration_pre_eigensolver = [&]() -> vibeqc_status {
    const cudaError_t fock_error = launch_fock_builder(density, true);
    if (fock_error != cudaSuccess) return cuda_status(fock_error);
    if (fock_only_iteration) return VIBEQC_STATUS_SUCCESS;

    vibeqc_status iteration_status = VIBEQC_STATUS_SUCCESS;
    if (unrestricted) {
      compute_uhf_energy_kernel<<<static_cast<unsigned>(batch_size),
                                  matrix_reduction_threads, 0,
                                  resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
          density, hcore, fock, nuclear_repulsion, active, energy);
      iteration_status = build_commutator_residual();
    } else {
      compute_energy_kernel<<<static_cast<unsigned>(batch_size),
                              matrix_reduction_threads, 0,
                              resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
          density, hcore, fock, nuclear_repulsion, active, energy);
      iteration_status = build_commutator_residual();
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;

    update_diis_kernel<<<static_cast<unsigned>(batch_size),
                         matrix_reduction_threads, 0, resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), static_cast<std::int32_t>(nbf),
        unrestricted ? 2 : 1,
        static_cast<std::uint32_t>(diis_history), fock, residual, active,
        fock_history, residual_history, diis_linear_system, diis_coefficients,
        diis_count, diis_head, eigensystem);
    if (unrestricted) {
      iteration_status = multiply_spin_matrices(
          eigensystem, true, false, orthogonalizer, false, temporary);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        iteration_status = multiply_spin_matrices(
            orthogonalizer, false, true, temporary, true, eigensystem);
      }
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        expand_spin_active_kernel<<<blocks_for(spin_batch_size), threads, 0,
                                    resources.stream_>>>(
            static_cast<std::int32_t>(batch_size), 2, active, spin_active);
      }
    } else {
      iteration_status = multiply_matrices(
          eigensystem, false, orthogonalizer, temporary);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        iteration_status = multiply_matrices(
            orthogonalizer, true, temporary, eigensystem);
      }
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    return cuda_status(cudaPeekAtLastError());
  };

  const auto launch_iteration_eigensolver = [
      &](CudaEigensolverFamily family) -> vibeqc_status {
    return launch_solver(
        resources, family, static_cast<int>(nbf),
        static_cast<int>(unrestricted ? spin_batch_size : batch_size),
        eigensystem, temporary, eigenvalues, lwork, solver_info,
        unrestricted ? spin_active : active,
        graph_eigensolver_profile_pointer);
  };

  const auto launch_iteration_post_eigensolver = [
      &](bool append_device_tail) -> vibeqc_status {
    vibeqc_status iteration_status = VIBEQC_STATUS_SUCCESS;
    if (unrestricted) {
      inspect_spin_solver_kernel<<<blocks_for(batch_size), threads, 0,
                                   resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), 2, solver_info, active,
          failed, converged);
      iteration_status = multiply_spin_matrices(
          orthogonalizer, false, false, eigensystem, true, coefficients);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        build_spin_density_kernel<<<blocks_for(spin_matrix_elements), threads,
                                    0, resources.stream_>>>(
            static_cast<std::int32_t>(batch_size), 2,
            static_cast<std::int32_t>(nbf), occupied, coefficients, active,
            next_density);
        if (reuse_converged_fock) {
          update_uhf_convergence_kernel<true><<<
              static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_>>>(
              static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance,
              options.density_tolerance, quartet_direct, energy,
              previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        } else {
          update_uhf_convergence_kernel<false><<<
              static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_>>>(
              static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance,
              options.density_tolerance, quartet_direct, energy,
              previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        }
      }
    } else {
      inspect_solver_kernel<<<blocks_for(batch_size), threads, 0,
                              resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), solver_info, active, failed,
          converged);
      iteration_status = multiply_matrices(
          orthogonalizer, false, eigensystem, coefficients);
      if (iteration_status == VIBEQC_STATUS_SUCCESS) {
        build_density_kernel<<<blocks_for(matrix_elements), threads, 0,
                               resources.stream_>>>(
            static_cast<std::int32_t>(batch_size),
            static_cast<std::int32_t>(nbf), occupied, coefficients, active,
            next_density);
        if (reuse_converged_fock) {
          update_convergence_kernel<true><<<
              static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_>>>(
              static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance,
              options.density_tolerance, quartet_direct, energy,
              previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        } else {
          update_convergence_kernel<false><<<
              static_cast<unsigned>(batch_size), matrix_reduction_threads, 0,
              resources.stream_>>>(
              static_cast<std::int32_t>(batch_size),
              static_cast<std::int32_t>(nbf), options.energy_tolerance,
              options.density_tolerance, quartet_direct, energy,
              previous_energy, next_density, density, active, converged,
              iterations, energy_change, density_rms);
        }
      }
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    if (append_device_tail) {
      tail_rhf_loop_kernel<<<1, 1, 0, resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), options.max_iterations,
          active, iterations);
    }
    return cuda_status(cudaPeekAtLastError());
  };

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
    status = launch_iteration_pre_eigensolver();
    if (status == VIBEQC_STATUS_SUCCESS && !fock_only_iteration &&
        !split_provider_iteration) {
      status = launch_iteration_eigensolver(graph_eigensolver_family);
    }
    if (status == VIBEQC_STATUS_SUCCESS && !fock_only_iteration &&
        !split_provider_iteration) {
      status = launch_iteration_post_eigensolver(true);
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      cudaGraph_t abandoned_graph = nullptr;
      (void)cudaStreamEndCapture(resources.stream_, &abandoned_graph);
      if (abandoned_graph != nullptr) (void)cudaGraphDestroy(abandoned_graph);
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs, status);
      return outputs;
    }
    cuda_error = cudaStreamEndCapture(resources.stream_, &resources.iteration_graph_);
    if (status != VIBEQC_STATUS_SUCCESS || cuda_error != cudaSuccess ||
        resources.iteration_graph_ == nullptr) {
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs, status != VIBEQC_STATUS_SUCCESS
                                       ? status
                                       : cuda_status(cuda_error));
      return outputs;
    }
    cuda_error = cudaGraphInstantiate(
        &resources.iteration_graph_exec_, resources.iteration_graph_,
        split_provider_iteration ? 0U : cudaGraphInstantiateFlagDeviceLaunch);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaGraphUpload(resources.iteration_graph_exec_, resources.stream_);
    }
    if (cuda_error == cudaSuccess && split_provider_iteration) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaStreamBeginCapture(resources.stream_,
                                            cudaStreamCaptureModeThreadLocal);
      }
      if (cuda_error == cudaSuccess) {
        status = launch_iteration_post_eigensolver(false);
      }
      if (cuda_error == cudaSuccess && status == VIBEQC_STATUS_SUCCESS) {
        cuda_error = cudaStreamEndCapture(
            resources.stream_, &resources.post_eigensolver_graph_);
      } else {
        cudaGraph_t abandoned_graph = nullptr;
        (void)cudaStreamEndCapture(resources.stream_, &abandoned_graph);
        if (abandoned_graph != nullptr) {
          (void)cudaGraphDestroy(abandoned_graph);
        }
      }
      if (cuda_error == cudaSuccess && status == VIBEQC_STATUS_SUCCESS &&
          resources.post_eigensolver_graph_ != nullptr) {
        cuda_error = cudaGraphInstantiate(
            &resources.post_eigensolver_graph_exec_,
            resources.post_eigensolver_graph_, 0U);
      }
      if (cuda_error == cudaSuccess && status == VIBEQC_STATUS_SUCCESS) {
        cuda_error = cudaGraphUpload(
            resources.post_eigensolver_graph_exec_, resources.stream_);
      }
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (status != VIBEQC_STATUS_SUCCESS || cuda_error != cudaSuccess) {
      if (use_cublas) plan.retry_without_cublas = true;
      fill_global_failure(outputs, status != VIBEQC_STATUS_SUCCESS
                                       ? status
                                       : cuda_status(cuda_error));
      return outputs;
    }
    plan.initialized = true;
  }
  if (inactive_eigensolver_profiling) {
    cuda_error = cudaMemsetAsync(inactive_eigensolver_profile_count, 0,
                                 sizeof(std::uint32_t), resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(
        bounded_fock_class_timer_elapsed, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint64_t),
        resources.stream_);
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing) {
    cuda_error = cudaMemsetAsync(
        bounded_fock_class_timer_launches, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
        resources.stream_);
  }
  if (cuda_error == cudaSuccess && split_provider_iteration) {
    std::vector<std::uint8_t> host_active(batch_size, 1U);
    for (std::uint32_t iteration = 0;
         iteration < options.max_iterations; ++iteration) {
      cuda_error = cudaGraphLaunch(
          resources.iteration_graph_exec_, resources.stream_);
      if (cuda_error != cudaSuccess) break;
      status = launch_iteration_eigensolver(ordinary_eigensolver_family);
      if (status != VIBEQC_STATUS_SUCCESS) break;
      cuda_error = cudaGraphLaunch(
          resources.post_eigensolver_graph_exec_, resources.stream_);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpyAsync(
            host_active.data(), active,
            batch_size * sizeof(std::uint8_t), cudaMemcpyDeviceToHost,
            resources.stream_);
      }
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaStreamSynchronize(resources.stream_);
      }
      if (cuda_error != cudaSuccess ||
          std::none_of(host_active.begin(), host_active.end(),
                       [](std::uint8_t value) { return value != 0; })) {
        break;
      }
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  } else if (cuda_error == cudaSuccess) {
    cuda_error = cudaGraphLaunch(
        resources.iteration_graph_exec_, resources.stream_);
  }
  if (cuda_error == cudaSuccess && direct_tile_validation &&
      resources.direct_tile_validation_ != nullptr) {
    cuda_error = cudaStreamSynchronize(resources.stream_);
    DirectTileValidationRecord host_validation{};
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(
          &host_validation, resources.direct_tile_validation_,
          sizeof(host_validation), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      if (host_validation.error == kDirectTileValidationNoError) {
        std::fprintf(stderr, "direct-tile-validation error=none\n");
        std::fflush(stderr);
      } else {
        const char* error_name = "none";
        switch (static_cast<DirectTileValidationError>(
            host_validation.error)) {
          case DirectTileValidationError::count_exceeds_capacity:
            error_name = "count-exceeds-capacity";
            break;
          case DirectTileValidationError::pair_out_of_bounds:
            error_name = "pair-out-of-bounds";
            break;
          case DirectTileValidationError::shell_out_of_bounds:
            error_name = "shell-out-of-bounds";
            break;
          case DirectTileValidationError::tile_out_of_bounds:
            error_name = "tile-out-of-bounds";
            break;
          case DirectTileValidationError::ao_range_invalid:
            error_name = "ao-range-invalid";
            break;
          default:
            break;
        }
        std::fprintf(
            stderr,
            "direct-tile-validation error=%s order=%u slot=%u tile=%u "
            "pairs=(%u,%u) shells=(%d,%d,%d,%d) direct_nbf=%u "
            "pair_counts=(%u,%u) ao=(%u,%u,%u,%u) count=%u capacity=%u "
            "partition_begin=%u\n",
            error_name, host_validation.angular_order, host_validation.slot,
            host_validation.tile, host_validation.first_pair,
            host_validation.second_pair, host_validation.shell[0],
            host_validation.shell[1], host_validation.shell[2],
            host_validation.shell[3], host_validation.direct_nbf,
            host_validation.first_pair_count, host_validation.second_pair_count,
            host_validation.i, host_validation.j, host_validation.k,
            host_validation.l, host_validation.active_tile_count,
            host_validation.partition_capacity,
            host_validation.partition_begin);
        std::fflush(stderr);
      }
    }
  }
  if (cuda_error == cudaSuccess && bounded_direct_count_diagnostic &&
      bounded_direct_streaming) {
    cuda_error = cudaStreamSynchronize(resources.stream_);
    std::array<std::uint32_t, detail::kDirectQuartetShellClassCount>
        host_counts{};
    std::array<std::uint32_t, detail::kDirectQuartetShellClassCount>
        host_overflow{};
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(
          host_counts.data(), bounded_direct_generated_task_counts,
          host_counts.size() * sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(
          host_overflow.data(), bounded_direct_generated_overflow,
          host_overflow.size() * sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      std::fprintf(
          stderr, "bounded-direct-count purpose=scf-fock capacity=%zu\n",
          plan.bounded_generated_task_capacity);
      for (std::size_t kernel_index = 0;
           kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const generated::ShellKernelMetadata& kernel =
            bounded_fock_kernels[kernel_index];
        const std::uint32_t class_capacity =
            plan.bounded_generated_task_offsets[kernel.shell_class + 1U] -
            plan.bounded_generated_task_offsets[kernel.shell_class];
        std::fprintf(
            stderr, "  %-4s count=%u capacity=%u overflow=%u\n",
            kernel.name, host_counts[kernel.shell_class], class_capacity,
            host_overflow[kernel.shell_class]);
      }
      std::fflush(stderr);
    }
  }
  if (cuda_error == cudaSuccess && bounded_fock_class_timing &&
      bounded_direct_streaming) {
    std::array<std::uint64_t, detail::kDirectQuartetShellClassCount>
        host_elapsed{};
    std::array<std::uint32_t, detail::kDirectQuartetShellClassCount>
        host_launches{};
    cuda_error = cudaMemcpyAsync(
        host_elapsed.data(), bounded_fock_class_timer_elapsed,
        host_elapsed.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
        resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(
          host_launches.data(), bounded_fock_class_timer_launches,
          host_launches.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
          resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      struct HostFockClassTiming {
        const char* name;
        unsigned shell_class;
        std::uint64_t elapsed_nanoseconds;
        std::uint32_t launches;
      };
      std::vector<HostFockClassTiming> timings;
      std::uint64_t total_elapsed_nanoseconds = 0U;
      timings.reserve(bounded_fock_kernel_count);
      for (std::size_t kernel_index = 0;
           kernel_index < bounded_fock_kernel_count; ++kernel_index) {
        const generated::ShellKernelMetadata& kernel =
            bounded_fock_kernels[kernel_index];
        const unsigned shell_class = kernel.shell_class;
        if (shell_class >= host_launches.size() ||
            host_launches[shell_class] == 0U) {
          continue;
        }
        timings.push_back({
            kernel.name, shell_class, host_elapsed[shell_class],
            host_launches[shell_class]});
        total_elapsed_nanoseconds += host_elapsed[shell_class];
      }
      std::sort(
          timings.begin(), timings.end(),
          [](const HostFockClassTiming& first,
             const HostFockClassTiming& second) {
            return first.elapsed_nanoseconds > second.elapsed_nanoseconds;
          });
      std::fprintf(
          stderr,
          "bounded-direct-fock-class-profile total_gpu_ms=%.6f classes=%zu\n",
          static_cast<double>(total_elapsed_nanoseconds) * 1.0e-6,
          timings.size());
      for (const HostFockClassTiming& timing : timings) {
        const double share = total_elapsed_nanoseconds == 0U
            ? 0.0
            : 100.0 * static_cast<double>(timing.elapsed_nanoseconds) /
                static_cast<double>(total_elapsed_nanoseconds);
        std::fprintf(
            stderr,
            "  %-4s class=%u launches=%u gpu_ms=%.6f share=%.2f%%\n",
            timing.name, timing.shell_class, timing.launches,
            static_cast<double>(timing.elapsed_nanoseconds) * 1.0e-6,
            share);
      }
      std::fflush(stderr);
    }
  }
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  if (bounded_direct_fock_only_diagnostic && bounded_direct_streaming) {
    // The isolated profile intentionally captures one Fock construction and
    // omits the eigensolve/convergence tail. Return after device timings are
    // copied so no final-Fock rebuild or analytic-force work contaminates it.
    fill_global_failure(outputs, VIBEQC_STATUS_NOT_CONVERGED);
    return outputs;
  }

  std::uint32_t host_final_fock_rebuild_count =
      static_cast<std::uint32_t>(batch_size);
  if (reuse_converged_fock) {
    // Partition on the device because density RMS is already per-system. This
    // permits a mixed bucket: tight systems retain P_n/F(P_n), while only
    // looser systems restore P_{n+1} and execute the exact legacy rebuild.
    cuda_error = cudaMemsetAsync(final_fock_rebuild_count, 0,
                                 sizeof(std::uint32_t), resources.stream_);
    if (cuda_error == cudaSuccess) {
      select_final_fock_rebuild_kernel<<<
          blocks_for(batch_size), threads, 0, resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          converged_fock_reuse_density_rms(options.density_tolerance),
          density_rms, converged, failed,
          final_fock_reuse_mask, active, final_fock_rebuild_count);
      copy_selected_matrices_kernel<<<
          blocks_for(spin_matrix_elements), threads, 0,
          resources.stream_>>>(
          static_cast<std::int32_t>(batch_size),
          static_cast<std::int32_t>(spin_count),
          static_cast<std::int32_t>(nbf), active, next_density, density);
      cuda_error = cudaMemcpyAsync(
          &host_final_fock_rebuild_count, final_fock_rebuild_count,
          sizeof(std::uint32_t), cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error == cudaSuccess) {
      // One post-Graph scalar fence avoids launching the expensive Fock
      // worker family when every system can reuse its retained matrix.
      cuda_error = cudaStreamSynchronize(resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
    if (host_final_fock_rebuild_count != 0) {
      cuda_error = launch_fock_builder(density, false);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
    select_converged_kernel<<<blocks_for(batch_size), threads, 0,
                              resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), converged, failed, active);
    if (quartet_direct && batch_size > 1 &&
        host_final_fock_rebuild_count != batch_size) {
      // Later device-tail launches overwrite the shared compact quartet list
      // after an early peer converges. Recreate only density transforms,
      // shell-pair bounds, and task metadata for all final snapshots; do not
      // evaluate any two-electron integrals or modify retained Fock matrices.
      cuda_error = launch_direct_quartet_metadata(density, false);
      if (cuda_error != cudaSuccess) {
        fill_global_failure(outputs, cuda_status(cuda_error));
        return outputs;
      }
    }
  } else {
    select_converged_kernel<<<blocks_for(batch_size), threads, 0,
                              resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), converged, failed, active);
    cuda_error = launch_fock_builder(density, false);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }

  // Diagonalize each un-extrapolated final Fock. Tight systems consume their
  // retained P_n/F(P_n); rebuilt systems consume P_{n+1}/F(P_{n+1}). The
  // active mask now contains every converged system for common finalization.
  if (unrestricted) {
    status = multiply_spin_matrices(
        fock, true, false, orthogonalizer, false, temporary);
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = multiply_spin_matrices(
          orthogonalizer, false, true, temporary, true, eigensystem);
    }
    if (status == VIBEQC_STATUS_SUCCESS) {
      expand_spin_active_kernel<<<blocks_for(spin_batch_size), threads, 0,
                                  resources.stream_>>>(
          static_cast<std::int32_t>(batch_size), 2, active, spin_active);
      status = launch_solver(resources, ordinary_eigensolver_family,
                             static_cast<int>(nbf),
                             static_cast<int>(spin_batch_size), eigensystem,
                             temporary, eigenvalues, lwork, solver_info,
                             spin_active);
    }
  } else {
    status = multiply_matrices(fock, false, orthogonalizer, temporary);
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = multiply_matrices(
          orthogonalizer, true, temporary, eigensystem);
    }
    if (status == VIBEQC_STATUS_SUCCESS) {
      status = launch_solver(resources, ordinary_eigensolver_family,
                             static_cast<int>(nbf),
                             static_cast<int>(batch_size), eigensystem,
                             temporary, eigenvalues, lwork, solver_info,
                             active);
    }
  }
  if (status != VIBEQC_STATUS_SUCCESS) {
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
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  } else {
    inspect_solver_kernel<<<blocks_for(batch_size), threads, 0,
                            resources.stream_>>>(
        static_cast<std::int32_t>(batch_size), solver_info, active, failed,
        converged);
    status = multiply_matrices(
        orthogonalizer, false, eigensystem, coefficients);
    if (status != VIBEQC_STATUS_SUCCESS) {
      fill_global_failure(outputs, status);
      return outputs;
    }
  }
  // Keep the converged density paired with the un-extrapolated F(P) that was
  // just diagonalized. The canonical coefficients and eigenvalues are needed
  // for the Pulay weighted density, but replacing P with C_occ C_occ^T would
  // require a second complete J/K rebuild before energy and force evaluation.
  // The accepted density update has already passed the requested SCF density
  // tolerance, so retaining P keeps all final energy/two-electron force terms
  // evaluated consistently at the same P and F(P).
  cuda_error = launch_direct_force_compaction();
  if (cuda_error != cudaSuccess) {
    fill_global_failure(outputs, cuda_status(cuda_error));
    return outputs;
  }
  if (shell_class_profiling && quartet_direct) {
    cuda_error = cudaMemsetAsync(
        shell_class_profile, 0,
        detail::kDirectQuartetShellClassCount *
            sizeof(DeviceShellClassProfileEntry),
        resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (shell_class_profiling && quartet_direct &&
      total_shell_quartet_tiles != 0) {
    profile_active_shell_quartet_tiles_kernel<<<
        blocks_for(total_shell_quartet_tiles), threads, 0,
        resources.stream_>>>(
        device_batch, total_shell_quartet_tiles,
        active_shell_quartet_tile_offsets,
        active_shell_quartet_tile_counts, active_shell_quartet_tiles,
        shell_class_profile);
  }
  if (unrestricted) {
    compute_uhf_energy_kernel<<<static_cast<unsigned>(batch_size),
                                matrix_reduction_threads, 0,
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
    compute_energy_kernel<<<static_cast<unsigned>(batch_size),
                            matrix_reduction_threads, 0,
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
  if (quartet_direct && !bounded_direct_streaming) {
    cuda_error = cudaMemsetAsync(
        persistent_force_task_heads, 0,
        kPersistentForceAngularOrderCount * sizeof(std::uint32_t),
        resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  nuclear_force_kernel<<<blocks_for(force_coordinate_count), threads, 0,
                          resources.stream_>>>(
      device_batch, active, forces);
  if (cooperative_one_electron_force) {
    constexpr std::size_t shared_bytes =
        3 * sizeof(OneElectronDerivativeHermiteCoefficients);
    one_electron_force_cooperative_kernel<<<
        static_cast<unsigned>(one_electron_force_elements), threads,
        shared_bytes, resources.stream_>>>(
        device_batch, ao_pair_first, ao_pair_second, pair_count,
        unrestricted ? total_density : density,
        unrestricted ? total_weighted_density : weighted_density, active,
        forces);
  } else {
    one_electron_force_scalar_kernel<<<
        blocks_for(one_electron_force_elements), threads, 0,
        resources.stream_>>>(
        device_batch, ao_pair_first, ao_pair_second, pair_count,
        unrestricted ? total_density : density,
        unrestricted ? total_weighted_density : weighted_density, active,
        forces);
  }
  const std::uint64_t explicit_generated_force_shell_class_mask =
      generated::enabled_shell_class_mask() & host_present_shell_class_mask;
  const std::uint64_t selected_force_shell_class_mask =
      (explicit_generated_force_shell_class_mask |
       (bounded_direct_streaming
            ? generated::enabled_fock_shell_class_mask() &
                  kStreamingFockShellClassMask
            : 0U)) &
      host_present_shell_class_mask;
  const std::uint64_t native_streaming_force_shell_class_mask =
      bounded_direct_streaming
      ? selected_force_shell_class_mask & kDdddShellClassMask &
            ~explicit_generated_force_shell_class_mask
      : 0U;
  const std::uint64_t generated_shell_class_mask =
      selected_force_shell_class_mask & ~native_streaming_force_shell_class_mask;
  const std::uint64_t generated_queued_force_shell_class_mask =
      generated_shell_class_mask;
  // Whole-task and subgroup-task workers use page-local primitive signatures
  // before advancing independent quartets in warp lockstep. Keep those exact
  // classes out of the unsorted first/retry arenas and route them through the
  // same bounded page stream used by their generated consumers.
  const std::uint64_t bounded_paged_force_shell_class_mask =
      bounded_direct_streaming
      ? generated_queued_force_shell_class_mask &
          kBoundedForceSignatureShellClassMask
      : 0U;
  const std::uint64_t bounded_first_wave_force_shell_class_mask =
      generated_queued_force_shell_class_mask &
      ~bounded_paged_force_shell_class_mask;
  const std::uint64_t covered_force_shell_class_mask =
      generated_shell_class_mask | native_streaming_force_shell_class_mask;
  const std::uint64_t uncovered_force_shell_class_mask =
      host_present_shell_class_mask & ~covered_force_shell_class_mask;
  std::size_t bounded_force_kernel_count = 0;
  const generated::ShellKernelMetadata* bounded_force_kernels =
      generated::selected_fock_shell_kernels(bounded_force_kernel_count);
  const auto launch_bounded_generated_force =
      [&](bool is_unrestricted,
          DirectScreeningPurpose purpose,
          const double* quartet_density) -> cudaError_t {
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_counts, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
        resources.stream_);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(
        bounded_direct_generated_overflow,
        bounded_force_kernel_count == 0 ? 1 : 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
        resources.stream_);
    if (error != cudaSuccess || bounded_force_kernel_count == 0) return error;
    if (bounded_first_wave_force_shell_class_mask == 0U) return cudaSuccess;
    error = cudaMemsetAsync(
        bounded_direct_cursor, 0, sizeof(unsigned long long),
        resources.stream_);
    if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(                           \
    unrestricted_value, purpose_value, task_offsets, selected_classes,   \
    selected_any)                                                        \
    compact_bounded_generated_tasks_kernel<                              \
        unrestricted_value, purpose_value, true><<<                      \
            plan.persistent_quartet_worker_blocks,                       \
            kBoundedDirectThreads, 0, resources.stream_>>>(               \
        device_batch, options.screening_tolerance, shell_pair_bounds,     \
        shell_pair_density_bounds, bounded_direct_shell_pair_order,       \
        bounded_direct_shell_pair_block_bounds,                           \
        bounded_direct_system_density_bounds, active, nullptr,            \
        bounded_first_wave_force_shell_class_mask, 0U,                   \
        selected_classes, selected_any,                                   \
        bounded_direct_cursor,                                            \
        bounded_direct_generated_tasks,                                   \
        bounded_direct_generated_task_counts,                             \
        task_offsets,                                                      \
        bounded_direct_generated_overflow)
    if (is_unrestricted) {
      if (purpose == DirectScreeningPurpose::Force) {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
            true, DirectScreeningPurpose::Force,
            bounded_direct_generated_task_offsets, nullptr, nullptr);
      } else {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
            true, DirectScreeningPurpose::Fock,
            bounded_direct_generated_task_offsets, nullptr, nullptr);
      }
    } else if (purpose == DirectScreeningPurpose::Force) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
          false, DirectScreeningPurpose::Force,
          bounded_direct_generated_task_offsets, nullptr, nullptr);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
          false, DirectScreeningPurpose::Fock,
          bounded_direct_generated_task_offsets, nullptr, nullptr);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    prepare_bounded_generated_retry_kernel<<<
        1, 1, 0,
        resources.stream_>>>(
        static_cast<std::uint32_t>(plan.bounded_generated_task_capacity),
        bounded_direct_generated_task_offsets,
        bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads,
        bounded_direct_generated_overflow,
        bounded_direct_generated_retry_mask,
        bounded_direct_generated_retry_task_offsets,
        bounded_direct_generated_retry_any,
        bounded_direct_count_diagnostic);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    if (bounded_direct_count_diagnostic) {
      std::array<std::uint32_t, detail::kDirectQuartetShellClassCount>
          host_counts{};
      std::array<std::uint32_t, detail::kDirectQuartetShellClassCount>
          host_overflow{};
      error = cudaMemcpyAsync(
          host_counts.data(), bounded_direct_generated_task_counts,
          host_counts.size() * sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, resources.stream_);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            host_overflow.data(), bounded_direct_generated_overflow,
            host_overflow.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost, resources.stream_);
      }
      if (error == cudaSuccess) {
        error = cudaStreamSynchronize(resources.stream_);
      }
      if (error != cudaSuccess) return error;
      std::fprintf(
          stderr, "bounded-direct-count purpose=%s capacity=%zu\n",
          purpose == DirectScreeningPurpose::Force ? "force" : "fock",
          plan.bounded_generated_task_capacity);
      for (std::size_t kernel_index = 0;
           kernel_index < bounded_force_kernel_count; ++kernel_index) {
        const generated::ShellKernelMetadata& kernel =
            bounded_force_kernels[kernel_index];
        if ((bounded_first_wave_force_shell_class_mask &
             (std::uint64_t{1} << kernel.shell_class)) == 0U) {
          continue;
        }
        const std::uint32_t class_capacity =
            plan.bounded_generated_task_offsets[kernel.shell_class + 1U] -
            plan.bounded_generated_task_offsets[kernel.shell_class];
        std::fprintf(
            stderr, "  %-4s count=%u capacity=%u overflow=%u\n",
            kernel.name, host_counts[kernel.shell_class], class_capacity,
            host_overflow[kernel.shell_class]);
      }
      std::fflush(stderr);
      return cudaSuccess;
    }
    const auto consume_generated_wave =
        [&](const std::uint32_t* task_offsets) -> cudaError_t {
      for (std::size_t kernel_index = 0;
           kernel_index < bounded_force_kernel_count; ++kernel_index) {
        const unsigned shell_class =
            bounded_force_kernels[kernel_index].shell_class;
        if ((bounded_first_wave_force_shell_class_mask &
             (std::uint64_t{1} << shell_class)) == 0U) {
          continue;
        }
        if (shell_class_profiling) {
          profile_bounded_generated_tasks_kernel<<<
              plan.persistent_quartet_worker_blocks, threads, 0,
              resources.stream_>>>(
              device_batch, bounded_direct_generated_tasks,
              task_offsets + shell_class,
              bounded_direct_generated_task_counts + shell_class,
              shell_class_profile);
          cudaError_t profile_error = cudaPeekAtLastError();
          if (profile_error != cudaSuccess) return profile_error;
        }
        cudaError_t launch_error = generated::launch_shell_class(
            shell_class, resources.stream_, is_unrestricted,
            plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks, task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets,
            device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density,
            forces, bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (launch_error != cudaSuccess) return launch_error;
      }
      return cudaSuccess;
    };
    error = consume_generated_wave(bounded_direct_generated_task_offsets);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(
        bounded_direct_generated_task_counts, 0,
        detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
        resources.stream_);
    if (error == cudaSuccess) {
      error = cudaMemsetAsync(
          bounded_direct_generated_overflow, 0,
          detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
          resources.stream_);
    }
    if (error == cudaSuccess) {
      error = cudaMemsetAsync(
          bounded_direct_cursor, 0, sizeof(unsigned long long),
          resources.stream_);
    }
    if (error != cudaSuccess) return error;
    if (is_unrestricted) {
      if (purpose == DirectScreeningPurpose::Force) {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
            true, DirectScreeningPurpose::Force,
            bounded_direct_generated_retry_task_offsets,
            bounded_direct_generated_retry_mask,
            bounded_direct_generated_retry_any);
      } else {
        VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
            true, DirectScreeningPurpose::Fock,
            bounded_direct_generated_retry_task_offsets,
            bounded_direct_generated_retry_mask,
            bounded_direct_generated_retry_any);
      }
    } else if (purpose == DirectScreeningPurpose::Force) {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
          false, DirectScreeningPurpose::Force,
          bounded_direct_generated_retry_task_offsets,
          bounded_direct_generated_retry_mask,
          bounded_direct_generated_retry_any);
    } else {
      VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE(
          false, DirectScreeningPurpose::Fock,
          bounded_direct_generated_retry_task_offsets,
          bounded_direct_generated_retry_mask,
          bounded_direct_generated_retry_any);
    }
#undef VIBEQC_LAUNCH_BOUNDED_GENERATED_FORCE
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return error;
    normalize_bounded_generated_task_counts_kernel<<<
        blocks_for(detail::kDirectQuartetShellClassCount), threads, 0,
        resources.stream_>>>(
        bounded_direct_generated_retry_task_offsets,
        bounded_direct_generated_task_counts,
        bounded_direct_generated_task_heads,
        bounded_direct_generated_overflow, false);
    error = cudaPeekAtLastError();
    return error == cudaSuccess
        ? consume_generated_wave(bounded_direct_generated_retry_task_offsets)
        : error;
  };
  const auto launch_bounded_overflow_force =
      [&](bool is_unrestricted,
          DirectScreeningPurpose purpose,
          const double* quartet_density) -> cudaError_t {
    for (std::size_t kernel_index = 0;
         kernel_index < bounded_force_kernel_count; ++kernel_index) {
      const unsigned shell_class =
          bounded_force_kernels[kernel_index].shell_class;
      if ((generated_queued_force_shell_class_mask &
           (std::uint64_t{1} << shell_class)) == 0U) {
        continue;
      }
      unsigned high_pair_class = 0U;
      while ((high_pair_class + 1U) * (high_pair_class + 2U) / 2U <=
             shell_class) {
        ++high_pair_class;
      }
      const unsigned low_pair_class = shell_class -
          high_pair_class * (high_pair_class + 1U) / 2U;
      const std::uint32_t page_capacity = static_cast<std::uint32_t>(
          plan.bounded_generated_task_capacity);
      if (page_capacity == 0U) return cudaErrorInvalidValue;
      const bool signature_paged =
          (bounded_paged_force_shell_class_mask &
           (std::uint64_t{1} << shell_class)) != 0U;
      const std::uint64_t page_domain =
          plan.bounded_generated_task_upper_bounds[shell_class] *
          (high_pair_class == low_pair_class ? 2U : 1U);
      for (std::uint64_t page_begin = 0U;
           page_begin < page_domain; page_begin += page_capacity) {
        cudaError_t error = cudaMemsetAsync(
            bounded_direct_generated_task_counts + shell_class, 0,
            sizeof(std::uint32_t), resources.stream_);
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(
              bounded_direct_generated_task_heads + shell_class, 0,
              sizeof(std::uint32_t), resources.stream_);
        }
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(
              bounded_direct_generated_retry_task_offsets + shell_class, 0,
              sizeof(std::uint32_t), resources.stream_);
        }
        if (error == cudaSuccess && signature_paged) {
          error = cudaMemsetAsync(
              bounded_force_signature_counts, 0,
              kBoundedForceSignatureBucketCount * sizeof(std::uint32_t),
              resources.stream_);
        }
        if (error != cudaSuccess) return error;
        const auto compact_page = [&](
            std::uint32_t* signature_counts,
            const std::uint32_t* signature_offsets,
            bool force_execution) -> cudaError_t {
#define VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(                              \
    unrestricted_value, purpose_value)                                   \
    compact_bounded_exact_class_force_wave_kernel<                        \
        unrestricted_value, purpose_value><<<                             \
            plan.persistent_quartet_worker_blocks,                        \
            kBoundedDirectThreads, 0, resources.stream_>>>(               \
        device_batch, bounded_stream_topology, shell_class,               \
        high_pair_class, low_pair_class, options.screening_tolerance,     \
        page_begin, page_capacity, bounded_direct_generated_tasks,        \
        bounded_direct_generated_task_counts + shell_class,               \
        bounded_direct_generated_task_heads + shell_class,                \
        bounded_direct_generated_overflow, force_execution,               \
        signature_counts, signature_offsets)
          if (is_unrestricted) {
            if (purpose == DirectScreeningPurpose::Force) {
              VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(
                  true, DirectScreeningPurpose::Force);
            } else {
              VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(
                  true, DirectScreeningPurpose::Fock);
            }
          } else if (purpose == DirectScreeningPurpose::Force) {
            VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(
                false, DirectScreeningPurpose::Force);
          } else {
            VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE(
                false, DirectScreeningPurpose::Fock);
          }
#undef VIBEQC_COMPACT_EXACT_OVERFLOW_FORCE
          return cudaPeekAtLastError();
        };
        if (signature_paged) {
          // Count and scatter the same exact candidate page without host
          // readback. The second scan trades a small compaction cost for
          // primitive-uniform batches across all lockstep force workers.
          error = compact_page(
              bounded_force_signature_counts, nullptr, true);
          if (error == cudaSuccess) {
            scan_bounded_force_signature_counts_kernel<<<
                kBoundedForceSignatureScanBlockCount,
                kBoundedForceSignatureScanThreads, 0,
                resources.stream_>>>(
                bounded_force_signature_counts,
                bounded_force_signature_offsets,
                bounded_force_signature_block_offsets);
            error = cudaPeekAtLastError();
          }
          if (error == cudaSuccess) {
            prefix_bounded_force_signature_blocks_kernel<<<
                1, kBoundedForceSignatureScanThreads, 0,
                resources.stream_>>>(
                bounded_force_signature_offsets,
                bounded_force_signature_block_offsets,
                bounded_direct_generated_task_counts + shell_class);
            error = cudaPeekAtLastError();
          }
          if (error == cudaSuccess) {
            error = cudaMemsetAsync(
                bounded_direct_generated_task_heads + shell_class, 0,
                sizeof(std::uint32_t), resources.stream_);
          }
          if (error == cudaSuccess) {
            error = compact_page(
                bounded_force_signature_counts,
                bounded_force_signature_offsets, true);
          }
        } else {
          error = compact_page(nullptr, nullptr, false);
        }
        if (error == cudaSuccess) {
          error = cudaMemsetAsync(
              bounded_direct_generated_task_heads + shell_class, 0,
              sizeof(std::uint32_t), resources.stream_);
        }
        if (error != cudaSuccess) return error;
        if (shell_class_profiling) {
          profile_bounded_generated_tasks_kernel<<<
              plan.persistent_quartet_worker_blocks, threads, 0,
              resources.stream_>>>(
              device_batch, bounded_direct_generated_tasks,
              bounded_direct_generated_retry_task_offsets + shell_class,
              bounded_direct_generated_task_counts + shell_class,
              shell_class_profile);
          error = cudaPeekAtLastError();
          if (error != cudaSuccess) return error;
        }
        error = generated::launch_shell_class(
            shell_class, resources.stream_, is_unrestricted,
            plan.persistent_quartet_worker_blocks,
            bounded_direct_generated_tasks,
            bounded_direct_generated_retry_task_offsets + shell_class,
            device_batch.shell_pair_primitive_offsets,
            device_batch.shell_primitive_pairs,
            device_batch.direct_ao_coefficients, device_batch.positions,
            options.screening_tolerance, schwarz_bounds, quartet_density,
            forces, bounded_direct_generated_task_counts + shell_class,
            bounded_direct_generated_task_heads + shell_class);
        if (error != cudaSuccess) return error;
      }
    }
    return cudaSuccess;
  };
  const auto launch_bounded_native_force =
      [&](bool is_unrestricted,
          DirectScreeningPurpose purpose,
          const double* quartet_density) -> cudaError_t {
    if ((native_streaming_force_shell_class_mask &
         kDdddShellClassMask) == 0U) {
      return cudaSuccess;
    }
    cudaError_t error = cudaMemsetAsync(
        bounded_direct_generated_task_heads + kDdddShellClass, 0,
        sizeof(std::uint32_t), resources.stream_);
    if (error != cudaSuccess) return error;
#define VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(unrestricted_value, purpose_value) \
    bounded_direct_dddd_streaming_kernel<                              \
        unrestricted_value, purpose_value, true><<<                    \
            plan.persistent_quartet_worker_blocks,                     \
            detail::kDirectQuartetThreads, 0, resources.stream_>>>(    \
        device_batch, bounded_stream_topology,                         \
        options.screening_tolerance, schwarz_bounds, quartet_density,  \
        active, forces,                                                \
        bounded_direct_generated_task_heads + kDdddShellClass,        \
        shell_class_profiling ? shell_class_profile : nullptr)
    if (is_unrestricted) {
      if (purpose == DirectScreeningPurpose::Force) {
        VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(
            true, DirectScreeningPurpose::Force);
      } else {
        VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(
            true, DirectScreeningPurpose::Fock);
      }
    } else if (purpose == DirectScreeningPurpose::Force) {
      VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(
          false, DirectScreeningPurpose::Force);
    } else {
      VIBEQC_LAUNCH_NATIVE_DDDD_FORCE(
          false, DirectScreeningPurpose::Fock);
    }
#undef VIBEQC_LAUNCH_NATIVE_DDDD_FORCE
    return cudaPeekAtLastError();
  };
  const auto launch_bounded_force =
      [&](bool is_unrestricted,
          DirectScreeningPurpose purpose,
          const double* quartet_density) -> cudaError_t {
    // Bounded production execution is class-specific by construction. Do not
    // silently turn a disabled or unsupported class into a whole-topology
    // generic force scan: that route is both unpredictable and prohibitively
    // slow on the large topologies that require bounded streaming.
    if (uncovered_force_shell_class_mask != 0U) {
      return cudaErrorNotSupported;
    }
    cudaError_t error = launch_bounded_generated_force(
        is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess || bounded_direct_count_diagnostic) return error;
    error = launch_bounded_overflow_force(
        is_unrestricted, purpose, quartet_density);
    if (error != cudaSuccess) return error;
    error = launch_bounded_native_force(
        is_unrestricted, purpose, quartet_density);
    return error;
  };
  if (quartet_direct &&
      plan.shell_quartet_tile_capacities[kGenericOrderFiveAngularOrder] != 0) {
    cuda_error = cudaMemsetAsync(
        generic_order5_tile_count, 0, sizeof(std::uint32_t),
        resources.stream_);
    if (cuda_error == cudaSuccess) {
      compact_generic_order5_tiles_kernel<<<
          blocks_for(plan.shell_quartet_tile_capacities[
              kGenericOrderFiveAngularOrder]),
          threads, 0, resources.stream_>>>(
          device_batch,
          active_shell_quartet_tile_counts + kGenericOrderFiveAngularOrder,
          active_shell_quartet_tiles +
              plan.shell_quartet_tile_offsets[kGenericOrderFiveAngularOrder],
          generated_shell_class_mask, nullptr, generic_order5_tile_count,
          generic_order5_tiles);
      cuda_error = cudaPeekAtLastError();
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (bounded_direct_fock_only_diagnostic && bounded_direct_streaming) {
    // Nuclear and one-electron forces above remain in the timing so this
    // diagnostic isolates only the bounded two-electron force tail.
  } else if (unrestricted && persistent_eri) {
    two_electron_uhf_force_kernel<<<blocks_for(persistent_force_elements),
                                    threads, 0, resources.stream_>>>(
        device_batch, density, active, forces);
  } else if (unrestricted && quartet_direct) {
    if (bounded_direct_streaming) {
      const DirectScreeningPurpose bounded_force_purpose =
          force_density_product_screening
          ? DirectScreeningPurpose::Force
          : DirectScreeningPurpose::Fock;
      cuda_error = launch_bounded_force(
          true, bounded_force_purpose,
          transformed_direct ? direct_density : density);
    } else {
      cuda_error = launch_generated_shell_class_forces(
          resources.stream_, plan.total_shell_quartet_tiles,
          plan.generated_shell_task_capacity,
          plan.shell_quartet_tile_capacities,
          active_shell_quartet_tile_offsets, device_batch,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          generated_shell_tasks, generated_shell_classes,
          generated_shell_task_offsets, generated_shell_task_counts,
          generated_shell_task_write_counts, generated_shell_task_heads,
          generated_low_order_signature_counts,
          generated_low_order_signature_offsets,
          generated_ppps_resident_tasks, generated_ppps_resident_ket_tasks,
          generated_ppps_resident_bra_counts,
          generated_ppps_resident_bra_offsets,
          generated_ppps_resident_bra_write_counts,
          generated_ppps_resident_signature_counts,
          generated_ppps_resident_signature_offsets,
          generated_ppps_resident_signatures, total_shell_pairs,
          resident_ppps_bra && resident_ppps_ket_task_capacity != 0,
          resident_ppps_signature_bucketing,
          psps_signature_bucketing,
          ppss_signature_bucketing,
          resident_ppps_block_threads,
          plan.persistent_quartet_worker_blocks, true,
          generated_shell_class_mask, options.screening_tolerance,
          schwarz_bounds, transformed_direct ? direct_density : density,
          forces);
      if (cuda_error == cudaSuccess) {
        launch_angular_force_quartets<true>(
            resources.stream_, plan.shell_quartet_tile_capacities,
            plan.shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generic_order5_tile_count, generic_order5_tiles,
            persistent_force_task_heads,
            plan.persistent_quartet_worker_blocks, psss_resident_tasks,
            psss_resident_ket_pairs, plan.resident_psss_task_count,
            plan.resident_psss_bra_primitive_pairs,
            options.screening_tolerance, shell_pair_bounds,
            shell_pair_density_bounds, force_density_product_screening,
            schwarz_bounds, transformed_direct ? direct_density : density,
            active, forces, generated_shell_class_mask);
        cuda_error = cudaPeekAtLastError();
      }
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
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
    if (bounded_direct_streaming) {
      const DirectScreeningPurpose bounded_force_purpose =
          force_density_product_screening
          ? DirectScreeningPurpose::Force
          : DirectScreeningPurpose::Fock;
      cuda_error = launch_bounded_force(
          false, bounded_force_purpose,
          transformed_direct ? direct_density : density);
    } else {
      cuda_error = launch_generated_shell_class_forces(
          resources.stream_, plan.total_shell_quartet_tiles,
          plan.generated_shell_task_capacity,
          plan.shell_quartet_tile_capacities,
          active_shell_quartet_tile_offsets, device_batch,
          active_shell_quartet_tile_counts, active_shell_quartet_tiles,
          generated_shell_tasks, generated_shell_classes,
          generated_shell_task_offsets, generated_shell_task_counts,
          generated_shell_task_write_counts, generated_shell_task_heads,
          generated_low_order_signature_counts,
          generated_low_order_signature_offsets,
          generated_ppps_resident_tasks, generated_ppps_resident_ket_tasks,
          generated_ppps_resident_bra_counts,
          generated_ppps_resident_bra_offsets,
          generated_ppps_resident_bra_write_counts,
          generated_ppps_resident_signature_counts,
          generated_ppps_resident_signature_offsets,
          generated_ppps_resident_signatures, total_shell_pairs,
          resident_ppps_bra && resident_ppps_ket_task_capacity != 0,
          resident_ppps_signature_bucketing,
          psps_signature_bucketing,
          ppss_signature_bucketing,
          resident_ppps_block_threads,
          plan.persistent_quartet_worker_blocks, false,
          generated_shell_class_mask, options.screening_tolerance,
          schwarz_bounds, transformed_direct ? direct_density : density,
          forces);
      if (cuda_error == cudaSuccess) {
        launch_angular_force_quartets<false>(
            resources.stream_, plan.shell_quartet_tile_capacities,
            plan.shell_quartet_tile_offsets, device_batch,
            active_shell_quartet_tile_counts, active_shell_quartet_tiles,
            generic_order5_tile_count, generic_order5_tiles,
            persistent_force_task_heads,
            plan.persistent_quartet_worker_blocks, psss_resident_tasks,
            psss_resident_ket_pairs, plan.resident_psss_task_count,
            plan.resident_psss_bra_primitive_pairs,
            options.screening_tolerance, shell_pair_bounds,
            shell_pair_density_bounds, force_density_product_screening,
            schwarz_bounds, transformed_direct ? direct_density : density,
            active, forces, generated_shell_class_mask);
        cuda_error = cudaPeekAtLastError();
      }
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  } else {
    two_electron_force_direct_kernel<<<
        blocks_for(direct_force_elements), threads, 0, resources.stream_>>>(
        device_batch, options.screening_tolerance, ao_pair_first,
        ao_pair_second, pair_count, schwarz_bounds, density, active,
        forces);
  }

  if (reuse_converged_fock) {
    // Energy and forces above consumed each system's selected consistent
    // snapshot. Advance only reused systems to the already accepted P_{n+1}
    // for their returned warm state; rebuilt systems already contain it.
    copy_selected_matrices_kernel<<<
        blocks_for(spin_matrix_elements), threads, 0, resources.stream_>>>(
        static_cast<std::int32_t>(batch_size),
        static_cast<std::int32_t>(spin_count),
        static_cast<std::int32_t>(nbf), final_fock_reuse_mask, next_density,
        density);
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
  std::uint32_t host_inactive_eigensolver_profile_count = 0U;
  std::vector<DeviceInactiveEigensolverProfileEntry>
      host_inactive_eigensolver_profile(
          inactive_eigensolver_profiling ? options.max_iterations : 0U);
  CudaRhfShellClassProfile host_shell_class_profile{};
  const bool collect_ppps_queue_profile =
      shell_class_profiling && quartet_direct && resident_ppps_bra &&
      resident_ppps_ket_task_capacity != 0U &&
      (generated_shell_class_mask &
       (std::uint64_t{1} << kPppsShellClass)) != 0U;
  std::vector<std::uint32_t> host_ppps_descriptor_counts(
      collect_ppps_queue_profile ? total_shell_pairs : 0U);
  std::vector<std::uint32_t> host_ppps_signatures(
      collect_ppps_queue_profile ? resident_ppps_ket_task_capacity : 0U);
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
  if (inactive_eigensolver_profiling) {
    cuda_error = cudaMemcpyAsync(
        &host_inactive_eigensolver_profile_count,
        inactive_eigensolver_profile_count, sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess &&
        !host_inactive_eigensolver_profile.empty()) {
      cuda_error = cudaMemcpyAsync(
          host_inactive_eigensolver_profile.data(),
          inactive_eigensolver_profile,
          host_inactive_eigensolver_profile.size() *
              sizeof(DeviceInactiveEigensolverProfileEntry),
          cudaMemcpyDeviceToHost, resources.stream_);
    }
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (shell_class_profiling && quartet_direct) {
    cuda_error = cudaMemcpyAsync(
        host_shell_class_profile.data(), shell_class_profile,
        host_shell_class_profile.size() *
            sizeof(CudaRhfShellClassProfileEntry),
        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error != cudaSuccess) {
      fill_global_failure(outputs, cuda_status(cuda_error));
      return outputs;
    }
  }
  if (collect_ppps_queue_profile) {
    cuda_error = cudaMemcpyAsync(
        host_ppps_descriptor_counts.data(),
        generated_ppps_resident_bra_counts,
        host_ppps_descriptor_counts.size() * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, resources.stream_);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(
          host_ppps_signatures.data(), generated_ppps_resident_signatures,
          host_ppps_signatures.size() * sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost, resources.stream_);
    }
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
  if (shell_class_profiling && quartet_direct) {
    plan.last_shell_class_profile = host_shell_class_profile;
  }
  if (inactive_eigensolver_profiling) {
    if (host_inactive_eigensolver_profile_count >
        host_inactive_eigensolver_profile.size()) {
      fill_global_failure(outputs, VIBEQC_STATUS_INTERNAL_ERROR);
      return outputs;
    }
    CudaInactiveEigensolverProfile profile;
    profile.reserve(host_inactive_eigensolver_profile_count);
    for (std::uint32_t index = 0;
         index < host_inactive_eigensolver_profile_count; ++index) {
      const DeviceInactiveEigensolverProfileEntry& input =
          host_inactive_eigensolver_profile[index];
      CudaInactiveEigensolverProfileEntry output;
      output.iteration = input.iteration;
      output.family = static_cast<CudaEigensolverFamily>(input.family);
      output.physical_system_count = input.physical_system_count;
      output.solver_batch_count = input.solver_batch_count;
      output.active_physical_count = input.active_physical_count;
      output.active_solver_count = input.active_solver_count;
      output.solver_elapsed_nanoseconds =
          input.solver_elapsed_nanoseconds;
      output.inactive_input_nonfinite_count =
          input.inactive_input_nonfinite_count;
      output.inactive_submission_nonfinite_count =
          input.inactive_submission_nonfinite_count;
      output.inactive_info_nonzero_count =
          input.inactive_info_nonzero_count;
      output.inactive_touch_flags = input.inactive_touch_flags;
      output.provider_invoked = input.provider_invoked != 0U;
      profile.push_back(output);
    }
    plan.last_inactive_eigensolver_profile = std::move(profile);
  }
  if (collect_ppps_queue_profile) {
    const unsigned multiprocessor_count = std::max(
        1U, plan.persistent_quartet_worker_blocks /
                kPersistentQuartetWarpsPerMultiprocessor);
    CudaPppsQueueProfile ppps_profile = build_ppps_queue_profile(
        host, host_ppps_descriptor_counts, host_ppps_signatures,
        multiprocessor_count);
    if (ppps_profile.descriptor_slots != 0U) {
      plan.last_ppps_queue_profile = std::move(ppps_profile);
    }
  }
  const bool no_system_failed =
      std::none_of(host_failed.begin(), host_failed.end(),
                   [](std::uint8_t value) { return value != 0; });
  if (no_system_failed) {
    plan.cached_positions = host.positions;
  } else if (geometry_changed) {
    // Never reuse an orthogonalizer from a calculation that reported a
    // numerical failure; retry the full geometry path on the next execution.
    plan.cached_positions.clear();
  }
  const bool all_systems_converged = no_system_failed &&
      std::all_of(host_converged.begin(), host_converged.end(),
                  [](std::uint8_t value) { return value != 0; });
  if (all_systems_converged) {
    plan.resident_warm_positions = host.positions;
    plan.resident_warm_density = host_density;
    plan.resident_previous_energy = host_energy;
  } else {
    // A failed or incomplete execution cannot provide an energy baseline for
    // the next warm density, even if the fleet retains another system's state.
    plan.resident_warm_positions.clear();
    plan.resident_warm_density.clear();
    plan.resident_previous_energy.clear();
  }

  for (std::size_t system = 0; system < batch_size; ++system) {
    RhfBucketItem& output = outputs[system];
    ScfResult& result = output.scf;
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
        ? VIBEQC_STATUS_NUMERICAL_FAILURE
        : (result.converged ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_SCF_NOT_CONVERGED);
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
      detail::direct_topology_requires_bounded_streaming(
          static_cast<std::size_t>(
              host.system_shell_quartet_offsets.back())),
      detail::direct_topology_requires_bounded_streaming(
          static_cast<std::size_t>(
              host.system_shell_quartet_offsets.back()))
          ? detail::kBoundedDirectQueueCapacity
          : 0,
  };
}

namespace {

std::vector<RhfBucketItem> run_hf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool unrestricted,
    bool shell_class_profiling,
    bool inactive_eigensolver_profiling) {
  if (plan == nullptr) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  HostBatch candidate;
  if (!pack_host_batch(systems, initial_densities, candidate, unrestricted)) {
    std::vector<RhfBucketItem> outputs(systems.size());
    fill_global_failure(outputs, VIBEQC_STATUS_INVALID_ARGUMENT);
    return outputs;
  }
  const std::optional<double> mixed_precision_fock_threshold =
      *plan != nullptr && (*plan)->quartet_direct
      ? configured_mixed_precision_fock_threshold(
            options.screening_tolerance)
      : std::nullopt;
  const bool mixed_precision_fock =
      mixed_precision_fock_threshold.has_value();
  const bool reuse_converged_fock =
      reuse_converged_fock_requested() && !mixed_precision_fock;
  const bool graph_native_eigensolver_override =
      graph_native_eigensolver_override_requested();
  if (*plan != nullptr && (*plan)->initialized &&
      ((*plan)->resources.device_id_ != device_id ||
       !same_topology((*plan)->topology, candidate) ||
       !same_options((*plan)->options, options) ||
       (*plan)->unrestricted != unrestricted ||
       (*plan)->shell_class_profiling != shell_class_profiling ||
       (*plan)->inactive_eigensolver_profiling !=
           inactive_eigensolver_profiling ||
       (*plan)->graph_native_eigensolver_override !=
           graph_native_eigensolver_override ||
       (*plan)->reuse_converged_fock != reuse_converged_fock ||
       (*plan)->mixed_precision_fock != mixed_precision_fock ||
       (*plan)->mixed_precision_fock_threshold !=
           mixed_precision_fock_threshold.value_or(0.0))) {
    delete *plan;
    *plan = nullptr;
  }
  if (*plan == nullptr) {
    *plan = new (std::nothrow) CudaRhfBucketPlan{};
    if (*plan == nullptr) {
      std::vector<RhfBucketItem> outputs(systems.size());
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
  }
  std::vector<RhfBucketItem> outputs = execute_hf_cuda_bucket(
      **plan, candidate, options, device_id, unrestricted,
      shell_class_profiling, inactive_eigensolver_profiling);
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
      fill_global_failure(outputs, VIBEQC_STATUS_OUT_OF_MEMORY);
      return outputs;
    }
    (*plan)->cublas_enabled = false;
    outputs = execute_hf_cuda_bucket(
        **plan, candidate, options, device_id, unrestricted,
        shell_class_profiling, inactive_eigensolver_profiling);
    if (!(*plan)->initialized) {
      delete *plan;
      *plan = nullptr;
    }
  }
  return outputs;
}

/** Generate Cartesian DF tensors without constructing the full four-center ERI. */
vibeqc_status build_cuda_density_fitting_integrals_impl(
    int device_id, const core::System& orbital_system,
    const core::System& auxiliary_system,
    integrals::DensityFittingIntegralData& output, std::string& detail) {
  if (device_id < 0) {
    detail = "CUDA density-fitting integral generation received an invalid device";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (orbital_system.atoms.size() != auxiliary_system.atoms.size()) {
    detail = "orbital and auxiliary systems must share geometry";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t atom = 0; atom < orbital_system.atoms.size(); ++atom) {
    if (orbital_system.atoms[atom].atomic_number !=
            auxiliary_system.atoms[atom].atomic_number ||
        orbital_system.atoms[atom].position !=
            auxiliary_system.atoms[atom].position) {
      detail = "orbital and auxiliary systems must share geometry";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }

  // Keep both bases Cartesian here.  The public spherical transform is a
  // separate, shared reference operation applied by the SCF preparation path.
  core::System combined;
  combined.atoms = orbital_system.atoms;
  combined.shells = orbital_system.shells;
  combined.shells.insert(combined.shells.end(), auxiliary_system.shells.begin(),
                         auxiliary_system.shells.end());
  // A zero-exponent s shell represents the implicit fourth center in a
  // three-/two-center Coulomb integral.  Its center is algebraically absent
  // from the result, but assigning atom zero keeps DeviceBatch well-formed.
  combined.shells.push_back({0, 0, {{0.0, 1.0}}});
  combined.charge = orbital_system.charge;
  combined.multiplicity = 1;
  combined.electron_count = 2;
  combined.basis_representation = VIBEQC_BASIS_CARTESIAN;

  HostBatch host;
  std::vector<const std::vector<double>*> no_warm(1, nullptr);
  if (!pack_host_batch({combined}, no_warm, host, false)) {
    detail = "combined Cartesian DF basis cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t orbital_count =
      molecule::cartesian_ao_count(orbital_system);
  const std::size_t auxiliary_count =
      molecule::cartesian_ao_count(auxiliary_system);
  const std::size_t dummy_index = orbital_count + auxiliary_count;
  if (host.nbf != dummy_index + 1U ||
      orbital_count == 0U || auxiliary_count == 0U) {
    detail = "Cartesian DF basis dimensions are inconsistent";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (auxiliary_count > std::numeric_limits<std::int32_t>::max() ||
      dummy_index > std::numeric_limits<std::int32_t>::max() ||
      host.nbf > std::numeric_limits<std::int32_t>::max()) {
    detail = "Cartesian DF basis exceeds CUDA index limits";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (orbital_count > std::numeric_limits<std::size_t>::max() / orbital_count ||
      orbital_count * orbital_count >
          std::numeric_limits<std::size_t>::max() / auxiliary_count) {
    detail = "CUDA DF tensor dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t metric_elements = auxiliary_count * auxiliary_count;
  const std::size_t three_center_elements =
      orbital_count * orbital_count * auxiliary_count;
  const std::size_t total_elements = metric_elements + three_center_elements;
  if (total_elements < metric_elements ||
      total_elements > std::numeric_limits<unsigned>::max() *
                            static_cast<std::size_t>(128U)) {
    detail = "CUDA DF integral launch dimensions are too large";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA device selection failed while generating DF integrals";
    return cuda_status(cuda_error);
  }
  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA stream creation failed while generating DF integrals";
    return cuda_status(cuda_error);
  }
  std::vector<void*> allocations;
  auto release = [&]() {
    for (void* pointer : allocations) (void)cudaFree(pointer);
    allocations.clear();
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }
  };
  auto upload = [&](const void* source, std::size_t bytes) -> void* {
    if (bytes == 0U) return nullptr;
    void* destination = nullptr;
    if (cudaMalloc(&destination, bytes) != cudaSuccess) return nullptr;
    if (source != nullptr &&
        cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      (void)cudaFree(destination);
      return nullptr;
    }
    allocations.push_back(destination);
    return destination;
  };
  auto upload_vector = [&](const auto& values) -> void* {
    return upload(values.data(), values.size() * sizeof(values[0]));
  };

  DeviceBatch device_batch{};
  device_batch.batch_size = 1;
  device_batch.nbf = static_cast<std::int32_t>(host.nbf);
  device_batch.direct_nbf = static_cast<std::int32_t>(host.direct_nbf);
  device_batch.total_atoms = static_cast<std::int64_t>(
      host.atomic_numbers.size());
  device_batch.total_shells = static_cast<std::int64_t>(host.shell_atoms.size());
  device_batch.atom_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.atom_offsets));
  device_batch.atom_systems = static_cast<const std::int32_t*>(
      upload_vector(host.atom_systems));
  device_batch.atomic_numbers = static_cast<const std::int32_t*>(
      upload_vector(host.atomic_numbers));
  device_batch.positions = static_cast<const double*>(
      upload_vector(host.positions));
  device_batch.shell_atoms = static_cast<const std::int32_t*>(
      upload_vector(host.shell_atoms));
  device_batch.shell_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.shell_angular));
  device_batch.shell_ao_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_ao_offsets));
  device_batch.shell_direct_ao_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_direct_ao_offsets));
  device_batch.shell_primitive_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_primitive_offsets));
  device_batch.ao_shells = static_cast<const std::int32_t*>(
      upload_vector(host.ao_shells));
  device_batch.ao_term_counts = static_cast<const std::uint8_t*>(
      upload_vector(host.ao_term_counts));
  device_batch.ao_term_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.ao_term_angular));
  device_batch.ao_term_coefficients = static_cast<const double*>(
      upload_vector(host.ao_term_coefficients));
  device_batch.direct_ao_shells = static_cast<const std::int32_t*>(
      upload_vector(host.direct_ao_shells));
  device_batch.direct_ao_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.direct_ao_angular));
  device_batch.direct_ao_coefficients = static_cast<const double*>(
      upload_vector(host.direct_ao_coefficients));
  device_batch.primitive_exponents = static_cast<const double*>(
      upload_vector(host.primitive_exponents));
  device_batch.primitive_coefficients = static_cast<const double*>(
      upload_vector(host.primitive_coefficients));
  const std::array<const void*, 18> metadata{
      device_batch.atom_offsets, device_batch.atom_systems,
      device_batch.atomic_numbers, device_batch.positions,
      device_batch.shell_atoms, device_batch.shell_angular,
      device_batch.shell_ao_offsets, device_batch.shell_direct_ao_offsets,
      device_batch.shell_primitive_offsets, device_batch.ao_shells,
      device_batch.ao_term_counts, device_batch.ao_term_angular,
      device_batch.ao_term_coefficients, device_batch.direct_ao_shells,
      device_batch.direct_ao_angular, device_batch.direct_ao_coefficients,
      device_batch.primitive_exponents, device_batch.primitive_coefficients};
  for (const void* pointer : metadata) {
    if (pointer == nullptr) {
      detail = "CUDA allocation failed while staging DF basis metadata";
      release();
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }

  double* device_metric = static_cast<double*>(
      upload(nullptr, metric_elements * sizeof(double)));
  double* device_three_center = static_cast<double*>(
      upload(nullptr, three_center_elements * sizeof(double)));
  if (device_metric == nullptr || device_three_center == nullptr) {
    detail = "CUDA allocation failed for DF integral output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  output = {};
  output.nbf = orbital_count;
  output.naux = auxiliary_count;
  output.ncoord = orbital_system.atoms.size() * 3U;
  output.metric.resize(metric_elements);
  output.three_center.resize(three_center_elements);
  output.metric_derivative.resize(output.ncoord * metric_elements);
  output.three_center_derivative.resize(output.ncoord * three_center_elements);
  constexpr unsigned threads = 128U;
  const unsigned blocks = static_cast<unsigned>(
      (total_elements + threads - 1U) / threads);
  build_cuda_df_integrals_kernel<false><<<blocks, threads, 0, stream>>>(
      device_batch, orbital_count, auxiliary_count, dummy_index,
      metric_elements, three_center_elements, 0, 1, -1,
      device_metric, device_three_center);
  cuda_error = cudaGetLastError();
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(output.metric.data(), device_metric,
                            metric_elements * sizeof(double),
                            cudaMemcpyDeviceToHost);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(output.three_center.data(), device_three_center,
                            three_center_elements * sizeof(double),
                            cudaMemcpyDeviceToHost);
  }
  for (std::size_t coordinate = 0;
       cuda_error == cudaSuccess && coordinate < output.ncoord; ++coordinate) {
    build_cuda_df_integrals_kernel<true><<<blocks, threads, 0, stream>>>(
        device_batch, orbital_count, auxiliary_count, dummy_index,
        metric_elements, three_center_elements, 0, 1,
        static_cast<std::int64_t>(coordinate), device_metric,
        device_three_center);
    cuda_error = cudaGetLastError();
    if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(
          output.metric_derivative.data() + coordinate * metric_elements,
          device_metric, metric_elements * sizeof(double),
          cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(
          output.three_center_derivative.data() +
              coordinate * three_center_elements,
          device_three_center, three_center_elements * sizeof(double),
          cudaMemcpyDeviceToHost);
    }
  }
  if (cuda_error != cudaSuccess) {
    detail = "CUDA kernel failed while generating DF integrals";
    release();
    return cuda_status(cuda_error);
  }
  release();
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status build_cuda_density_fitting_integrals_batch_impl(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems,
    std::vector<integrals::DensityFittingIntegralData>& outputs,
    std::string& detail, std::size_t output_budget_bytes) {
  outputs.clear();
  if (device_id < 0 || orbital_systems.empty() ||
      orbital_systems.size() != auxiliary_systems.size()) {
    detail = "CUDA DF integral batch dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t batch_size = orbital_systems.size();
  const std::size_t orbital_count =
      molecule::cartesian_ao_count(orbital_systems.front());
  const std::size_t auxiliary_count =
      molecule::cartesian_ao_count(auxiliary_systems.front());
  const std::size_t atom_count = orbital_systems.front().atoms.size();
  if (orbital_count == 0U || auxiliary_count == 0U || atom_count == 0U) {
    detail = "CUDA DF integral batch contains an empty basis or geometry";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (batch_size > static_cast<std::size_t>(
                       std::numeric_limits<std::int32_t>::max())) {
    detail = "CUDA DF integral batch exceeds the supported system count";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::vector<core::System> combined;
  combined.reserve(batch_size);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const core::System& orbital = orbital_systems[system];
    const core::System& auxiliary = auxiliary_systems[system];
    if (orbital.atoms.size() != atom_count ||
        auxiliary.atoms.size() != atom_count ||
        molecule::cartesian_ao_count(orbital) != orbital_count ||
        molecule::cartesian_ao_count(auxiliary) != auxiliary_count) {
      detail = "CUDA DF integral batch requires homogeneous AO dimensions";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      if (orbital.atoms[atom].atomic_number !=
              auxiliary.atoms[atom].atomic_number ||
          orbital.atoms[atom].position != auxiliary.atoms[atom].position) {
        detail = "orbital and auxiliary systems must share geometry";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
    }
    core::System item;
    item.atoms = orbital.atoms;
    item.shells = orbital.shells;
    item.shells.insert(item.shells.end(), auxiliary.shells.begin(),
                       auxiliary.shells.end());
    item.shells.push_back({0, 0, {{0.0, 1.0}}});
    item.charge = orbital.charge;
    item.multiplicity = 1;
    item.electron_count = 2;
    item.basis_representation = VIBEQC_BASIS_CARTESIAN;
    combined.push_back(std::move(item));
  }
  HostBatch host;
  std::vector<const std::vector<double>*> no_warm(batch_size, nullptr);
  if (!pack_host_batch(combined, no_warm, host, false) ||
      host.nbf == 0U) {
    detail = "combined Cartesian DF batch cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (orbital_count > std::numeric_limits<std::size_t>::max() - auxiliary_count ||
      orbital_count + auxiliary_count ==
          std::numeric_limits<std::size_t>::max()) {
    detail = "CUDA DF integral batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t expected_nbf = orbital_count + auxiliary_count + 1U;
  if (host.nbf != expected_nbf ||
      expected_nbf >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    detail = "combined Cartesian DF batch cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (auxiliary_count >
          std::numeric_limits<std::size_t>::max() / auxiliary_count ||
      orbital_count >
          std::numeric_limits<std::size_t>::max() / orbital_count) {
    detail = "CUDA DF integral batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t metric_elements = auxiliary_count * auxiliary_count;
  const std::size_t orbital_pair_count = orbital_count * orbital_count;
  if (orbital_pair_count >
      std::numeric_limits<std::size_t>::max() / auxiliary_count) {
    detail = "CUDA DF integral batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t three_center_elements =
      orbital_pair_count * auxiliary_count;
  if (orbital_count >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      auxiliary_count >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      orbital_count > std::numeric_limits<std::size_t>::max() - auxiliary_count) {
    detail = "CUDA DF integral batch exceeds CUDA index limits";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t dummy_index = orbital_count + auxiliary_count;
  if (dummy_index >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      dummy_index == std::numeric_limits<std::size_t>::max()) {
    detail = "CUDA DF integral batch exceeds CUDA index limits";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t per_system = metric_elements + three_center_elements;
  if (per_system < metric_elements ||
      batch_size > std::numeric_limits<std::size_t>::max() / per_system) {
    detail = "CUDA DF integral batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  constexpr unsigned threads = 128U;
  constexpr std::size_t kDefaultOutputChunkBytes = 64U * 1024U * 1024U;
  const std::size_t output_chunk_bytes =
      output_budget_bytes == 0U ? kDefaultOutputChunkBytes
                                : output_budget_bytes;
  const std::size_t chunk_systems = std::min(
      batch_size,
      std::max<std::size_t>(1U, output_chunk_bytes / per_system));
  const std::size_t chunk_elements = chunk_systems * per_system;
  if (chunk_elements > std::numeric_limits<unsigned>::max() *
                           static_cast<std::size_t>(threads) ||
      chunk_elements > std::numeric_limits<std::size_t>::max() /
                           sizeof(double)) {
    detail = "CUDA DF integral batch launch dimensions are too large";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA device selection failed while generating DF batch";
    return cuda_status(cuda_error);
  }
  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA stream creation failed while generating DF batch";
    return cuda_status(cuda_error);
  }
  std::vector<void*> allocations;
  auto release = [&]() {
    for (void* pointer : allocations) (void)cudaFree(pointer);
    allocations.clear();
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }
  };
  auto upload = [&](const void* source, std::size_t bytes) -> void* {
    if (bytes == 0U) return nullptr;
    void* destination = nullptr;
    if (cudaMalloc(&destination, bytes) != cudaSuccess) return nullptr;
    if (source != nullptr &&
        cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      (void)cudaFree(destination);
      return nullptr;
    }
    allocations.push_back(destination);
    return destination;
  };
  auto upload_vector = [&](const auto& values) -> void* {
    return upload(values.data(), values.size() * sizeof(values[0]));
  };

  DeviceBatch device_batch{};
  device_batch.batch_size = static_cast<std::int32_t>(batch_size);
  device_batch.nbf = static_cast<std::int32_t>(host.nbf);
  device_batch.direct_nbf = static_cast<std::int32_t>(host.direct_nbf);
  device_batch.total_atoms = static_cast<std::int64_t>(
      host.atomic_numbers.size());
  device_batch.total_shells = static_cast<std::int64_t>(host.shell_atoms.size());
  device_batch.atom_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.atom_offsets));
  device_batch.atom_systems = static_cast<const std::int32_t*>(
      upload_vector(host.atom_systems));
  device_batch.atomic_numbers = static_cast<const std::int32_t*>(
      upload_vector(host.atomic_numbers));
  device_batch.positions = static_cast<const double*>(
      upload_vector(host.positions));
  device_batch.shell_atoms = static_cast<const std::int32_t*>(
      upload_vector(host.shell_atoms));
  device_batch.shell_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.shell_angular));
  device_batch.shell_ao_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_ao_offsets));
  device_batch.shell_direct_ao_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_direct_ao_offsets));
  device_batch.shell_primitive_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_primitive_offsets));
  device_batch.ao_shells = static_cast<const std::int32_t*>(
      upload_vector(host.ao_shells));
  device_batch.ao_term_counts = static_cast<const std::uint8_t*>(
      upload_vector(host.ao_term_counts));
  device_batch.ao_term_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.ao_term_angular));
  device_batch.ao_term_coefficients = static_cast<const double*>(
      upload_vector(host.ao_term_coefficients));
  device_batch.direct_ao_shells = static_cast<const std::int32_t*>(
      upload_vector(host.direct_ao_shells));
  device_batch.direct_ao_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.direct_ao_angular));
  device_batch.direct_ao_coefficients = static_cast<const double*>(
      upload_vector(host.direct_ao_coefficients));
  device_batch.primitive_exponents = static_cast<const double*>(
      upload_vector(host.primitive_exponents));
  device_batch.primitive_coefficients = static_cast<const double*>(
      upload_vector(host.primitive_coefficients));
  const std::array<const void*, 18> metadata{
      device_batch.atom_offsets, device_batch.atom_systems,
      device_batch.atomic_numbers, device_batch.positions,
      device_batch.shell_atoms, device_batch.shell_angular,
      device_batch.shell_ao_offsets, device_batch.shell_direct_ao_offsets,
      device_batch.shell_primitive_offsets, device_batch.ao_shells,
      device_batch.ao_term_counts, device_batch.ao_term_angular,
      device_batch.ao_term_coefficients, device_batch.direct_ao_shells,
      device_batch.direct_ao_angular, device_batch.direct_ao_coefficients,
      device_batch.primitive_exponents, device_batch.primitive_coefficients};
  for (const void* pointer : metadata) {
    if (pointer == nullptr) {
      detail = "CUDA allocation failed while staging DF batch metadata";
      release();
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  double* device_metric = static_cast<double*>(
      upload(nullptr, chunk_elements * sizeof(double)));
  if (device_metric == nullptr) {
    detail = "CUDA allocation failed for DF batch output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  // Keep the packed three-center region adjacent to the metric region so one
  // allocation serves both tensors and derivative launches.
  // Keep the packed three-center region adjacent to the metric region. The
  // allocation is sized only for one bounded system chunk; host output
  // vectors retain the complete batch without requiring a full device copy.
  double* device_three_center = device_metric + chunk_systems * metric_elements;
  try {
    outputs.resize(batch_size);
    for (std::size_t system = 0; system < batch_size; ++system) {
      outputs[system].nbf = orbital_count;
      outputs[system].naux = auxiliary_count;
      outputs[system].ncoord = atom_count * 3U;
      outputs[system].metric.resize(metric_elements);
      outputs[system].three_center.resize(three_center_elements);
      outputs[system].metric_derivative.resize(
          outputs[system].ncoord * metric_elements);
      outputs[system].three_center_derivative.resize(
          outputs[system].ncoord * three_center_elements);
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for CUDA DF batch output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  std::vector<double> chunk_metric(chunk_systems * metric_elements);
  std::vector<double> chunk_three_center(chunk_systems * three_center_elements);
  auto launch_chunk = [&](std::size_t system_base, std::size_t systems_in_chunk,
                          std::int64_t coordinate) -> cudaError_t {
    const std::size_t launch_elements = systems_in_chunk * per_system;
    const unsigned blocks = static_cast<unsigned>(
        (launch_elements + threads - 1U) / threads);
    if (coordinate < 0) {
      build_cuda_df_integrals_kernel<false><<<blocks, threads, 0, stream>>>(
          device_batch, orbital_count, auxiliary_count, dummy_index,
          metric_elements, three_center_elements, system_base,
          systems_in_chunk, coordinate, device_metric, device_three_center);
    } else {
      build_cuda_df_integrals_kernel<true><<<blocks, threads, 0, stream>>>(
          device_batch, orbital_count, auxiliary_count, dummy_index,
          metric_elements, three_center_elements, system_base,
          systems_in_chunk, coordinate, device_metric, device_three_center);
    }
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error == cudaSuccess) {
      launch_error = cudaStreamSynchronize(stream);
    }
    return launch_error;
  };

  for (std::size_t system_base = 0;
       cuda_error == cudaSuccess && system_base < batch_size;
       system_base += chunk_systems) {
    const std::size_t systems_in_chunk =
        std::min(chunk_systems, batch_size - system_base);
    cuda_error = launch_chunk(system_base, systems_in_chunk, -1);
    if (cuda_error != cudaSuccess) break;
    cuda_error = cudaMemcpy(
        chunk_metric.data(), device_metric,
        systems_in_chunk * metric_elements * sizeof(double),
        cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) break;
    cuda_error = cudaMemcpy(
        chunk_three_center.data(), device_three_center,
        systems_in_chunk * three_center_elements * sizeof(double),
        cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) break;
    for (std::size_t local = 0; local < systems_in_chunk; ++local) {
      const std::size_t system = system_base + local;
      std::copy(chunk_metric.begin() + local * metric_elements,
                chunk_metric.begin() + (local + 1U) * metric_elements,
                outputs[system].metric.begin());
      std::copy(chunk_three_center.begin() + local * three_center_elements,
                chunk_three_center.begin() +
                    (local + 1U) * three_center_elements,
                outputs[system].three_center.begin());
    }
  }

  for (std::size_t coordinate = 0;
       cuda_error == cudaSuccess && coordinate < atom_count * 3U;
       ++coordinate) {
    for (std::size_t system_base = 0;
         cuda_error == cudaSuccess && system_base < batch_size;
         system_base += chunk_systems) {
      const std::size_t systems_in_chunk =
          std::min(chunk_systems, batch_size - system_base);
      cuda_error = launch_chunk(
          system_base, systems_in_chunk, static_cast<std::int64_t>(coordinate));
      if (cuda_error != cudaSuccess) break;
      cuda_error = cudaMemcpy(
          chunk_metric.data(), device_metric,
          systems_in_chunk * metric_elements * sizeof(double),
          cudaMemcpyDeviceToHost);
      if (cuda_error != cudaSuccess) break;
      cuda_error = cudaMemcpy(
          chunk_three_center.data(), device_three_center,
          systems_in_chunk * three_center_elements * sizeof(double),
          cudaMemcpyDeviceToHost);
      if (cuda_error != cudaSuccess) break;
      for (std::size_t local = 0; local < systems_in_chunk; ++local) {
        const std::size_t system = system_base + local;
        std::copy(
            chunk_metric.begin() + local * metric_elements,
            chunk_metric.begin() + (local + 1U) * metric_elements,
            outputs[system].metric_derivative.begin() +
                coordinate * metric_elements);
        std::copy(
            chunk_three_center.begin() + local * three_center_elements,
            chunk_three_center.begin() +
                (local + 1U) * three_center_elements,
            outputs[system].three_center_derivative.begin() +
                coordinate * three_center_elements);
      }
    }
  }
  release();
  if (cuda_error != cudaSuccess) {
    outputs.clear();
    detail = "CUDA kernel failed while generating DF batch derivatives";
    return cuda_status(cuda_error);
  }
  return VIBEQC_STATUS_SUCCESS;
}

/** Generate Cartesian overlap/Hcore matrices and their coordinate response. */
vibeqc_status build_cuda_one_electron_integrals_impl(
    int device_id, const core::System& system,
    integrals::IntegralData& output, std::string& detail) {
  if (device_id < 0) {
    detail = "CUDA one-electron integral generation received an invalid device";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  core::System cartesian_system = system;
  cartesian_system.basis_representation = VIBEQC_BASIS_CARTESIAN;
  HostBatch host;
  std::vector<const std::vector<double>*> no_warm(1, nullptr);
  if (!pack_host_batch({cartesian_system}, no_warm, host, false) ||
      host.nbf == 0U) {
    detail = "Cartesian one-electron basis cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (host.nbf > std::numeric_limits<std::int32_t>::max() ||
      host.nbf > std::numeric_limits<std::size_t>::max() / host.nbf) {
    detail = "Cartesian one-electron basis dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t matrix_elements = host.nbf * host.nbf;
  const std::size_t pair_count = host.nbf * (host.nbf + 1U) / 2U;
  if (pair_count > std::numeric_limits<unsigned>::max() *
                       static_cast<std::size_t>(128U)) {
    detail = "CUDA one-electron launch dimensions are too large";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::vector<std::int32_t> pair_first;
  std::vector<std::int32_t> pair_second;
  pair_first.reserve(pair_count);
  pair_second.reserve(pair_count);
  for (std::size_t row = 0; row < host.nbf; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      pair_first.push_back(static_cast<std::int32_t>(row));
      pair_second.push_back(static_cast<std::int32_t>(column));
    }
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA device selection failed while generating one-electron integrals";
    return cuda_status(cuda_error);
  }
  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA stream creation failed while generating one-electron integrals";
    return cuda_status(cuda_error);
  }
  std::vector<void*> allocations;
  auto release = [&]() {
    for (void* pointer : allocations) (void)cudaFree(pointer);
    allocations.clear();
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }
  };
  auto upload = [&](const void* source, std::size_t bytes) -> void* {
    if (bytes == 0U) return nullptr;
    void* destination = nullptr;
    if (cudaMalloc(&destination, bytes) != cudaSuccess) return nullptr;
    if (source != nullptr &&
        cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      (void)cudaFree(destination);
      return nullptr;
    }
    allocations.push_back(destination);
    return destination;
  };
  auto upload_vector = [&](const auto& values) -> void* {
    return upload(values.data(), values.size() * sizeof(values[0]));
  };

  DeviceBatch device_batch{};
  device_batch.batch_size = 1;
  device_batch.nbf = static_cast<std::int32_t>(host.nbf);
  device_batch.direct_nbf = static_cast<std::int32_t>(host.direct_nbf);
  device_batch.total_atoms = static_cast<std::int64_t>(
      host.atomic_numbers.size());
  device_batch.total_shells = static_cast<std::int64_t>(host.shell_atoms.size());
  device_batch.atom_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.atom_offsets));
  device_batch.atom_systems = static_cast<const std::int32_t*>(
      upload_vector(host.atom_systems));
  device_batch.atomic_numbers = static_cast<const std::int32_t*>(
      upload_vector(host.atomic_numbers));
  device_batch.positions = static_cast<const double*>(
      upload_vector(host.positions));
  device_batch.shell_atoms = static_cast<const std::int32_t*>(
      upload_vector(host.shell_atoms));
  device_batch.shell_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.shell_angular));
  device_batch.shell_ao_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_ao_offsets));
  device_batch.shell_direct_ao_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_direct_ao_offsets));
  device_batch.shell_primitive_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_primitive_offsets));
  device_batch.ao_shells = static_cast<const std::int32_t*>(
      upload_vector(host.ao_shells));
  device_batch.ao_term_counts = static_cast<const std::uint8_t*>(
      upload_vector(host.ao_term_counts));
  device_batch.ao_term_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.ao_term_angular));
  device_batch.ao_term_coefficients = static_cast<const double*>(
      upload_vector(host.ao_term_coefficients));
  device_batch.direct_ao_shells = static_cast<const std::int32_t*>(
      upload_vector(host.direct_ao_shells));
  device_batch.direct_ao_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.direct_ao_angular));
  device_batch.direct_ao_coefficients = static_cast<const double*>(
      upload_vector(host.direct_ao_coefficients));
  device_batch.primitive_exponents = static_cast<const double*>(
      upload_vector(host.primitive_exponents));
  device_batch.primitive_coefficients = static_cast<const double*>(
      upload_vector(host.primitive_coefficients));
  const std::array<const void*, 18> metadata{
      device_batch.atom_offsets, device_batch.atom_systems,
      device_batch.atomic_numbers, device_batch.positions,
      device_batch.shell_atoms, device_batch.shell_angular,
      device_batch.shell_ao_offsets, device_batch.shell_direct_ao_offsets,
      device_batch.shell_primitive_offsets, device_batch.ao_shells,
      device_batch.ao_term_counts, device_batch.ao_term_angular,
      device_batch.ao_term_coefficients, device_batch.direct_ao_shells,
      device_batch.direct_ao_angular, device_batch.direct_ao_coefficients,
      device_batch.primitive_exponents, device_batch.primitive_coefficients};
  for (const void* pointer : metadata) {
    if (pointer == nullptr) {
      detail = "CUDA allocation failed while staging one-electron metadata";
      release();
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  const auto* device_pair_first = static_cast<const std::int32_t*>(
      upload_vector(pair_first));
  const auto* device_pair_second = static_cast<const std::int32_t*>(
      upload_vector(pair_second));
  double* device_overlap = static_cast<double*>(
      upload(nullptr, matrix_elements * sizeof(double)));
  double* device_hcore = static_cast<double*>(
      upload(nullptr, matrix_elements * sizeof(double)));
  double* device_nuclear = static_cast<double*>(
      upload(nullptr, sizeof(double)));
  if (device_pair_first == nullptr || device_pair_second == nullptr ||
      device_overlap == nullptr || device_hcore == nullptr ||
      device_nuclear == nullptr) {
    detail = "CUDA allocation failed for one-electron integral output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  output = {};
  output.nbf = host.nbf;
  output.ncoord = system.atoms.size() * 3U;
  output.overlap.resize(matrix_elements);
  output.hcore.resize(matrix_elements);
  output.overlap_derivative.resize(output.ncoord * matrix_elements);
  output.hcore_derivative.resize(output.ncoord * matrix_elements);
  output.nuclear_repulsion_derivative.resize(output.ncoord);
  constexpr unsigned threads = 128U;
  const unsigned pair_blocks = static_cast<unsigned>(
      (pair_count + threads - 1U) / threads);
  build_cuda_one_electron_integrals_kernel<false><<<
      pair_blocks, threads, 0, stream>>>(
      device_batch, device_pair_first, device_pair_second, pair_count, -1,
      device_overlap, device_hcore);
  build_cuda_nuclear_repulsion_kernel<false><<<1, 1, 0, stream>>>(
      device_batch, -1, device_nuclear);
  cuda_error = cudaGetLastError();
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(output.overlap.data(), device_overlap,
                            matrix_elements * sizeof(double),
                            cudaMemcpyDeviceToHost);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(output.hcore.data(), device_hcore,
                            matrix_elements * sizeof(double),
                            cudaMemcpyDeviceToHost);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(&output.nuclear_repulsion, device_nuclear,
                            sizeof(double), cudaMemcpyDeviceToHost);
  }
  for (std::size_t coordinate = 0;
       cuda_error == cudaSuccess && coordinate < output.ncoord; ++coordinate) {
    build_cuda_one_electron_integrals_kernel<true><<<
        pair_blocks, threads, 0, stream>>>(
        device_batch, device_pair_first, device_pair_second, pair_count,
        static_cast<std::int64_t>(coordinate), device_overlap, device_hcore);
    build_cuda_nuclear_repulsion_kernel<true><<<1, 1, 0, stream>>>(
        device_batch, static_cast<std::int64_t>(coordinate), device_nuclear);
    cuda_error = cudaGetLastError();
    if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(
          output.overlap_derivative.data() + coordinate * matrix_elements,
          device_overlap, matrix_elements * sizeof(double),
          cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(
          output.hcore_derivative.data() + coordinate * matrix_elements,
          device_hcore, matrix_elements * sizeof(double),
          cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(
          output.nuclear_repulsion_derivative.data() + coordinate,
          device_nuclear, sizeof(double), cudaMemcpyDeviceToHost);
    }
  }
  if (cuda_error != cudaSuccess) {
    detail = "CUDA kernel failed while generating one-electron integrals";
    release();
    return cuda_status(cuda_error);
  }
  release();
  return VIBEQC_STATUS_SUCCESS;
}

/** Generate one-electron tensors for a homogeneous packed system batch. */
vibeqc_status build_cuda_one_electron_integrals_batch_impl(
    int device_id, const std::vector<core::System>& systems,
    std::vector<integrals::IntegralData>& outputs, std::string& detail) {
  outputs.clear();
  if (device_id < 0 || systems.empty()) {
    detail = "CUDA one-electron integral batch dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t batch_size = systems.size();
  if (batch_size >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    detail = "CUDA one-electron batch exceeds the supported system count";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::vector<core::System> cartesian_systems;
  try {
    cartesian_systems.reserve(batch_size);
    for (const core::System& system : systems) {
      core::System cartesian = system;
      cartesian.basis_representation = VIBEQC_BASIS_CARTESIAN;
      cartesian_systems.push_back(std::move(cartesian));
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed while staging one-electron batch";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  HostBatch host;
  std::vector<const std::vector<double>*> no_warm(batch_size, nullptr);
  if (!pack_host_batch(cartesian_systems, no_warm, host, false) ||
      host.nbf == 0U) {
    detail = "Cartesian one-electron batch cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t nbf = host.nbf;
  for (const core::System& system : cartesian_systems) {
    if (molecule::cartesian_ao_count(system) != nbf ||
        system.atoms.size() != systems.front().atoms.size()) {
      detail = "CUDA one-electron batch requires homogeneous AO dimensions";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }

  if (nbf > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      nbf > std::numeric_limits<std::size_t>::max() / nbf) {
    detail = "Cartesian one-electron batch dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t pair_count = nbf * (nbf + 1U) / 2U;
  if (batch_size > std::numeric_limits<std::size_t>::max() / pair_count ||
      batch_size > std::numeric_limits<std::size_t>::max() / matrix_elements) {
    detail = "CUDA one-electron batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t pair_launch_elements = batch_size * pair_count;
  const std::size_t matrix_batch_elements = batch_size * matrix_elements;
  if (pair_launch_elements > std::numeric_limits<unsigned>::max() *
                                  static_cast<std::size_t>(128U) ||
      matrix_batch_elements >
          std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    detail = "CUDA one-electron batch launch dimensions are too large";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  std::vector<std::int32_t> pair_first;
  std::vector<std::int32_t> pair_second;
  try {
    pair_first.reserve(pair_count);
    pair_second.reserve(pair_count);
    for (std::size_t row = 0; row < nbf; ++row) {
      for (std::size_t column = 0; column <= row; ++column) {
        pair_first.push_back(static_cast<std::int32_t>(row));
        pair_second.push_back(static_cast<std::int32_t>(column));
      }
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for one-electron batch pair indices";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA device selection failed while generating one-electron batch";
    return cuda_status(cuda_error);
  }
  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA stream creation failed while generating one-electron batch";
    return cuda_status(cuda_error);
  }
  std::vector<void*> allocations;
  auto release = [&]() {
    for (void* pointer : allocations) (void)cudaFree(pointer);
    allocations.clear();
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }
  };
  auto upload = [&](const void* source, std::size_t bytes) -> void* {
    if (bytes == 0U) return nullptr;
    void* destination = nullptr;
    if (cudaMalloc(&destination, bytes) != cudaSuccess) return nullptr;
    if (source != nullptr &&
        cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      (void)cudaFree(destination);
      return nullptr;
    }
    allocations.push_back(destination);
    return destination;
  };
  auto upload_vector = [&](const auto& values) -> void* {
    return upload(values.data(), values.size() * sizeof(values[0]));
  };

  DeviceBatch device_batch{};
  device_batch.batch_size = static_cast<std::int32_t>(batch_size);
  device_batch.nbf = static_cast<std::int32_t>(host.nbf);
  device_batch.direct_nbf = static_cast<std::int32_t>(host.direct_nbf);
  device_batch.total_atoms = static_cast<std::int64_t>(
      host.atomic_numbers.size());
  device_batch.total_shells = static_cast<std::int64_t>(host.shell_atoms.size());
  device_batch.atom_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.atom_offsets));
  device_batch.atom_systems = static_cast<const std::int32_t*>(
      upload_vector(host.atom_systems));
  device_batch.atomic_numbers = static_cast<const std::int32_t*>(
      upload_vector(host.atomic_numbers));
  device_batch.positions = static_cast<const double*>(
      upload_vector(host.positions));
  device_batch.shell_atoms = static_cast<const std::int32_t*>(
      upload_vector(host.shell_atoms));
  device_batch.shell_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.shell_angular));
  device_batch.shell_ao_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_ao_offsets));
  device_batch.shell_direct_ao_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_direct_ao_offsets));
  device_batch.shell_primitive_offsets = static_cast<const std::int64_t*>(
      upload_vector(host.shell_primitive_offsets));
  device_batch.ao_shells = static_cast<const std::int32_t*>(
      upload_vector(host.ao_shells));
  device_batch.ao_term_counts = static_cast<const std::uint8_t*>(
      upload_vector(host.ao_term_counts));
  device_batch.ao_term_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.ao_term_angular));
  device_batch.ao_term_coefficients = static_cast<const double*>(
      upload_vector(host.ao_term_coefficients));
  device_batch.direct_ao_shells = static_cast<const std::int32_t*>(
      upload_vector(host.direct_ao_shells));
  device_batch.direct_ao_angular = static_cast<const std::uint8_t*>(
      upload_vector(host.direct_ao_angular));
  device_batch.direct_ao_coefficients = static_cast<const double*>(
      upload_vector(host.direct_ao_coefficients));
  device_batch.primitive_exponents = static_cast<const double*>(
      upload_vector(host.primitive_exponents));
  device_batch.primitive_coefficients = static_cast<const double*>(
      upload_vector(host.primitive_coefficients));
  const std::array<const void*, 18> metadata{
      device_batch.atom_offsets, device_batch.atom_systems,
      device_batch.atomic_numbers, device_batch.positions,
      device_batch.shell_atoms, device_batch.shell_angular,
      device_batch.shell_ao_offsets, device_batch.shell_direct_ao_offsets,
      device_batch.shell_primitive_offsets, device_batch.ao_shells,
      device_batch.ao_term_counts, device_batch.ao_term_angular,
      device_batch.ao_term_coefficients, device_batch.direct_ao_shells,
      device_batch.direct_ao_angular, device_batch.direct_ao_coefficients,
      device_batch.primitive_exponents, device_batch.primitive_coefficients};
  for (const void* pointer : metadata) {
    if (pointer == nullptr) {
      detail = "CUDA allocation failed while staging one-electron batch metadata";
      release();
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  const auto* device_pair_first = static_cast<const std::int32_t*>(
      upload_vector(pair_first));
  const auto* device_pair_second = static_cast<const std::int32_t*>(
      upload_vector(pair_second));
  double* device_overlap = static_cast<double*>(
      upload(nullptr, matrix_batch_elements * sizeof(double)));
  double* device_hcore = static_cast<double*>(
      upload(nullptr, matrix_batch_elements * sizeof(double)));
  double* device_nuclear = static_cast<double*>(
      upload(nullptr, batch_size * sizeof(double)));
  if (device_pair_first == nullptr || device_pair_second == nullptr ||
      device_overlap == nullptr || device_hcore == nullptr ||
      device_nuclear == nullptr) {
    detail = "CUDA allocation failed for one-electron batch output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  try {
    outputs.resize(batch_size);
    for (integrals::IntegralData& output : outputs) {
      output.nbf = nbf;
      output.ncoord = systems.front().atoms.size() * 3U;
      output.overlap.resize(matrix_elements);
      output.hcore.resize(matrix_elements);
      output.overlap_derivative.resize(output.ncoord * matrix_elements);
      output.hcore_derivative.resize(output.ncoord * matrix_elements);
      output.nuclear_repulsion_derivative.resize(output.ncoord);
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for one-electron batch output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  std::vector<double> packed_overlap;
  std::vector<double> packed_hcore;
  std::vector<double> packed_nuclear;
  try {
    packed_overlap.resize(matrix_batch_elements);
    packed_hcore.resize(matrix_batch_elements);
    packed_nuclear.resize(batch_size);
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for one-electron batch staging";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  constexpr unsigned threads = 128U;
  const unsigned blocks = static_cast<unsigned>(
      (pair_launch_elements + threads - 1U) / threads);
  build_cuda_one_electron_integrals_kernel<false><<<
      blocks, threads, 0, stream>>>(
      device_batch, device_pair_first, device_pair_second, pair_count, -1,
      device_overlap, device_hcore);
  build_cuda_nuclear_repulsion_kernel<false><<<
      static_cast<unsigned>((batch_size + threads - 1U) / threads), threads,
      0, stream>>>(device_batch, -1, device_nuclear);
  cuda_error = cudaGetLastError();
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(packed_overlap.data(), device_overlap,
                            matrix_batch_elements * sizeof(double),
                            cudaMemcpyDeviceToHost);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(packed_hcore.data(), device_hcore,
                              matrix_batch_elements * sizeof(double),
                              cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      for (std::size_t system = 0; system < batch_size; ++system) {
        std::copy(packed_overlap.begin() + system * matrix_elements,
                  packed_overlap.begin() + (system + 1U) * matrix_elements,
                  outputs[system].overlap.begin());
        std::copy(packed_hcore.begin() + system * matrix_elements,
                  packed_hcore.begin() + (system + 1U) * matrix_elements,
                  outputs[system].hcore.begin());
      }
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(packed_nuclear.data(), device_nuclear,
                              batch_size * sizeof(double),
                              cudaMemcpyDeviceToHost);
      if (cuda_error == cudaSuccess) {
        for (std::size_t system = 0; system < batch_size; ++system) {
          outputs[system].nuclear_repulsion = packed_nuclear[system];
        }
      }
    }
  }
  if (cuda_error == cudaSuccess) {
    for (std::size_t coordinate = 0;
         coordinate < systems.front().atoms.size() * 3U;
         ++coordinate) {
      build_cuda_one_electron_integrals_kernel<true><<<
          blocks, threads, 0, stream>>>(
          device_batch, device_pair_first, device_pair_second, pair_count,
          static_cast<std::int64_t>(coordinate), device_overlap, device_hcore);
      build_cuda_nuclear_repulsion_kernel<true><<<
          static_cast<unsigned>((batch_size + threads - 1U) / threads),
          threads, 0, stream>>>(device_batch,
                                static_cast<std::int64_t>(coordinate),
                                device_nuclear);
      cuda_error = cudaGetLastError();
      if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
      if (cuda_error != cudaSuccess) break;
      cuda_error = cudaMemcpy(packed_overlap.data(), device_overlap,
                              matrix_batch_elements * sizeof(double),
                              cudaMemcpyDeviceToHost);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpy(packed_hcore.data(), device_hcore,
                                matrix_batch_elements * sizeof(double),
                                cudaMemcpyDeviceToHost);
      }
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpy(packed_nuclear.data(), device_nuclear,
                                batch_size * sizeof(double),
                                cudaMemcpyDeviceToHost);
      }
      if (cuda_error != cudaSuccess) break;
      for (std::size_t system = 0; system < batch_size; ++system) {
        std::copy(packed_overlap.begin() + system * matrix_elements,
                  packed_overlap.begin() + (system + 1U) * matrix_elements,
                  outputs[system].overlap_derivative.begin() +
                      coordinate * matrix_elements);
        std::copy(packed_hcore.begin() + system * matrix_elements,
                  packed_hcore.begin() + (system + 1U) * matrix_elements,
                  outputs[system].hcore_derivative.begin() +
                      coordinate * matrix_elements);
        outputs[system].nuclear_repulsion_derivative[coordinate] =
            packed_nuclear[system];
      }
    }
  }
  release();
  if (cuda_error != cudaSuccess) {
    outputs.clear();
    detail = "CUDA kernel failed while generating one-electron batch";
    return cuda_status(cuda_error);
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace

vibeqc_status build_cuda_density_fitting_integrals(
    int device_id, const core::System& orbital_system,
    const core::System& auxiliary_system,
    integrals::DensityFittingIntegralData& output, std::string& detail) {
  return build_cuda_density_fitting_integrals_impl(
      device_id, orbital_system, auxiliary_system, output, detail);
}

vibeqc_status build_cuda_density_fitting_integrals_batch(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems,
    std::vector<integrals::DensityFittingIntegralData>& outputs,
    std::string& detail, std::size_t output_budget_bytes) {
  return build_cuda_density_fitting_integrals_batch_impl(
      device_id, orbital_systems, auxiliary_systems, outputs, detail,
      output_budget_bytes);
}

vibeqc_status build_cuda_one_electron_integrals_batch(
    int device_id, const std::vector<core::System>& systems,
    std::vector<integrals::IntegralData>& outputs, std::string& detail) {
  return build_cuda_one_electron_integrals_batch_impl(
      device_id, systems, outputs, detail);
}

vibeqc_status build_cuda_one_electron_integrals(
    int device_id, const core::System& system,
    integrals::IntegralData& output, std::string& detail) {
  return build_cuda_one_electron_integrals_impl(
      device_id, system, output, detail);
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling,
    bool inactive_eigensolver_profiling) {
  return run_hf_cuda_bucket_cached(
      plan, systems, options, initial_densities, device_id, false,
      shell_class_profiling, inactive_eigensolver_profiling);
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan** plan,
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling,
    bool inactive_eigensolver_profiling) {
  return run_hf_cuda_bucket_cached(
      plan, systems, options, initial_densities, device_id, true,
      shell_class_profiling, inactive_eigensolver_profiling);
}

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan* plan) noexcept {
  delete plan;
}

void set_rhf_cuda_bucket_warm_start_updates(
    CudaRhfBucketPlan* plan, bool enabled) noexcept {
  if (plan == nullptr || plan->warm_start_updates_enabled == enabled) return;
  if (!enabled) {
    // Freeze only on the policy transition. Repeating the setter while fixed
    // must not replace the original post-cold dm0/seed with a later replay's
    // advanced resident state.
    plan->frozen_warm_positions = plan->resident_warm_positions;
    plan->frozen_warm_density = plan->resident_warm_density;
    plan->frozen_previous_energy = plan->resident_previous_energy;
  } else {
    plan->frozen_warm_positions.clear();
    plan->frozen_warm_density.clear();
    plan->frozen_previous_energy.clear();
  }
  plan->warm_start_updates_enabled = enabled;
}

void clear_rhf_cuda_bucket_warm_starts(CudaRhfBucketPlan* plan) noexcept {
  if (plan == nullptr) return;
  plan->resident_warm_positions.clear();
  plan->resident_warm_density.clear();
  plan->resident_previous_energy.clear();
  plan->frozen_warm_positions.clear();
  plan->frozen_warm_density.clear();
  plan->frozen_previous_energy.clear();
}

bool get_rhf_cuda_shell_class_profile(
    const CudaRhfBucketPlan* plan,
    CudaRhfShellClassProfile& profile) noexcept {
  if (plan == nullptr || !plan->last_shell_class_profile.has_value()) {
    return false;
  }
  profile = *plan->last_shell_class_profile;
  return true;
}

bool get_rhf_cuda_ppps_queue_profile(
    const CudaRhfBucketPlan* plan,
    CudaPppsQueueProfile& profile) noexcept {
  if (plan == nullptr || !plan->last_ppps_queue_profile.has_value()) {
    return false;
  }
  profile = *plan->last_ppps_queue_profile;
  return true;
}

bool get_rhf_cuda_eigensolver_diagnostic(
    const CudaRhfBucketPlan* plan,
    CudaEigensolverDiagnostic& diagnostic) noexcept {
  if (plan == nullptr || !plan->initialized) return false;
  diagnostic = plan->eigensolver_diagnostic;
  return true;
}

bool get_rhf_cuda_inactive_eigensolver_profile(
    const CudaRhfBucketPlan* plan,
    CudaInactiveEigensolverProfile& profile) noexcept {
  if (plan == nullptr ||
      !plan->last_inactive_eigensolver_profile.has_value()) {
    return false;
  }
  profile = *plan->last_inactive_eigensolver_profile;
  return true;
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling,
    bool inactive_eigensolver_profiling) {
  CudaRhfBucketPlan* plan = nullptr;
  std::vector<RhfBucketItem> outputs = run_rhf_cuda_bucket_cached(
      &plan, systems, options, initial_densities, device_id,
      shell_class_profiling, inactive_eigensolver_profiling);
  destroy_rhf_cuda_bucket_plan(plan);
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket(
    const std::vector<core::System>& systems,
    const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities,
    int device_id,
    bool shell_class_profiling,
    bool inactive_eigensolver_profiling) {
  CudaRhfBucketPlan* plan = nullptr;
  std::vector<RhfBucketItem> outputs = run_uhf_cuda_bucket_cached(
      &plan, systems, options, initial_densities, device_id,
      shell_class_profiling, inactive_eigensolver_profiling);
  destroy_rhf_cuda_bucket_plan(plan);
  return outputs;
}

ScfResult run_rhf_cuda(const core::System& system,
                       const ScfOptions& options,
                             int device_id,
                             const std::vector<double>* initial_density) {
  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  std::vector<RhfBucketItem> result =
      run_rhf_cuda_bucket(systems, options, initial_densities, device_id);
  if (result.empty()) throw std::runtime_error("CUDA RHF returned no result");
  if (result.front().status == VIBEQC_STATUS_CUDA_ERROR ||
      result.front().status == VIBEQC_STATUS_OUT_OF_MEMORY) {
    throw std::runtime_error("CUDA RHF execution failed");
  }
  return std::move(result.front().scf);
}

ScfResult run_uhf_cuda(const core::System& system,
                       const ScfOptions& options,
                             int device_id,
                             const std::vector<double>* initial_density) {
  const std::vector<core::System> systems{system};
  const std::vector<const std::vector<double>*> initial_densities{initial_density};
  std::vector<RhfBucketItem> result =
      run_uhf_cuda_bucket(systems, options, initial_densities, device_id);
  if (result.empty()) throw std::runtime_error("CUDA UHF returned no result");
  if (result.front().status == VIBEQC_STATUS_CUDA_ERROR ||
      result.front().status == VIBEQC_STATUS_OUT_OF_MEMORY) {
    throw std::runtime_error("CUDA UHF execution failed");
  }
  return std::move(result.front().scf);
}

}  // namespace vibeqc::scf

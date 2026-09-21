#pragma once

#include <cstddef>
#include <cstdint>

#include "scf/cuda/integral_limits.hpp"
#include "scf/cuda/launch_geometry.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda/scf_constants.hpp"
#include "scf/direct_task_layout.hpp"

namespace vibeqc::scf::cuda_execution {

/** Shared fixed launch, representation and screening contracts. Policy parsing remains in
 * rhf_policy.cpp. */
/**
 * IEEE-754 binary32 unit roundoff, mirroring the shared policy constant so the
 * host admission and this device gate resolve the identical cutoff. The
 * assertion below keeps the two definitions from drifting apart.
 */
constexpr double kMixedPrecisionFloat32UnitRoundoff = 5.9604644775390625e-08;
static_assert(kMixedPrecisionFloat32UnitRoundoff ==
              cuda_policy::kMixedPrecisionFloat32UnitRoundoff);
// Workload thresholds still shared with bucket/topology admission. They are
// intentionally left for the next profile-identity slice; this change first
// removes device-resource constants whose legality can be resolved now.
constexpr std::size_t kPersistentEriAoLimit = 16;
constexpr std::size_t kCublasMatrixProductAoThreshold = 17;
// Schwarz diagonal ERIs use the largest device call frame in the direct path.
// One thread per block prevents a full warp of those frames from exhausting
// the SM local-memory stack pool while preserving the dense AO-pair grid.
constexpr unsigned kSchwarzThreads = 1;
// Generated descriptors are a small staging cache, not topology.
// Dominant Fock classes stream directly from O(N_shell^2) shell-pair metadata;
// force classes that outgrow this cache are replayed losslessly through
// exact-class pages and the same generated consumers. Keeping the cache modest
// also leaves room for the large AOT module and CUDA Graph on a 32 GiB device.
constexpr std::size_t kBoundedGeneratedTasksPerShellPair = 1024;
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
static_assert(kPersistentFockAngularOrderCount <= detail::kDirectQuartetAngularOrderCount);
// Orders zero through two retain their specialized FP64 workers.  Keeping the
// mixed queue limited to the remaining partitions avoids duplicating the
// dominant low-order topology capacity solely for records that can never be
// routed to an FP32 recurrence.
constexpr unsigned kMixedFockMinimumAngularOrder =
    detail::kDirectQuartetMixedFockMinimumAngularOrder;
static_assert(kMixedFockMinimumAngularOrder < detail::kDirectQuartetAngularOrderCount);
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
constexpr std::uint64_t kDdddShellClassMask = std::uint64_t{1} << kDdddShellClass;
// The production profile covers the contiguous canonical class range from
// ssss through dddd. Generated resident-bra Fock kernels own classes 0..19;
// dddd Fock uses the native exact recurrence below because the generated
// value consumer is not numerically reliable for production tasks. A
// separately qualified generated force consumer may still own dddd gradients.
// Both routes enumerate pair-class segments directly and therefore avoid a
// whole-topology generic fallback scan.
constexpr std::uint64_t kCanonicalSpdShellClassMask = (std::uint64_t{1} << 21U) - 1U;
constexpr std::uint64_t kStreamingFockShellClassMask = kCanonicalSpdShellClassMask;
constexpr std::uint64_t kGeneratedStreamingFockShellClassMask =
    kStreamingFockShellClassMask & ~kDdddShellClassMask;
constexpr std::uint64_t kNativeStreamingFockShellClassMask = kDdddShellClassMask;
// Fixed-topology ssss/psss already have handwritten Fock consumers, while the
// generated dddd consumer is rejected above. Keep those bits out of the fixed
// mask so the established exact routes remain single-counted and correct.
constexpr std::uint64_t kFixedTopologyGeneratedFockExclusionMask =
    (std::uint64_t{1} << 0U) | (std::uint64_t{1} << 1U) | kDdddShellClassMask;
// The generated resident ppps consumer stages one pp primitive-pair list in
// shared memory.  Larger lists stay on the established ordinary task path.
constexpr unsigned kGeneratedPppsResidentMaximumBraPrimitivePairs = 64;
// Signature bucketing groups both ordered PPPS orientations by the ket
// primitive-pair count. Counts 0..63 are exact and 64 is an overflow bucket;
// bundled production bases currently use no more than 15 ket pairs here.
constexpr unsigned kPppsSignaturePrimitivePairBuckets = 65;
constexpr unsigned kPppsSignatureBucketCount = 2 * kPppsSignaturePrimitivePairBuckets;
// Whole-task and subgroup-task workers advance independent quartets in
// lockstep. Group both primitive-pair loop lengths and both pair orientations
// so each hardware warp executes a uniform recurrence slice. The same compact
// page-local histogram serves all selected scalar classes and PPPS without a
// topology-sized sort or a class-specific scientific fallback.
constexpr unsigned kBoundedForceSignatureOrientationCount = 4;
constexpr unsigned kBoundedForceSignatureBucketCount = kBoundedForceSignatureOrientationCount *
                                                       kPppsSignaturePrimitivePairBuckets *
                                                       kPppsSignaturePrimitivePairBuckets;
constexpr unsigned kBoundedForceSignatureScanThreads = 256;
constexpr unsigned kBoundedForceSignatureScanBlockCount =
    (kBoundedForceSignatureBucketCount + kBoundedForceSignatureScanThreads - 1U) /
    kBoundedForceSignatureScanThreads;
constexpr std::uint64_t kBoundedForceSignatureShellClassMask =
    (std::uint64_t{1} << kSsssShellClass) | (std::uint64_t{1} << kPsssShellClass) |
    (std::uint64_t{1} << kPspsShellClass) | (std::uint64_t{1} << kPpssShellClass) |
    (std::uint64_t{1} << kPppsShellClass) | (std::uint64_t{1} << kPpppShellClass) |
    (std::uint64_t{1} << kDsssShellClass) | (std::uint64_t{1} << kDspsShellClass) |
    (std::uint64_t{1} << kDsppShellClass) | (std::uint64_t{1} << kDpssShellClass) |
    (std::uint64_t{1} << kDppsShellClass) | (std::uint64_t{1} << kDpppShellClass) |
    (std::uint64_t{1} << kDpdsShellClass) | (std::uint64_t{1} << kDpdpShellClass) |
    (std::uint64_t{1} << kDdpsShellClass) | (std::uint64_t{1} << kDdppShellClass) |
    (std::uint64_t{1} << kDddpShellClass) | (std::uint64_t{1} << kDdddShellClass);
// ``ssss`` generated force mathematics and the current ``psss`` low-order path
// both reuse the exact bounded page scheduler. Keep both classes in this native-
// scheduler mask so neither is hidden by the standalone AOT force capability mask.
constexpr std::uint64_t kBoundedNativePagedForceShellClassMask =
    (std::uint64_t{1} << kSsssShellClass) | (std::uint64_t{1} << kPsssShellClass);
// The scalar PSPS and PPSS force workers assign one complete task to each
// lane. Group both canonical pair loop lengths so a warp advances through
// equal primitive work instead of serializing on the longest lane. Counts
// 0..63 are exact and 64 is the overflow bucket, matching the PPPS convention.
constexpr unsigned kLowOrderSignaturePrimitivePairBuckets = 65;
constexpr unsigned kLowOrderSignatureBucketsPerClass =
    kLowOrderSignaturePrimitivePairBuckets * kLowOrderSignaturePrimitivePairBuckets;
constexpr unsigned kLowOrderSignatureClassCount = 2;
constexpr unsigned kLowOrderSignatureElementCount =
    kLowOrderSignatureClassCount * kLowOrderSignatureBucketsPerClass;
static_assert(detail::kDirectQuartetThreads == 32);
// Generated order-five classes are removed from a compact generic fallback
// queue. Keeping the order explicit avoids coupling runtime selection to one
// generated shell class such as dppp.
constexpr unsigned kGenericOrderFiveAngularOrder = 5;
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

}  // namespace vibeqc::scf::cuda_execution

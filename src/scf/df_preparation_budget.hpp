#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vibeqc::scf {

/** Host live-set bound for one source-backed CUDA DF preparation item.
 * The caller selects the actual derivative exporter. Generated one-electron
 * response retains geometry instead of coordinate-major AO derivative arrays.
 * CPU/reference preparation remains outside this bounded CUDA source path.
 */
struct DfPreparationShape {
  std::size_t cartesian_nbf{}, public_nbf{}, atoms{};
  std::size_t orbital_shells{}, orbital_primitives{};
  std::size_t auxiliary_shells{}, auxiliary_primitives{};
  bool include_derivatives{}, generated_one_electron{};
};

struct DfPreparationStorage {
  std::size_t retained_bytes{};
  std::size_t peak_bytes{};
  std::size_t metadata_bytes{};
};

/** Saturate infeasible shapes instead of wrapping them into a small allowance. */
inline std::size_t df_preparation_bytes(long double bytes) noexcept {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  return bytes >= static_cast<long double>(maximum) ? maximum : static_cast<std::size_t>(bytes);
}

inline DfPreparationStorage df_preparation_storage(DfPreparationShape shape) noexcept {
  const long double c = shape.cartesian_nbf, n = shape.public_nbf, atoms = shape.atoms;
  const long double coordinates = 3 * atoms;
  const bool matrices = shape.include_derivatives && !shape.generated_one_electron;
  const long double copies = 1 + (matrices ? coordinates : 0);
  const long double nuclear = shape.include_derivatives ? coordinates : 0;
  const long double cartesian = 2 * c * c * copies + nuclear;
  const long double retained = 2 * n * n * copies + nuclear;
  const long double shells = shape.orbital_shells;
  // LP64 capacity allowance for basis copies, public expansions, packed pair
  // metadata, and the geometry owners bound later to generated force response.
  const long double metadata =
      512 * (1 + atoms + shells + shape.orbital_primitives + c + n + shells * (shells + 1) / 2 +
             shape.auxiliary_shells + shape.auxiliary_primitives);
  // The bridge owns packed S/H/nuclear values, a warm-density matrix and pair
  // indices. Transformation holds Cartesian and public outputs simultaneously
  // (even Cartesian return-by-value copies), plus two derivative row matrices.
  const long double staging = 4 * c * c + 1 + (matrices ? 2 * n * n : 0);
  return {df_preparation_bytes(sizeof(double) * retained + metadata),
          df_preparation_bytes(sizeof(double) * (cartesian + retained + staging) + 2 * metadata),
          df_preparation_bytes(metadata)};
}

/** Workload dimensions used by the DF resource resolver. They are deliberately
 * execution dimensions, not scientific controls, and therefore do not alter
 * integral mathematics or numerical thresholds. */
struct DfBudgetWorkload {
  std::size_t nbf{}, naux{}, atoms{}, batch{1}, diis_history{};
  bool forces{};
};

/** Optional live device envelope. A false live flag is the deterministic
 * CPU/probe-failure path; callers must not synthesize guessed free memory. */
struct DfResourceEnvelope {
  std::size_t free_bytes{}, total_bytes{};
  bool live{};
};

/** Fully resolved owner contract. The resolved bytes are replay/cache identity
 * for the prepared execution owner even though they are not scientific identity.
 * A positive requested_bytes is always a hard upper bound on total_bytes. */
struct DfResolvedBudget {
  static constexpr std::uint32_t policy_version = 3;
  std::size_t requested_bytes{};
  std::size_t total_bytes{};
  std::size_t value_bytes{};
  std::size_t response_bytes{};
  std::size_t reserved_headroom_bytes{};
  std::size_t observed_free_bytes{};
  std::size_t observed_total_bytes{};
  bool live_resource{};
  bool feasible{true};
  bool operator==(const DfResolvedBudget&) const = default;
};

inline std::size_t df_budget_bytes(long double bytes) noexcept {
  constexpr long double maximum = static_cast<long double>(std::numeric_limits<std::size_t>::max());
  return bytes >= maximum ? std::numeric_limits<std::size_t>::max()
                          : static_cast<std::size_t>(std::max<long double>(0, bytes));
}

inline std::size_t df_budget_ceiling(long double bytes) noexcept {
  constexpr long double maximum = static_cast<long double>(std::numeric_limits<std::size_t>::max());
  return bytes >= maximum ? std::numeric_limits<std::size_t>::max()
                          : static_cast<std::size_t>(std::ceil(std::max<long double>(0, bytes)));
}

/**
 * Estimate the value-owner floor for a full source-backed resident plan.
 *
 * The ordinary workload estimate intentionally describes preparation and
 * response staging.  It is not sufficient to admit the persistent CUDA J/K
 * owner: a resident plan keeps one full B tensor and three full-width K
 * panels live, in addition to the batched SCF/final-state reservations.  Keep
 * this estimate in the shared policy header so automatic resolution and the
 * planner agree on the admission boundary without making a low-memory device
 * borrow the response allowance.  The native planner remains authoritative and
 * may still select a streamed plan when basis metadata exceeds this shape-only
 * reserve.
 */
inline std::size_t df_resident_value_admission_floor(DfBudgetWorkload workload) noexcept {
  constexpr long double mib = 1024.0L * 1024.0L;
  const long double n = static_cast<long double>(std::max<std::size_t>(1, workload.nbf));
  const long double a = static_cast<long double>(std::max<std::size_t>(1, workload.naux));
  const long double batch = static_cast<long double>(std::max<std::size_t>(1, workload.batch));
  const long double diis =
      static_cast<long double>(std::min<std::size_t>(workload.diis_history, 12U));
  const long double matrix = n * n;
  const long double tensor = matrix * a;
  const long double matrix_bytes = matrix * sizeof(double);

  // Match the fixed DIIS reservation charged before tile selection. The
  // reserve is zero for the history sizes that do not allocate device DIIS.
  const long double diis_dimension = diis + 1.0L;
  const long double diis_bytes = diis < 2.0L
                                     ? 0.0L
                                     : batch *
                                               ((4.0L * diis + 6.0L) * matrix +
                                                diis_dimension * diis_dimension + diis_dimension) *
                                               sizeof(double) +
                                           batch * 2.0L * sizeof(std::uint32_t);

  const long double eigen_workspace = 1.0L * mib + 16.0L * matrix_bytes;
  const long double eigen_reservation =
      eigen_workspace + (3.0L * matrix + n) * sizeof(double) + sizeof(int) + sizeof(unsigned char);
  const long double snapshot =
      batch * (2.0L * (matrix + n) * sizeof(double) + 2.0L * (sizeof(std::uint64_t) + sizeof(int)));
  const long double final_validation =
      (9.0L * matrix + n) * sizeof(double) + (3.0L * 128.0L + 1.0L) * 128.0L;
  const long double control = batch * 154.0L;
  const long double solver = 1.0L * mib + 16.0L * a * a * sizeof(double) + batch * eigen_workspace;
  const long double metric = batch * a * a * sizeof(double);

  // Generated resident B plus the three full-width K panels, the persistent
  // SCF matrices, setup metric, and a conservative source/metadata margin.
  const long double setup_doubles = batch * (3.0L * a * a + 2.0L * a);
  const long double contraction_doubles =
      7.0L * batch * matrix + batch * a + 3.0L * tensor + batch * tensor;
  const long double one_electron_doubles = 23.0L * batch * matrix;
  const long double source_margin = 64.0L * mib + 16.0L * (matrix + a * a) * sizeof(double);
  return df_budget_bytes(
      diis_bytes + control + eigen_reservation + snapshot + final_validation + solver + metric +
      (setup_doubles + contraction_doubles + one_electron_doubles) * sizeof(double) +
      source_margin);
}

/** Resolve one value/response allowance without a fixed-size magic default.
 *
 * Live automatic mode bounds its workload target by available device memory,
 * not the probe-failure cap: a 1-GiB cap forces roomy multi-GiB tensors to
 * regenerate on every replay. When the live envelope can admit a complete
 * source-backed resident device value owner, raise the target to its admission
 * floor before splitting response capacity. This does not choose the
 * materialized host raw owner; that separate route changed SCF work counts
 * during qualification. Tight live envelopes retain the smaller target and
 * streamed fallback. Explicit positive budgets remain hard caps. If the probe
 * is unavailable, the same dimensions deterministically resolve to a
 * conservative 32 MiB..1 GiB envelope. Force response and value ownership
 * are proportional to their estimated staged work, not an unconditional 50/50.
 */
inline DfResolvedBudget resolve_df_budget(DfBudgetWorkload workload, DfResourceEnvelope resource,
                                          std::size_t requested_bytes) noexcept {
  constexpr std::size_t mib = 1024U * 1024U;
  constexpr std::size_t min_auto = 32U * mib;
  constexpr std::size_t max_fallback = 1024U * mib;
  constexpr std::size_t min_headroom = 256U * mib;

  const long double n = static_cast<long double>(std::max<std::size_t>(1, workload.nbf));
  const long double a = static_cast<long double>(std::max<std::size_t>(1, workload.naux));
  const long double atoms = static_cast<long double>(std::max<std::size_t>(1, workload.atoms));
  const long double batch = static_cast<long double>(std::max<std::size_t>(1, workload.batch));
  const long double diis =
      static_cast<long double>(std::min<std::size_t>(workload.diis_history, 12U));

  const long double value_demand =
      16.0L * mib + sizeof(double) * (4.0L * n * n * a + batch * (8.0L + 2.0L * diis) * n * n);
  const long double response_demand =
      workload.forces ? 8.0L * mib + sizeof(double) * 3.0L * atoms * (n * n + a * a + n * a) : 0.0L;
  const auto workload_target = std::max(df_budget_bytes(value_demand + response_demand), min_auto);
  const long double demand = value_demand + response_demand;
  long double response_fraction = demand > 0.0L ? response_demand / demand : 0.5L;
  response_fraction = std::clamp(response_fraction, 0.20L, 0.70L);
  const auto resident_value_floor = df_resident_value_admission_floor(workload);
  const auto resident_target =
      workload.forces ? df_budget_ceiling(static_cast<long double>(resident_value_floor) /
                                          (1.0L - response_fraction))
                      : resident_value_floor;

  DfResolvedBudget result;
  result.requested_bytes = requested_bytes;
  result.live_resource = resource.live;
  result.observed_free_bytes = resource.live ? resource.free_bytes : 0U;
  result.observed_total_bytes = resource.live ? resource.total_bytes : 0U;

  if (requested_bytes != 0U) {
    result.total_bytes = requested_bytes;
  } else if (resource.live) {
    const auto fractional = resource.total_bytes / 8U;
    const auto desired_headroom = std::max(min_headroom, fractional);
    // When the desired reservation exceeds free memory, the existing fallback
    // retains half that free envelope. Report that actual reservation, not an
    // impossible amount larger than the observed free-memory capacity.
    result.reserved_headroom_bytes = resource.free_bytes > desired_headroom
                                         ? desired_headroom
                                         : resource.free_bytes - resource.free_bytes / 2U;
    const auto after_absolute = resource.free_bytes - result.reserved_headroom_bytes;
    const auto available = after_absolute - after_absolute / 4U;
    // Keep host preparation source-backed while retaining the device-value
    // floor when it fits the live post-headroom envelope. Tight devices remain
    // on the bounded streamed target.
    const auto admitted_target =
        resident_target <= available ? std::max(workload_target, resident_target) : workload_target;
    result.total_bytes = std::min(admitted_target, available);
  } else {
    result.total_bytes = std::min(workload_target, max_fallback);
  }

  if (!workload.forces) {
    result.value_bytes = result.total_bytes;
    // A resolved zero is exhaustion, never a feasible implementation default.
    result.feasible = result.total_bytes != 0U;
    return result;
  }
  if (result.total_bytes < 2U) {
    result.value_bytes = result.total_bytes;
    result.feasible = false;
    return result;
  }

  auto response =
      static_cast<std::size_t>(static_cast<long double>(result.total_bytes) * response_fraction);
  response = std::clamp<std::size_t>(response, 1U, result.total_bytes - 1U);
  result.response_bytes = response;
  result.value_bytes = result.total_bytes - response;
  return result;
}

/** Partition an already resolved envelope; zero remaining bytes is exhaustion,
 * never a new automatic request. Preserve the original probe/headroom identity. */
inline DfResolvedBudget resolve_df_subbudget(DfBudgetWorkload workload,
                                             const DfResolvedBudget& envelope,
                                             std::size_t retained_bytes) noexcept {
  auto result = envelope;
  result.total_bytes = result.value_bytes = result.response_bytes = 0;
  result.feasible = false;
  if (!envelope.feasible || retained_bytes >= envelope.total_bytes) return result;
  const auto remaining = envelope.total_bytes - retained_bytes;
  const auto split = resolve_df_budget(workload, {}, remaining);
  result.total_bytes = split.total_bytes;
  result.value_bytes = split.value_bytes;
  result.response_bytes = split.response_bytes;
  result.feasible = split.feasible;
  return result;
}

/** Compatibility helpers for explicit-only callers. Zero no longer means a
 * hidden implementation default; production resolves zero with workload and
 * resource information through resolve_df_budget. */
inline std::size_t df_value_budget(std::size_t requested, bool forces) noexcept {
  if (!requested || !forces) return requested;
  return resolve_df_budget({1, 1, 1, 1, 0, true}, {}, requested).value_bytes;
}
inline std::size_t df_force_budget(std::size_t requested) noexcept {
  if (!requested) return 0U;
  return resolve_df_budget({1, 1, 1, 1, 0, true}, {}, requested).response_bytes;
}

}  // namespace vibeqc::scf

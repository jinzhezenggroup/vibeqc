#ifndef VIBEQC_SCF_TYPES_HPP
#define VIBEQC_SCF_TYPES_HPP

#include <memory>
#include <optional>
#include <vector>

#include "scf/fock_build.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::scf {

struct ScfHooks;

/** How the requested floating-point precision policy actually resolved. */
struct PrecisionProvenance {
  uint32_t policy_version{1};
  /** Requested mode (\p vibeqc_precision_mode). */
  int32_t requested_mode{VIBEQC_PRECISION_FP64};
  /** Effective Fock bits: 64 for FP64, 32 when a mixed route is active. */
  uint32_t effective_bits{64};
  /** Tile threshold for \p auto; zero when the mixed route is not active. */
  double mixed_precision_fock_threshold{0.0};
  /** The strict FP64 target refinement ran at the end of the run. */
  bool strict_refinement_applied{false};
  /** Accumulated FP32 Fock rounding the admission budget certified; 0 if none. */
  double mixed_precision_reserved_error{0.0};
  /** FP64 target-precision iterations run after the mixed iterative stage. */
  uint32_t refinement_iterations{0};
};

/** Numerical controls shared by the implemented mean-field solvers. */
struct ScfOptions {
  unsigned max_iterations{100};
  unsigned diis_history{8};
  double energy_tolerance{1.0e-10};
  double density_tolerance{1.0e-8};
  double screening_tolerance{1.0e-12};
  /** Select the DF solver; direct four-center remains the default. */
  vibeqc_density_fitting_mode density_fitting_mode{VIBEQC_DENSITY_FITTING_NONE};
  /** Relative cutoff used when factoring the auxiliary Coulomb metric. */
  double density_fitting_relative_threshold{1.0e-10};
  /** Byte budget for bounded DF plan/integral work; zero means implementation default. */
  std::size_t density_fitting_memory_budget_bytes{};
  /** Correlated energy consumers require values-only, bounded direct RHF. */
  bool export_physical_reference{false};
  std::size_t reference_memory_budget_bytes{};
  /**
   * Requested floating-point execution policy. \p std::nullopt (absent) preserves
   * the legacy VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD diagnostic switch; an
   * explicit \p VIBEQC_PRECISION_FP64 keeps the pure double path; an explicit
   * \p VIBEQC_PRECISION_AUTO derives the mixed Fock threshold from the
   * tolerances and finishes with a strict FP64 target refinement.
   */
  std::optional<vibeqc_precision_mode> precision_mode{};
  /** Resolved once; old internal callers may leave this unset for direct HF. */
  std::optional<ResolvedFockBuild> resolved_fock_build;
  /** Explicit synchronous CPU proposal/trace opt-in; null has no snapshot work. */
  ScfHooks* hooks{};
  /** Diagnostic proposal bridge rejects malformed seeds instead of normalizing them. */
  bool strict_initial_density{};
  /** False for an explicit energy-only endpoint. Backends must then omit
   * derivative evaluation and return an empty force vector. */
  bool compute_forces{true};
};

/** Internal mean-field result, including state retained for warm starts. */
/** Owned physical canonical RHF state used by bounded post-HF consumers. */
struct PhysicalReference {
  std::size_t nbf{};
  std::size_t nocc{};
  std::vector<double> overlap, hcore, fock, coefficients, orbital_energies, density;
  double energy{};
  double commutator_residual{};
  double canonical_density_drift{};
  double eigen_residual{};
  std::size_t numeric_capacity_bytes{};
};

struct ScfResult {
  double energy{};
  std::vector<double> forces;
  // RHF stores one N x N AO density; UHF stores alpha then beta matrices.
  // The state is explicit and remains private to prepared execution plans.
  std::vector<double> density;
  unsigned iterations{};
  double energy_change{};
  double density_rms{};
  bool converged{};
  bool initial_density_used{};
  /** CPU physical operator evaluations, counting a joint UHF J/K as one build. */
  std::size_t fock_builds{};
  /**
   * How the requested precision policy resolved. Set by the backend that can
   * report it (the CUDA mixed-precision route); the strict FP64 default
   * leaves the FP64 defaults in place.
   */
  PrecisionProvenance precision{};
  std::shared_ptr<const PhysicalReference> reference;
};

}  // namespace vibeqc::scf

#endif

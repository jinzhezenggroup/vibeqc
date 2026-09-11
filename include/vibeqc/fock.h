#ifndef VIBEQC_FOCK_H
#define VIBEQC_FOCK_H

#include "vibeqc/vibeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vibeqc_fock_plan vibeqc_fock_plan;
typedef int32_t vibeqc_fock_spin;
enum { VIBEQC_FOCK_RESTRICTED = 0, VIBEQC_FOCK_UNRESTRICTED = 1 };
typedef int32_t vibeqc_fock_operator;
enum { VIBEQC_FOCK_FULL_RANGE = 0, VIBEQC_FOCK_SHORT_RANGE = 1, VIBEQC_FOCK_LONG_RANGE = 2 };
typedef int32_t vibeqc_fock_approximation;
enum { VIBEQC_FOCK_EXACT = 0, VIBEQC_FOCK_DENSITY_FITTED = 1 };

/** Raw J/K coefficients are applied once by native Fock/energy assembly.
 * present is 0 or 1. A present term with coefficient zero still returns its
 * raw matrix. Reserved SR/LR operators currently fail explicitly. */
typedef struct vibeqc_fock_term {
  int32_t present;
  double coefficient;
  vibeqc_fock_operator op;
  double omega;
  vibeqc_fock_approximation approximation;
} vibeqc_fock_term;

/** Mathematical identity, independent of backend and memory placement.
 * Restricted D includes double occupation; unrestricted Da/Db have unit
 * occupation. Standard HF uses cJ=1, cK=-0.5 (restricted) or -1 (unrestricted).
 */
typedef struct vibeqc_fock_spec {
  uint32_t struct_size, abi_version, spec_version;
  vibeqc_fock_spin spin;
  uint32_t derivative_order;
  vibeqc_fock_term coulomb, exchange;
} vibeqc_fock_spec;

/** NULL controls use screening=1e-12, metric cutoff=1e-10 and the default
 * buffer allowance. An explicit screening zero means unscreened execution.
 * Screening must be finite and nonnegative. Metric cutoffs must be finite
 * and in [0, 1), including when no fitted term is requested.
 * A zero metric cutoff selects 1e-10; zero device bytes selects 256 MiB for
 * the independent CUDA source. These controls never authorize a change from
 * exact to fitted mathematics. */
typedef struct vibeqc_fock_controls {
  uint32_t struct_size, abi_version;
  double screening_tolerance, metric_relative_threshold;
  uint64_t device_budget_bytes;
} vibeqc_fock_controls;

/** Construct owned normalized source snapshots. system/auxiliary/context may
 * be destroyed after success. A null auxiliary uses the orbital basis for
 * fitted terms; unused auxiliary data does not affect exact-only identity.
 * Failure leaves *output NULL. Independent CUDA execution uses host SCF
 * control and CUDA integral consumers; ordinary HF APIs retain fused solvers.
 * A handle and its error/diagnostic state are not concurrently reentrant;
 * serialize every call on a handle and its destruction.
 */
VIBEQC_API vibeqc_status vibeqc_fock_plan_create(
    vibeqc_context* context, const vibeqc_system* system, const vibeqc_system* auxiliary,
    const vibeqc_fock_spec* spec, const vibeqc_fock_controls* controls, vibeqc_fock_plan** output);
VIBEQC_API void vibeqc_fock_plan_destroy(vibeqc_fock_plan* plan);
/** Valid until the next call on this plan; null handles return a fixed message. */
VIBEQC_API const char* vibeqc_fock_plan_last_error(const vibeqc_fock_plan* plan);

/** Caller-owned optional outputs. All matrix buffers use matrix_count=nbf^2
 * and row-major public AO order. Requested output buffers must be disjoint.
 * Absent raw terms leave their buffers untouched. Fock matrices include Hcore.
 * gradient is the fixed-density TWO-ELECTRON energy derivative only, with
 * gradient_count=3*Natom; it excludes one-electron, nuclear and Pulay terms.
 * Unrequested gradients use NULL and count zero. All outputs change only on
 * success. This is a fixed-density operation, with no density normalization.
 */
typedef struct vibeqc_fock_result {
  uint32_t struct_size, abi_version;
  uint64_t matrix_count, gradient_count;
  double* coulomb;
  double* exchange_alpha;
  double* exchange_beta;
  double* fock_alpha;
  double* fock_beta;
  double* gradient;
  double energy_one_electron, energy_two_electron, nuclear_repulsion;
} vibeqc_fock_result;

VIBEQC_API vibeqc_status vibeqc_fock_plan_evaluate(vibeqc_fock_plan* plan, const double* density,
                                                   uint64_t density_count, const double* beta,
                                                   uint64_t beta_count, vibeqc_fock_result* result);

/** Convergence controls only; NULL selects 100 iterations, DIIS history 8,
 * energy tolerance 1e-10 Hartree and density RMS tolerance 1e-8. Explicit
 * counts must be positive and tolerances finite and positive. */
typedef struct vibeqc_fock_scf_controls {
  uint32_t struct_size, abi_version;
  uint32_t max_iterations, diis_history;
  double energy_tolerance, density_tolerance;
} vibeqc_fock_scf_controls;

/** Optional caller-owned SCF outputs in public AO/atom order. Restricted
 * density_count is nbf^2; unrestricted is 2*nbf^2, alpha followed by beta.
 * NULL density with count zero omits its copy. NULL forces with count zero
 * requests energy-only execution; otherwise force_count must be 3*Natom and
 * the plan must support first derivatives. Forces are COMPLETE negative
 * energy derivatives, including one-electron, nuclear and overlap/Pulay terms.
 * Buffers and this descriptor must be disjoint. All outputs remain unchanged
 * on failure, including NOT_CONVERGED. The plan retains sources, not iterates.
 */
typedef struct vibeqc_fock_scf_result {
  uint32_t struct_size, abi_version;
  double* density;
  double* forces;
  uint64_t density_count, force_count;
  double energy, energy_change, density_rms;
  uint32_t iterations;
  int32_t initial_density_used;
  uint64_t fock_builds;
} vibeqc_fock_scf_result;

/** Solve the declared native J/K energy using the shared host SCF driver.
 * No XC is added. CUDA plans use CUDA integrals/J/K/response and host
 * DIIS/eigensolves. Optional initial_density uses the output density layout;
 * it must satisfy the existing Hermiticity, overlap-metric occupation and
 * electron/spin trace guards. NULL/zero requests the core-Hamiltonian guess.
 * Input density is copied before execution and may alias an output buffer.
 * Handles are not concurrently reentrant; serialize calls and destruction.
 */
VIBEQC_API vibeqc_status vibeqc_fock_plan_solve(vibeqc_fock_plan* plan,
                                                const vibeqc_fock_scf_controls* controls,
                                                const double* initial_density,
                                                uint64_t initial_density_count,
                                                vibeqc_fock_scf_result* result);

/** Stable v1 schedule identifiers for prepared source execution. */
enum {
  VIBEQC_FOCK_CPU_REFERENCE = 0,
  VIBEQC_FOCK_CUDA_FUSED = 1,
  VIBEQC_FOCK_STANDARD_DF = 2,
  VIBEQC_FOCK_CPU_INDEPENDENT = 3,
  VIBEQC_FOCK_CUDA_INDEPENDENT = 4
};

/** Requested and canonical semantics plus actual prepared source diagnostics.
 * The public independent plan always uses the common provider boundary even
 * when semantic resolution identifies a standard-HF preferred schedule.
 * source_schedule records the actual plan route and resolved_schedule records
 * that preference. CUDA module/driver/library-private storage is excluded.
 */
typedef struct vibeqc_fock_diagnostic {
  uint32_t struct_size, abi_version;
  vibeqc_fock_spec requested, resolved;
  vibeqc_backend backend;
  int32_t resolved_schedule, source_schedule;
  uint64_t nbf, coordinate_count, device_bytes, device_budget_bytes;
  uint64_t auxiliary_rank, auxiliary_tile;
  double screening_tolerance, metric_relative_threshold;
  int32_t df_streamed;
  char direct_schedule[96], df_value_backend[48], df_value_mapping[32], df_response_mapping[32];
  char one_electron_value_backend[32], one_electron_value_mapping[32];
  char one_electron_response_mapping[32];
} vibeqc_fock_diagnostic;
VIBEQC_API vibeqc_status vibeqc_fock_plan_diagnostic(const vibeqc_fock_plan* plan,
                                                     vibeqc_fock_diagnostic* diagnostic);

#ifdef __cplusplus
}
#endif
#endif

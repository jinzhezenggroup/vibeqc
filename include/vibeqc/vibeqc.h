#ifndef VIBEQC_VIBEQC_H
#define VIBEQC_VIBEQC_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(VIBEQC_BUILDING_LIBRARY)
#define VIBEQC_API __declspec(dllexport)
#else
#define VIBEQC_API __declspec(dllimport)
#endif
#else
#define VIBEQC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VIBEQC_ABI_VERSION 0u

typedef int32_t vibeqc_status;
enum {
  VIBEQC_STATUS_SUCCESS = 0,
  VIBEQC_STATUS_INVALID_ARGUMENT = 1,
  VIBEQC_STATUS_ABI_MISMATCH = 2,
  VIBEQC_STATUS_NOT_IMPLEMENTED = 3,
  VIBEQC_STATUS_NOT_CONVERGED = 4,
  /** Compatibility name retained for the original HF-only ABI. */
  VIBEQC_STATUS_SCF_NOT_CONVERGED = VIBEQC_STATUS_NOT_CONVERGED,
  VIBEQC_STATUS_NUMERICAL_FAILURE = 5,
  VIBEQC_STATUS_CUDA_ERROR = 6,
  VIBEQC_STATUS_OUT_OF_MEMORY = 7,
  VIBEQC_STATUS_INTERNAL_ERROR = 8,
  /**
   * A run has been prepared but the precision-provenance query ran before a
   * completed execution (or after one that threw). Not an error in the run
   * itself: the \p fp64 record is simply not yet populated. Serializing callers
   * treat this as "no provenance yet" (e.g. Python returns None).
   */
  VIBEQC_STATUS_PRECISION_UNAVAILABLE = 9
};

#include "vibeqc/generated_method_ids.h"

/** Broad algorithm family used for capability discovery and dispatch. */
typedef int32_t vibeqc_method_family;
enum {
  VIBEQC_METHOD_FAMILY_HARTREE_FOCK = 1,
  VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL = 2,
  VIBEQC_METHOD_FAMILY_COUPLED_CLUSTER = 3,
  VIBEQC_METHOD_FAMILY_PERTURBATION = 4,
  VIBEQC_METHOD_FAMILY_SEMIEMPIRICAL = 5
};

typedef uint32_t vibeqc_property_flags;
enum { VIBEQC_PROPERTY_ENERGY = 1u << 0, VIBEQC_PROPERTY_FORCES = 1u << 1 };

typedef int32_t vibeqc_backend;
enum {
  VIBEQC_BACKEND_CPU_REFERENCE = 0,
  VIBEQC_BACKEND_CUDA = 1,
  /** Reserved compatibility tag used by pre-device-resident prototypes. */
  VIBEQC_BACKEND_HYBRID_CUDA = 2
};

/** Density-fitting execution policy for Hartree-Fock methods. */
typedef int32_t vibeqc_density_fitting_mode;
enum {
  /** Preserve the existing direct four-center J/K path (the default). */
  VIBEQC_DENSITY_FITTING_NONE = 0,
  /** Use the independent CPU density-fitting reference implementation. */
  VIBEQC_DENSITY_FITTING_CPU_REFERENCE = 1,
  /** Require the accelerator-native density-fitting path. */
  VIBEQC_DENSITY_FITTING_CUDA = 2,
  /** Select CUDA when requested by the context, otherwise use CPU reference. */
  VIBEQC_DENSITY_FITTING_AUTO = 3
};

/**
 * Floating-point execution policy for the selected mean-field method.
 *
 * \p fp64 keeps the current bit-for-bit exact execution. \p auto enables the
 * profile-backed lower-precision contraction route, deriving its tile
 * threshold from the requested tolerances and always finishing with a strict
 * FP64 refinement of the converged density.
 */
typedef int32_t vibeqc_precision_mode;
enum {
  /** Preserve the existing double-precision execution (the default). */
  VIBEQC_PRECISION_FP64 = 0,
  /**
   * Select a lower-precision contraction route only when the accumulated-error
   * budget certifies it for the requested accuracy, and always finish with a
   * strict FP64 target refinement that continues exact iterations until the
   * requested criteria are met. When the budget cannot certify a cutoff the
   * policy keeps the FP64 operator instead of accumulating rounding.
   */
  VIBEQC_PRECISION_AUTO = 1
};

/** Semilocal grid/XC execution schedule for native CUDA KS. */
typedef int32_t vibeqc_xc_execution_schedule;
enum { VIBEQC_XC_EXECUTION_DEVICE_FUSED = 0, VIBEQC_XC_EXECUTION_HOST_UNFUSED = 1 };

typedef int32_t vibeqc_basis_representation;
enum {
  /** CCA-ordered Cartesian functions: 1, 3, 6, and 10 AOs for s-p-d-f. */
  VIBEQC_BASIS_CARTESIAN = 0,
  /** Real spherical functions in PySCF/libcint order: 1, 3, 5, and 7 AOs. */
  VIBEQC_BASIS_SPHERICAL = 1
};

/**
 * Setup-time CUDA DF metric and value/J/K plan evidence; peaks are estimates.
 *
 * Plan-slot order: system_index is the original input, bucket_id its owning bucket.
 * Peaks exclude generated-force staging and opaque provider allocations. Whole-HF
 * resource observations separately account for tracked response allocations.
 */
typedef struct vibeqc_density_fitting_metric_diagnostic {
  uint32_t bucket_id;
  uint32_t system_index;
  uint64_t effective_rank;
  double absolute_threshold;
  double condition_number;
  uint64_t solver_device_workspace_bytes;
  uint64_t solver_host_workspace_bytes;
  uint64_t device_resident_bytes;
  uint64_t peak_device_bytes;
  uint64_t host_resident_bytes;
  uint64_t peak_host_bytes;
  uint64_t auxiliary_tile;
  int32_t streamed;
} vibeqc_density_fitting_metric_diagnostic;

typedef struct vibeqc_context vibeqc_context;
typedef struct vibeqc_system vibeqc_system;
typedef struct vibeqc_calculation vibeqc_calculation;
typedef struct vibeqc_batch vibeqc_batch;

typedef struct vibeqc_d3_batch vibeqc_d3_batch;
typedef struct vibeqc_d4_batch vibeqc_d4_batch;

typedef int32_t vibeqc_d3_damping;
enum { VIBEQC_D3_DAMPING_BJ = 1 };

/** Geometry-only D3 system. Coordinates are Bohr and copied at prepare. */
typedef struct vibeqc_d3_system_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const int32_t* atomic_numbers;
  const double* coordinates;
  uint32_t atom_count;
} vibeqc_d3_system_descriptor;

/** Two-body D3(BJ) model. s9 must remain zero in the production v1 slice. */
typedef struct vibeqc_d3_bj_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  vibeqc_d3_damping damping;
  double s6;
  double s8;
  double a1;
  double a2;
  double s9;
  double cn_cutoff;
  double pair_cutoff;
  double pair_switch_width;
  uint64_t maximum_bytes;
} vibeqc_d3_bj_descriptor;

/** Optional changed geometry for one prepared D3 batch member. */
typedef struct vibeqc_d3_batch_input_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const double* coordinates;
  uint32_t coordinate_count;
} vibeqc_d3_batch_input_descriptor;

/** Caller-owned result buffer; gradient is dE/dR (not force). */
typedef struct vibeqc_d3_batch_item_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  vibeqc_status status;
  double energy;
  double* gradient;
  uint32_t gradient_count;
  vibeqc_backend executed_backend;
} vibeqc_d3_batch_item_result_descriptor;

/** Bounded production owner diagnostics. */
typedef struct vibeqc_d3_runtime_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  vibeqc_backend backend;
  uint64_t plan_host_bytes;
  uint64_t execution_host_bytes;
  uint64_t device_bytes;
  uint64_t table_bytes;
  uint64_t workspace_bytes;
  uint64_t maximum_bytes;
  uint64_t total_atoms;
  uint32_t system_count;
  uint32_t maximum_atoms;
} vibeqc_d3_runtime_diagnostic;

typedef int32_t vibeqc_d4_profile;
enum { VIBEQC_D4_PROFILE_STANDARD_EEQ = 1, VIBEQC_D4_PROFILE_R2SCAN3C_EEQ = 2 };

/** Molecular nonperiodic D4(BJ)-EEQ system. Coordinates are Bohr. */
typedef struct vibeqc_d4_system_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const int32_t* atomic_numbers;
  const double* coordinates;
  uint32_t atom_count;
  double total_charge;
} vibeqc_d4_system_descriptor;

/**
 * Explicit D4(BJ)-EEQ model identity. There is deliberately no generic GFN2
 * default: callers must provide the named-method parameters and EEQ profile.
 */
typedef struct vibeqc_d4_bj_eeq_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  vibeqc_d4_profile profile;
  double s6;
  double s8;
  double s9;
  double a1;
  double a2;
  double ga;
  double gc;
  double cn_cutoff;
  double pair_cutoff;
  double atm_cutoff;
  uint64_t maximum_bytes;
} vibeqc_d4_bj_eeq_descriptor;

typedef struct vibeqc_d4_batch_input_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const double* coordinates;
  uint32_t coordinate_count;
} vibeqc_d4_batch_input_descriptor;

/** Caller-owned D4 result buffers; gradient is dE/dR, charges are EEQ2019. */
typedef struct vibeqc_d4_batch_item_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  vibeqc_status status;
  double energy;
  double two_body_energy;
  double atm_energy;
  double* gradient;
  uint32_t gradient_count;
  double* charges;
  uint32_t charge_count;
  vibeqc_backend executed_backend;
} vibeqc_d4_batch_item_result_descriptor;

/** Bounded production owner and replay/scheduling evidence. */
typedef struct vibeqc_d4_runtime_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  vibeqc_backend backend;
  vibeqc_d4_profile profile;
  uint64_t plan_host_bytes;
  uint64_t execution_host_bytes;
  uint64_t device_bytes;
  uint64_t table_bytes;
  uint64_t workspace_bytes;
  uint64_t maximum_bytes;
  uint64_t total_atoms;
  uint32_t system_count;
  uint32_t maximum_atoms;
  uint32_t worker_blocks;
  uint32_t workspace_slots;
  uint64_t execution_count;
  uint64_t unchanged_geometry_replays;
  uint64_t changed_geometry_replays;
  uint64_t coordinate_h2d_bytes;
  uint64_t kernel_launches;
  int32_t atm_enabled;
} vibeqc_d4_runtime_diagnostic;

typedef uint32_t vibeqc_batch_flags;
enum {
  /** Retain each converged AO density for the next execution of the plan. */
  VIBEQC_BATCH_ENABLE_WARM_STARTS = 1u << 0,
  /**
   * Collect the final density-screened direct-J/K shell-class work profile.
   *
   * This diagnostic adds one untimed-by-default CUDA reduction after the
   * final compaction pass. Leave it disabled for production timing runs.
   */
  VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING = 1u << 1,
  /**
   * Collect one device-timed record for every SCF iteration eigensolve.
   *
   * The instrumentation is inserted into the device-tail CUDA Graph and is
   * intended only for diagnosing divergent fleets. Leave it disabled during
   * production endpoint timing.
   */
  VIBEQC_BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING = 1u << 2
};

/** Number of pair/pair-exchange-reduced s/p/d/f quartet shell classes. */
#define VIBEQC_DIRECT_SHELL_CLASS_COUNT 55u

/** Work retained for one shell class after final-density screening. */
typedef struct vibeqc_shell_class_profile_entry {
  uint64_t shell_quartets;
  uint64_t tiles;
  uint64_t ao_quartets;
  uint64_t primitive_quartets;
} vibeqc_shell_class_profile_entry;

#define VIBEQC_PPPS_PROFILE_BLOCK_SIZE_COUNT 4u
#define VIBEQC_PPPS_PROFILE_ORIENTATION_COUNT 2u
#define VIBEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT 65u

/**
 * Final-density statistics for the exact resident PPPS production queue.
 *
 * Block-size arrays are ordered as 32, 64, 128, and 256 threads. Orientation
 * arrays are ordered as 1110 then 1011. Primitive-pair buckets 0..63 are
 * exact; bucket 64 contains 64 or more primitive pairs.
 */
typedef struct vibeqc_ppps_queue_profile {
  uint64_t descriptor_slots;
  uint64_t non_empty_descriptors;
  uint64_t empty_descriptors;
  uint64_t tasks;
  uint64_t primitive_work;
  uint32_t ket_count_min;
  uint32_t ket_count_median;
  uint32_t ket_count_p90;
  uint32_t ket_count_p99;
  uint32_t ket_count_max;
  double lane_efficiency[VIBEQC_PPPS_PROFILE_BLOCK_SIZE_COUNT];
  double primitive_warp_efficiency;
  double task_tail_imbalance[VIBEQC_PPPS_PROFILE_BLOCK_SIZE_COUNT];
  double primitive_tail_imbalance[VIBEQC_PPPS_PROFILE_BLOCK_SIZE_COUNT];
  uint64_t orientation_tasks[VIBEQC_PPPS_PROFILE_ORIENTATION_COUNT];
  uint64_t orientation_primitive_work[VIBEQC_PPPS_PROFILE_ORIENTATION_COUNT];
  uint64_t bra_primitive_tasks[VIBEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
  uint64_t bra_primitive_work[VIBEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
  uint64_t ket_primitive_tasks[VIBEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
  uint64_t ket_primitive_work[VIBEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
} vibeqc_ppps_queue_profile;

typedef int32_t vibeqc_eigensolver_family;
enum {
  VIBEQC_EIGENSOLVER_SMALL_NATIVE = 0,
  VIBEQC_EIGENSOLVER_JACOBI_BATCHED = 1,
  VIBEQC_EIGENSOLVER_XSYEV_BATCHED = 2,
  VIBEQC_EIGENSOLVER_GRAPH_NATIVE = 3
};

typedef int32_t vibeqc_eigensolver_selection_source;
enum {
  VIBEQC_EIGENSOLVER_SELECTION_DIMENSION_POLICY = 0,
  VIBEQC_EIGENSOLVER_SELECTION_EXACT_PROBE = 1,
  VIBEQC_EIGENSOLVER_SELECTION_EXACT_PROBE_FALLBACK = 2,
  /** Explicit benchmark-only override of the Graph eigensolver family. */
  VIBEQC_EIGENSOLVER_SELECTION_BENCHMARK_OVERRIDE = 3
};

typedef int32_t vibeqc_xsyev_eligibility_reason;
enum {
  VIBEQC_XSYEV_ELIGIBLE = 0,
  VIBEQC_XSYEV_ZERO_DIMENSION = 1,
  VIBEQC_XSYEV_INVALID_LEADING_DIMENSION = 2,
  VIBEQC_XSYEV_DOCUMENTED_DIMENSION_LIMIT = 3,
  VIBEQC_XSYEV_SOLVER_BATCH_LIMIT = 4,
  VIBEQC_XSYEV_DOCUMENTED_PRODUCT_LIMIT = 5
};

typedef int32_t vibeqc_xsyev_graph_probe_stage;
enum {
  VIBEQC_XSYEV_PROBE_NONE = 0,
  VIBEQC_XSYEV_PROBE_API_ELIGIBILITY = 1,
  VIBEQC_XSYEV_PROBE_SELECT_DEVICE = 2,
  VIBEQC_XSYEV_PROBE_DEVICE_IDENTITY = 3,
  VIBEQC_XSYEV_PROBE_CREATE_STREAM = 4,
  VIBEQC_XSYEV_PROBE_CREATE_SOLVER = 5,
  VIBEQC_XSYEV_PROBE_CREATE_PARAMETERS = 6,
  VIBEQC_XSYEV_PROBE_ALLOCATE_DATA = 7,
  VIBEQC_XSYEV_PROBE_QUERY_WORKSPACE = 8,
  VIBEQC_XSYEV_PROBE_INSUFFICIENT_DEVICE_MEMORY = 9,
  VIBEQC_XSYEV_PROBE_ALLOCATE_WORKSPACE = 10,
  VIBEQC_XSYEV_PROBE_ORDINARY_EXECUTION = 11,
  VIBEQC_XSYEV_PROBE_ORDINARY_VALIDATION = 12,
  VIBEQC_XSYEV_PROBE_BEGIN_CAPTURE = 13,
  VIBEQC_XSYEV_PROBE_CAPTURE_PROVIDER = 14,
  VIBEQC_XSYEV_PROBE_END_CAPTURE = 15,
  VIBEQC_XSYEV_PROBE_INSTANTIATE_DEVICE_LAUNCH_GRAPH = 16,
  VIBEQC_XSYEV_PROBE_UPLOAD_GRAPH = 17,
  VIBEQC_XSYEV_PROBE_HOST_GRAPH_REPLAY = 18,
  VIBEQC_XSYEV_PROBE_HOST_GRAPH_VALIDATION = 19,
  VIBEQC_XSYEV_PROBE_DEVICE_TAIL_REPLAY = 20,
  VIBEQC_XSYEV_PROBE_DEVICE_TAIL_VALIDATION = 21
};

/** Exact setup-time eigensolver selection evidence for one workload bucket. */
typedef struct vibeqc_eigensolver_diagnostic {
  uint32_t bucket_id;
  vibeqc_eigensolver_family ordinary_family;
  vibeqc_eigensolver_family graph_family;
  vibeqc_eigensolver_selection_source selection_source;
  uint64_t matrix_dimension;
  uint64_t physical_system_count;
  uint64_t solver_batch_count;
  int32_t api_eligible;
  vibeqc_xsyev_eligibility_reason api_reason;
  uint64_t matrix_batch_product;
  vibeqc_xsyev_graph_probe_stage probe_failure_stage;
  uint64_t device_workspace_bytes;
  uint64_t host_workspace_bytes;
  uint64_t available_device_bytes;
  int32_t device_id;
  uint8_t device_uuid[16];
  char device_name[256];
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
  int32_t cuda_runtime_version;
  int32_t cuda_driver_version;
  int32_t cusolver_version;
  int32_t cuda_error;
  int32_t cusolver_error;
  int32_t ordinary_execution_passed;
  int32_t graph_capture_passed;
  int32_t host_graph_replay_passed;
  int32_t device_tail_replay_passed;
  int32_t graph_eligible;
  double maximum_eigenvalue_error;
  double maximum_residual;
  double maximum_orthogonality_error;
} vibeqc_eigensolver_diagnostic;

typedef uint32_t vibeqc_eigensolver_inactive_touch_flags;
enum {
  /** An inactive matrix was copied before the provider call. */
  VIBEQC_EIGENSOLVER_INACTIVE_TOUCH_COPY = 1u << 0,
  /** cuBLAS transformed an inactive matrix before the provider call. */
  VIBEQC_EIGENSOLVER_INACTIVE_TOUCH_CUBLAS_TRANSFORM = 1u << 1,
  /** The provider input was replaced with a finite identity matrix. */
  VIBEQC_EIGENSOLVER_INACTIVE_TOUCH_IDENTITY_SANITIZE = 1u << 2
};

/** Device-timed evidence for one eigensolve in the device-tail SCF loop. */
typedef struct vibeqc_inactive_eigensolver_profile_entry {
  uint32_t bucket_id;
  uint32_t iteration;
  vibeqc_eigensolver_family family;
  uint32_t physical_system_count;
  uint32_t solver_batch_count;
  uint32_t active_physical_count;
  uint32_t active_solver_count;
  uint64_t solver_elapsed_nanoseconds;
  /** Number of inactive matrices found non-finite before identity repair. */
  uint32_t inactive_input_nonfinite_count;
  /** Number of inactive matrices still non-finite when submitted. */
  uint32_t inactive_submission_nonfinite_count;
  uint32_t inactive_info_nonzero_count;
  vibeqc_eigensolver_inactive_touch_flags inactive_touch_flags;
  int32_t provider_invoked;
} vibeqc_inactive_eigensolver_profile_entry;

typedef struct vibeqc_context_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  int32_t device_id;
  vibeqc_backend backend;
} vibeqc_context_descriptor;

typedef struct vibeqc_atom {
  int32_t atomic_number;
  double x;
  double y;
  double z;
} vibeqc_atom;

typedef struct vibeqc_primitive {
  double exponent;
  double coefficient;
} vibeqc_primitive;

typedef struct vibeqc_shell {
  uint32_t atom_index;
  uint32_t angular_momentum;
  uint32_t primitive_offset;
  uint32_t primitive_count;
} vibeqc_shell;

typedef struct vibeqc_system_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const vibeqc_atom* atoms;
  uint32_t atom_count;
  const vibeqc_shell* shells;
  uint32_t shell_count;
  const vibeqc_primitive* primitives;
  uint32_t primitive_count;
  int32_t charge;
  uint32_t multiplicity;
  /** Optional in older ABI-0 descriptors; absent fields imply Cartesian. */
  vibeqc_basis_representation basis_representation;
} vibeqc_system_descriptor;

/** Native KS model snapshot, copied during preparation. Method selectors choose
 * the audited semilocal component family and spin; optional suffixes supply
 * resolved composition (v2) and execution schedule (v3). Legacy prefixes retain
 * unit semilocal XC, no K, and device-fused CUDA XC. */
typedef struct vibeqc_ks_options {
  uint32_t struct_size;
  uint32_t abi_version;
  /** Version 1: semilocal-scaled-v1/pbe-spin-c2-1e-18. */
  uint32_t scf_domain_version;
  /** Grid contract version. Version 1 is the deterministic reference
   * prescription with unit-radius fallback. Version 2 is a fully resolved
   * production prescription with sourced element radii. */
  uint32_t grid_version;
  uint32_t radial_points;
  uint32_t angular_polar;
  uint32_t angular_azimuth;
  uint32_t partition_iterations;
  double coincident_tolerance;
  uint64_t tile_points;
  /** Radii [0..118] in Bohr, indexed by atomic number; slot zero is unused.
   * Version 1 accepts NULL/zero as the historical unit-radius fallback.
   * Version 2 requires a positive finite entry for every element actually
   * materialized; zero/missing entries fail closed. */
  const double* element_radii;
  uint32_t element_radius_count;
  uint32_t reserved_v1_padding;
  /** Optional v2 suffix: 0 retains legacy defaults; 1 uses the coefficients
   * below. Scaled composition is CPU PBE-family only. Full-range exact J has cJ=1.
   * PBE0 is X=3/4, C=1, cK=-1/8 (RKS total D) or -1/4 (UKS spin D).
   * Values are supplied by resolved MethodIR, never inferred from a name. */
  uint32_t composition_version;
  double semilocal_exchange_scale;
  double semilocal_correlation_scale;
  double fock_exchange_coefficient;
  /** Optional v3 suffix. Older prefixes use DEVICE_FUSED. */
  vibeqc_xc_execution_schedule xc_execution_schedule;
} vibeqc_ks_options;

/** Pure capability query. Version 3 accepts the v1/v2 prefixes and v3 suffix. */
VIBEQC_API uint32_t vibeqc_ks_options_version(void);

typedef struct vibeqc_method_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  vibeqc_method method;
  uint32_t max_iterations;
  uint32_t diis_history;
  double energy_tolerance;
  double density_tolerance;
  double screening_tolerance;
  /** Optional fields are ignored when struct_size ends before this member. */
  vibeqc_density_fitting_mode density_fitting_mode;
  /** Optional prepared system carrying the auxiliary-basis shell topology. */
  const vibeqc_system* density_fitting_auxiliary_basis;
  /** Relative eigenvalue threshold used for the auxiliary metric. */
  double density_fitting_relative_threshold;
  /** Planner budget in bytes. Positive values are hard upper bounds; zero
   * selects the workload/device-aware resource policy. */
  uint64_t density_fitting_memory_budget_bytes;
  /**
   * Optional floating-point execution policy. Absent or zero callers keep the
   * double-precision default; \p auto enables the safe lower-precision route.
   */
  vibeqc_precision_mode precision_mode;
  /** Optional combined numeric capacity for correlated reference/energy phases.
   * Zero selects 256 MiB; this is not a process-RSS or CUDA-context bound. */
  uint64_t correlation_memory_budget_bytes;
  /** Positive MP2 absolute denominator threshold in Hartree; zero uses 1e-10. */
  double mp2_denominator_threshold;
  /** Optional KS snapshot. NULL/absent preserves the original default model.
   * The descriptor and pointees need only outlive the prepare call. */
  const vibeqc_ks_options* ks_options;
  /** RCCSD controls. These are an additive struct-size-gated extension. An
   * absent field preserves the documented default; zero also selects the
   * default except ccsd_diis_history=0, which explicitly disables DIIS. */
  uint32_t ccsd_max_iterations;
  uint32_t ccsd_diis_history;
  double ccsd_energy_tolerance;
  double ccsd_residual_tolerance;
  double ccsd_denominator_threshold;
  double ccsd_damping;
  double ccsd_level_shift;
  /** Frozen occupied orbitals are not implemented for the native RCCSD owner.
   * Zero means all occupied orbitals are correlated. */
  uint32_t ccsd_frozen_core;
} vibeqc_method_descriptor;

/**
 * Read-only record of how the requested precision policy resolved. Populated by
 * \p vibeqc_calculation_get_precision_provenance after a prepared run; callers
 * that predate this field never see it because the out-parameter is optional.
 */
typedef struct vibeqc_precision_provenance {
  uint32_t struct_size;
  uint32_t abi_version;
  /** Policy version the resolver honored. */
  uint32_t policy_version;
  /** Requested mode (\p vibeqc_precision_mode). */
  int32_t requested_mode;
  /** Effective Fock precision: 64 for FP64, 32 when a mixed route is active. */
  uint32_t effective_bits;
  /** Tile threshold for \p auto; zero when the mixed route is not active. */
  double mixed_precision_fock_threshold;
  /** The strict FP64 target refinement ran at the end of the run. */
  int32_t strict_refinement_applied;
  /**
   * Accumulated FP32 Fock rounding the \p auto admission budget certified, or
   * zero when the cutoff was uncertified (explicit diagnostic override) or the
   * mixed route did not run. A zero threshold with a zero reserved error means
   * the FP64 operator was kept because no cutoff fit the requested accuracy.
   */
  double mixed_precision_reserved_error;
  /**
   * FP64 target-precision iterations executed after the mixed iterative stage.
   * Zero when the mixed route did not run; the reported \p iterations count
   * includes these refinement iterations.
   */
  int32_t refinement_iterations;
} vibeqc_precision_provenance;

typedef struct vibeqc_correlation_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  double reference_energy;
  double opposite_spin_energy;
  double same_spin_energy;
  double minimum_absolute_denominator;
  double reference_residual;
  uint64_t numeric_capacity_bytes;
  uint64_t energy_tile_count;
  /** Actual MO transfer staging; not an assertion that the whole method is resident. */
  int32_t mo_host_staging;
  uint64_t correlation_owned_device_bytes;
  uint64_t correlation_provider_retained_bytes;
  uint64_t mo_transfer_bytes;
  double host_to_device_ms;
  double device_to_host_ms;
  double transform_library_ms;
  double tensor_kernel_ms;
  char equation_hash[65];
  /** Completed canonical orbital-response solve; zero for energy-only runs. */
  uint64_t response_iterations;
  uint64_t response_restarts;
  double response_absolute_residual;
  double response_relative_residual;
  uint64_t response_workspace_bytes;
  /** Peak numeric staging owned by the force derivative contraction. */
  uint64_t derivative_workspace_bytes;
  /** Conservative simultaneous endpoint numeric-capacity plan. */
  uint64_t planned_endpoint_peak_bytes;
  /** Observed endpoint peak; zero means measurement is unavailable, not zero usage. */
  uint64_t measured_endpoint_peak_bytes;
  /** Bit 0=response converged, 1=shell-streamed derivative, 2=no global derivative tensors. */
  uint64_t force_provenance_flags;
  char response_operator_hash[65];
  /** Actual high-water payload of GMRES-owned arrays, including its result.
   * Excludes input spans, operator callbacks, other MP2 stages and allocator
   * overhead. Zero when unmeasured; never a complete endpoint measurement.
   */
  uint64_t measured_response_workspace_peak_bytes;
  /** Successful allocation events in the same GMRES ownership domain. */
  uint64_t response_workspace_allocation_count;
  /** RCCSD-only fields. Zero for methods that do not publish iterative CC state. */
  uint64_t ccsd_iterations;
  uint64_t ccsd_diis_restarts;
  double ccsd_correlation_energy;
  double ccsd_energy_change;
  double ccsd_singles_residual_max;
  double ccsd_doubles_residual_max;
  double ccsd_replay_singles_residual_max;
  double ccsd_replay_doubles_residual_max;
  uint64_t ccsd_setup_h2d_bytes;
  uint64_t ccsd_scalar_d2h_bytes;
  uint64_t ccsd_amplitude_d2h_bytes;
  uint64_t ccsd_synchronizations;
  char ccsd_replay_equation_hash[65];
  /** Standard canonical noniterative (T) correction; zero for non-RCCSD(T) methods. */
  double ccsd_t_triples_energy;
  /** Number of triangular virtual triples evaluated by the native (T) owner. */
  uint64_t ccsd_t_virtual_triples;
  /** Peak temporary numeric workspace owned by the native (T) evaluator. */
  uint64_t ccsd_t_workspace_bytes;
  /** Audited standard-(T) inventory identity; empty for non-RCCSD(T) methods. */
  char ccsd_t_equation_hash[65];
} vibeqc_correlation_diagnostic;

/** Executable capabilities for one method identifier. */
typedef struct vibeqc_method_capabilities_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  vibeqc_method method;
  vibeqc_method_family family;
  vibeqc_property_flags supported_properties;
  int32_t available;
  int32_t supports_batch;
} vibeqc_method_capabilities_descriptor;

typedef struct vibeqc_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  double energy;
  double* forces;
  uint32_t force_count;
  uint32_t iterations;
  double energy_change;
  /** Legacy convergence residual: SCF density-update RMS for mean-field
   * methods; iterative correlated methods may report their own physical
   * residual aggregate here and expose method-specific components separately. */
  double density_rms;
  int32_t converged;
  vibeqc_backend executed_backend;
} vibeqc_result_descriptor;

/** Separate SCF measures, queried without extending existing result layouts. */
typedef struct vibeqc_scf_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  double density_rms;
  /** RMS of physical F D S - S D F evaluated for the reported energy.
   * UKS combines the alpha/beta matrix entries in one RMS. */
  double physical_residual_rms;
} vibeqc_scf_diagnostic;

/** A physical KS iteration before any optional final RKS validation rebuild.
 * The first energy_change is +infinity because no preceding energy exists.
 * Density change and residual are maxima of the spin RMS values (RKS has one
 * total-density matrix), distinct from the joined-spin legacy result RMS. */
typedef struct vibeqc_ks_iteration {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t iteration;
  int32_t occupation_stabilized;
  double nuclear_energy;
  double one_electron_energy;
  double hartree_energy;
  double xc_energy;
  double energy_change;
  double density_change_max;
  double physical_residual_max;
  double electrons[2];
} vibeqc_ks_iteration;

/** Completed KS state, copied without extending legacy result-array strides.
 * Electron counts are Tr(D_s S), not integrated grid densities. Final terms
 * refer to the returned physical state; CPU RKS's post-loop validation can
 * make them differ from the last iteration. All energies are in Hartree. */
typedef struct vibeqc_ks_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t scf_domain_version;
  uint32_t required_ao_order;
  uint32_t history_count;
  int32_t initial_density_used;
  uint64_t occupations[2];
  uint64_t grid_points;
  uint64_t tile_points;
  uint64_t fock_builds;
  double electrons[2];
  double nuclear_energy;
  double one_electron_energy;
  double hartree_energy;
  double xc_energy;
  double density_change_max;
  double physical_residual_max;
} vibeqc_ks_diagnostic;

/** Cumulative transport performed by one prepared CUDA KS owner.
 *
 * Counts are measured at native copy/synchronization sites rather than
 * inferred from SCF iterations. They include immutable setup, explicit seed
 * uploads, scalar iteration records, changed-geometry warm-density exports,
 * discarded warm attempts, and occupation-control proposals. Routine
 * iteration matrices remain resident and do not contribute matrix D2H bytes.
 * CPU and non-KS owners report this diagnostic as unavailable.
 */
typedef struct vibeqc_ks_transport_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t setup_h2d_bytes;
  uint64_t density_h2d_bytes;
  uint64_t scalar_d2h_bytes;
  uint64_t matrix_d2h_bytes;
  uint64_t synchronizations;
  uint64_t iterations;
  uint64_t occupation_stabilized_proposals;
} vibeqc_ks_transport_diagnostic;

/** Optional per-system coordinates for a prepared ragged batch execution. */
typedef struct vibeqc_batch_input_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  /** Flat xyz coordinates in Bohr, or NULL to use the prepared geometry. */
  const double* coordinates;
  uint32_t coordinate_count;
} vibeqc_batch_input_descriptor;

/** Per-system output. Each item owns an independent status and diagnostics. */
typedef struct vibeqc_batch_item_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  vibeqc_status status;
  double energy;
  double* forces;
  uint32_t force_count;
  uint32_t iterations;
  double energy_change;
  double density_rms;
  int32_t converged;
  vibeqc_backend executed_backend;
  uint32_t bucket_id;
  int32_t warm_start_used;
  int32_t warm_start_fallback;
} vibeqc_batch_item_result_descriptor;

/** Return the ABI version implemented by the loaded shared library. */
VIBEQC_API uint32_t vibeqc_get_abi_version(void);

/** Source/codegen identity, independent of checkout paths and selected kernels. */
VIBEQC_API const char* vibeqc_get_source_identity(void);

/** Hardware and runtime identity for safe user-local schedule reuse.
 * Device ordinals use the CUDA runtime's scheduler-assigned visibility. No
 * probe changes CUDA_VISIBLE_DEVICES. UUID/PCI address are intentionally absent
 * because matching GPUs on different cluster nodes may share a tuned profile.
 */
typedef struct vibeqc_cuda_tuning_device_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  char name[256];
  char official_profile[128];
  int32_t major, minor, warp_size;
  int32_t maximum_threads_per_block, maximum_threads_per_sm, maximum_blocks_per_sm;
  int32_t registers_per_sm, maximum_registers_per_thread, sm_count;
  uint64_t shared_memory_per_block, shared_memory_per_block_optin, shared_memory_per_sm;
  int32_t runtime_version, driver_version, toolkit_version;
  int32_t release_build, fast_compile;
  int32_t portable;
} vibeqc_cuda_tuning_device_descriptor;

/** Probe an allocated GPU; CPU builds return NOT_IMPLEMENTED without probing. */
VIBEQC_API vibeqc_status vibeqc_cuda_tuning_device(int32_t device_id,
                                                   vibeqc_cuda_tuning_device_descriptor* output);

/** Return a stable, process-lifetime error string for a status code. */
VIBEQC_API const char* vibeqc_status_message(vibeqc_status status);

/** Query whether a method is currently executable. */
VIBEQC_API vibeqc_status vibeqc_method_available(vibeqc_method method, int32_t* available);

/** Query method family, properties, and batch support without preparing work. */
VIBEQC_API vibeqc_status vibeqc_method_get_capabilities(
    vibeqc_method method, vibeqc_method_capabilities_descriptor* capabilities);

VIBEQC_API vibeqc_status vibeqc_context_create(const vibeqc_context_descriptor* descriptor,
                                               vibeqc_context** context);
VIBEQC_API void vibeqc_context_destroy(vibeqc_context* context);

/** Borrow the last native failure detail, valid until the next failing call
 * on this context or its destruction. Empty when no detail has been recorded. */
VIBEQC_API const char* vibeqc_context_get_last_detail(const vibeqc_context* context);

/** Backward-compatible alias for vibeqc_context_get_last_detail. */
VIBEQC_API const char* vibeqc_context_last_error(const vibeqc_context* context);

VIBEQC_API vibeqc_status vibeqc_system_create(vibeqc_context* context,
                                              const vibeqc_system_descriptor* descriptor,
                                              vibeqc_system** system);
VIBEQC_API void vibeqc_system_destroy(vibeqc_system* system);

/** Scalar Gaussian residual ECP: c r^(power-2) exp(-exponent r^2).
 * channel=-1 is local, 0..3 is a nonlocal projector difference.
 * Core counts are atom-major and must leave positive effective ionic charges.
 * All buffers are copied; existing all-electron system_create ABI is unchanged. */
typedef struct vibeqc_ecp_term {
  uint32_t atom_index;
  int32_t channel;
  uint32_t power;
  double exponent;
  double coefficient;
} vibeqc_ecp_term;
VIBEQC_API vibeqc_status vibeqc_system_create_ecp(vibeqc_context* context,
                                                  const vibeqc_system_descriptor* descriptor,
                                                  const int32_t* core_electrons,
                                                  const vibeqc_ecp_term* terms, size_t term_count,
                                                  vibeqc_system** system);
/** Part-major [local, nonlocal], each containing value then atom/xyz derivatives.
 * Derivatives are positive energy derivatives, not forces. Output is owned by caller. */
VIBEQC_API vibeqc_status vibeqc_system_ecp_integrals(vibeqc_context* context,
                                                     const vibeqc_system* system,
                                                     uint32_t radial_points, uint32_t polar_points,
                                                     int32_t derivatives, double* output,
                                                     size_t output_count);

/** Numeric staging and explicit transfer counters for the generic CUDA gradient.
 * Caller-owned weights/system and pre-existing HF plans are outside this arena. */
typedef struct vibeqc_one_electron_gradient_resources {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t device_bytes;
  uint64_t host_numeric_bytes;
  uint64_t host_to_device_bytes;
  uint64_t device_to_host_bytes;
  uint64_t synchronous_uploads;
  uint64_t stream_synchronizations;
} vibeqc_one_electron_gradient_resources;

/** Synchronously contract fixed, full row-major public-AO weights with dS/dT/dV.
 * matrix_count must be NAO*NAO; each null weight pointer means a zero channel.
 * Off-diagonal ownership combines W_ij+W_ji, including nonsymmetric weights.
 * Output is a 3*Natom energy gradient in atom/xyz order; nuclear repulsion is
 * excluded. schedule=0 selects AO threads, 1 shell-pair warps, 2 serial per-system
 * diagnostics, and 3 AO-pair warps with lanes owning nuclear centers. maximum_bytes
 * independently bounds numeric host/device staging.
 * A CUDA context is required; failures do not silently fall back to CPU.
 * Optional resources must carry the current struct_size/abi_version.
 */
VIBEQC_API vibeqc_status vibeqc_system_one_electron_gradient_cuda(
    vibeqc_context* context, const vibeqc_system* system, const double* overlap_weights,
    const double* kinetic_weights, const double* attraction_weights, size_t matrix_count,
    unsigned schedule, size_t maximum_bytes, double* gradient, size_t gradient_count,
    vibeqc_one_electron_gradient_resources* resources);

/** Physical Fock builds in the last CPU batch execution, including final
 * rebuilds. A joint UHF alpha/beta J/K evaluation counts once. Returns
 * NOT_IMPLEMENTED for an unexecuted item, CUDA, or an incompletely counted
 * warm-to-cold retry. The result is never inferred from iteration count. */
VIBEQC_API vibeqc_status vibeqc_batch_get_last_fock_builds(const vibeqc_batch* batch,
                                                           uint32_t index, uint64_t* builds);

/**
 * Synchronously write the normalized rectangular overlap <target AO|source AO>
 * in row-major order. output_count must equal target_nbf * source_nbf. Both
 * systems may independently select Cartesian/spherical AOs and geometries.
 * This explicit CPU evaluator allocates no ERI or nuclear-derivative tensors.
 * The context supplies error details; its accelerator selection is irrelevant.
 */
VIBEQC_API vibeqc_status vibeqc_system_cross_overlap_cpu(vibeqc_context* context,
                                                         const vibeqc_system* target,
                                                         const vibeqc_system* source,
                                                         double* output, size_t output_count);

/** Owned numeric staging and explicit transfers for a generic DF gradient.
 * Caller-owned systems/weights and opaque CUDA allocations are excluded. */
typedef struct vibeqc_df_gradient_resources {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t host_bytes, device_bytes, host_to_device_bytes, device_to_host_bytes;
  uint64_t weight_tile_elements, tiles, uploads, stream_synchronizations;
} vibeqc_df_gradient_resources;

/** Contract fixed external DF weights into an energy gradient on CUDA.
 * bar_a is full row-major [mu,nu,P]; bar_m is full row-major [P,Q]. Every
 * element is counted once; nonsymmetric inputs require no implicit factors.
 * Counts equal NAO*NAO*NAUX and NAUX*NAUX even when a null channel means zero.
 * Orbital/auxiliary systems share physical atom coordinates, with independently
 * assigned shell owners. Output is [atom,xyz], excluding all non-DF terms.
 * maximum_bytes bounds numeric host staging and device allocations separately;
 * maximum_tile_elements=0 selects an automatic bound. schedule=0 uses threads,
 * 1 uses deterministic serial traversal. Failures preserve caller output.
 */
VIBEQC_API vibeqc_status vibeqc_system_df_gradient_cuda(
    vibeqc_context* context, const vibeqc_system* orbital, const vibeqc_system* auxiliary,
    const double* bar_a, size_t count_a, const double* bar_m, size_t count_m, unsigned schedule,
    size_t maximum_bytes, size_t maximum_tile_elements, double* gradient, size_t gradient_count,
    vibeqc_df_gradient_resources* resources);

VIBEQC_API vibeqc_status vibeqc_calculation_prepare(vibeqc_context* context,
                                                    const vibeqc_system* system,
                                                    const vibeqc_method_descriptor* descriptor,
                                                    vibeqc_calculation** calculation);
VIBEQC_API void vibeqc_calculation_destroy(vibeqc_calculation* calculation);

/**
 * Execute synchronously. To request forces, the caller owns result->forces and
 * provides at least 3 * atom_count doubles. A NULL pointer with force_count=0
 * requests energy and diagnostics only. Coordinates and all reported
 * derivatives use atomic units (Bohr, Hartree, Hartree/Bohr).
 */
VIBEQC_API vibeqc_status vibeqc_calculation_execute(vibeqc_calculation* calculation,
                                                    vibeqc_result_descriptor* result);

/** Read separate density-update and physical SCF residual measures.
 * Available after a completed supported solve, including NOT_CONVERGED.
 * Returns NOT_IMPLEMENTED before execution, after a backend execution failure, or
 * when the method does not report a physical residual. Such returns leave
 * out untouched. A NULL out is an availability query; otherwise the caller
 * supplies struct_size and abi_version. Currently populated by LDA/PBE KS.
 */
VIBEQC_API vibeqc_status vibeqc_calculation_get_scf_diagnostic(
    const vibeqc_calculation* calculation, vibeqc_scf_diagnostic* out);

/** Query completed KS state and optionally copy its entire iteration history.
 * NULL history/zero capacity queries summary or availability only. To copy
 * history, allocate at least out->history_count initialized descriptors and
 * query again. Every supplied descriptor must have its current size/ABI.
 * Validation failures leave all outputs untouched. NOT_IMPLEMENTED means no
 * completed KS record: unsupported method, not executed, or failed evaluation.
 * Caller serializes all execution/query calls on the prepared owner. */
VIBEQC_API vibeqc_status vibeqc_calculation_get_ks_diagnostic(const vibeqc_calculation* calculation,
                                                              vibeqc_ks_diagnostic* out,
                                                              vibeqc_ks_iteration* history,
                                                              uint32_t history_capacity);

/** Query cumulative prepared-owner CUDA KS transport. Available immediately
 * after CUDA KS preparation so callers can separate setup from phase deltas. */
VIBEQC_API vibeqc_status vibeqc_calculation_get_ks_transport_diagnostic(
    const vibeqc_calculation* calculation, vibeqc_ks_transport_diagnostic* out);

/**
 * Read the precision policy that resolved for a prepared run. Both the
 * availability query (a NULL \p out) and the copy-out are gated on whether a
 * completed execution has populated the record:
 *
 * - Before any execution, and after an execution that threw, the query returns
 *   \p VIBEQC_STATUS_PRECISION_UNAVAILABLE (and leaves \p out untouched).
 * - After a normal execution return (converged or not) the resolved record is
 *   copied into \p out and SUCCESS is returned.
 *
 * The out-parameter must carry the current struct_size/abi_version. A NULL
 * \p out is a cheap availability probe that never writes. Adding this query
 * never changes existing descriptors.
 */
VIBEQC_API vibeqc_status vibeqc_calculation_get_precision_provenance(
    const vibeqc_calculation* calculation, vibeqc_precision_provenance* out);
/** Most recent successful correlated execution; failure/absence is explicit. */
VIBEQC_API vibeqc_status vibeqc_calculation_get_correlation_diagnostic(
    const vibeqc_calculation* calculation, vibeqc_correlation_diagnostic* diagnostic);

/**
 * Read one batch item's precision record by its original input index.
 * The descriptor and NULL availability probe follow the single-calculation
 * query. Records are cleared on each execution and populated independently for
 * SUCCESS/NOT_CONVERGED items; rejected or throwing items return
 * VIBEQC_STATUS_PRECISION_UNAVAILABLE without writing to out. An out-of-range
 * index returns VIBEQC_STATUS_INVALID_ARGUMENT.
 */
VIBEQC_API vibeqc_status vibeqc_batch_get_precision_provenance(const vibeqc_batch* batch,
                                                               uint32_t index,
                                                               vibeqc_precision_provenance* out);

/**
 * Prepare a persistent ragged fleet plan. Systems may have different atom,
 * shell, primitive, and AO counts; no global padding is introduced.
 */
VIBEQC_API vibeqc_status vibeqc_batch_prepare(vibeqc_context* context,
                                              const vibeqc_system* const* systems,
                                              uint32_t system_count,
                                              const vibeqc_method_descriptor* descriptor,
                                              vibeqc_batch_flags flags, vibeqc_batch** batch);

VIBEQC_API void vibeqc_batch_destroy(vibeqc_batch* batch);

VIBEQC_API uint32_t vibeqc_batch_get_system_count(const vibeqc_batch* batch);

/**
 * Copy the most recent final-density shell-class profile.
 *
 * The batch must have been prepared with
 * `VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING`, executed through the CUDA direct
 * J/K path, and `entry_count` must be at least
 * `VIBEQC_DIRECT_SHELL_CLASS_COUNT`. Entries use the canonical triangular class
 * encoding documented by VIBEQC's direct shell scheduler.
 */
VIBEQC_API vibeqc_status vibeqc_batch_get_last_shell_class_profile(
    const vibeqc_batch* batch, vibeqc_shell_class_profile_entry* entries, uint32_t entry_count);

/** Copy PPPS queue statistics from the most recent profiled CUDA execution. */
VIBEQC_API vibeqc_status vibeqc_batch_get_last_ppps_queue_profile(
    const vibeqc_batch* batch, vibeqc_ppps_queue_profile* profile);

/**
 * Copy setup-time eigensolver evidence for every bucket in the last execution.
 *
 * Pass `entries = NULL` and `entry_count = 0` to query the required count in
 * `written_count`. A later warm replay returns the cached setup decision and
 * never performs another capability probe.
 */
VIBEQC_API vibeqc_status vibeqc_batch_get_last_eigensolver_diagnostics(
    const vibeqc_batch* batch, vibeqc_eigensolver_diagnostic* entries, uint32_t entry_count,
    uint32_t* written_count);

/**
 * Copy CUDA density-fitting metric conditioning and allocation diagnostics
 * from the most recent batch execution. Pass `entries = NULL` and
 * `entry_count = 0` to query the required count in `written_count`.
 */
VIBEQC_API vibeqc_status vibeqc_batch_get_last_density_fitting_metric_diagnostics(
    const vibeqc_batch* batch, vibeqc_density_fitting_metric_diagnostic* entries,
    uint32_t entry_count, uint32_t* written_count);

/**
 * Copy per-iteration inactive-eigensolver evidence from the last execution.
 *
 * The batch must opt into
 * `VIBEQC_BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING`. Pass `entries = NULL`
 * and `entry_count = 0` to query the required count. Records are bucket-major
 * and iteration-ordered within each bucket.
 */
VIBEQC_API vibeqc_status vibeqc_batch_get_last_inactive_eigensolver_profile(
    const vibeqc_batch* batch, vibeqc_inactive_eigensolver_profile_entry* entries,
    uint32_t entry_count, uint32_t* written_count);

/** Portable scientific buffers for an HF seed. This is a live ABI descriptor,
 * never an on-disk representation. Density is row-major RHF total or UHF alpha
 * then beta. Coordinates are the source geometry, in Bohr and input atom order.
 * Source diagnostics do not establish target convergence.
 */
typedef struct vibeqc_hf_warm_state {
  uint32_t struct_size;
  uint32_t abi_version;
  double* density;
  uint64_t density_count;
  double* coordinates;
  uint64_t coordinate_count;
  double energy;
  double energy_change;
  double density_rms;
  int32_t iterations;
  int32_t present;
} vibeqc_hf_warm_state;

/** Query with both buffers null to obtain counts, then copy into owned buffers.
 * A missing retained seed returns present=0 and zero counts. */
VIBEQC_API vibeqc_status vibeqc_batch_get_hf_warm_state(const vibeqc_batch* batch, uint32_t index,
                                                        vibeqc_hf_warm_state* state);

/** Atomically import input-ordered seeds; present=0 preserves a neighbor.
 * The caller MUST verify source/target method, ordered nuclei, basis/AO, core,
 * spin, and provider identities before calling this low-level buffer API.
 * Native validation checks shape, finiteness, Hermiticity and source-metric
 * electron/spin occupations before mutation. Target execution normalizes the
 * warm guess in its current metric and recomputes SCF convergence normally.
 * Requires warm starts enabled; no runtime objects or convergence flags load.
 */
VIBEQC_API vibeqc_status vibeqc_batch_restore_hf_warm_states(vibeqc_batch* batch,
                                                             const vibeqc_hf_warm_state* states,
                                                             uint32_t count);

/** Discard all retained per-system converged-density warm starts. */
VIBEQC_API vibeqc_status vibeqc_batch_clear_warm_starts(vibeqc_batch* batch);

/** Input-ordered SCF diagnostics for the latest completed item evaluation.
 * Returns NOT_IMPLEMENTED before execution, after a rejected/failed item, or
 * when the method does not report a physical residual. A null out queries
 * availability. Every replay invalidates all prior records before validation.
 * This additive query preserves the legacy batch result array's exact stride.
 */
VIBEQC_API vibeqc_status vibeqc_batch_get_scf_diagnostic(const vibeqc_batch* batch, uint32_t index,
                                                         vibeqc_scf_diagnostic* out);

/** Input-ordered correlated-method diagnostic. The caller-size compatibility
 * contract matches vibeqc_calculation_get_correlation_diagnostic. A rejected
 * or backend-failed item has no record; a normal NOT_CONVERGED item may retain
 * its last finite correlation state and physical residual diagnostics. */
VIBEQC_API vibeqc_status vibeqc_batch_get_correlation_diagnostic(
    const vibeqc_batch* batch, uint32_t index, vibeqc_correlation_diagnostic* diagnostic);

/** Input-ordered counterpart of vibeqc_calculation_get_ks_diagnostic. Invalid
 * or numerically failed items have no record; valid nonconverged items retain
 * their actual history. Every replay invalidates records from its predecessor. */
VIBEQC_API vibeqc_status vibeqc_batch_get_ks_diagnostic(const vibeqc_batch* batch, uint32_t index,
                                                        vibeqc_ks_diagnostic* out,
                                                        vibeqc_ks_iteration* history,
                                                        uint32_t history_capacity);

/** Input-ordered cumulative CUDA KS transport. Geometry rebuilds retain the
 * retired owner's counters and add the replacement owner's setup. */
VIBEQC_API vibeqc_status vibeqc_batch_get_ks_transport_diagnostic(
    const vibeqc_batch* batch, uint32_t index, vibeqc_ks_transport_diagnostic* out);

/**
 * Enable or disable replacement of retained warm-start densities.
 *
 * Passing zero freezes the current snapshots so every later execution starts
 * from the same per-system dm0. Passing one restores the default behavior in
 * which each successful execution advances its retained density. Existing
 * snapshots are neither cleared nor created by this call.
 */
VIBEQC_API vibeqc_status vibeqc_batch_set_warm_start_updates(vibeqc_batch* batch, int32_t enabled);

/**
 * Execute all systems with failure isolation. A successful function return
 * means the batch was structurally valid; inspect each result.status for its
 * scientific outcome. `inputs` may be NULL with input_count=0 to reuse all
 * prepared geometries, otherwise it must contain one descriptor per system.
 * When every result has forces=NULL and force_count=0, execute energy only.
 * If any item requests forces, retain the whole-fleet force schedule and copy
 * only requested outputs. Output selection applies to this replay alone.
 */
VIBEQC_API vibeqc_status vibeqc_batch_execute(vibeqc_batch* batch,
                                              const vibeqc_batch_input_descriptor* inputs,
                                              uint32_t input_count,
                                              vibeqc_batch_item_result_descriptor* results,
                                              uint32_t result_count);

/** Canonical compact-table identities compiled into the D3 production owner. */
VIBEQC_API const char* vibeqc_d3_table_sha256(void);
VIBEQC_API const char* vibeqc_d3_radii_sha256(void);

/**
 * Prepare a standalone two-body D3(BJ) ragged fleet.
 *
 * The owner copies atomic numbers and prepared geometries. maximum_bytes bounds
 * the plan plus worst-case execution staging and, on CUDA, device ownership.
 */
VIBEQC_API vibeqc_status vibeqc_d3_batch_prepare(vibeqc_context* context,
                                                 const vibeqc_d3_system_descriptor* systems,
                                                 uint32_t system_count,
                                                 const vibeqc_d3_bj_descriptor* model,
                                                 vibeqc_d3_batch** batch);
VIBEQC_API void vibeqc_d3_batch_destroy(vibeqc_d3_batch* batch);
VIBEQC_API vibeqc_status vibeqc_d3_batch_get_diagnostic(const vibeqc_d3_batch* batch,
                                                        vibeqc_d3_runtime_diagnostic* diagnostic);

/**
 * Execute correction-only energy / analytic dE/dR with item failure isolation.
 *
 * inputs may be NULL with input_count=0. Otherwise there must be one descriptor
 * per prepared system; coordinates=NULL,count=0 means the original prepared
 * geometry for that member. A successful function return means the replay was
 * structurally valid; inspect each item status for scientific failures.
 */
VIBEQC_API vibeqc_status vibeqc_d3_batch_execute(vibeqc_d3_batch* batch,
                                                 const vibeqc_d3_batch_input_descriptor* inputs,
                                                 uint32_t input_count,
                                                 vibeqc_d3_batch_item_result_descriptor* results,
                                                 uint32_t result_count);

/** Audited identities compiled into the production D4(BJ)-EEQ provider. */
VIBEQC_API const char* vibeqc_d4_table_sha256(void);
VIBEQC_API const char* vibeqc_d4_charge_parameter_sha256(void);
VIBEQC_API const char* vibeqc_d4_derivative_identity(void);
VIBEQC_API const char* vibeqc_d4_provider_identity(void);
VIBEQC_API const char* vibeqc_d4_scheduler_identity(void);

/**
 * Prepare a standalone D4(BJ)-EEQ ragged fleet. Atomic numbers, charges and
 * prepared geometries are copied. maximum_bytes bounds all persistent owner
 * state plus worst-case execution staging/workspace.
 */
VIBEQC_API vibeqc_status vibeqc_d4_batch_prepare(vibeqc_context* context,
                                                 const vibeqc_d4_system_descriptor* systems,
                                                 uint32_t system_count,
                                                 const vibeqc_d4_bj_eeq_descriptor* model,
                                                 vibeqc_d4_batch** batch);
VIBEQC_API void vibeqc_d4_batch_destroy(vibeqc_d4_batch* batch);
VIBEQC_API vibeqc_status vibeqc_d4_batch_get_diagnostic(const vibeqc_d4_batch* batch,
                                                        vibeqc_d4_runtime_diagnostic* diagnostic);

/**
 * Execute complete molecular D4(BJ)-EEQ energy and analytic dE/dR.
 * inputs=NULL,input_count=0 replays prepared geometries. Otherwise each input
 * may independently select prepared or changed coordinates. Item scientific
 * failures are isolated; successful function return means the replay itself
 * was structurally valid.
 */
VIBEQC_API vibeqc_status vibeqc_d4_batch_execute(vibeqc_d4_batch* batch,
                                                 const vibeqc_d4_batch_input_descriptor* inputs,
                                                 uint32_t input_count,
                                                 vibeqc_d4_batch_item_result_descriptor* results,
                                                 uint32_t result_count);

#ifdef __cplusplus
}
#endif

#endif

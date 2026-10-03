#ifndef GENERATIVEQC_GENERATIVEQC_H
#define GENERATIVEQC_GENERATIVEQC_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(GENERATIVEQC_BUILDING_LIBRARY)
#define GENERATIVEQC_API __declspec(dllexport)
#else
#define GENERATIVEQC_API __declspec(dllimport)
#endif
#else
#define GENERATIVEQC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GENERATIVEQC_ABI_VERSION 0u

typedef int32_t generativeqc_status;
enum {
  GENERATIVEQC_STATUS_SUCCESS = 0,
  GENERATIVEQC_STATUS_INVALID_ARGUMENT = 1,
  GENERATIVEQC_STATUS_ABI_MISMATCH = 2,
  GENERATIVEQC_STATUS_NOT_IMPLEMENTED = 3,
  GENERATIVEQC_STATUS_NOT_CONVERGED = 4,
  /** Compatibility name retained for the original HF-only ABI. */
  GENERATIVEQC_STATUS_SCF_NOT_CONVERGED = GENERATIVEQC_STATUS_NOT_CONVERGED,
  GENERATIVEQC_STATUS_NUMERICAL_FAILURE = 5,
  GENERATIVEQC_STATUS_CUDA_ERROR = 6,
  GENERATIVEQC_STATUS_OUT_OF_MEMORY = 7,
  GENERATIVEQC_STATUS_INTERNAL_ERROR = 8,
  /**
   * A run has been prepared but the precision-provenance query ran before a
   * completed execution (or after one that threw). Not an error in the run
   * itself: the \p fp64 record is simply not yet populated. Serializing callers
   * treat this as "no provenance yet" (e.g. Python returns None).
   */
  GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE = 9
};

#include "generativeqc/generated_method_ids.h"

/** Broad algorithm family used for capability discovery and dispatch. */
typedef int32_t generativeqc_method_family;
enum {
  GENERATIVEQC_METHOD_FAMILY_HARTREE_FOCK = 1,
  GENERATIVEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL = 2,
  GENERATIVEQC_METHOD_FAMILY_COUPLED_CLUSTER = 3,
  GENERATIVEQC_METHOD_FAMILY_PERTURBATION = 4,
  GENERATIVEQC_METHOD_FAMILY_SEMIEMPIRICAL = 5
};

typedef uint32_t generativeqc_property_flags;
enum { GENERATIVEQC_PROPERTY_ENERGY = 1u << 0, GENERATIVEQC_PROPERTY_FORCES = 1u << 1 };

typedef int32_t generativeqc_backend;
enum {
  GENERATIVEQC_BACKEND_CPU_REFERENCE = 0,
  GENERATIVEQC_BACKEND_CUDA = 1,
  /** Reserved compatibility tag used by pre-device-resident prototypes. */
  GENERATIVEQC_BACKEND_HYBRID_CUDA = 2
};

/** Density-fitting execution policy for Hartree-Fock methods. */
typedef int32_t generativeqc_density_fitting_mode;
enum {
  /** Preserve the existing direct four-center J/K path (the default). */
  GENERATIVEQC_DENSITY_FITTING_NONE = 0,
  /** Use the independent CPU density-fitting reference implementation. */
  GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE = 1,
  /** Require the accelerator-native density-fitting path. */
  GENERATIVEQC_DENSITY_FITTING_CUDA = 2,
  /** Select CUDA when requested by the context, otherwise use CPU reference. */
  GENERATIVEQC_DENSITY_FITTING_AUTO = 3
};

/**
 * Floating-point execution policy for the selected mean-field method.
 *
 * \p fp64 keeps the current bit-for-bit exact execution. \p auto enables the
 * profile-backed lower-precision contraction route, deriving its tile
 * threshold from the requested tolerances and always finishing with a strict
 * FP64 refinement of the converged density.
 */
typedef int32_t generativeqc_precision_mode;
enum {
  /** Preserve the existing double-precision execution (the default). */
  GENERATIVEQC_PRECISION_FP64 = 0,
  /**
   * Select a lower-precision contraction route only when the accumulated-error
   * budget certifies it for the requested accuracy, and always finish with a
   * strict FP64 target refinement that continues exact iterations until the
   * requested criteria are met. When the budget cannot certify a cutoff the
   * policy keeps the FP64 operator instead of accumulating rounding.
   */
  GENERATIVEQC_PRECISION_AUTO = 1
};

/** Semilocal grid/XC execution schedule for native CUDA KS. */
typedef int32_t generativeqc_xc_execution_schedule;
enum { GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED = 0, GENERATIVEQC_XC_EXECUTION_HOST_UNFUSED = 1 };

typedef int32_t generativeqc_basis_representation;
enum {
  /** CCA-ordered Cartesian functions: 1, 3, 6, and 10 AOs for s-p-d-f. */
  GENERATIVEQC_BASIS_CARTESIAN = 0,
  /** Real spherical functions in PySCF/libcint order: 1, 3, 5, and 7 AOs. */
  GENERATIVEQC_BASIS_SPHERICAL = 1
};

/**
 * Setup-time CUDA DF metric and value/J/K plan evidence; peaks are estimates.
 *
 * Plan-slot order: system_index is the original input, bucket_id its owning bucket.
 * Peaks exclude generated-force staging and opaque provider allocations. Whole-HF
 * resource observations separately account for tracked response allocations.
 */
typedef struct generativeqc_density_fitting_metric_diagnostic {
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
} generativeqc_density_fitting_metric_diagnostic;

typedef struct generativeqc_context generativeqc_context;
typedef struct generativeqc_system generativeqc_system;
typedef struct generativeqc_calculation generativeqc_calculation;
typedef struct generativeqc_batch generativeqc_batch;

typedef struct generativeqc_d3_batch generativeqc_d3_batch;
typedef struct generativeqc_d4_batch generativeqc_d4_batch;
typedef struct generativeqc_nonlocal_plan generativeqc_nonlocal_plan;

typedef int32_t generativeqc_d3_damping;
enum { GENERATIVEQC_D3_DAMPING_BJ = 1, GENERATIVEQC_D3_DAMPING_ZERO = 2 };

/** Geometry-only D3 system. Coordinates are Bohr and copied at prepare. */
typedef struct generativeqc_d3_system_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const int32_t* atomic_numbers;
  const double* coordinates;
  uint32_t atom_count;
} generativeqc_d3_system_descriptor;

/**
 * Explicit molecular D3 model.
 *
 * Callers must provide the current complete descriptor layout. BJ, zero damping,
 * and BJ+ATM are accepted only by their separately qualified capability paths.
 * Zero-damping+ATM is deliberately unsupported.
 */
typedef struct generativeqc_d3_bj_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_d3_damping damping;
  double s6;
  double s8;
  double a1;
  double a2;
  double s9;
  double cn_cutoff;
  double pair_cutoff;
  double pair_switch_width;
  uint64_t maximum_bytes;
  /** Zero-damping radius scalings; ignored for BJ. */
  double rs6;
  double rs8;
  double alp;
  /** BJ-ATM cutoff/switch in bohr; ignored when s9 == 0. */
  double atm_cutoff;
  double atm_switch_width;
} generativeqc_d3_bj_descriptor;

/** Optional changed geometry for one prepared D3 batch member. */
typedef struct generativeqc_d3_batch_input_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const double* coordinates;
  uint32_t coordinate_count;
} generativeqc_d3_batch_input_descriptor;

/** Caller-owned result buffer; gradient is dE/dR (not force). */
typedef struct generativeqc_d3_batch_item_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_status status;
  double energy;
  double* gradient;
  uint32_t gradient_count;
  generativeqc_backend executed_backend;
} generativeqc_d3_batch_item_result_descriptor;

/** Bounded production owner diagnostics. */
typedef struct generativeqc_d3_runtime_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_backend backend;
  uint64_t plan_host_bytes;
  uint64_t execution_host_bytes;
  uint64_t device_bytes;
  uint64_t table_bytes;
  uint64_t workspace_bytes;
  uint64_t maximum_bytes;
  uint64_t total_atoms;
  uint32_t system_count;
  uint32_t maximum_atoms;
} generativeqc_d3_runtime_diagnostic;

typedef int32_t generativeqc_d4_profile;
enum { GENERATIVEQC_D4_PROFILE_STANDARD_EEQ = 1, GENERATIVEQC_D4_PROFILE_R2SCAN3C_EEQ = 2 };

/** Molecular nonperiodic D4(BJ)-EEQ system. Coordinates are Bohr. */
typedef struct generativeqc_d4_system_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const int32_t* atomic_numbers;
  const double* coordinates;
  uint32_t atom_count;
  double total_charge;
} generativeqc_d4_system_descriptor;

/**
 * Explicit D4(BJ)-EEQ model identity. There is deliberately no generic GFN2
 * default: callers must provide the named-method parameters and EEQ profile.
 */
typedef struct generativeqc_d4_bj_eeq_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_d4_profile profile;
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
} generativeqc_d4_bj_eeq_descriptor;

typedef struct generativeqc_d4_batch_input_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const double* coordinates;
  uint32_t coordinate_count;
} generativeqc_d4_batch_input_descriptor;

/** Caller-owned D4 result buffers; gradient is dE/dR, charges are EEQ2019. */
typedef struct generativeqc_d4_batch_item_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_status status;
  double energy;
  double two_body_energy;
  double atm_energy;
  double* gradient;
  uint32_t gradient_count;
  double* charges;
  uint32_t charge_count;
  generativeqc_backend executed_backend;
} generativeqc_d4_batch_item_result_descriptor;

/** Bounded production owner and replay/scheduling evidence. */
typedef struct generativeqc_d4_runtime_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_backend backend;
  generativeqc_d4_profile profile;
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
} generativeqc_d4_runtime_diagnostic;

/** Fixed-grid VV10/rVV10 kernel variant for the native nonlocal plan. */
typedef int32_t generativeqc_nonlocal_variant;
enum { GENERATIVEQC_NONLOCAL_VV10 = 1, GENERATIVEQC_NONLOCAL_RVV10 = 2 };

/**
 * Bounded fixed-grid nonlocal-correlation pair plan.
 *
 * Coordinates are Bohr; density and its Cartesian gradient use atomic units.
 * This low-level primitive does not imply a public KS/method capability.
 */
typedef struct generativeqc_nonlocal_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_nonlocal_variant variant;
  double b;
  double c;
  double coefficient;
  uint32_t point_count;
  uint32_t tile_points;
  uint64_t maximum_bytes;
} generativeqc_nonlocal_descriptor;

/** Caller-owned fixed-grid inputs for one prepared VV10/rVV10 execution. */
typedef struct generativeqc_nonlocal_input_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const double* coordinates;
  uint32_t coordinate_count;
  const double* weights;
  uint32_t weight_count;
  const double* density;
  uint32_t density_count;
  const double* density_gradient;
  uint32_t density_gradient_count;
} generativeqc_nonlocal_input_descriptor;

/**
 * Optional fixed-grid derivative outputs. vrho/vsigma are dE/d(rho,sigma)
 * before quadrature weights. point_derivative is the explicit pair-distance
 * derivative at fixed density features; weight_derivative differentiates both
 * quadrature legs. Null pointer plus zero count disables an output family.
 */
typedef struct generativeqc_nonlocal_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  double energy;
  double* vrho;
  uint32_t vrho_count;
  double* vsigma;
  uint32_t vsigma_count;
  double* point_derivative;
  uint32_t point_derivative_count;
  double* weight_derivative;
  uint32_t weight_derivative_count;
  generativeqc_backend executed_backend;
} generativeqc_nonlocal_result_descriptor;

/** Exact owned-capacity and pair-work diagnostics for the prepared plan. */
typedef struct generativeqc_nonlocal_runtime_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_backend backend;
  uint64_t workspace_bytes;
  uint64_t host_workspace_bytes;
  uint64_t device_workspace_bytes;
  uint64_t maximum_bytes;
  uint64_t pair_evaluations;
  uint64_t tiles;
  uint32_t point_count;
  uint32_t tile_points;
} generativeqc_nonlocal_runtime_diagnostic;

typedef uint32_t generativeqc_batch_flags;
enum {
  /** Retain each converged AO density for the next execution of the plan. */
  GENERATIVEQC_BATCH_ENABLE_WARM_STARTS = 1u << 0,
  /**
   * Collect the final density-screened direct-J/K shell-class work profile.
   *
   * This diagnostic adds one untimed-by-default CUDA reduction after the
   * final compaction pass. Leave it disabled for production timing runs.
   */
  GENERATIVEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING = 1u << 1,
  /**
   * Collect one device-timed record for every SCF iteration eigensolve.
   *
   * The instrumentation is inserted into the device-tail CUDA Graph and is
   * intended only for diagnosing divergent fleets. Leave it disabled during
   * production endpoint timing.
   */
  GENERATIVEQC_BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING = 1u << 2
};

/** Number of pair/pair-exchange-reduced s/p/d/f quartet shell classes. */
#define GENERATIVEQC_DIRECT_SHELL_CLASS_COUNT 55u

/** Work retained for one shell class after final-density screening. */
typedef struct generativeqc_shell_class_profile_entry {
  uint64_t shell_quartets;
  uint64_t tiles;
  uint64_t ao_quartets;
  uint64_t primitive_quartets;
} generativeqc_shell_class_profile_entry;

#define GENERATIVEQC_PPPS_PROFILE_BLOCK_SIZE_COUNT 4u
#define GENERATIVEQC_PPPS_PROFILE_ORIENTATION_COUNT 2u
#define GENERATIVEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT 65u

/**
 * Final-density statistics for the exact resident PPPS production queue.
 *
 * Block-size arrays are ordered as 32, 64, 128, and 256 threads. Orientation
 * arrays are ordered as 1110 then 1011. Primitive-pair buckets 0..63 are
 * exact; bucket 64 contains 64 or more primitive pairs.
 */
typedef struct generativeqc_ppps_queue_profile {
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
  double lane_efficiency[GENERATIVEQC_PPPS_PROFILE_BLOCK_SIZE_COUNT];
  double primitive_warp_efficiency;
  double task_tail_imbalance[GENERATIVEQC_PPPS_PROFILE_BLOCK_SIZE_COUNT];
  double primitive_tail_imbalance[GENERATIVEQC_PPPS_PROFILE_BLOCK_SIZE_COUNT];
  uint64_t orientation_tasks[GENERATIVEQC_PPPS_PROFILE_ORIENTATION_COUNT];
  uint64_t orientation_primitive_work[GENERATIVEQC_PPPS_PROFILE_ORIENTATION_COUNT];
  uint64_t bra_primitive_tasks[GENERATIVEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
  uint64_t bra_primitive_work[GENERATIVEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
  uint64_t ket_primitive_tasks[GENERATIVEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
  uint64_t ket_primitive_work[GENERATIVEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
} generativeqc_ppps_queue_profile;

typedef int32_t generativeqc_eigensolver_family;
enum {
  GENERATIVEQC_EIGENSOLVER_SMALL_NATIVE = 0,
  GENERATIVEQC_EIGENSOLVER_JACOBI_BATCHED = 1,
  GENERATIVEQC_EIGENSOLVER_XSYEV_BATCHED = 2,
  GENERATIVEQC_EIGENSOLVER_GRAPH_NATIVE = 3
};

typedef int32_t generativeqc_eigensolver_selection_source;
enum {
  GENERATIVEQC_EIGENSOLVER_SELECTION_DIMENSION_POLICY = 0,
  GENERATIVEQC_EIGENSOLVER_SELECTION_EXACT_PROBE = 1,
  GENERATIVEQC_EIGENSOLVER_SELECTION_EXACT_PROBE_FALLBACK = 2,
  /** Explicit benchmark-only override of the Graph eigensolver family. */
  GENERATIVEQC_EIGENSOLVER_SELECTION_BENCHMARK_OVERRIDE = 3
};

typedef int32_t generativeqc_xsyev_eligibility_reason;
enum {
  GENERATIVEQC_XSYEV_ELIGIBLE = 0,
  GENERATIVEQC_XSYEV_ZERO_DIMENSION = 1,
  GENERATIVEQC_XSYEV_INVALID_LEADING_DIMENSION = 2,
  GENERATIVEQC_XSYEV_DOCUMENTED_DIMENSION_LIMIT = 3,
  GENERATIVEQC_XSYEV_SOLVER_BATCH_LIMIT = 4,
  GENERATIVEQC_XSYEV_DOCUMENTED_PRODUCT_LIMIT = 5
};

typedef int32_t generativeqc_xsyev_graph_probe_stage;
enum {
  GENERATIVEQC_XSYEV_PROBE_NONE = 0,
  GENERATIVEQC_XSYEV_PROBE_API_ELIGIBILITY = 1,
  GENERATIVEQC_XSYEV_PROBE_SELECT_DEVICE = 2,
  GENERATIVEQC_XSYEV_PROBE_DEVICE_IDENTITY = 3,
  GENERATIVEQC_XSYEV_PROBE_CREATE_STREAM = 4,
  GENERATIVEQC_XSYEV_PROBE_CREATE_SOLVER = 5,
  GENERATIVEQC_XSYEV_PROBE_CREATE_PARAMETERS = 6,
  GENERATIVEQC_XSYEV_PROBE_ALLOCATE_DATA = 7,
  GENERATIVEQC_XSYEV_PROBE_QUERY_WORKSPACE = 8,
  GENERATIVEQC_XSYEV_PROBE_INSUFFICIENT_DEVICE_MEMORY = 9,
  GENERATIVEQC_XSYEV_PROBE_ALLOCATE_WORKSPACE = 10,
  GENERATIVEQC_XSYEV_PROBE_ORDINARY_EXECUTION = 11,
  GENERATIVEQC_XSYEV_PROBE_ORDINARY_VALIDATION = 12,
  GENERATIVEQC_XSYEV_PROBE_BEGIN_CAPTURE = 13,
  GENERATIVEQC_XSYEV_PROBE_CAPTURE_PROVIDER = 14,
  GENERATIVEQC_XSYEV_PROBE_END_CAPTURE = 15,
  GENERATIVEQC_XSYEV_PROBE_INSTANTIATE_DEVICE_LAUNCH_GRAPH = 16,
  GENERATIVEQC_XSYEV_PROBE_UPLOAD_GRAPH = 17,
  GENERATIVEQC_XSYEV_PROBE_HOST_GRAPH_REPLAY = 18,
  GENERATIVEQC_XSYEV_PROBE_HOST_GRAPH_VALIDATION = 19,
  GENERATIVEQC_XSYEV_PROBE_DEVICE_TAIL_REPLAY = 20,
  GENERATIVEQC_XSYEV_PROBE_DEVICE_TAIL_VALIDATION = 21
};

/** Exact setup-time eigensolver selection evidence for one workload bucket. */
typedef struct generativeqc_eigensolver_diagnostic {
  uint32_t bucket_id;
  generativeqc_eigensolver_family ordinary_family;
  generativeqc_eigensolver_family graph_family;
  generativeqc_eigensolver_selection_source selection_source;
  uint64_t matrix_dimension;
  uint64_t physical_system_count;
  uint64_t solver_batch_count;
  int32_t api_eligible;
  generativeqc_xsyev_eligibility_reason api_reason;
  uint64_t matrix_batch_product;
  generativeqc_xsyev_graph_probe_stage probe_failure_stage;
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
} generativeqc_eigensolver_diagnostic;

typedef uint32_t generativeqc_eigensolver_inactive_touch_flags;
enum {
  /** An inactive matrix was copied before the provider call. */
  GENERATIVEQC_EIGENSOLVER_INACTIVE_TOUCH_COPY = 1u << 0,
  /** cuBLAS transformed an inactive matrix before the provider call. */
  GENERATIVEQC_EIGENSOLVER_INACTIVE_TOUCH_CUBLAS_TRANSFORM = 1u << 1,
  /** The provider input was replaced with a finite identity matrix. */
  GENERATIVEQC_EIGENSOLVER_INACTIVE_TOUCH_IDENTITY_SANITIZE = 1u << 2
};

/** Device-timed evidence for one eigensolve in the device-tail SCF loop. */
typedef struct generativeqc_inactive_eigensolver_profile_entry {
  uint32_t bucket_id;
  uint32_t iteration;
  generativeqc_eigensolver_family family;
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
  generativeqc_eigensolver_inactive_touch_flags inactive_touch_flags;
  int32_t provider_invoked;
} generativeqc_inactive_eigensolver_profile_entry;

typedef struct generativeqc_context_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  int32_t device_id;
  generativeqc_backend backend;
} generativeqc_context_descriptor;

typedef struct generativeqc_atom {
  int32_t atomic_number;
  double x;
  double y;
  double z;
} generativeqc_atom;

typedef struct generativeqc_primitive {
  double exponent;
  double coefficient;
} generativeqc_primitive;

typedef struct generativeqc_shell {
  uint32_t atom_index;
  uint32_t angular_momentum;
  uint32_t primitive_offset;
  uint32_t primitive_count;
} generativeqc_shell;

typedef struct generativeqc_system_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const generativeqc_atom* atoms;
  uint32_t atom_count;
  const generativeqc_shell* shells;
  uint32_t shell_count;
  const generativeqc_primitive* primitives;
  uint32_t primitive_count;
  int32_t charge;
  uint32_t multiplicity;
  /** Optional in older ABI-0 descriptors; absent fields imply Cartesian. */
  generativeqc_basis_representation basis_representation;
} generativeqc_system_descriptor;

/** One semilocal XC component from the compiler-owned KS execution plan. */
typedef struct generativeqc_ks_semilocal_component {
  /** Stable compiler component identifier such as GGA_X_PBE. */
  const char* component_id;
  /** Physical coefficient carried by MethodIR. */
  double coefficient;
} generativeqc_ks_semilocal_component;

/** Exact-exchange operator carried by one KS execution-plan contribution. */
typedef int32_t generativeqc_ks_exchange_operator;
enum {
  GENERATIVEQC_KS_EXCHANGE_FULL_RANGE = 1,
  GENERATIVEQC_KS_EXCHANGE_SHORT_RANGE = 2,
  GENERATIVEQC_KS_EXCHANGE_LONG_RANGE = 3,
};

typedef struct generativeqc_ks_exchange_term {
  generativeqc_ks_exchange_operator operator_kind;
  /** Physical exact-exchange fraction. */
  double coefficient;
  /** Range parameter in bohr^-1; zero for full-range exchange. */
  double omega;
  /** Spin-convention-resolved coefficient applied to the native K build. */
  double fock_coefficient;
} generativeqc_ks_exchange_term;

/** Current compiler-to-native KS execution plan.
 *
 * This is intentionally a breaking, single-layout ABI: MethodIR is lowered once
 * into semantic primitive arrays instead of accumulating method-specific v2/v3/...
 * suffixes. Native preparation copies every pointee before returning.
 */
typedef struct generativeqc_ks_options {
  uint32_t struct_size;
  uint32_t abi_version;
  /** Exact compiler-owned numerical-domain identity. */
  const char* scf_domain;
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
  /** Radii [0..118] in Bohr, indexed by atomic number; slot zero is unused. */
  const double* element_radii;
  uint32_t element_radius_count;
  generativeqc_xc_execution_schedule xc_execution_schedule;
  /** 1 for RKS and 2 for UKS, copied directly from MethodIR. */
  uint32_t spin_channels;
  const generativeqc_ks_semilocal_component* semilocal_components;
  uint32_t semilocal_component_count;
  /** Semilocal range parameter in bohr^-1, or zero when absent. */
  double semilocal_range_omega;
  const generativeqc_ks_exchange_term* exchange_terms;
  uint32_t exchange_term_count;
  /** 0/1 optional MethodIR NonlocalCorrelation contribution. */
  uint32_t has_nonlocal_correlation;
  generativeqc_nonlocal_variant nonlocal_variant;
  double nonlocal_b;
  double nonlocal_c;
  double nonlocal_coefficient;
  uint64_t nonlocal_maximum_bytes;
} generativeqc_ks_options;

/** Current KS execution-plan ABI schema. No legacy prefix layouts are accepted. */
GENERATIVEQC_API uint32_t generativeqc_ks_options_version(void);

/** Explicit preliminary SCF; never an automatic/default selection. */
typedef int32_t generativeqc_initial_guess_kind;
enum { GENERATIVEQC_INITIAL_GUESS_HF = 1, GENERATIVEQC_INITIAL_GUESS_LDA = 2 };

/** CPU FP64, all-electron, restricted exact energy endpoints only. Zero-valued
 * controls select 32 iterations, DIIS 8, tolerances 1e-6/1e-4 and 256 MiB.
 * LDA uses an independent v1 coarse grid (defaults 8/6/12); HF has no grid.
 * maximum_numeric_bytes bounds preliminary numeric payloads, excluding object
 * headers/allocator/runtime overhead and the retained target owner. Compose
 * both owners through ResourceBudget for a whole-endpoint host capacity bound.
 */
typedef struct generativeqc_initial_guess_options {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_initial_guess_kind kind;
  uint32_t max_iterations;
  uint32_t diis_history;
  double energy_tolerance;
  double density_tolerance;
  uint64_t maximum_numeric_bytes;
  uint32_t radial_points;
  uint32_t angular_polar;
  uint32_t angular_azimuth;
} generativeqc_initial_guess_options;

/** 0 disabled, 1 existing-density bypass, 2 used, 3 preparation failure,
 * 4 preparation budget skipped, 5 seeded target failed and core was retried.
 * Target convergence remains reported by the ordinary result descriptor. */
typedef struct generativeqc_initial_guess_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t requested_kind;
  uint32_t outcome;
  uint32_t preliminary_iterations;
  uint64_t preliminary_fock_builds;
  uint32_t target_attempts;
  uint32_t discarded_target_iterations;
  uint64_t discarded_target_fock_builds;
  uint64_t preparation_numeric_capacity;
  double preparation_seconds;
  /** 0 when an exception prevented counting a discarded attempt completely. */
  uint32_t work_counters_complete;
} generativeqc_initial_guess_diagnostic;

GENERATIVEQC_API uint32_t generativeqc_initial_guess_options_version(void);
GENERATIVEQC_API generativeqc_status generativeqc_calculation_get_initial_guess_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_initial_guess_diagnostic* out);
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_initial_guess_diagnostic(
    const generativeqc_batch* batch, uint32_t index, generativeqc_initial_guess_diagnostic* out);

/** Current method preparation descriptor. Callers must provide this complete layout. */
typedef struct generativeqc_method_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_method method;
  uint32_t max_iterations;
  uint32_t diis_history;
  double energy_tolerance;
  double density_tolerance;
  double screening_tolerance;
  /** Density-fitting execution mode. */
  generativeqc_density_fitting_mode density_fitting_mode;
  /** Optional prepared system carrying the auxiliary-basis shell topology. */
  const generativeqc_system* density_fitting_auxiliary_basis;
  /** Relative eigenvalue threshold used for the auxiliary metric. */
  double density_fitting_relative_threshold;
  /** Planner budget in bytes. Positive values are hard upper bounds; zero
   * selects the workload/device-aware resource policy. */
  uint64_t density_fitting_memory_budget_bytes;
  /** Floating-point execution policy. FP64 keeps the strict double-precision
   * route; \p auto enables the qualified lower-precision route. */
  generativeqc_precision_mode precision_mode;
  /** Optional combined numeric capacity for correlated reference/energy phases.
   * Zero selects 256 MiB; this is not a process-RSS or CUDA-context bound. */
  uint64_t correlation_memory_budget_bytes;
  /** Positive MP2 absolute denominator threshold in Hartree; zero uses 1e-10. */
  double mp2_denominator_threshold;
  /** Optional KS snapshot. NULL preserves the method's built-in model.
   * The descriptor and pointees need only outlive the prepare call. */
  const generativeqc_ks_options* ks_options;
  /** RCCSD controls. Zero selects the documented default except
   * ccsd_diis_history=0, which explicitly disables DIIS. */
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
  /** Optional execution-only cold-start policy. Pointee is copied at preparation.
   * Existing explicit/imported/retained density always takes precedence. */
  const generativeqc_initial_guess_options* initial_guess;
} generativeqc_method_descriptor;

/**
 * Read-only record of how the requested precision policy resolved. Populated by
 * \p generativeqc_calculation_get_precision_provenance after a prepared run; callers
 * that predate this field never see it because the out-parameter is optional.
 */
typedef struct generativeqc_precision_provenance {
  uint32_t struct_size;
  uint32_t abi_version;
  /** Policy version the resolver honored. */
  uint32_t policy_version;
  /** Requested mode (\p generativeqc_precision_mode). */
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
  /** Mixed-stage Fock/operator applications actually executed for this item. */
  uint64_t mixed_stage_fock_builds;
  /** Strict-FP64 SCF-stage Fock/operator applications actually executed. */
  uint64_t strict_stage_fock_builds;
  /** Additional strict physical-Fock builds after SCF convergence. */
  uint64_t post_scf_fock_builds;
  /** Whole-execution provider retries before the returned attempt. */
  uint64_t execution_retries;
  /** Certified mixed-capable work census used by per-item admission. */
  uint64_t mixed_admission_census;
  /** Exact final physical-residual audits executed for this item. */
  uint64_t final_residual_audits;
  /** Final-Fock operator applications skipped by retained-state reuse. */
  uint64_t skipped_final_fock_builds;
  /** Nonzero only when the operator-work counters above are fully instrumented.
   * Numerical failures can leave partially executed stages uncounted; their
   * counters are not certified by this flag. */
  uint32_t operator_work_counters_valid;
} generativeqc_precision_provenance;

/** Read-only #990 incremental Direct-J/K work record for one completed SCF item. */
typedef struct generativeqc_incremental_direct_jk_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t policy_version;
  int32_t requested;
  int32_t active;
  int32_t quartet_work_counters_valid;
  uint64_t anchor_full_builds;
  uint64_t delta_builds;
  uint64_t periodic_rebuilds;
  uint64_t bypass_full_builds;
  uint64_t post_scf_full_builds;
  uint64_t anchor_updates;
  double max_abs_delta_density;
  uint64_t full_candidate_shell_quartets;
  uint64_t full_rejected_shell_quartets;
  uint64_t full_admitted_shell_quartets;
  uint64_t full_admitted_quartet_tiles;
  uint64_t delta_candidate_shell_quartets;
  uint64_t delta_rejected_shell_quartets;
  uint64_t delta_admitted_shell_quartets;
  uint64_t delta_admitted_quartet_tiles;
} generativeqc_incremental_direct_jk_diagnostic;

/** Version of the separate, variable-length precision-work query. */
#define GENERATIVEQC_PRECISION_WORK_DETAIL_VERSION 1u

typedef int32_t generativeqc_precision_work_event_kind;
enum {
  GENERATIVEQC_PRECISION_EVENT_MIXED_FOCK = 1,
  GENERATIVEQC_PRECISION_EVENT_STRICT_FOCK = 2,
  GENERATIVEQC_PRECISION_EVENT_POST_SCF_FOCK = 3,
  GENERATIVEQC_PRECISION_EVENT_FINAL_AUDIT = 4,
  GENERATIVEQC_PRECISION_EVENT_RETRY = 5,
  GENERATIVEQC_PRECISION_EVENT_FALLBACK = 6,
  GENERATIVEQC_PRECISION_EVENT_CONVERSION = 7
};

typedef int32_t generativeqc_precision_work_phase;
enum {
  GENERATIVEQC_PRECISION_PHASE_SCF = 1,
  GENERATIVEQC_PRECISION_PHASE_REFINEMENT = 2,
  GENERATIVEQC_PRECISION_PHASE_FINALIZATION = 3,
  GENERATIVEQC_PRECISION_PHASE_RETRY = 4
};

typedef int32_t generativeqc_precision_operator_kind;
enum {
  GENERATIVEQC_PRECISION_OPERATOR_COULOMB_J = 1,
  GENERATIVEQC_PRECISION_OPERATOR_EXCHANGE_K = 2,
  GENERATIVEQC_PRECISION_OPERATOR_XC = 3,
  GENERATIVEQC_PRECISION_OPERATOR_FOCK_ASSEMBLY = 4,
  GENERATIVEQC_PRECISION_OPERATOR_PHYSICAL_RESIDUAL = 5,
  GENERATIVEQC_PRECISION_OPERATOR_EIGENSOLVER = 6,
  GENERATIVEQC_PRECISION_OPERATOR_DENSITY_BUILD = 7,
  GENERATIVEQC_PRECISION_OPERATOR_DIIS = 8,
  GENERATIVEQC_PRECISION_OPERATOR_MATRIX_PRODUCT = 9,
  GENERATIVEQC_PRECISION_OPERATOR_DIAGNOSTICS = 10,
  GENERATIVEQC_PRECISION_OPERATOR_OCCUPATION_STABILIZATION = 11,
  GENERATIVEQC_PRECISION_OPERATOR_COULOMB_RECURRENCE = 12,
  GENERATIVEQC_PRECISION_OPERATOR_EXCHANGE_RECURRENCE = 13
};

typedef int32_t generativeqc_precision_dtype;
enum {
  GENERATIVEQC_PRECISION_DTYPE_UNKNOWN = 0,
  GENERATIVEQC_PRECISION_DTYPE_FP64 = 1,
  GENERATIVEQC_PRECISION_DTYPE_FP32 = 2,
  GENERATIVEQC_PRECISION_DTYPE_TF32 = 3,
  GENERATIVEQC_PRECISION_DTYPE_FP16 = 4,
  GENERATIVEQC_PRECISION_DTYPE_BF16 = 5
};

typedef int32_t generativeqc_precision_arithmetic_mode;
enum {
  GENERATIVEQC_PRECISION_ARITHMETIC_STRICT = 1,
  GENERATIVEQC_PRECISION_ARITHMETIC_MIXED = 2,
  GENERATIVEQC_PRECISION_ARITHMETIC_TF32 = 3,
  GENERATIVEQC_PRECISION_ARITHMETIC_FP16 = 4,
  GENERATIVEQC_PRECISION_ARITHMETIC_BF16 = 5
};

/** Summary for one execution-owned precision-work record.
 *
 * Query first with NULL row arrays, allocate exactly event_count and
 * operator_count initialized descriptors, then query again. Calls must be
 * serialized with execution. Each event is one observed logical call and has
 * an implicit count of one. Operator counts are logical applications, except
 * recurrence records count the actually evaluated AO-ERI values.
 *
 * owner_id/returned_solve_epoch/returned_state_generation identify native
 * state within the current process; they are not portable scientific labels.
 * complete and operator_inventory_complete remain zero for uninstrumented or
 * partial execution. In particular, aggregate counters are never expanded into
 * a synthetic timeline by this query. */
typedef struct generativeqc_precision_work_detail {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t detail_version;
  int32_t complete;
  int32_t operator_inventory_complete;
  uint32_t event_count;
  uint32_t operator_count;
  uint64_t conversion_count;
  uint64_t fallback_count;
  uint64_t owner_id;
  uint64_t returned_solve_epoch;
  uint64_t returned_state_generation;
} generativeqc_precision_work_detail;

typedef struct generativeqc_precision_work_event {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_precision_work_event_kind kind;
  generativeqc_precision_work_phase phase;
  uint64_t sequence;
  uint32_t iteration;
  uint64_t owner_id;
  uint64_t solve_epoch;
  uint64_t state_generation;
} generativeqc_precision_work_event;

typedef struct generativeqc_precision_operator_record {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_precision_operator_kind kind;
  generativeqc_precision_dtype storage_dtype;
  generativeqc_precision_dtype compute_dtype;
  generativeqc_precision_dtype accumulation_dtype;
  generativeqc_precision_dtype reduction_dtype;
  generativeqc_precision_arithmetic_mode arithmetic_mode;
  uint64_t count;
} generativeqc_precision_operator_record;

typedef struct generativeqc_correlation_diagnostic {
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
} generativeqc_correlation_diagnostic;

/** Additive observational RCCSD/RCCSD(T) phase/work record.
 * This descriptor does not alter correlation science or convergence contracts.
 * Nested phases are intentionally not additive: source is contained by provider,
 * while iteration/replay/update/DIIS are contained by solver.
 */
typedef struct generativeqc_cc_performance_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  double reference_seconds;
  double problem_seconds;
  double provider_seconds;
  double source_seconds;
  double solver_seconds;
  double iteration_seconds;
  double replay_seconds;
  double update_seconds;
  double diis_seconds;
  double triples_seconds;
  uint64_t source_scans;
  uint64_t source_reads;
  uint64_t source_values;
  uint64_t transform_fmas;
  uint64_t transform_stages;
  uint64_t mo_blocks;
  uint64_t cuda_transform_calls;
  uint64_t cuda_batch_calls;
  uint64_t iteration_graph_calls;
  uint64_t replay_graph_calls;
  uint64_t update_calls;
  uint64_t generated_error_checks;
  uint64_t diis_gram_calls;
  uint64_t diis_coefficient_calls;
  uint64_t diis_combine_calls;
} generativeqc_cc_performance_diagnostic;

/** Executable capabilities for one method identifier. */
typedef struct generativeqc_method_capabilities_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_method method;
  generativeqc_method_family family;
  generativeqc_property_flags supported_properties;
  int32_t available;
  int32_t supports_batch;
} generativeqc_method_capabilities_descriptor;

typedef struct generativeqc_result_descriptor {
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
  generativeqc_backend executed_backend;
} generativeqc_result_descriptor;

/** Separate SCF measures, queried without extending existing result layouts. */
typedef struct generativeqc_scf_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  double density_rms;
  /** RMS of physical F D S - S D F evaluated for the reported energy.
   * UKS combines the alpha/beta matrix entries in one RMS. */
  double physical_residual_rms;
} generativeqc_scf_diagnostic;

/** A physical KS iteration before any optional final RKS validation rebuild.
 * The first energy_change is +infinity because no preceding energy exists.
 * Density change and residual are maxima of the spin RMS values (RKS has one
 * total-density matrix), distinct from the joined-spin legacy result RMS. */
typedef struct generativeqc_ks_iteration {
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
} generativeqc_ks_iteration;

/** Completed KS state, copied without extending legacy result-array strides.
 * Electron counts are Tr(D_s S), not integrated grid densities. Final terms
 * refer to the returned physical state; xc_energy includes exact exchange for
 * hybrids. CPU RKS's post-loop validation can
 * make them differ from the last iteration. All energies are in Hartree. */
typedef struct generativeqc_ks_diagnostic {
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
} generativeqc_ks_diagnostic;

/** Cumulative transport performed by one prepared CUDA KS owner.
 *
 * Counts are measured at native copy/synchronization sites rather than
 * inferred from SCF iterations. They include immutable setup, explicit seed
 * uploads, scalar iteration records, changed-geometry warm-density exports,
 * discarded warm attempts, and occupation-control proposals. Routine
 * iteration matrices remain resident and do not contribute matrix D2H bytes.
 * CPU and non-KS owners report this diagnostic as unavailable.
 */
typedef struct generativeqc_ks_transport_diagnostic {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t setup_h2d_bytes;
  uint64_t density_h2d_bytes;
  uint64_t scalar_d2h_bytes;
  uint64_t matrix_d2h_bytes;
  uint64_t synchronizations;
  uint64_t iterations;
  uint64_t occupation_stabilized_proposals;
} generativeqc_ks_transport_diagnostic;

/** Optional sampled-AO geometry preparation and actual solve work. Maps are
 * immutable for one prepared geometry. Discovery time/bytes refer to its setup
 * and must not be counted again on warm solves; xc_evaluations is solve-local.
 * Point/AO counts describe ONE complete XC traversal, including every tile.
 * Numeric resource bounds charge full-capacity discovery, never mean AO counts.
 * These counters are evidence, not energy/force accuracy certificates. */
typedef struct generativeqc_ks_ao_selection_diagnostic_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t requested, selected, tiles, empty_tiles, min_active, max_active, active_sum;
  uint64_t discovery_ao_jet_values, point_ao_visits, point_ao_square_sum;
  uint64_t dense_point_ao_square_sum, discovery_d2h_bytes;
  uint64_t reserved_device_bytes, host_peak_bytes, xc_evaluations;
  double cutoff, discovery_seconds;
} generativeqc_ks_ao_selection_diagnostic_v1;

/** Optional per-system coordinates for a prepared ragged batch execution. */
typedef struct generativeqc_batch_input_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  /** Flat xyz coordinates in Bohr, or NULL to use the prepared geometry. */
  const double* coordinates;
  uint32_t coordinate_count;
} generativeqc_batch_input_descriptor;

/** Per-system output. Each item owns an independent status and diagnostics. */
typedef struct generativeqc_batch_item_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  generativeqc_status status;
  double energy;
  double* forces;
  uint32_t force_count;
  uint32_t iterations;
  double energy_change;
  double density_rms;
  int32_t converged;
  generativeqc_backend executed_backend;
  uint32_t bucket_id;
  int32_t warm_start_used;
  int32_t warm_start_fallback;
} generativeqc_batch_item_result_descriptor;

/** Return the ABI version implemented by the loaded shared library. */
GENERATIVEQC_API uint32_t generativeqc_get_abi_version(void);

/** Source/codegen identity, independent of checkout paths and selected kernels. */
GENERATIVEQC_API const char* generativeqc_get_source_identity(void);

/** Hardware and runtime identity for safe user-local schedule reuse.
 * Device ordinals use the CUDA runtime's scheduler-assigned visibility. No
 * probe changes CUDA_VISIBLE_DEVICES. UUID/PCI address are intentionally absent
 * because matching GPUs on different cluster nodes may share a tuned profile.
 */
typedef struct generativeqc_cuda_tuning_device_descriptor {
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
} generativeqc_cuda_tuning_device_descriptor;

/** Probe an allocated GPU; CPU builds return NOT_IMPLEMENTED without probing. */
GENERATIVEQC_API generativeqc_status generativeqc_cuda_tuning_device(
    int32_t device_id, generativeqc_cuda_tuning_device_descriptor* output);

/** Return a stable, process-lifetime error string for a status code. */
GENERATIVEQC_API const char* generativeqc_status_message(generativeqc_status status);

/** Query whether a method is currently executable. */
GENERATIVEQC_API generativeqc_status generativeqc_method_available(generativeqc_method method,
                                                                   int32_t* available);

/** Query method family, properties, and batch support without preparing work. */
GENERATIVEQC_API generativeqc_status generativeqc_method_get_capabilities(
    generativeqc_method method, generativeqc_method_capabilities_descriptor* capabilities);

GENERATIVEQC_API generativeqc_status generativeqc_context_create(
    const generativeqc_context_descriptor* descriptor, generativeqc_context** context);
GENERATIVEQC_API void generativeqc_context_destroy(generativeqc_context* context);

/** Borrow the last native failure detail, valid until the next failing call
 * on this context or its destruction. Empty when no detail has been recorded. */
GENERATIVEQC_API const char* generativeqc_context_get_last_detail(
    const generativeqc_context* context);

/** Backward-compatible alias for generativeqc_context_get_last_detail. */
GENERATIVEQC_API const char* generativeqc_context_last_error(const generativeqc_context* context);

GENERATIVEQC_API generativeqc_status generativeqc_system_create(
    generativeqc_context* context, const generativeqc_system_descriptor* descriptor,
    generativeqc_system** system);
GENERATIVEQC_API void generativeqc_system_destroy(generativeqc_system* system);

/** Scalar Gaussian residual ECP: c r^(power-2) exp(-exponent r^2).
 * channel=-1 is local, 0..3 is a nonlocal projector difference.
 * Core counts are atom-major and must leave positive effective ionic charges.
 * All buffers are copied; existing all-electron system_create ABI is unchanged. */
typedef struct generativeqc_ecp_term {
  uint32_t atom_index;
  int32_t channel;
  uint32_t power;
  double exponent;
  double coefficient;
} generativeqc_ecp_term;
GENERATIVEQC_API generativeqc_status generativeqc_system_create_ecp(
    generativeqc_context* context, const generativeqc_system_descriptor* descriptor,
    const int32_t* core_electrons, const generativeqc_ecp_term* terms, size_t term_count,
    generativeqc_system** system);
/** Part-major [local, nonlocal], each containing value then atom/xyz derivatives.
 * Derivatives are positive energy derivatives, not forces. Output is owned by caller. */
GENERATIVEQC_API generativeqc_status generativeqc_system_ecp_integrals(
    generativeqc_context* context, const generativeqc_system* system, uint32_t radial_points,
    uint32_t polar_points, int32_t derivatives, double* output, size_t output_count);

/** Numeric staging and explicit transfer counters for the generic CUDA gradient.
 * Caller-owned weights/system and pre-existing HF plans are outside this arena. */
typedef struct generativeqc_one_electron_gradient_resources {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t device_bytes;
  uint64_t host_numeric_bytes;
  uint64_t host_to_device_bytes;
  uint64_t device_to_host_bytes;
  uint64_t synchronous_uploads;
  uint64_t stream_synchronizations;
} generativeqc_one_electron_gradient_resources;

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
GENERATIVEQC_API generativeqc_status generativeqc_system_one_electron_gradient_cuda(
    generativeqc_context* context, const generativeqc_system* system, const double* overlap_weights,
    const double* kinetic_weights, const double* attraction_weights, size_t matrix_count,
    unsigned schedule, size_t maximum_bytes, double* gradient, size_t gradient_count,
    generativeqc_one_electron_gradient_resources* resources);

/** Physical Fock builds in the last CPU batch execution, including final
 * rebuilds. A joint UHF alpha/beta J/K evaluation counts once. Returns
 * NOT_IMPLEMENTED for an unexecuted item, CUDA, or an incompletely counted
 * warm-to-cold retry. The result is never inferred from iteration count. */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_last_fock_builds(
    const generativeqc_batch* batch, uint32_t index, uint64_t* builds);

/**
 * Synchronously write the normalized rectangular overlap <target AO|source AO>
 * in row-major order. output_count must equal target_nbf * source_nbf. Both
 * systems may independently select Cartesian/spherical AOs and geometries.
 * This explicit CPU evaluator allocates no ERI or nuclear-derivative tensors.
 * The context supplies error details; its accelerator selection is irrelevant.
 */
GENERATIVEQC_API generativeqc_status generativeqc_system_cross_overlap_cpu(
    generativeqc_context* context, const generativeqc_system* target,
    const generativeqc_system* source, double* output, size_t output_count);

/** Owned numeric staging and explicit transfers for a generic DF gradient.
 * Caller-owned systems/weights and opaque CUDA allocations are excluded. */
typedef struct generativeqc_df_gradient_resources {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t host_bytes, device_bytes, host_to_device_bytes, device_to_host_bytes;
  uint64_t weight_tile_elements, tiles, uploads, stream_synchronizations;
} generativeqc_df_gradient_resources;

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
GENERATIVEQC_API generativeqc_status generativeqc_system_df_gradient_cuda(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, const double* bar_a, size_t count_a, const double* bar_m,
    size_t count_m, unsigned schedule, size_t maximum_bytes, size_t maximum_tile_elements,
    double* gradient, size_t gradient_count, generativeqc_df_gradient_resources* resources);

GENERATIVEQC_API generativeqc_status generativeqc_calculation_prepare(
    generativeqc_context* context, const generativeqc_system* system,
    const generativeqc_method_descriptor* descriptor, generativeqc_calculation** calculation);
GENERATIVEQC_API void generativeqc_calculation_destroy(generativeqc_calculation* calculation);

/**
 * Execute synchronously. To request forces, the caller owns result->forces and
 * provides at least 3 * atom_count doubles. A NULL pointer with force_count=0
 * requests energy and diagnostics only. Coordinates and all reported
 * derivatives use atomic units (Bohr, Hartree, Hartree/Bohr).
 */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_execute(
    generativeqc_calculation* calculation, generativeqc_result_descriptor* result);

/** Read separate density-update and physical SCF residual measures.
 * Available after a completed supported solve, including NOT_CONVERGED.
 * Returns NOT_IMPLEMENTED before execution, after a backend execution failure, or
 * when the method does not report a physical residual. Such returns leave
 * out untouched. A NULL out is an availability query; otherwise the caller
 * supplies struct_size and abi_version. Currently populated by LDA/PBE KS.
 */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_get_scf_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_scf_diagnostic* out);

/** Query completed KS state and optionally copy its entire iteration history.
 * NULL history/zero capacity queries summary or availability only. To copy
 * history, allocate at least out->history_count initialized descriptors and
 * query again. Every supplied descriptor must have its current size/ABI.
 * Validation failures leave all outputs untouched. NOT_IMPLEMENTED means no
 * completed KS record: unsupported method, not executed, or failed evaluation.
 * Caller serializes all execution/query calls on the prepared owner. */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_get_ks_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_ks_diagnostic* out,
    generativeqc_ks_iteration* history, uint32_t history_capacity);

/** Query cumulative prepared-owner CUDA KS transport. Available immediately
 * after CUDA KS preparation so callers can separate setup from phase deltas. */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_get_ks_transport_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_ks_transport_diagnostic* out);

/**
 * Read the precision policy that resolved for a prepared run. Both the
 * availability query (a NULL \p out) and the copy-out are gated on whether a
 * completed execution has populated the record:
 *
 * - Before any execution, and after an execution that threw, the query returns
 *   \p GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE (and leaves \p out untouched).
 * - After a normal execution return (converged or not) the resolved record is
 *   copied into \p out and SUCCESS is returned.
 *
 * The out-parameter must carry the current complete descriptor size and
 * abi_version. A NULL \p out is a cheap availability probe that never writes.
 */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_get_precision_provenance(
    const generativeqc_calculation* calculation, generativeqc_precision_provenance* out);
/** Read #990 incremental Direct-J/K work from the same completed calculation. */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_get_incremental_direct_jk_diagnostic(
    const generativeqc_calculation* calculation,
    generativeqc_incremental_direct_jk_diagnostic* out);
/** Query ordered precision work without changing the aggregate descriptor.
 * Unsupported detail versions return NOT_IMPLEMENTED. Too-small row buffers
 * return INVALID_ARGUMENT without modifying any output descriptor. */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_get_precision_work(
    const generativeqc_calculation* calculation, uint32_t detail_version,
    generativeqc_precision_work_detail* out, generativeqc_precision_work_event* events,
    uint32_t event_capacity, generativeqc_precision_operator_record* operators,
    uint32_t operator_capacity);
/** Most recent successful correlated execution; failure/absence is explicit. */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_get_correlation_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_correlation_diagnostic* diagnostic);
/** Most recent RCCSD/RCCSD(T) phase/work record; NULL out probes availability. */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_get_cc_performance_diagnostic(
    const generativeqc_calculation* calculation, generativeqc_cc_performance_diagnostic* out);

/**
 * Read one batch item's precision record by its original input index.
 * The descriptor and NULL availability probe follow the single-calculation
 * query. Records are cleared on each execution and populated independently for
 * SUCCESS/NOT_CONVERGED items; rejected or throwing items return
 * GENERATIVEQC_STATUS_PRECISION_UNAVAILABLE without writing to out. An out-of-range
 * index returns GENERATIVEQC_STATUS_INVALID_ARGUMENT.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_precision_provenance(
    const generativeqc_batch* batch, uint32_t index, generativeqc_precision_provenance* out);
/** Input-indexed #990 incremental Direct-J/K work record. */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_incremental_direct_jk_diagnostic(
    const generativeqc_batch* batch, uint32_t index,
    generativeqc_incremental_direct_jk_diagnostic* out);
/** Original-index batch equivalent of generativeqc_calculation_get_precision_work. */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_precision_work(
    const generativeqc_batch* batch, uint32_t index, uint32_t detail_version,
    generativeqc_precision_work_detail* out, generativeqc_precision_work_event* events,
    uint32_t event_capacity, generativeqc_precision_operator_record* operators,
    uint32_t operator_capacity);

/**
 * Prepare a persistent ragged fleet plan. Systems may have different atom,
 * shell, primitive, and AO counts; no global padding is introduced.
 */
GENERATIVEQC_API generativeqc_status
generativeqc_batch_prepare(generativeqc_context* context, const generativeqc_system* const* systems,
                           uint32_t system_count, const generativeqc_method_descriptor* descriptor,
                           generativeqc_batch_flags flags, generativeqc_batch** batch);

GENERATIVEQC_API void generativeqc_batch_destroy(generativeqc_batch* batch);

GENERATIVEQC_API uint32_t generativeqc_batch_get_system_count(const generativeqc_batch* batch);

/**
 * Copy the most recent final-density shell-class profile.
 *
 * The batch must have been prepared with
 * `GENERATIVEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING`, executed through the CUDA direct
 * J/K path, and `entry_count` must be at least
 * `GENERATIVEQC_DIRECT_SHELL_CLASS_COUNT`. Entries use the canonical triangular class
 * encoding documented by GENERATIVEQC's direct shell scheduler.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_last_shell_class_profile(
    const generativeqc_batch* batch, generativeqc_shell_class_profile_entry* entries,
    uint32_t entry_count);

/** Copy PPPS queue statistics from the most recent profiled CUDA execution. */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_last_ppps_queue_profile(
    const generativeqc_batch* batch, generativeqc_ppps_queue_profile* profile);

/**
 * Copy setup-time eigensolver evidence for every bucket in the last execution.
 *
 * Pass `entries = NULL` and `entry_count = 0` to query the required count in
 * `written_count`. A later warm replay returns the cached setup decision and
 * never performs another capability probe.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_last_eigensolver_diagnostics(
    const generativeqc_batch* batch, generativeqc_eigensolver_diagnostic* entries,
    uint32_t entry_count, uint32_t* written_count);

/**
 * Copy CUDA density-fitting metric conditioning and allocation diagnostics
 * from the most recent batch execution. Pass `entries = NULL` and
 * `entry_count = 0` to query the required count in `written_count`.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_last_density_fitting_metric_diagnostics(
    const generativeqc_batch* batch, generativeqc_density_fitting_metric_diagnostic* entries,
    uint32_t entry_count, uint32_t* written_count);

/**
 * Copy per-iteration inactive-eigensolver evidence from the last execution.
 *
 * The batch must opt into
 * `GENERATIVEQC_BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING`. Pass `entries = NULL`
 * and `entry_count = 0` to query the required count. Records are bucket-major
 * and iteration-ordered within each bucket.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_last_inactive_eigensolver_profile(
    const generativeqc_batch* batch, generativeqc_inactive_eigensolver_profile_entry* entries,
    uint32_t entry_count, uint32_t* written_count);

/** Portable scientific buffers for an HF seed. This is a live ABI descriptor,
 * never an on-disk representation. Density is row-major RHF total or UHF alpha
 * then beta. Coordinates are the source geometry, in Bohr and input atom order.
 * Source diagnostics do not establish target convergence.
 */
typedef struct generativeqc_hf_warm_state {
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
} generativeqc_hf_warm_state;

/** Query with both buffers null to obtain counts, then copy into owned buffers.
 * A missing retained seed returns present=0 and zero counts. */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_hf_warm_state(
    const generativeqc_batch* batch, uint32_t index, generativeqc_hf_warm_state* state);

/** Atomically import input-ordered seeds; present=0 preserves a neighbor.
 * The caller MUST verify source/target method, ordered nuclei, basis/AO, core,
 * spin, and provider identities before calling this low-level buffer API.
 * Native validation checks shape, finiteness, Hermiticity and source-metric
 * electron/spin occupations before mutation. Target execution normalizes the
 * warm guess in its current metric and recomputes SCF convergence normally.
 * Requires warm starts enabled; no runtime objects or convergence flags load.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_restore_hf_warm_states(
    generativeqc_batch* batch, const generativeqc_hf_warm_state* states, uint32_t count);

/** Discard all retained per-system converged-density warm starts. */
GENERATIVEQC_API generativeqc_status
generativeqc_batch_clear_warm_starts(generativeqc_batch* batch);

/** Input-ordered SCF diagnostics for the latest completed item evaluation.
 * Returns NOT_IMPLEMENTED before execution, after a rejected/failed item, or
 * when the method does not report a physical residual. A null out queries
 * availability. Every replay invalidates all prior records before validation.
 * This additive query preserves the legacy batch result array's exact stride.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_scf_diagnostic(
    const generativeqc_batch* batch, uint32_t index, generativeqc_scf_diagnostic* out);

/** Input-ordered correlated-method diagnostic. The current complete descriptor
 * contract matches generativeqc_calculation_get_correlation_diagnostic. A rejected
 * or backend-failed item has no record; a normal NOT_CONVERGED item may retain
 * its last finite correlation state and physical residual diagnostics. */
GENERATIVEQC_API generativeqc_status
generativeqc_batch_get_correlation_diagnostic(const generativeqc_batch* batch, uint32_t index,
                                              generativeqc_correlation_diagnostic* diagnostic);
/** Input-ordered counterpart of generativeqc_calculation_get_cc_performance_diagnostic. */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_cc_performance_diagnostic(
    const generativeqc_batch* batch, uint32_t index, generativeqc_cc_performance_diagnostic* out);

/** Input-ordered counterpart of generativeqc_calculation_get_ks_diagnostic. Invalid
 * or numerically failed items have no record; valid nonconverged items retain
 * their actual history. Every replay invalidates records from its predecessor. */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_ks_diagnostic(
    const generativeqc_batch* batch, uint32_t index, generativeqc_ks_diagnostic* out,
    generativeqc_ks_iteration* history, uint32_t history_capacity);

/** Input-ordered cumulative CUDA KS transport. Geometry rebuilds retain the
 * retired owner's counters and add the replacement owner's setup. */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_ks_transport_diagnostic(
    const generativeqc_batch* batch, uint32_t index, generativeqc_ks_transport_diagnostic* out);

/** Host-only query for the current input-ordered KS result. Failed or stale
 * batch items have no record; this never accesses device state or runs AO work. */
GENERATIVEQC_API generativeqc_status generativeqc_batch_get_ks_ao_selection_diagnostic_v1(
    const generativeqc_batch* batch, uint32_t index,
    generativeqc_ks_ao_selection_diagnostic_v1* out);

/**
 * Enable or disable replacement of retained warm-start densities.
 *
 * Passing zero freezes the current snapshots so every later execution starts
 * from the same per-system dm0. Passing one restores the default behavior in
 * which each successful execution advances its retained density. Existing
 * snapshots are neither cleared nor created by this call.
 */
GENERATIVEQC_API generativeqc_status
generativeqc_batch_set_warm_start_updates(generativeqc_batch* batch, int32_t enabled);

/**
 * Execute all systems with failure isolation. A successful function return
 * means the batch was structurally valid; inspect each result.status for its
 * scientific outcome. `inputs` may be NULL with input_count=0 to reuse all
 * prepared geometries, otherwise it must contain one descriptor per system.
 * When every result has forces=NULL and force_count=0, execute energy only.
 * If any item requests forces, retain the whole-fleet force schedule and copy
 * only requested outputs. Output selection applies to this replay alone.
 */
GENERATIVEQC_API generativeqc_status generativeqc_batch_execute(
    generativeqc_batch* batch, const generativeqc_batch_input_descriptor* inputs,
    uint32_t input_count, generativeqc_batch_item_result_descriptor* results,
    uint32_t result_count);

/** Canonical compact-table identities compiled into the D3 production owner. */
GENERATIVEQC_API const char* generativeqc_d3_table_sha256(void);
GENERATIVEQC_API const char* generativeqc_d3_radii_sha256(void);
/** Stable executable-owner identities, separate from method/parameter identity. */
GENERATIVEQC_API const char* generativeqc_d3_provider_identity(void);
GENERATIVEQC_API const char* generativeqc_d3_scheduler_identity(void);
/** Prepared capability identity: d3.bj-two-body, d3.bj-atm, or d3.zero-two-body. */
GENERATIVEQC_API const char* generativeqc_d3_batch_variant_identity(
    const generativeqc_d3_batch* batch);

/**
 * Prepare a standalone explicitly selected D3 ragged fleet.
 *
 * The owner copies atomic numbers and prepared geometries. maximum_bytes bounds
 * the plan plus worst-case execution staging and, on CUDA, device ownership.
 */
GENERATIVEQC_API generativeqc_status generativeqc_d3_batch_prepare(
    generativeqc_context* context, const generativeqc_d3_system_descriptor* systems,
    uint32_t system_count, const generativeqc_d3_bj_descriptor* model,
    generativeqc_d3_batch** batch);
GENERATIVEQC_API void generativeqc_d3_batch_destroy(generativeqc_d3_batch* batch);
GENERATIVEQC_API generativeqc_status generativeqc_d3_batch_get_diagnostic(
    const generativeqc_d3_batch* batch, generativeqc_d3_runtime_diagnostic* diagnostic);

/**
 * Execute correction-only energy / analytic dE/dR with item failure isolation.
 *
 * inputs may be NULL with input_count=0. Otherwise there must be one descriptor
 * per prepared system; coordinates=NULL,count=0 means the original prepared
 * geometry for that member. A successful function return means the replay was
 * structurally valid; inspect each item status for scientific failures.
 */
GENERATIVEQC_API generativeqc_status generativeqc_d3_batch_execute(
    generativeqc_d3_batch* batch, const generativeqc_d3_batch_input_descriptor* inputs,
    uint32_t input_count, generativeqc_d3_batch_item_result_descriptor* results,
    uint32_t result_count);

/**
 * Evaluate the canonical r2SCAN-3c gCP correction on CPU.
 *
 * Coordinates are Bohr and gradient, when supplied, is dE/dR. Passing
 * gradient=NULL,gradient_count=0 requests energy only. The supported element
 * domain is the canonical H-Ar r2SCAN-3c profile.
 */
GENERATIVEQC_API const char* generativeqc_r2scan3c_gcp_provider_identity(void);
GENERATIVEQC_API generativeqc_status generativeqc_r2scan3c_gcp_evaluate(
    const int32_t* atomic_numbers, uint32_t atom_count, const double* coordinates,
    uint32_t coordinate_count, double* energy, double* gradient, uint32_t gradient_count);

/** Audited identities compiled into the production D4(BJ)-EEQ provider. */
GENERATIVEQC_API const char* generativeqc_d4_table_sha256(void);
GENERATIVEQC_API const char* generativeqc_d4_charge_parameter_sha256(void);
GENERATIVEQC_API const char* generativeqc_d4_derivative_identity(void);
GENERATIVEQC_API const char* generativeqc_d4_provider_identity(void);
GENERATIVEQC_API const char* generativeqc_d4_scheduler_identity(void);

/**
 * Prepare a standalone D4(BJ)-EEQ ragged fleet. Atomic numbers, charges and
 * prepared geometries are copied. maximum_bytes bounds all persistent owner
 * state plus worst-case execution staging/workspace.
 */
GENERATIVEQC_API generativeqc_status generativeqc_d4_batch_prepare(
    generativeqc_context* context, const generativeqc_d4_system_descriptor* systems,
    uint32_t system_count, const generativeqc_d4_bj_eeq_descriptor* model,
    generativeqc_d4_batch** batch);
GENERATIVEQC_API void generativeqc_d4_batch_destroy(generativeqc_d4_batch* batch);
GENERATIVEQC_API generativeqc_status generativeqc_d4_batch_get_diagnostic(
    const generativeqc_d4_batch* batch, generativeqc_d4_runtime_diagnostic* diagnostic);

/**
 * Execute complete molecular D4(BJ)-EEQ energy and analytic dE/dR.
 * inputs=NULL,input_count=0 replays prepared geometries. Otherwise each input
 * may independently select prepared or changed coordinates. Item scientific
 * failures are isolated; successful function return means the replay itself
 * was structurally valid.
 */
GENERATIVEQC_API generativeqc_status generativeqc_d4_batch_execute(
    generativeqc_d4_batch* batch, const generativeqc_d4_batch_input_descriptor* inputs,
    uint32_t input_count, generativeqc_d4_batch_item_result_descriptor* results,
    uint32_t result_count);

/**
 * Prepare a bounded fixed-grid VV10/rVV10 pair evaluator.
 *
 * CPU_REFERENCE and qualified CUDA backends retain O(N_grid) storage and never
 * materialize the full pair matrix. maximum_bytes bounds provider-owned peak
 * host/device workspace, including transactional output staging.
 */
GENERATIVEQC_API generativeqc_status generativeqc_nonlocal_plan_prepare(
    generativeqc_context* context, const generativeqc_nonlocal_descriptor* model,
    generativeqc_nonlocal_plan** plan);
GENERATIVEQC_API void generativeqc_nonlocal_plan_destroy(generativeqc_nonlocal_plan* plan);
GENERATIVEQC_API generativeqc_status generativeqc_nonlocal_plan_get_diagnostic(
    const generativeqc_nonlocal_plan* plan, generativeqc_nonlocal_runtime_diagnostic* diagnostic);

/** Evaluate fixed-grid energy and any requested derivative families. */
GENERATIVEQC_API generativeqc_status generativeqc_nonlocal_plan_execute(
    generativeqc_nonlocal_plan* plan, const generativeqc_nonlocal_input_descriptor* input,
    generativeqc_nonlocal_result_descriptor* result);

#ifdef __cplusplus
}
#endif

#endif

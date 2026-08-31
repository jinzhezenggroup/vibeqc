#ifndef VIBEQC_VIBEQC_H
#define VIBEQC_VIBEQC_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(VIBEQC_BUILDING_LIBRARY)
#    define VIBEQC_API __declspec(dllexport)
#  else
#    define VIBEQC_API __declspec(dllimport)
#  endif
#else
#  define VIBEQC_API __attribute__((visibility("default")))
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
  VIBEQC_STATUS_INTERNAL_ERROR = 8
};

typedef int32_t vibeqc_method;
enum {
  VIBEQC_METHOD_RHF = 1,
  VIBEQC_METHOD_UHF = 2,
  VIBEQC_METHOD_WB97M_V = 3,
  VIBEQC_METHOD_RCCSD_T = 4
};

/** Broad algorithm family used for capability discovery and dispatch. */
typedef int32_t vibeqc_method_family;
enum {
  VIBEQC_METHOD_FAMILY_HARTREE_FOCK = 1,
  VIBEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL = 2,
  VIBEQC_METHOD_FAMILY_COUPLED_CLUSTER = 3
};

typedef uint32_t vibeqc_property_flags;
enum {
  VIBEQC_PROPERTY_ENERGY = 1u << 0,
  VIBEQC_PROPERTY_FORCES = 1u << 1
};

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

typedef int32_t vibeqc_basis_representation;
enum {
  /** CCA-ordered Cartesian functions: 1, 3, 6, and 10 AOs for s-p-d-f. */
  VIBEQC_BASIS_CARTESIAN = 0,
  /** Real spherical functions in PySCF/libcint order: 1, 3, 5, and 7 AOs. */
  VIBEQC_BASIS_SPHERICAL = 1
};

/**
 * Setup-time CUDA density-fitting metric and allocation evidence.
 *
 * Records are returned in plan-slot order for the most recent CUDA DF batch
 * execution. `system_index` identifies the original prepared-batch input and
 * `bucket_id` identifies the fleet bucket that owns the plan.
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
  uint64_t bra_primitive_tasks[
      VIBEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
  uint64_t bra_primitive_work[
      VIBEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
  uint64_t ket_primitive_tasks[
      VIBEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
  uint64_t ket_primitive_work[
      VIBEQC_PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT];
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
  /** Planner budget in bytes; zero selects the implementation default. */
  uint64_t density_fitting_memory_budget_bytes;
} vibeqc_method_descriptor;

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
  double density_rms;
  int32_t converged;
  vibeqc_backend executed_backend;
} vibeqc_result_descriptor;

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

/** Return a stable, process-lifetime error string for a status code. */
VIBEQC_API const char* vibeqc_status_message(vibeqc_status status);

/** Query whether a method is currently executable. */
VIBEQC_API vibeqc_status vibeqc_method_available(vibeqc_method method, int32_t* available);

/** Query method family, properties, and batch support without preparing work. */
VIBEQC_API vibeqc_status vibeqc_method_get_capabilities(
    vibeqc_method method,
    vibeqc_method_capabilities_descriptor* capabilities);

VIBEQC_API vibeqc_status vibeqc_context_create(
    const vibeqc_context_descriptor* descriptor, vibeqc_context** context);
VIBEQC_API void vibeqc_context_destroy(vibeqc_context* context);

VIBEQC_API vibeqc_status vibeqc_system_create(
    vibeqc_context* context,
    const vibeqc_system_descriptor* descriptor,
    vibeqc_system** system);
VIBEQC_API void vibeqc_system_destroy(vibeqc_system* system);

VIBEQC_API vibeqc_status vibeqc_calculation_prepare(
    vibeqc_context* context,
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
VIBEQC_API vibeqc_status vibeqc_calculation_execute(
    vibeqc_calculation* calculation, vibeqc_result_descriptor* result);

/**
 * Prepare a persistent ragged fleet plan. Systems may have different atom,
 * shell, primitive, and AO counts; no global padding is introduced.
 */
VIBEQC_API vibeqc_status vibeqc_batch_prepare(
    vibeqc_context* context,
    const vibeqc_system* const* systems,
    uint32_t system_count,
    const vibeqc_method_descriptor* descriptor,
    vibeqc_batch_flags flags,
    vibeqc_batch** batch);

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
    const vibeqc_batch* batch,
    vibeqc_shell_class_profile_entry* entries,
    uint32_t entry_count);

/** Copy PPPS queue statistics from the most recent profiled CUDA execution. */
VIBEQC_API vibeqc_status vibeqc_batch_get_last_ppps_queue_profile(
    const vibeqc_batch* batch,
    vibeqc_ppps_queue_profile* profile);

/**
 * Copy setup-time eigensolver evidence for every bucket in the last execution.
 *
 * Pass `entries = NULL` and `entry_count = 0` to query the required count in
 * `written_count`. A later warm replay returns the cached setup decision and
 * never performs another capability probe.
 */
VIBEQC_API vibeqc_status vibeqc_batch_get_last_eigensolver_diagnostics(
    const vibeqc_batch* batch,
    vibeqc_eigensolver_diagnostic* entries,
    uint32_t entry_count,
    uint32_t* written_count);

/**
 * Copy CUDA density-fitting metric conditioning and allocation diagnostics
 * from the most recent batch execution. Pass `entries = NULL` and
 * `entry_count = 0` to query the required count in `written_count`.
 */
VIBEQC_API vibeqc_status
vibeqc_batch_get_last_density_fitting_metric_diagnostics(
    const vibeqc_batch* batch,
    vibeqc_density_fitting_metric_diagnostic* entries,
    uint32_t entry_count,
    uint32_t* written_count);

/**
 * Copy per-iteration inactive-eigensolver evidence from the last execution.
 *
 * The batch must opt into
 * `VIBEQC_BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING`. Pass `entries = NULL`
 * and `entry_count = 0` to query the required count. Records are bucket-major
 * and iteration-ordered within each bucket.
 */
VIBEQC_API vibeqc_status vibeqc_batch_get_last_inactive_eigensolver_profile(
    const vibeqc_batch* batch,
    vibeqc_inactive_eigensolver_profile_entry* entries,
    uint32_t entry_count,
    uint32_t* written_count);

/** Discard all retained per-system converged-density warm starts. */
VIBEQC_API vibeqc_status vibeqc_batch_clear_warm_starts(vibeqc_batch* batch);

/**
 * Enable or disable replacement of retained warm-start densities.
 *
 * Passing zero freezes the current snapshots so every later execution starts
 * from the same per-system dm0. Passing one restores the default behavior in
 * which each successful execution advances its retained density. Existing
 * snapshots are neither cleared nor created by this call.
 */
VIBEQC_API vibeqc_status vibeqc_batch_set_warm_start_updates(
    vibeqc_batch* batch, int32_t enabled);

/**
 * Execute all systems with failure isolation. A successful function return
 * means the batch was structurally valid; inspect each result.status for its
 * scientific outcome. `inputs` may be NULL with input_count=0 to reuse all
 * prepared geometries, otherwise it must contain one descriptor per system.
 */
VIBEQC_API vibeqc_status vibeqc_batch_execute(
    vibeqc_batch* batch,
    const vibeqc_batch_input_descriptor* inputs,
    uint32_t input_count,
    vibeqc_batch_item_result_descriptor* results,
    uint32_t result_count);

#ifdef __cplusplus
}
#endif

#endif

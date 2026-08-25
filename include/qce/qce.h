#ifndef QCE_QCE_H
#define QCE_QCE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(QCE_BUILDING_LIBRARY)
#    define QCE_API __declspec(dllexport)
#  else
#    define QCE_API __declspec(dllimport)
#  endif
#else
#  define QCE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define QCE_ABI_VERSION 0u

typedef int32_t qce_status;
enum {
  QCE_STATUS_SUCCESS = 0,
  QCE_STATUS_INVALID_ARGUMENT = 1,
  QCE_STATUS_ABI_MISMATCH = 2,
  QCE_STATUS_NOT_IMPLEMENTED = 3,
  QCE_STATUS_SCF_NOT_CONVERGED = 4,
  QCE_STATUS_NUMERICAL_FAILURE = 5,
  QCE_STATUS_CUDA_ERROR = 6,
  QCE_STATUS_OUT_OF_MEMORY = 7,
  QCE_STATUS_INTERNAL_ERROR = 8
};

typedef int32_t qce_method;
enum {
  QCE_METHOD_RHF = 1,
  QCE_METHOD_UHF = 2,
  QCE_METHOD_WB97M_V = 3,
  QCE_METHOD_RCCSD_T = 4
};

typedef int32_t qce_backend;
enum {
  QCE_BACKEND_CPU_REFERENCE = 0,
  QCE_BACKEND_CUDA = 1,
  /** Reserved compatibility tag used by pre-device-resident prototypes. */
  QCE_BACKEND_HYBRID_CUDA = 2
};

typedef int32_t qce_basis_representation;
enum {
  /** CCA-ordered Cartesian functions: 1, 3, 6, and 10 AOs for s-p-d-f. */
  QCE_BASIS_CARTESIAN = 0,
  /** Real spherical functions in PySCF/libcint order: 1, 3, 5, and 7 AOs. */
  QCE_BASIS_SPHERICAL = 1
};

typedef struct qce_context qce_context;
typedef struct qce_system qce_system;
typedef struct qce_calculation qce_calculation;
typedef struct qce_batch qce_batch;

typedef uint32_t qce_batch_flags;
enum {
  /** Retain each converged AO density for the next execution of the plan. */
  QCE_BATCH_ENABLE_WARM_STARTS = 1u << 0,
  /**
   * Collect the final density-screened direct-J/K shell-class work profile.
   *
   * This diagnostic adds one untimed-by-default CUDA reduction after the
   * final compaction pass. Leave it disabled for production timing runs.
   */
  QCE_BATCH_ENABLE_SHELL_CLASS_PROFILING = 1u << 1
};

/** Number of pair/pair-exchange-reduced s/p/d/f quartet shell classes. */
#define QCE_DIRECT_SHELL_CLASS_COUNT 55u

/** Work retained for one shell class after final-density screening. */
typedef struct qce_shell_class_profile_entry {
  uint64_t shell_quartets;
  uint64_t tiles;
  uint64_t ao_quartets;
  uint64_t primitive_quartets;
} qce_shell_class_profile_entry;

typedef struct qce_context_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  int32_t device_id;
  qce_backend backend;
} qce_context_descriptor;

typedef struct qce_atom {
  int32_t atomic_number;
  double x;
  double y;
  double z;
} qce_atom;

typedef struct qce_primitive {
  double exponent;
  double coefficient;
} qce_primitive;

typedef struct qce_shell {
  uint32_t atom_index;
  uint32_t angular_momentum;
  uint32_t primitive_offset;
  uint32_t primitive_count;
} qce_shell;

typedef struct qce_system_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  const qce_atom* atoms;
  uint32_t atom_count;
  const qce_shell* shells;
  uint32_t shell_count;
  const qce_primitive* primitives;
  uint32_t primitive_count;
  int32_t charge;
  uint32_t multiplicity;
  /** Optional in older ABI-0 descriptors; absent fields imply Cartesian. */
  qce_basis_representation basis_representation;
} qce_system_descriptor;

typedef struct qce_method_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  qce_method method;
  uint32_t max_iterations;
  uint32_t diis_history;
  double energy_tolerance;
  double density_tolerance;
  double screening_tolerance;
} qce_method_descriptor;

typedef struct qce_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  double energy;
  double* forces;
  uint32_t force_count;
  uint32_t iterations;
  double energy_change;
  double density_rms;
  int32_t converged;
  qce_backend executed_backend;
} qce_result_descriptor;

/** Optional per-system coordinates for a prepared ragged batch execution. */
typedef struct qce_batch_input_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  /** Flat xyz coordinates in Bohr, or NULL to use the prepared geometry. */
  const double* coordinates;
  uint32_t coordinate_count;
} qce_batch_input_descriptor;

/** Per-system output. Each item owns an independent status and diagnostics. */
typedef struct qce_batch_item_result_descriptor {
  uint32_t struct_size;
  uint32_t abi_version;
  qce_status status;
  double energy;
  double* forces;
  uint32_t force_count;
  uint32_t iterations;
  double energy_change;
  double density_rms;
  int32_t converged;
  qce_backend executed_backend;
  uint32_t bucket_id;
  int32_t warm_start_used;
  int32_t warm_start_fallback;
} qce_batch_item_result_descriptor;

/** Return the ABI version implemented by the loaded shared library. */
QCE_API uint32_t qce_get_abi_version(void);

/** Return a stable, process-lifetime error string for a status code. */
QCE_API const char* qce_status_message(qce_status status);

/** Query whether a method is currently executable. */
QCE_API qce_status qce_method_available(qce_method method, int32_t* available);

QCE_API qce_status qce_context_create(
    const qce_context_descriptor* descriptor, qce_context** context);
QCE_API void qce_context_destroy(qce_context* context);

QCE_API qce_status qce_system_create(
    qce_context* context,
    const qce_system_descriptor* descriptor,
    qce_system** system);
QCE_API void qce_system_destroy(qce_system* system);

QCE_API qce_status qce_calculation_prepare(
    qce_context* context,
    const qce_system* system,
    const qce_method_descriptor* descriptor,
    qce_calculation** calculation);
QCE_API void qce_calculation_destroy(qce_calculation* calculation);

/**
 * Execute synchronously. The caller owns result->forces and must provide at
 * least 3 * atom_count doubles. Coordinates and all reported derivatives use
 * atomic units (Bohr, Hartree, Hartree/Bohr).
 */
QCE_API qce_status qce_calculation_execute(
    qce_calculation* calculation, qce_result_descriptor* result);

/**
 * Prepare a persistent ragged fleet plan. Systems may have different atom,
 * shell, primitive, and AO counts; no global padding is introduced.
 */
QCE_API qce_status qce_batch_prepare(
    qce_context* context,
    const qce_system* const* systems,
    uint32_t system_count,
    const qce_method_descriptor* descriptor,
    qce_batch_flags flags,
    qce_batch** batch);

QCE_API void qce_batch_destroy(qce_batch* batch);

QCE_API uint32_t qce_batch_get_system_count(const qce_batch* batch);

/**
 * Copy the most recent final-density shell-class profile.
 *
 * The batch must have been prepared with
 * `QCE_BATCH_ENABLE_SHELL_CLASS_PROFILING`, executed through the CUDA direct
 * J/K path, and `entry_count` must be at least
 * `QCE_DIRECT_SHELL_CLASS_COUNT`. Entries use the canonical triangular class
 * encoding documented by QCE's direct shell scheduler.
 */
QCE_API qce_status qce_batch_get_last_shell_class_profile(
    const qce_batch* batch,
    qce_shell_class_profile_entry* entries,
    uint32_t entry_count);

/** Discard all retained per-system converged-density warm starts. */
QCE_API qce_status qce_batch_clear_warm_starts(qce_batch* batch);

/**
 * Execute all systems with failure isolation. A successful function return
 * means the batch was structurally valid; inspect each result.status for its
 * scientific outcome. `inputs` may be NULL with input_count=0 to reuse all
 * prepared geometries, otherwise it must contain one descriptor per system.
 */
QCE_API qce_status qce_batch_execute(
    qce_batch* batch,
    const qce_batch_input_descriptor* inputs,
    uint32_t input_count,
    qce_batch_item_result_descriptor* results,
    uint32_t result_count);

#ifdef __cplusplus
}
#endif

#endif

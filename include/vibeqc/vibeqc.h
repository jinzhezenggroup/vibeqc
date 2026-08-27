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

typedef int32_t vibeqc_basis_representation;
enum {
  /** CCA-ordered Cartesian functions: 1, 3, 6, and 10 AOs for s-p-d-f. */
  VIBEQC_BASIS_CARTESIAN = 0,
  /** Real spherical functions in PySCF/libcint order: 1, 3, 5, and 7 AOs. */
  VIBEQC_BASIS_SPHERICAL = 1
};

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
  VIBEQC_BATCH_ENABLE_SHELL_CLASS_PROFILING = 1u << 1
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

/** Discard all retained per-system converged-density warm starts. */
VIBEQC_API vibeqc_status vibeqc_batch_clear_warm_starts(vibeqc_batch* batch);

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

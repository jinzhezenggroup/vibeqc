#ifndef VIBEQC_XTB_RUNTIME_TYPES_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define VIBEQC_XTB_RUNTIME_TYPES_HPP

// Private native CPU/CUDA execution descriptors. VibeQC owns public entry
// points; the imported external C API and version header have been retired.

#include <stddef.h>
#include <stdint.h>

#define VIBEQC_XTB_API_VERSION 1u

/*
 * Electronic temperatures are k_B*T energy scales in Hartree, not kelvin.
 * This conversion matches the pinned xTB/tblite convention used by xtbloom.
 */
#define VIBEQC_XTB_KELVIN_TO_HARTREE 3.166808578545117e-6
#define VIBEQC_XTB_DEFAULT_ELECTRONIC_TEMPERATURE (300.0 * VIBEQC_XTB_KELVIN_TO_HARTREE)

/*
 * ABI tags are explicitly int32_t rather than enum-typed fields. This keeps
 * their object representation and function calling convention identical in C
 * and C++, including C99 builds compiled with options such as -fshort-enums.
 * The named enums below only provide debugger-friendly symbolic constants;
 * callers may pass any int32_t bit pattern and the library validates it.
 */
typedef int32_t vibeqc_xtb_status_t;
enum vibeqc_xtb_status_value {
  VIBEQC_XTB_STATUS_SUCCESS = 0,
  VIBEQC_XTB_STATUS_INVALID_ARGUMENT = 1,
  VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE = 2,
  VIBEQC_XTB_STATUS_NOT_SUPPORTED = 3,
  VIBEQC_XTB_STATUS_ALLOCATION_FAILED = 4,
  VIBEQC_XTB_STATUS_NOT_IMPLEMENTED = 5,
  VIBEQC_XTB_STATUS_INTERNAL_ERROR = 6,
  /* Per-system SCC reached max_scc_iterations without satisfying both tolerances. */
  VIBEQC_XTB_STATUS_SCC_NOT_CONVERGED = 7,
  /* Per-system generalized eigensolver failed or produced an unusable eigensystem. */
  VIBEQC_XTB_STATUS_EIGENSOLVER_FAILED = 8
};

typedef int32_t vibeqc_xtb_backend_t;
enum vibeqc_xtb_backend_value {
  /* Prefer CUDA when it is compiled in and a compatible device is present. */
  VIBEQC_XTB_BACKEND_AUTO = 0,
  VIBEQC_XTB_BACKEND_CPU = 1,
  VIBEQC_XTB_BACKEND_CUDA = 2,
  /* Reserved now so adding HIP kernels does not require redesigning the ABI. */
  VIBEQC_XTB_BACKEND_ROCM = 3
};

typedef int32_t vibeqc_xtb_memory_space_t;
enum vibeqc_xtb_memory_space_value {
  VIBEQC_XTB_MEMORY_HOST = 0,
  VIBEQC_XTB_MEMORY_CUDA_DEVICE = 1,
  VIBEQC_XTB_MEMORY_ROCM_DEVICE = 2
};

typedef int32_t vibeqc_xtb_model_t;
enum vibeqc_xtb_model_value { VIBEQC_XTB_MODEL_GFN1_XTB = 1, VIBEQC_XTB_MODEL_GFN2_XTB = 2 };

typedef int32_t vibeqc_xtb_scc_start_mode_t;
enum vibeqc_xtb_scc_start_mode_value {
  /* Restore the immutable initial electronic state before SCC. */
  VIBEQC_XTB_SCC_START_FRESH = 1,
  /* Strictly consume a checkpoint from the latest fully converged compatible batch call. */
  VIBEQC_XTB_SCC_START_WARM = 2
};

typedef int32_t vibeqc_xtb_scc_mixer_t;
enum vibeqc_xtb_scc_mixer_value {
  /* Johnson modified-Broyden mixing used by the GFN2 CPU and CUDA backends. */
  VIBEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN = 1
};

typedef int32_t vibeqc_xtb_determinism_t;
enum vibeqc_xtb_determinism_value {
  /* Use the production execution policy selected by the backend. */
  VIBEQC_XTB_DETERMINISM_DEFAULT = 0,
  /*
   * Request exact replay for an unchanged build, backend, numerical provider
   * or CUDA toolkit, device architecture, complete descriptors/options,
   * launch/bucket geometry, and FRESH/WARM sequence. This is not a bitwise
   * CPU/CUDA, cross-provider, cross-toolkit, or cross-architecture promise.
   */
  VIBEQC_XTB_DETERMINISM_REPRODUCIBLE = 1
};

typedef int32_t vibeqc_xtb_compute_flag_t;
enum vibeqc_xtb_compute_flag_value {
  VIBEQC_XTB_COMPUTE_ENERGY = 1 << 0,
  VIBEQC_XTB_COMPUTE_FORCES = 1 << 1,
  VIBEQC_XTB_COMPUTE_ATOMIC_CHARGES = 1 << 2,
  VIBEQC_XTB_COMPUTE_POINT_CHARGE_FORCES = 1 << 3,
  /* Reports per-system dipole moments through batch_result.dipole_moments. */
  VIBEQC_XTB_COMPUTE_DIPOLE_MOMENTS = 1 << 4,
  /* Reports per-system native-periodic dE/d(strain) through
   * batch_result.strain_derivatives. The nine row-major entries form the
   * symmetric derivative with respect to infinitesimal affine strain of the
   * direct-cell rows, in Hartree; each off-diagonal shear value is published
   * in both transposed positions. This output is released for CPU GFN2 XYZ
   * requests; CUDA and molecular requests reject it explicitly. */
  VIBEQC_XTB_COMPUTE_STRAIN_DERIVATIVES = 1 << 5,
  /* Bits 16-31 are reserved for future outputs and must be zero on input. */
};

typedef int32_t vibeqc_xtb_result_flag_t;
enum vibeqc_xtb_result_flag_value {
  /*
   * Set when atomic_potential_shifts or charge_response_matrix was supplied.
   * Forces then exclude coordinate derivatives of those caller-owned fields.
   */
  VIBEQC_XTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES = 1 << 0,
  /* Set when the requested per-system dipole moments were published. */
  VIBEQC_XTB_RESULT_DIPOLE_MOMENTS = 1 << 4,
  /* Set when native-periodic per-system strain derivatives were published. */
  VIBEQC_XTB_RESULT_STRAIN_DERIVATIVES = 1 << 5,
  /* Bits 16-31 are reserved; runtime-produced result flags are zero there. */
};

/*
 * Tag set for the generic interaction attachment slot (ABI-v3 batch suffix).
 * Tag values are intentionally spread over family ranges so future additions
 * never renumber an existing value. Both GFN2 backends implement the uniform
 * electric field; every other tag remains reserved and returns
 * VIBEQC_XTB_STATUS_NOT_IMPLEMENTED. VIBEQC_XTB_INTERACTION_NONE is not a valid
 * attachment.
 */
typedef int32_t vibeqc_xtb_interaction_type_t;
enum vibeqc_xtb_interaction_type_value {
  VIBEQC_XTB_INTERACTION_NONE = 0,
  /* External potentials (0x01xx). */
  VIBEQC_XTB_INTERACTION_ELECTRIC_FIELD = 0x0101,
  VIBEQC_XTB_INTERACTION_ELECTRIC_FIELD_GRADIENT = 0x0102,
  VIBEQC_XTB_INTERACTION_POINT_CHARGES_MULTIPOLE = 0x0103,
  VIBEQC_XTB_INTERACTION_ATOMIC_POTENTIAL_GRID = 0x0104,
  /* Self-consistent solvation models (0x02xx). */
  VIBEQC_XTB_INTERACTION_ALPB_SOLVATION = 0x0201,
  VIBEQC_XTB_INTERACTION_GBSA_SOLVATION = 0x0202,
  VIBEQC_XTB_INTERACTION_GB_SOLVATION = 0x0203,
  VIBEQC_XTB_INTERACTION_GBE_SOLVATION = 0x0204,
  VIBEQC_XTB_INTERACTION_DDX_SOLVATION = 0x0205,
  /* Dispersion models (0x03xx). */
  VIBEQC_XTB_INTERACTION_D3_DISPERSION = 0x0301,
  VIBEQC_XTB_INTERACTION_D4_VARIANT_DISPERSION = 0x0302,
  /* Structure-correction models (0x04xx). */
  VIBEQC_XTB_INTERACTION_HALOGEN_BOND = 0x0401
};

/*
 * Periodic-axis mask for the ABI-v4 native-lattice batch suffix.
 *
 * The individual x/y/z bits are reserved so later releases can describe
 * lower-dimensional boundary conditions without changing the field width.
 * This release accepts NONE for a molecular batch item and XYZ for a native
 * three-dimensional periodic item. Partial masks are not implemented.
 */
typedef int32_t vibeqc_xtb_periodic_axes_t;
enum vibeqc_xtb_periodic_axes_value {
  VIBEQC_XTB_PERIODIC_AXES_NONE = 0,
  VIBEQC_XTB_PERIODIC_AXIS_X = 1 << 0,
  VIBEQC_XTB_PERIODIC_AXIS_Y = 1 << 1,
  VIBEQC_XTB_PERIODIC_AXIS_Z = 1 << 2,
  VIBEQC_XTB_PERIODIC_AXES_XYZ = VIBEQC_XTB_PERIODIC_AXIS_X | VIBEQC_XTB_PERIODIC_AXIS_Y |
      VIBEQC_XTB_PERIODIC_AXIS_Z
};

/*
 * One attachment of an external interaction to one batch item.
 *
 * type selects the interaction; flags is reserved and must be zero;
 * system_index addresses one batch item in [0, batch_size); payload_offset
 * and payload_size locate the caller-owned payload block inside
 * interaction_payload. Every payload block starts with an int32_t
 * block_version so the byte layout of one tag can evolve independently of
 * this descriptor. The electric-field block (block_version 1) is
 * 32 bytes: int32_t version, int32_t reserved (zero), three finite doubles
 * holding the field vector in Hartree per elementary charge per bohr.
 */
typedef struct vibeqc_xtb_interaction {
  vibeqc_xtb_interaction_type_t type;
  uint32_t flags;
  int64_t system_index;
  uint64_t payload_offset;
  uint64_t payload_size;
} vibeqc_xtb_interaction_t;

#define VIBEQC_XTB_INTERACTION_V1_SIZE \
  (offsetof(vibeqc_xtb_interaction_t, payload_size) + sizeof(uint64_t))

/*
 * Keep all public ABI tag and flag aliases at their specified width.
 */
#if defined(__cplusplus)
static_assert(sizeof(vibeqc_xtb_status_t) == sizeof(int32_t), "vibeqc_xtb_status_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_backend_t) == sizeof(int32_t),
              "vibeqc_xtb_backend_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_memory_space_t) == sizeof(int32_t),
              "vibeqc_xtb_memory_space_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_model_t) == sizeof(int32_t), "vibeqc_xtb_model_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_scc_start_mode_t) == sizeof(int32_t),
              "vibeqc_xtb_scc_start_mode_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_scc_mixer_t) == sizeof(int32_t),
              "vibeqc_xtb_scc_mixer_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_determinism_t) == sizeof(int32_t),
              "vibeqc_xtb_determinism_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_compute_flag_t) == sizeof(int32_t),
              "vibeqc_xtb_compute_flag_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_result_flag_t) == sizeof(int32_t),
              "vibeqc_xtb_result_flag_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_interaction_type_t) == sizeof(int32_t),
              "vibeqc_xtb_interaction_type_t must be 32-bit");
static_assert(sizeof(vibeqc_xtb_periodic_axes_t) == sizeof(int32_t),
              "vibeqc_xtb_periodic_axes_t must be 32-bit");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(vibeqc_xtb_status_t) == sizeof(int32_t),
               "vibeqc_xtb_status_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_backend_t) == sizeof(int32_t),
               "vibeqc_xtb_backend_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_memory_space_t) == sizeof(int32_t),
               "vibeqc_xtb_memory_space_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_model_t) == sizeof(int32_t), "vibeqc_xtb_model_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_scc_start_mode_t) == sizeof(int32_t),
               "vibeqc_xtb_scc_start_mode_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_scc_mixer_t) == sizeof(int32_t),
               "vibeqc_xtb_scc_mixer_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_determinism_t) == sizeof(int32_t),
               "vibeqc_xtb_determinism_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_compute_flag_t) == sizeof(int32_t),
               "vibeqc_xtb_compute_flag_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_result_flag_t) == sizeof(int32_t),
               "vibeqc_xtb_result_flag_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_interaction_type_t) == sizeof(int32_t),
               "vibeqc_xtb_interaction_type_t must be 32-bit");
_Static_assert(sizeof(vibeqc_xtb_periodic_axes_t) == sizeof(int32_t),
               "vibeqc_xtb_periodic_axes_t must be 32-bit");
#endif

/* A byte-sized view of caller-owned input memory. The runtime never takes ownership. */
typedef struct vibeqc_xtb_const_buffer {
  const void* data;
  size_t size_bytes;
  vibeqc_xtb_memory_space_t memory_space;
  uint32_t reserved;
} vibeqc_xtb_const_buffer_t;

/* A byte-sized view of caller-owned output memory. The runtime never takes ownership. */
typedef struct vibeqc_xtb_buffer {
  void* data;
  size_t size_bytes;
  vibeqc_xtb_memory_space_t memory_space;
  uint32_t reserved;
} vibeqc_xtb_buffer_t;

/*
 * Pointer-bearing ABI images are architecture-local. A wasm32/native ILP32
 * caller and library use 32-bit pointers and size_t, while wasm64/native LP64
 * uses 64-bit pointers and size_t. Fixed-width counts and offsets below remain
 * 64-bit on both targets. Keep exact assertions for both supported widths so a
 * compiler or packing change fails at build time instead of corrupting views.
 */
#if UINTPTR_MAX == UINT64_MAX
#define VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE 24u
#define VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET 8u
#define VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET 16u
#define VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET 20u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V1_SIZE 328u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V2_SIZE 352u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET 352u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_INTERACTION_DESCRIPTORS_OFFSET 360u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_INTERACTION_PAYLOAD_OFFSET 384u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V3_SIZE 408u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_CELL_MATRICES_OFFSET 408u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_PERIODIC_AXES_OFFSET 432u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V4_SIZE 456u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V1_SIZE 184u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_DIPOLE_OFFSET 184u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_QUADRUPOLE_OFFSET 208u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_WIBERG_OFFSET 232u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_SPIN_OFFSET 256u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V2_SIZE 280u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_STRAIN_OFFSET 280u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V3_SIZE 304u
#elif UINTPTR_MAX == UINT32_MAX
#define VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE 16u
#define VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET 4u
#define VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET 8u
#define VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET 12u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V1_SIZE 232u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V2_SIZE 248u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET 248u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_INTERACTION_DESCRIPTORS_OFFSET 256u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_INTERACTION_PAYLOAD_OFFSET 272u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V3_SIZE 288u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_CELL_MATRICES_OFFSET 288u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_PERIODIC_AXES_OFFSET 304u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V4_SIZE 320u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V1_SIZE 128u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_DIPOLE_OFFSET 128u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_QUADRUPOLE_OFFSET 144u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_WIBERG_OFFSET 160u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_SPIN_OFFSET 176u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V2_SIZE 192u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_STRAIN_OFFSET 192u
#define VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V3_SIZE 208u
#endif

#if defined(__cplusplus)
#define VIBEQC_XTB_DETAIL_ABI_ASSERT(condition, message) static_assert((condition), message)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define VIBEQC_XTB_DETAIL_ABI_ASSERT(condition, message) _Static_assert((condition), message)
#endif

#if defined(VIBEQC_XTB_DETAIL_ABI_ASSERT) && defined(VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE)
VIBEQC_XTB_DETAIL_ABI_ASSERT(sizeof(vibeqc_xtb_const_buffer_t) ==
                                 VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE,
                             "vibeqc_xtb_const_buffer_t image must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(sizeof(vibeqc_xtb_buffer_t) == VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE,
                             "vibeqc_xtb_buffer_t image must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(offsetof(vibeqc_xtb_const_buffer_t, size_bytes) ==
                                 VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET,
                             "vibeqc_xtb_const_buffer_t size must follow its target-width pointer");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_const_buffer_t, memory_space) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET,
    "vibeqc_xtb_const_buffer_t memory tag offset must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_const_buffer_t, reserved) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET,
    "vibeqc_xtb_const_buffer_t reserved offset must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(offsetof(vibeqc_xtb_buffer_t, size_bytes) ==
                                 VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET,
                             "vibeqc_xtb_buffer_t size must follow its target-width pointer");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_buffer_t, memory_space) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET,
    "vibeqc_xtb_buffer_t memory tag offset must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_buffer_t, reserved) == VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET,
    "vibeqc_xtb_buffer_t reserved offset must match the target pointer width");
#endif

/*
 * Ragged molecular batch. All real-valued inputs use IEEE binary64 and atomic
 * units: bohr for positions and elementary-charge units for charges.
 *
 * atom_offsets contains batch_size + 1 int64_t values. atomic_numbers contains
 * total_atoms int32_t values. positions contains total_atoms * 3 doubles.
 * molecular_charges contains batch_size doubles and unpaired_electrons contains
 * batch_size int32_t values. The ABI-v2 spin_channels field, when present,
 * contains batch_size int32_t values equal to one (restricted) or two
 * (unrestricted). A missing or NULL spin_channels buffer preserves the ABI-v1
 * restricted default.
 *
 * External point charges participate in every SCC iteration. When
 * total_point_charges is nonzero, point_charge_offsets has batch_size + 1
 * int64_t values, point_charge_positions has total_point_charges * 3 doubles,
 * point_charge_values has total_point_charges doubles, and point_charge_gammas
 * has total_point_charges doubles. Gamma is the explicit point-site screening
 * parameter used in the softened Coulomb interaction; it is a model parameter,
 * not an optimizable point-charge degree of freedom.
 *
 * atomic_potential_shifts and the response matrix are optional advanced inputs
 * for periodic QM/MM coupling. For molecule i they define a per-atom SCC shift
 * b_i + A_i q_i and variational energy q_i^T b_i + 0.5 q_i^T A_i q_i.
 * atomic_potential_shifts contains total_atoms doubles. charge_response_offsets
 * contains batch_size + 1 int64_t values, and each row-major symmetric A_i is
 * packed consecutively in charge_response_matrix. Derivatives of b and A with
 * respect to coordinates are outside this runtime and are not included in forces.
 */
typedef struct vibeqc_xtb_batch {
  uint32_t struct_size;
  uint32_t api_version;
  int64_t batch_size;
  int64_t total_atoms;
  int64_t total_point_charges;
  int64_t total_charge_response_elements;
  vibeqc_xtb_const_buffer_t atom_offsets;
  vibeqc_xtb_const_buffer_t atomic_numbers;
  vibeqc_xtb_const_buffer_t positions;
  vibeqc_xtb_const_buffer_t molecular_charges;
  vibeqc_xtb_const_buffer_t unpaired_electrons;
  vibeqc_xtb_const_buffer_t point_charge_offsets;
  vibeqc_xtb_const_buffer_t point_charge_positions;
  vibeqc_xtb_const_buffer_t point_charge_values;
  vibeqc_xtb_const_buffer_t point_charge_gammas;
  vibeqc_xtb_const_buffer_t atomic_potential_shifts;
  vibeqc_xtb_const_buffer_t charge_response_offsets;
  vibeqc_xtb_const_buffer_t charge_response_matrix;
  /* ABI v2 optional suffix; NULL selects one restricted channel per system. */
  vibeqc_xtb_const_buffer_t spin_channels;
  /* ABI v3 optional suffix: generic external-interaction attachments.
   *
   * total_interactions counts vibeqc_xtb_interaction_t entries in
   * interaction_descriptors, each attaching one caller-owned payload block in
   * interaction_payload to one batch item. The suffix may carry any mix of
   * host and CUDA-device storage and may be present with zero interactions,
   * preserving ABI-v1/v2 behavior for callers that never use it. See
   * vibeqc_xtb_interaction_type_t for the reserved tag set and payload contract. */
  int64_t total_interactions;
  vibeqc_xtb_const_buffer_t interaction_descriptors;
  vibeqc_xtb_const_buffer_t interaction_payload;
  /* ABI v4 optional suffix: native lattice/PBC descriptors.
   *
   * When either buffer is active, both are required. cell_matrices contains
   * batch_size row-major 3x3 direct-cell matrices in bohr. The three rows are
   * the a, b, and c lattice vectors, so a fractional row vector u maps to the
   * Cartesian vector u[0]*a + u[1]*b + u[2]*c. periodic_axes contains
   * batch_size vibeqc_xtb_periodic_axes_t values. NONE requires the corresponding
   * nine cell entries to be exactly zero; XYZ requires a finite, right-handed,
   * nonsingular cell. Partial-axis masks are reserved but not implemented.
   *
   * Native PBC changes each model's complete topology and is distinct from the
   * caller-supplied b + A*q charge-response operator above. A V4 batch whose
   * masks are all NONE remains a molecular request. GFN2 CPU accepts complete
   * XYZ energy/force/charge requests and the additive strain-derivative result
   * suffix. GFN1 and CUDA native XYZ execution remain explicitly refused until
   * their periodic physics is connected. */
  vibeqc_xtb_const_buffer_t cell_matrices;
  vibeqc_xtb_const_buffer_t periodic_axes;
} vibeqc_xtb_batch_t;

#define VIBEQC_XTB_BATCH_V1_SIZE \
  (offsetof(vibeqc_xtb_batch_t, charge_response_matrix) + sizeof(vibeqc_xtb_const_buffer_t))
#define VIBEQC_XTB_BATCH_V2_SIZE \
  (offsetof(vibeqc_xtb_batch_t, spin_channels) + sizeof(vibeqc_xtb_const_buffer_t))
#define VIBEQC_XTB_BATCH_V3_SIZE \
  (offsetof(vibeqc_xtb_batch_t, interaction_payload) + sizeof(vibeqc_xtb_const_buffer_t))
#define VIBEQC_XTB_BATCH_V4_SIZE \
  (offsetof(vibeqc_xtb_batch_t, periodic_axes) + sizeof(vibeqc_xtb_const_buffer_t))

/*
 * electronic_temperature is k_B*T in Hartree. Bindings that accept kelvin
 * should multiply by VIBEQC_XTB_KELVIN_TO_HARTREE before populating this struct.
 *
 * The ABI-v2 scc_start_mode suffix is a strict per-call policy. FRESH restores
 * the immutable initial electronic state. WARM consumes the checkpoint from
 * the latest fully converged compatible public batch call; it never falls back
 * to FRESH. A V1/short prefix means FRESH.
 *
 * A compatible identity is a batch whose topology and compute policy
 * (requested-property flags, molecular charges, unpaired electrons, spin
 * channels, point-charge and periodic structure, SCC tolerances, maximum
 * iterations, electronic temperature, mixer algorithm/history/damping, and
 * determinism policy) exactly match the previous fully converged call on the
 * same context. This is the same compute-options identity used by CPU and
 * CUDA. Geometry is not part of the identity: a WARM
 * call reuses the previous converged electronic state as the initial SCC guess
 * for the new coordinates and reconverges. A WARM request with no such
 * compatible fully converged predecessor (first call, changed topology or
 * policy, or a preceding non-converged batch) is rejected with
 * VIBEQC_XTB_STATUS_INVALID_ARGUMENT before any caller output is modified.
 *
 * An accepted FRESH attempt consumes the preceding compatible checkpoint
 * before execution starts. If that attempt later fails, including a CUDA
 * failure discovered in stream order after enqueue, no stale checkpoint from
 * an older call survives; a subsequent strict WARM request is rejected.
 */
typedef struct vibeqc_xtb_compute_options {
  uint32_t struct_size;
  uint32_t api_version;
  vibeqc_xtb_model_t model;
  uint32_t flags;
  int32_t max_scc_iterations;
  uint32_t reserved;
  double charge_tolerance;
  double energy_tolerance;
  double electronic_temperature;
  /* ABI v2 optional suffix; absent suffix preserves strict FRESH semantics. */
  vibeqc_xtb_scc_start_mode_t scc_start_mode;
  uint32_t reserved_v2;
  /*
   * ABI v3 optional suffix. A caller must provide the complete suffix or the
   * library uses modified-Broyden history 8, damping 0.4, and default
   * execution. Partial v3 suffixes are ignored as a unit.
   */
  /* Currently only VIBEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN is accepted. */
  vibeqc_xtb_scc_mixer_t scc_mixer;
  /* Modified-Broyden history depth in [1, 64]. */
  int32_t scc_mixer_history;
  /* Linear damping factor, finite and in (0, 1]. */
  double scc_mixer_damping;
  /* VIBEQC_XTB_DETERMINISM_DEFAULT or VIBEQC_XTB_DETERMINISM_REPRODUCIBLE. */
  vibeqc_xtb_determinism_t determinism;
  uint32_t reserved_v3;
} vibeqc_xtb_compute_options_t;

#define VIBEQC_XTB_COMPUTE_OPTIONS_V1_SIZE \
  (offsetof(vibeqc_xtb_compute_options_t, electronic_temperature) + sizeof(double))
#define VIBEQC_XTB_COMPUTE_OPTIONS_V2_SIZE \
  (offsetof(vibeqc_xtb_compute_options_t, reserved_v2) + sizeof(uint32_t))
#define VIBEQC_XTB_COMPUTE_OPTIONS_V3_SIZE \
  (offsetof(vibeqc_xtb_compute_options_t, reserved_v3) + sizeof(uint32_t))

#if defined(__cplusplus)
static_assert(offsetof(vibeqc_xtb_compute_options_t, scc_start_mode) == 48u,
              "vibeqc_xtb_compute_options_t ABI-v2 suffix must start at byte 48");
static_assert(VIBEQC_XTB_COMPUTE_OPTIONS_V1_SIZE == 48u,
              "vibeqc_xtb_compute_options_t ABI-v1 prefix must remain 48 bytes");
static_assert(VIBEQC_XTB_COMPUTE_OPTIONS_V2_SIZE == 56u,
              "vibeqc_xtb_compute_options_t ABI-v2 image must remain 56 bytes");
static_assert(offsetof(vibeqc_xtb_compute_options_t, scc_mixer) == 56u,
              "vibeqc_xtb_compute_options_t ABI-v3 mixer must start at byte 56");
static_assert(offsetof(vibeqc_xtb_compute_options_t, scc_mixer_history) == 60u,
              "vibeqc_xtb_compute_options_t ABI-v3 history must start at byte 60");
static_assert(offsetof(vibeqc_xtb_compute_options_t, scc_mixer_damping) == 64u,
              "vibeqc_xtb_compute_options_t ABI-v3 damping must start at byte 64");
static_assert(offsetof(vibeqc_xtb_compute_options_t, determinism) == 72u,
              "vibeqc_xtb_compute_options_t ABI-v3 determinism must start at byte 72");
static_assert(offsetof(vibeqc_xtb_compute_options_t, reserved_v3) == 76u,
              "vibeqc_xtb_compute_options_t ABI-v3 reserved field must start at byte 76");
static_assert(VIBEQC_XTB_COMPUTE_OPTIONS_V3_SIZE == 80u,
              "vibeqc_xtb_compute_options_t ABI-v3 image must remain 80 bytes");
static_assert(sizeof(vibeqc_xtb_compute_options_t) == VIBEQC_XTB_COMPUTE_OPTIONS_V3_SIZE,
              "vibeqc_xtb_compute_options_t must not add trailing ABI padding");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(offsetof(vibeqc_xtb_compute_options_t, scc_start_mode) == 48u,
               "vibeqc_xtb_compute_options_t ABI-v2 suffix must start at byte 48");
_Static_assert(VIBEQC_XTB_COMPUTE_OPTIONS_V1_SIZE == 48u,
               "vibeqc_xtb_compute_options_t ABI-v1 prefix must remain 48 bytes");
_Static_assert(VIBEQC_XTB_COMPUTE_OPTIONS_V2_SIZE == 56u,
               "vibeqc_xtb_compute_options_t ABI-v2 image must remain 56 bytes");
_Static_assert(offsetof(vibeqc_xtb_compute_options_t, scc_mixer) == 56u,
               "vibeqc_xtb_compute_options_t ABI-v3 mixer must start at byte 56");
_Static_assert(offsetof(vibeqc_xtb_compute_options_t, scc_mixer_history) == 60u,
               "vibeqc_xtb_compute_options_t ABI-v3 history must start at byte 60");
_Static_assert(offsetof(vibeqc_xtb_compute_options_t, scc_mixer_damping) == 64u,
               "vibeqc_xtb_compute_options_t ABI-v3 damping must start at byte 64");
_Static_assert(offsetof(vibeqc_xtb_compute_options_t, determinism) == 72u,
               "vibeqc_xtb_compute_options_t ABI-v3 determinism must start at byte 72");
_Static_assert(offsetof(vibeqc_xtb_compute_options_t, reserved_v3) == 76u,
               "vibeqc_xtb_compute_options_t ABI-v3 reserved field must start at byte 76");
_Static_assert(VIBEQC_XTB_COMPUTE_OPTIONS_V3_SIZE == 80u,
               "vibeqc_xtb_compute_options_t ABI-v3 image must remain 80 bytes");
_Static_assert(sizeof(vibeqc_xtb_compute_options_t) == VIBEQC_XTB_COMPUTE_OPTIONS_V3_SIZE,
               "vibeqc_xtb_compute_options_t must not add trailing ABI padding");
#endif

#if defined(VIBEQC_XTB_DETAIL_ABI_ASSERT) && defined(VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE)
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    VIBEQC_XTB_BATCH_V1_SIZE == VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V1_SIZE,
    "vibeqc_xtb_batch_t ABI-v1 prefix must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(VIBEQC_XTB_BATCH_V2_SIZE == VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V2_SIZE,
                             "vibeqc_xtb_batch_t ABI-v2 image must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_batch_t, total_interactions) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET,
    "vibeqc_xtb_batch_t ABI-v3 suffix must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(offsetof(vibeqc_xtb_batch_t, interaction_descriptors) ==
                                 VIBEQC_XTB_DETAIL_EXPECTED_BATCH_INTERACTION_DESCRIPTORS_OFFSET,
                             "vibeqc_xtb_batch_t descriptors must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(offsetof(vibeqc_xtb_batch_t, interaction_payload) ==
                                 VIBEQC_XTB_DETAIL_EXPECTED_BATCH_INTERACTION_PAYLOAD_OFFSET,
                             "vibeqc_xtb_batch_t payload must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(VIBEQC_XTB_BATCH_V3_SIZE == VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V3_SIZE,
                             "vibeqc_xtb_batch_t ABI-v3 image must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_batch_t, cell_matrices) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BATCH_CELL_MATRICES_OFFSET,
    "vibeqc_xtb_batch_t cell matrices must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_batch_t, periodic_axes) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BATCH_PERIODIC_AXES_OFFSET,
    "vibeqc_xtb_batch_t periodic axes must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(VIBEQC_XTB_BATCH_V4_SIZE == VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V4_SIZE,
                             "vibeqc_xtb_batch_t ABI-v4 image must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(sizeof(vibeqc_xtb_batch_t) == VIBEQC_XTB_BATCH_V4_SIZE,
                             "vibeqc_xtb_batch_t must not add trailing ABI padding");
VIBEQC_XTB_DETAIL_ABI_ASSERT(VIBEQC_XTB_INTERACTION_V1_SIZE == 32u,
                             "vibeqc_xtb_interaction_t image must remain 32 bytes");
VIBEQC_XTB_DETAIL_ABI_ASSERT(sizeof(vibeqc_xtb_interaction_t) == VIBEQC_XTB_INTERACTION_V1_SIZE,
                             "vibeqc_xtb_interaction_t must not add trailing ABI padding");
VIBEQC_XTB_DETAIL_ABI_ASSERT(offsetof(vibeqc_xtb_interaction_t, flags) == 4u,
                             "vibeqc_xtb_interaction_t flags must start at byte 4");
VIBEQC_XTB_DETAIL_ABI_ASSERT(offsetof(vibeqc_xtb_interaction_t, system_index) == 8u,
                             "vibeqc_xtb_interaction_t system index must start at byte 8");
VIBEQC_XTB_DETAIL_ABI_ASSERT(offsetof(vibeqc_xtb_interaction_t, payload_offset) == 16u,
                             "vibeqc_xtb_interaction_t payload offset must start at byte 16");
VIBEQC_XTB_DETAIL_ABI_ASSERT(offsetof(vibeqc_xtb_interaction_t, payload_size) == 24u,
                             "vibeqc_xtb_interaction_t payload size must start at byte 24");
#endif

/*
 * Caller-allocated result buffers. At finite electronic temperature, energies
 * are the total electronic Helmholtz free energy E_internal - T*S_electronic
 * in Hartree, matching the variational xTB/tblite quantity. Forces are its
 * negative coordinate derivative in Hartree/bohr. At zero temperature this
 * quantity equals the internal energy.
 *
 * A NULL data pointer is valid for an output not requested by the compute
 * flags. The three SCC diagnostic buffers are always required for a nonempty
 * batch, independent of the requested property flags. scc_iterations and
 * per_system_status store batch_size int32_t values, while scc_converged stores
 * batch_size uint8_t values.
 *
 * A VIBEQC_XTB_STATUS_SUCCESS return means every diagnostic entry was committed,
 * not necessarily that every system converged. per_system_status is SUCCESS,
 * SCC_NOT_CONVERGED, or EIGENSOLVER_FAILED; scc_converged is exactly one only
 * for SUCCESS. A failed system's requested floating-point property slices are
 * filled with quiet NaNs and never contain a partially evaluated result.
 */
typedef struct vibeqc_xtb_batch_result {
  uint32_t struct_size;
  uint32_t api_version;
  uint32_t flags;
  uint32_t reserved;
  vibeqc_xtb_buffer_t energies;
  vibeqc_xtb_buffer_t forces;
  vibeqc_xtb_buffer_t atomic_charges;
  vibeqc_xtb_buffer_t point_charge_forces;
  vibeqc_xtb_buffer_t scc_iterations;
  vibeqc_xtb_buffer_t scc_converged;
  vibeqc_xtb_buffer_t per_system_status;
  /* ABI v2 optional suffix; absent suffix preserves ABI-v1 behavior.
   *
   * dipole_moments holds batch_size * 3 doubles (atomic units) and is filled
   * when VIBEQC_XTB_COMPUTE_DIPOLE_MOMENTS is requested. The remaining outputs
   * are ABI-reserved: their shape contract is unpublished, their buffers must
   * be NULL until the matching output is released, and requesting them is
   * rejected before execution. */
  vibeqc_xtb_buffer_t dipole_moments;
  vibeqc_xtb_buffer_t quadrupole_moments;
  vibeqc_xtb_buffer_t wiberg_orders;
  vibeqc_xtb_buffer_t spin_populations;
  /* ABI v3 optional suffix. strain_derivatives contains batch_size * 9
   * row-major doubles in Hartree. The matrix is symmetric by the public
   * infinitesimal-strain convention, with duplicated off-diagonal entries.
   * It is valid only when
   * VIBEQC_XTB_COMPUTE_STRAIN_DERIVATIVES is requested for an all-native XYZ
   * CPU GFN2 batch. Older result images remain valid and never expose this
   * member. */
  vibeqc_xtb_buffer_t strain_derivatives;
} vibeqc_xtb_batch_result_t;

#define VIBEQC_XTB_BATCH_RESULT_V1_SIZE \
  (offsetof(vibeqc_xtb_batch_result_t, per_system_status) + sizeof(vibeqc_xtb_buffer_t))
#define VIBEQC_XTB_BATCH_RESULT_V2_SIZE \
  (offsetof(vibeqc_xtb_batch_result_t, spin_populations) + sizeof(vibeqc_xtb_buffer_t))
#define VIBEQC_XTB_BATCH_RESULT_V3_SIZE \
  (offsetof(vibeqc_xtb_batch_result_t, strain_derivatives) + sizeof(vibeqc_xtb_buffer_t))

#if defined(VIBEQC_XTB_DETAIL_ABI_ASSERT) && defined(VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE)
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    VIBEQC_XTB_BATCH_RESULT_V1_SIZE == VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V1_SIZE,
    "vibeqc_xtb_batch_result_t ABI-v1 prefix must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_batch_result_t, dipole_moments) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_DIPOLE_OFFSET,
    "vibeqc_xtb_batch_result_t ABI-v2 suffix must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_batch_result_t, quadrupole_moments) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_QUADRUPOLE_OFFSET,
    "vibeqc_xtb_batch_result_t quadrupole outlet must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_batch_result_t, wiberg_orders) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_WIBERG_OFFSET,
    "vibeqc_xtb_batch_result_t Wiberg outlet must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_batch_result_t, spin_populations) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_SPIN_OFFSET,
    "vibeqc_xtb_batch_result_t spin outlet must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    VIBEQC_XTB_BATCH_RESULT_V2_SIZE == VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V2_SIZE,
    "vibeqc_xtb_batch_result_t ABI-v2 image must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    offsetof(vibeqc_xtb_batch_result_t, strain_derivatives) ==
        VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_STRAIN_OFFSET,
    "vibeqc_xtb_batch_result_t strain outlet must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(
    VIBEQC_XTB_BATCH_RESULT_V3_SIZE == VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V3_SIZE,
    "vibeqc_xtb_batch_result_t ABI-v3 image must match the target pointer width");
VIBEQC_XTB_DETAIL_ABI_ASSERT(sizeof(vibeqc_xtb_batch_result_t) == VIBEQC_XTB_BATCH_RESULT_V3_SIZE,
                             "vibeqc_xtb_batch_result_t must not add trailing ABI padding");
#endif

#undef VIBEQC_XTB_DETAIL_ABI_ASSERT
#undef VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE
#undef VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_SIZE_BYTES_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_MEMORY_SPACE_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BUFFER_RESERVED_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V1_SIZE
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V2_SIZE
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_TOTAL_INTERACTIONS_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_INTERACTION_DESCRIPTORS_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_INTERACTION_PAYLOAD_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V3_SIZE
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_CELL_MATRICES_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_PERIODIC_AXES_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_V4_SIZE
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V1_SIZE
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_DIPOLE_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_QUADRUPOLE_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_WIBERG_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_SPIN_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V2_SIZE
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_STRAIN_OFFSET
#undef VIBEQC_XTB_DETAIL_EXPECTED_BATCH_RESULT_V3_SIZE

#endif /* VIBEQC_XTB_RUNTIME_TYPES_HPP */

// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/cuda/gfn2_d4.cuh"
#include "backends/cuda/gfn2_electric_field.cuh"
#include "backends/cuda/gfn2_energy_force_execution.cuh"
#include "backends/cuda/gfn2_external_point_charges.cuh"
#include "backends/cuda/gfn2_geometry.cuh"
#include "backends/cuda/gfn2_inference_publication.cuh"
#include "backends/cuda/gfn2_preprocessing.cuh"
#include "backends/cuda/gfn2_public_result_bridge.cuh"
#include "backends/cuda/gfn2_scc_iteration_arena.cuh"
#include "backends/cuda/gfn2_scc_iteration_initialize.cuh"
#include "backends/cuda/gfn2_scc_iteration_reports.cuh"
#include "backends/cuda/gfn2_scc_loop.cuh"
#include "backends/cuda/gfn2_scc_setup_eigensolver.cuh"
#include "backends/cuda/gfn2_scc_setup_inputs.cuh"
#include "backends/cuda/gfn2_scc_setup_topology.hpp"
#include "backends/cuda/gfn2_terminal_classical_energy.cuh"
#include "dft/dispersion/d4_data.hpp"
#include "model/gfn2/aes2.hpp"
#include "model/gfn2/basis.hpp"
#include "model/gfn2/coordination.hpp"
#include "model/gfn2/d4.hpp"
#include "model/gfn2/eigensolver.hpp"
#include "model/gfn2/es2.hpp"
#include "model/gfn2/es3.hpp"
#include "model/gfn2/external_point_charges.hpp"
#include "model/gfn2/h0.hpp"
#include "model/gfn2/integrals.hpp"
#include "model/gfn2/mulliken.hpp"
#include "model/gfn2/periodic_embedding.hpp"
#include "model/gfn2/repulsion.hpp"
#include "model/gfn2/scc_driver.hpp"
#include "model/gfn2/scc_mixer.hpp"
#include "model/gfn2/wavefunction.hpp"
#include "runtime/backend.hpp"
#include "runtime/cuda_descriptor_validation.hpp"
#include "runtime/gfn2_cuda_execution.hpp"
#include "runtime/gfn2_cuda_topology_staging.hpp"
#include "runtime/molecular_request.hpp"
#include "runtime/nvidia_host_api.h"

namespace vibeqc::xtb::detail {
namespace {

using namespace vibeqc::xtb::detail::cuda;
using namespace vibeqc::xtb::detail::gfn2;

constexpr std::int32_t kDefaultMixerHistory = 8;
constexpr double kDefaultMixerDamping = 0.4;
constexpr std::uint64_t kInitialGeometryGeneration = 1u;
constexpr std::uint64_t kInitialStateGeneration = 1u;
constexpr std::size_t kArenaAlignment = 256u;
/* One committed physical superset serves every pair consumer.  D4 two-body
 * reaches 50 bohr; narrower roles re-evaluate positions with their own
 * inclusive 25/30-bohr predicates and never infer physical membership solely
 * from list presence. */
constexpr double kD4PairlistBuilderCutoffBohr = 50.0;

/* ABI-v3 mixer and reproducibility controls form one complete suffix. A
 * caller that supplies only a prefix of the suffix receives the established
 * production defaults; no individual field becomes visible early. */
bool has_compute_options_v3(const vibeqc_xtb_compute_options_t& options) noexcept {
  return options.struct_size >= VIBEQC_XTB_COMPUTE_OPTIONS_V3_SIZE;
}

vibeqc_xtb_scc_mixer_t public_scc_mixer(const vibeqc_xtb_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.scc_mixer
                                         : VIBEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN;
}

std::int32_t public_scc_mixer_history(const vibeqc_xtb_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.scc_mixer_history : kDefaultMixerHistory;
}

double public_scc_mixer_damping(const vibeqc_xtb_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.scc_mixer_damping : kDefaultMixerDamping;
}

vibeqc_xtb_determinism_t public_determinism(const vibeqc_xtb_compute_options_t& options) noexcept {
  return has_compute_options_v3(options) ? options.determinism : VIBEQC_XTB_DETERMINISM_DEFAULT;
}

vibeqc_xtb_status_t validate_public_execution_policy(const vibeqc_xtb_compute_options_t& options,
                                                     std::string& error) {
  if (!has_compute_options_v3(options)) return VIBEQC_XTB_STATUS_SUCCESS;
  if (options.scc_mixer != VIBEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN) {
    error = "CUDA GFN2 setup supports only modified-Broyden SCC mixing";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  if (options.scc_mixer_history < 1 || options.scc_mixer_history > 64) {
    error = "CUDA GFN2 SCC mixer history must be between 1 and 64";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  if (!std::isfinite(options.scc_mixer_damping) || options.scc_mixer_damping <= 0.0 ||
      options.scc_mixer_damping > 1.0) {
    error = "CUDA GFN2 SCC mixer damping must be finite and in (0, 1]";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  if (options.determinism != VIBEQC_XTB_DETERMINISM_DEFAULT &&
      options.determinism != VIBEQC_XTB_DETERMINISM_REPRODUCIBLE) {
    error = "CUDA GFN2 determinism policy is unknown";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  if (options.reserved_v3 != 0u) {
    error = "CUDA GFN2 compute-options reserved_v3 field must be zero";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_XTB_STATUS_SUCCESS;
}

template <typename T>
Gfn2SccSetupHostArray<T> setup_array(const std::vector<T>& values) noexcept {
  return {values.empty() ? nullptr : values.data(), static_cast<std::int64_t>(values.size())};
}

template <typename T>
Gfn2SccIterationHostArrayView<T> initialization_array(const T* values,
                                                      std::int64_t elements) noexcept {
  return {elements == 0 ? nullptr : values, elements};
}

bool checked_bytes(std::int64_t elements, std::size_t element_size, std::size_t& bytes) noexcept {
  if (elements < 0) {
    return false;
  }
  const auto count = static_cast<std::uint64_t>(elements);
  if (count > std::numeric_limits<std::size_t>::max() / element_size) {
    return false;
  }
  bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_address_range(const void* pointer, std::size_t bytes, AddressRange& range) noexcept {
  if (pointer == nullptr || bytes == 0u) return false;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > UINTPTR_MAX - begin) return false;
  range = {begin, begin + bytes};
  return true;
}

bool address_ranges_overlap(const AddressRange& first, const AddressRange& second) noexcept {
  return first.begin < second.end && second.begin < first.end;
}

bool address_range_contains(const AddressRange& owner, const AddressRange& child) noexcept {
  return owner.begin <= child.begin && child.end <= owner.end;
}

bool checked_elements(std::int64_t elements, std::int64_t factor, std::int64_t& product) noexcept {
  if (elements < 0 || factor < 0 ||
      (factor != 0 && elements > std::numeric_limits<std::int64_t>::max() / factor)) {
    return false;
  }
  product = elements * factor;
  return true;
}

bool checked_triangle(std::int64_t atoms, std::int64_t& pairs) noexcept {
  if (atoms < 0) {
    return false;
  }
  if (atoms <= 1) {
    pairs = 0;
    return true;
  }
  const std::int64_t lower = atoms - 1;
  if ((atoms & 1) == 0) {
    return checked_elements(atoms / 2, lower, pairs);
  }
  return checked_elements(atoms, lower / 2, pairs);
}

std::uint64_t hash_mix(std::uint64_t value) noexcept {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31u);
}

void hash_append(std::uint64_t& hash, std::uint64_t value) noexcept {
  hash ^= hash_mix(value + 0x9e3779b97f4a7c15ULL + (hash << 6u) + (hash >> 2u));
}

std::uint64_t double_bits(double value) noexcept {
  std::uint64_t bits = 0u;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename T>
void hash_append_vector(std::uint64_t& hash, const std::vector<T>& values) noexcept {
  hash_append(hash, values.size());
  for (const T value : values) {
    if constexpr (std::is_same_v<T, double>) {
      hash_append(hash, double_bits(value));
    } else {
      hash_append(hash, static_cast<std::uint64_t>(value));
    }
  }
}

class DeviceArena {
 public:
  DeviceArena() = default;
  ~DeviceArena() { reset(); }
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  DeviceArena(DeviceArena&&) = delete;
  DeviceArena& operator=(DeviceArena&&) = delete;

  cudaError_t allocate(std::size_t bytes) noexcept {
    reset();
    bytes_ = bytes;
    return bytes == 0u ? cudaSuccess : cudaMalloc(&pointer_, bytes);
  }

  void reset() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
    pointer_ = nullptr;
    bytes_ = 0u;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class PinnedArena {
 public:
  PinnedArena() = default;
  ~PinnedArena() { reset(); }
  PinnedArena(const PinnedArena&) = delete;
  PinnedArena& operator=(const PinnedArena&) = delete;
  PinnedArena(PinnedArena&&) = delete;
  PinnedArena& operator=(PinnedArena&&) = delete;

  cudaError_t allocate(std::size_t bytes, bool inject_allocation_failure = false) noexcept {
    if (bytes == 0u) {
      reset();
      return cudaSuccess;
    }

    /* Preserve the old image until a complete replacement exists. In
     * particular, a failed growth must not publish a nonzero capacity beside
     * a null pointer, because callers use capacity to decide whether the
     * retained image is reusable. */
    void* replacement = nullptr;
    const cudaError_t status =
        inject_allocation_failure ? cudaErrorMemoryAllocation : cudaMallocHost(&replacement, bytes);
    if (status != cudaSuccess) return status;

    void* const previous = pointer_;
    pointer_ = replacement;
    bytes_ = bytes;
    if (previous != nullptr) (void)cudaFreeHost(previous);
    return cudaSuccess;
  }

  void reset() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFreeHost(pointer_);
    }
    pointer_ = nullptr;
    bytes_ = 0u;
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

  /* Relinquish an image that cannot be proved idle. CUDA may still reference
   * it, so the caller deliberately leaves it allocated instead of risking a
   * use-after-free in a noexcept teardown path. */

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0u;
};

class CudaEvent {
 public:
  CudaEvent() = default;
  ~CudaEvent() { reset(); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&&) = delete;
  CudaEvent& operator=(CudaEvent&&) = delete;

  cudaError_t create(unsigned int flags) noexcept {
    reset();
    return cudaEventCreateWithFlags(&event_, flags);
  }

  void reset() noexcept {
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
    event_ = nullptr;
  }

  cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream() { reset(); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&&) = delete;
  CudaStream& operator=(CudaStream&&) = delete;

  cudaError_t create(unsigned int flags) noexcept {
    reset();
    return cudaStreamCreateWithFlags(&stream_, flags);
  }

  void reset() noexcept {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
    stream_ = nullptr;
  }

  cudaStream_t get() const noexcept { return stream_; }
  bool valid() const noexcept { return stream_ != nullptr; }

 private:
  cudaStream_t stream_ = nullptr;
};

bool align_up(std::size_t value, std::size_t alignment, std::size_t& aligned) noexcept {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  const std::size_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    return false;
  }
  aligned = (value + mask) & ~mask;
  return true;
}

class ArenaLayout {
 public:
  template <typename T, typename Count>
  std::size_t append(Count elements) {
    static_assert(std::is_integral_v<Count>, "arena element counts must be integral");
    if constexpr (std::is_signed_v<Count>) {
      if (elements < 0) {
        valid_ = false;
        return 0u;
      }
    }
    using UnsignedCount = std::make_unsigned_t<Count>;
    const auto count = static_cast<UnsignedCount>(elements);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      valid_ = false;
      return 0u;
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(T);
    const std::size_t alignment = std::max<std::size_t>(alignof(T), 8u);
    std::size_t offset = 0u;
    if (!align_up(bytes_, alignment, offset)) {
      valid_ = false;
      return 0u;
    }
    if (offset > std::numeric_limits<std::size_t>::max() - bytes) {
      valid_ = false;
      return 0u;
    }
    bytes_ = offset + bytes;
    return offset;
  }

  bool valid() const noexcept {
    std::size_t ignored = 0u;
    return valid_ && align_up(bytes_, kArenaAlignment, ignored);
  }
  std::size_t bytes() const noexcept {
    std::size_t aligned = 0u;
    return valid_ && align_up(bytes_, kArenaAlignment, aligned) ? aligned : 0u;
  }

 private:
  std::size_t bytes_ = 0u;
  bool valid_ = true;
};

template <typename T>
T* arena_pointer(void* arena, std::size_t offset) noexcept {
  return reinterpret_cast<T*>(static_cast<std::byte*>(arena) + offset);
}

template <typename T>
T* arena_pointer_if(void* arena, std::size_t offset, std::int64_t elements) noexcept {
  return elements == 0 ? nullptr : arena_pointer<T>(arena, offset);
}

/* Stable numerical leaves and transaction metadata passed by value to the
 * runtime-owned refresh kernels. CUDA types stay confined to this translation
 * unit; the public internal boundary above remains usable by a future HIP
 * implementation. */
struct NumericalRefreshDeviceSources {
  const double* positions = nullptr;
  const double* point_positions = nullptr;
  const double* point_values = nullptr;
  const double* point_gammas = nullptr;
  const double* periodic_shifts = nullptr;
  const double* periodic_response = nullptr;
  const vibeqc_xtb_interaction_t* interaction_descriptors = nullptr;
  const std::byte* interaction_payload = nullptr;
  /* Range/alignment validation always refers to the caller's declared
   * payload view.  When a HOST payload is compacted into the fixed staging
   * arena, field bytes are instead addressed by system_index below. */
  std::uintptr_t interaction_payload_address = 0u;
  std::uint64_t interaction_payload_bytes = 0u;
  std::int64_t total_interactions = 0;
  std::uint32_t prevalidated_interaction_error = kGfn2RequestErrorNone;
  std::uint8_t interaction_payload_compacted_by_system = 0u;
  /* A HOST payload paired with device descriptors is snapshotted at identical
   * offsets when it fits the released fixed-capacity arena.  Device semantic
   * validation still uses the caller view's address for alignment checks, but
   * reads bytes from this plan-owned device image after enqueue returns. */
  std::uint8_t interaction_payload_staged_at_offsets = 0u;
  std::uint8_t prevalidated_reserved_interaction = 0u;
};

/* The start-mode scalar is written by a kernel argument so steady-state
 * admission neither transfers host bytes nor retains a caller-owned pointer. */
struct NumericalRefreshDeviceBinding {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrices = 0;
  std::int64_t total_point_charges = 0;
  std::int64_t total_response_elements = 0;
  std::int64_t geometry_pair_elements = 0;
  std::int64_t es2_elements = 0;
  std::int64_t aes2_elements = 0;
  std::uint8_t d4_enabled = 0u;
  std::uint8_t point_enabled = 0u;
  std::uint8_t periodic_enabled = 0u;
  std::uint64_t plan_token = 0u;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* shell_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* geometry_pair_offsets = nullptr;
  const std::int64_t* es2_offsets = nullptr;
  const std::int64_t* point_offsets = nullptr;
  const std::int64_t* response_offsets = nullptr;
  const std::uint8_t* requested = nullptr;
  const std::uint8_t* preprocessing_published = nullptr;
  std::uint8_t* eligible = nullptr;
  /* Canonical eigensolver/SCC activity leaf snapshotted by overlap refactor. */
  std::uint8_t* factor_active = nullptr;
  std::uint64_t* committed_generations = nullptr;
  /* The immediately preceding successfully committed numerical epoch. Warm
   * reset uses this edge instead of rewriting checkpoint generations, so two
   * refreshes without an intervening inference cannot chain a stale state. */
  std::uint64_t* refresh_predecessor_generations = nullptr;
  /* Bound after the inference arena is built. A failed refresh must revoke
   * both the predecessor edge and its consumable warm checkpoint. */
  std::uint64_t* warm_checkpoint_generations = nullptr;

  double* candidate_positions = nullptr;
  double* candidate_point_positions = nullptr;
  double* candidate_point_values = nullptr;
  double* candidate_point_gammas = nullptr;
  double* candidate_periodic_shifts = nullptr;
  double* candidate_periodic_response = nullptr;

  double* committed_positions = nullptr;
  double* committed_point_positions = nullptr;
  double* committed_point_values = nullptr;
  double* committed_point_gammas = nullptr;
  double* committed_periodic_shifts = nullptr;
  double* committed_periodic_response = nullptr;

  /* Electric-field attachments are dense normalized numerical state. Exact
   * presence is retained separately so absent and attached-zero remain
   * distinct under strict WARM. */
  std::uint8_t* candidate_field_attached = nullptr;
  double* candidate_field_vectors = nullptr;
  double* candidate_field_atomic_potentials = nullptr;
  double* candidate_field_dipole_potentials = nullptr;
  std::uint8_t* committed_field_attached = nullptr;
  double* committed_field_vectors = nullptr;
  double* committed_field_atomic_potentials = nullptr;
  double* committed_field_dipole_potentials = nullptr;
  std::uint8_t* warm_checkpoint_field_attached = nullptr;
  double* warm_checkpoint_field_vectors = nullptr;
  std::int64_t total_interactions = 0;
  const vibeqc_xtb_interaction_t* interaction_descriptors = nullptr;
  const std::byte* interaction_payload = nullptr;
  std::uint64_t interaction_payload_bytes = 0u;
  std::uint32_t* interaction_request_error = nullptr;
  std::uint32_t* request_error = nullptr;
  std::uint32_t* field_system_errors = nullptr;
  std::uint32_t* field_plan_error = nullptr;

  const double* candidate_geometry_pairs = nullptr;
  const double* candidate_coordination = nullptr;
  const double* candidate_overlap = nullptr;
  const double* candidate_dipole = nullptr;
  const double* candidate_quadrupole = nullptr;
  const double* candidate_h0 = nullptr;
  const double* candidate_es2 = nullptr;
  const double* candidate_aes2 = nullptr;

  double* public_geometry_pairs = nullptr;
  double* public_coordination = nullptr;
  double* public_overlap = nullptr;
  double* public_dipole = nullptr;
  double* public_quadrupole = nullptr;
  double* public_h0 = nullptr;
  double* public_es2 = nullptr;
  double* public_aes2 = nullptr;

  const double* candidate_d4_coordination = nullptr;
  double* public_d4_coordination = nullptr;
  const double* candidate_point_shell = nullptr;
  double* public_point_shell = nullptr;

  const std::uint32_t* preprocessing_plan_error = nullptr;
  const std::uint32_t* d4_system_errors = nullptr;
  const std::uint32_t* d4_device_error = nullptr;
  const std::uint32_t* point_system_errors = nullptr;
  const std::uint32_t* point_plan_error = nullptr;
  std::uint32_t* periodic_system_errors = nullptr;
  std::uint32_t* periodic_plan_error = nullptr;
  const std::uint64_t* factor_generations = nullptr;
  const std::uint32_t* factor_statuses = nullptr;
  const std::uint64_t* geometry_epoch = nullptr;
};

static_assert(std::is_trivially_copyable_v<NumericalRefreshDeviceSources>);
static_assert(std::is_trivially_copyable_v<NumericalRefreshDeviceBinding>);

__global__ void stage_gfn2_numerical_inputs_kernel(NumericalRefreshDeviceBinding binding,
                                                   NumericalRefreshDeviceSources sources) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const bool use_source = binding.requested[system] == 1u;

  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  for (std::int64_t index = atom_begin * 3 + threadIdx.x; index < atom_end * 3;
       index += blockDim.x) {
    binding.candidate_positions[index] =
        use_source ? sources.positions[index] : binding.committed_positions[index];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    if (binding.periodic_enabled != 0u) {
      binding.candidate_periodic_shifts[atom] =
          use_source ? sources.periodic_shifts[atom] : binding.committed_periodic_shifts[atom];
    }
  }

  if (binding.point_enabled != 0u) {
    const std::int64_t point_begin = binding.point_offsets[system];
    const std::int64_t point_end = binding.point_offsets[system + 1];
    for (std::int64_t index = point_begin * 3 + threadIdx.x; index < point_end * 3;
         index += blockDim.x) {
      binding.candidate_point_positions[index] =
          use_source ? sources.point_positions[index] : binding.committed_point_positions[index];
    }
    for (std::int64_t point = point_begin + threadIdx.x; point < point_end; point += blockDim.x) {
      binding.candidate_point_values[point] =
          use_source ? sources.point_values[point] : binding.committed_point_values[point];
      binding.candidate_point_gammas[point] =
          use_source ? sources.point_gammas[point] : binding.committed_point_gammas[point];
    }
  }

  if (binding.periodic_enabled != 0u) {
    const std::int64_t response_begin = binding.response_offsets[system];
    const std::int64_t response_end = binding.response_offsets[system + 1];
    for (std::int64_t index = response_begin + threadIdx.x; index < response_end;
         index += blockDim.x) {
      binding.candidate_periodic_response[index] = use_source
                                                       ? sources.periodic_response[index]
                                                       : binding.committed_periodic_response[index];
    }
  }
}

std::uint32_t interaction_type_mask_host(std::int32_t type) noexcept {
  switch (type) {
    case VIBEQC_XTB_INTERACTION_ELECTRIC_FIELD:
      return UINT32_C(1) << 0;
    case VIBEQC_XTB_INTERACTION_ELECTRIC_FIELD_GRADIENT:
      return UINT32_C(1) << 1;
    case VIBEQC_XTB_INTERACTION_POINT_CHARGES_MULTIPOLE:
      return UINT32_C(1) << 2;
    case VIBEQC_XTB_INTERACTION_ATOMIC_POTENTIAL_GRID:
      return UINT32_C(1) << 3;
    case VIBEQC_XTB_INTERACTION_ALPB_SOLVATION:
      return UINT32_C(1) << 4;
    case VIBEQC_XTB_INTERACTION_GBSA_SOLVATION:
      return UINT32_C(1) << 5;
    case VIBEQC_XTB_INTERACTION_GB_SOLVATION:
      return UINT32_C(1) << 6;
    case VIBEQC_XTB_INTERACTION_GBE_SOLVATION:
      return UINT32_C(1) << 7;
    case VIBEQC_XTB_INTERACTION_DDX_SOLVATION:
      return UINT32_C(1) << 8;
    case VIBEQC_XTB_INTERACTION_D3_DISPERSION:
      return UINT32_C(1) << 9;
    case VIBEQC_XTB_INTERACTION_D4_VARIANT_DISPERSION:
      return UINT32_C(1) << 10;
    case VIBEQC_XTB_INTERACTION_HALOGEN_BOND:
      return UINT32_C(1) << 11;
    default:
      return 0u;
  }
}

__device__ bool known_interaction_type_device(std::int32_t type) noexcept {
  switch (type) {
    case VIBEQC_XTB_INTERACTION_ELECTRIC_FIELD:
    case VIBEQC_XTB_INTERACTION_ELECTRIC_FIELD_GRADIENT:
    case VIBEQC_XTB_INTERACTION_POINT_CHARGES_MULTIPOLE:
    case VIBEQC_XTB_INTERACTION_ATOMIC_POTENTIAL_GRID:
    case VIBEQC_XTB_INTERACTION_ALPB_SOLVATION:
    case VIBEQC_XTB_INTERACTION_GBSA_SOLVATION:
    case VIBEQC_XTB_INTERACTION_GB_SOLVATION:
    case VIBEQC_XTB_INTERACTION_GBE_SOLVATION:
    case VIBEQC_XTB_INTERACTION_DDX_SOLVATION:
    case VIBEQC_XTB_INTERACTION_D3_DISPERSION:
    case VIBEQC_XTB_INTERACTION_D4_VARIANT_DISPERSION:
    case VIBEQC_XTB_INTERACTION_HALOGEN_BOND:
      return true;
    default:
      return false;
  }
}

/* Normalize arbitrary descriptor/payload placement into one dense field per
 * system. The single-block scan is deliberately bounded by the public
 * descriptor count and performs no compact-payload assumption. The shared
 * request error codes are consumed by both the persistent-state admission gate
 * and the public result bridge. */
__global__ void normalize_gfn2_interactions_kernel(NumericalRefreshDeviceBinding binding,
                                                   NumericalRefreshDeviceSources sources) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (sources.prevalidated_interaction_error != kGfn2RequestErrorNone) {
    *binding.interaction_request_error = sources.prevalidated_interaction_error;
    return;
  }
  for (std::int64_t system = 0; system < binding.batch_size; ++system) {
    const bool retain = binding.requested[system] == 0u;
    binding.candidate_field_attached[system] =
        retain ? binding.committed_field_attached[system] : 0u;
    binding.candidate_field_vectors[3 * system] =
        retain ? binding.committed_field_vectors[3 * system] : 0.0;
    binding.candidate_field_vectors[3 * system + 1] =
        retain ? binding.committed_field_vectors[3 * system + 1] : 0.0;
    binding.candidate_field_vectors[3 * system + 2] =
        retain ? binding.committed_field_vectors[3 * system + 2] : 0.0;
  }
  bool reserved_tag = sources.prevalidated_reserved_interaction != 0u;
  for (std::int64_t index = 0; index < sources.total_interactions; ++index) {
    const vibeqc_xtb_interaction_t item = sources.interaction_descriptors[index];
    if (item.flags != 0u || item.type == VIBEQC_XTB_INTERACTION_NONE ||
        !known_interaction_type_device(item.type) || item.system_index < 0 ||
        item.system_index >= binding.batch_size ||
        item.payload_offset > sources.interaction_payload_bytes ||
        item.payload_size > sources.interaction_payload_bytes - item.payload_offset) {
      *binding.interaction_request_error = kGfn2RequestErrorInvalid;
      return;
    }
    const std::uintptr_t payload_base = sources.interaction_payload_address;
    if (item.payload_offset > static_cast<std::uint64_t>(UINTPTR_MAX - payload_base)) {
      *binding.interaction_request_error = kGfn2RequestErrorInvalid;
      return;
    }
    const std::uintptr_t block_address =
        payload_base + static_cast<std::uintptr_t>(item.payload_offset);
    if (item.type != VIBEQC_XTB_INTERACTION_ELECTRIC_FIELD) {
      if (item.payload_size < sizeof(std::int32_t) || block_address % alignof(std::int32_t) != 0u) {
        *binding.interaction_request_error = kGfn2RequestErrorInvalid;
        return;
      } else {
        reserved_tag = true;
      }
      continue;
    }
    if (item.payload_size != 32u || block_address % alignof(double) != 0u) {
      *binding.interaction_request_error = kGfn2RequestErrorInvalid;
      return;
    }
    const auto* block = sources.interaction_payload_compacted_by_system != 0u
                            ? sources.interaction_payload + 32u * item.system_index
                            : (sources.interaction_payload_staged_at_offsets != 0u
                                   ? sources.interaction_payload + item.payload_offset
                                   : reinterpret_cast<const std::byte*>(block_address));
    const auto version = *reinterpret_cast<const std::int32_t*>(block);
    const auto reserved = *reinterpret_cast<const std::int32_t*>(block + sizeof(std::int32_t));
    const auto* field = reinterpret_cast<const double*>(block + 2 * sizeof(std::int32_t));
    if (version != 1 || reserved != 0 || !isfinite(field[0]) || !isfinite(field[1]) ||
        !isfinite(field[2])) {
      *binding.interaction_request_error = kGfn2RequestErrorInvalid;
      return;
    }
  }
  for (std::int64_t lhs = 0; lhs < sources.total_interactions; ++lhs) {
    const vibeqc_xtb_interaction_t left = sources.interaction_descriptors[lhs];
    for (std::int64_t rhs = lhs + 1; rhs < sources.total_interactions; ++rhs) {
      const vibeqc_xtb_interaction_t right = sources.interaction_descriptors[rhs];
      if (left.system_index == right.system_index && left.type == right.type) {
        *binding.interaction_request_error = kGfn2RequestErrorInvalid;
        return;
      }
    }
  }
  if (reserved_tag) {
    *binding.interaction_request_error = kGfn2RequestErrorNotImplemented;
    return;
  }
  for (std::int64_t index = 0; index < sources.total_interactions; ++index) {
    const vibeqc_xtb_interaction_t item = sources.interaction_descriptors[index];
    if (binding.requested[item.system_index] == 0u) continue;
    const auto* block = sources.interaction_payload_compacted_by_system != 0u
                            ? sources.interaction_payload + 32u * item.system_index
                            : (sources.interaction_payload_staged_at_offsets != 0u
                                   ? sources.interaction_payload + item.payload_offset
                                   : reinterpret_cast<const std::byte*>(
                                         sources.interaction_payload_address +
                                         static_cast<std::uintptr_t>(item.payload_offset)));
    const auto* field = reinterpret_cast<const double*>(block + 2 * sizeof(std::int32_t));
    binding.candidate_field_attached[item.system_index] = 1u;
    binding.candidate_field_vectors[3 * item.system_index] = field[0];
    binding.candidate_field_vectors[3 * item.system_index + 1] = field[1];
    binding.candidate_field_vectors[3 * item.system_index + 2] = field[2];
  }
}

__global__ void merge_gfn2_request_error_kernel(const std::uint32_t* interaction_error,
                                                std::uint32_t* request_error) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::uint32_t error = *interaction_error;
    if (error != 0u) atomicMax(request_error, error);
  }
}

/* Strict WARM compatibility is a call-wide admission decision. It is checked
 * after interaction normalization has produced the candidate field image but
 * before preprocessing advances the epoch or any persistent cache publishes.
 * A separate validation pass avoids the peer race that would occur if one
 * block started consuming its checkpoint while another discovered a mismatch. */
__global__ void validate_gfn2_warm_field_identity_kernel(
    NumericalRefreshDeviceBinding binding, const std::uint8_t* checkpoint_field_attached,
    const double* checkpoint_field_vectors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.batch_size || *binding.request_error != kGfn2RequestErrorNone) return;
  const bool same_field =
      binding.candidate_field_attached[system] == checkpoint_field_attached[system] &&
      binding.candidate_field_vectors[3 * system] == checkpoint_field_vectors[3 * system] &&
      binding.candidate_field_vectors[3 * system + 1] == checkpoint_field_vectors[3 * system + 1] &&
      binding.candidate_field_vectors[3 * system + 2] == checkpoint_field_vectors[3 * system + 2];
  if (!same_field) {
    atomicCAS(binding.request_error, kGfn2RequestErrorNone, kGfn2RequestErrorWarmIncompatible);
  }
}

__global__ void validate_gfn2_periodic_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size || binding.periodic_enabled == 0u ||
      binding.preprocessing_published[system] == 0u ||
      atomicAdd(binding.periodic_plan_error, 0u) != 0u) {
    return;
  }
  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    if (!isfinite(binding.candidate_periodic_shifts[atom])) {
      atomicCAS(binding.periodic_system_errors + system, 0u, 1u);
    }
  }
  __syncthreads();
  if (atomicAdd(binding.periodic_system_errors + system, 0u) != 0u) return;

  const std::int64_t count = atom_end - atom_begin;
  const std::int64_t response_begin = binding.response_offsets[system];
  const std::int64_t response_end = binding.response_offsets[system + 1];
  if (count < 0 || count > 3037000499LL || response_begin < 0 || response_end < response_begin ||
      response_end - response_begin != count * count) {
    if (threadIdx.x == 0) atomicCAS(binding.periodic_plan_error, 0u, 1u);
    return;
  }
  for (std::int64_t local = threadIdx.x; local < count * count; local += blockDim.x) {
    const double value = binding.candidate_periodic_response[response_begin + local];
    if (!isfinite(value)) {
      atomicCAS(binding.periodic_system_errors + system, 0u, 2u);
      continue;
    }
    const std::int64_t row = local / count;
    const std::int64_t column = local - row * count;
    const double transpose =
        binding.candidate_periodic_response[response_begin + column * count + row];
    if (value != transpose) {
      atomicCAS(binding.periodic_system_errors + system, 0u, 3u);
    }
  }
}

__global__ void initialize_gfn2_refresh_sequence_kernel(std::uint32_t* sequence) {
  if (threadIdx.x == 0) *sequence = 1u;
}

/* Publish the common pre-factor gate into the eigensolver owner's canonical
 * activity mask. The overlap owner then commits factors only for these peers. */
__global__ void gate_gfn2_numerical_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  /* Request-level semantic rejection preserves the prior canonical activity
   * and factor-cache transaction. Candidate diagnostics may still have been
   * overwritten earlier on the stream, but no persistent eligibility changes. */
  const Gfn2DeviceAdmission admission{binding.request_error, 1, binding.plan_token};
  if (!gfn2_request_admitted(admission)) return;
  const bool plan_healthy =
      atomicAdd(const_cast<std::uint32_t*>(binding.preprocessing_plan_error), 0u) == 0u &&
      (binding.d4_enabled == 0u ||
       atomicAdd(const_cast<std::uint32_t*>(binding.d4_device_error), 0u) == 0u) &&
      (binding.point_enabled == 0u ||
       atomicAdd(const_cast<std::uint32_t*>(binding.point_plan_error), 0u) == 0u) &&
      (binding.periodic_enabled == 0u || atomicAdd(binding.periodic_plan_error, 0u) == 0u) &&
      gfn2_request_admitted(admission) && atomicAdd(binding.field_plan_error, 0u) == 0u;
  const bool peer_healthy =
      binding.requested[system] == 1u && binding.preprocessing_published[system] == 1u &&
      (binding.d4_enabled == 0u || binding.d4_system_errors[system] == 0u) &&
      (binding.point_enabled == 0u || binding.point_system_errors[system] == 0u) &&
      (binding.periodic_enabled == 0u || binding.periodic_system_errors[system] == 0u) &&
      binding.field_system_errors[system] == 0u;
  if (threadIdx.x == 0) {
    const std::uint8_t eligible = plan_healthy && peer_healthy ? 1u : 0u;
    binding.eligible[system] = eligible;
    binding.factor_active[system] = eligible;
  }
}

__global__ void commit_gfn2_numerical_refresh_kernel(NumericalRefreshDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const Gfn2DeviceAdmission admission{binding.request_error, 1, binding.plan_token};
  if (!gfn2_request_admitted(admission)) return;
  const std::uint64_t generation = *binding.geometry_epoch;
  const std::uint64_t previous_generation = binding.committed_generations[system];
  const bool publish = binding.eligible[system] == 1u && binding.factor_statuses[system] == 0u &&
                       binding.factor_generations[system] == generation;
  if (threadIdx.x == 0) {
    binding.eligible[system] = publish ? 1u : 0u;
    binding.refresh_predecessor_generations[system] = publish ? previous_generation : 0u;
    if (!publish) binding.warm_checkpoint_generations[system] = 0u;
  }
  if (!publish) return;

  if (threadIdx.x == 0) {
    binding.committed_field_attached[system] = binding.candidate_field_attached[system];
    binding.committed_field_vectors[3 * system] = binding.candidate_field_vectors[3 * system];
    binding.committed_field_vectors[3 * system + 1] =
        binding.candidate_field_vectors[3 * system + 1];
    binding.committed_field_vectors[3 * system + 2] =
        binding.candidate_field_vectors[3 * system + 2];
  }

  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  for (std::int64_t index = atom_begin * 3 + threadIdx.x; index < atom_end * 3;
       index += blockDim.x) {
    binding.committed_positions[index] = binding.candidate_positions[index];
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    binding.committed_field_atomic_potentials[atom] =
        binding.candidate_field_atomic_potentials[atom];
    binding.committed_field_dipole_potentials[3 * atom] =
        binding.candidate_field_dipole_potentials[3 * atom];
    binding.committed_field_dipole_potentials[3 * atom + 1] =
        binding.candidate_field_dipole_potentials[3 * atom + 1];
    binding.committed_field_dipole_potentials[3 * atom + 2] =
        binding.candidate_field_dipole_potentials[3 * atom + 2];
    binding.public_coordination[atom] = binding.candidate_coordination[atom];
    if (binding.d4_enabled != 0u) {
      binding.public_d4_coordination[atom] = binding.candidate_d4_coordination[atom];
    }
    if (binding.periodic_enabled != 0u) {
      binding.committed_periodic_shifts[atom] = binding.candidate_periodic_shifts[atom];
    }
  }

  const std::int64_t matrix_begin = binding.matrix_offsets[system];
  const std::int64_t matrix_end = binding.matrix_offsets[system + 1];
  for (std::int64_t index = matrix_begin + threadIdx.x; index < matrix_end; index += blockDim.x) {
    binding.public_overlap[index] = binding.candidate_overlap[index];
    binding.public_h0[index] = binding.candidate_h0[index];
  }
  /* Integral multipoles are global component-major arrays.  A peer owns its
   * matrix interval in every component plane; treating 3*M or 6*M as one
   * contiguous per-peer interval would publish bytes belonging to adjacent
   * peers and violate transactional rollback. */
  for (std::int64_t matrix = matrix_begin + threadIdx.x; matrix < matrix_end;
       matrix += blockDim.x) {
    for (std::int64_t component = 0; component < 3; ++component) {
      const std::int64_t index = component * binding.total_matrices + matrix;
      binding.public_dipole[index] = binding.candidate_dipole[index];
    }
    for (std::int64_t component = 0; component < 6; ++component) {
      const std::int64_t index = component * binding.total_matrices + matrix;
      binding.public_quadrupole[index] = binding.candidate_quadrupole[index];
    }
  }

  const std::int64_t pair_begin = binding.geometry_pair_offsets[system];
  const std::int64_t pair_end = binding.geometry_pair_offsets[system + 1];
  for (std::int64_t index = pair_begin * kGfn2GeometryPairDataElements + threadIdx.x;
       index < pair_end * kGfn2GeometryPairDataElements; index += blockDim.x) {
    binding.public_geometry_pairs[index] = binding.candidate_geometry_pairs[index];
  }
  for (std::int64_t index = pair_begin * kGfn2AES2PairDataElements + threadIdx.x;
       index < pair_end * kGfn2AES2PairDataElements; index += blockDim.x) {
    binding.public_aes2[index] = binding.candidate_aes2[index];
  }
  const std::int64_t es2_begin = binding.es2_offsets[system];
  const std::int64_t es2_end = binding.es2_offsets[system + 1];
  for (std::int64_t index = es2_begin + threadIdx.x; index < es2_end; index += blockDim.x) {
    binding.public_es2[index] = binding.candidate_es2[index];
  }

  if (binding.point_enabled != 0u) {
    const std::int64_t point_begin = binding.point_offsets[system];
    const std::int64_t point_end = binding.point_offsets[system + 1];
    for (std::int64_t index = point_begin * 3 + threadIdx.x; index < point_end * 3;
         index += blockDim.x) {
      binding.committed_point_positions[index] = binding.candidate_point_positions[index];
    }
    for (std::int64_t point = point_begin + threadIdx.x; point < point_end; point += blockDim.x) {
      binding.committed_point_values[point] = binding.candidate_point_values[point];
      binding.committed_point_gammas[point] = binding.candidate_point_gammas[point];
    }
    const std::int64_t shell_begin = binding.shell_offsets[system];
    const std::int64_t shell_end = binding.shell_offsets[system + 1];
    for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
      binding.public_point_shell[shell] = binding.candidate_point_shell[shell];
    }
  }
  if (binding.periodic_enabled != 0u) {
    const std::int64_t response_begin = binding.response_offsets[system];
    const std::int64_t response_end = binding.response_offsets[system + 1];
    for (std::int64_t index = response_begin + threadIdx.x; index < response_end;
         index += blockDim.x) {
      binding.committed_periodic_response[index] = binding.candidate_periodic_response[index];
    }
  }
  if (threadIdx.x == 0) binding.committed_generations[system] = generation;
}

struct WarmCheckpointPublicationDeviceBinding {
  std::int64_t batch_size = 0;
  Gfn2GeometryEpochDevice geometry_epoch{};
  const std::uint8_t* eligible = nullptr;
  const std::uint64_t* committed_generations = nullptr;
  const std::uint32_t* publication_plan_error = nullptr;
  const vibeqc_xtb_status_t* result_statuses = nullptr;
  const std::uint8_t* result_converged = nullptr;
  std::uint64_t* warm_checkpoint_generations = nullptr;
  std::uint32_t* batch_ready = nullptr;
  const std::uint8_t* committed_field_attached = nullptr;
  const double* committed_field_vectors = nullptr;
  std::uint8_t* checkpoint_field_attached = nullptr;
  double* checkpoint_field_vectors = nullptr;
  const std::uint32_t* request_error = nullptr;
};

static_assert(std::is_trivially_copyable_v<WarmCheckpointPublicationDeviceBinding>);

__global__ void publish_gfn2_warm_checkpoint_generation_kernel(
    WarmCheckpointPublicationDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= binding.batch_size) return;
  const Gfn2DeviceAdmission admission{binding.request_error, 1, 0u};
  if (!gfn2_request_mutation_allowed(admission)) return;
  const std::uint64_t epoch = *binding.geometry_epoch.value;
  const bool publish =
      atomicAdd(const_cast<std::uint32_t*>(binding.publication_plan_error), 0u) == 0u &&
      epoch != 0u && binding.eligible[system] == 1u &&
      binding.committed_generations[system] == epoch &&
      binding.result_statuses[system] == VIBEQC_XTB_STATUS_SUCCESS &&
      binding.result_converged[system] == 1u;
  binding.warm_checkpoint_generations[system] = publish ? epoch : 0u;
  if (publish) {
    binding.checkpoint_field_attached[system] = binding.committed_field_attached[system];
    binding.checkpoint_field_vectors[3 * system] = binding.committed_field_vectors[3 * system];
    binding.checkpoint_field_vectors[3 * system + 1] =
        binding.committed_field_vectors[3 * system + 1];
    binding.checkpoint_field_vectors[3 * system + 2] =
        binding.committed_field_vectors[3 * system + 2];
  }
  if (!publish) atomicExch(binding.batch_ready, 0u);
}

__global__ void invalidate_gfn2_warm_checkpoint_if_admitted_kernel(
    std::uint64_t* generations, std::int64_t elements, const std::uint32_t* request_error) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const Gfn2DeviceAdmission admission{request_error, 1, 0u};
  if (gfn2_request_mutation_allowed(admission) && index < elements) generations[index] = 0u;
}

__global__ void initialize_gfn2_warm_checkpoint_batch_if_admitted_kernel(
    std::uint32_t* batch_ready, Gfn2DeviceAdmission admission) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && gfn2_request_mutation_allowed(admission)) {
    *batch_ready = UINT32_MAX;
  }
}

/*
 * Terminal analytic forces consume physical topology-major arrays, whereas
 * the converged SCC state is system-major and spin-major. This stable binding
 * projects the committed state once after the SCC loop so every downstream
 * force primitive can retain its established physical-offset contract.
 */
struct StationaryForceProjectionDeviceBinding {
  std::uint8_t enabled = 0u;
  const std::uint32_t* request_error = nullptr;
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_shells = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t total_spin_atoms = 0;
  std::int64_t total_spin_shells = 0;
  std::int64_t total_spin_matrix_elements = 0;

  const std::int64_t* atom_offsets = nullptr;
  const std::int64_t* shell_offsets = nullptr;
  const std::int64_t* matrix_offsets = nullptr;
  const std::int64_t* atom_shell_offsets = nullptr;
  const std::int32_t* spin_channels = nullptr;
  const std::int64_t* spin_atom_offsets = nullptr;
  const std::int64_t* spin_shell_offsets = nullptr;
  const std::int64_t* spin_matrix_offsets = nullptr;
  const std::int64_t* spin_coupling_offsets = nullptr;
  const double* spin_coupling_matrices = nullptr;

  const double* packed_density = nullptr;
  const double* packed_energy_weighted_density = nullptr;
  const double* packed_shell_charges = nullptr;
  const double* packed_atomic_charges = nullptr;
  const double* packed_atomic_dipoles = nullptr;
  const double* packed_atomic_quadrupoles = nullptr;

  double* total_density = nullptr;
  double* total_energy_weighted_density = nullptr;
  double* spin_density = nullptr;
  double* shell_charges = nullptr;
  double* atomic_charges = nullptr;
  double* atomic_dipoles = nullptr;
  double* atomic_quadrupoles = nullptr;
  double* spin_shell_potentials = nullptr;
};

static_assert(std::is_trivially_copyable_v<StationaryForceProjectionDeviceBinding>);

__global__ void project_gfn2_stationary_force_state_kernel(
    StationaryForceProjectionDeviceBinding binding) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= binding.batch_size) return;
  const Gfn2DeviceAdmission admission{binding.request_error, 1, 0u};
  if (!gfn2_request_admitted(admission)) return;

  const std::int32_t channels = binding.spin_channels[system];
  const std::int64_t atom_begin = binding.atom_offsets[system];
  const std::int64_t atom_end = binding.atom_offsets[system + 1];
  const std::int64_t shell_begin = binding.shell_offsets[system];
  const std::int64_t shell_end = binding.shell_offsets[system + 1];
  const std::int64_t matrix_begin = binding.matrix_offsets[system];
  const std::int64_t matrix_end = binding.matrix_offsets[system + 1];
  const std::int64_t spin_atom_begin = binding.spin_atom_offsets[system];
  const std::int64_t spin_shell_begin = binding.spin_shell_offsets[system];
  const std::int64_t spin_matrix_begin = binding.spin_matrix_offsets[system];
  const std::int64_t physical_shells = shell_end - shell_begin;
  const std::int64_t physical_matrices = matrix_end - matrix_begin;

  for (std::int64_t local = threadIdx.x; local < physical_matrices; local += blockDim.x) {
    const double alpha_density = binding.packed_density[spin_matrix_begin + local];
    const double alpha_weighted = binding.packed_energy_weighted_density[spin_matrix_begin + local];
    if (channels == 1) {
      binding.total_density[matrix_begin + local] = alpha_density;
      binding.total_energy_weighted_density[matrix_begin + local] = alpha_weighted;
      binding.spin_density[matrix_begin + local] = 0.0;
    } else {
      const double beta_density =
          binding.packed_density[spin_matrix_begin + physical_matrices + local];
      const double beta_weighted =
          binding.packed_energy_weighted_density[spin_matrix_begin + physical_matrices + local];
      binding.total_density[matrix_begin + local] = alpha_density + beta_density;
      binding.total_energy_weighted_density[matrix_begin + local] = alpha_weighted + beta_weighted;
      binding.spin_density[matrix_begin + local] = alpha_density - beta_density;
    }
  }

  for (std::int64_t shell = shell_begin + threadIdx.x; shell < shell_end; shell += blockDim.x) {
    const std::int64_t local = shell - shell_begin;
    binding.shell_charges[shell] = binding.packed_shell_charges[spin_shell_begin + local];
    if (channels == 1) binding.spin_shell_potentials[shell] = 0.0;
  }
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    const std::int64_t local_atom = atom - atom_begin;
    const std::int64_t spin_atom = spin_atom_begin + local_atom;
    binding.atomic_charges[atom] = binding.packed_atomic_charges[spin_atom];
    for (std::int64_t component = 0; component < 3; ++component) {
      binding.atomic_dipoles[atom * 3 + component] =
          binding.packed_atomic_dipoles[spin_atom * 3 + component];
    }
    for (std::int64_t component = 0; component < 6; ++component) {
      binding.atomic_quadrupoles[atom * 6 + component] =
          binding.packed_atomic_quadrupoles[spin_atom * 6 + component];
    }
  }

  if (channels == 2) {
    /* v_mag = W*m uses the committed raw magnetization shell population.
     * Coupling rows are atom-local and retain the CPU column accumulation
     * order, avoiding a different roundoff path in force parity checks. */
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
      const std::int64_t atom_shell_begin = binding.atom_shell_offsets[atom];
      const std::int64_t atom_shell_end = binding.atom_shell_offsets[atom + 1];
      const std::int64_t atom_shells = atom_shell_end - atom_shell_begin;
      const std::int64_t coupling_begin = binding.spin_coupling_offsets[atom];
      for (std::int64_t shell = atom_shell_begin + threadIdx.x; shell < atom_shell_end;
           shell += blockDim.x) {
        const std::int64_t row = shell - atom_shell_begin;
        double potential = 0.0;
        for (std::int64_t column = 0; column < atom_shells; ++column) {
          const std::int64_t magnetization_shell =
              spin_shell_begin + physical_shells + atom_shell_begin - shell_begin + column;
          potential += binding.spin_coupling_matrices[coupling_begin + row * atom_shells + column] *
                       binding.packed_shell_charges[magnetization_shell];
        }
        binding.spin_shell_potentials[shell] = potential;
      }
    }
  }
}

cudaError_t project_gfn2_stationary_force_state_cuda(
    const StationaryForceProjectionDeviceBinding& binding, cudaStream_t stream) noexcept {
  if (binding.enabled == 0u) return cudaSuccess;
  if (binding.enabled != 1u || binding.batch_size <= 0 || binding.total_atoms <= 0 ||
      binding.total_shells <= 0 || binding.total_matrix_elements <= 0 ||
      binding.total_spin_atoms < binding.total_atoms ||
      binding.total_spin_shells < binding.total_shells ||
      binding.total_spin_matrix_elements < binding.total_matrix_elements ||
      binding.atom_offsets == nullptr || binding.shell_offsets == nullptr ||
      binding.matrix_offsets == nullptr || binding.atom_shell_offsets == nullptr ||
      binding.spin_channels == nullptr || binding.spin_atom_offsets == nullptr ||
      binding.spin_shell_offsets == nullptr || binding.spin_matrix_offsets == nullptr ||
      binding.spin_coupling_offsets == nullptr || binding.spin_coupling_matrices == nullptr ||
      binding.packed_density == nullptr || binding.packed_energy_weighted_density == nullptr ||
      binding.packed_shell_charges == nullptr || binding.packed_atomic_charges == nullptr ||
      binding.packed_atomic_dipoles == nullptr || binding.packed_atomic_quadrupoles == nullptr ||
      binding.total_density == nullptr || binding.total_energy_weighted_density == nullptr ||
      binding.spin_density == nullptr || binding.shell_charges == nullptr ||
      binding.atomic_charges == nullptr || binding.atomic_dipoles == nullptr ||
      binding.atomic_quadrupoles == nullptr || binding.spin_shell_potentials == nullptr ||
      binding.batch_size > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max())) {
    return cudaErrorInvalidValue;
  }
  project_gfn2_stationary_force_state_kernel<<<static_cast<unsigned int>(binding.batch_size), 256,
                                               0, stream>>>(binding);
  return cudaPeekAtLastError();
}

struct TopologyKey {
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> molecular_charges;
  std::vector<std::int32_t> unpaired_electrons;
  std::vector<std::int32_t> spin_channels;
  std::vector<std::int64_t> point_offsets;
  std::vector<std::int64_t> response_offsets;
  std::uint32_t flags = 0u;
  std::int32_t maximum_iterations = 0;
  double charge_tolerance = 0.0;
  double energy_tolerance = 0.0;
  double electronic_temperature = 0.0;
  vibeqc_xtb_scc_mixer_t scc_mixer = VIBEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN;
  std::int32_t scc_mixer_history = kDefaultMixerHistory;
  double scc_mixer_damping = kDefaultMixerDamping;
  vibeqc_xtb_determinism_t determinism = VIBEQC_XTB_DETERMINISM_DEFAULT;
  bool periodic_enabled = false;

  std::uint64_t fingerprint() const noexcept {
    std::uint64_t hash = 0x4750555854424b59ULL;
    hash_append_vector(hash, atom_offsets);
    hash_append_vector(hash, atomic_numbers);
    hash_append_vector(hash, molecular_charges);
    hash_append_vector(hash, unpaired_electrons);
    hash_append_vector(hash, spin_channels);
    hash_append_vector(hash, point_offsets);
    hash_append_vector(hash, response_offsets);
    hash_append(hash, flags);
    hash_append(hash, static_cast<std::uint32_t>(maximum_iterations));
    hash_append(hash, double_bits(charge_tolerance));
    hash_append(hash, double_bits(energy_tolerance));
    hash_append(hash, double_bits(electronic_temperature));
    hash_append(hash, static_cast<std::uint32_t>(scc_mixer));
    hash_append(hash, static_cast<std::uint32_t>(scc_mixer_history));
    hash_append(hash, double_bits(scc_mixer_damping));
    hash_append(hash, static_cast<std::uint32_t>(determinism));
    hash_append(hash, periodic_enabled ? 1u : 0u);
    return hash == 0u ? 1u : hash;
  }
};

/*
 * Build topology-derived host plans without downloading caller numerical
 * buffers.  The deterministic seed is used only while constructing stable
 * owners, descriptors, arenas, and provider workspaces.  A public CUDA
 * transaction must refresh the resulting Prepared candidate from the real
 * caller buffers before it may replace the committed runtime.
 */
vibeqc_xtb_status_t make_topology_only_seed(
    const Gfn2CudaTopologyHostSnapshot& snapshot, const vibeqc_xtb_compute_options_t& options,
    TopologyKey& key, std::vector<double>& positions, std::vector<double>& point_positions,
    std::vector<double>& point_values, std::vector<double>& point_gammas,
    std::vector<double>& periodic_shifts, std::vector<double>& periodic_response,
    std::string& error) {
  if (snapshot.batch_size <= 0 || snapshot.total_atoms < snapshot.batch_size ||
      snapshot.total_point_charges < 0 ||
      snapshot.atom_offsets.size() != static_cast<std::size_t>(snapshot.batch_size + 1) ||
      snapshot.atomic_numbers.size() != static_cast<std::size_t>(snapshot.total_atoms) ||
      snapshot.molecular_charges.size() != static_cast<std::size_t>(snapshot.batch_size) ||
      snapshot.unpaired_electrons.size() != static_cast<std::size_t>(snapshot.batch_size) ||
      snapshot.spin_channels.size() != static_cast<std::size_t>(snapshot.batch_size) ||
      snapshot.point_charge_offsets.size() != static_cast<std::size_t>(snapshot.batch_size + 1) ||
      snapshot.charge_response_offsets.size() !=
          static_cast<std::size_t>(snapshot.batch_size + 1)) {
    error = "CUDA topology staging returned an inconsistent host snapshot";
    return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
  }

  vibeqc_xtb_status_t status = validate_public_execution_policy(options, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

  key.atom_offsets = snapshot.atom_offsets;
  key.atomic_numbers = snapshot.atomic_numbers;
  key.molecular_charges = snapshot.molecular_charges;
  key.unpaired_electrons = snapshot.unpaired_electrons;
  key.spin_channels = snapshot.spin_channels;
  key.point_offsets = snapshot.point_charge_offsets;
  key.response_offsets = snapshot.charge_response_offsets;
  key.flags = options.flags;
  key.maximum_iterations = options.max_scc_iterations;
  key.charge_tolerance = options.charge_tolerance;
  key.energy_tolerance = options.energy_tolerance;
  key.electronic_temperature = options.electronic_temperature;
  key.scc_mixer = public_scc_mixer(options);
  key.scc_mixer_history = public_scc_mixer_history(options);
  key.scc_mixer_damping = public_scc_mixer_damping(options);
  key.determinism = public_determinism(options);
  key.periodic_enabled = snapshot.periodic_enabled;

  std::int64_t coordinate_elements = 0;
  std::int64_t point_coordinate_elements = 0;
  if (!checked_elements(snapshot.total_atoms, 3, coordinate_elements) ||
      !checked_elements(snapshot.total_point_charges, 3, point_coordinate_elements) ||
      snapshot.total_charge_response_elements < 0) {
    error = "CUDA topology-only seed extent overflows int64_t";
    return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
  }
  try {
    positions.assign(static_cast<std::size_t>(coordinate_elements), 0.0);
    for (std::int64_t system = 0; system < snapshot.batch_size; ++system) {
      const std::int64_t begin = snapshot.atom_offsets[static_cast<std::size_t>(system)];
      const std::int64_t end = snapshot.atom_offsets[static_cast<std::size_t>(system + 1)];
      for (std::int64_t atom = begin; atom < end; ++atom) {
        const std::int64_t local = atom - begin;
        positions[static_cast<std::size_t>(3 * atom)] = 8.0 * static_cast<double>(local);
      }
    }
    point_positions.assign(static_cast<std::size_t>(point_coordinate_elements), 0.0);
    point_values.assign(static_cast<std::size_t>(snapshot.total_point_charges), 0.0);
    point_gammas.assign(static_cast<std::size_t>(snapshot.total_point_charges), 1.0);
    periodic_shifts.assign(static_cast<std::size_t>(snapshot.total_atoms), 0.0);
    periodic_response.assign(static_cast<std::size_t>(snapshot.total_charge_response_elements),
                             0.0);
  } catch (const std::bad_alloc&) {
    error = "failed to allocate CUDA topology-only numerical seed";
    return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
  }
  return VIBEQC_XTB_STATUS_SUCCESS;
}

bool topology_snapshot_matches(const Gfn2CudaTopologyHostSnapshot& snapshot,
                               const vibeqc_xtb_compute_options_t& options,
                               const TopologyKey& key) noexcept {
  return snapshot.atom_offsets == key.atom_offsets &&
         snapshot.atomic_numbers == key.atomic_numbers &&
         snapshot.molecular_charges == key.molecular_charges &&
         snapshot.unpaired_electrons == key.unpaired_electrons &&
         snapshot.spin_channels == key.spin_channels &&
         snapshot.point_charge_offsets == key.point_offsets &&
         snapshot.charge_response_offsets == key.response_offsets &&
         snapshot.periodic_enabled == key.periodic_enabled && options.flags == key.flags &&
         options.max_scc_iterations == key.maximum_iterations &&
         options.charge_tolerance == key.charge_tolerance &&
         options.energy_tolerance == key.energy_tolerance &&
         options.electronic_temperature == key.electronic_temperature &&
         public_scc_mixer(options) == key.scc_mixer &&
         public_scc_mixer_history(options) == key.scc_mixer_history &&
         public_scc_mixer_damping(options) == key.scc_mixer_damping &&
         public_determinism(options) == key.determinism;
}

struct HostPlans {
  TopologyKey key;
  std::uint64_t fingerprint = 0u;
  std::uint64_t plan_token = 0u;
  std::uint64_t geometry_generation = kInitialGeometryGeneration;

  BasisPlan basis;
  IntegralPlan integrals;
  CoordinationPlan coordination;
  RepulsionPlan repulsion;
  H0Plan h0;
  WavefunctionLayout wavefunction_layout;
  ES2Plan es2;
  ES3Plan es3;
  AES2Plan aes2;
  MullikenPlan mulliken;
  EigensolverPlan eigensolver;
  SccMixerPlan mixer;
  D4Plan d4;
  ExternalPointChargePlan external;
  PeriodicEmbeddingPlan periodic;
  SccDriverPlan driver;
  bool d4_enabled = false;
  bool periodic_enabled = false;

  std::vector<double> positions;
  std::vector<double> point_positions;
  std::vector<double> point_values;
  std::vector<double> point_gammas;
  std::vector<double> periodic_shifts;
  std::vector<double> periodic_response;

  std::vector<double> coordination_numbers;
  std::vector<double> overlap;
  std::vector<double> dipole_integrals;
  std::vector<double> quadrupole_integrals;
  std::vector<double> core_hamiltonian;
  std::vector<double> integral_workspace;

  std::vector<double> geometry_pair_data;
  std::vector<std::uint64_t> geometry_generations;

  std::vector<double> es2_matrix;
  std::vector<double> es2_matrix_scratch;
  std::vector<double> es2_shell_scratch;
  std::vector<double> es2_batch_scratch;
  std::vector<double> es2_gradient_scratch;
  ES2Workspace es2_workspace{};
  ES2GeometryCache es2_cache{};

  std::vector<double> aes2_pairs;
  std::vector<double> aes2_pair_scratch;
  std::vector<double> aes2_potential_scratch;
  std::vector<double> aes2_batch_scratch;
  std::vector<double> aes2_gradient_scratch;
  std::vector<double> aes2_coordination_scratch;
  AES2Workspace aes2_workspace{};
  AES2GeometryCache aes2_cache{};

  std::vector<Gfn2D4DeviceElementData> d4_elements;
  std::vector<Gfn2D4DeviceReferenceData> d4_references;
  std::vector<double> d4_reference_c6;
  std::vector<double> d4_coordination;

  std::vector<double> explicit_point_shell_potential;
  PinnedArena wavefunction_storage;
  WavefunctionView wavefunction{};

  vibeqc_xtb_status_t build(TopologyKey&& new_key, std::vector<double>&& new_positions,
                            std::vector<double>&& new_point_positions,
                            std::vector<double>&& new_point_values,
                            std::vector<double>&& new_point_gammas,
                            std::vector<double>&& new_periodic_shifts,
                            std::vector<double>&& new_periodic_response, std::uint64_t token,
                            std::string& error) {
    key = std::move(new_key);
    positions = std::move(new_positions);
    point_positions = std::move(new_point_positions);
    point_values = std::move(new_point_values);
    point_gammas = std::move(new_point_gammas);
    periodic_shifts = std::move(new_periodic_shifts);
    periodic_response = std::move(new_periodic_response);
    fingerprint = key.fingerprint();
    plan_token = token;

    const std::int64_t batch = static_cast<std::int64_t>(key.molecular_charges.size());
    const std::int64_t atoms = static_cast<std::int64_t>(key.atomic_numbers.size());
    const std::int64_t points = static_cast<std::int64_t>(point_values.size());
    vibeqc_xtb_status_t status = make_basis_plan(batch, atoms, key.atom_offsets.data(),
                                                 key.atomic_numbers.data(), basis, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_integral_plan(basis, integrals, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_coordination_plan(batch, atoms, key.atom_offsets.data(),
                                    key.atomic_numbers.data(), coordination, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_repulsion_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(),
                                 repulsion, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_h0_plan(basis, integrals, key.atomic_numbers.data(), h0, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_wavefunction_layout(basis, key.atomic_numbers.data(),
                                      key.molecular_charges.data(), key.unpaired_electrons.data(),
                                      key.spin_channels.data(), wavefunction_layout, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_es2_plan(basis, key.atomic_numbers.data(), es2, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_es3_plan(basis, key.atomic_numbers.data(), es3, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_aes2_plan(basis, key.atomic_numbers.data(), aes2, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_mulliken_plan(basis, integrals, wavefunction_layout, mulliken, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_eigensolver_plan(wavefunction_layout, eigensolver, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = make_scc_mixer_plan(wavefunction_layout, key.scc_mixer_history, key.scc_mixer_damping,
                                 key.charge_tolerance, key.charge_tolerance, mixer, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

    d4_enabled = false;
    for (std::int64_t system = 0; system < batch; ++system) {
      d4_enabled = d4_enabled || key.atom_offsets[static_cast<std::size_t>(system + 1)] -
                                         key.atom_offsets[static_cast<std::size_t>(system)] >
                                     1;
    }
    if (d4_enabled) {
      status =
          make_d4_plan(batch, atoms, key.atom_offsets.data(), key.atomic_numbers.data(), d4, error);
      if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    }
    status = make_external_point_charge_plan(basis, key.atomic_numbers.data(), points,
                                             points == 0 ? nullptr : key.point_offsets.data(),
                                             external, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    periodic_enabled = key.periodic_enabled;
    if (periodic_enabled) {
      status = make_periodic_embedding_plan(batch, atoms, key.atom_offsets.data(), periodic, error);
      if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    }
    status =
        make_scc_driver_plan(wavefunction_layout, mulliken, es2, es3, aes2, eigensolver, mixer,
                             d4_enabled ? &d4 : nullptr, periodic_enabled ? &periodic : nullptr,
                             static_cast<std::uint64_t>(key.maximum_iterations),
                             key.electronic_temperature, key.energy_tolerance, driver, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

    const std::size_t atom_count = static_cast<std::size_t>(atoms);
    const std::size_t shells = static_cast<std::size_t>(basis.total_shells);
    const std::size_t matrices = static_cast<std::size_t>(integrals.total_matrix_elements);
    const std::size_t integral_doubles =
        (integrals.workspace_size_bytes + sizeof(double) - 1u) / sizeof(double);
    coordination_numbers.resize(atom_count);
    overlap.resize(matrices);
    dipole_integrals.resize(3u * matrices);
    quadrupole_integrals.resize(6u * matrices);
    core_hamiltonian.resize(matrices);
    integral_workspace.resize(std::max<std::size_t>(integral_doubles, 1u));

    status = evaluate_coordination_cpu(coordination, positions.data(), coordination_numbers.data(),
                                       error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = evaluate_overlap_cpu(basis, integrals, positions.data(), overlap.data(),
                                  integral_workspace.data(),
                                  integral_workspace.size() * sizeof(double), error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = evaluate_multipole_cpu(basis, integrals, positions.data(), dipole_integrals.data(),
                                    quadrupole_integrals.data(), integral_workspace.data(),
                                    integral_workspace.size() * sizeof(double), error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = evaluate_h0_cpu(basis, integrals, h0, positions.data(), coordination_numbers.data(),
                             overlap.data(), core_hamiltonian.data(), error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

    geometry_pair_data.assign(static_cast<std::size_t>(aes2.total_pairs()) *
                                  static_cast<std::size_t>(kGfn2GeometryPairDataElements),
                              0.0);
    geometry_generations.assign(static_cast<std::size_t>(batch), geometry_generation);

    es2_matrix.resize(static_cast<std::size_t>(es2.total_matrix_elements()));
    es2_matrix_scratch.resize(es2_matrix.size());
    es2_shell_scratch.resize(shells);
    es2_batch_scratch.resize(static_cast<std::size_t>(batch));
    es2_gradient_scratch.resize(3u * atom_count);
    es2_workspace = {es2_matrix_scratch.data(),   es2.total_matrix_elements(),
                     es2_shell_scratch.data(),    es2.total_shells(),
                     es2_batch_scratch.data(),    batch,
                     es2_gradient_scratch.data(), atoms * 3};
    status =
        update_es2_geometry_cache_cpu(es2, positions.data(), geometry_generation, es2_matrix.data(),
                                      es2_matrix.size(), es2_workspace, es2_cache, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

    aes2_pairs.resize(static_cast<std::size_t>(aes2.pair_data_elements()));
    aes2_pair_scratch.resize(aes2_pairs.size());
    aes2_potential_scratch.resize(static_cast<std::size_t>(aes2.potential_scratch_elements()));
    aes2_batch_scratch.resize(static_cast<std::size_t>(batch));
    aes2_gradient_scratch.resize(3u * atom_count);
    aes2_coordination_scratch.resize(atom_count);
    aes2_workspace = {aes2_pair_scratch.data(),         aes2.pair_data_elements(),
                      aes2_potential_scratch.data(),    aes2.potential_scratch_elements(),
                      aes2_batch_scratch.data(),        batch,
                      aes2_gradient_scratch.data(),     atoms * 3,
                      aes2_coordination_scratch.data(), atoms};
    status = update_aes2_geometry_cache_cpu(aes2, positions.data(), coordination_numbers.data(),
                                            geometry_generation, aes2_pairs.data(),
                                            aes2_pairs.size(), aes2_workspace, aes2_cache, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

    if (d4_enabled) {
      /* The setup owner needs only a stable initial CN image. The first CUDA
       * preprocessing transaction replaces these zeros before any D4
       * consumer is eligible; dense CPU D4 pair evaluation is intentionally
       * not part of cache construction anymore. */
      d4_coordination.assign(atom_count, 0.0);
      namespace canonical_d4 = ::vibeqc::dft::dispersion::data;
      d4_elements.reserve(canonical_d4::kElements.size());
      for (const auto& element : canonical_d4::kElements) {
        d4_elements.push_back({element.reference_offset, element.reference_count,
                               element.covalent_radius, element.electronegativity,
                               element.effective_charge, element.hardness, element.r4r2});
      }
      d4_references.reserve(canonical_d4::kReferences.size());
      for (const auto& reference : canonical_d4::kReferences) {
        d4_references.push_back(
            {reference.coordination_number, reference.charge, reference.gaussian_count});
      }
      const std::size_t reference_count = canonical_d4::kReferences.size();
      d4_reference_c6.resize(reference_count * reference_count);
      for (std::size_t first = 0; first < reference_count; ++first) {
        for (std::size_t second = 0; second < reference_count; ++second) {
          const std::size_t high = std::max(first, second);
          const std::size_t low = std::min(first, second);
          d4_reference_c6[first * reference_count + second] =
              canonical_d4::kReferenceC6[high * (high + 1u) / 2u + low];
        }
      }
    }

    explicit_point_shell_potential.resize(shells);
    status = evaluate_external_point_charge_potential_cpu(
        external, positions.data(), point_positions.empty() ? nullptr : point_positions.data(),
        point_values.empty() ? nullptr : point_values.data(),
        point_gammas.empty() ? nullptr : point_gammas.data(), explicit_point_shell_potential.data(),
        error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

    if (wavefunction_storage.allocate(wavefunction_layout.workspace_size_bytes) != cudaSuccess) {
      error = "failed to allocate pinned host wavefunction initialization storage";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    status = bind_wavefunction_view(wavefunction_layout, wavefunction_storage.get(),
                                    wavefunction_storage.bytes(), wavefunction, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = initialize_sad_multipole_state(wavefunction_layout, wavefunction, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  Gfn2SccSetupInputSources input_sources() const noexcept {
    Gfn2SccSetupInputSources sources{};
    sources.basis = &basis;
    sources.integrals = &integrals;
    sources.h0_plan = &h0;
    sources.wavefunction = &wavefunction_layout;
    sources.es2 = &es2;
    sources.es3 = &es3;
    sources.aes2 = &aes2;
    sources.mulliken = &mulliken;
    sources.mixer = &mixer;
    sources.driver = &driver;
    sources.eigensolver_options.deterministic_debug =
        key.determinism == VIBEQC_XTB_DETERMINISM_REPRODUCIBLE;
    sources.geometry_generation = geometry_generation;
    sources.atomic_numbers = setup_array(key.atomic_numbers);
    sources.positions = setup_array(positions);
    sources.covalent_radii = setup_array(coordination.covalent_radius);
    sources.h0 = setup_array(core_hamiltonian);
    sources.overlap = setup_array(overlap);
    sources.dipole_integrals = setup_array(dipole_integrals);
    sources.quadrupole_integrals = setup_array(quadrupole_integrals);
    sources.geometry_cache.pair_data = setup_array(geometry_pair_data);
    sources.geometry_cache.coordination_numbers = setup_array(coordination_numbers);
    sources.geometry_cache.system_generations = setup_array(geometry_generations);
    sources.es2_cache.coulomb_matrix = setup_array(es2_matrix);
    sources.aes2_cache.pair_data = setup_array(aes2_pairs);
    if (d4_enabled) {
      sources.d4.plan = &d4;
      sources.d4.elements = setup_array(d4_elements);
      sources.d4.references = setup_array(d4_references);
      sources.d4.reference_c6 = setup_array(d4_reference_c6);
      sources.d4.coordination_numbers = setup_array(d4_coordination);
    }
    if (!point_values.empty()) {
      sources.point_charges.plan = &external;
      sources.point_charges.positions = setup_array(point_positions);
      sources.point_charges.charges = setup_array(point_values);
      sources.point_charges.hardnesses = setup_array(point_gammas);
      sources.point_charges.shell_potential_cache = setup_array(explicit_point_shell_potential);
    }
    if (periodic_enabled) {
      sources.periodic.plan = &periodic;
      sources.periodic.shifts = setup_array(periodic_shifts);
      sources.periodic.response_matrices = setup_array(periodic_response);
    }
    return sources;
  }

  Gfn2SccIterationHostInitialization initialization() const noexcept {
    Gfn2SccIterationHostInitialization host{};
    host.mode = Gfn2SccIterationInitializationMode::kFresh;
    host.plan_token = plan_token;
    host.initialization_generation = kInitialStateGeneration;
    host.topology = {initialization_array(key.atom_offsets.data(),
                                          static_cast<std::int64_t>(key.atom_offsets.size())),
                     initialization_array(
                         wavefunction_layout.batch_shell_offsets.data(),
                         static_cast<std::int64_t>(wavefunction_layout.batch_shell_offsets.size())),
                     plan_token};
    host.wavefunction.plan_token = plan_token;
    host.wavefunction.population = {
        initialization_array(wavefunction.qsh, wavefunction_layout.qsh.element_count),
        initialization_array(wavefunction.qat, wavefunction_layout.qat.element_count),
        initialization_array(wavefunction.dipole, wavefunction_layout.dipole.element_count),
        initialization_array(wavefunction.quadrupole, wavefunction_layout.quadrupole.element_count),
        plan_token};
    return host;
  }
};

std::string cuda_error_message(const char* operation, cudaError_t status) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorName(status) << " ("
          << cudaGetErrorString(status) << ')';
  return message.str();
}

std::string setup_error_message(const char* operation, vibeqc_xtb_status_t status,
                                std::uint32_t error_code, std::uint32_t field, std::int64_t index) {
  std::ostringstream message;
  message << operation << " failed: status=" << status << " error=" << error_code
          << " field=" << field << " index=" << index;
  return message.str();
}

struct NumericalRefreshState {
  Gfn2PreprocessingDeviceBinding preprocessing{};
  NumericalRefreshDeviceBinding device{};
  /* Refresh reads preprocessing candidate positions; SCC/post/force/terminal
   * consumers read d4_cache with the exact committed-position identity. */
  Gfn2D4PairListDeviceCache d4_refresh_cache{};
  Gfn2D4PairListDeviceCache d4_cache{};
  Gfn2D4DeviceWorkspace d4_workspace{};
  Gfn2ExternalPointChargeDeviceBatch point_batch{};
  Gfn2ExternalPointChargeDeviceCache point_candidate{};
  Gfn2ExternalPointChargeDeviceWorkspace point_workspace{};
  Gfn2SccIterationDeviceActivity point_activity{};
  Gfn2ElectricFieldDeviceBatch field_batch{};
  Gfn2ElectricFieldDevicePotentials field_candidate_potentials{};
  Gfn2ElectricFieldDevicePotentials field_committed_potentials{};
  Gfn2ElectricFieldDevicePotentialView field_potential_view{};

  double* host_positions = nullptr;

  /*
   * Host position submissions first copy synchronously into this pinned image.
   * The device staging leaf is then populated asynchronously, so queued CUDA
   * work never retains a caller-owned host pointer after return.
   */
  double* owned_host_positions = nullptr;

  bool host_staging_poisoned = false;

  bool ready = false;
};

/*
 * The host-owned pinned snapshot is a single-flight resource.  A host function
 * on a private completion stream releases it after that stream has waited on
 * the owner-stream H2D event.  The submission thread therefore needs only one
 * acquire load; it never queries CUDA progress or waits for either stream.
 *
 * This state is deliberately separate from NumericalRefreshState because the
 * latter is reset as an aggregate while constructing a fixed-topology owner.
 */
struct NumericalHostUploadCompletion {
  std::atomic<bool> pending{false};
};
static_assert(std::atomic<bool>::is_always_lock_free,
              "CUDA host-upload completion must stay allocation-free and nonblocking");

void CUDART_CB release_numerical_host_upload(void* user_data) noexcept {
  auto* const completion = static_cast<NumericalHostUploadCompletion*>(user_data);
  completion->pending.store(false, std::memory_order_release);
}

/*
 * Stable terminal descriptors and their context-owned result arena.  These
 * views are constructed once for a fixed topology and copied by value into
 * allocation-free stream submissions.  The result buffers are deliberately
 * internal: #125 later bridges them to host or device C-API destinations.
 */
struct InferenceState {
  Gfn2GeometryEpochConsumerDevice epoch_consumer{};

  Gfn2TerminalClassicalEnergyDevicePlan terminal_plan{};
  Gfn2TerminalClassicalEnergyDeviceActivity terminal_activity{};
  Gfn2TerminalClassicalEnergyDeviceResults terminal_results{};
  Gfn2TerminalClassicalEnergyDeviceWorkspace terminal_workspace{};
  Gfn2TerminalClassicalEnergyDeviceDiagnostics terminal_diagnostics{};

  Gfn2InferencePublicationDevicePlan publication_plan{};
  Gfn2InferencePublicationDeviceInput publication_input{};
  Gfn2InferencePublicationDeviceResults publication_results{};
  Gfn2InferencePublicationDeviceWorkspace publication_workspace{};
  Gfn2InferencePublicationDeviceDiagnostics publication_diagnostics{};

  /* Per-peer generation of the last successfully published SCC checkpoint. */
  std::uint64_t* warm_checkpoint_generations = nullptr;
  /* Device aggregate: nonzero only when every peer published a checkpoint. */
  std::uint32_t* warm_checkpoint_batch_ready = nullptr;
  /* Single-flight asynchronous request control and the checkpoint token moved
   * out of public storage at accepted admission. */
  std::uint32_t* request_start_mode = nullptr;
  std::uint64_t* admitted_warm_checkpoint_generations = nullptr;
  std::uint8_t* warm_checkpoint_field_attached = nullptr;
  double* warm_checkpoint_field_vectors = nullptr;
  bool ready = false;
  bool warm_checkpoint_ready = false;
};

/*
 * Fixed-topology public result staging. Every possible HOST route has a stable
 * pinned slice, while one device control record seals aggregate publication
 * before any CUDA caller destination is touched. Per-call route descriptors
 * borrow these addresses but never outlive the synchronous cache transaction.
 */
struct PublicResultState {
  double* energies = nullptr;
  double* qm_forces = nullptr;
  double* atomic_charges = nullptr;
  double* point_forces = nullptr;
  double* dipole_moments = nullptr;
  std::int32_t* iterations = nullptr;
  std::uint8_t* converged = nullptr;
  vibeqc_xtb_status_t* system_statuses = nullptr;
  Gfn2PublicResultBridgeControl* host_control = nullptr;
  /* Pinned mirror copied under the existing public completion event. */
  std::uint32_t* warm_checkpoint_ready = nullptr;
  /* The request flag and canonical topology arrays share the plan-owned
   * result arena. They are reset/compared on the owner stream before
   * inference, then consumed by the public-result preflight gate. */
  std::uint32_t* request_topology_error = nullptr;
  const std::int64_t* expected_atom_offsets = nullptr;
  const std::int32_t* expected_atomic_numbers = nullptr;
  const double* expected_molecular_charges = nullptr;
  const std::int32_t* expected_unpaired_electrons = nullptr;
  const std::int32_t* expected_spin_channels = nullptr;
  const std::int64_t* expected_point_offsets = nullptr;
  const std::int64_t* expected_response_offsets = nullptr;
  double* staged_lattice_cells = nullptr;
  std::int32_t* staged_periodic_axes = nullptr;
  double* host_lattice_cells = nullptr;
  std::int32_t* host_periodic_axes = nullptr;
  Gfn2PublicResultBridgeDeviceStaging device_staging{};
  Gfn2PublicResultBridgeDeviceDiagnostics diagnostics{};
  std::uint32_t pending_result_flags = 0u;
  bool ready = false;
};

/*
 * The public bridge is intentionally represented as explicit host phases.
 * CUDA submission, completion observation, aggregate acceptance, and host
 * publication have different lifetime and failure boundaries; keeping those
 * boundaries visible lets the synchronous owner preserve today's ordering
 * while a later request owner can drive the same transaction incrementally.
 */
enum class PublicResultTransactionPhase : std::uint32_t {
  kEmpty = 0u,
  kPrepareSubmitted = 1u,
  kPrepareCompletionRecorded = 2u,
  kPrepareCompleted = 3u,
  kPrepareAccepted = 4u,
  kCommitSubmitted = 5u,
  kCommitCompletionRecorded = 6u,
  kCommitCompleted = 7u,
  kPublished = 8u,
};

/* Descriptor image retained across one public result transaction. All
 * referenced storage is owned by Prepared or borrowed from the caller until
 * synchronous return / asynchronous request completion. */
struct PublicResultTransaction {
  Gfn2PublicResultBridgeDevicePlan plan{};
  Gfn2PublicResultBridgeDeviceDestinations destinations{};
  PublicResultTransactionPhase phase = PublicResultTransactionPhase::kEmpty;
  bool prior_warm_checkpoint_ready = false;
};

}  // namespace

struct Gfn2CudaExecutionCache::Impl {
  struct EnergyForceBindings {
    Gfn2EnergyForceExecutionDevicePlan plan{};
    Gfn2EnergyForceExecutionDeviceInput input{};
    Gfn2EnergyForceExecutionDeviceResults results{};
    Gfn2EnergyForceExecutionDeviceIntermediates intermediates{};
    Gfn2EnergyForceExecutionDeviceWorkspace workspace{};
    Gfn2EnergyForceExecutionDeviceDiagnostics diagnostics{};
    StationaryForceProjectionDeviceBinding stationary_projection{};
  };

  struct Prepared {
    explicit Prepared(cudaStream_t owner_stream) noexcept : stream(owner_stream) {}
    ~Prepared() {
      /* Setup owners retain pinned provider images and the SCC initializer
       * retains its immutable device checkpoint. Numerical host-completion
       * functions also retain the address of this Prepared object's atomic
       * state. A failed candidate and a replaced cache therefore wait for the
       * owner and private completion streams before releasing any source image,
       * callback state, or arena referenced by queued work. CUDA does not
       * invoke a host function after a context failure; in that case pending
       * remains conservatively set and both failed synchronization attempts
       * still precede destruction. */
      if (submitted) {
        /* cudaStreamSynchronize(nullptr) waits for the selected legacy default
         * stream only; setup teardown never escalates to a device-wide fence. */
        (void)cudaStreamSynchronize(stream);
      }
      if (numerical_host_completion_stream.valid()) {
        (void)cudaStreamSynchronize(numerical_host_completion_stream.get());
      }
    }

    cudaStream_t stream = nullptr;
    bool submitted = false;
    HostPlans host;
    Gfn2SccSetupTopology topology_owner;
    Gfn2SccSetupInputs inputs_owner;
    Gfn2SccSetupEigensolver eigensolver_owner;
    Gfn2SccIterationInitializer initializer;
    DeviceArena topology_arena;
    DeviceArena input_arena;
    DeviceArena iteration_arena;
    DeviceArena eigensolver_setup_arena;
    PinnedArena provider_host_workspace;
    PinnedArena numerical_host_staging_arena;
    CudaStream numerical_host_completion_stream;
    CudaEvent numerical_host_upload_complete;
    CudaEvent numerical_host_release_complete;
    NumericalHostUploadCompletion numerical_host_upload_completion;
    DeviceArena force_immutable_arena;
    DeviceArena force_execution_arena;
    DeviceArena numerical_refresh_arena;
    DeviceArena inference_arena;
    DeviceArena public_result_device_arena;
    PinnedArena public_result_host_arena;
    PinnedArena candidate_validation_arena;
    CudaEvent public_result_completion_event;
    Gfn2RaggedTopologyView device_topology{};
    Gfn2WavefunctionLayoutView device_wavefunction{};
    Gfn2SccIterationDevicePlan plan_seed{};
    Gfn2SccIterationDeviceInput input_seed{};
    Gfn2SccIterationArenaRequirements iteration_requirements{};
    Gfn2SccIterationDeviceState state_seed{};
    Gfn2SccIterationDeviceWorkspace workspace_seed{};
    Gfn2SccIterationReportStorage report_storage{};
    Gfn2SccSetupEigensolverBinding eigensolver_binding{};
    Gfn2SccIterationInitializationReady ready{};
    Gfn2SccIterationBinding scc_binding{};
    /* Built once after setup validation and replayed by fresh/warm inference.
     * The owner retains a bounded fallback when conditional capture is not
     * supported by the selected CUDA/provider stack. */
    Gfn2SccLoopCudaGraphOwner scc_loop;
    const std::int32_t* atomic_numbers = nullptr;
    EnergyForceBindings energy_force{};
    NumericalRefreshState numerical{};
    InferenceState inference{};
    PublicResultState public_result{};
    bool energy_force_smoke_ready = false;
  };

  static vibeqc_xtb_status_t validate_prepared_admission_aliases(const Prepared& candidate,
                                                                 std::string& error) noexcept {
    const auto* const admission = candidate.public_result.request_topology_error;
    AddressRange admission_range{};
    AddressRange public_result_arena_range{};
    if (reinterpret_cast<std::uintptr_t>(admission) % alignof(std::uint32_t) != 0u ||
        !make_address_range(admission, sizeof(*admission), admission_range) ||
        !make_address_range(candidate.public_result_device_arena.get(),
                            candidate.public_result_device_arena.bytes(),
                            public_result_arena_range) ||
        !address_range_contains(public_result_arena_range, admission_range)) {
      error = "CUDA runtime admission scalar is outside its owned public-result range";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    const auto reject_overlap = [&](const char* name, const void* pointer, std::int64_t elements,
                                    std::size_t element_size) -> bool {
      if (elements == 0) return false;
      std::size_t bytes = 0u;
      AddressRange writable{};
      if (!checked_bytes(elements, element_size, bytes)) {
        error =
            std::string("CUDA runtime admission audit found an invalid writable range for ") + name;
        return true;
      }
      /* This pass audits cross-subsystem aliasing, while the component
       * validators own required/optional pointer canonicalization. A null
       * later-bound view therefore contributes no range here; a non-null
       * malformed range remains an owner-level construction failure. */
      if (pointer == nullptr) return false;
      if (!make_address_range(pointer, bytes, writable)) {
        error =
            std::string("CUDA runtime admission audit found an invalid writable range for ") + name;
        return true;
      }
      if (!address_ranges_overlap(admission_range, writable)) return false;
      error = std::string("CUDA runtime admission scalar aliases writable ") + name;
      return true;
    };
    const auto reject_arena = [&](const char* name, const auto& arena) -> bool {
      AddressRange writable{};
      if (!make_address_range(arena.get(), arena.bytes(), writable)) return false;
      if (!address_ranges_overlap(admission_range, writable)) return false;
      error = std::string("CUDA runtime admission scalar aliases writable ") + name + " arena";
      return true;
    };

    /* Bulk restores and several component composers write complete arenas.
     * Leaf validators prove their internal projections; the owner-level audit
     * additionally keeps the call-wide gate outside every such destination. */
    if (reject_arena("SCC iteration", candidate.iteration_arena) ||
        reject_arena("eigensolver cache", candidate.eigensolver_setup_arena) ||
        reject_arena("numerical refresh", candidate.numerical_refresh_arena) ||
        reject_arena("energy/force execution", candidate.force_execution_arena) ||
        reject_arena("inference", candidate.inference_arena)) {
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    const auto& numerical = candidate.numerical.device;
    const std::int64_t batch = numerical.batch_size;
    const std::int64_t atoms = numerical.total_atoms;
    const std::int64_t matrices = numerical.total_matrices;
    std::int64_t atom_coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t field_coordinates = 0;
    std::int64_t dipole_integrals = 0;
    std::int64_t quadrupole_integrals = 0;
    if (!checked_elements(atoms, 3, atom_coordinates) ||
        !checked_elements(numerical.total_point_charges, 3, point_coordinates) ||
        !checked_elements(batch, 3, field_coordinates) ||
        !checked_elements(matrices, 3, dipole_integrals) ||
        !checked_elements(matrices, 6, quadrupole_integrals)) {
      error = "CUDA runtime admission audit extent overflows int64_t";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
#define VIBEQC_XTB_REJECT_WRITE(name, pointer, elements, type) \
  if (reject_overlap(name, pointer, elements, sizeof(type)))   \
  return VIBEQC_XTB_STATUS_INVALID_ARGUMENT
    VIBEQC_XTB_REJECT_WRITE("numerical requested mask", numerical.requested, batch, std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("numerical eligibility mask", numerical.eligible, batch, std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("overlap factor activity", numerical.factor_active, batch,
                            std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("committed generations", numerical.committed_generations, batch,
                            std::uint64_t);
    VIBEQC_XTB_REJECT_WRITE("refresh predecessor generations",
                            numerical.refresh_predecessor_generations, batch, std::uint64_t);
    VIBEQC_XTB_REJECT_WRITE("warm checkpoint generations", numerical.warm_checkpoint_generations,
                            batch, std::uint64_t);
    VIBEQC_XTB_REJECT_WRITE("candidate positions", numerical.candidate_positions, atom_coordinates,
                            double);
    VIBEQC_XTB_REJECT_WRITE("committed positions", numerical.committed_positions, atom_coordinates,
                            double);
    VIBEQC_XTB_REJECT_WRITE("candidate point-charge positions", numerical.candidate_point_positions,
                            numerical.point_enabled != 0u ? point_coordinates : 0, double);
    VIBEQC_XTB_REJECT_WRITE("candidate point-charge values", numerical.candidate_point_values,
                            numerical.point_enabled != 0u ? numerical.total_point_charges : 0,
                            double);
    VIBEQC_XTB_REJECT_WRITE("candidate point-charge gammas", numerical.candidate_point_gammas,
                            numerical.point_enabled != 0u ? numerical.total_point_charges : 0,
                            double);
    VIBEQC_XTB_REJECT_WRITE("committed point-charge positions", numerical.committed_point_positions,
                            numerical.point_enabled != 0u ? point_coordinates : 0, double);
    VIBEQC_XTB_REJECT_WRITE("committed point-charge values", numerical.committed_point_values,
                            numerical.point_enabled != 0u ? numerical.total_point_charges : 0,
                            double);
    VIBEQC_XTB_REJECT_WRITE("committed point-charge gammas", numerical.committed_point_gammas,
                            numerical.point_enabled != 0u ? numerical.total_point_charges : 0,
                            double);
    VIBEQC_XTB_REJECT_WRITE("candidate periodic shifts", numerical.candidate_periodic_shifts,
                            numerical.periodic_enabled != 0u ? atoms : 0, double);
    VIBEQC_XTB_REJECT_WRITE(
        "candidate periodic response", numerical.candidate_periodic_response,
        numerical.periodic_enabled != 0u ? numerical.total_response_elements : 0, double);
    VIBEQC_XTB_REJECT_WRITE("committed periodic shifts", numerical.committed_periodic_shifts,
                            numerical.periodic_enabled != 0u ? atoms : 0, double);
    VIBEQC_XTB_REJECT_WRITE(
        "committed periodic response", numerical.committed_periodic_response,
        numerical.periodic_enabled != 0u ? numerical.total_response_elements : 0, double);
    VIBEQC_XTB_REJECT_WRITE("candidate field attachment", numerical.candidate_field_attached, batch,
                            std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("candidate field vectors", numerical.candidate_field_vectors,
                            field_coordinates, double);
    VIBEQC_XTB_REJECT_WRITE("candidate field atomic potentials",
                            numerical.candidate_field_atomic_potentials, atoms, double);
    VIBEQC_XTB_REJECT_WRITE("candidate field dipole potentials",
                            numerical.candidate_field_dipole_potentials, atom_coordinates, double);
    VIBEQC_XTB_REJECT_WRITE("committed field attachment", numerical.committed_field_attached, batch,
                            std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("committed field vectors", numerical.committed_field_vectors,
                            field_coordinates, double);
    VIBEQC_XTB_REJECT_WRITE("committed field atomic potentials",
                            numerical.committed_field_atomic_potentials, atoms, double);
    VIBEQC_XTB_REJECT_WRITE("committed field dipole potentials",
                            numerical.committed_field_dipole_potentials, atom_coordinates, double);
    VIBEQC_XTB_REJECT_WRITE("warm checkpoint field attachment",
                            numerical.warm_checkpoint_field_attached, batch, std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("warm checkpoint field vectors",
                            numerical.warm_checkpoint_field_vectors, field_coordinates, double);
    VIBEQC_XTB_REJECT_WRITE("interaction request diagnostic", numerical.interaction_request_error,
                            1, std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("field system diagnostics", numerical.field_system_errors, batch,
                            std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("field plan diagnostic", numerical.field_plan_error, 1, std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("periodic system diagnostics", numerical.periodic_system_errors,
                            numerical.periodic_enabled != 0u ? batch : 0, std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("periodic plan diagnostic", numerical.periodic_plan_error,
                            numerical.periodic_enabled != 0u ? 1 : 0, std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("public geometry pairs", numerical.public_geometry_pairs,
                            numerical.geometry_pair_elements, double);
    VIBEQC_XTB_REJECT_WRITE("public coordination", numerical.public_coordination, atoms, double);
    VIBEQC_XTB_REJECT_WRITE("public overlap", numerical.public_overlap, matrices, double);
    VIBEQC_XTB_REJECT_WRITE("public dipole integrals", numerical.public_dipole, dipole_integrals,
                            double);
    VIBEQC_XTB_REJECT_WRITE("public quadrupole integrals", numerical.public_quadrupole,
                            quadrupole_integrals, double);
    VIBEQC_XTB_REJECT_WRITE("public H0", numerical.public_h0, matrices, double);
    VIBEQC_XTB_REJECT_WRITE("public ES2 cache", numerical.public_es2, numerical.es2_elements,
                            double);
    VIBEQC_XTB_REJECT_WRITE("public AES2 cache", numerical.public_aes2, numerical.aes2_elements,
                            double);
    VIBEQC_XTB_REJECT_WRITE("public D4 coordination", numerical.public_d4_coordination,
                            numerical.d4_enabled != 0u ? atoms : 0, double);
    VIBEQC_XTB_REJECT_WRITE("public point-charge shell potential", numerical.public_point_shell,
                            numerical.point_enabled != 0u ? numerical.total_shells : 0, double);
    VIBEQC_XTB_REJECT_WRITE("overlap factor generations", numerical.factor_generations, batch,
                            std::uint64_t);
    VIBEQC_XTB_REJECT_WRITE("overlap factor statuses", numerical.factor_statuses, batch,
                            std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("geometry epoch", numerical.geometry_epoch, 1, std::uint64_t);

    const auto& mixer = candidate.state_seed.mixer;
    VIBEQC_XTB_REJECT_WRITE("warm mixer residual RMS", mixer.residual_rms, mixer.batch_elements,
                            double);
    VIBEQC_XTB_REJECT_WRITE("warm mixer residual maximum", mixer.residual_maximum,
                            mixer.batch_elements, double);
    VIBEQC_XTB_REJECT_WRITE("warm mixer iterations", mixer.iterations, mixer.batch_elements,
                            std::uint64_t);
    VIBEQC_XTB_REJECT_WRITE("warm mixer statuses", mixer.system_statuses, mixer.batch_elements,
                            vibeqc_xtb_status_t);
    VIBEQC_XTB_REJECT_WRITE("warm mixer convergence", mixer.residual_converged,
                            mixer.batch_elements, std::uint8_t);
    const auto& scc = candidate.state_seed.scc;
    VIBEQC_XTB_REJECT_WRITE("SCC free energies", scc.free_energies, scc.batch_elements, double);
    VIBEQC_XTB_REJECT_WRITE("SCC previous free energies", scc.previous_free_energies,
                            scc.batch_elements, double);
    VIBEQC_XTB_REJECT_WRITE("SCC free-energy changes", scc.free_energy_changes, scc.batch_elements,
                            double);
    VIBEQC_XTB_REJECT_WRITE("SCC residual RMS", scc.residual_rms, scc.batch_elements, double);
    VIBEQC_XTB_REJECT_WRITE("SCC iterations", scc.iterations, scc.batch_elements, std::uint64_t);
    VIBEQC_XTB_REJECT_WRITE("SCC statuses", scc.system_statuses, scc.batch_elements,
                            vibeqc_xtb_status_t);
    VIBEQC_XTB_REJECT_WRITE("SCC convergence", scc.converged, scc.batch_elements, std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("SCC report system diagnostics", candidate.report_storage.system_errors,
                            candidate.report_storage.system_error_elements, std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("SCC report device diagnostics", candidate.report_storage.device_errors,
                            candidate.report_storage.device_error_elements, std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("SCC report sequence latches",
                            candidate.report_storage.sequence_latches,
                            candidate.report_storage.sequence_latch_elements, std::uint32_t);

    const auto& projection = candidate.energy_force.stationary_projection;
    if (projection.enabled == 1u) {
      VIBEQC_XTB_REJECT_WRITE("stationary total density", projection.total_density,
                              projection.total_matrix_elements, double);
      VIBEQC_XTB_REJECT_WRITE("stationary energy-weighted density",
                              projection.total_energy_weighted_density,
                              projection.total_matrix_elements, double);
      VIBEQC_XTB_REJECT_WRITE("stationary spin density", projection.spin_density,
                              projection.total_matrix_elements, double);
      VIBEQC_XTB_REJECT_WRITE("stationary shell charges", projection.shell_charges,
                              projection.total_shells, double);
      VIBEQC_XTB_REJECT_WRITE("stationary atomic charges", projection.atomic_charges,
                              projection.total_atoms, double);
      VIBEQC_XTB_REJECT_WRITE("stationary atomic dipoles", projection.atomic_dipoles,
                              3 * projection.total_atoms, double);
      VIBEQC_XTB_REJECT_WRITE("stationary atomic quadrupoles", projection.atomic_quadrupoles,
                              6 * projection.total_atoms, double);
      VIBEQC_XTB_REJECT_WRITE("stationary spin-shell potentials", projection.spin_shell_potentials,
                              projection.total_shells, double);
    }

    const auto& inference = candidate.inference;
    VIBEQC_XTB_REJECT_WRITE("warm checkpoint aggregate", inference.warm_checkpoint_batch_ready, 1,
                            std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("warm checkpoint generation publication",
                            inference.warm_checkpoint_generations, batch, std::uint64_t);
    VIBEQC_XTB_REJECT_WRITE("warm checkpoint attachment publication",
                            inference.warm_checkpoint_field_attached, batch, std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("warm checkpoint field publication",
                            inference.warm_checkpoint_field_vectors, field_coordinates, double);
    VIBEQC_XTB_REJECT_WRITE("inference energy results", inference.publication_results.energies,
                            inference.publication_results.energy_elements, double);
    VIBEQC_XTB_REJECT_WRITE("inference force results", inference.publication_results.qm_forces,
                            inference.publication_results.qm_force_elements, double);
    VIBEQC_XTB_REJECT_WRITE("inference charge results",
                            inference.publication_results.atomic_charges,
                            inference.publication_results.atomic_charge_elements, double);
    VIBEQC_XTB_REJECT_WRITE("inference point-force results",
                            inference.publication_results.point_forces,
                            inference.publication_results.point_force_elements, double);
    VIBEQC_XTB_REJECT_WRITE("inference iteration results", inference.publication_results.iterations,
                            inference.publication_results.batch_elements, std::int32_t);
    VIBEQC_XTB_REJECT_WRITE("inference convergence results",
                            inference.publication_results.converged,
                            inference.publication_results.batch_elements, std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("inference status results",
                            inference.publication_results.system_statuses,
                            inference.publication_results.batch_elements, vibeqc_xtb_status_t);
    VIBEQC_XTB_REJECT_WRITE("inference dipole results",
                            inference.publication_results.dipole_moments,
                            inference.publication_results.dipole_moment_elements, double);
    VIBEQC_XTB_REJECT_WRITE("inference epoch snapshot",
                            inference.publication_workspace.epoch_snapshot,
                            inference.publication_workspace.epoch_snapshot_elements, std::uint64_t);
    VIBEQC_XTB_REJECT_WRITE("inference system diagnostics",
                            inference.publication_diagnostics.system_errors,
                            inference.publication_diagnostics.system_error_elements, std::uint32_t);
    VIBEQC_XTB_REJECT_WRITE("inference plan diagnostic",
                            inference.publication_diagnostics.plan_error,
                            inference.publication_diagnostics.plan_error_elements, std::uint32_t);

    const auto& public_result = candidate.public_result;
    VIBEQC_XTB_REJECT_WRITE("public-result energy staging", public_result.device_staging.energies,
                            public_result.device_staging.energy_elements, double);
    VIBEQC_XTB_REJECT_WRITE("public-result force staging", public_result.device_staging.qm_forces,
                            public_result.device_staging.qm_force_elements, double);
    VIBEQC_XTB_REJECT_WRITE("public-result charge staging",
                            public_result.device_staging.atomic_charges,
                            public_result.device_staging.atomic_charge_elements, double);
    VIBEQC_XTB_REJECT_WRITE("public-result point-force staging",
                            public_result.device_staging.point_forces,
                            public_result.device_staging.point_force_elements, double);
    VIBEQC_XTB_REJECT_WRITE("public-result iteration staging",
                            public_result.device_staging.iterations,
                            public_result.device_staging.batch_elements, std::int32_t);
    VIBEQC_XTB_REJECT_WRITE("public-result convergence staging",
                            public_result.device_staging.converged,
                            public_result.device_staging.batch_elements, std::uint8_t);
    VIBEQC_XTB_REJECT_WRITE("public-result status staging",
                            public_result.device_staging.system_statuses,
                            public_result.device_staging.batch_elements, vibeqc_xtb_status_t);
    VIBEQC_XTB_REJECT_WRITE("public-result dipole staging",
                            public_result.device_staging.dipole_moments,
                            public_result.device_staging.dipole_moment_elements, double);
    VIBEQC_XTB_REJECT_WRITE("public-result diagnostics", public_result.diagnostics.control,
                            public_result.diagnostics.control_elements,
                            Gfn2PublicResultBridgeControl);
#undef VIBEQC_XTB_REJECT_WRITE
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  static bool request_semantic_rejection(const PublicResultState& state) noexcept {
    const auto aggregate =
        static_cast<Gfn2PublicResultBridgeError>(state.host_control->aggregate_error);
    return aggregate == Gfn2PublicResultBridgeError::kRequestTopologyMismatch ||
           aggregate == Gfn2PublicResultBridgeError::kRequestInvalidArgument ||
           aggregate == Gfn2PublicResultBridgeError::kRequestNotSupported ||
           aggregate == Gfn2PublicResultBridgeError::kRequestNotImplemented ||
           aggregate == Gfn2PublicResultBridgeError::kRequestWarmIncompatible;
  }

  Impl(std::int32_t selected_device, void* selected_stream) noexcept
      : device_id(selected_device),
        stream(reinterpret_cast<cudaStream_t>(selected_stream)),
        topology_staging(selected_device, selected_stream) {}

  ~Impl() {
    std::lock_guard<std::mutex> lock(mutex);
    /* CUDA current-device state is thread-local. Context teardown can run on a
     * different thread or after the caller selected another device. Select the
     * owner only for teardown and restore the caller's selection afterwards;
     * destructors cannot report a restoration failure, so both operations are
     * necessarily best effort. */
    int caller_device = -1;
    const bool caller_device_known = cudaGetDevice(&caller_device) == cudaSuccess;
    bool owner_device_selected = caller_device_known && caller_device == device_id;
    bool restore_caller_device = false;
    if (!owner_device_selected && cudaSetDevice(device_id) == cudaSuccess) {
      owner_device_selected = true;
      restore_caller_device = caller_device_known && caller_device != device_id;
    }

    // Synchronous inference settles all work before returning. Fence again
    // before destroying handles in case setup failed after a submission.
    if (owner_device_selected && handles_created) (void)cudaStreamSynchronize(stream);
    if (prepared != nullptr) {
      prepared.reset();
    }

    if (blas != nullptr) (void)cublasDestroy(blas);
    if (solver_jacobi != nullptr) (void)cusolverDnDestroySyevjInfo(solver_jacobi);
    if (solver_parameters != nullptr) (void)cusolverDnDestroyParams(solver_parameters);
    if (solver != nullptr) (void)cusolverDnDestroy(solver);
    if (restore_caller_device) (void)cudaSetDevice(caller_device);
  }

  /* Exception-only safety net for asynchronous admission while mutex is held.
   * Normal status failures use the diagnostic-preserving settlement helpers;
   * this no-throw path exists so an allocation or string exception can never
   * strand queued work after the public Request rolls back its reservation. */

  vibeqc_xtb_status_t ensure_handles(std::string& error) {
    /* cudaSetDevice is intentionally repeated on every prepare. CUDA current
     * device selection is thread-local and is not preserved by this context. */
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice", cuda_status);
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }
    if (!ensure_cuda_gfn2_parameters(device_id, error)) {
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }
    if (handles_created) return VIBEQC_XTB_STATUS_SUCCESS;

    /* Publish context handles only after the entire construction succeeds.
     * This keeps retries leak-free after a partial provider failure. */
    cusolverDnHandle_t candidate_solver = nullptr;
    cusolverDnParams_t candidate_parameters = nullptr;
    syevjInfo_t candidate_jacobi = nullptr;
    cublasHandle_t candidate_blas = nullptr;
    const auto destroy_candidate = [&]() noexcept {
      if (candidate_blas != nullptr) (void)cublasDestroy(candidate_blas);
      if (candidate_jacobi != nullptr) (void)cusolverDnDestroySyevjInfo(candidate_jacobi);
      if (candidate_parameters != nullptr) (void)cusolverDnDestroyParams(candidate_parameters);
      if (candidate_solver != nullptr) (void)cusolverDnDestroy(candidate_solver);
    };

    cusolverStatus_t solver_status = cusolverDnCreate(&candidate_solver);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      error = "cusolverDnCreate failed";
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnCreateParams(&candidate_parameters);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      destroy_candidate();
      error = "cusolverDnCreateParams failed";
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnCreateSyevjInfo(&candidate_jacobi);
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status =
          cusolverDnXsyevjSetTolerance(candidate_jacobi, std::numeric_limits<double>::epsilon());
    }
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status = cusolverDnXsyevjSetMaxSweeps(candidate_jacobi, 100);
    }
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
      solver_status = cusolverDnXsyevjSetSortEig(candidate_jacobi, 1);
    }
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      destroy_candidate();
      error = "failed to configure the CUDA small-matrix Jacobi eigensolver";
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }
    cublasStatus_t blas_status = cublasCreate(&candidate_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      destroy_candidate();
      error = "cublasCreate failed";
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }
    solver_status = cusolverDnSetStream(candidate_solver, stream);
    blas_status = cublasSetStream(candidate_blas, stream);
    if (solver_status != CUSOLVER_STATUS_SUCCESS || blas_status != CUBLAS_STATUS_SUCCESS) {
      destroy_candidate();
      error = "failed to bind CUDA linear-algebra handles to the context stream";
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }
    solver = candidate_solver;
    solver_parameters = candidate_parameters;
    solver_jacobi = candidate_jacobi;
    blas = candidate_blas;
    handles_created = true;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t build_numerical_refresh_binding(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t shells = candidate.host.basis.total_shells;
    const std::int64_t orbitals = candidate.host.basis.total_orbitals;
    const std::int64_t primitives = candidate.host.basis.total_primitives;
    const std::int64_t matrices = candidate.host.integrals.total_matrix_elements;
    const std::int64_t points = candidate.host.external.total_point_charges;
    const std::int64_t response =
        static_cast<std::int64_t>(candidate.host.periodic_response.size());
    const std::int64_t geometry_pairs = candidate.plan_seed.geometry_batch.total_pairs;
    const std::int64_t es2_elements = candidate.plan_seed.es2_batch.total_matrix_elements;
    const std::int64_t aes2_pairs = candidate.plan_seed.aes2_batch.total_pairs;
    const std::uint64_t token = candidate.host.plan_token;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t geometry_pair_elements = 0;
    std::int64_t aes2_elements = 0;
    std::int64_t dipole_elements = 0;
    std::int64_t quadrupole_elements = 0;
    /* Fixed-topology capacities for the optional sparse pair-list gate.  The
     * cell arrays use a per-system trailing slot so adjacent systems never
     * race on the exclusive-end sentinel. */
    std::int64_t maximum_system_atoms = 0;
    for (std::int64_t system = 0; system < batch; ++system) {
      maximum_system_atoms =
          std::max(maximum_system_atoms,
                   candidate.host.basis.atom_offsets[static_cast<std::size_t>(system + 1)] -
                       candidate.host.basis.atom_offsets[static_cast<std::size_t>(system)]);
    }
    /* Per-system sparse/dense dispatch decisions, uploaded once at setup. Each
     * peer independently crosses the measured 40-atom crossover, so a
     * heterogeneous batch no longer applies one strategy to every member. D4
     * still forces the leaf to exist for small dense peers; dense mode then
     * publishes the full triangle into the same committed fixed-capacity
     * transaction used by sparse peers. */
    std::vector<std::int32_t> pairlist_system_modes(static_cast<std::size_t>(batch));
    for (std::int64_t system = 0; system < batch; ++system) {
      const std::int64_t atoms_per_system =
          candidate.host.basis.atom_offsets[static_cast<std::size_t>(system + 1)] -
          candidate.host.basis.atom_offsets[static_cast<std::size_t>(system)];
      pairlist_system_modes[static_cast<std::size_t>(system)] = static_cast<std::int32_t>(
          vibeqc::xtb::detail::cuda::gfn2_pairlist_use_sparse_for(atoms_per_system)
              ? vibeqc::xtb::detail::cuda::Gfn2PairListMode::kSparse
              : vibeqc::xtb::detail::cuda::Gfn2PairListMode::kDense);
    }
    std::int64_t scaled_cells = 0;
    if (!checked_elements(maximum_system_atoms, 8, scaled_cells)) {
      error = "numerical refresh sparse cell capacity overflows int64_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    std::int64_t sparse_cells_per_system = std::max<std::int64_t>(16, scaled_cells);
    std::int64_t sparse_neighbors_per_atom = maximum_system_atoms;
    std::int64_t sparse_pairs_per_system = 0;
    if (!checked_triangle(maximum_system_atoms, sparse_pairs_per_system)) {
      error = "numerical refresh sparse pair capacity overflows int64_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    /* The committed pair-list schema requires a positive fixed slot capacity
     * even when every system is a singleton. Reserve one inert pair slot per
     * peer; builders still publish pair_count=0, so no synthetic pair becomes
     * visible to D4 or coordination consumers. */
    sparse_pairs_per_system = std::max<std::int64_t>(1, sparse_pairs_per_system);
    std::int64_t sparse_cell_capacity = 0;
    std::int64_t sparse_neighbor_capacity = 0;
    std::int64_t sparse_pair_capacity = 0;
    if (!checked_elements(batch, sparse_cells_per_system, sparse_cell_capacity) ||
        !checked_elements(atoms, sparse_neighbors_per_atom, sparse_neighbor_capacity) ||
        !checked_elements(batch, sparse_pairs_per_system, sparse_pair_capacity) ||
        sparse_cell_capacity > std::numeric_limits<std::int64_t>::max() - batch) {
      error = "numerical refresh sparse pair-list capacity overflows int64_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(geometry_pairs, kGfn2GeometryPairDataElements, geometry_pair_elements) ||
        !checked_elements(aes2_pairs, kGfn2AES2PairDataElements, aes2_elements) ||
        !checked_elements(matrices, 3, dipole_elements) ||
        !checked_elements(matrices, 6, quadrupole_elements)) {
      error = "numerical refresh element count overflows int64_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }

    struct Offsets {
      std::size_t shell_pair_offsets = 0u;
      std::size_t shell_primitive_offsets = 0u;
      std::size_t angular_momenta = 0u;
      std::size_t primitive_exponents = 0u;
      std::size_t primitive_coefficients = 0u;
      std::size_t h0_atomic_radii = 0u;
      std::size_t h0_shell_levels = 0u;
      std::size_t h0_coordination_scale = 0u;
      std::size_t h0_polynomial = 0u;
      std::size_t h0_pair_scale = 0u;

      std::size_t committed_positions = 0u;
      std::size_t committed_point_positions = 0u;
      std::size_t committed_point_values = 0u;
      std::size_t committed_point_gammas = 0u;
      std::size_t committed_periodic_shifts = 0u;
      std::size_t committed_periodic_response = 0u;
      std::size_t committed_field_attached = 0u;
      std::size_t committed_field_vectors = 0u;
      std::size_t candidate_field_attached = 0u;
      std::size_t candidate_field_vectors = 0u;
      std::size_t committed_field_atomic_potentials = 0u;
      std::size_t committed_field_dipole_potentials = 0u;
      std::size_t candidate_field_atomic_potentials = 0u;
      std::size_t candidate_field_dipole_potentials = 0u;
      std::size_t field_system_errors = 0u;
      std::size_t field_plan_error = 0u;
      std::size_t interaction_request_error = 0u;
      std::size_t candidate_positions = 0u;
      std::size_t candidate_point_positions = 0u;
      std::size_t candidate_point_values = 0u;
      std::size_t candidate_point_gammas = 0u;
      std::size_t candidate_periodic_shifts = 0u;
      std::size_t candidate_periodic_response = 0u;
      std::size_t host_positions = 0u;
      std::size_t requested = 0u;
      std::size_t eligible = 0u;
      std::size_t committed_generations = 0u;
      std::size_t refresh_predecessor_generations = 0u;
      std::size_t geometry_epoch = 0u;

      std::size_t output_geometry_pairs = 0u;
      std::size_t output_coordination = 0u;
      std::size_t output_geometry_generations = 0u;
      std::size_t output_overlap = 0u;
      std::size_t output_dipole = 0u;
      std::size_t output_quadrupole = 0u;
      std::size_t output_h0 = 0u;
      std::size_t output_es2 = 0u;
      std::size_t output_aes2 = 0u;
      std::size_t output_operator_generations = 0u;
      std::size_t output_published = 0u;

      std::size_t positions_scratch = 0u;
      std::size_t geometry_candidate_pairs = 0u;
      std::size_t geometry_candidate_coordination = 0u;
      std::size_t geometry_candidate_generations = 0u;
      std::size_t geometry_pair_scratch = 0u;
      std::size_t geometry_coordination_scratch = 0u;
      std::size_t geometry_sequence = 0u;
      std::size_t overlap_candidate = 0u;
      std::size_t dipole_candidate = 0u;
      std::size_t quadrupole_candidate = 0u;
      std::size_t h0_candidate = 0u;
      std::size_t overlap_scratch = 0u;
      std::size_t dipole_scratch = 0u;
      std::size_t quadrupole_scratch = 0u;
      std::size_t h0_scratch = 0u;
      std::size_t integral_sequence = 0u;
      std::size_t es2_candidate = 0u;
      std::size_t es2_scratch = 0u;
      std::size_t aes2_candidate = 0u;
      std::size_t aes2_scratch = 0u;
      std::size_t geometry_system_errors = 0u;
      std::size_t geometry_device_error = 0u;
      std::size_t integral_system_errors = 0u;
      std::size_t integral_device_error = 0u;
      std::size_t es2_device_error = 0u;
      std::size_t aes2_system_errors = 0u;
      std::size_t aes2_device_error = 0u;
      std::size_t preprocessing_stages = 0u;
      std::size_t preprocessing_plan_error = 0u;
      std::size_t sparse_system_errors = 0u;
      std::size_t sparse_device_error = 0u;
      std::size_t sparse_coordination = 0u;
      std::size_t pairlist_pairs = 0u;
      std::size_t pairlist_offsets = 0u;
      std::size_t pairlist_pair_counts = 0u;
      std::size_t pairlist_neighbor_offsets = 0u;
      std::size_t pairlist_neighbor_counts = 0u;
      std::size_t pairlist_neighbors = 0u;
      std::size_t pairlist_generations = 0u;
      std::size_t pairlist_system_modes = 0u;
      std::size_t pairlist_meta = 0u;
      std::size_t pairlist_atom_cells = 0u;
      std::size_t pairlist_cell_counts = 0u;
      std::size_t pairlist_cell_offsets = 0u;
      std::size_t pairlist_cell_fill = 0u;
      std::size_t pairlist_cell_atoms = 0u;
      std::size_t pairlist_neighbor_cursor = 0u;
      std::size_t pairlist_neighbor_scratch = 0u;
      std::size_t pairlist_pair_cursor = 0u;
      std::size_t pairlist_sequence = 0u;

      /* Committed public pair-list view (preprocessing step 4).  Distinct
       * storage from the candidate so a failed peer can never expose a partial
       * candidate slice and consumers read only eligible, current peers. */
      std::size_t committed_pairs = 0u;
      std::size_t committed_pair_offsets = 0u;
      std::size_t committed_pair_counts = 0u;
      std::size_t committed_neighbor_offsets = 0u;
      std::size_t committed_neighbor_counts = 0u;
      std::size_t committed_neighbors = 0u;
      std::size_t committed_pair_generations = 0u;
      std::size_t committed_eligible_mask = 0u;

      std::size_t d4_coordination_scratch = 0u;
      std::size_t d4_system_errors = 0u;
      std::size_t d4_device_error = 0u;
      std::size_t point_candidate_shell = 0u;
      std::size_t point_shell_scratch = 0u;
      std::size_t point_system_errors = 0u;
      std::size_t point_plan_error = 0u;
      std::size_t point_sequence = 0u;
      std::size_t periodic_system_errors = 0u;
      std::size_t periodic_plan_error = 0u;
    } offset;

    struct HostStagingOffsets {
      std::size_t positions = 0u;
    } host_offset;

    ArenaLayout layout;
    offset.shell_pair_offsets = layout.append<std::int64_t>(
        static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()));
    offset.shell_primitive_offsets = layout.append<std::int64_t>(
        static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()));
    offset.angular_momenta = layout.append<std::uint8_t>(
        static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()));
    offset.primitive_exponents = layout.append<double>(primitives);
    offset.primitive_coefficients = layout.append<double>(primitives);
    offset.h0_atomic_radii = layout.append<double>(atoms);
    offset.h0_shell_levels = layout.append<double>(shells);
    offset.h0_coordination_scale = layout.append<double>(shells);
    offset.h0_polynomial = layout.append<double>(shells);
    offset.h0_pair_scale = layout.append<double>(candidate.host.h0.shell_pair_offsets.back());

    const auto append_numerical =
        [&](std::size_t& positions_offset, std::size_t& point_positions_offset,
            std::size_t& point_values_offset, std::size_t& point_gammas_offset,
            std::size_t& shifts_offset, std::size_t& response_offset) {
          positions_offset = layout.append<double>(coordinates);
          point_positions_offset = layout.append<double>(point_coordinates);
          point_values_offset = layout.append<double>(points);
          point_gammas_offset = layout.append<double>(points);
          shifts_offset = layout.append<double>(candidate.host.periodic_enabled ? atoms : 0);
          response_offset = layout.append<double>(candidate.host.periodic_enabled ? response : 0);
        };
    append_numerical(offset.committed_positions, offset.committed_point_positions,
                     offset.committed_point_values, offset.committed_point_gammas,
                     offset.committed_periodic_shifts, offset.committed_periodic_response);
    append_numerical(offset.candidate_positions, offset.candidate_point_positions,
                     offset.candidate_point_values, offset.candidate_point_gammas,
                     offset.candidate_periodic_shifts, offset.candidate_periodic_response);
    offset.host_positions = layout.append<double>(coordinates);
    offset.committed_field_attached = layout.append<std::uint8_t>(batch);
    offset.committed_field_vectors = layout.append<double>(3 * batch);
    offset.candidate_field_attached = layout.append<std::uint8_t>(batch);
    offset.candidate_field_vectors = layout.append<double>(3 * batch);
    offset.committed_field_atomic_potentials = layout.append<double>(atoms);
    offset.committed_field_dipole_potentials = layout.append<double>(coordinates);
    offset.candidate_field_atomic_potentials = layout.append<double>(atoms);
    offset.candidate_field_dipole_potentials = layout.append<double>(coordinates);
    offset.field_system_errors = layout.append<std::uint32_t>(batch);
    offset.field_plan_error = layout.append<std::uint32_t>(1);
    offset.interaction_request_error = layout.append<std::uint32_t>(1);
    offset.requested = layout.append<std::uint8_t>(batch);
    offset.eligible = layout.append<std::uint8_t>(batch);
    offset.committed_generations = layout.append<std::uint64_t>(batch);
    offset.refresh_predecessor_generations = layout.append<std::uint64_t>(batch);
    offset.geometry_epoch = layout.append<std::uint64_t>(1);

    offset.output_geometry_pairs = layout.append<double>(geometry_pair_elements);
    offset.output_coordination = layout.append<double>(atoms);
    offset.output_geometry_generations = layout.append<std::uint64_t>(batch);
    offset.output_overlap = layout.append<double>(matrices);
    offset.output_dipole = layout.append<double>(dipole_elements);
    offset.output_quadrupole = layout.append<double>(quadrupole_elements);
    offset.output_h0 = layout.append<double>(matrices);
    offset.output_es2 = layout.append<double>(es2_elements);
    offset.output_aes2 = layout.append<double>(aes2_elements);
    offset.output_operator_generations = layout.append<std::uint64_t>(batch);
    offset.output_published = layout.append<std::uint8_t>(batch);

    offset.positions_scratch = layout.append<double>(coordinates);
    offset.geometry_candidate_pairs = layout.append<double>(geometry_pair_elements);
    offset.geometry_candidate_coordination = layout.append<double>(atoms);
    offset.geometry_candidate_generations = layout.append<std::uint64_t>(batch);
    offset.geometry_pair_scratch = layout.append<double>(geometry_pair_elements);
    offset.geometry_coordination_scratch = layout.append<double>(atoms);
    offset.geometry_sequence = layout.append<std::uint32_t>(1);
    offset.overlap_candidate = layout.append<double>(matrices);
    offset.dipole_candidate = layout.append<double>(dipole_elements);
    offset.quadrupole_candidate = layout.append<double>(quadrupole_elements);
    offset.h0_candidate = layout.append<double>(matrices);
    offset.overlap_scratch = layout.append<double>(matrices);
    offset.dipole_scratch = layout.append<double>(dipole_elements);
    offset.quadrupole_scratch = layout.append<double>(quadrupole_elements);
    offset.h0_scratch = layout.append<double>(matrices);
    offset.integral_sequence = layout.append<std::uint32_t>(1);
    offset.es2_candidate = layout.append<double>(es2_elements);
    offset.es2_scratch = layout.append<double>(es2_elements);
    offset.aes2_candidate = layout.append<double>(aes2_elements);
    offset.aes2_scratch = layout.append<double>(aes2_elements);
    offset.geometry_system_errors = layout.append<std::uint32_t>(batch);
    offset.geometry_device_error = layout.append<std::uint32_t>(1);
    offset.integral_system_errors = layout.append<std::uint32_t>(batch);
    offset.integral_device_error = layout.append<std::uint32_t>(1);
    offset.es2_device_error = layout.append<std::uint32_t>(1);
    offset.aes2_system_errors = layout.append<std::uint32_t>(batch);
    offset.aes2_device_error = layout.append<std::uint32_t>(1);
    offset.preprocessing_stages = layout.append<std::uint32_t>(batch);
    offset.preprocessing_plan_error = layout.append<std::uint32_t>(1);

    /* Optional sparse pair-list gate.  The segments are always appended with
     * conservative fixed-topology capacities so a later dispatch toggle does
     * not require an arena rebuild; unused segments cost bytes only. */
    offset.sparse_system_errors = layout.append<std::uint32_t>(batch);
    offset.sparse_device_error = layout.append<std::uint32_t>(1);
    offset.sparse_coordination = layout.append<double>(atoms);
    offset.pairlist_pairs = layout.append<vibeqc::xtb::detail::Gfn2AtomPair>(sparse_pair_capacity);
    offset.pairlist_offsets = layout.append<std::int64_t>(batch + 1);
    offset.pairlist_pair_counts = layout.append<std::int64_t>(batch);
    offset.pairlist_neighbor_offsets = layout.append<std::int64_t>(atoms + 1);
    offset.pairlist_neighbor_counts = layout.append<std::int64_t>(atoms);
    offset.pairlist_neighbors = layout.append<std::int64_t>(sparse_neighbor_capacity);
    offset.pairlist_generations = layout.append<std::uint64_t>(batch);
    offset.pairlist_system_modes = layout.append<std::int32_t>(batch);
    offset.pairlist_meta = layout.append<vibeqc::xtb::detail::cuda::Gfn2PairListSystemMeta>(batch);
    offset.pairlist_atom_cells = layout.append<std::int64_t>(atoms);
    const std::int64_t sparse_cell_storage = sparse_cell_capacity + batch;
    offset.pairlist_cell_counts = layout.append<std::int64_t>(sparse_cell_storage);
    offset.pairlist_cell_offsets = layout.append<std::int64_t>(sparse_cell_storage);
    offset.pairlist_cell_fill = layout.append<std::int64_t>(sparse_cell_storage);
    offset.pairlist_cell_atoms = layout.append<std::int64_t>(atoms);
    offset.pairlist_neighbor_cursor = layout.append<std::int64_t>(atoms);
    offset.pairlist_neighbor_scratch = layout.append<std::int64_t>(sparse_neighbor_capacity);
    offset.pairlist_pair_cursor = layout.append<std::int64_t>(batch);
    offset.pairlist_sequence = layout.append<std::uint32_t>(1);

    /* Committed output pair-list storage: distinct fixed-capacity slices so a
     * failed peer's slice is never exposed and later peers never shift. */
    offset.committed_pairs = layout.append<vibeqc::xtb::detail::Gfn2AtomPair>(sparse_pair_capacity);
    offset.committed_pair_offsets = layout.append<std::int64_t>(batch + 1);
    offset.committed_pair_counts = layout.append<std::int64_t>(batch);
    offset.committed_neighbor_offsets = layout.append<std::int64_t>(atoms + 1);
    offset.committed_neighbor_counts = layout.append<std::int64_t>(atoms);
    offset.committed_neighbors = layout.append<std::int64_t>(sparse_neighbor_capacity);
    offset.committed_pair_generations = layout.append<std::uint64_t>(batch);
    offset.committed_eligible_mask = layout.append<std::uint8_t>(batch);

    if (candidate.host.d4_enabled) {
      offset.d4_coordination_scratch = layout.append<double>(atoms);
      offset.d4_system_errors = layout.append<std::uint32_t>(batch);
      offset.d4_device_error = layout.append<std::uint32_t>(1);
    }
    if (points != 0) {
      offset.point_candidate_shell = layout.append<double>(shells);
      offset.point_shell_scratch = layout.append<double>(shells);
      offset.point_system_errors = layout.append<std::uint32_t>(batch);
      offset.point_plan_error = layout.append<std::uint32_t>(1);
      offset.point_sequence = layout.append<std::uint32_t>(1);
    }
    if (candidate.host.periodic_enabled) {
      offset.periodic_system_errors = layout.append<std::uint32_t>(batch);
      offset.periodic_plan_error = layout.append<std::uint32_t>(1);
    }
    if (!layout.valid()) {
      error = "numerical refresh arena layout overflows size_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }

    ArenaLayout host_layout;
    host_offset.positions = host_layout.append<double>(coordinates);
    if (!host_layout.valid()) {
      error = "numerical host-staging arena layout overflows size_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }

    cudaError_t cuda_status = candidate.numerical_refresh_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh arena allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_staging_arena.allocate(host_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical pinned host-staging allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_completion_stream.create(cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-completion stream creation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_upload_complete.create(cudaEventDisableTiming);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-upload event creation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.numerical_host_release_complete.create(cudaEventDisableTiming);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-release event creation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.numerical_refresh_arena.get();
    cuda_status = cudaMemsetAsync(arena, 0, candidate.numerical_refresh_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh arena initialization", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = true;

    const auto upload = [&](std::size_t destination, const void* source, std::size_t bytes) {
      if (cuda_status == cudaSuccess && bytes != 0u) {
        cuda_status = cudaMemcpyAsync(static_cast<std::byte*>(arena) + destination, source, bytes,
                                      cudaMemcpyHostToDevice, stream);
      }
    };
    const auto upload_vector = [&](std::size_t destination, const auto& values) {
      using Value = typename std::decay_t<decltype(values)>::value_type;
      upload(destination, values.data(), values.size() * sizeof(Value));
    };
    upload_vector(offset.shell_pair_offsets, candidate.host.h0.shell_pair_offsets);
    upload_vector(offset.shell_primitive_offsets, candidate.host.basis.shell_primitive_offsets);
    upload_vector(offset.angular_momenta, candidate.host.basis.angular_momenta);
    upload_vector(offset.primitive_exponents, candidate.host.basis.primitive_exponents);
    upload_vector(offset.primitive_coefficients, candidate.host.basis.primitive_coefficients);
    upload_vector(offset.h0_atomic_radii, candidate.host.h0.atomic_radii);
    upload_vector(offset.h0_shell_levels, candidate.host.h0.shell_levels);
    upload_vector(offset.h0_coordination_scale, candidate.host.h0.shell_coordination_scale);
    upload_vector(offset.h0_polynomial, candidate.host.h0.shell_polynomial);
    upload_vector(offset.h0_pair_scale, candidate.host.h0.shell_pair_scale);
    upload_vector(offset.pairlist_system_modes, pairlist_system_modes);
    upload_vector(offset.committed_positions, candidate.host.positions);
    upload_vector(offset.committed_point_positions, candidate.host.point_positions);
    upload_vector(offset.committed_point_values, candidate.host.point_values);
    upload_vector(offset.committed_point_gammas, candidate.host.point_gammas);
    upload_vector(offset.committed_periodic_shifts, candidate.host.periodic_shifts);
    upload_vector(offset.committed_periodic_response, candidate.host.periodic_response);
    /* Setup caches start unpublished. The synchronous transaction refreshes
     * them from the real geometry before SCC or public result publication,
     * so epoch 1 cannot advertise zero-initialized pair-list/D4 leaves. */
    std::vector<std::uint64_t> initial_generations(static_cast<std::size_t>(batch), 0u);
    upload_vector(offset.committed_generations, initial_generations);
    constexpr std::uint64_t kUnpublishedGeometryEpoch = 0u;
    upload(offset.geometry_epoch, &kUnpublishedGeometryEpoch, sizeof(kUnpublishedGeometryEpoch));
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(arena_pointer<std::uint8_t>(arena, offset.requested), 1,
                                    static_cast<std::size_t>(batch), stream);
    }
    if (cuda_status == cudaSuccess) {
      /* The dedicated eligibility vector begins unpublished and is set only
       * by the final refresh gate. It then survives SCC activity derivation so
       * terminal publication can distinguish a refresh-rejected peer after the
       * SCC ledger becomes inactive through convergence or failure. */
      cuda_status = cudaMemsetAsync(arena_pointer<std::uint8_t>(arena, offset.eligible), 0,
                                    static_cast<std::size_t>(batch), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(
          arena_pointer<std::uint64_t>(arena, offset.refresh_predecessor_generations), 0,
          static_cast<std::size_t>(batch) * sizeof(std::uint64_t), stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical refresh setup upload", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    auto& numerical = candidate.numerical;
    numerical = {};
    numerical.host_positions = arena_pointer<double>(arena, offset.host_positions);
    void* const host_arena = candidate.numerical_host_staging_arena.get();
    numerical.owned_host_positions = arena_pointer<double>(host_arena, host_offset.positions);
    auto& binding = numerical.preprocessing;
    binding = {};
    binding.plan.abi_version = kGfn2PreprocessingAbiVersion;
    binding.plan.reserved = 0u;
    binding.plan.plan_token = token;
    binding.plan.geometry = candidate.plan_seed.geometry_batch;
    std::int64_t maximum_system_shells = 0;
    for (std::int64_t system = 0; system < batch; ++system) {
      maximum_system_shells =
          std::max(maximum_system_shells,
                   candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system + 1)] -
                       candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system)]);
    }
    binding.plan.integrals = {
        batch,
        atoms,
        shells,
        orbitals,
        primitives,
        matrices,
        candidate.host.h0.shell_pair_offsets.back(),
        maximum_system_shells,
        candidate.host.integrals.integral_cutoff,
        token,
        batch + 1,
        batch + 1,
        batch + 1,
        batch + 1,
        static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()),
        atoms + 1,
        shells + 1,
        static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()),
        shells,
        static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()),
        primitives,
        primitives,
        candidate.device_topology.atom_offsets,
        candidate.device_topology.batch_shell_offsets,
        candidate.device_topology.batch_orbital_offsets,
        candidate.device_topology.matrix_offsets,
        arena_pointer<std::int64_t>(arena, offset.shell_pair_offsets),
        candidate.device_topology.atom_shell_offsets,
        candidate.device_topology.shell_orbital_offsets,
        arena_pointer<std::int64_t>(arena, offset.shell_primitive_offsets),
        candidate.device_topology.shell_to_atom,
        arena_pointer<std::uint8_t>(arena, offset.angular_momenta),
        arena_pointer<double>(arena, offset.primitive_exponents),
        arena_pointer<double>(arena, offset.primitive_coefficients)};
    binding.plan.h0 = {atoms,
                       shells,
                       shells,
                       shells,
                       candidate.host.h0.shell_pair_offsets.back(),
                       token,
                       arena_pointer<double>(arena, offset.h0_atomic_radii),
                       arena_pointer<double>(arena, offset.h0_shell_levels),
                       arena_pointer<double>(arena, offset.h0_coordination_scale),
                       arena_pointer<double>(arena, offset.h0_polynomial),
                       arena_pointer<double>(arena, offset.h0_pair_scale)};
    binding.plan.es2 = candidate.plan_seed.es2_batch;
    binding.plan.aes2 = candidate.plan_seed.aes2_batch;
    binding.input = {arena_pointer<double>(arena, offset.candidate_positions), coordinates, token};
    binding.activity = {arena_pointer<std::uint8_t>(arena, offset.requested), batch,
                        arena_pointer<std::uint8_t>(arena, offset.output_published), batch, token};
    binding.output.geometry = {
        arena_pointer_if<double>(arena, offset.output_geometry_pairs, geometry_pair_elements),
        geometry_pair_elements,
        arena_pointer<double>(arena, offset.output_coordination),
        atoms,
        arena_pointer<std::uint64_t>(arena, offset.output_geometry_generations),
        batch,
        token};
    binding.output.overlap = arena_pointer<double>(arena, offset.output_overlap);
    binding.output.overlap_elements = matrices;
    binding.output.dipole_integrals = arena_pointer<double>(arena, offset.output_dipole);
    binding.output.dipole_elements = dipole_elements;
    binding.output.quadrupole_integrals = arena_pointer<double>(arena, offset.output_quadrupole);
    binding.output.quadrupole_elements = quadrupole_elements;
    binding.output.h0 = arena_pointer<double>(arena, offset.output_h0);
    binding.output.h0_elements = matrices;
    binding.output.es2 = {arena_pointer<double>(arena, offset.output_es2), es2_elements, 0u, token};
    binding.output.aes2 = {arena_pointer_if<double>(arena, offset.output_aes2, aes2_elements),
                           aes2_elements, 0u, token};
    binding.output.operator_generations =
        arena_pointer<std::uint64_t>(arena, offset.output_operator_generations);
    binding.output.generation_elements = batch;
    binding.output.plan_token = token;
    const bool pairlist_enabled =
        candidate.host.d4_enabled ||
        vibeqc::xtb::detail::cuda::gfn2_pairlist_use_sparse_for(maximum_system_atoms);
    const double pairlist_builder_cutoff =
        candidate.host.d4_enabled ? kD4PairlistBuilderCutoffBohr
                                  : vibeqc::xtb::detail::cuda::kDefaultPairlistCutoffBohr;
    if (pairlist_enabled) {
      /* Capacities were provisioned once from fixed topology above. D4 makes
       * the leaf mandatory and widens only the physical builder superset to
       * 50 bohr. This source GFN2 coordination view remains canonical at 25
       * bohr; the D4 binding later projects its 30/50/25-bohr role views. */
      binding.plan.pairlist = {
          static_cast<std::int64_t>(batch),
          static_cast<std::int64_t>(atoms),
          static_cast<std::int64_t>(batch + 1),
          pairlist_builder_cutoff,
          sparse_cells_per_system,
          sparse_neighbors_per_atom,
          sparse_pairs_per_system,
          vibeqc::xtb::detail::cuda::Gfn2PairListMode::kSparse,
          token,
          candidate.device_topology.atom_offsets,
          vibeqc::xtb::detail::cuda::kGfn2PairListAllowDenseFallback,
          arena_pointer<std::int32_t>(arena, offset.pairlist_system_modes),
          batch,
      };
    }
    binding.diagnostics = {
        arena_pointer<std::uint32_t>(arena, offset.geometry_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.geometry_device_error),
        arena_pointer<std::uint32_t>(arena, offset.integral_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.integral_device_error),
        arena_pointer<std::uint32_t>(arena, offset.es2_device_error),
        arena_pointer<std::uint32_t>(arena, offset.aes2_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.aes2_device_error),
        arena_pointer<std::uint32_t>(arena, offset.preprocessing_stages),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.preprocessing_plan_error),
        pairlist_enabled ? arena_pointer<std::uint32_t>(arena, offset.sparse_system_errors)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::uint32_t>(arena, offset.sparse_device_error)
                         : nullptr,
        token};
    binding.workspace.positions_scratch = arena_pointer<double>(arena, offset.positions_scratch);
    binding.workspace.position_elements = coordinates;
    binding.workspace.geometry_candidate = {
        arena_pointer_if<double>(arena, offset.geometry_candidate_pairs, geometry_pair_elements),
        geometry_pair_elements,
        arena_pointer<double>(arena, offset.geometry_candidate_coordination),
        atoms,
        arena_pointer<std::uint64_t>(arena, offset.geometry_candidate_generations),
        batch,
        token};
    binding.workspace.geometry = {
        arena_pointer_if<double>(arena, offset.geometry_pair_scratch, geometry_pair_elements),
        geometry_pair_elements,
        arena_pointer<double>(arena, offset.geometry_coordination_scratch),
        atoms,
        nullptr,
        0,
        arena_pointer<std::uint32_t>(arena, offset.geometry_sequence),
        1,
        token};
    binding.workspace.sparse_coordination =
        pairlist_enabled ? arena_pointer<double>(arena, offset.sparse_coordination) : nullptr;
    binding.workspace.sparse_coordination_elements = pairlist_enabled ? atoms : 0;
    binding.workspace.pairlist_candidate = {
        pairlist_enabled
            ? arena_pointer<vibeqc::xtb::detail::Gfn2AtomPair>(arena, offset.pairlist_pairs)
            : nullptr,
        pairlist_enabled ? sparse_pair_capacity : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_offsets) : nullptr,
        pairlist_enabled ? batch + 1 : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_pair_counts)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_offsets)
                         : nullptr,
        pairlist_enabled ? atoms + 1 : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_counts)
                         : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbors) : nullptr,
        pairlist_enabled ? sparse_neighbor_capacity : 0,
        pairlist_enabled ? arena_pointer<std::uint64_t>(arena, offset.pairlist_generations)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? token : 0u};
    binding.workspace.pairlist = {
        pairlist_enabled ? arena_pointer<vibeqc::xtb::detail::cuda::Gfn2PairListSystemMeta>(
                               arena, offset.pairlist_meta)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_atom_cells) : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_counts)
                         : nullptr,
        pairlist_enabled ? sparse_cell_capacity + batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_offsets)
                         : nullptr,
        pairlist_enabled ? sparse_cell_capacity + batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_fill) : nullptr,
        pairlist_enabled ? sparse_cell_capacity + batch : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_cell_atoms) : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_cursor)
                         : nullptr,
        pairlist_enabled ? atoms : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_neighbor_scratch)
                         : nullptr,
        pairlist_enabled ? sparse_neighbor_capacity : 0,
        pairlist_enabled ? arena_pointer<std::int64_t>(arena, offset.pairlist_pair_cursor)
                         : nullptr,
        pairlist_enabled ? batch : 0,
        pairlist_enabled ? arena_pointer<std::uint32_t>(arena, offset.pairlist_sequence) : nullptr,
        pairlist_enabled ? 1 : 0,
        pairlist_enabled ? token : 0u};
    /* Committed pair-list consumer view published through the final per-system
     * gate (preprocessing step 4).  Uses dedicated output storage so a failed
     * peer never exposes a partial candidate slice; only eligible peers with a
     * current committed generation are consumable.  A disabled leaf keeps the
     * canonical empty viewed with a zero token. */
    binding.output.pairlist =
        pairlist_enabled
            ? vibeqc::xtb::detail::
                  Gfn2PairListConsumerView{vibeqc::xtb::detail::Gfn2PlanMemorySpace::kCudaDevice,
                                           vibeqc::xtb::detail::Gfn2PairListState::kCommitted,
                                           vibeqc::xtb::detail::Gfn2PairListRole::kCoordination,
                                           vibeqc::xtb::detail::Gfn2PairMapKind::kExplicit,
                                           token,
                                           vibeqc::xtb::detail::cuda::kDefaultPairlistCutoffBohr,
                                           pairlist_builder_cutoff,
                                           static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(atoms),
                                           sparse_pairs_per_system,
                                           sparse_neighbors_per_atom,
                                           batch + 1,
                                           atoms + 1,
                                           sparse_pair_capacity,
                                           sparse_neighbor_capacity,
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_pair_offsets),
                                           arena_pointer<vibeqc::xtb::detail::Gfn2AtomPair>(
                                               arena, offset.committed_pairs),
                                           static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(atoms),
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_pair_counts),
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_neighbor_counts),
                                           arena_pointer<std::int64_t>(
                                               arena, offset.committed_neighbor_offsets),
                                           arena_pointer<std::int64_t>(arena,
                                                                       offset.committed_neighbors),
                                           static_cast<std::int64_t>(batch),
                                           static_cast<std::int64_t>(batch),
                                           0,
                                           arena_pointer<std::uint64_t>(
                                               arena, offset.committed_pair_generations),
                                           arena_pointer<std::uint8_t>(
                                               arena, offset.committed_eligible_mask),
                                           nullptr}
            : vibeqc::xtb::detail::Gfn2PairListConsumerView{};
    binding.workspace.overlap_candidate = arena_pointer<double>(arena, offset.overlap_candidate);
    binding.workspace.overlap_elements = matrices;
    binding.workspace.dipole_candidate = arena_pointer<double>(arena, offset.dipole_candidate);
    binding.workspace.dipole_elements = dipole_elements;
    binding.workspace.quadrupole_candidate =
        arena_pointer<double>(arena, offset.quadrupole_candidate);
    binding.workspace.quadrupole_elements = quadrupole_elements;
    binding.workspace.h0_candidate = arena_pointer<double>(arena, offset.h0_candidate);
    binding.workspace.h0_elements = matrices;
    binding.workspace.integrals = {arena_pointer<double>(arena, offset.overlap_scratch),
                                   matrices,
                                   arena_pointer<double>(arena, offset.dipole_scratch),
                                   dipole_elements,
                                   arena_pointer<double>(arena, offset.quadrupole_scratch),
                                   quadrupole_elements,
                                   arena_pointer<double>(arena, offset.h0_scratch),
                                   matrices,
                                   arena_pointer<std::uint32_t>(arena, offset.integral_sequence),
                                   1,
                                   token};
    binding.workspace.es2_candidate = {arena_pointer<double>(arena, offset.es2_candidate),
                                       es2_elements, 0u, token};
    binding.workspace.es2 = {arena_pointer<double>(arena, offset.es2_scratch),
                             es2_elements,
                             nullptr,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             0};
    binding.workspace.aes2_candidate = {
        arena_pointer_if<double>(arena, offset.aes2_candidate, aes2_elements), aes2_elements, 0u,
        token};
    binding.workspace.aes2 = {arena_pointer_if<double>(arena, offset.aes2_scratch, aes2_elements),
                              aes2_elements,
                              nullptr,
                              0,
                              nullptr,
                              0,
                              nullptr,
                              0,
                              nullptr,
                              0,
                              nullptr,
                              0};
    binding.workspace.plan_token = token;
    binding.geometry_epoch = {arena_pointer<std::uint64_t>(arena, offset.geometry_epoch), 1, token};
    binding.plan_token = token;

    const auto seal = seal_gfn2_preprocessing_binding_cuda(binding);
    if (!seal.success()) {
      error = "CUDA runtime preprocessing binding rejected its fixed arena projection";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    auto& device = numerical.device;
    device = {};
    device.batch_size = batch;
    device.total_atoms = atoms;
    device.total_shells = shells;
    device.total_matrices = matrices;
    device.total_point_charges = points;
    device.total_response_elements = response;
    device.geometry_pair_elements = geometry_pair_elements;
    device.es2_elements = es2_elements;
    device.aes2_elements = aes2_elements;
    device.d4_enabled = candidate.host.d4_enabled ? 1u : 0u;
    device.point_enabled = points != 0 ? 1u : 0u;
    device.periodic_enabled = candidate.host.periodic_enabled ? 1u : 0u;
    device.plan_token = token;
    device.atom_offsets = candidate.device_topology.atom_offsets;
    device.shell_offsets = candidate.device_topology.batch_shell_offsets;
    device.matrix_offsets = candidate.device_topology.matrix_offsets;
    device.geometry_pair_offsets = candidate.plan_seed.geometry_batch.pair_offsets;
    device.es2_offsets = candidate.plan_seed.es2_batch.matrix_offsets;
    device.point_offsets = candidate.plan_seed.explicit_point_charge_batch.point_charge_offsets;
    device.response_offsets = candidate.plan_seed.periodic_batch.matrix_offsets;
    device.requested = binding.activity.requested_mask;
    device.preprocessing_published = binding.activity.published_mask;
    device.eligible = arena_pointer<std::uint8_t>(arena, offset.eligible);
    device.factor_active = candidate.workspace_seed.ledger.active_mask;
    device.committed_generations =
        arena_pointer<std::uint64_t>(arena, offset.committed_generations);
    device.refresh_predecessor_generations =
        arena_pointer<std::uint64_t>(arena, offset.refresh_predecessor_generations);
    device.candidate_positions = arena_pointer<double>(arena, offset.candidate_positions);
    device.candidate_point_positions =
        arena_pointer_if<double>(arena, offset.candidate_point_positions, point_coordinates);
    device.candidate_point_values =
        arena_pointer_if<double>(arena, offset.candidate_point_values, points);
    device.candidate_point_gammas =
        arena_pointer_if<double>(arena, offset.candidate_point_gammas, points);
    device.candidate_periodic_shifts = arena_pointer_if<double>(
        arena, offset.candidate_periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    device.candidate_periodic_response = arena_pointer_if<double>(
        arena, offset.candidate_periodic_response, candidate.host.periodic_enabled ? response : 0);
    device.committed_positions = arena_pointer<double>(arena, offset.committed_positions);
    device.committed_point_positions =
        arena_pointer_if<double>(arena, offset.committed_point_positions, point_coordinates);
    device.committed_point_values =
        arena_pointer_if<double>(arena, offset.committed_point_values, points);
    device.committed_point_gammas =
        arena_pointer_if<double>(arena, offset.committed_point_gammas, points);
    device.committed_periodic_shifts = arena_pointer_if<double>(
        arena, offset.committed_periodic_shifts, candidate.host.periodic_enabled ? atoms : 0);
    device.committed_periodic_response = arena_pointer_if<double>(
        arena, offset.committed_periodic_response, candidate.host.periodic_enabled ? response : 0);
    device.committed_field_attached =
        arena_pointer<std::uint8_t>(arena, offset.committed_field_attached);
    device.committed_field_vectors = arena_pointer<double>(arena, offset.committed_field_vectors);
    device.candidate_field_attached =
        arena_pointer<std::uint8_t>(arena, offset.candidate_field_attached);
    device.candidate_field_vectors = arena_pointer<double>(arena, offset.candidate_field_vectors);
    device.candidate_field_atomic_potentials =
        arena_pointer<double>(arena, offset.candidate_field_atomic_potentials);
    device.candidate_field_dipole_potentials =
        arena_pointer<double>(arena, offset.candidate_field_dipole_potentials);
    device.committed_field_atomic_potentials =
        arena_pointer<double>(arena, offset.committed_field_atomic_potentials);
    device.committed_field_dipole_potentials =
        arena_pointer<double>(arena, offset.committed_field_dipole_potentials);
    device.interaction_request_error =
        arena_pointer<std::uint32_t>(arena, offset.interaction_request_error);
    device.field_system_errors = arena_pointer<std::uint32_t>(arena, offset.field_system_errors);
    device.field_plan_error = arena_pointer<std::uint32_t>(arena, offset.field_plan_error);
    device.candidate_geometry_pairs = binding.output.geometry.pair_data;
    device.candidate_coordination = binding.output.geometry.coordination_numbers;
    device.candidate_overlap = binding.output.overlap;
    device.candidate_dipole = binding.output.dipole_integrals;
    device.candidate_quadrupole = binding.output.quadrupole_integrals;
    device.candidate_h0 = binding.output.h0;
    device.candidate_es2 = binding.output.es2.coulomb_matrix;
    device.candidate_aes2 = binding.output.aes2.pair_data;
    device.public_geometry_pairs =
        const_cast<double*>(candidate.plan_seed.geometry_cache.pair_data);
    device.public_coordination =
        const_cast<double*>(candidate.plan_seed.geometry_cache.coordination_numbers);
    device.public_overlap = const_cast<double*>(candidate.input_seed.hamiltonian.overlap);
    device.public_dipole = const_cast<double*>(candidate.input_seed.hamiltonian.dipole_integrals);
    device.public_quadrupole =
        const_cast<double*>(candidate.input_seed.hamiltonian.quadrupole_integrals);
    device.public_h0 = const_cast<double*>(candidate.input_seed.hamiltonian.h0);
    device.public_es2 = candidate.plan_seed.es2_cache.coulomb_matrix;
    device.public_aes2 = candidate.plan_seed.aes2_cache.pair_data;
    device.preprocessing_plan_error = binding.diagnostics.plan_error;
    device.geometry_epoch = binding.geometry_epoch.value;

    if (candidate.host.d4_enabled) {
      numerical.d4_workspace.coordination_scratch =
          arena_pointer<double>(arena, offset.d4_coordination_scratch);
      numerical.d4_workspace.coordination_scratch_elements = atoms;
      numerical.d4_workspace.system_errors =
          arena_pointer<std::uint32_t>(arena, offset.d4_system_errors);
      numerical.d4_workspace.system_error_elements = batch;
      device.candidate_d4_coordination = numerical.d4_workspace.coordination_scratch;
      device.public_d4_coordination = candidate.plan_seed.d4_pairlist_cache.coordination_numbers;
      device.d4_system_errors = numerical.d4_workspace.system_errors;
      device.d4_device_error = arena_pointer<std::uint32_t>(arena, offset.d4_device_error);

      Gfn2PairListConsumerView d4_coordination_pairs{};
      Gfn2PairListConsumerView d4_two_body_pairs{};
      Gfn2PairListConsumerView d4_atm_pairs{};
      const auto d4_coordination_projection = project_gfn2_pair_list_role_binding(
          candidate.device_topology, binding.output.pairlist, Gfn2PairListRole::kD4Coordination,
          Gfn2PlanMemorySpace::kCudaDevice, d4_coordination_pairs);
      const auto d4_two_body_projection = project_gfn2_pair_list_role_binding(
          candidate.device_topology, binding.output.pairlist, Gfn2PairListRole::kD4TwoBody,
          Gfn2PlanMemorySpace::kCudaDevice, d4_two_body_pairs);
      const auto d4_atm_projection = project_gfn2_pair_list_role_binding(
          candidate.device_topology, binding.output.pairlist, Gfn2PairListRole::kD4Atm,
          Gfn2PlanMemorySpace::kCudaDevice, d4_atm_pairs);
      if (d4_coordination_projection.error != Gfn2PlanSchemaError::kSuccess ||
          d4_two_body_projection.error != Gfn2PlanSchemaError::kSuccess ||
          d4_atm_projection.error != Gfn2PlanSchemaError::kSuccess) {
        error = "CUDA runtime rejected a D4 role projection of the committed pair-list superset";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }

      /* The role views borrow one committed structural transaction. D4 CN is
       * built into scratch after that transaction commits, then the terminal
       * numerical gate alone copies it to the public outlet and advances the
       * common generation/eligibility authority. This keeps Graph replay on a
       * device epoch and prevents a D4-local failure from publishing stale CN. */
      numerical.d4_refresh_cache = {binding.workspace.positions_scratch,
                                    coordinates,
                                    device.public_d4_coordination,
                                    atoms,
                                    device.committed_generations,
                                    batch,
                                    device.eligible,
                                    batch,
                                    d4_coordination_pairs,
                                    d4_two_body_pairs,
                                    d4_atm_pairs,
                                    token};
      numerical.d4_cache = {device.committed_positions,
                            coordinates,
                            device.public_d4_coordination,
                            atoms,
                            device.committed_generations,
                            batch,
                            device.eligible,
                            batch,
                            d4_coordination_pairs,
                            d4_two_body_pairs,
                            d4_atm_pairs,
                            token};
      candidate.plan_seed.d4_pairlist_cache = numerical.d4_cache;
    }
    if (points != 0) {
      numerical.point_batch = candidate.plan_seed.explicit_point_charge_batch;
      numerical.point_batch.qm_positions = device.candidate_positions;
      numerical.point_batch.point_positions = device.candidate_point_positions;
      numerical.point_batch.point_charges = device.candidate_point_values;
      numerical.point_batch.point_hardnesses = device.candidate_point_gammas;
      numerical.point_candidate = {arena_pointer<double>(arena, offset.point_candidate_shell),
                                   shells, candidate.host.geometry_generation, token};
      numerical.point_workspace = {arena_pointer<double>(arena, offset.point_shell_scratch),
                                   shells};
      numerical.point_activity = {binding.activity.published_mask,
                                  arena_pointer<std::uint32_t>(arena, offset.point_sequence), batch,
                                  1, token};
      device.candidate_point_shell = numerical.point_candidate.shell_potentials;
      device.public_point_shell = candidate.plan_seed.explicit_point_charge_cache.shell_potentials;
      device.point_system_errors = arena_pointer<std::uint32_t>(arena, offset.point_system_errors);
      device.point_plan_error = arena_pointer<std::uint32_t>(arena, offset.point_plan_error);
    }
    if (candidate.host.periodic_enabled) {
      device.periodic_system_errors =
          arena_pointer<std::uint32_t>(arena, offset.periodic_system_errors);
      device.periodic_plan_error = arena_pointer<std::uint32_t>(arena, offset.periodic_plan_error);
    }
    numerical.field_batch = {batch, atoms, batch + 1, candidate.device_topology.atom_offsets,
                             token};
    numerical.field_candidate_potentials = {device.candidate_field_atomic_potentials, atoms,
                                            device.candidate_field_dipole_potentials, coordinates,
                                            token};
    numerical.field_committed_potentials = {device.committed_field_atomic_potentials, atoms,
                                            device.committed_field_dipole_potentials, coordinates,
                                            token};
    numerical.field_potential_view = {numerical.field_committed_potentials.atomic, atoms,
                                      numerical.field_committed_potentials.dipole, coordinates,
                                      token};

    /* Canonical runtime numerical pointers replace setup-only copies. */
    candidate.plan_seed.geometry_cache.geometry_generations = device.committed_generations;
    candidate.plan_seed.geometry_cache.generation_elements = batch;
    if (points != 0) {
      candidate.plan_seed.explicit_point_charge_batch.qm_positions = device.committed_positions;
      candidate.plan_seed.explicit_point_charge_batch.point_positions =
          device.committed_point_positions;
      candidate.plan_seed.explicit_point_charge_batch.point_charges = device.committed_point_values;
      candidate.plan_seed.explicit_point_charge_batch.point_hardnesses =
          device.committed_point_gammas;
    }
    if (candidate.host.periodic_enabled) {
      candidate.plan_seed.periodic_batch.shifts = device.committed_periodic_shifts;
      candidate.plan_seed.periodic_batch.response_matrices = device.committed_periodic_response;
    }

    std::vector<Gfn2SccCacheProvenanceBinding> provenance;
    provenance.reserve(
        static_cast<std::size_t>(candidate.plan_seed.provenance.cache_binding_count));
    const auto append_provenance = [&](Gfn2SccStageId stage) {
      Gfn2SccCacheProvenanceBinding record{};
      record.provenance = {Gfn2PlanMemorySpace::kCudaDevice,
                           Gfn2GenerationScope::kPerSystem,
                           token,
                           0u,
                           batch,
                           batch,
                           device.committed_generations};
      record.owner_stage = stage;
      provenance.push_back(record);
    };
    append_provenance(Gfn2SccStageId::kGeometry);
    append_provenance(Gfn2SccStageId::kES2Potential);
    append_provenance(Gfn2SccStageId::kAES2Potential);
    if (candidate.host.d4_enabled) append_provenance(Gfn2SccStageId::kD4Potential);
    if (points != 0) append_provenance(Gfn2SccStageId::kExplicitPointChargePotential);
    if (candidate.host.periodic_enabled) append_provenance(Gfn2SccStageId::kPeriodicPotential);
    if (static_cast<std::int64_t>(provenance.size()) !=
        candidate.plan_seed.provenance.cache_binding_count) {
      error = "runtime refresh provenance count disagrees with the setup owner";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaMemcpyAsync(
        const_cast<Gfn2SccCacheProvenanceBinding*>(candidate.plan_seed.provenance.cache_bindings),
        provenance.data(), provenance.size() * sizeof(provenance.front()), cudaMemcpyHostToDevice,
        stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA runtime provenance publication", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    numerical.ready = true;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t build_energy_force_bindings(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t shells = candidate.host.basis.total_shells;
    const std::int64_t orbitals = candidate.host.basis.total_orbitals;
    const std::int64_t primitives = candidate.host.basis.total_primitives;
    const std::int64_t matrices = candidate.host.integrals.total_matrix_elements;
    const std::int64_t points = candidate.host.external.total_point_charges;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t dipole_matrix_elements = 0;
    std::int64_t quadrupole_matrix_elements = 0;
    std::int64_t quadrupole_elements = 0;
    std::int64_t d4_weight_elements = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(matrices, 3, dipole_matrix_elements) ||
        !checked_elements(matrices, 6, quadrupole_matrix_elements) ||
        !checked_elements(atoms, 6, quadrupole_elements) ||
        !checked_elements(atoms, kGfn2D4MaximumReferences, d4_weight_elements)) {
      error = "force descriptor element count overflows int64_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    const bool force_mode = (candidate.host.key.flags &
                             (static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_FORCES) |
                              static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_POINT_CHARGE_FORCES) |
                              static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_DIPOLE_MOMENTS))) != 0u;
    const bool explicit_points = points != 0;
    const bool d4_enabled = candidate.host.d4_enabled;
    const bool periodic_enabled = candidate.host.periodic_enabled;
    const std::uint64_t token = candidate.host.plan_token;

    struct ImmutableOffsets {
      std::size_t atomic_numbers = 0u;
      std::size_t positions = 0u;
      std::size_t point_offsets = 0u;
      std::size_t shell_pair_offsets = 0u;
      std::size_t shell_primitive_offsets = 0u;
      std::size_t angular_momenta = 0u;
      std::size_t primitive_exponents = 0u;
      std::size_t primitive_coefficients = 0u;
      std::size_t h0_atomic_radii = 0u;
      std::size_t h0_shell_levels = 0u;
      std::size_t h0_coordination_scale = 0u;
      std::size_t h0_polynomial = 0u;
      std::size_t h0_pair_scale = 0u;
    } immutable;

    ArenaLayout immutable_layout;
    if (force_mode) {
      immutable.atomic_numbers = immutable_layout.append<std::int32_t>(atoms);
      immutable.positions = immutable_layout.append<double>(explicit_points ? 0 : coordinates);
      /* A zero-point plan still needs a valid all-zero ragged offset leaf for
       * force composition. Real point-charge plans reuse the SCC input owner. */
      immutable.point_offsets =
          immutable_layout.append<std::int64_t>(explicit_points ? 0 : batch + 1);
      immutable.shell_pair_offsets = immutable_layout.append<std::int64_t>(
          static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()));
      immutable.shell_primitive_offsets = immutable_layout.append<std::int64_t>(
          static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()));
      immutable.angular_momenta = immutable_layout.append<std::uint8_t>(
          static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()));
      immutable.primitive_exponents = immutable_layout.append<double>(primitives);
      immutable.primitive_coefficients = immutable_layout.append<double>(primitives);
      immutable.h0_atomic_radii = immutable_layout.append<double>(atoms);
      immutable.h0_shell_levels = immutable_layout.append<double>(shells);
      immutable.h0_coordination_scale = immutable_layout.append<double>(shells);
      immutable.h0_polynomial = immutable_layout.append<double>(shells);
      immutable.h0_pair_scale =
          immutable_layout.append<double>(candidate.host.h0.shell_pair_offsets.back());
    }
    if (!immutable_layout.valid()) {
      error = "force immutable arena layout overflows size_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.force_immutable_arena.allocate(immutable_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force immutable arena allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    const auto upload = [&](std::size_t offset, const void* source, std::size_t bytes) {
      if (bytes == 0u) return cudaSuccess;
      return cudaMemcpyAsync(
          static_cast<std::byte*>(candidate.force_immutable_arena.get()) + offset, source, bytes,
          cudaMemcpyHostToDevice, stream);
    };
    if (force_mode) {
      const auto upload_immutable = [&](std::size_t offset, const auto& source) {
        if (cuda_status != cudaSuccess) return;
        cuda_status = upload(offset, source.data(), source.size() * sizeof(source[0]));
      };
      upload_immutable(immutable.atomic_numbers, candidate.host.key.atomic_numbers);
      if (!explicit_points) {
        upload_immutable(immutable.positions, candidate.host.positions);
        upload_immutable(immutable.point_offsets, candidate.host.external.point_charge_offsets);
      }
      upload_immutable(immutable.shell_pair_offsets, candidate.host.h0.shell_pair_offsets);
      upload_immutable(immutable.shell_primitive_offsets,
                       candidate.host.basis.shell_primitive_offsets);
      upload_immutable(immutable.angular_momenta, candidate.host.basis.angular_momenta);
      upload_immutable(immutable.primitive_exponents, candidate.host.basis.primitive_exponents);
      upload_immutable(immutable.primitive_coefficients,
                       candidate.host.basis.primitive_coefficients);
      upload_immutable(immutable.h0_atomic_radii, candidate.host.h0.atomic_radii);
      upload_immutable(immutable.h0_shell_levels, candidate.host.h0.shell_levels);
      upload_immutable(immutable.h0_coordination_scale, candidate.host.h0.shell_coordination_scale);
      upload_immutable(immutable.h0_polynomial, candidate.host.h0.shell_polynomial);
      upload_immutable(immutable.h0_pair_scale, candidate.host.h0.shell_pair_scale);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force immutable upload", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = candidate.submitted || force_mode;

    enum SequenceIndex : std::int64_t {
      kTotalSequence,
      kPostSequence,
      kPostCompositionSequence,
      kPostBridgeSequence,
      kPostPeriodicSequence,
      kH0Sequence,
      kHamiltonianSequence,
      kIntegralSequence,
      kCoordinationSequence,
      kClassicalSequence,
      kClassicalPrimitiveSequence,
      kExternalSequence,
      kCompositionSequence,
      kSequenceCount,
    };
    enum MaskIndex : std::int64_t {
      kRequestedMask,
      kEnergySuccessMask,
      kPostSuccessMask,
      kElectronicSuccessMask,
      kCoordinationSuccessMask,
      kClassicalSuccessMask,
      kExternalSuccessMask,
      kH0SuccessMask,
      kHamiltonianSuccessMask,
      kClassicalSelectedMask,
      kPostActiveMask,
      kMaskCount,
    };
    enum SystemErrorIndex : std::int64_t {
      kExecutionSystemError,
      kTotalSystemError,
      kPostStageSystemError,
      kPostSystemError,
      kH0SystemError,
      kHamiltonianSystemError,
      kIntegralSystemError,
      kCoordinationSystemError,
      kClassicalSystemError,
      kExternalSystemError,
      kCompositionSystemError,
      kClassicalPrimitiveSystemError,
      kSystemErrorCount,
    };
    enum DeviceErrorIndex : std::int64_t {
      kExecutionDeviceError,
      kTotalDeviceError,
      kPostStageDeviceError,
      kPostDeviceError,
      kH0DeviceError,
      kHamiltonianDeviceError,
      kIntegralDeviceError,
      kCoordinationDeviceError,
      kClassicalDeviceError,
      kExternalDeviceError,
      kCompositionPlanError,
      kClassicalPrimitiveDeviceError,
      kPostAes2PeerError,
      kDeviceErrorCount,
    };

    struct ExecutionOffsets {
      std::size_t repulsion = 0u;
      std::size_t d4_atm = 0u;
      std::size_t staged_energy = 0u;
      std::size_t public_energy = 0u;
      std::size_t sequences = 0u;
      std::size_t plan_failure = 0u;
      std::size_t masks = 0u;
      std::size_t system_errors = 0u;
      std::size_t device_errors = 0u;

      std::size_t public_qm_force = 0u;
      std::size_t public_point_force = 0u;
      std::size_t staged_qm_force = 0u;
      std::size_t staged_point_force = 0u;
      std::size_t stationary_density = 0u;
      std::size_t stationary_weighted_density = 0u;
      std::size_t stationary_spin_density = 0u;
      std::size_t stationary_shell_charges = 0u;
      std::size_t stationary_atomic_charges = 0u;
      std::size_t stationary_atomic_dipoles = 0u;
      std::size_t stationary_atomic_quadrupoles = 0u;
      std::size_t stationary_spin_shell_potential = 0u;
      std::size_t post_complete_shell = 0u;
      std::size_t post_complete_atom = 0u;
      std::size_t post_complete_dipole = 0u;
      std::size_t post_complete_quadrupole = 0u;
      std::size_t post_shell_scalar = 0u;
      std::size_t post_es2_shell = 0u;
      std::size_t post_es3_shell = 0u;
      std::size_t post_aes2_atom = 0u;
      std::size_t post_aes2_dipole = 0u;
      std::size_t post_aes2_quadrupole = 0u;
      std::size_t post_d4_atom = 0u;
      std::size_t post_periodic_atom = 0u;
      std::size_t post_staged_shell = 0u;
      std::size_t post_staged_atom = 0u;
      std::size_t post_staged_dipole = 0u;
      std::size_t post_staged_quadrupole = 0u;
      std::size_t post_staged_shell_scalar = 0u;
      std::size_t overlap_adjoint = 0u;
      std::size_t coordination_adjoint = 0u;
      std::size_t dipole_adjoint = 0u;
      std::size_t quadrupole_adjoint = 0u;
      std::size_t electronic_gradient = 0u;
      std::size_t classical_force = 0u;
      std::size_t explicit_qm_force = 0u;
      std::size_t explicit_point_force = 0u;

      std::size_t post_es2_shell_scratch = 0u;
      std::size_t post_aes2_potential_scratch = 0u;
      std::size_t post_d4_weights = 0u;
      std::size_t post_d4_weight_cn = 0u;
      std::size_t post_d4_weight_charge = 0u;
      std::size_t post_d4_atom_scratch = 0u;
      std::size_t post_d4_coordination = 0u;
      std::size_t post_d4_batch = 0u;
      std::size_t post_d4_gradient = 0u;
      std::size_t post_periodic_potential = 0u;
      std::size_t post_periodic_raw_response = 0u;
      std::size_t post_composition_shell = 0u;
      std::size_t post_composition_atom = 0u;
      std::size_t post_composition_dipole = 0u;
      std::size_t post_composition_quadrupole = 0u;
      std::size_t post_bridge_shell = 0u;
      std::size_t h0_overlap_scratch = 0u;
      std::size_t h0_coordination_scratch = 0u;
      std::size_t h0_gradient_scratch = 0u;
      std::size_t hamiltonian_overlap_scratch = 0u;
      std::size_t hamiltonian_dipole_scratch = 0u;
      std::size_t hamiltonian_quadrupole_scratch = 0u;
      std::size_t integral_gradient_scratch = 0u;
      std::size_t coordination_gradient_scratch = 0u;
      std::size_t classical_gradient_scratch = 0u;
      std::size_t classical_force_scratch = 0u;
      std::size_t sparse_gradient_scratch = 0u;
      std::size_t sparse_sequence = 0u;
      std::size_t classical_coordination_adjoint = 0u;
      std::size_t classical_aes2_gradient = 0u;
      std::size_t classical_aes2_coordination = 0u;
      std::size_t classical_d4_weights = 0u;
      std::size_t classical_d4_weight_cn = 0u;
      std::size_t classical_d4_weight_charge = 0u;
      std::size_t classical_d4_atom_scratch = 0u;
      std::size_t classical_d4_coordination = 0u;
      std::size_t classical_d4_batch = 0u;
      std::size_t classical_d4_gradient = 0u;
      std::size_t classical_geometry_gradient = 0u;
      std::size_t external_qm_scratch = 0u;
      std::size_t external_point_scratch = 0u;
      std::size_t composition_qm_scratch = 0u;
      std::size_t composition_point_scratch = 0u;
    } execution;

    ArenaLayout execution_layout;
    execution.repulsion = execution_layout.append<double>(batch);
    execution.d4_atm = execution_layout.append<double>(d4_enabled ? batch : 0);
    execution.staged_energy = execution_layout.append<double>(batch);
    execution.public_energy = execution_layout.append<double>(batch);
    execution.sequences = execution_layout.append<std::uint32_t>(
        force_mode ? static_cast<std::int64_t>(kSequenceCount) : 1);
    execution.plan_failure = execution_layout.append<std::uint32_t>(1);
    execution.masks = execution_layout.append<std::uint8_t>(force_mode ? kMaskCount * batch : 0);
    execution.system_errors =
        execution_layout.append<std::uint32_t>(force_mode ? kSystemErrorCount * batch : 2 * batch);
    execution.device_errors = execution_layout.append<std::uint32_t>(
        force_mode ? static_cast<std::int64_t>(kDeviceErrorCount) : 2);
    if (force_mode) {
      execution.public_qm_force = execution_layout.append<double>(coordinates);
      execution.public_point_force = execution_layout.append<double>(point_coordinates);
      execution.staged_qm_force = execution_layout.append<double>(coordinates);
      execution.staged_point_force = execution_layout.append<double>(point_coordinates);
      execution.stationary_density = execution_layout.append<double>(matrices);
      execution.stationary_weighted_density = execution_layout.append<double>(matrices);
      execution.stationary_spin_density = execution_layout.append<double>(matrices);
      execution.stationary_shell_charges = execution_layout.append<double>(shells);
      execution.stationary_atomic_charges = execution_layout.append<double>(atoms);
      execution.stationary_atomic_dipoles = execution_layout.append<double>(coordinates);
      execution.stationary_atomic_quadrupoles =
          execution_layout.append<double>(quadrupole_elements);
      execution.stationary_spin_shell_potential = execution_layout.append<double>(shells);
      execution.post_complete_shell = execution_layout.append<double>(shells);
      execution.post_complete_atom = execution_layout.append<double>(atoms);
      execution.post_complete_dipole = execution_layout.append<double>(coordinates);
      execution.post_complete_quadrupole = execution_layout.append<double>(quadrupole_elements);
      execution.post_shell_scalar = execution_layout.append<double>(shells);
      execution.post_es2_shell = execution_layout.append<double>(shells);
      execution.post_es3_shell = execution_layout.append<double>(shells);
      execution.post_aes2_atom = execution_layout.append<double>(atoms);
      execution.post_aes2_dipole = execution_layout.append<double>(coordinates);
      execution.post_aes2_quadrupole = execution_layout.append<double>(quadrupole_elements);
      execution.post_d4_atom = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.post_periodic_atom = execution_layout.append<double>(periodic_enabled ? atoms : 0);
      execution.post_staged_shell = execution_layout.append<double>(shells);
      execution.post_staged_atom = execution_layout.append<double>(atoms);
      execution.post_staged_dipole = execution_layout.append<double>(coordinates);
      execution.post_staged_quadrupole = execution_layout.append<double>(quadrupole_elements);
      execution.post_staged_shell_scalar = execution_layout.append<double>(shells);
      execution.overlap_adjoint = execution_layout.append<double>(matrices);
      execution.coordination_adjoint = execution_layout.append<double>(atoms);
      execution.dipole_adjoint = execution_layout.append<double>(dipole_matrix_elements);
      execution.quadrupole_adjoint = execution_layout.append<double>(quadrupole_matrix_elements);
      execution.electronic_gradient = execution_layout.append<double>(coordinates);
      execution.classical_force = execution_layout.append<double>(coordinates);
      execution.explicit_qm_force =
          execution_layout.append<double>(explicit_points ? coordinates : 0);
      execution.explicit_point_force =
          execution_layout.append<double>(explicit_points ? point_coordinates : 0);

      execution.post_es2_shell_scratch = execution_layout.append<double>(shells);
      execution.post_aes2_potential_scratch =
          execution_layout.append<double>(candidate.host.aes2.potential_scratch_elements());
      execution.post_d4_weights =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.post_d4_weight_cn =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.post_d4_weight_charge =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.post_d4_atom_scratch = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.post_d4_coordination = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.post_d4_batch = execution_layout.append<double>(d4_enabled ? batch : 0);
      execution.post_d4_gradient = execution_layout.append<double>(d4_enabled ? coordinates : 0);
      execution.post_periodic_potential =
          execution_layout.append<double>(periodic_enabled ? atoms : 0);
      execution.post_periodic_raw_response =
          execution_layout.append<double>(periodic_enabled ? atoms : 0);
      execution.post_composition_shell = execution_layout.append<double>(shells);
      execution.post_composition_atom = execution_layout.append<double>(atoms);
      execution.post_composition_dipole = execution_layout.append<double>(coordinates);
      execution.post_composition_quadrupole = execution_layout.append<double>(quadrupole_elements);
      execution.post_bridge_shell = execution_layout.append<double>(shells);
      execution.h0_overlap_scratch = execution_layout.append<double>(matrices);
      execution.h0_coordination_scratch = execution_layout.append<double>(atoms);
      execution.h0_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.hamiltonian_overlap_scratch = execution_layout.append<double>(matrices);
      execution.hamiltonian_dipole_scratch =
          execution_layout.append<double>(dipole_matrix_elements);
      execution.hamiltonian_quadrupole_scratch =
          execution_layout.append<double>(quadrupole_matrix_elements);
      execution.integral_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.coordination_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.classical_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.classical_force_scratch = execution_layout.append<double>(coordinates);
      execution.sparse_gradient_scratch = execution_layout.append<double>(coordinates);
      execution.sparse_sequence = execution_layout.append<std::uint32_t>(1);
      execution.classical_coordination_adjoint = execution_layout.append<double>(atoms);
      execution.classical_aes2_gradient = execution_layout.append<double>(coordinates);
      execution.classical_aes2_coordination = execution_layout.append<double>(atoms);
      execution.classical_d4_weights =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.classical_d4_weight_cn =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.classical_d4_weight_charge =
          execution_layout.append<double>(d4_enabled ? d4_weight_elements : 0);
      execution.classical_d4_atom_scratch = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.classical_d4_coordination = execution_layout.append<double>(d4_enabled ? atoms : 0);
      execution.classical_d4_batch = execution_layout.append<double>(d4_enabled ? batch : 0);
      execution.classical_d4_gradient =
          execution_layout.append<double>(d4_enabled ? coordinates : 0);
      execution.classical_geometry_gradient = execution_layout.append<double>(coordinates);
      execution.external_qm_scratch =
          execution_layout.append<double>(explicit_points ? coordinates : 0);
      execution.external_point_scratch =
          execution_layout.append<double>(explicit_points ? point_coordinates : 0);
      execution.composition_qm_scratch = execution_layout.append<double>(coordinates);
      execution.composition_point_scratch = execution_layout.append<double>(point_coordinates);
    }
    if (!execution_layout.valid()) {
      error = "force execution arena layout overflows size_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate.force_execution_arena.allocate(execution_layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force execution arena allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = cudaMemsetAsync(candidate.force_execution_arena.get(), 0,
                                  candidate.force_execution_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("force execution arena initialization", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    void* const execution_arena = candidate.force_execution_arena.get();
    void* const immutable_arena = candidate.force_immutable_arena.get();
    candidate.atomic_numbers =
        force_mode ? arena_pointer<std::int32_t>(immutable_arena, immutable.atomic_numbers)
                   : nullptr;
    auto* const repulsion = arena_pointer<double>(execution_arena, execution.repulsion);
    auto* const d4_atm =
        arena_pointer_if<double>(execution_arena, execution.d4_atm, d4_enabled ? batch : 0);
    auto* const staged_energy = arena_pointer<double>(execution_arena, execution.staged_energy);
    auto* const public_energy = arena_pointer<double>(execution_arena, execution.public_energy);
    auto* const sequence_pool = arena_pointer<std::uint32_t>(execution_arena, execution.sequences);
    auto* const plan_failure =
        arena_pointer<std::uint32_t>(execution_arena, execution.plan_failure);
    auto* const system_error_pool =
        arena_pointer<std::uint32_t>(execution_arena, execution.system_errors);
    auto* const device_error_pool =
        arena_pointer<std::uint32_t>(execution_arena, execution.device_errors);
    const auto system_errors = [&](SystemErrorIndex index) {
      return system_error_pool + static_cast<std::int64_t>(index) * batch;
    };
    const auto device_error = [&](DeviceErrorIndex index) {
      return device_error_pool + static_cast<std::int64_t>(index);
    };

    auto& binding = candidate.energy_force;
    binding = {};
    binding.plan.compute_forces = force_mode ? 1u : 0u;
    binding.plan.scc_potential_components = candidate.plan_seed.enabled_components;
    binding.plan.scc_energy_components = candidate.plan_seed.enabled_components;
    binding.plan.plan_token = token;
    const std::uint32_t total_components =
        d4_enabled ? static_cast<std::uint32_t>(Gfn2TotalEnergyComponent::kD4Atm) : 0u;
    binding.plan.total_energy_batch = {batch, total_components, token};

    binding.input.total_energy = {candidate.state_seed.scc.free_energies,
                                  batch,
                                  repulsion,
                                  batch,
                                  d4_atm,
                                  d4_enabled ? batch : 0,
                                  token};
    binding.input.scc_state = {candidate.state_seed.scc.system_statuses,
                               candidate.state_seed.scc.converged, batch, token};
    binding.input.plan_token = token;
    binding.results.energy = {public_energy, batch, token};
    binding.results.plan_token = token;
    binding.intermediates.energy = {staged_energy, batch, token};
    binding.intermediates.plan_token = token;
    binding.workspace.total_energy = {sequence_pool + kTotalSequence, 1, token};
    binding.workspace.plan_failure = plan_failure;
    binding.workspace.plan_failure_elements = 1;
    binding.workspace.plan_token = token;
    binding.diagnostics.execution_system_errors = system_errors(kExecutionSystemError);
    binding.diagnostics.execution_device_error = device_error(kExecutionDeviceError);
    binding.diagnostics.total_energy_system_errors = system_errors(kTotalSystemError);
    binding.diagnostics.total_energy_device_error = device_error(kTotalDeviceError);
    binding.diagnostics.batch_elements = batch;
    binding.diagnostics.plan_token = token;

    if (force_mode) {
      auto* const mask_pool = arena_pointer<std::uint8_t>(execution_arena, execution.masks);
      const auto mask = [&](MaskIndex index) {
        return mask_pool + static_cast<std::int64_t>(index) * batch;
      };
      cuda_status = cudaMemsetAsync(mask(kRequestedMask), 1,
                                    static_cast<std::size_t>(batch) * sizeof(std::uint8_t), stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("force request-mask initialization", cuda_status);
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
      candidate.submitted = true;

      const auto* const atomic_numbers = candidate.atomic_numbers;
      /* All force descriptors consume the same runtime-owned committed
       * coordinates, including zero-point batches. This is the address updated
       * transactionally by #122 after overlap factorization succeeds. */
      const auto* const positions = candidate.numerical.device.committed_positions;

      std::int64_t maximum_system_shells = 0;
      for (std::int64_t system = 0; system < batch; ++system) {
        maximum_system_shells = std::max(
            maximum_system_shells,
            candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system + 1)] -
                candidate.host.basis.batch_shell_offsets[static_cast<std::size_t>(system)]);
      }
      binding.plan.integral_batch = {
          batch,
          atoms,
          shells,
          orbitals,
          primitives,
          matrices,
          candidate.host.h0.shell_pair_offsets.back(),
          maximum_system_shells,
          candidate.host.integrals.integral_cutoff,
          token,
          batch + 1,
          batch + 1,
          batch + 1,
          batch + 1,
          static_cast<std::int64_t>(candidate.host.h0.shell_pair_offsets.size()),
          atoms + 1,
          shells + 1,
          static_cast<std::int64_t>(candidate.host.basis.shell_primitive_offsets.size()),
          shells,
          static_cast<std::int64_t>(candidate.host.basis.angular_momenta.size()),
          primitives,
          primitives,
          candidate.device_topology.atom_offsets,
          candidate.device_topology.batch_shell_offsets,
          candidate.device_topology.batch_orbital_offsets,
          candidate.device_topology.matrix_offsets,
          arena_pointer<std::int64_t>(immutable_arena, immutable.shell_pair_offsets),
          candidate.device_topology.atom_shell_offsets,
          candidate.device_topology.shell_orbital_offsets,
          arena_pointer<std::int64_t>(immutable_arena, immutable.shell_primitive_offsets),
          candidate.device_topology.shell_to_atom,
          arena_pointer<std::uint8_t>(immutable_arena, immutable.angular_momenta),
          arena_pointer<double>(immutable_arena, immutable.primitive_exponents),
          arena_pointer<double>(immutable_arena, immutable.primitive_coefficients)};
      binding.plan.h0_plan = {
          atoms,
          shells,
          shells,
          shells,
          candidate.host.h0.shell_pair_offsets.back(),
          token,
          arena_pointer<double>(immutable_arena, immutable.h0_atomic_radii),
          arena_pointer<double>(immutable_arena, immutable.h0_shell_levels),
          arena_pointer<double>(immutable_arena, immutable.h0_coordination_scale),
          arena_pointer<double>(immutable_arena, immutable.h0_polynomial),
          arena_pointer<double>(immutable_arena, immutable.h0_pair_scale)};
      binding.plan.hamiltonian_batch = candidate.plan_seed.hamiltonian_batch;
      binding.plan.coordination_batch = candidate.plan_seed.geometry_batch;
      binding.plan.coordination_cache = candidate.plan_seed.geometry_cache;
      binding.plan.geometry_generation = candidate.host.geometry_generation;

      Gfn2ExternalPointChargeDeviceBatch external_batch{};
      if (explicit_points) {
        external_batch = candidate.plan_seed.explicit_point_charge_batch;
      } else {
        const auto* const point_offsets =
            arena_pointer<std::int64_t>(immutable_arena, immutable.point_offsets);
        external_batch = {batch,
                          atoms,
                          shells,
                          0,
                          candidate.device_topology.atom_offsets,
                          candidate.device_topology.batch_shell_offsets,
                          point_offsets,
                          candidate.device_topology.shell_to_atom,
                          nullptr,
                          positions,
                          nullptr,
                          nullptr,
                          nullptr,
                          token};
      }
      binding.plan.external_point_charge_batch = external_batch;
      const Gfn2PeriodicEmbeddingDeviceBatch periodic_batch = candidate.plan_seed.periodic_batch;

      auto& post_plan = binding.plan.post_scc_potential_plan;
      post_plan.enabled_components = candidate.plan_seed.enabled_components;
      post_plan.geometry_generation = candidate.host.geometry_generation;
      post_plan.plan_token = token;
      post_plan.potential_batch = candidate.plan_seed.potential_batch;
      post_plan.scalar_bridge_batch = candidate.plan_seed.scalar_bridge_batch;
      post_plan.es2_batch = candidate.plan_seed.es2_batch;
      post_plan.es2_cache = candidate.plan_seed.es2_cache;
      post_plan.es3_batch = candidate.plan_seed.es3_batch;
      post_plan.aes2_batch = candidate.plan_seed.aes2_batch;
      post_plan.aes2_cache = candidate.plan_seed.aes2_cache;
      post_plan.d4_batch = candidate.plan_seed.d4_batch;
      post_plan.d4_parameters = candidate.plan_seed.d4_parameters;
      post_plan.d4_cache = candidate.numerical.d4_cache;
      post_plan.external_point_charge_batch = external_batch;
      post_plan.external_point_charge_cache = candidate.plan_seed.explicit_point_charge_cache;
      post_plan.periodic_batch = periodic_batch;

      std::uint32_t classical_components =
          static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kRepulsion) |
          static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kES2) |
          static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kAES2);
      if (d4_enabled) {
        classical_components |=
            static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody) |
            static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM);
      }
      Gfn2D4DeviceBatch classical_d4_batch = candidate.plan_seed.d4_batch;
      if (d4_enabled) {
        /* The classical-force validator requires the repulsion and nested D4
         * views to share one canonical atomic-number leaf. The SCC setup owner
         * stores an equivalent copy in its input arena, but retaining both
         * addresses here would make the composed force plan cross-projection. */
        classical_d4_batch.atomic_numbers = atomic_numbers;
      }
      binding.plan.classical_plan = {batch,
                                     atoms,
                                     shells,
                                     classical_components,
                                     candidate.host.geometry_generation,
                                     token,
                                     candidate.device_topology.atom_offsets,
                                     atomic_numbers,
                                     candidate.plan_seed.geometry_batch,
                                     candidate.plan_seed.geometry_cache,
                                     candidate.plan_seed.es2_batch,
                                     candidate.plan_seed.es2_cache,
                                     candidate.plan_seed.aes2_batch,
                                     candidate.plan_seed.aes2_cache,
                                     classical_d4_batch,
                                     candidate.plan_seed.d4_parameters,
                                     candidate.numerical.d4_cache};
      std::uint32_t composition_components =
          static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kElectronicGradient) |
          static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kClassicalForce);
      if (explicit_points) {
        composition_components |=
            static_cast<std::uint32_t>(Gfn2ForceCompositionComponent::kExplicitPointChargeForce);
      }
      binding.plan.force_composition_batch = {batch,
                                              atoms,
                                              points,
                                              batch + 1,
                                              batch + 1,
                                              candidate.device_topology.atom_offsets,
                                              external_batch.point_charge_offsets,
                                              composition_components,
                                              token};

      /* Step 5 sparse CN VJP leaf: borrow the committed pair-list consumer view
       * and the pair-list dispatch batch from the preprocessing binding.  The
       * final per-system gate already published committed generations and
       * eligibility, so the H0 CN VJP can parity-gate a sparse result against
       * the dense reference.  A disabled leaf keeps zero tokens and disables
       * the differential path. */
      binding.plan.pairlist_committed = candidate.numerical.preprocessing.output.pairlist;
      binding.plan.pairlist_batch = candidate.numerical.preprocessing.plan.pairlist;

      /* Project the committed spin-major state into stable physical arrays.
       * Every terminal force descriptor below consumes these buffers, never
       * SCC iteration scratch or the alpha channel by itself. */
      auto* const stationary_density =
          arena_pointer<double>(execution_arena, execution.stationary_density);
      auto* const stationary_weighted_density =
          arena_pointer<double>(execution_arena, execution.stationary_weighted_density);
      auto* const stationary_spin_density =
          arena_pointer<double>(execution_arena, execution.stationary_spin_density);
      auto* const stationary_shell_charges =
          arena_pointer<double>(execution_arena, execution.stationary_shell_charges);
      auto* const stationary_atomic_charges =
          arena_pointer<double>(execution_arena, execution.stationary_atomic_charges);
      auto* const stationary_atomic_dipoles =
          arena_pointer<double>(execution_arena, execution.stationary_atomic_dipoles);
      auto* const stationary_atomic_quadrupoles =
          arena_pointer<double>(execution_arena, execution.stationary_atomic_quadrupoles);
      auto* const stationary_spin_shell_potential =
          arena_pointer<double>(execution_arena, execution.stationary_spin_shell_potential);
      const Gfn2SccPotentialDeviceTopologyMultipoles physical{stationary_shell_charges,
                                                              shells,
                                                              stationary_atomic_charges,
                                                              atoms,
                                                              stationary_atomic_dipoles,
                                                              coordinates,
                                                              stationary_atomic_quadrupoles,
                                                              quadrupole_elements,
                                                              token};

      auto& projection = binding.stationary_projection;
      projection = {};
      projection.enabled = 1u;
      projection.batch_size = batch;
      projection.total_atoms = atoms;
      projection.total_shells = shells;
      projection.total_matrix_elements = matrices;
      projection.total_spin_atoms = candidate.device_wavefunction.total_spin_atoms;
      projection.total_spin_shells = candidate.device_wavefunction.total_spin_shells;
      projection.total_spin_matrix_elements =
          candidate.device_wavefunction.total_spin_matrix_elements;
      projection.atom_offsets = candidate.device_topology.atom_offsets;
      projection.shell_offsets = candidate.device_topology.batch_shell_offsets;
      projection.matrix_offsets = candidate.device_topology.matrix_offsets;
      projection.atom_shell_offsets = candidate.device_topology.atom_shell_offsets;
      projection.spin_channels = candidate.device_wavefunction.spin_channels;
      projection.spin_atom_offsets = candidate.device_wavefunction.spin_atom_offsets;
      projection.spin_shell_offsets = candidate.device_wavefunction.spin_shell_offsets;
      projection.spin_matrix_offsets = candidate.device_wavefunction.spin_matrix_offsets;
      projection.spin_coupling_offsets = candidate.plan_seed.spin_batch.coupling_offsets;
      projection.spin_coupling_matrices = candidate.plan_seed.spin_batch.coupling_matrices;
      projection.packed_density = candidate.state_seed.density.density;
      projection.packed_energy_weighted_density =
          candidate.state_seed.density.energy_weighted_density;
      projection.packed_shell_charges = candidate.state_seed.raw_population.qsh;
      projection.packed_atomic_charges = candidate.state_seed.raw_population.qat;
      projection.packed_atomic_dipoles = candidate.state_seed.raw_population.dipole;
      projection.packed_atomic_quadrupoles = candidate.state_seed.raw_population.quadrupole;
      projection.total_density = stationary_density;
      projection.total_energy_weighted_density = stationary_weighted_density;
      projection.spin_density = stationary_spin_density;
      projection.shell_charges = stationary_shell_charges;
      projection.atomic_charges = stationary_atomic_charges;
      projection.atomic_dipoles = stationary_atomic_dipoles;
      projection.atomic_quadrupoles = stationary_atomic_quadrupoles;
      projection.spin_shell_potentials = stationary_spin_shell_potential;

      binding.input.force_activity = {mask(kRequestedMask),
                                      candidate.state_seed.scc.system_statuses, batch, token};
      binding.input.post_scc_potential = {
          {mask(kRequestedMask), candidate.state_seed.scc.system_statuses, batch, token},
          physical.shell_charges,
          physical.shell_elements,
          physical.atomic_charges,
          physical.atom_elements,
          physical.atomic_dipoles,
          physical.dipole_elements,
          physical.atomic_quadrupoles,
          physical.quadrupole_elements,
          token};
      binding.input.post_scc_potential.electric_field_potentials =
          candidate.numerical.field_potential_view;
      binding.input.h0 = {positions,
                          coordinates,
                          candidate.plan_seed.geometry_cache.coordination_numbers,
                          atoms,
                          candidate.input_seed.hamiltonian.overlap,
                          matrices,
                          stationary_density,
                          matrices,
                          stationary_weighted_density,
                          matrices,
                          token};

      auto* const post_complete_shell =
          arena_pointer<double>(execution_arena, execution.post_complete_shell);
      auto* const post_complete_atom =
          arena_pointer<double>(execution_arena, execution.post_complete_atom);
      auto* const post_complete_dipole =
          arena_pointer<double>(execution_arena, execution.post_complete_dipole);
      auto* const post_complete_quadrupole =
          arena_pointer<double>(execution_arena, execution.post_complete_quadrupole);
      auto* const post_shell_scalar =
          arena_pointer<double>(execution_arena, execution.post_shell_scalar);
      binding.input.hamiltonian = {stationary_density,
                                   matrices,
                                   post_shell_scalar,
                                   shells,
                                   post_complete_dipole,
                                   coordinates,
                                   post_complete_quadrupole,
                                   quadrupole_elements,
                                   token};
      binding.input.hamiltonian.spin_density = stationary_spin_density;
      binding.input.hamiltonian.spin_density_elements = matrices;
      binding.input.hamiltonian.spin_shell_scalar_potentials = stationary_spin_shell_potential;
      binding.input.hamiltonian.spin_shell_scalar_elements = shells;
      binding.input.classical = {positions,
                                 coordinates,
                                 candidate.plan_seed.geometry_cache.coordination_numbers,
                                 atoms,
                                 physical.shell_charges,
                                 physical.shell_elements,
                                 physical.atomic_charges,
                                 physical.atom_elements,
                                 physical.atomic_dipoles,
                                 physical.dipole_elements,
                                 physical.atomic_quadrupoles,
                                 physical.quadrupole_elements,
                                 token};
      binding.input.external_shell_charges = physical.shell_charges;
      binding.input.external_shell_elements = explicit_points ? shells : 0;
      binding.input.electric_field = {
          candidate.numerical.device.committed_field_vectors,
          3 * batch,
          candidate.numerical.device.committed_positions,
          coordinates,
          token,
      };
      binding.input.raw_atomic_charges = physical.atomic_charges;
      binding.input.raw_atomic_charge_elements = atoms;

      auto* const public_qm_force =
          arena_pointer<double>(execution_arena, execution.public_qm_force);
      auto* const public_point_force = arena_pointer_if<double>(
          execution_arena, execution.public_point_force, point_coordinates);
      auto* const staged_qm_force =
          arena_pointer<double>(execution_arena, execution.staged_qm_force);
      auto* const staged_point_force = arena_pointer_if<double>(
          execution_arena, execution.staged_point_force, point_coordinates);
      binding.results.forces = {public_qm_force, coordinates, public_point_force, point_coordinates,
                                token};

      binding.intermediates.post_scc_potential = {
          {post_complete_shell, shells, post_complete_atom, atoms, post_complete_dipole,
           coordinates, post_complete_quadrupole, quadrupole_elements, token},
          post_shell_scalar,
          shells,
          token};
      auto& post_intermediates = binding.intermediates.post_scc_potential_intermediates;
      post_intermediates.es2_shell =
          arena_pointer<double>(execution_arena, execution.post_es2_shell);
      post_intermediates.es2_shell_elements = shells;
      post_intermediates.es3_shell =
          arena_pointer<double>(execution_arena, execution.post_es3_shell);
      post_intermediates.es3_shell_elements = shells;
      post_intermediates.aes2_atomic =
          arena_pointer<double>(execution_arena, execution.post_aes2_atom);
      post_intermediates.aes2_atomic_elements = atoms;
      post_intermediates.aes2_dipole =
          arena_pointer<double>(execution_arena, execution.post_aes2_dipole);
      post_intermediates.aes2_dipole_elements = coordinates;
      post_intermediates.aes2_quadrupole =
          arena_pointer<double>(execution_arena, execution.post_aes2_quadrupole);
      post_intermediates.aes2_quadrupole_elements = quadrupole_elements;
      post_intermediates.d4_atomic =
          arena_pointer_if<double>(execution_arena, execution.post_d4_atom, d4_enabled ? atoms : 0);
      post_intermediates.d4_atomic_elements = d4_enabled ? atoms : 0;
      post_intermediates.periodic_atomic = arena_pointer_if<double>(
          execution_arena, execution.post_periodic_atom, periodic_enabled ? atoms : 0);
      post_intermediates.periodic_atomic_elements = periodic_enabled ? atoms : 0;
      post_intermediates.complete = {
          arena_pointer<double>(execution_arena, execution.post_staged_shell),
          shells,
          arena_pointer<double>(execution_arena, execution.post_staged_atom),
          atoms,
          arena_pointer<double>(execution_arena, execution.post_staged_dipole),
          coordinates,
          arena_pointer<double>(execution_arena, execution.post_staged_quadrupole),
          quadrupole_elements,
          token};
      post_intermediates.shell_scalar =
          arena_pointer<double>(execution_arena, execution.post_staged_shell_scalar);
      post_intermediates.shell_scalar_elements = shells;
      post_intermediates.plan_token = token;

      auto* const overlap_adjoint =
          arena_pointer<double>(execution_arena, execution.overlap_adjoint);
      auto* const electronic_gradient =
          arena_pointer<double>(execution_arena, execution.electronic_gradient);
      binding.intermediates.h0 = {
          overlap_adjoint,
          matrices,
          arena_pointer<double>(execution_arena, execution.coordination_adjoint),
          atoms,
          electronic_gradient,
          coordinates,
          token};
      binding.intermediates.hamiltonian = {
          overlap_adjoint,
          matrices,
          arena_pointer<double>(execution_arena, execution.dipole_adjoint),
          dipole_matrix_elements,
          arena_pointer<double>(execution_arena, execution.quadrupole_adjoint),
          quadrupole_matrix_elements,
          token};
      binding.intermediates.classical = {
          arena_pointer<double>(execution_arena, execution.classical_force), coordinates, token};
      binding.intermediates.explicit_qm_forces = arena_pointer_if<double>(
          execution_arena, execution.explicit_qm_force, explicit_points ? coordinates : 0);
      binding.intermediates.explicit_qm_force_elements = explicit_points ? coordinates : 0;
      binding.intermediates.explicit_point_forces = arena_pointer_if<double>(
          execution_arena, execution.explicit_point_force, explicit_points ? point_coordinates : 0);
      binding.intermediates.explicit_point_force_elements = explicit_points ? point_coordinates : 0;
      binding.intermediates.forces = {staged_qm_force, coordinates, staged_point_force,
                                      point_coordinates, token};

      auto& post_workspace = binding.workspace.post_scc_potential;
      post_workspace.es2.shell_scratch =
          arena_pointer<double>(execution_arena, execution.post_es2_shell_scratch);
      post_workspace.es2.shell_elements = shells;
      post_workspace.aes2.potential_scratch =
          arena_pointer<double>(execution_arena, execution.post_aes2_potential_scratch);
      post_workspace.aes2.potential_elements = candidate.host.aes2.potential_scratch_elements();
      post_workspace.aes2.scc_peer_error_scratch = device_error(kPostAes2PeerError);
      post_workspace.aes2.scc_peer_error_elements = 1;
      if (d4_enabled) {
        post_workspace.d4 = {
            arena_pointer<double>(execution_arena, execution.post_d4_weights),
            arena_pointer<double>(execution_arena, execution.post_d4_weight_cn),
            arena_pointer<double>(execution_arena, execution.post_d4_weight_charge),
            d4_weight_elements,
            arena_pointer<double>(execution_arena, execution.post_d4_atom_scratch),
            arena_pointer<double>(execution_arena, execution.post_d4_coordination),
            atoms,
            arena_pointer<double>(execution_arena, execution.post_d4_batch),
            batch,
            arena_pointer<double>(execution_arena, execution.post_d4_gradient),
            coordinates,
            system_errors(kPostStageSystemError),
            batch};
      }
      if (periodic_enabled) {
        post_workspace.periodic = {
            arena_pointer<double>(execution_arena, execution.post_periodic_potential),
            arena_pointer<double>(execution_arena, execution.post_periodic_raw_response),
            sequence_pool + kPostPeriodicSequence,
            atoms,
            1,
            token};
      }
      post_workspace.composition = {
          arena_pointer<double>(execution_arena, execution.post_composition_shell),
          shells,
          arena_pointer<double>(execution_arena, execution.post_composition_atom),
          atoms,
          arena_pointer<double>(execution_arena, execution.post_composition_dipole),
          coordinates,
          arena_pointer<double>(execution_arena, execution.post_composition_quadrupole),
          quadrupole_elements,
          sequence_pool + kPostCompositionSequence,
          1,
          token};
      post_workspace.scalar_bridge = {
          arena_pointer<double>(execution_arena, execution.post_bridge_shell), shells,
          sequence_pool + kPostBridgeSequence, 1, token};
      post_workspace.active_mask = mask(kPostActiveMask);
      post_workspace.active_elements = batch;
      post_workspace.sequence_active = sequence_pool + kPostSequence;
      post_workspace.sequence_elements = 1;
      post_workspace.stage_system_errors = system_errors(kPostStageSystemError);
      post_workspace.stage_system_error_elements = batch;
      post_workspace.stage_device_error = device_error(kPostStageDeviceError);
      post_workspace.stage_device_error_elements = 1;
      post_workspace.plan_token = token;

      binding.workspace.h0 = {
          arena_pointer<double>(execution_arena, execution.h0_overlap_scratch),
          matrices,
          arena_pointer<double>(execution_arena, execution.h0_coordination_scratch),
          atoms,
          arena_pointer<double>(execution_arena, execution.h0_gradient_scratch),
          coordinates,
          sequence_pool + kH0Sequence,
          1,
          token};
      binding.workspace.hamiltonian = {
          arena_pointer<double>(execution_arena, execution.hamiltonian_overlap_scratch),
          matrices,
          arena_pointer<double>(execution_arena, execution.hamiltonian_dipole_scratch),
          dipole_matrix_elements,
          arena_pointer<double>(execution_arena, execution.hamiltonian_quadrupole_scratch),
          quadrupole_matrix_elements,
          sequence_pool + kHamiltonianSequence,
          1,
          token};
      binding.workspace.integral = {
          arena_pointer<double>(execution_arena, execution.integral_gradient_scratch), coordinates,
          sequence_pool + kIntegralSequence, 1, token};
      binding.workspace.electronic = {mask(kH0SuccessMask), mask(kHamiltonianSuccessMask), batch,
                                      token};
      binding.workspace.coordination = {
          nullptr,
          0,
          nullptr,
          0,
          arena_pointer<double>(execution_arena, execution.coordination_gradient_scratch),
          coordinates,
          sequence_pool + kCoordinationSequence,
          1,
          token};

      Gfn2AES2DeviceWorkspace classical_aes2{};
      /* Only the AES2 reverse pass is used here. Assign by name because the
       * update/potential scratch fields precede the gradient fields. */
      classical_aes2.gradient_scratch =
          arena_pointer<double>(execution_arena, execution.classical_aes2_gradient);
      classical_aes2.gradient_elements = coordinates;
      classical_aes2.coordination_scratch =
          arena_pointer<double>(execution_arena, execution.classical_aes2_coordination);
      classical_aes2.coordination_elements = atoms;
      Gfn2D4DeviceWorkspace classical_d4{};
      if (d4_enabled) {
        classical_d4 = {
            arena_pointer<double>(execution_arena, execution.classical_d4_weights),
            arena_pointer<double>(execution_arena, execution.classical_d4_weight_cn),
            arena_pointer<double>(execution_arena, execution.classical_d4_weight_charge),
            d4_weight_elements,
            arena_pointer<double>(execution_arena, execution.classical_d4_atom_scratch),
            arena_pointer<double>(execution_arena, execution.classical_d4_coordination),
            atoms,
            arena_pointer<double>(execution_arena, execution.classical_d4_batch),
            batch,
            arena_pointer<double>(execution_arena, execution.classical_d4_gradient),
            coordinates,
            system_errors(kClassicalPrimitiveSystemError),
            batch};
      }
      const Gfn2GeometryDeviceWorkspace classical_geometry{
          nullptr,
          0,
          nullptr,
          0,
          arena_pointer<double>(execution_arena, execution.classical_geometry_gradient),
          coordinates,
          sequence_pool + kClassicalPrimitiveSequence,
          1,
          token};
      binding.workspace.classical = {
          arena_pointer<double>(execution_arena, execution.classical_gradient_scratch),
          coordinates,
          arena_pointer<double>(execution_arena, execution.classical_force_scratch),
          coordinates,
          arena_pointer<double>(execution_arena, execution.classical_coordination_adjoint),
          atoms,
          mask(kClassicalSelectedMask),
          batch,
          system_errors(kClassicalPrimitiveSystemError),
          batch,
          device_error(kClassicalPrimitiveDeviceError),
          1,
          sequence_pool + kClassicalSequence,
          1,
          classical_aes2,
          classical_d4,
          classical_geometry,
          token};
      binding.workspace.external_point_charge = {
          arena_pointer_if<double>(execution_arena, execution.external_qm_scratch,
                                   explicit_points ? coordinates : 0),
          explicit_points ? coordinates : 0,
          arena_pointer_if<double>(execution_arena, execution.external_point_scratch,
                                   explicit_points ? point_coordinates : 0),
          explicit_points ? point_coordinates : 0,
          sequence_pool + kExternalSequence,
          1,
          token};
      binding.workspace.force_composition = {
          arena_pointer<double>(execution_arena, execution.composition_qm_scratch),
          coordinates,
          arena_pointer_if<double>(execution_arena, execution.composition_point_scratch,
                                   point_coordinates),
          point_coordinates,
          sequence_pool + kCompositionSequence,
          1,
          token};
      binding.workspace.energy_success_mask = mask(kEnergySuccessMask);
      binding.workspace.post_scc_success_mask = mask(kPostSuccessMask);
      binding.workspace.electronic_success_mask = mask(kElectronicSuccessMask);
      binding.workspace.coordination_success_mask = mask(kCoordinationSuccessMask);
      binding.workspace.classical_success_mask = mask(kClassicalSuccessMask);
      binding.workspace.external_success_mask = mask(kExternalSuccessMask);
      binding.workspace.mask_elements = batch;
      /* Step 5 sparse CN VJP differential scratch and sequence. */
      binding.workspace.sparse_gradient_scratch =
          arena_pointer<double>(execution_arena, execution.sparse_gradient_scratch);
      binding.workspace.sparse_gradient_elements = coordinates;
      binding.workspace.sparse_sequence_active =
          arena_pointer<std::uint32_t>(execution_arena, execution.sparse_sequence);
      binding.workspace.sparse_sequence_elements = 1;

      binding.diagnostics.post_scc_potential = {system_errors(kPostSystemError),
                                                device_error(kPostDeviceError), batch, token};
      binding.diagnostics.electronic = {system_errors(kH0SystemError),
                                        device_error(kH0DeviceError),
                                        system_errors(kHamiltonianSystemError),
                                        device_error(kHamiltonianDeviceError),
                                        system_errors(kIntegralSystemError),
                                        device_error(kIntegralDeviceError),
                                        batch,
                                        token};
      binding.diagnostics.coordination_system_errors = system_errors(kCoordinationSystemError);
      binding.diagnostics.coordination_device_error = device_error(kCoordinationDeviceError);
      binding.diagnostics.classical_system_errors = system_errors(kClassicalSystemError);
      binding.diagnostics.classical_device_error = device_error(kClassicalDeviceError);
      binding.diagnostics.external_system_errors = system_errors(kExternalSystemError);
      binding.diagnostics.external_device_error = device_error(kExternalDeviceError);
      binding.diagnostics.force_composition_system_errors = system_errors(kCompositionSystemError);
      binding.diagnostics.force_composition_plan_error = device_error(kCompositionPlanError);

      /* The setup owner uploads host-generated geometry values, but the force
       * reverse passes compare their compact caches against CUDA arithmetic.
       * Build the geometry and CN-dependent AES2 caches once on the setup
       * stream; the numerical refresh transaction reuses these fixed public
       * addresses after geometry changes. */
      cuda_status = reset_gfn2_geometry_device_errors_cuda(
          batch, binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial geometry diagnostic reset", cuda_status);
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      cuda_status = update_gfn2_geometry_cache_cuda(
          binding.plan.coordination_batch, positions, candidate.host.geometry_generation,
          binding.plan.coordination_cache, candidate.workspace_seed.geometry_workspace,
          binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial geometry-cache construction", cuda_status);
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      cuda_status =
          reset_gfn2_aes2_device_errors_cuda(batch, binding.diagnostics.coordination_system_errors,
                                             binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial AES2 diagnostic reset", cuda_status);
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      cuda_status = update_gfn2_aes2_geometry_cache_cuda(
          candidate.plan_seed.aes2_batch, positions,
          binding.plan.coordination_cache.coordination_numbers, candidate.plan_seed.aes2_cache,
          candidate.workspace_seed.aes2_workspace, binding.diagnostics.coordination_system_errors,
          binding.diagnostics.coordination_device_error, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA initial AES2-cache construction", cuda_status);
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
    }

    /* Exercise the published descriptors, not merely their host validation.
     * All-bits-one is a NaN for the CUDA-supported IEEE-754 double format, so
     * a finite download proves that terminal publication actually ran. */
    std::size_t output_bytes = 0u;
    if (!checked_bytes(batch, sizeof(double), output_bytes)) {
      error = "energy smoke output extent overflows size_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = cudaMemsetAsync(binding.results.energy.total_energy, 0xff, output_bytes, stream);
    if (cuda_status == cudaSuccess && force_mode) {
      if (!checked_bytes(binding.results.forces.qm_force_elements, sizeof(double), output_bytes)) {
        error = "QM-force smoke output extent overflows size_t";
        return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
      }
      cuda_status = cudaMemsetAsync(binding.results.forces.qm_forces, 0xff, output_bytes, stream);
      if (cuda_status == cudaSuccess && binding.results.forces.point_force_elements != 0) {
        if (!checked_bytes(binding.results.forces.point_force_elements, sizeof(double),
                           output_bytes)) {
          error = "point-force smoke output extent overflows size_t";
          return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
        }
        cuda_status =
            cudaMemsetAsync(binding.results.forces.point_forces, 0xff, output_bytes, stream);
      }
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(candidate.state_seed.scc.free_energies, 0,
                                    static_cast<std::size_t>(batch) * sizeof(double), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          cudaMemsetAsync(candidate.state_seed.scc.system_statuses, 0,
                          static_cast<std::size_t>(batch) * sizeof(vibeqc_xtb_status_t), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemsetAsync(candidate.state_seed.scc.converged, 1,
                                    static_cast<std::size_t>(batch) * sizeof(std::uint8_t), stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA energy/force smoke initialization", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    if (binding.stationary_projection.enabled == 1u) {
      cuda_status = project_gfn2_stationary_force_state_cuda(binding.stationary_projection, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA stationary force projection smoke", cuda_status);
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
    }
    /* The binding is constructed before the first device numerical refresh, so
     * neither the general committed pair list nor the D4 committed pair-list/CN
     * bundle is eligible yet. Run the validation smoke on a plan copy with those
     * refresh-owned consumers disabled. The dense coordination VJP and non-D4
     * leaves still exercise the complete publication chain; the production plan
     * retains every D4 component and exercises it after the first real refresh
     * has atomically published the pair lists, CN, generation, and eligibility. */
    Gfn2EnergyForceExecutionDevicePlan validation_plan = binding.plan;
    validation_plan.pairlist_committed.plan_token = 0u;
    validation_plan.pairlist_batch.plan_token = 0u;
    constexpr std::uint32_t kD4SccPotential =
        static_cast<std::uint32_t>(Gfn2SccPotentialComponent::kD4TwoBody);
    constexpr std::uint32_t kD4SccEnergy =
        static_cast<std::uint32_t>(Gfn2SccClassicalEnergyComponent::kD4TwoBody);
    constexpr std::uint32_t kD4AtmEnergy =
        static_cast<std::uint32_t>(Gfn2TotalEnergyComponent::kD4Atm);
    constexpr std::uint32_t kD4ClassicalForces =
        static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4TwoBody) |
        static_cast<std::uint32_t>(Gfn2ClassicalForceComponent::kD4ATM);
    validation_plan.scc_potential_components &= ~kD4SccPotential;
    validation_plan.scc_energy_components &= ~kD4SccEnergy;
    validation_plan.post_scc_potential_plan.enabled_components &= ~kD4SccPotential;
    validation_plan.total_energy_batch.enabled_components &= ~kD4AtmEnergy;
    validation_plan.classical_plan.enabled_components &= ~kD4ClassicalForces;
    Gfn2EnergyForceExecutionDeviceInput validation_input = binding.input;
    validation_input.total_energy.d4_atm = nullptr;
    validation_input.total_energy.d4_atm_elements = 0;
    cuda_status = execute_gfn2_energy_force_cuda(validation_plan, validation_input, binding.results,
                                                 binding.intermediates, binding.workspace,
                                                 binding.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA energy/force composed smoke", cuda_status);
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    /* The smoke borrows the initialized SCC storage only long enough to drive
     * the terminal chain. Restore the immutable device checkpoint so the
     * published runtime still starts at iteration zero. */
    const auto restore_diagnostic = candidate.initializer.upload_async(
        candidate.iteration_arena.get(), candidate.iteration_arena.bytes(), candidate.ready,
        stream);
    if (!restore_diagnostic.success()) {
      error = setup_error_message("CUDA SCC fresh-state restoration", restore_diagnostic.status,
                                  static_cast<std::uint32_t>(restore_diagnostic.error),
                                  static_cast<std::uint32_t>(restore_diagnostic.field),
                                  restore_diagnostic.index);
      return restore_diagnostic.status;
    }
    candidate.submitted = true;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t build_inference_bindings(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t points = candidate.host.external.total_point_charges;
    const std::uint64_t token = candidate.host.plan_token;
    const std::uint32_t requested = candidate.host.key.flags;
    const bool d4_enabled = candidate.host.d4_enabled;
    const bool energy_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_DIPOLE_MOMENTS)) != 0u;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t d4_weight_elements = 0;
    if (requested == 0u || !checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(atoms, kGfn2D4MaximumReferences, d4_weight_elements)) {
      error = "inference result extent or requested-property mask is invalid";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    struct Offsets {
      std::size_t atomic_numbers = 0u;
      std::size_t terminal_repulsion_candidate = 0u;
      std::size_t terminal_d4_candidate = 0u;
      std::size_t terminal_d4_weights = 0u;
      std::size_t terminal_d4_weight_cn = 0u;
      std::size_t terminal_d4_weight_charge = 0u;
      std::size_t terminal_d4_atom_scratch = 0u;
      std::size_t terminal_d4_coordination_adjoints = 0u;
      std::size_t terminal_d4_batch_scratch = 0u;
      std::size_t terminal_d4_gradient_scratch = 0u;
      std::size_t terminal_epoch_snapshot = 0u;
      std::size_t terminal_repulsion_error = 0u;
      std::size_t terminal_d4_system_errors = 0u;
      std::size_t terminal_d4_device_error = 0u;
      std::size_t terminal_system_errors = 0u;
      std::size_t terminal_plan_error = 0u;

      std::size_t result_energies = 0u;
      std::size_t result_qm_forces = 0u;
      std::size_t result_atomic_charges = 0u;
      std::size_t result_point_forces = 0u;
      std::size_t result_dipole_moments = 0u;
      std::size_t result_iterations = 0u;
      std::size_t result_converged = 0u;
      std::size_t result_system_statuses = 0u;
      std::size_t publication_epoch_snapshot = 0u;
      std::size_t publication_system_errors = 0u;
      std::size_t publication_plan_error = 0u;
      std::size_t warm_checkpoint_generations = 0u;
      std::size_t warm_checkpoint_batch_ready = 0u;
      std::size_t request_start_mode = 0u;
      std::size_t admitted_warm_checkpoint_generations = 0u;
      std::size_t warm_checkpoint_field_attached = 0u;
      std::size_t warm_checkpoint_field_vectors = 0u;
    } offset;

    ArenaLayout layout;
    offset.atomic_numbers =
        layout.append<std::int32_t>(candidate.atomic_numbers == nullptr ? atoms : 0);
    offset.terminal_repulsion_candidate = layout.append<double>(batch);
    offset.terminal_d4_candidate = layout.append<double>(d4_enabled ? batch : 0);
    offset.terminal_d4_weights = layout.append<double>(d4_enabled ? d4_weight_elements : 0);
    offset.terminal_d4_weight_cn = layout.append<double>(d4_enabled ? d4_weight_elements : 0);
    offset.terminal_d4_weight_charge = layout.append<double>(d4_enabled ? d4_weight_elements : 0);
    offset.terminal_d4_atom_scratch = layout.append<double>(d4_enabled ? atoms : 0);
    offset.terminal_d4_coordination_adjoints = layout.append<double>(d4_enabled ? atoms : 0);
    offset.terminal_d4_batch_scratch = layout.append<double>(d4_enabled ? batch : 0);
    offset.terminal_d4_gradient_scratch = layout.append<double>(d4_enabled ? coordinates : 0);
    offset.terminal_epoch_snapshot = layout.append<std::uint64_t>(1);
    offset.terminal_repulsion_error = layout.append<std::uint32_t>(1);
    offset.terminal_d4_system_errors = layout.append<std::uint32_t>(d4_enabled ? batch : 0);
    offset.terminal_d4_device_error = layout.append<std::uint32_t>(d4_enabled ? 1 : 0);
    offset.terminal_system_errors = layout.append<std::uint32_t>(batch);
    offset.terminal_plan_error = layout.append<std::uint32_t>(1);

    offset.result_energies = layout.append<double>(energy_requested ? batch : 0);
    offset.result_qm_forces = layout.append<double>(force_requested ? coordinates : 0);
    offset.result_atomic_charges =
        layout.append<double>((charges_requested || dipoles_requested) ? atoms : 0);
    offset.result_point_forces =
        layout.append<double>(point_forces_requested ? point_coordinates : 0);
    offset.result_dipole_moments = layout.append<double>(dipoles_requested ? 3 * batch : 0);
    offset.result_iterations = layout.append<std::int32_t>(batch);
    offset.result_converged = layout.append<std::uint8_t>(batch);
    offset.result_system_statuses = layout.append<vibeqc_xtb_status_t>(batch);
    offset.publication_epoch_snapshot = layout.append<std::uint64_t>(1);
    offset.publication_system_errors = layout.append<std::uint32_t>(batch);
    offset.publication_plan_error = layout.append<std::uint32_t>(1);
    offset.warm_checkpoint_generations = layout.append<std::uint64_t>(batch);
    offset.warm_checkpoint_batch_ready = layout.append<std::uint32_t>(1);
    offset.request_start_mode = layout.append<std::uint32_t>(1);
    offset.admitted_warm_checkpoint_generations = layout.append<std::uint64_t>(batch);
    offset.warm_checkpoint_field_attached = layout.append<std::uint8_t>(batch);
    offset.warm_checkpoint_field_vectors = layout.append<double>(3 * batch);
    if (!layout.valid()) {
      error = "inference arena layout overflows size_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }

    cudaError_t cuda_status = candidate.inference_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA inference arena allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.inference_arena.get();
    cuda_status = cudaMemsetAsync(arena, 0, candidate.inference_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA inference arena initialization", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const std::int32_t* terminal_atomic_numbers = candidate.atomic_numbers;
    if (terminal_atomic_numbers == nullptr) {
      auto* const destination = arena_pointer<std::int32_t>(arena, offset.atomic_numbers);
      cuda_status = cudaMemcpyAsync(destination, candidate.host.key.atomic_numbers.data(),
                                    static_cast<std::size_t>(atoms) * sizeof(std::int32_t),
                                    cudaMemcpyHostToDevice, stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA terminal atomic-number upload", cuda_status);
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
      terminal_atomic_numbers = destination;
    }
    candidate.submitted = true;

    auto& inference = candidate.inference;
    inference = {};
    inference.epoch_consumer = {
        candidate.numerical.preprocessing.geometry_epoch,
        candidate.numerical.device.committed_generations,
        candidate.numerical.device.eligible,
        batch,
        token,
    };

    const Gfn2RepulsionDeviceBatch repulsion{
        batch,
        atoms,
        candidate.device_topology.atom_offsets,
        terminal_atomic_numbers,
        candidate.numerical.device.committed_positions,
    };
    Gfn2D4DeviceBatch terminal_d4 = candidate.plan_seed.d4_batch;
    if (d4_enabled) terminal_d4.atomic_numbers = repulsion.atomic_numbers;
    inference.terminal_plan = {
        kGfn2TerminalClassicalEnergyAbiVersion,
        d4_enabled ? kGfn2TerminalClassicalEnergyAllComponents : 0u,
        token,
        repulsion,
        d4_enabled ? terminal_d4 : Gfn2D4DeviceBatch{},
        d4_enabled ? candidate.plan_seed.d4_parameters : Gfn2D4DeviceParameters{},
        d4_enabled ? candidate.numerical.d4_cache : Gfn2D4PairListDeviceCache{},
        candidate.numerical.preprocessing.geometry_epoch,
        candidate.numerical.device.committed_generations,
        batch,
    };
    inference.terminal_activity = {
        candidate.numerical.device.eligible, batch, token, {nullptr, 0, 0}};
    inference.terminal_results = {
        const_cast<double*>(candidate.energy_force.input.total_energy.repulsion),
        batch,
        d4_enabled ? const_cast<double*>(candidate.energy_force.input.total_energy.d4_atm)
                   : nullptr,
        d4_enabled ? batch : 0,
        token,
    };
    inference.terminal_workspace.repulsion_candidate =
        arena_pointer<double>(arena, offset.terminal_repulsion_candidate);
    inference.terminal_workspace.repulsion_elements = batch;
    inference.terminal_workspace.d4_atm_candidate =
        arena_pointer_if<double>(arena, offset.terminal_d4_candidate, d4_enabled ? batch : 0);
    inference.terminal_workspace.d4_atm_elements = d4_enabled ? batch : 0;
    if (d4_enabled) {
      inference.terminal_workspace.d4 = {
          arena_pointer<double>(arena, offset.terminal_d4_weights),
          arena_pointer<double>(arena, offset.terminal_d4_weight_cn),
          arena_pointer<double>(arena, offset.terminal_d4_weight_charge),
          d4_weight_elements,
          arena_pointer<double>(arena, offset.terminal_d4_atom_scratch),
          arena_pointer<double>(arena, offset.terminal_d4_coordination_adjoints),
          atoms,
          arena_pointer<double>(arena, offset.terminal_d4_batch_scratch),
          batch,
          arena_pointer<double>(arena, offset.terminal_d4_gradient_scratch),
          coordinates,
          arena_pointer<std::uint32_t>(arena, offset.terminal_d4_system_errors),
          batch,
      };
    }
    inference.terminal_workspace.epoch_snapshot =
        arena_pointer<std::uint64_t>(arena, offset.terminal_epoch_snapshot);
    inference.terminal_workspace.epoch_snapshot_elements = 1;
    inference.terminal_workspace.plan_token = token;
    inference.terminal_diagnostics = {
        arena_pointer<std::uint32_t>(arena, offset.terminal_repulsion_error),
        arena_pointer_if<std::uint32_t>(arena, offset.terminal_d4_system_errors,
                                        d4_enabled ? batch : 0),
        d4_enabled ? batch : 0,
        arena_pointer_if<std::uint32_t>(arena, offset.terminal_d4_device_error, d4_enabled ? 1 : 0),
        arena_pointer<std::uint32_t>(arena, offset.terminal_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.terminal_plan_error),
        1,
        token,
    };

    const auto input_if_requested = [](const double* pointer, bool enabled) {
      return enabled ? pointer : nullptr;
    };
    inference.publication_plan = {
        kGfn2InferencePublicationAbiVersion,
        requested,
        token,
        static_cast<std::uint64_t>(candidate.host.key.maximum_iterations),
        batch,
        atoms,
        points,
        candidate.device_topology.atom_offsets,
        points == 0 ? nullptr
                    : candidate.plan_seed.explicit_point_charge_batch.point_charge_offsets,
        candidate.numerical.preprocessing.geometry_epoch,
        candidate.numerical.device.committed_generations,
        batch,
    };
    inference.publication_input = {
        {nullptr, 0, 0},
        candidate.numerical.device.eligible,
        batch,
        candidate.state_seed.scc.iterations,
        candidate.state_seed.scc.converged,
        candidate.state_seed.scc.system_statuses,
        batch,
        input_if_requested(candidate.energy_force.results.energy.total_energy, energy_requested),
        energy_requested ? batch : 0,
        input_if_requested(candidate.energy_force.results.forces.qm_forces, force_requested),
        force_requested ? coordinates : 0,
        input_if_requested(candidate.energy_force.stationary_projection.enabled == 1u
                               ? candidate.energy_force.stationary_projection.atomic_charges
                               : candidate.workspace_seed.physical_topology.atomic_charges,
                           charges_requested || dipoles_requested),
        (charges_requested || dipoles_requested) ? atoms : 0,
        input_if_requested(candidate.energy_force.results.forces.point_forces,
                           point_forces_requested),
        point_forces_requested ? point_coordinates : 0,
        inference.terminal_diagnostics.system_errors,
        batch,
        inference.terminal_diagnostics.plan_error,
        candidate.energy_force.diagnostics.execution_system_errors,
        batch,
        candidate.energy_force.workspace.plan_failure,
        token,
        input_if_requested(candidate.numerical.device.committed_positions, dipoles_requested),
        dipoles_requested ? coordinates : 0,
        input_if_requested(candidate.energy_force.stationary_projection.atomic_dipoles,
                           dipoles_requested),
        dipoles_requested ? coordinates : 0,
    };
    inference.publication_results = {
        arena_pointer_if<double>(arena, offset.result_energies, energy_requested ? batch : 0),
        energy_requested ? batch : 0,
        arena_pointer_if<double>(arena, offset.result_qm_forces, force_requested ? coordinates : 0),
        force_requested ? coordinates : 0,
        arena_pointer_if<double>(arena, offset.result_atomic_charges,
                                 charges_requested ? atoms : 0),
        charges_requested ? atoms : 0,
        arena_pointer_if<double>(arena, offset.result_point_forces,
                                 point_forces_requested ? point_coordinates : 0),
        point_forces_requested ? point_coordinates : 0,
        arena_pointer<std::int32_t>(arena, offset.result_iterations),
        arena_pointer<std::uint8_t>(arena, offset.result_converged),
        arena_pointer<vibeqc_xtb_status_t>(arena, offset.result_system_statuses),
        batch,
        token,
        arena_pointer_if<double>(arena, offset.result_dipole_moments,
                                 dipoles_requested ? 3 * batch : 0),
        dipoles_requested ? 3 * batch : 0,
    };
    inference.publication_workspace = {
        arena_pointer<std::uint64_t>(arena, offset.publication_epoch_snapshot), 1, token};
    inference.publication_diagnostics = {
        arena_pointer<std::uint32_t>(arena, offset.publication_system_errors),
        batch,
        arena_pointer<std::uint32_t>(arena, offset.publication_plan_error),
        1,
        token,
    };
    inference.warm_checkpoint_generations =
        arena_pointer<std::uint64_t>(arena, offset.warm_checkpoint_generations);
    inference.warm_checkpoint_batch_ready =
        arena_pointer<std::uint32_t>(arena, offset.warm_checkpoint_batch_ready);
    inference.request_start_mode = arena_pointer<std::uint32_t>(arena, offset.request_start_mode);
    inference.admitted_warm_checkpoint_generations =
        arena_pointer<std::uint64_t>(arena, offset.admitted_warm_checkpoint_generations);
    inference.warm_checkpoint_field_attached =
        arena_pointer<std::uint8_t>(arena, offset.warm_checkpoint_field_attached);
    inference.warm_checkpoint_field_vectors =
        arena_pointer<double>(arena, offset.warm_checkpoint_field_vectors);
    inference.ready = true;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t build_public_result_state(Prepared& candidate, std::string& error) {
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t atoms = candidate.host.basis.total_atoms;
    const std::int64_t points = candidate.host.external.total_point_charges;
    const std::uint32_t requested = candidate.host.key.flags;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t lattice_elements = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates) ||
        !checked_elements(batch, 9, lattice_elements)) {
      error = "public CUDA result staging extent overflows int64_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    const bool energy_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (requested & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_DIPOLE_MOMENTS)) != 0u;

    struct HostOffsets {
      std::size_t energies = 0u;
      std::size_t qm_forces = 0u;
      std::size_t atomic_charges = 0u;
      std::size_t point_forces = 0u;
      std::size_t dipole_moments = 0u;
      std::size_t iterations = 0u;
      std::size_t converged = 0u;
      std::size_t system_statuses = 0u;
      std::size_t control = 0u;
      std::size_t warm_checkpoint_ready = 0u;
      std::size_t lattice_cells = 0u;
      std::size_t periodic_axes = 0u;
    } host_offset;
    struct DeviceOffsets {
      std::size_t energies = 0u;
      std::size_t qm_forces = 0u;
      std::size_t atomic_charges = 0u;
      std::size_t point_forces = 0u;
      std::size_t dipole_moments = 0u;
      std::size_t iterations = 0u;
      std::size_t converged = 0u;
      std::size_t system_statuses = 0u;
      std::size_t control = 0u;
      std::size_t request_topology_error = 0u;
      std::size_t expected_atom_offsets = 0u;
      std::size_t expected_atomic_numbers = 0u;
      std::size_t expected_molecular_charges = 0u;
      std::size_t expected_unpaired_electrons = 0u;
      std::size_t expected_spin_channels = 0u;
      std::size_t expected_point_offsets = 0u;
      std::size_t expected_response_offsets = 0u;
      std::size_t lattice_cells = 0u;
      std::size_t periodic_axes = 0u;
    } device_offset;
    ArenaLayout host_layout;
    host_offset.energies = host_layout.append<double>(energy_requested ? batch : 0);
    host_offset.qm_forces = host_layout.append<double>(force_requested ? coordinates : 0);
    host_offset.atomic_charges = host_layout.append<double>(charges_requested ? atoms : 0);
    host_offset.point_forces =
        host_layout.append<double>(point_forces_requested ? point_coordinates : 0);
    host_offset.dipole_moments = host_layout.append<double>(dipoles_requested ? 3 * batch : 0);
    host_offset.iterations = host_layout.append<std::int32_t>(batch);
    host_offset.converged = host_layout.append<std::uint8_t>(batch);
    host_offset.system_statuses = host_layout.append<vibeqc_xtb_status_t>(batch);
    host_offset.control = host_layout.append<Gfn2PublicResultBridgeControl>(1);
    host_offset.warm_checkpoint_ready = host_layout.append<std::uint32_t>(1);
    host_offset.lattice_cells = host_layout.append<double>(lattice_elements);
    host_offset.periodic_axes = host_layout.append<std::int32_t>(batch);

    ArenaLayout device_layout;
    device_offset.energies = device_layout.append<double>(energy_requested ? batch : 0);
    device_offset.qm_forces = device_layout.append<double>(force_requested ? coordinates : 0);
    device_offset.atomic_charges = device_layout.append<double>(charges_requested ? atoms : 0);
    device_offset.point_forces =
        device_layout.append<double>(point_forces_requested ? point_coordinates : 0);
    device_offset.dipole_moments = device_layout.append<double>(dipoles_requested ? 3 * batch : 0);
    device_offset.iterations = device_layout.append<std::int32_t>(batch);
    device_offset.converged = device_layout.append<std::uint8_t>(batch);
    device_offset.system_statuses = device_layout.append<vibeqc_xtb_status_t>(batch);
    device_offset.control = device_layout.append<Gfn2PublicResultBridgeControl>(1);
    device_offset.request_topology_error = device_layout.append<std::uint32_t>(1);
    device_offset.expected_atom_offsets = device_layout.append<std::int64_t>(batch + 1);
    device_offset.expected_atomic_numbers = device_layout.append<std::int32_t>(atoms);
    device_offset.expected_molecular_charges = device_layout.append<double>(batch);
    device_offset.expected_unpaired_electrons = device_layout.append<std::int32_t>(batch);
    device_offset.expected_spin_channels = device_layout.append<std::int32_t>(batch);
    device_offset.expected_point_offsets = device_layout.append<std::int64_t>(batch + 1);
    device_offset.expected_response_offsets = device_layout.append<std::int64_t>(batch + 1);
    device_offset.lattice_cells = device_layout.append<double>(lattice_elements);
    device_offset.periodic_axes = device_layout.append<std::int32_t>(batch);
    if (!host_layout.valid() || !device_layout.valid()) {
      error = "public CUDA result staging layout overflows size_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.public_result_host_arena.allocate(host_layout.bytes());
    if (cuda_status == cudaSuccess) {
      cuda_status = candidate.public_result_device_arena.allocate(device_layout.bytes());
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = candidate.public_result_completion_event.create(cudaEventDisableTiming);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA public result staging allocation", cuda_status);
      return cuda_status == cudaErrorMemoryAllocation ? VIBEQC_XTB_STATUS_ALLOCATION_FAILED
                                                      : VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    cuda_status = cudaMemsetAsync(candidate.public_result_device_arena.get(), 0,
                                  candidate.public_result_device_arena.bytes(), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA public result control initialization", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = true;

    void* const host_arena = candidate.public_result_host_arena.get();
    auto& state = candidate.public_result;
    state = {};
    state.energies =
        arena_pointer_if<double>(host_arena, host_offset.energies, energy_requested ? batch : 0);
    state.qm_forces = arena_pointer_if<double>(host_arena, host_offset.qm_forces,
                                               force_requested ? coordinates : 0);
    state.atomic_charges = arena_pointer_if<double>(host_arena, host_offset.atomic_charges,
                                                    charges_requested ? atoms : 0);
    state.point_forces = arena_pointer_if<double>(host_arena, host_offset.point_forces,
                                                  point_forces_requested ? point_coordinates : 0);
    state.dipole_moments = arena_pointer_if<double>(host_arena, host_offset.dipole_moments,
                                                    dipoles_requested ? 3 * batch : 0);
    state.iterations = arena_pointer<std::int32_t>(host_arena, host_offset.iterations);
    state.converged = arena_pointer<std::uint8_t>(host_arena, host_offset.converged);
    state.system_statuses =
        arena_pointer<vibeqc_xtb_status_t>(host_arena, host_offset.system_statuses);
    state.host_control =
        arena_pointer<Gfn2PublicResultBridgeControl>(host_arena, host_offset.control);
    state.warm_checkpoint_ready =
        arena_pointer<std::uint32_t>(host_arena, host_offset.warm_checkpoint_ready);
    void* const device_arena = candidate.public_result_device_arena.get();
    state.request_topology_error =
        arena_pointer<std::uint32_t>(device_arena, device_offset.request_topology_error);
    state.expected_atom_offsets =
        arena_pointer<std::int64_t>(device_arena, device_offset.expected_atom_offsets);
    state.expected_atomic_numbers =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_atomic_numbers);
    state.expected_molecular_charges =
        arena_pointer<double>(device_arena, device_offset.expected_molecular_charges);
    state.expected_unpaired_electrons =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_unpaired_electrons);
    state.expected_spin_channels =
        arena_pointer<std::int32_t>(device_arena, device_offset.expected_spin_channels);
    state.expected_point_offsets =
        arena_pointer<std::int64_t>(device_arena, device_offset.expected_point_offsets);
    state.expected_response_offsets =
        arena_pointer<std::int64_t>(device_arena, device_offset.expected_response_offsets);
    state.staged_lattice_cells = arena_pointer<double>(device_arena, device_offset.lattice_cells);
    state.staged_periodic_axes =
        arena_pointer<std::int32_t>(device_arena, device_offset.periodic_axes);
    state.host_lattice_cells = arena_pointer<double>(host_arena, host_offset.lattice_cells);
    state.host_periodic_axes = arena_pointer<std::int32_t>(host_arena, host_offset.periodic_axes);
    state.device_staging = {
        arena_pointer_if<double>(device_arena, device_offset.energies,
                                 energy_requested ? batch : 0),
        energy_requested ? batch : 0,
        arena_pointer_if<double>(device_arena, device_offset.qm_forces,
                                 force_requested ? coordinates : 0),
        force_requested ? coordinates : 0,
        arena_pointer_if<double>(device_arena, device_offset.atomic_charges,
                                 charges_requested ? atoms : 0),
        charges_requested ? atoms : 0,
        arena_pointer_if<double>(device_arena, device_offset.point_forces,
                                 point_forces_requested ? point_coordinates : 0),
        point_forces_requested ? point_coordinates : 0,
        arena_pointer<std::int32_t>(device_arena, device_offset.iterations),
        arena_pointer<std::uint8_t>(device_arena, device_offset.converged),
        arena_pointer<vibeqc_xtb_status_t>(device_arena, device_offset.system_statuses),
        batch,
        candidate.host.plan_token,
        arena_pointer_if<double>(device_arena, device_offset.dipole_moments,
                                 dipoles_requested ? 3 * batch : 0),
        dipoles_requested ? 3 * batch : 0,
    };
    state.diagnostics = {
        arena_pointer<Gfn2PublicResultBridgeControl>(device_arena, device_offset.control),
        1,
        candidate.host.plan_token,
    };
    const auto upload = [&](std::size_t offset, const auto& values) {
      if (cuda_status != cudaSuccess || values.empty()) return;
      cuda_status =
          cudaMemcpyAsync(static_cast<std::byte*>(device_arena) + offset, values.data(),
                          values.size() * sizeof(values[0]), cudaMemcpyHostToDevice, stream);
    };
    upload(device_offset.expected_atom_offsets, candidate.host.key.atom_offsets);
    upload(device_offset.expected_atomic_numbers, candidate.host.key.atomic_numbers);
    upload(device_offset.expected_molecular_charges, candidate.host.key.molecular_charges);
    upload(device_offset.expected_unpaired_electrons, candidate.host.key.unpaired_electrons);
    upload(device_offset.expected_spin_channels, candidate.host.key.spin_channels);
    upload(device_offset.expected_point_offsets, candidate.host.key.point_offsets);
    upload(device_offset.expected_response_offsets, candidate.host.key.response_offsets);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-topology comparison upload", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    state.ready = true;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t validate_candidate_setup(Prepared& candidate, std::string& error) {
    const auto& binding = candidate.energy_force;
    const std::int64_t batch = candidate.host.basis.batch_size;
    const std::int64_t setup_systems = candidate.eigensolver_binding.setup_system_error_elements;
    const std::int64_t factor_statuses = candidate.eigensolver_binding.cache.status_elements;
    const std::int64_t qm_force_elements =
        binding.plan.compute_forces == 1u ? binding.results.forces.qm_force_elements : 0;
    const std::int64_t point_force_elements =
        binding.plan.compute_forces == 1u ? binding.results.forces.point_force_elements : 0;
    struct Offsets {
      std::size_t setup_device_error = 0u;
      std::size_t setup_system_errors = 0u;
      std::size_t factor_statuses = 0u;
      std::size_t execution_system_errors = 0u;
      std::size_t execution_device_error = 0u;
      std::size_t plan_failure = 0u;
      std::size_t energies = 0u;
      std::size_t qm_forces = 0u;
      std::size_t point_forces = 0u;
      std::size_t converged = 0u;
    } offset;
    ArenaLayout layout;
    offset.setup_device_error = layout.append<std::uint32_t>(1);
    offset.setup_system_errors = layout.append<std::uint32_t>(setup_systems);
    offset.factor_statuses = layout.append<std::uint32_t>(factor_statuses);
    offset.execution_system_errors = layout.append<std::uint32_t>(batch);
    offset.execution_device_error = layout.append<std::uint32_t>(1);
    offset.plan_failure = layout.append<std::uint32_t>(1);
    offset.energies = layout.append<double>(batch);
    offset.qm_forces = layout.append<double>(qm_force_elements);
    offset.point_forces = layout.append<double>(point_force_elements);
    offset.converged = layout.append<std::uint8_t>(batch);
    if (!layout.valid()) {
      error = "CUDA candidate validation staging layout overflows size_t";
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cudaError_t cuda_status = candidate.candidate_validation_arena.allocate(layout.bytes());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA candidate validation staging allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    void* const arena = candidate.candidate_validation_arena.get();
    auto* const setup_device_error = arena_pointer<std::uint32_t>(arena, offset.setup_device_error);
    auto* const setup_system_errors =
        arena_pointer_if<std::uint32_t>(arena, offset.setup_system_errors, setup_systems);
    auto* const factor_status_values =
        arena_pointer_if<std::uint32_t>(arena, offset.factor_statuses, factor_statuses);
    auto* const execution_system_errors =
        arena_pointer<std::uint32_t>(arena, offset.execution_system_errors);
    auto* const execution_device_error =
        arena_pointer<std::uint32_t>(arena, offset.execution_device_error);
    auto* const plan_failure = arena_pointer<std::uint32_t>(arena, offset.plan_failure);
    auto* const energies = arena_pointer<double>(arena, offset.energies);
    auto* const qm_forces = arena_pointer_if<double>(arena, offset.qm_forces, qm_force_elements);
    auto* const point_forces =
        arena_pointer_if<double>(arena, offset.point_forces, point_force_elements);
    auto* const converged = arena_pointer<std::uint8_t>(arena, offset.converged);
    const auto enqueue = [&](void* destination, const void* source, std::int64_t elements,
                             std::size_t element_size) {
      return elements == 0 ? cudaSuccess
                           : cudaMemcpyAsync(destination, source,
                                             static_cast<std::size_t>(elements) * element_size,
                                             cudaMemcpyDeviceToHost, stream);
    };
    cuda_status = enqueue(setup_device_error, candidate.eigensolver_binding.setup_device_error, 1,
                          sizeof(std::uint32_t));
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(setup_system_errors, candidate.eigensolver_binding.setup_system_errors,
                            setup_systems, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          enqueue(factor_status_values, candidate.eigensolver_binding.cache.factor_statuses,
                  factor_statuses, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(execution_system_errors, binding.diagnostics.execution_system_errors,
                            batch, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(execution_device_error, binding.diagnostics.execution_device_error, 1,
                            sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(plan_failure, binding.workspace.plan_failure, 1, sizeof(std::uint32_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(energies, binding.results.energy.total_energy, batch, sizeof(double));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          enqueue(qm_forces, binding.results.forces.qm_forces, qm_force_elements, sizeof(double));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = enqueue(point_forces, binding.results.forces.point_forces, point_force_elements,
                            sizeof(double));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status =
          enqueue(converged, candidate.state_seed.scc.converged, batch, sizeof(std::uint8_t));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaEventRecord(candidate.public_result_completion_event.get(), stream);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaEventSynchronize(candidate.public_result_completion_event.get());
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA candidate validation completion", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    candidate.submitted = false;

    const auto any_nonzero = [](const std::uint32_t* values, std::int64_t elements) {
      return elements != 0 && std::any_of(values, values + elements,
                                          [](std::uint32_t value) { return value != 0u; });
    };
    if (*setup_device_error != 0u || any_nonzero(setup_system_errors, setup_systems) ||
        any_nonzero(factor_status_values, factor_statuses)) {
      error = "CUDA eigensolver setup reported an asynchronous factorization failure";
      return VIBEQC_XTB_STATUS_EIGENSOLVER_FAILED;
    }

    const auto first_system_error =
        std::find_if(execution_system_errors, execution_system_errors + batch,
                     [](std::uint32_t value) { return value != 0u; });
    if (*execution_device_error != 0u || *plan_failure != 0u ||
        first_system_error != execution_system_errors + batch) {
      std::ostringstream message;
      message << "CUDA energy/force smoke reported device_error=" << *execution_device_error
              << " plan_failure=" << *plan_failure;
      if (first_system_error != execution_system_errors + batch) {
        message << " system=" << std::distance(execution_system_errors, first_system_error)
                << " system_error=" << *first_system_error;
      }
      error = message.str();
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const auto all_finite = [](const double* values, std::int64_t elements) {
      return elements == 0 || std::all_of(values, values + elements,
                                          [](double value) { return std::isfinite(value); });
    };
    if (!all_finite(energies, batch) || !all_finite(qm_forces, qm_force_elements) ||
        !all_finite(point_forces, point_force_elements)) {
      error = "CUDA energy/force smoke did not publish every requested output";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    if (std::any_of(converged, converged + batch, [](std::uint8_t value) { return value != 0u; })) {
      error = "CUDA energy/force smoke failed to restore the fresh SCC state";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    candidate.energy_force_smoke_ready = true;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t build_candidate(TopologyKey&& key, std::vector<double>&& positions,
                                      std::vector<double>&& point_positions,
                                      std::vector<double>&& point_values,
                                      std::vector<double>&& point_gammas,
                                      std::vector<double>&& periodic_shifts,
                                      std::vector<double>&& periodic_response,
                                      std::unique_ptr<Prepared>& output, std::string& error) {
    auto candidate = std::make_unique<Prepared>(stream);
    const std::uint64_t fingerprint = key.fingerprint();
    std::uint64_t token = hash_mix(fingerprint ^ next_plan_token++ ^ 0x112112112ULL);
    if (token == 0u) token = next_plan_token++;
    vibeqc_xtb_status_t status = candidate->host.build(
        std::move(key), std::move(positions), std::move(point_positions), std::move(point_values),
        std::move(point_gammas), std::move(periodic_shifts), std::move(periodic_response), token,
        error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

    auto topology_diagnostic = Gfn2SccSetupTopology::create(
        candidate->host.basis, candidate->host.integrals, candidate->host.wavefunction_layout,
        token, candidate->topology_owner);
    if (!topology_diagnostic.success()) {
      error = setup_error_message("CUDA topology owner construction", topology_diagnostic.status,
                                  static_cast<std::uint32_t>(topology_diagnostic.error),
                                  static_cast<std::uint32_t>(topology_diagnostic.field),
                                  topology_diagnostic.index);
      return topology_diagnostic.status;
    }
    cudaError_t cuda_status = candidate->topology_arena.allocate(
        candidate->topology_owner.requirements().immutable_device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA topology arena allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    topology_diagnostic = candidate->topology_owner.bind_device_arena_and_upload_async(
        candidate->topology_arena.get(), candidate->topology_arena.bytes(),
        candidate->device_topology, candidate->device_wavefunction, stream);
    if (!topology_diagnostic.success()) {
      error = setup_error_message("CUDA topology upload", topology_diagnostic.status,
                                  static_cast<std::uint32_t>(topology_diagnostic.error),
                                  static_cast<std::uint32_t>(topology_diagnostic.field),
                                  topology_diagnostic.index);
      return topology_diagnostic.status;
    }
    candidate->submitted = true;

    auto input_diagnostic = Gfn2SccSetupInputs::create(candidate->host.input_sources(),
                                                       candidate->topology_owner.host_topology(),
                                                       token, candidate->inputs_owner);
    if (!input_diagnostic.success()) {
      error = setup_error_message("CUDA input owner construction", input_diagnostic.status,
                                  static_cast<std::uint32_t>(input_diagnostic.error),
                                  static_cast<std::uint32_t>(input_diagnostic.field),
                                  input_diagnostic.index);
      return input_diagnostic.status;
    }
    cuda_status =
        candidate->input_arena.allocate(candidate->inputs_owner.requirements().device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA input arena allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    input_diagnostic = candidate->inputs_owner.bind_device_arena_and_upload_async(
        candidate->device_topology, candidate->device_wavefunction, candidate->input_arena.get(),
        candidate->input_arena.bytes(), candidate->plan_seed, candidate->input_seed, stream);
    if (!input_diagnostic.success()) {
      error = setup_error_message("CUDA input upload", input_diagnostic.status,
                                  static_cast<std::uint32_t>(input_diagnostic.error),
                                  static_cast<std::uint32_t>(input_diagnostic.field),
                                  input_diagnostic.index);
      return input_diagnostic.status;
    }
    /* Uniform-field storage is part of every fixed CUDA plan so FRESH calls
     * can attach, change, or detach a field without changing the SCC arena.
     * Numerical refresh binds the address-stable values after that arena is
     * allocated; declaring the topology here lets the arena query reserve the
     * appended field energy/diagnostic storage. */
    candidate->plan_seed.electric_field_batch = {
        candidate->host.basis.batch_size,
        candidate->host.basis.total_atoms,
        candidate->host.basis.batch_size + 1,
        candidate->device_topology.atom_offsets,
        token,
    };

    Gfn2EigensolverOptions eigensolver_options = candidate->plan_seed.eigensolver_options;
    eigensolver_options.jacobi = solver_jacobi;
    auto eigensolver_diagnostic = Gfn2SccSetupEigensolver::create(
        candidate->topology_owner, candidate->host.overlap.data(),
        static_cast<std::int64_t>(candidate->host.overlap.size()),
        candidate->host.geometry_generation, token, solver, solver_parameters, blas,
        eigensolver_options, candidate->eigensolver_owner);
    if (!eigensolver_diagnostic.success()) {
      error = setup_error_message(
          "CUDA eigensolver owner construction", eigensolver_diagnostic.status,
          static_cast<std::uint32_t>(eigensolver_diagnostic.error),
          static_cast<std::uint32_t>(eigensolver_diagnostic.field), eigensolver_diagnostic.index);
      return eigensolver_diagnostic.status;
    }
    const auto& eigensolver_requirements = candidate->eigensolver_owner.requirements();
    const auto arena_diagnostic = query_gfn2_scc_iteration_arena_requirements_cuda(
        candidate->plan_seed, eigensolver_requirements.provider, candidate->iteration_requirements);
    if (!arena_diagnostic.success()) {
      error = "CUDA SCC iteration arena query rejected the composed production plan";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    cuda_status =
        candidate->iteration_arena.allocate(candidate->iteration_requirements.total_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA SCC iteration arena allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    cuda_status = candidate->provider_host_workspace.allocate(
        eigensolver_requirements.provider.solver_host_workspace_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA SCC provider workspace allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    const auto bind_diagnostic = bind_gfn2_scc_iteration_arena_cuda(
        candidate->plan_seed, eigensolver_requirements.provider, candidate->iteration_requirements,
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(),
        candidate->provider_host_workspace.get(), candidate->provider_host_workspace.bytes(),
        candidate->state_seed, candidate->workspace_seed, candidate->report_storage);
    if (!bind_diagnostic.success()) {
      error = "CUDA SCC iteration arena binding rejected its own sealed requirements";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    status = build_numerical_refresh_binding(*candidate, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    candidate->input_seed.electric_field = {
        candidate->numerical.device.committed_field_vectors,
        3 * candidate->host.basis.batch_size,
        candidate->numerical.device.committed_positions,
        3 * candidate->host.basis.total_atoms,
        token,
    };
    candidate->input_seed.electric_field_potentials = candidate->numerical.field_potential_view;
    candidate->plan_seed.classical_energy_batch.electric_field =
        candidate->plan_seed.electric_field_batch;
    /* Report projection rebuilds the composed input descriptors from the
     * seed while preserving runtime-owned numerical leaves. Keeping the field
     * diagnostics explicitly bound here makes the first captured iteration
     * and every replay use the arena's appended field slots. */
    candidate->input_seed.classical_energy.electric_field_multipoles = {
        candidate->workspace_seed.physical_topology.atomic_charges,
        candidate->workspace_seed.physical_topology.atom_elements,
        candidate->workspace_seed.physical_topology.atomic_dipoles,
        candidate->workspace_seed.physical_topology.dipole_elements,
        token,
    };
    candidate->input_seed.classical_energy.electric_field_potentials =
        candidate->numerical.field_potential_view;
    candidate->input_seed.free_energy.electric_field =
        candidate->workspace_seed.staged_classical_energy.electric_field;
    candidate->input_seed.free_energy.electric_field_elements =
        candidate->workspace_seed.staged_classical_energy.electric_field_elements;
    cuda_status =
        candidate->eigensolver_setup_arena.allocate(eigensolver_requirements.setup_device_bytes);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA eigensolver setup arena allocation", cuda_status);
      return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
    }
    eigensolver_diagnostic = candidate->eigensolver_owner.bind_and_factor_overlap_async(
        candidate->device_topology, candidate->plan_seed, candidate->iteration_requirements,
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(),
        candidate->workspace_seed, candidate->provider_host_workspace.get(),
        candidate->provider_host_workspace.bytes(), candidate->eigensolver_setup_arena.get(),
        candidate->eigensolver_setup_arena.bytes(), candidate->eigensolver_binding, stream);
    if (!eigensolver_diagnostic.success()) {
      error = setup_error_message(
          "CUDA overlap factorization submission", eigensolver_diagnostic.status,
          static_cast<std::uint32_t>(eigensolver_diagnostic.error),
          static_cast<std::uint32_t>(eigensolver_diagnostic.field), eigensolver_diagnostic.index);
      return eigensolver_diagnostic.status;
    }
    candidate->plan_seed.eigensolver_batch = candidate->eigensolver_binding.batch;
    candidate->plan_seed.eigensolver_provider = candidate->eigensolver_binding.provider;
    candidate->plan_seed.overlap_cache = candidate->eigensolver_binding.cache;
    candidate->plan_seed.eigensolver_options = candidate->eigensolver_binding.options;

    Gfn2SccIterationHostInitialization initialization = candidate->host.initialization();
    /* Mixed-spin fresh checkpoints must carry the same setup-sealed host
     * layout that produced the device plan; physical offsets alone cannot
     * reconstruct per-system spin-major extents. */
    initialization.wavefunction_layout = candidate->topology_owner.host_wavefunction_layout();
    auto initialization_diagnostic = Gfn2SccIterationInitializer::create(
        candidate->plan_seed, candidate->iteration_requirements, candidate->iteration_arena.get(),
        candidate->iteration_arena.bytes(), candidate->state_seed, candidate->workspace_seed,
        candidate->report_storage, initialization, candidate->initializer);
    if (!initialization_diagnostic.success()) {
      error =
          setup_error_message("CUDA SCC initializer construction", initialization_diagnostic.status,
                              static_cast<std::uint32_t>(initialization_diagnostic.error),
                              static_cast<std::uint32_t>(initialization_diagnostic.field),
                              initialization_diagnostic.index);
      return initialization_diagnostic.status;
    }
    initialization_diagnostic = candidate->initializer.upload_async(
        candidate->iteration_arena.get(), candidate->iteration_arena.bytes(), candidate->ready,
        stream);
    if (!initialization_diagnostic.success()) {
      error = setup_error_message("CUDA SCC initializer upload", initialization_diagnostic.status,
                                  static_cast<std::uint32_t>(initialization_diagnostic.error),
                                  static_cast<std::uint32_t>(initialization_diagnostic.field),
                                  initialization_diagnostic.index);
      return initialization_diagnostic.status;
    }
    const auto report_diagnostic = build_gfn2_scc_iteration_report_binding_cuda(
        candidate->report_storage, candidate->plan_seed, candidate->input_seed,
        candidate->state_seed, candidate->workspace_seed, candidate->scc_binding);
    if (report_diagnostic.error != Gfn2SccIterationBindingError::kSuccess) {
      error = "CUDA SCC report factory rejected the composed runtime binding";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    status = build_energy_force_bindings(*candidate, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = build_inference_bindings(*candidate, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    /* Numerical refresh is built first because inference consumes its epoch
     * descriptors. Close the reverse failure-invalidation and field-identity
     * edges only after the inference arena owns the stable checkpoint arrays. */
    candidate->numerical.device.warm_checkpoint_generations =
        candidate->inference.warm_checkpoint_generations;
    candidate->numerical.device.warm_checkpoint_field_attached =
        candidate->inference.warm_checkpoint_field_attached;
    candidate->numerical.device.warm_checkpoint_field_vectors =
        candidate->inference.warm_checkpoint_field_vectors;
    status = build_public_result_state(*candidate, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    candidate->numerical.device.request_error = candidate->public_result.request_topology_error;
    candidate->numerical.preprocessing.admission = {candidate->public_result.request_topology_error,
                                                    1, candidate->host.plan_token};
    candidate->scc_binding.input.admission = {candidate->public_result.request_topology_error, 1,
                                              candidate->host.plan_token};
    candidate->energy_force.input.admission = {candidate->public_result.request_topology_error, 1,
                                               candidate->host.plan_token};
    candidate->energy_force.stationary_projection.request_error =
        candidate->public_result.request_topology_error;
    candidate->inference.terminal_activity.admission = {
        candidate->public_result.request_topology_error, 1, candidate->host.plan_token};
    candidate->inference.publication_input.admission = {
        candidate->public_result.request_topology_error, 1, candidate->host.plan_token};
    const auto preprocessing_reseal =
        seal_gfn2_preprocessing_binding_cuda(candidate->numerical.preprocessing);
    if (!preprocessing_reseal.success()) {
      error = "CUDA runtime preprocessing admission binding is invalid";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    status = validate_prepared_admission_aliases(*candidate, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

    status = validate_candidate_setup(*candidate, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    const Gfn2SccLoopGraphBuildResult loop_graph =
        candidate->scc_loop.build(candidate->scc_binding, candidate->inference.epoch_consumer);
    if (!loop_graph.success()) {
      std::ostringstream message;
      message << "CUDA SCC conditional Graph setup rejected the production binding: status="
              << static_cast<std::uint32_t>(loop_graph.status)
              << " iteration_status=" << static_cast<std::uint32_t>(loop_graph.iteration.status)
              << " stage=" << static_cast<std::uint32_t>(loop_graph.iteration.stage)
              << " cuda=" << static_cast<int>(loop_graph.cuda_status);
      error = message.str();
      return loop_graph.iteration.status == Gfn2SccIterationLaunchStatus::kInvalidBinding
                 ? VIBEQC_XTB_STATUS_INVALID_ARGUMENT
                 : VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    /* The setup owner factors its deterministic topology-only seed at
     * generation 1 so setup and graph validation can exercise a usable
     * overlap cache.  The externally visible numerical runtime, however,
     * starts unpublished at epoch 0.  Invalidate only the seed provenance
     * before publishing the candidate so the first real epoch-1 refresh must
     * refactor the caller geometry instead of mistaking the seed factor for a
     * cache hit. */
    cuda_status = cudaMemsetAsync(
        candidate->eigensolver_binding.cache.geometry_generations, 0,
        static_cast<std::size_t>(candidate->host.basis.batch_size) * sizeof(std::uint64_t), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA topology-only overlap cache invalidation", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    candidate->submitted = true;

    output = std::move(candidate);
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t validate_public_request_pointers(const vibeqc_xtb_batch_t& batch,
                                                       const vibeqc_xtb_compute_options_t& options,
                                                       const vibeqc_xtb_batch_result_t* result,
                                                       std::string& error) {
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    std::int64_t cell_elements = 0;
    std::int64_t dipole_elements = 0;
    const bool lattice_active =
        batch.struct_size >= VIBEQC_XTB_BATCH_V4_SIZE &&
        (batch.cell_matrices.data != nullptr || batch.cell_matrices.size_bytes != 0u ||
         batch.periodic_axes.data != nullptr || batch.periodic_axes.size_bytes != 0u);
    const bool interaction_suffix_present = batch.struct_size >= VIBEQC_XTB_BATCH_V3_SIZE;
    const bool interaction_descriptors_active =
        interaction_suffix_present && (batch.interaction_descriptors.data != nullptr ||
                                       batch.interaction_descriptors.size_bytes != 0u);
    const bool interaction_payload_active =
        interaction_suffix_present &&
        (batch.interaction_payload.data != nullptr || batch.interaction_payload.size_bytes != 0u);
    if (!checked_elements(batch.total_atoms, 3, coordinates) ||
        !checked_elements(batch.total_point_charges, 3, point_coordinates) ||
        !checked_elements(batch.batch_size, 3, dipole_elements) ||
        (lattice_active && !checked_elements(batch.batch_size, 9, cell_elements))) {
      error = "CUDA public descriptor coordinate extent overflows int64_t";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    const bool point_topology_active = batch.total_point_charges != 0 ||
                                       batch.point_charge_offsets.data != nullptr ||
                                       batch.point_charge_offsets.size_bytes != 0u;
    const bool shift_active = batch.atomic_potential_shifts.data != nullptr ||
                              batch.atomic_potential_shifts.size_bytes != 0u;
    const bool response_active = batch.total_charge_response_elements != 0 ||
                                 batch.charge_response_offsets.data != nullptr ||
                                 batch.charge_response_offsets.size_bytes != 0u ||
                                 batch.charge_response_matrix.data != nullptr ||
                                 batch.charge_response_matrix.size_bytes != 0u;
    CudaValidatedConstBuffer validated_const{};
    CudaValidatedBuffer validated_output{};
    const auto validate_const = [&](const char* name, const vibeqc_xtb_const_buffer_t& buffer,
                                    std::int64_t elements, std::size_t element_size,
                                    std::size_t alignment) -> vibeqc_xtb_status_t {
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, element_size, bytes)) {
        error = std::string(name) + " extent overflows size_t";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      return validate_cuda_const_buffer(device_id, name, buffer, bytes, alignment,
                                        CudaManagedMemoryPolicy::kReject, validated_const, error);
    };
    const auto validate_output = [&](const char* name, const vibeqc_xtb_buffer_t& buffer,
                                     std::int64_t elements, std::size_t element_size,
                                     std::size_t alignment) -> vibeqc_xtb_status_t {
      std::size_t bytes = 0u;
      if (!checked_bytes(elements, element_size, bytes)) {
        error = std::string(name) + " extent overflows size_t";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      return validate_cuda_buffer(device_id, name, buffer, bytes, alignment,
                                  CudaManagedMemoryPolicy::kReject, validated_output, error);
    };

    vibeqc_xtb_status_t status =
        validate_const("atom_offsets", batch.atom_offsets, batch.batch_size + 1,
                       sizeof(std::int64_t), alignof(std::int64_t));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("atomic_numbers", batch.atomic_numbers, batch.total_atoms,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status =
        validate_const("positions", batch.positions, coordinates, sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("molecular_charges", batch.molecular_charges, batch.batch_size,
                            sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("unpaired_electrons", batch.unpaired_electrons, batch.batch_size,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    const bool spin_present =
        batch.struct_size >= VIBEQC_XTB_BATCH_V2_SIZE &&
        (batch.spin_channels.data != nullptr || batch.spin_channels.size_bytes != 0u);
    const vibeqc_xtb_const_buffer_t absent_spin{};
    const vibeqc_xtb_const_buffer_t& spin_buffer = spin_present ? batch.spin_channels : absent_spin;
    status = validate_const("spin_channels", spin_buffer, spin_present ? batch.batch_size : 0,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_offsets", batch.point_charge_offsets,
                            point_topology_active ? batch.batch_size + 1 : 0, sizeof(std::int64_t),
                            alignof(std::int64_t));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_positions", batch.point_charge_positions,
                            point_coordinates, sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_values", batch.point_charge_values,
                            batch.total_point_charges, sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("point_charge_gammas", batch.point_charge_gammas,
                            batch.total_point_charges, sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("atomic_potential_shifts", batch.atomic_potential_shifts,
                            shift_active ? batch.total_atoms : 0, sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("charge_response_offsets", batch.charge_response_offsets,
                            response_active ? batch.batch_size + 1 : 0, sizeof(std::int64_t),
                            alignof(std::int64_t));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("charge_response_matrix", batch.charge_response_matrix,
                            response_active ? batch.total_charge_response_elements : 0,
                            sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    const vibeqc_xtb_const_buffer_t absent_interaction{};
    const vibeqc_xtb_const_buffer_t& interaction_descriptor_buffer =
        interaction_descriptors_active ? batch.interaction_descriptors : absent_interaction;
    const vibeqc_xtb_const_buffer_t& interaction_payload_buffer =
        interaction_payload_active ? batch.interaction_payload : absent_interaction;
    /* HOST descriptors are a public byte store and are decoded with memcpy;
     * only device-resident descriptors require natural alignment for typed
     * device loads. */
    const std::size_t interaction_descriptor_alignment =
        interaction_descriptor_buffer.memory_space == VIBEQC_XTB_MEMORY_HOST
            ? 1u
            : alignof(vibeqc_xtb_interaction_t);
    status = validate_const("interaction_descriptors", interaction_descriptor_buffer,
                            interaction_descriptors_active ? batch.total_interactions : 0,
                            sizeof(vibeqc_xtb_interaction_t), interaction_descriptor_alignment);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_cuda_const_buffer(
        device_id, "interaction_payload", interaction_payload_buffer,
        interaction_payload_active ? interaction_payload_buffer.size_bytes : 0u,
        alignof(std::int32_t), CudaManagedMemoryPolicy::kReject, validated_const, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    const vibeqc_xtb_const_buffer_t absent_lattice{};
    const vibeqc_xtb_const_buffer_t& cell_buffer =
        lattice_active ? batch.cell_matrices : absent_lattice;
    const vibeqc_xtb_const_buffer_t& axes_buffer =
        lattice_active ? batch.periodic_axes : absent_lattice;
    status = validate_const("cell_matrices", cell_buffer, lattice_active ? cell_elements : 0,
                            sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_const("periodic_axes", axes_buffer, lattice_active ? batch.batch_size : 0,
                            sizeof(std::int32_t), alignof(std::int32_t));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    if (result == nullptr) return VIBEQC_XTB_STATUS_SUCCESS;

    const bool energy_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_DIPOLE_MOMENTS)) != 0u;
    status = validate_output("energies", result->energies, energy_requested ? batch.batch_size : 0,
                             sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_output("forces", result->forces, force_requested ? coordinates : 0,
                             sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status =
        validate_output("atomic_charges", result->atomic_charges,
                        charges_requested ? batch.total_atoms : 0, sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_output("point_charge_forces", result->point_charge_forces,
                             point_forces_requested ? point_coordinates : 0, sizeof(double),
                             alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    const vibeqc_xtb_buffer_t absent_output{};
    const vibeqc_xtb_buffer_t& dipole_buffer =
        result->struct_size >= VIBEQC_XTB_BATCH_RESULT_V2_SIZE ? result->dipole_moments
                                                               : absent_output;
    status =
        validate_output("dipole_moments", dipole_buffer, dipoles_requested ? dipole_elements : 0,
                        sizeof(double), alignof(double));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_output("scc_iterations", result->scc_iterations, batch.batch_size,
                             sizeof(std::int32_t), alignof(std::int32_t));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = validate_output("scc_converged", result->scc_converged, batch.batch_size,
                             sizeof(std::uint8_t), alignof(std::uint8_t));
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    return validate_output("per_system_status", result->per_system_status, batch.batch_size,
                           sizeof(vibeqc_xtb_status_t), alignof(vibeqc_xtb_status_t));
  }

  vibeqc_xtb_status_t reset_request_topology_error_locked(Prepared& current, std::string& error) {
    if (!current.public_result.ready || current.public_result.request_topology_error == nullptr) {
      error = "CUDA fixed-topology request gate is not initialized";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const cudaError_t cuda_status = cudaMemsetAsync(current.public_result.request_topology_error, 0,
                                                    sizeof(std::uint32_t), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fixed-topology request gate reset", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t stage_numerical_ingress_locked(
      Prepared& current, const vibeqc_xtb_const_buffer_t& positions, bool& host_upload_enqueued,
      std::string& error) {
    host_upload_enqueued = false;
    if (!current.numerical.ready) {
      error = "CUDA GFN2 numerical refresh requires a prepared fixed topology";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    auto& numerical = current.numerical;
    auto& device = numerical.device;
    if (device.refresh_predecessor_generations == nullptr ||
        device.warm_checkpoint_generations == nullptr) {
      error = "CUDA GFN2 numerical refresh has an incomplete warm-checkpoint binding";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    auto& preprocessing = numerical.preprocessing;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for numerical refresh", cuda_status);
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }

    std::size_t position_bytes = 0u;
    if (!checked_bytes(device.total_atoms * 3, sizeof(double), position_bytes)) {
      error = "CUDA GFN2 position extent overflows size_t";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    const bool supported_space = positions.memory_space == VIBEQC_XTB_MEMORY_HOST ||
                                 positions.memory_space == VIBEQC_XTB_MEMORY_CUDA_DEVICE;
    if (positions.reserved != 0u || positions.data == nullptr ||
        positions.size_bytes < position_bytes || !supported_space) {
      error = "positions is not a sufficiently large host/CUDA buffer";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    const bool uses_host_staging = positions.memory_space == VIBEQC_XTB_MEMORY_HOST;
    if (uses_host_staging) {
      if (numerical.host_positions == nullptr || numerical.owned_host_positions == nullptr) {
        error = "positions has no runtime host-staging projection";
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
      cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
      cuda_status = cudaStreamIsCapturing(stream, &capture_status);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA numerical host-upload capture query", cuda_status);
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
      if (capture_status != cudaStreamCaptureStatusNone) {
        error = "CUDA Graph numerical refresh requires CUDA-device input buffers";
        return VIBEQC_XTB_STATUS_NOT_SUPPORTED;
      }
      if (numerical.host_staging_poisoned) {
        error = "CUDA numerical host staging is unavailable after an earlier completion failure";
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
      if (current.numerical_host_upload_completion.pending.load(std::memory_order_acquire)) {
        error = "a previous CUDA numerical host upload is still in flight";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      std::memcpy(numerical.owned_host_positions, positions.data, position_bytes);
    }

    const auto seal_host_uploads = [&]() -> cudaError_t {
      if (!host_upload_enqueued) return cudaSuccess;
      auto& completion = current.numerical_host_upload_completion;
      completion.pending.store(true, std::memory_order_release);
      cudaError_t completion_status =
          cudaEventRecord(current.numerical_host_upload_complete.get(), stream);
      if (completion_status == cudaSuccess) {
        completion_status = cudaStreamWaitEvent(current.numerical_host_completion_stream.get(),
                                                current.numerical_host_upload_complete.get(), 0u);
      }
      if (completion_status == cudaSuccess) {
        completion_status = cudaLaunchHostFunc(current.numerical_host_completion_stream.get(),
                                               release_numerical_host_upload, &completion);
      }
      if (completion_status == cudaSuccess) {
        completion_status = cudaEventRecord(current.numerical_host_release_complete.get(),
                                            current.numerical_host_completion_stream.get());
      }
      if (completion_status == cudaSuccess) {
        current.submitted = true;
      } else {
        numerical.host_staging_poisoned = true;
      }
      return completion_status;
    };

    NumericalRefreshDeviceSources sources{};
    if (uses_host_staging) {
      cuda_status = cudaMemcpyAsync(numerical.host_positions, numerical.owned_host_positions,
                                    position_bytes, cudaMemcpyHostToDevice, stream);
      if (cuda_status != cudaSuccess) {
        const cudaError_t completion_status = seal_host_uploads();
        error = cuda_error_message("positions", cuda_status);
        if (completion_status != cudaSuccess) {
          error += "; host-upload completion enqueue also failed";
        }
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
      host_upload_enqueued = true;
      current.submitted = true;
      sources.positions = numerical.host_positions;
    } else {
      sources.positions = static_cast<const double*>(positions.data);
    }

    std::size_t mask_bytes = 0u;
    if (!checked_bytes(device.batch_size, sizeof(std::uint8_t), mask_bytes)) {
      const cudaError_t completion_status = seal_host_uploads();
      error = "numerical refresh activity extent overflows size_t";
      if (completion_status != cudaSuccess) {
        error += "; host-upload completion enqueue also failed";
      }
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    auto* const requested = const_cast<std::uint8_t*>(preprocessing.activity.requested_mask);
    cuda_status = cudaMemsetAsync(requested, 1, mask_bytes, stream);
    if (cuda_status != cudaSuccess) {
      const cudaError_t completion_status = seal_host_uploads();
      error = cuda_error_message("CUDA numerical refresh activity staging", cuda_status);
      if (completion_status != cudaSuccess) {
        error += "; host-upload completion enqueue also failed";
      }
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;

    cuda_status = seal_host_uploads();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical host-upload completion enqueue", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    cuda_status =
        cudaMemsetAsync(device.interaction_request_error, 0, sizeof(std::uint32_t), stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA interaction request-gate reset", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    normalize_gfn2_interactions_kernel<<<1, 1, 0, stream>>>(device, sources);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA interaction semantic normalization", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    merge_gfn2_request_error_kernel<<<1, 1, 0, stream>>>(
        device.interaction_request_error, current.public_result.request_topology_error);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA interaction request-gate publication", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    stage_gfn2_numerical_inputs_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                         stream>>>(device, sources);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical input staging", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;

    error.clear();
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t execute_numerical_body_locked(Prepared& current,
                                                    cudaStream_t execution_stream,
                                                    std::string& error) {
    if (!current.numerical.ready) {
      error = "CUDA GFN2 numerical refresh requires a prepared fixed topology";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    auto& numerical = current.numerical;
    auto& device = numerical.device;
    auto& preprocessing = numerical.preprocessing;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for numerical refresh", cuda_status);
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }

    cuda_status = reset_gfn2_electric_field_device_errors_cuda(
        device.batch_size, device.field_system_errors, device.field_plan_error, execution_stream);
    if (cuda_status == cudaSuccess) {
      const Gfn2ElectricFieldDeviceInput field_input{
          device.candidate_field_vectors, 3 * device.batch_size, device.candidate_positions,
          3 * device.total_atoms, device.plan_token};
      cuda_status = refresh_gfn2_electric_field_potentials_cuda(
          numerical.field_batch, field_input, numerical.field_candidate_potentials,
          device.field_system_errors, device.field_plan_error, execution_stream);
    }
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA electric-field potential refresh", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? VIBEQC_XTB_STATUS_INVALID_ARGUMENT
                                                  : VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    const auto preprocessing_launch =
        compose_gfn2_preprocessing_epoch_cuda(preprocessing, execution_stream);
    if (!preprocessing_launch.success()) {
      error = "CUDA numerical preprocessing composer rejected the runtime binding";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    if (device.d4_enabled != 0u) {
      cuda_status = reset_gfn2_d4_device_errors_cuda(
          device.batch_size, const_cast<std::uint32_t*>(device.d4_system_errors),
          const_cast<std::uint32_t*>(device.d4_device_error), execution_stream);
      if (cuda_status == cudaSuccess) {
        /* compose_gfn2_preprocessing_epoch_cuda commits the physical pair
         * superset earlier on this stream. The D4 refresh therefore observes
         * current role views without host polling, and its epoch overload is
         * safe under CUDA Graph replay. */
        cuda_status = update_gfn2_d4_pairlist_cache_cuda(
            current.plan_seed.d4_batch, current.plan_seed.d4_parameters,
            preprocessing.geometry_epoch, numerical.d4_refresh_cache, numerical.d4_workspace,
            const_cast<std::uint32_t*>(device.d4_device_error), execution_stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA D4 numerical refresh", cuda_status);
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
    }
    if (device.point_enabled != 0u) {
      cuda_status = reset_gfn2_external_point_charge_scc_errors_cuda(
          device.batch_size, const_cast<std::uint32_t*>(device.point_system_errors),
          const_cast<std::uint32_t*>(device.point_plan_error), execution_stream);
      if (cuda_status == cudaSuccess) {
        initialize_gfn2_refresh_sequence_kernel<<<1, 1, 0, execution_stream>>>(
            const_cast<std::uint32_t*>(numerical.point_activity.sequence_active));
        cuda_status = cudaPeekAtLastError();
      }
      if (cuda_status == cudaSuccess) {
        cuda_status = update_gfn2_external_point_charge_scc_potential_cache_cuda(
            numerical.point_batch, numerical.point_activity, numerical.point_candidate,
            kInitialGeometryGeneration, numerical.point_workspace,
            const_cast<std::uint32_t*>(device.point_system_errors),
            const_cast<std::uint32_t*>(device.point_plan_error), execution_stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA explicit-point-charge numerical refresh", cuda_status);
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
    }
    if (device.periodic_enabled != 0u) {
      cuda_status = cudaMemsetAsync(
          device.periodic_system_errors, 0,
          static_cast<std::size_t>(device.batch_size) * sizeof(std::uint32_t), execution_stream);
      if (cuda_status == cudaSuccess) {
        cuda_status =
            cudaMemsetAsync(device.periodic_plan_error, 0, sizeof(std::uint32_t), execution_stream);
      }
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic refresh diagnostic reset", cuda_status);
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
      validate_gfn2_periodic_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                              execution_stream>>>(device);
      cuda_status = cudaPeekAtLastError();
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA periodic numerical validation", cuda_status);
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
    }

    gate_gfn2_numerical_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 1, 0,
                                         execution_stream>>>(device);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical pre-factor publication gate", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const auto factor = current.eigensolver_owner.refactor_overlap_from_device_epoch_async(
        current.eigensolver_setup_arena.get(), current.eigensolver_setup_arena.bytes(),
        current.eigensolver_binding, preprocessing.output.overlap,
        preprocessing.output.overlap_elements, preprocessing.geometry_epoch,
        current.public_result.request_topology_error, execution_stream);
    if (!factor.success()) {
      error = setup_error_message("CUDA numerical overlap refactor", factor.status,
                                  static_cast<std::uint32_t>(factor.error),
                                  static_cast<std::uint32_t>(factor.field), factor.index);
      return factor.status;
    }
    device.factor_generations = current.eigensolver_binding.cache.geometry_generations;
    device.factor_statuses = current.eigensolver_binding.cache.factor_statuses;
    commit_gfn2_numerical_refresh_kernel<<<static_cast<unsigned int>(device.batch_size), 256, 0,
                                           execution_stream>>>(device);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA numerical final publication", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    error.clear();
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t enqueue_inference_tail_locked(Prepared& current, bool capture_bounded_scc,
                                                    cudaStream_t execution_stream,
                                                    std::string& error) {
    auto& inference = current.inference;
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for CUDA GFN2 inference", cuda_status);
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }

    /* The synchronous path launches the production device-resident early-stop
     * Graph. The asynchronous request Graph instead captures the bounded
     * iteration DAG at setup. Nesting a device-launched/tail-launched SCC
     * Graph inside a conditional Graph did not provide a reliable completion
     * boundary on CUDA 12.9/Blackwell; the bounded body keeps one prebuilt
     * request-Graph launch per call and introduces no steady-state allocation,
     * host polling, transfer, or synchronization. */
    const Gfn2SccLoopLaunchResult loop =
        capture_bounded_scc ? launch_gfn2_restricted_scc_loop_cuda(
                                  current.scc_binding, inference.epoch_consumer, execution_stream)
                            : current.scc_loop.launch(execution_stream);
    if (!loop.success()) {
      std::ostringstream message;
      message << "CUDA SCC loop submission failed: mode="
              << static_cast<std::uint32_t>(loop.execution_mode)
              << " status=" << static_cast<std::uint32_t>(loop.iteration.status)
              << " stage=" << static_cast<std::uint32_t>(loop.iteration.stage)
              << " cuda=" << static_cast<int>(loop.iteration.cuda_status);
      error = message.str();
      if (loop.iteration.status == Gfn2SccIterationLaunchStatus::kInvalidBinding) {
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      if (loop.iteration.status == Gfn2SccIterationLaunchStatus::kCublasError ||
          loop.iteration.status == Gfn2SccIterationLaunchStatus::kCusolverError) {
        return VIBEQC_XTB_STATUS_EIGENSOLVER_FAILED;
      }
      if (loop.iteration.status == Gfn2SccIterationLaunchStatus::kProviderCaptureUnsupported) {
        return VIBEQC_XTB_STATUS_NOT_SUPPORTED;
      }
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    cuda_status = evaluate_gfn2_terminal_classical_energy_cuda(
        inference.terminal_plan, inference.terminal_activity, inference.terminal_results,
        inference.terminal_workspace, inference.terminal_diagnostics, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA terminal classical energy", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? VIBEQC_XTB_STATUS_INVALID_ARGUMENT
                                                  : VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    if (current.energy_force.stationary_projection.enabled == 1u) {
      cuda_status = project_gfn2_stationary_force_state_cuda(
          current.energy_force.stationary_projection, execution_stream);
      if (cuda_status != cudaSuccess) {
        error = cuda_error_message("CUDA stationary force projection", cuda_status);
        return cuda_status == cudaErrorInvalidValue ? VIBEQC_XTB_STATUS_INVALID_ARGUMENT
                                                    : VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
    }

    cuda_status = execute_gfn2_energy_force_cuda(
        current.energy_force.plan, current.energy_force.input, current.energy_force.results,
        current.energy_force.intermediates, current.energy_force.workspace,
        current.energy_force.diagnostics, inference.epoch_consumer, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA terminal energy/force execution", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? VIBEQC_XTB_STATUS_INVALID_ARGUMENT
                                                  : VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    cuda_status = publish_gfn2_inference_results_cuda(
        inference.publication_plan, inference.publication_input, inference.publication_results,
        inference.publication_workspace, inference.publication_diagnostics, execution_stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA internal inference publication", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? VIBEQC_XTB_STATUS_INVALID_ARGUMENT
                                                  : VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    initialize_gfn2_warm_checkpoint_batch_if_admitted_kernel<<<1, 1, 0, execution_stream>>>(
        inference.warm_checkpoint_batch_ready,
        {current.public_result.request_topology_error, 1, current.host.plan_token});
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA warm-checkpoint aggregate initialization", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    const WarmCheckpointPublicationDeviceBinding checkpoint{
        current.host.basis.batch_size,
        inference.epoch_consumer.epoch,
        inference.epoch_consumer.eligible_mask,
        inference.epoch_consumer.committed_generations,
        inference.publication_diagnostics.plan_error,
        inference.publication_results.system_statuses,
        inference.publication_results.converged,
        inference.warm_checkpoint_generations,
        inference.warm_checkpoint_batch_ready,
        current.numerical.device.committed_field_attached,
        current.numerical.device.committed_field_vectors,
        inference.warm_checkpoint_field_attached,
        inference.warm_checkpoint_field_vectors,
        current.public_result.request_topology_error,
    };
    constexpr int kThreads = 256;
    const auto blocks = static_cast<unsigned int>(
        (static_cast<std::uint64_t>(checkpoint.batch_size) + kThreads - 1u) / kThreads);
    publish_gfn2_warm_checkpoint_generation_kernel<<<blocks, kThreads, 0, execution_stream>>>(
        checkpoint);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA warm-checkpoint publication", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    current.submitted = true;
    error.clear();
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t execute_inference_body_locked(Prepared& current, std::string& error) {
    return enqueue_inference_tail_locked(current, false, stream, error);
  }

  vibeqc_xtb_status_t refresh_numerical_locked(Prepared& current,
                                               const vibeqc_xtb_const_buffer_t& positions,
                                               std::string& error) {
    bool host_upload_enqueued = false;
    vibeqc_xtb_status_t status =
        stage_numerical_ingress_locked(current, positions, host_upload_enqueued, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = execute_numerical_body_locked(current, stream, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    if (host_upload_enqueued) {
      const cudaError_t release_status =
          cudaStreamWaitEvent(stream, current.numerical_host_release_complete.get(), 0u);
      if (release_status != cudaSuccess) {
        current.numerical.host_staging_poisoned = true;
        error = cuda_error_message("CUDA numerical host-release tail ordering", release_status);
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
      current.submitted = true;
    }
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t execute_inference_locked(Prepared& current, std::string& error) {
    if (!current.inference.ready || !current.numerical.ready) {
      error = "CUDA GFN2 inference requires a prepared numerical/runtime binding";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    if (current.numerical.device.refresh_predecessor_generations == nullptr ||
        current.inference.warm_checkpoint_generations == nullptr ||
        current.inference.warm_checkpoint_batch_ready == nullptr) {
      error = "CUDA GFN2 inference has an incomplete warm-checkpoint binding";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    auto& inference = current.inference;

    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("cudaSetDevice for CUDA GFN2 inference", cuda_status);
      return VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE;
    }

    /*
     * Public admission requires FRESH SCC. Invalidate every old checkpoint
     * before restoring the canonical fresh state so a failed attempt cannot
     * leave a stale warm token associated with the new request.
     */
    const auto checkpoint_elements = current.host.basis.batch_size;
    constexpr int kInvalidateThreads = 256;
    const auto invalidate_blocks = static_cast<unsigned int>(
        (static_cast<std::uint64_t>(checkpoint_elements) + kInvalidateThreads - 1u) /
        kInvalidateThreads);
    invalidate_gfn2_warm_checkpoint_if_admitted_kernel<<<invalidate_blocks, kInvalidateThreads, 0,
                                                         stream>>>(
        inference.warm_checkpoint_generations, checkpoint_elements,
        current.public_result.request_topology_error);
    cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA fresh warm-checkpoint invalidation", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    current.submitted = true;
    inference.warm_checkpoint_ready = false;
    const auto diagnostic = current.initializer.upload_if_admitted_async(
        current.iteration_arena.get(), current.iteration_arena.bytes(), current.ready,
        current.public_result.request_topology_error, stream);
    if (!diagnostic.success()) {
      error = setup_error_message("CUDA SCC fresh-state restore", diagnostic.status,
                                  static_cast<std::uint32_t>(diagnostic.error),
                                  static_cast<std::uint32_t>(diagnostic.field), diagnostic.index);
      return diagnostic.status;
    }
    current.submitted = true;

    const vibeqc_xtb_status_t body_status = execute_inference_body_locked(current, error);
    if (body_status != VIBEQC_XTB_STATUS_SUCCESS) return body_status;
    inference.warm_checkpoint_ready = true;
    error.clear();
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  /*
   * A synchronous public call must not return while either the owner stream
   * can still read caller CUDA pointers or the private host-upload callback
   * can still access Prepared-owned lease state.  Failure paths use the same
   * disable-timing event as successful publication and fall back to an exact
   * owner-stream fence if recording that event itself fails.
   */
  vibeqc_xtb_status_t settle_public_submissions_locked(Prepared& current,
                                                       vibeqc_xtb_status_t primary_status,
                                                       std::string& error) {
    cudaError_t owner_status = cudaSuccess;
    if (current.submitted) {
      {
        owner_status = cudaEventRecord(current.public_result_completion_event.get(), stream);
        if (owner_status == cudaSuccess) {
          owner_status = cudaEventSynchronize(current.public_result_completion_event.get());
        }
        if (owner_status != cudaSuccess) {
          const cudaError_t fallback_status = cudaStreamSynchronize(stream);
          if (fallback_status == cudaSuccess) owner_status = cudaSuccess;
        }
        if (owner_status == cudaSuccess) current.submitted = false;
      }
    }

    cudaError_t host_status = cudaSuccess;
    if (current.numerical_host_completion_stream.valid()) {
      host_status = cudaStreamSynchronize(current.numerical_host_completion_stream.get());
      if (host_status == cudaSuccess) {
        /* The stream fence is after the release callback, so this store cannot
         * race a previous lease generation or clear a future one. */
        current.numerical_host_upload_completion.pending.store(false, std::memory_order_release);
      } else {
        current.numerical.host_staging_poisoned = true;
      }
    }
    if (owner_status == cudaSuccess && host_status == cudaSuccess) return primary_status;

    std::ostringstream message;
    if (!error.empty()) message << error << "; additionally, ";
    message << "CUDA public transaction settlement failed";
    if (owner_status != cudaSuccess) {
      message << " owner_stream=" << cudaGetErrorString(owner_status);
    }
    if (host_status != cudaSuccess) {
      message << " host_completion_stream=" << cudaGetErrorString(host_status);
    }
    error = message.str();
    return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
  }

  vibeqc_xtb_status_t enqueue_public_result_prepare_locked(
      Prepared& current, const vibeqc_xtb_batch_t& batch,
      const vibeqc_xtb_compute_options_t& options, vibeqc_xtb_batch_result_t& result,
      PublicResultTransaction& transaction, std::string& error) {
    const bool prior_warm_checkpoint_ready = transaction.prior_warm_checkpoint_ready;
    transaction = {};
    transaction.prior_warm_checkpoint_ready = prior_warm_checkpoint_ready;
    if (!current.public_result.ready || !current.inference.ready ||
        current.public_result_completion_event.get() == nullptr) {
      error = "CUDA public result bridge requires a complete prepared runtime";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const std::int64_t batch_size = current.host.basis.batch_size;
    const std::int64_t atoms = current.host.basis.total_atoms;
    const std::int64_t points = current.host.external.total_point_charges;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates)) {
      error = "CUDA public result extent overflows int64_t";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const std::uint64_t token = current.host.plan_token;
    const bool external_operator = batch.atomic_potential_shifts.data != nullptr ||
                                   batch.atomic_potential_shifts.size_bytes != 0u ||
                                   batch.charge_response_matrix.data != nullptr ||
                                   batch.charge_response_matrix.size_bytes != 0u;
    const std::uint32_t result_flags =
        (external_operator ? static_cast<std::uint32_t>(
                                 VIBEQC_XTB_RESULT_FORCES_EXCLUDE_EXTERNAL_OPERATOR_DERIVATIVES)
                           : 0u) |
        (((options.flags & VIBEQC_XTB_COMPUTE_DIPOLE_MOMENTS) != 0u)
             ? static_cast<std::uint32_t>(VIBEQC_XTB_RESULT_DIPOLE_MOMENTS)
             : 0u);
    const Gfn2PublicResultBridgeDevicePlan plan{
        kGfn2PublicResultBridgeAbiVersion,
        options.flags,
        result_flags,
        0u,
        token,
        batch_size,
        atoms,
        points,
    };
    const auto& inference = current.inference;
    Gfn2PublicResultBridgeDeviceInput input{};
    input.energies = inference.publication_results.energies;
    input.energy_elements = inference.publication_results.energy_elements;
    input.qm_forces = inference.publication_results.qm_forces;
    input.qm_force_elements = inference.publication_results.qm_force_elements;
    input.atomic_charges = inference.publication_results.atomic_charges;
    input.atomic_charge_elements = inference.publication_results.atomic_charge_elements;
    input.point_forces = inference.publication_results.point_forces;
    input.point_force_elements = inference.publication_results.point_force_elements;
    input.iterations = inference.publication_results.iterations;
    input.converged = inference.publication_results.converged;
    input.system_statuses = inference.publication_results.system_statuses;
    input.batch_elements = inference.publication_results.batch_elements;
    input.publication_plan_error = inference.publication_diagnostics.plan_error;
    input.request_topology_error = current.public_result.request_topology_error;
    input.publication_epoch_snapshot = inference.publication_workspace.epoch_snapshot;
    input.current_geometry_epoch = current.numerical.preprocessing.geometry_epoch.value;
    input.plan_token = token;
    input.dipole_moments = inference.publication_results.dipole_moments;
    input.dipole_moment_elements = inference.publication_results.dipole_moment_elements;

    Gfn2PublicResultBridgeDeviceDestinations destinations{};
    Gfn2PublicResultBridgeHostStaging staging{};
    const auto bind = [](const vibeqc_xtb_buffer_t& output, std::int64_t elements, void* host_stage,
                         Gfn2PublicResultBridgeDestination& destination,
                         Gfn2PublicResultBridgeHostBuffer& host) {
      if (elements == 0) {
        destination = {};
        host = {};
        return;
      }
      const bool host_route = output.memory_space == VIBEQC_XTB_MEMORY_HOST;
      destination.route =
          host_route ? Gfn2PublicResultRoute::kHost : Gfn2PublicResultRoute::kCudaDevice;
      destination.device_data = host_route ? nullptr : output.data;
      destination.elements = elements;
      host.data = host_route ? host_stage : nullptr;
      host.elements = host_route ? elements : 0;
    };
    const bool energy_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_DIPOLE_MOMENTS)) != 0u;
    const vibeqc_xtb_buffer_t absent_output{};
    const auto& dipole_output = result.struct_size >= VIBEQC_XTB_BATCH_RESULT_V2_SIZE
                                    ? result.dipole_moments
                                    : absent_output;
    auto& public_state = current.public_result;
    bind(result.energies, energy_requested ? batch_size : 0, public_state.energies,
         destinations.energies, staging.energies);
    bind(result.forces, force_requested ? coordinates : 0, public_state.qm_forces,
         destinations.qm_forces, staging.qm_forces);
    bind(result.atomic_charges, charges_requested ? atoms : 0, public_state.atomic_charges,
         destinations.atomic_charges, staging.atomic_charges);
    bind(result.point_charge_forces, point_forces_requested ? point_coordinates : 0,
         public_state.point_forces, destinations.point_forces, staging.point_forces);
    bind(dipole_output, dipoles_requested ? 3 * batch_size : 0, public_state.dipole_moments,
         destinations.dipole_moments, staging.dipole_moments);
    bind(result.scc_iterations, batch_size, public_state.iterations, destinations.iterations,
         staging.iterations);
    bind(result.scc_converged, batch_size, public_state.converged, destinations.converged,
         staging.converged);
    bind(result.per_system_status, batch_size, public_state.system_statuses,
         destinations.system_statuses, staging.system_statuses);
    destinations.plan_token = token;
    staging.control = public_state.host_control;
    staging.control_elements = 1;
    staging.pending_result_flags = &public_state.pending_result_flags;
    staging.plan_token = token;

    cudaError_t cuda_status =
        prepare_gfn2_public_results_cuda(plan, input, public_state.device_staging, destinations,
                                         staging, public_state.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA public result bridge submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? VIBEQC_XTB_STATUS_INVALID_ARGUMENT
                                                  : VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    cuda_status =
        cudaMemcpyAsync(public_state.warm_checkpoint_ready, inference.warm_checkpoint_batch_ready,
                        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA warm-checkpoint readiness download", cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    transaction.plan = plan;
    transaction.destinations = destinations;
    transaction.phase = PublicResultTransactionPhase::kPrepareSubmitted;
    error.clear();
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  /*
   * Append one completion marker after a submitted public bridge phase. The
   * commit phase retains the historical stream-fence fallback when recording
   * the event itself fails; prepare keeps its existing strict event behavior.
   */
  vibeqc_xtb_status_t record_public_result_completion_locked(
      Prepared& current, PublicResultTransaction& transaction,
      PublicResultTransactionPhase submitted_phase, PublicResultTransactionPhase recorded_phase,
      PublicResultTransactionPhase completed_phase, bool fallback_to_stream,
      const char* failure_context, std::string& error) {
    if (transaction.phase != submitted_phase ||
        current.public_result_completion_event.get() == nullptr) {
      error = "CUDA public result completion marker has an invalid transaction phase";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    const cudaError_t record_status =
        cudaEventRecord(current.public_result_completion_event.get(), stream);
    if (record_status == cudaSuccess) {
      transaction.phase = recorded_phase;
      return VIBEQC_XTB_STATUS_SUCCESS;
    }
    if (fallback_to_stream && cudaStreamSynchronize(stream) == cudaSuccess) {
      current.submitted = false;
      transaction.phase = completed_phase;
      return VIBEQC_XTB_STATUS_SUCCESS;
    }

    error = cuda_error_message(failure_context, record_status);
    return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
  }

  /* Completion observation never publishes host bytes or result.flags. */
  vibeqc_xtb_status_t wait_public_result_completion_locked(
      Prepared& current, PublicResultTransaction& transaction,
      PublicResultTransactionPhase recorded_phase, PublicResultTransactionPhase completed_phase,
      const char* failure_context, std::string& error) {
    if (transaction.phase == completed_phase) return VIBEQC_XTB_STATUS_SUCCESS;
    if (transaction.phase != recorded_phase ||
        current.public_result_completion_event.get() == nullptr) {
      error = "CUDA public result completion wait has an invalid transaction phase";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    const cudaError_t cuda_status =
        cudaEventSynchronize(current.public_result_completion_event.get());
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message(failure_context, cuda_status);
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    current.submitted = false;
    transaction.phase = completed_phase;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  /* Read the pinned aggregate only after prepare completion is observed. */
  vibeqc_xtb_status_t accept_public_result_prepare_locked(Prepared& current,
                                                          PublicResultTransaction& transaction,
                                                          std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kPrepareCompleted) {
      error = "CUDA public result prepare acceptance precedes completion";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const auto& public_state = current.public_result;
    const auto aggregate =
        static_cast<Gfn2PublicResultBridgeError>(public_state.host_control->aggregate_error);
    if (aggregate != Gfn2PublicResultBridgeError::kSuccess) {
      if (aggregate == Gfn2PublicResultBridgeError::kRequestTopologyMismatch) {
        error = "the batch topology does not match the fixed CUDA plan topology";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestInvalidArgument) {
        error = "CUDA stream-ordered request validation rejected an invalid descriptor";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotSupported) {
        error = "CUDA stream-ordered request validation rejected unsupported periodic axes";
        return VIBEQC_XTB_STATUS_NOT_SUPPORTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotImplemented) {
        error = "the CUDA request uses a validated feature that is not implemented yet";
        return VIBEQC_XTB_STATUS_NOT_IMPLEMENTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestWarmIncompatible) {
        error = "CUDA strict WARM SCC start requires an identical electric-field attachment";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      std::ostringstream message;
      message << "CUDA public result bridge rejected inference: aggregate_error="
              << static_cast<std::uint32_t>(aggregate) << " publication_plan_error="
              << public_state.host_control->internal_publication_plan_error
              << " publication_epoch=" << public_state.host_control->publication_epoch_snapshot
              << " current_epoch=" << public_state.host_control->current_geometry_epoch;
      error = message.str();
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    transaction.phase = PublicResultTransactionPhase::kPrepareAccepted;
    error.clear();
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  /* The asynchronous path submits the device-gated commit before returning,
   * so its host acceptance occurs only after that commit event completes. The
   * device kernel itself reads the same aggregate control and is a no-op on a
   * rejected prepare image. */
  vibeqc_xtb_status_t accept_public_result_after_commit_locked(Prepared& current,
                                                               PublicResultTransaction& transaction,
                                                               std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kCommitCompleted) {
      error = "CUDA public result acceptance precedes asynchronous commit completion";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const auto& public_state = current.public_result;
    const auto aggregate =
        static_cast<Gfn2PublicResultBridgeError>(public_state.host_control->aggregate_error);
    if (aggregate != Gfn2PublicResultBridgeError::kSuccess) {
      if (aggregate == Gfn2PublicResultBridgeError::kRequestTopologyMismatch) {
        error = "the batch topology does not match the fixed CUDA plan topology";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestInvalidArgument) {
        error = "CUDA stream-ordered request validation rejected an invalid descriptor";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotSupported) {
        error = "CUDA stream-ordered request validation rejected unsupported periodic axes";
        return VIBEQC_XTB_STATUS_NOT_SUPPORTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestNotImplemented) {
        error = "the CUDA request uses a validated feature that is not implemented yet";
        return VIBEQC_XTB_STATUS_NOT_IMPLEMENTED;
      }
      if (aggregate == Gfn2PublicResultBridgeError::kRequestWarmIncompatible) {
        error = "CUDA strict WARM SCC start requires an identical electric-field attachment";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      std::ostringstream message;
      message << "CUDA public result bridge rejected inference: aggregate_error="
              << static_cast<std::uint32_t>(aggregate) << " publication_plan_error="
              << public_state.host_control->internal_publication_plan_error
              << " publication_epoch=" << public_state.host_control->publication_epoch_snapshot
              << " current_epoch=" << public_state.host_control->current_geometry_epoch;
      error = message.str();
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    error.clear();
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  vibeqc_xtb_status_t enqueue_public_result_commit_locked(Prepared& current,
                                                          PublicResultTransaction& transaction,
                                                          bool allow_device_gated_prepare,
                                                          std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kPrepareAccepted &&
        !(allow_device_gated_prepare &&
          transaction.phase == PublicResultTransactionPhase::kPrepareSubmitted)) {
      error = "CUDA public result commit has no accepted prepared image";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const cudaError_t cuda_status = commit_gfn2_public_results_cuda(
        transaction.plan, current.public_result.device_staging, transaction.destinations,
        current.public_result.diagnostics, stream);
    if (cuda_status != cudaSuccess) {
      error = cuda_error_message("CUDA caller-device result commit submission", cuda_status);
      return cuda_status == cudaErrorInvalidValue ? VIBEQC_XTB_STATUS_INVALID_ARGUMENT
                                                  : VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    /* From this point onward caller CUDA bytes may be modified. No later
     * failure is recoverable by pretending that the transaction rolled back. */
    current.submitted = true;
    transaction.phase = PublicResultTransactionPhase::kCommitSubmitted;
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  /* Host result publication is a separate, non-CUDA phase after commit wait. */
  vibeqc_xtb_status_t publish_public_results_locked(
      Prepared& current, const vibeqc_xtb_compute_options_t& options,
      vibeqc_xtb_batch_result_t& result, PublicResultTransaction& transaction,
      bool commit_host_outputs, bool publish_descriptor_flags,
      std::uint32_t& completed_result_flags, std::string& error) {
    if (transaction.phase != PublicResultTransactionPhase::kCommitCompleted) {
      error = "CUDA public result host publication precedes commit completion";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    if (!commit_host_outputs) return VIBEQC_XTB_STATUS_INTERNAL_ERROR;

    const std::int64_t batch_size = current.host.basis.batch_size;
    const std::int64_t atoms = current.host.basis.total_atoms;
    const std::int64_t points = current.host.external.total_point_charges;
    std::int64_t coordinates = 0;
    std::int64_t point_coordinates = 0;
    if (!checked_elements(atoms, 3, coordinates) ||
        !checked_elements(points, 3, point_coordinates)) {
      error = "CUDA public result extent changed after commit acceptance";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
    const bool energy_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ENERGY)) != 0u;
    const bool force_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_FORCES)) != 0u;
    const bool charges_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ATOMIC_CHARGES)) != 0u;
    const bool point_forces_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_POINT_CHARGE_FORCES)) != 0u;
    const bool dipoles_requested =
        (options.flags & static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_DIPOLE_MOMENTS)) != 0u;
    const vibeqc_xtb_buffer_t absent_output{};
    const auto& dipole_output = result.struct_size >= VIBEQC_XTB_BATCH_RESULT_V2_SIZE
                                    ? result.dipole_moments
                                    : absent_output;
    auto& public_state = current.public_result;
    const auto commit_host = [](const vibeqc_xtb_buffer_t& output, const void* staging_data,
                                std::int64_t elements, std::size_t element_size) {
      if (elements != 0 && output.memory_space == VIBEQC_XTB_MEMORY_HOST) {
        std::memcpy(output.data, staging_data, static_cast<std::size_t>(elements) * element_size);
      }
    };
    commit_host(result.energies, public_state.energies, energy_requested ? batch_size : 0,
                sizeof(double));
    commit_host(result.forces, public_state.qm_forces, force_requested ? coordinates : 0,
                sizeof(double));
    commit_host(result.atomic_charges, public_state.atomic_charges, charges_requested ? atoms : 0,
                sizeof(double));
    commit_host(result.point_charge_forces, public_state.point_forces,
                point_forces_requested ? point_coordinates : 0, sizeof(double));
    commit_host(dipole_output, public_state.dipole_moments, dipoles_requested ? 3 * batch_size : 0,
                sizeof(double));
    commit_host(result.scc_iterations, public_state.iterations, batch_size, sizeof(std::int32_t));
    commit_host(result.scc_converged, public_state.converged, batch_size, sizeof(std::uint8_t));
    commit_host(result.per_system_status, public_state.system_statuses, batch_size,
                sizeof(vibeqc_xtb_status_t));
    completed_result_flags = public_state.pending_result_flags;
    if (publish_descriptor_flags) result.flags = completed_result_flags;
    current.inference.warm_checkpoint_ready = *public_state.warm_checkpoint_ready != 0u;
    transaction.phase = PublicResultTransactionPhase::kPublished;
    error.clear();
    return VIBEQC_XTB_STATUS_SUCCESS;
  }

  /* Complete the one request-owned publication transaction in a single place.
   * In particular, a deferred stream-ordered failure must not leave the host
   * checkpoint-ready bit resurrected by an otherwise successful D2H result
   * bridge. The device token was already consumed by FRESH/WARM admission, so
   * keeping the host bit false makes every later strict WARM reject safely. */

  std::int32_t device_id = -1;
  cudaStream_t stream = nullptr;
  Gfn2CudaTopologyStaging topology_staging;
  cusolverDnHandle_t solver = nullptr;
  cusolverDnParams_t solver_parameters = nullptr;
  syevjInfo_t solver_jacobi = nullptr;
  cublasHandle_t blas = nullptr;
  bool handles_created = false;
  std::uint64_t next_plan_token = 1u;
  std::unique_ptr<Prepared> prepared;
  mutable std::mutex mutex;
};

Gfn2CudaExecutionCache::Gfn2CudaExecutionCache(std::int32_t device_id, void* stream)
    : impl_(std::make_unique<Impl>(device_id, stream)) {}

Gfn2CudaExecutionCache::~Gfn2CudaExecutionCache() = default;

vibeqc_xtb_status_t execute_restricted_gfn2_cuda_impl(Gfn2CudaExecutionCache& cache,
                                                      const vibeqc_xtb_batch_t& batch,
                                                      const vibeqc_xtb_compute_options_t& options,
                                                      vibeqc_xtb_batch_result_t& result,
                                                      std::string& error) {
  if (cache.impl_ == nullptr) {
    error = "CUDA GFN2 execution cache has no implementation";
    return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
  }
  const auto contract_status = validate_molecular_request(batch, options, error);
  if (contract_status != VIBEQC_XTB_STATUS_SUCCESS) return contract_status;
  auto& implementation = *cache.impl_;
  std::lock_guard<std::mutex> lock(implementation.mutex);

  ScopedCudaDevice device(implementation.device_id, error);
  if (!device.ok()) return device.status();
  const auto finish = [&](vibeqc_xtb_status_t status) -> vibeqc_xtb_status_t {
    std::string restore_error;
    const vibeqc_xtb_status_t restore_status = device.restore(restore_error);
    if (restore_status == VIBEQC_XTB_STATUS_SUCCESS) return status;
    if (status != VIBEQC_XTB_STATUS_SUCCESS && !error.empty()) {
      error += "; additionally, " + restore_error;
    } else {
      error = std::move(restore_error);
    }
    return restore_status;
  };

  /* Keep every transaction-owned CUDA object inside this scope.  It ends
   * before finish() restores the caller's current device, so failed candidate
   * teardown and host-callback settlement always run on the context device. */
  const vibeqc_xtb_status_t transaction_status = [&]() -> vibeqc_xtb_status_t {
    vibeqc_xtb_status_t status =
        validate_cuda_stream_owner(implementation.device_id, implementation.stream, true, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = implementation.validate_public_request_pointers(batch, options, &result, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = implementation.ensure_handles(error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;

    const Gfn2CudaTopologyStagingDiagnostic staged =
        implementation.topology_staging.stage_and_validate(batch, error);
    if (!staged.success()) return staged.status;
    bool topology_candidate_pending =
        staged.disposition == Gfn2CudaTopologyStageDisposition::kCandidate;
    const auto abort_topology_candidate = [&]() noexcept {
      if (topology_candidate_pending) {
        implementation.topology_staging.abort_candidate();
        topology_candidate_pending = false;
      }
    };

    const Gfn2CudaTopologyHostSnapshot* topology =
        topology_candidate_pending ? implementation.topology_staging.candidate_snapshot()
                                   : implementation.topology_staging.committed_snapshot();
    if (topology == nullptr) {
      abort_topology_candidate();
      error = "CUDA topology staging did not expose the selected canonical snapshot";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }

    Gfn2CudaExecutionCache::Impl::Prepared* working = implementation.prepared.get();
    const bool reuse_runtime =
        working != nullptr && topology_snapshot_matches(*topology, options, working->host.key);
    std::unique_ptr<Gfn2CudaExecutionCache::Impl::Prepared> candidate;
    if (!reuse_runtime) {
      TopologyKey key;
      std::vector<double> seed_positions;
      std::vector<double> seed_point_positions;
      std::vector<double> seed_point_values;
      std::vector<double> seed_point_gammas;
      std::vector<double> seed_periodic_shifts;
      std::vector<double> seed_periodic_response;
      status = make_topology_only_seed(*topology, options, key, seed_positions,
                                       seed_point_positions, seed_point_values, seed_point_gammas,
                                       seed_periodic_shifts, seed_periodic_response, error);
      if (status == VIBEQC_XTB_STATUS_SUCCESS) {
        try {
          status = implementation.build_candidate(
              std::move(key), std::move(seed_positions), std::move(seed_point_positions),
              std::move(seed_point_values), std::move(seed_point_gammas),
              std::move(seed_periodic_shifts), std::move(seed_periodic_response), candidate, error);
        } catch (const std::bad_alloc&) {
          error = "failed to allocate a CUDA GFN2 runtime candidate";
          status = VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
        } catch (const std::exception& exception) {
          error = exception.what();
          status = VIBEQC_XTB_STATUS_INTERNAL_ERROR;
        } catch (...) {
          error = "unknown exception while constructing a CUDA GFN2 runtime candidate";
          status = VIBEQC_XTB_STATUS_INTERNAL_ERROR;
        }
      }
      if (status != VIBEQC_XTB_STATUS_SUCCESS) {
        abort_topology_candidate();
        return status;
      }
      working = candidate.get();
    }

    const auto fail_working_transaction = [&](vibeqc_xtb_status_t failure) {
      const vibeqc_xtb_status_t settled =
          implementation.settle_public_submissions_locked(*working, failure, error);
      abort_topology_candidate();
      return settled;
    };

    status = implementation.reset_request_topology_error_locked(*working, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return fail_working_transaction(status);

    const bool prior_warm_checkpoint_ready = working->inference.warm_checkpoint_ready;
    status = implementation.refresh_numerical_locked(*working, batch.positions, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.execute_inference_locked(*working, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return fail_working_transaction(status);
    /* Public synchronous readiness is finalized only after the completion
     * event and aggregate bridge diagnostics are known to have succeeded. */
    PublicResultTransaction public_result_transaction;
    public_result_transaction.prior_warm_checkpoint_ready = prior_warm_checkpoint_ready;
    working->inference.warm_checkpoint_ready = false;
    status = implementation.enqueue_public_result_prepare_locked(*working, batch, options, result,
                                                                 public_result_transaction, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.record_public_result_completion_locked(
        *working, public_result_transaction, PublicResultTransactionPhase::kPrepareSubmitted,
        PublicResultTransactionPhase::kPrepareCompletionRecorded,
        PublicResultTransactionPhase::kPrepareCompleted, false, "CUDA public inference completion",
        error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.wait_public_result_completion_locked(
        *working, public_result_transaction,
        PublicResultTransactionPhase::kPrepareCompletionRecorded,
        PublicResultTransactionPhase::kPrepareCompleted, "CUDA public inference completion", error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return fail_working_transaction(status);
    status = implementation.accept_public_result_prepare_locked(*working, public_result_transaction,
                                                                error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) {
      if (Gfn2CudaExecutionCache::Impl::request_semantic_rejection(working->public_result)) {
        working->inference.warm_checkpoint_ready =
            public_result_transaction.prior_warm_checkpoint_ready;
      }
      return fail_working_transaction(status);
    }

    /* The completed prepare is accepted before the private host-upload stream
     * is settled and before any caller CUDA output is touched. */
    status = implementation.settle_public_submissions_locked(*working, status, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) {
      abort_topology_candidate();
      return status;
    }
    if (topology_candidate_pending) {
      const Gfn2CudaTopologyStagingDiagnostic prepared_commit =
          implementation.topology_staging.prepare_candidate_commit(error);
      if (!prepared_commit.success()) return fail_working_transaction(prepared_commit.status);
      if (!implementation.topology_staging.candidate_publishable()) {
        error = "prepared CUDA topology candidate is not publishable";
        return fail_working_transaction(VIBEQC_XTB_STATUS_INTERNAL_ERROR);
      }
    }

    status = implementation.enqueue_public_result_commit_locked(*working, public_result_transaction,
                                                                false, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return fail_working_transaction(status);

    /* A successful launch is the caller-device commit point. Ownership moves
     * must now follow even if stream completion later reports a hard fault;
     * claiming rollback could not restore caller CUDA bytes. */
    if (candidate != nullptr) implementation.prepared = std::move(candidate);
    bool ownership_published = true;
    if (topology_candidate_pending) {
      ownership_published = implementation.topology_staging.publish_candidate();
      topology_candidate_pending = false;
      if (!ownership_published) {
        error = "CUDA topology publication invariant failed after caller-device commit acceptance";
      }
    }
    constexpr const char* kCommitCompletionFailure =
        "CUDA caller-device result commit failed after its kernel was accepted; caller CUDA "
        "outputs may have been modified";
    status = implementation.record_public_result_completion_locked(
        *working, public_result_transaction, PublicResultTransactionPhase::kCommitSubmitted,
        PublicResultTransactionPhase::kCommitCompletionRecorded,
        PublicResultTransactionPhase::kCommitCompleted, true, kCommitCompletionFailure, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    status = implementation.wait_public_result_completion_locked(
        *working, public_result_transaction,
        PublicResultTransactionPhase::kCommitCompletionRecorded,
        PublicResultTransactionPhase::kCommitCompleted, kCommitCompletionFailure, error);
    if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
    std::uint32_t completed_result_flags = 0u;
    status = implementation.publish_public_results_locked(
        *working, options, result, public_result_transaction, ownership_published, true,
        completed_result_flags, error);
    return status;
  }();
  return finish(transaction_status);
}

vibeqc_xtb_status_t execute_restricted_gfn2_cuda(Gfn2CudaExecutionCache& cache,
                                                 const vibeqc_xtb_batch_t& batch,
                                                 const vibeqc_xtb_compute_options_t& options,
                                                 vibeqc_xtb_batch_result_t& result,
                                                 std::string& error) {
  return execute_restricted_gfn2_cuda_impl(cache, batch, options, result, error);
}

}  // namespace vibeqc::xtb::detail

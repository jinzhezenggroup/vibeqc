#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "dft/dispersion/d4_cuda.hpp"
#include "dft/dispersion/d4_eeq_data.hpp"
#include "dft/dispersion/d4_eeq_r2scan3c_c6.hpp"
#include "dft/dispersion/d4_runtime.hpp"
#include "generated_d4_derivative.hpp"

#if defined(VIBEQC_TEST_HOOKS)
namespace {
thread_local bool fail_next_d4_after_coordinate_upload = false;
}
extern "C" void d4_cuda_fail_after_coordinate_upload_for_test_v1() {
  fail_next_d4_after_coordinate_upload = true;
}
#endif

namespace vibeqc::dft::dispersion {

struct D4CudaOwner {
  int device_id{-1};
  cudaStream_t stream{};
  std::uint32_t systems{};
  std::uint64_t atoms{};
  std::uint32_t workers{};
  std::size_t eeq_workspace_stride{};
  std::size_t dqdr_stride{};

  std::uint32_t* offsets{};
  std::int32_t* atomic_numbers{};
  double* total_charges{};
  double* coordinates{};
  std::uint8_t* active{};
  std::uint8_t* want_gradient{};
  std::uint8_t* fixed_active{};
  D4Status* statuses{};
  D4Status* fixed_statuses{};
  double* energies{};
  double* gradients{};
  double* charges{};
  double* dedq{};
  std::uint32_t* next_system{};
  double* eeq_workspace{};
  double* dqdr_workspace{};
  double* fixed_workspace{};
  data::D4ElementData* elements{};
  data::D4ReferenceData* references{};
  double* reference_c6{};
  eeq_data::EEQChargeElementData* charge_elements{};
};

namespace {

class DeviceScope {
 public:
  explicit DeviceScope(int requested) {
    error_ = cudaGetDevice(&previous_);
    if (error_ == cudaSuccess && previous_ != requested) {
      error_ = cudaSetDevice(requested);
      restore_ = error_ == cudaSuccess;
    }
  }
  ~DeviceScope() {
    if (restore_) cudaSetDevice(previous_);
  }
  [[nodiscard]] cudaError_t error() const noexcept { return error_; }

 private:
  int previous_{-1};
  bool restore_{false};
  cudaError_t error_{cudaSuccess};
};

vibeqc_status cuda_failure(cudaError_t error, const char* action, std::string& detail) {
  detail = std::string(action) + ": " + cudaGetErrorString(error);
  return error == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                            : VIBEQC_STATUS_CUDA_ERROR;
}

template <typename T>
bool allocate(T*& pointer, std::size_t count, std::string& detail) {
  if (!count) {
    pointer = nullptr;
    return true;
  }
  const auto error = cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T));
  if (error == cudaSuccess) return true;
  detail = std::string("D4 CUDA allocation failed: ") + cudaGetErrorString(error);
  return false;
}

template <typename T>
bool upload(cudaStream_t stream, T* destination, const T* source, std::size_t count,
            std::string& detail) {
  if (!count) return true;
  const auto error =
      cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) return true;
  detail = std::string("D4 CUDA setup upload failed: ") + cudaGetErrorString(error);
  return false;
}

__device__ void zero_item(std::uint32_t system, const std::uint32_t* offsets, double* energies,
                          double* gradients, double* charges) {
  const std::size_t begin = offsets[system];
  const std::size_t end = offsets[system + 1];
  energies[2 * system] = 0.0;
  energies[2 * system + 1] = 0.0;
  for (std::size_t atom = begin; atom < end; ++atom) {
    charges[atom] = 0.0;
    gradients[3 * atom] = 0.0;
    gradients[3 * atom + 1] = 0.0;
    gradients[3 * atom + 2] = 0.0;
  }
}

__global__ void eeq_prepare_kernel(std::uint32_t systems, const std::uint32_t* offsets,
                                   const std::int32_t* atomic_numbers, const double* total_charges,
                                   const double* coordinates, const std::uint8_t* active,
                                   EEQTables eeq_tables, std::uint32_t* next_system,
                                   double* eeq_workspace, std::size_t eeq_workspace_stride,
                                   double* dqdr_workspace, std::size_t dqdr_stride,
                                   D4Status* statuses, std::uint8_t* fixed_active, double* charges,
                                   double* energies, double* gradients) {
  if (threadIdx.x != 0) return;
  const auto slot = static_cast<std::size_t>(blockIdx.x);
  double* workspace = eeq_workspace + slot * eeq_workspace_stride;
  double* dqdr = dqdr_workspace + slot * dqdr_stride;
  while (true) {
    const std::uint32_t system = atomicAdd(next_system, 1u);
    if (system >= systems) return;
    if (!active[system]) {
      statuses[system] = D4Status::success;
      fixed_active[system] = 0;
      zero_item(system, offsets, energies, gradients, charges);
      continue;
    }
    const std::size_t begin = offsets[system];
    const int n = static_cast<int>(offsets[system + 1] - offsets[system]);
    const auto status = evaluate_eeq2019_with_tables(
        n, atomic_numbers + begin, coordinates + 3 * begin, total_charges[system], eeq_tables,
        workspace, eeq2019_workspace_elements(n), charges + begin, dqdr);
    statuses[system] = status;
    fixed_active[system] = status == D4Status::success ? 1 : 0;
    if (status != D4Status::success) zero_item(system, offsets, energies, gradients, charges);
  }
}

__global__ void eeq_compose_kernel(std::uint32_t systems, const std::uint32_t* offsets,
                                   const std::int32_t* atomic_numbers, const double* total_charges,
                                   const double* coordinates, const std::uint8_t* active,
                                   const std::uint8_t* want_gradient,
                                   const D4Status* fixed_statuses, EEQTables eeq_tables,
                                   std::uint32_t* next_system, double* eeq_workspace,
                                   std::size_t eeq_workspace_stride, double* dqdr_workspace,
                                   std::size_t dqdr_stride, D4Status* statuses, double* energies,
                                   double* gradients, double* charges, const double* dedq) {
  if (threadIdx.x != 0) return;
  const auto slot = static_cast<std::size_t>(blockIdx.x);
  double* workspace = eeq_workspace + slot * eeq_workspace_stride;
  double* dqdr = dqdr_workspace + slot * dqdr_stride;
  while (true) {
    const std::uint32_t system = atomicAdd(next_system, 1u);
    if (system >= systems) return;
    if (!active[system]) {
      statuses[system] = D4Status::success;
      continue;
    }
    if (statuses[system] != D4Status::success) {
      zero_item(system, offsets, energies, gradients, charges);
      continue;
    }
    if (fixed_statuses[system] != D4Status::success) {
      statuses[system] = fixed_statuses[system];
      zero_item(system, offsets, energies, gradients, charges);
      continue;
    }
    const std::size_t begin = offsets[system];
    const int n = static_cast<int>(offsets[system + 1] - offsets[system]);
    if (!want_gradient[system]) {
      for (int coordinate = 0; coordinate < 3 * n; ++coordinate)
        gradients[3 * begin + coordinate] = 0.0;
      continue;
    }
    auto status = evaluate_eeq2019_with_tables(
        n, atomic_numbers + begin, coordinates + 3 * begin, total_charges[system], eeq_tables,
        workspace, eeq2019_workspace_elements(n), charges + begin, dqdr);
    if (status == D4Status::success)
      status = ::vibeqc::generated::d4::compose_eeq_gradient(
          n, dqdr, dedq + begin, gradients + 3 * begin, gradients + 3 * begin);
    statuses[system] = status;
    if (status != D4Status::success) zero_item(system, offsets, energies, gradients, charges);
  }
}

}  // namespace

D4CudaOwner* create_d4_cuda_owner(int device_id, std::span<const std::uint32_t> offsets,
                                  std::span<const std::int32_t> atomic_numbers,
                                  std::span<const double> total_charges,
                                  std::span<const double> default_coordinates, D4EEQProfile profile,
                                  const D4ResourceUsage& resources, std::string& detail,
                                  vibeqc_status& status) {
  status = VIBEQC_STATUS_CUDA_ERROR;
  DeviceScope scope(device_id);
  if (scope.error() != cudaSuccess) {
    status = cuda_failure(scope.error(), "select D4 CUDA device", detail);
    return nullptr;
  }

  auto owner = std::make_unique<D4CudaOwner>();
  owner->device_id = device_id;
  owner->systems = static_cast<std::uint32_t>(offsets.size() - 1);
  owner->atoms = atomic_numbers.size();
  owner->workers = resources.worker_blocks;
  owner->eeq_workspace_stride =
      eeq2019_workspace_elements(static_cast<int>(resources.maximum_atoms));
  owner->dqdr_stride = 3u * resources.maximum_atoms * resources.maximum_atoms;
  if (!owner->workers || !owner->eeq_workspace_stride || !owner->dqdr_stride) {
    detail = "invalid D4 CUDA EEQ worker/workspace schedule";
    status = VIBEQC_STATUS_INTERNAL_ERROR;
    return nullptr;
  }

  auto error = cudaStreamCreateWithFlags(&owner->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    status = cuda_failure(error, "create D4 nonblocking stream", detail);
    return nullptr;
  }

  const auto atoms = static_cast<std::size_t>(owner->atoms);
  const auto systems = static_cast<std::size_t>(owner->systems);
  const auto c6_count = eeq_data::kReferenceC6Standard.size();
  if (!allocate(owner->offsets, offsets.size(), detail) ||
      !allocate(owner->atomic_numbers, atoms, detail) ||
      !allocate(owner->total_charges, systems, detail) ||
      !allocate(owner->coordinates, 3 * atoms, detail) ||
      !allocate(owner->active, systems, detail) ||
      !allocate(owner->want_gradient, systems, detail) ||
      !allocate(owner->fixed_active, systems, detail) ||
      !allocate(owner->statuses, systems, detail) ||
      !allocate(owner->fixed_statuses, systems, detail) ||
      !allocate(owner->energies, 2 * systems, detail) ||
      !allocate(owner->gradients, 3 * atoms, detail) || !allocate(owner->charges, atoms, detail) ||
      !allocate(owner->dedq, atoms, detail) || !allocate(owner->next_system, 1, detail) ||
      !allocate(owner->eeq_workspace,
                static_cast<std::size_t>(owner->workers) * owner->eeq_workspace_stride, detail) ||
      !allocate(owner->dqdr_workspace,
                static_cast<std::size_t>(owner->workers) * owner->dqdr_stride, detail) ||
      !allocate(owner->fixed_workspace, d4_cuda_workspace_elements(atoms), detail) ||
      !allocate(owner->elements, eeq_data::kElementCount, detail) ||
      !allocate(owner->references, eeq_data::kReferenceCount, detail) ||
      !allocate(owner->reference_c6, c6_count, detail) ||
      !allocate(owner->charge_elements, eeq_data::kElementCount, detail)) {
    status = VIBEQC_STATUS_OUT_OF_MEMORY;
    destroy_d4_cuda_owner(owner.release());
    return nullptr;
  }

  const double* c6 = profile == D4EEQProfile::r2scan3c ? eeq_data::kReferenceC6R2SCAN3C.data()
                                                       : eeq_data::kReferenceC6Standard.data();
  if (!upload(owner->stream, owner->offsets, offsets.data(), offsets.size(), detail) ||
      !upload(owner->stream, owner->atomic_numbers, atomic_numbers.data(), atoms, detail) ||
      !upload(owner->stream, owner->total_charges, total_charges.data(), systems, detail) ||
      !upload(owner->stream, owner->coordinates, default_coordinates.data(), 3 * atoms, detail) ||
      !upload(owner->stream, owner->elements, eeq_data::kElements.data(), eeq_data::kElementCount,
              detail) ||
      !upload(owner->stream, owner->references, eeq_data::kReferences.data(),
              eeq_data::kReferenceCount, detail) ||
      !upload(owner->stream, owner->reference_c6, c6, c6_count, detail) ||
      !upload(owner->stream, owner->charge_elements, eeq_data::kChargeElements.data(),
              eeq_data::kElementCount, detail)) {
    destroy_d4_cuda_owner(owner.release());
    return nullptr;
  }
  error = cudaStreamSynchronize(owner->stream);
  if (error != cudaSuccess) {
    status = cuda_failure(error, "synchronize D4 setup", detail);
    destroy_d4_cuda_owner(owner.release());
    return nullptr;
  }
  status = VIBEQC_STATUS_SUCCESS;
  detail.clear();
  return owner.release();
}

void destroy_d4_cuda_owner(D4CudaOwner* owner) noexcept {
  if (!owner) return;
  DeviceScope scope(owner->device_id);
  if (scope.error() == cudaSuccess) {
    if (owner->stream) cudaStreamSynchronize(owner->stream);
    cudaFree(owner->charge_elements);
    cudaFree(owner->reference_c6);
    cudaFree(owner->references);
    cudaFree(owner->elements);
    cudaFree(owner->fixed_workspace);
    cudaFree(owner->dqdr_workspace);
    cudaFree(owner->eeq_workspace);
    cudaFree(owner->next_system);
    cudaFree(owner->dedq);
    cudaFree(owner->charges);
    cudaFree(owner->gradients);
    cudaFree(owner->energies);
    cudaFree(owner->fixed_statuses);
    cudaFree(owner->statuses);
    cudaFree(owner->fixed_active);
    cudaFree(owner->want_gradient);
    cudaFree(owner->active);
    cudaFree(owner->coordinates);
    cudaFree(owner->total_charges);
    cudaFree(owner->atomic_numbers);
    cudaFree(owner->offsets);
    if (owner->stream) cudaStreamDestroy(owner->stream);
  }
  delete owner;
}

vibeqc_status execute_d4_cuda(D4CudaOwner* owner, const D4Parameters& parameters,
                              D4EEQProfile profile, std::span<const double> coordinates,
                              bool coordinates_changed, std::span<const std::uint8_t> active,
                              std::span<const std::uint8_t> want_gradient,
                              std::vector<D4Status>& statuses,
                              std::vector<double>& energy_components,
                              std::vector<double>& gradients, std::vector<double>& charges,
                              D4RuntimeCounters& counters, std::string& detail) {
  if (!owner || coordinates.size() != 3 * owner->atoms || active.size() != owner->systems ||
      want_gradient.size() != owner->systems || statuses.size() != owner->systems ||
      energy_components.size() != 2 * owner->systems || gradients.size() != 3 * owner->atoms ||
      charges.size() != owner->atoms) {
    detail = "invalid D4 CUDA replay shape";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  DeviceScope scope(owner->device_id);
  if (scope.error() != cudaSuccess)
    return cuda_failure(scope.error(), "select D4 CUDA replay device", detail);

  // Queued copies borrow caller storage until the stream completes.
  struct FailureDrain {
    cudaStream_t stream;
    bool armed{true};
    ~FailureDrain() {
      if (armed) (void)cudaStreamSynchronize(stream);
    }
  } failure_drain{owner->stream};

  auto copy_h2d = [&](void* destination, const void* source, std::size_t bytes,
                      const char* action) -> vibeqc_status {
    if (!bytes) return VIBEQC_STATUS_SUCCESS;
    const auto error =
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, owner->stream);
    return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS : cuda_failure(error, action, detail);
  };
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  if (coordinates_changed) {
    status = copy_h2d(owner->coordinates, coordinates.data(), coordinates.size_bytes(),
                      "upload D4 changed coordinates");
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    counters.coordinate_h2d_bytes += coordinates.size_bytes();
#if defined(VIBEQC_TEST_HOOKS)
    if (fail_next_d4_after_coordinate_upload) {
      fail_next_d4_after_coordinate_upload = false;
      detail = "injected D4 CUDA failure after coordinate upload";
      return VIBEQC_STATUS_CUDA_ERROR;
    }
#endif
  }
  status = copy_h2d(owner->active, active.data(), active.size_bytes(), "upload D4 active mask");
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  status = copy_h2d(owner->want_gradient, want_gradient.data(), want_gradient.size_bytes(),
                    "upload D4 gradient mask");
  if (status != VIBEQC_STATUS_SUCCESS) return status;

  for (const auto& clear :
       {std::pair<void*, std::size_t>{owner->energies, energy_components.size() * sizeof(double)},
        {owner->gradients, gradients.size() * sizeof(double)},
        {owner->charges, charges.size() * sizeof(double)},
        {owner->dedq, charges.size() * sizeof(double)},
        {owner->next_system, sizeof(std::uint32_t)}}) {
    const auto error = cudaMemsetAsync(clear.first, 0, clear.second, owner->stream);
    if (error != cudaSuccess) return cuda_failure(error, "clear D4 CUDA publication", detail);
  }

  const EEQTables eeq_tables{owner->elements, owner->charge_elements, eeq_data::kElementCount};
  eeq_prepare_kernel<<<owner->workers, 1, 0, owner->stream>>>(
      owner->systems, owner->offsets, owner->atomic_numbers, owner->total_charges,
      owner->coordinates, owner->active, eeq_tables, owner->next_system, owner->eeq_workspace,
      owner->eeq_workspace_stride, owner->dqdr_workspace, owner->dqdr_stride, owner->statuses,
      owner->fixed_active, owner->charges, owner->energies, owner->gradients);
  ++counters.kernel_launches;
  auto error = cudaPeekAtLastError();
  if (error != cudaSuccess) return cuda_failure(error, "launch D4 EEQ prepare kernel", detail);

  const bool r2scan = profile == D4EEQProfile::r2scan3c;
  const D4Tables d4_tables{D4ReferenceModel::eeq,
                           owner->elements,
                           owner->references,
                           owner->reference_c6,
                           eeq_data::kElementCount,
                           eeq_data::kReferenceCount,
                           eeq_data::kReferenceC6Standard.size(),
                           r2scan ? 2.0 : 3.0,
                           r2scan ? 1.0 : 2.0};
  const D4CudaBatch batch{owner->systems,     static_cast<std::uint32_t>(owner->atoms),
                          owner->offsets,     owner->atomic_numbers,
                          owner->coordinates, owner->charges,
                          owner->fixed_active};
  const D4CudaResult result{owner->fixed_statuses, owner->energies, owner->gradients, owner->dedq};
  error = launch_d4_fixed_charge_batched_cuda(
      batch, parameters, d4_tables, owner->fixed_workspace,
      d4_cuda_workspace_elements(static_cast<std::size_t>(owner->atoms)), result, owner->stream);
  counters.kernel_launches += parameters.s9 != 0.0 ? 7u : 5u;
  if (error != cudaSuccess)
    return cuda_failure(error, "launch cooperative fixed-charge D4 scheduler", detail);

  const bool gradients_requested = std::any_of(want_gradient.begin(), want_gradient.end(),
                                               [](std::uint8_t value) { return value != 0; });
  error = cudaMemsetAsync(owner->next_system, 0, sizeof(std::uint32_t), owner->stream);
  if (error != cudaSuccess) return cuda_failure(error, "reset D4 EEQ compose queue", detail);
  eeq_compose_kernel<<<owner->workers, 1, 0, owner->stream>>>(
      owner->systems, owner->offsets, owner->atomic_numbers, owner->total_charges,
      owner->coordinates, owner->active, owner->want_gradient, owner->fixed_statuses, eeq_tables,
      owner->next_system, owner->eeq_workspace, owner->eeq_workspace_stride, owner->dqdr_workspace,
      owner->dqdr_stride, owner->statuses, owner->energies, owner->gradients, owner->charges,
      owner->dedq);
  ++counters.kernel_launches;
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return cuda_failure(error, "launch D4 EEQ VJP compose kernel", detail);
  (void)gradients_requested;

  auto copy_d2h = [&](void* destination, const void* source, std::size_t bytes,
                      const char* action) -> vibeqc_status {
    if (!bytes) return VIBEQC_STATUS_SUCCESS;
    const auto copy_error =
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, owner->stream);
    return copy_error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                                     : cuda_failure(copy_error, action, detail);
  };
  status = copy_d2h(statuses.data(), owner->statuses, statuses.size() * sizeof(D4Status),
                    "download D4 statuses");
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  status = copy_d2h(energy_components.data(), owner->energies,
                    energy_components.size() * sizeof(double), "download D4 energy components");
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  status = copy_d2h(charges.data(), owner->charges, charges.size() * sizeof(double),
                    "download D4 EEQ charges");
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (gradients_requested) {
    status = copy_d2h(gradients.data(), owner->gradients, gradients.size() * sizeof(double),
                      "download D4 gradients");
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  error = cudaStreamSynchronize(owner->stream);
  if (error != cudaSuccess) return cuda_failure(error, "synchronize D4 CUDA replay", detail);
  failure_drain.armed = false;
  detail.clear();
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::dft::dispersion

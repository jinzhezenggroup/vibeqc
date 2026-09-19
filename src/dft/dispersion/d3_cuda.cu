#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <string>

#include "d3_data.hpp"
#include "dft/dispersion/d3_runtime.hpp"

namespace vibeqc::dft::dispersion {

struct D3CudaOwner {
  int device_id{-1};
  cudaStream_t stream{};
  std::uint32_t systems{};
  std::uint64_t atoms{};
  std::uint32_t* offsets{};
  std::int32_t* atomic_numbers{};
  double* coordinates{};
  std::uint8_t* active{};
  std::uint8_t* want_gradient{};
  D3Status* statuses{};
  double* energies{};
  double* gradients{};
  double* workspace{};
  d3_data::ElementData* elements{};
  d3_data::PairData* pairs{};
  double* reference_cn{};
  double* reference_c6{};
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
  detail = std::string("D3 CUDA allocation failed: ") + cudaGetErrorString(error);
  return false;
}

template <typename T>
bool upload(cudaStream_t stream, T* destination, const T* source, std::size_t count,
            std::string& detail) {
  if (!count) return true;
  const auto error =
      cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) return true;
  detail = std::string("D3 CUDA setup upload failed: ") + cudaGetErrorString(error);
  return false;
}

__global__ void d3_ragged_kernel(std::uint32_t systems, const std::uint32_t* offsets,
                                 const std::int32_t* atomic_numbers, const double* coordinates,
                                 const std::uint8_t* active, const std::uint8_t* want_gradient,
                                 D3Parameters parameters, D3Tables tables, double* workspace,
                                 D3Status* statuses, double* energies, double* gradients) {
  const auto system = static_cast<std::uint32_t>(blockIdx.x);
  if (system >= systems || threadIdx.x != 0) return;
  if (!active[system]) {
    statuses[system] = D3Status::success;
    energies[system] = 0.0;
    return;
  }

  const std::size_t begin = offsets[system];
  const std::size_t end = offsets[system + 1];
  const std::size_t atoms = end - begin;
  double energy = 0.0;
  double* gradient = want_gradient[system] ? gradients + 3 * begin : nullptr;
  const auto status =
      evaluate_d3_bj(atoms, atomic_numbers + begin, coordinates + 3 * begin, parameters, tables,
                     workspace + 16 * begin, 16 * atoms, &energy, gradient);
  statuses[system] = status;
  energies[system] = status == D3Status::success ? energy : 0.0;
}

}  // namespace

D3CudaOwner* create_d3_cuda_owner(int device_id, std::span<const std::uint32_t> offsets,
                                  std::span<const std::int32_t> atomic_numbers,
                                  const D3ResourceUsage& resources, std::string& detail,
                                  vibeqc_status& status) {
  status = VIBEQC_STATUS_CUDA_ERROR;
  DeviceScope scope(device_id);
  if (scope.error() != cudaSuccess) {
    status = cuda_failure(scope.error(), "select D3 CUDA device", detail);
    return nullptr;
  }

  auto owner = std::make_unique<D3CudaOwner>();
  owner->device_id = device_id;
  owner->systems = static_cast<std::uint32_t>(offsets.size() - 1);
  owner->atoms = atomic_numbers.size();

  auto error = cudaStreamCreateWithFlags(&owner->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    status = cuda_failure(error, "create D3 nonblocking stream", detail);
    return nullptr;
  }

  const auto atoms = static_cast<std::size_t>(owner->atoms);
  const auto systems = static_cast<std::size_t>(owner->systems);
  if (!allocate(owner->offsets, offsets.size(), detail) ||
      !allocate(owner->atomic_numbers, atoms, detail) ||
      !allocate(owner->coordinates, 3 * atoms, detail) ||
      !allocate(owner->active, systems, detail) ||
      !allocate(owner->want_gradient, systems, detail) ||
      !allocate(owner->statuses, systems, detail) || !allocate(owner->energies, systems, detail) ||
      !allocate(owner->gradients, 3 * atoms, detail) ||
      !allocate(owner->workspace, d3_ragged_workspace_elements(atoms), detail) ||
      !allocate(owner->elements, d3_data::kElements.size(), detail) ||
      !allocate(owner->pairs, d3_data::kPairs.size(), detail) ||
      !allocate(owner->reference_cn, d3_data::kReferenceCn.size(), detail) ||
      !allocate(owner->reference_c6, d3_data::kReferenceC6.size(), detail)) {
    status = VIBEQC_STATUS_OUT_OF_MEMORY;
    destroy_d3_cuda_owner(owner.release());
    return nullptr;
  }

  if (!upload(owner->stream, owner->offsets, offsets.data(), offsets.size(), detail) ||
      !upload(owner->stream, owner->atomic_numbers, atomic_numbers.data(), atoms, detail) ||
      !upload(owner->stream, owner->elements, d3_data::kElements.data(), d3_data::kElements.size(),
              detail) ||
      !upload(owner->stream, owner->pairs, d3_data::kPairs.data(), d3_data::kPairs.size(),
              detail) ||
      !upload(owner->stream, owner->reference_cn, d3_data::kReferenceCn.data(),
              d3_data::kReferenceCn.size(), detail) ||
      !upload(owner->stream, owner->reference_c6, d3_data::kReferenceC6.data(),
              d3_data::kReferenceC6.size(), detail)) {
    destroy_d3_cuda_owner(owner.release());
    return nullptr;
  }

  error = cudaStreamSynchronize(owner->stream);
  if (error != cudaSuccess) {
    status = cuda_failure(error, "synchronize D3 setup", detail);
    destroy_d3_cuda_owner(owner.release());
    return nullptr;
  }

  (void)resources;
  status = VIBEQC_STATUS_SUCCESS;
  return owner.release();
}

void destroy_d3_cuda_owner(D3CudaOwner* owner) noexcept {
  if (!owner) return;
  DeviceScope scope(owner->device_id);
  if (scope.error() == cudaSuccess) {
    if (owner->stream) cudaStreamSynchronize(owner->stream);
    cudaFree(owner->reference_c6);
    cudaFree(owner->reference_cn);
    cudaFree(owner->pairs);
    cudaFree(owner->elements);
    cudaFree(owner->workspace);
    cudaFree(owner->gradients);
    cudaFree(owner->energies);
    cudaFree(owner->statuses);
    cudaFree(owner->want_gradient);
    cudaFree(owner->active);
    cudaFree(owner->coordinates);
    cudaFree(owner->atomic_numbers);
    cudaFree(owner->offsets);
    if (owner->stream) cudaStreamDestroy(owner->stream);
  }
  delete owner;
}

vibeqc_status execute_d3_cuda(D3CudaOwner* owner, const D3Parameters& parameters,
                              std::span<const double> coordinates,
                              std::span<const std::uint8_t> active,
                              std::span<const std::uint8_t> want_gradient,
                              std::vector<D3Status>& statuses, std::vector<double>& energies,
                              std::vector<double>& gradients, std::string& detail) {
  if (!owner || coordinates.size() != 3 * owner->atoms || active.size() != owner->systems ||
      want_gradient.size() != owner->systems) {
    detail = "invalid D3 CUDA replay shape";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  DeviceScope scope(owner->device_id);
  if (scope.error() != cudaSuccess)
    return cuda_failure(scope.error(), "select D3 CUDA replay device", detail);

  auto copy_h2d = [&](void* destination, const void* source, std::size_t bytes,
                      const char* action) -> vibeqc_status {
    const auto error =
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, owner->stream);
    return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS : cuda_failure(error, action, detail);
  };

  vibeqc_status status = copy_h2d(owner->coordinates, coordinates.data(), coordinates.size_bytes(),
                                  "upload D3 changed coordinates");
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  status = copy_h2d(owner->active, active.data(), active.size_bytes(), "upload D3 active mask");
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  status = copy_h2d(owner->want_gradient, want_gradient.data(), want_gradient.size_bytes(),
                    "upload D3 gradient mask");
  if (status != VIBEQC_STATUS_SUCCESS) return status;

  // Failed, inactive and energy-only rows are copied with successful peers.
  // Initialize every publication slot rather than returning stale device data.
  const auto clear =
      cudaMemsetAsync(owner->gradients, 0, gradients.size() * sizeof(double), owner->stream);
  if (clear != cudaSuccess) return cuda_failure(clear, "clear D3 gradient publication", detail);
  const D3Tables tables{owner->elements, owner->pairs, owner->reference_cn, owner->reference_c6};
  d3_ragged_kernel<<<owner->systems, 1, 0, owner->stream>>>(
      owner->systems, owner->offsets, owner->atomic_numbers, owner->coordinates, owner->active,
      owner->want_gradient, parameters, tables, owner->workspace, owner->statuses, owner->energies,
      owner->gradients);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return cuda_failure(error, "launch D3 ragged CUDA kernel", detail);

  auto copy_d2h = [&](void* destination, const void* source, std::size_t bytes,
                      const char* action) -> vibeqc_status {
    const auto copy_error =
        cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, owner->stream);
    return copy_error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                                     : cuda_failure(copy_error, action, detail);
  };

  status = copy_d2h(statuses.data(), owner->statuses, statuses.size() * sizeof(D3Status),
                    "download D3 statuses");
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  status = copy_d2h(energies.data(), owner->energies, energies.size() * sizeof(double),
                    "download D3 energies");
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (std::any_of(want_gradient.begin(), want_gradient.end(),
                  [](std::uint8_t value) { return value != 0; })) {
    status = copy_d2h(gradients.data(), owner->gradients, gradients.size() * sizeof(double),
                      "download D3 gradients");
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }

  error = cudaStreamSynchronize(owner->stream);
  return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                              : cuda_failure(error, "synchronize D3 CUDA replay", detail);
}

}  // namespace vibeqc::dft::dispersion

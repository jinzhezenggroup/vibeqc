#ifndef VIBEQC_RUNTIME_RESOURCE_CUDA_CUH
#define VIBEQC_RUNTIME_RESOURCE_CUDA_CUH

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <new>

#include "runtime/resource_ledger.hpp"

namespace vibeqc::runtime {

/** Keep partial device uploads owned across host staging/vector failures.
 * The callable outlives this noncopyable guard and may also run explicitly
 * on a status-return path; it must clear released pointers and stream handles.
 */
template <class Cleanup>
class ResourceScopeExit {
 public:
  explicit ResourceScopeExit(Cleanup& cleanup) noexcept : cleanup_(cleanup) {}
  ResourceScopeExit(const ResourceScopeExit&) = delete;
  ResourceScopeExit& operator=(const ResourceScopeExit&) = delete;
  ~ResourceScopeExit() { cleanup_(); }

 private:
  Cleanup& cleanup_;
};

/** Charge the actual requested CUDA bytes before returning a native pointer.
 * A rejected allocation has the ordinary typed CUDA OOM status. No policy
 * changes the equation or silently selects another backend here.
 */
template <class Allocate, class Release>
cudaError_t resource_cuda_allocate(void** output, std::size_t bytes, Allocate allocate,
                                   Release release) {
  auto ledger = active_device_resource_ledger;
  if (!ledger) return allocate();
  if (output == nullptr) return cudaErrorInvalidValue;
  *output = nullptr;
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != ledger->device) return cudaErrorInvalidDevice;
  // Reserve while locked, but do not hold the registry lock across a CUDA
  // call: a driver callback may destroy another registered resource.
  {
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    if (bytes > ledger->limit - ledger->live) {
      ++ledger->rejected;
      return cudaErrorMemoryAllocation;
    }
    ledger->live += bytes;
  }
  status = allocate();
  const bool device_allocation_failed = status == cudaErrorMemoryAllocation;
  if (status == cudaSuccess && *output != nullptr) {
    try {
      std::lock_guard<std::mutex> lock(device_resource_mutex);
      if (device_allocation_generation == std::numeric_limits<std::uint64_t>::max())
        throw std::bad_alloc();
      // Another thread may reuse an address after cudaFree returned but
      // before old bookkeeping finished. Retire that generation; a delayed
      // free must not remove the new allocation at the same address.
      const auto old = device_allocation_owners.find(*output);
      if (old != device_allocation_owners.end()) {
        old->second.ledger->live -= old->second.bytes;
        device_allocation_owners.erase(old);
      }
      device_allocation_owners.emplace(
          *output, DeviceAllocationOwner{ledger, bytes, ++device_allocation_generation});
      ledger->peak = std::max(ledger->peak, ledger->live);
      ++ledger->allocations;
      return status;
    } catch (const std::bad_alloc&) {
      (void)release();
      *output = nullptr;
      status = cudaErrorMemoryAllocation;
    }
  }
  {
    std::lock_guard<std::mutex> lock(device_resource_mutex);
    ledger->live -= bytes;
    // Registry metadata can fail on the host after a successful CUDA call.
    // Preserve an unknown-space OOM in that case instead of authorizing a
    // device-specific retry based on a host bookkeeping failure.
    if (device_allocation_failed) ++ledger->rejected;
  }
  return status;
}

inline cudaError_t resource_cuda_malloc(void** output, std::size_t bytes) {
  return resource_cuda_allocate(
      output, bytes, [&] { return cudaMalloc(output, bytes); }, [&] { return cudaFree(*output); });
}

template <class T>
cudaError_t resource_cuda_malloc(T** output, std::size_t bytes) {
  return resource_cuda_malloc(reinterpret_cast<void**>(output), bytes);
}

inline cudaError_t resource_cuda_malloc_async(void** output, std::size_t bytes,
                                              cudaStream_t stream) {
  return resource_cuda_allocate(
      output, bytes, [&] { return cudaMallocAsync(output, bytes, stream); },
      [&] {
        const auto status = cudaFreeAsync(*output, stream);
        (void)cudaStreamSynchronize(stream);
        return status;
      });
}

template <class T>
cudaError_t resource_cuda_malloc_async(T** output, std::size_t bytes, cudaStream_t stream) {
  return resource_cuda_malloc_async(reinterpret_cast<void**>(output), bytes, stream);
}

/** Forget a successfully released pointer even when its scope has ended. */
inline std::uint64_t resource_cuda_generation(void* pointer) {
  std::lock_guard<std::mutex> lock(device_resource_mutex);
  const auto found = device_allocation_owners.find(pointer);
  return found == device_allocation_owners.end() ? 0 : found->second.generation;
}

inline void resource_cuda_forget(void* pointer, std::uint64_t generation) {
  std::lock_guard<std::mutex> lock(device_resource_mutex);
  const auto found = device_allocation_owners.find(pointer);
  if (found == device_allocation_owners.end() || found->second.generation != generation) return;
  found->second.ledger->live -= found->second.bytes;
  device_allocation_owners.erase(found);
}

inline cudaError_t resource_cuda_free(void* pointer) {
  const auto generation = resource_cuda_generation(pointer);
  const auto status = cudaFree(pointer);
  if (status == cudaSuccess) resource_cuda_forget(pointer, generation);
  return status;
}

inline cudaError_t resource_cuda_free_async(void* pointer, cudaStream_t stream) {
  const auto generation = resource_cuda_generation(pointer);
  auto status = cudaFreeAsync(pointer, stream);
  // The next owner can use a different stream. Do not release its logical
  // reservation until the previous physical use is complete. Unbudgeted
  // stream-ordered destruction keeps its original asynchronous behavior.
  if (status == cudaSuccess && generation != 0) status = cudaStreamSynchronize(stream);
  if (status == cudaSuccess) resource_cuda_forget(pointer, generation);
  return status;
}

}  // namespace vibeqc::runtime

#endif

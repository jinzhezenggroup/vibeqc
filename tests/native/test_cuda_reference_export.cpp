// Fault injection for asynchronous host staging; no CUDA device is accessed.
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>

#include "tensor/cuda_error.hpp"

namespace {
void* pending_destination = nullptr;
bool freed_before_drain = false;
unsigned copies = 0;

cudaError_t mock_copy(void* destination, const void* source, std::size_t bytes, cudaMemcpyKind,
                      cudaStream_t) {
  ++copies;
  if (copies <= 3) {
    std::memcpy(destination, source, bytes);  // convergence flags and iteration count
    return cudaSuccess;
  }
  if (copies == 4) {
    pending_destination = destination;
    return cudaSuccess;
  }
  return cudaErrorInvalidValue;  // next copy fails while the first is pending
}

cudaError_t mock_synchronize(cudaStream_t) {
  pending_destination = nullptr;
  return cudaSuccess;
}
}  // namespace

// Observe destruction without dereferencing freed storage or relying on an
// allocator to overwrite it. Both sized and unsized vector deletion are covered.
void operator delete(void* pointer) noexcept {
  if (pointer && pointer == pending_destination) freed_before_drain = true;
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

#define cudaMemcpyAsync mock_copy
#define cudaStreamSynchronize mock_synchronize
#include "scf/cuda/reference_export.cuh"
#undef cudaMemcpyAsync
#undef cudaStreamSynchronize

int main() {
  using vibeqc::scf::reference_detail::download;
  const std::array<const double*, 5> matrices{};
  const std::array<const double*, 3> scalars{};
  std::uint8_t converged = 0, failed = 0;
  std::uint32_t iterations = 1;
  vibeqc::scf::ScfResult result;
  try {
    // An unconverged solve must return its status before dimension checking or
    // any dense allocation, even when a reference could never fit in memory.
    auto status = download(nullptr, std::numeric_limits<std::size_t>::max(), 1, 0, matrices,
                           nullptr, scalars, &converged, &failed, &iterations, result);
    if (status != VIBEQC_STATUS_NOT_CONVERGED || result.reference)
      throw std::runtime_error("unconverged reference was allocated/published");
    copies = 0;
    converged = 1;
    bool threw = false;
    try {
      download(nullptr, 2, 1, 0, matrices, nullptr, scalars, &converged, &failed, &iterations,
               result);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    if (!threw || copies != 5 || pending_destination || freed_before_drain || result.reference)
      throw std::runtime_error("failed reference copy did not drain before freeing staging");
    std::cout << "Reference convergence preflight and exceptional staging lifetime passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

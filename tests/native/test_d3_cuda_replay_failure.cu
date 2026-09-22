#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/dispersion/d3_runtime.hpp"

#if defined(VIBEQC_D3_REPLAY_TEST_INTERPOSE)
namespace fault_injection {
enum class FailurePoint {
  none,
  second_upload,
  gradient_clear,
  launch_status,
  second_download,
  gradient_download,
};

FailurePoint failure = FailurePoint::none;
unsigned h2d_calls = 0;
unsigned d2h_calls = 0;
unsigned memsets = 0;
unsigned syncs = 0;
unsigned failure_drains = 0;
bool queued_host_reference = false;
bool injected = false;

void reset(FailurePoint point) {
  failure = point;
  h2d_calls = d2h_calls = memsets = syncs = failure_drains = 0;
  queued_host_reference = false;
  injected = false;
}
}  // namespace fault_injection

extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind,
                                              cudaStream_t);
extern "C" cudaError_t __real_cudaMemsetAsync(void*, int, std::size_t, cudaStream_t);
extern "C" cudaError_t __real_cudaGetLastError();
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);

extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* out, const void* in, std::size_t bytes,
                                              cudaMemcpyKind kind, cudaStream_t stream) {
  using namespace fault_injection;
  if (kind == cudaMemcpyHostToDevice) {
    ++h2d_calls;
    if (h2d_calls == 1) queued_host_reference = true;
    if (failure == FailurePoint::second_upload && h2d_calls == 2) {
      injected = true;
      return cudaErrorInvalidValue;
    }
  } else if (kind == cudaMemcpyDeviceToHost) {
    ++d2h_calls;
    if (d2h_calls == 1) queued_host_reference = true;
    if (failure == FailurePoint::second_download && d2h_calls == 2) {
      injected = true;
      return cudaErrorInvalidValue;
    }
    if (failure == FailurePoint::gradient_download && d2h_calls == 3) {
      injected = true;
      return cudaErrorInvalidValue;
    }
  }
  return __real_cudaMemcpyAsync(out, in, bytes, kind, stream);
}

extern "C" cudaError_t __wrap_cudaMemsetAsync(void* out, int value, std::size_t bytes,
                                              cudaStream_t stream) {
  using namespace fault_injection;
  ++memsets;
  if (failure == FailurePoint::gradient_clear && memsets == 1) {
    injected = true;
    return cudaErrorInvalidValue;
  }
  return __real_cudaMemsetAsync(out, value, bytes, stream);
}

extern "C" cudaError_t __wrap_cudaGetLastError() {
  using namespace fault_injection;
  if (failure == FailurePoint::launch_status) {
    failure = FailurePoint::none;
    injected = true;
    return cudaErrorInvalidConfiguration;
  }
  return __real_cudaGetLastError();
}

extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
  using namespace fault_injection;
  ++syncs;
  const auto status = __real_cudaStreamSynchronize(stream);
  if (queued_host_reference) {
    queued_host_reference = false;
    if (injected) ++failure_drains;
  }
  return status;
}
#endif

namespace {

using vibeqc::dft::dispersion::create_d3_cuda_owner;
using vibeqc::dft::dispersion::D3CudaOwner;
using vibeqc::dft::dispersion::D3ModelParameters;
using vibeqc::dft::dispersion::D3Parameters;
using vibeqc::dft::dispersion::D3ResourceUsage;
using vibeqc::dft::dispersion::D3Status;
using vibeqc::dft::dispersion::destroy_d3_cuda_owner;
using vibeqc::dft::dispersion::execute_d3_cuda;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Owner {
  D3CudaOwner* value{};
  ~Owner() { destroy_d3_cuda_owner(value); }
};

Owner make_owner() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return {};
  constexpr std::array<std::uint32_t, 2> offsets{0, 8};
  constexpr std::array<std::int32_t, 8> numbers{6, 6, 6, 6, 6, 6, 6, 6};
  D3ResourceUsage resources{};
  resources.maximum_atoms = 8;
  resources.total_atoms = 8;
  resources.system_count = 1;
  resources.maximum_bytes = std::numeric_limits<std::uint64_t>::max();
  std::string detail;
  vibeqc_status status = VIBEQC_STATUS_INTERNAL_ERROR;
  auto* owner = create_d3_cuda_owner(0, offsets, numbers, resources, detail, status);
  require(owner && status == VIBEQC_STATUS_SUCCESS, detail.c_str());
  return {owner};
}

D3ModelParameters model() {
  D3ModelParameters parameters{};
  parameters.bj = D3Parameters{1.0, 0.7875, 0.4289, 4.4407, 0.0, 0.0, 0.0, 0.0};
  return parameters;
}

vibeqc_status replay(D3CudaOwner* owner, bool gradient, std::string& detail) {
  std::vector<double> xyz(24);
  for (std::size_t atom = 0; atom < 8; ++atom) xyz[3 * atom] = 2.0 * atom;
  std::array<std::uint8_t, 1> active{1};
  std::array<std::uint8_t, 1> want_gradient{static_cast<std::uint8_t>(gradient)};
  std::vector<D3Status> statuses(1);
  std::vector<double> energies(1);
  std::vector<double> gradients(24);
  return execute_d3_cuda(owner, model(), xyz, active, want_gradient, statuses, energies, gradients,
                         detail);
}

#if defined(VIBEQC_D3_REPLAY_TEST_INTERPOSE)
void expect_failure_drain(D3CudaOwner* owner, fault_injection::FailurePoint point, bool gradient) {
  fault_injection::reset(point);
  std::string detail;
  const auto status = replay(owner, gradient, detail);
  require(status == VIBEQC_STATUS_CUDA_ERROR, "fault-injected D3 replay unexpectedly succeeded");
  require(fault_injection::injected, "D3 replay did not reach requested injected CUDA failure");
  require(!fault_injection::queued_host_reference,
          "D3 replay returned while stream retained a caller host reference");
  require(fault_injection::failure_drains == 1,
          "D3 replay failure did not drain retained stream exactly once");
}
#endif

}  // namespace

int main() {
  try {
    auto owner = make_owner();
    if (!owner.value) return 77;

#if defined(VIBEQC_D3_REPLAY_TEST_INTERPOSE)
    using fault_injection::FailurePoint;
    expect_failure_drain(owner.value, FailurePoint::second_upload, false);
    expect_failure_drain(owner.value, FailurePoint::gradient_clear, false);
    expect_failure_drain(owner.value, FailurePoint::launch_status, false);
    expect_failure_drain(owner.value, FailurePoint::second_download, false);
    expect_failure_drain(owner.value, FailurePoint::gradient_download, true);

    fault_injection::reset(FailurePoint::none);
    std::string detail;
    require(replay(owner.value, true, detail) == VIBEQC_STATUS_SUCCESS, detail.c_str());
    require(fault_injection::syncs == 1, "successful D3 replay synchronized more than once");
#else
    std::string detail;
    require(replay(owner.value, true, detail) == VIBEQC_STATUS_SUCCESS, detail.c_str());
#endif

    std::cout << "D3 CUDA replay failures drain retained stream before host buffers expire\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

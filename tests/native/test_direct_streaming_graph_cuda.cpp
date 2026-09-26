#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "runtime/cuda_provider.hpp"
#include "scf/cuda/direct_queue_scan.hpp"
#include "scf/direct_task_layout.hpp"

namespace {

void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

/** Return a graph whose captured arguments must survive this call's return. */
cudaGraphExec_t capture_flags(cudaStream_t stream, std::uint32_t* flags, std::uint64_t mask,
                              unsigned long long instantiate_flags) {
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  vibeqc::scf::cuda_execution::launch_reset_bounded_generated_streaming_flags_kernel(
      1, 64, 0, stream, mask, flags);
  check(cudaGetLastError());
  check(cudaStreamEndCapture(stream, &graph));
  const auto status = cudaGraphInstantiate(&executable, graph, instantiate_flags);
  check(cudaGraphDestroy(graph));
  check(status);
  check(cudaGraphUpload(executable, stream));
  check(cudaStreamSynchronize(stream));
  return executable;
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  cudaStream_t stream{};
  std::uint32_t* flags{};
  cudaGraphExec_t executable{};
  try {
    constexpr auto count = vibeqc::scf::detail::kDirectQuartetShellClassCount;
    std::array<std::uint32_t, count> actual{};
    check(cudaStreamCreate(&stream));
    check(cudaMalloc(reinterpret_cast<void**>(&flags), sizeof(actual)));
    for (const bool device_launch : {false, true}) {
      // CuMetal implements host replay; device launch is an NVIDIA capability.
      if (device_launch &&
          vibeqc::runtime::active_cuda_provider().kind != vibeqc::runtime::CudaProviderKind::Nvidia)
        continue;
      for (const auto mask : {std::uint64_t{0}, std::uint64_t{0x15}, std::uint64_t{1} << 54U,
                              std::uint64_t{1} << 63U, std::numeric_limits<std::uint64_t>::max()}) {
        executable = capture_flags(
            stream, flags, mask,
            device_launch ? static_cast<unsigned long long>(cudaGraphInstantiateFlagDeviceLaunch)
                          : 0ULL);
        for (unsigned replay = 0; replay < 3; ++replay) {
          // Poison every slot: replay must restore selected AND unselected
          // flags without any live host staging array from capture_flags.
          check(cudaMemsetAsync(flags, 0xff, sizeof(actual), stream));
          check(cudaGraphLaunch(executable, stream));
          check(cudaMemcpyAsync(actual.data(), flags, sizeof(actual), cudaMemcpyDeviceToHost,
                                stream));
          check(cudaStreamSynchronize(stream));
          for (std::size_t shell_class = 0; shell_class < count; ++shell_class) {
            if (actual[shell_class] != ((mask >> shell_class) & 1U))
              throw std::runtime_error("captured streaming partition changed on replay");
          }
        }
        check(cudaGraphExecDestroy(executable));
        executable = nullptr;
      }
    }
    check(cudaFree(flags));
    flags = nullptr;
    check(cudaStreamDestroy(stream));
    std::puts("validated streaming flags in host/device-launch graphs and repeated replay");
    return 0;
  } catch (const std::exception& error) {
    if (executable) (void)cudaGraphExecDestroy(executable);
    if (flags) (void)cudaFree(flags);
    if (stream) (void)cudaStreamDestroy(stream);
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

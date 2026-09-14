#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "scf/cuda/df_scf_library.hpp"

namespace {
void require(bool value, const char* detail) {
  if (!value) throw std::runtime_error(detail);
}
struct Resources {
  cudaStream_t stream{};
  void* byte{};
  ~Resources() {
    if (byte) (void)cudaFree(byte);
    if (stream) (void)cudaStreamDestroy(stream);
  }
};
}  // namespace

int main() {
  try {
    using vibeqc::scf::cuda_df::recover_scf_capture;
    Resources resources;
    require(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking) == cudaSuccess,
            "create capture recovery stream");
    require(cudaMalloc(&resources.byte, 1) == cudaSuccess, "allocate recovery sentinel");
    require(
        cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
        "begin rejected capture");
    // CUDA reserves ABI values 900/901 for unsupported/invalidated stream
    // capture. Use the values so compatible providers need not name the enums.
    require(static_cast<int>(cudaStreamSynchronize(resources.stream)) == 900,
            "capture fixture did not reject synchronization");
    cudaGraph_t graph{};
    const auto ended = cudaStreamEndCapture(resources.stream, &graph);
    if (graph) (void)cudaGraphDestroy(graph);
    require(static_cast<int>(ended) == 901, "capture did not invalidate");
    bool rejected = false;
    std::string detail;
    require(recover_scf_capture(resources.stream, ended, VIBEQC_STATUS_CUDA_ERROR, rejected,
                                detail) == VIBEQC_STATUS_SUCCESS &&
                rejected && detail.empty() && cudaPeekAtLastError() == cudaSuccess,
            "expected capture error poisoned ordinary execution");
    require(cudaMemsetAsync(resources.byte, 0, 1, resources.stream) == cudaSuccess &&
                cudaStreamSynchronize(resources.stream) == cudaSuccess,
            "ordinary work failed after capture rejection");
    for (auto error : {cudaErrorInvalidValue, cudaErrorIllegalAddress, cudaErrorMemoryAllocation}) {
      rejected = false;
      const auto expected = error == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                               : VIBEQC_STATUS_CUDA_ERROR;
      require(recover_scf_capture(resources.stream, error, VIBEQC_STATUS_SUCCESS, rejected,
                                  detail) == expected &&
                  !rejected && !detail.empty(),
              "capture recovery suppressed an unrelated error");
    }
    for (auto failure : {VIBEQC_STATUS_CUDA_ERROR, VIBEQC_STATUS_OUT_OF_MEMORY}) {
      rejected = false;
      detail = "library failure without a CUDA runtime error";
      require(recover_scf_capture(resources.stream, cudaSuccess, failure, rejected, detail) ==
                      failure &&
                  !rejected && detail == "library failure without a CUDA runtime error",
              "capture recovery suppressed an unexplained library failure");
    }
    require(recover_scf_capture(resources.stream, ended, VIBEQC_STATUS_OUT_OF_MEMORY, rejected,
                                detail) == VIBEQC_STATUS_OUT_OF_MEMORY &&
                !rejected,
            "capture invalidation suppressed a library allocation failure");
    require(
        cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
        "begin active capture guard");
    require(recover_scf_capture(resources.stream, cudaSuccess, VIBEQC_STATUS_SUCCESS, rejected,
                                detail) == VIBEQC_STATUS_CUDA_ERROR &&
                !rejected,
            "ordinary recovery accepted an active capture");
    graph = nullptr;
    require(cudaStreamEndCapture(resources.stream, &graph) == cudaSuccess,
            "end active capture guard");
    if (graph) (void)cudaGraphDestroy(graph);
    std::cout << "DF capture rejection preserves ordinary execution and unrelated errors\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

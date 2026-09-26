// Optional CUPTI audit of cuBLAS's retained allocations. Compile independently
// of VibeQC; run only in an allocated GPU job. A supplied workspace does not
// release the provider's default retained buffers. The callback records driver
// allocation APIs without calling CUDA recursively from inside CUPTI.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

std::atomic<const char*> phase{"initialization"};
template <class T>
void check(T status, const char* operation) {
  if (int(status) != 0)
    throw std::runtime_error(std::string(operation) + " status " + std::to_string(int(status)));
}
void CUPTIAPI allocation_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId,
                                  const void* raw) {
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
  auto data = static_cast<const CUpti_CallbackData*>(raw);
  if (data->callbackSite != CUPTI_API_ENTER) return;
  if (!std::strstr(data->functionName, "MemAlloc") && !std::strstr(data->functionName, "MemCreate"))
    return;
  if (!std::strcmp(data->functionName, "cuMemAlloc_v2")) {
    auto params = static_cast<const cuMemAlloc_v2_params*>(data->functionParams);
    std::printf("%s %s bytes=%zu\n", phase.load(), data->functionName, params->bytesize);
  } else if (!std::strcmp(data->functionName, "cuMemAllocAsync")) {
    auto params = static_cast<const cuMemAllocAsync_params*>(data->functionParams);
    std::printf("%s %s bytes=%zu\n", phase.load(), data->functionName, params->bytesize);
  } else {
    std::printf("%s %s size-not-decoded\n", phase.load(), data->functionName);
  }
}
void memory(const char* label) {
  size_t available = 0, total = 0;
  check(cudaMemGetInfo(&available, &total), "cudaMemGetInfo");
  std::printf("%s free=%zu total=%zu\n", label, available, total);
}
int main() {
  CUpti_SubscriberHandle subscriber{};
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  double* storage = nullptr;
  bool subscribed = false;
  auto cleanup = [&] {
    if (handle) cublasDestroy(handle);
    if (storage) cudaFree(storage);
    if (stream) cudaStreamDestroy(stream);
    if (subscribed) cuptiUnsubscribe(subscriber);
  };
  try {
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaFree(nullptr), "initialize CUDA");
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
    std::printf("Slurm job=%s GPU=%s architecture=sm_%d%d\n",
                std::getenv("SLURM_JOB_ID") ? std::getenv("SLURM_JOB_ID") : "not-recorded",
                device.name, device.major, device.minor);
    check(cuptiSubscribe(&subscriber, allocation_callback, nullptr), "cuptiSubscribe");
    subscribed = true;
    check(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "cuptiEnableDomain");
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    check(cudaMalloc(reinterpret_cast<void**>(&storage), 4 * 1024 * 1024 + 1024),
          "tensor/workspace allocation");
    memory("before create");
    phase = "cublasCreate";
    check(cublasCreate(&handle), "cublasCreate");
    memory("after create");
    phase = "bind workspace";
    check(cublasSetStream(handle, stream), "cublasSetStream");
    check(cublasSetWorkspace(handle, storage + 128, 4 * 1024 * 1024), "cublasSetWorkspace");
    memory("after workspace");
    double inputs[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    check(cudaMemcpy(storage, inputs, sizeof(inputs), cudaMemcpyHostToDevice), "input copy");
    double alpha = 1, beta = 0;
    phase = "gemm";
    check(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, storage, 2, storage + 4, 2,
                      &beta, storage + 8, 2),
          "cublasDgemm");
    check(cudaStreamSynchronize(stream), "GEMM synchronization");
    memory("after gemm");
    phase = "strided batched gemm";
    check(cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, storage, 2,
                                    0, storage + 4, 2, 0, &beta, storage + 8, 2, 4, 2),
          "cublasDgemmStridedBatched");
    check(cudaStreamSynchronize(stream), "batched GEMM synchronization");
    memory("after batched gemm");
    phase = "destroy";
    check(cublasDestroy(handle), "cublasDestroy");
    handle = nullptr;
    memory("after destroy");
    cleanup();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    cleanup();
    return 1;
  }
}

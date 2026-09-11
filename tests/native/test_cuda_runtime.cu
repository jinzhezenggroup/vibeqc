#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cub/block/block_scan.cuh>
#include <vector>

extern "C" __global__ void vibeqc_test_indexing_kernel(const float* input, float* output,
                                                       std::size_t size) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) output[index] = 2.0F * input[index] + 1.0F;
}

extern "C" __global__ void vibeqc_test_block_scan_kernel(const int* input, int* exclusive,
                                                         int* aggregate) {
  using BlockScan = cub::BlockScan<int, 32>;
  __shared__ typename BlockScan::TempStorage storage;
  int prefix = 0;
  int total = 0;
  BlockScan(storage).ExclusiveSum(input[threadIdx.x], prefix, total);
  exclusive[threadIdx.x] = prefix;
  if (threadIdx.x == 0) *aggregate = total;
}

extern "C" __global__ void vibeqc_test_atomic_kernel(float* output) { atomicAdd(output, 1.0F); }

namespace {

bool check_cuda(cudaError_t error, const char* expression) {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "FAIL: %s: %s\n", expression, cudaGetErrorString(error));
  return false;
}

bool check_cublas(cublasStatus_t status, const char* expression) {
  if (status == CUBLAS_STATUS_SUCCESS) return true;
  std::fprintf(stderr, "FAIL: %s: cuBLAS status %d\n", expression, static_cast<int>(status));
  return false;
}

bool test_device_properties() {
  cudaDeviceProp properties{};
  if (!check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")) {
    return false;
  }
  if (properties.maxThreadsPerMultiProcessor <= 0 || properties.maxBlocksPerMultiProcessor <= 0 ||
      properties.regsPerMultiprocessor <= 0) {
    std::fprintf(stderr, "FAIL: invalid occupancy properties threads=%d blocks=%d regs=%d\n",
                 properties.maxThreadsPerMultiProcessor, properties.maxBlocksPerMultiProcessor,
                 properties.regsPerMultiprocessor);
    return false;
  }
  return true;
}

bool test_indexing(cudaStream_t stream) {
  constexpr std::size_t size = 67;
  std::vector<float> input(size);
  std::vector<float> output(size, 0.0F);
  for (std::size_t i = 0; i < size; ++i) input[i] = static_cast<float>(i) * 0.25F;

  float* device_input = nullptr;
  float* device_output = nullptr;
  bool ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_input), size * sizeof(float)),
                       "cudaMalloc(index input)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), size * sizeof(float)),
                       "cudaMalloc(index output)") &&
            check_cuda(cudaMemcpyAsync(device_input, input.data(), size * sizeof(float),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync(index input)");
  if (ok) {
    vibeqc_test_indexing_kernel<<<3, 32, 0, stream>>>(device_input, device_output, size);
    ok = check_cuda(cudaGetLastError(), "indexing kernel launch") &&
         check_cuda(cudaMemcpyAsync(output.data(), device_output, size * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync(index output)") &&
         check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(indexing)");
  }

  cudaFree(device_input);
  cudaFree(device_output);
  if (!ok) return false;

  for (std::size_t i = 0; i < size; ++i) {
    const float expected = 2.0F * input[i] + 1.0F;
    // Negate the passing predicate so NaN is a failure as well as large error.
    if (!(std::fabs(output[i] - expected) <= 1.0e-6F)) {
      std::fprintf(stderr, "FAIL: indexing[%zu]=%.7g expected=%.7g\n", i,
                   static_cast<double>(output[i]), static_cast<double>(expected));
      return false;
    }
  }
  return true;
}

bool test_block_scan(cudaStream_t stream) {
  constexpr int threads = 32;
  std::vector<int> input(threads);
  std::vector<int> output(threads, -1);
  for (int i = 0; i < threads; ++i) input[i] = i + 1;
  int aggregate = -1;

  int* device_input = nullptr;
  int* device_output = nullptr;
  int* device_aggregate = nullptr;
  bool ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_input), threads * sizeof(int)),
                       "cudaMalloc(scan input)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_output), threads * sizeof(int)),
                       "cudaMalloc(scan output)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_aggregate), sizeof(int)),
                       "cudaMalloc(scan aggregate)") &&
            check_cuda(cudaMemcpyAsync(device_input, input.data(), threads * sizeof(int),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync(scan input)");
  if (ok) {
    vibeqc_test_block_scan_kernel<<<1, threads, 0, stream>>>(device_input, device_output,
                                                             device_aggregate);
    ok = check_cuda(cudaGetLastError(), "BlockScan launch") &&
         check_cuda(cudaMemcpyAsync(output.data(), device_output, threads * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync(scan output)") &&
         check_cuda(cudaMemcpyAsync(&aggregate, device_aggregate, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync(scan aggregate)") &&
         check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(BlockScan)");
  }

  cudaFree(device_input);
  cudaFree(device_output);
  cudaFree(device_aggregate);
  if (!ok) return false;

  int running = 0;
  for (int i = 0; i < threads; ++i) {
    if (output[i] != running) {
      std::fprintf(stderr, "FAIL: BlockScan[%d]=%d expected=%d\n", i, output[i], running);
      return false;
    }
    running += input[i];
  }
  if (aggregate != running) {
    std::fprintf(stderr, "FAIL: BlockScan aggregate=%d expected=%d\n", aggregate, running);
    return false;
  }
  return true;
}

bool test_atomic(cudaStream_t stream) {
  float* device_value = nullptr;
  float value = 0.0F;
  bool ok = check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_value), sizeof(float)),
                       "cudaMalloc(atomic)") &&
            check_cuda(cudaMemsetAsync(device_value, 0, sizeof(float), stream),
                       "cudaMemsetAsync(atomic)");
  if (ok) {
    vibeqc_test_atomic_kernel<<<1, 32, 0, stream>>>(device_value);
    ok = check_cuda(cudaGetLastError(), "atomicAdd launch") &&
         check_cuda(
             cudaMemcpyAsync(&value, device_value, sizeof(float), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync(atomic)") &&
         check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(atomic)");
  }
  cudaFree(device_value);
  return ok && std::fabs(value - 32.0F) <= 1.0e-5F;
}

bool test_cublas(cudaStream_t stream) {
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  const std::vector<float> a = {1.0F, 3.0F, 2.0F, 4.0F};
  const std::vector<float> identity = {1.0F, 0.0F, 0.0F, 1.0F};
  std::vector<float> c(4, 0.0F);
  const float alpha = 1.0F;
  const float beta = 0.0F;

  bool ok = check_cublas(cublasCreate(&handle), "cublasCreate") &&
            check_cublas(cublasSetStream(handle, stream), "cublasSetStream") &&
            check_cublas(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED),
                         "cublasSetAtomicsMode") &&
            check_cuda(cudaMalloc(&workspace, 4096), "cudaMalloc(cuBLAS workspace)") &&
            check_cublas(cublasSetWorkspace(handle, workspace, 4096), "cublasSetWorkspace") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), 4 * sizeof(float)),
                       "cudaMalloc(SGEMM A)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), 4 * sizeof(float)),
                       "cudaMalloc(SGEMM B)") &&
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_c), 4 * sizeof(float)),
                       "cudaMalloc(SGEMM C)");
  if (ok) {
    // Exercise the same control used by generated tensor/grid consumers.
    cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
    ok = check_cublas(cublasGetAtomicsMode(handle, &mode), "cublasGetAtomicsMode") &&
         mode == CUBLAS_ATOMICS_NOT_ALLOWED;
  }
  if (ok) {
    ok = check_cuda(
             cudaMemcpyAsync(device_a, a.data(), 4 * sizeof(float), cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(SGEMM A)") &&
         check_cuda(cudaMemcpyAsync(device_b, identity.data(), 4 * sizeof(float),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync(SGEMM B)") &&
         check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, device_a, 2,
                                  device_b, 2, &beta, device_c, 2),
                      "cublasSgemm") &&
         check_cuda(
             cudaMemcpyAsync(c.data(), device_c, 4 * sizeof(float), cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync(SGEMM C)") &&
         check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(SGEMM)");
  }

  cudaFree(workspace);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  if (handle != nullptr) cublasDestroy(handle);
  if (!ok) return false;

  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!(std::fabs(c[i] - a[i]) <= 1.0e-5F)) return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!check_cuda(cudaSetDevice(0), "cudaSetDevice(0)")) return 1;
  if (!test_device_properties()) return 2;

  cudaStream_t stream = nullptr;
  if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags")) {
    return 3;
  }

  const bool ok = test_indexing(stream) && test_block_scan(stream) && test_atomic(stream) &&
                  test_cublas(stream);
  cudaStreamDestroy(stream);
  if (!ok) return 4;

  std::printf("PASS: CUDA runtime contracts\n");
  return 0;
}
